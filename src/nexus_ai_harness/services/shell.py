from __future__ import annotations

import os
import re
import secrets
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from nexus_ai_harness.protocols.sandbox import (
    SandboxResult,
    SandboxViolation,
    ShellResult,
)

# --- command-string scanning (for sandbox command policy) -------------------

_OPERATORS = (
    "&>>",
    "<<<",
    "<<-",
    "&&",
    "||",
    ";;",
    "|&",
    ">>",
    "<<",
    "<&",
    ">&",
    "&>",
    ">|",
    "<>",
    ";",
    "&",
    "|",
    "(",
    ")",
    "<",
    ">",
    "\n",
)
"""Shell operators, longest first so a greedy match picks ``&&`` over ``&``."""

_REDIRECTS = frozenset(
    {"&>>", "<<<", "<<-", "<<", ">>", "<&", ">&", "&>", ">|", "<>", "<", ">"}
)
"""Operators whose following word is a redirection target, not a command."""

_PREFIX_KEYWORDS = frozenset(
    {"if", "then", "else", "elif", "do", "while", "until", "!", "{", "time"}
)
"""Reserved words after which the next word is still in command position."""

_END_KEYWORDS = frozenset({"fi", "done", "}"})
"""Reserved words that close a compound command and run nothing themselves."""

_LOOP_KEYWORDS = frozenset({"for", "select"})
"""Reserved words that assign their next word and list data up to a separator."""

WRITE_REDIRECTS = frozenset({">", ">>", ">|", "<>", "&>", "&>>", ">&"})
"""Redirection operators that open their target for writing."""

_DATA_REDIRECTS = frozenset({"<<", "<<-", "<<<"})
"""Redirections whose following word is data (a heredoc delimiter or string)."""

_DIR_CHANGERS = frozenset({"cd", "pushd", "popd"})
"""Commands that change the directory relative redirection targets resolve in."""

_INERT_BUILTINS = frozenset(
    {
        "cd",
        "pwd",
        "echo",
        "printf",
        "true",
        "false",
        ":",
        "exit",
        "return",
        "shift",
        "export",
        "set",
        "umask",
        "test",
        "[",
    }
)
"""Builtins that run no other program, so the policy need not check them."""

_CODE_BUILTINS = frozenset(
    {
        "eval",
        "source",
        ".",
        "trap",
        "alias",
        "enable",
        "hash",
        "coproc",
        "let",
        "[[",
        "case",
    }
)
"""Words that evaluate text as code or arithmetic the scanner cannot follow."""

_EXEC_BUILTINS: dict[str, frozenset[str]] = {
    "exec": frozenset({"-a"}),
    "command": frozenset(),
    "builtin": frozenset(),
}
"""Builtins that run the next word as a program, with their valued options."""

_WRAPPERS: dict[str, tuple[frozenset[str], int]] = {
    "env": (frozenset({"-u", "--unset", "-C", "--chdir"}), 0),
    "nohup": (frozenset(), 0),
    "nice": (frozenset({"-n", "--adjustment"}), 0),
    "timeout": (frozenset({"-s", "--signal", "-k", "--kill-after"}), 1),
    "stdbuf": (frozenset({"-i", "-o", "-e"}), 0),
    "xargs": (frozenset({"-I", "-n", "-P", "-L", "-s", "-E", "-d", "-a"}), 0),
}
"""Programs that run another program: (options taking a value, positionals)."""

_FIND_EXEC = frozenset({"-exec", "-execdir", "-ok", "-okdir"})
"""``find`` actions whose next word is a program to run."""

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PLAIN_PATH = re.compile(r"[A-Za-z0-9_./+-]+")
_FD_VARIABLE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\+?=")
_SIMPLE_BRACED = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]|[@*#?$!-]")
_SPECIAL_PARAMS = "@*#?$!-0123456789"


@dataclass
class CommandScan:
    """What a static scan of a shell command string could determine.

    Attributes:
        programs: Every program name the string runs that the scanner could
            identify, in order (inert builtins such as ``cd`` are omitted).
        opaque: Reasons the string may run code the scanner cannot see, such
            as a command substitution or a computed program name. Empty when
            ``programs`` is the complete set of programs the string runs.
        assignments: Variables the string sets (``NAME=value`` prefixes and
            statements, ``export``, ``env NAME=value``, ``for NAME in``,
            ``{NAME}>file``). A variable such as ``PATH`` or
            ``GIT_EXTERNAL_DIFF`` can change what a permitted program executes,
            so an allowlist may refuse these too.
        redirects: ``(operator, target)`` for each file redirection whose
            target is literal text, e.g. ``(">", "out.txt")``. Heredoc and
            here-string data and ``2>&1``-style descriptor copies are omitted;
            a computed target is reported in :attr:`opaque` instead.
        directories: The target of each ``cd``/``pushd``/``popd``, in order,
            or ``None`` where it is not a single literal directory (``cd``
            alone, ``cd -``, ``popd``, a computed path). After any of these a
            relative redirection target no longer resolves against the
            starting directory.
        repeats: Whether the string has a ``while``/``until`` loop or defines a
            function, so its commands may run more times than they appear.
    """

    programs: list[str] = field(default_factory=list)
    opaque: list[str] = field(default_factory=list)
    assignments: list[str] = field(default_factory=list)
    redirects: list[tuple[str, str]] = field(default_factory=list)
    directories: list[str | None] = field(default_factory=list)
    repeats: bool = False


@dataclass
class _Word:
    text: str = ""
    quoted: bool = False
    dynamic: bool = False
    globbed: bool = False
    literal_prefix: int = -1


@dataclass
class _Token:
    op: str | None = None
    word: _Word | None = None


def scan_command(command: str) -> CommandScan:
    """Statically find the programs a shell command string would run.

    Splits the string into simple commands at ``;``, ``&``, ``|``, ``&&``,
    ``||``, newlines, and parentheses, drops redirections and variable
    assignments, looks through wrappers (``env``, ``nohup``, ``timeout``,
    ``xargs``, ``exec``, ``command``, ``find -exec``), and records each
    remaining program name. Anything that can run code the scan cannot see (a
    command or process substitution, a heredoc, ``eval``/``source``, arithmetic,
    a computed program name) is reported in :attr:`CommandScan.opaque` rather
    than guessed at, so a caller enforcing an allowlist can refuse it.

    The scan is conservative, not a full shell parser: an unusual construct may
    be reported opaque (or a harmless word listed as a program), never the
    other way round for the constructs it recognizes. It checks names only;
    what a name resolves to on ``PATH`` is outside its view.

    Args:
        command: The shell command string.

    Returns:
        The identified programs and any reasons the scan is incomplete.
    """
    scan = CommandScan()
    words: list[_Word] = []
    redirect: str | None = None
    previous: _Token | None = None
    for token in _tokenize(command, scan):
        if token.op == ")" and previous is not None and previous.op == "(":
            scan.repeats = True  # ``name() { ...; }`` defines a function
        previous = token
        if token.word is not None:
            if redirect is not None:
                _redirection(redirect, token.word, scan)
                redirect = None
            else:
                words.append(token.word)
            continue
        op = token.op or ""
        if op in _REDIRECTS:
            redirect = op
            continue
        _analyze(words, scan)
        words = []
    _analyze(words, scan)
    return scan


def _redirection(op: str, target: _Word, scan: CommandScan) -> None:
    """Record a redirection's target, or why it cannot be known statically."""
    if op in _DATA_REDIRECTS:
        return
    text = target.text
    if op in ("<&", ">&") and not target.dynamic and (text.isdigit() or text == "-"):
        return  # a descriptor copy or close such as 2>&1, not a file
    if op == "<&":
        return
    if target.dynamic:
        scan.opaque.append("redirection target is computed at run time")
    elif text.startswith("~") and not target.quoted:
        scan.opaque.append("redirection target uses '~' expansion")
    else:
        scan.redirects.append((op, text))


def _tokenize(command: str, scan: CommandScan) -> list[_Token]:
    """Split ``command`` into words and operators, noting opaque constructs."""
    return _Lexer(command, scan).tokens()


class _Lexer:
    """A single quote-aware, left-to-right pass over a command string."""

    def __init__(self, command: str, scan: CommandScan) -> None:
        self._s = command
        self._scan = scan
        self._i = 0
        self._word: _Word | None = None
        self._tokens: list[_Token] = []
        self._heredocs: list[tuple[str, bool, bool]] = []
        self._await_delimiter: bool | None = None

    def tokens(self) -> list[_Token]:
        s, n = self._s, len(self._s)
        while self._i < n:
            c = s[self._i]
            if c in " \t\r":
                self._flush()
                self._i += 1
            elif c == "#" and self._word is None:
                while self._i < n and s[self._i] != "\n":
                    self._i += 1
            elif c in ";&|()<>\n":
                self._operator(c)
            elif c == "'":
                word = self._begin(quoted=True)
                end = s.find("'", self._i + 1)
                if end == -1:
                    self._scan.opaque.append("unterminated quote")
                    end = n
                word.text += s[self._i + 1 : end]
                self._i = end + 1
            elif c == '"':
                word = self._begin(quoted=True)
                self._i = _double_quoted(s, self._i + 1, word, self._scan)
            elif c == "\\":
                if s.startswith("\n", self._i + 1):  # line continuation
                    self._i += 2
                    continue
                self._begin(quoted=True).text += s[self._i + 1 : self._i + 2]
                self._i += 2
            elif c == "`":
                self._scan.opaque.append("command substitution")
                self._begin(quoted=True).dynamic = True
                self._i += 1
            elif c == "$":
                word = self._begin(quoted=True)
                self._i = _expansion(s, self._i, word, self._scan)
            else:
                word = self._begin()
                if c in "*?[{}" and not _standalone(s, self._i):
                    word.globbed = True
                word.text += c
                self._i += 1
        self._flush()
        return self._tokens

    def _begin(self, *, quoted: bool = False) -> _Word:
        """Return the word being built, starting one if needed."""
        if self._word is None:
            self._word = _Word()
        if quoted:
            self._word.quoted = True
            if self._word.literal_prefix < 0:
                self._word.literal_prefix = len(self._word.text)
        return self._word

    def _flush(self) -> None:
        word = self._word
        if word is None:
            return
        self._word = None
        if word.globbed and word.text not in ("[", "[["):
            word.dynamic = True  # glob or brace expansion
        if self._await_delimiter is not None:
            self._heredocs.append((word.text, word.quoted, self._await_delimiter))
            self._await_delimiter = None
        self._tokens.append(_Token(word=word))

    def _operator(self, c: str) -> None:
        s, i = self._s, self._i
        word = self._word
        if c in "<>" and word is not None and not word.quoted:
            if word.text.isdigit():
                self._word = None  # an fd number such as the 2 in 2>&1
            elif match := _FD_VARIABLE.fullmatch(word.text):
                self._scan.assignments.append(match.group(1))  # {fd}>file
                self._word = None
        self._flush()
        if s.startswith("((", i):
            self._scan.opaque.append("arithmetic command '(('")
        if c in "<>" and s.startswith("(", i + 1):
            self._scan.opaque.append("process substitution")
        op = next(o for o in _OPERATORS if s.startswith(o, i))
        if op == ";;":
            self._scan.opaque.append("case statement")
        elif op in ("<<", "<<-"):
            self._await_delimiter = op == "<<-"
        self._tokens.append(_Token(op=op))
        self._i = i + len(op)
        if op == "\n" and self._heredocs:
            self._skip_heredoc_bodies()

    def _skip_heredoc_bodies(self) -> None:
        """Consume pending heredoc bodies, which are data rather than commands."""
        s = self._s
        for delimiter, quoted, strip_tabs in self._heredocs:
            body: list[str] = []
            while self._i < len(s):
                end = s.find("\n", self._i)
                end = len(s) if end == -1 else end
                line = s[self._i : end]
                self._i = end + 1
                if (line.lstrip("\t") if strip_tabs else line) == delimiter:
                    break
                body.append(line)
            text = "\n".join(body)
            if not quoted and any(m in text for m in ("$(", "`", "$[", "${")):
                self._scan.opaque.append("expansion inside an unquoted heredoc")
        self._heredocs = []


def _standalone(command: str, i: int) -> bool:
    """Whether the brace at ``i`` is a whole word (the ``{ ...; }`` keyword)."""
    if command[i] not in "{}":
        return False
    before = command[i - 1] if i > 0 else " "
    after = command[i + 1] if i + 1 < len(command) else " "
    return before in " \t\r\n;&|()" and after in " \t\r\n;&|()"


def _double_quoted(command: str, i: int, current: _Word, scan: CommandScan) -> int:
    """Consume a double-quoted span starting after its quote; return the end."""
    n = len(command)
    while i < n:
        c = command[i]
        if c == '"':
            return i + 1
        if c == "\\" and i + 1 < n and command[i + 1] in '$`"\\\n':
            current.text += command[i + 1]
            i += 2
        elif c == "`":
            scan.opaque.append("command substitution")
            current.dynamic = True
            i += 1
        elif c == "$":
            i = _expansion(command, i, current, scan, in_quotes=True)
        else:
            current.text += c
            i += 1
    scan.opaque.append("unterminated quote")
    return n


def _expansion(
    command: str,
    i: int,
    current: _Word,
    scan: CommandScan,
    *,
    in_quotes: bool = False,
) -> int:
    """Consume a ``$`` expansion at ``i``; return the index just past it."""
    n = len(command)
    nxt = command[i + 1] if i + 1 < n else ""
    if nxt == "(":
        scan.opaque.append("command substitution or arithmetic expansion")
    elif nxt == "[":
        scan.opaque.append("arithmetic expansion")
    elif nxt == "{":
        end = command.find("}", i + 2)
        if end == -1 or not _SIMPLE_BRACED.fullmatch(command[i + 2 : end]):
            scan.opaque.append("complex parameter expansion")
            end = n - 1 if end == -1 else end
        current.dynamic = True
        return end + 1
    elif nxt == "'" and not in_quotes:
        current.dynamic = True
        end = i + 2
        while end < n and command[end] != "'":
            end += 2 if command[end] == "\\" else 1
        return end + 1
    elif nxt and (nxt in _SPECIAL_PARAMS or _NAME.match(nxt)):
        current.dynamic = True
        if nxt in _SPECIAL_PARAMS:
            return i + 2
        match = _NAME.match(command, i + 1)
        return match.end() if match else i + 2
    else:
        current.text += "$"
        return i + 1
    current.dynamic = True
    return i + 2


def _analyze(words: list[_Word], scan: CommandScan) -> None:
    """Record the program run by one simple command's words."""
    i = 0
    while i < len(words):
        word = words[i]
        bare = not word.quoted and not word.dynamic
        if bare and word.text in _LOOP_KEYWORDS:
            # The loop variable is assigned (and exported under ``set -a`` or if
            # already exported, e.g. ``for PATH in ...``); the rest is data.
            if i + 1 < len(words):
                variable = words[i + 1]
                if variable.dynamic or not _NAME.fullmatch(variable.text):
                    scan.opaque.append(f"'{word.text}' over a computed variable")
                scan.assignments.append(variable.text)
            return
        if bare and word.text == "function":
            scan.repeats = True
            i += 2
            continue
        if bare and word.text in ("while", "until"):
            scan.repeats = True
        if bare and (word.text in _PREFIX_KEYWORDS or word.text in _END_KEYWORDS):
            i += 1
            if word.text == "time" and i < len(words) and words[i].text == "-p":
                i += 1
            continue
        match = _ASSIGNMENT.match(word.text)
        if match and 0 <= match.end() <= _prefix(word):
            scan.assignments.append(word.text.split("=", 1)[0].rstrip("+"))
            i += 1
            continue
        break
    _command(words[i:], scan)


def _prefix(word: _Word) -> int:
    """Length of the leading text that was written unquoted and unexpanded."""
    return len(word.text) if word.literal_prefix < 0 else word.literal_prefix


def _command(words: list[_Word], scan: CommandScan) -> None:
    """Record ``words[0]`` as a program, following wrappers to what they run."""
    while words:
        word = words[0]
        name = word.text
        if word.dynamic:
            shown = f": {name!r}" if name else ""
            scan.opaque.append(f"program name is computed at run time{shown}")
            return
        if name in _CODE_BUILTINS:
            scan.opaque.append(f"{name!r} evaluates code the policy cannot inspect")
            return
        if name in _EXEC_BUILTINS:
            words = _skip_options(words[1:], _EXEC_BUILTINS[name], 0, scan)
            continue
        if name in _DIR_CHANGERS:
            scan.directories.append(_directory(name, words[1:]))
        if name in _INERT_BUILTINS:
            _check_inert(name, words[1:], scan)
            return
        scan.programs.append(name)
        base = os.path.basename(name)
        if base == "find":
            for index, arg in enumerate(words):
                if arg.text in _FIND_EXEC:
                    _command(words[index + 1 :], scan)
            return
        spec = _WRAPPERS.get(base)
        if spec is None:
            return
        words = _skip_options(
            words[1:], spec[0], spec[1], scan, assignments=base == "env"
        )


def _directory(name: str, args: list[_Word]) -> str | None:
    """Return the literal directory a ``cd``/``pushd`` enters, if knowable."""
    if name == "popd":
        return None
    operands = [a for a in args if a.text not in ("-L", "-P", "-e", "-@")]
    if len(operands) != 1 or operands[0].dynamic:
        return None
    text = operands[0].text
    if not text or text[0] in "-+" or (text.startswith("~") and not operands[0].quoted):
        return None
    return text


def _check_inert(name: str, args: list[_Word], scan: CommandScan) -> None:
    """Flag the few inert-builtin forms that assign to a computed variable name."""
    texts = [a.text for a in args]
    if name == "printf" and "-v" in texts:
        scan.opaque.append("'printf -v' assigns to a computed variable")
    elif name in ("test", "[") and ("-v" in texts or "-R" in texts):
        scan.opaque.append(f"'{name} -v' evaluates a variable subscript")
    elif name == "set" and any(
        a == "allexport" or (a.startswith("-") and not a.startswith("--") and "a" in a)
        for a in texts
    ):
        scan.opaque.append("'set -a' exports every later assignment")
    elif name == "export":
        for arg in args:
            variable = arg.text.split("=", 1)[0]
            if arg.text.startswith("-"):
                continue
            if not _NAME.fullmatch(variable):
                scan.opaque.append("'export' of a computed variable name")
            scan.assignments.append(variable)


def _env_split_string(text: str) -> bool:
    """Whether an ``env`` argument is ``-S``/``--split-string`` (or bundles it)."""
    if text.startswith("--"):
        name = text[2:].split("=", 1)[0]
        return len(name) >= 1 and "split-string".startswith(name)
    return text.startswith("-") and "S" in text


def _skip_options(
    words: list[_Word],
    valued: frozenset[str],
    positionals: int,
    scan: CommandScan,
    *,
    assignments: bool = False,
) -> list[_Word]:
    """Drop a wrapper's options, its fixed positionals, and ``env`` assignments."""
    i = 0
    while i < len(words):
        text = words[i].text
        if text == "--":
            i += 1
            break
        if assignments and _env_split_string(text):
            # ``env -S "cmd args"`` runs a program named inside an option value.
            scan.opaque.append("'env -S' runs a command the policy cannot inspect")
            return []
        if text in valued:
            i += 2
        elif text.startswith("-") and len(text) > 1:
            i += 1
        elif assignments and "=" in text:
            scan.assignments.append(text.split("=", 1)[0])
            i += 1
        else:
            break
    return words[i + positionals :]


# --- running a command string with persistent cwd/env ----------------------

_IGNORED_ENV = frozenset({"PWD", "OLDPWD", "SHLVL", "_", "BASH_TRAPSIG"})
"""Variables the shell itself maintains, never carried between calls."""

_WRAPPER_PREFIX = "__nexus_"
"""Prefix of the driver's own variables, which ``set -a`` could export."""

_MAX_ENV_VALUE = 32_768
"""Largest variable value carried between calls, to keep argv bounded."""

_WRAPPER = """exec 2>&1
__nexus_cmd=$1
__nexus_dir=$2
shift 2
for __nexus_kv in "$@"; do
  case $__nexus_kv in
    *=*) export "$__nexus_kv" ;;
    *) unset "$__nexus_kv" ;;
  esac
done
unset __nexus_kv
set --
printf '%s\\n' '{tag}:ENV0'
{env} -0
printf '\\n%s\\n' '{tag}:BEGIN'
cd -- "$__nexus_dir" || printf '%s\\n' "note: could not enter $__nexus_dir"
unset __nexus_dir
trap '__nexus_rc=$?; printf "\\n%s\\n" "{tag}:STATE"; pwd -P 2>/dev/null; \
printf "\\n%s\\n" "{tag}:ENV"; {env} -0; exit "$__nexus_rc"' EXIT
eval "$__nexus_cmd"
"""
"""POSIX ``sh`` driver: restore state, run the command, then report the new state.

The command is passed as an argument and run with ``eval`` so a syntax error in
it is reported like any other failure (and the state trailer still runs). The
trailer marks where ``pwd`` ends so a failed ``pwd`` (a deleted cwd) is not
mistaken for the env dump, and ``{env}`` is an absolute path so a command that
breaks ``PATH`` cannot blank the snapshot.
"""


def _env_program() -> str:
    """Return the ``env`` invocation for the driver, immune to ``PATH`` changes."""
    found = shutil.which("env")
    # Embedded in a single-quoted trap, so only plain path characters are safe.
    if found and _PLAIN_PATH.fullmatch(found):
        return found
    return "command -p env"


def default_interpreter() -> str:
    """Return the shell used for command strings: ``bash`` if found, else ``sh``."""
    return shutil.which("bash") or "/bin/sh"


@dataclass(frozen=True)
class ShellInvocation:
    """One command string wrapped so its final cwd and env can be observed.

    Build it with :meth:`build`, execute :attr:`argv` (directly, or through a
    sandbox), then turn the raw capture into a :class:`ShellResult` with
    :meth:`parse`.

    Attributes:
        argv: The argument vector that runs the command via the interpreter.
        tag: The random marker delimiting the state report in the output.
    """

    argv: list[str]
    tag: str

    @classmethod
    def build(
        cls,
        command: str,
        *,
        cwd: str,
        env: Mapping[str, str | None] | None = None,
        interpreter: str | None = None,
    ) -> ShellInvocation:
        """Wrap ``command`` to start in ``cwd`` with ``env`` applied.

        Args:
            command: The shell command string to run.
            cwd: The directory to start in.
            env: Variables to export (or unset, when ``None``) first.
            interpreter: The shell to use (default :func:`default_interpreter`).

        Returns:
            The invocation, ready to execute.
        """
        tag = f"__NEXUS_{secrets.token_hex(8)}"
        pairs = [
            name if value is None else f"{name}={value}"
            for name, value in (env or {}).items()
            if _NAME.fullmatch(name)
        ]
        script = _WRAPPER.replace("{tag}", tag).replace("{env}", _env_program())
        shell = interpreter or default_interpreter()
        return cls(
            argv=[shell, "-c", script, "nexus-shell", command, cwd, *pairs], tag=tag
        )

    def parse(
        self, raw: SandboxResult, *, confine: Callable[[str], Path], root: Path
    ) -> ShellResult:
        """Split the raw capture into the command's output and its new state.

        Args:
            raw: The captured result of running :attr:`argv`.
            confine: Resolves a path under the root, raising
                :class:`SandboxViolation` if it escapes.
            root: Where the cwd is reset to if the command left the root.

        Returns:
            The command's combined output plus its final cwd and env changes.
        """
        out = raw.stdout
        stderr = raw.stderr
        env0 = f"{self.tag}:ENV0\n"
        begin = f"\n{self.tag}:BEGIN\n"
        start = out.find(env0)
        split = out.find(begin, start) if start != -1 else -1
        if split == -1:  # the driver never got going; report the raw output
            return ShellResult(raw.returncode, out, stderr, raw.timed_out)
        before = _parse_env(out[start + len(env0) : split])
        body = out[split + len(begin) :]
        marker = f"\n{self.tag}:STATE\n"
        state = body.rfind(marker)
        if state == -1:
            return ShellResult(
                raw.returncode, out[:start] + body, stderr, raw.timed_out
            )
        trailer = body[state + len(marker) :]
        body = out[:start] + body[:state]
        cwd_line, found, env_blob = trailer.partition(f"\n{self.tag}:ENV\n")
        if not found:  # the trailer was cut short; trust none of it
            cwd_line, env_blob = "", ""
        cwd_line = cwd_line.strip("\n")
        try:
            # An empty or malformed line means ``pwd`` failed: the cwd is unknown.
            cwd = str(confine(cwd_line)) if cwd_line and "\0" not in cwd_line else None
        except ValueError:
            cwd = None
        except SandboxViolation:
            cwd = str(root)
            note = f"note: {cwd_line} is outside the workspace; cwd reset to {root}"
            stderr = f"{stderr}\n{note}".strip()
        return ShellResult(
            raw.returncode,
            body,
            stderr,
            raw.timed_out,
            cwd=cwd,
            env=_env_changes(before, _parse_env(env_blob)),
        )


def _parse_env(blob: str) -> dict[str, str]:
    """Parse ``env -0`` output into a mapping, skipping malformed entries."""
    env: dict[str, str] = {}
    for entry in blob.split("\0"):
        name, sep, value = entry.partition("=")
        if sep and _NAME.fullmatch(name):
            env[name] = value
    return env


def _env_changes(
    before: dict[str, str], after: dict[str, str]
) -> dict[str, str | None]:
    """Diff two environments into sets (a value) and unsets (``None``).

    An empty ``after`` means the snapshot failed rather than that the command
    unset everything, so it yields no changes.
    """
    if not after:
        return {}
    changes: dict[str, str | None] = {
        name: value
        for name, value in after.items()
        if name not in _IGNORED_ENV
        and not name.startswith(_WRAPPER_PREFIX)
        and before.get(name) != value
        and len(value) <= _MAX_ENV_VALUE
    }
    changes.update(
        {
            name: None
            for name in before
            if name not in after
            and name not in _IGNORED_ENV
            and not name.startswith(_WRAPPER_PREFIX)
        }
    )
    return changes
