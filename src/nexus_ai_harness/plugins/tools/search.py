from __future__ import annotations

import os
import re
import stat
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

from nexus_ai_harness.plugins.tools._paths import display, resolve, root
from nexus_ai_harness.plugins.tools._text import split_lines
from nexus_ai_harness.protocols.context import Context
from nexus_ai_harness.protocols.sandbox import SandboxViolation
from nexus_ai_harness.protocols.tool import Tool

SKIPPED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        ".eggs",
    }
)
"""Directories ``grep`` and ``glob`` skip unless the pattern names them."""

_DEFAULT_RESULTS = 100
"""Matches (or paths) returned when the model does not pass ``max_results``."""

_MAX_RESULTS = 1000
"""Hard cap on ``max_results``."""

_MAX_CONTEXT = 10
"""Hard cap on ``grep`` context lines."""

_MAX_LINE_CHARS = 500
"""Longest matched line ``grep`` shows before clipping it."""

_MAX_OUTPUT = 50_000
"""Cap on one search result, cut between whole lines."""

_MAX_FILE_BYTES = 5_000_000
"""Files larger than this are skipped by ``grep``, to keep a search bounded."""

_BINARY_SNIFF = 8192
"""Leading bytes checked for a NUL to decide a file is binary."""

_MAX_BRACE_EXPANSIONS = 64
"""Cap on alternatives a ``{a,b}`` glob may expand to."""


def _count(value: object, default: int, cap: int, *, minimum: int = 1) -> int:
    """Coerce a model-supplied count into ``[minimum, cap]``, else ``default``."""
    # bool is a subclass of int, so exclude it explicitly.
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return min(max(value, minimum), cap)


def _expand_braces(pattern: str) -> list[str]:
    """Expand the first top-level ``{a,b}`` group in ``pattern``, recursively."""
    depth = 0
    start = -1
    for i, c in enumerate(pattern):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}" and depth:
            depth -= 1
            if depth == 0:
                inner = pattern[start + 1 : i]
                options = _split_top_level(inner)
                if len(options) < 2:
                    continue
                head, tail = pattern[:start], pattern[i + 1 :]
                out: list[str] = []
                for option in options:
                    out.extend(_expand_braces(head + option + tail))
                    if len(out) > _MAX_BRACE_EXPANSIONS:
                        return out[:_MAX_BRACE_EXPANSIONS]
                return out
    return [pattern]


def _split_top_level(text: str) -> list[str]:
    """Split ``text`` on commas that are not inside a nested brace group."""
    parts, depth, current = [], 0, ""
    for c in text:
        if c == "," and depth == 0:
            parts.append(current)
            current = ""
            continue
        depth += {"{": 1, "}": -1}.get(c, 0)
        current += c
    parts.append(current)
    return parts


def _translate(pattern: str) -> str:
    """Translate one brace-free glob into a regex matching a POSIX relative path."""
    out: list[str] = []
    i, n = 0, len(pattern)
    while i < n:
        c = pattern[i]
        if pattern.startswith("**", i):
            at_segment = i == 0 or pattern[i - 1] == "/"
            if at_segment and pattern.startswith("**/", i):
                out.append("(?:[^/]+/)*")
                i += 3
            else:
                out.append(".*")
                i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        elif c == "[":
            end = pattern.find("]", i + 2 if pattern.startswith("[!", i) else i + 1)
            if end == -1:
                out.append(re.escape(c))
                i += 1
                continue
            body = pattern[i + 1 : end].replace("\\", "\\\\")
            if body.startswith("!"):
                body = "^" + body[1:]
            out.append(f"[{body}]")
            i = end + 1
        else:
            out.append(re.escape(c))
            i += 1
    return "".join(out)


def _normalize_glob(pattern: str) -> str:
    """Strip a leading ``/`` or ``./`` so the glob is relative to the base."""
    pattern = pattern.strip().lstrip("/")
    while pattern.startswith("./"):
        pattern = pattern[2:].lstrip("/")
    return pattern


def compile_glob(pattern: str) -> re.Pattern[str]:
    """Compile a glob into a regex over POSIX paths relative to the search base.

    ``*`` and ``?`` stay within one path segment, ``**`` spans segments (so
    ``**/*.py`` matches Python files at any depth, the base included), ``[...]``
    is a character class, and ``{a,b}`` expands to alternatives.

    Args:
        pattern: The glob, e.g. ``src/**/*.{py,pyi}``.

    Returns:
        A compiled regex to ``fullmatch`` against relative paths.

    Raises:
        re.error: If a ``[...]`` class is invalid, e.g. ``[z-a]``.
    """
    alternatives = _expand_braces(_normalize_glob(pattern) or "*")
    return re.compile("|".join(f"(?:{_translate(a)})" for a in alternatives))


def _walk(base: Path, confine_root: Path, pattern: str) -> Iterator[Path]:
    """Yield regular files under ``base`` in full path sort order, confined to root.

    Skips the :data:`SKIPPED_DIRS` (unless ``pattern`` names one as a path
    segment), does not descend into symlinked directories, and drops symlinked
    files whose target lies outside ``confine_root``. Files and directories at
    each level sort together the way their full paths would (a directory sorts
    as if its name were followed by ``/``), so the stream is in true path
    order and a caller may stop as soon as it has enough matches.
    """
    named = re.split(r"[/{},]", pattern)

    def visit(current: Path) -> Iterator[Path]:
        try:
            entries = sorted(os.scandir(current), key=_entry_sort_key)
        except OSError:
            return
        for entry in entries:
            try:
                is_symlink = entry.is_symlink()
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                continue
            path = Path(entry.path)
            if is_symlink:
                try:
                    target = path.resolve()
                except OSError:
                    continue
                if target.is_relative_to(confine_root) and target.is_file():
                    yield path
            elif is_dir:
                if entry.name in SKIPPED_DIRS and entry.name not in named:
                    continue
                yield from visit(path)
            elif _is_regular(path):
                yield path
            # else: a FIFO or other non-regular file would block the read forever

    yield from visit(base)


def _entry_sort_key(entry: os.DirEntry[str]) -> str:
    """Sort key matching full-path order: a directory sorts as ``name + "/"``."""
    try:
        return f"{entry.name}/" if entry.is_dir(follow_symlinks=False) else entry.name
    except OSError:
        return entry.name


def _is_regular(path: Path) -> bool:
    """Whether ``path`` (not followed if a symlink) is a regular file."""
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


def _clip(lines: list[str], footer: str | None) -> str:
    """Join result lines under :data:`_MAX_OUTPUT`, cutting between lines."""
    kept: list[str] = []
    size = 0
    for line in lines:
        if kept and size + len(line) + 1 > _MAX_OUTPUT:
            footer = (
                f"... (output truncated after {len(kept)} lines; narrow the search)"
            )
            break
        kept.append(line)
        size += len(line) + 1
    if footer:
        kept.append(footer)
    return "\n".join(kept)


class Grep(Tool):
    """Search file contents by regular expression, confined to the sandbox root.

    Pure Python (no external ``grep``/``rg``), read-only, and bounded: binary
    files, files over a few megabytes, and junk directories such as ``.git`` and
    ``node_modules`` are skipped, and results stop at ``max_results`` matches.
    """

    name = "grep"
    description = (
        "Search file contents with a regular expression (Python `re` syntax) "
        "and return matching lines as `path:line:text`, paths relative to the "
        "workspace root. Searches `path` (a file or directory, default the "
        "root) recursively, optionally only files matching `glob` (e.g. '*.py' "
        "or 'src/**/*.ts'). Skips binary files and directories like .git, "
        "node_modules, __pycache__, and .venv. Read-only."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regular expression to search for.",
            },
            "path": {
                "type": "string",
                "description": "File or directory to search (default '.').",
            },
            "glob": {
                "type": "string",
                "description": (
                    "Only search files matching this glob; a pattern without "
                    "'/' matches the file name at any depth."
                ),
            },
            "case_insensitive": {
                "type": "boolean",
                "description": "Match case-insensitively (default false).",
            },
            "context": {
                "type": "integer",
                "description": "Lines of context around each match (default 0).",
            },
            "files_only": {
                "type": "boolean",
                "description": "List matching file paths instead of lines.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum matches (or files) to return (default 100).",
            },
        },
        "required": ["pattern"],
    }

    def run(self, arguments: dict, ctx: Context) -> str:
        pattern = str(arguments.get("pattern", ""))
        if not pattern:
            return "error: pattern is required"
        flags = re.IGNORECASE if arguments.get("case_insensitive") is True else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as exc:
            return f"error: invalid regular expression: {exc}"
        name_glob = arguments.get("glob")
        name_glob = _normalize_glob(str(name_glob)) if name_glob else None
        try:
            matcher = compile_glob(name_glob) if name_glob else None
        except re.error as exc:
            return f"error: invalid glob: {exc}"
        max_results = _count(
            arguments.get("max_results"), _DEFAULT_RESULTS, _MAX_RESULTS
        )
        context = _count(arguments.get("context"), 0, _MAX_CONTEXT, minimum=0)
        files_only = arguments.get("files_only") is True
        try:
            base = root(ctx)
            target = resolve(str(arguments.get("path") or "."), ctx)
            files = self._files(target, base, name_glob, matcher)
            return self._search(files, base, regex, context, files_only, max_results)
        except SandboxViolation as exc:
            return f"error: {exc}"
        except (OSError, ValueError) as exc:  # ValueError e.g. embedded NUL
            return f"error: {exc}"

    @staticmethod
    def _files(
        target: Path,
        base: Path,
        name_glob: str | None,
        matcher: re.Pattern[str] | None,
    ) -> Iterator[Path]:
        """The files to search: ``target`` itself, or those under it."""
        if not target.exists():
            raise FileNotFoundError(
                f"no such file or directory: {display(target, base)}"
            )
        if target.is_file():
            return iter([target])
        if not target.is_dir():
            raise OSError(f"not a regular file: {display(target, base)}")
        by_name = name_glob is not None and "/" not in name_glob

        def wanted(path: Path) -> bool:
            if matcher is None:
                return True
            rel = path.name if by_name else path.relative_to(target).as_posix()
            return matcher.fullmatch(rel) is not None

        return (p for p in _walk(target, base, name_glob or "") if wanted(p))

    @staticmethod
    def _search(
        files: Iterator[Path],
        base: Path,
        regex: re.Pattern[str],
        context: int,
        files_only: bool,
        max_results: int,
    ) -> str:
        """Scan ``files`` and render the first ``max_results`` hits."""
        out: list[str] = []
        hits = 0
        capped = False
        for path in files:
            lines = _text_lines(path)
            if lines is None:
                continue
            matched = [i for i, line in enumerate(lines) if regex.search(line)]
            if not matched:
                continue
            shown = display(path, base)
            if files_only:
                out.append(shown)
                hits += 1
            else:
                room = max_results - hits
                if len(matched) > room:
                    matched = matched[:room]
                    capped = True
                out.extend(_render(shown, lines, matched, context, bool(out)))
                hits += len(matched)
            if hits >= max_results:
                capped = True  # unknown whether more exist, so say so
                break
        if not out:
            return "no matches"
        footer = None
        if capped:
            noun = "files" if files_only else "matches"
            footer = (
                f"... (stopped at max_results={max_results} {noun}; there may be "
                "more: narrow the pattern, path, or glob, or raise max_results)"
            )
        return _clip(out, footer)


def _text_lines(path: Path) -> list[str] | None:
    """Return ``path``'s lines, or ``None`` if it is binary, huge, or unreadable."""
    try:
        if path.stat().st_size > _MAX_FILE_BYTES:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    if b"\0" in data[:_BINARY_SNIFF]:
        return None
    return split_lines(data.decode("utf-8", errors="replace"))


def _render(
    shown: str, lines: list[str], matched: list[int], context: int, separate: bool
) -> list[str]:
    """Render matches as ``path:n:text`` with ``path-n-text`` context lines."""
    out: list[str] = []
    hits = set(matched)
    last = -1
    for index in matched:
        low = max(index - context, last + 1)
        high = min(index + context, len(lines) - 1)
        if context and (out or separate) and low > last + 1:
            out.append("--")
        for n in range(low, high + 1):
            text = lines[n]
            if len(text) > _MAX_LINE_CHARS:
                text = text[:_MAX_LINE_CHARS] + "..."
            sep = ":" if n in hits else "-"
            out.append(f"{shown}{sep}{n + 1}{sep}{text}")
        last = high
    return out


class Glob(Tool):
    """Find files by glob pattern, confined to the sandbox root."""

    name = "glob"
    description = (
        "Find files whose path matches a glob pattern and return them sorted, "
        "relative to the workspace root. `*` and `?` match within one path "
        "segment, `**` matches any number of directories (e.g. '**/*.py' for "
        "every Python file, 'src/**/test_*.py'), and `{a,b}` gives "
        "alternatives. Patterns are relative to `path` (default the root). "
        "Skips directories like .git, node_modules, __pycache__, and .venv "
        "unless the pattern names them. Returns files only. Read-only."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern, e.g. '**/*.py'.",
            },
            "path": {
                "type": "string",
                "description": "Directory to search from (default '.').",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum paths to return (default 100).",
            },
        },
        "required": ["pattern"],
    }

    def run(self, arguments: dict, ctx: Context) -> str:
        pattern = str(arguments.get("pattern", "")).strip()
        if not pattern:
            return "error: pattern is required"
        max_results = _count(
            arguments.get("max_results"), _DEFAULT_RESULTS, _MAX_RESULTS
        )
        try:
            base = root(ctx)
            target = resolve(str(arguments.get("path") or "."), ctx)
            if not target.is_dir():
                return f"error: not a directory: {display(target, base)}"
            try:
                matcher = compile_glob(pattern)
            except re.error as exc:
                return f"error: invalid glob: {exc}"
            found: list[str] = []
            capped = False
            for p in _walk(target, base, pattern):
                if not matcher.fullmatch(p.relative_to(target).as_posix()):
                    continue
                found.append(display(p, base))
                if len(found) > max_results:
                    capped = True
                    break
        except SandboxViolation as exc:
            return f"error: {exc}"
        except (OSError, ValueError) as exc:  # ValueError e.g. embedded NUL
            return f"error: {exc}"
        if not found:
            return "no files match"
        footer = None
        if capped:
            found = found[:max_results]
            footer = (
                f"... (stopped at max_results={max_results} files; there may be "
                "more: narrow the pattern or raise max_results)"
            )
        return _clip(found, footer)
