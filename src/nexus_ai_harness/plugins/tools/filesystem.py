from __future__ import annotations

import contextlib
import math
import os
import stat
import tempfile
import threading
from pathlib import Path
from typing import ClassVar

from nexus_ai_harness.plugins.tools._paths import confine, display, resolve, root
from nexus_ai_harness.plugins.tools._text import count_breaks, iter_lines, split_lines
from nexus_ai_harness.protocols.context import Context
from nexus_ai_harness.protocols.sandbox import SandboxViolation
from nexus_ai_harness.protocols.tool import Tool

# Kept under their historical private names for callers that imported them.
_confine = confine
_resolve = resolve

_DEFAULT_LINE_LIMIT = 2000
"""Lines ``read_file`` returns when the model does not pass ``limit``."""

_MAX_LINE_CHARS = 2000
"""Longest single line ``read_file`` shows before clipping it."""

_MAX_READ_CHARS = 50_000
"""Cap on one ``read_file`` result, so a large range can't blow up context."""

_MAX_LIST_CHARS = 10_000
"""Cap on one ``list_dir`` result."""

_MAX_COUNT_BYTES = 20_000_000
"""Largest file ``read_file`` scans to the end to report its line count.

Larger files stop reading once the requested window is filled.
"""

_SNIPPET_CONTEXT = 3
"""Lines of surrounding context ``edit_file`` echoes around a change."""

_WRITE_LOCKS = tuple(threading.Lock() for _ in range(64))
"""Striped locks serializing writes to one path within this process.

The loop runs a response's tool calls concurrently, so two ``edit_file`` calls on
one file would otherwise both read the original and the later replace would
silently drop the other's edit. A fixed stripe keeps memory bounded.
"""


def _write_lock(path: Path) -> threading.Lock:
    """Return the lock guarding read-modify-write cycles on ``path``."""
    return _WRITE_LOCKS[hash(str(path)) % len(_WRITE_LOCKS)]


def _require_regular(path: Path) -> None:
    """Refuse FIFOs, devices, and sockets, which could block or never end."""
    if path.exists() and not path.is_file() and not path.is_dir():
        raise OSError(f"not a regular file: {path}")


def _positive_int(value: object, default: int) -> int:
    """Coerce a model-supplied count to a positive int, else ``default``."""
    # bool is a subclass of int, so exclude it explicitly.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    if not math.isfinite(value):
        return default
    number = int(value)
    return number if number >= 1 else default


def _number(lineno: int, text: str) -> str:
    """Format one line the way ``cat -n`` does: right-aligned number, tab, text."""
    return f"{lineno:>6}\t{text}"


def _join_capped(lines: list[str], limit: int, noun: str) -> str:
    """Join ``lines`` up to ``limit`` chars, cutting only between whole lines."""
    kept: list[str] = []
    size = 0
    for line in lines:
        if kept and size + len(line) + 1 > limit:
            break
        kept.append(line)
        size += len(line) + 1
    omitted = len(lines) - len(kept)
    if omitted:
        kept.append(f"... ({omitted} more {noun} not shown)")
    return "\n".join(kept)


class ReadFile(Tool):
    """Read a range of lines from a text file, confined to the sandbox root."""

    name = "read_file"
    description = (
        "Read a UTF-8 text file and return its lines numbered like `cat -n` "
        "(line number, a tab, then the line). The numbers are not part of the "
        "file. Returns up to 2000 lines starting at `offset`; for a longer file "
        "the result says which `offset` to pass to read the next range. The "
        "path is confined to the workspace root."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read, relative to the root.",
            },
            "offset": {
                "type": "integer",
                "description": "1-based line number to start reading at (default 1).",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to return (default 2000).",
            },
            "line_numbers": {
                "type": "boolean",
                "description": "Prefix each line with its number (default true).",
            },
        },
        "required": ["path"],
    }

    def run(self, arguments: dict, ctx: Context) -> str:
        path = str(arguments.get("path", ""))
        offset = _positive_int(arguments.get("offset"), 1)
        limit = _positive_int(arguments.get("limit"), _DEFAULT_LINE_LIMIT)
        numbered = arguments.get("line_numbers") is not False
        try:
            resolved = resolve(path, ctx)
            return self._read(resolved, offset, limit, numbered)
        except SandboxViolation as exc:
            return f"error: {exc}"
        except OSError as exc:
            return f"error: {exc}"

    @staticmethod
    def _read(resolved: Path, offset: int, limit: int, numbered: bool) -> str:
        """Stream the file, keeping only the requested window of lines."""
        _require_regular(resolved)
        # Counting every line of a huge file just for the footer costs a full
        # scan, so past this size the read stops once the window is filled.
        count_all = resolved.stat().st_size <= _MAX_COUNT_BYTES
        shown: list[str] = []
        size = 0
        total = 0
        last = offset - 1
        clipped = False
        stopped = False
        # Streamed in chunks and leniently decoded: neither a huge file nor one
        # huge line loads whole, and a bad byte becomes a replacement char.
        with resolved.open(encoding="utf-8", errors="replace", newline="") as f:
            for total, (text, cut) in enumerate(
                iter_lines(f, _MAX_LINE_CHARS), start=1
            ):
                if total < offset:
                    continue
                if clipped or total >= offset + limit:
                    if not count_all:
                        stopped = True
                        break
                    continue
                if cut:
                    text += "... (line truncated)"
                line = _number(total, text) if numbered else text
                if shown and size + len(line) + 1 > _MAX_READ_CHARS:
                    clipped = True
                    continue
                shown.append(line)
                size += len(line) + 1
                last = total
        if total == 0:
            return "(empty file)"
        if offset > total:
            return f"error: offset {offset} is past the end of the file ({total} lines)"
        body = "\n".join(shown)
        if last < total:
            extent = "; the file has more" if stopped else f" of {total}"
            body += (
                f"\n... (showing lines {offset}-{last}{extent}; call read_file "
                f"with offset={last + 1} to read more)"
            )
        return body


class WriteFile(Tool):
    """Write text to a file, creating parent directories within the root."""

    name = "write_file"
    description = (
        "Write UTF-8 text to a file, overwriting it if it exists. Parent "
        "directories are created as needed. The path is confined to the "
        "workspace root. To change part of an existing file, prefer edit_file."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to write, relative to the root.",
            },
            "content": {
                "type": "string",
                "description": "The text to write to the file.",
            },
        },
        "required": ["path", "content"],
    }

    def run(self, arguments: dict, ctx: Context) -> str:
        path = str(arguments.get("path", ""))
        content = str(arguments.get("content", ""))
        try:
            resolved = resolve(path, ctx)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            data = content.encode("utf-8")
            with _write_lock(resolved):
                resolved.write_bytes(data)
        except SandboxViolation as exc:
            return f"error: {exc}"
        except OSError as exc:
            return f"error: {exc}"
        return f"wrote {len(data)} bytes to {resolved}"


class _EditError(Exception):
    """A requested edit that cannot be applied; its message goes to the model."""


class EditFile(Tool):
    """Replace exact text in an existing file, confined to the sandbox root.

    Each edit replaces ``old_string`` with ``new_string``. ``old_string`` must
    occur exactly once unless ``replace_all`` is set, so an edit never lands
    somewhere the model did not intend. Several edits can be passed as
    ``edits``; they apply in order to the running result and are all-or-nothing:
    if any edit fails, the file is left untouched. The file is replaced
    atomically, keeping its permissions.
    """

    name = "edit_file"
    description = (
        "Edit an existing UTF-8 text file by exact string replacement. "
        "`old_string` must match the file exactly (whitespace and indentation "
        "included, without read_file's line-number prefixes) and must be unique "
        "in the file unless `replace_all` is true; include surrounding lines to "
        "make it unique. Pass `edits` (a list of {old_string, new_string, "
        "replace_all}) to apply several edits in order, atomically: if any edit "
        "fails, nothing is written. The path is confined to the workspace root."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to edit, relative to the root.",
            },
            "old_string": {
                "type": "string",
                "description": "The exact text to replace.",
            },
            "new_string": {
                "type": "string",
                "description": "The text to replace it with.",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace every occurrence (default false).",
            },
            "edits": {
                "type": "array",
                "description": (
                    "Several edits to apply in order, instead of old_string/new_string."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"},
                        "replace_all": {"type": "boolean"},
                    },
                    "required": ["old_string", "new_string"],
                },
            },
        },
        "required": ["path"],
    }

    def run(self, arguments: dict, ctx: Context) -> str:
        path = str(arguments.get("path", ""))
        try:
            edits = self._edits(arguments)
            resolved = resolve(path, ctx)
            _require_regular(resolved)
            # Held across read and replace so concurrent edits compose.
            with _write_lock(resolved):
                original = resolved.read_bytes()
                try:
                    text = original.decode("utf-8")
                except UnicodeDecodeError:
                    raise _EditError(f"{path} is not valid UTF-8 text") from None
                updated, count, first = self._apply(text, edits, path)
                _atomic_write(resolved, updated.encode("utf-8"))
        except _EditError as exc:
            return f"error: {exc}"
        except SandboxViolation as exc:
            return f"error: {exc}"
        except OSError as exc:
            return f"error: {exc}"
        shown = display(resolved, root(ctx))
        noun = "replacement" if count == 1 else "replacements"
        if len(edits) > 1:
            return f"edited {shown}: {len(edits)} edits, {count} {noun}"
        snippet = _snippet(updated, first, edits[0][1])
        return f"edited {shown}: {count} {noun}\n{snippet}"

    @staticmethod
    def _edits(arguments: dict) -> list[tuple[str, str, bool]]:
        """Normalize the single-edit or ``edits`` form into a list of edits."""
        raw = arguments.get("edits")
        if raw is not None:
            if "old_string" in arguments or "new_string" in arguments:
                raise _EditError("pass either old_string/new_string or edits, not both")
            if not isinstance(raw, list) or not raw:
                raise _EditError("edits must be a non-empty list")
            items = raw
        else:
            items = [arguments]
        edits: list[tuple[str, str, bool]] = []
        for index, item in enumerate(items, start=1):
            where = f"edit {index}: " if raw is not None else ""
            if not isinstance(item, dict):
                raise _EditError(f"{where}each edit must be an object")
            old, new = item.get("old_string"), item.get("new_string")
            if not isinstance(old, str) or not isinstance(new, str):
                raise _EditError(f"{where}old_string and new_string are required")
            edits.append((old, new, item.get("replace_all") is True))
        return edits

    @staticmethod
    def _apply(
        text: str, edits: list[tuple[str, str, bool]], path: str
    ) -> tuple[str, int, int]:
        """Apply ``edits`` in order; return the text, replacements, first offset."""
        crlf = "\r\n" in text
        total = 0
        first = -1
        for index, (old, new, replace_all) in enumerate(edits, start=1):
            where = f"edit {index}: " if len(edits) > 1 else ""
            if not old:
                raise _EditError(
                    f"{where}old_string must not be empty "
                    "(use write_file to create or overwrite a file)"
                )
            if old == new:
                raise _EditError(f"{where}old_string and new_string are identical")
            count = text.count(old)
            if count == 0 and crlf and "\n" in old:
                # The model usually writes \n; match it against a CRLF file's lines.
                old, new = _to_crlf(old), _to_crlf(new)
                count = text.count(old)
            if count == 0:
                raise _EditError(
                    f"{where}old_string not found in {path}; it must match the "
                    "file exactly, including whitespace and indentation"
                )
            if count > 1 and not replace_all:
                raise _EditError(
                    f"{where}old_string occurs {count} times in {path}; add "
                    "surrounding context to make it unique, or set replace_all"
                )
            if first < 0:
                first = text.find(old)
            text = text.replace(old, new) if replace_all else text.replace(old, new, 1)
            total += count if replace_all else 1
        return text, total, first


def _to_crlf(text: str) -> str:
    """Convert ``text``'s line endings to CRLF."""
    return text.replace("\r\n", "\n").replace("\n", "\r\n")


def _snippet(text: str, offset: int, new: str) -> str:
    """Show the numbered lines around a change so the model can check it."""
    lines = split_lines(text)
    start = count_breaks(text[: max(offset, 0)])
    end = start + count_breaks(new)
    low = max(start - _SNIPPET_CONTEXT, 0)
    high = min(end + _SNIPPET_CONTEXT + 1, len(lines))
    numbered = [_number(n + 1, lines[n][:_MAX_LINE_CHARS]) for n in range(low, high)]
    return _join_capped(numbered, _MAX_LIST_CHARS, "lines")


def _atomic_write(path: Path, data: bytes) -> None:
    """Replace ``path`` with ``data`` atomically, preserving its permissions."""
    mode = stat.S_IMODE(path.stat().st_mode)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


class ListDir(Tool):
    """List the entries of a directory, confined to the sandbox root."""

    name = "list_dir"
    description = (
        "List the entries of a directory (defaults to the root). Directories "
        "are marked with a trailing '/'. The path is confined to the workspace "
        "root."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory to list, relative to the root (default '.').",
            }
        },
    }

    def run(self, arguments: dict, ctx: Context) -> str:
        path = str(arguments.get("path") or ".")
        try:
            resolved = resolve(path, ctx)
            entries = sorted(resolved.iterdir(), key=lambda p: p.name)
        except SandboxViolation as exc:
            return f"error: {exc}"
        except OSError as exc:
            return f"error: {exc}"
        if not entries:
            return "(empty)"
        names = [f"{e.name}/" if e.is_dir() else e.name for e in entries]
        return _join_capped(names, _MAX_LIST_CHARS, "entries")
