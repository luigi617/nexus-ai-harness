from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar

from protocols.context import Context
from protocols.sandbox import Sandbox, SandboxViolation
from protocols.tool import Tool

_MAX_OUTPUT = 10_000
"""Cap on returned text so a large file can't blow up the model's context."""


def _truncate(text: str) -> str:
    """Cap ``text`` at ``_MAX_OUTPUT`` chars with a clear marker when clipped."""
    if len(text) <= _MAX_OUTPUT:
        return text
    return text[:_MAX_OUTPUT] + "\n... (truncated)"


def _confine(path: str, root: Path) -> Path:
    """Resolve ``path`` under ``root``, rejecting anything that escapes it.

    Mirrors the confinement discipline of ``FileMemoryStore._path``: the
    resolved real path (symlinks included) must live under ``root``. Absolute
    paths and ``..`` traversal that land outside the root are rejected.

    Args:
        path: A model- or tool-supplied path, absolute or relative to ``root``.
        root: The directory the result must stay within.

    Returns:
        The resolved, confined absolute path.

    Raises:
        SandboxViolation: If the resolved path escapes ``root``.
    """
    base = root.resolve()
    candidate = Path(path)
    combined = candidate if candidate.is_absolute() else base / candidate
    resolved = combined.resolve()
    if not resolved.is_relative_to(base):  # is_relative_to is True when equal to base
        raise SandboxViolation(f"path escapes root: {path!r}")
    return resolved


def _resolve(path: str, ctx: Context) -> Path:
    """Resolve ``path`` via the active :class:`Sandbox`, or the cwd if none.

    Args:
        path: The path to confine.
        ctx: The run context; a ``Sandbox`` is used when registered.

    Returns:
        The confined absolute path.

    Raises:
        SandboxViolation: If ``path`` escapes the confinement root.
    """
    sandbox: Sandbox | None = ctx.get(Sandbox)
    if sandbox is not None:
        return sandbox.resolve_path(path)
    return _confine(path, Path(os.getcwd()))


class ReadFile(Tool):
    """Read the contents of a text file, confined to the sandbox root."""

    name = "read_file"
    description = (
        "Read a UTF-8 text file and return its contents. The path is confined "
        "to the workspace root; very large files are truncated."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read, relative to the root.",
            }
        },
        "required": ["path"],
    }

    def run(self, arguments: dict, ctx: Context) -> str:
        path = str(arguments.get("path", ""))
        try:
            resolved = _resolve(path, ctx)
            # Bounded read so a huge file can't exhaust memory, and lenient decode
            # so a non-UTF-8 file yields replacement text instead of raising.
            with resolved.open("rb") as handle:
                raw = handle.read(_MAX_OUTPUT + 1)
            # Gate the marker on bytes read, not decoded chars: multibyte text
            # decodes shorter, so a char check would drop the tail unmarked.
            if len(raw) > _MAX_OUTPUT:
                text = raw[:_MAX_OUTPUT].decode("utf-8", errors="replace")
                return text + "\n... (truncated)"
            return raw.decode("utf-8", errors="replace")
        except SandboxViolation as exc:
            return f"error: {exc}"
        except OSError as exc:
            return f"error: {exc}"


class WriteFile(Tool):
    """Write text to a file, creating parent directories within the root."""

    name = "write_file"
    description = (
        "Write UTF-8 text to a file, overwriting it if it exists. Parent "
        "directories are created as needed. The path is confined to the "
        "workspace root."
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
            resolved = _resolve(path, ctx)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            data = content.encode("utf-8")
            resolved.write_bytes(data)
        except SandboxViolation as exc:
            return f"error: {exc}"
        except OSError as exc:
            return f"error: {exc}"
        return f"wrote {len(data)} bytes to {resolved}"


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
            resolved = _resolve(path, ctx)
            entries = sorted(resolved.iterdir(), key=lambda p: p.name)
        except SandboxViolation as exc:
            return f"error: {exc}"
        except OSError as exc:
            return f"error: {exc}"
        if not entries:
            return "(empty)"
        names = [f"{e.name}/" if e.is_dir() else e.name for e in entries]
        return _truncate("\n".join(names))
