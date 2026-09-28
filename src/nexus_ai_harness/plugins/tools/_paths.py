from __future__ import annotations

import os
from pathlib import Path

from nexus_ai_harness.protocols.context import Context
from nexus_ai_harness.protocols.sandbox import Sandbox, SandboxViolation


def confine(path: str, root: Path) -> Path:
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


def resolve(path: str, ctx: Context) -> Path:
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
    return confine(path, Path(os.getcwd()))


def root(ctx: Context) -> Path:
    """Return the confinement root: the sandbox root, or the cwd if none."""
    return resolve(".", ctx)


def display(path: Path, base: Path) -> str:
    """Render ``path`` relative to ``base`` (POSIX separators) when inside it."""
    try:
        return path.relative_to(base).as_posix() or "."
    except ValueError:
        return str(path)
