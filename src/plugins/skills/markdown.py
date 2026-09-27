from __future__ import annotations

import warnings
from pathlib import Path

from protocols.skill import Skill


def _read(path: Path) -> str:
    # utf-8-sig transparently strips a leading BOM so it can't hide the fence.
    return path.read_text(encoding="utf-8-sig")


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split ``---``-delimited frontmatter from the markdown body.

    Parses only flat ``key: value`` pairs (the ``name`` and ``description``
    skills need) — enough to avoid a YAML dependency; nested/multi-line YAML is
    not supported and the body must not begin with a ``---`` line. A file with
    no frontmatter yields an empty mapping and the whole text as the body.
    """
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines(keepends=True)
    # Skip the opening fence and scan for its closer.
    for i in range(1, len(lines)):
        if lines[i].rstrip() == "---":
            meta: dict[str, str] = {}
            for raw in lines[1:i]:
                key, sep, value = raw.partition(":")
                if sep and key.strip():
                    meta[key.strip()] = value.strip().strip("'\"")
            return meta, "".join(lines[i + 1 :]).lstrip("\n")
    return {}, text  # unterminated fence → treat as no frontmatter


class MarkdownSkill(Skill):
    """A :class:`~protocols.skill.Skill` backed by a ``SKILL.md`` file.

    Frontmatter (``name``, ``description``) is read eagerly at construction so
    the catalog is cheap to build; the instruction body is read lazily on
    :meth:`instructions`, so a large skill costs nothing until it is invoked.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        meta, _ = _parse_frontmatter(_read(self.path))
        # Per-instance identity (Tool declares these as ClassVars, but a file-backed
        # skill carries its own); fall back to the folder name then the file stem.
        self.name = meta.get("name") or self.path.parent.name or self.path.stem  # type: ignore[misc]
        self.description = meta.get("description", "")  # type: ignore[misc]

    def instructions(self) -> str:
        """Read and return the markdown body (frontmatter stripped)."""
        _, body = _parse_frontmatter(_read(self.path))
        return body.strip()


def load_skills(directory: str | Path) -> list[MarkdownSkill]:
    """Load every ``SKILL.md`` under ``directory`` (recursively), sorted by path.

    Matches the Claude Code layout where each skill is a folder containing a
    ``SKILL.md``. Returns an empty list if the directory does not exist. A file
    that can't be read (e.g. non-UTF-8 or permission-denied) is skipped with a
    warning so one bad skill doesn't sink the rest.
    """
    root = Path(directory).expanduser()  # honor ~ so a "~/skills" dir resolves
    if not root.is_dir():
        return []
    skills: list[MarkdownSkill] = []
    for path in sorted(root.rglob("SKILL.md")):
        try:
            skills.append(MarkdownSkill(path))
        except (OSError, UnicodeDecodeError) as exc:
            warnings.warn(f"skipping unreadable skill {path}: {exc}", stacklevel=2)
    return skills
