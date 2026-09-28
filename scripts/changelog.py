#!/usr/bin/env python3
"""Build CHANGELOG.md from per-PR JSON fragments in .changes/unreleased/.

Each PR adds ``.changes/unreleased/<id>.json`` containing
``{"id": <id>, "description": "..."}``. At release time the fragments are
rendered into a new CHANGELOG.md section and deleted.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CHANGELOG = REPO / "CHANGELOG.md"
UNRELEASED = REPO / ".changes" / "unreleased"
PR_URL = "https://github.com/luigi617/nexus-ai-harness/pull/{id}"
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+((a|b|rc)\d+)?$")


class ChangelogError(Exception):
    """A fragment or the changelog is invalid."""


@dataclass
class Fragment:
    """One changelog entry.

    Attributes:
        id: Unique id, normally the PR number; also the file's stem.
        description: One-line summary of the change.
    """

    id: str
    description: str

    def render(self) -> str:
        """Returns the entry as a markdown bullet, linking numeric ids to PRs."""
        if self.id.isdigit():
            ref = f"[#{self.id}]({PR_URL.format(id=self.id)})"
        else:
            ref = self.id
        return f"- {self.description} ({ref})"


def load_fragment(path: Path) -> Fragment:
    """Parses and validates one fragment file."""
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ChangelogError(f"{path.name}: invalid JSON ({e})") from e
    if not isinstance(data, dict) or set(data) != {"id", "description"}:
        raise ChangelogError(
            f"{path.name}: must be an object with exactly 'id' and 'description'"
        )
    frag_id, description = data["id"], data["description"]
    if isinstance(frag_id, bool) or not isinstance(frag_id, (int, str)):
        raise ChangelogError(f"{path.name}: 'id' must be a string or integer")
    frag_id = str(frag_id).strip()
    if frag_id != path.stem:
        raise ChangelogError(f"{path.name}: 'id' {frag_id!r} must match file name")
    if not isinstance(description, str) or not description.strip():
        raise ChangelogError(f"{path.name}: 'description' must be a non-empty string")
    return Fragment(id=frag_id, description=" ".join(description.split()))


def load_fragments(directory: Path = UNRELEASED) -> list[Fragment]:
    """Loads every fragment, ordered by id (numeric ids first, ascending)."""
    paths = sorted(directory.glob("*.json")) if directory.is_dir() else []
    fragments = [load_fragment(p) for p in paths]
    return sorted(
        fragments,
        key=lambda f: (0, int(f.id), "") if f.id.isdigit() else (1, 0, f.id),
    )


def render_section(version: str, date: str, fragments: list[Fragment]) -> str:
    """Returns the CHANGELOG.md section for a release."""
    lines = [f"## [{version}] - {date}", "", *(f.render() for f in fragments)]
    return "\n".join(lines) + "\n"


def insert_section(changelog: str, section: str) -> str:
    """Inserts a section above the newest release, or at the end if none."""
    match = re.search(r"^## \[", changelog, flags=re.M)
    if match is None:
        return changelog.rstrip("\n") + "\n\n" + section
    return changelog[: match.start()] + section + "\n" + changelog[match.start() :]


def extract_section(changelog: str, version: str) -> str:
    """Returns the body of a release's section, without its heading."""
    pattern = rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)"
    match = re.search(pattern, changelog, flags=re.M | re.S)
    return match.group(1).strip() if match else ""


def release(
    version: str,
    date: str,
    changelog_path: Path = CHANGELOG,
    directory: Path = UNRELEASED,
) -> list[Fragment]:
    """Moves all fragments into a new CHANGELOG.md section and deletes them."""
    if not VERSION_RE.match(version):
        raise ChangelogError(f"invalid version {version!r} (expected X.Y.Z)")
    changelog = changelog_path.read_text()
    if re.search(rf"^## \[{re.escape(version)}\]", changelog, flags=re.M):
        raise ChangelogError(f"CHANGELOG.md already has a section for {version}")
    fragments = load_fragments(directory)
    if not fragments:
        raise ChangelogError(f"no changelog fragments in {directory}")
    section = render_section(version, date, fragments)
    changelog_path.write_text(insert_section(changelog, section))
    for fragment in fragments:
        (directory / f"{fragment.id}.json").unlink()
    return fragments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="validate the unreleased fragments")
    rel = sub.add_parser("release", help="write a release section from fragments")
    rel.add_argument("version", help="release version, e.g. 0.2.0")
    rel.add_argument("--date", default=datetime.date.today().isoformat())
    notes = sub.add_parser("notes", help="print a release's CHANGELOG.md section")
    notes.add_argument("version")
    args = parser.parse_args(argv)

    try:
        if args.command == "check":
            print(f"{len(load_fragments())} valid changelog fragment(s)")
        elif args.command == "release":
            fragments = release(args.version, args.date)
            print(f"Added {len(fragments)} entries to CHANGELOG.md for {args.version}")
        else:
            body = extract_section(CHANGELOG.read_text(), args.version)
            if not body:
                raise ChangelogError(f"CHANGELOG.md has no entry for {args.version}")
            print(body)
    except ChangelogError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
