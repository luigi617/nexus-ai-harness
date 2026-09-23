from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from core.ids import new_id
from protocols.memory import MemoryItem, MemoryStore


@dataclass
class FileMemoryItem(MemoryItem):
    """A MemoryItem that also tracks the file it lives in."""

    id: str = field(default_factory=lambda: new_id("mem"))
    created_at: float = field(default_factory=time.time)
    path: Path | None = None


class FileMemoryStore(MemoryStore):
    """One markdown file per memory with a small frontmatter header.

    Format (``<dir>/<id>.md``)::

        ---
        id: mem_ab12
        created_at: 1700000000.0
        ---
        the fact text

    """

    def __init__(self, directory: str | Path = "~/.nexus-ai-harness/memory") -> None:
        self._dir = Path(directory).expanduser()
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, text: str, id: str | None = None) -> FileMemoryItem:
        created = time.time()
        if id is not None:
            existing = self.get(id)
            if existing is not None:
                created = existing.created_at
        item = FileMemoryItem(
            text=text.strip(),
            id=id or new_id("mem"),
            created_at=created,
        )
        item.path = self._path(item.id)
        item.path.write_text(self._serialize(item), encoding="utf-8")
        return item

    def get(self, id: str) -> FileMemoryItem | None:
        path = self._path(id)
        return self._parse(path) if path.exists() else None

    def search(self, query: str, limit: int = 5) -> list[FileMemoryItem]:
        terms = [t for t in query.lower().split() if t]
        scored: list[tuple[int, float, FileMemoryItem]] = []
        for item in self.all():
            haystack = item.text.lower()
            hits = sum(haystack.count(t) for t in terms)
            if hits or not terms:
                scored.append((hits, item.created_at, item))
        # most relevant first, then most recent
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        return [item for _, _, item in scored[:limit]]

    def all(self) -> list[FileMemoryItem]:
        parsed = (self._parse(p) for p in self._dir.glob("*.md"))
        items = [item for item in parsed if item is not None]
        items.sort(key=lambda i: i.created_at, reverse=True)
        return items

    def delete(self, id: str) -> bool:
        path = self._path(id)
        if path.exists():
            path.unlink()
            return True
        return False

    # --- persistence helpers -------------------------------------------------

    def _path(self, id: str) -> Path:
        # The id is model-controlled; confine it to the store dir so it can't escape.
        candidate = (self._dir / f"{id}.md").resolve()
        if candidate.parent != self._dir.resolve():
            raise ValueError(f"invalid memory id: {id!r}")
        return candidate

    @staticmethod
    def _serialize(item: FileMemoryItem) -> str:
        return f"---\nid: {item.id}\ncreated_at: {item.created_at}\n---\n{item.text}\n"

    @staticmethod
    def _parse(path: Path) -> FileMemoryItem | None:
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return None
        header: dict[str, str] = {}
        body = raw
        if raw.startswith("---\n"):
            _, _, rest = raw.partition("---\n")
            front, sep, body = rest.partition("\n---\n")
            if sep:
                for line in front.splitlines():
                    key, _, value = line.partition(":")
                    header[key.strip()] = value.strip()
        try:
            created = float(header.get("created_at", "0") or 0)
        except ValueError:
            created = 0.0
        return FileMemoryItem(
            text=body.strip(),
            id=header.get("id") or path.stem,
            created_at=created,
            path=path,
        )
