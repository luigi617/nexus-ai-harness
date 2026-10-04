from __future__ import annotations

import contextlib
import importlib
import json
import math
import os
import re
import stat
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import ModuleType

from nexus_ai_harness.core.ids import new_id
from nexus_ai_harness.protocols.memory import MemoryItem, MemoryStore
from nexus_ai_harness.services.fs import replace_retrying

_FORMAT_VERSION = "2"
"""Header ``format`` value marking a file whose body is stored verbatim."""

_DELIMITER = "---"
_LOCK_NAME = ".lock"
_HEADER_LINE = re.compile(r"([A-Za-z_][A-Za-z0-9_-]*)[ \t]*:(.*)")
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_TOKEN = re.compile(r"\w+")
_TEMP_NAME = re.compile(r"\..+\.md\..+\.tmp")

_BM25_K1 = 1.5
_BM25_B = 0.75

_STALE_TEMP_SECONDS = 3600.0
"""Age after which a leftover temp file is assumed orphaned by a crashed writer."""

_DIR_LOCKS: dict[Path, threading.RLock] = {}
_DIR_LOCKS_GUARD = threading.Lock()


def _dir_lock(root: Path) -> threading.RLock:
    """Return the process-wide lock shared by every store on ``root``."""
    with _DIR_LOCKS_GUARD:
        return _DIR_LOCKS.setdefault(root, threading.RLock())


def _load_fcntl() -> ModuleType | None:
    try:
        return importlib.import_module("fcntl")
    except ImportError:  # e.g. Windows: fall back to in-process locking only
        return None


_FCNTL = _load_fcntl()


def _tokenize(text: str) -> list[str]:
    """Split ``text`` into case-insensitive word tokens."""
    return _TOKEN.findall(text.casefold())


@dataclass
class FileMemoryItem(MemoryItem):
    """A MemoryItem that also tracks the file it lives in.

    Attributes:
        created_at: When the memory was first saved, as a Unix timestamp.
        path: The markdown file backing the memory, if known.
    """

    id: str = field(default_factory=lambda: new_id("mem"))
    created_at: float = field(default_factory=time.time)
    path: Path | None = None


@dataclass
class _Entry:
    """A parsed memory file plus its search statistics, keyed by file identity."""

    signature: tuple[int, int, int]
    item: FileMemoryItem
    terms: Counter[str]
    length: int


class FileMemoryStore(MemoryStore):
    """One markdown file per memory with a small frontmatter header.

    Format (``<dir>/<id>.md``)::

        ---
        id: "mem_ab12"
        created_at: 1700000000.0
        format: 2
        ---
        the fact text

    The file name is the authoritative id; the header ``id`` is informational.
    Only a frontmatter block that opens on the file's first line is recognized,
    and it ends at the first ``---`` line, so the body may itself contain
    ``---`` lines. Text is stored verbatim: the writer appends one newline
    after the body and the reader removes exactly one. Files without
    ``format: 2`` (written by older versions or by hand) still load, with their
    body stripped of surrounding whitespace as before.

    Writes go to a unique temp file that is fsynced and then atomically renamed
    over the target, so readers never see a partial file. Saves and deletes
    also take an exclusive ``fcntl.flock`` on ``<dir>/.lock`` where available,
    serializing writers across processes. Every store on the same directory in
    one process also shares a thread lock, so where ``flock`` is unavailable
    writers within one process are still serialized. Temp files orphaned by a
    crashed writer are removed when a store is opened, once they are an hour
    old.

    Parsed files are cached in-process and revalidated by inode, size, and
    modification time, so repeated searches only ``stat`` unchanged files and
    edits from other processes are still picked up.
    """

    def __init__(self, directory: str | Path = "~/.nexus-ai-harness/memory") -> None:
        self._dir = Path(directory).expanduser()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._root = self._dir.resolve()
        self._cache: dict[str, _Entry] = {}
        self._mutex = threading.RLock()
        # Shared per directory so separate instances in one process don't race.
        self._write_lock = _dir_lock(self._root)
        self._remove_stale_temps()

    def save(self, text: str, id: str | None = None) -> FileMemoryItem:
        """Create or overwrite a memory.

        Args:
            text: The memory's text, stored verbatim.
            id: The id of the memory to overwrite; omit it to create a new one.

        Returns:
            The stored memory. An overwrite keeps the original ``created_at``,
            and on a case-insensitive filesystem also the id's original casing.

        Raises:
            ValueError: If ``id`` is not a string, is blank, contains control
                characters, or is not a plain file name (for example ``./x``).
        """
        item_id = new_id("mem") if id is None else id
        path = self._path(item_id)
        with self._write_lock, self._file_lock():
            existing = self._load(path)
            created = existing.item.created_at if existing is not None else time.time()
            if existing is not None:
                # Case-insensitive filesystems alias ids; keep the one on disk.
                item_id = existing.item.id
                path = path.with_name(f"{item_id}.md")
            item = FileMemoryItem(text=text, id=item_id, created_at=created, path=path)
            self._write_atomic(path, self._serialize(item))
            self._load(path)
        return replace(item)

    def get(self, id: str) -> FileMemoryItem | None:
        entry = self._load(self._path(id))
        return replace(entry.item) if entry is not None else None

    def search(self, query: str, limit: int = 5) -> list[FileMemoryItem]:
        """Rank memories against ``query`` with BM25 over word tokens.

        Matching is case-insensitive and on whole words, so ``java`` does not
        match ``javascript``. Ties break by most recent, then by id.

        Args:
            query: Keywords to look for. A blank query returns the most recent
                memories.
            limit: The maximum number of memories to return.

        Returns:
            Matching memories, most relevant first; memories that share no
            term with the query are omitted.
        """
        entries = self._entries()
        if not query.strip():
            ranked = sorted(entries, key=_recency)
        else:
            scored = _bm25(list(dict.fromkeys(_tokenize(query))), entries)
            scored.sort(key=lambda s: (-s[0], *_recency(s[1])))
            ranked = [entry for _, entry in scored]
        return [replace(entry.item) for entry in ranked[: max(limit, 0)]]

    def all(self) -> list[FileMemoryItem]:
        return [replace(entry.item) for entry in sorted(self._entries(), key=_recency)]

    def delete(self, id: str) -> bool:
        path = self._path(id)
        with self._write_lock, self._file_lock():
            with self._mutex:
                self._cache.pop(path.name, None)
            try:
                path.unlink()
            except FileNotFoundError:
                return False
            self._fsync_dir()
        return True

    # --- persistence helpers -------------------------------------------------

    def _path(self, id: str) -> Path:
        # The id is model-controlled; confine it to the store dir so it can't escape.
        if not isinstance(id, str) or not id.strip() or _CONTROL_CHARS.search(id):
            raise ValueError(f"invalid memory id: {id!r}")
        name = f"{id}.md"
        candidate = (self._dir / name).resolve()
        # Requiring an exact name keeps ids canonical: no "./x" aliasing "x".
        # Casefold the name check only: a case-insensitive filesystem resolves
        # to the on-disk spelling, which must still be accepted as that id.
        if (
            candidate.parent != self._root
            or candidate.name.casefold() != name.casefold()
        ):
            raise ValueError(f"invalid memory id: {id!r}")
        return candidate

    def _remove_stale_temps(self) -> None:
        # Temps from a writer killed before its replace; age spares in-flight ones.
        cutoff = time.time() - _STALE_TEMP_SECONDS
        with contextlib.suppress(OSError), os.scandir(self._dir) as it:
            for entry in it:
                if not _TEMP_NAME.fullmatch(entry.name):
                    continue
                with contextlib.suppress(OSError):
                    if entry.is_file(follow_symlinks=False) and (
                        entry.stat(follow_symlinks=False).st_mtime < cutoff
                    ):
                        os.unlink(entry.path)

    def _entries(self) -> list[_Entry]:
        with os.scandir(self._dir) as it:
            names = [e.name for e in it if e.name.endswith(".md")]
        with self._mutex:
            for gone in self._cache.keys() - set(names):
                del self._cache[gone]
        loaded = (self._load(self._root / name) for name in names)
        return [entry for entry in loaded if entry is not None]

    def _load(self, path: Path) -> _Entry | None:
        """Return the parsed entry for ``path``, reparsing only if the file changed."""
        try:
            st = path.stat()
        except OSError:
            st = None
        if st is None or not stat.S_ISREG(st.st_mode):
            with self._mutex:
                self._cache.pop(path.name, None)
            return None
        # A replace yields a new inode, so this catches rewrites within one mtime tick.
        signature = (st.st_ino, st.st_size, st.st_mtime_ns)
        with self._mutex:
            cached = self._cache.get(path.name)
            if cached is not None and cached.signature == signature:
                return cached
        # Stat precedes the read so a racing rewrite only causes a later reparse.
        item = self._parse(path)
        if item is None:
            with self._mutex:
                self._cache.pop(path.name, None)
            return None
        terms = Counter(_tokenize(item.text))
        entry = _Entry(signature, item, terms, sum(terms.values()))
        with self._mutex:
            self._cache[path.name] = entry
        return entry

    @contextlib.contextmanager
    def _file_lock(self) -> Iterator[None]:
        if _FCNTL is None:
            yield
            return
        fd = os.open(self._dir / _LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            # Filesystems without flock support degrade to unlocked writes.
            with contextlib.suppress(OSError):
                _FCNTL.flock(fd, _FCNTL.LOCK_EX)
            yield
        finally:
            os.close(fd)  # closing the descriptor releases the lock

    def _write_atomic(self, path: Path, data: str) -> None:
        # Unique per writer so concurrent saves never share a temp; .tmp skips the scan.
        fd, tmp = tempfile.mkstemp(
            dir=self._root, prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data.encode("utf-8"))
                f.flush()
                os.fsync(f.fileno())
            replace_retrying(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        self._fsync_dir()

    def _fsync_dir(self) -> None:
        # Persists the rename itself; not supported (or needed) on every platform.
        with contextlib.suppress(OSError, AttributeError):
            fd = os.open(self._dir, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

    @staticmethod
    def _serialize(item: FileMemoryItem) -> str:
        return (
            f"{_DELIMITER}\n"
            f"id: {json.dumps(item.id)}\n"
            f"created_at: {item.created_at!r}\n"
            f"format: {_FORMAT_VERSION}\n"
            f"{_DELIMITER}\n"
            f"{item.text}\n"
        )

    @staticmethod
    def _parse(path: Path) -> FileMemoryItem | None:
        try:
            raw = path.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        header, body = _split_frontmatter(raw)
        if header.get("format") == _FORMAT_VERSION:
            body = body.removesuffix("\n")
        else:
            # Legacy and hand-written files keep the old universal-newline + strip.
            body = body.replace("\r\n", "\n").replace("\r", "\n").strip()
        try:
            created = float(header.get("created_at", "0") or 0)
        except ValueError:
            created = 0.0
        return FileMemoryItem(
            text=body,
            id=_canonical_id(path, header.get("id")),
            created_at=created,
            path=path,
        )


def _canonical_id(path: Path, header_id: str | None) -> str:
    """Return the id for ``path``: its stem, unless the header names a case alias.

    On a case-insensitive filesystem ``Foo.md`` is also reachable as ``foo.md``;
    the header records the spelling it was saved under, so every read reports
    that one. The header only wins when it names this very file, so it cannot
    inject an unrelated id.
    """
    stem = path.stem
    if not header_id or header_id == stem or header_id.casefold() != stem.casefold():
        return stem
    try:
        return header_id if path.with_name(f"{header_id}.md").samefile(path) else stem
    except (OSError, ValueError):
        return stem


def _split_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    """Split a leading frontmatter block from the body.

    The block must open on the first line and close at the first ``---`` line.
    Anything else, such as an unterminated block or a line that isn't
    ``key: value``, means the whole text is body.

    Returns:
        The header fields (first occurrence of each key wins) and the body.
    """
    first, sep, rest = raw.partition("\n")
    if not sep or first.rstrip("\r") != _DELIMITER:
        return {}, raw
    header: dict[str, str] = {}
    while True:
        line, sep, remainder = rest.partition("\n")
        line = line.rstrip("\r")
        if line == _DELIMITER:
            return header, remainder
        if not sep:
            return {}, raw
        if line.strip():
            match = _HEADER_LINE.fullmatch(line)
            if match is None:
                return {}, raw
            header.setdefault(match.group(1), _decode_value(match.group(2).strip()))
        rest = remainder


def _decode_value(value: str) -> str:
    if value.startswith('"'):
        with contextlib.suppress(ValueError):
            decoded = json.loads(value)
            if isinstance(decoded, str):
                return decoded
    return value


def _recency(entry: _Entry) -> tuple[float, str]:
    return (-entry.item.created_at, entry.item.id)


def _bm25(terms: list[str], entries: list[_Entry]) -> list[tuple[float, _Entry]]:
    """Score each entry against ``terms`` with Okapi BM25, dropping non-matches."""
    if not terms or not entries:
        return []
    count = len(entries)
    average = sum(e.length for e in entries) / count or 1.0
    idf: dict[str, float] = {}
    for term in terms:
        df = sum(1 for e in entries if term in e.terms)
        idf[term] = math.log(1 + (count - df + 0.5) / (df + 0.5))
    scored: list[tuple[float, _Entry]] = []
    for entry in entries:
        norm = _BM25_K1 * (1 - _BM25_B + _BM25_B * entry.length / average)
        score = 0.0
        for term in terms:
            tf = entry.terms.get(term, 0)
            if tf:
                score += idf[term] * tf * (_BM25_K1 + 1) / (tf + norm)
        if score > 0:
            scored.append((score, entry))
    return scored
