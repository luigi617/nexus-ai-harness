from __future__ import annotations

import contextlib
import re

import pytest

from plugins.memory import FileMemoryStore
from plugins.tools import Forget, Recall, Remember
from protocols.memory import MemoryStore
from tests.conftest import make_ctx


def store(tmp_path):
    return FileMemoryStore(tmp_path)


def test_store_satisfies_protocol(tmp_path):
    assert isinstance(store(tmp_path), MemoryStore)


def test_save_creates_then_get_returns(tmp_path):
    s = store(tmp_path)
    item = s.save("user likes dark mode")
    assert s.get(item.id).text == "user likes dark mode"


def test_save_with_id_upserts_and_preserves_created_at(tmp_path):
    s = store(tmp_path)
    a = s.save("v1")
    b = s.save("v2", id=a.id)
    assert b.id == a.id
    assert b.created_at == a.created_at
    assert b.text == "v2"
    assert len(s.all()) == 1


def test_search_ranks_by_relevance(tmp_path):
    s = store(tmp_path)
    s.save("python is a language")
    s.save("bedrock runs in us-east-1")
    hits = s.search("python language")
    assert hits[0].text == "python is a language"


def test_search_miss_returns_empty(tmp_path):
    s = store(tmp_path)
    s.save("something")
    assert s.search("kubernetes") == []


def test_delete(tmp_path):
    s = store(tmp_path)
    item = s.save("x")
    assert s.delete(item.id) is True
    assert s.delete(item.id) is False
    assert s.get(item.id) is None


def test_persistence_across_instances(tmp_path):
    store(tmp_path).save("durable")
    assert len(FileMemoryStore(tmp_path).all()) == 1


# --- tools ---------------------------------------------------------------


def test_tools_roundtrip_via_ctx(tmp_path):
    s = store(tmp_path)
    ctx = make_ctx(s)
    out = Remember().run({"text": "prefers vim"}, ctx)
    assert out.startswith("remembered")
    listing = Recall().run({"query": "vim"}, ctx)
    mem_id = re.search(r"\((mem_\w+)\)", listing).group(1)
    assert Forget().run({"id": mem_id}, ctx).startswith("forgot")
    assert s.all() == []


def test_tools_error_without_store():
    ctx = make_ctx()
    assert "no memory store" in Remember().run({"text": "x"}, ctx)
    assert "no memory store" in Recall().run({"query": "x"}, ctx)


def test_remember_update_reports_updated(tmp_path):
    s = store(tmp_path)
    ctx = make_ctx(s)
    item = s.save("original")
    out = Remember().run({"text": "new", "id": item.id}, ctx)
    assert out.startswith("updated")


# --- corrupt / malformed store files -------------------------------------


def test_parse_no_frontmatter_uses_stem_and_zero_created_at(tmp_path):
    s = store(tmp_path)
    (tmp_path / "plain.md").write_text("just body text", encoding="utf-8")
    item = s.get("plain")
    assert item is not None
    assert item.id == "plain"  # id falls back to the filename stem
    assert item.text == "just body text"
    assert item.created_at == 0.0


def test_parse_unterminated_header_does_not_raise(tmp_path):
    s = store(tmp_path)
    (tmp_path / "unterminated.md").write_text(
        "---\nid: foo\nmore body", encoding="utf-8"
    )
    item = s.get("unterminated")
    # No closing '---': frontmatter unparsed, id=stem, created_at=0.0, body empty.
    assert item is not None
    assert item.id == "unterminated"
    assert item.text == ""
    assert item.created_at == 0.0


def test_parse_non_numeric_created_at_defaults_to_zero(tmp_path):
    s = store(tmp_path)
    (tmp_path / "m1.md").write_text(
        "---\nid: m1\ncreated_at: notanumber\n---\nbody", encoding="utf-8"
    )
    item = s.get("m1")
    assert item is not None
    assert item.id == "m1"
    assert item.text == "body"
    assert item.created_at == 0.0  # ValueError swallowed -> 0.0


def test_unreadable_file_is_skipped_not_fatal(tmp_path):
    s = store(tmp_path)
    good = s.save("readable")
    # A directory named like a memory file raises OSError on read_text.
    (tmp_path / "baddir.md").mkdir()
    assert s.get("baddir") is None  # _parse returns None on OSError
    ids = {i.id for i in s.all()}
    assert ids == {good.id}  # the unreadable entry is silently skipped
    # search iterates every *.md via _parse and must survive the bad entry too
    assert [i.id for i in s.search("readable")] == [good.id]


# --- search behavior -----------------------------------------------------


def _monotonic_time(monkeypatch):
    """Make plugins.memory.file.time.time strictly increasing per call."""
    import plugins.memory.file as filemod

    state = {"t": 0.0}

    def fake_time() -> float:
        state["t"] += 1.0
        return state["t"]

    monkeypatch.setattr(filemod.time, "time", fake_time)


def test_search_empty_query_returns_all_recent_first(tmp_path, monkeypatch):
    _monotonic_time(monkeypatch)
    s = store(tmp_path)
    s.save("alpha")
    s.save("beta")
    s.save("gamma")  # saved last -> newest
    hits = s.search("")  # blank query -> terms == [] -> return everything
    assert [h.text for h in hits] == ["gamma", "beta", "alpha"]  # created_at desc


def test_search_empty_query_respects_limit(tmp_path, monkeypatch):
    _monotonic_time(monkeypatch)
    s = store(tmp_path)
    for i in range(5):
        s.save(f"fact-{i}")
    hits = s.search("", limit=2)
    assert len(hits) == 2
    assert [h.text for h in hits] == ["fact-4", "fact-3"]  # newest first, capped


def test_search_recency_breaks_ties_on_equal_hits(tmp_path, monkeypatch):
    _monotonic_time(monkeypatch)
    s = store(tmp_path)
    s.save("python rocks")
    s.save("python rules")  # newer, same single keyword hit
    hits = s.search("python")
    assert [h.text for h in hits] == ["python rules", "python rocks"]


def test_search_uses_substring_counts(tmp_path):
    # 'java' matches inside 'javascript' (substring); more occurrences score higher.
    s = store(tmp_path)
    s.save("java once")  # 'java'.count == 1
    s.save("javascript javascript everywhere")  # 'java' substring counted twice
    hits = s.search("java")
    assert hits[0].text == "javascript javascript everywhere"
    assert {h.text for h in hits} == {
        "java once",
        "javascript javascript everywhere",
    }


# --- path traversal is rejected ------------------------------------------


def test_save_rejects_traversal_id(tmp_path):
    # A '..' id must be refused, not written outside the store directory.
    store_dir = tmp_path / "store"
    s = FileMemoryStore(store_dir)
    with pytest.raises(ValueError):
        s.save("pwned", id="../escaped")
    assert not (tmp_path / "escaped.md").exists()
    assert list(store_dir.glob("*.md")) == []


def test_delete_rejects_traversal_id(tmp_path):
    store_dir = tmp_path / "store"
    s = FileMemoryStore(store_dir)
    victim = tmp_path / "victim.md"
    victim.write_text("secret", encoding="utf-8")
    with pytest.raises(ValueError):
        s.delete("../victim")
    assert victim.exists()  # the outside file is untouched


def test_remember_tool_rejects_traversal_id(tmp_path):
    store_dir = tmp_path / "store"
    s = FileMemoryStore(store_dir)
    ctx = make_ctx(s)
    out = Remember().run({"text": "x", "id": "../escaped"}, ctx)
    assert out.startswith("error")
    assert not (tmp_path / "escaped.md").exists()


# --- Recall tool guard / limit edges -------------------------------------


def test_recall_no_matches_reports_none_found(tmp_path):
    s = store(tmp_path)
    s.save("something unrelated")
    assert Recall().run({"query": "nonexistent"}, make_ctx(s)) == (
        "no relevant memories found"
    )


def test_recall_limit_caps_results(tmp_path):
    s = store(tmp_path)
    for i in range(5):
        s.save(f"cat fact {i}")
    out = Recall().run({"query": "cat", "limit": 2}, make_ctx(s))
    assert len(out.splitlines()) == 2


def test_recall_invalid_limit_falls_back_to_five(tmp_path):
    s = store(tmp_path)
    for i in range(6):
        s.save(f"cat fact {i}")
    ctx = make_ctx(s)
    for bad in (0, -1, "x"):
        out = Recall().run({"query": "cat", "limit": bad}, ctx)
        assert len(out.splitlines()) == 5  # invalid -> default 5


def test_recall_bool_limit_falls_back_to_default(tmp_path):
    # A bool limit is rejected (isinstance(True, int) trap), so the default 5 applies.
    s = store(tmp_path)
    for i in range(3):
        s.save(f"cat fact {i}")
    out = Recall().run({"query": "cat", "limit": True}, make_ctx(s))
    assert len(out.splitlines()) == 3


# --- Forget tool guard branches ------------------------------------------


def test_forget_without_store_errors():
    assert "no memory store" in Forget().run({"id": "x"}, make_ctx())


def test_forget_blank_id_errors(tmp_path):
    ctx = make_ctx(store(tmp_path))
    assert Forget().run({"id": "  "}, ctx) == "error: no id provided"


def test_forget_missing_id_reports_no_memory(tmp_path):
    ctx = make_ctx(store(tmp_path))
    assert Forget().run({"id": "mem_missing"}, ctx) == "no memory with id 'mem_missing'"


def test_forget_success_reports_forgot(tmp_path):
    s = store(tmp_path)
    item = s.save("bye")
    assert Forget().run({"id": item.id}, make_ctx(s)) == f"forgot ({item.id})"


# --- Remember tool edges -------------------------------------------------


def test_remember_empty_text_errors_and_stores_nothing(tmp_path):
    s = store(tmp_path)
    ctx = make_ctx(s)
    out = Remember().run({"text": "   "}, ctx)
    assert out == "error: nothing to remember (empty text)"
    assert s.all() == []


def test_remember_nonexistent_id_creates_and_reports_remembered(tmp_path):
    s = store(tmp_path)
    ctx = make_ctx(s)
    out = Remember().run({"text": "x", "id": "mem_doesnotexist"}, ctx)
    assert out.startswith("remembered")  # not "updated" — the id did not exist
    assert not out.startswith("updated")
    stored = s.get("mem_doesnotexist")
    assert stored is not None and stored.text == "x"


# --- FileMemoryStore path confinement ------------------------------------


def test_save_rejects_or_confines_path_traversal_id(tmp_path):
    # A '..' id must never write outside the store directory.
    store_dir = tmp_path / "mem"
    s = FileMemoryStore(store_dir)
    outside = tmp_path / "escape.md"

    with contextlib.suppress(ValueError, OSError):
        s.save("secret", id="../escape")  # rejecting the id is also acceptable
    assert not outside.exists()


# --- Recall limit validation ---------------------------------------------


class _RecordingStore(MemoryStore):
    """A MemoryStore that records the limit forwarded to search()."""

    def __init__(self) -> None:
        self.forwarded_limit: object = None

    def save(self, text, id=None):  # pragma: no cover - unused
        raise NotImplementedError

    def get(self, id):  # pragma: no cover - unused
        return None

    def search(self, query, limit=5):
        self.forwarded_limit = limit
        return []

    def all(self):  # pragma: no cover - unused
        return []

    def delete(self, id):  # pragma: no cover - unused
        return False


def test_recall_treats_bool_limit_as_invalid():
    # A bool limit is not a valid count; Recall must fall back to the default (5).
    store = _RecordingStore()
    Recall().run({"query": "x", "limit": True}, make_ctx(store))
    assert store.forwarded_limit == 5
