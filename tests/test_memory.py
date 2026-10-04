from __future__ import annotations

import contextlib
import re

import pytest

from nexus_ai_harness.plugins.memory import FileMemoryStore
from nexus_ai_harness.plugins.tools import Forget, Recall, Remember
from nexus_ai_harness.protocols.memory import MemoryStore
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


def test_parse_unterminated_header_keeps_whole_file_as_body(tmp_path):
    s = store(tmp_path)
    (tmp_path / "unterminated.md").write_text(
        "---\nid: foo\nmore body", encoding="utf-8"
    )
    item = s.get("unterminated")
    # No closing '---': not frontmatter, so nothing is dropped from the body.
    assert item is not None
    assert item.id == "unterminated"
    assert item.text == "---\nid: foo\nmore body"
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
    import nexus_ai_harness.plugins.memory.file as filemod

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


def test_search_matches_whole_words_only(tmp_path):
    # 'java' must not match inside 'javascript'.
    s = store(tmp_path)
    s.save("java once")
    s.save("javascript javascript everywhere")
    assert [h.text for h in s.search("java")] == ["java once"]


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


# --- round-trip-safe format ----------------------------------------------


def test_body_with_delimiter_lines_cannot_inject_header(tmp_path):
    s = store(tmp_path)
    text = "---\nid: evil\ncreated_at: 1\n---\nreal body\n---\ntail"
    item = s.save(text)
    loaded = FileMemoryStore(tmp_path).get(item.id)
    assert loaded is not None
    assert loaded.text == text
    assert loaded.id == item.id
    assert loaded.created_at == item.created_at


@pytest.mark.parametrize(
    "text",
    ["  padded  ", "\n\nleading blank lines", "trailing newlines\n\n", "a\r\nb", ""],
)
def test_text_round_trips_verbatim(tmp_path, text):
    s = store(tmp_path)
    item = s.save(text)
    assert item.text == text
    assert FileMemoryStore(tmp_path).get(item.id).text == text


def test_serialized_file_quotes_id_and_marks_format(tmp_path):
    s = store(tmp_path)
    item = s.save("body", id="mem_1")
    raw = (tmp_path / "mem_1.md").read_text(encoding="utf-8")
    assert raw == (
        f'---\nid: "mem_1"\ncreated_at: {item.created_at!r}\nformat: 2\n---\nbody\n'
    )


def test_filename_is_authoritative_over_header_id(tmp_path):
    (tmp_path / "real.md").write_text(
        "---\nid: other\ncreated_at: 5\n---\nbody", encoding="utf-8"
    )
    item = store(tmp_path).get("real")
    assert item is not None
    assert item.id == "real"
    assert item.created_at == 5.0


def test_legacy_file_still_loads(tmp_path):
    # Exactly what the pre-format-2 serializer wrote, with a '---' line in the body.
    (tmp_path / "mem_old.md").write_text(
        "---\nid: mem_old\ncreated_at: 1700000000.0\n---\nfirst\n---\nsecond\n",
        encoding="utf-8",
    )
    item = store(tmp_path).get("mem_old")
    assert item is not None
    assert item.text == "first\n---\nsecond"
    assert item.created_at == 1700000000.0


def test_legacy_crlf_file_is_normalized(tmp_path):
    (tmp_path / "win.md").write_bytes(
        b"---\r\nid: win\r\ncreated_at: 3.0\r\n---\r\n  line one\r\nline two\r\n"
    )
    item = store(tmp_path).get("win")
    assert item is not None
    assert item.text == "line one\nline two"
    assert item.created_at == 3.0


def test_resaving_legacy_file_keeps_created_at(tmp_path):
    (tmp_path / "mem_old.md").write_text(
        "---\nid: mem_old\ncreated_at: 42.0\n---\nold\n", encoding="utf-8"
    )
    s = store(tmp_path)
    item = s.save("new", id="mem_old")
    assert item.created_at == 42.0
    assert FileMemoryStore(tmp_path).get("mem_old").text == "new"


def test_non_utf8_file_is_skipped(tmp_path):
    s = store(tmp_path)
    good = s.save("fine")
    (tmp_path / "binary.md").write_bytes(b"\xff\xfe\x00junk")
    assert s.get("binary") is None
    assert [i.id for i in s.all()] == [good.id]


@pytest.mark.parametrize("bad", ["", " ", "\t\n", "a\nb", "nul\x00"])
def test_blank_or_control_char_ids_are_rejected(tmp_path, bad):
    s = store(tmp_path)
    with pytest.raises(ValueError):
        s.save("x", id=bad)
    with pytest.raises(ValueError):
        s.get(bad)
    with pytest.raises(ValueError):
        s.delete(bad)
    assert list(tmp_path.glob("*.md")) == []


# --- atomic, locked writes -----------------------------------------------


def test_failed_replace_keeps_original_and_cleans_temp(tmp_path, monkeypatch):
    import nexus_ai_harness.plugins.memory.file as filemod

    s = store(tmp_path)
    item = s.save("original")

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(filemod.os, "replace", boom)
    with pytest.raises(OSError):
        s.save("replacement", id=item.id)
    monkeypatch.undo()
    assert FileMemoryStore(tmp_path).get(item.id).text == "original"
    assert list(tmp_path.glob("*.tmp")) == []


def test_save_leaves_no_temp_or_lock_entries_in_listing(tmp_path):
    s = store(tmp_path)
    s.save("one")
    s.save("two")
    assert list(tmp_path.glob("*.tmp")) == []
    assert len(s.all()) == 2


def test_writes_take_exclusive_flock(tmp_path, monkeypatch):
    import nexus_ai_harness.plugins.memory.file as filemod

    calls = []

    class FakeFcntl:
        LOCK_EX = 2

        @staticmethod
        def flock(fd, op):
            calls.append(op)

    monkeypatch.setattr(filemod, "_FCNTL", FakeFcntl)
    s = store(tmp_path)
    item = s.save("x")
    s.delete(item.id)
    assert calls == [FakeFcntl.LOCK_EX, FakeFcntl.LOCK_EX]
    assert (tmp_path / ".lock").exists()


def test_flock_failure_degrades_to_unlocked_write(tmp_path, monkeypatch):
    import nexus_ai_harness.plugins.memory.file as filemod

    class BrokenFcntl:
        LOCK_EX = 2

        @staticmethod
        def flock(fd, op):
            raise OSError("flock unsupported")

    monkeypatch.setattr(filemod, "_FCNTL", BrokenFcntl)
    s = store(tmp_path)
    assert s.get(s.save("still saved").id).text == "still saved"


def test_works_without_fcntl(tmp_path, monkeypatch):
    import nexus_ai_harness.plugins.memory.file as filemod

    monkeypatch.setattr(filemod, "_FCNTL", None)
    s = store(tmp_path)
    item = s.save("no fcntl")
    assert s.get(item.id).text == "no fcntl"
    assert s.delete(item.id) is True


def test_concurrent_threads_never_observe_partial_files(tmp_path):
    import threading

    s = store(tmp_path)
    item = s.save("seed")
    values = {"seed"} | {f"value {i} " + "x" * (i * 500) for i in range(20)}
    errors = []

    def writer(i):
        try:
            FileMemoryStore(tmp_path).save(f"value {i} " + "x" * (i * 500), item.id)
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    def reader():
        reader_store = FileMemoryStore(tmp_path)
        for _ in range(50):
            got = reader_store.get(item.id)
            if got is None or got.text not in values:
                errors.append(AssertionError(f"partial read: {got!r}"))

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
    threads += [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert s.get(item.id).text in values
    assert list(tmp_path.glob("*.tmp")) == []


def _save_many_in_process(directory: str, worker: int) -> None:
    s = FileMemoryStore(directory)
    for i in range(10):
        s.save(f"worker {worker} fact {i}")
        s.save(f"worker {worker} shared {i}", id="shared")


def test_concurrent_processes_do_not_corrupt_store(tmp_path):
    import multiprocessing

    # Seeded first: the lock must keep every concurrent update on this created_at.
    seed = store(tmp_path).save("seed", id="shared")
    mp = multiprocessing.get_context("spawn")
    procs = [
        mp.Process(target=_save_many_in_process, args=(str(tmp_path), w))
        for w in range(3)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    s = store(tmp_path)
    texts = {i.text for i in s.all()}
    assert {f"worker {w} fact {i}" for w in range(3) for i in range(10)} <= texts
    assert s.get("shared").text.startswith("worker ")
    assert s.get("shared").created_at == seed.created_at
    assert len(s.all()) == 31
    assert list(tmp_path.glob("*.tmp")) == []


# --- search cache --------------------------------------------------------


def _count_parses(monkeypatch):
    original = FileMemoryStore._parse
    calls = []

    def counting(path):
        calls.append(path.name)
        return original(path)

    monkeypatch.setattr(FileMemoryStore, "_parse", staticmethod(counting))
    return calls


def test_repeated_search_does_not_reparse_unchanged_files(tmp_path, monkeypatch):
    s = store(tmp_path)
    for i in range(3):
        s.save(f"cached fact {i}")
    calls = _count_parses(monkeypatch)
    s.search("cached")
    s.search("fact")
    s.all()
    assert calls == []


def test_cache_picks_up_edits_from_another_instance(tmp_path):
    s = store(tmp_path)
    item = s.save("old wording")
    assert s.search("old")
    other = FileMemoryStore(tmp_path)
    other.save("new phrasing entirely", id=item.id)
    assert s.search("old") == []
    assert [h.text for h in s.search("phrasing")] == ["new phrasing entirely"]
    other.delete(item.id)
    assert s.all() == []
    assert s.get(item.id) is None


def test_returned_items_do_not_alias_the_cache(tmp_path):
    s = store(tmp_path)
    item = s.save("immutable")
    s.all()[0].text = "mutated"
    s.search("immutable")[0].text = "mutated"
    item.text = "mutated"
    assert s.get(item.id).text == "immutable"


# --- search scoring ------------------------------------------------------


def test_search_is_case_insensitive_and_ignores_punctuation(tmp_path):
    s = store(tmp_path)
    s.save("User prefers Dark-Mode, always.")
    assert [h.text for h in s.search("dark MODE")] == [
        "User prefers Dark-Mode, always."
    ]


def test_search_rare_terms_outweigh_common_ones(tmp_path, monkeypatch):
    _monotonic_time(monkeypatch)
    s = store(tmp_path)
    s.save("rust is fast")
    for i in range(5):
        s.save(f"python note {i}")
    s.save("python python here")  # newest, repeats the common term
    hits = s.search("python rust")
    assert hits[0].text == "rust is fast"


def test_search_ties_break_by_id_when_created_at_equal(tmp_path, monkeypatch):
    import nexus_ai_harness.plugins.memory.file as filemod

    monkeypatch.setattr(filemod.time, "time", lambda: 1.0)
    s = store(tmp_path)
    s.save("same text", id="b")
    s.save("same text", id="a")
    s.save("same text", id="c")
    assert [h.id for h in s.search("text")] == ["a", "b", "c"]
    assert [h.id for h in s.all()] == ["a", "b", "c"]


def test_search_punctuation_only_query_matches_nothing(tmp_path):
    s = store(tmp_path)
    s.save("something")
    assert s.search("!!!") == []


def test_search_non_positive_limit_returns_empty(tmp_path):
    s = store(tmp_path)
    s.save("match me")
    assert s.search("match", limit=0) == []
    assert s.search("match", limit=-1) == []


# --- review follow-ups: canonical ids, shared locks, stale temps ---------


@pytest.mark.parametrize("bad", ["./x", "a/../x", "sub/x", "x/"])
def test_non_canonical_ids_are_rejected(tmp_path, bad):
    s = store(tmp_path)
    with pytest.raises(ValueError):
        s.save("hello", id=bad)
    with pytest.raises(ValueError):
        s.get(bad)
    assert list(tmp_path.glob("*.md")) == []


def test_saved_id_matches_id_reported_by_reads(tmp_path):
    s = store(tmp_path)
    item = s.save("hello", id="x")
    assert item.id == "x"
    assert s.get("x").id == "x"
    assert [i.id for i in s.all()] == ["x"]


@pytest.mark.parametrize("bad", [5, 1.5, ["x"], {"a": 1}])
def test_non_string_id_raises_value_error(tmp_path, bad):
    s = store(tmp_path)
    with pytest.raises(ValueError):
        s.save("x", id=bad)
    with pytest.raises(ValueError):
        s.get(bad)


def test_remember_tool_accepts_numeric_id(tmp_path):
    s = store(tmp_path)
    out = Remember().run({"text": "fact", "id": 5}, make_ctx(s))
    assert out == "remembered (5)"
    assert s.get("5").text == "fact"


def _case_insensitive(directory) -> bool:
    probe = directory / "CaseProbe.md"
    probe.write_text("", encoding="utf-8")
    try:
        return (directory / "caseprobe.md").exists()
    finally:
        probe.unlink()


def test_case_aliased_ids_report_the_saved_spelling(tmp_path):
    if not _case_insensitive(tmp_path):
        pytest.skip("filesystem is case-sensitive")
    s = store(tmp_path)
    first = s.save("hi", id="Foo")
    assert s.get("foo").id == "Foo"
    again = s.save("updated", id="foo")
    assert again.id == "Foo"
    assert again.created_at == first.created_at
    assert [(i.id, i.text) for i in s.all()] == [("Foo", "updated")]


def test_header_case_alias_ignored_when_it_names_another_file(tmp_path):
    if _case_insensitive(tmp_path):
        pytest.skip("filesystem is case-insensitive")
    (tmp_path / "foo.md").write_text(
        '---\nid: "Foo"\ncreated_at: 1\nformat: 2\n---\nbody\n', encoding="utf-8"
    )
    assert store(tmp_path).get("foo").id == "foo"


@pytest.mark.flaky(reruns=2, reruns_delay=0.3)
def test_stores_on_one_directory_share_the_in_process_lock(tmp_path, monkeypatch):
    import threading
    import time as _time

    import nexus_ai_harness.plugins.memory.file as filemod

    monkeypatch.setattr(filemod, "_FCNTL", None)
    original = FileMemoryStore._write_atomic
    active = []
    overlaps = []
    guard = threading.Lock()

    def slow_write(self, path, data):
        with guard:
            active.append(1)
            if len(active) > 1:
                overlaps.append(len(active))
        _time.sleep(0.02)
        try:
            original(self, path, data)
        finally:
            with guard:
                active.pop()

    monkeypatch.setattr(FileMemoryStore, "_write_atomic", slow_write)
    seed = store(tmp_path).save("seed", id="shared")
    stores = [FileMemoryStore(tmp_path) for _ in range(4)]
    threads = [
        threading.Thread(target=st.save, args=(f"v{n}", "shared"))
        for n, st in enumerate(stores)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert overlaps == []
    assert store(tmp_path).get("shared").created_at == seed.created_at


def test_save_blocks_while_another_holder_has_the_flock(tmp_path):
    import os
    import threading

    import nexus_ai_harness.plugins.memory.file as filemod

    if filemod._FCNTL is None:
        pytest.skip("fcntl is unavailable")
    fcntl = filemod._FCNTL
    s = store(tmp_path)
    fd = os.open(tmp_path / ".lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        # A separate open file description conflicts, just as another process would.
        fcntl.flock(fd, fcntl.LOCK_EX)
        done = threading.Event()
        worker = threading.Thread(target=lambda: (s.save("x", id="a"), done.set()))
        worker.start()
        assert not done.wait(0.3)
        assert FileMemoryStore(tmp_path).get("a") is None
    finally:
        os.close(fd)
    worker.join(timeout=10)
    assert done.is_set()
    assert s.get("a").text == "x"


def test_opening_store_removes_only_stale_temp_files(tmp_path):
    import os
    import time as _time

    stale = tmp_path / ".mem_1.md.abcd.tmp"
    fresh = tmp_path / ".mem_2.md.efgh.tmp"
    stale.write_text("old", encoding="utf-8")
    fresh.write_text("new", encoding="utf-8")
    old = _time.time() - 2 * 3600
    os.utime(stale, (old, old))
    (tmp_path / "keep.md").write_text("note", encoding="utf-8")
    os.utime(tmp_path / "keep.md", (old, old))
    s = store(tmp_path)
    assert not stale.exists()
    assert fresh.exists()
    assert s.get("keep").text == "note"
