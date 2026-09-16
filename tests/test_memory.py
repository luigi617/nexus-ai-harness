from __future__ import annotations

import re

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
