from __future__ import annotations

import pytest

from graph import Graph


def test_add_node_is_idempotent_and_backfills_data():
    g = Graph()
    g.add_node("a")
    g.add_node("a", data="payload")  # same id, now with data
    assert len(g) == 1
    assert g.node("a").data == "payload"


def test_add_edge_requires_existing_endpoints():
    g = Graph()
    g.add_node("a")
    with pytest.raises(KeyError):
        g.add_edge("a", "missing")


def test_successors_predecessors_and_dedup():
    g = Graph()
    for n in "abc":
        g.add_node(n)
    g.add_edge("a", "b")
    g.add_edge("a", "b")  # duplicate edge: succ/pred not double-counted
    g.add_edge("a", "c")
    assert g.successors("a") == ["b", "c"]
    assert g.predecessors("b") == ["a"]


def test_leaves():
    g = Graph()
    for n in "abc":
        g.add_node(n)
    g.add_edge("a", "b")
    assert set(g.leaves()) == {"b", "c"}  # nodes with no successors


def test_levels_group_into_dependency_waves():
    g = Graph()
    for n in "abcd":
        g.add_node(n)
    g.add_edge("a", "c")
    g.add_edge("b", "c")
    g.add_edge("c", "d")
    assert g.levels() == [["a", "b"], ["c"], ["d"]]


def test_topological_order_flattens_levels():
    g = Graph()
    for n in "abc":
        g.add_node(n)
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    assert g.topological_order() == ["a", "b", "c"]


def test_levels_detects_cycle():
    g = Graph()
    for n in "ab":
        g.add_node(n)
    g.add_edge("a", "b")
    g.add_edge("b", "a")
    with pytest.raises(ValueError, match="cycle"):
        g.levels()


def test_attach_renames_on_collision_and_wires_needs():
    g = Graph()
    g.add_node("root", data=1)
    child = g.attach("root", data=2, needs="root")  # id collision → renamed
    assert child != "root"
    assert g.predecessors(child) == ["root"]


def test_attach_unknown_dependency_raises():
    g = Graph()
    with pytest.raises(KeyError):
        g.attach("x", needs="nope")
