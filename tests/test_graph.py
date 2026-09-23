from __future__ import annotations

import pytest

from graph import Graph
from graph.node import Node


def test_add_node_is_idempotent_and_backfills_data():
    g = Graph()
    g.add_node("a")
    g.add_node("a", data="payload")  # same id, now with data
    assert len(g) == 1
    assert g.node("a").data == "payload"


def test_add_node_does_not_clobber_existing_non_none_data():
    # The backfill guard is one-directional: an existing payload is never overwritten.
    g = Graph()
    g.add_node("a", data=1)
    returned = g.add_node("a", data=2)
    assert g.node("a").data == 1  # first payload preserved, not clobbered
    assert returned is g.node("a")  # the pre-existing node instance is returned
    assert len(g) == 1


def test_add_node_accepts_a_node_instance():
    # The isinstance(node, Node) branch: a pre-built Node is stored as-is.
    g = Graph()
    n = Node(id="z", data=5)
    returned = g.add_node(n)
    assert returned is n
    assert g.node("z") is n
    assert g.node("z").data == 5


def test_contains_operator_reflects_membership():
    g = Graph()
    g.add_node("a")
    assert "a" in g
    assert "missing" not in g


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


def test_attach_wires_kwargs_edges_with_role_and_name_metadata():
    g = Graph()
    g.add_node("root", data=1)
    child = g.attach("child", data=2, kwargs={"x": "root"})
    assert g.predecessors(child) == ["root"]
    in_edges = g.in_edges(child)
    assert len(in_edges) == 1
    edge = in_edges[0]
    assert edge.source == "root"
    assert edge.target == child
    assert edge.metadata == {"role": "kwarg", "name": "x"}


def test_attach_unknown_kwarg_dependency_raises_naming_kwarg_and_dep():
    g = Graph()
    with pytest.raises(KeyError) as exc:
        g.attach("child", kwargs={"x": "missing"})
    message = str(exc.value)
    assert "x" in message  # the kwarg name
    assert "missing" in message  # the unknown dependency


def test_attach_wires_multiple_needs_as_fan_in():
    g = Graph()
    g.add_node("a")
    g.add_node("b")
    child = g.attach("c", needs=["a", "b"])
    assert set(g.predecessors(child)) == {"a", "b"}
    assert len(g.in_edges(child)) == 2


def test_attach_with_none_needs_creates_isolated_node():
    g = Graph()
    child = g.attach("d", needs=None)
    assert child == "d"
    assert g.predecessors("d") == []
    assert "d" in g


def test_attach_same_id_repeatedly_yields_distinct_ids():
    g = Graph()
    g.add_node("a")
    first = g.attach("a")
    second = g.attach("a")
    third = g.attach("a")
    assert len({"a", first, second, third}) == 4  # all four ids are distinct


def test_attach_collision_rename_mints_a_fresh_unique_id():
    # attach() bumps the suffix until unique, so it skips a pre-existing "<id>#<len>".
    g = Graph()
    g.add_node("x")
    g.add_node("x#2")
    returned = g.attach("x", needs="x")
    assert returned not in ("x", "x#2")  # a genuinely fresh id
    assert len(g) == 3
    assert g.predecessors("x#2") == []  # the pre-existing node is untouched
    assert g.predecessors(returned) == ["x"]  # the edge wired onto the new node


def test_levels_and_topological_order_on_empty_graph():
    assert Graph().levels() == []
    assert Graph().topological_order() == []


def test_disconnected_nodes_form_a_single_sorted_wave():
    g = Graph()
    for n in "abc":
        g.add_node(n)
    assert g.levels() == [["a", "b", "c"]]  # no edges -> one wave, sorted
    assert g.topological_order() == ["a", "b", "c"]
    assert set(g.leaves()) == {"a", "b", "c"}  # nothing has a successor


# --- to_text rendering ---------------------------------------------------------

_TOP_10 = "┌" + "─" * 12 + "┐"
_BOTTOM_10 = "└" + "─" * 12 + "┘"


def test_to_text_renders_exact_box_for_a_short_label():
    g = Graph()
    g.add_node("n1", data="hi")
    lines = g.to_text(width=10).splitlines()
    assert lines == [_TOP_10, "│ " + "hi".ljust(10) + " │", _BOTTOM_10]
    # Every line is width + 4 chars; borders match the content line length.
    assert {len(line) for line in lines} == {14}


def test_to_text_truncates_a_label_longer_than_width():
    g = Graph()
    g.add_node("n1", data="abcdefghijkl")  # 12 chars > width 10
    content = g.to_text(width=10).splitlines()[1]
    # Truncated to width-3 chars plus an ellipsis.
    assert content == "│ " + ("abcdefg" + "...") + " │"


def test_to_text_flattens_newlines_in_a_label():
    g = Graph()
    g.add_node("n1", data="a\nb")
    content = g.to_text(width=10).splitlines()[1]
    assert content == "│ " + "a b".ljust(10) + " │"


def test_to_text_applies_a_custom_label_callable_to_node_data():
    g = Graph()
    g.add_node("n1", data=5)
    content = g.to_text(label=lambda data: f"L{data}", width=10).splitlines()[1]
    assert content == "│ " + "L5".ljust(10) + " │"


def test_to_text_places_a_connector_between_each_pair_of_boxes():
    g = Graph()
    for n in "abc":
        g.add_node(n, data=n)
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    text = g.to_text(width=10)
    connector = " " * ((10 + 4) // 2) + "│"
    # Exactly len(order) - 1 connector lines between the three boxes.
    assert text.splitlines().count(connector) == 2


def test_to_text_of_empty_graph_is_empty_string():
    assert Graph().to_text() == ""


# --- attach() id uniqueness ----------------------------------------------


def test_attach_generates_a_unique_id_on_collision():
    # attach() must mint a unique id even when its first rename candidate exists.
    g = Graph()
    g.add_node("a")
    g.add_node("a#2")  # exactly the name attach() will try next (len == 2)
    assert len(g) == 2

    new_id = g.attach("a")
    assert len(g) == 3  # a genuinely new node was created
    assert new_id not in ("a", "a#2")
