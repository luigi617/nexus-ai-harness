from __future__ import annotations

import pytest

from harness.registry import Registry
from plugins.guards import MaxIterations
from plugins.loops import AgenticLoop
from protocols.loop import Loop
from protocols.tool import Tool
from tests.conftest import RecordingTool


def test_get_resolves_by_type():
    loop = AgenticLoop()
    r = Registry()
    r.add(loop)
    assert r.get(Loop) is loop


def test_get_returns_none_when_absent():
    assert Registry().get(Loop) is None


def test_all_returns_every_plugin_of_kind():
    t1, t2 = RecordingTool("a"), RecordingTool("b")
    r = Registry()
    r.add(t1)
    r.add(t2)
    assert set(r.all(Tool)) == {t1, t2}


def test_kind_discriminates_structurally_overlapping_protocols():
    # Tool and Loop both have `run` + `kind`; resolution must not confuse them.
    tool = RecordingTool()
    r = Registry()
    r.add(tool)
    assert r.get(Loop) is None
    assert r.get(Tool) is tool


def test_get_returns_last_registered():
    first, second = RecordingTool("x"), RecordingTool("x")
    r = Registry()
    r.add(first)
    r.add(second)
    assert r.get(Tool) is second


def test_add_rejects_non_plugin():
    with pytest.raises(TypeError):
        Registry().add(object())


def test_plugins_returns_all():
    a, b = AgenticLoop(), MaxIterations(3)
    r = Registry()
    r.add(a)
    r.add(b)
    assert set(r.plugins()) == {a, b}
