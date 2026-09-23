from __future__ import annotations

import pytest

from harness.registry import PluginStatus, Registry
from plugins.guards import MaxIterations
from plugins.loops import AgenticLoop
from protocols.lifecycle import Lifecycle
from protocols.loop import Loop
from protocols.plugin import Plugin
from protocols.tool import Tool
from tests.conftest import RecordingTool


class LifecycleTool(Plugin, Lifecycle):
    """A registrable lifecycle plugin for exercising started-state tracking."""


def test_get_resolves_by_type():
    loop = AgenticLoop()
    r = Registry()
    r.add(loop)
    assert r.get(Loop) is loop


def test_get_returns_none_when_absent():
    assert Registry().get(Loop) is None


def test_all_returns_every_plugin_of_type():
    t1, t2 = RecordingTool("a"), RecordingTool("b")
    r = Registry()
    r.add(t1)
    r.add(t2)
    assert set(r.all(Tool)) == {t1, t2}


def test_type_discriminates_method_overlapping_bases():
    # Tool and Loop both expose `run`, but resolution by type must not confuse them.
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


def test_plugins_register_as_registered():
    plugin = LifecycleTool()
    r = Registry()
    r.add(plugin)
    assert r.status_of(plugin) is PluginStatus.REGISTERED
    assert r.is_started(plugin) is False
    assert r.unstarted(Lifecycle) == [plugin]
    assert r.started(Lifecycle) == []


def test_status_moves_a_plugin_between_the_partitions():
    plugin = LifecycleTool()
    r = Registry()
    r.add(plugin)
    r.set_status(plugin, PluginStatus.STARTED)
    assert r.is_started(plugin) is True
    assert r.unstarted(Lifecycle) == []
    assert r.started(Lifecycle) == [plugin]
    r.set_status(plugin, PluginStatus.REGISTERED)
    assert r.is_started(plugin) is False
    assert r.unstarted(Lifecycle) == [plugin]  # stopping makes it startable again


def test_failed_plugin_is_neither_startable_nor_started():
    plugin = LifecycleTool()
    r = Registry()
    r.add(plugin)
    r.set_status(plugin, PluginStatus.FAILED)
    assert r.is_started(plugin) is False
    assert r.unstarted(Lifecycle) == []  # not retried
    assert r.started(Lifecycle) == []


def test_partitions_keep_registration_order():
    a, b, c = LifecycleTool(), LifecycleTool(), LifecycleTool()
    r = Registry()
    for plugin in (a, b, c):
        r.add(plugin)
    r.set_status(a, PluginStatus.STARTED)
    r.set_status(c, PluginStatus.STARTED)
    assert r.started(Lifecycle) == [a, c]  # order of registration, not of marking
    assert r.unstarted(Lifecycle) == [b]


def test_remove_discards_status_with_the_entry():
    plugin = LifecycleTool()
    r = Registry()
    r.add(plugin)
    r.set_status(plugin, PluginStatus.STARTED)
    r.remove(plugin)
    assert r.status_of(plugin) is None  # state gone with the entry

    r.add(plugin)  # re-registering starts fresh
    assert r.status_of(plugin) is PluginStatus.REGISTERED
