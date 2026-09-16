from __future__ import annotations

from dataclasses import dataclass

from harness.context import RunContext
from harness.registry import Registry
from harness.session import Session


@dataclass
class Counter:
    n: int = 0


def make():
    return RunContext(Session(), Registry())


def test_lazy_default_created_on_first_access():
    assert make().state(Counter).n == 0


def test_state_persists_and_returns_same_instance():
    ctx = make()
    ctx.state(Counter).n = 9
    assert ctx.state(Counter).n == 9
    assert ctx.state(Counter) is ctx.state(Counter)


def test_state_is_isolated_per_session():
    a, b = make(), make()
    a.state(Counter).n = 5
    assert b.state(Counter).n == 0
