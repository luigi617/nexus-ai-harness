from __future__ import annotations

from dataclasses import dataclass

from harness.context import RunContext
from harness.registry import Registry
from harness.session import Session


@dataclass
class Counter:
    n: int = 0


@dataclass
class Flag:
    on: bool = False


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


def test_two_contexts_over_the_same_session_share_one_state_instance():
    # State is keyed to the Session, not RunContext, so two contexts over one share it.
    session = Session()
    ctx_a = RunContext(session, Registry())
    ctx_b = RunContext(session, Registry())
    ctx_a.state(Counter).n = 7
    assert ctx_b.state(Counter).n == 7
    assert ctx_b.state(Counter) is ctx_a.state(Counter)


def test_state_keyed_by_type_returns_distinct_instances_per_class():
    ctx = make()
    counter = ctx.state(Counter)
    flag = ctx.state(Flag)
    # Different requested classes yield different objects, each of its own type.
    assert counter is not flag
    assert isinstance(counter, Counter)
    assert isinstance(flag, Flag)
    # Mutating one type's state never bleeds into another's.
    counter.n = 3
    assert ctx.state(Flag).on is False


def test_session_state_creates_lazy_default_and_is_identity_stable():
    # Exercise Session.state() directly, not via RunContext as the other tests do.
    session = Session()
    first = session.state(Counter)
    assert first.n == 0  # lazily constructed default
    first.n = 42
    assert session.state(Counter) is first  # identity stable across calls
    assert session.state(Counter).n == 42
