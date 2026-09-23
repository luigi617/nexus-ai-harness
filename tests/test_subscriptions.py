from __future__ import annotations

import pytest

from core.events import Event, MessageAdded, ModelCallStarted
from core.message import Message
from core.subscription import Subscription
from harness.registry import Registry
from protocols.hook import Hook
from protocols.plugin import Plugin
from tests.conftest import make_ctx


class Listener(Plugin):
    """A plugin that records the events routed to its bound handler."""

    def __init__(self) -> None:
        self.seen: list[Event] = []

    def handle(self, event: Event, ctx: object) -> None:
        self.seen.append(event)


class RecordingHook(Hook):
    def __init__(self, sink: list[Event]) -> None:
        self._sink = sink

    def on(self, event: Event, ctx: object) -> None:
        self._sink.append(event)


def _msg() -> MessageAdded:
    return MessageAdded(Message(role="user", content="hi"))


def test_on_infers_owner_from_bound_method():
    listener = Listener()
    sub = make_ctx().on(MessageAdded, listener.handle)
    assert sub.owner is listener


def test_on_has_no_owner_for_plain_function_outside_start():
    def handler(event: Event, ctx: object) -> None: ...

    assert make_ctx().on(MessageAdded, handler).owner is None


def test_subscribers_match_by_event_type():
    listener = Listener()
    r = Registry()
    r.subscribe(MessageAdded, listener.handle, listener)
    assert len(r.subscribers(_msg())) == 1
    assert r.subscribers(ModelCallStarted(history=[])) == []


def test_subscribers_match_event_subclasses():
    class Derived(MessageAdded): ...

    listener = Listener()
    r = Registry()
    r.subscribe(MessageAdded, listener.handle, listener)
    derived = Derived(Message(role="user", content="hi"))
    assert r.subscribers(derived)[0].handler == listener.handle


def test_subscription_remove_is_idempotent():
    r = Registry()
    sub = r.subscribe(MessageAdded, Listener().handle, None)
    sub.remove()
    sub.remove()  # a second removal must not raise
    assert r.subscriptions() == []


def test_remove_drops_only_the_named_plugins_subscriptions():
    keep, drop = Listener(), Listener()
    r = Registry()
    r.subscribe(MessageAdded, keep.handle, keep)
    r.subscribe(MessageAdded, drop.handle, drop)
    r.remove(drop)
    owners = [s.owner for s in r.subscriptions()]
    assert owners == [keep]


def test_remove_drops_every_subscription_of_one_owner():
    owner = Listener()
    r = Registry()
    r.subscribe(MessageAdded, owner.handle, owner)
    r.subscribe(ModelCallStarted, owner.handle, owner)
    r.remove(owner)
    assert r.subscriptions() == []


def test_ownerless_subscription_survives_plugin_removal_but_yields_to_its_handle():
    def handler(event: Event, ctx: object) -> None: ...

    owned = Listener()
    r = Registry()
    sub = r.subscribe(MessageAdded, handler, None)
    r.subscribe(MessageAdded, owned.handle, owned)
    r.remove(owned)
    assert r.subscriptions() == [sub]  # no owner, so plugin removal spares it
    sub.remove()
    assert r.subscriptions() == []


def test_ctx_on_routes_matching_events_to_handler():
    listener = Listener()
    ctx = make_ctx()
    ctx.on(MessageAdded, listener.handle)
    ctx.emit(_msg())
    ctx.emit(ModelCallStarted(history=[]))
    assert [type(e).__name__ for e in listener.seen] == ["MessageAdded"]


def test_emit_reaches_both_hooks_and_subscriptions():
    hook_seen: list[Event] = []
    listener = Listener()
    ctx = make_ctx(RecordingHook(hook_seen))
    ctx.on(MessageAdded, listener.handle)
    ctx.emit(_msg())
    assert len(hook_seen) == 1
    assert len(listener.seen) == 1


def test_handler_may_remove_a_subscription_mid_emit():
    # subscribers() returns a fresh list so a mid-emit cancel can't corrupt iteration.
    other = Listener()
    ctx = make_ctx()
    victim: list[Subscription] = []

    def self_removing(event: Event, _ctx: object) -> None:
        victim[0].remove()

    victim.append(ctx.on(MessageAdded, self_removing))
    ctx.on(MessageAdded, other.handle)
    ctx.emit(_msg())  # must not raise
    ctx.emit(_msg())
    assert len(other.seen) == 2


def test_handler_removing_a_different_handler_mid_emit_still_fires_it_this_round():
    # emit() iterates a snapshot, so a removed but pending handler still fires once.
    ctx = make_ctx()
    fired: list[str] = []
    victim: list[Subscription] = []

    def a(event: Event, _ctx: object) -> None:
        fired.append("a")
        victim[0].remove()  # cancel b during a's dispatch

    def b(event: Event, _ctx: object) -> None:
        fired.append("b")

    ctx.on(MessageAdded, a)  # registered first, so dispatched first
    victim.append(ctx.on(MessageAdded, b))

    ctx.emit(_msg())
    assert fired == ["a", "b"]  # b still fired despite a removing it (snapshot)

    fired.clear()
    ctx.emit(_msg())
    assert fired == ["a"]  # b's removal is now visible


def test_handler_adding_a_subscription_mid_emit_does_not_fire_it_until_next_emit():
    # Snapshot is taken before dispatch, so subscriptions added mid-emit won't fire yet.
    ctx = make_ctx()
    fired: list[str] = []

    def late(event: Event, _ctx: object) -> None:
        fired.append("late")

    def adder(event: Event, _ctx: object) -> None:
        fired.append("adder")
        ctx.on(MessageAdded, late)

    ctx.on(MessageAdded, adder)

    ctx.emit(_msg())
    assert fired == ["adder"]  # late was added mid-emit, so it did not fire yet

    fired.clear()
    ctx.emit(_msg())
    assert fired == ["adder", "late"]  # now the newly-added handler fires


def test_fork_inherits_owned_subscription_so_it_fires_in_the_child():
    owner = Listener()
    ctx = make_ctx(owner)
    ctx.on(MessageAdded, owner.handle)
    ctx.fork().emit(_msg())
    assert len(owner.seen) == 1


def test_fork_inherits_ownerless_subscription():
    seen: list[Event] = []
    ctx = make_ctx()
    ctx.on(MessageAdded, lambda event, _ctx: seen.append(event))
    ctx.fork().emit(_msg())
    assert len(seen) == 1


def test_fork_excluding_the_owner_drops_its_subscription():
    owner, other = Listener(), Listener()
    ctx = make_ctx(owner, other)
    ctx.on(MessageAdded, owner.handle)
    ctx.fork(plugins=[other]).emit(_msg())  # owner not in the child
    assert owner.seen == []


def test_async_handler_raises_instead_of_silently_doing_nothing():
    async def handler(event: Event, ctx: object) -> None: ...

    ctx = make_ctx()
    ctx.on(MessageAdded, handler)
    with pytest.raises(TypeError, match="synchronous"):
        ctx.emit(_msg())


def test_async_generator_handler_also_raises():
    async def handler(event: Event, ctx: object):
        yield  # an async generator, not a coroutine

    ctx = make_ctx()
    ctx.on(MessageAdded, handler)
    with pytest.raises(TypeError, match="synchronous"):
        ctx.emit(_msg())
