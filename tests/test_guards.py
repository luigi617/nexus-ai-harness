from __future__ import annotations

import asyncio

import pytest

from core.message import Message
from core.response import Response
from core.run import RunState
from harness.harness import NexusAIHarness
from harness.validation import MissingDependencyError
from plugins.guards import BudgetGuard, MaxIterations, Timeout
from plugins.hooks import CostCounter, CostState, ElapsedState, IterationState
from plugins.loops import AgenticLoop
from services.guard_chain import GuardChain
from tests.conftest import ScriptedModel, make_ctx


def test_max_iterations_halts_at_limit():
    ctx = make_ctx()
    g = MaxIterations(3)
    ctx.state(IterationState).index = 2
    assert not g.check(ctx).stop
    ctx.state(IterationState).index = 3
    assert g.check(ctx).stop


def test_timeout_halts_past_budget():
    ctx = make_ctx()
    g = Timeout(10)
    ctx.state(ElapsedState).elapsed = 9.9
    assert not g.check(ctx).stop
    ctx.state(ElapsedState).elapsed = 10.0
    assert g.check(ctx).stop


def test_budget_guard_halts_over_cost():
    ctx = make_ctx()
    g = BudgetGuard(0.5)
    ctx.state(CostState).total = 0.49
    assert not g.check(ctx).stop
    ctx.state(CostState).total = 0.5
    assert g.check(ctx).stop


def test_guard_chain_stops_on_first_halt():
    ctx = make_ctx(MaxIterations(1), Timeout(100))
    ctx.state(IterationState).index = 5
    decision = GuardChain().check(ctx)
    assert decision.stop and "iterations" in decision.reason


def test_guard_chain_proceeds_when_all_pass():
    ctx = make_ctx(MaxIterations(10), Timeout(100))
    assert not GuardChain().check(ctx).stop


# --- empty and multi-guard chains (finding 10) ---------------------------


def test_guard_chain_with_no_guards_proceeds():
    # The default configuration of any loop with no guards registered.
    decision = GuardChain().check(make_ctx())
    assert decision.stop is False


def test_guard_chain_continues_past_a_proceeding_guard_to_a_later_halt():
    # First guard proceeds, second halts; chain must return the halting guard's reason.
    ctx = make_ctx(MaxIterations(100), Timeout(10))
    ctx.state(IterationState).index = 0  # MaxIterations proceeds
    ctx.state(ElapsedState).elapsed = 20  # Timeout halts
    decision = GuardChain().check(ctx)
    assert decision.stop
    assert decision.reason == "exceeded 10s time budget"


# --- guards fail open when their required Hook is missing (finding 3) ----


def _harness_with(*guards) -> NexusAIHarness:
    h = NexusAIHarness().use(AgenticLoop()).use(ScriptedModel(Response(text="hi")))
    for guard in guards:
        h.use(guard)
    return h


def test_validate_flags_max_iterations_missing_counter():
    with pytest.raises(MissingDependencyError) as exc:
        _harness_with(MaxIterations(2)).validate()
    assert ("MaxIterations", "IterationCounter") in exc.value.missing


def test_validate_flags_budget_guard_missing_cost_counter():
    with pytest.raises(MissingDependencyError) as exc:
        _harness_with(BudgetGuard(1.0)).validate()
    assert ("BudgetGuard", "CostCounter") in exc.value.missing


def test_validate_flags_timeout_missing_elapsed_time():
    with pytest.raises(MissingDependencyError) as exc:
        _harness_with(Timeout(1.0)).validate()
    assert ("Timeout", "ElapsedTime") in exc.value.missing


def test_run_auto_validates_a_guards_requires():
    # A guard missing its hook fails closed with MissingDependencyError, not silently.
    with pytest.raises(MissingDependencyError):
        _harness_with(MaxIterations(2)).run_sync("q")


# --- guards must not fail open when their required Hook is missing --------


def test_start_validates_declared_guard_dependency():
    # A guard registered without its required hook must be caught at start().
    harness = (
        NexusAIHarness()
        .use(AgenticLoop())
        .use(ScriptedModel())
        .use(MaxIterations(3))  # requires IterationCounter, which is absent
    )
    with pytest.raises(MissingDependencyError):
        asyncio.run(harness.start())


# --- BudgetGuard must account the terminating response's cost -------------


def test_budget_guard_rejects_single_over_budget_response():
    # A single response that blows the budget must not report as a clean completion.
    model = ScriptedModel(Response(text="expensive", cost=10.0))
    ctx = make_ctx(model, CostCounter(), BudgetGuard(0.5))
    ctx.add_message(Message(role="user", content="hi"))
    asyncio.run(AgenticLoop().run(ctx))
    assert ctx.state(RunState).stop_reason != "completed"
