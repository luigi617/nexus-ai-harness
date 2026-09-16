from __future__ import annotations

from tests.conftest import make_ctx

from plugins.guards import BudgetGuard, MaxIterations, Timeout
from plugins.hooks import CostState, ElapsedState, IterationState
from services.guard_chain import GuardChain


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
