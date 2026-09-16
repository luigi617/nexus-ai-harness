from __future__ import annotations

from core.invoke import invoke
from protocols.mediator import Context


async def apply_interventions(ctx: Context) -> None:
    """Drain and apply any queued interventions at a loop's iteration boundary."""
    for intervention in ctx.take_interventions():
        await invoke(intervention.apply, ctx)
