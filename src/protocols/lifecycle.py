from __future__ import annotations

from protocols.context import Context


class Lifecycle:
    """Mixin base for a plugin that owns resources needing setup and teardown.

    Subclass it (alongside a plugin base, e.g. ``class DbTool(Tool,
    Lifecycle)``) to acquire resources such as HTTP clients, connections, or
    background tasks when the harness starts and release them when it stops.

    The harness calls ``start`` on those plugins in registration order before
    the first run, and ``stop`` in reverse order on shutdown; register a
    dependency before the plugin that needs it so it is initialized first and
    torn down last. Override only the hook you need — both default to no-ops —
    and either may be ``def`` or ``async def``; the harness adapts.
    """

    async def start(self, ctx: Context) -> None:
        """Acquire resources before the harness runs. Override to implement.

        Args:
            ctx: A context over the registry, so setup can resolve other
                plugins (via ``ctx.get`` / ``ctx.all``) for dependency-aware
                initialization. Its session is ephemeral — distinct from the
                one each run uses — so events or state raised here don't reach
                later runs.
        """

    async def stop(self) -> None:
        """Release resources acquired in :meth:`start`. Override to implement."""
