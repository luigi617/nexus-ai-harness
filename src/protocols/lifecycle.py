from __future__ import annotations

from typing import Protocol, runtime_checkable

from protocols.context import Context


@runtime_checkable
class Lifecycle(Protocol):
    """A plugin that owns resources needing explicit setup and teardown.

    Lifecycle is orthogonal to a plugin's ``kind``: any plugin (a ``Model``,
    ``Tool``, ``Loop``, …) may also implement it to acquire resources such as
    HTTP clients, connections, or background tasks when the harness starts and
    release them when it stops.

    The harness calls ``start`` on every registered lifecycle plugin in
    registration order before the first run, and ``stop`` in reverse order on
    shutdown; register a dependency before the plugin that needs it so it is
    initialized first and torn down last. Both methods may be ``def`` or
    ``async def``, and either may be omitted — the harness runs whichever is
    present. Note that ``isinstance(plugin, Lifecycle)`` requires *both*
    methods, while the harness honors a partial implementation.
    """

    async def start(self, ctx: Context) -> None:
        """Acquire resources before the harness runs.

        Args:
            ctx: A context over the registry, so setup can resolve other
                plugins (via ``ctx.get`` / ``ctx.all``) for dependency-aware
                initialization. Its session is ephemeral — distinct from the
                one each run uses — so events or state raised here don't reach
                later runs.
        """
        ...

    async def stop(self) -> None:
        """Release resources acquired in :meth:`start`."""
        ...
