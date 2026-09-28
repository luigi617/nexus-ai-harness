from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable

from nexus_ai_harness.protocols.context import Context
from nexus_ai_harness.protocols.plugin import Plugin
from nexus_ai_harness.protocols.tool import Tool


class ToolProvider(Plugin):
    """A plugin that contributes :class:`~protocols.tool.Tool` s at runtime.

    Unlike a statically registered ``Tool``, a provider is asked for its tools
    when a run starts, so the set can depend on live state — a discovered MCP
    server's advertised tools, a filesystem scan, a feature flag. The loop
    expands every registered provider and merges the result with ``ctx.all(Tool)``
    into a single tool set for the turn, so provided tools are indistinguishable
    from statically registered ones once expanded.
    """

    @abstractmethod
    def provide_tools(self, ctx: Context) -> list[Tool] | Awaitable[list[Tool]]:
        """Return the tools this provider contributes for the current run.

        Called once per run when the loop assembles its tool set. May be ``def``
        or ``async def`` — the harness adapts — so a provider that must reach a
        remote endpoint can await it.

        Args:
            ctx: The run-scoped context, so discovery can resolve other plugins
                (e.g. a connected client via ``ctx.get``) or read run state.

        Returns:
            The tools to add to the run alongside ``ctx.all(Tool)``; empty when
            the provider currently has nothing to contribute.
        """
