from __future__ import annotations

from collections.abc import Awaitable
from typing import ClassVar

from nexus_ai_harness.core.invocation import Invocation, InvocationOutcome
from nexus_ai_harness.protocols.context import Context
from nexus_ai_harness.protocols.plugin import Plugin


class Interceptor(Plugin):
    """Runs a side effect around every invocation of the plugin type it wraps.

    An older override of ``before(self, ctx)`` or ``after(self, ctx)``, without
    invocation metadata, still works: the harness detects which signature an
    override uses and calls it accordingly.
    """

    target: ClassVar[type[Plugin]]

    def before(self, invocation: Invocation, ctx: Context) -> Awaitable[None] | None:
        """React ahead of each wrapped invocation."""

    def after(
        self, invocation: Invocation, outcome: InvocationOutcome, ctx: Context
    ) -> Awaitable[None] | None:
        """React after each wrapped invocation, even one that raised.

        Every ``after`` still runs when the invocation
        raises; the first error is re-raised once they have all run.
        """
