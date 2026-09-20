from __future__ import annotations

from collections.abc import Awaitable
from typing import ClassVar

from protocols.context import Context
from protocols.plugin import Plugin


class Interceptor(Plugin):
    """Runs a side effect around every invocation of the plugin type it wraps."""

    target: ClassVar[type[Plugin]]

    def before(self, ctx: Context) -> Awaitable[None] | None:
        """React ahead of each wrapped invocation."""

    def after(self, ctx: Context) -> Awaitable[None] | None:
        """React after each wrapped invocation, even one that raised.

        Every ``after`` still runs when the invocation
        raises; the first error is re-raised once they have all run.
        """
