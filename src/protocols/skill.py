from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable
from typing import ClassVar

from core.events import SkillInvoked
from core.invoke import call
from protocols.context import Context
from protocols.tool import Tool


class Skill(Tool):
    """A named instruction set the model loads on demand by calling it as a tool.

    Calling the skill returns its full ``instructions`` into the conversation.
    A skill *is* a :class:`~protocols.tool.Tool`, so it's registered and used
    like any other (``harness.use(my_skill)``) and needs no separate loader.
    Progressive disclosure falls out for free: the model sees only each skill's
    ``name`` and one-line ``description`` in its tool list until it calls one, at
    which point the body is loaded to steer the following turns. Authors supply
    ``name``, ``description``, and :meth:`instructions`; calling the skill takes
    no arguments, and the base :meth:`run` handles loading and the event.
    """

    # No arguments: invoking the skill *is* the request to load it.
    parameters: ClassVar[dict] = {"type": "object", "properties": {}}

    @abstractmethod
    def instructions(self) -> str | Awaitable[str]:
        """Return the full instruction body, loaded into context on invocation.

        Read lazily — only called when the skill is invoked — so a file-backed
        skill need not hold its body in memory until then. May be ``def`` or
        ``async def``; the harness adapts.
        """

    async def run(self, arguments: dict, ctx: Context) -> str:
        # Emit before reading the body so a failed lazy load is still observable.
        ctx.emit(SkillInvoked(self.name))
        # Adapt sync/async directly: instructions() is an internal step of this
        # tool call, not a second entrypoint that should re-run interceptors.
        body = await call(self.instructions)
        return f"Loaded skill {self.name!r}. Follow these instructions:\n\n{body}"
