from __future__ import annotations

from abc import abstractmethod

from nexus_ai_harness.core.events import Event
from nexus_ai_harness.protocols.context import Context
from nexus_ai_harness.protocols.plugin import Plugin


class Hook(Plugin):
    @abstractmethod
    def on(self, event: Event, ctx: Context) -> None: ...
