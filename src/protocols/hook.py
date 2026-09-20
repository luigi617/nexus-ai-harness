from __future__ import annotations

from abc import abstractmethod

from core.events import Event
from protocols.context import Context
from protocols.plugin import Plugin


class Hook(Plugin):
    @abstractmethod
    def on(self, event: Event, ctx: Context) -> None: ...
