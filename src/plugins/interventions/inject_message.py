from __future__ import annotations

from dataclasses import dataclass

from core.message import Message
from protocols.context import Context
from protocols.intervention import Intervention


@dataclass
class InjectMessage(Intervention):
    """Steer a running agent by adding a message before its next turn.

    The next model call sees the injected message.
    """

    content: str
    role: str = "user"

    def apply(self, ctx: Context) -> None:
        ctx.add_message(Message(role=self.role, content=self.content))
