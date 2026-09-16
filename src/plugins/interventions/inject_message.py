from __future__ import annotations

from dataclasses import dataclass

from core.message import Message
from protocols.mediator import Context


@dataclass
class InjectMessage:
    """Steer a running agent by adding a message before its next turn, so the
    next model call sees it. ``session.submit(InjectMessage("focus on X"))``.
    """

    content: str
    role: str = "user"

    def apply(self, ctx: Context) -> None:
        ctx.add_message(Message(role=self.role, content=self.content))
