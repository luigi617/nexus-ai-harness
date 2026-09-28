from __future__ import annotations

from abc import abstractmethod

from nexus_ai_harness.core.message import Message
from nexus_ai_harness.protocols.plugin import Plugin


class TokenEstimator(Plugin):
    """Estimates how many input tokens a list of messages costs a model."""

    @abstractmethod
    def estimate(self, messages: list[Message]) -> int:
        """Estimate the prompt tokens ``messages`` would consume.

        Called on every turn, so it must be synchronous and cheap. It need not
        be exact; context managers use it to decide when to compact.

        Args:
            messages: The messages as they would be sent, oldest first.

        Returns:
            A non-negative token estimate.
        """
