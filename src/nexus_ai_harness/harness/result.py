from __future__ import annotations

from dataclasses import dataclass

from nexus_ai_harness.core.run import CostState, RunState
from nexus_ai_harness.harness.session import Session


@dataclass
class RunResult:
    """The outcome of a run."""

    output: str
    session: Session

    @property
    def stop_reason(self) -> str:
        """How the loop ended."""
        return self.session.state(RunState).stop_reason

    @property
    def cost(self) -> float:
        """The session's running cost in USD; ``0.0`` unless CostCounter is used."""
        return self.session.state(CostState).total

    @property
    def usage(self) -> dict[str, int]:
        """The session's running token totals; zero unless CostCounter is used.

        Uses the same ``input_tokens``/``output_tokens`` keys as ``Response.usage``.
        """
        state = self.session.state(CostState)
        return {
            "input_tokens": state.input_tokens,
            "output_tokens": state.output_tokens,
        }
