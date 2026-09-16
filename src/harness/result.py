from __future__ import annotations

from dataclasses import dataclass

from core.run import RunState
from harness.session import Session


@dataclass
class RunResult:
    """The outcome of a run."""

    output: str
    session: Session

    @property
    def stop_reason(self) -> str:
        """How the loop ended."""
        return self.session.state(RunState).stop_reason
