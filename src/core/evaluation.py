from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvaluationResult:
    """Result of one structured evaluation.

    ``answers`` maps each question key to its answer object. An answer's shape
    depends on its ``type``: ``noul`` carries ``noul`` (0-1); ``choice`` carries
    ``choice``/``probabilities``/``confidence``; ``score`` carries
    ``score``/``legend``/``probabilities``/``confidence``.
    """

    model: str = ""
    answers: dict[str, dict] = field(default_factory=dict)
    # token usage
    usage: dict = field(default_factory=dict)
    cost: float = 0.0
