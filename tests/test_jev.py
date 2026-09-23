from __future__ import annotations

from core.evaluation import EvaluationResult
from plugins.evaluators import JevEvaluator, choice, noul, score
from protocols.evaluator import Evaluator
from protocols.model import Model


def test_jev_is_an_evaluator_not_a_model():
    jev = JevEvaluator()
    assert isinstance(jev, Evaluator)
    assert not isinstance(jev, Model)  # invisible to ctx.get(Model)
    assert jev.provider == "typesafe"
    assert jev.name == "jev-latest"


def test_question_builders():
    assert noul("urgent?") == {"type": "noul", "instructions": "urgent?"}
    assert noul("urgent?", {"true": "is", "false": "not"}) == {
        "type": "noul",
        "instructions": "urgent?",
        "criteria": {"true": "is", "false": "not"},
    }
    assert choice("pick", {"a": "first", "b": "second"}) == {
        "type": "choice",
        "instructions": "pick",
        "criteria": {"a": "first", "b": "second"},
    }
    assert score("rate", ["low", "high"]) == {
        "type": "score",
        "instructions": "rate",
        "criteria": ["low", "high"],
    }


def test_parse_extracts_answers_and_usage():
    response = {
        "model": "jev-1.13.0",
        "answers": {"is_urgent": {"type": "noul", "noul": 0.95}},
        "usage": {"input_tokens": 296, "output_tokens": 20},
    }
    result = JevEvaluator._parse(response)
    assert isinstance(result, EvaluationResult)
    assert result.model == "jev-1.13.0"
    assert result.answers == {"is_urgent": {"type": "noul", "noul": 0.95}}
    assert result.usage == {"input_tokens": 296, "output_tokens": 20}


def test_cost_zero_without_pricing():
    jev = JevEvaluator()
    assert jev._cost({"input_tokens": 999, "output_tokens": 999}) == 0.0


def test_base_url_override_strips_trailing_slash():
    jev = JevEvaluator(base_url="https://proxy.local/v1/")
    assert jev.base_url == "https://proxy.local/v1"
