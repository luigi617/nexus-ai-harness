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


# --- evaluate() ----------------------------------------------------------


def test_evaluate_assembles_payload_auth_and_wires_cost(monkeypatch):
    import plugins.evaluators.jev as jevmod

    captured: dict = {}

    def fake_post(url, payload, headers, timeout):
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        captured["timeout"] = timeout
        return {
            "model": "jev-1.13.0",
            "answers": {"a": {"type": "noul", "noul": 0.5}},
            "usage": {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
        }

    monkeypatch.setattr(jevmod, "post_json", fake_post)

    jev = JevEvaluator(api_key="sekret", temperature=0.2)
    jev.pricing = {jev.name: (3.0, 15.0)}  # so cost is provably non-zero
    result = jev.evaluate({"draft": "hi"}, {"a": noul("ok?")}, None)

    assert captured["url"] == "https://api.typesafe.ai/v1/systemone"
    assert captured["payload"]["state"] == {"draft": "hi"}
    assert captured["payload"]["model"] == "jev-latest"
    assert captured["payload"]["questions"] == {"a": noul("ok?")}
    assert captured["payload"]["temperature"] == 0.2  # merged **params
    assert captured["headers"]["Authorization"] == "Bearer sekret"
    assert captured["timeout"] == 60.0

    # Response flows into the result.
    assert result.model == "jev-1.13.0"
    assert result.answers == {"a": {"type": "noul", "noul": 0.5}}
    assert result.usage == {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
    # evaluate() must wire cost = _cost(usage): (1M*3 + 1M*15)/1e6 == 18.0
    assert result.cost == 18.0


def test_evaluate_omits_auth_header_without_key(monkeypatch):
    import plugins.evaluators.jev as jevmod

    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    captured: dict = {}

    def fake_post(url, payload, headers, timeout):
        captured["headers"] = headers
        return {"model": "m", "answers": {}, "usage": {}}

    monkeypatch.setattr(jevmod, "post_json", fake_post)

    jev = JevEvaluator(api_key=None)
    jev.evaluate("state", {}, None)
    assert captured["headers"] == {}  # no Authorization when no key


# --- _parse / _cost defaults ---------------------------------------------


def test_parse_defaults_on_empty_response():
    result = JevEvaluator._parse({})
    assert result.model == ""
    assert result.answers == {}
    assert result.usage == {"input_tokens": 0, "output_tokens": 0}


def test_cost_applies_pricing_and_defaults_missing_tokens():
    jev = JevEvaluator()
    jev.pricing = {jev.name: (3.0, 15.0)}
    assert jev._cost({"input_tokens": 1_000_000, "output_tokens": 1_000_000}) == 18.0
    assert jev._cost({"input_tokens": 1_000_000}) == 3.0  # missing output -> 0
    assert jev._cost({}) == 0.0
