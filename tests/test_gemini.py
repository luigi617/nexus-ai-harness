from __future__ import annotations

from core.message import Message
from plugins.models.gemini import GeminiModel


def test_to_contents_maps_roles_and_tool_flow():
    history = [
        Message(role="system", content="sys"),
        Message(role="user", content="hi"),
        Message(
            role="assistant",
            content="thinking",
            tool_calls=[{"id": "t1", "name": "calc", "arguments": {"x": 1}}],
        ),
        Message(role="tool", tool_use_id="t1", content="4", name="calc"),
    ]
    system, contents = GeminiModel._to_contents(history)

    assert system == "sys"
    assert contents[0] == {"role": "user", "parts": [{"text": "hi"}]}

    model_turn = contents[1]
    assert model_turn["role"] == "model"
    assert {"text": "thinking"} in model_turn["parts"]
    assert model_turn["parts"][1] == {
        "functionCall": {"name": "calc", "args": {"x": 1}}
    }

    result_turn = contents[2]
    assert result_turn["role"] == "user"
    assert result_turn["parts"][0] == {
        "functionResponse": {"name": "calc", "response": {"result": "4"}}
    }


def test_parse_extracts_text_tool_calls_and_usage():
    response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "hello"},
                        {"functionCall": {"name": "search", "args": {"q": "x"}}},
                    ]
                }
            }
        ],
        "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 5},
    }
    parsed = GeminiModel._parse(response)
    assert parsed.text == "hello"
    assert parsed.tool_calls == [
        {"id": "call_search_1", "name": "search", "arguments": {"q": "x"}}
    ]
    assert parsed.usage == {"input_tokens": 12, "output_tokens": 5}


def test_cost_uses_pricing_table():
    m = GeminiModel(model="gemini-1.5-flash")
    cost = m._cost({"input_tokens": 1_000_000, "output_tokens": 1_000_000})
    assert abs(cost - (0.075 + 0.3)) < 1e-9


def test_cost_zero_for_unknown_model():
    m = GeminiModel(model="unpriced")
    assert m._cost({"input_tokens": 999, "output_tokens": 999}) == 0.0
