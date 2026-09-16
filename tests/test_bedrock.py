from __future__ import annotations

from core.message import Message
from plugins.models.bedrock import BedrockModel


def test_to_converse_splits_system_and_batches_tool_results():
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
    system, messages = BedrockModel._to_converse(history)

    assert system == [{"text": "sys"}]
    assert messages[0] == {"role": "user", "content": [{"text": "hi"}]}

    assistant = messages[1]
    assert assistant["role"] == "assistant"
    assert {"text": "thinking"} in assistant["content"]
    tool_use = assistant["content"][1]["toolUse"]
    assert tool_use == {"toolUseId": "t1", "name": "calc", "input": {"x": 1}}

    # the tool result is folded into a following user turn, linked by toolUseId
    result_turn = messages[2]
    assert result_turn["role"] == "user"
    assert result_turn["content"][0]["toolResult"]["toolUseId"] == "t1"
    assert result_turn["content"][0]["toolResult"]["content"] == [{"text": "4"}]


def test_parse_extracts_text_tool_calls_and_usage():
    response = {
        "output": {
            "message": {
                "content": [
                    {"text": "hello"},
                    {
                        "toolUse": {
                            "toolUseId": "u1",
                            "name": "search",
                            "input": {"q": "x"},
                        }
                    },
                ]
            }
        },
        "usage": {"inputTokens": 12, "outputTokens": 5},
    }
    parsed = BedrockModel._parse(response)
    assert parsed.text == "hello"
    assert parsed.tool_calls == [
        {"id": "u1", "name": "search", "arguments": {"q": "x"}}
    ]
    assert parsed.usage == {"input_tokens": 12, "output_tokens": 5}


def test_cost_uses_pricing_table():
    p = BedrockModel(model="us.anthropic.claude-3-5-haiku-20241022-v1:0")
    cost = p._cost({"input_tokens": 1_000_000, "output_tokens": 1_000_000})
    assert abs(cost - (0.8 + 4.0)) < 1e-9  # $/1M in + $/1M out


def test_cost_zero_for_unknown_model():
    p = BedrockModel(model="some-unpriced-model")
    assert p._cost({"input_tokens": 999, "output_tokens": 999}) == 0.0
