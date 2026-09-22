from __future__ import annotations

from core.message import Message
from plugins.models.anthropic import AnthropicModel


def test_to_messages_splits_system_and_batches_tool_results():
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
    system, messages = AnthropicModel._to_messages(history)

    assert system == "sys"
    assert messages[0] == {"role": "user", "content": [{"type": "text", "text": "hi"}]}

    assistant = messages[1]
    assert assistant["role"] == "assistant"
    assert {"type": "text", "text": "thinking"} in assistant["content"]
    tool_use = assistant["content"][1]
    assert tool_use == {
        "type": "tool_use",
        "id": "t1",
        "name": "calc",
        "input": {"x": 1},
    }

    result_turn = messages[2]
    assert result_turn["role"] == "user"
    assert result_turn["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "t1",
        "content": "4",
    }


def test_to_messages_merges_consecutive_same_role_turns():
    history = [
        Message(role="user", content="task"),
        Message(role="user", content="summary"),
        Message(role="user", content="tail"),
    ]
    system, messages = AnthropicModel._to_messages(history)
    assert system == ""
    assert [m["role"] for m in messages] == ["user"]
    assert len(messages[0]["content"]) == 3


def test_parse_extracts_text_tool_calls_and_usage():
    response = {
        "content": [
            {"type": "text", "text": "hello"},
            {"type": "tool_use", "id": "u1", "name": "search", "input": {"q": "x"}},
        ],
        "usage": {"input_tokens": 12, "output_tokens": 5},
    }
    parsed = AnthropicModel._parse(response)
    assert parsed.text == "hello"
    assert parsed.tool_calls == [
        {"id": "u1", "name": "search", "arguments": {"q": "x"}}
    ]
    assert parsed.usage == {"input_tokens": 12, "output_tokens": 5}


def test_cost_uses_pricing_table():
    m = AnthropicModel(model="claude-3-5-haiku-20241022")
    cost = m._cost({"input_tokens": 1_000_000, "output_tokens": 1_000_000})
    assert abs(cost - (0.8 + 4.0)) < 1e-9


def test_cost_zero_for_unknown_model():
    m = AnthropicModel(model="unpriced")
    assert m._cost({"input_tokens": 999, "output_tokens": 999}) == 0.0
