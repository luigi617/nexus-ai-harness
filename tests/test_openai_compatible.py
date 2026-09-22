from __future__ import annotations

import json

from core.message import Message
from plugins.models import (
    DeepSeekModel,
    GLMModel,
    GroqModel,
    MiniMaxModel,
    OpenAIModel,
    QwenModel,
    XAIModel,
)


def test_to_messages_maps_roles_and_tool_calls():
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
    messages = OpenAIModel._to_messages(history)

    assert messages[0] == {"role": "system", "content": "sys"}
    assert messages[1] == {"role": "user", "content": "hi"}

    assistant = messages[2]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == "thinking"
    call = assistant["tool_calls"][0]
    assert call["id"] == "t1"
    assert call["function"]["name"] == "calc"
    # arguments are serialized to a JSON string per the OpenAI wire format
    assert json.loads(call["function"]["arguments"]) == {"x": 1}

    result = messages[3]
    assert result == {"role": "tool", "tool_call_id": "t1", "content": "4"}


def test_parse_extracts_text_tool_calls_and_usage():
    response = {
        "choices": [
            {
                "message": {
                    "content": "hello",
                    "tool_calls": [
                        {
                            "id": "u1",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": '{"q": "x"}',
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5},
    }
    parsed = OpenAIModel._parse(response)
    assert parsed.text == "hello"
    assert parsed.tool_calls == [
        {"id": "u1", "name": "search", "arguments": {"q": "x"}}
    ]
    assert parsed.usage == {"input_tokens": 12, "output_tokens": 5}


def test_parse_tolerates_null_content_and_bad_json_args():
    response = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {"id": "u1", "function": {"name": "f", "arguments": "not-json"}}
                    ],
                }
            }
        ]
    }
    parsed = OpenAIModel._parse(response)
    assert parsed.text == ""
    assert parsed.tool_calls == [{"id": "u1", "name": "f", "arguments": {}}]


def test_cost_uses_pricing_table():
    m = OpenAIModel(model="gpt-4o-mini")
    cost = m._cost({"input_tokens": 1_000_000, "output_tokens": 1_000_000})
    assert abs(cost - (0.15 + 0.6)) < 1e-9


def test_cost_zero_for_unknown_model():
    m = OpenAIModel(model="mystery-model")
    assert m._cost({"input_tokens": 999, "output_tokens": 999}) == 0.0


def test_provider_defaults_are_distinct():
    cases = {
        OpenAIModel: ("openai", "https://api.openai.com/v1", "OPENAI_API_KEY"),
        DeepSeekModel: ("deepseek", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
        MiniMaxModel: ("minimax", "https://api.minimax.chat/v1", "MINIMAX_API_KEY"),
        QwenModel: (
            "qwen",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "DASHSCOPE_API_KEY",
        ),
        GLMModel: ("glm", "https://open.bigmodel.cn/api/paas/v4", "ZHIPUAI_API_KEY"),
        GroqModel: ("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
        XAIModel: ("xai", "https://api.x.ai/v1", "XAI_API_KEY"),
    }
    for cls, (provider, base_url, key_env) in cases.items():
        assert cls.provider == provider
        assert cls.base_url == base_url
        assert cls.api_key_env == key_env


def test_base_url_override_and_trailing_slash_stripped():
    m = DeepSeekModel(model="deepseek-chat", base_url="https://proxy.local/v1/")
    assert m.base_url == "https://proxy.local/v1"
