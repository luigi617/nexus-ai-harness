from __future__ import annotations

import json
from types import SimpleNamespace

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
from plugins.models.openai_compatible import OpenAICompatibleModel


def _tool(name="calc", description="d", parameters=None):
    return SimpleNamespace(
        name=name, description=description, parameters=parameters or {}
    )


def _patch_post_json(monkeypatch, response):
    captured: dict = {}

    def fake_post_json(url, payload, headers, timeout):
        captured.update(url=url, payload=payload, headers=headers, timeout=timeout)
        return response

    monkeypatch.setattr("plugins.models.openai_compatible.post_json", fake_post_json)
    return captured


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


def test_empty_assistant_turn_is_dropped():
    # A natural-exit turn (empty, no tool calls) must never be replayed to the API.
    history = [
        Message(role="user", content="hi"),
        Message(role="assistant", content="", tool_calls=[]),
        Message(role="user", content="again"),
    ]
    messages = OpenAIModel._to_messages(history)
    assert [m["role"] for m in messages] == ["user", "user"]


def test_assistant_turn_with_tool_calls_but_no_text_is_kept():
    history = [
        Message(
            role="assistant",
            content="",
            tool_calls=[{"id": "t1", "name": "calc", "arguments": {}}],
        )
    ]
    messages = OpenAIModel._to_messages(history)
    assert len(messages) == 1
    assert messages[0]["content"] is None
    assert messages[0]["tool_calls"][0]["id"] == "t1"


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

    # No two providers share a provider id, base URL, or key env var.
    providers = [cls.provider for cls in cases]
    base_urls = [cls.base_url for cls in cases]
    key_envs = [cls.api_key_env for cls in cases]
    assert len(set(providers)) == len(providers)
    assert len(set(base_urls)) == len(base_urls)
    assert len(set(key_envs)) == len(key_envs)


def test_base_url_override_and_trailing_slash_stripped():
    m = DeepSeekModel(model="deepseek-chat", base_url="https://proxy.local/v1/")
    assert m.base_url == "https://proxy.local/v1"


# --- _generate: routing, auth header, params, cost -----------------------

_CANNED = {
    "choices": [{"message": {"content": "hi"}}],
    "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000},
}


def test_generate_routes_to_chat_completions_with_auth_and_params(monkeypatch):
    captured = _patch_post_json(monkeypatch, _CANNED)

    model = OpenAIModel(model="gpt-4o-mini", api_key="sk-test", temperature=0.3)
    tool = _tool()
    result = model._generate([Message(role="user", content="hi")], [tool])

    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["url"].endswith("/chat/completions")
    assert captured["url"].startswith("https://api.openai.com/v1")
    assert captured["headers"]["Authorization"] == "Bearer sk-test"

    payload = captured["payload"]
    assert payload["model"] == "gpt-4o-mini"
    assert payload["max_tokens"] == 1024
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert payload["temperature"] == 0.3  # **self.params merged
    assert payload["tools"] == [OpenAIModel._tool_spec(tool)]

    assert abs(result.cost - (0.15 + 0.6)) < 1e-9


def test_generate_subclass_routes_to_its_base_url(monkeypatch):
    captured = _patch_post_json(monkeypatch, _CANNED)

    model = DeepSeekModel(model="deepseek-chat", api_key="ds-key")
    model._generate([Message(role="user", content="hi")], [])

    assert captured["url"] == "https://api.deepseek.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer ds-key"
    # no tools passed -> no tools key on the payload
    assert "tools" not in captured["payload"]


def test_generate_omits_authorization_when_no_api_key(monkeypatch):
    monkeypatch.setattr("plugins.models.base.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    captured = _patch_post_json(monkeypatch, _CANNED)

    model = OpenAIModel(model="gpt-4o-mini")
    model._generate([Message(role="user", content="hi")], [])
    assert "Authorization" not in captured["headers"]


def test_all_thin_providers_route_to_their_host(monkeypatch):
    expected = {
        OpenAIModel: "https://api.openai.com/v1/chat/completions",
        DeepSeekModel: "https://api.deepseek.com/v1/chat/completions",
        MiniMaxModel: "https://api.minimax.chat/v1/chat/completions",
        QwenModel: (
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
        ),
        GLMModel: "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        GroqModel: "https://api.groq.com/openai/v1/chat/completions",
        XAIModel: "https://api.x.ai/v1/chat/completions",
    }
    for cls, url in expected.items():
        captured = _patch_post_json(monkeypatch, _CANNED)
        model = cls(model="x", api_key="k")
        # base_url survives construction (rstrip is a no-op here)
        assert model.base_url == url.removesuffix("/chat/completions")
        model._generate([Message(role="user", content="hi")], [])
        assert captured["url"] == url


def test_direct_openai_compatible_uses_overridden_base_url(monkeypatch):
    captured = _patch_post_json(monkeypatch, _CANNED)
    model = OpenAICompatibleModel(
        model="x", api_key="k", base_url="https://proxy.local/v9/"
    )
    model._generate([Message(role="user", content="hi")], [])
    assert captured["url"] == "https://proxy.local/v9/chat/completions"


# --- provider descriptions and pricing tables ----------------------------


def test_thin_provider_descriptions_populated():
    assert (
        DeepSeekModel(model="deepseek-chat").description == "general-purpose chat model"
    )
    assert (
        DeepSeekModel(model="deepseek-reasoner").description
        == "reasoning-optimized model"
    )
    assert XAIModel(model="grok-4").description == "xAI flagship reasoning model"


def test_openai_pricing_yields_expected_cost():
    m = OpenAIModel(model="gpt-4o")
    cost = m._cost({"input_tokens": 1_000_000, "output_tokens": 1_000_000})
    assert abs(cost - (2.5 + 10.0)) < 1e-9


# --- _to_messages and _parse edge cases ----------------------------------


def test_to_messages_tool_only_assistant_sets_content_none():
    history = [
        Message(
            role="assistant",
            content="",
            tool_calls=[{"id": "t1", "name": "f", "arguments": {"a": 1}}],
        ),
    ]
    messages = OpenAIModel._to_messages(history)
    assert messages[0]["content"] is None
    assert len(messages[0]["tool_calls"]) == 1
    assert messages[0]["tool_calls"][0]["id"] == "t1"


def test_to_messages_assistant_with_content_and_no_tool_calls():
    history = [Message(role="assistant", content="just text")]
    messages = OpenAIModel._to_messages(history)
    assert messages[0]["content"] == "just text"
    assert "tool_calls" not in messages[0]


def test_parse_empty_choices_returns_empty_response():
    parsed = OpenAIModel._parse({})
    assert parsed.text == ""
    assert parsed.tool_calls == []
    assert parsed.usage == {"input_tokens": 0, "output_tokens": 0}


def test_tool_spec_defaults_empty_schema():
    spec = OpenAIModel._tool_spec(_tool(parameters={}))
    assert spec["function"]["parameters"] == {
        "type": "object",
        "properties": {},
    }
