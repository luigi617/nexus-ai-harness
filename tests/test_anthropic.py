from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

from core.message import Message
from core.response import Response
from plugins.models.anthropic import ANTHROPIC_VERSION, AnthropicModel
from plugins.models.base import BaseModel


def _tool(name="calc", description="d", parameters=None):
    return SimpleNamespace(
        name=name, description=description, parameters=parameters or {}
    )


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


def test_empty_assistant_turn_is_dropped():
    # An empty assistant turn would become content:[], which the API rejects.
    history = [
        Message(role="user", content="hi"),
        Message(role="assistant", content="", tool_calls=[]),
        Message(role="user", content="again"),
    ]
    _, messages = AnthropicModel._to_messages(history)
    # the empty assistant turn is gone, so the two user turns merge into one
    assert [m["role"] for m in messages] == ["user"]
    assert len(messages[0]["content"]) == 2


def test_assistant_tool_call_without_text_is_kept():
    _, messages = AnthropicModel._to_messages(
        [
            Message(
                role="assistant",
                content="",
                tool_calls=[{"id": "t1", "name": "calc", "arguments": {}}],
            )
        ]
    )
    assert len(messages) == 1
    assert messages[0]["content"][0]["type"] == "tool_use"


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
    mid, (pin, pout) = next(iter(AnthropicModel.pricing.items()))
    m = AnthropicModel(model=mid)
    cost = m._cost({"input_tokens": 1_000_000, "output_tokens": 1_000_000})
    assert abs(cost - (pin + pout)) < 1e-9


def test_cost_zero_for_unknown_model():
    m = AnthropicModel(model="unpriced")
    assert m._cost({"input_tokens": 999, "output_tokens": 999}) == 0.0


# --- _generate: request construction, headers, cost wiring ---------------


def _patch_post_json(monkeypatch, response):
    captured: dict = {}

    def fake_post_json(url, payload, headers, timeout):
        captured.update(url=url, payload=payload, headers=headers, timeout=timeout)
        return response

    monkeypatch.setattr("plugins.models.anthropic.post_json", fake_post_json)
    return captured


def test_generate_builds_request_headers_and_wires_cost(monkeypatch):
    canned = {
        "content": [{"type": "text", "text": "hi"}],
        "usage": {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
    }
    captured = _patch_post_json(monkeypatch, canned)

    mid, (pin, pout) = next(iter(AnthropicModel.pricing.items()))
    model = AnthropicModel(
        model=mid,
        api_key="sk-test",
        temperature=0.5,
    )
    history = [
        Message(role="system", content="sys"),
        Message(role="user", content="hi"),
    ]
    tool = _tool(parameters={"type": "object", "properties": {"x": {}}})
    result = model._generate(history, [tool])

    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["url"].endswith("/messages")
    assert captured["timeout"] == 60.0

    headers = captured["headers"]
    assert headers["anthropic-version"] == ANTHROPIC_VERSION
    assert headers["x-api-key"] == "sk-test"

    payload = captured["payload"]
    assert payload["model"] == mid
    assert payload["max_tokens"] == 1024
    assert payload["messages"][0]["role"] == "user"
    assert payload["system"] == "sys"  # present because a system msg exists
    assert payload["temperature"] == 0.5  # **self.params merged in
    assert payload["tools"] == [AnthropicModel._tool_spec(tool)]

    # cost = 1M*input + 1M*output for the selected pricing row
    assert abs(result.cost - (pin + pout)) < 1e-9
    assert result.text == "hi"


def test_generate_omits_api_key_system_and_tools_when_absent(monkeypatch):
    monkeypatch.setattr("plugins.models.base.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    captured = _patch_post_json(
        monkeypatch,
        {"content": [{"type": "text", "text": "ok"}], "usage": {}},
    )

    model = AnthropicModel(model="claude-3-5-haiku-20241022")
    model._generate([Message(role="user", content="hi")], [])

    assert "x-api-key" not in captured["headers"]
    assert "system" not in captured["payload"]  # no system message
    assert "tools" not in captured["payload"]  # no tools


def test_tool_spec_defaults_empty_schema():
    spec = AnthropicModel._tool_spec(_tool(parameters={}))
    assert spec["input_schema"] == {"type": "object", "properties": {}}
    assert spec["name"] == "calc"


# --- _to_messages edge cases: multi-system, ordering, final flush --------


def test_to_messages_joins_multiple_system_messages():
    history = [
        Message(role="system", content="a"),
        Message(role="system", content="b"),
        Message(role="user", content="hi"),
    ]
    system, _ = AnthropicModel._to_messages(history)
    assert system == "a\nb"


def test_to_messages_final_tool_results_flush_into_trailing_user_turn():
    history = [
        Message(role="user", content="go"),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                {"id": "t1", "name": "a", "arguments": {}},
                {"id": "t2", "name": "b", "arguments": {}},
            ],
        ),
        Message(role="tool", tool_use_id="t1", content="r1", name="a"),
        Message(role="tool", tool_use_id="t2", content="r2", name="b"),
    ]
    _, messages = AnthropicModel._to_messages(history)
    # both tool results batched into the final trailing user turn
    last = messages[-1]
    assert last["role"] == "user"
    ids = [b["tool_use_id"] for b in last["content"]]
    assert ids == ["t1", "t2"]


def test_to_messages_system_after_tool_results_flushes_first():
    history = [
        Message(
            role="assistant",
            content="",
            tool_calls=[{"id": "t1", "name": "a", "arguments": {}}],
        ),
        Message(role="tool", tool_use_id="t1", content="r1", name="a"),
        Message(role="system", content="late"),
        Message(role="assistant", content="after"),
    ]
    system, messages = AnthropicModel._to_messages(history)
    assert system == "late"
    roles = [m["role"] for m in messages]
    # tool_result user turn is ordered before the trailing assistant turn
    assert roles == ["assistant", "user", "assistant"]
    assert messages[1]["content"][0]["type"] == "tool_result"


# --- BaseModel.complete and constructor resolution -----------------------


class _FakeModel(BaseModel):
    provider = "fake"
    base_url = "https://fake.test/v1"
    api_key_env = "FAKE_API_KEY"
    descriptions: ClassVar[dict] = {"m1": "desc one"}

    def _generate(self, history, tools):
        self.received_tools = tools
        return Response(text="ok")


class _FakeCtx:
    def __init__(self, tools):
        self._tools = tools
        self.asked = None

    def all(self, cls):
        self.asked = cls
        return self._tools


def test_complete_forwards_tools_from_context():
    from protocols.tool import Tool

    tools = [_tool("one"), _tool("two")]
    ctx = _FakeCtx(tools)
    model = _FakeModel(model="m1")
    result = model.complete([Message(role="user", content="hi")], ctx)

    assert ctx.asked is Tool  # pulled the Tool list from the context
    assert model.received_tools is tools  # forwarded exactly
    assert result.text == "ok"


def test_init_api_key_precedence(monkeypatch):
    monkeypatch.setattr("plugins.models.base.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("FAKE_API_KEY", "envkey")

    assert _FakeModel(model="m1").api_key == "envkey"  # from env
    assert _FakeModel(model="m1", api_key="argkey").api_key == "argkey"  # arg wins


def test_init_empty_api_key_env_yields_none(monkeypatch):
    monkeypatch.setattr("plugins.models.base.load_dotenv", lambda *a, **k: None)

    class _NoEnvModel(_FakeModel):
        api_key_env = ""

    assert _NoEnvModel(model="m1").api_key is None


def test_init_base_url_override_stripped_and_description_and_params(monkeypatch):
    monkeypatch.setattr("plugins.models.base.load_dotenv", lambda *a, **k: None)
    m = _FakeModel(model="m1", base_url="https://proxy.local/v2/", foo=1, bar="b")
    assert m.base_url == "https://proxy.local/v2"
    assert m.description == "desc one"
    assert m.params == {"foo": 1, "bar": "b"}

    unknown = _FakeModel(model="unknown")
    assert unknown.description == ""
