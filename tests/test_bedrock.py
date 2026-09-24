from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

from core.message import Message
from plugins.models import bedrock
from plugins.models.bedrock import DEFAULT_REGION, BedrockModel


def _tool(name="calc", description="d", parameters=None):
    return SimpleNamespace(
        name=name, description=description, parameters=parameters or {}
    )


def _no_dotenv(monkeypatch):
    """Neutralize .env loading so region/key resolution is deterministic."""
    monkeypatch.setattr(bedrock, "load_dotenv", lambda *a, **k: None)


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


def test_empty_assistant_turn_is_dropped():
    # An empty assistant turn would yield an empty content list, which Converse rejects.
    history = [
        Message(role="user", content="hi"),
        Message(role="assistant", content="", tool_calls=[]),
        Message(role="user", content="again"),
    ]
    _, messages = BedrockModel._to_converse(history)
    assert [m["role"] for m in messages] == ["user"]  # user turns merge
    assert len(messages[0]["content"]) == 2


def test_to_converse_merges_consecutive_same_role_turns():
    # An injected summary sits next to the first user turn; Converse requires
    # alternating roles, so consecutive same-role turns must be merged.
    history = [
        Message(role="system", content="sys"),
        Message(role="user", content="task"),
        Message(role="user", content="summary"),
        Message(role="user", content="tail"),
    ]
    system, messages = BedrockModel._to_converse(history)
    assert system == [{"text": "sys"}]
    assert [m["role"] for m in messages] == ["user"]  # collapsed to one turn
    assert len(messages[0]["content"]) == 3


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
    mid, (pin, pout) = next(iter(BedrockModel.pricing.items()))
    p = BedrockModel(model=mid)
    cost = p._cost({"input_tokens": 1_000_000, "output_tokens": 1_000_000})
    assert abs(cost - (pin + pout)) < 1e-9  # $/1M in + $/1M out


def test_cost_zero_for_unknown_model():
    p = BedrockModel(model="some-unpriced-model")
    assert p._cost({"input_tokens": 999, "output_tokens": 999}) == 0.0


# --- __init__: region precedence -----------------------------------------


def test_region_defaults_to_us_east_1(monkeypatch):
    _no_dotenv(monkeypatch)
    monkeypatch.delenv("AWS_REGION", raising=False)
    assert BedrockModel(model="m").region == DEFAULT_REGION == "us-east-1"


def test_region_from_env_when_no_arg(monkeypatch):
    _no_dotenv(monkeypatch)
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    assert BedrockModel(model="m").region == "eu-west-1"


def test_region_arg_wins_over_env(monkeypatch):
    _no_dotenv(monkeypatch)
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    assert BedrockModel(model="m", region="ap-south-1").region == "ap-south-1"


# --- _get_client: caching, region wiring, api_key global mutation --------


def test_get_client_is_lazy_and_cached(monkeypatch):
    _no_dotenv(monkeypatch)
    fake_client = MagicMock()
    monkeypatch.setattr(bedrock.boto3, "client", fake_client)

    model = BedrockModel(model="m", region="us-east-1")
    # constructor does not create the client yet
    assert fake_client.call_count == 0

    c1 = model._get_client()
    c2 = model._get_client()
    assert c1 is c2  # cached
    assert fake_client.call_count == 1  # boto3.client called once
    args, kwargs = fake_client.call_args
    assert args == ("bedrock-runtime",)
    assert kwargs["region_name"] == "us-east-1"


def test_get_client_wires_timeout_into_boto_config(monkeypatch):
    _no_dotenv(monkeypatch)
    factory = MagicMock()
    monkeypatch.setattr(bedrock.boto3, "client", factory)

    BedrockModel(model="m", region="us-east-1", timeout=5.0)._get_client()
    config = factory.call_args.kwargs["config"]
    assert config.connect_timeout == 5.0
    assert config.read_timeout == 5.0


def test_get_client_sets_global_bearer_token_env(monkeypatch):
    _no_dotenv(monkeypatch)
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.setattr(bedrock.boto3, "client", MagicMock())

    model = BedrockModel(model="m", api_key="secret-token")
    model._get_client()
    # documented side effect: the key is written into process env (global)
    assert os.environ["AWS_BEARER_TOKEN_BEDROCK"] == "secret-token"


# --- _generate: converse kwargs and cost wiring --------------------------


def _fake_converse_client(monkeypatch, response):
    client = MagicMock()
    client.converse.return_value = response
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(bedrock.boto3, "client", factory)
    return client


def test_generate_builds_converse_kwargs_and_wires_cost(monkeypatch):
    _no_dotenv(monkeypatch)
    canned = {
        "output": {"message": {"content": [{"text": "hi"}]}},
        "usage": {"inputTokens": 1_000_000, "outputTokens": 1_000_000},
    }
    client = _fake_converse_client(monkeypatch, canned)

    mid, (pin, pout) = next(iter(BedrockModel.pricing.items()))
    model = BedrockModel(model=mid, region="us-east-1")
    history = [
        Message(role="system", content="sys"),
        Message(role="user", content="hi"),
    ]
    tool = _tool()
    result = model._generate(history, [tool])

    kwargs = client.converse.call_args.kwargs
    assert kwargs["modelId"] == mid
    assert kwargs["inferenceConfig"]["maxTokens"] == 1024
    assert kwargs["system"] == [{"text": "sys"}]  # present: system msg exists
    assert kwargs["toolConfig"] == {
        "tools": [BedrockModel._tool_spec(tool)]
    }  # present: tools given
    assert abs(result.cost - (pin + pout)) < 1e-9


def test_generate_omits_system_and_toolconfig_when_absent(monkeypatch):
    _no_dotenv(monkeypatch)
    client = _fake_converse_client(
        monkeypatch,
        {"output": {"message": {"content": [{"text": "ok"}]}}, "usage": {}},
    )
    model = BedrockModel(model="m", region="us-east-1")
    model._generate([Message(role="user", content="hi")], [])

    kwargs = client.converse.call_args.kwargs
    assert "system" not in kwargs
    assert "toolConfig" not in kwargs


# --- _to_converse edge cases: multi-system, ordering, final flush --------


def test_to_converse_keeps_multiple_system_messages_separate():
    history = [
        Message(role="system", content="a"),
        Message(role="system", content="b"),
        Message(role="user", content="hi"),
    ]
    system, _ = BedrockModel._to_converse(history)
    assert system == [{"text": "a"}, {"text": "b"}]


def test_to_converse_final_tool_results_flush_into_trailing_user_turn():
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
    _, messages = BedrockModel._to_converse(history)
    last = messages[-1]
    assert last["role"] == "user"
    ids = [b["toolResult"]["toolUseId"] for b in last["content"]]
    assert ids == ["t1", "t2"]


def test_to_converse_tool_result_ordered_before_later_assistant_turn():
    history = [
        Message(
            role="assistant",
            content="",
            tool_calls=[{"id": "t1", "name": "a", "arguments": {}}],
        ),
        Message(role="tool", tool_use_id="t1", content="r1", name="a"),
        Message(role="assistant", content="after"),
    ]
    _, messages = BedrockModel._to_converse(history)
    roles = [m["role"] for m in messages]
    assert roles == ["assistant", "user", "assistant"]
    assert "toolResult" in messages[1]["content"][0]


# --- suspected source bug: timeout kwarg pollutes params (SEE bugsFound) --


def test_bedrock_generate_excludes_timeout_from_inference_config(monkeypatch):
    # timeout is a real constructor param, so it must not leak into
    # inferenceConfig (which would be an invalid Converse member).
    _no_dotenv(monkeypatch)
    client = _fake_converse_client(
        monkeypatch,
        {"output": {"message": {"content": [{"text": "ok"}]}}, "usage": {}},
    )
    model = BedrockModel(model="m", region="us-east-1", timeout=30)
    model._generate([Message(role="user", content="hi")], [])
    assert client.converse.call_args.kwargs["inferenceConfig"] == {"maxTokens": 1024}


# --- BedrockModel timeout kwarg ------------------------------------------


def test_bedrock_honors_timeout_kwarg(monkeypatch):
    # A timeout kwarg must be honored, not swept into params where it would
    # pollute inferenceConfig (Converse rejects unknown members).
    _no_dotenv(monkeypatch)
    model = BedrockModel(model="us.anthropic.claude-opus-4-8", timeout=30)
    assert model.timeout == 30
    assert "timeout" not in model.params
