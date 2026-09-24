from __future__ import annotations

from types import SimpleNamespace

from core.message import Message
from plugins.models.gemini import GeminiModel


def _tool(name="calc", description="d", parameters=None):
    return SimpleNamespace(
        name=name, description=description, parameters=parameters or {}
    )


def _patch_post_json(monkeypatch, response):
    captured: dict = {}

    def fake_post_json(url, payload, headers, timeout):
        captured.update(url=url, payload=payload, headers=headers, timeout=timeout)
        return response

    monkeypatch.setattr("plugins.models.gemini.post_json", fake_post_json)
    return captured


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


def test_empty_assistant_turn_is_dropped():
    # An empty assistant turn would yield an empty parts list, which the API rejects.
    history = [
        Message(role="user", content="hi"),
        Message(role="assistant", content="", tool_calls=[]),
        Message(role="user", content="again"),
    ]
    _, contents = GeminiModel._to_contents(history)
    assert [c["role"] for c in contents] == ["user"]  # user turns merge
    assert len(contents[0]["parts"]) == 2


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
    mid, (pin, pout) = next(iter(GeminiModel.pricing.items()))
    m = GeminiModel(model=mid)
    cost = m._cost({"input_tokens": 1_000_000, "output_tokens": 1_000_000})
    assert abs(cost - (pin + pout)) < 1e-9


def test_cost_zero_for_unknown_model():
    m = GeminiModel(model="unpriced")
    assert m._cost({"input_tokens": 999, "output_tokens": 999}) == 0.0


# --- _generate: URL construction, headers, cost wiring -------------------


def test_generate_builds_url_headers_and_wires_cost(monkeypatch):
    canned = {
        "candidates": [{"content": {"parts": [{"text": "hi"}]}}],
        "usageMetadata": {
            "promptTokenCount": 1_000_000,
            "candidatesTokenCount": 1_000_000,
        },
    }
    captured = _patch_post_json(monkeypatch, canned)

    mid, (pin, pout) = next(iter(GeminiModel.pricing.items()))
    model = GeminiModel(model=mid, api_key="gk-test", topP=0.9)
    history = [
        Message(role="system", content="sys"),
        Message(role="user", content="hi"),
    ]
    tool = _tool()
    result = model._generate(history, [tool])

    assert captured["url"] == (
        "https://generativelanguage.googleapis.com/v1beta"
        f"/models/{mid}:generateContent"
    )
    assert captured["headers"]["x-goog-api-key"] == "gk-test"

    payload = captured["payload"]
    assert payload["systemInstruction"] == {"parts": [{"text": "sys"}]}
    assert payload["tools"] == [
        {"functionDeclarations": [GeminiModel._tool_spec(tool)]}
    ]
    assert payload["generationConfig"]["maxOutputTokens"] == 1024
    assert payload["generationConfig"]["topP"] == 0.9  # **self.params merged

    assert abs(result.cost - (pin + pout)) < 1e-9


def test_generate_omits_api_key_system_and_tools_when_absent(monkeypatch):
    monkeypatch.setattr("plugins.models.base.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    captured = _patch_post_json(
        monkeypatch,
        {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]},
    )

    model = GeminiModel(model="gemini-1.5-flash")
    model._generate([Message(role="user", content="hi")], [])

    assert "x-goog-api-key" not in captured["headers"]
    assert "systemInstruction" not in captured["payload"]
    assert "tools" not in captured["payload"]


def test_tool_spec_defaults_empty_schema():
    spec = GeminiModel._tool_spec(_tool(parameters={}))
    assert spec["parameters"] == {"type": "object", "properties": {}}


# --- _parse defensive branches -------------------------------------------


def test_parse_empty_candidates_returns_empty_response():
    parsed = GeminiModel._parse({"candidates": []})
    assert parsed.text == ""
    assert parsed.tool_calls == []
    assert parsed.usage == {"input_tokens": 0, "output_tokens": 0}


def test_parse_two_function_calls_ids_encode_part_index():
    response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"functionCall": {"name": "f", "args": {"a": 1}}},
                        {"functionCall": {"name": "g", "args": {}}},
                    ]
                }
            }
        ]
    }
    parsed = GeminiModel._parse(response)
    assert [c["id"] for c in parsed.tool_calls] == ["call_f_0", "call_g_1"]
    assert [c["name"] for c in parsed.tool_calls] == ["f", "g"]
    # missing usageMetadata defaults to zeros
    assert parsed.usage == {"input_tokens": 0, "output_tokens": 0}


def test_parse_candidate_without_content_is_safe():
    parsed = GeminiModel._parse({"candidates": [{}]})
    assert parsed.text == ""
    assert parsed.tool_calls == []
