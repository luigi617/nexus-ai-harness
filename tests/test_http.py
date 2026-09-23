from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

import pytest

from plugins.models._http import post_json


class _FakeResponse:
    """Minimal context-manager stand-in for the urlopen return value."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


def test_post_json_builds_post_request_and_decodes_body(monkeypatch):
    captured: dict = {}

    def fake_urlopen(request, timeout=None):
        captured["request"] = request
        captured["timeout"] = timeout
        return _FakeResponse(b'{"ok": true}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    payload = {"a": 1, "b": ["x"]}
    result = post_json(
        "https://api.test/v1/thing",
        payload,
        {"X-Custom": "v"},
        timeout=12.5,
    )

    # success path returns the decoded + json.loads'd body
    assert result == {"ok": True}

    request = captured["request"]
    assert request.get_method() == "POST"
    assert request.full_url == "https://api.test/v1/thing"
    # body is the JSON-encoded payload
    assert request.data == json.dumps(payload).encode()
    # Content-Type is injected and the caller header is preserved (not clobbered)
    assert request.get_header("Content-type") == "application/json"
    assert request.get_header("X-custom") == "v"
    # timeout is propagated to urlopen
    assert captured["timeout"] == 12.5


def test_post_json_caller_header_can_override_content_type(monkeypatch):
    captured: dict = {}

    def fake_urlopen(request, timeout=None):
        captured["request"] = request
        return _FakeResponse(b"{}")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    # A caller Content-Type header wins over the default (dict merge order).
    post_json("https://api.test/x", {}, {"Content-Type": "application/xml"})
    assert captured["request"].get_header("Content-type") == "application/xml"


def test_post_json_default_timeout_is_60(monkeypatch):
    captured: dict = {}

    def fake_urlopen(request, timeout=None):
        captured["timeout"] = timeout
        return _FakeResponse(b"{}")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    post_json("https://api.test/x", {}, {})
    assert captured["timeout"] == 60.0


def test_post_json_raises_runtimeerror_on_http_error(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            "https://api.test/v1/thing",
            429,
            "Too Many Requests",
            {},
            io.BytesIO(b"boom"),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError) as excinfo:
        post_json("https://api.test/v1/thing", {"a": 1}, {})

    message = str(excinfo.value)
    assert "429" in message
    assert "https://api.test/v1/thing" in message
    assert "boom" in message


def test_post_json_does_not_wrap_transport_urlerror(monkeypatch):
    # Contract: only HTTPError becomes RuntimeError; transport URLError surfaces raw.
    def fake_urlopen(request, timeout=None):
        raise urllib.error.URLError("conn refused")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(urllib.error.URLError):
        post_json("https://api.test/x", {}, {})
