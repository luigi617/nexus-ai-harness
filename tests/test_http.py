from __future__ import annotations

import http.client
import io
import json
import ssl
import urllib.error
import urllib.request

import pytest

from nexus_ai_harness.core.errors import (
    ContextLengthExceeded,
    ModelAPIError,
    RateLimitError,
)
from nexus_ai_harness.plugins.models._http import post_json
from nexus_ai_harness.plugins.models.retry import RetryPolicy


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


def test_post_json_http_error_is_a_typed_model_api_error(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise _http_error(429, b"boom")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RateLimitError) as excinfo:
        post_json("https://api.test/x", {}, {})
    assert isinstance(excinfo.value, ModelAPIError)
    assert excinfo.value.status == 429


# --- retries ---------------------------------------------------------------


def _http_error(status: int, body: bytes = b"err", headers: dict | None = None):
    return urllib.error.HTTPError(
        "https://api.test/x", status, "error", headers or {}, io.BytesIO(body)
    )


def _script(monkeypatch, *outcomes):
    """Make urlopen raise or return each outcome in turn; returns the call log."""
    calls: list[int] = []
    queue = list(outcomes)

    def fake_urlopen(request, timeout=None):
        calls.append(1)
        outcome = queue.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return _FakeResponse(outcome)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return calls


def _policy(sleeps: list[float], **overrides) -> RetryPolicy:
    settings: dict = {"max_retries": 3, "backoff_base": 1.0, "jitter": 0.0}
    settings.update(overrides)
    return RetryPolicy(sleep=sleeps.append, clock=lambda: 0.0, **settings)


def test_post_json_retries_transient_status_then_succeeds(monkeypatch):
    calls = _script(monkeypatch, _http_error(503), _http_error(502), b'{"ok": 1}')
    sleeps: list[float] = []
    result = post_json("https://api.test/x", {}, {}, retry=_policy(sleeps))
    assert result == {"ok": 1}
    assert len(calls) == 3
    assert sleeps == [1.0, 2.0]  # exponential backoff between attempts


def test_post_json_honors_retry_after_seconds(monkeypatch):
    _script(monkeypatch, _http_error(429, headers={"Retry-After": "7"}), b"{}")
    sleeps: list[float] = []
    post_json("https://api.test/x", {}, {}, retry=_policy(sleeps))
    assert sleeps == [7.0]


def test_post_json_honors_retry_after_http_date(monkeypatch):
    date = "Thu, 01 Jan 1970 00:00:12 GMT"
    _script(monkeypatch, _http_error(429, headers={"Retry-After": date}), b"{}")
    sleeps: list[float] = []
    post_json("https://api.test/x", {}, {}, retry=_policy(sleeps))
    assert sleeps == [12.0]  # the policy clock reads 0.0, the epoch


def test_post_json_raises_rate_limit_error_when_retries_run_out(monkeypatch):
    calls = _script(monkeypatch, *(_http_error(429, b"slow down") for _ in range(3)))
    sleeps: list[float] = []
    with pytest.raises(RateLimitError) as excinfo:
        post_json("https://api.test/x", {}, {}, retry=_policy(sleeps, max_retries=2))
    error = excinfo.value
    assert (error.status, error.body, error.attempts) == (429, "slow down", 3)
    assert error.retryable is True
    assert len(calls) == 3
    assert sleeps == [1.0, 2.0]


def test_post_json_gives_up_when_retry_after_is_too_long(monkeypatch):
    calls = _script(monkeypatch, _http_error(429, headers={"Retry-After": "3600"}))
    sleeps: list[float] = []
    with pytest.raises(RateLimitError) as excinfo:
        post_json("https://api.test/x", {}, {}, retry=_policy(sleeps))
    assert excinfo.value.retry_after == 3600.0
    assert len(calls) == 1
    assert sleeps == []


def test_post_json_does_not_retry_client_errors(monkeypatch):
    calls = _script(monkeypatch, _http_error(401, b"bad key"))
    sleeps: list[float] = []
    with pytest.raises(ModelAPIError) as excinfo:
        post_json("https://api.test/x", {}, {}, retry=_policy(sleeps))
    assert excinfo.value.status == 401
    assert excinfo.value.retryable is False
    assert len(calls) == 1
    assert sleeps == []


def test_post_json_raises_context_length_exceeded_without_retrying(monkeypatch):
    body = b'{"error": {"message": "prompt is too long: 300000 tokens"}}'
    calls = _script(monkeypatch, _http_error(400, body))
    with pytest.raises(ContextLengthExceeded):
        post_json("https://api.test/x", {}, {}, retry=_policy([]))
    assert len(calls) == 1


def test_post_json_respects_x_should_retry_header(monkeypatch):
    calls = _script(monkeypatch, _http_error(500, headers={"x-should-retry": "false"}))
    with pytest.raises(ModelAPIError) as excinfo:
        post_json("https://api.test/x", {}, {}, retry=_policy([]))
    assert excinfo.value.retryable is False
    assert len(calls) == 1

    calls = _script(
        monkeypatch, _http_error(400, headers={"x-should-retry": "true"}), b"{}"
    )
    assert post_json("https://api.test/x", {}, {}, retry=_policy([])) == {}
    assert len(calls) == 2


@pytest.mark.parametrize(
    "transient",
    [
        urllib.error.URLError("conn refused"),
        TimeoutError("read timed out"),
        ConnectionResetError("reset"),
        http.client.RemoteDisconnected("hung up"),
        http.client.IncompleteRead(b"par"),
    ],
)
def test_post_json_retries_transport_failures(monkeypatch, transient):
    calls = _script(monkeypatch, transient, b"{}")
    sleeps: list[float] = []
    assert post_json("https://api.test/x", {}, {}, retry=_policy(sleeps)) == {}
    assert len(calls) == 2
    assert sleeps == [1.0]


def test_post_json_reraises_transport_failure_once_retries_run_out(monkeypatch):
    calls = _script(monkeypatch, *[urllib.error.URLError("down")] * 2)
    with pytest.raises(urllib.error.URLError):
        post_json("https://api.test/x", {}, {}, retry=_policy([], max_retries=1))
    assert len(calls) == 2


def test_post_json_does_not_retry_certificate_failures(monkeypatch):
    cert_error = urllib.error.URLError(ssl.SSLCertVerificationError("bad cert"))
    calls = _script(monkeypatch, cert_error)
    with pytest.raises(urllib.error.URLError):
        post_json("https://api.test/x", {}, {}, retry=_policy([]))
    assert len(calls) == 1


def test_post_json_without_policy_makes_a_single_attempt(monkeypatch):
    calls = _script(monkeypatch, _http_error(503))
    with pytest.raises(ModelAPIError) as excinfo:
        post_json("https://api.test/x", {}, {})
    assert excinfo.value.attempts == 1
    assert len(calls) == 1
