from __future__ import annotations

from email.message import Message as HTTPHeaders

import pytest

from nexus_ai_harness.core.errors import (
    ContextLengthExceeded,
    ModelAPIError,
    RateLimitError,
)
from nexus_ai_harness.plugins.models import (
    AnthropicModel,
    GeminiModel,
    OpenAICompatibleModel,
    RetryPolicy,
)
from nexus_ai_harness.plugins.models.retry import (
    NO_RETRY,
    classify_error,
    is_context_length_error,
    is_retryable_status,
    parse_retry_after,
)

# --- RetryPolicy.delay ---------------------------------------------------


def test_delay_backs_off_exponentially_up_to_the_cap():
    policy = RetryPolicy(max_retries=10, backoff_base=1.0, backoff_cap=5.0, jitter=0)
    assert [policy.delay(n) for n in range(1, 6)] == [1.0, 2.0, 4.0, 5.0, 5.0]


def test_delay_jitter_shortens_the_wait_by_at_most_the_fraction():
    policy = RetryPolicy(backoff_base=2.0, jitter=0.25, rand=lambda: 1.0)
    assert policy.delay(1) == pytest.approx(1.5)
    no_jitter_draw = RetryPolicy(backoff_base=2.0, jitter=0.25, rand=lambda: 0.0)
    assert no_jitter_draw.delay(1) == 2.0


def test_delay_is_none_once_retries_are_exhausted():
    policy = RetryPolicy(max_retries=2, jitter=0)
    assert policy.delay(2) is not None
    assert policy.delay(3) is None
    assert NO_RETRY.delay(1) is None


def test_delay_honors_retry_after_instead_of_backoff():
    policy = RetryPolicy(backoff_base=1.0, jitter=0)
    assert policy.delay(1, retry_after=7.0) == 7.0


def test_delay_gives_up_when_retry_after_exceeds_the_limit():
    policy = RetryPolicy(max_retry_after=10.0)
    assert policy.delay(1, retry_after=11.0) is None


def test_policy_rejects_invalid_settings():
    with pytest.raises(ValueError):
        RetryPolicy(max_retries=-1)
    with pytest.raises(ValueError):
        RetryPolicy(jitter=1.5)


def test_resolve_defaults_and_max_retries_override():
    assert RetryPolicy.resolve(None, None) == RetryPolicy()
    custom = RetryPolicy(backoff_base=0.1)
    resolved = RetryPolicy.resolve(custom, 0)
    assert resolved.max_retries == 0
    assert resolved.backoff_base == 0.1  # the rest of the policy is kept


# --- parse_retry_after ---------------------------------------------------


def test_parse_retry_after_seconds_is_case_insensitive():
    assert parse_retry_after({"Retry-After": "3"}) == 3.0
    assert parse_retry_after({"retry-after": "1.5"}) == 1.5


def test_parse_retry_after_http_date_uses_the_clock():
    headers = {"Retry-After": "Wed, 21 Oct 2015 07:28:10 GMT"}
    now = 1445412480.0  # 2015-10-21 07:28:00 UTC
    assert parse_retry_after(headers, now=now) == pytest.approx(10.0)


def test_parse_retry_after_past_date_is_zero():
    headers = {"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}
    assert parse_retry_after(headers, now=1445412490.0) == 0.0


def test_parse_retry_after_prefers_milliseconds_header():
    headers = {"retry-after-ms": "250", "Retry-After": "9"}
    assert parse_retry_after(headers) == 0.25


def test_parse_retry_after_reads_http_message_headers():
    headers = HTTPHeaders()
    headers["Retry-After"] = "4"
    assert parse_retry_after(headers) == 4.0


@pytest.mark.parametrize("headers", [None, {}, {"Retry-After": "soon"}])
def test_parse_retry_after_absent_or_malformed_is_none(headers):
    assert parse_retry_after(headers) is None


# --- classification ------------------------------------------------------


@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 504, 529])
def test_transient_statuses_are_retryable(status):
    assert is_retryable_status(status)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422, 501])
def test_permanent_statuses_are_not_retryable(status):
    assert not is_retryable_status(status)


@pytest.mark.parametrize(
    "body",
    [
        '{"error": {"code": "context_length_exceeded"}}',
        "This model's maximum context length is 128000 tokens.",
        "prompt is too long: 210000 tokens > 200000 maximum",
        "Input is too long for requested model.",
        "The input token count exceeds the maximum number of tokens allowed",
    ],
)
def test_context_length_bodies_are_recognized(body):
    assert is_context_length_error(body)
    error = classify_error("m", status=400, body=body, retryable=False)
    assert isinstance(error, ContextLengthExceeded)


def test_context_length_is_never_retryable():
    error = classify_error(
        "m", status=400, body="prompt is too long", retryable=True, attempts=1
    )
    assert isinstance(error, ContextLengthExceeded)
    assert error.retryable is False


def test_unrelated_bad_request_is_a_plain_api_error():
    error = classify_error("m", status=400, body="invalid tool schema", retryable=False)
    assert type(error) is ModelAPIError


def test_rate_limit_error_carries_its_metadata():
    error = classify_error(
        "slow down", status=429, body="b", retryable=True, retry_after=2.0, attempts=3
    )
    assert isinstance(error, RateLimitError)
    assert (error.status, error.body, error.retryable) == (429, "b", True)
    assert (error.retry_after, error.attempts) == (2.0, 3)
    assert str(error) == "slow down"


def test_typed_errors_stay_catchable_as_runtime_error():
    error = classify_error("m", status=500, body="", retryable=True)
    assert isinstance(error, RuntimeError)


# --- model configuration -------------------------------------------------


def test_models_default_to_retrying():
    model = OpenAICompatibleModel(model="m", api_key="k")
    assert model.retry == RetryPolicy()
    assert model.retry.max_retries == 3


def test_max_retries_configures_the_model_without_leaking_into_params():
    policy = RetryPolicy(backoff_base=0.5)
    model = AnthropicModel(model="m", api_key="k", max_retries=0, retry=policy)
    assert model.retry.max_retries == 0
    assert model.retry.backoff_base == 0.5
    assert "max_retries" not in model.params
    assert "retry" not in model.params


@pytest.mark.parametrize(
    "cls,module",
    [
        (OpenAICompatibleModel, "openai_compatible"),
        (AnthropicModel, "anthropic"),
        (GeminiModel, "gemini"),
    ],
)
def test_backends_pass_their_policy_to_post_json(monkeypatch, cls, module):
    captured: dict = {}

    def fake_post_json(url, payload, headers, timeout, retry=None):
        captured["retry"] = retry
        return {}

    monkeypatch.setattr(
        f"nexus_ai_harness.plugins.models.{module}.post_json", fake_post_json
    )
    policy = RetryPolicy(max_retries=5)
    cls(model="m", api_key="k", retry=policy).complete([], [], None)
    assert captured["retry"] is policy
