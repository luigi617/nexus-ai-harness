from __future__ import annotations

import http.client
import json
import ssl
import urllib.error
import urllib.request
from typing import Any

from nexus_ai_harness.core.errors import ModelAPIError
from nexus_ai_harness.plugins.models.retry import (
    NO_RETRY,
    RetryPolicy,
    classify_error,
    is_retryable_status,
    parse_retry_after,
)


def post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float = 60.0,
    *,
    retry: RetryPolicy | None = None,
) -> dict[str, Any]:
    """POST ``payload`` as JSON and return the decoded JSON response.

    Uses the standard library only, so model backends add no dependencies.
    Transient failures (throttling, 408/409, most 5xx, dropped connections and
    timeouts) are retried according to ``retry``; without one, a single attempt
    is made.

    Args:
        url: The endpoint to POST to.
        payload: The JSON-serializable request body.
        headers: Extra request headers; these win over the default Content-Type.
        timeout: Per-attempt socket timeout in seconds.
        retry: The retry policy; ``None`` disables retrying.

    Returns:
        The decoded JSON response body.

    Raises:
        ModelAPIError: On an HTTP error status, once retries are exhausted. It is
            a ``RateLimitError`` for 429 and ``ContextLengthExceeded`` when the
            body reports a context overflow.
        urllib.error.URLError: On a transport failure, once retries are exhausted.
        TimeoutError: When a response read times out, once retries are exhausted.
    """
    policy = retry or NO_RETRY
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    attempt = 0
    while True:
        attempt += 1
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            error = _http_error(exc, url, attempt, policy)
            delay = (
                policy.delay(attempt, error.retry_after) if error.retryable else None
            )
            if delay is None:
                raise error from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            delay = policy.delay(attempt) if _is_transient(exc) else None
            if delay is None:
                raise
        except http.client.HTTPException:  # e.g. the server hung up mid-response
            delay = policy.delay(attempt)
            if delay is None:
                raise
        policy.sleep(delay)


def _http_error(
    exc: urllib.error.HTTPError, url: str, attempt: int, policy: RetryPolicy
) -> ModelAPIError:
    body = exc.read().decode(errors="replace")
    retryable = is_retryable_status(exc.code)
    should_retry = (exc.headers.get("x-should-retry") if exc.headers else None) or ""
    if should_retry.lower() in ("true", "false"):  # provider's explicit verdict wins
        retryable = should_retry.lower() == "true"
    return classify_error(
        f"HTTP {exc.code} from {url}: {body}",
        status=exc.code,
        body=body,
        retryable=retryable,
        retry_after=parse_retry_after(exc.headers, now=policy.clock()),
        attempts=attempt,
    )


def _is_transient(exc: OSError) -> bool:
    reason = getattr(exc, "reason", None)
    # A bad certificate or hostname will fail identically on every attempt.
    return not isinstance(exc, ssl.SSLCertVerificationError) and not isinstance(
        reason, ssl.SSLCertVerificationError
    )
