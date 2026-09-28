from __future__ import annotations

import random
import re
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from email.message import Message
from email.utils import parsedate_to_datetime

from nexus_ai_harness.core.errors import (
    ContextLengthExceeded,
    ModelAPIError,
    RateLimitError,
)
from nexus_ai_harness.core.invoke import offload_cancelled

Headers = Mapping[str, str] | Message

# Phrases providers use when a prompt overflows the context window.
_CONTEXT_LENGTH_PATTERN = re.compile(
    r"context[_ ]length[_ ]exceeded"
    r"|maximum context length"
    r"|context window"
    r"|prompt is too long"
    r"|input is too long"
    r"|too many (input )?tokens"
    r"|exceeds? the (maximum|max) number of tokens"
    r"|exceed context limit",
    re.IGNORECASE,
)

# Longest single sleep while an abort check is active, so aborts land promptly.
_ABORT_POLL_SECONDS = 0.25

_abort_check: ContextVar[Callable[[], bool] | None] = ContextVar(
    "_abort_check", default=None
)


@dataclass(frozen=True)
class RetryPolicy:
    """How a model backend retries transient request failures.

    Failed attempts wait ``backoff_base * 2 ** (attempt - 1)`` seconds, capped at
    ``backoff_cap`` and shortened by up to ``jitter`` (a fraction) so concurrent
    clients spread out. A provider's ``Retry-After`` replaces the computed wait;
    one longer than ``max_retry_after`` ends the retries instead of stalling.

    Attributes:
        max_retries: Retries after the first attempt; ``0`` disables retrying.
        backoff_base: Wait in seconds before the first retry, before jitter.
        backoff_cap: Upper bound in seconds on any computed backoff wait.
        max_retry_after: Longest provider-requested wait, in seconds, to honor.
        jitter: Fraction of each computed wait that is randomly removed.
        sleep: Blocks for the given number of seconds; injectable for tests.
        clock: Returns the current Unix time; used for HTTP-date ``Retry-After``.
        rand: Returns a float in ``[0, 1)``; injectable for deterministic tests.
    """

    max_retries: int = 3
    backoff_base: float = 1.0
    backoff_cap: float = 30.0
    max_retry_after: float = 60.0
    jitter: float = 0.25
    sleep: Callable[[float], None] = field(default=time.sleep, compare=False)
    clock: Callable[[], float] = field(default=time.time, compare=False)
    rand: Callable[[], float] = field(default=random.random, compare=False)

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if not 0.0 <= self.jitter <= 1.0:
            raise ValueError("jitter must be between 0 and 1")

    @classmethod
    def resolve(cls, retry: RetryPolicy | None, max_retries: int | None) -> RetryPolicy:
        """Combine a model's ``retry`` and ``max_retries`` arguments into a policy.

        Args:
            retry: An explicit policy, or ``None`` for the defaults.
            max_retries: Overrides the policy's ``max_retries`` when given.

        Returns:
            The policy the model should use.
        """
        policy = retry or cls()
        if max_retries is not None:
            policy = replace(policy, max_retries=max_retries)
        return policy

    def delay(self, attempt: int, retry_after: float | None = None) -> float | None:
        """Return how long to wait before retrying after failed ``attempt``.

        Args:
            attempt: The 1-based number of the attempt that just failed.
            retry_after: The provider-requested wait in seconds, if any.

        Returns:
            Seconds to sleep, or ``None`` when no further retry should be made.
        """
        if attempt > self.max_retries:
            return None
        if retry_after is not None:
            return retry_after if retry_after <= self.max_retry_after else None
        backoff = min(self.backoff_cap, self.backoff_base * 2 ** (attempt - 1))
        return backoff * (1.0 - self.jitter * self.rand())


NO_RETRY = RetryPolicy(max_retries=0)


@contextmanager
def abort_when(check: Callable[[], bool]) -> Iterator[None]:
    """Stop retrying requests made inside this block once ``check`` is true.

    ``BaseModel.complete`` wires this to ``ctx.interrupted``, so an interrupted
    run stops sending (billed) retries. Cancellation of the task awaiting a
    thread-offloaded call is honored without it.

    Args:
        check: Returns ``True`` once no further attempt should be made.
    """
    token = _abort_check.set(check)
    try:
        yield
    finally:
        _abort_check.reset(token)


def wait_before_retry(policy: RetryPolicy, delay: float) -> bool:
    """Sleep ``delay`` seconds before a retry unless the caller aborts first.

    Args:
        policy: Supplies the injectable ``sleep``.
        delay: The wait in seconds.

    Returns:
        ``True`` to go ahead with the retry, ``False`` if it was aborted.
    """
    check = _abort_check.get()
    cancelled = offload_cancelled()
    if check is None and cancelled is None:  # nothing can abort; one plain sleep
        policy.sleep(delay)
        return True

    def aborted() -> bool:
        return bool(offload_cancelled()) or (check is not None and check())

    remaining = delay
    while not aborted():
        if remaining <= 0:
            return True
        step = min(remaining, _ABORT_POLL_SECONDS)
        policy.sleep(step)
        remaining -= step
    return False


def parse_retry_after(
    headers: Headers | None, now: float | None = None
) -> float | None:
    """Read the wait a provider requested from its response headers.

    Understands ``retry-after-ms`` and ``Retry-After`` as either delta-seconds or
    an HTTP-date.

    Args:
        headers: The error response headers; ``None`` when there were none.
        now: The current Unix time, used to turn an HTTP-date into a delay.

    Returns:
        The wait in seconds (never negative), or ``None`` if absent or malformed.
    """
    if not headers:
        return None
    millis = _header(headers, "retry-after-ms")
    if millis is not None:
        try:
            return max(0.0, float(millis) / 1000)
        except ValueError:
            pass
    value = _header(headers, "retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    current = time.time() if now is None else now
    return max(0.0, when.timestamp() - current)


def is_retryable_status(status: int) -> bool:
    """Whether an HTTP status marks a failure that may succeed on retry."""
    return status in (408, 409, 429) or (status >= 500 and status != 501)


def is_context_length_error(body: str) -> bool:
    """Whether an error body says the prompt overflowed the context window."""
    return bool(_CONTEXT_LENGTH_PATTERN.search(body))


def classify_error(
    message: str,
    *,
    status: int | None,
    body: str,
    retryable: bool,
    retry_after: float | None = None,
    attempts: int = 1,
    rate_limited: bool = False,
) -> ModelAPIError:
    """Build the most specific typed error for a failed model request.

    Args:
        message: The human-readable error message.
        status: The HTTP status code, if known.
        body: The provider's error body, inspected for context overflow.
        retryable: Whether the failure is transient.
        retry_after: The provider-requested wait in seconds, if any.
        attempts: How many requests were made.
        rate_limited: Marks throttling a provider reports without an HTTP 429.

    Returns:
        A ``ContextLengthExceeded``, ``RateLimitError``, or ``ModelAPIError``.
    """
    cls: type[ModelAPIError] = ModelAPIError
    if status == 429 or rate_limited:
        cls = RateLimitError
    elif (status is None or 400 <= status < 500) and is_context_length_error(body):
        cls, retryable = ContextLengthExceeded, False  # resending cannot fit it
    return cls(
        message,
        status=status,
        body=body,
        retryable=retryable,
        retry_after=retry_after,
        attempts=attempts,
    )


def _header(headers: Headers, name: str) -> str | None:
    value = headers.get(name)
    if value is None:  # plain dicts are case-sensitive, unlike HTTPMessage
        lowered = name.lower()
        for key, candidate in headers.items():
            if key.lower() == lowered:
                return candidate
    return value
