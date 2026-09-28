from __future__ import annotations

import contextlib


class ModelAPIError(RuntimeError):
    """A model provider rejected or failed a completion request.

    Subclasses ``RuntimeError`` so code written against the earlier untyped
    errors keeps catching it.

    Attributes:
        status: The HTTP status code, or ``None`` when the provider gave none.
        body: The raw error body returned by the provider.
        retryable: Whether retrying the same request could plausibly succeed.
        retry_after: Seconds the provider asked the caller to wait, if any.
        attempts: How many requests were made before giving up.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        body: str = "",
        retryable: bool = False,
        retry_after: float | None = None,
        attempts: int = 1,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.body = body
        self.retryable = retryable
        self.retry_after = retry_after
        self.attempts = attempts


class RateLimitError(ModelAPIError):
    """The provider throttled the request (HTTP 429 or equivalent)."""


class ContextLengthExceeded(ModelAPIError):
    """The request's prompt did not fit in the model's context window."""


_MODEL_FAILURE = "_nexus_model_failure"


def mark_model_failure(exc: BaseException) -> None:
    """Record that ``exc`` was raised by a model call itself.

    ``Context.invoke`` marks exceptions escaping a ``Model`` method (but not its
    interceptors), so a loop can tell a failed completion, even one made inside
    a router or context manager, apart from any other error.

    Args:
        exc: The exception to mark.
    """
    # An exception type without a __dict__ simply stays unmarked.
    with contextlib.suppress(AttributeError, TypeError):
        setattr(exc, _MODEL_FAILURE, True)


def is_model_failure(exc: BaseException) -> bool:
    """Whether ``exc`` was raised by a model call, per :func:`mark_model_failure`.

    Args:
        exc: The exception to inspect.

    Returns:
        ``True`` for exceptions raised by a ``Model`` method or typed as
        ``ModelAPIError``.
    """
    return isinstance(exc, ModelAPIError) or bool(getattr(exc, _MODEL_FAILURE, False))


def failure_attempts(exc: BaseException) -> int | None:
    """How many requests a failed model call made, if the error records it.

    Args:
        exc: The exception raised by the model call.

    Returns:
        The attempt count, or ``None`` when unknown.
    """
    attempts = getattr(exc, "attempts", None)
    return attempts if isinstance(attempts, int) else None
