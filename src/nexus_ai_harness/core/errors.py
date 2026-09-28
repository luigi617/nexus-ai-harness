from __future__ import annotations


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
