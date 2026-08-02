"""Pacing and rate-limit recovery for the embeddings wrapper."""

import time

import pytest
from langchain_core.embeddings import Embeddings

from src.rag.embeddings import (
    RateLimitedEmbeddings,
    _is_rate_limit,
    _retry_after_seconds,
)

AZURE_429 = (
    "Error code: 429 - {'error': {'code': 'RateLimitReached', 'message': "
    "'Your requests to text-embedding-3-large have exceeded the call rate "
    "limit for your current AIServices S0 pricing tier. "
    "Please retry after 3 seconds.'}}"
)


class FlakyEmbeddings(Embeddings):
    """Fake provider that raises a rate-limit error a fixed number of times."""

    def __init__(self, fail_times: int = 0, message: str = AZURE_429):
        self.calls: list[int] = []
        self.fail_times = fail_times
        self.failures_emitted = 0
        self.message = message

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.failures_emitted < self.fail_times:
            self.failures_emitted += 1
            raise RuntimeError(self.message)
        self.calls.append(len(texts))
        return [[1.0, 2.0, 3.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def test_inputs_are_split_into_batches_preserving_order():
    provider = FlakyEmbeddings()
    limiter = RateLimitedEmbeddings(provider, requests_per_second=1000, batch_size=10)

    vectors = limiter.embed_documents([f"text {i}" for i in range(25)])

    assert provider.calls == [10, 10, 5]
    assert len(vectors) == 25


def test_requests_are_paced_to_the_configured_rate():
    provider = FlakyEmbeddings()
    limiter = RateLimitedEmbeddings(provider, requests_per_second=10.0, batch_size=1)

    start = time.monotonic()
    limiter.embed_documents(["a", "b", "c"])
    elapsed = time.monotonic() - start

    # Three requests at 10/s cannot all land inside the first 100ms window.
    assert elapsed >= 0.15, f"requests were not paced (took {elapsed:.3f}s)"


def test_rate_limit_errors_are_retried():
    provider = FlakyEmbeddings(fail_times=2, message="Error code: 429 - retry after 1")
    limiter = RateLimitedEmbeddings(provider, requests_per_second=1000, max_retries=5)

    vectors = limiter.embed_documents(["a", "b"])

    assert provider.failures_emitted == 2
    assert len(vectors) == 2


def test_rate_limit_error_is_reraised_once_retries_are_exhausted():
    provider = FlakyEmbeddings(fail_times=99, message="Error code: 429 - retry after 1")
    limiter = RateLimitedEmbeddings(provider, requests_per_second=1000, max_retries=2)

    with pytest.raises(RuntimeError, match="429"):
        limiter.embed_documents(["a"])


def test_non_rate_limit_errors_are_not_retried():
    """A bad request or auth failure must surface immediately, not after 6 waits."""
    provider = FlakyEmbeddings(fail_times=99, message="401 invalid api key")
    limiter = RateLimitedEmbeddings(provider, requests_per_second=1000, max_retries=5)

    with pytest.raises(RuntimeError, match="401"):
        limiter.embed_documents(["a"])

    assert provider.failures_emitted == 1, "non-429 errors must not be retried"


def test_retry_after_is_recovered_from_the_azure_message_body():
    assert _retry_after_seconds(RuntimeError(AZURE_429)) == 3.0
    assert _retry_after_seconds(RuntimeError("boom")) is None


def test_retry_after_prefers_the_response_header():
    class Response:
        headers = {"retry-after": "42"}

    error = RuntimeError(AZURE_429)
    error.response = Response()

    assert _retry_after_seconds(error) == 42.0


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (AZURE_429, True),
        ("429 Too Many Requests", True),
        ("Rate limit exceeded", True),
        ("401 unauthorized", False),
        ("connection reset", False),
    ],
)
def test_rate_limit_detection(message, expected):
    assert _is_rate_limit(RuntimeError(message)) is expected


def test_status_code_attribute_is_recognised():
    error = RuntimeError("something opaque")
    error.status_code = 429

    assert _is_rate_limit(error) is True
