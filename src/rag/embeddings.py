"""Rate-limited embeddings decorator.

Managed embedding endpoints (Azure OpenAI S0, Bedrock, Vertex) enforce both a
requests-per-minute and a tokens-per-minute quota. The provider SDKs retry on
HTTP 429, but only reactively and only a couple of times by default -- which is
useless against a window that stays closed for 60 seconds.

This module adds the two things a production ingest pipeline needs:

* **Proactive pacing** -- a token-bucket limiter spaces requests so the quota is
  never exhausted in the first place.
* **A retry backstop** -- exponential backoff that honours the provider's
  ``Retry-After`` hint when it sends one.

The wrapper implements :class:`~langchain_core.embeddings.Embeddings`, so it is
a drop-in replacement anywhere an embeddings instance is accepted (vector
stores, semantic splitters, retrievers).
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.rate_limiters import InMemoryRateLimiter

logger = logging.getLogger("pipeline.embeddings")

# Azure surfaces the wait hint in the message body rather than in a header the
# SDK exposes, so we recover it by pattern instead.
_RETRY_AFTER_RE = re.compile(r"retry after (\d+)", re.IGNORECASE)


def _retry_after_seconds(error: BaseException) -> float | None:
    """Extracts the provider's suggested wait, if it advertises one."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        raw = headers.get("retry-after") or headers.get("Retry-After")
        if raw:
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass

    match = _RETRY_AFTER_RE.search(str(error))
    return float(match.group(1)) if match else None


def _is_rate_limit(error: BaseException) -> bool:
    """Duck-typed 429 check that avoids a hard dependency on any provider SDK."""
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if status == 429:
        return True
    return "429" in str(error) or "rate limit" in str(error).lower()


class RateLimitedEmbeddings(Embeddings):
    """Paces and retries calls to an underlying embeddings provider.

    Args:
        embeddings: The provider instance to wrap.
        requests_per_second: Sustained request rate. Derive it from the
            deployment quota, e.g. an Azure deployment rated at 60 RPM should
            use ``1.0`` (leave headroom if other clients share the quota).
        batch_size: Maximum inputs per request. Caps the tokens in flight so a
            single oversized batch cannot blow the TPM window on its own.
        max_retries: Attempts made before a rate-limit error is re-raised.
        max_backoff: Ceiling for the computed backoff, in seconds.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        *,
        requests_per_second: float = 1.0,
        batch_size: int = 128,
        max_retries: int = 8,
        max_backoff: float = 120.0,
    ) -> None:
        self._embeddings = embeddings
        self._batch_size = max(1, batch_size)
        self._max_retries = max(1, max_retries)
        self._max_backoff = max_backoff
        self._limiter = InMemoryRateLimiter(
            requests_per_second=requests_per_second,
            check_every_n_seconds=0.1,
            # A bucket of 1 forbids bursting: the very pattern that trips the
            # quota is several requests landing in the same instant.
            max_bucket_size=1,
        )

    @property
    def inner(self) -> Embeddings:
        """The wrapped provider, for callers that need the raw instance."""
        return self._embeddings

    def _call_with_retry(self, fn: Any, *args: Any) -> Any:
        """Runs `fn` under the limiter, retrying rate-limit failures."""
        for attempt in range(1, self._max_retries + 1):
            self._limiter.acquire(blocking=True)
            try:
                return fn(*args)
            except Exception as error:
                if not _is_rate_limit(error) or attempt == self._max_retries:
                    raise

                # Trust the provider's hint when present; otherwise back off
                # exponentially from 2s.
                wait = _retry_after_seconds(error) or min(
                    2.0**attempt, self._max_backoff
                )
                wait = min(wait, self._max_backoff)
                logger.warning(
                    f"Rate limit hit (attempt {attempt}/{self._max_retries}); "
                    f"backing off {wait:.0f}s before retrying."
                )
                time.sleep(wait)

        raise RuntimeError("Unreachable: retry loop exited without a result.")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeds `texts` in paced batches, preserving input order."""
        vectors: list[list[float]] = []
        total_batches = (len(texts) + self._batch_size - 1) // self._batch_size

        for index in range(0, len(texts), self._batch_size):
            batch = texts[index : index + self._batch_size]
            batch_number = index // self._batch_size + 1
            if total_batches > 1:
                logger.debug(
                    f"Embedding batch {batch_number}/{total_batches} "
                    f"({len(batch)} inputs)."
                )
            vectors.extend(
                self._call_with_retry(self._embeddings.embed_documents, batch)
            )

        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embeds a single query string under the same pacing policy."""
        return self._call_with_retry(self._embeddings.embed_query, text)
