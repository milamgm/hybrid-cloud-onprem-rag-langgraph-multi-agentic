"""Distributed TTL context cache for RAG bodies that must not be checkpointed."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from langchain_core.documents import Document


class ContextUnavailable(RuntimeError):
    """Raised when ephemeral context expired or crossed an isolation boundary."""


class ContextCache(Protocol):
    def put(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        classification: str,
        documents: list[Document],
    ) -> str: ...

    def get(self, *, handle: str, tenant_id: str, thread_id: str) -> list[Document]: ...

    def delete(self, *, handle: str, tenant_id: str, thread_id: str) -> None: ...


@dataclass
class InMemoryContextCache:
    """Deterministic test implementation with the same isolation contract."""

    _items: dict[str, tuple[str, str, list[Document]]]
    _allowed_classifications: frozenset[str]

    def __init__(
        self,
        *,
        allowed_classifications: frozenset[str] = frozenset({"public", "internal"}),
    ) -> None:
        self._items = {}
        self._allowed_classifications = allowed_classifications

    def put(self, *, tenant_id, thread_id, classification, documents) -> str:
        if classification not in self._allowed_classifications:
            raise ValueError(
                f"Classification {classification!r} is not permitted in the context cache."
            )
        handle = str(uuid4())
        self._items[handle] = (tenant_id, thread_id, documents)
        return handle

    def get(self, *, handle, tenant_id, thread_id) -> list[Document]:
        item = self._items.get(handle)
        if not item or item[0] != tenant_id or item[1] != thread_id:
            raise ContextUnavailable(
                "RAG context is unavailable for this tenant/thread."
            )
        return item[2]

    def delete(self, *, handle, tenant_id, thread_id) -> None:
        self.get(handle=handle, tenant_id=tenant_id, thread_id=thread_id)
        self._items.pop(handle, None)


class RedisContextCache:
    """Shared ephemeral cache for hybrid deployments; Redis is not authoritative."""

    def __init__(self, client, *, ttl_seconds: int = 300) -> None:
        if not 30 <= ttl_seconds <= 3600:
            raise ValueError("RAG context TTL must be between 30 and 3600 seconds.")
        self._client = client
        self._ttl_seconds = ttl_seconds

    @classmethod
    def from_environment(cls) -> RedisContextCache:
        import redis

        mode = os.getenv("INFRASTRUCTURE_MODE", "on_premise").lower()
        variable = (
            "RAG_CONTEXT_REDIS_URI_CLOUD"
            if mode == "cloud"
            else "RAG_CONTEXT_REDIS_URI_ONPREM"
        )
        uri = os.getenv(variable) or os.getenv("REDIS_URI")
        if not uri:
            raise ValueError(f"{variable} is required for the RAG context cache.")
        if os.getenv(
            "DEPLOYMENT_ENVIRONMENT", "development"
        ) == "production" and not uri.startswith("rediss://"):
            raise ValueError("Production Redis context cache requires TLS (rediss://).")
        return cls(
            redis.Redis.from_url(uri, decode_responses=True),
            ttl_seconds=int(os.getenv("RAG_CONTEXT_TTL_SECONDS", "300")),
        )

    def put(self, *, tenant_id, thread_id, classification, documents) -> str:
        allowed = {
            value.strip()
            for value in os.getenv(
                "RAG_CACHE_ALLOWED_CLASSIFICATIONS", "public,internal"
            ).split(",")
        }
        if classification not in allowed:
            raise ValueError(
                f"Classification {classification!r} is not permitted in the context cache."
            )
        handle = str(uuid4())
        key = self._key(handle, tenant_id, thread_id)
        payload = {
            "tenant_id": tenant_id,
            "thread_id": thread_id,
            "documents": [
                {"page_content": document.page_content, "metadata": document.metadata}
                for document in documents
            ],
        }
        self._client.setex(
            key, self._ttl_seconds, json.dumps(payload, separators=(",", ":"))
        )
        return handle

    def get(self, *, handle, tenant_id, thread_id) -> list[Document]:
        payload = self._client.get(self._key(handle, tenant_id, thread_id))
        if not payload:
            raise ContextUnavailable("RAG context expired or is unavailable.")
        data = json.loads(payload)
        if data["tenant_id"] != tenant_id or data["thread_id"] != thread_id:
            raise ContextUnavailable("RAG context isolation check failed.")
        return [Document(**document) for document in data["documents"]]

    def delete(self, *, handle, tenant_id, thread_id) -> None:
        self._client.delete(self._key(handle, tenant_id, thread_id))

    @staticmethod
    def _key(handle: str, tenant_id: str, thread_id: str) -> str:
        scope = hashlib.sha256(f"{tenant_id}\x00{thread_id}".encode()).hexdigest()[:24]
        return f"onyx:rag-context:v1:{scope}:{handle}"
