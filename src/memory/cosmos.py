"""Small LangGraph BaseStore adapter for Azure Cosmos DB for NoSQL."""

from __future__ import annotations

import asyncio
import math
import os
from datetime import UTC, datetime
from typing import Any

from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    PutOp,
    SearchItem,
    SearchOp,
)


class CosmosStore(BaseStore):
    """Namespace-isolated JSON store with optional semantic recall.

    Cosmos performs the authoritative JSON persistence. For the small serverless
    development profile semantic ranking is calculated over stored embeddings in
    the adapter; a production deployment can replace this with Cosmos vector
    indexing without changing the MemoryManager contract.
    """

    def __init__(self, container: Any, *, embedder: Any = None) -> None:
        self._container = container
        self._embedder = embedder

    @classmethod
    def from_environment(cls) -> CosmosStore:
        from azure.cosmos import CosmosClient
        from azure.identity import DefaultAzureCredential

        endpoint = os.getenv("COSMOS_ENDPOINT")
        if not endpoint:
            raise ValueError("COSMOS_ENDPOINT is required for Cosmos long-term memory.")
        key = os.getenv("COSMOS_KEY")
        client = (
            CosmosClient(endpoint, key)
            if key
            else CosmosClient(endpoint, credential=DefaultAzureCredential())
        )
        database = client.get_database_client(
            os.getenv("COSMOS_DATABASE", "onyx-memory")
        )
        container = database.get_container_client(
            os.getenv("COSMOS_CONTAINER", "memories")
        )

        def embed(texts: list[str]) -> list[list[float]]:
            from src.config.config import get_embeddings

            return get_embeddings().embed_documents(texts)

        return cls(container, embedder=embed)

    def batch(self, ops):
        results = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(self._get(op.namespace, op.key))
            elif isinstance(op, SearchOp):
                results.append(self._search(op))
            elif isinstance(op, PutOp):
                results.append(self._put(op))
            elif isinstance(op, ListNamespacesOp):
                results.append([])
            else:
                raise TypeError(f"Unsupported Cosmos operation: {type(op)!r}")
        return results

    async def abatch(self, ops):
        return await asyncio.to_thread(self.batch, ops)

    def _get(self, namespace: tuple[str, ...], key: str) -> Item | None:
        try:
            document = self._container.read_item(
                key, partition_key=_namespace_key(namespace)
            )
        except Exception as error:
            if error.__class__.__name__ == "CosmosResourceNotFoundError":
                return None
            raise
        return _item(document)

    def _put(self, op: PutOp) -> None:
        partition = _namespace_key(op.namespace)
        if op.value is None:
            try:
                self._container.delete_item(op.key, partition_key=partition)
            except Exception as error:
                if error.__class__.__name__ != "CosmosResourceNotFoundError":
                    raise
            return None
        now = datetime.now(UTC).isoformat()
        existing = self._get(op.namespace, op.key)
        document = {
            "id": op.key,
            "namespace": list(op.namespace),
            "namespace_key": partition,
            "value": op.value,
            "created_at": existing.created_at.isoformat() if existing else now,
            "updated_at": now,
        }
        if op.index and self._embedder:
            fields = " ".join(str(op.value.get(field, "")) for field in op.index)
            document["embedding"] = self._embedder([fields])[0]
        self._container.upsert_item(document)
        return None

    def _search(self, op: SearchOp) -> list[SearchItem]:
        prefix = _namespace_key(op.namespace_prefix)
        documents = list(
            self._container.query_items(
                query="SELECT * FROM c WHERE c.namespace_key = @namespace_key",
                parameters=[{"name": "@namespace_key", "value": prefix}],
                enable_cross_partition_query=False,
            )
        )
        filtered = [document for document in documents if _matches(document, op.filter)]
        query_vector = (
            self._embedder([op.query])[0] if op.query and self._embedder else None
        )
        scored = [(_score(document, query_vector), document) for document in filtered]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            _search_item(document, score)
            for score, document in scored[op.offset : op.offset + op.limit]
        ]


def _namespace_key(namespace: tuple[str, ...]) -> str:
    return "\x1f".join(namespace)


def _item(document: dict[str, Any]) -> Item:
    return Item(
        namespace=tuple(document["namespace"]),
        key=document["id"],
        value=document["value"],
        created_at=document["created_at"],
        updated_at=document["updated_at"],
    )


def _search_item(document: dict[str, Any], score: float) -> SearchItem:
    return SearchItem(
        namespace=tuple(document["namespace"]),
        key=document["id"],
        value=document["value"],
        created_at=document["created_at"],
        updated_at=document["updated_at"],
        score=score,
    )


def _matches(document: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    value = document["value"]
    for field, expected in filters.items():
        actual = value.get(field)
        if isinstance(expected, dict):
            for operator, target in expected.items():
                comparisons = {
                    "$eq": actual == target,
                    "$ne": actual != target,
                    "$gt": actual is not None and actual > target,
                    "$gte": actual is not None and actual >= target,
                    "$lt": actual is not None and actual < target,
                    "$lte": actual is not None and actual <= target,
                }
                if not comparisons.get(operator, False):
                    return False
        elif actual != expected:
            return False
    return True


def _score(document: dict[str, Any], query_vector: list[float] | None) -> float:
    if not query_vector or not document.get("embedding"):
        return 0.0
    document_vector = document["embedding"]
    numerator = sum(
        left * right for left, right in zip(query_vector, document_vector, strict=True)
    )
    left_norm = math.sqrt(sum(item * item for item in query_vector))
    right_norm = math.sqrt(sum(item * item for item in document_vector))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0
