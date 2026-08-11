"""Persistent cross-thread memory store for portable hybrid deployment."""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from langgraph.store.base import IndexConfig

from src.graph.persistence import _require_postgres_tls


def _memory_connection_string() -> str:
    mode = os.getenv("INFRASTRUCTURE_MODE", "on_premise").lower()
    variable = (
        "LONG_TERM_MEMORY_DATABASE_URL_CLOUD"
        if mode == "cloud"
        else "LONG_TERM_MEMORY_DATABASE_URL_ONPREM"
    )
    value = os.getenv(variable)
    if not value:
        raise ValueError(f"{variable} is required for persistent long-term memory.")
    _require_postgres_tls(value, "long-term memory")
    return value


def _embed_memory(texts: Sequence[str]) -> list[list[float]]:
    from src.config.config import get_embeddings

    return get_embeddings().embed_documents(list(texts))


@contextmanager
def open_memory_store() -> Iterator[object]:
    """Yield PostgreSQL/pgvector memory; in-memory is explicit test-only."""
    backend = os.getenv("LONG_TERM_MEMORY_BACKEND", "postgres").lower()
    if backend == "memory":
        if os.getenv("DEPLOYMENT_ENVIRONMENT") == "production":
            raise ValueError("In-memory long-term memory is forbidden in production.")
        from langgraph.store.memory import InMemoryStore

        yield InMemoryStore()
        return
    if backend != "postgres":
        raise ValueError("LONG_TERM_MEMORY_BACKEND must be 'postgres' or 'memory'.")

    from langgraph.store.postgres import PostgresStore

    from src.config.config import VECTOR_SIZE

    index = IndexConfig(embed=_embed_memory, dims=VECTOR_SIZE, fields=["memory_text"])
    with PostgresStore.from_conn_string(
        _memory_connection_string(), index=index
    ) as store:
        yield store


def initialize_memory_schema() -> None:
    """Run BaseStore/pgvector migrations from a controlled deployment job."""
    from langgraph.store.postgres import PostgresStore

    from src.config.config import VECTOR_SIZE

    index = IndexConfig(embed=_embed_memory, dims=VECTOR_SIZE, fields=["memory_text"])
    with PostgresStore.from_conn_string(
        _memory_connection_string(), index=index
    ) as store:
        store.setup()
