"""Encrypted durable checkpoint lifecycle for cloud and on-prem deployments."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from urllib.parse import parse_qs, urlparse


def _require_postgres_tls(connection_string: str, purpose: str) -> None:
    if os.getenv("DEPLOYMENT_ENVIRONMENT") != "production":
        return
    sslmode = parse_qs(urlparse(connection_string).query).get("sslmode", [""])[0]
    if sslmode not in {"require", "verify-ca", "verify-full"}:
        raise ValueError(
            f"Production {purpose} PostgreSQL must use sslmode=require, "
            "verify-ca or verify-full."
        )


def _connection_string() -> str:
    mode = os.getenv("INFRASTRUCTURE_MODE", "on_premise").lower()
    variable = (
        "LANGGRAPH_CHECKPOINT_DATABASE_URL_CLOUD"
        if mode == "cloud"
        else "LANGGRAPH_CHECKPOINT_DATABASE_URL_ONPREM"
    )
    connection_string = os.getenv(variable)
    if not connection_string:
        raise ValueError(f"{variable} is required for durable LangGraph checkpoints.")
    _require_postgres_tls(connection_string, "checkpoint")
    return connection_string


def _redis_connection_string() -> str:
    mode = os.getenv("INFRASTRUCTURE_MODE", "on_premise").lower()
    variable = (
        "LANGGRAPH_CHECKPOINT_REDIS_URI_CLOUD"
        if mode == "cloud"
        else "LANGGRAPH_CHECKPOINT_REDIS_URI_ONPREM"
    )
    connection_string = os.getenv(variable) or os.getenv("REDIS_URI")
    if not connection_string:
        raise ValueError(f"{variable} is required for Redis checkpoints.")
    if os.getenv(
        "DEPLOYMENT_ENVIRONMENT"
    ) == "production" and not connection_string.startswith("rediss://"):
        raise ValueError("Production Redis checkpoints require TLS (rediss://).")
    return connection_string


def _encrypted_serializer():
    if not os.getenv("LANGGRAPH_AES_KEY"):
        raise ValueError("LANGGRAPH_AES_KEY must be supplied by Key Vault or Vault.")
    from langgraph.checkpoint.serde.encrypted import EncryptedSerializer

    return EncryptedSerializer.from_pycryptodome_aes()


@contextmanager
def open_checkpointer() -> Iterator[object]:
    """Yield Redis in cloud, PostgreSQL on-prem, or explicit test memory."""
    mode = os.getenv("INFRASTRUCTURE_MODE", "on_premise").lower()
    default_backend = "redis" if mode == "cloud" else "postgres"
    backend = os.getenv("LANGGRAPH_CHECKPOINTER", default_backend).lower()
    if backend == "memory":
        if os.getenv("DEPLOYMENT_ENVIRONMENT") == "production":
            raise ValueError("In-memory checkpointing is forbidden in production.")
        from langgraph.checkpoint.memory import InMemorySaver

        yield InMemorySaver()
        return
    if backend != "postgres":
        if backend != "redis":
            raise ValueError(
                "LANGGRAPH_CHECKPOINTER must be 'postgres', 'redis' or 'memory'."
            )
        from langgraph.checkpoint.redis import RedisSaver

        with RedisSaver.from_conn_string(
            _redis_connection_string(),
            ttl={
                "default_ttl": float(os.getenv("CHECKPOINT_TTL_MINUTES", "1440")),
                "refresh_on_read": False,
            },
        ) as saver:
            yield saver
        return
    import psycopg
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row

    serializer = _encrypted_serializer()
    with psycopg.connect(
        _connection_string(),
        autocommit=True,
        prepare_threshold=0,
        row_factory=dict_row,
    ) as connection:
        yield PostgresSaver(connection, serde=serializer)


def initialize_checkpoint_schema() -> None:
    """Run checkpoint migrations as a deployment job, never at request startup."""
    backend = os.getenv(
        "LANGGRAPH_CHECKPOINTER",
        "redis"
        if os.getenv("INFRASTRUCTURE_MODE", "on_premise").lower() == "cloud"
        else "postgres",
    ).lower()
    if backend == "redis":
        from langgraph.checkpoint.redis import RedisSaver

        with RedisSaver.from_conn_string(_redis_connection_string()) as saver:
            saver.setup()
        return
    import psycopg
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row

    serializer = _encrypted_serializer()
    with psycopg.connect(
        _connection_string(),
        autocommit=True,
        prepare_threshold=0,
        row_factory=dict_row,
    ) as connection:
        PostgresSaver(connection, serde=serializer).setup()
