"""Encrypted durable checkpoint lifecycle for cloud and on-prem deployments."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from urllib.parse import parse_qs, urlparse

_CHECKPOINT_ALLOWED_TYPES = (
    ("src.events.contracts", "ForensicFinding"),
    ("src.events.contracts", "CaseEvidenceBundle"),
    ("src.events.contracts", "CoreBankingEvidence"),
    ("src.events.contracts", "InvestigationReport"),
    ("src.events.contracts", "PolicyCitation"),
    ("src.events.contracts", "TransactionFacts"),
    ("src.events.contracts", "TransactionRiskAlert"),
    ("src.hitl.models", "ApprovalDecision"),
    ("src.state.schema", "AgentInput"),
    ("src.state.schema", "AgentOutput"),
    ("src.state.schema", "AgentState"),
    ("src.state.schema", "CitationReference"),
    ("src.state.schema", "SecurityEvent"),
)


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


def _cosmos_connection_info() -> dict[str, str | None]:
    endpoint = os.getenv("LANGGRAPH_CHECKPOINT_COSMOS_ENDPOINT") or os.getenv(
        "COSMOS_ENDPOINT"
    )
    if not endpoint:
        raise ValueError(
            "LANGGRAPH_CHECKPOINT_COSMOS_ENDPOINT or COSMOS_ENDPOINT is required "
            "for Cosmos DB checkpoints."
        )
    return {
        "endpoint": endpoint,
        # An omitted key deliberately selects DefaultAzureCredential/managed identity.
        "key": os.getenv("LANGGRAPH_CHECKPOINT_COSMOS_KEY") or os.getenv("COSMOS_KEY"),
        "database_name": os.getenv(
            "LANGGRAPH_CHECKPOINT_COSMOS_DATABASE", "onyx_checkpoints"
        ),
        "container_name": os.getenv(
            "LANGGRAPH_CHECKPOINT_COSMOS_CONTAINER", "forensic_threads"
        ),
    }


def _checkpoint_backend() -> str:
    mode = os.getenv("INFRASTRUCTURE_MODE", "on_premise").lower()
    default_backend = "redis" if mode == "cloud" else "postgres"
    return (os.getenv("LANGGRAPH_CHECKPOINTER") or default_backend).strip().lower()


def _encrypted_serializer():
    if not os.getenv("LANGGRAPH_AES_KEY"):
        raise ValueError("LANGGRAPH_AES_KEY must be supplied by Key Vault or Vault.")
    from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    strict = (
        os.getenv("DEPLOYMENT_ENVIRONMENT", "development").lower() == "production"
        or os.getenv("LANGGRAPH_STRICT_MSGPACK", "false").lower() == "true"
    )
    serde = JsonPlusSerializer(
        pickle_fallback=False,
        allowed_msgpack_modules=_CHECKPOINT_ALLOWED_TYPES if strict else True,
        allowed_json_modules=_CHECKPOINT_ALLOWED_TYPES if strict else True,
    )
    return EncryptedSerializer.from_pycryptodome_aes(serde=serde)


@asynccontextmanager
async def open_async_checkpointer() -> AsyncIterator[object]:
    """Yield an async durable saver for API/worker graph execution.

    HITL graphs are long-lived and commonly resumed from async HTTP or event
    workers. Using the async saver avoids running blocking Postgres/Redis
    checkpoint operations on the event loop. Schema/index creation remains
    explicit here and should still be promoted to a deployment migration in
    tightly controlled production environments.
    """

    backend = _checkpoint_backend()
    if backend == "memory":
        if os.getenv("DEPLOYMENT_ENVIRONMENT") == "production":
            raise ValueError("In-memory checkpointing is forbidden in production.")
        from langgraph.checkpoint.memory import InMemorySaver

        yield InMemorySaver()
        return
    if backend == "redis":
        from langgraph.checkpoint.redis.aio import AsyncRedisSaver

        async with AsyncRedisSaver.from_conn_string(
            _redis_connection_string(),
            ttl={
                "default_ttl": float(os.getenv("CHECKPOINT_TTL_MINUTES", "1440")),
                "refresh_on_read": False,
            },
        ) as saver:
            yield saver
        return
    if backend == "cosmos":
        from langchain_azure_cosmosdb import CosmosDBSaver

        async with CosmosDBSaver.from_conn_info(
            **_cosmos_connection_info(),
            serde=_encrypted_serializer(),
        ) as saver:
            yield saver
        return
    if backend != "postgres":
        raise ValueError(
            "LANGGRAPH_CHECKPOINTER must be 'postgres', 'redis', 'cosmos' or 'memory'."
        )

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    async with AsyncPostgresSaver.from_conn_string(
        _connection_string(),
        pipeline=False,
        serde=_encrypted_serializer(),
    ) as saver:
        await saver.setup()
        yield saver


@contextmanager
def open_checkpointer() -> Iterator[object]:
    """Yield a sync Redis/PostgreSQL saver or explicit test memory.

    Cosmos is intentionally async-only here because its sync integration does
    not accept the application-level encrypted serializer.
    """
    backend = _checkpoint_backend()
    if backend == "memory":
        if os.getenv("DEPLOYMENT_ENVIRONMENT") == "production":
            raise ValueError("In-memory checkpointing is forbidden in production.")
        from langgraph.checkpoint.memory import InMemorySaver

        yield InMemorySaver()
        return
    if backend == "cosmos":
        # The official sync Cosmos saver does not expose the custom serializer
        # supported by its async counterpart. Keep the sync path fail-closed so
        # sensitive forensic state uses application-level encryption.
        raise ValueError(
            "LANGGRAPH_CHECKPOINTER=cosmos requires open_async_checkpointer(); "
            "use CosmosDBSaver with the encrypted serializer in async workers."
        )
    if backend != "postgres":
        if backend != "redis":
            raise ValueError(
                "LANGGRAPH_CHECKPOINTER must be 'postgres', 'redis', 'cosmos' or 'memory'."
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
    backend = _checkpoint_backend()
    if backend == "cosmos":
        # Cosmos creates its database/container through the official saver. This
        # is a deployment-time operation, not a request-startup migration.
        from langchain_azure_cosmosdb import CosmosDBSaverSync

        with CosmosDBSaverSync(**_cosmos_connection_info()) as saver:
            del saver
        return
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
