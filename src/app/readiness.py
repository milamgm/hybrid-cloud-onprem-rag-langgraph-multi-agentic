"""Read-only readiness checks for cloud and on-premise profiles."""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class ReadinessReport:
    mode: str
    checks: tuple[ReadinessCheck, ...]

    @property
    def ready(self) -> bool:
        return all(check.ok for check in self.checks)


def _required(name: str) -> ReadinessCheck:
    value = os.getenv(name, "").strip()
    return ReadinessCheck(name, bool(value), "configured" if value else "missing")


def _configured_or_default(name: str, default: str) -> ReadinessCheck:
    value = os.getenv(name, "").strip()
    return ReadinessCheck(
        name,
        True,
        "configured" if value else f"using default {default}",
    )


def _tcp(name: str, url: str | None, default_port: int) -> ReadinessCheck:
    if not url:
        return ReadinessCheck(name, False, "URL missing")
    parsed = urlparse(url.replace("postgresql+psycopg://", "postgresql://"))
    host = parsed.hostname
    port = parsed.port or default_port
    if not host:
        return ReadinessCheck(name, False, "invalid URL")
    try:
        with socket.create_connection((host, port), timeout=3):
            return ReadinessCheck(name, True, f"reachable on port {port}")
    except OSError as error:
        return ReadinessCheck(name, False, f"unreachable on port {port}: {error}")


def _postgres(mode: str) -> ReadinessCheck:
    variable = "PG_CONNECTION_CLOUD" if mode == "cloud" else "PG_CONNECTION_ONPREM"
    connection_string = os.getenv(variable)
    if not connection_string:
        return ReadinessCheck("postgres", False, f"{variable} missing")
    try:
        import psycopg

        dsn = connection_string.replace("postgresql+psycopg://", "postgresql://")
        with psycopg.connect(dsn, connect_timeout=5) as connection:
            vector = connection.execute(
                "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')"
            ).fetchone()[0]
        detail = (
            "connected; pgvector enabled" if vector else "connected; pgvector missing"
        )
        return ReadinessCheck("postgres", bool(vector), detail)
    except Exception as error:
        return ReadinessCheck("postgres", False, f"connection failed: {error}")


def _redis(mode: str) -> ReadinessCheck:
    variable = (
        "RAG_CONTEXT_REDIS_URI_CLOUD"
        if mode == "cloud"
        else "RAG_CONTEXT_REDIS_URI_ONPREM"
    )
    uri = os.getenv(variable) or os.getenv("REDIS_URI")
    if not uri:
        if os.getenv("RAG_CONTEXT_CACHE", "memory") == "memory":
            return ReadinessCheck("context_cache", True, "in-memory development cache")
        return ReadinessCheck("redis", False, f"{variable} or REDIS_URI missing")
    try:
        import redis

        client = redis.Redis.from_url(uri, socket_connect_timeout=3)
        client.ping()
        return ReadinessCheck("redis", True, "connected")
    except Exception as error:
        return ReadinessCheck("redis", False, f"connection failed: {error}")


def check_readiness() -> ReadinessReport:
    """Probe required configuration and network dependencies without writes."""
    mode = os.getenv("INFRASTRUCTURE_MODE", "on_premise").lower()
    checks: list[ReadinessCheck] = [_postgres(mode), _redis(mode)]
    if mode == "cloud":
        checks.extend(
            [
                _required("AZURE_FOUNDRY_ENDPOINT"),
                _required("AZURE_FOUNDRY_API_KEY"),
                _configured_or_default(
                    "AZURE_OPENAI_DEPLOYMENT_NAME", "text-embedding-3-large"
                ),
                _configured_or_default("AZURE_CHAT_DEPLOYMENT", "gpt-5.4-mini"),
            ]
        )
        if os.getenv("RAG_SECURITY_PROFILE", "development") == "managed":
            checks.extend(
                [
                    _required("AZURE_CONTENT_SAFETY_ENDPOINT"),
                    _required("AZURE_CONTENT_SAFETY_KEY"),
                ]
            )
    else:
        chat_url = os.getenv("ONPREM_LITELLM_BASE_URL") or os.getenv(
            "ONPREM_CHAT_BASE_URL", "http://127.0.0.1:1234/v1"
        )
        checks.append(_tcp("onprem_chat", chat_url, 1234))
    return ReadinessReport(mode=mode, checks=tuple(checks))
