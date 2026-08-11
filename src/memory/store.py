"""Governed cross-thread memory backed by a LangGraph persistent BaseStore."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from uuid import uuid4

from langgraph.store.base import BaseStore


class MemoryKind(StrEnum):
    """Long-term memory classes with different write policies."""

    PRESENTATION = "presentation"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


@dataclass(frozen=True)
class MemoryWrite:
    """A reviewed memory write with provenance, purpose and expiry."""

    tenant_id: str
    subject_id: str
    kind: MemoryKind
    memory_text: str
    source_ref: str
    purpose: str
    legal_basis: str
    approved_by: str
    classification: str = "internal"
    retention_days: int = 90
    logical_key: str | None = None


@dataclass(frozen=True)
class MemoryRecord:
    """Immutable version stored under a tenant/subject namespace."""

    memory_id: str
    logical_key: str
    version: int
    supersedes: str | None
    kind: str
    memory_text: str
    source_ref: str
    purpose: str
    legal_basis: str
    approved_by: str
    classification: str
    created_at: str
    expires_at: str
    integrity_hmac: str


class MemoryManager:
    """Validate, version, retrieve and erase governed long-term memory.

    Memory creation is never inferred automatically from chat. A trusted
    workflow must submit a ``MemoryWrite`` carrying provenance and approval.
    Versions are append-only; readers select the newest valid version per
    logical key.
    """

    _PRESENTATION_VALUES = {
        "answer_language": frozenset({"en", "es", "fr", "de", "pt"}),
        "response_format": frozenset({"concise", "bullets", "detailed"}),
    }

    def __init__(self, store: BaseStore, *, integrity_key: bytes | None = None) -> None:
        configured_key = os.getenv("MEMORY_INTEGRITY_KEY", "").encode()
        self._integrity_key = integrity_key or configured_key
        if not self._integrity_key:
            if os.getenv("DEPLOYMENT_ENVIRONMENT") == "production":
                raise ValueError(
                    "MEMORY_INTEGRITY_KEY must be supplied by Key Vault or Vault."
                )
            self._integrity_key = b"development-only-memory-integrity-key"
        if (
            os.getenv("DEPLOYMENT_ENVIRONMENT") == "production"
            and len(self._integrity_key) < 32
        ):
            raise ValueError("MEMORY_INTEGRITY_KEY must contain at least 32 bytes.")
        self._store = store

    @staticmethod
    def _namespace(
        tenant_id: str, subject_id: str, kind: MemoryKind
    ) -> tuple[str, ...]:
        return ("tenants", tenant_id, "subjects", subject_id, "memory", kind.value)

    def commit(self, request: MemoryWrite) -> MemoryRecord:
        """Append an approved memory version and return immutable metadata."""
        self._validate_write(request)
        namespace = self._namespace(request.tenant_id, request.subject_id, request.kind)
        logical_key = self._logical_key(request)
        previous = self._latest_version(namespace, logical_key)
        now = datetime.now(UTC)
        unsigned = {
            "memory_id": str(uuid4()),
            "logical_key": logical_key,
            "version": previous.version + 1 if previous else 1,
            "supersedes": previous.memory_id if previous else None,
            "kind": request.kind.value,
            "memory_text": request.memory_text,
            "source_ref": request.source_ref,
            "purpose": request.purpose,
            "legal_basis": request.legal_basis,
            "approved_by": request.approved_by,
            "classification": request.classification,
            "created_at": now.isoformat(),
            "expires_at": (now + timedelta(days=request.retention_days)).isoformat(),
        }
        record = MemoryRecord(
            **unsigned,
            integrity_hmac=self._sign(unsigned),
        )
        self._store.put(
            namespace,
            record.memory_id,
            asdict(record),
            index=["memory_text"] if request.kind is MemoryKind.SEMANTIC else False,
        )
        return record

    def presentation_preferences(
        self, tenant_id: str, subject_id: str
    ) -> dict[str, str]:
        """Load current presentation preferences without model inference."""
        namespace = self._namespace(tenant_id, subject_id, MemoryKind.PRESENTATION)
        preferences: dict[str, str] = {}
        for key in self._PRESENTATION_VALUES:
            record = self._latest_version(namespace, key)
            if record and not self._expired(record):
                _, value = record.memory_text.split("=", 1)
                preferences[key] = value
        return preferences

    def recall(
        self, tenant_id: str, subject_id: str, query: str, *, limit: int = 3
    ) -> list[MemoryRecord]:
        """Semantic recall within exactly one tenant and subject namespace."""
        namespace = self._namespace(tenant_id, subject_id, MemoryKind.SEMANTIC)
        records: list[MemoryRecord] = []
        seen: set[str] = set()
        for item in self._store.search(
            namespace, query=query, limit=max(limit * 3, 10)
        ):
            record = self._validated_record(item.value)
            if record.logical_key not in seen and not self._expired(record):
                latest = self._latest_version(namespace, record.logical_key)
                if latest and not self._expired(latest):
                    records.append(latest)
                    seen.add(record.logical_key)
            if len(records) == limit:
                break
        return records

    def forget(
        self, tenant_id: str, subject_id: str, kind: MemoryKind, logical_key: str
    ) -> int:
        """Erase every version of one exact tenant-scoped logical memory."""
        namespace = self._namespace(tenant_id, subject_id, kind)
        deleted = 0
        while batch := self._store.search(
            namespace, filter={"logical_key": logical_key}, limit=100
        ):
            for item in batch:
                record = self._validated_record(item.value)
                if record.logical_key != logical_key:
                    raise ValueError("Memory store returned a cross-key record.")
                self._store.delete(namespace, item.key)
                deleted += 1
        return deleted

    def purge_expired(self, tenant_id: str, subject_id: str, kind: MemoryKind) -> int:
        """Delete expired versions; call from the deployment retention job."""
        namespace = self._namespace(tenant_id, subject_id, kind)
        deleted = 0
        now = datetime.now(UTC).isoformat()
        while batch := self._store.search(
            namespace, filter={"expires_at": {"$lte": now}}, limit=100
        ):
            for item in batch:
                record = self._validated_record(item.value)
                if self._expired(record):
                    self._store.delete(namespace, item.key)
                    deleted += 1
        return deleted

    def _latest_version(
        self, namespace: tuple[str, ...], logical_key: str
    ) -> MemoryRecord | None:
        latest: MemoryRecord | None = None
        offset = 0
        while batch := self._store.search(
            namespace,
            filter={"logical_key": logical_key},
            limit=100,
            offset=offset,
        ):
            for item in batch:
                record = self._validated_record(item.value)
                if record.logical_key != logical_key:
                    raise ValueError("Memory store returned a cross-key record.")
                if latest is None or record.version > latest.version:
                    latest = record
            offset += len(batch)
        return latest

    def _validate_write(self, request: MemoryWrite) -> None:
        required = {
            "tenant_id": request.tenant_id,
            "subject_id": request.subject_id,
            "memory_text": request.memory_text,
            "source_ref": request.source_ref,
            "purpose": request.purpose,
            "legal_basis": request.legal_basis,
            "approved_by": request.approved_by,
        }
        missing = [name for name, value in required.items() if not value.strip()]
        if missing:
            raise ValueError(f"Memory write is missing governance fields: {missing}")
        if request.classification not in {
            "public",
            "internal",
            "confidential",
            "restricted",
        }:
            raise ValueError("Memory classification is not recognized.")
        if request.classification == "restricted":
            raise ValueError(
                "Restricted data is not permitted in long-term agent memory."
            )
        if not 1 <= request.retention_days <= 365:
            raise ValueError("Memory retention must be between 1 and 365 days.")
        if request.kind is MemoryKind.PRESENTATION:
            if "=" not in request.memory_text:
                raise ValueError("Presentation memory must use key=value.")
            key, value = request.memory_text.split("=", 1)
            if (
                key not in self._PRESENTATION_VALUES
                or value not in self._PRESENTATION_VALUES[key]
            ):
                raise ValueError("Presentation memory is outside the approved schema.")
            if request.logical_key and request.logical_key != key:
                raise ValueError(
                    "Presentation logical_key must match its approved key."
                )

    @staticmethod
    def _logical_key(request: MemoryWrite) -> str:
        if request.kind is MemoryKind.PRESENTATION:
            return request.memory_text.split("=", 1)[0]
        return request.logical_key or str(uuid4())

    def _sign(self, unsigned: dict) -> str:
        payload = json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        return hmac.new(self._integrity_key, payload, hashlib.sha256).hexdigest()

    def _validated_record(self, value: dict) -> MemoryRecord:
        record = MemoryRecord(**value)
        unsigned = asdict(record)
        supplied_signature = unsigned.pop("integrity_hmac")
        if not hmac.compare_digest(self._sign(unsigned), supplied_signature):
            raise ValueError(f"Memory integrity check failed for {record.memory_id}.")
        return record

    @staticmethod
    def _expired(record: MemoryRecord) -> bool:
        return datetime.fromisoformat(record.expires_at) <= datetime.now(UTC)
