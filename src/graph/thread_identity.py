"""Identity-scoped public thread identifiers for checkpoint access."""

from __future__ import annotations

import hashlib


def checkpoint_thread_id(
    tenant_id: str,
    subject_id: str,
    public_thread_id: str,
) -> str:
    """Derive an opaque checkpoint key from verified identity claims.

    The public investigation/thread id remains stable for APIs and audit
    records, while the checkpointer never receives a caller-controlled id that
    could collide across tenants or subjects.
    """

    values = {
        "tenant_id": tenant_id,
        "subject_id": subject_id,
        "public_thread_id": public_thread_id,
    }
    for name, value in values.items():
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        if len(value) > 256 or any(
            ord(character) < 32 or ord(character) == 127 for character in value
        ):
            raise ValueError(f"{name} exceeds the safe identifier bounds")
    value = f"{tenant_id}\x00{subject_id}\x00{public_thread_id}".encode()
    return hashlib.sha256(value).hexdigest()


def case_checkpoint_thread_id(tenant_id: str, case_id: str) -> str:
    """Derive a case-owned checkpoint key, independent of customer or reviewer.

    A fraud analyst must be able to resume a case concerning another person;
    including either identity in the key would make that control-plane flow
    incorrect.  Tenant isolation and case identity are the stable boundary.
    """
    for name, value in {"tenant_id": tenant_id, "case_id": case_id}.items():
        if (
            not isinstance(value, str)
            or not value
            or value != value.strip()
            or len(value) > 256
            or any(ord(char) < 32 or ord(char) == 127 for char in value)
        ):
            raise ValueError(f"{name} exceeds the safe identifier bounds")
    return hashlib.sha256(
        f"case\x00{tenant_id}\x00{case_id}".encode()
    ).hexdigest()
