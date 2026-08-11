"""Secret retrieval through workload identity, never application source code.

Cloud mode reads Azure Key Vault using DefaultAzureCredential, which resolves a
managed or federated workload identity in Azure. On-premise mode authenticates
to HashiCorp Vault with a projected OIDC JWT and preserves Vault lease data so
callers can renew or revoke dynamic credentials deliberately.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SecretLease:
    """A secret value and, when applicable, its short-lived Vault lease."""

    value: str | None = field(default=None, repr=False)
    data: Mapping[str, str] = field(default_factory=dict, repr=False)
    lease_id: str | None = None
    expires_at: datetime | None = None
    renewable: bool = False
    version: str | None = None

    @property
    def expiring(self) -> bool:
        """Whether the secret should be refreshed before its next use."""
        return bool(self.expires_at and self.expires_at <= datetime.now(UTC))


class SecretProvider:
    """Read static or dynamic secrets from the selected deployment provider."""

    def __init__(self) -> None:
        self._mode = os.getenv("INFRASTRUCTURE_MODE", "on_premise").lower()
        self._vault = None

    def get_secret(self, name: str) -> SecretLease:
        """Read the current version of a named static secret."""
        if self._mode == "cloud":
            return self._get_azure_secret(name)
        return self._get_vault_kv_secret(name)

    def get_dynamic_secret(self, path: str) -> SecretLease:
        """Request a short-lived credential from a Vault dynamic secrets engine."""
        if self._mode == "cloud":
            raise ValueError(
                "Azure Key Vault stores secrets but does not issue Vault-style dynamic "
                "credential leases. Use a managed identity for Azure resources."
            )
        response = self._get_vault_client().read(path)
        if not response or not response.get("data"):
            raise LookupError(f"Vault returned no secret for path {path!r}.")
        return self._lease_from_vault(response)

    def renew(
        self, lease: SecretLease, *, increment_seconds: int | None = None
    ) -> SecretLease:
        """Renew a Vault lease and return its updated expiry metadata."""
        if self._mode == "cloud":
            raise ValueError(
                "Azure Key Vault secrets are rotated by version, not renewed."
            )
        if not lease.lease_id or not lease.renewable:
            raise ValueError(
                "The supplied secret does not have a renewable Vault lease."
            )
        response = self._get_vault_client().sys.renew_lease(
            lease.lease_id, increment=increment_seconds
        )
        return self._lease_from_vault(response, data=lease.data)

    def revoke(self, lease: SecretLease) -> None:
        """Immediately revoke a Vault dynamic credential lease."""
        if self._mode == "cloud":
            raise ValueError(
                "Azure Key Vault secret versions cannot be revoked as leases."
            )
        if not lease.lease_id:
            raise ValueError("The supplied secret does not have a Vault lease ID.")
        self._get_vault_client().sys.revoke_lease(lease.lease_id)

    @staticmethod
    def _single_value(data: Mapping[str, Any], path: str) -> str:
        values = [value for value in data.values() if isinstance(value, str)]
        if len(values) != 1:
            raise ValueError(
                f"Vault path {path!r} has multiple fields; use a dedicated credential "
                "adapter instead of selecting a secret implicitly."
            )
        return values[0]

    def _get_azure_secret(self, name: str) -> SecretLease:
        vault_url = os.getenv("AZURE_KEY_VAULT_URL")
        if not vault_url:
            raise ValueError("Cloud secret retrieval requires AZURE_KEY_VAULT_URL.")
        try:
            from azure.identity import DefaultAzureCredential
            from azure.keyvault.secrets import SecretClient
        except ImportError as error:  # pragma: no cover - dependency contract
            raise RuntimeError(
                "Install azure-identity and azure-keyvault-secrets for Azure Key Vault."
            ) from error

        credential = DefaultAzureCredential(
            managed_identity_client_id=os.getenv("AZURE_CLIENT_ID") or None
        )
        secret = SecretClient(vault_url=vault_url, credential=credential).get_secret(
            name
        )
        return SecretLease(value=secret.value, version=secret.properties.version)

    def _get_vault_kv_secret(self, name: str) -> SecretLease:
        mount_point = os.getenv("VAULT_KV_MOUNT", "secret")
        response = self._get_vault_client().secrets.kv.v2.read_secret_version(
            path=name, mount_point=mount_point
        )
        return SecretLease(value=self._single_value(response["data"]["data"], name))

    def _get_vault_client(self):
        if self._vault is not None:
            return self._vault
        vault_addr = os.getenv("VAULT_ADDR")
        role = os.getenv("VAULT_JWT_ROLE")
        if not vault_addr or not role:
            raise ValueError(
                "On-premise secret retrieval requires VAULT_ADDR and VAULT_JWT_ROLE."
            )
        try:
            import hvac
        except ImportError as error:  # pragma: no cover - dependency contract
            raise RuntimeError(
                "Install hvac for HashiCorp Vault secret retrieval."
            ) from error

        client = hvac.Client(url=vault_addr, verify=os.getenv("VAULT_CACERT") or True)
        jwt = self._read_workload_jwt()
        client.auth.jwt.jwt_login(role=role, jwt=jwt)
        if not client.is_authenticated():
            raise PermissionError("Vault rejected the workload OIDC token.")
        self._vault = client
        return client

    @staticmethod
    def _read_workload_jwt() -> str:
        token = os.getenv("VAULT_JWT")
        if token:
            return token
        token_file = os.getenv("VAULT_JWT_FILE")
        if token_file:
            return Path(token_file).read_text(encoding="utf-8").strip()
        raise ValueError(
            "Vault JWT authentication requires VAULT_JWT_FILE (a projected workload "
            "token) or VAULT_JWT. Do not use VAULT_TOKEN in production."
        )

    @staticmethod
    def _lease_from_vault(
        response: dict[str, Any], *, data: Mapping[str, str] | None = None
    ) -> SecretLease:
        duration = int(response.get("lease_duration", 0))
        expires_at = (
            datetime.now(UTC) + timedelta(seconds=duration) if duration else None
        )
        return SecretLease(
            data=data
            or {
                key: value
                for key, value in response["data"].items()
                if isinstance(value, str)
            },
            lease_id=response.get("lease_id") or None,
            expires_at=expires_at,
            renewable=bool(response.get("renewable", False)),
        )
