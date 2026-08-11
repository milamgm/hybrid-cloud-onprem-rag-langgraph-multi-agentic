from __future__ import annotations

from src.security.secrets import SecretLease, SecretProvider


class FakeVault:
    def __init__(self):
        self.sys = self
        self.revoked: str | None = None

    def read(self, path):
        assert path == "database/creds/onyx-readonly"
        return {
            "data": {"username": "v-onyx-readonly", "password": "short-lived-password"},
            "lease_id": "database/creds/onyx-readonly/abc",
            "lease_duration": 300,
            "renewable": True,
        }

    def renew_lease(self, lease_id, increment=None):
        assert lease_id == "database/creds/onyx-readonly/abc"
        assert increment == 120
        return {"lease_id": lease_id, "lease_duration": 120, "renewable": True}

    def revoke_lease(self, lease_id):
        self.revoked = lease_id


def test_dynamic_vault_credentials_keep_lease_metadata(monkeypatch):
    monkeypatch.setenv("INFRASTRUCTURE_MODE", "on_premise")
    provider = SecretProvider()
    fake_vault = FakeVault()
    monkeypatch.setattr(provider, "_get_vault_client", lambda: fake_vault)

    lease = provider.get_dynamic_secret("database/creds/onyx-readonly")
    renewed = provider.renew(lease, increment_seconds=120)
    provider.revoke(lease)

    assert lease.value is None
    assert lease.data == {
        "username": "v-onyx-readonly",
        "password": "short-lived-password",
    }
    assert lease.renewable is True
    assert renewed.renewable is True
    assert fake_vault.revoked == lease.lease_id


def test_azure_key_vault_does_not_claim_vault_style_dynamic_leases(monkeypatch):
    monkeypatch.setenv("INFRASTRUCTURE_MODE", "cloud")
    provider = SecretProvider()

    try:
        provider.get_dynamic_secret("database/creds/readonly")
    except ValueError as error:
        assert "does not issue Vault-style dynamic" in str(error)
    else:
        raise AssertionError("Expected an Azure dynamic-secret error")


def test_static_vault_secret_requires_one_explicit_value(monkeypatch):
    monkeypatch.setenv("INFRASTRUCTURE_MODE", "on_premise")
    provider = SecretProvider()

    try:
        provider._single_value({"username": "onyx", "password": "secret"}, "app/db")
    except ValueError as error:
        assert "multiple fields" in str(error)
    else:
        raise AssertionError("Expected an ambiguous secret error")


def test_lease_value_is_not_rendered_in_repr():
    assert "short-lived-password" not in repr(
        SecretLease(
            value="short-lived-password", data={"password": "short-lived-password"}
        )
    )
