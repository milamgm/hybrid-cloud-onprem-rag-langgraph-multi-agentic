# HashiCorp Vault

Vault is the on-premise secret authority. The application authenticates with a
short-lived OIDC workload JWT, not a Vault root token or a long-lived AppRole
secret. Use Keycloak or Microsoft Entra ID as the JWT issuer and configure
Vault's JWT auth method to validate its discovery URL, audience and bound
claims.

Before starting this production-shaped compose file, provide `tls/tls.crt` and
`tls/tls.key`, set `api_addr` in `config/vault.hcl` to the real internal URL,
initialize and unseal Vault through your approved operator process. Do not use
`vault server -dev` outside an isolated development environment.

Configure the workload with a projected token:

```env
VAULT_ADDR=https://vault.example.internal:8200
VAULT_CACERT=/var/run/secrets/onyx/ca.crt
VAULT_JWT_ROLE=onyx-workload
VAULT_JWT_FILE=/var/run/secrets/onyx/workload.jwt
VAULT_KV_MOUNT=secret
```

Apply `policies/onyx-app.hcl` to `onyx-app`. Bind the JWT role to the exact
issuer, audience and subject of the workload. The database secrets engine role
`onyx-readonly` should use a short default TTL and a bounded max TTL; its leases
are renewed or revoked through `SecretProvider`.
