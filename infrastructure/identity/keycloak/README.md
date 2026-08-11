# Keycloak workload identity

The imported realm creates an OIDC confidential client named `onyx-workload`
with service accounts enabled and an access-token audience of `vault`. Create
the client credential in Keycloak after import and deliver it only through the
deployment secret mechanism. Prefer workload federation (Kubernetes projected
tokens, cloud workload identity or mTLS) over a reusable client secret.

Configure Vault JWT auth with Keycloak's realm discovery URL, expected audience
`vault`, and claims that uniquely identify `onyx-workload`. Do not grant Vault
policies from unbounded realm roles or email claims.
