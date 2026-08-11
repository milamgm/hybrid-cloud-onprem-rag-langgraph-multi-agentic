# Static application configuration only. Dynamic database credentials are read
# through database/creds/<role>, which creates a short-lived renewable lease.
path "secret/data/onyx/*" {
  capabilities = ["read"]
}

path "database/creds/onyx-readonly" {
  capabilities = ["read"]
}
