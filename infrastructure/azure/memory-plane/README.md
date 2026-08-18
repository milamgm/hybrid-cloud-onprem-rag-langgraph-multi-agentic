# Azure memory-plane deployment

`deploy.sh` provisions the development profile for the
architecture:

- short-term checkpoints: Azure Managed Redis `Balanced_B0`, with a TTL;
- long-term memory: one-region, serverless Azure Cosmos DB for NoSQL;
- audit transport: one Basic Event Hubs namespace and one hub;
- audit archive: Standard locally redundant Blob Storage with a 30-day append
  immutability policy.

It requires Azure CLI login and an active subscription:

```bash
az login
export AZURE_SUBSCRIPTION_ID="..."
export AZURE_LOCATION="westeurope"
bash infrastructure/azure/memory-plane/deploy.sh
```

The script does not lock the Blob policy. Locking is a deliberate compliance
decision because retained data then prevents normal deletion of the container.
Do not use this dev profile for production without private networking, managed
identity/RBAC, customer data residency review, backup/restore testing and an
independent Event Hubs consumer that archives events to Blob.
