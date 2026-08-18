#!/usr/bin/env bash
set -euo pipefail

# Development profile for the two memory planes and audit transport.
# The resource group is intentionally shared so a test run can be removed as one unit.
: "${AZURE_SUBSCRIPTION_ID:?Set AZURE_SUBSCRIPTION_ID before running az commands}"

LOCATION="${AZURE_LOCATION:-westeurope}"
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-onyx-memory-dev}"
SUFFIX="${ONYX_RESOURCE_SUFFIX:-$(date +%s)}"
REDIS_NAME="${ONYX_REDIS_NAME:-onyx-redis-${SUFFIX}}"
COSMOS_NAME="${ONYX_COSMOS_NAME:-onyx-cosmos-${SUFFIX}}"
EVENTHUB_NAMESPACE="${ONYX_EVENTHUB_NAMESPACE:-onyx-audit-${SUFFIX}}"
STORAGE_NAME="${ONYX_STORAGE_NAME:-onyxaudit${SUFFIX}}"

az account set --subscription "$AZURE_SUBSCRIPTION_ID"
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --tags workload=onyx environment=dev

# Short-term / transactional plane: Azure Managed Redis, minimum Balanced SKU.
az redisenterprise create \
  --name "$REDIS_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --sku Balanced_B0
az redisenterprise database create \
  --cluster-name "$REDIS_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --name default \
  --client-protocol Encrypted \
  --clustering-policy OSSCluster \
  --eviction-policy VolatileLRU \
  --port 10000

# Long-term / cross-thread plane: Cosmos DB serverless, one region and eventual consistency.
az cosmosdb create \
  --name "$COSMOS_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --default-consistency-level Eventual \
  --locations regionName="$LOCATION" failoverPriority=0 isZoneRedundant=False \
  --capabilities EnableServerless
az cosmosdb sql database create \
  --account-name "$COSMOS_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --name onyx-memory
az cosmosdb sql container create \
  --account-name "$COSMOS_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --database-name onyx-memory \
  --name memories \
  --partition-key-path /namespace_key

# Independent audit transport: Basic Event Hubs (one-day retention, one hub).
az eventhubs namespace create \
  --name "$EVENTHUB_NAMESPACE" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --sku Basic \
  --capacity 1
az eventhubs eventhub create \
  --name agent-audit \
  --namespace-name "$EVENTHUB_NAMESPACE" \
  --resource-group "$RESOURCE_GROUP" \
  --partition-count 1 \
  --message-retention 1

# Immutable archival target. Locking the policy is intentionally a separate,
# explicit operation because it makes the container non-deletable while retained.
az storage account create \
  --name "$STORAGE_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --sku Standard_LRS \
  --kind StorageV2 \
  --https-only true \
  --min-tls-version TLS1_2 \
  --allow-blob-public-access false
az storage container create \
  --name audit \
  --account-name "$STORAGE_NAME" \
  --auth-mode login
az storage container immutability-policy create \
  --account-name "$STORAGE_NAME" \
  --container-name audit \
  --period 30 \
  --allow-protected-append-writes true \
  --auth-mode login

cat <<EOF
Created resource group: $RESOURCE_GROUP
Set these application variables (use managed identity in production):
  INFRASTRUCTURE_MODE=cloud
  LANGGRAPH_CHECKPOINTER=redis
  LANGGRAPH_CHECKPOINT_REDIS_URI_CLOUD=rediss://$REDIS_NAME.redis.azure.net:10000
  LONG_TERM_MEMORY_BACKEND=cosmos
  COSMOS_ENDPOINT=$(az cosmosdb show --name "$COSMOS_NAME" --resource-group "$RESOURCE_GROUP" --query documentEndpoint -o tsv)
  COSMOS_DATABASE=onyx-memory
  COSMOS_CONTAINER=memories
  AUDIT_BACKEND=eventhub
  AZURE_EVENT_HUB_NAMESPACE=$EVENTHUB_NAMESPACE.servicebus.windows.net
  AZURE_EVENT_HUB_NAME=agent-audit
  AUDIT_ARCHIVE_STORAGE_ACCOUNT=$STORAGE_NAME

To remove the dev environment later:
  az group delete --name "$RESOURCE_GROUP" --yes --no-wait
EOF
