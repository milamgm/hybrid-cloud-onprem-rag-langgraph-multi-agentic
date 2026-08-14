# Governed hybrid agent state and memory

This implementation separates data by lifecycle and control boundary. It does
not treat LangGraph state, a cache and long-term memory as interchangeable.

| Plane | Purpose | Authoritative system | Retention |
|---|---|---|---|
| Short-term execution state | Resume one bounded workflow/thread | Encrypted LangGraph PostgreSQL (on-prem) or Redis checkpointer (cloud) | TTL plus delete by scoped thread ID |
| Ephemeral RAG context | Carry retrieved document bodies between nodes | Redis TTL cache (non-authoritative) | 30–3600 seconds and explicit end-of-run deletion |
| Governed long-term memory | Approved facts/preferences across threads | PostgreSQL + pgvector (on-prem) or Cosmos DB serverless (cloud) | Per-record expiry, erasure and append-only versions |
| Governance evidence | Decisions and audit references | Event Hubs -> immutable Blob/SIEM, independent from state | Governance policy; never mixed with user memory |

Redis is not written through to PostgreSQL. The systems contain different data:
Redis holds transient retrieved chunks; the checkpointer holds bounded execution
state; the long-term store accepts only an explicit, approved `MemoryWrite`.

## Hybrid deployment mapping

The application contracts stay identical in both modes.

| Capability | Azure cloud | On-premises |
|---|---|---|
| Checkpoints | Azure Managed Redis, encrypted and TTL-bound | PostgreSQL HA cluster, TLS and dedicated role |
| Ephemeral context | Azure Managed Redis, private endpoint and TLS | Redis Enterprise or Redis HA, ACLs and TLS |
| Long-term memory | Azure Cosmos DB serverless (JSON namespaces; vector adapter) | PostgreSQL with pgvector |
| Audit transport/archive | Basic Event Hubs + Standard_LRS Blob WORM | Kafka/RabbitMQ + immutable SIEM/WORM |
| Keys | Azure Key Vault workload identity | HashiCorp Vault federated workload identity |

Use separate databases and least-privilege roles for checkpoints and long-term
memory. Apply platform encryption at rest, private networking, backups and
regional placement according to the data classification and residency policy.

## Enforced controls

- The trusted API must supply `tenant_id`, `subject_id`, `thread_id`, roles,
  request correlation and data classification. Caller thread IDs are hashed
  with tenant and subject scope before reaching the checkpointer.
- Checkpoint payloads use LangGraph's AES encrypted serializer and cap message
  history. The first graph node preserves system instructions, summarizes old
  turns above the configured threshold and trims the active token budget.
  Retrieved document bodies never enter `AgentState`.
- Redis handles are opaque, tenant/thread-bound, classification-allowlisted and
  fail closed on expiry or scope mismatch. Context is deleted after evidence is
  recorded, including blocked executions. Confidential and restricted content
  is denied by default and requires a documented risk exception.
- Long-term writes require source, purpose, legal basis, approver and retention.
  Restricted data is rejected. Records are append-only versions protected by
  an HMAC key held in Key Vault or Vault.
- The model cannot write long-term memory. A separate trusted workflow calls
  `MemoryManager.commit` after policy/consent approval.
- Direct prompt attacks, retrieved-document attacks and invalid output are
  deterministic graph gates. Governance evidence stores references and
  decisions, not prompt/document bodies. Audit delivery is buffered and
  independent, so expiring a checkpoint cannot erase the audit trail.

## Deployment lifecycle

Schema creation is a controlled migration step, not application startup:

```python
from src.graph import initialize_checkpoint_schema
from src.memory import initialize_memory_schema

initialize_checkpoint_schema()
initialize_memory_schema()
```

At service startup, open managed connections and inject the resulting adapters:

```python
from src.graph import build_rag_graph, open_checkpointer
from src.memory import MemoryManager, RedisContextCache, open_memory_store

with open_checkpointer() as checkpointer, open_memory_store() as store:
    memory = MemoryManager(store)
    context_cache = RedisContextCache.from_environment()
    graph = build_rag_graph(dependencies, checkpointer=checkpointer)
```

The composition root must place `context_cache` and, in production, an
`AuditService.from_environment()` sink in `RAGDependencies`; the example above
omits construction of the other existing security dependencies.

For the inexpensive Azure development profile, run
`infrastructure/azure/memory-plane/deploy.sh` and set
`LANGGRAPH_CHECKPOINTER=redis`, `LONG_TERM_MEMORY_BACKEND=cosmos` and
`AUDIT_BACKEND=eventhub`.

Run a scheduled retention job that enumerates subjects from the authoritative
identity/data catalog and calls `MemoryManager.purge_expired` for each memory
kind. A data-subject erasure workflow calls `MemoryManager.forget` for all
logical memories and `RAGAgent.delete_thread` for known threads. Redis expiry is
a safety net; normal graph completion performs immediate deletion.

Rotate `LANGGRAPH_AES_KEY` and `MEMORY_INTEGRITY_KEY` through a versioned
decrypt/verify migration before retiring an old key. Replacing either key
without migration intentionally makes existing records unreadable or invalid.

## Operational rule

`LANGGRAPH_CHECKPOINTER=memory` and `LONG_TERM_MEMORY_BACKEND=memory` are test
options and fail closed when `DEPLOYMENT_ENVIRONMENT=production`. Production
PostgreSQL URLs must explicitly set `sslmode`; production Redis must use
`rediss://`. Run restore tests, tenant-isolation tests and retention evidence
checks as release gates.
# Event-driven forensic investigation

The transaction-risk path is implemented around versioned Pydantic contracts
and broker-neutral ports in `src/events`. A mock XGBoost publisher emits a
`risk.transaction.alert.v1` event asynchronously. The alert consumer validates
the envelope, claims its idempotency key, opens a LangGraph case, and
acknowledges the source message only after the graph has published an evidence
collection request. It does **not** notify a human yet.

The stages are deliberately separate:

```text
XGBoost -> risk alert stream -> case intake -> LangGraph case start
                                      |
                                      v
                     evidence collection requested
                                      |
         read-only core/AML/CRM gateway + governed policy RAG
                                      |
                         validated evidence ready
                                      |
                  reasoning -> cited case report -> review event
                                      |
                   LangGraph interrupt / analyst work queue
                                      |
                                      v
                         approval granted/rejected event
                                      |
                                      v
                         execution order requested (approved only)
```

The evidence worker is the only component that talks to banking data sources.
It uses repository-owned named operations with tenant, customer and transaction
parameters, a read-only service account and database row-level security. The
LLM has neither database credentials nor a raw-SQL tool: it reasons only over a
bounded `CaseEvidenceBundle`. Policy RAG is likewise a curated internal corpus
and returns cited snippets/references, not internet search results.

The graph emits a versioned approval-granted event after the external approval
service authenticates an authorised human decision, and emits an
execution-order command only for an approved decision. Separate consumers must
enforce their idempotency keys. This keeps a model or reasoning retry from
becoming a financial side effect.

## Deployment bindings

`EventTopologySettings.from_env()` chooses the topology from
`INFRASTRUCTURE_MODE`, with explicit per-layer overrides:

| Layer | Cloud default | On-prem default |
| --- | --- | --- |
| Risk-alert stream | Azure Event Hubs | Apache Kafka |
| Transactional event delivery | Azure Service Bus | RabbitMQ |
| Reactive routing | Azure Event Grid | NATS |

AWS SQS/EventBridge, IBM MQ, and Knative are also represented by thin publisher
bindings in `src/events/adapters.py`. SDK clients are injected at the
composition root, so credentials, TLS, retry policy, partition keys, consumer
groups, and checkpoint stores remain deployment concerns.

Production consumers must use manual acknowledgement/checkpointing after the
side effect, a durable idempotency ledger keyed by event id, bounded retries,
and a dead-letter path. The in-memory bus is only for tests and local demos.

## Human-in-the-loop brake

The forensic graph now requires a checkpointer and pauses in the
`human_approval` node with LangGraph's dynamic `interrupt()` **after** it has
published the analyst work-queue event containing the structured case report.
The interrupt payload is JSON-safe and contains the approval id, report and
evidence references, risk score, finding codes, and proposed idempotency key;
it does not contain unrestricted checkpoint
contents. The external control plane calls
`HumanApprovalService.resume(...)` with the same public `thread_id`. The
service derives a tenant/case-scoped checkpoint key, verifies the approver role
and approval id, and resumes using `Command(resume=...)`. Customer identity and
reviewer identity are deliberately not checkpoint-key inputs: a reviewer must
be able to decide a case about another customer.

The graph publishes `forensic.execution.order_requested.v1` only after an
approved decision. A separate executor remains responsible for applying the
financial side effect and must enforce the same idempotency key. Rejection
ends the graph without publishing an execution command.

For async APIs and workers use `open_async_checkpointer()` with
`AsyncPostgresSaver` on PostgreSQL, `AsyncRedisSaver` on Azure Managed Redis /
Redis Stack, or the official `CosmosDBSaver` integration. PostgreSQL and
Cosmos checkpoints use the configured encrypted serializer; Cosmos can use
Microsoft Entra managed identity when its key is blank. Cosmos database and
container provisioning belongs in IaC and uses the `/partition_key` container
partition key. Redis checkpoint indices must be initialized before serving;
PostgreSQL migrations remain a deployment step. Azure Redis should use Managed
Redis where applicable, TLS and Private Link; on-premise Redis must provide the
RedisJSON/RediSearch capabilities required by the current Redis checkpointer.
The synchronous helper intentionally rejects the Cosmos backend so callers do
not silently lose application-level checkpoint encryption.

`src/hitl/api.py` provides an optional FastAPI router. The same service can be
called from an Azure Container Apps worker or Celery task, keeping HTTP,
queueing and graph execution separate. LangSmith Studio can inspect threads and
interrupts when the graph is deployed through the LangGraph/Agent Server API;
on-premise tracing is available through the optional Langfuse callback without
placing trace payloads in the checkpoint state.
