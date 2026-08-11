# Governed hybrid agent state and memory

This implementation separates data by lifecycle and control boundary. It does
not treat LangGraph state, a cache and long-term memory as interchangeable.

| Plane | Purpose | Authoritative system | Retention |
|---|---|---|---|
| Short-term execution state | Resume one bounded workflow/thread | Encrypted LangGraph PostgreSQL checkpointer | Policy-defined; delete by scoped thread ID |
| Ephemeral RAG context | Carry retrieved document bodies between nodes | Redis TTL cache (non-authoritative) | 30–3600 seconds and explicit end-of-run deletion |
| Governed long-term memory | Approved facts/preferences across threads | LangGraph PostgreSQL `BaseStore` + pgvector | Per-record expiry, erasure and append-only versions |
| Governance evidence | Decisions and control evidence | Existing evidence ledger + Azure Monitor/SIEM | Governance policy; never mixed with user memory |

Redis is not written through to PostgreSQL. The systems contain different data:
Redis holds transient retrieved chunks; the checkpointer holds bounded execution
state; the long-term store accepts only an explicit, approved `MemoryWrite`.

## Hybrid deployment mapping

The application contracts stay identical in both modes.

| Capability | Azure cloud | On-premises |
|---|---|---|
| Checkpoints | Azure Database for PostgreSQL, private endpoint and TLS | PostgreSQL HA cluster, TLS and dedicated role |
| Ephemeral context | Azure Managed Redis, private endpoint and TLS | Redis Enterprise or Redis HA, ACLs and TLS |
| Long-term memory | Azure Database for PostgreSQL with pgvector | PostgreSQL with pgvector |
| Keys | Azure Key Vault workload identity | HashiCorp Vault federated workload identity |

Use separate databases and least-privilege roles for checkpoints and long-term
memory. Apply platform encryption at rest, private networking, backups and
regional placement according to the data classification and residency policy.

## Enforced controls

- The trusted API must supply `tenant_id`, `subject_id`, `thread_id`, roles,
  request correlation and data classification. Caller thread IDs are hashed
  with tenant and subject scope before reaching the checkpointer.
- Checkpoint payloads use LangGraph's AES encrypted serializer and cap message
  history. Retrieved document bodies never enter `AgentState`.
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
  decisions, not prompt/document bodies.

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

The composition root must place `context_cache` in `RAGDependencies`; the
example above omits construction of the other existing security dependencies.

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
