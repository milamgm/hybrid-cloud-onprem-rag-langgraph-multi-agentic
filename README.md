# Onyx

## Ejecutar la aplicación

Onyx incluye una interfaz Streamlit que compone el grafo RAG completo. Los dos
perfiles usan tablas distintas porque sus embeddings tienen dimensiones
incompatibles (Azure 3072, BGE-M3 1024).

### On-premise de desarrollo

```bash
cp .env.onprem.example .env.onprem
docker compose --env-file .env.onprem up -d
# Inicia el servidor local de LM Studio en 127.0.0.1:1234.
ONYX_ENV_FILE=.env.onprem uv run python main.py check
ONYX_ENV_FILE=.env.onprem uv run python -m scripts.ingest data/raw
ONYX_ENV_FILE=.env.onprem uv run python -m streamlit run main.py
```

La primera ingestión descarga BGE-M3 y puede tardar. El perfil inicial usa CPU,
PyMuPDF, no carga el reranker y conserva los checkpoints sólo durante el proceso.

### Cloud desde el equipo local

```bash
cp .env.cloud.example .env.cloud
# Completa PostgreSQL y Foundry; añade Content Safety/Language para managed.
ONYX_ENV_FILE=.env.cloud uv run python main.py check
ONYX_ENV_FILE=.env.cloud uv run python -m scripts.ingest data/raw
ONYX_ENV_FILE=.env.cloud uv run python -m streamlit run main.py
```

El perfil cloud de desarrollo consume Azure AI Foundry y Azure PostgreSQL, pero
mantiene el contexto y checkpoints en el proceso local. Para producción deben
configurarse Redis/Cosmos, identidades administradas, auditoría Event Hubs y los
controles descritos más abajo.

`RAG_SECURITY_PROFILE=development` sólo está permitido fuera de producción y
no debe utilizarse con datos reales. `managed` activa Azure Content Safety y
Azure AI Language en cloud o los NIM de NeMo on-premise. Si la región del
recurso Foundry no ofrece PII, configura `AZURE_LANGUAGE_ENDPOINT` y
`AZURE_LANGUAGE_KEY` con un recurso Language compatible.

## Deployment Workflows

Agentic Fraud Investigation Workflow

An XGBoost transaction-risk alert starts one investigation. The investigation team does
not receive database credentials, arbitrary SQL, write tools, or web search.
Each specialist gets only the following case-scoped, read-only tools:

| Agent | Tools | Responsibility |
| --- | --- | --- |
| Transaction analyst | `read_transaction_case_context` | Assesses the alerted transaction using core-banking evidence. |
| Customer-risk analyst | `read_customer_risk_profile`, `read_subject_screening` | Reviews KYC/CDD risk and bounded sanctions/adverse-media screening. |
| Network analyst | `read_transaction_network` | Assesses counterparties, linkage and related-activity indicators. |
| Policy-compliance analyst | `search_internal_policy` | Maps the evidence to the curated internal AML/fraud procedures. |
| Case lead | `read_specialist_case_dossier` | Reconciles specialist findings into a cited investigation report. |

Every tool is closed over the alert's tenant, customer and transaction IDs, so
the model cannot expand the data scope. The tool adapters call repository-owned
parameterised read operations; no agent has a database connection or a raw-SQL
tool. The report remains a recommendation: a human approval is mandatory before
an execution-order event can be published.

Cloud forensic flow:

```text
XGBoost -> Event Hubs -> Azure Function starter -> Durable Functions orchestration
  -> collect evidence activity -> four specialist agents (parallel) -> case-lead agent
  -> review event -> wait for human approval external event -> execution order (approved only)
```

Azure Durable Functions owns the long-running cloud case lifecycle, retries and
the wait for human approval. Its orchestrator is deterministic; database reads,
LLM/tool calls and event publication run in idempotent activity functions. The
Azure Functions entrypoint is `function_app.py`; it must be configured at
startup with real least-privilege gateways through
`FORENSIC_DURABLE_ACTIVITIES_FACTORY=package.module:factory` (or directly with
`configure_durable_forensic_activities`), and deliberately fails closed if that
composition is absent.

On-premises forensic flow:

```text
XGBoost -> Kafka -> LangGraph investigation -> evidence worker -> specialist agents
  -> case-lead report -> HITL approval -> RabbitMQ execution command
```

In both environments, production composition injects `ForensicInvestigationTeam`
instead of `MockForensicReasoner`. The mock remains only for development and
unit tests.

Application Workflow Node by Node

The application runs two LangGraph workflows. The RAG graph answers interactive
questions; the forensic graph investigates XGBoost fraud alerts. Both are
deterministic: node order is fixed in code and the model never chooses the next
step. The graph topology is identical on Azure and on-premises; only the
infrastructure behind each node changes.

RAG graph (`src/graph/rag_graph.py`, nodes in `src/nodes/rag.py`)

```text
START -> manage_context -> guard_input -> retrieve -> guard_retrieval
      -> generate -> validate_output -> record_evidence -> cleanup_context -> END
```

`guard_input`, `retrieve` and `guard_retrieval` short-circuit straight to
`record_evidence` when a control blocks the run, so a blocked request is still
audited and its ephemeral context is still deleted.

| Node | What it does | Cloud (Azure) | On-premises |
| --- | --- | --- | --- |
| `manage_context` | Summarizes old turns and trims the active buffer past the token/message threshold. | Azure AI Foundry chat model (summarizer) | LM Studio / vLLM / NVIDIA NIM via LiteLLM |
| `guard_input` | Audits the received prompt and enforces direct prompt-injection rules. | `PromptInjectionMiddleware`, `AuditService` -> Event Hubs / Blob WORM | Same middleware; audit -> OTel Collector -> SIEM/WORM |
| `retrieve` | Vector + text retrieval, then stores documents in the ephemeral context cache under a handle and builds citation markers. | Azure Database for PostgreSQL + pgvector; Azure Managed Redis (TTL) for the handle | PostgreSQL + pgvector; Redis with TTL, ACLs and TLS |
| `guard_retrieval` | Re-runs injection enforcement over the retrieved (untrusted) document text. | `PromptInjectionMiddleware`, Redis-backed `ContextCache` | Same middleware, local Redis |
| `generate` | Generates a cited answer from the cached documents, summary, history and approved long-term memories. | Azure AI Foundry chat model, direct or via APIM | LiteLLM Proxy -> LM Studio (dev) / vLLM / NIM (prod) |
| `validate_output` | Validates the answer against output policy and requires valid citation markers. | `OutputValidationMiddleware`, Azure AI Content Safety | `OutputValidationMiddleware`, NeMo Guardrails / NIM |
| `record_evidence` | Records the allow/block policy decision plus an audit event. | `GovernanceService`, `AuditService` -> Event Hubs -> Blob WORM | OTel Collector -> SIEM / WORM storage |
| `cleanup_context` | Deletes the retrieval handle so retrieved context is never permanent. | Azure Managed Redis | Redis |

Long-term memory approved for sharing across threads is read into state before
the run: Azure Cosmos DB for NoSQL in the cloud, PostgreSQL with pgvector
on-premises. Checkpoints live in Redis (cloud) or PostgreSQL with TLS (on-prem).

Forensic graph (`src/graph/forensic_graph.py`)

The graph is entered once per stage; the entry router reads `stage` and jumps to
the right node, so each event delivery advances the case exactly one step.

```text
START -(alert_received)-> request_evidence -> END
START -(evidence_ready)-> prepare_case_for_review -> human_approval
                                                  -> request_execution_order -> END
START -(approved)------> request_execution_order -> END
```

| Node | What it does | Cloud (Azure) | On-premises |
| --- | --- | --- | --- |
| `request_evidence` | Publishes `EvidenceCollectionRequested` for the alert; no reads happen inside the graph. | Event Hubs publisher | Kafka publisher |
| (evidence worker) | `EvidenceRequestedConsumer` calls `CaseEvidenceCollector`, which fans out to parameterised read-only gateways and publishes `EvidenceReady`. | Named queries over Azure PostgreSQL, policy RAG over pgvector | Same gateways over local PostgreSQL/pgvector |
| `prepare_case_for_review` | Runs `ForensicInvestigationTeam`, builds the cited `InvestigationReport`, publishes `ApprovalRequested`. | Azure AI Foundry chat model; Event Hubs | LiteLLM -> vLLM / NIM; Kafka |
| `human_approval` | `interrupt()` pauses the case; on resume it validates the decision and optionally publishes `HumanApprovalGranted`. No side effect precedes the interrupt, so re-entry is safe. | HITL API behind APIM/Entra ID; Redis checkpointer | HITL API behind Keycloak/OPA; PostgreSQL checkpointer |
| `request_execution_order` | Only reachable after an approved decision; publishes the idempotent `ExecutionOrderRequested`. | Event Hubs | RabbitMQ (transactional delivery) |

Investigation agents and their tools (`src/agents/forensic.py`)

`prepare_case_for_review` delegates to the investigation team. The four
specialists run in parallel because their tools read disjoint sources; the case
lead runs afterwards and cannot reach any business system.

| Agent | Tools | Backing gateway |
| --- | --- | --- |
| `transaction_analyst` | `read_transaction_case_context` | `ParameterizedCoreBankingGateway` (named query) |
| `customer_risk_analyst` | `read_customer_risk_profile`, `read_subject_screening` | `ParameterizedCustomerRiskGateway`, `ParameterizedScreeningGateway` |
| `network_analyst` | `read_transaction_network` | `ParameterizedNetworkGateway` |
| `policy_compliance_analyst` | `search_internal_policy` | `PolicyRAGGateway` over the curated policy corpus in pgvector |
| `case_lead` | `read_specialist_case_dossier` | In-memory dossier of the four specialist assessments only |

Every tool is closed over the alert's tenant, customer and transaction IDs, so
the model cannot widen the data scope. There is no database connection, raw-SQL
tool, web search or write tool anywhere in the team.

Cloud durable variant

On Azure the same steps run as Durable Functions instead of a checkpointed
LangGraph run, because the wait for human approval can last days.
`function_app.py` holds the triggers and `src/cloud/durable_forensics.py` the
orchestrator and activities.

| Step | Function | Notes |
| --- | --- | --- |
| `start_fraud_case` | Event Hub trigger | Validates the XGBoost alert and starts one instance per `alert_id`. |
| `fraud_case_orchestrator` | Orchestration trigger | Deterministic generator only: no LLM, clock, random ID, database or HTTP work. |
| `collect_forensic_evidence` | Activity | `CaseEvidenceCollector` over the scoped read gateways. |
| `run_forensic_agents` | Activity | The five-agent team plus report assembly. |
| `publish_forensic_review` | Activity | Publishes `ApprovalRequested` to Event Hubs. |
| `submit_human_approval` | HTTP trigger | Raises the `human_approval` external event after APIM/Entra has authenticated and authorised the approver. |
| `publish_forensic_execution` | Activity | Publishes the idempotent execution order, approved cases only. |

Activities must be composed at startup through
`FORENSIC_DURABLE_ACTIVITIES_FACTORY=package.module:factory`; without it the
activities fail closed rather than falling back to mock data.

Cloud Workflow (Azure)

RAG ingestion: documents are loaded with Docling, split into chunks, and embeddings are generated using Azure AI Foundry text-embedding-3-large. The corpus and its vectors are persisted in Azure Database for PostgreSQL with pgvector; retrieval combines vector and text search.

Agent execution: LangGraph orchestrates the RAG graph. The chat model is consumed from Azure AI Foundry, either directly or through Azure API Management (APIM), which handles authentication and quotas.

Temporary state: execution/thread checkpoints are stored by default in Azure Managed Redis with a TTL. Retrieved RAG context is also stored in Redis with a TTL and deleted when the execution finishes; Redis is not used as permanent memory.

Governed long-term memory: approved memories shared across threads are stored in Azure Cosmos DB for NoSQL (see the dedicated section below).

Events and auditing: risk and audit events flow through Azure Event Hubs; evidence is archived in Azure Blob Storage using append/WORM policies. Observability data is exported via OpenTelemetry to Azure Monitor.

Security and privacy: secrets are managed through Azure Key Vault and managed identities; workload identity is handled with Microsoft Entra ID; PII detection uses Azure Language, while content filtering is handled by Azure AI Content Safety.

Summary flow:

Documents -> Docling/chunks -> Azure Foundry embeddings -> PostgreSQL/pgvector
                                                             |
Interactive RAG question -> LangGraph -> retrieval + reranking -> Azure Foundry chat -> response
              |
              +-> Redis: checkpoints and ephemeral context
              +-> Cosmos DB: approved long-term memory
              +-> Event Hubs -> Blob WORM: evidence/audit




On-Premises Workflow

RAG ingestion: documents are loaded with Docling, split into chunks, and embeddings are generated locally using BAAI/bge-m3 (CUDA GPU when available). The corpus and its vectors are stored in PostgreSQL with pgvector; in development, the database is deployed using Docker Compose.

Agent execution: LangGraph orchestrates the RAG graph. The chat model is served through LM Studio in development, or through vLLM/NVIDIA NIM in production; LiteLLM Proxy provides an OpenAI-compatible endpoint.

Temporary state: checkpoints are stored by default in PostgreSQL with TLS. Retrieved RAG context uses Redis with TTL, ACLs, and TLS; it is deleted after execution and is not considered authoritative persistence.
Governed long-term memory: approved memories shared across threads are stored in PostgreSQL with pgvector, separately from checkpoint state.

Events and auditing: the event pipeline uses Apache Kafka for streams, RabbitMQ for transactional delivery, and NATS for reactive routing. Evidence is sent through the OpenTelemetry Collector to the SIEM and/or WORM storage.

Security and privacy: HashiCorp Vault manages secrets using short-lived OIDC tokens issued by Keycloak (or Entra ID); OPA acts as the policy decision point for tool access; NeMo Guardrails/NVIDIA NIM applies content safety controls.

Summary flow:

Documents -> Docling/chunks -> BAAI/bge-m3 -> PostgreSQL/pgvector
                                               |
Interactive RAG question -> LangGraph -> retrieval + reranking -> LiteLLM -> LM Studio/vLLM/NIM -> response
              |
              +-> PostgreSQL: checkpoints
              +-> Redis: ephemeral context
              +-> PostgreSQL: approved long-term memory
              +-> Kafka/RabbitMQ/NATS -> SIEM/WORM: events and audit
