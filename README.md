Deployment Workflows

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
Question -> LangGraph -> retrieval + reranking -> Azure Foundry chat -> response
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
Question -> LangGraph -> retrieval + reranking -> LiteLLM -> LM Studio/vLLM/NIM -> response
              |
              +-> PostgreSQL: checkpoints
              +-> Redis: ephemeral context
              +-> PostgreSQL: approved long-term memory
              +-> Kafka/RabbitMQ/NATS -> SIEM/WORM: events and audit