"""Global Configuration: Environment variables, checkpointer, and dynamic Microsoft Foundry / On-Premise RAG singletons."""

import os
from dotenv import load_dotenv
from pydantic import SecretStr
from langgraph.checkpoint.redis import RedisSaver

# Load target environment architecture flags (Mid-2026 specs)
load_dotenv()

# Read the core deployment strategy: "cloud" or "on_premise"
INFRASTRUCTURE_MODE = os.getenv("INFRASTRUCTURE_MODE", "on_premise").lower()

# Externalize infrastructure connection strings
REDIS_URI = os.getenv("REDIS_URI", "redis://localhost:6379")

# Select PG connection based on infrastructure mode
if INFRASTRUCTURE_MODE == "cloud":
    PG_CONNECTION = os.getenv(
        "PG_CONNECTION_CLOUD",
        "postgresql+psycopg://mi_usuario:mi_password@azure-postgres.postgres.database.azure.com:5432/langgraph_db",
    )
else:
    PG_CONNECTION = os.getenv(
        "PG_CONNECTION_ONPREM",
        "postgresql+psycopg://onyx_user:onyx_password@localhost:5432/onyx_db",
    )


# ── Dynamic Embeddings Factory (Dynamic Injection Pattern) ────
def _init_embeddings():
    """Dynamically initializes embeddings based on infrastructure mode."""
    if INFRASTRUCTURE_MODE == "cloud":
        from langchain_azure_ai.embeddings import AzureAIOpenAIApiEmbeddingsModel

        api_key_env = os.getenv("AZURE_FOUNDRY_API_KEY")
        api_key_secret = SecretStr(api_key_env) if api_key_env else None

        return AzureAIOpenAIApiEmbeddingsModel(
            endpoint=os.getenv("AZURE_FOUNDRY_ENDPOINT"),
            api_key=api_key_secret,
            model="text-embedding-3-large",
        )
    elif INFRASTRUCTURE_MODE == "on_premise":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )
    else:
        raise ValueError(
            f"[config] Invalid INFRASTRUCTURE_MODE token: {INFRASTRUCTURE_MODE}"
        )


# Initialize the shared embedding instance dynamically at application startup
EMBEDDINGS = _init_embeddings()

# Shared indexer instance placeholder for lazy initialization tracking
_hybrid_indexer = None


def get_indexer():
    """Returns the native Indexing API orchestrator mapping directly to the main framework."""
    global _hybrid_indexer
    if _hybrid_indexer is None:
        from langchain_core.indexing import index
        from langchain_classic.indexes import SQLRecordManager
        from langchain_postgres.v2.vectorstores import PGVectorStore
        from langchain_postgres.v2.engine import PGEngine

        TABLE_NAME = "onyx_corporate_knowledge"

        # Vector dimensions: 1024 for bge-m3 (on_premise), 3072 for text-embedding-3-large (cloud)
        vector_size = 3072 if INFRASTRUCTURE_MODE == "cloud" else 1024

        # PGEngine uses SQLAlchemy async — requires psycopg_async or asyncpg driver.
        # Convert the sync psycopg driver if present in the connection string.
        pg_async_connection = PG_CONNECTION.replace(
            "postgresql+psycopg://", "postgresql+psycopg_async://"
        )
        engine = PGEngine.from_connection_string(pg_async_connection)
        try:
            engine.init_vectorstore_table(
                table_name=TABLE_NAME,
                vector_size=vector_size,
            )
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise

        vector_store = PGVectorStore.create_sync(
            engine=engine,
            table_name=TABLE_NAME,
            embedding_service=EMBEDDINGS,
        )

        # Initialize the persistent tracking record manager
        record_manager = SQLRecordManager(
            namespace="postgres/onyx_corporate_knowledge",
            db_url=PG_CONNECTION,
        )
        record_manager.create_schema()

        # Define native workflow using partial application for clean pipeline ingestion
        class NativeIndexer:
            def add_documents(self, documents):
                return index(
                    documents,
                    record_manager=record_manager,
                    vector_store=vector_store,
                    cleanup="incremental",
                    source_id_key="source",
                )

        _hybrid_indexer = NativeIndexer()

    return _hybrid_indexer
