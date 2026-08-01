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
PG_CONNECTION = os.getenv(
    "PG_CONNECTION",
    "postgresql+psycopg://mi_usuario:mi_password@localhost:5432/langgraph_db",
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
        # Conector nativo directo para el modelo BGE-M3 sin dependencias de OpenAI
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
    """Returns the HybridIndexer singleton instance."""
    global _hybrid_indexer
    if _hybrid_indexer is None:
        from src.rag.indexers import HybridIndexer

        _hybrid_indexer = HybridIndexer(
            embeddings=EMBEDDINGS,
            connection_string=PG_CONNECTION,
        )
    return _hybrid_indexer
