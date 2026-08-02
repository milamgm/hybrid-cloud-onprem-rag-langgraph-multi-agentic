"""Global configuration: environment flags, embeddings, and index singletons.

The deployment target is selected by ``INFRASTRUCTURE_MODE``:

* ``cloud`` -- Azure AI Foundry embeddings against Azure PostgreSQL. Embedding
  calls are metered, so they are paced and retried (see :mod:`src.rag.embeddings`).
* ``on_premise`` -- BAAI/bge-m3 on local hardware against a Docker Postgres. No
  pacing: the only limit is the GPU.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from pydantic import SecretStr

load_dotenv()

logger = logging.getLogger("pipeline.config")

INFRASTRUCTURE_MODE = os.getenv("INFRASTRUCTURE_MODE", "on_premise").lower()
REDIS_URI = os.getenv("REDIS_URI", "redis://localhost:6379")

TABLE_NAME = os.getenv("RAG_TABLE_NAME", "onyx_corporate_knowledge")

if INFRASTRUCTURE_MODE == "cloud":
    PG_CONNECTION = os.getenv("PG_CONNECTION_CLOUD")
else:
    PG_CONNECTION = os.getenv("PG_CONNECTION_ONPREM")

if not PG_CONNECTION:
    raise ValueError(
        f"[config] No PostgreSQL connection configured for "
        f"INFRASTRUCTURE_MODE={INFRASTRUCTURE_MODE!r}. Set "
        f"{'PG_CONNECTION_CLOUD' if INFRASTRUCTURE_MODE == 'cloud' else 'PG_CONNECTION_ONPREM'} "
        f"in the environment."
    )


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


# ── Embedding quota tuning ────────────────────────────────────
# Derive EMBEDDINGS_REQUESTS_PER_SECOND from the deployment quota: a deployment
# rated at N requests/minute supports N/60, and leaving ~20% headroom keeps room
# for retrieval traffic running alongside an ingest. Raise the deployment quota
# at https://aka.ms/oai/quotaincrease rather than pushing this value past it.
EMBEDDINGS_REQUESTS_PER_SECOND = _env_float("EMBEDDINGS_REQUESTS_PER_SECOND", 1.0)

# Inputs per request. Bounds the tokens in flight so one batch cannot exhaust
# the tokens-per-minute window by itself.
EMBEDDINGS_BATCH_SIZE = _env_int("EMBEDDINGS_BATCH_SIZE", 128)

EMBEDDINGS_MAX_RETRIES = _env_int("EMBEDDINGS_MAX_RETRIES", 6)
EMBEDDINGS_TIMEOUT = _env_float("EMBEDDINGS_TIMEOUT", 120.0)

# text-embedding-3-large is natively 3072-dim and supports Matryoshka
# truncation. Lowering this shrinks storage and speeds up index scans at a small
# recall cost; it must match the vector column width, so changing it on a
# populated table requires a re-index.
CLOUD_EMBEDDING_DIMENSIONS = _env_int("EMBEDDINGS_DIMENSIONS", 3072)
ONPREM_EMBEDDING_DIMENSIONS = 1024  # Fixed by BAAI/bge-m3.

# Base model names, used to resolve the matching tokenizer. Distinct from the
# Azure deployment name, which is a routing label and carries no tokenizer.
CLOUD_EMBEDDING_MODEL = "text-embedding-3-large"
ONPREM_EMBEDDING_MODEL = "BAAI/bge-m3"

# Token budget per chunk. Both embedding models accept far more (8191 / 8192),
# but retrieval quality peaks well below the ceiling: a chunk holding one idea
# ranks better than a page holding six, because its vector is not an average of
# unrelated content. 400-512 is the current consensus starting point.
MAX_CHUNK_TOKENS = _env_int("MAX_CHUNK_TOKENS", 512)

VECTOR_SIZE = (
    CLOUD_EMBEDDING_DIMENSIONS
    if INFRASTRUCTURE_MODE == "cloud"
    else ONPREM_EMBEDDING_DIMENSIONS
)


# ── Embeddings factory ────────────────────────────────────────
def _build_cloud_embeddings() -> Embeddings:
    """Azure AI Foundry embeddings, paced to stay inside the deployment quota."""
    from langchain_azure_ai.embeddings import AzureAIOpenAIApiEmbeddingsModel

    from src.rag.embeddings import RateLimitedEmbeddings

    api_key = os.getenv("AZURE_FOUNDRY_API_KEY")
    if not api_key:
        raise ValueError("[config] AZURE_FOUNDRY_API_KEY is required in cloud mode.")

    # Azure routes on the deployment name, which need not match the base model.
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "text-embedding-3-large")

    client = AzureAIOpenAIApiEmbeddingsModel(
        endpoint=os.getenv("AZURE_FOUNDRY_ENDPOINT"),
        api_key=SecretStr(api_key),
        model=deployment,
        # Keep the provider's own batching aligned with the limiter's, so one
        # embed_documents call maps to one paced request.
        chunk_size=EMBEDDINGS_BATCH_SIZE,
        # The SDK honours the server's Retry-After between these attempts; the
        # limiter below is the outer backstop.
        max_retries=EMBEDDINGS_MAX_RETRIES,
        request_timeout=EMBEDDINGS_TIMEOUT,
        dimensions=(
            CLOUD_EMBEDDING_DIMENSIONS if CLOUD_EMBEDDING_DIMENSIONS != 3072 else None
        ),
    )

    logger.info(
        f"Azure embeddings ready: deployment={deployment} "
        f"dims={CLOUD_EMBEDDING_DIMENSIONS} "
        f"rate={EMBEDDINGS_REQUESTS_PER_SECOND}/s batch={EMBEDDINGS_BATCH_SIZE}"
    )

    return RateLimitedEmbeddings(
        client,
        requests_per_second=EMBEDDINGS_REQUESTS_PER_SECOND,
        batch_size=EMBEDDINGS_BATCH_SIZE,
        max_retries=EMBEDDINGS_MAX_RETRIES,
    )


def _build_onprem_embeddings() -> Embeddings:
    """Local bge-m3 embeddings. Falls back to CPU when no GPU is present."""
    import torch
    from langchain_huggingface import HuggingFaceEmbeddings

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("No CUDA device detected; embedding on CPU will be slow.")

    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


_embeddings: Embeddings | None = None


def get_embeddings() -> Embeddings:
    """Returns the process-wide embeddings singleton, building it on first use.

    Construction is deferred so that importing this module does not load a
    multi-gigabyte local model or open a network client.
    """
    global _embeddings
    if _embeddings is None:
        if INFRASTRUCTURE_MODE == "cloud":
            _embeddings = _build_cloud_embeddings()
        elif INFRASTRUCTURE_MODE == "on_premise":
            _embeddings = _build_onprem_embeddings()
        else:
            raise ValueError(
                f"[config] Invalid INFRASTRUCTURE_MODE: {INFRASTRUCTURE_MODE!r}. "
                f"Expected 'cloud' or 'on_premise'."
            )
    return _embeddings


# ── Tokenizer ─────────────────────────────────────────────────
_tokenizer = None


def get_tokenizer():
    """Returns the tokenizer of the active embedding model.

    The chunker measures its token budget with this. It must be the embedding
    model's own tokenizer: a budget counted with a different tokenizer is a
    guess, and chunks that overflow are silently truncated by the provider,
    losing their tail with no error raised.
    """
    global _tokenizer
    if _tokenizer is None:
        if INFRASTRUCTURE_MODE == "cloud":
            import tiktoken
            from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

            try:
                encoding = tiktoken.encoding_for_model(CLOUD_EMBEDDING_MODEL)
            except KeyError:
                # Unknown model names fall back to the encoding every current
                # OpenAI embedding model uses.
                logger.warning(
                    f"No tiktoken encoding registered for {CLOUD_EMBEDDING_MODEL}; "
                    f"falling back to cl100k_base."
                )
                encoding = tiktoken.get_encoding("cl100k_base")

            _tokenizer = OpenAITokenizer(
                tokenizer=encoding, max_tokens=MAX_CHUNK_TOKENS
            )
        else:
            from docling_core.transforms.chunker.tokenizer.huggingface import (
                HuggingFaceTokenizer,
            )

            _tokenizer = HuggingFaceTokenizer.from_pretrained(
                model_name=ONPREM_EMBEDDING_MODEL,
                max_tokens=MAX_CHUNK_TOKENS,
            )

        logger.info(
            f"Tokenizer ready: {type(_tokenizer).__name__} "
            f"max_tokens={MAX_CHUNK_TOKENS}"
        )

    return _tokenizer


# ── Index singleton ───────────────────────────────────────────
_indexer = None


def get_indexer():
    """Returns the tracked indexer singleton backed by pgvector.

    Wraps LangChain's :func:`~langchain_core.indexing.index` API with a
    :class:`SQLRecordManager`, so re-ingesting an unchanged document skips the
    embedding calls entirely instead of paying for them again.
    """
    global _indexer
    if _indexer is None:
        from langchain_classic.indexes import SQLRecordManager
        from langchain_core.indexing import index
        from langchain_postgres.v2.engine import PGEngine
        from langchain_postgres.v2.vectorstores import PGVectorStore

        embeddings = get_embeddings()

        # PGEngine runs on SQLAlchemy's async stack and needs an async driver.
        pg_async_connection = PG_CONNECTION.replace(
            "postgresql+psycopg://", "postgresql+psycopg_async://"
        )
        engine = PGEngine.from_connection_string(pg_async_connection)

        # init_vectorstore_table has no IF NOT EXISTS clause, so a pre-existing
        # table surfaces as a constraint error we can safely absorb.
        try:
            engine.init_vectorstore_table(
                table_name=TABLE_NAME,
                vector_size=VECTOR_SIZE,
            )
        except Exception as error:
            if "already exists" not in str(error).lower():
                raise
            logger.info(f"Reusing existing vector table '{TABLE_NAME}'.")

        vector_store = PGVectorStore.create_sync(
            engine=engine,
            table_name=TABLE_NAME,
            embedding_service=embeddings,
        )

        record_manager = SQLRecordManager(
            namespace=f"postgres/{TABLE_NAME}",
            db_url=PG_CONNECTION,
        )
        record_manager.create_schema()

        class TrackedIndexer:
            """Adapter exposing the indexing API the ingest pipeline expects."""

            def add_documents(self, documents) -> dict:
                return index(
                    documents,
                    record_manager=record_manager,
                    vector_store=vector_store,
                    # Scoped per source_id present in the batch: stale chunks of
                    # an updated document are removed, other sources untouched.
                    cleanup="incremental",
                    source_id_key="source",
                )

            @property
            def store(self):
                return vector_store

        _indexer = TrackedIndexer()

    return _indexer
