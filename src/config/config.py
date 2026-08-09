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

# Inputs per request. This governs feasibility, not merely throughput: a single
# request whose token count exceeds the per-minute quota can never succeed, no
# matter how long the client backs off. Keep batch_size * avg tokens per chunk
# comfortably below the deployment's TPM.
EMBEDDINGS_BATCH_SIZE = _env_int("EMBEDDINGS_BATCH_SIZE", 16)

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

# ── Retrieval ─────────────────────────────────────────────────
# Postgres text-search configuration for the BM25 arm. It governs stemming and
# stop words, so it must match the corpus language: 'english' will not stem
# Spanish, silently degrading keyword recall on a Spanish corpus.
FTS_LANGUAGE = os.getenv("FTS_LANGUAGE", "pg_catalog.english")

# Documents handed to the caller.
RETRIEVER_K = _env_int("RETRIEVER_K", 5)

# Candidates pulled from the index before reranking. Generous on purpose: the
# reranker can only reorder what recall retrieved, so a relevant chunk missing
# here is lost permanently.
RETRIEVER_FETCH_K = _env_int("RETRIEVER_FETCH_K", 50)

# Cross-encoder that rescores (query, document) pairs. bge-reranker-v2-m3 is the
# companion model to bge-m3 and is multilingual. It runs locally in both
# infrastructure modes -- it is small, and keeping it off the metered endpoint
# means reranking costs no API quota.
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() == "true"

# ── Generation ────────────────────────────────────────────────
# Cloud: an Azure AI Foundry chat deployment. The deployment name is a routing
# label chosen when deploying and need not match the base model name.
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-5.4-mini")
AZURE_CHAT_ENDPOINT = os.getenv("AZURE_CHAT_ENDPOINT") or os.getenv(
    "AZURE_FOUNDRY_ENDPOINT"
)
AZURE_APIM_GATEWAY_URL = os.getenv("AZURE_APIM_GATEWAY_URL")
AZURE_APIM_SUBSCRIPTION_KEY = os.getenv("AZURE_APIM_SUBSCRIPTION_KEY")

# On-premise: LM Studio during local development (default port 1234); vLLM or
# NVIDIA NIM in production. All expose OpenAI Chat Completions, so switching
# only changes these variables and the LiteLLM upstream configuration.
ONPREM_CHAT_BASE_URL = os.getenv(
    "ONPREM_CHAT_BASE_URL", "http://127.0.0.1:1234/v1"
)
ONPREM_CHAT_MODEL = os.getenv("ONPREM_CHAT_MODEL", "ministral-3-3b-instruct-2512")
# Self-hosted servers ignore the key, but the OpenAI client requires one.
ONPREM_CHAT_API_KEY = os.getenv("ONPREM_CHAT_API_KEY", "not-needed")
ONPREM_LITELLM_BASE_URL = os.getenv("ONPREM_LITELLM_BASE_URL")
ONPREM_LITELLM_API_KEY = os.getenv("ONPREM_LITELLM_API_KEY")
ONPREM_LITELLM_MODEL = os.getenv("ONPREM_LITELLM_MODEL", "onprem-primary")

# Low by default: the model's job is to restate retrieved text faithfully, not
# to be creative. Sampling variance here shows up as invented detail.
GENERATION_TEMPERATURE = _env_float("GENERATION_TEMPERATURE", 0.0)
GENERATION_MAX_TOKENS = _env_int("GENERATION_MAX_TOKENS", 1024)
GENERATION_TIMEOUT = _env_float("GENERATION_TIMEOUT", 120.0)

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
    """Returns the hybrid indexer singleton (dense vectors + full-text search)."""
    global _indexer
    if _indexer is None:
        from src.rag.indexers import HybridIndexer

        _indexer = HybridIndexer(
            embeddings=get_embeddings(),
            connection_string=PG_CONNECTION,
            table_name=TABLE_NAME,
            vector_size=VECTOR_SIZE,
            tsv_language=FTS_LANGUAGE,
            candidate_pool=RETRIEVER_FETCH_K,
        )

    return _indexer


# ── Reranker ──────────────────────────────────────────────────
_reranker = None


def get_reranker():
    """Returns the cross-encoder singleton, or None when reranking is disabled.

    Loaded lazily and kept process-wide: the model weights are hundreds of
    megabytes and re-loading them per query would dominate latency.
    """
    global _reranker
    if not RERANKER_ENABLED:
        return None

    if _reranker is None:
        import torch
        from sentence_transformers import CrossEncoder

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _reranker = CrossEncoder(RERANKER_MODEL, device=device)
        logger.info(f"Reranker ready: {RERANKER_MODEL} on {device}")

    return _reranker


# ── Chat model ────────────────────────────────────────────────
_chat_model = None


def get_chat_model():
    """Returns the chat model singleton for the active infrastructure mode.

    Both branches target the same protocol -- OpenAI Chat Completions -- so the
    generator is identical in either deployment. Moving from LM Studio to vLLM
    or NVIDIA NIM on-premise changes ``ONPREM_CHAT_BASE_URL`` and
    ``ONPREM_CHAT_MODEL``, and nothing else.
    """
    global _chat_model
    if _chat_model is None:
        if INFRASTRUCTURE_MODE == "cloud":
            if bool(AZURE_APIM_GATEWAY_URL) != bool(AZURE_APIM_SUBSCRIPTION_KEY):
                raise ValueError(
                    "[config] Set both AZURE_APIM_GATEWAY_URL and "
                    "AZURE_APIM_SUBSCRIPTION_KEY to use Azure API Management."
                )

            if AZURE_APIM_GATEWAY_URL:
                from langchain_openai import ChatOpenAI

                _chat_model = ChatOpenAI(
                    model=AZURE_CHAT_DEPLOYMENT,
                    base_url=AZURE_APIM_GATEWAY_URL,
                    api_key=AZURE_APIM_SUBSCRIPTION_KEY,
                    default_headers={
                        "Ocp-Apim-Subscription-Key": AZURE_APIM_SUBSCRIPTION_KEY
                    },
                    temperature=GENERATION_TEMPERATURE,
                    max_tokens=GENERATION_MAX_TOKENS,
                    timeout=GENERATION_TIMEOUT,
                )
                logger.info("Chat model ready through Azure API Management.")
                return _chat_model

            from azure.core.credentials import AzureKeyCredential
            from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel

            api_key = os.getenv("AZURE_FOUNDRY_API_KEY")
            if not api_key:
                raise ValueError(
                    "[config] AZURE_FOUNDRY_API_KEY is required in cloud mode."
                )

            _chat_model = AzureAIOpenAIApiChatModel(
                endpoint=AZURE_CHAT_ENDPOINT,
                credential=AzureKeyCredential(api_key),
                model=AZURE_CHAT_DEPLOYMENT,
                temperature=GENERATION_TEMPERATURE,
                max_tokens=GENERATION_MAX_TOKENS,
            )
            logger.info(f"Chat model ready: Azure deployment={AZURE_CHAT_DEPLOYMENT}")
        else:
            from langchain_openai import ChatOpenAI

            if bool(ONPREM_LITELLM_BASE_URL) != bool(ONPREM_LITELLM_API_KEY):
                raise ValueError(
                    "[config] Set both ONPREM_LITELLM_BASE_URL and "
                    "ONPREM_LITELLM_API_KEY to use LiteLLM."
                )

            if ONPREM_LITELLM_BASE_URL:
                _chat_model = ChatOpenAI(
                    model=ONPREM_LITELLM_MODEL,
                    base_url=ONPREM_LITELLM_BASE_URL,
                    api_key=ONPREM_LITELLM_API_KEY,
                    temperature=GENERATION_TEMPERATURE,
                    max_tokens=GENERATION_MAX_TOKENS,
                    timeout=GENERATION_TIMEOUT,
                )
                logger.info("Chat model ready through the on-premise LiteLLM gateway.")
                return _chat_model

            _chat_model = ChatOpenAI(
                model=ONPREM_CHAT_MODEL,
                base_url=ONPREM_CHAT_BASE_URL,
                api_key=ONPREM_CHAT_API_KEY,
                temperature=GENERATION_TEMPERATURE,
                max_tokens=GENERATION_MAX_TOKENS,
                timeout=GENERATION_TIMEOUT,
            )
            logger.info(
                f"Chat model ready: {ONPREM_CHAT_MODEL} at {ONPREM_CHAT_BASE_URL}"
            )

    return _chat_model


# ── Generator ─────────────────────────────────────────────────
def get_generator(**kwargs):
    """Builds the grounded answer generator.

    Args:
        **kwargs: Passed through to :class:`~src.rag.generator.Generator`
            (``system_prompt``).
    """
    from src.rag.generator import Generator

    return Generator(llm=get_chat_model(), **kwargs)


# ── Retriever ─────────────────────────────────────────────────
def get_retriever(k: int | None = None, fetch_k: int | None = None, **kwargs):
    """Builds the retrieval pipeline: hybrid recall, then cross-encoder rerank.

    Args:
        k: Documents returned. Defaults to ``RETRIEVER_K``.
        fetch_k: Candidates retrieved before reranking. Defaults to
            ``RETRIEVER_FETCH_K``.
        **kwargs: Passed through to :class:`~src.rag.retriever.HybridRetriever`
            (``score_threshold``, ``filter``).
    """
    from src.rag.retriever import HybridRetriever

    return HybridRetriever(
        indexer=get_indexer(),
        reranker=get_reranker(),
        k=k if k is not None else RETRIEVER_K,
        fetch_k=fetch_k if fetch_k is not None else RETRIEVER_FETCH_K,
        **kwargs,
    )
