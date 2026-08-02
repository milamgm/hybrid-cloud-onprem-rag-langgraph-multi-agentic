from __future__ import annotations

import hashlib
import uuid

from langchain_classic.indexes import SQLRecordManager
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.indexing import index

from src.config.config import PG_CONNECTION, VECTOR_SIZE

# Fixed namespace for generating deterministic UUIDs (uuid5).
# This guarantees that re-ingesting the same chunk produces the same ID,
# enabling incremental upserts without duplicates.
_CHUNK_NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


def stable_chunk_id(source: str, content: str) -> str:
    """Generates a deterministic UUID based on document source and chunk content.

    Uses uuid5 (SHA-1 based) to ensure the exact same chunk always receives the
    same ID, enabling idempotent upserts in vector databases.
    """
    h = hashlib.sha256(f"{source}:{content}".encode()).hexdigest()[:32]
    return str(uuid.uuid5(_CHUNK_NAMESPACE, f"{source}:{h}"))


# ── Hybrid Indexer (pgvector + FTS with SQLRecordManager Tracking) ─────


class HybridIndexer:
    """Indexer combining pgvector + Postgres FTS with native RRF and tracking.

    A single Postgres table containing:
      - langchain_id        UUID PRIMARY KEY
      - content             TEXT
      - embedding           vector(1024)
      - langchain_metadata  JSONB
      - tsv                 tsvector (automatically generated via trigger)

    Hybrid search via HybridSearchConfig using reciprocal_rank_fusion.
    Tracks document state incrementally using SQLRecordManager.

    Usage:
        indexer = HybridIndexer(embeddings=get_embeddings())
        indexer.add_documents(chunks)              # Incremental smart upsert
        results = indexer.search(query, top_k=5)   # Hybrid RRF search

    Note:
        Not yet wired into the ingest pipeline, which uses the dense-only
        indexer from :func:`src.config.config.get_indexer`. This class writes to
        its own table, so switching over requires a full re-ingest.
    """

    TABLE_NAME = "rag_documents"
    RECORD_MANAGER_TABLE = "rag_record_manager"

    def __init__(
        self,
        embeddings: Embeddings,
        connection_string: str | None = None,
        table_name: str = TABLE_NAME,
        vector_size: int | None = None,
    ):
        # Defaults come from the environment-driven config, never from literals:
        # a hardcoded DSN leaks credentials and silently drifts from the
        # deployment the rest of the pipeline talks to.
        connection_string = connection_string or PG_CONNECTION
        vector_size = vector_size if vector_size is not None else VECTOR_SIZE

        from langchain_postgres.v2.engine import PGEngine
        from langchain_postgres.v2.hybrid_search_config import (
            HybridSearchConfig,
            reciprocal_rank_fusion,
        )
        from langchain_postgres.v2.vectorstores import PGVectorStore

        self._embeddings = embeddings
        self._connection_string = connection_string
        self._table_name = table_name
        self._vector_size = vector_size

        # Hybrid search config: RRF with tsvector mapped to the 'tsv' column
        self._hybrid_config = HybridSearchConfig(
            tsv_column="tsv",
            tsv_lang="pg_catalog.english",  # Change to pg_catalog.spanish if documents are in Spanish
            fusion_function=reciprocal_rank_fusion,
            primary_top_k=20,
            secondary_top_k=20,
            index_name="rag_documents_tsv_gin",
            index_type="GIN",
        )

        # Create the database engine for connection pooling
        self._engine = PGEngine.from_connection_string(connection_string)

        # Create table with hybrid schema (includes tsv column + GIN index).
        # init_vectorstore_table lacks an 'IF NOT EXISTS' clause, so we catch the constraint error.
        try:
            self._engine.init_vectorstore_table(
                table_name=table_name,
                vector_size=self._vector_size,  # 1024 for bge-m3, 3072 for text-embedding-3-large
                id_column="langchain_id",
                hybrid_search_config=self._hybrid_config,
                overwrite_existing=False,
            )
        except Exception as e:
            if "already exists" in str(e).lower():
                print(
                    f"[indexers] Table '{table_name}' already exists, reusing infrastructure."
                )
            else:
                raise

        # Initialize the underlying PGVectorStore instance
        self._store = PGVectorStore.from_texts(
            texts=[],
            embedding=embeddings,
            engine=self._engine,
            table_name=table_name,
            id_column="langchain_id",
            hybrid_search_config=self._hybrid_config,
        )

        # Ensure the GIN index is applied on the tsv column for fast keyword retrieval
        try:
            self._store.apply_hybrid_search_index()
        except Exception as e:
            if "already exists" in str(e).lower():
                pass  # GIN index is already initialized
            else:
                raise

        # Configure the professional SQLRecordManager to prevent duplicate embeddings
        self._record_manager = SQLRecordManager(
            namespace=f"pgvector/{self._table_name}", db_url=self._connection_string
        )
        self._record_manager.create_schema()

    # ── Incremental Production Indexing ────────────────────────
    def add_documents(self, documents: list[Document]) -> dict:
        """Upserts documents into pgvector and tracks state via SQLRecordManager.

        Calculates content hashes dynamically. Automatically handles garbage collection
        by deleting stale chunks and completely avoids redundant embedding API calls.
        """
        # Inject deterministic chunk IDs into metadata for tracking consistency
        for doc in documents:
            if "chunk_id" not in doc.metadata:
                doc.metadata["chunk_id"] = stable_chunk_id(
                    doc.metadata.get("source", "Unknown"), doc.page_content
                )

        # Execute LangChain's industry-standard high-level indexing API
        indexing_result = index(
            docs_source=documents,
            record_manager=self._record_manager,
            vector_store=self._store,
            cleanup="incremental",  # Cleans up old chunks if the source document gets updated
            source_id_key="source",  # Uses the source metadata attribute as the logical document tracking key
        )
        return indexing_result

    def delete(self, ids: list[str]) -> None:
        """Deletes raw vector entries directly by their specific IDs."""
        self._store.delete(ids=ids)

    def delete_by_source(self, source: str) -> int:
        """Manually drops all chunks belonging to a single source document. Returns deletion count."""
        import psycopg

        conn = psycopg.connect(
            self._connection_string.replace("postgresql+psycopg://", "postgresql://")
        )
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self._table_name} "
                    f"WHERE langchain_metadata->>'source' = %s",
                    (source,),
                )
                return cur.rowcount
        finally:
            conn.close()

    # ── High-Performance Retrieval ──────────────────────────────
    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Hybrid search (RRF: pgvector dense + FTS sparse). Returns [(chunk_id, rrf_score)]."""
        results = self._store.similarity_search_with_score(
            query, k=top_k, hybrid_search_config=self._hybrid_config
        )
        return [
            (doc.metadata.get("chunk_id", ""), float(score)) for doc, score in results
        ]

    def search_documents(self, query: str, top_k: int = 5) -> list[Document]:
        """Hybrid search returning complete, unmarshalled Document objects with metadata."""
        return self._store.similarity_search(
            query, k=top_k, hybrid_search_config=self._hybrid_config
        )

    def get_document(self, chunk_id: str) -> Document | None:
        """Retrieves a single isolated Document by its exact chunk_id (langchain_id)."""
        docs = self._store.get_by_ids([chunk_id])
        return docs[0] if docs else None

    @property
    def store(self):
        """Direct access hook to the underlying raw PGVectorStore v2 layer."""
        return self._store

    @property
    def count(self) -> int:
        """Returns the total number of document nodes currently active inside the database."""
        import psycopg

        conn = psycopg.connect(
            self._connection_string.replace("postgresql+psycopg://", "postgresql://")
        )
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT count(*) FROM {self._table_name}")
                return cur.fetchone()[0]
        finally:
            conn.close()
