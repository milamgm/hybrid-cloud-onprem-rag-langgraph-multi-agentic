"""Hybrid index: dense vectors + Postgres full-text search in one table.

**Why hybrid.** Dense retrieval matches meaning; it is what finds the paraphrase
of a question. It is also unreliable on the exact tokens that matter most in
corporate documents -- product codes, acronyms, part numbers, surnames -- because
those carry little semantic signal and get averaged away in an embedding. BM25
matches them exactly and cheaply. Running both and fusing the results recovers
what either alone misses; published benchmarks put the lift around 7% NDCG over
the better single retriever.

**Why RRF and not score averaging.** BM25 scores are unbounded positive numbers;
cosine similarity lives in [-1, 1]. Averaging them is meaningless -- whichever
scale happens to be larger dominates the ranking. Reciprocal Rank Fusion throws
the scores away and fuses on *rank* alone (``1 / (k + rank)`` summed across
lists), which is scale-free and needs no tuning per corpus.

This module is the **write path and the recall stage**. It owns the schema,
the incremental upsert, and a raw hybrid search that returns a broad candidate
pool. Narrowing that pool to a precise answer set is the retriever's job -- see
:mod:`src.rag.retriever`.
"""

from __future__ import annotations

import hashlib
import logging
import uuid

from langchain_classic.indexes import SQLRecordManager
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.indexing import index

logger = logging.getLogger("pipeline.indexers")

# Fixed namespace for deterministic chunk UUIDs (uuid5): re-ingesting the same
# chunk yields the same id, so upserts stay idempotent.
_CHUNK_NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


def stable_chunk_id(source: str, content: str) -> str:
    """Returns a deterministic UUID for a chunk, keyed by its source and text."""
    digest = hashlib.sha256(f"{source}:{content}".encode()).hexdigest()[:32]
    return str(uuid.uuid5(_CHUNK_NAMESPACE, f"{source}:{digest}"))


class SchemaMismatchError(RuntimeError):
    """Raised when the target table cannot hold what this indexer writes."""


class HybridIndexer:
    """Indexes chunks into pgvector with a parallel full-text index.

    The table carries both representations of every chunk:

    ==================  ==========================================
    ``langchain_id``    UUID primary key (deterministic per chunk)
    ``content``         chunk text
    ``embedding``       dense vector
    ``langchain_meta``  JSON metadata (source, page, headings)
    ``tsv``             tsvector, maintained by Postgres, GIN-indexed
    ==================  ==========================================

    Writes go through LangChain's indexing API with a
    :class:`SQLRecordManager`, so re-ingesting an unchanged document is a no-op
    rather than a re-embedding: content hashes are compared first, and only
    genuinely new or changed chunks reach the embedding provider. On a metered
    endpoint that is the difference between a cheap re-run and a rate limit.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        connection_string: str,
        table_name: str,
        vector_size: int,
        *,
        tsv_language: str = "pg_catalog.english",
        candidate_pool: int = 50,
        rrf_k: float = 60.0,
    ):
        """
        Args:
            embeddings: Embedding model. Must produce `vector_size` dimensions.
            connection_string: SQLAlchemy DSN (sync psycopg driver).
            table_name: Target table. Created if absent.
            vector_size: Embedding width. Must match the model exactly.
            tsv_language: Postgres text-search configuration. Governs stemming
                and stop words, so it must match the corpus language --
                'pg_catalog.english' will not stem Spanish correctly.
            candidate_pool: Rows each retrieval arm contributes before fusion.
                Deliberately wide: recall lost here cannot be recovered by any
                downstream reranker.
            rrf_k: RRF smoothing constant. 60 is the value from the original
                paper and is not usually worth tuning.
        """
        from langchain_postgres.v2.engine import PGEngine
        from langchain_postgres.v2.hybrid_search_config import (
            HybridSearchConfig,
            reciprocal_rank_fusion,
        )
        from langchain_postgres.v2.vectorstores import PGVectorStore

        self._connection_string = connection_string
        self._table_name = table_name
        self._vector_size = vector_size
        self._tsv_language = tsv_language
        self._candidate_pool = candidate_pool
        self._rrf_k = rrf_k
        self._reciprocal_rank_fusion = reciprocal_rank_fusion
        self._HybridSearchConfig = HybridSearchConfig

        self._hybrid_config = self._build_hybrid_config(candidate_pool)

        # PGEngine drives SQLAlchemy's async stack and needs an async driver.
        self._engine = PGEngine.from_connection_string(
            connection_string.replace(
                "postgresql+psycopg://", "postgresql+psycopg_async://"
            )
        )

        self._ensure_table()

        self._store = PGVectorStore.create_sync(
            engine=self._engine,
            embedding_service=embeddings,
            table_name=table_name,
            hybrid_search_config=self._hybrid_config,
        )

        # GIN index on the tsvector column. Without it every keyword query is a
        # sequential scan, which is invisible on 200 rows and fatal on 2 million.
        try:
            self._store.apply_hybrid_search_index()
        except Exception as error:
            if "already exists" not in str(error).lower():
                raise
            logger.debug("Full-text GIN index already present.")

        self._record_manager = SQLRecordManager(
            namespace=f"pgvector/{table_name}", db_url=connection_string
        )
        self._record_manager.create_schema()

    # ── Schema ────────────────────────────────────────────────
    def _build_hybrid_config(self, fetch_top_k: int):
        """Builds a fusion config that actually returns `fetch_top_k` rows.

        reciprocal_rank_fusion defaults to fetch_top_k=4 internally, independent
        of primary_top_k/secondary_top_k. Without this override, asking for 50
        candidates silently yields 4 -- the arms retrieve wide and the fusion
        throws the result away.
        """
        return self._HybridSearchConfig(
            tsv_column="tsv",
            tsv_lang=self._tsv_language,
            fusion_function=self._reciprocal_rank_fusion,
            fusion_function_parameters={
                "rrf_k": self._rrf_k,
                "fetch_top_k": fetch_top_k,
            },
            primary_top_k=fetch_top_k,
            secondary_top_k=fetch_top_k,
            index_name=f"{self._table_name}_tsv_gin",
            index_type="GIN",
        )

    def _existing_columns(self) -> dict[str, str]:
        """Returns {column: type} for the target table, empty if it does not exist."""
        import psycopg

        dsn = self._connection_string.replace("postgresql+psycopg://", "postgresql://")
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT column_name, udt_name FROM information_schema.columns "
                "WHERE table_name = %s",
                (self._table_name,),
            )
            return {name: udt for name, udt in cur.fetchall()}

    def _vector_dimensions(self) -> int | None:
        """Returns the declared width of the embedding column, if any."""
        import psycopg

        dsn = self._connection_string.replace("postgresql+psycopg://", "postgresql://")
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT a.atttypmod FROM pg_attribute a "
                "JOIN pg_class c ON a.attrelid = c.oid "
                "WHERE c.relname = %s AND a.attname = 'embedding'",
                (self._table_name,),
            )
            row = cur.fetchone()
            return row[0] if row and row[0] and row[0] > 0 else None

    def _ensure_table(self) -> None:
        """Creates the table, or verifies an existing one can hold our data.

        Writing into a table with the wrong vector width or no tsv column fails
        later, deep inside a batch, with an opaque driver error. Checking up
        front turns that into one actionable message.
        """
        columns = self._existing_columns()

        if not columns:
            self._engine.init_vectorstore_table(
                table_name=self._table_name,
                vector_size=self._vector_size,
                hybrid_search_config=self._hybrid_config,
            )
            logger.info(
                f"Created hybrid table '{self._table_name}' "
                f"(vector({self._vector_size}) + tsv)."
            )
            return

        problems = []
        if "tsv" not in columns:
            problems.append("missing the 'tsv' column required for full-text search")

        existing_dims = self._vector_dimensions()
        if existing_dims is not None and existing_dims != self._vector_size:
            problems.append(
                f"embedding column is vector({existing_dims}), "
                f"but the configured model produces {self._vector_size}"
            )

        if problems:
            raise SchemaMismatchError(
                f"Table '{self._table_name}' cannot be used: "
                + "; ".join(problems)
                + ". This usually means the table predates hybrid search, or was "
                "built under a different INFRASTRUCTURE_MODE. Point "
                "RAG_TABLE_NAME at a new table and re-ingest, or drop the old "
                "one. Existing rows cannot be migrated -- vectors from a "
                "different model are not comparable."
            )

        logger.info(f"Reusing hybrid table '{self._table_name}'.")

    # ── Write path ────────────────────────────────────────────
    def add_documents(self, documents: list[Document]) -> dict:
        """Upserts chunks, skipping any whose content is already indexed."""
        for doc in documents:
            doc.metadata.setdefault(
                "chunk_id",
                stable_chunk_id(
                    doc.metadata.get("source", "unknown"), doc.page_content
                ),
            )

        return index(
            documents,
            record_manager=self._record_manager,
            vector_store=self._store,
            # Scoped to the source ids present in this batch: stale chunks of an
            # updated document are deleted, other documents are untouched.
            cleanup="incremental",
            source_id_key="source",
        )

    def delete_by_source(self, source: str) -> int:
        """Deletes every chunk belonging to one source document."""
        import psycopg

        dsn = self._connection_string.replace("postgresql+psycopg://", "postgresql://")
        with psycopg.connect(dsn) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self._table_name} "  # noqa: S608 - identifier is config, not input
                    f"WHERE langchain_metadata->>'source' = %s",
                    (source,),
                )
                return cur.rowcount

    # ── Recall stage ──────────────────────────────────────────
    def search(
        self, query: str, k: int = 50, filter: dict | None = None
    ) -> list[tuple[Document, float]]:
        """Runs both retrieval arms and returns the RRF-fused candidates.

        Args:
            query: Natural-language query. Used verbatim by both arms -- the
                dense arm embeds it, the sparse arm tokenizes it.
            k: Candidates to return. Keep this wide (tens, not units); this is
                a recall stage, and a reranker cannot rank what was never
                retrieved.
            filter: Optional metadata equality filter.

        Returns:
            (document, rrf_score) pairs, best first. The score is a fusion
            artefact, not a similarity -- it is comparable within one result
            set and meaningless across queries.
        """
        config = self._build_hybrid_config(max(k, self._candidate_pool))
        return self._store.similarity_search_with_score(
            query, k=k, filter=filter, hybrid_search_config=config
        )

    @property
    def store(self):
        """The underlying PGVectorStore, for callers needing the raw layer."""
        return self._store

    @property
    def count(self) -> int:
        """Number of chunks currently indexed."""
        import psycopg

        dsn = self._connection_string.replace("postgresql+psycopg://", "postgresql://")
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {self._table_name}")  # noqa: S608
            return cur.fetchone()[0]
