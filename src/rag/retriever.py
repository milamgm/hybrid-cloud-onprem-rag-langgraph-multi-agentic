"""Retrieval: the read path, as a two-stage pipeline.

    query -> [recall] hybrid search, ~50 candidates -> [precision] rerank -> top 5

**Why two stages.** The two have opposite failure modes, and no single model is
good at both jobs. Retrieval is a *bi-encoder* problem: the document was
embedded offline, the query is embedded now, and the two vectors are compared.
That comparison is fast enough to scan millions of rows, but the document's
vector was computed without ever seeing the query, so it cannot represent which
parts of the document matter for *this* question.

A **cross-encoder** reranker reads the query and the document *together* in one
forward pass, so it can judge whether a specific passage answers a specific
question. That is far more accurate and far too slow to run over a whole corpus
-- it is O(candidates), not O(corpus). Hence the funnel: retrieve wide and
cheaply, then rank narrow and expensively.

**Recall is the binding constraint.** A reranker can only reorder what the
recall stage handed it. A relevant chunk missing from the candidate pool is
gone for good, so `fetch_k` should be generous (tens) even though `k` is small.
Widening `fetch_k` costs one cross-encoder batch; missing the answer costs the
answer.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

logger = logging.getLogger("pipeline.retriever")


class HybridRetriever(BaseRetriever):
    """Hybrid recall followed by optional cross-encoder reranking.

    Implements LangChain's :class:`BaseRetriever`, so it composes with chains
    and agents and inherits ``invoke``/``ainvoke``, batching, and tracing.

    Attributes:
        indexer: Provides ``search(query, k, filter) -> [(Document, score)]``.
        reranker: A ``sentence_transformers.CrossEncoder``, or None to return
            the fused order unchanged.
        k: Documents returned to the caller.
        fetch_k: Candidates pulled from the index before reranking. Ignored
            when no reranker is configured.
        score_threshold: Drop documents whose rerank score falls below this.
            None keeps the top `k` regardless of absolute score.
        filter: Metadata equality filter applied at the index level.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    indexer: Any
    reranker: Any = None
    k: int = 5
    fetch_k: int = 50
    score_threshold: float | None = None
    filter: dict | None = Field(default=None)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        # Without a reranker there is nothing to narrow, so retrieving more than
        # k candidates would be wasted embedding and I/O work.
        candidate_count = self.fetch_k if self.reranker is not None else self.k

        candidates = self.indexer.search(query, k=candidate_count, filter=self.filter)
        if not candidates:
            logger.debug(f"No candidates for query: {query!r}")
            return []

        documents = [doc for doc, _ in candidates]

        if self.reranker is None:
            return documents[: self.k]

        return self._rerank(query, documents)

    def _rerank(self, query: str, documents: list[Document]) -> list[Document]:
        """Scores every (query, document) pair and keeps the best `k`."""
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.reranker.predict(pairs)

        ranked = sorted(
            zip(documents, scores, strict=True), key=lambda pair: pair[1], reverse=True
        )

        if self.score_threshold is not None:
            kept = [(doc, s) for doc, s in ranked if s >= self.score_threshold]
            if not kept:
                # Returning nothing is usually the honest answer, but callers
                # that cannot handle it get the single best candidate plus a
                # warning they can act on.
                logger.warning(
                    f"No document scored above {self.score_threshold} for "
                    f"query {query!r}; best was {ranked[0][1]:.3f}."
                )
                return []
            ranked = kept

        selected = ranked[: self.k]

        # Scores are attached rather than returned separately so they survive
        # into whatever consumes the documents (citations, debugging, evals).
        results = []
        for doc, score in selected:
            enriched = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "rerank_score": float(score)},
            )
            results.append(enriched)

        logger.debug(
            f"Reranked {len(documents)} candidates to {len(results)}; "
            f"top score {selected[0][1]:.3f}"
        )
        return results
