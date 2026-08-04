"""Retrieval funnel: recall width, rerank ordering, thresholds, filters.

These use fakes rather than Postgres and a real cross-encoder, so they run in
milliseconds and assert on the retriever's own logic. The integration behaviour
(that hybrid search actually fuses two arms) is not something a fake can prove.
"""

import pytest
from langchain_core.documents import Document

from src.rag.retriever import HybridRetriever


class FakeIndexer:
    """Records how it was called and replays a fixed candidate list."""

    def __init__(self, documents):
        self.documents = documents
        self.calls = []

    def search(self, query, k=50, filter=None):
        self.calls.append({"query": query, "k": k, "filter": filter})
        docs = self.documents
        if filter:
            docs = [
                d
                for d in docs
                if all(d.metadata.get(key) == value for key, value in filter.items())
            ]
        # Fused scores descend; their absolute value carries no meaning.
        return [(d, 1.0 / (i + 1)) for i, d in enumerate(docs[:k])]


class FakeReranker:
    """Scores by looking up a keyword, so expected orderings are explicit."""

    def __init__(self, scores_by_keyword):
        self.scores_by_keyword = scores_by_keyword
        self.pairs_seen = []

    def predict(self, pairs):
        self.pairs_seen = pairs
        scores = []
        for _query, text in pairs:
            score = 0.0
            for keyword, value in self.scores_by_keyword.items():
                if keyword in text:
                    score = max(score, value)
            scores.append(score)
        return scores


@pytest.fixture
def documents():
    return [
        Document(page_content="alpha content", metadata={"source": "a.pdf"}),
        Document(page_content="beta content", metadata={"source": "b.pdf"}),
        Document(page_content="gamma content", metadata={"source": "c.pdf"}),
        Document(page_content="delta content", metadata={"source": "d.pdf"}),
    ]


def test_recall_stage_fetches_wider_than_it_returns(documents):
    """A reranker cannot rank what recall never retrieved."""
    indexer = FakeIndexer(documents)
    retriever = HybridRetriever(
        indexer=indexer, reranker=FakeReranker({"alpha": 1.0}), k=2, fetch_k=50
    )

    retriever.invoke("anything")

    assert indexer.calls[0]["k"] == 50, "must retrieve fetch_k, not k"


def test_without_a_reranker_no_extra_candidates_are_fetched(documents):
    """With nothing to narrow, a wide fetch would be pure waste."""
    indexer = FakeIndexer(documents)
    retriever = HybridRetriever(indexer=indexer, reranker=None, k=2, fetch_k=50)

    results = retriever.invoke("anything")

    assert indexer.calls[0]["k"] == 2
    assert len(results) == 2


def test_reranker_reorders_the_candidate_list(documents):
    """The whole point: fused rank order is not final relevance order."""
    indexer = FakeIndexer(documents)
    # 'delta' arrives last from recall but is the most relevant.
    reranker = FakeReranker({"delta": 0.99, "alpha": 0.10})
    retriever = HybridRetriever(indexer=indexer, reranker=reranker, k=2, fetch_k=10)

    results = retriever.invoke("query")

    assert [d.metadata["source"] for d in results] == ["d.pdf", "a.pdf"]


def test_rerank_scores_are_attached_to_the_documents(documents):
    """Scores must survive downstream for citations, debugging, and evals."""
    indexer = FakeIndexer(documents)
    retriever = HybridRetriever(
        indexer=indexer, reranker=FakeReranker({"beta": 0.77}), k=1, fetch_k=10
    )

    results = retriever.invoke("query")

    assert results[0].metadata["rerank_score"] == pytest.approx(0.77)
    assert results[0].metadata["source"] == "b.pdf"


def test_reranker_receives_query_document_pairs(documents):
    """A cross-encoder needs both sides; passing documents alone would be a bug."""
    indexer = FakeIndexer(documents)
    reranker = FakeReranker({"alpha": 1.0})
    retriever = HybridRetriever(indexer=indexer, reranker=reranker, k=1, fetch_k=10)

    retriever.invoke("my question")

    assert all(pair[0] == "my question" for pair in reranker.pairs_seen)
    assert len(reranker.pairs_seen) == len(documents)


def test_score_threshold_drops_weak_matches(documents):
    indexer = FakeIndexer(documents)
    reranker = FakeReranker({"alpha": 0.9, "beta": 0.2})
    retriever = HybridRetriever(
        indexer=indexer, reranker=reranker, k=4, fetch_k=10, score_threshold=0.5
    )

    results = retriever.invoke("query")

    assert [d.metadata["source"] for d in results] == ["a.pdf"]


def test_threshold_above_every_score_returns_nothing(documents):
    """Returning nothing beats returning confident-looking noise."""
    indexer = FakeIndexer(documents)
    retriever = HybridRetriever(
        indexer=indexer,
        reranker=FakeReranker({"alpha": 0.3}),
        k=4,
        fetch_k=10,
        score_threshold=0.9,
    )

    assert retriever.invoke("query") == []


def test_metadata_filter_reaches_the_index(documents):
    """Filtering at the index is cheaper than filtering after retrieval."""
    indexer = FakeIndexer(documents)
    retriever = HybridRetriever(
        indexer=indexer,
        reranker=None,
        k=5,
        filter={"source": "c.pdf"},
    )

    results = retriever.invoke("query")

    assert indexer.calls[0]["filter"] == {"source": "c.pdf"}
    assert [d.metadata["source"] for d in results] == ["c.pdf"]


def test_empty_index_returns_empty_list(documents):
    retriever = HybridRetriever(
        indexer=FakeIndexer([]), reranker=FakeReranker({}), k=5, fetch_k=10
    )

    assert retriever.invoke("query") == []


def test_k_larger_than_available_documents_is_not_an_error(documents):
    retriever = HybridRetriever(
        indexer=FakeIndexer(documents),
        reranker=FakeReranker({"alpha": 1.0}),
        k=99,
        fetch_k=99,
    )

    assert len(retriever.invoke("query")) == len(documents)
