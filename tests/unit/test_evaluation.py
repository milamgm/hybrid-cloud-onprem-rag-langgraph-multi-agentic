"""Metric correctness. A wrong metric is worse than no metric: it hides the bug."""

import math

import pytest
from langchain_core.documents import Document

from src.evaluation.retrieval import EvalCase, evaluate, evaluate_case


def case(question="q", relevant=("a.pdf",)):
    return EvalCase(question=question, relevant_sources=list(relevant))


class StubRetriever:
    """Returns a fixed source list per question."""

    def __init__(self, sources_by_question):
        self.sources_by_question = sources_by_question

    def invoke(self, question):
        return [
            Document(page_content="", metadata={"source": s})
            for s in self.sources_by_question[question]
        ]


def test_perfect_retrieval_scores_one():
    result = evaluate_case(case(), ["a.pdf", "b.pdf", "c.pdf"], k=3)

    assert result.hit_rank == 1
    assert result.recall == 1.0
    assert result.reciprocal_rank == 1.0
    assert result.ndcg == 1.0


def test_complete_miss_scores_zero():
    result = evaluate_case(case(), ["x.pdf", "y.pdf"], k=3)

    assert result.hit_rank is None
    assert result.recall == 0.0
    assert result.reciprocal_rank == 0.0
    assert result.ndcg == 0.0


def test_reciprocal_rank_reflects_position():
    """Finding the answer at rank 3 is worth a third of finding it at rank 1."""
    result = evaluate_case(case(), ["x.pdf", "y.pdf", "a.pdf"], k=5)

    assert result.hit_rank == 3
    assert result.reciprocal_rank == pytest.approx(1 / 3)


def test_recall_counts_distinct_sources_not_chunks():
    """Several chunks of one document must not inflate that document's recall."""
    result = evaluate_case(
        case(relevant=("a.pdf", "b.pdf")), ["a.pdf", "a.pdf", "a.pdf"], k=5
    )

    assert result.recall == 0.5, "a.pdf repeated is still one of two sources"


def test_ndcg_stays_within_bounds_when_one_source_yields_many_chunks():
    """Regression: retrieval returns chunks, relevance is defined per source.

    Counting five chunks of one relevant document as five separate hits made DCG
    exceed its own ideal, producing nDCG values above 1 (observed: 2.779) --
    a metric that looks precise and means nothing.
    """
    result = evaluate_case(case(relevant=("a.pdf",)), ["a.pdf"] * 5, k=5)

    assert 0.0 <= result.ndcg <= 1.0
    assert result.ndcg == pytest.approx(1.0)
    assert result.recall == 1.0


@pytest.mark.parametrize(
    "retrieved",
    [
        ["a.pdf"] * 5,
        ["a.pdf", "a.pdf", "x.pdf"],
        ["x.pdf", "a.pdf", "a.pdf", "b.pdf"],
        ["x.pdf", "y.pdf", "z.pdf"],
    ],
)
def test_ndcg_is_always_a_fraction(retrieved):
    result = evaluate_case(case(relevant=("a.pdf", "b.pdf")), retrieved, k=5)

    assert 0.0 <= result.ndcg <= 1.0


def test_duplicate_chunks_do_not_change_the_reported_rank():
    """hit_rank is about what the user sees, so it counts chunks, not sources."""
    result = evaluate_case(case(relevant=("a.pdf",)), ["x.pdf", "x.pdf", "a.pdf"], k=5)

    assert result.hit_rank == 3


def test_recall_ignores_ordering_but_ndcg_does_not():
    """This is exactly why both are reported."""
    early = evaluate_case(case(relevant=("a.pdf",)), ["a.pdf", "x.pdf", "y.pdf"], k=3)
    late = evaluate_case(case(relevant=("a.pdf",)), ["x.pdf", "y.pdf", "a.pdf"], k=3)

    assert early.recall == late.recall == 1.0
    assert early.ndcg > late.ndcg


def test_ndcg_matches_the_hand_computed_value():
    """Relevant at rank 2 of 2 relevant: DCG = 1/log2(3), ideal = 1 + 1/log2(3)."""
    result = evaluate_case(case(relevant=("a.pdf", "b.pdf")), ["x.pdf", "a.pdf"], k=2)

    dcg = 1 / math.log2(3)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert result.ndcg == pytest.approx(dcg / ideal)


def test_cutoff_k_truncates_the_retrieved_list():
    """A hit beyond k does not count -- the user never sees it."""
    result = evaluate_case(case(), ["x.pdf", "y.pdf", "a.pdf"], k=2)

    assert result.hit_rank is None
    assert result.recall == 0.0


def test_sources_match_on_filename_not_full_path():
    """Golden sets written with bare filenames must match indexed absolute paths."""
    result = evaluate_case(case(relevant=("a.pdf",)), ["/data/raw/a.pdf"], k=3)

    assert result.hit_rank == 1


def test_multi_source_question_needs_both_for_full_recall():
    partial = evaluate_case(case(relevant=("a.pdf", "b.pdf")), ["a.pdf", "x.pdf"], k=5)
    complete = evaluate_case(case(relevant=("a.pdf", "b.pdf")), ["a.pdf", "b.pdf"], k=5)

    assert partial.recall == 0.5
    assert complete.recall == 1.0
    # MRR is identical for both -- it only sees the first hit. This is the
    # blind spot that makes reporting recall alongside it necessary.
    assert partial.reciprocal_rank == complete.reciprocal_rank == 1.0


def test_evaluate_aggregates_across_cases():
    cases = [case("q1", ("a.pdf",)), case("q2", ("b.pdf",))]
    retriever = StubRetriever({"q1": ["a.pdf"], "q2": ["z.pdf"]})

    report = evaluate(retriever, cases, k=3)

    assert report.num_cases == 2
    assert report.recall_at_k == 0.5
    assert report.mrr == 0.5
    assert len(report.failures) == 1
    assert report.failures[0].question == "q2"


def test_empty_golden_set_is_rejected():
    with pytest.raises(ValueError, match="empty golden set"):
        evaluate(StubRetriever({}), [], k=5)
