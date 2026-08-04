"""Retrieval evaluation: metrics over a golden set.

Every knob in this pipeline -- chunk size, `k`, `fetch_k`, whether reranking
helps at all -- is currently set to a defensible default rather than a measured
optimum. This module is what turns those into measurements.

**What it measures.** Retrieval quality only, not answer quality. Whether the
right chunks reach the model is separable from whether the model then uses them
well, and it is the half you can fix by tuning retrieval.

**The metrics, and what each one is blind to:**

``recall@k``
    Fraction of relevant documents that appear anywhere in the top k. Ignores
    where. Use it to size `k` and `fetch_k`: if recall@50 is not near 1.0, the
    reranker is being starved and no amount of reranking will fix it.

``mrr``
    Reciprocal of the rank of the *first* relevant hit, averaged. Rewards
    putting one good answer at the top. Blind to everything after that hit, so
    it flatters a system that finds one document and misses three.

``ndcg@k``
    Discounted gain over *all* relevant hits, normalised against the perfect
    ordering. The one to optimise when several chunks matter. Catches the case
    MRR misses.

Report all three: a change that lifts MRR while dropping recall has traded
breadth for a single lucky hit, and only reading both reveals it.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("pipeline.evaluation")


@dataclass
class EvalCase:
    """One question and the sources that should answer it.

    Attributes:
        question: The query, phrased as a user would actually type it.
        relevant_sources: Source paths that count as correct. Matching at
            source granularity rather than chunk id keeps the golden set stable
            across re-chunking -- chunk ids change whenever chunking changes,
            which would silently invalidate the whole set.
        notes: Free text, e.g. why this case is interesting.
    """

    question: str
    relevant_sources: list[str]
    notes: str = ""


@dataclass
class CaseResult:
    """Per-question outcome, kept so failures can be inspected individually."""

    question: str
    retrieved_sources: list[str]
    relevant_sources: list[str]
    hit_rank: int | None
    recall: float
    reciprocal_rank: float
    ndcg: float


@dataclass
class EvalReport:
    """Aggregate scores plus the per-case detail behind them."""

    k: int
    num_cases: int
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def failures(self) -> list[CaseResult]:
        """Cases where nothing relevant was retrieved at all."""
        return [c for c in self.cases if c.hit_rank is None]

    def summary(self) -> str:
        lines = [
            f"cases={self.num_cases}  k={self.k}",
            f"  recall@{self.k}: {self.recall_at_k:.3f}",
            f"  MRR       : {self.mrr:.3f}",
            f"  nDCG@{self.k}  : {self.ndcg_at_k:.3f}",
        ]
        if self.failures:
            lines.append(f"  total misses: {len(self.failures)}")
            for case in self.failures:
                lines.append(f"    - {case.question}")
        return "\n".join(lines)


def load_golden_set(path: str | Path) -> list[EvalCase]:
    """Reads a golden set from JSON."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        EvalCase(
            question=entry["question"],
            relevant_sources=entry["relevant_sources"],
            notes=entry.get("notes", ""),
        )
        for entry in raw
    ]


def _normalise(source: str) -> str:
    """Compares on filename, so absolute vs relative paths do not cause misses."""
    return Path(source).name


def _dcg(gains: list[float]) -> float:
    return sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))


def evaluate_case(case: EvalCase, retrieved_sources: list[str], k: int) -> CaseResult:
    """Scores one question against what was retrieved for it."""
    relevant = {_normalise(s) for s in case.relevant_sources}
    retrieved = [_normalise(s) for s in retrieved_sources[:k]]

    # Rank is reported over what the user actually sees, chunks included.
    hit_rank = next(
        (i + 1 for i, source in enumerate(retrieved) if source in relevant), None
    )

    # Gains are computed over *distinct* sources, in first-seen order. Relevance
    # here is defined per source document, so a document that happened to chunk
    # into five pieces must score once, not five times. Without this the DCG can
    # exceed its own ideal and nDCG leaves [0, 1] -- the metric silently stops
    # meaning anything.
    seen: set[str] = set()
    distinct = [s for s in retrieved if not (s in seen or seen.add(s))]

    gains = [1.0 if source in relevant else 0.0 for source in distinct]

    found = {source for source in distinct if source in relevant}
    recall = len(found) / len(relevant) if relevant else 0.0

    ideal = _dcg([1.0] * min(len(relevant), k))
    ndcg = _dcg(gains) / ideal if ideal else 0.0

    return CaseResult(
        question=case.question,
        retrieved_sources=retrieved,
        relevant_sources=sorted(relevant),
        hit_rank=hit_rank,
        recall=recall,
        reciprocal_rank=1.0 / hit_rank if hit_rank else 0.0,
        ndcg=ndcg,
    )


def evaluate(retriever: Any, cases: list[EvalCase], k: int = 5) -> EvalReport:
    """Runs every case through `retriever` and aggregates the scores.

    Args:
        retriever: Anything with ``invoke(question) -> list[Document]``, so a
            bare indexer, the full reranking retriever, or a competing
            configuration can all be measured with the same code.
        cases: The golden set.
        k: Cut-off for recall and nDCG.

    Returns:
        Aggregate metrics plus per-case detail.
    """
    if not cases:
        raise ValueError("Cannot evaluate against an empty golden set.")

    results: list[CaseResult] = []
    for case in cases:
        documents = retriever.invoke(case.question)
        sources = [doc.metadata.get("source", "") for doc in documents]
        results.append(evaluate_case(case, sources, k))

    count = len(results)
    return EvalReport(
        k=k,
        num_cases=count,
        recall_at_k=sum(r.recall for r in results) / count,
        mrr=sum(r.reciprocal_rank for r in results) / count,
        ndcg_at_k=sum(r.ndcg for r in results) / count,
        cases=results,
    )
