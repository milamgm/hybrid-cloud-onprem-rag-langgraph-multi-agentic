"""Measures retrieval quality against a golden set.

Usage:
    python -m scripts.evaluate                          # current configuration
    python -m scripts.evaluate --compare                # with vs without reranking
    python -m scripts.evaluate --k 10 --fetch-k 100
    python -m scripts.evaluate --golden-set my_set.json

The --compare mode is the one that answers "is reranking earning its latency?".
Run it before accepting any default in this pipeline as correct.
"""

import argparse
import logging
import sys

from src.config.config import get_indexer, get_reranker, get_retriever
from src.evaluation.retrieval import evaluate, load_golden_set
from src.rag.retriever import HybridRetriever

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

DEFAULT_GOLDEN_SET = "tests/fixtures/golden_set.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.evaluate",
        description="Measure retrieval quality against a golden set.",
    )
    parser.add_argument("--golden-set", default=DEFAULT_GOLDEN_SET)
    parser.add_argument("--k", type=int, default=5, help="Documents scored per query.")
    parser.add_argument(
        "--fetch-k", type=int, default=50, help="Candidates retrieved before reranking."
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Score hybrid-only against hybrid+reranking.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print per-question results."
    )
    args = parser.parse_args()

    cases = load_golden_set(args.golden_set)
    print(f"Golden set: {args.golden_set} ({len(cases)} cases)\n")

    if not args.compare:
        report = evaluate(
            get_retriever(k=args.k, fetch_k=args.fetch_k), cases, k=args.k
        )
        print(report.summary())
        if args.verbose:
            _print_cases(report)
        return 0

    # Same indexer for both arms, so the only variable is the reranker.
    indexer = get_indexer()

    baseline = HybridRetriever(indexer=indexer, reranker=None, k=args.k)
    reranked = HybridRetriever(
        indexer=indexer, reranker=get_reranker(), k=args.k, fetch_k=args.fetch_k
    )

    print("── hybrid only (no reranking) ──")
    base_report = evaluate(baseline, cases, k=args.k)
    print(base_report.summary())

    print("\n── hybrid + cross-encoder reranking ──")
    rerank_report = evaluate(reranked, cases, k=args.k)
    print(rerank_report.summary())

    print("\n── delta ──")
    for name, before, after in [
        (f"recall@{args.k}", base_report.recall_at_k, rerank_report.recall_at_k),
        ("MRR", base_report.mrr, rerank_report.mrr),
        (f"nDCG@{args.k}", base_report.ndcg_at_k, rerank_report.ndcg_at_k),
    ]:
        change = after - before
        arrow = "+" if change > 0 else ""
        print(f"  {name:<10} {before:.3f} -> {after:.3f}  ({arrow}{change:.3f})")

    # Reranking reorders a fixed candidate pool, so it cannot add documents that
    # recall missed. A recall change here means fetch_k differed between arms.
    if rerank_report.recall_at_k < base_report.recall_at_k:
        print(
            "\n  NOTE: recall dropped. Reranking pushed relevant documents out of "
            "the top k -- inspect the failing cases before trusting it."
        )

    if args.verbose:
        _print_cases(rerank_report)

    return 0


def _print_cases(report) -> None:
    print("\n── per case ──")
    for case in report.cases:
        rank = f"rank {case.hit_rank}" if case.hit_rank else "MISS"
        print(f"  [{rank:>7}] recall={case.recall:.2f} ndcg={case.ndcg:.2f}  {case.question}")
        if case.hit_rank is None:
            print(f"            expected {case.relevant_sources}")
            print(f"            got      {case.retrieved_sources[:3]}")


if __name__ == "__main__":
    sys.exit(main())
