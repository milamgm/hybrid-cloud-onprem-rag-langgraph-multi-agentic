"""Document ingestion pipeline: load, chunk, and sync into the vector index.

Usage:
    python -m scripts.ingest <file_or_directory> [...] [--dry-run]

Each source file is indexed as its own transaction. That keeps memory bounded
on large corpora and, more importantly, makes the run resumable: a provider
outage on file 40 does not discard the work already committed for files 1-39,
and re-running skips them via the record manager's content hashes.
"""

import os

# Must precede any torch import: caps fragmentation during Docling's GPU passes.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import logging
import sys
from pathlib import Path

from src.config.config import get_indexer, get_tokenizer
from src.rag.loaders import SUPPORTED_EXTENSIONS, load_any
from src.rag.splitters import chunk_document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pipeline.ingest")


def collect_target_files(paths: list[str]) -> list[Path]:
    """Resolves paths into a sorted, deduplicated list of ingestible files."""
    resolved: set[Path] = set()

    for path_str in paths:
        target = Path(path_str)
        if target.is_dir():
            logger.info(f"Scanning directory: {target.resolve()}")
            resolved.update(
                file
                for file in target.rglob("*")
                if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        elif target.is_file():
            if target.suffix.lower() in SUPPORTED_EXTENSIONS:
                resolved.add(target)
            else:
                logger.warning(f"Skipping unsupported extension: {target.name}")
        else:
            logger.error(f"Path does not exist: {path_str}")

    return sorted(resolved)


def ingest_file(file_path: Path, indexer, tokenizer, dry_run: bool = False) -> dict:
    """Loads, chunks, and indexes a single file. Returns its sync metrics."""
    empty = {"num_added": 0, "num_updated": 0, "num_skipped": 0, "num_deleted": 0}

    logger.info(f"Loading: {file_path.name}")
    parsed = load_any(file_path)
    if parsed.is_empty:
        logger.warning(f"No extractable content in {file_path.name}; skipping.")
        return empty

    logger.info(
        f"Parsed {file_path.name} "
        f"({'structured' if parsed.is_structured else 'text-only fallback'})."
    )

    chunks = chunk_document(parsed, tokenizer)
    if not chunks:
        logger.warning(f"Chunking produced nothing for {file_path.name}; skipping.")
        return empty

    # The record manager keys incremental cleanup off this field, so a chunk
    # without it would be untrackable.
    for chunk in chunks:
        chunk.metadata.setdefault("source", str(file_path))

    if dry_run:
        logger.info(f"[dry-run] Would index {len(chunks)} chunk(s).")
        return empty

    metrics = indexer.add_documents(chunks)
    logger.info(
        f"Synced {file_path.name} -> "
        f"added={metrics.get('num_added', 0)} "
        f"updated={metrics.get('num_updated', 0)} "
        f"skipped={metrics.get('num_skipped', 0)} "
        f"deleted={metrics.get('num_deleted', 0)}"
    )
    return metrics


def run_ingestion(paths: list[str], dry_run: bool = False) -> int:
    """Ingests every discovered file. Returns the count of added/updated chunks."""
    target_files = collect_target_files(paths)
    if not target_files:
        logger.error("Aborting: no ingestible files found.")
        return 0

    logger.info(f"Discovered {len(target_files)} file(s) to ingest.")

    # Chunking needs only the tokenizer. The embedding client is built lazily by
    # get_indexer(), so a --dry-run never touches the embedding provider at all.
    tokenizer = get_tokenizer()
    indexer = None if dry_run else get_indexer()

    totals = {"num_added": 0, "num_updated": 0, "num_skipped": 0, "num_deleted": 0}
    failed: list[str] = []

    for position, file_path in enumerate(target_files, start=1):
        logger.info(f"── [{position}/{len(target_files)}] {file_path.name} ──")
        try:
            metrics = ingest_file(file_path, indexer, tokenizer, dry_run=dry_run)
            for key in totals:
                totals[key] += metrics.get(key, 0)
        except Exception as error:
            # One bad file must not sink the run; committed files stay committed.
            logger.exception(f"Failed to ingest {file_path.name}: {error}")
            failed.append(file_path.name)

    logger.info(
        f"Totals -> added={totals['num_added']} updated={totals['num_updated']} "
        f"skipped={totals['num_skipped']} deleted={totals['num_deleted']}"
    )
    if failed:
        logger.warning(f"{len(failed)} file(s) failed: {', '.join(failed)}")

    return totals["num_added"] + totals["num_updated"]


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.ingest",
        description="Load, chunk, and index documents into the vector store.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Files or directories to ingest (directories are scanned recursively).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and chunk without writing to the index.",
    )
    args = parser.parse_args()

    try:
        synced = run_ingestion(args.paths, dry_run=args.dry_run)
    except Exception as error:
        logger.critical(f"Ingestion aborted: {error}", exc_info=error)
        return 1

    logger.info(f"Ingestion complete. {synced} chunk(s) added or updated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
