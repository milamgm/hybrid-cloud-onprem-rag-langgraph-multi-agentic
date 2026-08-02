import sys
import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from src.config.config import get_indexer
from src.rag.splitters import chunk_documents
from src.rag.loaders import load_any

# Initialize system logger using strict console formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pipeline.ingest")


def collect_target_files(paths: List[str]) -> List[Path]:
    """Scans provided paths and collects files matching corporate extensions."""
    supported_extensions = {".pdf", ".html", ".htm", ".xlsx", ".docx", ".pptx"}
    resolved_files: List[Path] = []

    for path_str in paths:
        target_path = Path(path_str)
        if target_path.is_dir():
            logger.info(f"Scanning target directory: {target_path.resolve()}")
            resolved_files.extend(
                file
                for file in target_path.rglob("*")
                if file.suffix.lower() in supported_extensions
            )
        elif target_path.is_file():
            if target_path.suffix.lower() in supported_extensions:
                resolved_files.append(target_path)
            else:
                logger.warning(
                    f"Skipping unsupported file extension: {target_path.name}"
                )
        else:
            logger.error(f"Provided path descriptor does not exist: {path_str}")

    return resolved_files


def run_ingestion(paths: List[str]) -> int:
    """Orchestrates document loading, text splitting, and index synchronization."""
    indexer = get_indexer()
    global_document_pool: List[Document] = []

    target_files = collect_target_files(paths)
    if not target_files:
        logger.error("Ingestion sequence aborted: Zero valid files discovered.")
        return 0

    # Stage 1: Parse and load file nodes into raw document objects
    for file_path in target_files:
        try:
            logger.info(f"Parsing storage node: {file_path.name}")
            parsed_documents = load_any(file_path)
            if not parsed_documents:
                logger.warning(f"File node parsed as empty matrix: {file_path.name}")
                continue

            logger.info(
                f"Successfully loaded {len(parsed_documents)} document object nodes."
            )

            # Stage 2: Tokenize and split raw texts into clean semantic chunks
            chunked_segments = chunk_documents(parsed_documents)
            logger.info(
                f"Transformed document metrics: {len(chunked_segments)} atomic chunks generated."
            )
            global_document_pool.extend(chunked_segments)
        except Exception as error:
            logger.error(
                f"Critical failure processing node {file_path.name}: {str(error)}"
            )
            continue

    if not global_document_pool:
        logger.error(
            "Pipeline halted: No available chunks ready for database synchronization."
        )
        return 0

    # Stage 3: Audit trail metadata normalization
    for chunk in global_document_pool:
        if "source" not in chunk.metadata:
            chunk.metadata["source"] = "unknown_origin"

    logger.info(
        f"Initiating transactional sync for {len(global_document_pool)} records..."
    )

    # Stage 4: Commit records via the underlying tracking indexer
    try:
        sync_metrics = indexer.add_documents(global_document_pool)
        logger.info("Database synchronization completed successfully.")
        logger.info(
            f"Metrics -> Added: {sync_metrics.get('num_added', 0)} | "
            f"Updated: {sync_metrics.get('num_updated', 0)} | "
            f"Skipped: {sync_metrics.get('num_skipped', 0)} | "
            f"Deleted: {sync_metrics.get('num_deleted', 0)}"
        )
        return sync_metrics.get("num_added", 0) + sync_metrics.get("num_updated", 0)
    except Exception as network_error:
        logger.critical(
            f"Database transaction aborted due to network or constraint anomalies: {str(network_error)}"
        )
        raise network_error


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Syntax Error: Insufficient parameters.")
        print("Usage: python -m scripts.ingest <file_or_directory_path> [...]")
        sys.exit(1)

    execution_arguments = sys.argv[1:]

    # Drop legacy cleanup flags safely
    if "--cleanup" in execution_arguments:
        flag_index = execution_arguments.index("--cleanup")
        execution_arguments = (
            execution_arguments[:flag_index] + execution_arguments[flag_index + 2 :]
        )

    try:
        total_synchronized = run_ingestion(execution_arguments)
        logger.info(
            f"Pipeline successfully terminated. Clean sync count: {total_synchronized} nodes updated."
        )
    except Exception as runtime_error:
        logger.critical(
            f"Process abnormally terminated via fatal execution signal: {str(runtime_error)}"
        )
        sys.exit(1)
