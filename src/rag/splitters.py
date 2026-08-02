import re
import logging
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker

logger = logging.getLogger("pipeline.splitters")

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]+")
_DOTRUN_RE = re.compile(r"\.{4,}")
_WS_RE = re.compile(r"[ \t]+")


def _sanitize(text: str) -> str:
    text = _CONTROL_RE.sub(" ", text)
    text = _DOTRUN_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def chunk_documents(
    docs: list[Document],
    embeddings: Embeddings,
    breakpoint_threshold_amount: float = 75,
    min_chunk_size: int = 100,
) -> list[Document]:
    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=breakpoint_threshold_amount,
        min_chunk_size=min_chunk_size,
    )

    result: list[Document] = []

    for doc in docs:
        try:
            if doc.metadata.get("pre_chunked"):
                if doc.page_content.strip():
                    result.append(doc)
                continue

            sanitized_content = _sanitize(doc.page_content)
            if not sanitized_content.strip():
                continue

            clean_doc = Document(page_content=sanitized_content, metadata=doc.metadata)
            chunks = splitter.split_documents([clean_doc])
            result.extend(chunks)

        except Exception as e:
            logger.error(
                f"Critical error processing document {doc.metadata.get('source', 'Unknown')}: {str(e)}"
            )

    return result
