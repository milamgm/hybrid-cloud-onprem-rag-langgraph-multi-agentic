"""Structure-aware, tokenizer-aware chunking.

Strategy, and why it is this one:

**Structure first.** Docling's ``HybridChunker`` cuts along the boundaries the
document itself declares -- headings, tables, list groups -- rather than at
arbitrary character offsets. It then makes two refinement passes against a real
tokenizer: split what overflows the budget, merge undersized neighbours that
share a heading. Cutting along native boundaries is the single highest-leverage
chunking decision; character-count splitting routinely severs a table from its
header or a claim from its qualifier.

**Tokenizer-aligned.** The token budget is measured with the *same* tokenizer
the embedding model uses. Measuring in characters, or with a different
tokenizer, means the budget is a guess -- chunks silently overflow the model's
context and get truncated, losing their tail.

**Heading-contextualized.** Each chunk is embedded as
``chunker.contextualize(chunk)``, which prepends the section headings the chunk
sits under. A paragraph reading "this must be approved before release" is
near-useless in isolation; prefixed with "4.2 Deployment Approval" it is
retrievable. This is what replaces the overlap that lexical splitters need.

**No embedding calls.** Chunking is pure computation. The previous
``SemanticChunker`` approach embedded every sentence just to place boundaries,
which cost one or more embedding API requests per document *before* indexing had
embedded anything -- the direct cause of the provider rate limiting this
pipeline hit. Industry measurements put semantic chunking at ~2-3 points of
recall over recursive splitting, which does not pay for that cost; structural
boundaries capture most of the same benefit for free.

The lexical path below is the fallback for documents that arrived without
structure (the PyMuPDF loader). It is token-aware and overlapped, because
without headings there is no context to prepend.
"""

from __future__ import annotations

import logging
import re

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.loaders import ParsedDocument

logger = logging.getLogger("pipeline.splitters")

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]+")
_DOTRUN_RE = re.compile(r"\.{4,}")
_WS_RE = re.compile(r"[ \t]+")

# Overlap for the lexical fallback only. Structural chunks do not need it: the
# heading prefix supplies the context that overlap would otherwise recover.
FALLBACK_OVERLAP_RATIO = 0.15


def _sanitize(text: str) -> str:
    """Strips control characters, dot leaders, and whitespace runs."""
    text = _CONTROL_RE.sub(" ", text)
    text = _DOTRUN_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def _chunk_structured(
    parsed: ParsedDocument, tokenizer: BaseTokenizer
) -> list[Document]:
    """Chunks a DoclingDocument along its own structure."""
    chunker = HybridChunker(
        tokenizer=tokenizer,
        # Merge consecutive undersized chunks that share a heading, so a section
        # split across short paragraphs stays one retrievable unit.
        merge_peers=True,
        # A table spanning several chunks repeats its header row in each, so no
        # chunk is a grid of numbers with no column names.
        repeat_table_header=True,
    )

    documents: list[Document] = []
    for position, chunk in enumerate(chunker.chunk(parsed.docling_document)):
        # contextualize() is the text to embed: chunk text prefixed with the
        # headings it lives under.
        text = _sanitize(chunker.contextualize(chunk))
        if not text:
            continue

        meta = chunk.meta
        pages = sorted({prov.page_no for item in meta.doc_items for prov in item.prov})
        labels = sorted({str(item.label) for item in meta.doc_items})

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(parsed.source),
                    "loader": "docling",
                    "doc_type": parsed.source.suffix.lstrip(".").lower(),
                    "chunker": "docling-hybrid",
                    "chunk_index": position,
                    # Headings make the chunk citable and enable metadata filters.
                    # (meta.captions is deprecated in docling-core; caption text
                    # already reaches the chunk body via contextualize().)
                    "headings": list(meta.headings or []),
                    # Real page numbers, for citations back to the source PDF.
                    "pages": pages,
                    "page": pages[0] if pages else None,
                    "element_types": labels,
                },
            )
        )

    return documents


def _chunk_lexical(
    parsed: ParsedDocument, tokenizer: BaseTokenizer, max_tokens: int
) -> list[Document]:
    """Chunks unstructured text by token count, with overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=int(max_tokens * FALLBACK_OVERLAP_RATIO),
        # Budget is enforced in tokens, matching the embedding model, not in
        # characters -- the two diverge badly on tables and non-English text.
        length_function=tokenizer.count_tokens,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )

    sanitized: list[Document] = []
    for doc in parsed.text_documents:
        text = _sanitize(doc.page_content)
        if text:
            sanitized.append(Document(page_content=text, metadata=dict(doc.metadata)))

    chunks = splitter.split_documents(sanitized)
    for position, chunk in enumerate(chunks):
        chunk.metadata.setdefault("source", str(parsed.source))
        chunk.metadata["chunker"] = "recursive-token"
        chunk.metadata["chunk_index"] = position

    return chunks


def chunk_document(parsed: ParsedDocument, tokenizer: BaseTokenizer) -> list[Document]:
    """Chunks one parsed document, dispatching on how it was parsed.

    Args:
        parsed: Output of :func:`src.rag.loaders.load_any`.
        tokenizer: Tokenizer of the embedding model that will index these
            chunks. Alignment is required for the token budget to mean anything.

    Returns:
        Chunks ready to embed, each carrying source, page, and heading metadata.
    """
    if parsed.is_empty:
        logger.warning(f"Nothing to chunk in {parsed.source.name}.")
        return []

    if parsed.is_structured:
        chunks = _chunk_structured(parsed, tokenizer)
        strategy = "structural"
    else:
        chunks = _chunk_lexical(parsed, tokenizer, tokenizer.get_max_tokens())
        strategy = "lexical fallback"

    if not chunks:
        logger.warning(f"Chunking produced nothing for {parsed.source.name}.")
        return chunks

    counts = [tokenizer.count_tokens(c.page_content) for c in chunks]
    logger.info(
        f"{parsed.source.name}: {len(chunks)} chunks ({strategy}), "
        f"tokens min={min(counts)} avg={sum(counts) // len(counts)} max={max(counts)}"
    )
    return chunks


def chunk_documents(
    parsed_documents: list[ParsedDocument], tokenizer: BaseTokenizer
) -> list[Document]:
    """Chunks several parsed documents, skipping any that fail."""
    result: list[Document] = []
    for parsed in parsed_documents:
        try:
            result.extend(chunk_document(parsed, tokenizer))
        except Exception as error:
            logger.error(f"Failed to chunk {parsed.source.name}: {error}")
    return result
