"""Chunking behaviour: structure-aware primary path, lexical fallback."""

from pathlib import Path

import pytest
import tiktoken
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from docling_core.types.doc import DocItemLabel, DoclingDocument
from langchain_core.documents import Document

from src.rag.loaders import ParsedDocument
from src.rag.splitters import _sanitize, chunk_document, chunk_documents

MAX_TOKENS = 128


@pytest.fixture
def tokenizer():
    return OpenAITokenizer(
        tokenizer=tiktoken.get_encoding("cl100k_base"), max_tokens=MAX_TOKENS
    )


@pytest.fixture
def structured_doc():
    """A DoclingDocument with two headed sections and a nested subsection."""
    doc = DoclingDocument(name="handbook")

    doc.add_heading("1 Deployment Policy", level=1)
    doc.add_text(
        label=DocItemLabel.TEXT,
        text=(
            "Every release must be approved before it reaches production. "
            "The approval is recorded in the change management system and "
            "retained for seven years."
        ),
    )

    doc.add_heading("2 Incident Response", level=1)
    doc.add_text(
        label=DocItemLabel.TEXT,
        text=(
            "On detecting an incident the on-call engineer opens a bridge "
            "within fifteen minutes and notifies the service owner."
        ),
    )
    doc.add_heading("2.1 Escalation", level=2)
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="Unresolved incidents escalate to the director after one hour.",
    )

    return doc


@pytest.fixture
def parsed_structured(structured_doc):
    return ParsedDocument(
        source=Path("data/raw/handbook.pdf"), docling_document=structured_doc
    )


def test_structural_chunking_makes_no_embedding_calls(parsed_structured, tokenizer):
    """The point of the rewrite: chunking is pure computation.

    The previous SemanticChunker embedded every sentence to place boundaries,
    spending embedding-provider quota before indexing began. Nothing here can
    make a network call -- the tokenizer is local and no embeddings object is
    even accepted by the API.
    """
    chunks = chunk_document(parsed_structured, tokenizer)

    assert chunks
    assert all(c.metadata["chunker"] == "docling-hybrid" for c in chunks)


def test_chunks_are_prefixed_with_their_headings(parsed_structured, tokenizer):
    """Heading context is what makes an isolated paragraph retrievable."""
    chunks = chunk_document(parsed_structured, tokenizer)

    approval = next(c for c in chunks if "seven years" in c.page_content)
    assert "Deployment Policy" in approval.page_content

    escalation = next(c for c in chunks if "director" in c.page_content)
    assert "Incident Response" in escalation.page_content
    assert "Escalation" in escalation.page_content


def test_headings_are_preserved_as_metadata(parsed_structured, tokenizer):
    """Headings in metadata enable citations and metadata filtering."""
    chunks = chunk_document(parsed_structured, tokenizer)

    assert all("headings" in c.metadata for c in chunks)
    all_headings = {h for c in chunks for h in c.metadata["headings"]}
    assert "1 Deployment Policy" in all_headings


def test_sections_do_not_bleed_into_each_other(parsed_structured, tokenizer):
    """Cutting on structure means a chunk belongs to exactly one section."""
    chunks = chunk_document(parsed_structured, tokenizer)

    approval = next(c for c in chunks if "seven years" in c.page_content)
    assert "on-call engineer" not in approval.page_content


def test_chunks_respect_the_token_budget(tokenizer):
    """The budget is measured with the embedding model's own tokenizer."""
    doc = DoclingDocument(name="long")
    doc.add_heading("1 Overview", level=1)
    doc.add_text(
        label=DocItemLabel.TEXT,
        text=" ".join(
            f"Sentence number {i} describes a distinct operational control."
            for i in range(200)
        ),
    )
    parsed = ParsedDocument(source=Path("long.pdf"), docling_document=doc)

    chunks = chunk_document(parsed, tokenizer)

    assert len(chunks) > 1, "an oversized section must be split"
    # contextualize() prepends headings on top of the body budget, so allow a
    # small margin for the heading tokens themselves.
    assert all(tokenizer.count_tokens(c.page_content) <= MAX_TOKENS + 32 for c in chunks)


def test_source_metadata_is_set_for_incremental_cleanup(parsed_structured, tokenizer):
    chunks = chunk_document(parsed_structured, tokenizer)

    assert all(c.metadata["source"] == "data/raw/handbook.pdf" for c in chunks)
    assert all(isinstance(c.metadata["chunk_index"], int) for c in chunks)


# ── Lexical fallback (unstructured input) ─────────────────────


def test_lexical_fallback_chunks_by_tokens(tokenizer):
    """Documents that lost their structure are still chunked to budget."""
    text = " ".join(
        f"Paragraph {i} covers an unrelated operational topic in detail."
        for i in range(150)
    )
    parsed = ParsedDocument(
        source=Path("scan.pdf"),
        text_documents=[
            Document(page_content=text, metadata={"source": "scan.pdf", "page": 1})
        ],
    )

    chunks = chunk_document(parsed, tokenizer)

    assert len(chunks) > 1
    assert all(c.metadata["chunker"] == "recursive-token" for c in chunks)
    assert all(tokenizer.count_tokens(c.page_content) <= MAX_TOKENS for c in chunks)
    assert all(c.metadata["page"] == 1 for c in chunks)


def test_empty_input_yields_no_chunks(tokenizer):
    empty = ParsedDocument(source=Path("blank.pdf"))
    assert chunk_document(empty, tokenizer) == []

    whitespace = ParsedDocument(
        source=Path("blank.pdf"),
        text_documents=[Document(page_content="  \n\t ", metadata={})],
    )
    assert chunk_document(whitespace, tokenizer) == []


def test_chunk_documents_skips_failures_and_keeps_going(parsed_structured, tokenizer):
    """One malformed document must not abort a corpus-wide ingest."""
    broken = ParsedDocument(source=Path("broken.pdf"), docling_document="not a document")

    chunks = chunk_documents([broken, parsed_structured], tokenizer)

    assert chunks, "the healthy document should still have been chunked"
    assert all(c.metadata["source"] == "data/raw/handbook.pdf" for c in chunks)


def test_sanitize_strips_control_chars_and_dot_leaders():
    assert _sanitize("Section\x00 1 ....... page 4") == "Section 1 page 4"
    assert _sanitize("a\t\t  b") == "a b"
