"""Document loading.

Docling is the primary parser: it recovers the document's *structure* (headings,
tables, lists, page provenance) as a ``DoclingDocument``, not just a flat string.
That structure is what the chunker cuts along, so the loader deliberately hands
it downstream intact rather than flattening to markdown first.

PyMuPDF is the fallback. It is fast and dependency-light but text-only, so
documents that take this path lose their structure and are chunked lexically.
"""

from __future__ import annotations

import gc
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from docling_core.types.doc import DoclingDocument
from langchain_core.documents import Document

logger = logging.getLogger("pipeline.loaders")


@dataclass
class ParsedDocument:
    """A parsed source file, in whichever representation the parser produced.

    Exactly one of the two payloads is populated:

    * ``docling_document`` -- the structured parse. Chunked by structure.
    * ``text_documents`` -- the flat fallback parse. Chunked lexically.
    """

    source: Path
    docling_document: DoclingDocument | None = None
    text_documents: list[Document] = field(default_factory=list)

    @property
    def is_structured(self) -> bool:
        return self.docling_document is not None

    @property
    def is_empty(self) -> bool:
        return self.docling_document is None and not self.text_documents


def _build_pipeline_options():
    import torch
    from docling.datamodel.accelerator_options import (
        AcceleratorDevice,
        AcceleratorOptions,
    )
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    device = (
        AcceleratorDevice.CUDA if torch.cuda.is_available() else AcceleratorDevice.CPU
    )

    return PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(
            device=device,
            cuda_use_flash_attention2=False,
        ),
        artifacts_path=None,
        do_ocr=False,
        # Table structure is what makes tables survive chunking as tables.
        do_table_structure=True,
        do_code_enrichment=False,
        do_formula_enrichment=False,
        do_picture_classification=False,
        do_picture_description=False,
        generate_picture_images=False,
    )


def _load_pdf_docling(path: Path) -> DoclingDocument:
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=_build_pipeline_options()
            ),
        },
    )
    return converter.convert(str(path)).document


def _load_pdf_pymupdf(path: Path) -> list[Document]:
    """Text-only fallback. One Document per page, so page provenance survives."""
    import fitz

    docs: list[Document] = []
    with fitz.open(str(path)) as pdf:
        for page_num in range(len(pdf)):
            text = pdf[page_num].get_text()
            if text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(path),
                            "loader": "pymupdf",
                            "doc_type": "pdf",
                            # 1-based to match what a reader sees in a PDF viewer.
                            "page": page_num + 1,
                        },
                    )
                )
    return docs


def load_pdf(path: Path) -> ParsedDocument:
    """Parses a PDF, preferring the structured Docling path."""
    if os.environ.get("RAG_DOCLING", "1") == "0":
        logger.info(f"RAG_DOCLING=0; using PyMuPDF for {path.name}.")
        return ParsedDocument(source=path, text_documents=_load_pdf_pymupdf(path))

    try:
        dl_doc = _load_pdf_docling(path)
        return ParsedDocument(source=path, docling_document=dl_doc)
    except Exception as error:
        logger.warning(
            f"Docling failed on {path.name}: {error}. "
            f"Falling back to PyMuPDF (structure will be lost)."
        )
        return ParsedDocument(source=path, text_documents=_load_pdf_pymupdf(path))
    finally:
        # Docling's layout models are GPU-resident; release between documents so
        # a long corpus does not accumulate allocator fragmentation.
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


_LOADERS = {
    ".pdf": load_pdf,
}

# Single source of truth for what the pipeline can ingest.
SUPPORTED_EXTENSIONS = frozenset(_LOADERS)


def load_any(path: str | Path) -> ParsedDocument:
    """Parses `path` with the loader registered for its extension."""
    p = Path(path)
    loader = _LOADERS.get(p.suffix.lower())
    if loader is None:
        raise ValueError(
            f"No loader registered for '{p.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return loader(p)
