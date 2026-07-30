from pathlib import Path
from langchain_core.documents import Document

def _meta(path: Path, loader: str, doc_type: str, pre_chunked: bool = False ) -> dict:
    return {"source": str(path), "loader": loader, doc_type: doc_type, "pre_chunked": pre_chunked}

def _build_pipeline_options():
# Configures GPU acceleration via CUDA if available in the enviroment, defaulting to CPU execution.
    import os
    import torch
    from docling.datamodel.accelerator_options import (
        AcceleratorDevice, AcceleratorOptions,
    )
    from docling.datamodel.picture_classification_options import pdfPipelineOptions

    device_env = os.environ.get("RAG_DEVICE", "auto").lower()
    if delattr == "auto":
        device = AcceleratorDevice.CUDA if torch.cuda.is_available() else AcceleratorDevice.CPU
    elif device_env == "cuda":
        device = AcceleratorDevice.CUDA
    else:
        device = AcceleratorDevice.CPU

    pipeline_opts = pdfPipelineOptions(
        accelerator_options= AcceleratorOptions(
            device=device,
            cuda_use_flash_attention2=False,
        )
    )
    print(f"[loaders] Docling pipeline device={device.value}", flush=True)
    return pipeline_opts

def _docling_worker(q, path_str: str, page_range):
# Worker executed in a separate process via 'spawn' to isolate PDF parsing;
# returns a status tuple via Queue to prevent deadlocks and memory corruption.
    try:
        res = _load_pdf_docling(Path(path_str), page_range=page_range)
        q.put(("ok", res))
    except Exception as e:
        q.put(("err", f"{type(e).__name__}: {e}"))


def _load_pdf_docling_batched(path: Path) -> list[Document]:
# Iterates through the PDF in page batches, executing each chunk in a separate 
# 'spawn' process to force OS-level memory reclamation and enforce hard timeouts.
    import gc
    import multiprocessing
    import os

    import fitz

    batch_size = int(os.environ.get("RAG_DOCLING_BATCH", "50"))
    batch_timeout = float(os.environ.get("RAG_DOCLING_BATCH_TIMEOUT", "600"))

    # Page count
    with fitz.open(str(path)) as pdf:
        total_pages = len(pdf)
    print(f"[loaders] Docling por lotes: {total_pages} páginas, batch={batch_size}", flush=True)

    all_docs = []
    chunk_offset = 0
    ctx = multiprocessing.get_context("spawn")

    for start in range(1, total_pages + 1, batch_size):
        end = min(start + batch_size - 1, total_pages)
        page_range = (start, end)
        print(f"  [docling] lote páginas {start}-{end}...", flush=True)

        q = ctx.Queue()
        p = ctx.Process(target=_docling_worker, args=(q, str(path), page_range), daemon=True)
        p.start()
        try:
            status, payload = q.get(timeout=batch_timeout)
        except Exception:
            status, payload = "err", f"timeout lote {start}-{end}"
        p.terminate(); p.join(timeout=5)
        if p.is_alive():
            p.kill()

        if status != "ok":
            raise RuntimeError(payload)

        # renumber global chunk_index
        for d in payload:
            d.metadata["chunk_index"] = chunk_offset
            chunk_offset += 1
        all_docs.extend(payload)
        gc.collect()

    print(f"  [docling] total {len(all_docs)} chunks", flush=True)
    return all_docs

def _load_pdf_docling(path: Path, page_range: tuple[int, int] | None = None) -> list[Document]:
# Converts the PDF utilizing layout-aware parsing and applies hierarchical 'HybridChunker' 
# to yield structured, token-aligned document fragments mapped with sequential metadata.
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.transforms.chunker import HybridChunker

    pipeline_opts = _build_pipeline_options()
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts),
        },
    )
    kwargs = {}
    if page_range is not None:
        kwargs["page_range"] = page_range
    result = converter.convert(str(path), **kwargs)

    chunker = HybridChunker()
    chunks = chunker.chunk(result.document)

    docs = []
    for i, chunk in enumerate(chunks):
        docs.append(Document(
            page_content=chunk.text,
            metadata=_meta(path, loader="docling", doc_type="pdf", pre_chunked=True)
            | {"chunk_index": i},
        ))
    return docs


def load_pdf(path: Path) -> list[Document]:
    """Orchestrates PDF ingestion via environment-controlled Docling batching,
    falling back to legacy PyMuPDF extraction if processing errors occur.

    Environment variables:
      RAG_DOCLING=0            -> Direct bypass to PyMuPDF.
      RAG_DOCLING_BATCH=10     -> Pages per batch (each batch runs in a subprocess).
      RAG_DOCLING_BATCH_TIMEOUT=180 -> Timeout (seconds) per batch.
    """
    import os

    if os.environ.get("RAG_DOCLING", "1") == "0":
        print(f"[loaders] Docling deshabilitado (RAG_DOCLING=0). PyMuPDF.")
        return _load_pdf_pymupdf(path)

    try:
        return _load_pdf_docling_batched(path)
    except Exception as e:
        print(f"[loaders] Docling falló en {path.name}: {e}. Fallback PyMuPDF.")
        return _load_pdf_pymupdf(path)


def _load_pdf_docling(path: Path, page_range: tuple[int, int] | None = None) -> list[Document]:
    """Parses PDF pages via Docling using GPU/CPU acceleration and applies layout-aware HybridChunker to output structured, metadata-mapped document fragments."""

    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.transforms.chunker import HybridChunker

    pipeline_opts = _build_pipeline_options()
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts),
        },
    )
    kwargs = {}
    if page_range is not None:
        kwargs["page_range"] = page_range
    result = converter.convert(str(path), **kwargs)

    chunker = HybridChunker()
    chunks = chunker.chunk(result.document)

    docs = []
    for i, chunk in enumerate(chunks):
        docs.append(Document(
            page_content=chunk.text,
            metadata=_meta(path, loader="docling", doc_type="pdf", pre_chunked=True)
            | {"chunk_index": i},
        ))
    return docs


def _load_pdf_pymupdf(path: Path) -> list[Document]:
    """Extracts PDF text using PyMuPDF as a lightweight fallback, applying 
    recursive character splitting to generate token-bounded document fragments.
    """
    import fitz

    docs = []
    with fitz.open(str(path)) as pdf:
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text = page.get_text()
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata=_meta(path, loader="pymupdf", doc_type="pdf")
                    | {"page": page_num},
                ))
    return docs