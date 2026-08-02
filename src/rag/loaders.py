from pathlib import Path
from langchain_core.documents import Document


def _meta(path: Path, loader: str, doc_type: str, pre_chunked: bool = False) -> dict:
    return {
        "source": str(path),
        "loader": loader,
        doc_type: doc_type,
        "pre_chunked": pre_chunked,
    }


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
        do_table_structure=True,
        do_code_enrichment=False,
        do_formula_enrichment=False,
        do_picture_classification=False,
        do_picture_description=False,
        generate_picture_images=False,
    )


def _load_pdf_docling(path: Path) -> list[Document]:
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.transforms.chunker import HybridChunker

    pipeline_opts = _build_pipeline_options()
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts),
        },
    )
    result = converter.convert(str(path))

    chunker = HybridChunker()
    chunks = chunker.chunk(result.document)

    docs = []
    for i, chunk in enumerate(chunks):
        docs.append(
            Document(
                page_content=chunk.text,
                metadata=_meta(path, loader="docling", doc_type="pdf", pre_chunked=True)
                | {"chunk_index": i},
            )
        )
    return docs


def _load_pdf_pymupdf(path: Path) -> list[Document]:
    import fitz

    docs = []
    with fitz.open(str(path)) as pdf:
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text = page.get_text()
            if text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata=_meta(path, loader="pymupdf", doc_type="pdf")
                        | {"page": page_num},
                    )
                )
    return docs


def load_pdf(path: Path) -> list[Document]:
    import os
    import torch
    import gc

    if os.environ.get("RAG_DOCLING", "1") == "0":
        return _load_pdf_pymupdf(path)

    try:
        docs = _load_pdf_docling(path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return docs
    except Exception as e:
        print(f"[loaders] Docling falló en {path.name}: {e}. Fallback PyMuPDF.")
        return _load_pdf_pymupdf(path)


_LOADERS = {
    ".pdf": load_pdf,
}


def load_any(path: str | Path) -> list[Document]:
    p = Path(path)
    loader = _LOADERS.get(p.suffix.lower())
    if loader is None:
        raise ValueError(f"Sin loader para la extensión {p.suffix}")
    return loader(p)
