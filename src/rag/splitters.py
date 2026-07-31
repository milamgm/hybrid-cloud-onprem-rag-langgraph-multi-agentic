import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

# Setup token counter using tiktoken if available, otherwise fallback to character length
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    _token_len = lambda text: len(_enc.encode(text))
except Exception:
    _token_len = len

# Sanitization before tokenization.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]+")
_DOTRUN_RE = re.compile(r"\.{4,}")
_WS_RE = re.compile(r"[ \t]+")

def _sanitize(text: str) -> str:
    text = _CONTROL_RE.sub(" ", text)
    text = _DOTRUN_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text

# Splitter configuration
_default_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=60,
    length_function=_token_len,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# Splits documents while respecting Docling's pre_chunked flag.
# If doc.metadata["pre_chunked"] is True → included as-is (preventing format breakage)
# Otherwise → sanitizes plain text and applies native RecursiveCharacterTextSplitter.
def chunk_documents(
    docs: list[Document],
    splitter: RecursiveCharacterTextSplitter | None = None,
) -> list[Document]:
    _splitter = splitter or _default_splitter
    result: list[Document] = []

    for doc in docs:
        try:
            # CASE A: Document is already pre-chunked by Docling HybridChunker
            if doc.metadata.get("pre_chunked"):
                # Skip aggressive whitespace sanitization to avoid breaking Docling's tables or Markdown format
                if doc.page_content.strip():
                    result.append(doc)
            
            # CASE B: Plain text or traditional PDF requiring standard chunking
            else:
                # 1. Sanitize content (only for documents not processed by Docling)
                sanitized_content = _sanitize(doc.page_content)
                if not sanitized_content.strip():
                    continue
                    
                clean_doc = Document(page_content=sanitized_content, metadata=doc.metadata)
                
                # 2. Splitter cuts safely (the "" separator acts as a native fallback)
                chunks = _splitter.split_documents([clean_doc])
                result.extend(chunks)
                
        except Exception as e:
            # Production fallback: log the error and keep the pipeline running if a document fails
            logger.error(f"Critical error processing document {doc.metadata.get('source', 'Unknown')}: {str(e)}")

    return result
