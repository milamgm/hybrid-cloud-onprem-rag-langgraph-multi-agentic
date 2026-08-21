"""Executable application composition for the governed RAG workflow."""

from src.app.composition import RAGApplication, build_rag_application
from src.app.readiness import ReadinessReport, check_readiness

__all__ = [
    "RAGApplication",
    "ReadinessReport",
    "build_rag_application",
    "check_readiness",
]
