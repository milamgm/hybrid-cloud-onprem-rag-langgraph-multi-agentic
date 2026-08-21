"""Human-in-the-loop control plane for resumable LangGraph threads."""

from src.hitl.models import (
    ApprovalDecision,
    HITLResumeResult,
    HITLStatus,
    HumanApprovalPrompt,
)

__all__ = [
    "ApprovalDecision",
    "HITLResumeResult",
    "HITLStatus",
    "HumanApprovalPrompt",
    "HumanApprovalService",
]


def __getattr__(name: str):
    if name == "HumanApprovalService":
        from src.hitl.service import HumanApprovalService

        return HumanApprovalService
    raise AttributeError(name)
