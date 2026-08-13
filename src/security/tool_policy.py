"""Authorization middleware for agent tool calls.

Cloud deployments enforce remote-tool access at API Management with Entra ID.
On-premise deployments ask Open Policy Agent (OPA) for a decision before a tool
is invoked. The middleware remains independent from the agent state until tool
execution is wired into the workflow.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any

from src.privacy.middleware import _post_json


class ToolAuthorizationDenied(PermissionError):
    """Raised when policy rejects a proposed agent tool call."""


class ToolApprovalRequired(PermissionError):
    """Raised when a call is allowed only after a recorded human approval."""


@dataclass(frozen=True)
class ToolCall:
    """The complete policy input for one proposed tool invocation."""

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    subject: str = "anonymous"
    roles: tuple[str, ...] = ()
    data_classification: str = "internal"
    requested_limit: int | None = None
    approval_id: str | None = None


@dataclass(frozen=True)
class ToolDecision:
    """Policy result returned before a tool may execute."""

    allowed: bool
    requires_approval: bool = False
    reason: str = ""


class ToolAuthorizationMiddleware:
    """Authorize agent tool calls and require approval for sensitive actions."""

    def __init__(self) -> None:
        self._mode = os.getenv("INFRASTRUCTURE_MODE", "on_premise").lower()

    def decide(self, call: ToolCall) -> ToolDecision:
        """Return the policy decision without executing the requested tool."""
        if self._mode == "cloud":
            return self._cloud_decision(call)
        return self._opa_decision(call)

    def enforce(self, call: ToolCall) -> ToolDecision:
        """Return a permitted decision or stop the proposed tool invocation."""
        decision = self.decide(call)
        if not decision.allowed:
            raise ToolAuthorizationDenied(
                decision.reason or "Tool call denied by policy."
            )
        if decision.requires_approval and not call.approval_id:
            raise ToolApprovalRequired(
                decision.reason or "Human approval is required before this tool call."
            )
        return decision

    def _opa_decision(self, call: ToolCall) -> ToolDecision:
        opa_url = os.getenv("OPA_URL", "http://localhost:8181").rstrip("/")
        response = _post_json(
            f"{opa_url}/v1/data/onyx/tools/decision",
            {"input": asdict(call)},
            {},
        )
        result = response.get("result")
        if not isinstance(result, dict):
            raise ValueError("OPA returned no decision for the tool call.")
        return ToolDecision(
            allowed=bool(result.get("allow", False)),
            requires_approval=bool(result.get("requires_approval", False)),
            reason=str(result.get("reason", "")),
        )

    @staticmethod
    def _cloud_decision(call: ToolCall) -> ToolDecision:
        """Apply the application-side allowlist before APIM/IAM enforcement.

        APIM validates the caller identity, claims, quotas and operation access
        at the remote-tool boundary. This local check prevents accidental use
        of tools not registered for the agent, including before an HTTP call is
        made to APIM.
        """
        allowed_tools = {"search_corpus", "search_web"}
        if call.tool_name not in allowed_tools:
            return ToolDecision(False, reason="Tool is not allowlisted for this agent.")
        if call.tool_name == "search_web" and "tool.web.search" not in call.roles:
            return ToolDecision(False, reason="Missing tool.web.search role.")
        if call.requested_limit is not None and call.requested_limit > 5:
            return ToolDecision(
                False, reason="Requested limit exceeds the policy maximum."
            )
        return ToolDecision(True)
