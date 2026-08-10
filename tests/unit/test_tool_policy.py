from __future__ import annotations

import pytest

import src.security.tool_policy as tool_policy
from src.security.tool_policy import (
    ToolApprovalRequired,
    ToolAuthorizationMiddleware,
    ToolCall,
)


def test_opa_policy_request_contains_full_tool_context(monkeypatch):
    monkeypatch.setenv("INFRASTRUCTURE_MODE", "on_premise")
    monkeypatch.setenv("OPA_URL", "http://opa.example")

    def fake_post(url, payload, headers):
        assert url == "http://opa.example/v1/data/onyx/tools/decision"
        assert payload["input"]["tool_name"] == "search_corpus"
        assert payload["input"]["arguments"] == {"query": "vacation policy"}
        assert payload["input"]["roles"] == ("employee",)
        assert headers == {}
        return {"result": {"allow": True, "requires_approval": False}}

    monkeypatch.setattr(tool_policy, "_post_json", fake_post)
    decision = ToolAuthorizationMiddleware().decide(
        ToolCall(
            tool_name="search_corpus",
            arguments={"query": "vacation policy"},
            roles=("employee",),
        )
    )

    assert decision.allowed is True
    assert decision.requires_approval is False


def test_approval_stops_a_sensitive_call_until_it_is_recorded(monkeypatch):
    middleware = ToolAuthorizationMiddleware()
    monkeypatch.setattr(
        middleware,
        "decide",
        lambda call: tool_policy.ToolDecision(
            allowed=True,
            requires_approval=True,
            reason="Approval required.",
        ),
    )

    with pytest.raises(ToolApprovalRequired, match="Approval required"):
        middleware.enforce(ToolCall(tool_name="send_email"))


def test_cloud_web_search_requires_its_role(monkeypatch):
    monkeypatch.setenv("INFRASTRUCTURE_MODE", "cloud")
    middleware = ToolAuthorizationMiddleware()

    assert middleware.decide(ToolCall(tool_name="search_web")).allowed is False
    assert (
        middleware.decide(
            ToolCall(tool_name="search_web", roles=("tool.web.search",))
        ).allowed
        is True
    )
