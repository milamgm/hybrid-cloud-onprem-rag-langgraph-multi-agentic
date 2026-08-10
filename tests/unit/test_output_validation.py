from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

import src.security.output_validation as output_validation


@dataclass(frozen=True)
class FakePrivacyResult:
    text: str
    entities: tuple[str, ...] = ()


class FakePrivacy:
    def sanitize(self, text: str, *, language: str) -> FakePrivacyResult:
        assert language == "es"
        return FakePrivacyResult(text.replace("ana@example.com", "<PII>"), ("EMAIL",))


class ToolResponse(BaseModel):
    operation: str
    limit: int


def test_validates_json_pydantic_json_schema_citations_and_pii(monkeypatch):
    monkeypatch.setenv("INFRASTRUCTURE_MODE", "cloud")
    monkeypatch.setenv("AZURE_CONTENT_SAFETY_ENDPOINT", "https://safety.example")
    monkeypatch.setenv("AZURE_CONTENT_SAFETY_KEY", "secret")
    monkeypatch.setattr(
        output_validation,
        "_post_json",
        lambda *args: {"categoriesAnalysis": [{"category": "Violence", "severity": 0}]},
    )
    middleware = output_validation.OutputValidationMiddleware(privacy=FakePrivacy())

    result = middleware.validate(
        '{"operation":"search","limit":2,"owner":"ana@example.com","note":"source [1]"}',
        response_model=ToolResponse,
        json_schema={"type": "object", "required": ["operation", "limit"]},
        available_citations=[1],
        require_citations=True,
    )

    assert result.allowed
    assert result.parsed_json == {
        "operation": "search",
        "limit": 2,
        "owner": "<PII>",
        "note": "source [1]",
    }
    assert result.citations == (1,)
    assert result.pii_entities == ("EMAIL",)


def test_reports_invalid_schema_unknown_citation_and_business_policy(monkeypatch):
    monkeypatch.setenv("INFRASTRUCTURE_MODE", "cloud")
    monkeypatch.setenv("AZURE_CONTENT_SAFETY_ENDPOINT", "https://safety.example")
    monkeypatch.setenv("AZURE_CONTENT_SAFETY_KEY", "secret")
    monkeypatch.setattr(
        output_validation, "_post_json", lambda *args: {"categoriesAnalysis": []}
    )
    middleware = output_validation.OutputValidationMiddleware(privacy=FakePrivacy())

    result = middleware.validate(
        '{"operation":"delete","limit":"many","note":"source [8]"}',
        response_model=ToolResponse,
        available_citations=[1],
        require_citations=True,
        business_policy=lambda text, parsed: "delete needs a human approval",
    )

    assert result.allowed is False
    assert any("Pydantic schema" in violation for violation in result.violations)
    assert "grounding: unknown citation markers [8]" in result.violations
    assert "business policy: delete needs a human approval" in result.violations


def test_azure_content_safety_blocks_harmful_output(monkeypatch):
    monkeypatch.setenv("INFRASTRUCTURE_MODE", "cloud")
    monkeypatch.setenv("AZURE_CONTENT_SAFETY_ENDPOINT", "https://safety.example")
    monkeypatch.setenv("AZURE_CONTENT_SAFETY_KEY", "secret")
    monkeypatch.setattr(
        output_validation,
        "_post_json",
        lambda *args: {"categoriesAnalysis": [{"category": "Violence", "severity": 4}]},
    )
    middleware = output_validation.OutputValidationMiddleware(privacy=FakePrivacy())

    result = middleware.validate("unsafe output", redact_pii=False)

    assert result.allowed is False
    assert result.violations == ("Azure Content Safety: Violence",)
