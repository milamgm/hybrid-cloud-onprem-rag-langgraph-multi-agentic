"""Validate and sanitize model output before it reaches a user or a tool.

This module deliberately keeps deterministic controls (JSON schema, citation
references and business rules) separate from model-based moderation. Azure AI
Content Safety and NeMo Guardrails determine whether content is safe; neither
one proves that an answer is correctly cited or matches an action contract.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from typing import Any

from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema import validate as validate_json_schema
from pydantic import BaseModel, ValidationError

from src.privacy.middleware import PrivacyMiddleware, _post_json

_CITATION_PATTERN = re.compile(r"\[(\d+)]")
BusinessPolicy = Callable[[str, Any | None], str | None]
GroundingValidator = Callable[[str, Collection[int]], str | None]


class OutputValidationViolation(ValueError):
    """Raised when output must not be returned or executed."""


@dataclass(frozen=True)
class OutputValidationResult:
    """The sanitized output and the checks that were applied to it."""

    text: str
    parsed_json: Any | None = None
    citations: tuple[int, ...] = ()
    pii_entities: tuple[str, ...] = ()
    violations: tuple[str, ...] = ()

    @property
    def allowed(self) -> bool:
        """Whether the output passed every configured blocking check."""
        return not self.violations


class OutputValidationMiddleware:
    """Apply output controls in a fixed, auditable order.

    ``validate`` returns a decision for logging or observability; callers that
    will render output or execute an action must use ``enforce``.
    """

    def __init__(self, *, privacy: PrivacyMiddleware | None = None) -> None:
        self._mode = os.getenv("INFRASTRUCTURE_MODE", "on_premise").lower()
        self._privacy = privacy or PrivacyMiddleware()
        self._nemo_rails = None

    def validate(
        self,
        text: str,
        *,
        response_model: type[BaseModel] | None = None,
        json_schema: Mapping[str, Any] | None = None,
        available_citations: Collection[int] = (),
        require_citations: bool = False,
        grounding_validator: GroundingValidator | None = None,
        business_policy: BusinessPolicy | None = None,
        language: str = "es",
        redact_pii: bool = True,
    ) -> OutputValidationResult:
        """Check output without returning it to an untrusted destination.

        ``grounding_validator`` receives the sanitized text and the cited
        markers. It is the integration point for an entailment verifier that
        has access to the retrieved passages; citation syntax alone is not a
        proof of factual grounding.
        """
        sanitized = self._sanitize_text(text)
        pii_entities: tuple[str, ...] = ()
        if redact_pii:
            pii_result = self._privacy.sanitize(sanitized, language=language)
            sanitized = pii_result.text
            pii_entities = pii_result.entities

        violations: list[str] = []
        parsed_json = self._validate_structure(
            sanitized, response_model, json_schema, violations
        )
        citations = tuple(
            sorted({int(marker) for marker in _CITATION_PATTERN.findall(sanitized)})
        )
        self._validate_citations(
            citations,
            available_citations,
            require_citations,
            violations,
        )

        if grounding_validator:
            violation = grounding_validator(sanitized, citations)
            if violation:
                violations.append(f"grounding: {violation}")

        if business_policy:
            violation = business_policy(sanitized, parsed_json)
            if violation:
                violations.append(f"business policy: {violation}")

        self._validate_content_safety(sanitized, violations)
        return OutputValidationResult(
            text=sanitized,
            parsed_json=parsed_json,
            citations=citations,
            pii_entities=pii_entities,
            violations=tuple(violations),
        )

    def enforce(self, text: str, **kwargs: Any) -> OutputValidationResult:
        """Return safe output or raise before it can be rendered or executed."""
        result = self.validate(text, **kwargs)
        if not result.allowed:
            raise OutputValidationViolation(
                "Output validation blocked: " + "; ".join(result.violations)
            )
        return result

    @staticmethod
    def _sanitize_text(text: str) -> str:
        """Remove control characters that are unsafe across text sinks.

        HTML encoding belongs at the rendering boundary; applying it here would
        corrupt JSON and Markdown. This layer therefore only normalizes the
        transport-neutral control characters and delegates PII redaction to its
        configured provider.
        """
        return "".join(
            character for character in text if character >= " " or character in "\n\t"
        )

    @staticmethod
    def _validate_structure(
        text: str,
        response_model: type[BaseModel] | None,
        json_schema: Mapping[str, Any] | None,
        violations: list[str],
    ) -> Any | None:
        if not response_model and not json_schema:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as error:
            violations.append(f"JSON is invalid: {error.msg}")
            return None

        if response_model:
            try:
                response_model.model_validate(parsed)
            except ValidationError as error:
                violations.append(
                    f"Pydantic schema rejected output: {error.errors()[0]['msg']}"
                )
        if json_schema:
            try:
                validate_json_schema(parsed, json_schema)
            except JsonSchemaValidationError as error:
                violations.append(f"JSON Schema rejected output: {error.message}")
        return parsed

    @staticmethod
    def _validate_citations(
        citations: Collection[int],
        available_citations: Collection[int],
        require_citations: bool,
        violations: list[str],
    ) -> None:
        available = set(available_citations)
        if require_citations and not citations:
            violations.append("grounding: output has no citations")
        unknown = sorted(set(citations) - available)
        if unknown:
            violations.append(f"grounding: unknown citation markers {unknown}")

    def _validate_content_safety(self, text: str, violations: list[str]) -> None:
        if self._mode == "cloud":
            self._validate_azure_content_safety(text, violations)
        else:
            self._validate_nemo_content_safety(text, violations)

    @staticmethod
    def _validate_azure_content_safety(text: str, violations: list[str]) -> None:
        endpoint = os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT")
        key = os.getenv("AZURE_CONTENT_SAFETY_KEY")
        if not endpoint or not key:
            raise ValueError(
                "Cloud output validation requires AZURE_CONTENT_SAFETY_ENDPOINT "
                "and AZURE_CONTENT_SAFETY_KEY."
            )
        response = _post_json(
            f"{endpoint.rstrip('/')}/contentsafety/text:analyze?api-version=2024-09-01",
            {
                "text": text,
                "categories": ["Hate", "Sexual", "SelfHarm", "Violence"],
                "outputType": "FourSeverityLevels",
            },
            {"Ocp-Apim-Subscription-Key": key},
        )
        threshold = int(os.getenv("AZURE_CONTENT_SAFETY_BLOCK_SEVERITY", "4"))
        categories = [
            item["category"]
            for item in response.get("categoriesAnalysis", [])
            if int(item.get("severity", 0)) >= threshold
        ]
        if categories:
            violations.append("Azure Content Safety: " + ", ".join(categories))

    def _validate_nemo_content_safety(self, text: str, violations: list[str]) -> None:
        rails = self._get_nemo_rails()
        try:
            from nemoguardrails.rails.llm.options import RailStatus, RailType
        except ImportError as error:  # pragma: no cover - guarded by _get_nemo_rails
            raise RuntimeError(
                "Install nemoguardrails to validate on-premise output."
            ) from error
        result = rails.check(
            [{"role": "assistant", "content": text}], rail_types=[RailType.OUTPUT]
        )
        if result.status == RailStatus.BLOCKED:
            violations.append("NeMo Guardrails content safety")

    def _get_nemo_rails(self):
        if self._nemo_rails is not None:
            return self._nemo_rails
        try:
            from nemoguardrails import LLMRails, RailsConfig
        except ImportError as error:
            raise RuntimeError(
                "Install nemoguardrails to validate on-premise output."
            ) from error

        base_url = os.getenv(
            "NEMO_CONTENT_SAFETY_NIM_BASE_URL", "http://localhost:8124/v1"
        )
        model = os.getenv(
            "NEMO_CONTENT_SAFETY_NIM_MODEL",
            "nvidia/llama-3.1-nemotron-safety-guard-8b-v3",
        )
        yaml_content = f"""
models:
  - type: content_safety
    engine: nim
    model: {model!r}
    parameters:
      base_url: {base_url!r}
rails:
  output:
    flows:
      - content safety check output $model=content_safety
prompts:
  - task: content_safety_check_output $model=content_safety
    content: |
      Check the assistant response against this safety policy: violence, sexual
      content, self-harm, hate, harassment, illegal activity, fraud, malware,
      privacy, dangerous advice and manipulation.

      <BEGIN CONVERSATION>
      user: {{ user_input }}
      response: agent: {{ bot_response }}
      <END CONVERSATION>

      Return only JSON with "Response Safety" set to "safe" or "unsafe" and,
      if unsafe, "Safety Categories".
    output_parser: nemoguard_parse_response_safety
    max_tokens: 50
"""
        self._nemo_rails = LLMRails(RailsConfig.from_content(yaml_content=yaml_content))
        return self._nemo_rails
