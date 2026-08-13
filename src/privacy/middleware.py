"""Provider-neutral privacy middleware.

Cloud uses Azure AI Language for PII and Azure Content Safety for moderation.
On-premise uses Presidio's local REST services.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib.request import Request, urlopen


class PrivacyViolation(ValueError):
    """Raised when text violates a blocking privacy or safety policy."""


@dataclass(frozen=True)
class PrivacyResult:
    """Sanitized text plus the PII entity categories that were removed."""

    text: str
    entities: tuple[str, ...] = ()


def _post_json(url: str, payload: dict[str, object], headers: dict[str, str]) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urlopen(request, timeout=20) as response:  # noqa: S310 - configured endpoint
        return json.loads(response.read().decode("utf-8"))


class PrivacyMiddleware:
    """Redacts PII before model calls and blocks unsafe text when configured."""

    def __init__(self) -> None:
        self._mode = os.getenv("INFRASTRUCTURE_MODE", "on_premise").lower()

    def sanitize(self, text: str, *, language: str = "es") -> PrivacyResult:
        """Return text safe to pass to a model; empty text is returned unchanged."""
        if not text:
            return PrivacyResult(text)
        if self._mode == "cloud":
            self._check_azure_content_safety(text)
            return self._sanitize_azure(text, language)
        return self._sanitize_presidio(text, language)

    def _check_azure_content_safety(self, text: str) -> None:
        """Block harmful text when Azure Content Safety is configured.

        Content Safety is deliberately optional here so local development can
        continue using only Azure Language PII or Presidio. A severity of 4 or
        higher is blocked by default; tune it with the environment variable.
        """
        endpoint = os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT")
        key = os.getenv("AZURE_CONTENT_SAFETY_KEY")
        if not endpoint or not key:
            return

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
        violations = [
            item
            for item in response.get("categoriesAnalysis", [])
            if int(item.get("severity", 0)) >= threshold
        ]
        if violations:
            categories = ", ".join(item["category"] for item in violations)
            raise PrivacyViolation(
                f"Azure Content Safety blocked the text; categories={categories}"
            )

    def _sanitize_presidio(self, text: str, language: str) -> PrivacyResult:
        analyzer = os.environ["PRESIDIO_ANALYZER_URL"].rstrip("/")
        anonymizer = os.environ["PRESIDIO_ANONYMIZER_URL"].rstrip("/")
        findings = _post_json(
            f"{analyzer}/analyze", {"text": text, "language": language}, {}
        )
        entities = tuple(sorted({item["entity_type"] for item in findings}))
        if not findings:
            return PrivacyResult(text)
        redacted = _post_json(
            f"{anonymizer}/anonymize",
            {
                "text": text,
                "analyzer_results": findings,
                "anonymizers": {"DEFAULT": {"type": "replace", "new_value": "<PII>"}},
            },
            {},
        )
        return PrivacyResult(redacted["text"], entities)

    def _sanitize_azure(self, text: str, language: str) -> PrivacyResult:
        endpoint = os.getenv("AZURE_LANGUAGE_ENDPOINT") or os.getenv(
            "AZURE_FOUNDRY_ENDPOINT"
        )
        key = os.getenv("AZURE_LANGUAGE_KEY") or os.getenv("AZURE_FOUNDRY_API_KEY")
        if not endpoint or not key:
            raise ValueError(
                "Cloud PII requires AZURE_FOUNDRY_ENDPOINT and "
                "AZURE_FOUNDRY_API_KEY (or the legacy AZURE_LANGUAGE_* pair)."
            )

        # Foundry's OpenAI-compatible variable includes this suffix, while the
        # Azure Language runtime endpoint starts at the resource root.
        endpoint = endpoint.removesuffix("/openai/v1").rstrip("/")
        response = _post_json(
            f"{endpoint}/language/:analyze-text?api-version=2024-11-01",
            {
                "kind": "PiiEntityRecognition",
                "parameters": {"modelVersion": "latest"},
                "analysisInput": {
                    "documents": [{"id": "1", "language": language, "text": text}]
                },
            },
            {"Ocp-Apim-Subscription-Key": key},
        )
        document = response["results"]["documents"][0]
        entities = tuple(
            sorted({item["category"] for item in document.get("entities", [])})
        )
        return PrivacyResult(document.get("redactedText", text), entities)
