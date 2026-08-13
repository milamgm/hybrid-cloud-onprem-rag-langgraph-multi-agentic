"""Provider-neutral prompt-injection and jailbreak middleware.

Cloud uses Azure Prompt Shields. On-premise uses NVIDIA NemoGuard
JailbreakDetect NIM through NeMo Guardrails.
The middleware is intentionally independent from the agent state so it can
be tested before being inserted around retrieval and generation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from src.privacy.middleware import _post_json


class InjectionViolation(ValueError):
    """Raised when a prompt or retrieved document contains an attack."""


@dataclass(frozen=True)
class InjectionResult:
    """Security decision for a user prompt and retrieved documents."""

    user_prompt_attack: bool = False
    document_attacks: tuple[int, ...] = ()

    @property
    def allowed(self) -> bool:
        """Whether the prompt and all retrieved documents are safe."""
        return not self.user_prompt_attack and not self.document_attacks


class PromptInjectionMiddleware:
    """Detect direct and indirect prompt injection before model execution."""

    def __init__(self) -> None:
        self._mode = os.getenv("INFRASTRUCTURE_MODE", "on_premise").lower()
        self._nemo_rails = None

    def check(
        self, user_prompt: str, documents: list[str] | tuple[str, ...] = ()
    ) -> InjectionResult:
        """Inspect the user prompt and RAG documents without changing them."""
        if self._mode == "cloud":
            return self._check_azure(user_prompt, documents)
        return self._check_nemo(user_prompt, documents)

    def enforce(
        self, user_prompt: str, documents: list[str] | tuple[str, ...] = ()
    ) -> InjectionResult:
        """Inspect content and raise if a direct or indirect attack is found."""
        result = self.check(user_prompt, documents)
        if not result.allowed:
            locations = []
            if result.user_prompt_attack:
                locations.append("user prompt")
            locations.extend(f"document {index}" for index in result.document_attacks)
            raise InjectionViolation(
                "Prompt injection or jailbreak detected in " + ", ".join(locations)
            )
        return result

    def _check_azure(
        self, user_prompt: str, documents: list[str] | tuple[str, ...]
    ) -> InjectionResult:
        endpoint = os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT")
        key = os.getenv("AZURE_CONTENT_SAFETY_KEY")
        if not endpoint or not key:
            raise ValueError(
                "Cloud prompt protection requires "
                "AZURE_CONTENT_SAFETY_ENDPOINT and AZURE_CONTENT_SAFETY_KEY."
            )

        response = _post_json(
            f"{endpoint.rstrip('/')}/contentsafety/text:shieldPrompt?"
            "api-version=2024-09-01",
            {"userPrompt": user_prompt, "documents": list(documents)},
            {"Ocp-Apim-Subscription-Key": key},
        )
        user_attack = bool(
            response.get("userPromptAnalysis", {}).get("attackDetected", False)
        )
        document_attacks = tuple(
            index
            for index, analysis in enumerate(response.get("documentsAnalysis", []), 1)
            if analysis.get("attackDetected", False)
        )
        return InjectionResult(user_attack, document_attacks)

    def _check_nemo(
        self, user_prompt: str, documents: list[str] | tuple[str, ...]
    ) -> InjectionResult:
        rails = self._get_nemo_rails()
        try:
            from nemoguardrails.rails.llm.options import RailStatus, RailType
        except ImportError as error:  # pragma: no cover - guarded by _get_nemo_rails
            raise RuntimeError(
                "Install nemoguardrails to use on-premise prompt protection."
            ) from error

        def blocked(text: str) -> bool:
            result = rails.check(
                [{"role": "user", "content": text}],
                rail_types=[RailType.INPUT],
            )
            return result.status == RailStatus.BLOCKED

        return InjectionResult(
            user_prompt_attack=blocked(user_prompt),
            document_attacks=tuple(
                index
                for index, document in enumerate(documents, 1)
                if blocked(document)
            ),
        )

    def _get_nemo_rails(self):
        if self._nemo_rails is not None:
            return self._nemo_rails
        try:
            from nemoguardrails import LLMRails, RailsConfig
        except ImportError as error:
            raise RuntimeError(
                "Install nemoguardrails to use on-premise prompt protection."
            ) from error

        nim_base_url = os.getenv(
            "NEMO_JAILBREAK_NIM_BASE_URL", "http://localhost:8123/v1/"
        )
        api_key_env = "NVIDIA_API_KEY"
        if not os.getenv(api_key_env):
            raise ValueError(
                "On-premise prompt protection requires NVIDIA_API_KEY for "
                "NemoGuard JailbreakDetect NIM."
            )

        # NVIDIA's model-based production rail: a dedicated NIM classifies
        # attacks. It does not use the application LLM or a self-check prompt.
        yaml_content = f"""
rails:
  input:
    flows:
      - jailbreak detection model
  config:
    jailbreak_detection:
      nim_base_url: {nim_base_url!r}
      nim_server_endpoint: "/v1/security/nvidia/nemoguard-jailbreak-detect"
      api_key_env_var: {api_key_env}
"""
        config = RailsConfig.from_content(yaml_content=yaml_content)
        self._nemo_rails = LLMRails(config)
        return self._nemo_rails
