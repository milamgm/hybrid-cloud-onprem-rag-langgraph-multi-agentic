"""Deterministic controls for a workstation-only development profile."""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass

from src.security.injection import InjectionResult, InjectionViolation

_ATTACK_PATTERNS = (
    re.compile(r"\bignore\s+(all\s+)?(previous|prior|system|developer)\b", re.I),
    re.compile(r"\b(reveal|print|show)\s+(the\s+)?(system|developer)\s+prompt\b", re.I),
    re.compile(r"\b(jailbreak|prompt\s+injection)\b", re.I),
    re.compile(r"<\s*/?\s*(system|developer)\s*>", re.I),
)
_CITATION_PATTERN = re.compile(r"\[(\d+)]")


class DevelopmentInjectionGuard:
    """Block common injection strings without claiming production coverage."""

    @staticmethod
    def _attacked(text: str) -> bool:
        return any(pattern.search(text) for pattern in _ATTACK_PATTERNS)

    def check(
        self, user_prompt: str, documents: list[str] | tuple[str, ...] = ()
    ) -> InjectionResult:
        return InjectionResult(
            user_prompt_attack=self._attacked(user_prompt),
            document_attacks=tuple(
                index
                for index, document in enumerate(documents, 1)
                if self._attacked(document)
            ),
        )

    def enforce(
        self, user_prompt: str, documents: list[str] | tuple[str, ...] = ()
    ) -> InjectionResult:
        result = self.check(user_prompt, documents)
        if not result.allowed:
            raise InjectionViolation("Development injection guard blocked the content.")
        return result


@dataclass(frozen=True)
class DevelopmentValidationResult:
    """Subset of the managed output-validator result consumed by the graph."""

    allowed: bool
    text: str
    violations: tuple[str, ...] = ()


class DevelopmentOutputValidation:
    """Enforce citation integrity and basic output hygiene on a workstation."""

    def validate(
        self,
        text: str,
        *,
        available_citations: Collection[int] = (),
        require_citations: bool = False,
        **_: object,
    ) -> DevelopmentValidationResult:
        sanitized = "".join(
            character for character in text if character >= " " or character in "\n\t"
        )
        citations = {int(marker) for marker in _CITATION_PATTERN.findall(sanitized)}
        available = set(available_citations)
        violations: list[str] = []
        if require_citations and not citations:
            violations.append("grounding: output has no citations")
        if unknown := sorted(citations - available):
            violations.append(f"grounding: unknown citation markers {unknown}")
        return DevelopmentValidationResult(
            allowed=not violations,
            text=sanitized,
            violations=tuple(violations),
        )
