from __future__ import annotations

from pathlib import Path

from src.governance.evidence import EvidenceLedger, GovernanceEvent, GovernanceService


def test_evidence_chain_records_versions_decisions_and_reviews(tmp_path: Path):
    ledger = EvidenceLedger(tmp_path / "governance.jsonl")
    governance = GovernanceService(ledger)

    version = governance.register_version(
        "rag-generator",
        version="2026.08.11",
        artifact_sha256="a" * 64,
        policy_version="tool-policy@3",
    )
    decision = governance.record_policy_decision(
        "tool.search_web",
        policy_id="tool-policy",
        policy_version="3",
        decision="allow",
        reason_code="role_and_limit_valid",
    )
    governance.record_risk_review(
        "rag-generator",
        review_id="risk-42",
        risk_level="medium",
        reviewer_id="risk-team",
        outcome="approved_with_controls",
    )

    assert version.previous_hash is None
    assert decision.previous_hash == version.event_hash
    assert ledger.verify() is True


def test_evidence_rejects_raw_sensitive_content(tmp_path: Path):
    ledger = EvidenceLedger(tmp_path / "governance.jsonl")

    try:
        ledger.append(GovernanceEvent("incident_reported", "rag", {"prompt": "secret"}))
    except ValueError as error:
        assert "must not contain" in str(error)
    else:
        raise AssertionError("Expected sensitive evidence to be rejected")


def test_evidence_detects_tampering(tmp_path: Path):
    path = tmp_path / "governance.jsonl"
    ledger = EvidenceLedger(path)
    ledger.append(
        GovernanceEvent("metric_recorded", "rag", {"name": "coverage", "value": 0.9})
    )
    path.write_text(
        path.read_text(encoding="utf-8").replace("coverage", "tampered"),
        encoding="utf-8",
    )

    assert ledger.verify() is False
