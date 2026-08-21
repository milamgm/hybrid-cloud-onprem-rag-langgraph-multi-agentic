"""External control-plane service for pausing and resuming graph threads."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from inspect import isawaitable
from typing import Any, Protocol

from langgraph.types import Command

from src.audit.events import AuditService
from src.hitl.models import ApprovalDecision, HITLResumeResult, HITLStatus
from src.hitl.tracing import case_graph_config


class ResumableGraph(Protocol):
    async def ainvoke(self, input: Any, *, config: dict[str, Any]) -> Any:
        """Resume a compiled graph with a command."""


class HITLConflict(RuntimeError):
    """Raised when a thread is not waiting for the requested approval."""


class HITLAuthorizationError(PermissionError):
    """Raised when the verified principal lacks the approval role."""


class HITLResumeTimeout(TimeoutError):
    """Raised when the graph does not finish the resume within the SLA."""


class HumanApprovalService:
    """Validate reviewer authority and resume one persisted graph thread.

    The service receives identity claims from the API gateway or worker. It
    never trusts approver identity from the request body and never exposes the
    raw checkpoint state to the caller.
    """

    def __init__(
        self,
        graph: ResumableGraph,
        *,
        audit: AuditService | None = None,
        required_role: str | None = None,
    ) -> None:
        self._graph = graph
        self._audit = audit or AuditService()
        self._required_role = required_role or os.getenv(
            "HITL_APPROVER_ROLE", "risk.investigation.approve"
        )
        try:
            self._resume_timeout_seconds = float(
                os.getenv("HITL_RESUME_TIMEOUT_SECONDS", "30")
            )
        except ValueError as exc:
            raise ValueError("HITL_RESUME_TIMEOUT_SECONDS must be numeric") from exc
        if not 1 <= self._resume_timeout_seconds <= 300:
            raise ValueError("HITL_RESUME_TIMEOUT_SECONDS must be between 1 and 300")
        # Prevent two requests hitting the same worker from both passing the
        # interrupt check. Cross-replica delivery still relies on partitioning
        # by thread and the deterministic event idempotency key.
        self._resume_locks: dict[str, asyncio.Lock] = {}

    async def status(
        self,
        *,
        thread_id: str,
        tenant_id: str,
    ) -> HITLStatus:
        config = self._config(tenant_id, thread_id)
        snapshot = await self._get_state(config)
        values = dict(getattr(snapshot, "values", {}) or {})
        if not values:
            return HITLStatus(
                thread_id=thread_id,
                investigation_id=thread_id,
                status="not_found",
            )

        interrupts = _interrupt_payloads(snapshot)
        stage = values.get("stage")
        if interrupts:
            status = "awaiting_human"
        elif stage == "rejected":
            status = "rejected"
        elif stage == "approved":
            status = "execution_pending"
        elif stage in {"execution_requested", "completed"}:
            status = "completed"
        else:
            status = "running"
        checkpoint_id = _checkpoint_id(snapshot)
        return HITLStatus(
            thread_id=thread_id,
            investigation_id=str(values.get("investigation_id", thread_id)),
            status=status,
            approval_request_id=_optional_string(values.get("approval_request_id")),
            interrupts=tuple(interrupts),
            checkpoint_id=checkpoint_id,
        )

    async def resume(
        self,
        *,
        thread_id: str,
        tenant_id: str,
        approver_ref: str,
        approver_roles: Iterable[str],
        approval_request_id: str,
        decision: str,
        reason: str,
    ) -> HITLResumeResult:
        roles = tuple(approver_roles)
        if self._required_role not in roles:
            raise HITLAuthorizationError(
                "verified principal lacks the HITL approval role"
            )
        if decision not in {"approved", "rejected"}:
            raise ValueError("decision must be approved or rejected")

        config = self._config(tenant_id, thread_id)
        internal_thread_id = config["configurable"]["thread_id"]
        lock = self._resume_locks.setdefault(internal_thread_id, asyncio.Lock())
        async with lock:
            snapshot = await self._get_state(config)
            values = dict(getattr(snapshot, "values", {}) or {})
            interrupts = _interrupt_payloads(snapshot)
            if not values:
                raise HITLConflict("thread is not waiting for human approval")
            expected_approval_id = values.get("approval_request_id")
            if expected_approval_id != approval_request_id:
                raise HITLConflict(
                    "approval_request_id does not match the persisted thread"
                )

            request_id = str(values.get("request_id", approval_request_id))
            if not interrupts:
                persisted_decision = values.get("human_decision")
                persisted_value = (
                    persisted_decision.get("decision")
                    if isinstance(persisted_decision, Mapping)
                    else getattr(persisted_decision, "decision", None)
                )
                if values.get("stage") != "approved" or (
                    decision != "approved" or persisted_value != "approved"
                ):
                    raise HITLConflict("thread is not waiting for human approval")
                # The approval was checkpointed but command publication failed;
                # continue from the approved state without asking the reviewer
                # to approve a second time.
                raw_result = await self._invoke_with_timeout(
                    {
                        "alert": values.get("alert"),
                        "investigation_id": values.get("investigation_id"),
                        "request_id": values.get("request_id"),
                        "stage": "approved",
                    },
                    config=config,
                )
                return self._resume_result(
                    raw_result,
                    tenant_id=tenant_id,
                    subject_id=approver_ref,
                    thread_id=thread_id,
                    request_id=request_id,
                    approval_request_id=approval_request_id,
                    decision=decision,
                )

            human_decision = ApprovalDecision(
                approval_request_id=approval_request_id,
                decision=decision,
                approver_ref=approver_ref,
                reason=reason,
                decided_at=datetime.now(UTC),
            )
            self._audit.record(
                "human.approval.submitted",
                tenant_id=tenant_id,
                subject_id=approver_ref,
                thread_id=thread_id,
                request_id=request_id,
                payload={
                    "approval_request_id": approval_request_id,
                    "decision": decision,
                    "approver_ref": approver_ref,
                },
            )

            raw_result = await self._invoke_with_timeout(
                Command(resume=human_decision.model_dump(mode="python")),
                config=config,
            )
            return self._resume_result(
                raw_result,
                tenant_id=tenant_id,
                subject_id=approver_ref,
                thread_id=thread_id,
                request_id=request_id,
                approval_request_id=approval_request_id,
                decision=decision,
            )

    def _resume_result(
        self,
        raw_result: Any,
        *,
        tenant_id: str,
        subject_id: str,
        thread_id: str,
        request_id: str,
        approval_request_id: str,
        decision: str,
    ) -> HITLResumeResult:
        state = _public_state(raw_result)
        resulting_stage = str(state.get("stage", "completed"))
        status = {
            "approved": "approved",
            "rejected": "rejected",
            "execution_requested": "execution_requested",
        }.get(resulting_stage, "completed")
        event_ids = tuple(str(value) for value in state.get("event_ids", ()))
        self._audit.record(
            "human.approval.resumed",
            tenant_id=tenant_id,
            subject_id=subject_id,
            thread_id=thread_id,
            request_id=request_id,
            payload={
                "approval_request_id": approval_request_id,
                "decision": decision,
                "resulting_stage": resulting_stage,
                "event_ids": event_ids,
            },
        )
        return HITLResumeResult(
            thread_id=thread_id,
            status=status,
            approval_request_id=approval_request_id,
            state=state,
            event_ids=event_ids,
        )

    @staticmethod
    def _config(tenant_id: str, thread_id: str) -> dict[str, Any]:
        return case_graph_config(tenant_id=tenant_id, case_id=thread_id)

    async def _invoke_with_timeout(
        self, input_value: Any, *, config: dict[str, Any]
    ) -> Any:
        try:
            return await asyncio.wait_for(
                self._graph.ainvoke(input_value, config=config),
                timeout=self._resume_timeout_seconds,
            )
        except TimeoutError as exc:
            raise HITLResumeTimeout(
                "graph resume exceeded the configured HITL timeout"
            ) from exc

    async def _get_state(self, config: dict[str, Any]) -> Any:
        getter = getattr(self._graph, "aget_state", None)
        if getter is not None:
            result = getter(config)
            return await result if isawaitable(result) else result
        result = self._graph.get_state(config)
        return await result if isawaitable(result) else result


def _interrupt_payloads(snapshot: Any) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for task in getattr(snapshot, "tasks", ()) or ():
        for item in getattr(task, "interrupts", ()) or ():
            value = getattr(item, "value", item)
            if isinstance(value, Mapping):
                payloads.append(dict(value))
            else:
                payloads.append({"value": str(value)})
    if payloads:
        return payloads
    raw = getattr(snapshot, "values", {}) or {}
    for item in raw.get("__interrupt__", ()) if isinstance(raw, Mapping) else ():
        value = getattr(item, "value", item)
        payloads.append(
            dict(value) if isinstance(value, Mapping) else {"value": str(value)}
        )
    return payloads


def _checkpoint_id(snapshot: Any) -> str | None:
    config = getattr(snapshot, "config", {}) or {}
    configurable = config.get("configurable", {}) if isinstance(config, Mapping) else {}
    value = configurable.get("checkpoint_id")
    return str(value) if value else None


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None


def _public_state(raw_result: Any) -> dict[str, object]:
    """Return only safe state fields, never raw checkpoint contents."""

    if hasattr(raw_result, "value"):
        raw_result = raw_result.value
    if not isinstance(raw_result, Mapping):
        return {}
    values = raw_result.get("values", raw_result)
    if not isinstance(values, Mapping):
        return {}
    allowed = {"stage", "investigation_id", "approval_request_id", "event_ids"}
    return {key: values[key] for key in allowed if key in values}
