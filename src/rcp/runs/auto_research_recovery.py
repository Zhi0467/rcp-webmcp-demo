from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

from rcp.runs.auto_research import (
    AutoResearchEndingSignal,
    AutoResearchRunRequest,
    auto_research_failure_signal,
)
from rcp.storage import EpisodeRecord

if TYPE_CHECKING:
    from rcp.background import BackgroundAgentTasks
    from rcp.storage import AppStore


class AutoResearchSettlement(Protocol):
    operation_id: str
    store: AppStore


AutoResearchRecoveryFailure = Literal[
    "provider",
    "network",
    "rate_limit",
    "session_limit",
    "missing_checkpoint",
    "continuation_unavailable",
    "structural_unrecoverable",
]


class AutoResearchOrchestratorTerminalFailure(RuntimeError):
    """Typed verdict for a structurally unrecoverable orchestrator turn.

    Provider diagnostics and arbitrary exception prose must never construct this verdict.
    """

    def __init__(self, diagnostic: str) -> None:
        self.diagnostic = " ".join(diagnostic.split())[:2000]
        super().__init__(self.diagnostic)


def record_structural_failure(
    background: BackgroundAgentTasks,
    *,
    operation_id: str,
    diagnostic: str,
) -> None:
    background.store.record_agent_task_receipt(
        operation_id,
        "auto_research_orchestrator_failure",
        {
            "classification": "structural_unrecoverable",
            "recoverable": False,
            "diagnostic": " ".join(diagnostic.split())[:2000],
        },
        tier="summary",
    )


def reconcile_auto_research_task_settlement(
    background: BackgroundAgentTasks,
    episode: EpisodeRecord,
    request: AutoResearchRunRequest,
    execution: AutoResearchSettlement,
) -> AutoResearchEndingSignal | None:
    """Schedule default recovery or atomically fence one typed terminal failure."""

    store = background.store
    task = store.agent_task(execution.operation_id)
    current = store.episode(episode.episode_id)
    if task is None or current is None or task.status not in {"failed", "interrupted"}:
        return None
    if current.mode != "auto_research":
        raise ValueError("Auto-research settlement received another episode mode")
    if request.role == "worker":
        return None

    structural = any(
        receipt.category == "auto_research_orchestrator_failure"
        and receipt.payload.get("classification") == "structural_unrecoverable"
        and receipt.payload.get("recoverable") is False
        for receipt in store.agent_task_receipts(task.operation_id)
    )
    if structural and request.role == "orchestrator":
        return auto_research_failure_signal(
            store,
            current.episode_id,
            diagnostic=task.error or "The auto_research orchestrator failed structurally.",
        )

    failure_kind, retry_mode = _recoverable_failure(store, task.operation_id, request)
    store.schedule_auto_research_task_recovery(
        task.operation_id,
        failure_kind=failure_kind,
        retry_mode=retry_mode,
        diagnostic=task.error or "The auto_research actor turn failed.",
    )
    return None


def reconcile_due_auto_research_recoveries(
    background: BackgroundAgentTasks,
    *,
    as_of: str | None = None,
) -> int:
    """Attempt every due durable recovery once; callers provide the process heartbeat."""

    store = background.store
    reconciled = 0
    for recovery in store.due_auto_research_recoveries(as_of=as_of):
        if recovery.status != "pending":
            continue
        try:
            child = store.auto_research_task_recovery_child(recovery.operation_id)
            if child is None:
                child = background.retry(recovery.operation_id)
            store.complete_auto_research_recovery(
                recovery.recovery_id,
                admitted_operation_id=child.operation_id,
                expected_operation_id=recovery.operation_id,
            )
        except Exception as exc:
            child = store.auto_research_task_recovery_child(recovery.operation_id)
            task = store.agent_task(recovery.operation_id)
            if child is not None:
                store.complete_auto_research_recovery(
                    recovery.recovery_id,
                    admitted_operation_id=child.operation_id,
                    expected_operation_id=recovery.operation_id,
                )
            elif (
                task is not None
                and task.status in {"paused", "interrupted", "failed"}
                and not task.can_retry
            ):
                store.abandon_auto_research_recovery(
                    recovery.operation_id,
                    diagnostic=str(exc) or "The exact recovery continuation is unavailable.",
                )
            else:
                store.defer_auto_research_recovery(recovery.recovery_id, diagnostic=str(exc))
        reconciled += 1
    return reconciled


def reconcile_orphaned_auto_research_failures(
    background: BackgroundAgentTasks,
) -> list[AutoResearchEndingSignal]:
    """Rebuild recovery decisions for failures/interruption persisted before restart."""

    endings: list[AutoResearchEndingSignal] = []
    for task in background.store.auto_research_recovery_candidates():
        request = AutoResearchRunRequest.model_validate(task.request)
        episode = background.store.episode(request.episode_id)
        if episode is None or episode.mode != "auto_research":
            continue
        execution = _StoredSettlement(task.operation_id, background.store)
        ending = reconcile_auto_research_task_settlement(
            background,
            episode,
            request,
            execution,
        )
        if ending is not None:
            endings.append(ending)
    return endings


class _StoredSettlement:
    def __init__(self, operation_id: str, store) -> None:
        self.operation_id = operation_id
        self.store = store


def _recoverable_failure(
    store,
    operation_id: str,
    request: AutoResearchRunRequest,
) -> tuple[AutoResearchRecoveryFailure, Literal["exact", "clean"]]:
    receipts = store.agent_task_receipts(operation_id)
    if any(
        receipt.category == "continuation_context_unavailable"
        and receipt.payload.get("retry_required") is True
        for receipt in receipts
    ):
        return "continuation_unavailable", "clean" if request.role == "orchestrator" else "exact"
    terminal = next(
        (
            receipt.payload.get("classification")
            for receipt in reversed(receipts)
            if receipt.category == "provider_terminal_error"
        ),
        None,
    )
    if terminal == "session_limit":
        return "session_limit", "clean" if request.role == "orchestrator" else "exact"
    task = store.agent_task(operation_id)
    if task is None or not task.native_session_id or not task.stage_root:
        return "missing_checkpoint", "clean" if request.role == "orchestrator" else "exact"
    return "provider", "exact"
