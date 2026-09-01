"""Restart reconciliation for admitted Auto-research child commands."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from rcp.agents.command_protocol import ExperimentKickoffArguments, SpawnArguments
from rcp.runs.auto_research import (
    AutoResearchCommandContext,
    AutoResearchCommandEffectResult,
    AutoResearchCommandFile,
    AutoResearchRunRequest,
)
from rcp.runs.auto_research_admission import (
    start_auto_research_child_work,
)
from rcp.runs.auto_research_experiments import (
    AutoResearchExperimentAction,
    AutoResearchExperimentCoordinator,
    AutoResearchExperimentLimitInvalid,
)
from rcp.service import RunRequest
from rcp.storage import (
    AgentCommandInvocationRecord,
    AppStore,
    AutoResearchChildAdmissionRecord,
    AutoResearchCommandFileRecord,
)

if TYPE_CHECKING:
    from rcp.background import BackgroundAgentTasks


AutoResearchWorkerRequestFactory = Callable[
    [AutoResearchCommandContext, SpawnArguments, str, str],
    RunRequest,
]
AutoResearchSeatNodeType = Callable[[str, str, str], str | None]


@dataclass(frozen=True)
class AutoResearchChildAdmissionDeferral:
    """Why one accepted admission could not be reflected on this pass."""

    admission_id: str
    episode_id: str
    child_kind: str
    reason: str


@dataclass(frozen=True)
class AutoResearchChildAdmissionReconciliation:
    examined: int = 0
    reflected: int = 0
    cancelled: int = 0
    deferred: int = 0
    deferrals: tuple[AutoResearchChildAdmissionDeferral, ...] = ()


def reconcile_pending_auto_research_child_admissions(
    store: AppStore,
    background: BackgroundAgentTasks,
    experiment_coordinator: AutoResearchExperimentCoordinator,
    *,
    worker_request_factory: AutoResearchWorkerRequestFactory,
    seat_node_type: AutoResearchSeatNodeType,
    episode_id: str | None = None,
) -> AutoResearchChildAdmissionReconciliation:
    """Re-drive command starts that committed before their child route did.

    The mutable orchestrator workspace is never consulted.  Spawn instructions
    and optional Experiment goals come only from the file snapshot committed in
    the same transaction as the accepted admission and command start.
    """

    examined = reflected = cancelled = 0
    deferrals: list[AutoResearchChildAdmissionDeferral] = []
    for admission in store.pending_auto_research_child_admissions(episode_id):
        examined += 1
        command: AgentCommandInvocationRecord | None = None
        try:
            try:
                command = store.auto_research_child_admission_command(admission.admission_id)
            except RuntimeError as exc:
                raise ValueError(
                    "The accepted child admission has an inconsistent command ledger."
                ) from exc
            if command is None:
                raise ValueError("The accepted child admission lost its originating command.")
            context = _command_context(store, admission, command)
            command_file = _recorded_command_file(store, admission, command)
            context = replace(context, command_file=command_file)
            if admission.child_kind == "work":
                outcome = _reconcile_spawn(
                    store,
                    background,
                    admission,
                    command,
                    context,
                    worker_request_factory=worker_request_factory,
                    seat_node_type=seat_node_type,
                )
            else:
                outcome = _reconcile_experiment(
                    store,
                    experiment_coordinator,
                    admission,
                    command,
                    context,
                )
            _finish_command_if_unknown(store, command, outcome)
        except Exception as exc:
            current = store.auto_research_child_admission(admission.admission_id)
            if current is not None and current.state == "reflected":
                reflected += 1
                continue
            if current is not None and current.state == "cancelled":
                cancelled += 1
                continue
            if not isinstance(exc, ValueError):
                # Infrastructure, provider, and canonical-state availability can
                # change. Keep the atomic admission and unknown command intact so
                # Finish remains blocked and a later reconciliation reuses the
                # same child id and immutable file snapshot.
                #
                # A deferral blocks Finish for as long as it lasts, so it never
                # stays silent: a genuine outage and a defect in this path look
                # identical from the outside, and the caller reports the reason.
                deferrals.append(
                    AutoResearchChildAdmissionDeferral(
                        admission_id=admission.admission_id,
                        episode_id=admission.episode_id,
                        child_kind=admission.child_kind,
                        reason=f"{type(exc).__name__}: {' '.join(str(exc).split())[:400]}",
                    )
                )
                continue
            outcome = _failed_outcome(exc)
            disposition = _settle_failed_admission(store, admission)
            if disposition == "cancelled":
                cancelled += 1
            else:
                reflected += 1
            if command is not None and disposition == "cancelled":
                _finish_command_if_unknown(store, command, outcome)
            continue
        reflected += 1
    return AutoResearchChildAdmissionReconciliation(
        examined=examined,
        reflected=reflected,
        cancelled=cancelled,
        deferred=len(deferrals),
        deferrals=tuple(deferrals),
    )


def _command_context(
    store: AppStore,
    admission: AutoResearchChildAdmissionRecord,
    command: AgentCommandInvocationRecord,
) -> AutoResearchCommandContext:
    expected_verb = "spawn" if admission.child_kind == "work" else "episode"
    expected_child_id = _planned_child_id(
        admission.episode_id,
        admission.child_kind,
        command.idempotency_key,
    )
    if (
        command.episode_id != admission.episode_id
        or command.verb != expected_verb
        or command.idempotency_key is None
        or (command.exited_at is not None and command.status != "unavailable")
        or admission.admission_id != expected_child_id
        or admission.child_id != expected_child_id
    ):
        raise ValueError(
            "The accepted child admission does not name a recoverable keyed child command."
        )
    task = store.agent_task(command.operation_id)
    episode = store.episode(admission.episode_id)
    if (
        task is None
        or episode is None
        or episode.mode != "auto_research"
        or task.kind != "auto_research"
        or task.project_id != admission.project_id
        or task.episode_id != admission.episode_id
        or store.auto_research_invocation_role(task.operation_id) != "orchestrator"
    ):
        raise ValueError("The admitted child command lost its canonical orchestrator parent.")
    request = AutoResearchRunRequest.model_validate(task.request)
    if request.role != "orchestrator":
        raise ValueError("The admitted child command parent is not the orchestrator.")
    return AutoResearchCommandContext(
        episode=episode,
        task=task,
        request=request,
    )


def _recorded_command_file(
    store: AppStore,
    admission: AutoResearchChildAdmissionRecord,
    command: AgentCommandInvocationRecord,
) -> AutoResearchCommandFile | None:
    stored = store.auto_research_command_file(command.command_id)
    arguments = command.start_payload.get("arguments")
    if not isinstance(arguments, dict):
        raise ValueError("The admitted child command has no durable arguments.")
    filename = (
        arguments.get("instruction_file")
        if admission.child_kind == "work"
        else arguments.get("goal_file")
    )
    if filename is None:
        if admission.child_kind == "work" or stored is not None:
            raise ValueError("The admitted child command lost its required file binding.")
        return None
    if not isinstance(filename, str) or stored is None:
        raise ValueError("The admitted child command lost its immutable file snapshot.")
    expected_kind = "instruction" if admission.child_kind == "work" else "goal"
    metadata = command.start_payload.get("command_file")
    if not isinstance(metadata, dict) or not _command_file_matches(
        stored,
        kind=expected_kind,
        filename=filename,
        metadata=metadata,
    ):
        raise ValueError("The admitted child command file does not match its audit record.")
    return AutoResearchCommandFile(
        kind=stored.kind,
        filename=stored.filename,
        text=stored.content,
        sha256=stored.sha256,
    )


def _command_file_matches(
    stored: AutoResearchCommandFileRecord,
    *,
    kind: str,
    filename: str,
    metadata: dict[str, object],
) -> bool:
    return (
        stored.kind == kind
        and stored.filename == filename
        and metadata.get("filename") == stored.filename
        and metadata.get("sha256") == stored.sha256
        and metadata.get("byte_length") == len(stored.content.encode("utf-8"))
    )


def _reconcile_spawn(
    store: AppStore,
    background: BackgroundAgentTasks,
    admission: AutoResearchChildAdmissionRecord,
    command: AgentCommandInvocationRecord,
    context: AutoResearchCommandContext,
    *,
    worker_request_factory: AutoResearchWorkerRequestFactory,
    seat_node_type: AutoResearchSeatNodeType,
) -> AutoResearchCommandEffectResult:
    if command.start_payload.get("planned_worker_id") != admission.child_id:
        raise ValueError("The admitted Spawn command has an invalid deterministic worker id.")
    arguments = SpawnArguments.model_validate(command.start_payload.get("arguments"))
    snapshot = context.command_file
    if snapshot is None or snapshot.filename != arguments.instruction_file:
        raise ValueError("The admitted Spawn command lost its instruction snapshot.")
    node_type = seat_node_type(
        admission.project_id,
        admission.episode_id,
        arguments.seat_node_id,
    )
    if node_type is None or node_type.casefold() not in {"experiment", "blocker"}:
        raise ValueError("Auto-research workers may be seated only on Experiments and Blockers.")
    if store.auto_research_child_work(admission.child_id) is not None:
        raise ValueError("The admitted Spawn already has an unreflected child route.")
    request = worker_request_factory(
        context,
        arguments,
        snapshot.text,
        admission.child_id,
    )
    _validate_worker_request(arguments, snapshot.text, admission.child_id, request)
    worker = start_auto_research_child_work(
        background,
        admission.episode_id,
        request,
        admitted_by_operation_id=context.task.operation_id,
        worker_id=admission.child_id,
        instruction=snapshot.text,
        instruction_sha256=snapshot.sha256,
        admission_id=admission.admission_id,
    )
    return AutoResearchCommandEffectResult(
        message="Ordinary Work was seated and queued after recovery.",
        result={
            "worker_id": admission.child_id,
            "current_operation_id": worker.operation_id,
            "status": worker.status,
            "disposition": "created",
        },
    )


def _validate_worker_request(
    arguments: SpawnArguments,
    instruction: str,
    worker_id: str,
    request: RunRequest,
) -> None:
    if (
        request.mode != "work"
        or request.trigger != "orchestrator"
        or request.patch_kind != "work"
        or request.chat_scope != "node"
        or request.node_id != arguments.seat_node_id
        or request.message != instruction
        or request.chat_id != worker_id
        or request.provider is None
        or request.run_on is None
        or request.session_id is not None
        or request.watcher_ids
        or request.result_view is not None
    ):
        raise ValueError(
            "The resolved worker request changed its ordinary fresh Work mode, seat, or instruction."
        )


def _reconcile_experiment(
    store: AppStore,
    experiment_coordinator: AutoResearchExperimentCoordinator,
    admission: AutoResearchChildAdmissionRecord,
    command: AgentCommandInvocationRecord,
    context: AutoResearchCommandContext,
) -> AutoResearchCommandEffectResult:
    if command.start_payload.get("planned_episode_effect_id") != admission.child_id:
        raise ValueError("The admitted Experiment command has an invalid deterministic episode id.")
    if store.auto_research_child_experiment(admission.child_id) is not None:
        raise ValueError("The admitted Experiment already has an unreflected child route.")
    arguments = ExperimentKickoffArguments.model_validate(command.start_payload.get("arguments"))
    snapshot = context.command_file
    if arguments.goal_file is None:
        if snapshot is not None:
            raise ValueError("The admitted Experiment received an unexpected goal snapshot.")
        goal = goal_sha256 = None
    else:
        if snapshot is None or snapshot.filename != arguments.goal_file:
            raise ValueError("The admitted Experiment lost its immutable goal snapshot.")
        goal = snapshot.text
        goal_sha256 = snapshot.sha256
    action = experiment_coordinator.kick_off(
        auto_research_episode_id=admission.episode_id,
        parent_operation_id=context.task.operation_id,
        child_episode_id=admission.child_id,
        node_id=arguments.node_id,
        goal=goal,
        goal_sha256=goal_sha256,
        invocation_limit=arguments.invocation_limit,
        admission_id=admission.admission_id,
    )
    return _experiment_outcome(action)


def _experiment_outcome(
    action: AutoResearchExperimentAction,
) -> AutoResearchCommandEffectResult:
    result: dict[str, object] = {
        "disposition": action.disposition,
        "episode_id": action.episode_id,
        "status": action.status,
        "experiment_allowance": action.allowance.model_dump(mode="json"),
    }
    if action.operation_id is not None:
        result["operation_id"] = action.operation_id
    return AutoResearchCommandEffectResult(
        message=(
            "Experiment episode was created and queued after recovery."
            if action.disposition == "created"
            else "Experiment replacement was recovered."
        ),
        result=result,
    )


def _finish_command(
    store: AppStore,
    command: AgentCommandInvocationRecord,
    outcome: AutoResearchCommandEffectResult,
) -> None:
    payload: dict[str, object] = {"result": outcome.result}
    if outcome.message:
        payload["diagnostic"] = outcome.message
    store.finish_agent_command(
        command.command_id,
        status=outcome.status,
        payload=payload,
        message=outcome.message or f"Agent command completed with {outcome.status}.",
    )


def _finish_command_if_unknown(
    store: AppStore,
    command: AgentCommandInvocationRecord,
    outcome: AutoResearchCommandEffectResult,
) -> None:
    current = store.agent_command(command.command_id)
    if current is not None and current.exited_at is None:
        _finish_command(store, current, outcome)


def _settle_failed_admission(
    store: AppStore,
    admission: AutoResearchChildAdmissionRecord,
) -> str:
    current = store.auto_research_child_admission(admission.admission_id)
    if current is not None and current.state == "accepted":
        try:
            current = store.cancel_auto_research_child_admission(admission.admission_id)
        except ValueError:
            current = store.auto_research_child_admission(admission.admission_id)
    return "reflected" if current is not None and current.state == "reflected" else "cancelled"


def _diagnostic(exc: Exception) -> str:
    detail = " ".join(str(exc).split())[:1_698].rstrip(".")
    return (
        f"The child was not started because {detail}."
        if detail
        else "The child was not started because its request is no longer valid."
    )


def _failed_outcome(exc: Exception) -> AutoResearchCommandEffectResult:
    if isinstance(exc, AutoResearchExperimentLimitInvalid):
        return AutoResearchCommandEffectResult(
            status="invalid",
            message=str(exc),
            result={
                "disposition": "limit_too_high",
                "experiment_allowance": exc.allowance.model_dump(mode="json"),
            },
        )
    return AutoResearchCommandEffectResult(
        status="invalid" if isinstance(exc, ValueError) else "unavailable",
        message=_diagnostic(exc),
        result={"disposition": "cancelled"},
    )


def _planned_child_id(
    episode_id: str,
    child_kind: str,
    idempotency_key: str | None,
) -> str | None:
    if idempotency_key is None:
        return None
    verb = "spawn" if child_kind == "work" else "episode"
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rcp:auto_research:{episode_id}:{verb}:{idempotency_key}",
        )
    )
