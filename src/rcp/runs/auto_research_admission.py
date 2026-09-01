"""Admission, recovery, and child lifecycle for an Auto-research episode.

Auto-research policy: seating an orchestrator on its own graph branch, seating
and waking workers and child Experiments, proving what a previous process
committed, and settling a Stop. It is the largest single job type and none of it
generalises.

It takes the engine because admission ends in ordinary task creation and launch.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from rcp.core.models import AuthorizedHuman
from rcp.core.transition_models import GraphHeadRef, GraphTargetRef
from rcp.runs.auto_research import (
    AutoResearchRunRequest,
    AutoResearchStartRequest,
    AutoResearchWakeAdmission,
    PendingAutoResearchMail,
    auto_research_exhaustion_signal,
    auto_research_root_request,
    request_auto_research_stop,
    settle_auto_research_stop,
)
from rcp.runs.auto_research import (
    pending_auto_research_mail as _episode_pending_mail,
)
from rcp.runs.experiment_admission import experiment_start_message
from rcp.runs.task_policy import AgentTaskContinuation, resolved_dispatch_authority, skill_update
from rcp.service import RunRequest
from rcp.skill_registry import SkillSelection
from rcp.storage import (
    AgentTaskRecord,
    AutoResearchChildExperimentRecord,
    AutoResearchChildWorkRecord,
    AutoResearchStateRecord,
    EpisodeInvocationCeilingReached,
    EpisodeNotRunning,
    EpisodeRecord,
)
from rcp.transport import RemoteRunStage

if TYPE_CHECKING:
    from rcp.background import BackgroundAgentTasks


@dataclass(frozen=True)
class _CommittedAutoResearchDispatch:
    kind: Literal["actor_wake", "child_work", "child_experiment"]
    episode_id: str
    operation_id: str
    child_id: str | None = None
    continuation: Literal["fresh", "resume", "graph_repair", "message_wake", "watcher_wake"] = (
        "fresh"
    )


@dataclass(frozen=True)
class AutoResearchChildResumeResult:
    """Outcome of an orchestrator's exact child-allocation Resume request."""

    disposition: Literal["resumed", "resume_unavailable"]
    child_kind: Literal["work", "experiment"]
    child_id: str
    current_operation_id: str
    task: AgentTaskRecord | None = None
    reason: str | None = None
    replacement_command: Literal["spawn", "episode --kick-off-experiment"] | None = None


def start_auto_research(
    tasks: BackgroundAgentTasks,
    project_id: str,
    request: AutoResearchStartRequest,
    *,
    authorized_by: AuthorizedHuman,
    graph_base_head: GraphHeadRef,
    ensure_graph_target: Callable[[EpisodeRecord], None],
    episode_id: str | None = None,
    operation_id: str | None = None,
) -> tuple[EpisodeRecord, AgentTaskRecord]:
    """Reserve SQLite identity, establish its branch, then spend the first invocation."""

    episode, task, run_request = reserve_auto_research(
        tasks,
        project_id,
        request,
        authorized_by=authorized_by,
        graph_base_head=graph_base_head,
        episode_id=episode_id,
        operation_id=operation_id,
    )
    try:
        ensure_graph_target(episode)
    except Exception as exc:
        _fail_reserved_auto_research_root(tasks, episode, task, exc)
        raise
    episode = tasks.store.activate_auto_research_reservation(
        episode.episode_id,
        task.operation_id,
    )
    return episode, tasks.launch_admitted(task.operation_id)


def reserve_auto_research(
    tasks: BackgroundAgentTasks,
    project_id: str,
    request: AutoResearchStartRequest,
    *,
    authorized_by: AuthorizedHuman,
    graph_base_head: GraphHeadRef,
    episode_id: str | None = None,
    operation_id: str | None = None,
) -> tuple[EpisodeRecord, AgentTaskRecord, AutoResearchRunRequest]:
    """Atomically reserve one episode/root before any canonical branch publication."""

    if not authorized_by.display_name.strip():
        raise ValueError("Auto-research requires a named human authorizer snapshot.")
    episode_id = episode_id or str(uuid.uuid4())
    if graph_base_head.target.kind != "main":
        raise ValueError("Auto-research must branch from an exact main graph head.")
    graph_target = GraphTargetRef(kind="branch", branch_id=episode_id)
    operation_id = operation_id or str(uuid.uuid4())
    run_request = auto_research_root_request(request, episode_id=episode_id).model_copy(
        update={"actor_operation_id": operation_id}
    )
    dispatch_authority = resolved_dispatch_authority(
        tasks.store,
        tasks.dispatch_authority_resolver,
        "auto_research",
        run_request,
        project_id=project_id,
        operation_id=operation_id,
    )
    assert dispatch_authority is not None
    request_data = run_request.model_dump(mode="json")
    estimate, samples = tasks.store.agent_task_estimate(
        project_id,
        "auto_research",
        request_data,
    )
    now = tasks.store.now()
    episode = EpisodeRecord(
        episode_id=episode_id,
        project_id=project_id,
        mode="auto_research",
        graph_target=graph_target,
        graph_base_head=graph_base_head,
        status="queued",
        invocation_ceiling=request.invocation_ceiling,
        authorized_by=authorized_by,
        created_at=now,
        updated_at=now,
    )
    task = AgentTaskRecord(
        operation_id=operation_id,
        project_id=project_id,
        episode_id=episode_id,
        graph_target=graph_target,
        kind="auto_research",
        status="queued",
        request=request_data,
        created_at=now,
        updated_at=now,
        status_message="Waiting for the Auto-research orchestrator to start.",
        estimate_seconds=estimate,
        estimate_samples=samples,
        phase="queued",
        last_activity_at=now,
        authorized_by=authorized_by,
        dispatch_authority=dispatch_authority,
    )
    stored_episode, stored_task = tasks.store.create_auto_research_episode_with_root_task(
        episode,
        AutoResearchStateRecord(
            episode_id=episode_id,
            starting_instruction=request.starting_instruction,
            created_at=now,
            updated_at=now,
        ),
        task,
        activate=False,
    )
    return stored_episode, stored_task, run_request


def reconcile_reserved_auto_research_roots(
    tasks: BackgroundAgentTasks,
    ensure_graph_target: Callable[[EpisodeRecord], None],
) -> list[str]:
    """Finish branch creation and launch roots reserved before a process interruption."""

    started: list[str] = []
    for episode, task, _request in proven_reserved_auto_research_roots(tasks):
        try:
            ensure_graph_target(episode)
        except Exception as exc:
            _fail_reserved_auto_research_root(tasks, episode, task, exc)
            continue
        tasks.store.activate_auto_research_reservation(
            episode.episode_id,
            task.operation_id,
        )
        tasks.launch_admitted(task.operation_id)
        started.append(task.operation_id)
    return started


def proven_reserved_auto_research_roots(
    tasks: BackgroundAgentTasks,
) -> list[tuple[EpisodeRecord, AgentTaskRecord, AutoResearchRunRequest]]:
    reserved: list[tuple[EpisodeRecord, AgentTaskRecord, AutoResearchRunRequest]] = []
    for project in tasks.store.projects():
        for episode in tasks.store.episodes(project.project_id):
            if (
                episode.mode != "auto_research"
                or episode.root_operation_id is None
                or episode.status not in {"queued", "running"}
            ):
                continue
            task = tasks.store.agent_task(episode.root_operation_id)
            if (
                task is None
                or task.kind != "auto_research"
                or task.episode_id != episode.episode_id
                or task.project_id != episode.project_id
                or task.graph_target != episode.graph_target
                or task.parent_operation_id is not None
                or task.status != "queued"
                or not tasks.store.agent_task_dispatch_was_proven_not_started(task.operation_id)
            ):
                continue
            try:
                request = AutoResearchRunRequest.model_validate(task.request)
            except ValueError:
                continue
            if (
                request.episode_id != episode.episode_id
                or request.role != "orchestrator"
                or request.actor_operation_id != task.operation_id
                or request.wake_cause is not None
            ):
                continue
            reserved.append((episode, task, request))
    return reserved


def _fail_reserved_auto_research_root(
    tasks: BackgroundAgentTasks,
    episode: EpisodeRecord,
    task: AgentTaskRecord,
    error: Exception,
) -> None:
    diagnostic = (
        f"Auto-research could not establish its exact graph branch before provider launch: {error}"
    )
    tasks.store.fail_agent_task(task.operation_id, diagnostic)
    from rcp.runs.episodes.wrapup import EpisodeWrapupSpec, begin_episode_report_wrapup

    begin_episode_report_wrapup(
        tasks.store,
        EpisodeWrapupSpec(
            episode_id=episode.episode_id,
            ending="failed",
            partial=True,
            continuation_operation_id=task.operation_id,
            receipt={"reason": "graph_branch_unavailable_before_launch"},
            diagnostic=diagnostic,
        ),
    )


def start_auto_research_turn(
    tasks: BackgroundAgentTasks,
    episode_id: str,
    request: AutoResearchRunRequest,
    *,
    parent_operation_id: str | None = None,
    operation_id: str | None = None,
    mail_delivery: PendingAutoResearchMail | None = None,
    wake_admission: AutoResearchWakeAdmission | None = None,
) -> AgentTaskRecord | None:
    """Admit one operational actor turn from the episode invocation budget."""

    episode = auto_research_for_request(tasks, episode_id, request)
    operation_id = operation_id or str(uuid.uuid4())
    parent = _auto_research_parent(tasks, episode, parent_operation_id)
    parent_role = tasks.store.auto_research_invocation_role(parent.operation_id)
    if parent_role not in {"orchestrator", "worker"}:
        raise ValueError("Auto-research continuation parent has no canonical actor role.")
    parent_request = AutoResearchRunRequest.model_validate(parent.request)
    if parent_request.role != parent_role:
        raise ValueError("Auto-research continuation parent role disagrees with its durable actor.")
    parent_actor_id = parent_request.actor_operation_id or parent.operation_id
    requested_actor_id = request.actor_operation_id
    if request.wake_cause is not None:
        if request.role != parent_role:
            raise ValueError("An Auto-research wake cannot change its canonical actor role.")
        if requested_actor_id is not None and requested_actor_id != parent_actor_id:
            raise ValueError("An Auto-research wake cannot change its canonical actor identity.")
        if parent_role == "worker" and request.control_node_id != parent_request.control_node_id:
            raise ValueError("An Auto-research worker wake cannot change its canonical seat.")
        request = request.model_copy(update={"actor_operation_id": parent_actor_id})
    elif request.role == "worker":
        if parent_role != "orchestrator":
            raise ValueError("Only the Auto-research orchestrator may seat a worker.")
        if requested_actor_id is not None and requested_actor_id != operation_id:
            raise ValueError(
                "A new Auto-research worker must use its own canonical actor identity."
            )
        if request.session_id is not None:
            raise ValueError("A new Auto-research worker must start a fresh native session.")
        request = request.model_copy(update={"actor_operation_id": operation_id})
    else:
        if parent_role != "orchestrator" or parent_actor_id != episode.root_operation_id:
            raise ValueError("Only the root Auto-research actor may continue as orchestrator.")
        if requested_actor_id is not None and requested_actor_id != parent_actor_id:
            raise ValueError(
                "An Auto-research orchestrator turn cannot change its canonical actor identity."
            )
        request = request.model_copy(update={"actor_operation_id": parent_actor_id})

    authority_origin = tasks.store.agent_task(parent_actor_id)
    if authority_origin is None or authority_origin.dispatch_authority is None:
        raise ValueError(
            "Authority refused action 'dispatch': the canonical Auto-research actor has no "
            "durable authority binding."
        )
    canonical_scope = authority_origin.dispatch_authority.scope.run_truth_scope
    if request.run_truth_scope is not None and sorted(set(request.run_truth_scope)) != (
        canonical_scope
    ):
        raise ValueError(
            "Authority refused action 'dispatch': an Auto-research actor cannot change its "
            "project-wide run truth scope."
        )
    request = request.model_copy(update={"run_truth_scope": list(canonical_scope)})
    if episode.status != "running" or episode.stop_requested_at is not None:
        raise EpisodeNotRunning("the Auto-research episode is not admitting new work")
    if episode.invocations_used >= episode.invocation_ceiling:
        auto_research_admission_exhausted(tasks, episode)
        raise EpisodeInvocationCeilingReached(
            "the Auto-research operational invocation ceiling is exhausted"
        )

    existing_actor = request.actor_operation_id != operation_id
    stage_host: str | None = None
    stage_root: str | None = None
    if existing_actor:
        binding = tasks.store.auto_research_actor_binding(parent.operation_id)
        if (
            binding.actor_operation_id != request.actor_operation_id
            or binding.role != request.role
            or binding.control_node_id != request.control_node_id
        ):
            raise ValueError(
                "An Auto-research continuation must preserve its canonical actor role and seat."
            )
        if not binding.native_session_id or not binding.stage_root:
            raise ValueError(
                "An Auto-research continuation requires the actor's exact saved session and stage."
            )
        if request.session_id not in {None, binding.native_session_id}:
            raise ValueError(
                "An Auto-research continuation cannot change its saved native session."
            )
        request = request.model_copy(update={"session_id": binding.native_session_id})
        stage_host = binding.stage_host
        stage_root = binding.stage_root

    request_data = request.model_dump(mode="json")
    estimate, samples = tasks.store.agent_task_estimate(
        episode.project_id,
        "auto_research",
        request_data,
    )
    continuation: AgentTaskContinuation = {
        None: "fresh",
        "watcher": "watcher_wake",
        "graph_condition": "graph_condition_wake",
        "message": "message_wake",
        "lifecycle": "lifecycle_wake",
    }[request.wake_cause]
    if request.wake_cause is None and existing_actor:
        continuation = "auto_research_continuation"
    if request.wake_cause is not None:
        assert stage_root is not None
    if request.wake_cause == "message":
        if (
            mail_delivery is None
            or not mail_delivery.messages
            or mail_delivery.episode_id != episode_id
            or mail_delivery.recipient_task_id != parent_actor_id
        ):
            raise ValueError(
                "An Auto-research message wake requires its exact non-empty mail batch."
            )
        if wake_admission is not None:
            raise ValueError(
                "Auto-research message wake admission is owned by the mail transaction."
            )
    elif request.wake_cause in {"watcher", "graph_condition"}:
        if wake_admission is None:
            raise ValueError(
                "Auto-research watcher wakes require their atomic wake-admission hook."
            )
    elif request.wake_cause == "lifecycle":
        if parent_role != "orchestrator" or parent_actor_id != episode.root_operation_id:
            raise ValueError(
                "Auto-research lifecycle delivery may wake only the root orchestrator."
            )
        if mail_delivery is not None:
            raise ValueError(
                "Auto-research lifecycle mail is claimed by the lifecycle transaction."
            )
        if wake_admission is None:
            raise ValueError(
                "Auto-research lifecycle wakes require their atomic wake-admission hook."
            )
    elif mail_delivery is not None:
        raise ValueError("Only an Auto-research message wake may claim a mail batch.")
    elif wake_admission is not None:
        raise ValueError("Only an Auto-research watcher wake may use wake admission.")
    assert episode.authorized_by is not None
    return tasks._create_and_spawn(
        episode.project_id,
        "auto_research",
        request,
        parent=parent,
        continuation=continuation,
        stage_host=stage_host,
        stage_root=stage_root,
        estimate_seconds=estimate,
        estimate_samples=samples,
        operation_id=operation_id,
        authorized_by=episode.authorized_by,
        auto_research_mail_delivery=mail_delivery,
        auto_research_wake_admission=wake_admission,
    )


def ensure_auto_research_wake_spawned(
    tasks: BackgroundAgentTasks,
    episode_id: str,
    *,
    operation_id: str,
) -> AgentTaskRecord:
    """Dispatch one already-committed automatic actor wake exactly once in-process."""

    existing = tasks._require_operation(operation_id)
    request = tasks._request_from_record(existing)
    if not isinstance(request, AutoResearchRunRequest) or request.wake_cause is None:
        raise ValueError("The committed task is not an automatic Auto-research wake.")
    _validate_existing_auto_research_wake(
        tasks,
        episode_id,
        operation_id,
        existing,
        request,
    )
    return tasks.launch_admitted(existing.operation_id)


def reconcile_committed_auto_research_dispatches(tasks: BackgroundAgentTasks) -> list[str]:
    """Start exact paid child/wake rows durably proven never to have run."""

    started: list[str] = []
    for dispatch in proven_committed_auto_research_dispatches(tasks):
        if dispatch.kind == "actor_wake":
            task = ensure_auto_research_wake_spawned(
                tasks,
                dispatch.episode_id,
                operation_id=dispatch.operation_id,
            )
        elif dispatch.kind == "child_work":
            assert dispatch.child_id is not None
            task = ensure_auto_research_child_work_spawned(
                tasks,
                dispatch.episode_id,
                dispatch.child_id,
                operation_id=dispatch.operation_id,
                continuation=dispatch.continuation,
            )
        else:
            assert dispatch.child_id is not None
            task = ensure_auto_research_child_experiment_spawned(
                tasks,
                dispatch.episode_id,
                dispatch.child_id,
                operation_id=dispatch.operation_id,
                continuation=dispatch.continuation,
            )
        started.append(task.operation_id)
    return list(dict.fromkeys(started))


def proven_committed_auto_research_dispatches(
    tasks: BackgroundAgentTasks,
) -> list[_CommittedAutoResearchDispatch]:
    """Find admitted queued rows whose dispatch attempt durably never started."""

    dispatches: list[_CommittedAutoResearchDispatch] = []
    for project in tasks.store.projects():
        episodes = [
            item
            for item in tasks.store.episodes(project.project_id)
            if item.mode == "auto_research"
        ]
        for episode in episodes:
            for task in tasks.store.auto_research_tasks(episode.episode_id):
                if (
                    task.status != "queued"
                    or not tasks.store.agent_task_dispatch_was_proven_not_started(task.operation_id)
                ):
                    continue
                try:
                    request = tasks._request_from_record(task)
                except (TypeError, ValueError):
                    continue
                if not isinstance(request, AutoResearchRunRequest):
                    continue
                if request.wake_cause is None:
                    continue
                try:
                    _validate_existing_auto_research_wake(
                        tasks,
                        episode.episode_id,
                        task.operation_id,
                        task,
                        request,
                    )
                except (KeyError, ValueError):
                    continue
                dispatches.append(
                    _CommittedAutoResearchDispatch(
                        kind="actor_wake",
                        episode_id=episode.episode_id,
                        operation_id=task.operation_id,
                    )
                )
            delivered_operation_ids = {
                message.delivery_operation_id
                for message in tasks.store.auto_research_messages(episode.episode_id)
                if message.delivery_operation_id is not None
            }
            for route in tasks.store.auto_research_child_works(episode.episode_id):
                task = tasks.store.agent_task(route.current_operation_id)
                if (
                    task is None
                    or task.status != "queued"
                    or not tasks.store.agent_task_dispatch_was_proven_not_started(task.operation_id)
                ):
                    continue
                if route.root_operation_id == task.operation_id:
                    continuation: Literal["fresh", "resume", "message_wake"] = "fresh"
                    validator = _validate_existing_child_work_fresh
                elif task.operation_id in delivered_operation_ids:
                    continuation = "message_wake"
                    validator = _validate_existing_child_work_message_wake
                elif tasks.store.auto_research_child_resume_command_owns_operation(
                    episode.episode_id,
                    child_kind="work",
                    child_id=route.worker_id,
                    operation_id=task.operation_id,
                ):
                    continuation = "resume"
                    validator = _validate_existing_child_work_resume
                else:
                    continue
                try:
                    validator(
                        tasks,
                        episode.episode_id,
                        route.worker_id,
                        task.operation_id,
                        task,
                    )
                except (KeyError, ValueError):
                    continue
                dispatches.append(
                    _CommittedAutoResearchDispatch(
                        kind="child_work",
                        episode_id=episode.episode_id,
                        operation_id=task.operation_id,
                        child_id=route.worker_id,
                        continuation=continuation,
                    )
                )
            for route in tasks.store.auto_research_child_experiments(episode.episode_id):
                child_episode = tasks.store.episode(route.child_episode_id)
                if route.state != "running" or child_episode is None:
                    continue
                for task in tasks.store.episode_tasks(route.child_episode_id):
                    if (
                        task.status != "queued"
                        or not tasks.store.agent_task_dispatch_was_proven_not_started(
                            task.operation_id
                        )
                    ):
                        continue
                    try:
                        request = tasks._request_from_record(task)
                    except (TypeError, ValueError):
                        continue
                    if task.operation_id == child_episode.root_operation_id:
                        continuation = "fresh"
                        experiment_validator = _validate_existing_child_experiment_fresh
                    elif isinstance(request, RunRequest) and request.trigger == "watcher":
                        continuation = "watcher_wake"
                        experiment_validator = _validate_existing_child_experiment_watcher_wake
                    elif (
                        tasks.store.agent_task_continuation_cause(task.operation_id)
                        == "graph_repair"
                    ):
                        continuation = "graph_repair"
                        experiment_validator = _validate_existing_child_experiment_graph_repair
                    elif tasks.store.auto_research_child_resume_command_owns_operation(
                        episode.episode_id,
                        child_kind="experiment",
                        child_id=route.child_episode_id,
                        operation_id=task.operation_id,
                    ):
                        continuation = "resume"
                        experiment_validator = _validate_existing_child_experiment_resume
                    else:
                        continue
                    try:
                        experiment_validator(
                            tasks,
                            episode.episode_id,
                            route.child_episode_id,
                            task.operation_id,
                            task,
                        )
                    except (KeyError, ValueError):
                        continue
                    dispatches.append(
                        _CommittedAutoResearchDispatch(
                            kind="child_experiment",
                            episode_id=episode.episode_id,
                            operation_id=task.operation_id,
                            child_id=route.child_episode_id,
                            continuation=continuation,
                        )
                    )
    return dispatches


def start_auto_research_child_work(
    tasks: BackgroundAgentTasks,
    episode_id: str,
    request: RunRequest,
    *,
    admitted_by_operation_id: str,
    worker_id: str,
    instruction: str,
    instruction_sha256: str,
    admission_id: str | None = None,
) -> AgentTaskRecord:
    """Atomically spend B and launch one routed ordinary node Work task."""

    episode = _auto_research_parent_episode(tasks, episode_id)
    if (
        request.mode != "work"
        or request.trigger != "orchestrator"
        or request.patch_kind != "work"
        or request.chat_scope != "node"
        or not request.node_id
        or request.message != instruction
        or request.chat_id != worker_id
        or request.session_id is not None
        or request.result_view is not None
        or request.watcher_ids
    ):
        raise ValueError(
            "An Auto-research spawn must be a fresh ordinary node Work request with its "
            "exact snapshotted instruction and stable worker conversation id."
        )
    tasks._validate_request_type("node_chat", request)
    operation_id = worker_id
    request_data = request.model_dump(mode="json")
    estimate, samples = tasks.store.agent_task_estimate(
        episode.project_id,
        "node_chat",
        request_data,
    )
    dispatch_authority = resolved_dispatch_authority(
        tasks.store,
        tasks.dispatch_authority_resolver,
        "node_chat",
        request,
        project_id=episode.project_id,
        operation_id=operation_id,
    )
    assert episode.authorized_by is not None
    now = tasks.store.now()
    task = AgentTaskRecord(
        operation_id=operation_id,
        project_id=episode.project_id,
        episode_id=episode_id,
        graph_target=episode.graph_target,
        kind="node_chat",
        status="queued",
        request=request_data,
        created_at=now,
        updated_at=now,
        status_message="Waiting for the spawned Work task to start.",
        estimate_seconds=estimate,
        estimate_samples=samples,
        phase="queued",
        last_activity_at=now,
        authorized_by=episode.authorized_by,
        dispatch_authority=dispatch_authority,
    )
    route = AutoResearchChildWorkRecord(
        worker_id=worker_id,
        episode_id=episode_id,
        project_id=episode.project_id,
        control_node_id=request.node_id,
        root_operation_id=operation_id,
        current_operation_id=operation_id,
        admitted_by_operation_id=admitted_by_operation_id,
        instruction=instruction,
        instruction_sha256=instruction_sha256,
        created_at=now,
        updated_at=now,
    )
    _, stored = tasks.store.create_auto_research_child_work(
        route,
        task,
        admission_id=admission_id,
    )
    return ensure_auto_research_child_work_spawned(
        tasks,
        episode_id,
        worker_id,
        operation_id=stored.operation_id,
        continuation="fresh",
    )


def ensure_auto_research_child_work_spawned(
    tasks: BackgroundAgentTasks,
    episode_id: str,
    worker_id: str,
    *,
    operation_id: str,
    continuation: Literal["fresh", "resume", "message_wake"],
) -> AgentTaskRecord:
    """Dispatch one already-committed child Work row exactly once in this process.

    Child admission and process launch are deliberately separate durability
    boundaries.  A command replay that finds the first boundary committed
    must therefore repair the second rather than merely return the queued
    row.  The in-process worker registry is the dispatch claim; terminal or
    already-live tasks are returned without another launch.
    """

    existing = tasks._require_operation(operation_id)
    request = tasks._request_from_record(existing)
    if not isinstance(request, RunRequest):
        raise ValueError("The routed child Work task lost its ordinary Work request.")
    if continuation == "fresh":
        _validate_existing_child_work_fresh(
            tasks,
            episode_id,
            worker_id,
            operation_id,
            existing,
        )
    elif continuation == "resume":
        _validate_existing_child_work_resume(
            tasks,
            episode_id,
            worker_id,
            operation_id,
            existing,
        )
    else:
        _validate_existing_child_work_message_wake(
            tasks,
            episode_id,
            worker_id,
            operation_id,
            existing,
        )
    return tasks.launch_admitted(existing.operation_id)


def auto_research_child_work_task(
    tasks: BackgroundAgentTasks,
    episode_id: str,
    worker_id: str,
) -> tuple[AutoResearchChildWorkRecord, AgentTaskRecord]:
    """Resolve a routed worker and exactly its current attempt."""

    route = tasks.store.auto_research_child_work(worker_id)
    if route is None:
        raise KeyError(worker_id)
    if route.episode_id != episode_id:
        raise ValueError("The worker is not registered to this Auto-research episode.")
    current = tasks._require_operation(route.current_operation_id)
    if (
        current.project_id != route.project_id
        or current.episode_id != route.episode_id
        or current.kind != "node_chat"
    ):
        raise ValueError("The worker route lost its ordinary Work task lineage.")
    return route, current


def start_auto_research_child_work_message_wake(
    tasks: BackgroundAgentTasks,
    episode_id: str,
    worker_id: str,
    message_ids: list[str],
    *,
    operation_id: str | None = None,
    created_at: str | None = None,
) -> AgentTaskRecord | None:
    """Spend B to deliver mail through the routed Work task's exact saved session."""

    episode = _auto_research_parent_episode(tasks, episode_id)
    _, current = auto_research_child_work_task(tasks, episode_id, worker_id)
    if current.status != "succeeded":
        return None
    if not current.native_session_id or not current.stage_root:
        return None
    request = tasks._request_from_record(current)
    if not isinstance(request, RunRequest):
        raise ValueError("The routed worker task is not an ordinary Work request.")
    request = request.model_copy(
        update={
            "session_id": current.native_session_id,
            "message": None,
            "watcher_ids": [],
            "result_view": None,
        }
    )
    operation_id = operation_id or str(uuid.uuid4())
    dispatch_authority = resolved_dispatch_authority(
        tasks.store,
        tasks.dispatch_authority_resolver,
        "node_chat",
        request,
        project_id=current.project_id,
        parent=current,
        operation_id=operation_id,
        continuation="message_wake",
    )
    request_data = request.model_dump(mode="json")
    estimate, samples = tasks.store.agent_task_estimate(
        current.project_id,
        "node_chat",
        request_data,
    )
    assert episode.authorized_by is not None
    now = created_at or tasks.store.now()
    task = AgentTaskRecord(
        operation_id=operation_id,
        project_id=current.project_id,
        episode_id=episode_id,
        graph_target=episode.graph_target,
        kind="node_chat",
        status="queued",
        request=request_data,
        created_at=now,
        updated_at=now,
        status_message="Waiting for the spawned Work task to receive its message batch.",
        attempt=current.attempt + 1,
        parent_operation_id=current.operation_id,
        native_session_id=current.native_session_id,
        stage_host=current.stage_host,
        stage_root=current.stage_root,
        estimate_seconds=estimate,
        estimate_samples=samples,
        phase="queued",
        last_activity_at=now,
        authorized_by=episode.authorized_by,
        dispatch_authority=dispatch_authority,
    )
    stored = tasks.store.create_auto_research_child_work_message_wake_task(
        task,
        worker_id=worker_id,
        message_ids=message_ids,
    )
    if stored is None:
        return None
    return ensure_auto_research_child_work_spawned(
        tasks,
        episode_id,
        worker_id,
        operation_id=stored.operation_id,
        continuation="message_wake",
    )


def pause_auto_research_child_work(
    tasks: BackgroundAgentTasks,
    episode_id: str,
    worker_id: str,
) -> AgentTaskRecord:
    """Gracefully pause the current attempt of one routed Work child."""

    route, current = auto_research_child_work_task(tasks, episode_id, worker_id)
    if current.status == "paused":
        return current
    if current.status == "pausing":
        tasks._signal_agent_task_pause(current.operation_id)
        return current
    if route.stop_requested_at is not None:
        raise ValueError("The worker has already been stopped and cannot be paused.")
    return tasks.pause(current.operation_id)


def stop_auto_research_child_work(
    tasks: BackgroundAgentTasks,
    episode_id: str,
    worker_id: str,
) -> AgentTaskRecord:
    """Durably stop one route and gracefully pause its current live attempt."""

    route, current = auto_research_child_work_task(tasks, episode_id, worker_id)
    tasks.store.request_auto_research_child_work_stop(route.worker_id)
    current = tasks._require_operation(current.operation_id)
    if current.status in {"queued", "running"}:
        return tasks.pause(current.operation_id)
    if current.status == "pausing":
        tasks._signal_agent_task_pause(current.operation_id)
    return current


def resume_auto_research_child_work(
    tasks: BackgroundAgentTasks,
    episode_id: str,
    worker_id: str,
    *,
    operation_id: str | None = None,
) -> AutoResearchChildResumeResult:
    """Resume a routed Work attempt only from its exact usable checkpoint."""

    route, previous = auto_research_child_work_task(tasks, episode_id, worker_id)
    if operation_id is not None:
        existing = tasks.store.agent_task(operation_id)
        if existing is not None:
            existing = ensure_auto_research_child_work_spawned(
                tasks,
                episode_id,
                worker_id,
                operation_id=operation_id,
                continuation="resume",
            )
            return AutoResearchChildResumeResult(
                disposition="resumed",
                child_kind="work",
                child_id=worker_id,
                current_operation_id=existing.operation_id,
                task=existing,
            )
    problem = (
        "the worker was stopped"
        if route.stop_requested_at is not None
        else _exact_child_resume_problem(tasks, previous)
    )
    if problem is not None:
        return AutoResearchChildResumeResult(
            disposition="resume_unavailable",
            child_kind="work",
            child_id=worker_id,
            current_operation_id=previous.operation_id,
            reason=problem,
            replacement_command="spawn",
        )
    episode = _auto_research_parent_episode(tasks, episode_id)
    assert previous.native_session_id is not None
    assert previous.stage_root is not None
    request = tasks._request_from_record(previous)
    if not isinstance(request, RunRequest):
        raise ValueError("The routed worker task is not an ordinary Work request.")
    request = request.model_copy(update={"session_id": previous.native_session_id})
    operation_id = operation_id or str(uuid.uuid4())
    dispatch_authority = resolved_dispatch_authority(
        tasks.store,
        tasks.dispatch_authority_resolver,
        "node_chat",
        request,
        project_id=previous.project_id,
        parent=previous,
        operation_id=operation_id,
        continuation="resume",
    )
    assert episode.authorized_by is not None
    now = tasks.store.now()
    task = AgentTaskRecord(
        operation_id=operation_id,
        project_id=previous.project_id,
        episode_id=episode_id,
        graph_target=episode.graph_target,
        kind="node_chat",
        status="queued",
        request=request.model_dump(mode="json"),
        created_at=now,
        updated_at=now,
        status_message="Waiting for the spawned Work task to resume.",
        attempt=previous.attempt + 1,
        parent_operation_id=previous.operation_id,
        native_session_id=previous.native_session_id,
        stage_host=previous.stage_host,
        stage_root=previous.stage_root,
        estimate_seconds=previous.estimate_seconds,
        estimate_samples=previous.estimate_samples,
        phase="queued",
        last_activity_at=now,
        authorized_by=episode.authorized_by,
        dispatch_authority=dispatch_authority,
    )
    try:
        _, stored = tasks.store.create_auto_research_child_work_recovery(worker_id, task)
    except ValueError:
        existing = tasks.store.agent_task(operation_id)
        if existing is None:
            raise
        _validate_existing_child_work_resume(
            tasks,
            episode_id,
            worker_id,
            operation_id,
            existing,
        )
        existing = ensure_auto_research_child_work_spawned(
            tasks,
            episode_id,
            worker_id,
            operation_id=operation_id,
            continuation="resume",
        )
        return AutoResearchChildResumeResult(
            disposition="resumed",
            child_kind="work",
            child_id=worker_id,
            current_operation_id=existing.operation_id,
            task=existing,
        )
    spawned = ensure_auto_research_child_work_spawned(
        tasks,
        episode_id,
        worker_id,
        operation_id=stored.operation_id,
        continuation="resume",
    )
    return AutoResearchChildResumeResult(
        disposition="resumed",
        child_kind="work",
        child_id=worker_id,
        current_operation_id=spawned.operation_id,
        task=spawned,
    )


def start_auto_research_child_experiment(
    tasks: BackgroundAgentTasks,
    route: AutoResearchChildExperimentRecord,
    request: RunRequest,
    *,
    admission_id: str | None = None,
) -> AgentTaskRecord:
    """Launch invocation one through the ordinary Experiment task stream."""

    parent = _auto_research_parent_episode(tasks, route.auto_research_episode_id)
    if set(route.request) != {"goal", "invocation_limit"}:
        raise ValueError("The child Experiment route has an invalid launch intent.")
    goal = route.request["goal"]
    invocation_limit = route.request["invocation_limit"]
    if goal is not None and (not isinstance(goal, str) or not goal.strip()):
        raise ValueError("The child Experiment route has an invalid goal snapshot.")
    if invocation_limit is not None and (
        not isinstance(invocation_limit, int)
        or isinstance(invocation_limit, bool)
        or invocation_limit < 1
    ):
        raise ValueError("The child Experiment route has an invalid invocation limit.")
    expected_goal_sha256 = (
        hashlib.sha256(goal.encode("utf-8")).hexdigest() if isinstance(goal, str) else None
    )
    if (
        route.state != "running"
        or route.project_id != parent.project_id
        or request.mode != "work"
        or request.trigger != "orchestrator"
        or request.patch_kind != "experiment_loop"
        or request.chat_scope != "node"
        or request.node_id != route.control_node_id
        or request.control_node_id != route.control_node_id
        or request.control_episode_id != route.child_episode_id
        or request.control_invocation != 1
        or request.message != experiment_start_message(goal, route.control_node_id)
        or (invocation_limit is not None and request.control_invocation_ceiling != invocation_limit)
        or request.session_id is not None
        or request.watcher_ids
    ):
        raise ValueError(
            "An Auto-research child Experiment must be its routed fresh invocation one."
        )
    if route.goal_sha256 != expected_goal_sha256:
        raise ValueError("The child Experiment goal changed after command admission.")
    operation_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rcp:auto-research-child-experiment:{route.child_episode_id}",
        )
    )
    dispatch_authority = resolved_dispatch_authority(
        tasks.store,
        tasks.dispatch_authority_resolver,
        "node_chat",
        request,
        project_id=route.project_id,
        operation_id=operation_id,
    )
    request_data = request.model_dump(mode="json")
    estimate, samples = tasks.store.agent_task_estimate(
        route.project_id,
        "node_chat",
        request_data,
    )
    assert parent.authorized_by is not None
    now = tasks.store.now()
    task = AgentTaskRecord(
        operation_id=operation_id,
        project_id=route.project_id,
        episode_id=route.child_episode_id,
        graph_target=parent.graph_target,
        kind="node_chat",
        status="queued",
        request=request_data,
        created_at=now,
        updated_at=now,
        status_message="Waiting for the bounded Experiment to start.",
        estimate_seconds=estimate,
        estimate_samples=samples,
        phase="queued",
        last_activity_at=now,
        authorized_by=parent.authorized_by,
        dispatch_authority=dispatch_authority,
    )
    stored = tasks.store.create_experiment_episode_with_invocation(
        task,
        request.watcher_ids,
        auto_research_route=route,
        auto_research_admission_id=admission_id,
    )
    return ensure_auto_research_child_experiment_spawned(
        tasks,
        route.auto_research_episode_id,
        route.child_episode_id,
        operation_id=stored.operation_id,
        continuation="fresh",
    )


def ensure_auto_research_child_experiment_spawned(
    tasks: BackgroundAgentTasks,
    parent_episode_id: str,
    child_episode_id: str,
    *,
    operation_id: str,
    continuation: Literal["fresh", "resume", "graph_repair", "watcher_wake"],
) -> AgentTaskRecord:
    """Dispatch one committed child Experiment attempt exactly once in-process."""

    existing = tasks._require_operation(operation_id)
    request = tasks._request_from_record(existing)
    if not isinstance(request, RunRequest) or request.patch_kind != "experiment_loop":
        raise ValueError("The child Experiment task lost its Work request contract.")
    if continuation == "fresh":
        _validate_existing_child_experiment_fresh(
            tasks,
            parent_episode_id,
            child_episode_id,
            operation_id,
            existing,
        )
    elif continuation == "resume":
        _validate_existing_child_experiment_resume(
            tasks,
            parent_episode_id,
            child_episode_id,
            operation_id,
            existing,
        )
    elif continuation == "graph_repair":
        _validate_existing_child_experiment_graph_repair(
            tasks,
            parent_episode_id,
            child_episode_id,
            operation_id,
            existing,
        )
    else:
        _validate_existing_child_experiment_watcher_wake(
            tasks,
            parent_episode_id,
            child_episode_id,
            operation_id,
            existing,
        )
    return tasks.launch_admitted(existing.operation_id)


def resume_auto_research_child_experiment(
    tasks: BackgroundAgentTasks,
    parent_episode_id: str,
    child_episode_id: str,
    *,
    operation_id: str | None = None,
) -> AutoResearchChildResumeResult:
    """Resume the newest child Experiment attempt without spending E again."""

    route = tasks.store.auto_research_child_experiment(child_episode_id)
    if route is None:
        raise KeyError(child_episode_id)
    if route.auto_research_episode_id != parent_episode_id:
        raise ValueError("The Experiment is not registered to this Auto-research episode.")
    if operation_id is not None:
        existing = tasks.store.agent_task(operation_id)
        if existing is not None:
            existing = ensure_auto_research_child_experiment_spawned(
                tasks,
                parent_episode_id,
                child_episode_id,
                operation_id=operation_id,
                continuation="resume",
            )
            return AutoResearchChildResumeResult(
                disposition="resumed",
                child_kind="experiment",
                child_id=child_episode_id,
                current_operation_id=existing.operation_id,
                task=existing,
            )
    episode_tasks = tasks.store.episode_tasks(child_episode_id)
    if not episode_tasks:
        raise ValueError("The child Experiment route has no task to resume.")
    previous = episode_tasks[-1]
    problem = (
        f"the child Experiment route is {route.state}"
        if route.state != "running"
        else _exact_child_resume_problem(tasks, previous)
    )
    if problem is None:
        problem = tasks.store.experiment_episode_recovery_context_problem(previous.operation_id)
    if problem is not None:
        return AutoResearchChildResumeResult(
            disposition="resume_unavailable",
            child_kind="experiment",
            child_id=child_episode_id,
            current_operation_id=previous.operation_id,
            reason=problem,
            replacement_command="episode --kick-off-experiment",
        )
    assert previous.native_session_id is not None
    request = tasks._request_from_record(previous)
    if not isinstance(request, RunRequest) or request.patch_kind != "experiment_loop":
        raise ValueError("The child Experiment task lost its Work request contract.")
    request = request.model_copy(update={"session_id": previous.native_session_id})
    operation_id = operation_id or str(uuid.uuid4())
    try:
        resumed = tasks._create_and_spawn(
            previous.project_id,
            previous.kind,
            request,
            parent=previous,
            continuation="resume",
            estimate_seconds=previous.estimate_seconds,
            estimate_samples=previous.estimate_samples,
            stage_host=previous.stage_host,
            stage_root=previous.stage_root,
            operation_id=operation_id,
        )
    except ValueError:
        existing = tasks.store.agent_task(operation_id)
        if existing is None:
            raise
        _validate_existing_child_experiment_resume(
            tasks,
            parent_episode_id,
            child_episode_id,
            operation_id,
            existing,
        )
        resumed = ensure_auto_research_child_experiment_spawned(
            tasks,
            parent_episode_id,
            child_episode_id,
            operation_id=operation_id,
            continuation="resume",
        )
    assert resumed is not None
    return AutoResearchChildResumeResult(
        disposition="resumed",
        child_kind="experiment",
        child_id=child_episode_id,
        current_operation_id=resumed.operation_id,
        task=resumed,
    )


def stop_auto_research(tasks: BackgroundAgentTasks, episode_id: str) -> EpisodeRecord:
    """Persist Stop without cancelling the already-authorized actor turn."""

    before = tasks.store.episode(episode_id)
    if before is None or before.mode != "auto_research":
        raise KeyError(episode_id)
    stopped = request_auto_research_stop(tasks.store, episode_id)
    if (
        before.stop_requested_at is None
        and stopped.stop_requested_at is not None
        and stopped.root_operation_id is not None
    ):
        tasks.store.record_agent_task_event(
            stopped.root_operation_id,
            "Auto-research Stop requested; current turns will finish and no new work will start.",
        )
    return settle_auto_research_stop(tasks.store, episode_id) or stopped


def pending_auto_research_mail(
    tasks: BackgroundAgentTasks,
    *,
    episode_id: str,
    recipient_task_id: str,
) -> PendingAutoResearchMail:
    return _episode_pending_mail(
        tasks.store,
        episode_id=episode_id,
        recipient_task_id=recipient_task_id,
    )


def retry_auto_research_task(
    tasks: BackgroundAgentTasks,
    previous: AgentTaskRecord,
    original: AutoResearchRunRequest,
    *,
    provider: str | None,
    model: str | None,
    reasoning: str | None,
    run_on: str | None,
    skills: SkillSelection | None,
) -> AgentTaskRecord:
    """Recover one paid Auto-research allocation without changing its binding."""

    requested = {
        "provider": provider,
        "model": model,
        "reasoning": reasoning,
        "run_on": run_on,
    }
    changed = [
        key
        for key, value in requested.items()
        if value is not None and value != getattr(original, key)
    ]
    if changed:
        raise ValueError(
            "Auto-research recovery cannot change its pinned " + ", ".join(changed) + "."
        )
    session_limit = tasks._failure_is_session_limit(previous)
    continuation_unavailable = tasks._continuation_context_is_unavailable(previous)
    owned_checkpoint = bool(
        previous.native_session_id and previous.stage_root and tasks._session_is_rcp_owned(previous)
    )
    clean_orchestrator_retry = original.role == "orchestrator" and (
        session_limit or continuation_unavailable or not owned_checkpoint
    )
    problem: str | None = None
    if previous.stage_root:
        if previous.stage_host:
            if (
                RemoteRunStage(previous.stage_host).directory_exists(previous.stage_root)
                is not True
            ):
                problem = "the saved provider workspace is unavailable"
        else:
            stage = Path(previous.stage_root)
            if not stage.is_dir() or stage.is_symlink():
                problem = "the saved provider workspace is unavailable"
    elif not clean_orchestrator_retry:
        problem = "the prior task has no complete RCP-owned session and stage"
    if not clean_orchestrator_retry:
        if session_limit:
            problem = "the native provider session reached its limit"
        elif continuation_unavailable:
            problem = "the saved continuation context is unavailable"
        elif not owned_checkpoint:
            problem = "the prior task has no complete RCP-owned session and stage"
    if problem is not None:
        episode = tasks.store.episode(original.episode_id)
        if episode is not None and episode.stop_requested_at is not None:
            tasks.store.abandon_auto_research_recovery(
                previous.operation_id,
                diagnostic=problem,
            )
            settle_auto_research_stop(tasks.store, episode.episode_id, diagnostic=problem)
        raise ValueError(
            "Auto-research recovery cannot start a fresh provider session because "
            f"{problem}. Its original allocation and operational history were preserved."
        )
    if clean_orchestrator_retry:
        session_id = None
        classification = (
            "session_limit"
            if session_limit
            else "continuation_unavailable"
            if continuation_unavailable
            else "checkpoint_missing"
        )
    else:
        assert previous.native_session_id is not None
        session_id = previous.native_session_id
        classification = None
    request = AutoResearchRunRequest.model_validate(
        {
            **original.model_dump(mode="json"),
            "session_id": session_id,
            **skill_update(skills, mode="json"),
        }
    )
    retried = tasks._create_and_spawn(
        previous.project_id,
        previous.kind,
        request,
        parent=previous,
        continuation="retry",
        estimate_seconds=previous.estimate_seconds,
        estimate_samples=previous.estimate_samples,
        stage_host=previous.stage_host,
        stage_root=previous.stage_root,
    )
    if classification is not None:
        tasks.store.record_agent_task_receipt(
            retried.operation_id,
            "auto_research_orchestrator_clean_retry",
            {
                "classification": classification,
                "same_allocation": True,
                "actor_operation_id": original.actor_operation_id,
                "retry_mode": "clean_native_session",
            },
            tier="summary",
        )
        tasks.store.record_agent_task_event(
            retried.operation_id,
            "The orchestrator is retrying this same paid allocation with a clean native "
            "session after its prior continuation became unavailable.",
            level="warning",
        )
    return retried


def pause_auto_research_worker(
    tasks: BackgroundAgentTasks,
    operation_id: str,
    episode_id: str,
) -> AgentTaskRecord:
    """Commit the episode gate before signalling the worker process."""

    record = tasks.store.request_auto_research_worker_pause(operation_id, episode_id)
    with tasks._controls_lock:
        control = tasks._controls.get(operation_id)
    if control is not None:
        control.request_pause()
    return record


def auto_research_for_request(
    tasks: BackgroundAgentTasks,
    episode_id: str,
    request: AutoResearchRunRequest,
) -> EpisodeRecord:
    if request.episode_id != episode_id:
        raise ValueError("Auto-research request does not match its episode lineage.")
    episode = tasks.store.episode(episode_id)
    if (
        episode is None
        or episode.mode != "auto_research"
        or tasks.store.auto_research_state(episode_id) is None
    ):
        raise KeyError(episode_id)
    if (
        episode.graph_target.kind != "branch"
        or episode.graph_target.branch_id != episode.episode_id
        or episode.graph_base_head is None
        or episode.graph_base_head.target.kind != "main"
    ):
        raise ValueError("The Auto-research episode has no exact canonical graph-branch binding.")
    return episode


def _auto_research_parent_episode(tasks: BackgroundAgentTasks, episode_id: str) -> EpisodeRecord:
    episode = tasks.store.episode(episode_id)
    if (
        episode is None
        or episode.mode != "auto_research"
        or tasks.store.auto_research_state(episode_id) is None
    ):
        raise KeyError(episode_id)
    if (
        episode.graph_target.kind != "branch"
        or episode.graph_target.branch_id != episode.episode_id
        or episode.graph_base_head is None
        or episode.graph_base_head.target.kind != "main"
    ):
        raise ValueError("The Auto-research episode has no exact canonical graph-branch binding.")
    if episode.authorized_by is None:
        raise ValueError("The Auto-research episode lost its human authorizer snapshot.")
    return episode


def _auto_research_parent(
    tasks: BackgroundAgentTasks,
    episode: EpisodeRecord,
    operation_id: str | None,
) -> AgentTaskRecord:
    parent_id = operation_id or episode.root_operation_id
    if parent_id is None:
        raise ValueError("Auto-research has no root operation for child lineage.")
    parent = tasks._require_operation(parent_id)
    if parent.project_id != episode.project_id or parent.episode_id != episode.episode_id:
        raise ValueError("Auto-research child parent is outside the episode lineage.")
    return parent


def auto_research_admission_exhausted(tasks: BackgroundAgentTasks, episode: EpisodeRecord) -> None:
    # Hitting the ceiling refuses the *next* paid invocation.  It must not
    # revoke authority from an invocation that was already admitted at the
    # ceiling: that turn may still finish, emit its final patch, or declare
    # completion.  Normal task settlement performs the terminal
    # completion/exhaustion choice once all admitted work is quiescent.
    if not tasks.store.auto_research_is_quiescent(episode.episode_id):
        return
    auto_research_exhaustion_signal(
        tasks.store,
        episode.episode_id,
        diagnostic="The Auto-research operational invocation ceiling was exhausted.",
    )
    current = tasks.store.episode(episode.episode_id)
    assert current is not None
    if current.root_operation_id is not None:
        with suppress(Exception):
            tasks.store.record_agent_task_event(
                current.root_operation_id,
                "Auto-research operational invocation ceiling exhausted.",
                level="warning",
            )
    if tasks.on_auto_research_admission_exhausted is not None:
        with suppress(Exception):
            tasks.on_auto_research_admission_exhausted(current)


def _validate_existing_auto_research_wake(
    tasks: BackgroundAgentTasks,
    episode_id: str,
    operation_id: str,
    existing: AgentTaskRecord,
    request: AutoResearchRunRequest,
) -> AgentTaskRecord:
    """Prove a queued wake retains its paid allocation and exact delivery binding."""

    episode = tasks.store.episode(episode_id)
    invocation = tasks.store.auto_research_invocation(operation_id)
    parent = (
        tasks.store.agent_task(existing.parent_operation_id)
        if existing.parent_operation_id is not None
        else None
    )
    parent_invocation = (
        tasks.store.auto_research_invocation(parent.operation_id) if parent is not None else None
    )
    messages = [
        item
        for item in tasks.store.auto_research_messages(episode_id)
        if item.delivery_operation_id == operation_id
    ]
    lifecycle = tasks.store.auto_research_lifecycle_delivery(operation_id)
    watchers = [
        item
        for item in tasks.store.watchers(existing.project_id)
        if item.notification_operation_id == operation_id
    ]
    actor_operation_id = request.actor_operation_id
    binding_is_exact = False
    if request.wake_cause == "message":
        binding_is_exact = bool(messages) and all(
            item.episode_id == episode_id and item.recipient_task_id == actor_operation_id
            for item in messages
        )
    elif request.wake_cause == "lifecycle":
        binding_is_exact = bool(lifecycle) and all(
            item.episode_id == episode_id for item in lifecycle
        )
        binding_is_exact = binding_is_exact and all(
            item.episode_id == episode_id and item.recipient_task_id == actor_operation_id
            for item in messages
        )
    elif request.wake_cause in {"watcher", "graph_condition"}:
        binding_is_exact = bool(watchers) and {item.watcher_id for item in watchers} == set(
            request.watcher_ids
        )
        binding_is_exact = binding_is_exact and all(
            item.episode_id == episode_id
            and item.origin_task_kind == "auto_research"
            and item.chat_id == actor_operation_id
            for item in watchers
        )
    if (
        episode is None
        or episode.mode != "auto_research"
        or existing.operation_id != operation_id
        or existing.project_id != episode.project_id
        or existing.episode_id != episode_id
        or existing.kind != "auto_research"
        or existing.parent_operation_id is None
        or existing.authorized_by != episode.authorized_by
        or parent is None
        or parent.project_id != existing.project_id
        or parent.episode_id != episode_id
        or parent.kind != "auto_research"
        or invocation is None
        or invocation.episode_id != episode_id
        or invocation.operation_id != operation_id
        or invocation.allocation_operation_id != operation_id
        or invocation.role != request.role
        or invocation.actor_operation_id != actor_operation_id
        or invocation.control_node_id != request.control_node_id
        or parent_invocation is None
        or parent_invocation.episode_id != episode_id
        or parent_invocation.role != invocation.role
        or parent_invocation.actor_operation_id != invocation.actor_operation_id
        or parent_invocation.control_node_id != invocation.control_node_id
        or not request.session_id
        or existing.native_session_id != request.session_id
        or existing.native_session_id != parent.native_session_id
        or not existing.stage_root
        or existing.stage_root != parent.stage_root
        or (existing.stage_host or "") != (parent.stage_host or "")
        or not binding_is_exact
    ):
        raise ValueError(
            "The committed Auto-research wake lost its allocation or delivery binding."
        )
    return parent


def _validate_existing_child_experiment_fresh(
    tasks: BackgroundAgentTasks,
    parent_episode_id: str,
    child_episode_id: str,
    operation_id: str,
    existing: AgentTaskRecord,
) -> None:
    """Prove a committed row is the child Experiment's immutable invocation one."""

    route = tasks.store.auto_research_child_experiment(child_episode_id)
    episode = tasks.store.episode(child_episode_id)
    request = tasks._request_from_record(existing)
    goal = route.request.get("goal") if route is not None else None
    invocation_limit = route.request.get("invocation_limit") if route is not None else None
    expected_goal_sha256 = (
        hashlib.sha256(goal.encode("utf-8")).hexdigest() if isinstance(goal, str) else None
    )
    if (
        route is None
        or route.auto_research_episode_id != parent_episode_id
        or route.child_episode_id != child_episode_id
        or route.state != "running"
        or episode is None
        or episode.mode != "experiment_loop"
        or episode.root_operation_id != operation_id
        or episode.project_id != route.project_id
        or episode.control_node_id != route.control_node_id
        or existing.operation_id != operation_id
        or existing.project_id != route.project_id
        or existing.episode_id != child_episode_id
        or existing.kind != "node_chat"
        or existing.parent_operation_id is not None
        or not isinstance(request, RunRequest)
        or request.mode != "work"
        or request.trigger != "orchestrator"
        or request.patch_kind != "experiment_loop"
        or request.chat_scope != "node"
        or request.node_id != route.control_node_id
        or request.control_node_id != route.control_node_id
        or request.control_episode_id != child_episode_id
        or request.control_invocation != 1
        or request.message != experiment_start_message(goal, route.control_node_id)
        or (invocation_limit is not None and request.control_invocation_ceiling != invocation_limit)
        or route.goal_sha256 != expected_goal_sha256
        or request.session_id is not None
    ):
        raise ValueError("The deterministic Experiment operation belongs to another fresh launch.")


def _validate_existing_child_experiment_resume(
    tasks: BackgroundAgentTasks,
    parent_episode_id: str,
    child_episode_id: str,
    operation_id: str,
    existing: AgentTaskRecord,
) -> None:
    """Prove a deterministic operation id is this child Experiment Resume."""

    route = tasks.store.auto_research_child_experiment(child_episode_id)
    parent = (
        tasks.store.agent_task(existing.parent_operation_id)
        if existing.parent_operation_id is not None
        else None
    )
    request = tasks._request_from_record(existing)
    if (
        route is None
        or route.auto_research_episode_id != parent_episode_id
        or route.child_episode_id != child_episode_id
        or existing.operation_id != operation_id
        or existing.project_id != route.project_id
        or existing.episode_id != child_episode_id
        or existing.kind != "node_chat"
        or parent is None
        or parent.project_id != existing.project_id
        or parent.episode_id != child_episode_id
        or existing.attempt != parent.attempt + 1
        or existing.native_session_id != parent.native_session_id
        or (existing.stage_host or "") != (parent.stage_host or "")
        or existing.stage_root != parent.stage_root
        or not isinstance(request, RunRequest)
        or request.patch_kind != "experiment_loop"
        or request.control_episode_id != child_episode_id
        or request.session_id != parent.native_session_id
    ):
        raise ValueError(
            "The deterministic Experiment Resume operation belongs to another recovery."
        )


def _validate_existing_child_experiment_graph_repair(
    tasks: BackgroundAgentTasks,
    parent_episode_id: str,
    child_episode_id: str,
    operation_id: str,
    existing: AgentTaskRecord,
) -> None:
    """Prove a committed child continuation is the same patch-only graph repair."""

    _validate_existing_child_experiment_resume(
        tasks,
        parent_episode_id,
        child_episode_id,
        operation_id,
        existing,
    )
    route = tasks.store.auto_research_child_experiment(child_episode_id)
    parent = tasks._require_operation(existing.parent_operation_id or "")
    request = tasks._request_from_record(existing)
    graph_update = parent.result.get("graph_update") if isinstance(parent.result, dict) else None
    if (
        route is None
        or route.state != "running"
        or tasks.store.agent_task_continuation_cause(operation_id) != "graph_repair"
        or not isinstance(request, RunRequest)
        or request.message is not None
        or parent.status != "succeeded"
        or not isinstance(graph_update, dict)
        or graph_update.get("status") != "rejected"
        or graph_update.get("repairable") is not False
    ):
        raise ValueError(
            "The committed Experiment graph repair lost its patch-only recovery binding."
        )


def _validate_existing_child_experiment_watcher_wake(
    tasks: BackgroundAgentTasks,
    parent_episode_id: str,
    child_episode_id: str,
    operation_id: str,
    existing: AgentTaskRecord,
) -> None:
    """Prove a paid child Experiment watcher allocation and exact session binding."""

    route = tasks.store.auto_research_child_experiment(child_episode_id)
    episode = tasks.store.episode(child_episode_id)
    request = tasks._request_from_record(existing)
    invocations = {
        item.operation_id: item.invocation_number
        for item in tasks.store.episode_invocations(child_episode_id)
    }
    watchers = [
        item
        for item in tasks.store.watchers(existing.project_id)
        if item.notification_operation_id == operation_id
    ]
    if (
        route is None
        or route.auto_research_episode_id != parent_episode_id
        or route.child_episode_id != child_episode_id
        or route.state != "running"
        or episode is None
        or episode.mode != "experiment_loop"
        or episode.project_id != route.project_id
        or episode.control_node_id != route.control_node_id
        or existing.operation_id != operation_id
        or existing.project_id != route.project_id
        or existing.episode_id != child_episode_id
        or existing.kind != "node_chat"
        or existing.parent_operation_id is not None
        or not isinstance(request, RunRequest)
        or request.mode != "work"
        or request.trigger != "watcher"
        or request.patch_kind != "experiment_loop"
        or request.control_node_id != route.control_node_id
        or request.control_episode_id != child_episode_id
        or request.control_invocation != invocations.get(operation_id)
        or not request.session_id
        or request.session_id != existing.native_session_id
        or not existing.stage_root
        or set(request.watcher_ids) != {item.watcher_id for item in watchers}
        or not watchers
        or any(
            item.project_id != route.project_id
            or item.origin_task_kind != "node_chat"
            or item.chat_id != request.chat_id
            or item.continuation.control_episode_id != child_episode_id
            for item in watchers
        )
    ):
        raise ValueError(
            "The committed child Experiment watcher wake lost its allocation, watcher, "
            "or native-session binding."
        )


def _validate_existing_child_work_fresh(
    tasks: BackgroundAgentTasks,
    episode_id: str,
    worker_id: str,
    operation_id: str,
    existing: AgentTaskRecord,
) -> None:
    """Prove a committed row is the routed worker's immutable fresh launch."""

    route = tasks.store.auto_research_child_work(worker_id)
    request = tasks._request_from_record(existing)
    if (
        route is None
        or route.episode_id != episode_id
        or route.worker_id != worker_id
        or route.root_operation_id != operation_id
        or existing.operation_id != operation_id
        or existing.project_id != route.project_id
        or existing.episode_id != episode_id
        or existing.kind != "node_chat"
        or existing.parent_operation_id is not None
        or not isinstance(request, RunRequest)
        or request.mode != "work"
        or request.trigger != "orchestrator"
        or request.patch_kind != "work"
        or request.chat_scope != "node"
        or request.chat_id != worker_id
        or request.node_id != route.control_node_id
        or request.message != route.instruction
        or hashlib.sha256(route.instruction.encode("utf-8")).hexdigest() != route.instruction_sha256
        or request.session_id is not None
    ):
        raise ValueError("The deterministic worker operation belongs to another fresh launch.")


def _validate_existing_child_work_resume(
    tasks: BackgroundAgentTasks,
    episode_id: str,
    worker_id: str,
    operation_id: str,
    existing: AgentTaskRecord,
) -> None:
    """Prove a deterministic operation id is this worker's exact Resume."""

    route = tasks.store.auto_research_child_work_for_operation(operation_id)
    parent = (
        tasks.store.agent_task(existing.parent_operation_id)
        if existing.parent_operation_id is not None
        else None
    )
    request = tasks._request_from_record(existing)
    if (
        route is None
        or route.worker_id != worker_id
        or route.episode_id != episode_id
        or existing.operation_id != operation_id
        or existing.project_id != route.project_id
        or existing.episode_id != episode_id
        or existing.kind != "node_chat"
        or parent is None
        or parent.project_id != existing.project_id
        or parent.episode_id != episode_id
        or existing.attempt != parent.attempt + 1
        or existing.native_session_id != parent.native_session_id
        or (existing.stage_host or "") != (parent.stage_host or "")
        or existing.stage_root != parent.stage_root
        or not isinstance(request, RunRequest)
        or request.session_id != parent.native_session_id
        or request.chat_id != parent.request.get("chat_id")
    ):
        raise ValueError("The deterministic worker Resume operation belongs to another recovery.")


def _validate_existing_child_work_message_wake(
    tasks: BackgroundAgentTasks,
    episode_id: str,
    worker_id: str,
    operation_id: str,
    existing: AgentTaskRecord,
) -> None:
    """Prove one queued ordinary Work continuation owns its exact claimed mail."""

    route = tasks.store.auto_research_child_work_for_operation(operation_id)
    parent = (
        tasks.store.agent_task(existing.parent_operation_id)
        if existing.parent_operation_id is not None
        else None
    )
    request = tasks._request_from_record(existing)
    messages = [
        item
        for item in tasks.store.auto_research_messages(episode_id)
        if item.delivery_operation_id == operation_id
    ]
    episode_invocations = {
        item.operation_id for item in tasks.store.episode_invocations(episode_id)
    }
    pinned_request_fields = (
        "provider",
        "model",
        "reasoning",
        "run_on",
        "run_truth_scope",
        "chat_scope",
        "node_id",
        "chat_id",
        "mode",
        "patch_kind",
    )
    if (
        route is None
        or route.worker_id != worker_id
        or route.episode_id != episode_id
        or route.current_operation_id != operation_id
        or existing.operation_id != operation_id
        or existing.project_id != route.project_id
        or existing.episode_id != episode_id
        or existing.kind != "node_chat"
        or operation_id not in episode_invocations
        or parent is None
        or parent.project_id != existing.project_id
        or parent.episode_id != episode_id
        or parent.kind != "node_chat"
        or parent.status != "succeeded"
        or existing.attempt != parent.attempt + 1
        or not existing.native_session_id
        or existing.native_session_id != parent.native_session_id
        or (existing.stage_host or "") != (parent.stage_host or "")
        or existing.stage_root != parent.stage_root
        or not existing.stage_root
        or not isinstance(request, RunRequest)
        or request.session_id != parent.native_session_id
        or request.trigger != "orchestrator"
        or request.mode != "work"
        or request.patch_kind != "work"
        or request.message is not None
        or request.watcher_ids
        or request.result_view is not None
        or any(
            existing.request.get(field) != parent.request.get(field)
            for field in pinned_request_fields
        )
        or not messages
        or any(
            item.episode_id != episode_id
            or item.recipient_task_id != worker_id
            or item.delivered_at != existing.created_at
            for item in messages
        )
    ):
        raise ValueError(
            "The committed child Work message wake lost its allocation or mail binding."
        )


def _exact_child_resume_problem(tasks: BackgroundAgentTasks, record: AgentTaskRecord) -> str | None:
    if record.status not in {"paused", "interrupted", "failed"}:
        return "only a paused, interrupted, or failed attempt can be resumed"
    if tasks._failure_is_session_limit(record):
        return "the saved provider session reached its limit"
    if tasks._continuation_context_is_unavailable(record):
        return "the saved continuation context is unavailable"
    if (
        not record.native_session_id
        or not record.stage_root
        or not tasks._session_is_rcp_owned(record)
    ):
        return "the attempt has no complete RCP-owned session and stage"
    if record.stage_host:
        try:
            available = RemoteRunStage(record.stage_host).directory_exists(record.stage_root)
        except Exception as exc:
            raise OSError(
                "The saved provider workspace could not be checked because its remote "
                "infrastructure is unavailable."
            ) from exc
        if available is None:
            raise OSError(
                "The saved provider workspace could not be checked because its remote "
                "infrastructure is unavailable."
            )
    else:
        try:
            stage = Path(record.stage_root)
            available = stage.is_dir() and not stage.is_symlink()
        except Exception:
            available = False
    if available is not True:
        return "the saved provider workspace is unavailable"
    return None
