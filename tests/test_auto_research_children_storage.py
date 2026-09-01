from __future__ import annotations

import hashlib
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from rcp.core.authority import AgentDispatchAuthority, AgentDispatchScope
from rcp.core.models import AuthorizedHuman
from rcp.core.transition_models import GraphHeadRef, GraphTargetRef
from rcp.limits import AUTO_RESEARCH_APPLY_MAX_PER_TURN
from rcp.storage import (
    AgentTaskRecord,
    AppStore,
    AutoResearchActorBusy,
    AutoResearchApplyResultRecord,
    AutoResearchChildAdmissionRecord,
    AutoResearchChildExperimentRecord,
    AutoResearchChildWorkRecord,
    AutoResearchCommandFileRecord,
    AutoResearchExperimentAllowanceReached,
    AutoResearchLifecycleNoticeRecord,
    AutoResearchMessageRecord,
    AutoResearchStateRecord,
    EpisodeInvocationCeilingReached,
    EpisodeNotRunning,
    EpisodeRecord,
    ProjectRecord,
)
from rcp.storage.auto_research_children import (
    AutoResearchInboxHarvestTooLarge,
    AutoResearchInboxNoticeUnacknowledgeable,
)


def _insert_unbounded_legacy_lifecycle_notice(
    store: AppStore,
    record: AutoResearchLifecycleNoticeRecord,
) -> AutoResearchLifecycleNoticeRecord:
    """Simulate a notice persisted before per-command response bounds existed."""

    with store.connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            INSERT INTO auto_research_lifecycle_notices (
                notice_id, episode_id, source_kind, source_id, source_event,
                source_attempt, state, payload_json, created_at, delivered_at,
                delivery_operation_id, acknowledged_at, acknowledged_by
            ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, NULL, NULL, NULL, NULL)
            """,
            (
                record.notice_id,
                record.episode_id,
                record.source_kind,
                record.source_id,
                record.source_event,
                record.source_attempt,
                json.dumps(record.payload, sort_keys=True, separators=(",", ":")),
                record.created_at,
            ),
        )
    return record


def _identity(store: AppStore) -> AuthorizedHuman:
    owner = store.local_owner
    assert owner is not None
    if owner.display_name is None:
        owner = store.rename_space_user(owner.user_id, "Researcher")
    return AuthorizedHuman(
        space_id=store.space_id,
        user_id=owner.user_id,
        display_name=owner.display_name,
    )


def _project(store: AppStore) -> None:
    store.upsert_project(
        ProjectRecord(
            project_id="project",
            locator="/tmp/project/research.yaml",
            name="project",
            state_location="/tmp/project/.research",
            state_remote=False,
            added_at=store.now(),
        )
    )


def _auto_parent(
    store: AppStore,
    *,
    ceiling: int = 4,
) -> tuple[EpisodeRecord, AgentTaskRecord]:
    now = store.now()
    authorizer = _identity(store)
    episode_id = str(uuid.uuid4())
    graph_target = GraphTargetRef(kind="branch", branch_id=episode_id)
    episode = EpisodeRecord(
        episode_id=episode_id,
        project_id="project",
        mode="auto_research",
        graph_target=graph_target,
        graph_base_head=GraphHeadRef(revision=0),
        status="queued",
        invocation_ceiling=ceiling,
        authorized_by=authorizer,
        created_at=now,
        updated_at=now,
    )
    root_id = str(uuid.uuid4())
    root = AgentTaskRecord(
        operation_id=root_id,
        project_id="project",
        episode_id=episode.episode_id,
        graph_target=graph_target,
        kind="auto_research",
        status="queued",
        request={
            "episode_id": episode.episode_id,
            "role": "orchestrator",
            "actor_operation_id": root_id,
            "run_truth_scope": ["repo"],
        },
        created_at=now,
        updated_at=now,
        status_message="Queued",
        authorized_by=authorizer,
        dispatch_authority=AgentDispatchAuthority(
            profile="orchestrator",
            task_contract="orchestrate",
            scope=AgentDispatchScope(
                run_truth_scope=["repo"],
                episode_id=episode.episode_id,
                patch_kind="work",
            ),
        ),
    )
    stored_episode, stored_root = store.create_auto_research_episode_with_root_task(
        episode,
        AutoResearchStateRecord(
            episode_id=episode.episode_id,
            starting_instruction="Run the research plan.",
            created_at=now,
            updated_at=now,
        ),
        root,
    )
    assert stored_episode.graph_target == stored_root.graph_target == graph_target
    assert stored_episode.graph_base_head == GraphHeadRef(revision=0)
    return stored_episode, stored_root


def _work_pair(
    store: AppStore,
    episode: EpisodeRecord,
    root: AgentTaskRecord,
    *,
    worker_id: str,
    operation_id: str | None = None,
    chat_id: str | None = None,
) -> tuple[AutoResearchChildWorkRecord, AgentTaskRecord]:
    now = store.now()
    operation_id = operation_id or str(uuid.uuid4())
    instruction = f"Investigate {worker_id}."
    task = AgentTaskRecord(
        operation_id=operation_id,
        project_id=episode.project_id,
        episode_id=episode.episode_id,
        graph_target=episode.graph_target,
        kind="node_chat",
        status="queued",
        request={
            "provider": "codex",
            "model": "gpt-5",
            "reasoning": "medium",
            "run_on": "local",
            "run_truth_scope": ["repo"],
            "chat_scope": "node",
            "mode": "work",
            "trigger": "orchestrator",
            "patch_kind": "work",
            "node_id": "exp/seat",
            "chat_id": chat_id or f"chat-{worker_id}",
            "message": instruction,
            "session_id": None,
            "watcher_ids": [],
        },
        created_at=now,
        updated_at=now,
        status_message="Queued",
        authorized_by=episode.authorized_by,
    )
    route = AutoResearchChildWorkRecord(
        worker_id=worker_id,
        episode_id=episode.episode_id,
        project_id=episode.project_id,
        control_node_id="exp/seat",
        root_operation_id=operation_id,
        current_operation_id=operation_id,
        admitted_by_operation_id=root.operation_id,
        instruction=instruction,
        instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
        created_at=now,
        updated_at=now,
    )
    return route, task


def _experiment_task(
    store: AppStore,
    episode_id: str,
    authorizer: AuthorizedHuman | None,
    *,
    node_id: str,
    operation_id: str | None = None,
    invocation: int = 1,
    ceiling: int = 10,
    trigger: str = "orchestrator",
    parent_operation_id: str | None = None,
    attempt: int = 1,
    session_id: str | None = None,
    stage_root: str | None = None,
) -> AgentTaskRecord:
    now = store.now()
    existing_episode = store.episode(episode_id)
    if existing_episode is None:
        auto_parents = [
            episode for episode in store.episodes("project") if episode.mode == "auto_research"
        ]
        assert len(auto_parents) == 1
        graph_target = auto_parents[0].graph_target
    else:
        graph_target = existing_episode.graph_target
    return AgentTaskRecord(
        operation_id=operation_id or str(uuid.uuid4()),
        project_id="project",
        episode_id=episode_id,
        graph_target=graph_target,
        kind="node_chat",
        status="queued",
        request={
            "provider": "codex",
            "model": "gpt-5",
            "reasoning": "medium",
            "run_on": "local",
            "run_truth_scope": ["repo"],
            "chat_id": f"experiment-{episode_id}",
            "node_id": node_id,
            "message": "Run the bounded experiment.",
            "mode": "work",
            "trigger": trigger,
            "patch_kind": "experiment_loop",
            "control_node_id": node_id,
            "control_revision": 1,
            "control_episode_id": episode_id,
            "control_invocation": invocation,
            "control_invocation_ceiling": ceiling,
            "control_decision_bundle": [],
            "control_completion_criteria": ["The run completes."],
            "watcher_ids": [],
            "session_id": session_id,
            "workflow_ids": [],
            "skill_ids": [],
            "invoked_workflow_ids": [],
            "invoked_skill_ids": [],
            "resolved_skill_packages": [],
        },
        created_at=now,
        updated_at=now,
        status_message="Queued",
        attempt=attempt,
        parent_operation_id=parent_operation_id,
        native_session_id=session_id,
        stage_root=stage_root,
        authorized_by=authorizer,
    )


def _experiment_route(
    store: AppStore,
    parent: EpisodeRecord,
    root: AgentTaskRecord,
    task: AgentTaskRecord,
    *,
    state: str = "running",
    replaces_episode_id: str | None = None,
) -> AutoResearchChildExperimentRecord:
    node_id = str(task.request["control_node_id"])
    return AutoResearchChildExperimentRecord(
        child_episode_id=str(task.episode_id),
        auto_research_episode_id=parent.episode_id,
        project_id=parent.project_id,
        control_node_id=node_id,
        state=state,
        replaces_episode_id=replaces_episode_id,
        request={"goal": task.request["message"]},
        goal_sha256=hashlib.sha256(str(task.request["message"]).encode()).hexdigest(),
        parent_operation_id=root.operation_id,
        created_at=task.created_at,
        updated_at=task.updated_at,
    )


def _admission(
    store: AppStore,
    parent: EpisodeRecord,
    *,
    admission_id: str,
    child_kind: str,
    child_id: str,
) -> AutoResearchChildAdmissionRecord:
    now = store.now()
    return AutoResearchChildAdmissionRecord(
        admission_id=admission_id,
        episode_id=parent.episode_id,
        project_id=parent.project_id,
        child_kind=child_kind,
        child_id=child_id,
        state="accepted",
        created_at=now,
        updated_at=now,
    )


def _orchestrator_wake(
    store: AppStore,
    parent: EpisodeRecord,
    predecessor: AgentTaskRecord,
    *,
    operation_id: str | None = None,
) -> AgentTaskRecord:
    now = store.now()
    request = dict(predecessor.request)
    request.update(
        {
            "role": "orchestrator",
            "actor_operation_id": parent.root_operation_id,
            "wake_cause": "lifecycle",
        }
    )
    return predecessor.model_copy(
        update={
            "operation_id": operation_id or str(uuid.uuid4()),
            "status": "queued",
            "request": request,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "finished_at": None,
            "status_message": "Queued",
            "error": None,
            "attempt": 1,
            "parent_operation_id": predecessor.operation_id,
        }
    )


def _child_work_mail_wake(
    store: AppStore,
    current: AgentTaskRecord,
    *,
    operation_id: str | None = None,
) -> AgentTaskRecord:
    now = store.now()
    assert current.native_session_id is not None and current.stage_root is not None
    request = dict(current.request)
    request.update(
        {
            "session_id": current.native_session_id,
            "message": None,
            "trigger": "orchestrator",
            "watcher_ids": [],
            "result_view": None,
        }
    )
    return current.model_copy(
        update={
            "operation_id": operation_id or str(uuid.uuid4()),
            "status": "queued",
            "request": request,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "finished_at": None,
            "status_message": "Queued",
            "error": None,
            "attempt": current.attempt + 1,
            "parent_operation_id": current.operation_id,
        }
    )


def _completed_child_project_rows(
    store: AppStore,
) -> tuple[
    EpisodeRecord,
    AutoResearchChildWorkRecord,
    AutoResearchChildExperimentRecord,
    AutoResearchChildAdmissionRecord,
]:
    _project(store)
    parent, root = _auto_parent(store, ceiling=3)
    work_route, work_task = _work_pair(
        store,
        parent,
        root,
        worker_id="worker-project-migration",
    )
    store.create_auto_research_child_work(work_route, work_task)

    child_id = str(uuid.uuid4())
    child_task = _experiment_task(
        store,
        child_id,
        parent.authorized_by,
        node_id="exp/project-migration",
    )
    experiment_route = _experiment_route(store, parent, root, child_task)
    store.create_experiment_episode_with_invocation(
        child_task,
        auto_research_route=experiment_route,
    )
    admission = store.record_auto_research_child_admission(
        _admission(
            store,
            parent,
            admission_id="pending-project-migration-admission",
            child_kind="work",
            child_id="future-project-migration-worker",
        )
    )

    store.complete_agent_task(work_task.operation_id, applied_revision=None, result={})
    store.complete_agent_task(child_task.operation_id, applied_revision=None, result={})
    store.complete_agent_task(root.operation_id, applied_revision=None, result={})
    return parent, work_route, experiment_route, admission


def test_child_work_admission_spends_b_and_reflects_route_atomically(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    episode, root = _auto_parent(store, ceiling=2)
    route, task = _work_pair(store, episode, root, worker_id="worker-one")
    admission = store.record_auto_research_child_admission(
        _admission(
            store,
            episode,
            admission_id="spawn-command",
            child_kind="work",
            child_id=route.worker_id,
        )
    )

    stored, stored_task = store.create_auto_research_child_work(
        route,
        task,
        admission_id=admission.admission_id,
    )

    assert stored == route
    assert stored_task.kind == "node_chat"
    assert store.episode_budget_meter(episode.episode_id).invocations_used == 2
    assert [item.operation_id for item in store.episode_invocations(episode.episode_id)] == [
        root.operation_id,
        task.operation_id,
    ]
    assert store.auto_research_child_admission(admission.admission_id).state == "reflected"  # type: ignore[union-attr]

    store.complete_agent_task(task.operation_id, applied_revision=None, result={})
    notices = store.auto_research_lifecycle_notices(episode.episode_id)
    assert [(notice.source_kind, notice.source_id, notice.source_event) for notice in notices] == [
        ("worker", route.worker_id, "succeeded")
    ]


def test_concurrent_child_work_admission_cannot_overspend_b(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    episode, root = _auto_parent(store, ceiling=2)
    pairs = [_work_pair(store, episode, root, worker_id=f"worker-{index}") for index in range(2)]

    def admit(pair: tuple[AutoResearchChildWorkRecord, AgentTaskRecord]) -> str:
        try:
            store.create_auto_research_child_work(*pair)
        except EpisodeInvocationCeilingReached:
            return "exhausted"
        return "admitted"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = sorted(executor.map(admit, pairs))

    assert results == ["admitted", "exhausted"]
    assert store.episode_budget_meter(episode.episode_id).invocations_used == 2
    assert len(store.auto_research_child_works(episode.episode_id)) == 1


def test_exact_child_work_recovery_reuses_allocation_and_binding(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    episode, root = _auto_parent(store, ceiling=3)
    route, task = _work_pair(store, episode, root, worker_id="worker-one")
    store.create_auto_research_child_work(route, task)
    store.checkpoint_agent_task(
        task.operation_id,
        native_session_id="saved-session",
        stage_root="/tmp/saved-child-stage",
    )
    store.fail_agent_task(task.operation_id, "network failed")
    current = store.agent_task(task.operation_id)
    assert current is not None
    failure_notice = store.auto_research_lifecycle_notices(episode.episode_id)[0]
    assert failure_notice.source_event == "failed"
    assert failure_notice.payload["resume_available"] is True
    now = store.now()
    recovery_request = dict(current.request)
    recovery_request["session_id"] = "saved-session"
    recovery = current.model_copy(
        update={
            "operation_id": str(uuid.uuid4()),
            "status": "queued",
            "request": recovery_request,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "finished_at": None,
            "status_message": "Queued",
            "error": None,
            "attempt": 2,
            "parent_operation_id": task.operation_id,
        }
    )

    recovered_route, recovered = store.create_auto_research_child_work_recovery(
        route.worker_id,
        recovery,
    )

    assert recovered.parent_operation_id == task.operation_id
    assert recovered_route.current_operation_id == recovered.operation_id
    with store.connection() as connection:
        allocation = connection.execute(
            """
            SELECT allocation_operation_id FROM auto_research_child_work_attempts
            WHERE operation_id = ?
            """,
            (recovered.operation_id,),
        ).fetchone()
    assert allocation is not None
    assert allocation["allocation_operation_id"] == task.operation_id
    assert store.episode_budget_meter(episode.episode_id).invocations_used == 2
    assert len(store.episode_invocations(episode.episode_id)) == 2


def test_child_work_recovery_inherits_a_message_wake_allocation(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    episode, root = _auto_parent(store, ceiling=4)
    route, task = _work_pair(store, episode, root, worker_id="worker-mail-recovery")
    store.create_auto_research_child_work(route, task)
    store.checkpoint_agent_task(
        task.operation_id,
        native_session_id="saved-session",
        stage_root="/tmp/saved-child-stage",
    )
    store.complete_agent_task(task.operation_id, applied_revision=None, result={})
    current = store.agent_task(task.operation_id)
    assert current is not None
    message = store.record_auto_research_message(
        AutoResearchMessageRecord(
            message_id="wake-mail",
            episode_id=episode.episode_id,
            sender_role="orchestrator",
            sender_task_id=root.operation_id,
            recipient_task_id=route.worker_id,
            control_node_id=route.control_node_id,
            body="Continue the bounded check.",
            created_at=store.now(),
        )
    )
    wake = _child_work_mail_wake(store, current)
    admitted = store.create_auto_research_child_work_message_wake_task(
        wake,
        worker_id=route.worker_id,
        message_ids=[message.message_id],
    )
    assert admitted is not None
    store.fail_agent_task(admitted.operation_id, "network failed")
    failed = store.agent_task(admitted.operation_id)
    assert failed is not None
    recovery_request = dict(failed.request)
    recovery_request["session_id"] = "saved-session"
    now = store.now()
    recovery = failed.model_copy(
        update={
            "operation_id": str(uuid.uuid4()),
            "status": "queued",
            "request": recovery_request,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "finished_at": None,
            "status_message": "Queued",
            "error": None,
            "attempt": failed.attempt + 1,
            "parent_operation_id": failed.operation_id,
        }
    )

    store.create_auto_research_child_work_recovery(route.worker_id, recovery)

    with store.connection() as connection:
        allocation = connection.execute(
            """
            SELECT allocation_operation_id FROM auto_research_child_work_attempts
            WHERE operation_id = ?
            """,
            (recovery.operation_id,),
        ).fetchone()
    assert allocation is not None
    assert allocation["allocation_operation_id"] == wake.operation_id


def test_session_limit_notice_requires_replacement_even_with_a_saved_checkpoint(
    tmp_path,
) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    episode, root = _auto_parent(store, ceiling=2)
    route, task = _work_pair(store, episode, root, worker_id="worker-session-limit")
    store.create_auto_research_child_work(route, task)
    store.checkpoint_agent_task(
        task.operation_id,
        native_session_id="spent-session",
        stage_root="/tmp/spent-session-stage",
    )
    store.record_agent_task_receipt(
        task.operation_id,
        "provider_terminal_error",
        {"classification": "session_limit"},
    )

    store.fail_agent_task(task.operation_id, "You've hit your session limit.")

    notice = store.auto_research_lifecycle_notices(episode.episode_id)[0]
    assert notice.payload["resume_available"] is False
    assert notice.payload["replacement_command"] == "spawn"


def test_ordinary_child_work_mail_preserves_parent_worker_star_topology(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    parent, root = _auto_parent(store, ceiling=5)
    first_route, first_task = _work_pair(store, parent, root, worker_id="worker-one")
    second_route, second_task = _work_pair(store, parent, root, worker_id="worker-two")
    store.create_auto_research_child_work(first_route, first_task)
    store.create_auto_research_child_work(second_route, second_task)

    outbound = store.record_auto_research_message(
        AutoResearchMessageRecord(
            message_id="to-child",
            episode_id=parent.episode_id,
            sender_role="orchestrator",
            sender_task_id=root.operation_id,
            recipient_task_id=first_route.worker_id,
            control_node_id=first_route.control_node_id,
            body="Check the latest evidence.",
            created_at=store.now(),
        )
    )
    reply = store.record_auto_research_message(
        AutoResearchMessageRecord(
            message_id="from-child",
            episode_id=parent.episode_id,
            sender_role="worker",
            sender_task_id=first_task.operation_id,
            recipient_task_id=root.operation_id,
            control_node_id=first_route.control_node_id,
            body="The evidence is inconclusive.",
            created_at=store.now(),
        )
    )

    assert outbound.recipient_task_id == first_route.worker_id
    assert reply.recipient_task_id == root.operation_id
    with pytest.raises(ValueError, match="reply only"):
        store.record_auto_research_message(
            reply.model_copy(
                update={
                    "message_id": "child-to-peer",
                    "recipient_task_id": second_route.worker_id,
                }
            )
        )
    with pytest.raises(ValueError, match="sender role"):
        store.record_auto_research_message(
            reply.model_copy(
                update={
                    "message_id": "unregistered-child",
                    "sender_task_id": str(uuid.uuid4()),
                }
            )
        )


def test_child_work_mail_wake_spends_b_updates_route_and_claims_exact_prefix(
    tmp_path,
) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    parent, root = _auto_parent(store, ceiling=5)
    route, task = _work_pair(store, parent, root, worker_id="worker-mail")
    store.create_auto_research_child_work(route, task)
    store.checkpoint_agent_task(
        task.operation_id,
        native_session_id="worker-session",
        stage_root="/tmp/worker-mail-stage",
    )
    store.complete_agent_task(task.operation_id, applied_revision=None, result={})
    current = store.agent_task(task.operation_id)
    assert current is not None
    first = store.record_auto_research_message(
        AutoResearchMessageRecord(
            message_id="mail-one",
            episode_id=parent.episode_id,
            sender_role="orchestrator",
            sender_task_id=root.operation_id,
            recipient_task_id=route.worker_id,
            control_node_id=route.control_node_id,
            body="First message.",
            created_at="2026-08-16T00:00:01+00:00",
        )
    )
    second = store.record_auto_research_message(
        first.model_copy(
            update={
                "message_id": "mail-two",
                "body": "Second message.",
                "created_at": "2026-08-16T00:00:02+00:00",
            }
        )
    )
    wake = _child_work_mail_wake(store, current)
    meter_before = store.episode_budget_meter(parent.episode_id)

    assert (
        store.create_auto_research_child_work_message_wake_task(
            wake,
            worker_id=route.worker_id,
            message_ids=[second.message_id],
        )
        is None
    )
    assert store.agent_task(wake.operation_id) is None
    assert store.episode_budget_meter(parent.episode_id) == meter_before

    admitted = store.create_auto_research_child_work_message_wake_task(
        wake,
        worker_id=route.worker_id,
        message_ids=[first.message_id],
    )

    assert admitted is not None
    assert admitted.attempt == current.attempt + 1
    updated_route = store.auto_research_child_work(route.worker_id)
    assert updated_route is not None
    assert updated_route.root_operation_id == route.root_operation_id
    assert updated_route.current_operation_id == wake.operation_id
    assert store.auto_research_child_work_for_operation(wake.operation_id) == updated_route
    assert store.episode_budget_meter(parent.episode_id).invocations_used == (
        meter_before.invocations_used + 1
    )
    assert store.auto_research_message(first.message_id).delivery_operation_id == (  # type: ignore[union-attr]
        wake.operation_id
    )
    assert store.pending_auto_research_messages(parent.episode_id, route.worker_id) == [second]

    store.complete_agent_task(admitted.operation_id, applied_revision=None, result={})
    worker_notices = [
        notice
        for notice in store.auto_research_lifecycle_notices(parent.episode_id)
        if notice.source_kind == "worker"
        and notice.source_id == route.worker_id
        and notice.source_event == "succeeded"
    ]
    assert [notice.source_attempt for notice in worker_notices] == [1, 2]


def test_child_work_mail_wake_lineage_failure_leaves_mail_and_budget_unchanged(
    tmp_path,
) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    parent, root = _auto_parent(store, ceiling=4)
    route, task = _work_pair(store, parent, root, worker_id="worker-lineage")
    store.create_auto_research_child_work(route, task)
    store.checkpoint_agent_task(
        task.operation_id,
        native_session_id="worker-session",
        stage_root="/tmp/worker-lineage-stage",
    )
    store.complete_agent_task(task.operation_id, applied_revision=None, result={})
    current = store.agent_task(task.operation_id)
    assert current is not None
    message = store.record_auto_research_message(
        AutoResearchMessageRecord(
            message_id="lineage-mail",
            episode_id=parent.episode_id,
            sender_role="orchestrator",
            sender_task_id=root.operation_id,
            recipient_task_id=route.worker_id,
            body="Continue.",
            created_at=store.now(),
        )
    )
    wake = _child_work_mail_wake(store, current)
    changed_request = dict(wake.request)
    changed_request["chat_id"] = "another-chat"
    invalid = wake.model_copy(update={"request": changed_request})
    meter_before = store.episode_budget_meter(parent.episode_id)

    with pytest.raises(ValueError, match="exact saved Work session"):
        store.create_auto_research_child_work_message_wake_task(
            invalid,
            worker_id=route.worker_id,
            message_ids=[message.message_id],
        )

    assert store.agent_task(invalid.operation_id) is None
    assert store.episode_budget_meter(parent.episode_id) == meter_before
    assert store.pending_auto_research_messages(parent.episode_id, route.worker_id) == [message]


def test_concurrent_child_work_mail_wakes_claim_once_and_admit_one_lineage(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    parent, root = _auto_parent(store, ceiling=4)
    route, task = _work_pair(store, parent, root, worker_id="worker-race")
    store.create_auto_research_child_work(route, task)
    store.checkpoint_agent_task(
        task.operation_id,
        native_session_id="worker-session",
        stage_root="/tmp/worker-race-stage",
    )
    store.complete_agent_task(task.operation_id, applied_revision=None, result={})
    current = store.agent_task(task.operation_id)
    assert current is not None
    message = store.record_auto_research_message(
        AutoResearchMessageRecord(
            message_id="race-mail",
            episode_id=parent.episode_id,
            sender_role="orchestrator",
            sender_task_id=root.operation_id,
            recipient_task_id=route.worker_id,
            body="Continue once.",
            created_at=store.now(),
        )
    )
    candidates = [
        _child_work_mail_wake(store, current, operation_id=str(uuid.uuid4())) for _ in range(2)
    ]

    def admit(candidate: AgentTaskRecord) -> str | None:
        admitted = store.create_auto_research_child_work_message_wake_task(
            candidate,
            worker_id=route.worker_id,
            message_ids=[message.message_id],
        )
        return admitted.operation_id if admitted is not None else None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(admit, candidates))

    winner_ids = [result for result in results if result is not None]
    assert len(winner_ids) == 1
    winner_id = winner_ids[0]
    assert store.auto_research_child_work(route.worker_id).current_operation_id == winner_id  # type: ignore[union-attr]
    assert store.episode_budget_meter(parent.episode_id).invocations_used == 3
    assert [
        candidate.operation_id
        for candidate in candidates
        if store.agent_task(candidate.operation_id)
    ] == [winner_id]


def test_child_experiments_share_atomic_five_times_b_allowance(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    parent, root = _auto_parent(store, ceiling=1)

    child_ids: list[str] = []
    for index in range(5):
        child_id = str(uuid.uuid4())
        task = _experiment_task(
            store,
            child_id,
            parent.authorized_by,
            node_id=f"exp/{index}",
        )
        route = _experiment_route(store, parent, root, task)
        store.create_experiment_episode_with_invocation(task, auto_research_route=route)
        child_ids.append(child_id)

    allowance = store.auto_research_experiment_allowance(parent.episode_id)
    assert allowance.model_dump() == {"total": 5, "used": 5, "remaining": 0}

    refused_id = str(uuid.uuid4())
    refused_task = _experiment_task(
        store,
        refused_id,
        parent.authorized_by,
        node_id="exp/refused",
    )
    refused_route = _experiment_route(store, parent, root, refused_task)
    with pytest.raises(AutoResearchExperimentAllowanceReached):
        store.create_experiment_episode_with_invocation(
            refused_task,
            auto_research_route=refused_route,
        )

    assert store.episode(refused_id) is None
    assert store.agent_task(refused_task.operation_id) is None
    assert store.auto_research_child_experiment(refused_id) is None
    assert child_ids == [
        item.child_episode_id for item in store.auto_research_child_experiments(parent.episode_id)
    ]


def test_child_transition_and_lifecycle_notice_roll_back_together(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    parent, root = _auto_parent(store, ceiling=2)
    route, task = _work_pair(store, parent, root, worker_id="worker-atomic")
    store.create_auto_research_child_work(route, task)
    store.record_auto_research_lifecycle_notice(
        AutoResearchLifecycleNoticeRecord(
            notice_id="conflicting-notice",
            episode_id=parent.episode_id,
            source_kind="worker",
            source_id=route.worker_id,
            source_event="succeeded",
            source_attempt=1,
            payload={"conflict": True},
            created_at=store.now(),
        )
    )

    with pytest.raises(ValueError, match="different facts"):
        store.complete_agent_task(task.operation_id, applied_revision=None, result={})

    assert store.agent_task(task.operation_id).status == "queued"  # type: ignore[union-attr]


def test_gracefully_stopped_child_work_notifies_the_sleeping_orchestrator(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    parent, root = _auto_parent(store, ceiling=2)
    route, task = _work_pair(store, parent, root, worker_id="worker-stopped")
    store.create_auto_research_child_work(route, task)
    store.checkpoint_agent_task(
        task.operation_id,
        native_session_id="worker-session",
        stage_root="/tmp/worker-stopped-stage",
    )
    store.request_auto_research_child_work_stop(route.worker_id)

    store.pause_agent_task(task.operation_id, detail="Worker stopped at its checkpoint.")

    notice = store.auto_research_lifecycle_notices(parent.episode_id)[0]
    assert notice.source_kind == "worker"
    assert notice.source_id == route.worker_id
    assert notice.source_event == "paused"
    assert notice.payload["status"] == "paused"
    assert notice.payload["resume_available"] is False
    assert notice.payload["replacement_command"] == "spawn"


def test_exact_experiment_recovery_does_not_spend_e_again(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    parent, root = _auto_parent(store, ceiling=2)
    child_id = str(uuid.uuid4())
    task = _experiment_task(store, child_id, parent.authorized_by, node_id="exp/one")
    route = _experiment_route(store, parent, root, task)
    store.create_experiment_episode_with_invocation(task, auto_research_route=route)
    store.checkpoint_agent_task(
        task.operation_id,
        native_session_id="saved-session",
        stage_root="/tmp/saved-experiment-stage",
    )
    store.fail_agent_task(task.operation_id, "network failed")
    recovery = _experiment_task(
        store,
        child_id,
        parent.authorized_by,
        node_id="exp/one",
        parent_operation_id=task.operation_id,
        attempt=2,
        session_id="saved-session",
        stage_root="/tmp/saved-experiment-stage",
    )

    store.create_experiment_recovery_task(recovery)

    assert store.auto_research_experiment_allowance(parent.episode_id).used == 1
    assert store.episode(child_id).invocations_used == 1  # type: ignore[union-attr]
    notice = store.auto_research_lifecycle_notices(parent.episode_id)[0]
    assert notice.source_kind == "experiment_task"
    assert notice.payload["episode_id"] == child_id
    assert store.auto_research_child_experiment(child_id).state == "running"  # type: ignore[union-attr]


def test_stopping_child_experiment_pause_keeps_exact_resume_available(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    parent, root = _auto_parent(store, ceiling=2)
    child_id = str(uuid.uuid4())
    task = _experiment_task(store, child_id, parent.authorized_by, node_id="exp/stopping")
    route = _experiment_route(store, parent, root, task)
    store.create_experiment_episode_with_invocation(task, auto_research_route=route)
    store.checkpoint_agent_task(
        task.operation_id,
        native_session_id="saved-session",
        stage_root="/tmp/stopping-experiment-stage",
    )
    store.request_episode_stop(child_id)

    store.pause_agent_task(task.operation_id, detail="Experiment stopped at its checkpoint.")

    notice = store.auto_research_lifecycle_notices(parent.episode_id)[0]
    assert notice.source_kind == "experiment_task"
    assert notice.source_event == "paused"
    assert notice.payload["resume_available"] is True
    assert "replacement_command" not in notice.payload


def test_stopped_child_experiment_terminalizes_route_and_notifies_parent(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    parent, root = _auto_parent(store, ceiling=2)
    child_id = str(uuid.uuid4())
    task = _experiment_task(store, child_id, parent.authorized_by, node_id="exp/stop")
    route = _experiment_route(store, parent, root, task)
    store.create_experiment_episode_with_invocation(task, auto_research_route=route)
    store.complete_agent_task(task.operation_id, applied_revision=None, result={})
    store.request_episode_stop(child_id)

    store.mark_episode_stop_skipped(child_id, diagnostic="Replaced by a fresh episode.")

    assert store.auto_research_child_experiment(child_id).state == "terminal"  # type: ignore[union-attr]
    notice = store.auto_research_lifecycle_notices(parent.episode_id)[0]
    assert notice.source_kind == "experiment_episode"
    assert notice.source_event == "stopped"
    assert notice.payload["ending"] == "stopped"


def test_lifecycle_notice_dedup_harvest_clear_and_delivery_are_durable(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    parent, _ = _auto_parent(store)
    now = store.now()
    first = AutoResearchLifecycleNoticeRecord(
        notice_id="notice-one",
        episode_id=parent.episode_id,
        source_kind="worker",
        source_id="worker-one",
        source_event="failed",
        source_attempt=1,
        payload={"resume_available": True},
        created_at=now,
    )

    assert store.record_auto_research_lifecycle_notice(first) == first
    duplicate = first.model_copy(update={"notice_id": "another-generated-id"})
    assert store.record_auto_research_lifecycle_notice(duplicate).notice_id == first.notice_id
    with pytest.raises(ValueError, match="different facts"):
        store.record_auto_research_lifecycle_notice(
            duplicate.model_copy(update={"payload": {"resume_available": False}})
        )

    delivered = store.claim_auto_research_lifecycle_notices(
        parent.episode_id,
        "wake-operation",
    )
    assert [notice.state for notice in delivered] == ["delivered"]
    assert store.pending_auto_research_lifecycle_notices(parent.episode_id) == []

    second = first.model_copy(
        update={
            "notice_id": "notice-two",
            "source_event": "settled",
            "payload": {"status": "succeeded"},
        }
    )
    third = first.model_copy(
        update={
            "notice_id": "notice-three",
            "source_id": "worker-two",
            "payload": {"status": "failed"},
        }
    )
    store.record_auto_research_lifecycle_notice(second)
    store.record_auto_research_lifecycle_notice(third)
    harvested = store.harvest_auto_research_lifecycle_notices(
        parent.episode_id,
        acknowledged_by="orchestrator-turn",
        limit=1,
    )
    assert len(harvested) == 1
    assert harvested[0].notice_id == "notice-three"
    assert harvested[0].payload
    cleared = store.clear_auto_research_lifecycle_notices(
        parent.episode_id,
        acknowledged_by="orchestrator-turn",
    )
    assert cleared == ["notice-two"]
    states = {
        notice.notice_id: notice.state
        for notice in store.auto_research_lifecycle_notices(parent.episode_id)
    }
    assert states == {
        "notice-one": "delivered",
        "notice-three": "acknowledged",
        "notice-two": "acknowledged",
    }

    late = first.model_copy(
        update={
            "notice_id": "notice-late",
            "source_id": "worker-late",
            "created_at": store.now(),
        }
    )
    store.record_auto_research_lifecycle_notice(late)
    assert [
        notice.notice_id
        for notice in store.pending_auto_research_lifecycle_notices(parent.episode_id)
    ] == ["notice-late"]


def test_keyed_inbox_receipt_replays_empty_snapshot_without_clearing_later_notice(
    tmp_path,
) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    parent, _ = _auto_parent(store)

    already_delivered = store.record_auto_research_lifecycle_notice(
        AutoResearchLifecycleNoticeRecord(
            notice_id="already-delivered",
            episode_id=parent.episode_id,
            source_kind="worker",
            source_id="worker-delivered",
            source_event="settled",
            payload={"status": "failed"},
            created_at=store.now(),
        )
    )
    claimed = store.claim_auto_research_lifecycle_notices(
        parent.episode_id,
        "lifecycle-wake",
    )
    assert [notice.notice_id for notice in claimed] == [already_delivered.notice_id]

    empty = store.process_auto_research_lifecycle_inbox(
        parent.episode_id,
        effect_id="empty-clear",
        mode="clear",
        acknowledged_by="orchestrator-turn",
    )
    assert empty.count == 0
    assert empty.notice_ids == []
    assert empty.notices == []
    delivered_record = store.auto_research_lifecycle_notices(parent.episode_id)[0]
    assert delivered_record.state == "delivered"
    assert delivered_record.acknowledged_at is None

    late = AutoResearchLifecycleNoticeRecord(
        notice_id="late-after-empty",
        episode_id=parent.episode_id,
        source_kind="worker",
        source_id="worker-late",
        source_event="failed",
        payload={"diagnostic": "network"},
        created_at=store.now(),
    )
    store.record_auto_research_lifecycle_notice(late)
    replay = store.process_auto_research_lifecycle_inbox(
        parent.episode_id,
        effect_id="empty-clear",
        mode="clear",
        acknowledged_by="orchestrator-turn",
    )

    assert replay == empty
    assert store.auto_research_inbox_receipt("empty-clear") == empty
    assert store.pending_auto_research_lifecycle_notices(parent.episode_id) == [late]

    harvested = store.process_auto_research_lifecycle_inbox(
        parent.episode_id,
        effect_id="harvest-late",
        mode="harvest",
        acknowledged_by="orchestrator-turn",
    )
    assert harvested.notice_ids == [late.notice_id]
    assert harvested.count == 1
    assert harvested.notices[0].payload == late.payload
    assert harvested.notices[0].state == "acknowledged"


def test_keyed_inbox_bounds_the_exact_prefix_before_acknowledging_it(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    parent, _ = _auto_parent(store)
    small = store.record_auto_research_lifecycle_notice(
        AutoResearchLifecycleNoticeRecord(
            notice_id="notice-small",
            episode_id=parent.episode_id,
            source_kind="worker",
            source_id="worker-small",
            source_event="settled",
            payload={"status": "succeeded"},
            created_at=store.now(),
        )
    )
    oversized = _insert_unbounded_legacy_lifecycle_notice(
        store,
        AutoResearchLifecycleNoticeRecord(
            notice_id="notice-oversized",
            episode_id=parent.episode_id,
            source_kind="worker",
            source_id="worker-large",
            source_event="failed",
            payload={"diagnostic": "x" * 40_000},
            created_at=store.now(),
        ),
    )

    receipt = store.process_auto_research_lifecycle_inbox(
        parent.episode_id,
        effect_id="bounded-harvest",
        mode="harvest",
        acknowledged_by="orchestrator-turn",
    )

    assert receipt.notice_ids == [small.notice_id]
    assert receipt.notices[0].payload == small.payload
    assert store.auto_research_inbox_receipt(receipt.effect_id) == receipt
    remaining = store.pending_auto_research_lifecycle_notices(parent.episode_id)
    assert remaining == [oversized]
    stored = {
        notice.notice_id: notice
        for notice in store.auto_research_lifecycle_notices(parent.episode_id)
    }
    assert stored[small.notice_id].state == "acknowledged"
    assert stored[oversized.notice_id].state == "pending"


def test_keyed_inbox_refuses_an_oversized_first_harvest_without_a_receipt(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    parent, _ = _auto_parent(store)
    oversized = _insert_unbounded_legacy_lifecycle_notice(
        store,
        AutoResearchLifecycleNoticeRecord(
            notice_id="notice-oversized-first",
            episode_id=parent.episode_id,
            source_kind="worker",
            source_id="worker-large",
            source_event="failed",
            payload={"diagnostic": "x" * 40_000},
            created_at="2026-08-17T00:00:01+00:00",
        ),
    )
    small = store.record_auto_research_lifecycle_notice(
        AutoResearchLifecycleNoticeRecord(
            notice_id="notice-small-second",
            episode_id=parent.episode_id,
            source_kind="worker",
            source_id="worker-small",
            source_event="settled",
            payload={"status": "succeeded"},
            created_at="2026-08-17T00:00:02+00:00",
        )
    )

    with pytest.raises(AutoResearchInboxHarvestTooLarge, match="--clear"):
        store.process_auto_research_lifecycle_inbox(
            parent.episode_id,
            effect_id="oversized-first-harvest",
            mode="harvest",
            acknowledged_by="orchestrator-turn",
        )

    assert store.auto_research_inbox_receipt("oversized-first-harvest") is None
    assert store.pending_auto_research_lifecycle_notices(parent.episode_id) == [
        oversized,
        small,
    ]
    assert all(
        notice.acknowledged_at is None
        for notice in store.auto_research_lifecycle_notices(parent.episode_id)
    )


def test_new_lifecycle_notices_truncate_diagnostics_to_remain_harvestable(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    parent, _ = _auto_parent(store)

    original = AutoResearchLifecycleNoticeRecord(
        notice_id="bounded-new-notice",
        episode_id=parent.episode_id,
        source_kind="worker",
        source_id="worker-large",
        source_event="failed",
        payload={"diagnostic": "x" * 40_000},
        created_at=store.now(),
    )
    stored = store.record_auto_research_lifecycle_notice(original)

    assert stored.payload == {
        "diagnostic": "x" * 2_000,
        "diagnostic_truncated": True,
    }
    assert store.record_auto_research_lifecycle_notice(original) == stored
    receipt = store.process_auto_research_lifecycle_inbox(
        parent.episode_id,
        effect_id="bounded-new-harvest",
        mode="harvest",
        acknowledged_by="orchestrator-turn",
    )
    assert receipt.notice_ids == [stored.notice_id]
    assert receipt.notices[0].payload == stored.payload


def test_oversized_harvest_does_not_recommend_clear_when_the_full_clear_cannot_fit(
    tmp_path,
) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    parent, _ = _auto_parent(store)
    _insert_unbounded_legacy_lifecycle_notice(
        store,
        AutoResearchLifecycleNoticeRecord(
            notice_id="000-oversized-body",
            episode_id=parent.episode_id,
            source_kind="worker",
            source_id="worker-large",
            source_event="failed",
            payload={"diagnostic": "x" * 40_000},
            created_at="2026-08-17T00:00:00+00:00",
        ),
    )
    for index in range(36):
        _insert_unbounded_legacy_lifecycle_notice(
            store,
            AutoResearchLifecycleNoticeRecord(
                notice_id=f"{index + 1:03d}-" + ("n" * 1_024),
                episode_id=parent.episode_id,
                source_kind="worker",
                source_id=f"worker-{index}",
                source_event="settled",
                payload={"status": "succeeded"},
                created_at=f"2026-08-17T00:00:{index + 1:02d}+00:00",
            ),
        )

    with pytest.raises(AutoResearchInboxNoticeUnacknowledgeable, match="complete Clear"):
        store.process_auto_research_lifecycle_inbox(
            parent.episode_id,
            effect_id="harvest-and-clear-too-large",
            mode="harvest",
            acknowledged_by="orchestrator-turn",
        )

    assert store.auto_research_inbox_receipt("harvest-and-clear-too-large") is None
    assert all(
        notice.acknowledged_at is None
        for notice in store.auto_research_lifecycle_notices(parent.episode_id)
    )


def test_lifecycle_wake_spends_once_and_atomically_claims_notice_and_root_mail(
    tmp_path,
) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    parent, root = _auto_parent(store, ceiling=3)
    store.complete_agent_task(root.operation_id, applied_revision=None, result={})
    notice = store.record_auto_research_lifecycle_notice(
        AutoResearchLifecycleNoticeRecord(
            notice_id="wake-notice",
            episode_id=parent.episode_id,
            source_kind="worker",
            source_id="worker-one",
            source_event="failed",
            payload={"resume_available": True},
            created_at=store.now(),
        )
    )
    message = store.record_auto_research_message(
        AutoResearchMessageRecord(
            message_id="wake-mail",
            episode_id=parent.episode_id,
            sender_role="human",
            authorized_by=parent.authorized_by,
            recipient_task_id=root.operation_id,
            body="Also inspect the new run.",
            created_at=store.now(),
        )
    )
    wake = _orchestrator_wake(store, parent, root)

    admitted = store.create_auto_research_lifecycle_wake_task(
        wake,
        lifecycle_notice_ids=[notice.notice_id],
        message_ids=[message.message_id],
    )

    assert admitted is not None
    assert store.episode_budget_meter(parent.episode_id).invocations_used == 2
    delivery = store.auto_research_lifecycle_delivery(wake.operation_id)
    assert [item.notice_id for item in delivery] == [notice.notice_id]
    assert delivery[0].state == "delivered"
    assert store.auto_research_message(message.message_id).delivery_operation_id == (  # type: ignore[union-attr]
        wake.operation_id
    )

    store.fail_agent_task(wake.operation_id, "connection dropped")
    late = store.record_auto_research_lifecycle_notice(
        notice.model_copy(
            update={
                "notice_id": "notice-after-failure",
                "source_id": "worker-two",
                "created_at": store.now(),
            }
        )
    )
    recovery = _orchestrator_wake(store, parent, wake).model_copy(update={"attempt": 2})
    store.create_auto_research_recovery_task(recovery)

    assert store.episode_budget_meter(parent.episode_id).invocations_used == 2
    assert store.auto_research_lifecycle_delivery(recovery.operation_id) == []
    assert store.pending_auto_research_lifecycle_notices(parent.episode_id) == [late]


def test_lifecycle_wake_busy_or_exhausted_rolls_back_notice_claim(tmp_path) -> None:
    busy_store = AppStore(tmp_path / "busy.sqlite3")
    _project(busy_store)
    parent, root = _auto_parent(busy_store, ceiling=2)
    notice = busy_store.record_auto_research_lifecycle_notice(
        AutoResearchLifecycleNoticeRecord(
            notice_id="busy-notice",
            episode_id=parent.episode_id,
            source_kind="worker",
            source_id="worker-busy",
            source_event="failed",
            payload={},
            created_at=busy_store.now(),
        )
    )
    with pytest.raises(AutoResearchActorBusy):
        busy_store.create_auto_research_lifecycle_wake_task(
            _orchestrator_wake(busy_store, parent, root),
            lifecycle_notice_ids=[notice.notice_id],
        )
    assert busy_store.pending_auto_research_lifecycle_notices(parent.episode_id) == [notice]
    assert busy_store.episode_budget_meter(parent.episode_id).invocations_used == 1

    exhausted_store = AppStore(tmp_path / "exhausted.sqlite3")
    _project(exhausted_store)
    exhausted_parent, exhausted_root = _auto_parent(exhausted_store, ceiling=1)
    exhausted_store.complete_agent_task(
        exhausted_root.operation_id,
        applied_revision=None,
        result={},
    )
    exhausted_notice = exhausted_store.record_auto_research_lifecycle_notice(
        notice.model_copy(
            update={
                "notice_id": "exhausted-notice",
                "episode_id": exhausted_parent.episode_id,
                "source_id": "worker-exhausted",
                "created_at": exhausted_store.now(),
            }
        )
    )
    with pytest.raises(EpisodeInvocationCeilingReached):
        exhausted_store.create_auto_research_lifecycle_wake_task(
            _orchestrator_wake(exhausted_store, exhausted_parent, exhausted_root),
            lifecycle_notice_ids=[exhausted_notice.notice_id],
        )
    assert exhausted_store.pending_auto_research_lifecycle_notices(exhausted_parent.episode_id) == [
        exhausted_notice
    ]


def test_command_start_and_exact_file_snapshot_commit_once_on_key_replay(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    parent, root = _auto_parent(store)
    command_id = "original-command"
    content = '{"operations":[]}'
    snapshot = AutoResearchCommandFileRecord(
        command_id=command_id,
        episode_id=parent.episode_id,
        operation_id=root.operation_id,
        kind="apply",
        filename="patch.json",
        sha256=hashlib.sha256(content.encode()).hexdigest(),
        content=content,
        created_at=store.now(),
    )
    started = store.start_agent_command(
        operation_id=root.operation_id,
        command_id=command_id,
        episode_id=parent.episode_id,
        verb="apply",
        idempotency_key="same-key",
        payload={"filename": "patch.json"},
        file_snapshot=snapshot,
    )

    changed = '{"operations":[{"kind":"new"}]}'
    replay = store.start_agent_command(
        operation_id=root.operation_id,
        command_id="replay-command",
        episode_id=parent.episode_id,
        verb="apply",
        idempotency_key="same-key",
        payload={"filename": "patch.json"},
        file_snapshot=snapshot.model_copy(
            update={
                "command_id": "replay-command",
                "content": changed,
                "sha256": hashlib.sha256(changed.encode()).hexdigest(),
            }
        ),
    )

    assert replay.command_id == started.command_id == command_id
    assert store.auto_research_command_file(command_id) == snapshot
    assert store.auto_research_command_file("replay-command") is None


def test_concurrent_apply_starts_atomically_admit_only_one_remaining_slot(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    parent, root = _auto_parent(store)
    for index in range(AUTO_RESEARCH_APPLY_MAX_PER_TURN - 1):
        store.start_agent_command(
            operation_id=root.operation_id,
            command_id=f"seed-apply-command-{index}",
            episode_id=parent.episode_id,
            verb="apply",
            idempotency_key=f"seed-apply-key-{index}",
            payload={"request_id": f"seed-apply-request-{index}"},
            apply_admission_limit=AUTO_RESEARCH_APPLY_MAX_PER_TURN,
        )

    barrier = threading.Barrier(2)

    def start(index: int):
        command_id = f"racing-apply-command-{index}"
        content = f'{{"candidate":{index}}}'
        snapshot = AutoResearchCommandFileRecord(
            command_id=command_id,
            episode_id=parent.episode_id,
            operation_id=root.operation_id,
            kind="apply",
            filename="patch.json",
            sha256=hashlib.sha256(content.encode()).hexdigest(),
            content=content,
            created_at=store.now(),
        )
        barrier.wait()
        return store.start_agent_command(
            operation_id=root.operation_id,
            command_id=command_id,
            episode_id=parent.episode_id,
            verb="apply",
            idempotency_key=f"racing-apply-key-{index}",
            payload={"request_id": f"racing-apply-request-{index}"},
            file_snapshot=snapshot,
            apply_admission_limit=AUTO_RESEARCH_APPLY_MAX_PER_TURN,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(start, index) for index in range(2)]
        invocations = [future.result() for future in futures]

    assert (
        store.auto_research_apply_admission_count(root.operation_id)
        == AUTO_RESEARCH_APPLY_MAX_PER_TURN
    )
    admitted = [item for item in invocations if item.start_payload["apply_admitted"] is True]
    refused = [item for item in invocations if item.start_payload["apply_admitted"] is False]
    assert len(admitted) == len(refused) == 1
    assert store.auto_research_command_file(admitted[0].command_id) is not None
    assert store.auto_research_command_file(refused[0].command_id) is None


def test_apply_results_are_immutable_and_ordered_per_turn(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    parent, root = _auto_parent(store)
    records = [
        AutoResearchApplyResultRecord(
            apply_id=f"apply-{index}",
            episode_id=parent.episode_id,
            operation_id=root.operation_id,
            patch_sha256=hashlib.sha256(str(index).encode()).hexdigest(),
            result={"disposition": disposition},
            created_at=f"2026-08-16T00:00:0{index}+00:00",
        )
        for index, disposition in enumerate(("applied", "valid_empty"), start=1)
    ]
    for record in records:
        store.save_auto_research_apply_result(record)

    assert store.auto_research_apply_results(root.operation_id) == records
    assert store.save_auto_research_apply_result(records[0]) == records[0]
    assert (
        store.save_auto_research_apply_result(
            records[0].model_copy(
                update={
                    "result": {"disposition": "invalid"},
                    "created_at": "2026-08-16T01:00:00+00:00",
                }
            )
        )
        == records[0]
    )
    with pytest.raises(ValueError, match="another durable result"):
        store.save_auto_research_apply_result(
            records[0].model_copy(update={"patch_sha256": hashlib.sha256(b"different").hexdigest()})
        )


def test_pending_experiment_replacement_terminal_outcomes_notify_atomically(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    parent, root = _auto_parent(store)
    cancelled_task = _experiment_task(
        store,
        str(uuid.uuid4()),
        parent.authorized_by,
        node_id="exp/cancelled",
    )
    cancelled_route = _experiment_route(
        store,
        parent,
        root,
        cancelled_task,
        state="pending",
        replaces_episode_id="old-cancelled",
    )
    store.reserve_auto_research_experiment_replacement(cancelled_route)

    cancelled = store.cancel_auto_research_experiment_replacement(
        cancelled_route.child_episode_id,
        diagnostic="The orchestrator cancelled replacement.",
    )

    assert cancelled.state == "cancelled"
    cancelled_notice = store.auto_research_lifecycle_notices(parent.episode_id)[0]
    assert cancelled_notice.source_kind == "experiment_replacement"
    assert cancelled_notice.source_event == "cancelled"

    failed_task = _experiment_task(
        store,
        str(uuid.uuid4()),
        parent.authorized_by,
        node_id="exp/failed",
    )
    failed_route = _experiment_route(
        store,
        parent,
        root,
        failed_task,
        state="pending",
        replaces_episode_id="old-failed",
    )
    store.reserve_auto_research_experiment_replacement(failed_route)
    failed = store.fail_auto_research_experiment_replacement(
        failed_route.child_episode_id,
        diagnostic="The node never became ready.",
    )

    assert failed.state == "cancelled"
    assert [
        (notice.source_id, notice.source_event)
        for notice in store.auto_research_lifecycle_notices(parent.episode_id)
    ] == [
        (cancelled_route.child_episode_id, "cancelled"),
        (failed_route.child_episode_id, "failed"),
    ]


def test_pending_experiment_replacement_activation_notifies_atomically(
    tmp_path,
    monkeypatch,
) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    parent, root = _auto_parent(store)
    direct_task = _experiment_task(
        store,
        str(uuid.uuid4()),
        parent.authorized_by,
        node_id="exp/direct",
    )
    store.create_experiment_episode_with_invocation(
        direct_task,
        auto_research_route=_experiment_route(store, parent, root, direct_task),
    )
    assert store.auto_research_lifecycle_notices(parent.episode_id) == []

    replacement_task = _experiment_task(
        store,
        str(uuid.uuid4()),
        parent.authorized_by,
        node_id="exp/replacement",
    )
    replacement_route = _experiment_route(
        store,
        parent,
        root,
        replacement_task,
        state="pending",
        replaces_episode_id="predecessor-episode",
    )
    store.reserve_auto_research_experiment_replacement(replacement_route)
    original_insert = store._insert_auto_research_lifecycle_notice

    def fail_advanced_notice(connection, notice):
        if notice.source_kind == "experiment_replacement" and notice.source_event == "advanced":
            raise RuntimeError("synthetic lifecycle failure")
        return original_insert(connection, notice)

    monkeypatch.setattr(store, "_insert_auto_research_lifecycle_notice", fail_advanced_notice)
    with pytest.raises(RuntimeError, match="synthetic lifecycle failure"):
        store.create_experiment_episode_with_invocation(
            replacement_task,
            auto_research_route=replacement_route.model_copy(update={"state": "running"}),
        )

    rolled_back = store.auto_research_child_experiment(replacement_task.episode_id)
    assert rolled_back is not None and rolled_back.state == "pending"
    assert store.episode(replacement_task.episode_id) is None
    assert store.auto_research_lifecycle_notices(parent.episode_id) == []

    monkeypatch.setattr(store, "_insert_auto_research_lifecycle_notice", original_insert)
    store.create_experiment_episode_with_invocation(
        replacement_task,
        auto_research_route=replacement_route.model_copy(update={"state": "running"}),
    )

    notices = store.auto_research_lifecycle_notices(parent.episode_id)
    assert len(notices) == 1
    notice = notices[0]
    assert (
        notice.source_kind,
        notice.source_id,
        notice.source_event,
        notice.source_attempt,
    ) == (
        "experiment_replacement",
        replacement_task.episode_id,
        "advanced",
        1,
    )
    assert notice.payload == {
        "episode_id": replacement_task.episode_id,
        "status": "running",
        "replaces_episode_id": "predecessor-episode",
    }

    with pytest.raises(ValueError, match="already in use"):
        store.create_experiment_episode_with_invocation(
            replacement_task,
            auto_research_route=replacement_route.model_copy(update={"state": "running"}),
        )
    assert store.auto_research_lifecycle_notices(parent.episode_id) == [notice]


def test_finish_blocker_query_reports_all_categories_without_mutation(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    parent, root = _auto_parent(store, ceiling=3)
    work_route, work_task = _work_pair(store, parent, root, worker_id="worker-one")
    store.create_auto_research_child_work(work_route, work_task)

    child_id = str(uuid.uuid4())
    child_task = _experiment_task(store, child_id, parent.authorized_by, node_id="exp/one")
    child_route = _experiment_route(store, parent, root, child_task)
    store.create_experiment_episode_with_invocation(
        child_task,
        auto_research_route=child_route,
    )
    store.fence_episode_ending(child_id, "completed")
    pending_id = str(uuid.uuid4())
    pending_task = _experiment_task(
        store,
        pending_id,
        parent.authorized_by,
        node_id="exp/two",
    )
    pending_route = _experiment_route(
        store,
        parent,
        root,
        pending_task,
        state="pending",
        replaces_episode_id="prior-episode",
    )
    store.reserve_auto_research_experiment_replacement(pending_route)
    notice = AutoResearchLifecycleNoticeRecord(
        notice_id="notice-one",
        episode_id=parent.episode_id,
        source_kind="worker",
        source_id="worker-old",
        source_event="settled",
        payload={"status": "failed"},
        created_at=store.now(),
    )
    store.record_auto_research_lifecycle_notice(notice)
    delivered = store.claim_auto_research_lifecycle_notices(
        parent.episode_id,
        root.operation_id,
    )
    assert [item.notice_id for item in delivered] == [notice.notice_id]
    assert delivered[0].state == "delivered"
    pending_notice = store.record_auto_research_lifecycle_notice(
        AutoResearchLifecycleNoticeRecord(
            notice_id="notice-pending",
            episode_id=parent.episode_id,
            source_kind="worker",
            source_id="worker-new",
            source_event="settled",
            payload={"status": "failed"},
            created_at=store.now(),
        )
    )
    admission = _admission(
        store,
        parent,
        admission_id="accepted-not-reflected",
        child_kind="work",
        child_id="future-worker",
    )
    store.record_auto_research_child_admission(admission)

    before = (
        store.auto_research_child_work(work_route.worker_id),
        store.auto_research_child_experiment(pending_id),
        store.auto_research_lifecycle_notices(parent.episode_id),
        store.auto_research_child_admission(admission.admission_id),
    )
    blockers = store.auto_research_finish_blockers(parent.episode_id)
    after = (
        store.auto_research_child_work(work_route.worker_id),
        store.auto_research_child_experiment(pending_id),
        store.auto_research_lifecycle_notices(parent.episode_id),
        store.auto_research_child_admission(admission.admission_id),
    )

    assert {blocker.kind for blocker in blockers} == {
        "spawned_work",
        "experiment_episode",
        "experiment_replacement",
        "lifecycle_notice",
        "child_admission",
    }
    experiment_blocker = next(
        blocker for blocker in blockers if blocker.kind == "experiment_episode"
    )
    lifecycle_blockers = [blocker for blocker in blockers if blocker.kind == "lifecycle_notice"]
    assert experiment_blocker.state == "wrapping_up"
    assert experiment_blocker.action == "wait for report settlement"
    assert [blocker.blocker_id for blocker in lifecycle_blockers] == [pending_notice.notice_id]
    lifecycle_blocker = lifecycle_blockers[0]
    assert lifecycle_blocker.state == "pending"
    assert lifecycle_blocker.action == ("inbox --key <key> --harvest or inbox --key <key> --clear")
    assert before == after


def test_guarded_finish_receipt_replays_exact_snapshot_and_new_key_sees_live_state(
    tmp_path,
) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    parent, root = _auto_parent(store, ceiling=2)
    notice = store.record_auto_research_lifecycle_notice(
        AutoResearchLifecycleNoticeRecord(
            notice_id="notice-exact-snapshot",
            episode_id=parent.episode_id,
            source_kind="worker",
            source_id="worker-one",
            source_event="settled",
            payload={"status": "failed"},
            created_at=store.now(),
        )
    )
    first_id = str(uuid.uuid4())

    first = store.guard_auto_research_finish(
        parent.episode_id,
        effect_id=first_id,
        actor_operation_id=root.operation_id,
    )
    store.clear_auto_research_lifecycle_notices(
        parent.episode_id,
        acknowledged_by=root.operation_id,
    )
    replay = store.guard_auto_research_finish(
        parent.episode_id,
        effect_id=first_id,
        actor_operation_id=root.operation_id,
    )
    completed = store.guard_auto_research_finish(
        parent.episode_id,
        effect_id=str(uuid.uuid4()),
        actor_operation_id=root.operation_id,
    )

    assert first == replay == store.auto_research_finish_receipt(first_id)
    assert first.disposition == "blocked"
    assert first.blocker_count == 1
    assert first.result["blockers"] == [
        {
            "kind": "lifecycle_notice",
            "blocker_id": notice.notice_id,
            "state": "pending",
            "action": "inbox --key <key> --harvest or inbox --key <key> --clear",
        }
    ]
    assert completed.disposition == "completed"
    assert completed.result == {
        "episode_id": parent.episode_id,
        "status": "wrapping_up",
        "ending": "completed",
    }
    with pytest.raises(ValueError, match="another command"):
        store.guard_auto_research_finish(
            parent.episode_id,
            effect_id=first_id,
            actor_operation_id="another-actor",
        )


def test_guarded_finish_and_child_admission_are_one_serializable_decision(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    parent, root = _auto_parent(store, ceiling=2)
    route, task = _work_pair(store, parent, root, worker_id="racing-worker")
    barrier = threading.Barrier(2)

    def finish() -> tuple[str, object]:
        barrier.wait()
        receipt = store.guard_auto_research_finish(
            parent.episode_id,
            effect_id=str(uuid.uuid4()),
            actor_operation_id=root.operation_id,
        )
        episode = store.episode(parent.episode_id)
        assert episode is not None
        return (receipt.disposition, episode.status)

    def admit() -> tuple[str, object]:
        barrier.wait()
        try:
            store.create_auto_research_child_work(route, task)
        except EpisodeNotRunning:
            return "rejected", None
        return "admitted", None

    with ThreadPoolExecutor(max_workers=2) as executor:
        finish_result = executor.submit(finish)
        admit_result = executor.submit(admit)
        outcomes = {finish_result.result()[0], admit_result.result()[0]}

    assert outcomes in ({"completed", "rejected"}, {"blocked", "admitted"})
    final = store.episode(parent.episode_id)
    assert final is not None
    if final.ending == "completed":
        assert store.auto_research_child_work(route.worker_id) is None
        assert final.status == "wrapping_up"
    else:
        assert final.status == "running"
        assert store.auto_research_child_work(route.worker_id) is not None


def test_guarded_finish_and_command_start_admission_are_one_serializable_decision(
    tmp_path,
) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    parent, root = _auto_parent(store, ceiling=2)
    admission = _admission(
        store,
        parent,
        admission_id="racing-command-admission",
        child_kind="work",
        child_id="racing-command-worker",
    )
    barrier = threading.Barrier(2)

    def finish() -> str:
        barrier.wait()
        receipt = store.guard_auto_research_finish(
            parent.episode_id,
            effect_id="racing-finish-effect",
            actor_operation_id=root.operation_id,
        )
        return receipt.disposition

    def admit() -> str:
        barrier.wait()
        try:
            store.start_agent_command(
                operation_id=root.operation_id,
                command_id="racing-spawn-command",
                episode_id=parent.episode_id,
                verb="spawn",
                idempotency_key="racing-spawn-key",
                payload={
                    "request_id": "a" * 32,
                    "arguments": {
                        "seat_node_id": "blk/race",
                        "instruction_file": "worker.md",
                    },
                    "planned_worker_id": admission.child_id,
                },
                child_admission=admission,
            )
        except EpisodeNotRunning:
            return "rejected"
        return (
            "accepted"
            if store.auto_research_child_admission(admission.admission_id) is not None
            else "rejected"
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        finish_result = executor.submit(finish)
        admission_result = executor.submit(admit)
        outcomes = {finish_result.result(), admission_result.result()}

    stored_admission = store.auto_research_child_admission(admission.admission_id)
    command = store.agent_command("racing-spawn-command")
    if outcomes == {"completed", "rejected"}:
        assert stored_admission is None
        assert command is not None and command.exited_at is None
    else:
        assert outcomes == {"blocked", "accepted"}
        assert stored_admission is not None and stored_admission.state == "accepted"
        assert command is not None and command.exited_at is None


def test_routed_experiment_recovery_respects_parent_stop_fence(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    parent, root = _auto_parent(store, ceiling=2)
    child_id = str(uuid.uuid4())
    task = _experiment_task(store, child_id, parent.authorized_by, node_id="exp/recovery")
    route = _experiment_route(store, parent, root, task)
    store.create_experiment_episode_with_invocation(task, auto_research_route=route)
    store.fail_agent_task(task.operation_id, "network failed")
    store.request_episode_stop(parent.episode_id)
    recovery = _experiment_task(
        store,
        child_id,
        parent.authorized_by,
        node_id="exp/recovery",
        parent_operation_id=task.operation_id,
        attempt=2,
    )

    with pytest.raises(EpisodeNotRunning, match="Auto-research episode"):
        store.create_experiment_recovery_task(recovery)

    assert store.auto_research_experiment_allowance(parent.episode_id).used == 1
    assert store.agent_task(recovery.operation_id) is None


def test_project_deletion_removes_every_auto_research_child_registry(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    parent, root = _auto_parent(store, ceiling=3)
    work_route, work_task = _work_pair(store, parent, root, worker_id="worker-delete")
    store.create_auto_research_child_work(work_route, work_task)
    store.complete_agent_task(work_task.operation_id, applied_revision=None, result={})

    child_id = str(uuid.uuid4())
    child_task = _experiment_task(
        store,
        child_id,
        parent.authorized_by,
        node_id="exp/delete",
    )
    child_route = _experiment_route(store, parent, root, child_task)
    store.create_experiment_episode_with_invocation(
        child_task,
        auto_research_route=child_route,
    )
    store.complete_agent_task(child_task.operation_id, applied_revision=None, result={})
    store.complete_agent_task(root.operation_id, applied_revision=None, result={})

    pending_task = _experiment_task(
        store,
        str(uuid.uuid4()),
        parent.authorized_by,
        node_id="exp/pending-delete",
    )
    store.reserve_auto_research_experiment_replacement(
        _experiment_route(
            store,
            parent,
            root,
            pending_task,
            state="pending",
            replaces_episode_id=child_id,
        )
    )
    store.record_auto_research_child_admission(
        _admission(
            store,
            parent,
            admission_id="delete-admission",
            child_kind="work",
            child_id="future-delete-worker",
        )
    )
    store.record_auto_research_lifecycle_notice(
        AutoResearchLifecycleNoticeRecord(
            notice_id="delete-notice",
            episode_id=parent.episode_id,
            source_kind="worker",
            source_id=work_route.worker_id,
            source_event="settled",
            payload={"status": "succeeded"},
            created_at=store.now(),
        )
    )
    apply_record = AutoResearchApplyResultRecord(
        apply_id="delete-apply",
        episode_id=parent.episode_id,
        operation_id=root.operation_id,
        patch_sha256=hashlib.sha256(b"delete").hexdigest(),
        result={"disposition": "applied"},
        created_at=store.now(),
    )
    store.save_auto_research_apply_result(apply_record)
    snapshot_text = '{"operations":[]}'
    snapshot = AutoResearchCommandFileRecord(
        command_id="delete-command",
        episode_id=parent.episode_id,
        operation_id=root.operation_id,
        kind="apply",
        filename="patch.json",
        sha256=hashlib.sha256(snapshot_text.encode()).hexdigest(),
        content=snapshot_text,
        created_at=store.now(),
    )
    store.start_agent_command(
        operation_id=root.operation_id,
        command_id=snapshot.command_id,
        episode_id=parent.episode_id,
        verb="apply",
        idempotency_key="delete-key",
        payload={"filename": snapshot.filename},
        file_snapshot=snapshot,
    )
    store.process_auto_research_lifecycle_inbox(
        parent.episode_id,
        effect_id="delete-inbox",
        mode="harvest",
        acknowledged_by=root.operation_id,
        limit=1,
    )
    finish_receipt = store.guard_auto_research_finish(
        parent.episode_id,
        effect_id="delete-finish",
        actor_operation_id=root.operation_id,
    )
    assert finish_receipt.disposition == "blocked"

    counts = store.delete_project_records(parent.project_id)

    assert "auto_research_child_work" not in counts
    assert store.auto_research_child_work(work_route.worker_id) is None
    assert store.auto_research_child_experiment(child_id) is None
    assert store.auto_research_child_admission("delete-admission") is None
    assert store.auto_research_lifecycle_notices(parent.episode_id) == []
    assert store.auto_research_inbox_receipt("delete-inbox") is None
    assert store.auto_research_finish_receipt("delete-finish") is None
    assert store.auto_research_apply_result(apply_record.apply_id) is None
    assert store.auto_research_command_file(snapshot.command_id) is None
    assert store.project(parent.project_id) is None


def test_project_identity_migration_moves_child_registries_and_deletion_cleans_them(
    tmp_path,
) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    parent, work_route, experiment_route, admission = _completed_child_project_rows(store)
    canonical_project_id = str(uuid.uuid4())

    store.migrate_project_identity("project", canonical_project_id, store.space_id)

    migrated_work = store.auto_research_child_work(work_route.worker_id)
    migrated_experiment = store.auto_research_child_experiment(experiment_route.child_episode_id)
    migrated_admission = store.auto_research_child_admission(admission.admission_id)
    assert migrated_work is not None and migrated_work.project_id == canonical_project_id
    assert (
        migrated_experiment is not None and migrated_experiment.project_id == canonical_project_id
    )
    assert migrated_admission is not None and migrated_admission.project_id == canonical_project_id
    with store.connection() as connection:
        for table in (
            "auto_research_child_work",
            "auto_research_child_experiments",
            "auto_research_child_admissions",
        ):
            assert (
                connection.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE project_id = ?",
                    ("project",),
                ).fetchone()[0]
                == 0
            )

    store.delete_project_records(canonical_project_id)

    assert store.auto_research_child_work(work_route.worker_id) is None
    assert store.auto_research_child_experiment(experiment_route.child_episode_id) is None
    assert store.auto_research_child_admission(admission.admission_id) is None
    assert store.project(canonical_project_id) is None
    assert store.project(parent.project_id) is None


def test_legacy_project_data_migration_moves_child_registries_idempotently(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _, work_route, experiment_route, admission = _completed_child_project_rows(store)
    canonical_project_id = str(uuid.uuid4())
    with store.connection() as connection:
        connection.execute("DELETE FROM projects WHERE project_id = 'project'")
    store.upsert_project(
        ProjectRecord(
            project_id=canonical_project_id,
            home_space_id=store.space_id,
            locator="/tmp/canonical-project/research.yaml",
            name="canonical-project",
            state_location="/tmp/canonical-project/.research",
            state_remote=False,
            added_at=store.now(),
        )
    )

    store.migrate_legacy_project_data("project", canonical_project_id)
    store.migrate_legacy_project_data("project", canonical_project_id)

    migrated_work = store.auto_research_child_work(work_route.worker_id)
    migrated_experiment = store.auto_research_child_experiment(experiment_route.child_episode_id)
    migrated_admission = store.auto_research_child_admission(admission.admission_id)
    assert migrated_work is not None and migrated_work.project_id == canonical_project_id
    assert (
        migrated_experiment is not None and migrated_experiment.project_id == canonical_project_id
    )
    assert migrated_admission is not None and migrated_admission.project_id == canonical_project_id
    with store.connection() as connection:
        for table in (
            "auto_research_child_work",
            "auto_research_child_experiments",
            "auto_research_child_admissions",
        ):
            assert (
                connection.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE project_id = ?",
                    ("project",),
                ).fetchone()[0]
                == 0
            )
