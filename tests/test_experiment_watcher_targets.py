from __future__ import annotations

import uuid

import pytest

from rcp.core.models import AuthorizedHuman
from rcp.core.transition_models import GraphHeadRef, GraphTargetRef
from rcp.runs.experiment_loop import experiment_watcher_output_name
from rcp.storage import AgentTaskRecord, AppStore, WatcherContinuation, WatcherRecord

_PROJECT_ID = "project"
_CONTROL_NODE_ID = "exp/shared"


def _identity(store: AppStore) -> AuthorizedHuman:
    owner = store.local_owner
    assert owner is not None
    return AuthorizedHuman(
        space_id=store.space_id,
        user_id=owner.user_id,
        display_name=owner.display_name or "Test researcher",
    )


def _loop_task(
    store: AppStore,
    operation_id: str,
    episode_id: str,
    *,
    invocation: int = 1,
    watcher_ids: list[str] | None = None,
    graph_target: GraphTargetRef | None = None,
) -> AgentTaskRecord:
    target = graph_target or GraphTargetRef()
    now = store.now()
    ids = list(watcher_ids or [])
    return AgentTaskRecord(
        operation_id=operation_id,
        project_id=_PROJECT_ID,
        episode_id=episode_id,
        graph_target=target,
        kind="node_chat",
        status="queued",
        request={
            "chat_id": f"chat-{episode_id}",
            "node_id": _CONTROL_NODE_ID,
            "provider": "codex",
            "model": "gpt-5",
            "reasoning": "medium",
            "run_on": "laptop",
            "run_truth_scope": ["state"],
            "mode": "work",
            "trigger": "watcher" if ids else "experiment_run",
            "patch_kind": "experiment_loop",
            "control_node_id": _CONTROL_NODE_ID,
            "control_revision": 0,
            "control_episode_id": episode_id,
            "control_invocation": invocation,
            "control_invocation_ceiling": 3,
            "control_decision_bundle": [],
            "control_completion_criteria": [],
            "workflow_ids": [],
            "skill_ids": [],
            "invoked_workflow_ids": [],
            "invoked_skill_ids": [],
            "resolved_skill_packages": [],
            "watcher_ids": ids,
        },
        created_at=now,
        updated_at=now,
        status_message="Queued loop invocation.",
        authorized_by=_identity(store),
    )


def _retarget_episode(
    store: AppStore,
    episode_id: str,
    graph_target: GraphTargetRef,
) -> None:
    if graph_target.kind == "main":
        return
    with store.connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            UPDATE episodes
            SET graph_target_json = ?, graph_base_head_json = ?
            WHERE episode_id = ?
            """,
            (
                graph_target.model_dump_json(),
                GraphHeadRef(revision=0).model_dump_json(),
                episode_id,
            ),
        )
        connection.execute(
            "UPDATE graph_runs SET graph_target_json = ? WHERE episode_id = ?",
            (graph_target.model_dump_json(), episode_id),
        )


def _create_episode(
    store: AppStore,
    operation_id: str,
    graph_target: GraphTargetRef,
) -> tuple[str, AgentTaskRecord]:
    episode_id = str(uuid.uuid4())
    # The public child-Experiment creator normally receives branch identity from
    # its Auto-research parent. This storage-focused fixture retargets the fully
    # created rows atomically so it can exercise both directions without building
    # unrelated orchestration state.
    root = _loop_task(store, operation_id, episode_id)
    store.create_experiment_episode_with_invocation(root)
    _retarget_episode(store, episode_id, graph_target)
    stored = store.agent_task(operation_id)
    assert stored is not None and stored.graph_target == graph_target
    return episode_id, stored


def _watcher(
    store: AppStore,
    watcher_id: str,
    root: AgentTaskRecord,
    *,
    status: str = "completed",
) -> WatcherRecord:
    episode_id = root.episode_id
    assert episode_id is not None
    continuation = WatcherContinuation(
        provider="codex",
        model="gpt-5",
        reasoning="medium",
        run_on="laptop",
        run_truth_scope=["state"],
        patch_kind="experiment_loop",
        control_node_id=_CONTROL_NODE_ID,
        control_revision=0,
        control_episode_id=episode_id,
        control_invocation=1,
        control_invocation_ceiling=3,
        control_decision_bundle=[],
        control_completion_criteria=[],
    )
    created_at = store.now()
    return WatcherRecord(
        watcher_id=watcher_id,
        project_id=_PROJECT_ID,
        origin_operation_id=root.operation_id,
        origin_task_kind="node_chat",
        chat_id=str(root.request["chat_id"]),
        node_id=_CONTROL_NODE_ID,
        episode_id=episode_id,
        graph_target=root.graph_target,
        execution_host="",
        check_command="true",
        log_path=f"/tmp/{watcher_id}.log",
        cwd="/tmp",
        continuation=continuation,
        status=status,
        created_at=created_at,
        completed_at=created_at if status == "completed" else None,
    )


def _terminalize_for_new_admission(store: AppStore, episode_id: str) -> None:
    now = store.now()
    with store.connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            UPDATE episodes
            SET status = 'completed', ending = 'completed', updated_at = ?, ended_at = ?
            WHERE episode_id = ?
            """,
            (now, now, episode_id),
        )


def _watcher_snapshot(store: AppStore, graph_target: GraphTargetRef) -> str:
    with store.connection() as connection:
        return store._experiment_watcher_snapshot_token(  # noqa: SLF001
            connection,
            _PROJECT_ID,
            _CONTROL_NODE_ID,
            graph_target,
        )


def test_same_node_watcher_snapshot_and_file_identity_are_target_local(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    main = GraphTargetRef()
    branch = GraphTargetRef(kind="branch", branch_id="branch-owner")
    main_root = _loop_task(store, "main-root", str(uuid.uuid4()), graph_target=main)
    _branch_episode_id, branch_root = _create_episode(store, "branch-root", branch)

    empty_main = _watcher_snapshot(store, main)
    empty_branch = _watcher_snapshot(store, branch)
    assert empty_main == empty_branch

    store.create_watchers([_watcher(store, "main-watcher", main_root, status="active")])
    main_only = _watcher_snapshot(store, main)
    assert main_only != empty_main
    assert _watcher_snapshot(store, branch) == empty_branch

    store.create_watchers([_watcher(store, "branch-watcher", branch_root, status="active")])
    assert _watcher_snapshot(store, main) == main_only
    assert _watcher_snapshot(store, branch) != empty_branch
    assert experiment_watcher_output_name(_CONTROL_NODE_ID, main) != experiment_watcher_output_name(
        _CONTROL_NODE_ID,
        branch,
    )


@pytest.mark.parametrize("watcher_target_kind", ["main", "branch"])
def test_newer_other_target_episode_cannot_claim_or_adopt_completed_watcher(
    tmp_path,
    watcher_target_kind: str,
) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    branch = GraphTargetRef(kind="branch", branch_id="branch-owner")
    watcher_target = GraphTargetRef() if watcher_target_kind == "main" else branch
    newer_target = branch if watcher_target_kind == "main" else GraphTargetRef()

    old_episode_id, old_root = _create_episode(store, "old-root", watcher_target)
    store.complete_agent_task(old_root.operation_id, applied_revision=None, result={})
    pending = _watcher(store, "pending-old-target", old_root)
    store.create_watchers([pending])
    _terminalize_for_new_admission(store, old_episode_id)

    newer_episode_id, newer_root = _create_episode(store, "newer-root", newer_target)
    assert (
        store.experiment_loop_runtime(
            _PROJECT_ID,
            _CONTROL_NODE_ID,
        ).episode_id
        == newer_episode_id
    )
    exact_old = store.experiment_loop_runtime_for_target(
        _PROJECT_ID,
        _CONTROL_NODE_ID,
        watcher_target,
    )
    exact_new = store.experiment_loop_runtime_for_target(
        _PROJECT_ID,
        _CONTROL_NODE_ID,
        newer_target,
    )
    assert exact_old.episode_id == old_episode_id
    assert exact_old.watcher_completion_pending is True
    assert exact_new.episode_id == newer_episode_id
    assert exact_new.watcher_completion_pending is False
    group = store.completed_experiment_watcher_group(
        _PROJECT_ID,
        _CONTROL_NODE_ID,
        graph_target=watcher_target,
    )
    assert group is not None and [item.watcher_id for item in group] == [pending.watcher_id]
    assert (
        store.completed_experiment_watcher_group(
            _PROJECT_ID,
            _CONTROL_NODE_ID,
            graph_target=newer_target,
        )
        is None
    )

    forged = _loop_task(
        store,
        "cross-target-wake",
        newer_episode_id,
        invocation=2,
        watcher_ids=[pending.watcher_id],
        graph_target=newer_target,
    )
    with pytest.raises(ValueError, match="different graph targets"):
        store.create_experiment_watcher_invocation(forged, [pending.watcher_id])

    assert store.agent_task(forged.operation_id) is None
    assert store.watcher(pending.watcher_id).notified is False
    assert store.episode(newer_episode_id).invocations_used == 1
    assert store.agent_task(newer_root.operation_id).applied_revision is None


def test_exact_branch_stop_mutates_only_selected_episode_and_target_watchers(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    branch = GraphTargetRef(kind="branch", branch_id="branch-owner")
    branch_episode_id, branch_root = _create_episode(store, "branch-root", branch)
    store.complete_agent_task(branch_root.operation_id, applied_revision=None, result={})
    branch_watcher = _watcher(store, "branch-watcher", branch_root, status="active")
    store.create_watchers([branch_watcher])
    _terminalize_for_new_admission(store, branch_episode_id)

    main_episode_id, main_root = _create_episode(store, "newer-main-root", GraphTargetRef())
    main_watcher = _watcher(store, "main-watcher", main_root, status="active")
    store.create_watchers([main_watcher])
    # Recreate the narrow race the exact API defends: the selected branch was
    # validated first, while a newer same-node episode now exists on main.
    with store.connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("DROP INDEX episodes_one_live_experiment_control")
        connection.execute(
            """
            UPDATE episodes
            SET status = 'running', ending = NULL, ended_at = NULL
            WHERE episode_id = ?
            """,
            (branch_episode_id,),
        )

    stopped = store.request_experiment_loop_stop(
        _PROJECT_ID,
        _CONTROL_NODE_ID,
        episode_id=branch_episode_id,
        graph_target=branch,
    )

    assert stopped is not None and stopped.episode_id == branch_episode_id
    assert stopped.graph_target == branch
    assert stopped.stop_requested_at is not None
    assert stopped.stop_settled_at is not None
    main = store.episode(main_episode_id)
    assert main is not None and main.status == "running"
    assert main.stop_requested_at is None and main.stop_settled_at is None
    assert store.watcher(branch_watcher.watcher_id).status == "stopped"
    assert store.watcher(main_watcher.watcher_id).status == "active"
    assert store.agent_task(main_root.operation_id).status == "queued"
