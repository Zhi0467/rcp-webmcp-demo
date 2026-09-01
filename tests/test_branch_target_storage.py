from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from rcp.core.authority import AgentDispatchAuthority, AgentDispatchScope
from rcp.core.transition_models import GraphHeadRef, GraphTargetRef
from rcp.runs.auto_research import AutoResearchRunRequest
from rcp.runs.episodes.wrapup import EpisodeWrapupSpec, begin_episode_report_wrapup
from rcp.storage import (
    AgentTaskRecord,
    AppStore,
    AutoResearchStateRecord,
    EpisodeRecord,
    ProjectRecord,
    WatcherContinuation,
    WatcherRecord,
)

from .helpers import fabricated_authorizer


def _store(tmp_path: Path) -> AppStore:
    store = AppStore(tmp_path / "rcp.sqlite3")
    store.upsert_project(
        ProjectRecord(
            project_id="project",
            locator=str(tmp_path / "research.yaml"),
            name="Project",
            state_location=str(tmp_path / ".research"),
            state_remote=False,
            added_at=store.now(),
        )
    )
    return store


def _orchestrator_authority(episode_id: str) -> AgentDispatchAuthority:
    return AgentDispatchAuthority(
        profile="orchestrator",
        task_contract="orchestrate",
        scope=AgentDispatchScope(
            run_truth_scope=["repo-a"],
            episode_id=episode_id,
            patch_kind="work",
        ),
    )


def _worker_authority(episode_id: str) -> AgentDispatchAuthority:
    return AgentDispatchAuthority(
        profile="ordinary",
        task_contract="work_auto",
        scope=AgentDispatchScope(
            run_truth_scope=["repo-a"],
            episode_id=episode_id,
            patch_kind="work",
        ),
    )


def _create_auto_episode(
    store: AppStore,
    *,
    episode_id: str | None = None,
    stage_root: str | None = None,
) -> tuple[EpisodeRecord, AgentTaskRecord]:
    episode_id = episode_id or str(uuid.uuid4())
    operation_id = f"root-{episode_id}"
    target = GraphTargetRef(kind="branch", branch_id=episode_id)
    base = GraphHeadRef(revision=0)
    authorizer = fabricated_authorizer("Branch owner")
    now = store.now()
    episode = EpisodeRecord(
        episode_id=episode_id,
        project_id="project",
        mode="auto_research",
        graph_target=target,
        graph_base_head=base,
        status="queued",
        invocation_ceiling=4,
        authorized_by=authorizer,
        created_at=now,
        updated_at=now,
    )
    task = AgentTaskRecord(
        operation_id=operation_id,
        project_id="project",
        episode_id=episode_id,
        graph_target=target,
        kind="auto_research",
        status="queued",
        request=AutoResearchRunRequest(
            episode_id=episode_id,
            role="orchestrator",
            actor_operation_id=operation_id,
            provider="codex",
            model="",
            reasoning="medium",
            run_on="local",
            run_truth_scope=["repo-a"],
        ).model_dump(mode="json"),
        created_at=now,
        updated_at=now,
        status_message="Queued",
        native_session_id="branch-session" if stage_root is not None else None,
        stage_root=stage_root,
        authorized_by=authorizer,
        dispatch_authority=_orchestrator_authority(episode_id),
    )
    return store.create_auto_research_episode_with_root_task(
        episode,
        AutoResearchStateRecord(
            episode_id=episode_id,
            created_at=now,
            updated_at=now,
        ),
        task,
    )


def _watcher(
    store: AppStore,
    episode: EpisodeRecord,
    root: AgentTaskRecord,
    *,
    watcher_id: str,
    status: str = "active",
) -> WatcherRecord:
    return WatcherRecord(
        watcher_id=watcher_id,
        project_id=episode.project_id,
        origin_operation_id=root.operation_id,
        origin_task_kind="auto_research",
        chat_id="shared-actor",
        episode_id=episode.episode_id,
        graph_target=episode.graph_target,
        execution_host="",
        check_command="true",
        log_path=f"/tmp/{watcher_id}.log",
        cwd="/tmp",
        continuation=WatcherContinuation(
            provider="codex",
            run_on="local",
            run_truth_scope=["repo-a"],
            patch_kind="work",
        ),
        status=status,
        created_at=store.now(),
        completed_at=store.now() if status == "completed" else None,
    )


def _merge_task(
    store: AppStore,
    episode: EpisodeRecord,
    operation_id: str,
    *,
    graph_target: GraphTargetRef | None = None,
    authority: AgentDispatchAuthority | None = None,
) -> AgentTaskRecord:
    now = store.now()
    return AgentTaskRecord(
        operation_id=operation_id,
        project_id=episode.project_id,
        episode_id=episode.episode_id,
        graph_target=graph_target or episode.graph_target,
        kind="branch_merge",
        status="queued",
        request={"branch_id": episode.episode_id},
        created_at=now,
        updated_at=now,
        status_message="Queued for human-authorized graph merge",
        authorized_by=episode.authorized_by,
        dispatch_authority=authority or _orchestrator_authority(episode.episode_id),
    )


def test_branch_target_round_trips_through_episode_task_watcher_and_report(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    stage = tmp_path / "branch-stage"
    stage.mkdir()
    episode, root = _create_auto_episode(store, stage_root=str(stage))
    watcher = _watcher(store, episode, root, watcher_id="branch-watcher")
    store.create_watchers([watcher])

    stored_episode = store.episode(episode.episode_id)
    stored_root = store.agent_task(root.operation_id)
    stored_watcher = store.watcher(watcher.watcher_id)
    assert stored_episode is not None
    assert stored_root is not None
    assert stored_watcher is not None
    assert stored_episode.graph_target == stored_root.graph_target == stored_watcher.graph_target
    assert stored_episode.graph_base_head == GraphHeadRef(revision=0)

    with pytest.raises(ValueError, match="watcher identity conflicts"):
        store._validate_idempotent_watcher(
            stored_watcher,
            watcher.model_copy(update={"graph_target": GraphTargetRef()}),
        )

    store.complete_agent_task(root.operation_id, applied_revision=None, result={})
    admission = begin_episode_report_wrapup(
        store,
        EpisodeWrapupSpec(
            episode_id=episode.episode_id,
            ending="completed",
            partial=False,
            continuation_operation_id=root.operation_id,
            receipt={"result": "bounded"},
        ),
    )
    assert admission.task is not None
    assert admission.task.graph_target == episode.graph_target
    assert admission.episode.graph_base_head == episode.graph_base_head


def test_cross_target_auto_research_continuation_is_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    episode, root = _create_auto_episode(store)
    worker_id = str(uuid.uuid4())
    other_target = GraphTargetRef(kind="branch", branch_id=str(uuid.uuid4()))
    now = store.now()
    continuation = AgentTaskRecord(
        operation_id=worker_id,
        project_id=episode.project_id,
        episode_id=episode.episode_id,
        graph_target=other_target,
        kind="auto_research",
        status="queued",
        request=AutoResearchRunRequest(
            episode_id=episode.episode_id,
            role="worker",
            actor_operation_id=worker_id,
            control_node_id="blk/worker-seat",
            run_truth_scope=["repo-a"],
        ).model_dump(mode="json"),
        created_at=now,
        updated_at=now,
        status_message="Queued",
        parent_operation_id=root.operation_id,
        authorized_by=episode.authorized_by,
        dispatch_authority=_worker_authority(episode.episode_id),
    )

    with pytest.raises(ValueError, match="cannot change its graph target"):
        store.create_auto_research_agent_task(continuation, role="worker")
    assert store.agent_task(worker_id) is None


def test_main_chat_cannot_reuse_a_branch_bound_conversation_or_session(tmp_path: Path) -> None:
    store = _store(tmp_path)
    episode, _root = _create_auto_episode(store)
    chat_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    stage = tmp_path / "branch-chat"
    stage.mkdir()
    now = store.now()
    branch_task = AgentTaskRecord(
        operation_id="branch-chat",
        project_id=episode.project_id,
        episode_id=episode.episode_id,
        graph_target=episode.graph_target,
        kind="node_chat",
        status="succeeded",
        request={
            "chat_id": chat_id,
            "session_id": None,
            "patch_kind": "work",
            "mode": "work",
        },
        created_at=now,
        updated_at=now,
        status_message="Stored branch conversation.",
        native_session_id=session_id,
        stage_root=str(stage),
    )
    with store.connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        store._insert_agent_task(connection, branch_task, continuation_cause="fresh")

    def main_task(operation_id: str, *, requested_chat_id: str, requested_session: str | None):
        return AgentTaskRecord(
            operation_id=operation_id,
            project_id=episode.project_id,
            kind="node_chat",
            status="queued",
            request={
                "chat_id": requested_chat_id,
                "session_id": requested_session,
                "patch_kind": "work",
                "mode": "work",
            },
            created_at=now,
            updated_at=now,
            status_message="Queued main conversation.",
        )

    with pytest.raises(ValueError, match="another graph target"):
        store.create_agent_task(
            main_task("main-same-chat", requested_chat_id=chat_id, requested_session=None)
        )
    with pytest.raises(ValueError, match="another conversation or graph target"):
        store.create_agent_task(
            main_task(
                "main-same-session",
                requested_chat_id=str(uuid.uuid4()),
                requested_session=session_id,
            )
        )

    assert store.agent_task("main-same-chat") is None
    assert store.agent_task("main-same-session") is None


def test_completed_watcher_delivery_groups_do_not_cross_graph_targets(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first, first_root = _create_auto_episode(store)
    first_watcher = _watcher(
        store,
        first,
        first_root,
        watcher_id="first-branch-watcher",
        status="completed",
    )
    store.create_watchers([first_watcher])
    store.complete_agent_task(first_root.operation_id, applied_revision=None, result={})
    store.mark_episode_stop_skipped(first.episode_id)

    second, second_root = _create_auto_episode(store)
    second_watcher = _watcher(
        store,
        second,
        second_root,
        watcher_id="second-branch-watcher",
        status="completed",
    )
    store.create_watchers([second_watcher])

    groups = store.completed_watcher_groups()
    assert {tuple(item.watcher_id for item in group) for group in groups} == {
        (first_watcher.watcher_id,),
        (second_watcher.watcher_id,),
    }
    assert {group[0].graph_target.key for group in groups} == {
        first.graph_target.key,
        second.graph_target.key,
    }


def test_branch_merge_task_requires_ended_quiescent_branch_and_exact_authority(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    episode, root = _create_auto_episode(store)

    with pytest.raises(ValueError, match="quiescent"):
        store.create_branch_merge_task(_merge_task(store, episode, "merge-active"))

    store.complete_agent_task(root.operation_id, applied_revision=None, result={})
    with pytest.raises(ValueError, match="ended Auto-research branch"):
        store.create_branch_merge_task(_merge_task(store, episode, "merge-not-ended"))

    store.mark_episode_stop_skipped(episode.episode_id)

    with pytest.raises(ValueError, match="visible attributed branch root"):
        store.create_branch_merge_task(
            _merge_task(store, episode, "merge-main", graph_target=GraphTargetRef())
        )

    other_target = GraphTargetRef(kind="branch", branch_id=str(uuid.uuid4()))
    with pytest.raises(ValueError, match="ended Auto-research branch"):
        store.create_branch_merge_task(
            _merge_task(store, episode, "merge-cross-target", graph_target=other_target)
        )

    wrong_authority = AgentDispatchAuthority(
        profile="ordinary",
        task_contract="work_auto",
        scope=AgentDispatchScope(
            run_truth_scope=["repo-a"],
            episode_id=episode.episode_id,
            patch_kind="work",
        ),
    )
    with pytest.raises(ValueError, match="exact graph-only orchestrator authority"):
        store.create_branch_merge_task(
            _merge_task(store, episode, "merge-wrong-authority", authority=wrong_authority)
        )

    accepted = store.create_branch_merge_task(_merge_task(store, episode, "merge-accepted"))
    assert accepted.graph_target == episode.graph_target
    assert accepted.dispatch_authority == _orchestrator_authority(episode.episode_id)
    merge_authority = store.agent_task_authority(episode.project_id, accepted.operation_id)
    assert merge_authority.apply_target == GraphTargetRef()

    replay = _merge_task(store, episode, accepted.operation_id).model_copy(
        update={
            "created_at": accepted.created_at,
            "updated_at": accepted.updated_at,
        }
    )
    replayed = store.create_branch_merge_task(replay)
    assert replayed.operation_id == accepted.operation_id

    with pytest.raises(ValueError, match="another merge is already active"):
        store.create_branch_merge_task(_merge_task(store, episode, "merge-duplicate"))
