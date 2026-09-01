from __future__ import annotations

import json
import uuid

import pytest
from pydantic import ValidationError

from rcp.core.authority import AgentDispatchAuthority, AgentDispatchScope
from rcp.core.models import AuthorizedHuman
from rcp.core.transition_models import GraphHeadRef, GraphTargetRef
from rcp.runs.auto_research import (
    AutoResearchRunRequest,
    AutoResearchStartRequest,
    auto_research_completion_signal,
    auto_research_wrapup_spec,
    request_auto_research_stop,
    settle_auto_research_stop,
)
from rcp.storage import (
    AgentTaskRecord,
    AppStore,
    AutoResearchStateRecord,
    EpisodeRecord,
    GraphWatcherRecord,
    ProjectRecord,
    WatcherContinuation,
)


def _authority(episode_id: str) -> AgentDispatchAuthority:
    return AgentDispatchAuthority(
        profile="orchestrator",
        task_contract="orchestrate",
        scope=AgentDispatchScope(
            run_truth_scope=["repo"],
            episode_id=episode_id,
            patch_kind="work",
        ),
    )


def _episode(tmp_path) -> tuple[AppStore, EpisodeRecord, AgentTaskRecord]:
    store = AppStore(tmp_path / "rcp.sqlite3")
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
    now = store.now()
    authorized_by = AuthorizedHuman(
        space_id=str(uuid.uuid4()),
        user_id=str(uuid.uuid4()),
        display_name="Researcher",
    )
    graph_target = GraphTargetRef(kind="branch", branch_id="episode")
    episode = EpisodeRecord(
        episode_id="episode",
        project_id="project",
        mode="auto_research",
        graph_target=graph_target,
        graph_base_head=GraphHeadRef(revision=0),
        status="queued",
        invocation_ceiling=3,
        authorized_by=authorized_by,
        created_at=now,
        updated_at=now,
    )
    request = AutoResearchRunRequest(
        episode_id=episode.episode_id,
        role="orchestrator",
        actor_operation_id="root",
        provider="codex",
        model="gpt-5",
        reasoning="medium",
        run_on="local",
        run_truth_scope=["repo"],
    )
    root = AgentTaskRecord(
        operation_id="root",
        project_id=episode.project_id,
        episode_id=episode.episode_id,
        graph_target=graph_target,
        kind="auto_research",
        status="queued",
        request=request.model_dump(mode="json"),
        created_at=now,
        updated_at=now,
        status_message="queued",
        authorized_by=authorized_by,
        dispatch_authority=_authority(episode.episode_id),
    )
    return (
        store,
        *store.create_auto_research_episode_with_root_task(
            episode,
            AutoResearchStateRecord(
                episode_id=episode.episode_id,
                starting_instruction="Decide what the evidence supports.",
                created_at=now,
                updated_at=now,
            ),
            root,
        ),
    )


def test_auto_research_request_contract_has_only_operational_roles_and_budget() -> None:
    assert AutoResearchStartRequest(invocation_ceiling=1).invocation_ceiling == 1
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        AutoResearchStartRequest(invocation_ceiling=0)
    with pytest.raises(ValidationError, match="orchestrator|worker"):
        AutoResearchRunRequest.model_validate({"episode_id": "episode", "role": "report"})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AutoResearchRunRequest.model_validate(
            {"episode_id": "episode", "role": "orchestrator", "ending": "completed"}
        )


def test_wrapup_selects_the_root_actors_exact_recovery_child_and_compact_receipt(tmp_path) -> None:
    store, episode, root = _episode(tmp_path)
    stage = str(tmp_path / "orchestrator-stage")
    store.checkpoint_agent_task(
        root.operation_id,
        native_session_id="session-1",
        stage_root=stage,
    )
    store.fail_agent_task(root.operation_id, "temporary provider failure")
    root = store.agent_task(root.operation_id)
    assert root is not None
    request = AutoResearchRunRequest.model_validate(root.request).model_copy(
        update={"session_id": "session-1"}
    )
    now = store.now()
    recovery = store.create_auto_research_recovery_task(
        AgentTaskRecord(
            operation_id="root-retry",
            project_id=episode.project_id,
            episode_id=episode.episode_id,
            graph_target=episode.graph_target,
            kind="auto_research",
            status="succeeded",
            request=request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="recovered",
            attempt=2,
            parent_operation_id=root.operation_id,
            native_session_id="session-1",
            stage_root=stage,
            authorized_by=episode.authorized_by,
            dispatch_authority=root.dispatch_authority,
        )
    )
    watcher = store.create_watchers(
        [
            GraphWatcherRecord(
                watcher_id="episode-watcher",
                project_id=episode.project_id,
                origin_operation_id=recovery.operation_id,
                origin_task_kind="auto_research",
                graph_target=episode.graph_target,
                chat_id=episode.episode_id,
                episode_id=episode.episode_id,
                continuation=WatcherContinuation(
                    provider="codex",
                    run_on="local",
                    patch_kind="work",
                ),
                condition={"node_id": "claim", "status_in": ["active"]},
                armed_revision=1,
                status="active",
                created_at=store.now(),
            )
        ]
    )[0]

    signal = auto_research_completion_signal(store, episode.episode_id)
    spec = auto_research_wrapup_spec(store, signal)

    assert spec.continuation_operation_id == recovery.operation_id
    assert spec.ending == "completed"
    assert spec.partial is False
    assert spec.receipt["starting_instruction"] == "Decide what the evidence supports."
    assert spec.receipt["operational_meter"] == {
        "ceiling": 3,
        "used": 1,
        "remaining": 2,
        "observed_input_tokens": 0,
        "observed_generated_tokens": 0,
    }
    assert spec.receipt["experiment_allowance"] == {
        "total": 15,
        "used": 0,
        "remaining": 15,
    }
    assert spec.receipt["child_work"] == []
    assert spec.receipt["child_experiments"] == []
    assert spec.receipt["lifecycle"] == {
        "counts": {"pending": 0, "delivered": 0, "acknowledged": 0},
        "facts": [],
        "omitted_fact_count": 0,
    }
    assert (
        not {
            "events",
            "history",
            "messages",
            "research",
            "tasks",
            "transcript",
        }
        & spec.receipt.keys()
    )
    assert len(json.dumps(spec.receipt).encode("utf-8")) < 32_000
    assert store.watcher(watcher.watcher_id).status == "stopped"  # type: ignore[union-attr]


def test_stop_settles_without_a_wrapup_spec(tmp_path) -> None:
    store, episode, root = _episode(tmp_path)
    store.complete_agent_task(root.operation_id, applied_revision=None, result={})

    stopping = request_auto_research_stop(store, episode.episode_id)
    stopped = settle_auto_research_stop(store, episode.episode_id)

    assert stopping.status == "stopping"
    assert stopped is not None
    assert stopped.status == "stopped"
    assert stopped.ending == "stopped"
    assert stopped.wrapup_state == "skipped"
