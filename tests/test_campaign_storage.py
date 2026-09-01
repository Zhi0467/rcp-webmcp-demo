from __future__ import annotations

import json
import uuid

import pytest

import rcp.storage as storage_package
from rcp.core.authority import AgentDispatchAuthority, AgentDispatchScope
from rcp.core.models import AuthorizedHuman
from rcp.core.transition_models import GraphHeadRef, GraphTargetRef
from rcp.providers import ProviderUsage
from rcp.storage import (
    AgentTaskRecord,
    AppStore,
    AutoResearchActorBusy,
    AutoResearchMessageRecord,
    AutoResearchStateRecord,
    EpisodeInvocationCeilingReached,
    EpisodeNotRunning,
    EpisodeRecord,
    EpisodeWrapupRecord,
    GraphWatcherRecord,
    ProjectRecord,
    WatcherContinuation,
)
from rcp.storage.episodes import compact_episode_receipt


def _authorizer() -> AuthorizedHuman:
    return AuthorizedHuman(
        space_id=str(uuid.uuid4()),
        user_id=str(uuid.uuid4()),
        display_name="Researcher",
    )


def _project(store: AppStore, project_id: str = "project") -> None:
    store.upsert_project(
        ProjectRecord(
            project_id=project_id,
            locator=f"/tmp/{project_id}/research.yaml",
            name=project_id,
            state_location=f"/tmp/{project_id}/.research",
            state_remote=False,
            added_at=store.now(),
        )
    )


def _authority(episode_id: str, role: str) -> AgentDispatchAuthority:
    return AgentDispatchAuthority(
        profile="orchestrator" if role == "orchestrator" else "ordinary",
        task_contract="orchestrate" if role == "orchestrator" else "work_auto",
        scope=AgentDispatchScope(
            run_truth_scope=["repo"],
            episode_id=episode_id,
            patch_kind="work",
        ),
    )


def _task(
    store: AppStore,
    episode: EpisodeRecord,
    *,
    operation_id: str,
    role: str,
    parent_operation_id: str | None,
    actor_operation_id: str | None = None,
    control_node_id: str | None = None,
    status: str = "succeeded",
    attempt: int = 1,
    session_id: str | None = None,
    stage_root: str | None = None,
) -> AgentTaskRecord:
    actor = actor_operation_id or operation_id
    request: dict[str, object] = {
        "episode_id": episode.episode_id,
        "role": role,
        "actor_operation_id": actor,
        "run_truth_scope": ["repo"],
    }
    if control_node_id is not None:
        request["control_node_id"] = control_node_id
    if session_id is not None:
        request["session_id"] = session_id
    now = store.now()
    return AgentTaskRecord(
        operation_id=operation_id,
        project_id=episode.project_id,
        episode_id=episode.episode_id,
        graph_target=episode.graph_target,
        kind="auto_research",
        status=status,
        request=request,
        created_at=now,
        updated_at=now,
        status_message=status,
        attempt=attempt,
        parent_operation_id=parent_operation_id,
        native_session_id=session_id,
        stage_root=stage_root,
        authorized_by=episode.authorized_by,
        dispatch_authority=_authority(episode.episode_id, role),
    )


def _episode(
    store: AppStore,
    *,
    episode_id: str = "episode-a",
    project_id: str = "project",
    ceiling: int = 4,
    episode_status: str = "queued",
    root_status: str = "succeeded",
    root_stage: str | None = None,
) -> tuple[EpisodeRecord, AgentTaskRecord]:
    now = store.now()
    authorizer = _authorizer()
    episode = EpisodeRecord(
        episode_id=episode_id,
        project_id=project_id,
        mode="auto_research",
        graph_target=GraphTargetRef(kind="branch", branch_id=episode_id),
        graph_base_head=GraphHeadRef(revision=0),
        status=episode_status,
        invocation_ceiling=ceiling,
        authorized_by=authorizer,
        created_at=now,
        updated_at=now,
    )
    root = _task(
        store,
        episode,
        operation_id=f"{episode_id}-root",
        role="orchestrator",
        parent_operation_id=None,
        status="queued",
        stage_root=root_stage,
    )
    saved_episode, saved_root = store.create_auto_research_episode_with_root_task(
        episode,
        AutoResearchStateRecord(
            episode_id=episode_id,
            starting_instruction="Investigate the evidence.",
            created_at=now,
            updated_at=now,
        ),
        root,
    )
    if root_status == "succeeded":
        store.complete_agent_task(saved_root.operation_id, applied_revision=None, result={})
    elif root_status == "failed":
        store.fail_agent_task(saved_root.operation_id, "fixture failure")
    elif root_status != "queued":
        raise ValueError(f"unsupported root fixture status: {root_status}")
    refreshed_root = store.agent_task(saved_root.operation_id)
    assert refreshed_root is not None
    return saved_episode, refreshed_root


def _table_names(store: AppStore) -> set[str]:
    with store.connection() as connection:
        return {
            str(row["name"])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }


def test_fresh_schema_and_public_storage_surface_use_only_episode_vocabulary(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    tables = _table_names(store)

    assert {
        "auto_research_episodes",
        "auto_research_invocations",
        "auto_research_messages",
        "auto_research_recoveries",
    } <= tables
    assert not {name for name in tables if name.startswith("campaign")}
    assert not hasattr(storage_package, "CampaignRecord")
    with store.connection() as connection:
        for table in (
            "auto_research_episodes",
            "auto_research_invocations",
            "auto_research_messages",
            "auto_research_recoveries",
        ):
            columns = {
                str(row["name"]) for row in connection.execute(f"PRAGMA table_info({table})")
            }
            assert "campaign_id" not in columns


def test_auto_research_root_and_paid_turn_share_the_generic_operational_meter(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    episode, root = _episode(store)
    worker = store.create_auto_research_agent_task(
        _task(
            store,
            episode,
            operation_id="worker",
            role="worker",
            parent_operation_id=root.operation_id,
            control_node_id="exp/check",
        ),
        role="worker",
    )

    state = store.auto_research_state(episode.episode_id)
    assert state is not None
    assert state.episode_id == episode.episode_id
    assert state.starting_instruction == "Investigate the evidence."
    assert store.episode_budget_meter(episode.episode_id).invocations_used == 2
    assert [item.operation_id for item in store.episode_invocations(episode.episode_id)] == [
        root.operation_id,
        worker.operation_id,
    ]
    invocation = store.auto_research_invocation(worker.operation_id)
    assert invocation is not None
    assert invocation.episode_id == episode.episode_id
    assert invocation.allocation_operation_id == worker.operation_id
    assert invocation.actor_operation_id == worker.operation_id
    assert invocation.role == "worker"
    assert store.agent_task_profile(root.operation_id) == "orchestrator"
    assert store.agent_task_profile(worker.operation_id) == "ordinary"
    assert all(
        "campaign_id" not in task.request for task in store.episode_tasks(episode.episode_id)
    )


def test_auto_research_root_cannot_bypass_new_episode_invariants(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)

    with pytest.raises(ValueError, match="unused queued episode"):
        _episode(store, episode_status="running")

    now = store.now()
    invalid_episode = EpisodeRecord(
        episode_id="episode-b",
        project_id="project",
        mode="auto_research",
        graph_target=GraphTargetRef(kind="branch", branch_id="episode-b"),
        graph_base_head=GraphHeadRef(revision=0),
        status="queued",
        invocation_ceiling=4,
        authorized_by=_authorizer(),
        created_at=now,
        updated_at=now,
    )
    with pytest.raises(ValueError, match="visible queued task"):
        store.create_auto_research_episode_with_root_task(
            invalid_episode,
            AutoResearchStateRecord(
                episode_id=invalid_episode.episode_id,
                starting_instruction="Investigate the evidence.",
                created_at=now,
                updated_at=now,
            ),
            _task(
                store,
                invalid_episode,
                operation_id="episode-b-root",
                role="orchestrator",
                parent_operation_id=None,
                status="succeeded",
            ),
        )


def test_exact_recovery_reuses_its_allocation_without_spending_an_invocation(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    episode, root = _episode(store)
    worker = store.create_auto_research_agent_task(
        _task(
            store,
            episode,
            operation_id="failed-worker",
            role="worker",
            parent_operation_id=root.operation_id,
            control_node_id="exp/check",
            status="failed",
            session_id="worker-session",
            stage_root="/tmp/worker-stage",
        ),
        role="worker",
    )
    meter_before = store.episode_budget_meter(episode.episode_id)
    recovery = store.create_auto_research_recovery_task(
        _task(
            store,
            episode,
            operation_id="worker-retry",
            role="worker",
            parent_operation_id=worker.operation_id,
            actor_operation_id=worker.operation_id,
            control_node_id="exp/check",
            status="queued",
            attempt=2,
            session_id="worker-session",
            stage_root="/tmp/worker-stage",
        )
    )

    assert store.episode_budget_meter(episode.episode_id) == meter_before
    assert len(store.episode_invocations(episode.episode_id)) == 2
    recovery_invocation = store.auto_research_invocation(recovery.operation_id)
    assert recovery_invocation is not None
    assert recovery_invocation.allocation_operation_id == worker.operation_id
    assert len(store.auto_research_tasks(episode.episode_id)) == 3
    with pytest.raises(ValueError, match="already has a recovery child"):
        store.create_auto_research_recovery_task(
            recovery.model_copy(
                update={
                    "operation_id": "duplicate-retry",
                    "created_at": store.now(),
                    "updated_at": store.now(),
                }
            )
        )


def test_clean_root_retry_needs_the_same_stage_but_not_a_dead_session(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    episode, root = _episode(
        store,
        root_status="failed",
        root_stage="/tmp/orchestrator-stage",
    )
    recovery = store.create_auto_research_recovery_task(
        _task(
            store,
            episode,
            operation_id="clean-root-retry",
            role="orchestrator",
            parent_operation_id=root.operation_id,
            actor_operation_id=root.operation_id,
            status="queued",
            attempt=2,
            stage_root=root.stage_root,
        )
    )

    assert recovery.native_session_id is None
    assert recovery.stage_root == root.stage_root
    assert store.episode_budget_meter(episode.episode_id).invocations_used == 1


def test_paid_wake_cannot_overtake_a_recoverable_actor_leaf(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    episode, root = _episode(store)
    worker = store.create_auto_research_agent_task(
        _task(
            store,
            episode,
            operation_id="paused-worker",
            role="worker",
            parent_operation_id=root.operation_id,
            control_node_id="exp/check",
            status="paused",
            session_id="worker-session",
            stage_root="/tmp/worker-stage",
        ),
        role="worker",
    )
    meter_before = store.episode_budget_meter(episode.episode_id)
    with pytest.raises(AutoResearchActorBusy) as exc_info:
        store.create_auto_research_agent_task(
            _task(
                store,
                episode,
                operation_id="wake",
                role="worker",
                parent_operation_id=worker.operation_id,
                actor_operation_id=worker.operation_id,
                control_node_id="exp/check",
                session_id="worker-session",
                stage_root="/tmp/worker-stage",
            ),
            role="worker",
        )
    assert exc_info.value.operation_id == worker.operation_id
    assert store.episode_budget_meter(episode.episode_id) == meter_before


def test_message_and_command_ledgers_use_episode_identity(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    episode, root = _episode(store)
    message = store.record_auto_research_message(
        AutoResearchMessageRecord(
            message_id="human-message",
            episode_id=episode.episode_id,
            sender_role="human",
            authorized_by=episode.authorized_by,
            recipient_task_id=root.operation_id,
            body="Check the new result.",
            created_at=store.now(),
        )
    )
    command = store.start_agent_command(
        operation_id=root.operation_id,
        command_id="command-a",
        episode_id=episode.episode_id,
        verb="inspect",
        idempotency_key="inspect-once",
        payload={"node": "result"},
    )

    assert message.episode_id == episode.episode_id
    assert store.pending_auto_research_messages(episode.episode_id, root.operation_id) == [message]
    assert command.episode_id == episode.episode_id
    assert store.agent_command_by_key(episode.episode_id, "inspect-once") == command
    event = store.agent_task_events(root.operation_id)[-1]
    assert event.episode_id == episode.episode_id
    assert "campaign_id" not in (event.payload or {})


def test_stop_fence_closes_admission_and_retires_current_and_new_watchers(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    episode, root = _episode(store)
    first = GraphWatcherRecord(
        watcher_id="watch-before-stop",
        project_id=episode.project_id,
        origin_operation_id=root.operation_id,
        origin_task_kind="auto_research",
        chat_id=root.operation_id,
        episode_id=episode.episode_id,
        graph_target=episode.graph_target,
        continuation=WatcherContinuation(provider="codex", run_on="local", patch_kind="work"),
        condition={"node_id": "claim", "status_in": ["active"]},
        armed_revision=1,
        status="active",
        created_at=store.now(),
    )
    store.create_watchers([first])
    store.request_episode_stop(episode.episode_id)
    assert store.settle_auto_research_watchers(episode.episode_id) == 1
    assert store.watcher(first.watcher_id).status == "stopped"  # type: ignore[union-attr]
    with pytest.raises(EpisodeNotRunning):
        store.create_auto_research_agent_task(
            _task(
                store,
                episode,
                operation_id="late-worker",
                role="worker",
                parent_operation_id=root.operation_id,
                control_node_id="exp/check",
            ),
            role="worker",
        )
    second = first.model_copy(update={"watcher_id": "watch-after-stop"})
    assert store.create_watchers([second])[0].status == "stopped"
    settled = store.mark_episode_stop_skipped(episode.episode_id)
    assert settled.status == "stopped"
    assert settled.wrapup_state == "skipped"


def test_auto_research_stop_and_watcher_settlement_is_atomic(tmp_path, monkeypatch) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    episode, root = _episode(store)
    watcher = GraphWatcherRecord(
        watcher_id="watch-atomic-stop",
        project_id=episode.project_id,
        origin_operation_id=root.operation_id,
        origin_task_kind="auto_research",
        chat_id=root.operation_id,
        episode_id=episode.episode_id,
        graph_target=episode.graph_target,
        continuation=WatcherContinuation(provider="codex", run_on="local", patch_kind="work"),
        condition={"node_id": "claim", "status_in": ["active"]},
        armed_revision=1,
        status="active",
        created_at=store.now(),
    )
    store.create_watchers([watcher])
    before_episode = store.episode(episode.episode_id)
    before_watcher = store.watcher(watcher.watcher_id)
    assert before_episode is not None and before_watcher is not None

    def fail_after_episode_update(connection, *, episode_id: str, now: str) -> int:
        row = connection.execute(
            "SELECT status, stop_requested_at FROM episodes WHERE episode_id = ?",
            (episode_id,),
        ).fetchone()
        assert row is not None
        assert row["status"] == "stopping"
        assert row["stop_requested_at"] == now
        raise RuntimeError("injected Auto-research Stop settlement failure")

    monkeypatch.setattr(
        store, "_settle_auto_research_watchers_in_connection", fail_after_episode_update
    )
    with pytest.raises(RuntimeError, match="Stop settlement failure"):
        store.request_auto_research_stop_and_settle_watchers(episode.episode_id)

    assert store.episode(episode.episode_id) == before_episode
    assert store.watcher(watcher.watcher_id) == before_watcher

    monkeypatch.undo()
    settled = store.request_auto_research_stop_and_settle_watchers(episode.episode_id)

    assert settled == store.episode(episode.episode_id)
    assert settled.status == "stopping"
    assert settled.stop_requested_at is not None
    stopped_watcher = store.watcher(watcher.watcher_id)
    assert stopped_watcher is not None
    assert stopped_watcher.status == "stopped"
    assert stopped_watcher.notified is True


def test_auto_research_ending_fence_and_watcher_settlement_is_atomic(tmp_path, monkeypatch) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    episode, root = _episode(store, episode_id="ending-atomic", root_status="failed")
    watcher = GraphWatcherRecord(
        watcher_id="watch-atomic-ending",
        project_id=episode.project_id,
        origin_operation_id=root.operation_id,
        origin_task_kind="auto_research",
        chat_id=root.operation_id,
        episode_id=episode.episode_id,
        graph_target=episode.graph_target,
        continuation=WatcherContinuation(provider="codex", run_on="local", patch_kind="work"),
        condition={"node_id": "claim", "status_in": ["active"]},
        armed_revision=1,
        status="active",
        created_at=store.now(),
    )
    store.create_watchers([watcher])
    before_episode = store.episode(episode.episode_id)
    before_watcher = store.watcher(watcher.watcher_id)
    assert before_episode is not None and before_watcher is not None

    def fail_after_episode_update(connection, *, episode_id: str, now: str) -> int:
        row = connection.execute(
            "SELECT status, ending, ending_diagnostic FROM episodes WHERE episode_id = ?",
            (episode_id,),
        ).fetchone()
        assert row is not None
        assert row["status"] == "wrapping_up"
        assert row["ending"] == "failed"
        assert row["ending_diagnostic"] == "terminal"
        raise RuntimeError("injected Auto-research ending settlement failure")

    monkeypatch.setattr(
        store, "_settle_auto_research_watchers_in_connection", fail_after_episode_update
    )
    with pytest.raises(RuntimeError, match="ending settlement failure"):
        store.fence_auto_research_ending_and_settle_watchers(
            episode.episode_id,
            "failed",
            diagnostic="terminal",
        )

    assert store.episode(episode.episode_id) == before_episode
    assert store.watcher(watcher.watcher_id) == before_watcher

    monkeypatch.undo()
    fenced = store.fence_auto_research_ending_and_settle_watchers(
        episode.episode_id,
        "failed",
        diagnostic="terminal",
    )

    assert fenced == store.episode(episode.episode_id)
    assert fenced.status == "wrapping_up"
    assert fenced.ending == "failed"
    assert fenced.ending_diagnostic == "terminal"
    stopped_watcher = store.watcher(watcher.watcher_id)
    assert stopped_watcher is not None
    assert stopped_watcher.status == "stopped"
    assert stopped_watcher.notified is True


def test_non_stop_ending_fence_closes_auto_research_recovery_admission(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    episode, root = _episode(store, episode_id="scheduled", root_status="failed")
    store.schedule_auto_research_task_recovery(
        root.operation_id,
        failure_kind="transport",
        retry_mode="clean",
        diagnostic="Retry the root.",
    )
    _project(store, "candidate-project")
    candidate_episode, candidate_root = _episode(
        store,
        episode_id="candidate",
        project_id="candidate-project",
        root_status="failed",
    )

    store.fence_episode_ending(episode.episode_id, "failed", diagnostic="terminal")
    store.fence_episode_ending(candidate_episode.episode_id, "failed", diagnostic="terminal")

    assert store.due_auto_research_recoveries(as_of="9999-12-31T23:59:59+00:00") == []
    assert candidate_root.operation_id not in {
        task.operation_id for task in store.auto_research_recovery_candidates()
    }
    with pytest.raises(EpisodeNotRunning, match="no longer accepts recovery"):
        store.schedule_auto_research_task_recovery(
            root.operation_id,
            failure_kind="transport",
            retry_mode="clean",
            diagnostic="Do not reopen the ending.",
        )


def test_operational_ceiling_has_no_hidden_report_reservation(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    episode, root = _episode(store, ceiling=1)
    meter = store.episode_budget_meter(episode.episode_id)
    assert meter.invocations_used == 1
    assert meter.invocations_remaining == 0
    with pytest.raises(EpisodeInvocationCeilingReached):
        store.create_auto_research_agent_task(
            _task(
                store,
                episode,
                operation_id="over-ceiling",
                role="worker",
                parent_operation_id=root.operation_id,
                control_node_id="exp/check",
            ),
            role="worker",
        )


def test_legacy_campaign_tables_migrate_once_then_move_to_private_archives(tmp_path) -> None:
    path = tmp_path / "rcp.sqlite3"
    store = AppStore(path)
    _project(store)
    authorizer = _authorizer()
    now = store.now()
    with store.connection() as connection:
        connection.executescript(
            """
            CREATE TABLE campaigns (
                campaign_id TEXT PRIMARY KEY, project_id TEXT NOT NULL,
                root_operation_id TEXT, status TEXT NOT NULL, starting_instruction TEXT,
                invocation_ceiling INTEGER NOT NULL, invocations_used INTEGER NOT NULL,
                authorized_space_id TEXT, authorized_user_id TEXT,
                authorized_display_name TEXT, stop_requested_at TEXT, ending TEXT,
                error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, ended_at TEXT
            );
            CREATE TABLE campaign_invocations (
                campaign_id TEXT NOT NULL, operation_id TEXT NOT NULL UNIQUE,
                role TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE campaign_reports (
                report_id TEXT PRIMARY KEY, campaign_id TEXT NOT NULL,
                operation_id TEXT NOT NULL UNIQUE, ending TEXT NOT NULL,
                sha256 TEXT NOT NULL, html TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE campaign_messages (
                message_id TEXT PRIMARY KEY, campaign_id TEXT NOT NULL,
                sender_role TEXT NOT NULL, sender_task_id TEXT,
                authorized_space_id TEXT, authorized_user_id TEXT,
                authorized_display_name TEXT, recipient_task_id TEXT NOT NULL,
                control_node_id TEXT, body TEXT NOT NULL, created_at TEXT NOT NULL,
                delivered_at TEXT, delivery_operation_id TEXT
            );
            CREATE TABLE campaign_recoveries (
                recovery_id TEXT PRIMARY KEY, campaign_id TEXT NOT NULL,
                operation_id TEXT, purpose TEXT NOT NULL, failure_kind TEXT NOT NULL,
                retry_mode TEXT NOT NULL, attempts INTEGER NOT NULL,
                max_attempts INTEGER NOT NULL, status TEXT NOT NULL,
                next_attempt_at TEXT, diagnostic TEXT NOT NULL,
                admitted_operation_id TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            """
        )
        connection.execute(
            """
            INSERT INTO graph_runs (
                operation_id, project_id, episode_id, kind, status, request_json,
                created_at, updated_at, status_message, attempt,
                estimate_seconds, estimate_samples, phase, visible
            ) VALUES ('legacy-root', 'project', NULL, 'campaign', 'failed', ?, ?, ?,
                      'failed', 1, 300, 0, 'failed', 1)
            """,
            (
                json.dumps(
                    {
                        "campaign_id": "legacy-episode",
                        "role": "orchestrator",
                        "actor_operation_id": "legacy-root",
                        "run_truth_scope": ["repo"],
                    }
                ),
                now,
                now,
            ),
        )
        connection.execute(
            """
            INSERT INTO campaigns VALUES (
                'legacy-episode', 'project', 'legacy-root', 'running',
                'Legacy instruction', 3, 1, ?, ?, ?, NULL, NULL, NULL, ?, ?, NULL
            )
            """,
            (
                authorizer.space_id,
                authorizer.user_id,
                authorizer.display_name,
                now,
                now,
            ),
        )
        connection.execute(
            "INSERT INTO campaign_invocations VALUES ('legacy-episode', 'legacy-root', 'orchestrator', ?)",
            (now,),
        )
        connection.execute(
            """
            INSERT INTO campaign_messages VALUES (
                'legacy-message', 'legacy-episode', 'human', NULL, ?, ?, ?,
                'legacy-root', NULL, 'Preserved message', ?, NULL, NULL
            )
            """,
            (
                authorizer.space_id,
                authorizer.user_id,
                authorizer.display_name,
                now,
            ),
        )
        connection.execute(
            """
            INSERT INTO campaign_recoveries VALUES (
                'task:legacy-root', 'legacy-episode', 'legacy-root', 'task',
                'transport', 'clean', 0, 3, 'pending', ?, 'Retry later', NULL, ?, ?
            )
            """,
            (now, now, now),
        )

    migrated = AppStore(path)
    episode = migrated.episode("legacy-episode")
    assert episode is not None and episode.mode == "auto_research"
    assert migrated.auto_research_state("legacy-episode").starting_instruction == (  # type: ignore[union-attr]
        "Legacy instruction"
    )
    assert migrated.auto_research_invocation("legacy-root") is not None
    assert migrated.auto_research_message("legacy-message") is not None
    assert migrated.auto_research_recovery("task:legacy-root") is not None
    names = _table_names(migrated)
    assert not {name for name in names if name.startswith("campaign")}
    assert {
        "_legacy_campaigns_archive",
        "_legacy_campaign_invocations_archive",
        "_legacy_campaign_reports_archive",
        "_legacy_campaign_messages_archive",
        "_legacy_campaign_recoveries_archive",
    } <= names
    reopened = AppStore(path)
    assert reopened.auto_research_state("legacy-episode") is not None
    with reopened.connection() as connection:
        assert (
            connection.execute(
                "SELECT body FROM _legacy_campaign_messages_archive WHERE message_id = 'legacy-message'"
            ).fetchone()["body"]
            == "Preserved message"
        )


def test_new_auto_research_writes_never_create_legacy_tables_or_keys(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    episode, root = _episode(store)
    store.schedule_auto_research_task_recovery(
        root.operation_id,
        failure_kind="transport",
        retry_mode="clean",
        diagnostic="Retry the root.",
    )
    with store.connection() as connection:
        names = {
            str(row["name"])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert not {name for name in names if name.startswith("campaign")}
        request = json.loads(
            connection.execute(
                "SELECT request_json FROM graph_runs WHERE operation_id = ?",
                (root.operation_id,),
            ).fetchone()["request_json"]
        )
        assert request["episode_id"] == episode.episode_id
        assert "campaign_id" not in request
        recovery_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(auto_research_recoveries)")
        }
        assert "episode_id" in recovery_columns
        assert "campaign_id" not in recovery_columns


def test_project_deletion_removes_canonical_auto_research_children(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    episode, root = _episode(store, root_status="failed")
    store.record_auto_research_message(
        AutoResearchMessageRecord(
            message_id="delete-message",
            episode_id=episode.episode_id,
            sender_role="human",
            authorized_by=episode.authorized_by,
            recipient_task_id=root.operation_id,
            body="Delete this preserved mail with its project.",
            created_at=store.now(),
        )
    )
    store.schedule_auto_research_task_recovery(
        root.operation_id,
        failure_kind="transport",
        retry_mode="clean",
        diagnostic="Delete this recovery with its project.",
    )

    counts = store.delete_project_records(episode.project_id)

    assert counts["auto_research_recoveries"] == 1
    assert counts["auto_research_messages"] == 1
    assert counts["auto_research_invocations"] == 1
    assert counts["auto_research_episodes"] == 1
    assert counts["episode_invocations"] == 1
    assert counts["episodes"] == 1
    assert store.episode(episode.episode_id) is None
    assert store.agent_task(root.operation_id) is None


def test_public_usage_snapshot_excludes_hidden_episode_report_usage(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    episode, root = _episode(store)
    store.checkpoint_agent_task(
        root.operation_id,
        native_session_id="root-session",
        stage_root="/tmp/root-stage",
    )
    receipt_json, receipt_sha256 = compact_episode_receipt(
        {"ending": "completed", "episode_id": episode.episode_id}
    )
    wrapup = EpisodeWrapupRecord(
        episode_id=episode.episode_id,
        ending="completed",
        partial=False,
        concluding_operation_id=root.operation_id,
        allocation_operation_id="hidden-report",
        provider="codex",
        run_on="local",
        execution_host="",
        native_session_id="root-session",
        stage_root="/tmp/root-stage",
        skill_id="episode-report",
        skill_version="1",
        output_name="episode-report.html",
        output_path="/tmp/root-stage/episode-report.html",
        receipt_json=receipt_json,
        receipt_sha256=receipt_sha256,
        state="pending",
        created_at=store.now(),
        updated_at=store.now(),
    )
    hidden = AgentTaskRecord(
        operation_id="hidden-report",
        project_id=episode.project_id,
        episode_id=episode.episode_id,
        graph_target=episode.graph_target,
        kind="episode_report",
        status="queued",
        request={
            "episode_id": episode.episode_id,
            "provider": "codex",
            "run_on": "local",
            "execution_host": "",
        },
        created_at=store.now(),
        updated_at=store.now(),
        status_message="hidden",
        parent_operation_id=root.operation_id,
        native_session_id="root-session",
        stage_root="/tmp/root-stage",
        visible=False,
    )
    store.begin_episode_wrapup(episode.episode_id, wrapup, hidden)
    for operation_id, key, tokens in (
        (root.operation_id, "operational", 100),
        (hidden.operation_id, "hidden-report", 900),
    ):
        store.record_agent_usage(
            operation_id,
            ProviderUsage(
                provider_profile="codex.turn.v1",
                provider_event_type="turn.completed",
                dedupe_key=key,
                processed_input_tokens=tokens,
                generated_tokens=tokens // 10,
            ),
        )

    assert len(store.agent_usage(episode.project_id)) == 2
    snapshot = store.agent_usage_snapshot(episode.project_id)
    assert snapshot.input_processed.total_tokens == 100
    assert snapshot.generated.total_tokens == 10
    assert snapshot.counted_records == 1
    assert all(cell.task_kind != "episode_report" for cell in snapshot.input_processed.cells)
    assert all(cell.task_kind != "episode_report" for cell in snapshot.generated.cells)
    meter = store.episode_budget_meter(episode.episode_id)
    assert meter.observed_input_tokens == 100
    assert meter.observed_generated_tokens == 10
    assert [task.operation_id for task in store.episode_tasks(episode.episode_id)] == [
        root.operation_id
    ]
    assert {
        task.operation_id for task in store.episode_tasks(episode.episode_id, include_hidden=True)
    } == {root.operation_id, hidden.operation_id}
