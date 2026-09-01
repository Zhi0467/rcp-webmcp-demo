from __future__ import annotations

import json
import sqlite3

import pytest

from rcp.core.models import AuthorizedHuman
from rcp.storage import AgentTaskRecord, AppStore, EpisodeRecord, ProjectRecord


def _authorizer(store: AppStore) -> AuthorizedHuman:
    owner = store.local_owner
    assert owner is not None
    if owner.display_name is None:
        owner = store.rename_space_user(owner.user_id, "Episode owner")
    return AuthorizedHuman(
        space_id=store.space_id,
        user_id=owner.user_id,
        display_name=owner.display_name,
    )


def _project(project_id: str) -> ProjectRecord:
    return ProjectRecord(
        project_id=project_id,
        locator=f"/tmp/{project_id}/research.yaml",
        name=project_id,
        state_location=f"/tmp/{project_id}/.research",
        state_remote=False,
        added_at="2026-08-14T00:00:00+00:00",
    )


def _episode(store: AppStore, episode_id: str, project_id: str) -> EpisodeRecord:
    now = store.now()
    return EpisodeRecord(
        episode_id=episode_id,
        project_id=project_id,
        mode="experiment_loop",
        control_node_id=f"{project_id}-experiment-node",
        status="queued",
        invocation_ceiling=1,
        authorized_by=_authorizer(store),
        created_at=now,
        updated_at=now,
    )


def _root_task(store: AppStore, episode_id: str, project_id: str) -> AgentTaskRecord:
    now = store.now()
    return AgentTaskRecord(
        operation_id=f"{episode_id}-operation",
        project_id=project_id,
        episode_id=episode_id,
        kind="node_chat",
        status="queued",
        request={},
        created_at=now,
        updated_at=now,
        status_message="Queued",
    )


def test_fresh_lineage_schema_and_events_use_only_episode_vocabulary(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    episode = _episode(store, "episode", "project")
    task = _root_task(store, episode.episode_id, episode.project_id)
    store.create_episode_with_invocation(episode, task)
    store.record_agent_task_event(task.operation_id, "Episode event")

    stored = store.agent_task(task.operation_id)
    event = store.agent_task_events(task.operation_id)[0]
    assert stored is not None
    assert stored.episode_id == episode.episode_id
    assert event.episode_id == episode.episode_id
    with store.connection() as connection:
        columns = {
            table: {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}
            for table in ("graph_runs", "graph_run_events", "watchers")
        }
    assert "episode_id" in columns["graph_runs"]
    assert "campaign_id" not in columns["graph_runs"]
    assert "episode_id" in columns["graph_run_events"]
    assert "campaign_id" not in columns["graph_run_events"]
    assert "episode_id" in columns["watchers"]
    assert "experiment_episode_id" not in columns["watchers"]
    with pytest.raises(ValueError):
        AgentTaskRecord.model_validate({**task.model_dump(), "campaign_id": "legacy"})
    with pytest.raises(ValueError):
        AgentTaskRecord.model_validate({**task.model_dump(), "kind": "campaign"})


def test_existing_lineage_columns_json_watchers_and_usage_migrate_once(tmp_path) -> None:
    path = tmp_path / "rcp.sqlite3"
    store = AppStore(path)
    created_at = "2026-08-14T00:00:00+00:00"
    legacy_request = {
        "campaign_id": "auto-episode",
        "nested": {"campaign_id": "auto-episode"},
    }
    legacy_dispatch = {
        "profile": "orchestrator",
        "task_contract": "orchestrate",
        "scope": {
            "run_truth_scope": ["repo"],
            "campaign_id": "auto-episode",
            "patch_kind": "work",
        },
    }
    continuation = {
        "provider": "codex",
        "run_on": "local",
        "patch_kind": "work",
    }
    with store.connection() as connection:
        for operation_id, episode_id, kind, request, dispatch in (
            (
                "auto-origin",
                "auto-episode",
                "campaign",
                legacy_request,
                legacy_dispatch,
            ),
            ("plain-origin", None, "refresh", {}, None),
            ("auto-notification", "auto-episode", "campaign", legacy_request, None),
        ):
            connection.execute(
                """
                INSERT INTO graph_runs (
                    operation_id, project_id, episode_id, kind, status, request_json,
                    created_at, updated_at, status_message, dispatch_authority_json
                ) VALUES (?, 'project', ?, ?, 'succeeded', ?, ?, ?, 'done', ?)
                """,
                (
                    operation_id,
                    episode_id,
                    kind,
                    json.dumps(request),
                    created_at,
                    created_at,
                    json.dumps(dispatch) if dispatch is not None else None,
                ),
            )
        connection.execute(
            """
            INSERT INTO graph_run_events (
                operation_id, created_at, level, message, payload_json
            ) VALUES ('auto-origin', ?, 'info', 'legacy event', ?)
            """,
            (created_at, json.dumps({"campaign_id": "auto-episode"})),
        )
        for watcher_id, origin_operation_id, notification_operation_id in (
            ("origin-watcher", "auto-origin", None),
            ("notification-watcher", "plain-origin", "auto-notification"),
        ):
            connection.execute(
                """
                INSERT INTO watchers (
                    watcher_id, project_id, origin_operation_id, origin_task_kind,
                    chat_id, execution_host, check_command, log_path, cwd,
                    continuation_json, status, created_at, notification_operation_id
                ) VALUES (?, 'project', ?, 'campaign', 'chat', '', 'true', '/tmp/log',
                          '/tmp', ?, 'completed', ?, ?)
                """,
                (
                    watcher_id,
                    origin_operation_id,
                    json.dumps(continuation),
                    created_at,
                    notification_operation_id,
                ),
            )
        connection.execute(
            """
            INSERT INTO agent_usage (
                usage_id, project_id, operation_id, task_kind, provider, model,
                provider_profile, provider_event_type, dedupe_key, counted,
                count_reason, created_at, processed_input_tokens, generated_tokens,
                cached_input_tokens, cache_creation_input_tokens, cache_write_input_tokens,
                reasoning_output_tokens, reported_input_tokens, reported_output_tokens,
                reported_total_tokens, provider_fields_json
            ) VALUES (
                'usage', 'project', 'auto-origin', 'campaign', 'codex', NULL,
                'default', 'usage', 'dedupe', 1, 'reported', ?, 0, 0, 0, 0, 0, 0,
                NULL, NULL, NULL, '{}'
            )
            """,
            (created_at,),
        )
        connection.execute("ALTER TABLE graph_runs RENAME COLUMN episode_id TO campaign_id")
        connection.execute("ALTER TABLE graph_run_events RENAME COLUMN episode_id TO campaign_id")
        connection.execute("ALTER TABLE watchers RENAME COLUMN episode_id TO experiment_episode_id")

    migrated = AppStore(path)
    task = migrated.agent_task("auto-origin")
    assert task is not None
    assert task.kind == "auto_research"
    assert task.episode_id == "auto-episode"
    assert task.request == {
        "episode_id": "auto-episode",
        "nested": {"episode_id": "auto-episode"},
    }
    assert task.dispatch_authority is not None
    assert task.dispatch_authority.scope.episode_id == "auto-episode"
    event = migrated.agent_task_events("auto-origin")[0]
    assert event.episode_id == "auto-episode"
    assert event.payload == {"episode_id": "auto-episode"}
    assert migrated.watcher("origin-watcher").episode_id == "auto-episode"
    assert migrated.watcher("notification-watcher").episode_id == "auto-episode"
    assert migrated.watcher("origin-watcher").origin_task_kind == "auto_research"
    with migrated.connection() as connection:
        assert (
            connection.execute(
                "SELECT task_kind FROM agent_usage WHERE usage_id = 'usage'"
            ).fetchone()[0]
            == "auto_research"
        )
        for table, legacy_column in (
            ("graph_runs", "campaign_id"),
            ("graph_run_events", "campaign_id"),
            ("watchers", "experiment_episode_id"),
        ):
            columns = {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}
            assert legacy_column not in columns
            assert "episode_id" in columns
    assert AppStore(path).agent_task("auto-origin") == task


@pytest.mark.parametrize(
    "request_payload",
    [
        {"campaign_id": "old", "episode_id": "new"},
        {"campaign_id": "old", "nested": {"episode_id": "new"}},
    ],
)
def test_lineage_json_migration_rejects_both_parent_keys(tmp_path, request_payload) -> None:
    path = tmp_path / "rcp.sqlite3"
    store = AppStore(path)
    now = store.now()
    with store.connection() as connection:
        connection.execute(
            """
            INSERT INTO graph_runs (
                operation_id, project_id, kind, status, request_json,
                created_at, updated_at, status_message
            ) VALUES ('ambiguous', 'project', 'refresh', 'succeeded', ?, ?, ?, 'done')
            """,
            (json.dumps(request_payload), now, now),
        )

    with pytest.raises(ValueError, match="both campaign_id and episode_id"):
        AppStore(path)


def test_project_deletion_removes_episode_parent_and_child_rows(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    project_id = "delete-project"
    store.upsert_project(_project(project_id))
    episode = _episode(store, "delete-episode", project_id)
    task = _root_task(store, episode.episode_id, project_id)
    store.create_episode_with_invocation(episode, task)
    store.fail_agent_task(task.operation_id, "settled")

    counts = store.delete_project_records(project_id)

    assert counts["episode_invocations"] == 1
    assert counts["episodes"] == 1
    assert counts["graph_runs"] == 1
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM episode_invocations").fetchone()[0] == 0
