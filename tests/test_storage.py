import hashlib
import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest

from rcp.artifacts import AgentArtifactDescriptor
from rcp.core.models import DISPLAY_NAME_MAX_LENGTH, AuthorizedHuman
from rcp.limits import AGENT_TASK_RECEIPT_RETENTION_COUNTS
from rcp.providers import ProviderUsage
from rcp.service import RunRequest, resolve_dispatch_authority
from rcp.storage import (
    AgentTaskRecord,
    AppStore,
    ChatSessionContextRecord,
    ExperimentLoopRuntime,
    ProjectRecord,
    SpaceUserRecord,
    TeamAuthenticationError,
    WatcherContinuation,
    WatcherRecord,
)


def _project(project_id: str) -> ProjectRecord:
    return ProjectRecord(
        project_id=project_id,
        locator=f"/tmp/{project_id}/research.yaml",
        name=project_id,
        state_location=f"/tmp/{project_id}/.research",
        state_remote=False,
        added_at="2026-07-31T00:00:00+00:00",
    )


def _task(store: AppStore, project_id: str, operation_id: str, status: str) -> None:
    now = store.now()
    store.create_agent_task(
        AgentTaskRecord(
            operation_id=operation_id,
            project_id=project_id,
            kind="refresh",
            status=status,
            request={},
            created_at=now,
            updated_at=now,
            status_message=status,
        )
    )


def _snapshot(value: str) -> tuple[str, str]:
    content = json.dumps({"value": value}, separators=(",", ":"), sort_keys=True)
    return content, hashlib.sha256(content.encode("utf-8")).hexdigest()


def test_space_identity_is_canonical_durable_and_distinct_per_store(tmp_path) -> None:
    first_path = tmp_path / "first" / "rcp.sqlite3"
    first = AppStore(first_path)
    space_id = first.space_id

    parsed = uuid.UUID(space_id)
    assert str(parsed) == space_id
    assert parsed.version == 4
    assert AppStore(first_path).space_id == space_id
    assert AppStore(tmp_path / "second" / "rcp.sqlite3").space_id != space_id


def test_existing_database_receives_one_durable_space_identity(tmp_path) -> None:
    path = tmp_path / "rcp.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE legacy_data (value TEXT NOT NULL)")
        connection.execute("INSERT INTO legacy_data(value) VALUES ('preserved')")

    first_store = AppStore(path)
    first = first_store.space_id
    owner = first_store.local_owner
    assert owner is not None
    second_store = AppStore(path)
    second = second_store.space_id

    with sqlite3.connect(path) as connection:
        identities = connection.execute(
            "SELECT singleton, space_id, space_kind, space_name FROM space_identity"
        ).fetchall()
        legacy = connection.execute("SELECT value FROM legacy_data").fetchone()
    assert identities == [(1, first, "personal", None)]
    assert second == first
    assert second_store.space_kind == "personal"
    assert second_store.local_owner == owner
    assert legacy == ("preserved",)


def test_team_space_initialization_requires_and_preserves_one_named_bootstrap(tmp_path) -> None:
    path = tmp_path / "rcp.sqlite3"

    for invalid_name in ("", "   ", "two\nlines"):
        with pytest.raises(ValueError, match="space name"):
            AppStore.initialize_team_space(path, invalid_name)
        assert not path.exists()

    store, bootstrap = AppStore.initialize_team_space(path, "  Systems Lab  ")

    assert store.space_kind == "team"
    assert store.space_name == "Systems Lab"
    assert bootstrap.startswith("rcp_bootstrap_")
    assert AppStore(path).space_name == "Systems Lab"
    with store.connection() as connection:
        identity = connection.execute(
            "SELECT space_id, space_kind, space_name FROM space_identity WHERE singleton = 1"
        ).fetchone()
        bootstrap_rows = connection.execute(
            "SELECT code_id, code_hash FROM team_bootstrap_codes"
        ).fetchall()
    assert tuple(identity) == (store.space_id, "team", "Systems Lab")
    assert len(bootstrap_rows) == 1
    assert bootstrap not in path.read_text(encoding="utf-8", errors="ignore")

    with pytest.raises(ValueError, match="already contains a space"):
        AppStore.initialize_team_space(path, "Replacement")
    with store.connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM team_bootstrap_codes").fetchone()[0] == 1


def test_unclaimed_team_init_reissues_only_with_the_same_name(tmp_path) -> None:
    store, first_code = AppStore.initialize_team_space(tmp_path / "rcp.sqlite3", "Lab")
    original_space_id = store.space_id

    with pytest.raises(ValueError, match="already contains a space"):
        AppStore.initialize_team_space(store.path, "Different lab")

    recovered, replacement_code = AppStore.initialize_team_space(store.path, "Lab")
    assert recovered.space_id == original_space_id
    assert replacement_code != first_code
    with pytest.raises(TeamAuthenticationError) as old_code:
        recovered.enroll_team_member(first_code, "Alice")
    assert old_code.value.code == "enrollment_code_invalid"
    member, _token = recovered.enroll_team_member(replacement_code, "Alice")
    assert member.display_name == "Alice"

    with pytest.raises(ValueError, match="already contains a space"):
        AppStore.initialize_team_space(store.path, "Lab")


def test_only_a_team_space_name_is_mutable_while_space_identity_is_not(tmp_path) -> None:
    team, _bootstrap = AppStore.initialize_team_space(tmp_path / "team.sqlite3", "Original name")
    original_id = team.space_id

    assert team.rename_space("Renamed team") == "Renamed team"
    assert AppStore(team.path).space_name == "Renamed team"
    with (
        team.connection() as connection,
        pytest.raises(sqlite3.IntegrityError, match="space identity is immutable"),
    ):
        connection.execute(
            "UPDATE space_identity SET space_id = ? WHERE singleton = 1",
            (str(uuid.uuid4()),),
        )
    with (
        team.connection() as connection,
        pytest.raises(sqlite3.IntegrityError, match="space identity is immutable"),
    ):
        connection.execute("UPDATE space_identity SET space_kind = 'personal' WHERE singleton = 1")
    assert team.space_id == original_id
    assert team.space_kind == "team"

    personal = AppStore(tmp_path / "personal.sqlite3")
    assert personal.space_name is None
    with pytest.raises(ValueError, match="Only a team space"):
        personal.rename_space("Not allowed")
    assert personal.space_name is None


def test_concurrent_initialization_converges_on_one_space_identity(tmp_path) -> None:
    path = tmp_path / "rcp.sqlite3"

    def initialize(_: int) -> tuple[str, str, str]:
        store = AppStore(path)
        owner = store.local_owner
        assert owner is not None
        return store.space_id, store.space_kind, owner.user_id

    with ThreadPoolExecutor(max_workers=4) as executor:
        identities = list(executor.map(initialize, range(4)))

    assert len(set(identities)) == 1
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM space_identity").fetchone() == (1,)


def test_space_identity_survives_complete_database_relocation(tmp_path) -> None:
    original_dir = tmp_path / "original"
    original_store = AppStore(original_dir / "rcp.sqlite3")
    original_owner = original_store.local_owner
    assert original_owner is not None
    original = (original_store.space_id, original_store.space_kind, original_owner.user_id)
    relocated_dir = tmp_path / "relocated"

    original_dir.rename(relocated_dir)

    relocated_store = AppStore(relocated_dir / "rcp.sqlite3")
    relocated_owner = relocated_store.local_owner
    assert relocated_owner is not None
    assert (
        relocated_store.space_id,
        relocated_store.space_kind,
        relocated_owner.user_id,
    ) == original


def test_s111_identity_migrates_to_personal_with_one_unnamed_owner(tmp_path) -> None:
    path = tmp_path / "rcp.sqlite3"
    space_id = str(uuid.uuid4())
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE space_identity (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                space_id TEXT NOT NULL UNIQUE
            )
            """
        )
        connection.execute(
            "INSERT INTO space_identity(singleton, space_id) VALUES (1, ?)",
            (space_id,),
        )
        connection.execute(
            """
            CREATE TRIGGER space_identity_immutable
            BEFORE UPDATE OF singleton, space_id ON space_identity
            BEGIN
                SELECT RAISE(ABORT, 'space identity is immutable');
            END
            """
        )

    migrated = AppStore(path)
    owner = migrated.local_owner

    assert migrated.space_id == space_id
    assert migrated.space_kind == "personal"
    assert owner is not None
    assert owner.identity_kind == "local_owner"
    assert owner.display_name is None
    assert uuid.UUID(owner.user_id).version == 4
    assert AppStore(path).local_owner == owner
    with (
        migrated.connection() as connection,
        pytest.raises(sqlite3.IntegrityError, match="space identity is immutable"),
    ):
        connection.execute("UPDATE space_identity SET space_kind = 'team' WHERE singleton = 1")


def test_legacy_database_cannot_be_reclassified_as_team_during_migration(tmp_path) -> None:
    path = tmp_path / "rcp.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE legacy_data (value TEXT NOT NULL)")

    with pytest.raises(ValueError, match="existing RCP database migrates to personal"):
        AppStore(path, space_kind="team")

    with sqlite3.connect(path) as connection:
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'space_identity'"
            ).fetchone()
            is None
        )
    assert AppStore(path).space_kind == "personal"


def test_explicit_team_space_preprovisions_distinct_members_with_duplicate_names(
    tmp_path,
) -> None:
    path = tmp_path / "rcp.sqlite3"
    store = AppStore(path, space_kind="team")

    assert store.space_kind == "team"
    assert store.local_owner is None
    assert store.space_users() == []

    first = store.preprovision_team_member("  Same Name  ")
    second = store.preprovision_team_member("Same Name")

    assert first.display_name == second.display_name == "Same Name"
    assert first.user_id != second.user_id
    assert first.identity_kind == second.identity_kind == "team_member"
    assert uuid.UUID(first.user_id).version == uuid.UUID(second.user_id).version == 4
    restarted = AppStore(path)
    assert restarted.space_kind == "team"
    assert {user.user_id: user for user in restarted.space_users()} == {
        first.user_id: first,
        second.user_id: second,
    }

    renamed = restarted.rename_space_user(first.user_id, "  Renamed Member ")
    assert renamed.display_name == "Renamed Member"
    assert renamed.updated_at != first.updated_at
    assert renamed.model_dump(exclude={"display_name", "updated_at"}) == first.model_dump(
        exclude={"display_name", "updated_at"}
    )
    assert restarted.space_user(second.user_id) == second


def test_space_user_names_reject_blank_without_changing_identity(tmp_path) -> None:
    team = AppStore(tmp_path / "team.sqlite3", space_kind="team")
    member = team.preprovision_team_member()

    with pytest.raises(ValueError, match="display name must not be blank"):
        team.rename_space_user(member.user_id, "  ")
    assert team.space_user(member.user_id) == member

    for invalid_name, message in (
        ("line one\nline two", "single line"),
        # A line separator and a paragraph separator are single characters that
        # still split the name across lines, so they are rejected like "\n".
        ("line one\u2028line two", "single line"),
        ("line one\u2029line two", "single line"),
        ("x" * (DISPLAY_NAME_MAX_LENGTH + 1), f"at most {DISPLAY_NAME_MAX_LENGTH} characters"),
    ):
        with pytest.raises(ValueError, match=message):
            team.rename_space_user(member.user_id, invalid_name)
        assert team.space_user(member.user_id) == member

    with pytest.raises(ValueError, match="display name must not be blank"):
        team.preprovision_team_member("\t")
    assert team.space_users() == [member]

    personal = AppStore(tmp_path / "personal.sqlite3")
    with pytest.raises(ValueError, match="Only a team space"):
        personal.preprovision_team_member("Member")
    assert len(personal.space_users()) == 1

    # The limit is inclusive, so the longest legal name is still accepted. This
    # runs last because, unlike every rejection above, it does change identity.
    longest_legal = "x" * DISPLAY_NAME_MAX_LENGTH
    assert team.rename_space_user(member.user_id, longest_legal).display_name == longest_legal


@pytest.mark.parametrize(
    ("created_kind", "requested_kind"),
    [("personal", "team"), ("team", "personal")],
)
def test_explicit_space_kind_mismatch_fails_without_changing_stored_kind(
    tmp_path,
    created_kind,
    requested_kind,
) -> None:
    path = tmp_path / "rcp.sqlite3"
    original = AppStore(path, space_kind=created_kind)
    original_users = original.space_users()

    with pytest.raises(ValueError, match=f"RCP space is {created_kind}"):
        AppStore(path, space_kind=requested_kind)

    reopened = AppStore(path)
    assert reopened.space_kind == created_kind
    assert reopened.space_users() == original_users


@pytest.mark.parametrize("persisted_kind", [None, "organization"])
def test_missing_or_invalid_stored_space_kind_is_never_silently_replaced(
    tmp_path,
    persisted_kind,
) -> None:
    path = tmp_path / "rcp.sqlite3"
    space_id = str(uuid.uuid4())
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE space_identity (
                singleton INTEGER PRIMARY KEY,
                space_id TEXT NOT NULL,
                space_kind TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO space_identity(singleton, space_id, space_kind)
            VALUES (1, ?, ?)
            """,
            (space_id, persisted_kind),
        )

    with pytest.raises(RuntimeError, match="space kind is invalid"):
        AppStore(path)

    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT space_id, space_kind FROM space_identity").fetchall() == [
            (space_id, persisted_kind)
        ]
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'space_users'"
            ).fetchone()
            is None
        )


def test_space_and_user_identity_fields_are_immutable(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    owner = store.local_owner
    assert owner is not None

    with (
        store.connection() as connection,
        pytest.raises(sqlite3.IntegrityError, match="space identity is immutable"),
    ):
        connection.execute("UPDATE space_identity SET space_kind = 'team' WHERE singleton = 1")
    with (
        store.connection() as connection,
        pytest.raises(sqlite3.IntegrityError, match="space user identity is immutable"),
    ):
        connection.execute(
            "UPDATE space_users SET identity_kind = 'team_member' WHERE user_id = ?",
            (owner.user_id,),
        )

    reopened = AppStore(store.path)
    assert reopened.space_kind == "personal"
    assert reopened.local_owner == owner


def test_space_user_record_rejects_noncanonical_identity_and_extra_fields(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    owner = store.local_owner
    assert owner is not None

    with pytest.raises(ValueError, match="canonical UUIDv4"):
        SpaceUserRecord.model_validate({**owner.model_dump(), "user_id": str(uuid.uuid1())})
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        SpaceUserRecord.model_validate({**owner.model_dump(), "role": "admin"})


@pytest.mark.parametrize("persisted", [None, "not-a-uuid", str(uuid.uuid1())])
def test_existing_space_identity_is_never_silently_replaced(tmp_path, persisted) -> None:
    path = tmp_path / "rcp.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE space_identity (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                space_id TEXT NOT NULL UNIQUE
            )
            """
        )
        if persisted is not None:
            connection.execute(
                "INSERT INTO space_identity(singleton, space_id) VALUES (1, ?)",
                (persisted,),
            )

    with pytest.raises(RuntimeError, match="space identity"):
        AppStore(path)

    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT space_id FROM space_identity").fetchall() == (
            [] if persisted is None else [(persisted,)]
        )


def test_opening_project_does_not_reorder_catalog(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    older = _project("older")
    newer = _project("newer")
    tied_a = _project("tied-a")
    tied_b = _project("tied-b")
    older.added_at = "2026-07-30T00:00:00+00:00"
    newer.added_at = "2026-07-31T00:00:00+00:00"
    tied_a.added_at = "2026-08-01T00:00:00+00:00"
    tied_b.added_at = tied_a.added_at
    tied_a.name = "Same name"
    tied_b.name = "same name"
    store.upsert_project(older)
    store.upsert_project(newer)
    store.upsert_project(tied_b)
    store.upsert_project(tied_a)

    expected_order = ["tied-a", "tied-b", "newer", "older"]
    assert [project.project_id for project in store.projects()] == expected_order

    store.update_project_summary(
        "older",
        revision=1,
        primary_question="Question",
        attention_count=0,
        last_refresh_at=None,
        reachable=True,
        error=None,
    )

    assert store.project("older").last_opened_at is not None
    assert [project.project_id for project in store.projects()] == expected_order


def test_existing_paper_draft_database_gains_ancestor_without_losing_draft(tmp_path) -> None:
    path = tmp_path / "rcp.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE paper_drafts (
                project_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                base_hash TEXT,
                updated_at TEXT NOT NULL,
                cursor_state TEXT
            );
            INSERT INTO paper_drafts(project_id, content, base_hash, updated_at, cursor_state)
            VALUES ('project', '# Existing draft', 'old-base', '2026-08-01T00:00:00+00:00', NULL);
            """
        )

    store = AppStore(path)

    with store.connection() as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(paper_drafts)")}
        draft = connection.execute(
            "SELECT content, base_hash, ancestor_content FROM paper_drafts WHERE project_id = ?",
            ("project",),
        ).fetchone()
    assert "ancestor_content" in columns
    assert draft is not None
    assert draft["content"] == "# Existing draft"
    assert draft["base_hash"] == "old-base"
    assert draft["ancestor_content"] is None


class _TracingAppStore(AppStore):
    """Count projection reads without changing the production connection path."""

    def __init__(self, path) -> None:
        self.select_count = 0
        super().__init__(path)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        with super().connection() as connection:
            connection.set_trace_callback(self._count_select)
            yield connection

    def _count_select(self, statement: str) -> None:
        if statement.lstrip().upper().startswith("SELECT"):
            self.select_count += 1


class _WriteTracingAppStore(AppStore):
    """Signal when a store writer reaches its transaction boundary."""

    def __init__(self, path) -> None:
        self.write_attempted = threading.Event()
        super().__init__(path)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        with super().connection() as connection:
            connection.set_trace_callback(self._trace_statement)
            yield connection

    def _trace_statement(self, statement: str) -> None:
        if statement.strip().upper() == "BEGIN IMMEDIATE":
            self.write_attempted.set()


def _create_experiment_runtime_fixture(
    store: AppStore,
    *,
    project_id: str,
    control_node_id: str,
    operation_id: str,
    status: str,
    arm_watcher: bool = False,
) -> tuple[str, str]:
    episode_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{project_id}:{control_node_id}:episode"))
    chat_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{project_id}:{control_node_id}:chat"))
    request: dict[str, object] = {
        "provider": "codex",
        "model": "",
        "reasoning": "medium",
        "run_on": "local",
        "run_truth_scope": ["repo-a"],
        "chat_id": chat_id,
        "node_id": control_node_id,
        "message": "Continue the bounded experiment.",
        "mode": "work",
        "trigger": "experiment_run",
        "patch_kind": "experiment_loop",
        "control_node_id": control_node_id,
        "control_revision": 7,
        "control_episode_id": episode_id,
        "control_invocation": 1,
        "control_invocation_ceiling": 3,
        "control_decision_bundle": [],
        "control_completion_criteria": ["Detached work has finished."],
        "watcher_ids": [],
    }
    now = store.now()
    owner = store.local_owner
    assert owner is not None
    if owner.display_name is None:
        owner = store.rename_space_user(owner.user_id, "Experiment owner")
    authorized_by = AuthorizedHuman(
        space_id=store.space_id,
        user_id=owner.user_id,
        display_name=owner.display_name,
    )
    store.create_experiment_episode_with_invocation(
        AgentTaskRecord(
            operation_id=operation_id,
            project_id=project_id,
            episode_id=episode_id,
            kind="node_chat",
            status="queued",
            request=request,
            created_at=now,
            updated_at=now,
            status_message="queued",
            phase="queued",
            last_activity_at=now,
            authorized_by=authorized_by,
        )
    )
    store.commit_experiment_episode_turn(
        episode_id=episode_id,
        project_id=project_id,
        control_node_id=control_node_id,
        provider="codex",
        execution_machine="local",
        execution_host="",
        native_session_id=f"session-{operation_id}",
        stage_host=None,
        stage_root=f"/tmp/{operation_id}",
        chat_id=chat_id,
        operation_id=operation_id,
        invocation=1,
        graph_result="applied",
        watcher_ids=[],
        context_baseline={},
    )
    if status == "running":
        store.mark_agent_task_running(operation_id)
    elif status == "succeeded":
        store.complete_agent_task(operation_id, applied_revision=None, result={})
    elif status == "failed":
        store.fail_agent_task(operation_id, "fixture failure")
    else:
        raise ValueError(f"Unsupported Experiment fixture status: {status}")
    if arm_watcher:
        store.create_watchers(
            [
                WatcherRecord(
                    watcher_id=f"watcher-{operation_id}",
                    project_id=project_id,
                    origin_operation_id=operation_id,
                    origin_task_kind="node_chat",
                    chat_id=chat_id,
                    node_id=control_node_id,
                    execution_host="",
                    check_command="true",
                    log_path=f"/tmp/{operation_id}.log",
                    cwd="/tmp",
                    continuation=WatcherContinuation(
                        provider="codex",
                        model="",
                        reasoning="medium",
                        run_on="local",
                        run_truth_scope=["repo-a"],
                        patch_kind="experiment_loop",
                        control_node_id=control_node_id,
                        control_revision=7,
                        control_episode_id=episode_id,
                        control_invocation=1,
                        control_invocation_ceiling=3,
                        control_decision_bundle=[],
                        control_completion_criteria=["Detached work has finished."],
                    ),
                    created_at=now,
                )
            ]
        )
    return episode_id, operation_id


_PROJECT_ID_TABLES = (
    "projects",
    "paper_drafts",
    "writing_sessions",
    "chat_session_contexts",
    "graph_runs",
    "episodes",
    "agent_usage",
    "watchers",
)


def _seed_project_identity_rows(
    store: AppStore,
    project_id: str,
    *,
    label: str,
) -> ProjectRecord:
    record = ProjectRecord(
        project_id=project_id,
        locator=f"/tmp/{label}/research.yaml",
        name=f"Project {label}",
        state_location=f"ssh://host/{label}/.research",
        state_remote=True,
        added_at="2026-08-01T01:02:03+00:00",
        last_opened_at="2026-08-02T01:02:03+00:00",
        revision=7,
        primary_question="Does identity survive?",
        attention_count=3,
        last_refresh_at="2026-08-03T01:02:03+00:00",
        reachable=False,
        error="saved diagnostic",
    )
    store.upsert_project(record)
    with store.connection() as connection:
        connection.execute(
            """
            INSERT INTO paper_drafts (
                project_id, content, base_hash, updated_at, cursor_state, ancestor_content
            ) VALUES (?, '# Draft', 'base', '2026-08-04T01:02:03+00:00', 'cursor', '# Ancestor')
            """,
            (project_id,),
        )
        connection.execute(
            """
            INSERT INTO writing_sessions (
                native_session_id, provider, execution_machine, project_id, title, model,
                reasoning, created_at, last_resumed_at, introduction_hash_examined,
                graph_revision_examined, research_md_hash_examined
            ) VALUES (?, 'codex', 'laptop', ?, 'Session title', 'model', 'high', ?, ?,
                      'introduction', 7, 'research')
            """,
            (
                f"session-{label}",
                project_id,
                "2026-08-04T01:02:03+00:00",
                "2026-08-05T01:02:03+00:00",
            ),
        )
    snapshot_json, snapshot_digest = _snapshot(label)
    store.commit_chat_session_context(
        provider="codex",
        execution_machine="laptop",
        native_session_id=f"chat-session-{label}",
        project_id=project_id,
        kind="project_chat",
        chat_id=f"chat-{label}",
        node_id=None,
        protocol_version=1,
        snapshot_json=snapshot_json,
        snapshot_sha256=snapshot_digest,
        committed_operation_id=f"chat-operation-{label}",
        expected_snapshot_sha256=None,
    )
    _, operation_id = _create_experiment_runtime_fixture(
        store,
        project_id=project_id,
        control_node_id=f"exp/{label}",
        operation_id=f"operation-{label}",
        status="succeeded",
        arm_watcher=True,
    )
    store.record_agent_usage(
        operation_id,
        ProviderUsage(
            provider_profile="codex.turn.v1",
            provider_event_type="turn.completed",
            dedupe_key=f"usage-{label}",
            processed_input_tokens=101,
            generated_tokens=17,
            cached_input_tokens=23,
            provider_fields={"label": label},
        ),
    )
    return record


def _project_identity_rows(store: AppStore) -> dict[str, list[dict[str, object]]]:
    with store.connection() as connection:
        rows = {
            table: [
                dict(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY rowid")
            ]
            for table in _PROJECT_ID_TABLES
        }
        rows["project_aliases"] = [
            dict(row)
            for row in connection.execute("SELECT * FROM project_aliases ORDER BY alias_id")
        ]
    return rows


def _insert_destination_conflict(
    store: AppStore,
    table: str,
    *,
    old_project_id: str,
    canonical_project_id: str,
) -> None:
    if table == "projects":
        store.upsert_project(
            ProjectRecord(
                project_id=canonical_project_id,
                locator="/tmp/destination/research.yaml",
                name="Destination",
                state_location="/tmp/destination/.research",
                state_remote=False,
                added_at="2026-08-06T01:02:03+00:00",
            )
        )
        return

    primary_keys = {
        "writing_sessions": "native_session_id",
        "chat_session_contexts": "native_session_id",
        "graph_runs": "operation_id",
        "episodes": "episode_id",
        "agent_usage": "usage_id",
        "watchers": "watcher_id",
    }
    with store.connection() as connection:
        source = connection.execute(
            f"SELECT * FROM {table} WHERE project_id = ? LIMIT 1", (old_project_id,)
        ).fetchone()
        assert source is not None
        cloned = dict(source)
        cloned["project_id"] = canonical_project_id
        primary_key = primary_keys.get(table)
        if primary_key is not None:
            cloned[primary_key] = f"destination-{cloned[primary_key]}"
        columns = list(cloned)
        connection.execute(
            f"INSERT INTO {table} ({', '.join(columns)}) "
            f"VALUES ({', '.join('?' for _ in columns)})",
            tuple(cloned[column] for column in columns),
        )


def test_existing_project_table_gains_nullable_home_and_alias_storage(tmp_path) -> None:
    path = tmp_path / "rcp.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE projects (
                project_id TEXT PRIMARY KEY,
                locator TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                state_location TEXT NOT NULL,
                state_remote INTEGER NOT NULL,
                added_at TEXT NOT NULL,
                last_opened_at TEXT,
                revision INTEGER,
                primary_question TEXT,
                attention_count INTEGER NOT NULL DEFAULT 0,
                last_refresh_at TEXT,
                reachable INTEGER,
                error TEXT
            );
            INSERT INTO projects (
                project_id, locator, name, state_location, state_remote, added_at
            ) VALUES (
                'legacy-id', '/tmp/legacy/research.yaml', 'Legacy', '/tmp/legacy/.research',
                0, '2026-08-01T00:00:00+00:00'
            );
            """
        )

    store = AppStore(path)

    with store.connection() as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(projects)")}
        aliases_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'project_aliases'"
        ).fetchone()
    assert "home_space_id" in columns
    assert aliases_table is not None
    assert store.project("legacy-id").home_space_id is None
    assert store.project_aliases() == {}


def test_project_identity_migration_moves_all_rows_and_is_idempotent(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    old_project_id = "derived-project-id"
    canonical_project_id = str(uuid.uuid4())
    home_space_id = store.space_id
    original = _seed_project_identity_rows(store, old_project_id, label="source")
    before = _project_identity_rows(store)

    migrated = store.migrate_project_identity(
        old_project_id,
        canonical_project_id,
        home_space_id,
    )

    assert migrated.project_id == canonical_project_id
    assert migrated.home_space_id == home_space_id
    assert migrated.model_dump(exclude={"project_id", "home_space_id"}) == original.model_dump(
        exclude={"project_id", "home_space_id"}
    )
    assert store.resolve_project_id(old_project_id) == canonical_project_id
    assert store.resolve_project_id(canonical_project_id) == canonical_project_id
    assert store.project(old_project_id) == migrated
    assert store.project(canonical_project_id) == migrated
    assert store.project_aliases() == {old_project_id: canonical_project_id}
    assert [project.project_id for project in store.projects()] == [canonical_project_id]

    after = _project_identity_rows(store)
    for table in _PROJECT_ID_TABLES:
        assert all(row["project_id"] == canonical_project_id for row in after[table])
        before_without_identity = [
            {key: value for key, value in row.items() if key not in {"project_id", "home_space_id"}}
            for row in before[table]
        ]
        after_without_identity = [
            {key: value for key, value in row.items() if key not in {"project_id", "home_space_id"}}
            for row in after[table]
        ]
        assert after_without_identity == before_without_identity

    retry = store.migrate_project_identity(
        old_project_id,
        canonical_project_id,
        home_space_id,
    )
    assert retry == migrated
    assert _project_identity_rows(store) == after


@pytest.mark.parametrize("field", ["canonical_project_id", "home_space_id"])
def test_project_identity_migration_rejects_noncanonical_uuid4(tmp_path, field) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _seed_project_identity_rows(store, "derived-project-id", label="source")
    values = {
        "canonical_project_id": str(uuid.uuid4()),
        "home_space_id": store.space_id,
    }
    values[field] = str(uuid.uuid1())
    before = _project_identity_rows(store)

    with pytest.raises(ValueError, match="canonical UUIDv4"):
        store.migrate_project_identity(
            "derived-project-id",
            values["canonical_project_id"],
            values["home_space_id"],
        )

    assert _project_identity_rows(store) == before


@pytest.mark.parametrize("field", ["project_id", "home_space_id"])
def test_project_record_rejects_invalid_canonical_nameplate_ids(field) -> None:
    values = {
        **_project(str(uuid.uuid4())).model_dump(),
        "home_space_id": str(uuid.uuid4()),
    }
    values[field] = str(uuid.uuid1())
    with pytest.raises(ValueError, match="canonical UUIDv4"):
        ProjectRecord.model_validate(values)


@pytest.mark.parametrize("conflict_table", _PROJECT_ID_TABLES)
def test_project_identity_destination_conflict_rolls_back_every_table(
    tmp_path,
    conflict_table,
) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    old_project_id = "derived-project-id"
    canonical_project_id = str(uuid.uuid4())
    _seed_project_identity_rows(store, old_project_id, label="source")
    _insert_destination_conflict(
        store,
        conflict_table,
        old_project_id=old_project_id,
        canonical_project_id=canonical_project_id,
    )
    before = _project_identity_rows(store)

    with pytest.raises(ValueError, match="destination"):
        store.migrate_project_identity(old_project_id, canonical_project_id, store.space_id)

    assert _project_identity_rows(store) == before
    assert store.project_aliases() == {}
    assert store.project(old_project_id).project_id == old_project_id


def test_project_identity_alias_collision_rolls_back_without_changes(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    old_project_id = "derived-project-id"
    canonical_project_id = str(uuid.uuid4())
    conflicting_project_id = str(uuid.uuid4())
    _seed_project_identity_rows(store, old_project_id, label="source")
    with store.connection() as connection:
        connection.execute(
            """
            INSERT INTO project_aliases(alias_id, canonical_project_id)
            VALUES (?, ?)
            """,
            (old_project_id, conflicting_project_id),
        )
    before = _project_identity_rows(store)

    with pytest.raises(ValueError, match="already resolves"):
        store.migrate_project_identity(old_project_id, canonical_project_id, store.space_id)

    assert _project_identity_rows(store) == before


def test_project_deletion_preserves_alias_for_canonical_reregistration(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    old_project_id = "derived-project-id"
    canonical_project_id = str(uuid.uuid4())
    home_space_id = store.space_id
    _seed_project_identity_rows(store, old_project_id, label="source")
    store.migrate_project_identity(old_project_id, canonical_project_id, home_space_id)

    store.delete_project_records(canonical_project_id)

    assert store.project(old_project_id) is None
    assert store.resolve_project_id(old_project_id) == canonical_project_id
    assert store.project_aliases() == {old_project_id: canonical_project_id}

    recovered_id = store.resolve_project_id(old_project_id)
    recovered = _project(recovered_id).model_copy(update={"home_space_id": home_space_id})
    store.upsert_project(recovered)
    assert store.project(old_project_id) == store.project(canonical_project_id) == recovered


def test_brief_database_write_contention_waits_then_succeeds(tmp_path) -> None:
    store = _WriteTracingAppStore(tmp_path / "rcp.sqlite3")

    with store.connection() as connection:
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 30_000

    with ThreadPoolExecutor(max_workers=1) as executor:
        with sqlite3.connect(store.path) as blocker:
            blocker.execute("BEGIN IMMEDIATE")
            future = executor.submit(_task, store, "project-a", "operation-a", "queued")

            assert store.write_attempted.wait(timeout=2)
            assert not future.done()
            blocker.commit()

        future.result(timeout=5)

    assert store.agent_task("operation-a") is not None


def test_experiment_runtime_batch_matches_scalar_for_active_stopped_and_empty(
    tmp_path,
) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    project_id = "runtime-project"
    active_episode, _ = _create_experiment_runtime_fixture(
        store,
        project_id=project_id,
        control_node_id="exp/active",
        operation_id="active-loop",
        status="succeeded",
        arm_watcher=True,
    )
    stopped_episode, stopped_operation = _create_experiment_runtime_fixture(
        store,
        project_id=project_id,
        control_node_id="exp/stopped",
        operation_id="stopped-loop",
        status="failed",
    )
    store.record_experiment_episode_diagnostic(
        episode_id=stopped_episode,
        project_id=project_id,
        control_node_id="exp/stopped",
        diagnostic="The saved native session is unavailable.",
    )
    store.request_experiment_loop_stop(project_id, "exp/stopped")

    runtimes = store.experiment_loop_runtimes(
        project_id,
        ["exp/active", "exp/stopped", "exp/empty"],
    )
    project_runtimes = store.project_experiment_loop_runtimes(project_id)

    assert set(project_runtimes) == {"exp/active", "exp/stopped"}
    assert project_runtimes["exp/active"] == runtimes["exp/active"]
    assert project_runtimes["exp/stopped"] == runtimes["exp/stopped"]
    assert runtimes["exp/active"] == store.experiment_loop_runtime(project_id, "exp/active")
    assert runtimes["exp/stopped"] == store.experiment_loop_runtime(project_id, "exp/stopped")
    assert (
        runtimes["exp/empty"]
        == store.experiment_loop_runtime(project_id, "exp/empty")
        == ExperimentLoopRuntime()
    )
    assert runtimes["exp/active"].episode_id == active_episode
    assert runtimes["exp/active"].active is True
    assert runtimes["exp/active"].task_active is False
    assert runtimes["exp/active"].detached_work_active is True
    assert runtimes["exp/active"].session_bound is True
    assert runtimes["exp/active"].model == ""
    assert runtimes["exp/stopped"].episode_id == stopped_episode
    assert runtimes["exp/stopped"].active is False
    assert runtimes["exp/stopped"].task_active is False
    assert runtimes["exp/stopped"].stop_requested is True
    assert runtimes["exp/stopped"].stop_settled is True
    assert "experiment_recovery_abandoned" in {
        receipt.category for receipt in store.agent_task_receipts(stopped_operation)
    }


def test_experiment_runtime_batch_select_count_is_constant(tmp_path) -> None:
    store = _TracingAppStore(tmp_path / "rcp.sqlite3")
    project_id = "runtime-project"
    control_node_ids = []
    for index in range(12):
        control_node_id = f"exp/runtime-{index}"
        control_node_ids.append(control_node_id)
        _create_experiment_runtime_fixture(
            store,
            project_id=project_id,
            control_node_id=control_node_id,
            operation_id=f"loop-{index}",
            status="running",
        )

    store.select_count = 0
    store.experiment_loop_runtimes(project_id, control_node_ids[:1])
    one_experiment_selects = store.select_count
    store.select_count = 0
    runtimes = store.experiment_loop_runtimes(project_id, control_node_ids)
    all_experiment_selects = store.select_count

    assert set(runtimes) == set(control_node_ids)
    assert one_experiment_selects == all_experiment_selects == 5

    store.select_count = 0
    assert set(store.project_experiment_loop_runtimes(project_id)) == set(control_node_ids)
    assert store.select_count == 5

    store.select_count = 0
    assert store.active_experiment_control_ids(project_id) == set(control_node_ids)
    assert store.select_count == 5


def test_multiple_active_agent_tasks_can_share_a_project(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    store.upsert_project(_project("project"))

    _task(store, "project", "first-run", "running")
    _task(store, "project", "second-run", "queued")

    assert store.agent_task("first-run") is not None
    assert store.agent_task("second-run") is not None


def test_has_active_chat_task_is_scoped_to_project_kind_and_chat(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    now = store.now()
    tasks = [
        ("project-a-chat-a", "project-a", "project_chat", "chat-a", "running"),
        ("project-a-chat-b", "project-a", "project_chat", "chat-b", "queued"),
        ("project-b-chat-a", "project-b", "project_chat", "chat-a", "pausing"),
        ("finished", "project-a", "node_chat", "finished-chat", "succeeded"),
    ]
    for operation_id, project_id, kind, chat_id, status in tasks:
        store.create_agent_task(
            AgentTaskRecord(
                operation_id=operation_id,
                project_id=project_id,
                kind=kind,
                status=status,
                request={"chat_id": chat_id},
                created_at=now,
                updated_at=now,
                status_message=status,
            )
        )

    assert store.has_active_chat_task("project-a", "project_chat", "chat-a")
    assert store.has_active_chat_task("project-a", "project_chat", "chat-b")
    assert store.has_active_chat_task("project-b", "project_chat", "chat-a")
    assert not store.has_active_chat_task("project-a", "node_chat", "chat-a")
    assert not store.has_active_chat_task("project-a", "node_chat", "finished-chat")
    assert not store.has_active_chat_task("project-a", "project_chat", "missing")


def test_create_chat_task_rejects_only_an_active_turn_in_the_same_conversation(
    tmp_path,
) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    now = store.now()

    def create(operation_id: str, chat_id: str) -> None:
        store.create_agent_task(
            AgentTaskRecord(
                operation_id=operation_id,
                project_id="project",
                kind="project_chat",
                status="queued",
                request={"chat_id": chat_id},
                created_at=now,
                updated_at=now,
                status_message="queued",
            )
        )

    create("chat-a-first", "chat-a")
    create("chat-b", "chat-b")

    with pytest.raises(ValueError, match="already active in this conversation"):
        create("chat-a-overlap", "chat-a")

    assert store.agent_task("chat-b") is not None
    assert store.agent_task("chat-a-overlap") is None


def test_native_chat_session_origin_requires_the_exact_rcp_binding(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    now = store.now()
    store.create_agent_task(
        AgentTaskRecord(
            operation_id="completed-chat-turn",
            project_id="project",
            kind="node_chat",
            status="succeeded",
            request={
                "chat_id": "chat-a",
                "node_id": "rq/example",
                "provider": "codex",
                "run_on": "local",
            },
            created_at=now,
            updated_at=now,
            status_message="succeeded",
            native_session_id="native-session",
        )
    )

    assert store.has_chat_native_session_origin(
        "project",
        "node_chat",
        "chat-a",
        "rq/example",
        "codex",
        "local",
        "native-session",
    )
    assert not store.has_chat_native_session_origin(
        "project",
        "node_chat",
        "chat-b",
        "rq/example",
        "codex",
        "local",
        "native-session",
    )


def test_chat_session_context_commit_reads_and_cas_updates_without_task_history(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    first_json, first_digest = _snapshot("first")

    first = store.commit_chat_session_context(
        provider="codex",
        execution_machine="laptop",
        native_session_id="native-session",
        project_id="project",
        kind="node_chat",
        chat_id="chat",
        node_id="rq/question",
        protocol_version=1,
        snapshot_json=first_json,
        snapshot_sha256=first_digest,
        committed_operation_id="first-operation",
        expected_snapshot_sha256=None,
    )

    assert first == store.chat_session_context("codex", "laptop", "native-session")
    assert first == store.validate_chat_session_context_binding(
        "codex",
        "laptop",
        "native-session",
        project_id="project",
        kind="node_chat",
        chat_id="chat",
        node_id="rq/question",
    )
    assert first.snapshot_json == first_json
    assert first.snapshot_sha256 == first_digest
    assert first.committed_operation_id == "first-operation"
    assert first.created_at == first.updated_at

    second_json, second_digest = _snapshot("second")
    second = store.commit_chat_session_context(
        provider="codex",
        execution_machine="laptop",
        native_session_id="native-session",
        project_id="project",
        kind="node_chat",
        chat_id="chat",
        node_id="rq/question",
        protocol_version=2,
        snapshot_json=second_json,
        snapshot_sha256=second_digest,
        committed_operation_id="second-operation",
        expected_snapshot_sha256=first_digest,
    )

    assert second.created_at == first.created_at
    assert second.protocol_version == 2
    assert second.snapshot_json == second_json
    assert second.snapshot_sha256 == second_digest
    assert second.committed_operation_id == "second-operation"


def test_chat_session_context_cas_rejects_missing_stale_or_invalid_snapshots(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    first_json, first_digest = _snapshot("first")

    with pytest.raises(ValueError, match="prior baseline is missing"):
        store.commit_chat_session_context(
            provider="codex",
            execution_machine="laptop",
            native_session_id="native-session",
            project_id="project",
            kind="project_chat",
            chat_id="chat",
            node_id=None,
            protocol_version=1,
            snapshot_json=first_json,
            snapshot_sha256=first_digest,
            committed_operation_id="operation",
            expected_snapshot_sha256="missing-digest",
        )

    store.commit_chat_session_context(
        provider="codex",
        execution_machine="laptop",
        native_session_id="native-session",
        project_id="project",
        kind="project_chat",
        chat_id="chat",
        node_id=None,
        protocol_version=1,
        snapshot_json=first_json,
        snapshot_sha256=first_digest,
        committed_operation_id="first-operation",
        expected_snapshot_sha256=None,
    )
    second_json, second_digest = _snapshot("second")
    with pytest.raises(ValueError, match="prior digest changed"):
        store.commit_chat_session_context(
            provider="codex",
            execution_machine="laptop",
            native_session_id="native-session",
            project_id="project",
            kind="project_chat",
            chat_id="chat",
            node_id=None,
            protocol_version=2,
            snapshot_json=second_json,
            snapshot_sha256=second_digest,
            committed_operation_id="stale-operation",
            expected_snapshot_sha256="stale-digest",
        )
    with pytest.raises(ValueError, match="does not match"):
        store.commit_chat_session_context(
            provider="codex",
            execution_machine="laptop",
            native_session_id="native-session",
            project_id="project",
            kind="project_chat",
            chat_id="chat",
            node_id=None,
            protocol_version=2,
            snapshot_json=second_json,
            snapshot_sha256="wrong-digest",
            committed_operation_id="invalid-operation",
            expected_snapshot_sha256=first_digest,
        )
    assert store.chat_session_context("codex", "laptop", "native-session").snapshot_sha256 == (
        first_digest
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("project_id", "other-project"),
        ("kind", "project_chat"),
        ("chat_id", "other-chat"),
        ("node_id", "rq/other"),
    ],
)
def test_chat_session_context_rejects_immutable_binding_conflicts(tmp_path, field, value) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    snapshot_json, snapshot_digest = _snapshot("first")
    store.commit_chat_session_context(
        provider="codex",
        execution_machine="laptop",
        native_session_id="native-session",
        project_id="project",
        kind="node_chat",
        chat_id="chat",
        node_id="rq/question",
        protocol_version=1,
        snapshot_json=snapshot_json,
        snapshot_sha256=snapshot_digest,
        committed_operation_id="operation",
        expected_snapshot_sha256=None,
    )
    binding = {
        "project_id": "project",
        "kind": "node_chat",
        "chat_id": "chat",
        "node_id": "rq/question",
    }
    binding[field] = value

    with pytest.raises(ValueError, match=f"immutable binding conflict: {field}"):
        store.validate_chat_session_context_binding("codex", "laptop", "native-session", **binding)
    second_json, second_digest = _snapshot("second")
    with pytest.raises(ValueError, match=f"immutable binding conflict: {field}"):
        store.commit_chat_session_context(
            provider="codex",
            execution_machine="laptop",
            native_session_id="native-session",
            **binding,
            protocol_version=2,
            snapshot_json=second_json,
            snapshot_sha256=second_digest,
            committed_operation_id="conflicting-operation",
            expected_snapshot_sha256=snapshot_digest,
        )


@pytest.mark.parametrize(
    ("provider", "execution_machine"),
    [("claude", "laptop"), ("codex", "remote")],
)
def test_chat_session_context_rejects_provider_or_machine_conflicts(
    tmp_path, provider, execution_machine
) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    snapshot_json, snapshot_digest = _snapshot("first")
    store.commit_chat_session_context(
        provider="codex",
        execution_machine="laptop",
        native_session_id="native-session",
        project_id="project",
        kind="project_chat",
        chat_id="chat",
        node_id=None,
        protocol_version=1,
        snapshot_json=snapshot_json,
        snapshot_sha256=snapshot_digest,
        committed_operation_id="operation",
        expected_snapshot_sha256=None,
    )

    with pytest.raises(ValueError, match="provider or execution-machine conflict"):
        store.chat_session_context(provider, execution_machine, "native-session")
    with pytest.raises(ValueError, match="provider or execution-machine conflict"):
        store.commit_chat_session_context(
            provider=provider,
            execution_machine=execution_machine,
            native_session_id="native-session",
            project_id="project",
            kind="project_chat",
            chat_id="chat",
            node_id=None,
            protocol_version=2,
            snapshot_json=snapshot_json,
            snapshot_sha256=snapshot_digest,
            committed_operation_id="conflicting-operation",
            expected_snapshot_sha256=None,
        )


def test_chat_session_context_record_forbids_extra_fields(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    snapshot_json, snapshot_digest = _snapshot("first")
    record = store.commit_chat_session_context(
        provider="codex",
        execution_machine="laptop",
        native_session_id="native-session",
        project_id="project",
        kind="project_chat",
        chat_id="chat",
        node_id=None,
        protocol_version=1,
        snapshot_json=snapshot_json,
        snapshot_sha256=snapshot_digest,
        committed_operation_id="operation",
        expected_snapshot_sha256=None,
    )

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        ChatSessionContextRecord.model_validate({**record.model_dump(), "transcript": []})


def test_project_record_deletion_is_atomic_complete_and_project_scoped(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    store.upsert_project(_project("delete-me"))
    store.upsert_project(_project("keep-me"))
    _task(store, "delete-me", "delete-run", "failed")
    _task(store, "keep-me", "keep-run", "failed")
    owner = store.local_owner
    assert owner is not None
    store.seat_project_member("delete-me", owner.user_id)
    store.seat_project_member("keep-me", owner.user_id)

    for operation_id in ("delete-run", "keep-run"):
        store.record_agent_task_event(operation_id, "event")
        store.record_agent_task_receipt(operation_id, "receipt", {"value": True})
        store.record_agent_task_patch_output(operation_id, "{}")
        store.record_agent_task_contract(operation_id, "base", "contract", "digest")
    with store.connection() as connection:
        for project_id in ("delete-me", "keep-me"):
            connection.execute(
                """
                INSERT INTO paper_drafts(project_id, content, updated_at)
                VALUES (?, 'draft', '2026-07-31T00:00:00+00:00')
                """,
                (project_id,),
            )
            connection.execute(
                """
                INSERT INTO writing_sessions (
                    native_session_id, provider, execution_machine, project_id,
                    model, created_at, last_resumed_at, introduction_hash_examined,
                    graph_revision_examined, research_md_hash_examined
                ) VALUES (?, 'provider', 'local', ?, 'model', ?, ?, 'intro', 1, 'research')
                """,
                (
                    f"{project_id}-session",
                    project_id,
                    "2026-07-31T00:00:00+00:00",
                    "2026-07-31T00:00:00+00:00",
                ),
            )
            snapshot_json, snapshot_digest = _snapshot(project_id)
            connection.execute(
                """
                INSERT INTO chat_session_contexts (
                    provider, execution_machine, native_session_id, project_id,
                    kind, chat_id, node_id, protocol_version, snapshot_json,
                    snapshot_sha256, committed_operation_id, created_at, updated_at
                ) VALUES ('provider', 'local', ?, ?, 'project_chat', ?, NULL, 1, ?, ?, ?, ?, ?)
                """,
                (
                    f"{project_id}-chat-session",
                    project_id,
                    f"{project_id}-chat",
                    snapshot_json,
                    snapshot_digest,
                    f"{project_id}-operation",
                    "2026-07-31T00:00:00+00:00",
                    "2026-07-31T00:00:00+00:00",
                ),
            )

    counts = store.delete_project_records("delete-me")

    assert counts == {
        "project_members": 1,
        "paper_drafts": 1,
        "writing_sessions": 1,
        "chat_session_contexts": 1,
        "watchers": 0,
        "graph_watcher_reconciliation": 0,
        "experiment_episode_state": 0,
        "result_views": 0,
        "auto_research_recoveries": 0,
        "auto_research_messages": 0,
        "auto_research_invocations": 0,
        "auto_research_episodes": 0,
        "episode_reports": 0,
        "episode_report_attempts": 0,
        "episode_wrapups": 0,
        "episode_invocations": 0,
        "episodes": 0,
        "graph_run_outputs": 1,
        "graph_run_events": 1,
        "graph_run_receipts": 2,
        "graph_run_contracts": 1,
        "graph_runs": 1,
        "projects": 1,
    }
    with store.connection() as connection:
        for table in (
            "paper_drafts",
            "writing_sessions",
            "chat_session_contexts",
            "graph_runs",
            "graph_run_outputs",
            "graph_run_contracts",
            "graph_run_events",
            "graph_run_receipts",
        ):
            expected = 2 if table == "graph_run_receipts" else 1
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == expected
    assert store.project("delete-me") is None
    assert store.project("keep-me") is not None
    assert store.project_members("delete-me") == []
    assert [record.project_id for record in store.project_members("keep-me")] == ["keep-me"]

    reopened = AppStore(store.path)
    assert reopened.project("delete-me") is None
    assert reopened.project("keep-me") is not None


def test_chat_session_context_project_id_migrates_with_legacy_project_data(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    project_id = str(uuid.uuid4())
    snapshot_json, snapshot_digest = _snapshot("legacy")
    store.commit_chat_session_context(
        provider="codex",
        execution_machine="laptop",
        native_session_id="native-session",
        project_id="legacy-project",
        kind="project_chat",
        chat_id="chat",
        node_id=None,
        protocol_version=1,
        snapshot_json=snapshot_json,
        snapshot_sha256=snapshot_digest,
        committed_operation_id="operation",
        expected_snapshot_sha256=None,
    )

    with pytest.raises(ValueError, match="not an exact canonical project registration"):
        store.migrate_legacy_project_data("legacy-project", project_id)
    assert (
        store.validate_chat_session_context_binding(
            "codex",
            "laptop",
            "native-session",
            project_id="legacy-project",
            kind="project_chat",
            chat_id="chat",
            node_id=None,
        )
        is not None
    )

    store.upsert_project(_project(project_id).model_copy(update={"home_space_id": store.space_id}))
    store.migrate_legacy_project_data("legacy-project", project_id)

    migrated = store.validate_chat_session_context_binding(
        "codex",
        "laptop",
        "native-session",
        project_id=project_id,
        kind="project_chat",
        chat_id="chat",
        node_id=None,
    )
    assert migrated is not None
    assert migrated.project_id == project_id


_LEGACY_PROJECT_DATA_TABLES = (
    "paper_drafts",
    "writing_sessions",
    "chat_session_contexts",
    "result_views",
    "graph_runs",
    "episodes",
    "agent_usage",
    "watchers",
)


def _seed_legacy_project_data_rows(store: AppStore, project_id: str, *, label: str) -> str:
    _seed_project_identity_rows(store, project_id, label=label)
    operation_id = f"operation-{label}"
    created_at = "2026-08-12T01:02:03+00:00"
    user_id = str(uuid.uuid4())
    dispatch_authority_json = json.dumps(
        {
            "profile": "ordinary",
            "scope": {"patch_kind": "refresh", "run_truth_scope": ["repo-a"]},
            "task_contract": "scratch_patch",
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    with store.connection() as connection:
        connection.execute(
            """
            INSERT INTO result_views (
                view_id, project_id, experiment_id, chat_id,
                origin_operation_id, latest_operation_id,
                provider, model, reasoning, run_on,
                native_session_id, stage_host, stage_root, source_name,
                content_sha256, size_bytes, html, created_at, updated_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'codex', 'gpt', 'high', 'laptop', ?, '', ?,
                      'result.html', 'sha256', 7, 'result', ?, ?,
                      '2026-08-13T01:02:03+00:00')
            """,
            (
                f"legacy-view-{label}",
                project_id,
                f"legacy-experiment-{label}",
                f"legacy-chat-{label}",
                operation_id,
                operation_id,
                f"legacy-native-session-{label}",
                f"/tmp/legacy-stage-{label}",
                created_at,
                created_at,
            ),
        )
        connection.execute(
            """
            UPDATE graph_runs
            SET dispatch_authority_json = ?,
                authorized_space_id = ?, authorized_user_id = ?,
                authorized_display_name = 'Original Authorizer'
            WHERE operation_id = ?
            """,
            (
                dispatch_authority_json,
                store.space_id,
                user_id,
                operation_id,
            ),
        )
    return operation_id


def _legacy_project_data_state(store: AppStore) -> dict[str, list[dict[str, object]]]:
    tables = ("projects", "project_aliases", *_LEGACY_PROJECT_DATA_TABLES)
    with store.connection() as connection:
        return {
            table: [
                dict(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY rowid")
            ]
            for table in tables
        }


@pytest.mark.parametrize(
    ("collision", "diagnostic"),
    [
        ("canonical", "registered canonical project"),
        ("foreign_alias", "belongs to canonical project"),
    ],
)
def test_legacy_project_data_migration_refuses_owned_source_identity_atomically(
    tmp_path,
    collision,
    diagnostic,
) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    target_project_id = str(uuid.uuid4())
    store.upsert_project(
        _project(target_project_id).model_copy(update={"home_space_id": store.space_id})
    )
    if collision == "canonical":
        legacy_id = str(uuid.uuid4())
        store.upsert_project(
            _project(legacy_id).model_copy(update={"home_space_id": store.space_id})
        )
        operation_id = _seed_legacy_project_data_rows(store, legacy_id, label=collision)
    else:
        legacy_id = "foreign-project-alias"
        operation_id = _seed_legacy_project_data_rows(store, legacy_id, label=collision)
        foreign_project_id = str(uuid.uuid4())
        store.upsert_project(
            _project(foreign_project_id).model_copy(update={"home_space_id": store.space_id})
        )
        with store.connection() as connection:
            connection.execute("DELETE FROM projects WHERE project_id = ?", (legacy_id,))
            connection.execute(
                """
                INSERT INTO project_aliases(alias_id, canonical_project_id)
                VALUES (?, ?)
                """,
                (legacy_id, foreign_project_id),
            )
    before = _legacy_project_data_state(store)
    source_authority = store.agent_task_authority(legacy_id, operation_id)
    source_authorizer = store.agent_task_authorizer(operation_id)

    with pytest.raises(ValueError, match=diagnostic):
        store.migrate_legacy_project_data(legacy_id, target_project_id)

    assert _legacy_project_data_state(store) == before
    assert store.agent_task_authority(legacy_id, operation_id) == source_authority
    assert store.agent_task_authorizer(operation_id) == source_authorizer
    with pytest.raises(KeyError, match=operation_id):
        store.agent_task_authority(target_project_id, operation_id)


def test_legacy_project_data_migration_allows_same_target_alias_idempotently(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    project_id = str(uuid.uuid4())
    legacy_id = "same-project-alias"
    store.upsert_project(_project(project_id).model_copy(update={"home_space_id": store.space_id}))
    with store.connection() as connection:
        connection.execute(
            """
            INSERT INTO project_aliases(alias_id, canonical_project_id)
            VALUES (?, ?)
            """,
            (legacy_id, project_id),
        )
    snapshot_json, snapshot_digest = _snapshot("same-target-alias")
    store.commit_chat_session_context(
        provider="codex",
        execution_machine="laptop",
        native_session_id="same-target-session",
        project_id=legacy_id,
        kind="project_chat",
        chat_id="chat",
        node_id=None,
        protocol_version=1,
        snapshot_json=snapshot_json,
        snapshot_sha256=snapshot_digest,
        committed_operation_id="operation",
        expected_snapshot_sha256=None,
    )

    store.migrate_legacy_project_data(legacy_id, project_id)
    store.migrate_legacy_project_data(legacy_id, project_id)

    migrated = store.chat_session_context("codex", "laptop", "same-target-session")
    assert migrated is not None
    assert migrated.project_id == project_id
    assert store.project_aliases() == {legacy_id: project_id}


def test_agent_usage_is_counted_once_and_snapshot_uses_weighted_cache_share(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    store.upsert_project(_project("project"))
    now = store.now()
    store.create_agent_task(
        AgentTaskRecord(
            operation_id="refresh-operation",
            project_id="project",
            kind="refresh",
            status="succeeded",
            request={"provider": "codex", "model": "gpt"},
            created_at=now,
            updated_at=now,
            status_message="done",
        )
    )
    usage = ProviderUsage(
        provider_profile="codex.turn.v1",
        provider_event_type="turn.completed",
        dedupe_key="turn-1",
        processed_input_tokens=1_000,
        generated_tokens=100,
        cached_input_tokens=400,
        provider_fields={"input_tokens": 1_000},
    )

    counted = store.record_agent_usage("refresh-operation", usage)
    duplicate = store.record_agent_usage("refresh-operation", usage)

    assert counted.counted is True
    assert duplicate.counted is False
    assert duplicate.count_reason == "duplicate"
    snapshot = store.agent_usage_snapshot("project")
    assert snapshot.counted_records == 1
    assert snapshot.excluded_records == 1
    assert snapshot.input_processed.total_tokens == 1_000
    assert snapshot.generated.total_tokens == 100
    assert snapshot.input_processed.cache_share == 0.4
    assert snapshot.input_processed.block_tokens == 50
    assert snapshot.input_processed.cells[0].task_kind == "refresh"


def test_agent_usage_snapshot_counts_latest_input_context_once_per_native_session(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    store.upsert_project(_project("project"))
    now = store.now()
    tasks = [
        AgentTaskRecord(
            operation_id=operation_id,
            project_id="project",
            kind="node_chat",
            status="succeeded",
            request={"provider": provider, "model": "model"},
            created_at=now,
            updated_at=now,
            status_message="done",
            native_session_id="shared-session",
        )
        for operation_id, provider in (
            ("codex-first", "codex"),
            ("codex-latest", "codex"),
            ("claude-latest", "claude"),
        )
    ]
    for task in tasks:
        store.create_agent_task(task)

    usages = [
        ("codex-first", "codex.turn.v1", "codex-1", 1_000, 400, 100),
        ("codex-latest", "codex.turn.v1", "codex-2", 2_000, 1_500, 200),
        ("claude-latest", "claude.query.v1", "claude-1", 500, 200, 50),
    ]
    for operation_id, profile, dedupe_key, input_tokens, cached_tokens, output_tokens in usages:
        store.record_agent_usage(
            operation_id,
            ProviderUsage(
                provider_profile=profile,
                provider_event_type="turn.completed",
                dedupe_key=dedupe_key,
                processed_input_tokens=input_tokens,
                generated_tokens=output_tokens,
                cached_input_tokens=cached_tokens,
            ),
        )

    with store.connection() as connection:
        for index, (operation_id, *_rest) in enumerate(usages):
            connection.execute(
                "UPDATE agent_usage SET created_at = ? WHERE operation_id = ?",
                (f"2026-08-04T00:00:0{index}+00:00", operation_id),
            )

    snapshot = store.agent_usage_snapshot("project")

    assert snapshot.input_processed.total_tokens == 2_500
    assert snapshot.input_processed.cached_tokens == 1_700
    assert snapshot.input_processed.cache_share == 1_700 / 2_500
    assert snapshot.generated.total_tokens == 350
    assert snapshot.input_processed.cells[0].counted_records == 1
    assert snapshot.input_processed.cells[1].counted_records == 1
    assert {cell.counted_records for cell in snapshot.generated.cells} == {1, 2}


@pytest.mark.parametrize("status", ["queued", "running", "pausing"])
def test_project_record_deletion_refuses_active_task(tmp_path, status) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    store.upsert_project(_project("project"))
    _task(store, "project", "operation", status)

    with pytest.raises(ValueError, match="Pause the active agent task"):
        store.project_deletion_stages("project")
    with pytest.raises(ValueError, match="Pause the active agent task"):
        store.delete_project_records("project")

    assert store.project("project") is not None
    assert store.agent_task("operation") is not None


def test_v02_graph_run_migrates_to_recoverable_interrupted_agent_task(tmp_path) -> None:
    path = tmp_path / "rcp.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE graph_runs (
                operation_id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                request_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                status_message TEXT NOT NULL,
                error TEXT,
                applied_revision INTEGER
            );
            CREATE UNIQUE INDEX graph_runs_active_project
                ON graph_runs(project_id)
                WHERE status IN ('queued', 'running');
            CREATE TABLE graph_run_receipts (
                receipt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                tier TEXT NOT NULL,
                category TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            """
        )
        connection.execute(
            """
            INSERT INTO graph_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "old-operation",
                "project",
                "seed",
                "running",
                json.dumps({"provider": "codex", "run_on": "laptop"}),
                "2026-07-27T10:00:00+00:00",
                "2026-07-27T10:00:01+00:00",
                "2026-07-27T10:00:01+00:00",
                None,
                "Working",
                None,
                None,
            ),
        )
        connection.execute(
            """
            INSERT INTO graph_run_receipts (
                operation_id, created_at, tier, category, payload_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                "old-operation",
                "2026-07-27T10:00:01+00:00",
                "summary",
                "operation_created",
                '{"kind":"seed"}',
            ),
        )

    store = AppStore(path)
    store.interrupt_active_agent_tasks()
    record = store.agent_task("old-operation")

    assert record is not None
    assert record.status == "interrupted"
    assert record.can_resume is False
    assert record.can_retry is True
    assert record.attempt == 1
    assert "Resume" in record.status_message
    assert record.result is None
    assert record.authorized_by is None
    assert store.agent_task_authorizer("old-operation") is None
    assert store.agent_task_events("old-operation")[0].level == "warning"
    assert [receipt.category for receipt in store.agent_task_receipts("old-operation")] == [
        "operation_created",
        "operation_interrupted",
    ]
    with store.connection() as connection:
        indexes = {row[1] for row in connection.execute("PRAGMA index_list(graph_runs)")}
        columns = {row[1] for row in connection.execute("PRAGMA table_info(graph_runs)")}
    assert "graph_runs_active_project" not in indexes
    assert "agent_tasks_active_project" not in indexes
    assert "graph_runs_episode" in indexes
    assert {
        "episode_id",
        "authorized_space_id",
        "authorized_user_id",
        "authorized_display_name",
    } <= columns
    assert "campaign_worker_handoffs_cleared_at" not in columns


def test_agent_task_authorizer_snapshot_round_trips_and_survives_restart(tmp_path) -> None:
    path = tmp_path / "rcp.sqlite3"
    store = AppStore(path)
    authorizer = AuthorizedHuman(
        space_id=store.space_id,
        user_id=str(uuid.uuid4()),
        display_name="Original Name",
    )
    now = store.now()

    created = store.create_agent_task(
        AgentTaskRecord(
            operation_id="authorized-operation",
            project_id="project",
            kind="refresh",
            status="succeeded",
            request={},
            created_at=now,
            updated_at=now,
            status_message="done",
            authorized_by=authorizer,
        )
    )

    assert created.authorized_by == authorizer
    assert store.agent_task_authorizer("authorized-operation") == authorizer
    restarted = AppStore(path)
    assert restarted.agent_task("authorized-operation").authorized_by == authorizer
    assert restarted.agent_task_authorizer("authorized-operation") == authorizer
    with restarted.connection() as connection:
        row = connection.execute(
            """
            SELECT authorized_space_id, authorized_user_id, authorized_display_name
            FROM graph_runs WHERE operation_id = 'authorized-operation'
            """
        ).fetchone()
    assert tuple(row) == (authorizer.space_id, authorizer.user_id, authorizer.display_name)


def test_space_user_rename_does_not_rewrite_agent_task_authorizer_snapshot(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    owner = store.local_owner
    assert owner is not None
    named_owner = store.rename_space_user(owner.user_id, "First Name")
    authorizer = AuthorizedHuman(
        space_id=store.space_id,
        user_id=named_owner.user_id,
        display_name=named_owner.display_name,
    )
    now = store.now()
    store.create_agent_task(
        AgentTaskRecord(
            operation_id="before-rename",
            project_id="project",
            kind="refresh",
            status="succeeded",
            request={},
            created_at=now,
            updated_at=now,
            status_message="done",
            authorized_by=authorizer,
        )
    )

    renamed_owner = store.rename_space_user(owner.user_id, "Second Name")

    assert renamed_owner.display_name == "Second Name"
    assert store.agent_task("before-rename").authorized_by == authorizer
    assert store.agent_task_authorizer("before-rename") == authorizer


@pytest.mark.parametrize("corruption", ["partial", "invalid"])
def test_agent_task_authorizer_snapshot_corruption_fails_closed(tmp_path, corruption) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _task(store, "project", "operation", "succeeded")
    with store.connection() as connection:
        if corruption == "partial":
            connection.execute(
                """
                UPDATE graph_runs SET authorized_space_id = ? WHERE operation_id = 'operation'
                """,
                (store.space_id,),
            )
        else:
            connection.execute(
                """
                UPDATE graph_runs
                SET authorized_space_id = ?, authorized_user_id = 'not-a-uuid',
                    authorized_display_name = 'Name'
                WHERE operation_id = 'operation'
                """,
                (store.space_id,),
            )

    with pytest.raises(RuntimeError, match="authorizer snapshot"):
        store.agent_task("operation")
    with pytest.raises(RuntimeError, match="authorizer snapshot"):
        store.agent_task_authorizer("operation")


def test_child_agent_tasks_use_only_the_explicitly_supplied_authorizer(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    parent_authorizer = AuthorizedHuman(
        space_id=store.space_id,
        user_id=str(uuid.uuid4()),
        display_name="Parent Authorizer",
    )
    child_authorizer = AuthorizedHuman(
        space_id=store.space_id,
        user_id=str(uuid.uuid4()),
        display_name="Child Authorizer",
    )
    now = store.now()
    request = RunRequest(run_truth_scope=["repo-a"])
    dispatch_authority = resolve_dispatch_authority("refresh", request)
    assert dispatch_authority is not None
    store.create_agent_task(
        AgentTaskRecord(
            operation_id="parent",
            project_id="project",
            kind="refresh",
            status="failed",
            request=request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="failed",
            authorized_by=parent_authorizer,
            dispatch_authority=dispatch_authority,
        )
    )
    store.create_agent_task(
        AgentTaskRecord(
            operation_id="explicit-child",
            project_id="project",
            kind="refresh",
            status="succeeded",
            request=request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="done",
            parent_operation_id="parent",
            authorized_by=child_authorizer,
            dispatch_authority=dispatch_authority,
        )
    )
    store.create_agent_task(
        AgentTaskRecord(
            operation_id="legacy-child",
            project_id="project",
            kind="refresh",
            status="succeeded",
            request=request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="done",
            parent_operation_id="parent",
            dispatch_authority=dispatch_authority,
        )
    )

    assert store.agent_task_authorizer("parent") == parent_authorizer
    assert store.agent_task_authorizer("explicit-child") == child_authorizer
    assert store.agent_task_authorizer("legacy-child") is None


def test_agent_task_authorizer_distinguishes_unknown_from_legacy_task(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _task(store, "project", "legacy-operation", "succeeded")

    assert store.agent_task_authorizer("legacy-operation") is None
    with pytest.raises(KeyError, match="unknown-operation"):
        store.agent_task_authorizer("unknown-operation")


def test_agent_task_attribution_schema_has_episode_lineage_without_duplicate_identity_fields(
    tmp_path,
) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    with store.connection() as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(graph_runs)")}

    assert {
        "authorized_space_id",
        "authorized_user_id",
        "authorized_display_name",
        "episode_id",
    } <= columns
    assert {
        "orchestrator_profile_id",
        "parent_task_id",
        "worker_id",
    }.isdisjoint(columns)


def test_existing_watcher_database_opens_and_gains_scheduling_columns(tmp_path) -> None:
    path = tmp_path / "rcp.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE watchers (
                watcher_id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                origin_operation_id TEXT NOT NULL,
                origin_task_kind TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                node_id TEXT,
                execution_host TEXT NOT NULL,
                check_command TEXT NOT NULL,
                log_path TEXT NOT NULL,
                cwd TEXT NOT NULL,
                continuation_json TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_checked_at TEXT,
                last_exit_code INTEGER,
                last_error TEXT,
                completed_at TEXT,
                notified INTEGER NOT NULL DEFAULT 0,
                notification_operation_id TEXT
            );
            """
        )
        connection.execute(
            "INSERT INTO watchers ("
            "watcher_id, project_id, origin_operation_id, origin_task_kind, chat_id, node_id,"
            "execution_host, check_command, log_path, cwd, continuation_json, status, created_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "old-watcher",
                "project",
                "old-operation",
                "node_chat",
                "chat",
                None,
                "",
                "squeue -j 1",
                "/logs/1.out",
                "/work",
                json.dumps({"provider": "codex", "run_on": "laptop", "patch_kind": "work"}),
                "active",
                "2026-08-01T10:00:00+00:00",
            ),
        )

    store = AppStore(path)
    record = store.watcher("old-watcher")

    assert record is not None
    assert record.next_check_at is None
    assert record.consecutive_error_count == 0
    assert record.group_id is None
    assert record.stopped_by is None
    assert store.pollable_watchers() == [record]


def test_patch_recovery_output_is_bounded(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    now = store.now()
    store.create_agent_task(
        AgentTaskRecord(
            operation_id="operation",
            project_id="project",
            kind="refresh",
            status="failed",
            request={},
            created_at=now,
            updated_at=now,
            status_message="failed",
        )
    )

    store.record_agent_task_patch_output("operation", '{"kind":"refresh"}')

    assert store.agent_task_patch_output("operation") == '{"kind":"refresh"}'
    with pytest.raises(ValueError, match="2 MB"):
        store.record_agent_task_patch_output("operation", "x" * 2_000_001)


def test_agent_task_admission_persists_exact_launch_intent(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    now = store.now()
    parent = store.create_agent_task(
        AgentTaskRecord(
            operation_id="parent",
            project_id="project",
            kind="refresh",
            status="failed",
            request={},
            created_at=now,
            updated_at=now,
            status_message="failed",
        )
    )
    child = AgentTaskRecord(
        operation_id="child",
        project_id="project",
        kind="refresh",
        status="queued",
        request={},
        created_at=now,
        updated_at=now,
        status_message="queued",
        attempt=2,
        parent_operation_id=parent.operation_id,
    )

    store.create_agent_task(child, continuation_cause="retry")

    assert store.agent_task_admission_intent(child.operation_id) == {
        "kind": "refresh",
        "attempt": 2,
        "parent_operation_id": parent.operation_id,
        "continuation_cause": "retry",
        "admission_committed": True,
    }
    assert store.agent_task_continuation_cause(child.operation_id) == "retry"
    assert store.agent_task_dispatch_was_proven_not_started(child.operation_id) is True
    receipts = store.agent_task_receipts(child.operation_id)
    assert [receipt.category for receipt in receipts] == ["operation_admitted"]


def test_agent_task_admission_rolls_back_when_its_intent_cannot_persist(
    tmp_path,
    monkeypatch,
) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    now = store.now()

    def fail_receipt(*_args, **_kwargs):
        raise RuntimeError("simulated admission receipt failure")

    monkeypatch.setattr(store, "_insert_agent_task_receipt", fail_receipt)
    with pytest.raises(RuntimeError, match="simulated admission receipt failure"):
        store.create_agent_task(
            AgentTaskRecord(
                operation_id="operation",
                project_id="project",
                kind="refresh",
                status="queued",
                request={},
                created_at=now,
                updated_at=now,
                status_message="queued",
            )
        )

    assert store.agent_task("operation") is None


def test_malformed_agent_task_admission_intent_fails_closed(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    now = store.now()
    store.create_agent_task(
        AgentTaskRecord(
            operation_id="operation",
            project_id="project",
            kind="refresh",
            status="queued",
            request={},
            created_at=now,
            updated_at=now,
            status_message="queued",
        )
    )
    with store.connection() as connection:
        connection.execute(
            """
            UPDATE graph_run_receipts SET payload_json = ?
            WHERE operation_id = ? AND category = 'operation_admitted'
            """,
            ('{"kind":"refresh","attempt":1,"continuation_cause":"fresh"}', "operation"),
        )

    with pytest.raises(ValueError, match="admission intent is malformed"):
        store.agent_task_admission_intent("operation")
    assert store.agent_task_dispatch_was_proven_not_started("operation") is False


def test_inconsistent_or_duplicate_agent_task_admission_intent_fails_closed(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    now = store.now()
    store.create_agent_task(
        AgentTaskRecord(
            operation_id="operation",
            project_id="project",
            kind="refresh",
            status="queued",
            request={},
            created_at=now,
            updated_at=now,
            status_message="queued",
        )
    )
    inconsistent = {
        "kind": "seed",
        "attempt": 1,
        "parent_operation_id": None,
        "continuation_cause": "fresh",
        "admission_committed": True,
    }
    with store.connection() as connection:
        connection.execute(
            """
            UPDATE graph_run_receipts SET payload_json = ?
            WHERE operation_id = ? AND category = 'operation_admitted'
            """,
            (json.dumps(inconsistent), "operation"),
        )

    with pytest.raises(ValueError, match="does not match its task"):
        store.agent_task_admission_intent("operation")
    with pytest.raises(ValueError, match="does not match its task"):
        store.agent_task_continuation_cause("operation")
    assert store.agent_task_dispatch_was_proven_not_started("operation") is False

    with store.connection() as connection:
        store._insert_agent_task_receipt(
            connection,
            "operation",
            "operation_admitted",
            json.dumps(inconsistent),
            tier="summary",
            created_at=now,
        )
    with pytest.raises(ValueError, match="multiple admission intents"):
        store.agent_task_admission_intent("operation")
    assert store.agent_task_dispatch_was_proven_not_started("operation") is False


def test_legacy_committed_dispatch_intent_remains_readable(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    now = store.now()
    store.create_agent_task(
        AgentTaskRecord(
            operation_id="operation",
            project_id="project",
            kind="refresh",
            status="queued",
            request={},
            created_at=now,
            updated_at=now,
            status_message="queued",
        )
    )
    with store.connection() as connection:
        connection.execute(
            """
            DELETE FROM graph_run_receipts
            WHERE operation_id = ? AND category = 'operation_admitted'
            """,
            ("operation",),
        )
    legacy_intent = {
        "kind": "refresh",
        "attempt": 1,
        "has_parent": False,
        "continuation_cause": "fresh",
        "resumed": False,
        "admission_committed": True,
    }
    store.record_agent_task_receipt(
        "operation",
        "operation_created",
        legacy_intent,
    )

    assert store.agent_task_admission_intent("operation") == legacy_intent
    assert store.agent_task_dispatch_was_proven_not_started("operation") is False

    store.record_agent_task_receipt(
        "operation",
        "operation_dispatch_attempt",
        {"dispatch_attempt_id": "legacy-failed-attempt"},
        tier="diagnostic",
    )
    store.record_agent_task_receipt(
        "operation",
        "operation_dispatch_failed_before_start",
        {"dispatch_attempt_id": "legacy-failed-attempt"},
        tier="diagnostic",
    )
    assert store.agent_task_dispatch_was_proven_not_started("operation") is True

    store.record_agent_task_receipt(
        "operation",
        "operation_dispatch_attempt",
        {"dispatch_attempt_id": "legacy-unknown-attempt"},
        tier="diagnostic",
    )
    assert store.agent_task_dispatch_was_proven_not_started("operation") is False

    store.record_agent_task_receipt(
        "operation",
        "operation_dispatch_attempt",
        {"dispatch_attempt_id": "legacy-started-attempt"},
        tier="diagnostic",
    )
    store.record_agent_task_receipt(
        "operation",
        "operation_dispatch_started",
        {"dispatch_attempt_id": "legacy-started-attempt"},
        tier="diagnostic",
    )
    assert store.agent_task_dispatch_was_proven_not_started("operation") is False

    store.record_agent_task_receipt(
        "operation",
        "operation_created",
        legacy_intent,
    )
    with pytest.raises(ValueError, match="multiple legacy admission intents"):
        store.agent_task_admission_intent("operation")
    assert store.agent_task_dispatch_was_proven_not_started("operation") is False


def test_agent_task_events_and_tiered_receipts_are_bounded_per_operation(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    now = store.now()
    store.create_agent_task(
        AgentTaskRecord(
            operation_id="operation",
            project_id="project",
            kind="refresh",
            status="failed",
            request={},
            created_at=now,
            updated_at=now,
            status_message="failed",
        )
    )

    for index in range(225):
        store.record_agent_task_event("operation", f"event {index}")
    for tier, count in (("summary", 70), ("diagnostic", 40), ("trace", 20)):
        for index in range(count):
            store.record_agent_task_receipt(
                "operation",
                f"{tier}-{index}",
                {"index": index},
                tier=tier,
            )

    events = store.agent_task_events("operation", limit=500)
    receipts = store.agent_task_receipts("operation")
    assert len(events) == 200
    assert events[0].message == "event 25"
    assert sum(receipt.tier == "summary" for receipt in receipts) == 64
    assert sum(receipt.tier == "diagnostic" for receipt in receipts) == 32
    assert sum(receipt.tier == "trace" for receipt in receipts) == 16
    assert next(receipt for receipt in receipts if receipt.tier == "summary").category == (
        "operation_admitted"
    )
    assert next(
        receipt
        for receipt in receipts
        if receipt.tier == "summary" and receipt.category != "operation_admitted"
    ).category == ("summary-7")
    assert store.agent_task_admission_intent("operation") is not None


def test_dispatch_proof_receipts_survive_diagnostic_retention(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    now = store.now()

    def queued(operation_id: str) -> None:
        store.create_agent_task(
            AgentTaskRecord(
                operation_id=operation_id,
                project_id="project",
                kind="refresh",
                status="queued",
                request={},
                created_at=now,
                updated_at=now,
                status_message="queued",
            )
        )

    queued("started")
    store.record_agent_task_receipt(
        "started",
        "operation_dispatch_attempt",
        {"dispatch_attempt_id": "started-attempt"},
        tier="diagnostic",
    )
    store.record_agent_task_receipt(
        "started",
        "operation_dispatch_started",
        {"dispatch_attempt_id": "started-attempt"},
        tier="diagnostic",
    )
    queued("failed-before-start")
    store.record_agent_task_receipt(
        "failed-before-start",
        "operation_dispatch_attempt",
        {"dispatch_attempt_id": "failed-attempt"},
        tier="diagnostic",
    )
    store.record_agent_task_receipt(
        "failed-before-start",
        "operation_dispatch_failed_before_start",
        {"dispatch_attempt_id": "failed-attempt"},
        tier="diagnostic",
    )
    for operation_id in ("started", "failed-before-start"):
        for index in range(AGENT_TASK_RECEIPT_RETENTION_COUNTS["diagnostic"] + 5):
            store.record_agent_task_receipt(
                operation_id,
                f"diagnostic-{index}",
                {"index": index},
                tier="diagnostic",
            )

    started_categories = {receipt.category for receipt in store.agent_task_receipts("started")}
    assert {
        "operation_dispatch_attempt",
        "operation_dispatch_started",
    } <= started_categories
    assert store.agent_task_dispatch_was_proven_not_started("started") is False
    failed_categories = {
        receipt.category for receipt in store.agent_task_receipts("failed-before-start")
    }
    assert {
        "operation_dispatch_attempt",
        "operation_dispatch_failed_before_start",
    } <= failed_categories
    assert store.agent_task_dispatch_was_proven_not_started("failed-before-start") is True


@pytest.mark.parametrize(
    ("attempt_payload", "outcome_payload"),
    [
        ('{"dispatch_attempt_id":null}', '{"dispatch_attempt_id":null}'),
        ("[]", '{"dispatch_attempt_id":"attempt"}'),
        ('{"dispatch_attempt_id":"attempt"}', '{"dispatch_attempt_id":""}'),
        ("not-json", '{"dispatch_attempt_id":"attempt"}'),
    ],
)
def test_malformed_dispatch_proof_receipts_fail_closed(
    tmp_path,
    attempt_payload: str,
    outcome_payload: str,
) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    now = store.now()
    store.create_agent_task(
        AgentTaskRecord(
            operation_id="operation",
            project_id="project",
            kind="refresh",
            status="queued",
            request={},
            created_at=now,
            updated_at=now,
            status_message="queued",
        )
    )
    store.record_agent_task_receipt(
        "operation",
        "operation_dispatch_attempt",
        {"dispatch_attempt_id": "attempt"},
        tier="diagnostic",
    )
    store.record_agent_task_receipt(
        "operation",
        "operation_dispatch_failed_before_start",
        {"dispatch_attempt_id": "attempt"},
        tier="diagnostic",
    )
    with store.connection() as connection:
        connection.execute(
            """
            UPDATE graph_run_receipts SET payload_json = ?
            WHERE operation_id = ? AND category = 'operation_dispatch_attempt'
            """,
            (attempt_payload, "operation"),
        )
        connection.execute(
            """
            UPDATE graph_run_receipts SET payload_json = ?
            WHERE operation_id = ? AND category = 'operation_dispatch_failed_before_start'
            """,
            (outcome_payload, "operation"),
        )

    assert store.agent_task_dispatch_was_proven_not_started("operation") is False


def test_oversized_agent_task_receipt_omits_values_but_keeps_safe_metadata(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    now = store.now()
    store.create_agent_task(
        AgentTaskRecord(
            operation_id="operation",
            project_id="project",
            kind="seed",
            status="failed",
            request={},
            created_at=now,
            updated_at=now,
            status_message="failed",
        )
    )
    raw_evidence = "sensitive transcript fragment" * 1000

    store.record_agent_task_receipt(
        "operation",
        "context_assembled",
        {"raw_evidence": raw_evidence, "session_count": 12},
        tier="diagnostic",
    )

    receipt = next(
        item
        for item in store.agent_task_receipts("operation")
        if item.category == "context_assembled"
    )
    assert receipt.payload == {
        "omitted": True,
        "reason": "payload_exceeded_limit",
        "byte_length": len(
            json.dumps(
                {"raw_evidence": raw_evidence, "session_count": 12},
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            ).encode("utf-8")
        ),
        "keys": ["raw_evidence", "session_count"],
    }
    assert raw_evidence not in json.dumps(receipt.payload)


def test_agent_task_contract_content_is_durable_beyond_receipt_limit(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    now = store.now()
    store.create_agent_task(
        AgentTaskRecord(
            operation_id="operation",
            project_id="project",
            kind="seed",
            status="failed",
            request={},
            created_at=now,
            updated_at=now,
            status_message="failed",
        )
    )
    content = "immutable pointer contract\n" + "x" * 20_000
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()

    store.record_agent_task_contract("operation", "base", content, digest)
    store.record_agent_task_contract("operation", "base", content, digest)

    assert store.agent_task_contract("operation", "base") == content
    contracts = store.agent_task_contracts("operation")
    assert len(contracts) == 1
    assert contracts[0].operation_id == "operation"
    assert contracts[0].role == "base"
    assert contracts[0].sha256 == digest
    assert contracts[0].content == content
    with pytest.raises(ValueError, match="immutable"):
        store.record_agent_task_contract("operation", "base", content + "changed", digest)


def test_agent_task_result_messages_are_bounded(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    now = store.now()
    store.create_agent_task(
        AgentTaskRecord(
            operation_id="operation",
            project_id="project",
            kind="paper_coach",
            status="running",
            request={"message": "Review this."},
            created_at=now,
            updated_at=now,
            status_message="running",
        )
    )

    store.complete_agent_task(
        "operation",
        applied_revision=None,
        result={"messages": ["x" * 20_000 for _ in range(40)]},
    )

    record = store.agent_task("operation")
    assert record is not None
    assert record.status == "succeeded"
    assert record.applied_revision is None
    assert record.result is not None
    messages = record.result["messages"]
    assert isinstance(messages, list)
    assert len(json.dumps(record.result).encode("utf-8")) < 64 * 1024
    assert all(isinstance(message, str) and len(message) <= 16_000 for message in messages)


def test_agent_task_result_keeps_a_bounded_latest_tail_of_graph_updates(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    now = store.now()
    store.create_agent_task(
        AgentTaskRecord(
            operation_id="graph-updates",
            project_id="project",
            kind="node_chat",
            status="running",
            request={"message": "Apply the updates."},
            created_at=now,
            updated_at=now,
            status_message="running",
        )
    )
    updates = [
        {
            "status": "applied",
            "applied_revision": revision,
            "change_summary": [f"update-{revision}-" + "🔬" * 1600 for _ in range(2)],
            "proposal_ids": [f"proposal-{revision}-" + "p" * 400 for _ in range(8)],
            "validation_messages": ["🔬" * 1600 for _ in range(2)],
            "correction_rounds": 0,
            "repairable": False,
        }
        for revision in range(100)
    ]
    latest = {
        "status": "applied",
        "applied_revision": 101,
        "change_summary": ["Applied the final end-of-turn Patch."],
        "proposal_ids": [],
        "validation_messages": [],
        "correction_rounds": 0,
        "repairable": False,
    }

    store.complete_agent_task(
        "graph-updates",
        applied_revision=101,
        result={"messages": [], "graph_update": latest, "graph_updates": updates},
    )

    record = store.agent_task("graph-updates")
    assert record is not None
    assert record.applied_revision == 101
    assert record.result is not None
    assert record.result["graph_update"] == latest
    retained = record.result["graph_updates"]
    assert isinstance(retained, list)
    revisions = [update["applied_revision"] for update in retained]
    assert revisions == list(range(100 - len(revisions), 100))
    assert 0 < len(revisions) <= 32
    assert (
        len(json.dumps(record.result, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        < 64 * 1024
    )


def test_work_graph_repair_admission_rolls_back_claim_and_child_together(
    tmp_path, monkeypatch
) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    now = store.now()
    chat_id = str(uuid.uuid4())
    request = RunRequest(
        chat_scope="project",
        chat_id=chat_id,
        message="Complete operational work.",
        mode="work",
        patch_kind="work",
    )
    authority = resolve_dispatch_authority("project_chat", request)
    assert authority is not None
    authorizer = AuthorizedHuman(
        space_id=store.space_id,
        user_id=str(uuid.uuid4()),
        display_name="Test researcher",
    )
    store.create_agent_task(
        AgentTaskRecord(
            operation_id="work-parent",
            project_id="project",
            kind="project_chat",
            status="succeeded",
            request=request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="Graph update rejected.",
            native_session_id="repair-session",
            stage_root=str(tmp_path / "repair-stage"),
            result={
                "messages": ["Operational work completed."],
                "graph_update": {
                    "status": "rejected",
                    "validation_messages": ["Repair the Patch."],
                    "repairable": True,
                },
            },
            authorized_by=authorizer,
            dispatch_authority=authority,
        )
    )
    repair_request = request.model_copy(update={"message": None, "session_id": "repair-session"})

    def child(operation_id: str) -> AgentTaskRecord:
        return AgentTaskRecord(
            operation_id=operation_id,
            project_id="project",
            kind="project_chat",
            status="queued",
            request=repair_request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="Waiting to repair the graph update.",
            attempt=2,
            parent_operation_id="work-parent",
            native_session_id="repair-session",
            stage_root=str(tmp_path / "repair-stage"),
            authorized_by=authorizer,
            dispatch_authority=authority,
        )

    original_insert = store._insert_agent_task

    def fail_after_child_insert(connection, record, *, continuation_cause) -> None:
        original_insert(connection, record, continuation_cause=continuation_cause)
        raise RuntimeError("simulated graph repair insert failure")

    monkeypatch.setattr(store, "_insert_agent_task", fail_after_child_insert)
    with pytest.raises(RuntimeError, match="simulated graph repair insert failure"):
        store.create_agent_task_graph_repair("work-parent", child("work-repair-failed"))

    parent = store.agent_task("work-parent")
    assert parent is not None and parent.result is not None
    assert parent.result["graph_update"]["repairable"] is True
    assert store.agent_task("work-repair-failed") is None

    monkeypatch.setattr(store, "_insert_agent_task", original_insert)
    admitted = store.create_agent_task_graph_repair("work-parent", child("work-repair"))

    assert admitted.operation_id == "work-repair"
    parent = store.agent_task("work-parent")
    assert parent is not None and parent.result is not None
    assert parent.result["graph_update"]["repairable"] is False
    children = [
        task for task in store.agent_tasks("project") if task.parent_operation_id == "work-parent"
    ]
    assert [task.operation_id for task in children] == ["work-repair"]


def test_resumable_paused_chat_query_is_exact_and_child_attempt_resolves_it(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    now = store.now()
    chat_id = "1f16a63a-06c8-42b6-a856-c9329e2e9007"
    request = RunRequest(
        chat_scope="node",
        chat_id=chat_id,
        node_id="rq/one",
        mode="discuss",
    )
    dispatch_authority = resolve_dispatch_authority("node_chat", request)
    assert dispatch_authority is not None
    store.create_agent_task(
        AgentTaskRecord(
            operation_id="paused-chat",
            project_id="project",
            kind="node_chat",
            status="paused",
            request=request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="paused",
            native_session_id="native-session",
            stage_host="",
            stage_root="/tmp/paused-chat",
            dispatch_authority=dispatch_authority,
        )
    )

    assert store.has_resumable_paused_chat_task("project", "node_chat", chat_id)
    assert not store.has_resumable_paused_chat_task("other", "node_chat", chat_id)
    assert not store.has_resumable_paused_chat_task("project", "project_chat", chat_id)
    assert not store.has_resumable_paused_chat_task(
        "project", "node_chat", "701929b2-50f5-41e6-9614-41e4a82b9e34"
    )

    store.create_agent_task(
        AgentTaskRecord(
            operation_id="retried-chat",
            project_id="project",
            kind="node_chat",
            status="succeeded",
            request=request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="complete",
            parent_operation_id="paused-chat",
            dispatch_authority=dispatch_authority,
        )
    )

    assert not store.has_resumable_paused_chat_task("project", "node_chat", chat_id)


def test_agent_task_result_retains_only_valid_bounded_artifact_descriptors(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    now = store.now()
    store.create_agent_task(
        AgentTaskRecord(
            operation_id="artifact-operation",
            project_id="project",
            kind="project_chat",
            status="running",
            request={},
            created_at=now,
            updated_at=now,
            status_message="running",
        )
    )
    descriptor = AgentArtifactDescriptor(
        artifact_id="a" * 24,
        name="preview.html",
        media_type="text/html",
    )

    store.complete_agent_task(
        "artifact-operation",
        applied_revision=None,
        result={
            "messages": ["answer"],
            "artifacts": [
                descriptor.model_dump(mode="json"),
                {"artifact_id": "bad", "name": "raw", "media_type": "application/octet-stream"},
            ],
        },
    )

    record = store.agent_task("artifact-operation")
    assert record is not None and record.result is not None
    assert record.result == {
        "messages": ["answer"],
        "artifacts": [descriptor.model_dump(mode="json")],
    }
