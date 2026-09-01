from __future__ import annotations

import os
import tempfile
import threading
import uuid
from io import BytesIO, StringIO
from pathlib import Path

from fastapi.testclient import TestClient

from rcp.api import create_app
from rcp.background import BackgroundAgentTasks
from rcp.core.models import AuthorizedHuman
from rcp.server_ops.cli import CallerIdentity, ServerEventEmitter
from rcp.server_ops.members import MemberRemovalCoordinator, prepare_member_remove_command
from rcp.server_ops.models import ServerCommandRequest, ServerStepEvent
from rcp.server_runtime import ServerMetadata
from rcp.storage import AgentTaskRecord, AppStore, EpisodeRecord, ProjectRecord


class _CoordinatorControl:
    def __init__(self, coordinator: MemberRemovalCoordinator) -> None:
        self.coordinator = coordinator
        self.advances = 0

    def member_removal_plan(self, member_id: str):
        return self.coordinator.plan(member_id)

    def advance_member_removal(self, member_id: str, *, boundary_sha256: str):
        self.advances += 1
        return self.coordinator.advance(member_id, boundary_sha256=boundary_sha256)


class _RecordingBackground:
    def __init__(self, store: AppStore) -> None:
        self.store = store
        self.paused: list[str] = []

    def request_member_removal_pause(self, operation_id: str):
        self.paused.append(operation_id)
        return self.store.request_agent_task_pause(
            operation_id,
            requested_by="member_removal",
        )


def _team(tmp_path):
    store, bootstrap = AppStore.initialize_team_space(tmp_path / "rcp.sqlite3", "Team Lab")
    alice, _alice_token = store.enroll_team_member(bootstrap, "Alice")
    _invitation, code = store.create_team_invitation(alice.user_id)
    bob, _bob_token = store.enroll_team_member(code, "Bob")
    metadata = ServerMetadata.create(
        tmp_path,
        host="team.example",
        port=8421,
        owner_kind="cli",
        control_socket=tmp_path / "control.sock",
    )
    background = _RecordingBackground(store)
    coordinator = MemberRemovalCoordinator(store, background, metadata)  # type: ignore[arg-type]
    return store, alice, bob, metadata, background, coordinator


def _execute(prepared):
    emitter = ServerEventEmitter(
        prepared.plan,
        machine_readable=True,
        stream=StringIO(),
    )
    prepared.execute(emitter, BytesIO())
    final = emitter.events[-1]
    assert isinstance(final, ServerStepEvent)
    return final.step


def _field(step, name: str) -> object:
    """Read one rendered console field by name."""
    for field in step.fields:
        if field.name == name:
            return field.value
    raise AssertionError(f"{name} was not rendered; got {[f.name for f in step.fields]}")


def test_cli_previews_exact_boundary_before_confirmation(tmp_path) -> None:
    store, _alice, bob, _metadata, _background, coordinator = _team(tmp_path)
    control = _CoordinatorControl(coordinator)
    prepared = prepare_member_remove_command(
        ServerCommandRequest(command="server member remove", member_id=bob.user_id),
        CallerIdentity(uid=501, username="rcp", host="team.example"),
        control_factory=lambda _layout: control,
    )

    final = _execute(prepared)

    assert final.state == "operator_action_needed"
    assert final.resume_argv[:5] == (
        "rcp",
        "server",
        "member",
        "remove",
        bob.user_id,
    )
    assert final.resume_argv[-2] == "--confirm-boundary"
    assert final.resume_argv[-1] == coordinator.plan(bob.user_id).snapshot.boundary_sha256
    assert control.advances == 0
    assert store.space_user(bob.user_id).removal_started_at is None
    # The console names the state it is about to change, so each state is pinned.
    assert _field(final, "removal_state") == "active"


def test_confirmed_cli_removes_member_and_preserves_history(tmp_path) -> None:
    store, _alice, bob, _metadata, _background, coordinator = _team(tmp_path)
    control = _CoordinatorControl(coordinator)
    boundary = coordinator.plan(bob.user_id).snapshot.boundary_sha256
    prepared = prepare_member_remove_command(
        ServerCommandRequest(
            command="server member remove",
            member_id=bob.user_id,
            member_confirmed_boundary=boundary,
        ),
        CallerIdentity(uid=501, username="rcp", host="team.example"),
        control_factory=lambda _layout: control,
    )

    final = _execute(prepared)

    assert final.state == "succeeded"
    assert control.advances == 1
    tombstone = store.space_user(bob.user_id)
    assert tombstone is not None
    assert tombstone.display_name == "Bob"
    assert tombstone.removal_started_at is not None
    assert tombstone.removed_at is not None


def test_preview_reports_the_state_the_member_is_actually_in(tmp_path) -> None:
    """An operator re-running the command must see the fence already happened.

    The preview renders one removal_state, and it is the only place the operator
    learns whether access is still open.
    """

    store, _alice, bob, _metadata, _background, coordinator = _team(tmp_path)
    control = _CoordinatorControl(coordinator)
    preview = store.member_removal_preview(bob.user_id)
    store.begin_member_removal(bob.user_id, expected_boundary_sha256=preview.boundary_sha256)

    prepared = prepare_member_remove_command(
        ServerCommandRequest(command="server member remove", member_id=bob.user_id),
        CallerIdentity(uid=501, username="rcp", host="team.example"),
        control_factory=lambda _layout: control,
    )
    emitter = ServerEventEmitter(prepared.plan, machine_readable=True, stream=StringIO())
    prepared.execute(emitter, BytesIO())
    steps = [event.step for event in emitter.events if isinstance(event, ServerStepEvent)]

    # The operator sees the fence that a crash already applied, then its completion.
    assert _field(steps[0], "removal_state") == "access_fenced"
    assert _field(steps[0], "member_name") == "Bob"
    assert _field(steps[-1], "removal_state") == "removed"


def test_reentry_reconciles_a_crash_after_the_access_fence(tmp_path) -> None:
    store, _alice, bob, _metadata, _background, coordinator = _team(tmp_path)
    preview = store.member_removal_preview(bob.user_id)
    store.begin_member_removal(
        bob.user_id,
        expected_boundary_sha256=preview.boundary_sha256,
    )

    result = coordinator.reconcile_pending()

    assert len(result) == 1
    assert result[0].step.state == "succeeded"
    assert store.space_user(bob.user_id).removed_at is not None


def test_installed_service_startup_reconciles_a_fenced_member(tmp_path) -> None:
    store, _alice, bob, _metadata, _background, _coordinator = _team(tmp_path)
    preview = store.member_removal_preview(bob.user_id)
    store.begin_member_removal(
        bob.user_id,
        expected_boundary_sha256=preview.boundary_sha256,
    )
    with tempfile.TemporaryDirectory(prefix="rcp-member-", dir="/tmp") as raw_runtime:
        runtime = Path(raw_runtime)
        os.chown(runtime, os.geteuid(), os.getegid())
        runtime.chmod(0o700)
        metadata = ServerMetadata.create(
            tmp_path,
            host="127.0.0.1",
            port=8421,
            owner_kind="cli",
            control_socket=runtime / "control.sock",
        )
        app = create_app(data_dir=tmp_path, instance_metadata=metadata)

        with TestClient(app):
            tombstone = app.state.services.store.space_user(bob.user_id)
            assert tombstone is not None
            assert tombstone.removed_at is not None


def test_live_task_settles_before_member_tombstone_completes(tmp_path) -> None:
    store, alice, bob, _metadata, background, coordinator = _team(tmp_path)
    project_id = str(uuid.uuid4())
    now = store.now()
    store.upsert_project(
        ProjectRecord(
            project_id=project_id,
            home_space_id=store.space_id,
            locator=f"/tmp/{project_id}/research.yaml",
            name="Shared",
            state_location=f"/tmp/{project_id}/.research",
            state_remote=False,
            added_at=now,
        )
    )
    store.seat_project_member(project_id, alice.user_id)
    store.seat_project_member(project_id, bob.user_id)
    operation_id = str(uuid.uuid4())
    store.create_agent_task(
        AgentTaskRecord(
            operation_id=operation_id,
            project_id=project_id,
            kind="refresh",
            status="running",
            request={},
            created_at=now,
            updated_at=now,
            status_message="Running",
            authorized_by=AuthorizedHuman(
                space_id=store.space_id,
                user_id=bob.user_id,
                display_name="Bob",
            ),
        )
    )
    boundary = coordinator.plan(bob.user_id).snapshot.boundary_sha256

    first = coordinator.advance(bob.user_id, boundary_sha256=boundary)

    assert first.step.state == "operator_action_needed"
    assert background.paused == [operation_id]
    assert store.agent_task(operation_id).status == "pausing"
    assert store.space_user(bob.user_id).removal_started_at is not None
    assert store.space_user(bob.user_id).removed_at is None

    store.complete_agent_task(operation_id, applied_revision=None, result={})
    second = coordinator.reconcile_pending()
    assert second[0].step.state == "succeeded"
    assert store.space_user(bob.user_id).removed_at is not None


def test_member_removal_presses_the_existing_episode_stop_fence(tmp_path) -> None:
    store, alice, bob, _metadata, _background, coordinator = _team(tmp_path)
    project_id = str(uuid.uuid4())
    episode_id = str(uuid.uuid4())
    now = store.now()
    store.upsert_project(
        ProjectRecord(
            project_id=project_id,
            home_space_id=store.space_id,
            locator=f"/tmp/{project_id}/research.yaml",
            name="Shared",
            state_location=f"/tmp/{project_id}/.research",
            state_remote=False,
            added_at=now,
        )
    )
    store.seat_project_member(project_id, alice.user_id)
    store.seat_project_member(project_id, bob.user_id)
    store.create_episode(
        EpisodeRecord(
            episode_id=episode_id,
            project_id=project_id,
            mode="experiment_loop",
            control_node_id="exp/removal-fence",
            status="queued",
            invocation_ceiling=1,
            authorized_by=AuthorizedHuman(
                space_id=store.space_id,
                user_id=bob.user_id,
                display_name="Bob",
            ),
            created_at=now,
            updated_at=now,
        )
    )
    boundary = coordinator.plan(bob.user_id).snapshot.boundary_sha256

    first = coordinator.advance(bob.user_id, boundary_sha256=boundary)

    assert first.step.state == "operator_action_needed"
    stopping = store.episode(episode_id)
    assert stopping is not None
    assert stopping.status == "stopping"
    assert stopping.stop_requested_at is not None
    assert store.space_user(bob.user_id).removed_at is None

    store.mark_episode_stop_skipped(
        episode_id,
        diagnostic="Authorizing member removed",
    )
    second = coordinator.reconcile_pending()
    assert second[0].step.state == "succeeded"
    assert store.space_user(bob.user_id).removed_at is not None


def test_background_member_pause_does_not_signal_a_live_provider_control(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    owner = store.rename_space_user(store.local_owner.user_id, "Owner")
    project_id = str(uuid.uuid4())
    now = store.now()
    store.upsert_project(
        ProjectRecord(
            project_id=project_id,
            home_space_id=store.space_id,
            locator=f"/tmp/{project_id}/research.yaml",
            name="Project",
            state_location=f"/tmp/{project_id}/.research",
            state_remote=False,
            added_at=now,
        )
    )
    operation_id = str(uuid.uuid4())
    store.create_agent_task(
        AgentTaskRecord(
            operation_id=operation_id,
            project_id=project_id,
            kind="refresh",
            status="running",
            request={},
            created_at=now,
            updated_at=now,
            status_message="Running",
            authorized_by=AuthorizedHuman(
                space_id=store.space_id,
                user_id=owner.user_id,
                display_name="Owner",
            ),
        )
    )

    async def unused_stream(*_args, **_kwargs):
        if False:
            yield ""

    background = BackgroundAgentTasks(store, unused_stream)
    background._workers[operation_id] = threading.current_thread()

    paused = background.request_member_removal_pause(operation_id)

    assert paused.status == "pausing"
    assert operation_id not in background._controls
