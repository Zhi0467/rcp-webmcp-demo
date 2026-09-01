from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from rcp.api.app import create_app
from rcp.core.models import AuthorizedHuman, Patch
from rcp.history import HistoryManager
from rcp.service import ChatMessage, ChatTranscript, RunRequest
from rcp.storage import (
    AgentTaskRecord,
    AppStore,
    WatcherContinuation,
    WatcherRecord,
)

from .helpers import seed_patch


def _assert_name_required(response) -> None:
    assert response.status_code == 428, response.text
    assert response.json()["detail"] == {
        "code": "identity_name_required",
        "message": (
            "Choose an RCP display name before this action. The name will be copied into "
            "permanent project history as a snapshot."
        ),
    }


def _experiment_patch() -> Patch:
    return Patch(
        kind="refresh",
        author="agent",
        summary="Added an experiment for identity-admission testing.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": "exp/identity-gate",
                        "type": "experiment",
                        "title": "Identity gate",
                        "objective": "Exercise patch-capable admission.",
                        "completion_criteria": ["The fixture exits."],
                        "invocation_ceiling": 2,
                    }
                ],
            }
        ],
    )


def _agent_task(
    store: AppStore,
    *,
    operation_id: str,
    project_id: str,
    authorized_by: AuthorizedHuman | None,
    status: str = "succeeded",
    request: dict[str, object] | None = None,
) -> AgentTaskRecord:
    now = store.now()
    return AgentTaskRecord(
        operation_id=operation_id,
        project_id=project_id,
        kind="project_chat",
        status=status,
        request=request or {},
        created_at=now,
        updated_at=now,
        status_message="Test task.",
        authorized_by=authorized_by,
    )


def _watcher(
    *,
    watcher_id: str,
    project_id: str,
    origin_operation_id: str,
) -> WatcherRecord:
    return WatcherRecord(
        watcher_id=watcher_id,
        project_id=project_id,
        origin_operation_id=origin_operation_id,
        origin_task_kind="project_chat",
        chat_id="watcher-chat",
        check_command="true",
        log_path="watcher.log",
        cwd=".",
        continuation=WatcherContinuation(
            provider="codex",
            run_on="laptop",
            run_truth_scope=["repo-a"],
        ),
        status="completed",
        created_at="2026-08-11T00:00:00Z",
    )


def test_health_and_personal_identity_get_patch_are_durable(tmp_path) -> None:
    data_dir = tmp_path / "personal"
    app = create_app(data_dir=data_dir)
    client = TestClient(app)

    health = client.get("/api/health")
    original = client.get("/api/identity")

    assert health.status_code == 200
    assert health.json()["space_id"] == app.state.space_id
    assert health.json()["space_kind"] == app.state.space_kind == "personal"
    assert original.status_code == 200
    assert original.json()["space_id"] == app.state.space_id
    assert original.json()["space_kind"] == "personal"
    assert original.json()["user"]["identity_kind"] == "local_owner"
    assert original.json()["user"]["display_name"] is None
    user_id = original.json()["user"]["user_id"]

    for forged in (
        {"display_name": "Owner", "user_id": str(uuid.uuid4())},
        {"display_name": "Owner", "identity_kind": "team_member"},
        {"display_name": "   "},
        {"display_name": "line one\nline two"},
        {"display_name": "x" * 121},
        {"display_name": 42},
    ):
        assert client.patch("/api/identity", json=forged).status_code == 422
    assert client.get("/api/identity").json()["user"]["display_name"] is None

    renamed = client.patch("/api/identity", json={"display_name": "  Researcher  "})
    assert renamed.status_code == 200
    assert renamed.json()["user"]["user_id"] == user_id
    assert renamed.json()["user"]["display_name"] == "Researcher"

    restarted = TestClient(create_app(data_dir=data_dir)).get("/api/identity")
    assert restarted.status_code == 200
    assert restarted.json()["space_id"] == app.state.space_id
    assert restarted.json()["user"]["user_id"] == user_id
    assert restarted.json()["user"]["display_name"] == "Researcher"


def test_team_identity_uses_only_the_trusted_resolver(tmp_path) -> None:
    data_dir = tmp_path / "team"
    store = AppStore(data_dir / "rcp.sqlite3", space_kind="team")
    first = store.preprovision_team_member("Same name")
    second = store.preprovision_team_member("Same name")
    selected = [first.user_id]

    def resolve(_request, _store):
        return selected[0]

    app = create_app(data_dir=data_dir, trusted_principal_resolver=resolve)
    client = TestClient(app)

    assert client.get("/api/health").json()["space_kind"] == "team"
    assert client.get("/api/identity").json()["user"]["user_id"] == first.user_id

    forged = client.patch(
        "/api/identity",
        json={"display_name": "Forged", "user_id": second.user_id},
    )
    assert forged.status_code == 422
    assert AppStore(data_dir / "rcp.sqlite3").space_user(first.user_id) == first
    assert AppStore(data_dir / "rcp.sqlite3").space_user(second.user_id) == second

    renamed = client.patch("/api/identity", json={"display_name": "First renamed"})
    assert renamed.status_code == 200
    assert renamed.json()["user"]["user_id"] == first.user_id
    selected[0] = second.user_id
    assert client.get("/api/identity").json()["user"]["user_id"] == second.user_id

    missing = TestClient(create_app(data_dir=data_dir)).get("/api/identity")
    assert missing.status_code == 401
    assert missing.json()["detail"]["code"] == "team_identity_required"

    invalid_app = create_app(
        data_dir=data_dir,
        trusted_principal_resolver=lambda _request, _store: str(uuid.uuid4()),
    )
    invalid = TestClient(invalid_app).get("/api/identity")
    assert invalid.status_code == 403
    assert invalid.json()["detail"]["code"] == "team_identity_invalid"


def test_unnamed_personal_owner_gates_all_agent_api_admissions(
    manifest,
    tmp_path,
    monkeypatch,
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    client = TestClient(app)
    project_id = app.state.default_project_id
    service = app.state.service
    fixture_history = HistoryManager(service.manifest, service.history.workspace)
    fixture_history.append(seed_patch())
    fixture_history.append(_experiment_patch())
    store = app.state.background_tasks.store
    started_kinds: list[str] = []

    def fake_start(
        project_id,
        kind,
        request,
        *,
        operation_id=None,
        authorized_by=None,
        stage_host=None,
        stage_root=None,
    ):
        assert stage_host is None
        assert stage_root is None
        started_kinds.append(kind)
        now = store.now()
        return AgentTaskRecord(
            operation_id=operation_id or str(uuid.uuid4()),
            project_id=project_id,
            kind=kind,
            status="queued",
            request=request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="Queued.",
            authorized_by=authorized_by,
        )

    monkeypatch.setattr(app.state.background_tasks, "start", fake_start)

    assert client.get(f"/api/projects/{project_id}").status_code == 200
    _assert_name_required(
        client.post(
            f"/api/projects/{project_id}/tasks/project_chat",
            json={
                "chat_id": str(uuid.uuid4()),
                "message": "Explain the project.",
                "mode": "discuss",
            },
        )
    )
    _assert_name_required(
        client.post(
            f"/api/projects/{project_id}/tasks/paper_coach",
            json={"message": "Review the introduction."},
        )
    )
    paper = client.post(f"/api/projects/{project_id}/paper/create")
    assert paper.status_code == 200
    assert started_kinds == []

    current_revision = service.history.state().revision
    _assert_name_required(
        client.post(
            f"/api/projects/{project_id}/sync",
            json={"base_revision": current_revision},
        )
    )
    for kind in ("seed", "refresh"):
        _assert_name_required(client.post(f"/api/projects/{project_id}/tasks/{kind}", json={}))

    def unexpected_claim(*_args, **_kwargs):
        raise AssertionError("attachment claim ran before identity admission")

    monkeypatch.setattr("rcp.api.app.ChatAttachmentStore.claim", unexpected_claim)
    _assert_name_required(
        client.post(
            f"/api/projects/{project_id}/tasks/project_chat",
            json={
                "chat_id": str(uuid.uuid4()),
                "message": "Change the graph.",
                "mode": "work",
                "attachment_set_id": str(uuid.uuid4()),
                "attachment_client_id": str(uuid.uuid4()),
            },
        )
    )
    _assert_name_required(
        client.post(
            f"/api/projects/{project_id}/experiments/exp%2Fidentity-gate/run",
            json={"chat_id": str(uuid.uuid4())},
        )
    )

    now = store.now()
    work_request = RunRequest(
        provider="codex",
        run_on="laptop",
        chat_scope="project",
        chat_id=str(uuid.uuid4()),
        message="Change the graph.",
        mode="work",
    ).model_dump(mode="json")
    operation_ids = {
        "resume": "identity-gate-resume",
        "retry": "identity-gate-retry",
        "repair-graph-update": "identity-gate-repair",
    }
    for operation_id in operation_ids.values():
        store.create_agent_task(
            AgentTaskRecord(
                operation_id=operation_id,
                project_id=project_id,
                kind="project_chat",
                status="paused",
                request=work_request,
                created_at=now,
                updated_at=now,
                status_message="Paused.",
            )
        )
    for action, operation_id in operation_ids.items():
        _assert_name_required(
            client.post(f"/api/projects/{project_id}/tasks/{operation_id}/{action}", json={})
        )

    discuss_record = AgentTaskRecord(
        operation_id="identity-gate-discuss",
        project_id=project_id,
        kind="project_chat",
        status="paused",
        request=RunRequest(
            provider="codex",
            run_on="laptop",
            chat_scope="project",
            chat_id=str(uuid.uuid4()),
            message="Explain the graph.",
            mode="discuss",
        ).model_dump(mode="json"),
        created_at=now,
        updated_at=now,
        status_message="Paused.",
    )
    coach_record = AgentTaskRecord(
        operation_id="identity-gate-coach",
        project_id=project_id,
        kind="paper_coach",
        status="failed",
        request={"message": "Review the introduction."},
        created_at=now,
        updated_at=now,
        status_message="Failed.",
    )
    store.create_agent_task(discuss_record)
    store.create_agent_task(coach_record)
    recovery_authorizers: dict[str, AuthorizedHuman | None] = {}

    def capture_recovery(label, record):
        def recover(_operation_id, *, authorized_by=None, **_kwargs):
            recovery_authorizers[label] = authorized_by
            return record

        return recover

    monkeypatch.setattr(
        app.state.background_tasks,
        "resume",
        capture_recovery("discuss", discuss_record),
    )
    monkeypatch.setattr(
        app.state.background_tasks,
        "retry",
        capture_recovery("coach", coach_record),
    )
    monkeypatch.setattr(
        app.state.background_tasks,
        "repair_graph_update",
        lambda _operation_id, **_kwargs: discuss_record,
    )
    _assert_name_required(
        client.post(f"/api/projects/{project_id}/tasks/{discuss_record.operation_id}/resume")
    )
    _assert_name_required(
        client.post(
            f"/api/projects/{project_id}/tasks/{coach_record.operation_id}/retry",
            json={},
        )
    )
    assert recovery_authorizers == {}
    assert (
        client.post(
            f"/api/projects/{project_id}/tasks/{discuss_record.operation_id}/repair-graph-update"
        ).status_code
        == 202
    )

    paused_record = store.agent_task(operation_ids["resume"])
    assert paused_record is not None
    monkeypatch.setattr(app.state.background_tasks, "pause", lambda _operation_id: paused_record)
    assert (
        client.post(f"/api/projects/{project_id}/tasks/{operation_ids['resume']}/pause").status_code
        == 202
    )
    assert len(store.agent_tasks(project_id)) == len(operation_ids) + 2

    named = client.patch("/api/identity", json={"display_name": "Researcher"})
    assert named.status_code == 200
    resumed_discuss = client.post(
        f"/api/projects/{project_id}/tasks/{discuss_record.operation_id}/resume"
    )
    retried_coach = client.post(
        f"/api/projects/{project_id}/tasks/{coach_record.operation_id}/retry",
        json={},
    )
    assert resumed_discuss.status_code == retried_coach.status_code == 202
    assert recovery_authorizers["discuss"] is not None
    assert recovery_authorizers["coach"] is not None
    assert recovery_authorizers["discuss"].display_name == "Researcher"
    assert recovery_authorizers["coach"].display_name == "Researcher"
    admitted = client.post(f"/api/projects/{project_id}/tasks/seed", json={})
    assert admitted.status_code == 202
    assert started_kinds[-1] == "seed"


def test_team_patch_admission_rejects_missing_or_invalid_principal_before_task_creation(
    manifest,
    tmp_path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "team"
    store = AppStore(data_dir / "rcp.sqlite3", space_kind="team")
    member = store.preprovision_team_member("Team member")

    missing_app = create_app(str(manifest.path), data_dir=data_dir)
    missing_client = TestClient(missing_app)
    project_id = missing_app.state.default_project_id
    missing = missing_client.post(f"/api/projects/{project_id}/tasks/seed", json={})
    assert missing.status_code == 401
    assert missing.json()["detail"]["code"] == "team_identity_required"
    assert missing_app.state.background_tasks.store.agent_tasks(project_id) == []

    invalid_app = create_app(
        str(manifest.path),
        data_dir=data_dir,
        trusted_principal_resolver=lambda _request, _store: str(uuid.uuid4()),
    )
    invalid_client = TestClient(invalid_app)
    invalid = invalid_client.post(f"/api/projects/{project_id}/tasks/seed", json={})
    assert invalid.status_code == 403
    assert invalid.json()["detail"]["code"] == "team_identity_invalid"
    assert invalid_app.state.background_tasks.store.agent_tasks(project_id) == []

    valid_app = create_app(
        str(manifest.path),
        data_dir=data_dir,
        trusted_principal_resolver=lambda _request, _store: member,
    )
    valid_store = valid_app.state.background_tasks.store

    def fake_start(
        project_id,
        kind,
        request,
        *,
        operation_id=None,
        authorized_by=None,
        stage_host=None,
        stage_root=None,
    ):
        assert stage_host is None
        assert stage_root is None
        now = valid_store.now()
        return AgentTaskRecord(
            operation_id=operation_id or str(uuid.uuid4()),
            project_id=project_id,
            kind=kind,
            status="queued",
            request=request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="Queued.",
            authorized_by=authorized_by,
        )

    monkeypatch.setattr(valid_app.state.background_tasks, "start", fake_start)
    admitted = TestClient(valid_app).post(f"/api/projects/{project_id}/tasks/seed", json={})
    assert admitted.status_code == 202


def test_personal_sync_and_task_records_keep_immutable_identity_snapshots(
    manifest,
    tmp_path,
    monkeypatch,
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    client = TestClient(app)
    project_id = app.state.default_project_id
    service = app.state.service
    fixture_history = HistoryManager(service.manifest, service.history.workspace)
    fixture_history.append(seed_patch())
    assert client.patch("/api/identity", json={"display_name": "First name"}).status_code == 200

    node = service.history.state().nodes["rq/learning-after-shift"]
    synced = client.post(
        f"/api/projects/{project_id}/sync",
        json={
            "base_revision": service.history.state().revision,
            "nodes": [
                {
                    "node_id": node.id,
                    "base_updated_rev": node.updated_rev,
                    "changes": {"title": "Human-synced question"},
                }
            ],
        },
    )
    assert synced.status_code == 200, synced.text
    sync_authorizer = service.history.load_patches()[-1].authorized_by
    assert sync_authorizer is not None
    assert sync_authorizer.display_name == "First name"

    monkeypatch.setattr(
        app.state.background_tasks,
        "_spawn_record",
        lambda record, _request, **_kwargs: record,
    )
    seed = client.post(f"/api/projects/{project_id}/tasks/seed", json={})
    assert seed.status_code == 202, seed.text
    seed_record = app.state.background_tasks.store.agent_task(seed.json()["operation_id"])
    assert seed_record is not None
    assert seed_record.authorized_by == sync_authorizer
    app.state.background_tasks.store.complete_agent_task(
        seed_record.operation_id,
        applied_revision=None,
        result={},
    )

    assert client.patch("/api/identity", json={"display_name": "Later name"}).status_code == 200
    refresh = client.post(f"/api/projects/{project_id}/tasks/refresh", json={})
    assert refresh.status_code == 202, refresh.text
    refresh_record = app.state.background_tasks.store.agent_task(refresh.json()["operation_id"])
    assert refresh_record is not None
    assert refresh_record.authorized_by is not None
    assert refresh_record.authorized_by.display_name == "Later name"
    assert seed_record.authorized_by.display_name == "First name"
    app.state.background_tasks.store.complete_agent_task(
        refresh_record.operation_id,
        applied_revision=None,
        result={},
    )

    discuss = client.post(
        f"/api/projects/{project_id}/tasks/project_chat",
        json={"chat_id": str(uuid.uuid4()), "message": "Explain.", "mode": "discuss"},
    )
    coach = client.post(
        f"/api/projects/{project_id}/tasks/paper_coach",
        json={"message": "Review the introduction."},
    )
    assert discuss.status_code == coach.status_code == 202
    assert discuss.json()["authorized_by"]["display_name"] == "Later name"
    assert coach.json()["authorized_by"]["display_name"] == "Later name"


def test_team_sync_and_tasks_use_only_current_trusted_member_snapshot(
    manifest,
    tmp_path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "team"
    setup_store = AppStore(data_dir / "rcp.sqlite3", space_kind="team")
    alice = setup_store.preprovision_team_member("Alice")
    bob = setup_store.preprovision_team_member("Bob")
    selected = [alice.user_id]
    app = create_app(
        str(manifest.path),
        data_dir=data_dir,
        trusted_principal_resolver=lambda _request, store: store.space_user(selected[0]),
    )
    client = TestClient(app)
    project_id = app.state.default_project_id
    service = app.state.service
    fixture_history = HistoryManager(service.manifest, service.history.workspace)
    fixture_history.append(seed_patch())

    node = service.history.state().nodes["rq/learning-after-shift"]
    synced = client.post(
        f"/api/projects/{project_id}/sync",
        json={
            "base_revision": service.history.state().revision,
            "nodes": [
                {
                    "node_id": node.id,
                    "base_updated_rev": node.updated_rev,
                    "changes": {"title": "Alice's edit"},
                }
            ],
        },
    )
    assert synced.status_code == 200, synced.text
    sync_authorizer = service.history.load_patches()[-1].authorized_by
    assert sync_authorizer is not None
    assert sync_authorizer.user_id == alice.user_id
    assert sync_authorizer.display_name == "Alice"

    monkeypatch.setattr(
        app.state.background_tasks,
        "_spawn_record",
        lambda record, _request, **_kwargs: record,
    )
    selected[0] = bob.user_id
    forged = client.post(
        f"/api/projects/{project_id}/tasks/seed",
        json={
            "user_id": alice.user_id,
            "authorized_by": sync_authorizer.model_dump(mode="json"),
        },
    )
    assert forged.status_code == 202, forged.text
    forged_record = app.state.background_tasks.store.agent_task(forged.json()["operation_id"])
    assert forged_record is not None
    assert forged_record.authorized_by is not None
    assert forged_record.authorized_by.user_id == bob.user_id
    assert forged_record.authorized_by.display_name == "Bob"
    app.state.background_tasks.store.complete_agent_task(
        forged_record.operation_id,
        applied_revision=None,
        result={},
    )

    selected[0] = alice.user_id
    renamed = client.patch("/api/identity", json={"display_name": "Alice renamed"})
    assert renamed.status_code == 200
    future = client.post(f"/api/projects/{project_id}/tasks/refresh", json={})
    assert future.status_code == 202, future.text
    future_record = app.state.background_tasks.store.agent_task(future.json()["operation_id"])
    assert future_record is not None
    assert future_record.authorized_by is not None
    assert future_record.authorized_by.display_name == "Alice renamed"
    assert sync_authorizer.display_name == "Alice"


def test_resume_retry_and_repair_capture_the_current_actor_instead_of_parent(
    manifest,
    tmp_path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "team"
    setup_store = AppStore(data_dir / "rcp.sqlite3", space_kind="team")
    first = setup_store.preprovision_team_member("First")
    second = setup_store.preprovision_team_member("Second")
    selected = [second.user_id]
    app = create_app(
        str(manifest.path),
        data_dir=data_dir,
        trusted_principal_resolver=lambda _request, store: store.space_user(selected[0]),
    )
    client = TestClient(app)
    project_id = app.state.default_project_id
    store = app.state.background_tasks.store
    parent_authorizer = AuthorizedHuman(
        space_id=store.space_id,
        user_id=first.user_id,
        display_name="First",
    )
    work_request = RunRequest(
        provider="codex",
        run_on="laptop",
        chat_scope="project",
        chat_id=str(uuid.uuid4()),
        message="Change the graph.",
        mode="work",
    ).model_dump(mode="json")
    records = {
        "resume": _agent_task(
            store,
            operation_id="current-actor-resume",
            project_id=project_id,
            authorized_by=parent_authorizer,
            status="paused",
            request=work_request,
        ),
        "retry": _agent_task(
            store,
            operation_id="current-actor-retry",
            project_id=project_id,
            authorized_by=parent_authorizer,
            status="failed",
            request=work_request,
        ),
        "repair-graph-update": _agent_task(
            store,
            operation_id="current-actor-repair",
            project_id=project_id,
            authorized_by=parent_authorizer,
            status="failed",
            request=work_request,
        ),
    }
    for record in records.values():
        store.create_agent_task(record)

    captured: dict[str, AuthorizedHuman | None] = {}

    def capture(action, record):
        def call(_operation_id, *, authorized_by=None, **_kwargs):
            captured[action] = authorized_by
            return record

        return call

    monkeypatch.setattr(
        app.state.background_tasks,
        "resume",
        capture("resume", records["resume"]),
    )
    monkeypatch.setattr(
        app.state.background_tasks,
        "retry",
        capture("retry", records["retry"]),
    )
    monkeypatch.setattr(
        app.state.background_tasks,
        "repair_graph_update",
        capture("repair-graph-update", records["repair-graph-update"]),
    )

    for action, record in records.items():
        response = client.post(
            f"/api/projects/{project_id}/tasks/{record.operation_id}/{action}",
            json={"user_id": first.user_id},
        )
        assert response.status_code == 202, response.text
        assert captured[action] is not None
        assert captured[action].user_id == second.user_id
        assert captured[action].display_name == "Second"


def test_automatic_watcher_delivery_inherits_one_origin_authorizer_and_refuses_ambiguity(
    manifest,
    tmp_path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "team"
    setup_store = AppStore(data_dir / "rcp.sqlite3", space_kind="team")
    first = setup_store.preprovision_team_member("First")
    second = setup_store.preprovision_team_member("Second")
    app = create_app(
        str(manifest.path),
        data_dir=data_dir,
        trusted_principal_resolver=lambda _request, _store: second,
    )
    project_id = app.state.default_project_id
    store = app.state.background_tasks.store
    first_authorizer = AuthorizedHuman(
        space_id=store.space_id,
        user_id=first.user_id,
        display_name="First",
    )
    second_authorizer = AuthorizedHuman(
        space_id=store.space_id,
        user_id=second.user_id,
        display_name="Second",
    )
    for operation_id, authorizer in (
        ("watcher-origin-first", first_authorizer),
        ("watcher-origin-second", second_authorizer),
        ("watcher-origin-legacy", None),
    ):
        store.create_agent_task(
            _agent_task(
                store,
                operation_id=operation_id,
                project_id=project_id,
                authorized_by=authorizer,
            )
        )

    captured: list[AuthorizedHuman] = []

    def fake_delivery(
        _tasks, _project_id, _kind, _request, _watcher_ids, *, authorized_by, **_kwargs
    ):
        captured.append(authorized_by)
        return None

    monkeypatch.setattr("rcp.api.app.start_watcher_notification", fake_delivery)
    callback = app.state.watcher_poller.on_completed
    same_authorizer = [
        _watcher(
            watcher_id="same-authorizer-one",
            project_id=project_id,
            origin_operation_id="watcher-origin-first",
        ),
        _watcher(
            watcher_id="same-authorizer-two",
            project_id=project_id,
            origin_operation_id="watcher-origin-first",
        ),
    ]
    mixed = [
        _watcher(
            watcher_id="mixed-first",
            project_id=project_id,
            origin_operation_id="watcher-origin-first",
        ),
        _watcher(
            watcher_id="mixed-second",
            project_id=project_id,
            origin_operation_id="watcher-origin-second",
        ),
    ]
    legacy = [
        _watcher(
            watcher_id="legacy",
            project_id=project_id,
            origin_operation_id="watcher-origin-legacy",
        )
    ]
    missing = [
        _watcher(
            watcher_id="missing",
            project_id=project_id,
            origin_operation_id="watcher-origin-missing",
        )
    ]
    for watcher_group in (same_authorizer, mixed, legacy, missing):
        for watcher in watcher_group:
            store.create_watchers([watcher])

    callback(same_authorizer)
    assert captured == [first_authorizer]

    callback(mixed)
    callback(legacy)
    callback(missing)

    assert captured == [first_authorizer]
    assert store.watcher("mixed-first").stop_reason is not None
    assert "different human authorizers" in store.watcher("mixed-first").stop_reason
    assert "predates durable human attribution" in store.watcher("legacy").stop_reason
    assert "originating task is unavailable" in store.watcher("missing").stop_reason
    assert all(
        store.watcher(watcher_id).status == "stopped" and store.watcher(watcher_id).notified
        for watcher_id in ("mixed-first", "mixed-second", "legacy", "missing")
    )


def test_reopened_poller_terminalizes_legacy_watcher_once_without_wake(
    manifest,
    tmp_path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "data"
    app = create_app(str(manifest.path), data_dir=data_dir)
    project_id = app.state.default_project_id
    store = app.state.background_tasks.store
    owner = store.local_owner
    assert owner is not None
    owner = store.rename_space_user(owner.user_id, "Pre-upgrade researcher")
    assert owner.display_name is not None
    authorizer = AuthorizedHuman(
        space_id=store.space_id,
        user_id=owner.user_id,
        display_name=owner.display_name,
    )
    origin = _agent_task(
        store,
        operation_id="pre-upgrade-work",
        project_id=project_id,
        authorized_by=authorizer,
        request={"mode": "work", "chat_id": "watcher-chat"},
    )
    store.create_agent_task(origin)
    store.create_watchers(
        [
            _watcher(
                watcher_id="pre-upgrade-watcher",
                project_id=project_id,
                origin_operation_id=origin.operation_id,
            )
        ]
    )
    with store.connection() as connection:
        connection.execute(
            """
            UPDATE graph_runs
            SET authorized_space_id = NULL, authorized_user_id = NULL,
                authorized_display_name = NULL
            WHERE operation_id = ?
            """,
            (origin.operation_id,),
        )

    reopened = create_app(str(manifest.path), data_dir=data_dir)
    wake_attempts: list[list[str]] = []

    def capture_wake(_tasks, _project_id, _kind, _request, watcher_ids, **_kwargs):
        wake_attempts.append(watcher_ids)
        return None

    monkeypatch.setattr("rcp.api.app.start_watcher_notification", capture_wake)

    first_pass = reopened.state.watcher_poller.poll_once()
    second_pass = reopened.state.watcher_poller.poll_once()

    terminal = reopened.state.background_tasks.store.watcher("pre-upgrade-watcher")
    assert first_pass and [item.watcher_id for item in first_pass[0]] == ["pre-upgrade-watcher"]
    assert second_pass == []
    assert wake_attempts == []
    assert terminal is not None
    assert terminal.status == "stopped"
    assert terminal.notified is True
    assert terminal.stop_reason is not None
    assert "predates durable human attribution" in terminal.stop_reason
    assert [
        task.operation_id for task in reopened.state.background_tasks.store.agent_tasks(project_id)
    ] == [origin.operation_id]


def test_old_project_url_alias_is_canonicalized_for_tasks_chat_and_paper(
    manifest,
    tmp_path,
    monkeypatch,
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    client = TestClient(app)
    project_id = app.state.default_project_id
    old_project_id = "legacy-project-url"
    store = app.state.background_tasks.store
    with store.connection() as connection:
        connection.execute(
            "INSERT INTO project_aliases(alias_id, canonical_project_id) VALUES (?, ?)",
            (old_project_id, project_id),
        )

    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    client = TestClient(app)
    store = app.state.background_tasks.store

    def refuse_request_path_sqlite_lookup(_project_id: str) -> str:
        raise AssertionError("project URL alias resolution must use the catalog snapshot")

    monkeypatch.setattr(store, "resolve_project_id", refuse_request_path_sqlite_lookup)

    task = _agent_task(
        store,
        operation_id="old-url-task",
        project_id=project_id,
        authorized_by=None,
    )
    store.create_agent_task(task)
    listed = client.get(f"/api/projects/{old_project_id}/tasks?proof=kept")
    detail = client.get(f"/api/projects/{old_project_id}/tasks/{task.operation_id}")
    assert listed.status_code == 200
    assert [item["operation_id"] for item in listed.json()] == [task.operation_id]
    assert detail.status_code == 200
    assert detail.json()["project_id"] == project_id

    chat_id = str(uuid.uuid4())
    transcript = ChatTranscript(
        chat_id=chat_id,
        kind="project_chat",
        node_id=None,
        title="Legacy URL chat",
        updated_at="2026-08-11T00:00:00+00:00",
        message_count=1,
        last_message_preview="Still here.",
        messages=[
            ChatMessage(
                message_id="legacy-url-message",
                role="assistant",
                text="Still here.",
                timestamp="2026-08-11T00:00:00+00:00",
            )
        ],
    )
    monkeypatch.setattr(app.state.service, "chat_transcript", lambda _chat_id: transcript)
    chat = client.get(f"/api/projects/{old_project_id}/chats/{chat_id}?proof=kept")
    assert chat.status_code == 200
    assert chat.json()["chat_id"] == chat_id

    created_paper = client.post(f"/api/projects/{old_project_id}/paper/create")
    read_paper = client.get(f"/api/projects/{old_project_id}/paper?proof=kept")
    assert created_paper.status_code == 200
    assert read_paper.status_code == 200
    assert read_paper.json() == created_paper.json()
    assert client.get("/api/health?proof=kept").status_code == 200
