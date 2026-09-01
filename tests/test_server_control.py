from __future__ import annotations

import io
import json
import os
import socket
import stat
import struct
import tempfile
import uuid
from dataclasses import replace
from pathlib import Path
from shutil import rmtree
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from rcp.api import create_app
from rcp.limits import (
    SERVER_CONTROL_BACKUP_CAPTURE_TIMEOUT_SECONDS,
    SERVER_CONTROL_IO_TIMEOUT_SECONDS,
    SERVER_CONTROL_PROJECT_PROVISION_TIMEOUT_SECONDS,
    SERVER_CONTROL_PROVIDER_CHECK_TIMEOUT_SECONDS,
    SERVER_CONTROL_UPDATE_MAINTENANCE_TIMEOUT_SECONDS,
    SERVER_CONTROL_UPDATE_VERIFY_TIMEOUT_SECONDS,
)
from rcp.server_ops import control
from rcp.server_ops.control import (
    SERVER_CONTROL_MAX_REQUEST_BYTES,
    SERVER_CONTROL_OPERATIONS,
    SERVER_CONTROL_SOCKET_MODE,
    ServerControlClient,
    ServerControlError,
    ServerControlPeer,
    ServerControlProbeResult,
    ServerControlProjectTransferActivationResult,
    ServerControlProjectTransferUploadResult,
    ServerControlRequest,
    ServerControlServer,
)
from rcp.server_ops.layout import DEFAULT_SERVER_LAYOUT
from rcp.server_runtime import (
    ServerMetadata,
    ServerMetadataError,
    installed_control_socket_path,
    published_server_metadata,
)
from rcp.storage import AppStore
from rcp.transfer.target import upload_target_transfer_archive
from tests.test_project_transfer_request_storage import _archive_bound_pair
from tests.test_transfer_import import _archive_fixture


@pytest.fixture
def control_root() -> Path:
    path = Path(tempfile.mkdtemp(prefix="rcp-control-", dir="/tmp"))
    os.chown(path, os.geteuid(), os.getegid())
    path.chmod(0o700)
    try:
        yield path
    finally:
        rmtree(path)


def _team_app(tmp_path: Path, control_root: Path):
    data_dir = tmp_path / "data"
    AppStore.initialize_team_space(data_dir / "rcp.sqlite3", "Control lab")
    metadata = ServerMetadata.create(
        data_dir,
        host="127.0.0.1",
        port=8421,
        owner_kind="cli",
        control_socket=control_root / "control.sock",
    )
    return data_dir, metadata, create_app(data_dir=data_dir, instance_metadata=metadata)


def test_installed_app_refuses_to_open_before_pending_restoration_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service_home = tmp_path / "home"
    server_root = service_home / "server"
    layout = replace(
        DEFAULT_SERVER_LAYOUT,
        service_home=service_home,
        server_root=server_root,
        source_checkout=server_root / "source",
        releases_root=server_root / "releases",
        data_dir=server_root / "data",
        projects_root=server_root / "projects",
        credentials_root=server_root / "credentials",
        update_checkpoints_root=server_root / "update-checkpoints",
        restore_operations_root=server_root / "restore-operations",
        codex_state_root=service_home / ".codex",
        claude_state_root=service_home / ".claude",
        ssh_state_root=service_home / ".ssh",
    )
    layout.update_checkpoints_root.mkdir(parents=True)
    journal = layout.update_checkpoints_root / "checkpoint-fixture" / "rollback-journal.json"
    monkeypatch.setattr(
        "rcp.api.app._installed_rollback_journals",
        lambda _root: (journal,),
    )

    with pytest.raises(RuntimeError, match="restoration is incomplete"):
        create_app(data_dir=layout.data_dir, server_layout=layout)

    assert not layout.data_dir.exists()


def test_team_lifespan_publishes_private_socket_without_opening_a_second_store(
    tmp_path: Path,
    control_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir, metadata, app = _team_app(tmp_path, control_root)
    socket_path = Path(metadata.control_socket or "")

    with published_server_metadata(data_dir, metadata), TestClient(app):
        info = socket_path.lstat()
        assert stat.S_ISSOCK(info.st_mode)
        assert stat.S_IMODE(info.st_mode) == SERVER_CONTROL_SOCKET_MODE
        assert (info.st_uid, info.st_gid) == (os.geteuid(), os.getegid())

        def refuse_second_store(*_args, **_kwargs):
            raise AssertionError("the control client tried to open SQLite")

        monkeypatch.setattr(AppStore, "__init__", refuse_second_store)
        result = ServerControlClient.from_data_dir(
            data_dir,
            expected_server_uid=os.geteuid(),
        ).probe()

        assert result == ServerControlProbeResult(
            instance_id=metadata.instance_id,
            pid=os.getpid(),
            data_dir_id=metadata.data_dir_id,
            space_id=app.state.space_id,
            operations=SERVER_CONTROL_OPERATIONS,
        )
        assert set(result.model_dump()) == {
            "instance_id",
            "pid",
            "data_dir_id",
            "space_id",
            "space_kind",
            "operations",
            "pending_member_removals",
        }

    assert not os.path.lexists(socket_path)


def test_control_probe_can_report_a_known_incomplete_operation_set() -> None:
    result = ServerControlProbeResult(
        instance_id=str(uuid.uuid4()),
        pid=os.getpid(),
        data_dir_id="d" * 64,
        space_id=str(uuid.uuid4()),
        operations=("probe",),
    )

    assert result.operations == ("probe",)
    with pytest.raises(ValueError, match="registry order"):
        ServerControlProbeResult(
            instance_id=str(uuid.uuid4()),
            pid=os.getpid(),
            data_dir_id="d" * 64,
            space_id=str(uuid.uuid4()),
            operations=("probe", "provider_readiness_check", "provider_readiness_plan"),
        )


def test_transfer_upload_control_shapes_bind_request_and_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_id = str(uuid.uuid4())
    project_id = str(uuid.uuid4())
    instance_id = str(uuid.uuid4())
    space_id = str(uuid.uuid4())
    identity = {
        "instance_id": instance_id,
        "pid": os.getpid(),
        "data_dir_id": "d" * 64,
        "space_id": space_id,
        "request_id": request_id,
        "project_id": project_id,
        "archive_sha256": "a" * 64,
        "archive_size_bytes": 42,
        "lease_boundary_sha256": "b" * 64,
    }

    plan_request = ServerControlRequest(
        request_id=str(uuid.uuid4()),
        instance_id=instance_id,
        operation="project_transfer_upload_plan",
        selector_kind="request",
        selector_id=request_id,
    )
    assert plan_request.boundary_sha256 is None
    plan = ServerControlProjectTransferUploadResult(**identity, state="active")
    assert control._validated_control_result(plan_request, plan) == plan

    complete_request = ServerControlRequest(
        request_id=str(uuid.uuid4()),
        instance_id=instance_id,
        operation="project_transfer_upload_complete",
        selector_kind="request",
        selector_id=request_id,
        boundary_sha256=identity["lease_boundary_sha256"],
    )
    complete = ServerControlProjectTransferUploadResult(**identity, state="complete")
    assert control._validated_control_result(complete_request, complete) == complete
    consumed = complete.model_copy(update={"state": "consumed"})
    assert control._validated_control_result(complete_request, consumed) == consumed

    activation_request = ServerControlRequest(
        request_id=str(uuid.uuid4()),
        instance_id=instance_id,
        operation="project_transfer_activate",
        selector_kind="request",
        selector_id=request_id,
        boundary_sha256=identity["lease_boundary_sha256"],
    )
    activation = ServerControlProjectTransferActivationResult(
        instance_id=instance_id,
        pid=os.getpid(),
        data_dir_id=identity["data_dir_id"],
        space_id=space_id,
        target_request_id=request_id,
        source_request_id=str(uuid.uuid4()),
        project_id=project_id,
        archive_sha256=identity["archive_sha256"],
        upload_lease_boundary_sha256=identity["lease_boundary_sha256"],
        archive_manifest_sha256="c" * 64,
        target_manifest_sha256="d" * 64,
        publication_sha256="e" * 64,
        activated_at="2026-08-31T20:00:00+00:00",
    )
    assert control._validated_control_result(activation_request, activation) == activation

    metadata = ServerMetadata.create(
        tmp_path / "data",
        host="127.0.0.1",
        port=8421,
        owner_kind="cli",
        control_socket=tmp_path / "control.sock",
    )
    client = ServerControlClient(metadata, expected_server_uid=os.geteuid())
    sent: list[ServerControlRequest] = []
    responses = iter((plan, complete, activation))

    def fake_exchange(request: ServerControlRequest):
        sent.append(request)
        return next(responses)

    monkeypatch.setattr(client, "_exchange", fake_exchange)
    assert client.project_transfer_upload_plan(request_id=request_id) == plan
    assert (
        client.complete_project_transfer_upload(
            request_id=request_id,
            lease_boundary_sha256=identity["lease_boundary_sha256"],
        )
        == complete
    )
    assert (
        client.activate_project_transfer(
            request_id=request_id,
            lease_boundary_sha256=identity["lease_boundary_sha256"],
        )
        == activation
    )
    assert sent[0].operation == "project_transfer_upload_plan"
    assert sent[0].selector_kind == "request"
    assert sent[0].selector_id == request_id
    assert sent[0].boundary_sha256 is None
    assert sent[1].operation == "project_transfer_upload_complete"
    assert sent[1].selector_kind == "request"
    assert sent[1].selector_id == request_id
    assert sent[1].boundary_sha256 == identity["lease_boundary_sha256"]
    assert sent[2].operation == "project_transfer_activate"
    assert sent[2].selector_kind == "request"
    assert sent[2].selector_id == request_id
    assert sent[2].boundary_sha256 == identity["lease_boundary_sha256"]

    with pytest.raises(ValueError, match="request selector"):
        ServerControlRequest(
            request_id=str(uuid.uuid4()),
            instance_id=instance_id,
            operation="project_transfer_upload_plan",
            selector_kind="project",
            selector_id=project_id,
        )
    with pytest.raises(ValueError, match="confirmed upload boundary"):
        ServerControlRequest(
            request_id=str(uuid.uuid4()),
            instance_id=instance_id,
            operation="project_transfer_upload_complete",
            selector_kind="request",
            selector_id=request_id,
        )
    with pytest.raises(ValueError, match="another request or lease boundary"):
        control._validated_control_result(
            complete_request,
            complete.model_copy(update={"lease_boundary_sha256": "c" * 64}),
        )


def test_running_team_service_owns_the_upload_lease_and_completion(
    tmp_path: Path,
    control_root: Path,
) -> None:
    _source, store, _source_request, request = _archive_bound_pair(tmp_path)
    data_dir = store.path.parent
    data_dir.chmod(0o700)
    metadata = ServerMetadata.create(
        data_dir,
        host="127.0.0.1",
        port=8421,
        owner_kind="cli",
        control_socket=control_root / "transfer-control.sock",
    )
    app = create_app(data_dir=data_dir, instance_metadata=metadata)

    with published_server_metadata(data_dir, metadata), TestClient(app):
        client = ServerControlClient.from_data_dir(
            data_dir,
            expected_server_uid=os.geteuid(),
        )
        plan = client.project_transfer_upload_plan(request_id=request.request_id)
        assert plan.state == "active"
        payload = b"one sealed transfer archive"
        upload_target_transfer_archive(
            data_dir,
            request.request_id,
            archive_sha256=plan.archive_sha256,
            archive_size_bytes=plan.archive_size_bytes,
            source=io.BytesIO(payload),
        )
        completed = client.complete_project_transfer_upload(
            request_id=request.request_id,
            lease_boundary_sha256=plan.lease_boundary_sha256,
        )

        assert completed.state == "complete"
        assert (
            client.project_transfer_upload_plan(request_id=request.request_id).state == "complete"
        )


def test_running_team_service_imports_and_compound_activates_the_uploaded_archive(
    manifest,
    tmp_path: Path,
    control_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _archive_fixture(
        manifest,
        tmp_path / "fixture",
        monkeypatch,
        seal_archive=True,
    )
    store = fixture["target"]
    sealed = fixture["sealed_archive_path"]
    assert isinstance(store, AppStore)
    assert isinstance(sealed, Path)
    data_dir = store.path.parent
    data_dir.chmod(0o700)
    metadata = ServerMetadata.create(
        data_dir,
        host="127.0.0.1",
        port=8421,
        owner_kind="cli",
        control_socket=control_root / "transfer-activation-control.sock",
    )
    app = create_app(data_dir=data_dir, instance_metadata=metadata)
    request_id = fixture["archive"].target_request_id

    with published_server_metadata(data_dir, metadata), TestClient(app):
        client = ServerControlClient.from_data_dir(
            data_dir,
            expected_server_uid=os.geteuid(),
        )
        plan = client.project_transfer_upload_plan(request_id=request_id)
        upload_target_transfer_archive(
            data_dir,
            request_id,
            archive_sha256=plan.archive_sha256,
            archive_size_bytes=plan.archive_size_bytes,
            source=io.BytesIO(sealed.read_bytes()),
        )
        completed = client.complete_project_transfer_upload(
            request_id=request_id,
            lease_boundary_sha256=plan.lease_boundary_sha256,
        )
        activated = client.activate_project_transfer(
            request_id=request_id,
            lease_boundary_sha256=completed.lease_boundary_sha256,
        )

    assert activated.state == "activated"
    assert activated.project_id == fixture["archive"].project_id
    assert store.project(activated.project_id) is not None
    assert store.project_transfer_request(request_id).phase == "target_activated"
    assert store.target_project_transfer_upload(request_id).status == "consumed"
    assert not (data_dir / "transfer-inbox" / f"{request_id}.rcp-transfer").exists()


def test_update_maintenance_refuses_an_active_upload_before_closing_admission(
    tmp_path: Path,
    control_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, store, _source_request, request = _archive_bound_pair(tmp_path / "fixture")
    data_dir = store.path.parent
    data_dir.chmod(0o700)
    server_root = data_dir.parent
    service_home = tmp_path / "home"
    layout = replace(
        DEFAULT_SERVER_LAYOUT,
        service_home=service_home,
        server_root=server_root,
        source_checkout=server_root / "source",
        releases_root=server_root / "releases",
        data_dir=data_dir,
        projects_root=server_root / "projects",
        credentials_root=server_root / "credentials",
        update_checkpoints_root=server_root / "update-checkpoints",
        restore_operations_root=server_root / "restore-operations",
        codex_state_root=service_home / ".codex",
        claude_state_root=service_home / ".claude",
        ssh_state_root=service_home / ".ssh",
        config_path=tmp_path / "etc" / "server.toml",
        current_release=tmp_path / "etc" / "current",
        runtime_dir=control_root,
        control_socket=control_root / "transfer-control.sock",
        cli_wrapper=tmp_path / "bin" / "rcp",
        systemd_unit=tmp_path / "etc" / "rcp.service",
    )
    layout.update_checkpoints_root.mkdir()
    layout.restore_operations_root.mkdir()
    layout.update_checkpoints_root.chmod(0o700)
    layout.restore_operations_root.chmod(0o700)
    metadata = ServerMetadata.create(
        data_dir,
        host="127.0.0.1",
        port=8421,
        owner_kind="cli",
        control_socket=layout.control_socket,
    )
    app = create_app(
        data_dir=data_dir,
        instance_metadata=metadata,
        server_layout=layout,
    )
    assert app.state.server_control is not None
    handler = app.state.server_control.handler
    service_peer = ServerControlPeer(pid=os.getpid(), uid=os.geteuid(), gid=os.getegid())
    root_peer = ServerControlPeer(pid=os.getpid(), uid=0, gid=0)
    plan = handler(
        ServerControlRequest(
            request_id=str(uuid.uuid4()),
            instance_id=metadata.instance_id,
            operation="project_transfer_upload_plan",
            selector_kind="request",
            selector_id=request.request_id,
        ),
        service_peer,
    )
    assert isinstance(plan, ServerControlProjectTransferUploadResult)
    monkeypatch.setattr(
        "rcp.api.app.SERVER_CONTROL_UPDATE_MAINTENANCE_TIMEOUT_SECONDS",
        0.01,
    )

    with pytest.raises(ServerControlError, match="active project transfer upload") as caught:
        handler(
            ServerControlRequest(
                request_id=str(uuid.uuid4()),
                instance_id=metadata.instance_id,
                operation="update_maintenance_enter",
                selector_id=str(uuid.uuid4()),
                boundary_sha256="a" * 64,
            ),
            root_peer,
        )

    assert caught.value.code == "operation_refused"
    assert not app.state.background_admission_gate.closed
    payload = b"one sealed transfer archive"
    upload_target_transfer_archive(
        data_dir,
        request.request_id,
        archive_sha256=plan.archive_sha256,
        archive_size_bytes=plan.archive_size_bytes,
        source=io.BytesIO(payload),
    )
    completed = handler(
        ServerControlRequest(
            request_id=str(uuid.uuid4()),
            instance_id=metadata.instance_id,
            operation="project_transfer_upload_complete",
            selector_kind="request",
            selector_id=request.request_id,
            boundary_sha256=plan.lease_boundary_sha256,
        ),
        service_peer,
    )
    assert isinstance(completed, ServerControlProjectTransferUploadResult)
    assert completed.state == "complete"


def test_control_socket_is_refused_for_a_personal_or_non_cli_app(
    tmp_path: Path, control_root: Path
) -> None:
    personal_data = tmp_path / "personal"
    metadata = ServerMetadata.create(
        personal_data,
        host="127.0.0.1",
        port=8421,
        owner_kind="cli",
        control_socket=control_root / "control.sock",
    )
    with pytest.raises(ValueError, match="only to an installed CLI-owned team service"):
        create_app(data_dir=personal_data, instance_metadata=metadata)

    team_data = tmp_path / "team"
    AppStore.initialize_team_space(team_data / "rcp.sqlite3", "Control lab")
    desktop = replace(
        metadata,
        owner_kind="desktop",
        data_dir_id=ServerMetadata.create(
            team_data, host="127.0.0.1", port=8421, owner_kind="desktop"
        ).data_dir_id,
    )
    with pytest.raises(ValueError, match="only to an installed CLI-owned team service"):
        create_app(data_dir=team_data, instance_metadata=desktop)


def test_update_maintenance_blocks_new_machine_operations(
    tmp_path: Path,
    control_root: Path,
) -> None:
    _data_dir, metadata, app = _team_app(tmp_path, control_root)
    app.state.background_admission_gate.close_and_wait(timeout=1)
    request = ServerControlRequest(
        request_id=str(uuid.uuid4()),
        instance_id=metadata.instance_id,
        operation="project_provision_plan",
        selector_kind="request",
        selector_id=str(uuid.uuid4()),
    )

    with pytest.raises(ServerControlError, match="maintenance") as caught:
        app.state.server_control.handler(
            request,
            ServerControlPeer(pid=os.getpid(), uid=os.geteuid(), gid=os.getegid()),
        )

    assert caught.value.code == "operation_refused"


def test_update_maintenance_blocks_get_routes_that_can_mutate(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path / "data")
    app.state.runtime_admission_gate.close_and_wait(timeout=1)

    with TestClient(app) as client:
        response = client.get("/api/health")

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "server_update_maintenance"


def test_update_control_operations_require_a_root_peer(
    tmp_path: Path,
    control_root: Path,
) -> None:
    _data_dir, metadata, app = _team_app(tmp_path, control_root)
    request = ServerControlRequest(
        request_id=str(uuid.uuid4()),
        instance_id=metadata.instance_id,
        operation="update_maintenance_enter",
        selector_id=str(uuid.uuid4()),
        boundary_sha256="a" * 64,
    )

    with pytest.raises(ServerControlError, match="root server coordinator") as caught:
        app.state.server_control.handler(
            request,
            ServerControlPeer(pid=os.getpid(), uid=1, gid=1),
        )

    assert caught.value.code == "operation_refused"


def test_provider_check_uses_its_bounded_operation_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = ServerMetadata.create(
        tmp_path / "data",
        host="127.0.0.1",
        port=8421,
        owner_kind="cli",
        control_socket=tmp_path / "control.sock",
    )
    observed: list[float] = []

    class RefusingSocket:
        def settimeout(self, timeout: float) -> None:
            observed.append(timeout)

        def connect(self, _path: str) -> None:
            raise RuntimeError("stop after observing the timeout")

        def close(self) -> None:
            pass

    monkeypatch.setattr(control.socket, "socket", lambda *_args: RefusingSocket())
    client = ServerControlClient(
        metadata,
        expected_server_uid=os.geteuid(),
    )

    with pytest.raises(RuntimeError, match="stop after observing"):
        client.probe()
    with pytest.raises(RuntimeError, match="stop after observing"):
        client.capture_backup_sqlite()
    with pytest.raises(RuntimeError, match="stop after observing"):
        client.check_provider_readiness(
            selector_kind="request",
            selector_id=str(uuid.uuid4()),
            boundary_sha256="a" * 64,
            target_id="b" * 64,
        )
    with pytest.raises(RuntimeError, match="stop after observing"):
        client.advance_project_provision(
            request_id=str(uuid.uuid4()),
            boundary_sha256="a" * 64,
            target_id="b" * 64,
        )
    with pytest.raises(RuntimeError, match="stop after observing"):
        client.complete_project_transfer_upload(
            request_id=str(uuid.uuid4()),
            lease_boundary_sha256="d" * 64,
        )
    operation_id = str(uuid.uuid4())
    with pytest.raises(RuntimeError, match="stop after observing"):
        client.enter_update_maintenance(
            operation_id=operation_id,
            receipt_sha256="c" * 64,
        )
    with pytest.raises(RuntimeError, match="stop after observing"):
        client.verify_update_candidate(
            operation_id=operation_id,
            receipt_sha256="c" * 64,
        )
    with pytest.raises(RuntimeError, match="stop after observing"):
        client.release_update_fence(
            operation_id=operation_id,
            receipt_sha256="c" * 64,
        )
    with pytest.raises(RuntimeError, match="stop after observing"):
        client.abort_update_maintenance(
            operation_id=operation_id,
            receipt_sha256="c" * 64,
        )

    assert observed == [
        SERVER_CONTROL_IO_TIMEOUT_SECONDS,
        SERVER_CONTROL_BACKUP_CAPTURE_TIMEOUT_SECONDS,
        SERVER_CONTROL_PROVIDER_CHECK_TIMEOUT_SECONDS,
        SERVER_CONTROL_PROJECT_PROVISION_TIMEOUT_SECONDS,
        SERVER_CONTROL_PROJECT_PROVISION_TIMEOUT_SECONDS,
        SERVER_CONTROL_UPDATE_MAINTENANCE_TIMEOUT_SECONDS,
        SERVER_CONTROL_UPDATE_VERIFY_TIMEOUT_SECONDS,
        SERVER_CONTROL_UPDATE_VERIFY_TIMEOUT_SECONDS,
        SERVER_CONTROL_UPDATE_VERIFY_TIMEOUT_SECONDS,
    ]


def test_installed_control_socket_is_discovered_only_for_the_service_account(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rcp.server_ops.config as config_module
    import rcp.server_ops.layout as layout_module
    import rcp.server_runtime as runtime_module

    data_dir = tmp_path / "data"
    socket_path = Path("/run/rcp/control.sock")
    config_path = tmp_path / "server.toml"
    config_path.touch()
    monkeypatch.setattr(
        layout_module,
        "DEFAULT_SERVER_LAYOUT",
        replace(DEFAULT_SERVER_LAYOUT, config_path=config_path),
    )
    monkeypatch.setattr(
        config_module,
        "load_installed_server_config",
        lambda path: SimpleNamespace(
            service_account="rcp",
            paths=SimpleNamespace(data_dir=str(data_dir), control_socket=str(socket_path)),
        ),
    )
    account = SimpleNamespace(pw_uid=os.geteuid(), pw_gid=os.getegid())
    monkeypatch.setattr(runtime_module.pwd, "getpwnam", lambda _name: account)

    assert installed_control_socket_path(data_dir) == socket_path
    assert installed_control_socket_path(tmp_path / "other-data") is None

    account.pw_uid = os.geteuid() + 1
    with pytest.raises(ServerMetadataError, match="configured service account"):
        installed_control_socket_path(data_dir)


def test_unauthorized_os_peer_is_rejected_before_request_dispatch(control_root: Path) -> None:
    metadata = ServerMetadata.create(
        control_root / "data",
        host="127.0.0.1",
        port=8421,
        owner_kind="cli",
        control_socket=control_root / "control.sock",
    )
    dispatched = False

    def handler(_request, _peer):
        nonlocal dispatched
        dispatched = True
        raise AssertionError("unauthorized peers must not reach dispatch")

    server = ServerControlServer(
        control_root / "control.sock",
        instance_id=metadata.instance_id,
        owner_uid=os.geteuid(),
        owner_gid=os.getegid(),
        handler=handler,
        peer_resolver=lambda _connection: ServerControlPeer(
            pid=os.getpid(), uid=os.geteuid() + 1, gid=os.getegid()
        ),
    )
    server.start()
    try:
        with pytest.raises(ServerControlError) as caught:
            ServerControlClient(metadata, expected_server_uid=os.geteuid()).probe()
        assert caught.value.code == "unauthorized_peer"
        assert dispatched is False
    finally:
        server.stop()


@pytest.mark.parametrize(
    ("payload", "expected_code"),
    [
        (struct.pack("!I", 1) + b"{", "invalid_request"),
        (struct.pack("!I", SERVER_CONTROL_MAX_REQUEST_BYTES + 1), "oversized_request"),
        (
            lambda instance_id: _framed_json(
                {
                    "protocol_version": control.SERVER_CONTROL_PROTOCOL_VERSION,
                    "request_id": str(uuid.uuid4()),
                    "instance_id": instance_id,
                    "operation": "probe",
                    "member_id": str(uuid.uuid4()),
                }
            ),
            "invalid_request",
        ),
    ],
)
def test_malformed_oversized_and_member_claim_requests_fail_closed(
    control_root: Path,
    payload,
    expected_code: str,
) -> None:
    server, metadata = _standalone_server(control_root)
    server.start()
    try:
        raw = payload(metadata.instance_id) if callable(payload) else payload
        response = _raw_request(Path(metadata.control_socket or ""), raw)
        assert response["ok"] is False
        assert response["error"]["code"] == expected_code
    finally:
        server.stop()


def test_root_machine_peer_reaches_probe_without_becoming_a_member(control_root: Path) -> None:
    observed: list[ServerControlPeer] = []
    server, metadata = _standalone_server(
        control_root,
        peer_resolver=lambda _connection: ServerControlPeer(pid=os.getpid(), uid=0, gid=0),
        observed=observed,
    )
    server.start()
    try:
        result = ServerControlClient(metadata, expected_server_uid=os.geteuid()).probe()
        assert result.space_kind == "team"
        assert observed == [ServerControlPeer(pid=os.getpid(), uid=0, gid=0)]
        assert result.pending_member_removals == ()
        assert "member_id" not in json.dumps(result.model_dump())
        assert "user" not in json.dumps(result.model_dump())
    finally:
        server.stop()


def test_restart_recovers_a_safe_stale_socket(control_root: Path) -> None:
    socket_path = control_root / "control.sock"
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(socket_path))
    os.chmod(socket_path, SERVER_CONTROL_SOCKET_MODE)
    stale.close()

    server, metadata = _standalone_server(control_root)
    server.start()
    try:
        assert ServerControlClient(
            metadata, expected_server_uid=os.geteuid()
        ).probe().instance_id == (metadata.instance_id)
    finally:
        server.stop()
    assert not os.path.lexists(socket_path)


def test_shutdown_does_not_remove_a_replacement_socket(control_root: Path) -> None:
    server, metadata = _standalone_server(control_root)
    socket_path = Path(metadata.control_socket or "")
    server.start()
    socket_path.unlink()
    replacement = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        replacement.bind(str(socket_path))
        os.chmod(socket_path, SERVER_CONTROL_SOCKET_MODE)
        server.stop()
        assert socket_path.exists()
        assert stat.S_ISSOCK(socket_path.lstat().st_mode)
    finally:
        replacement.close()
        socket_path.unlink(missing_ok=True)


def _standalone_server(
    control_root: Path,
    *,
    peer_resolver=control.unix_peer_identity,
    observed: list[ServerControlPeer] | None = None,
) -> tuple[ServerControlServer, ServerMetadata]:
    metadata = ServerMetadata.create(
        control_root / "data",
        host="127.0.0.1",
        port=8421,
        owner_kind="cli",
        control_socket=control_root / "control.sock",
    )

    def handler(request: ServerControlRequest, peer: ServerControlPeer):
        if observed is not None:
            observed.append(peer)
        return ServerControlProbeResult(
            instance_id=request.instance_id,
            pid=os.getpid(),
            data_dir_id=metadata.data_dir_id,
            space_id=str(uuid.uuid4()),
            operations=SERVER_CONTROL_OPERATIONS,
        )

    return (
        ServerControlServer(
            control_root / "control.sock",
            instance_id=metadata.instance_id,
            owner_uid=os.geteuid(),
            owner_gid=os.getegid(),
            handler=handler,
            peer_resolver=peer_resolver,
        ),
        metadata,
    )


def _framed_json(value: object) -> bytes:
    body = json.dumps(value, separators=(",", ":")).encode("utf-8")
    return struct.pack("!I", len(body)) + body


def _raw_request(path: Path, payload: bytes) -> dict[str, object]:
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(5)
    try:
        connection.connect(str(path))
        connection.sendall(payload)
        header = _receive_exact(connection, 4)
        (size,) = struct.unpack("!I", header)
        return json.loads(_receive_exact(connection, size))
    finally:
        connection.close()


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    body = b""
    while len(body) < size:
        chunk = connection.recv(size - len(body))
        if not chunk:
            raise AssertionError("control server closed an incomplete response")
        body += chunk
    return body
