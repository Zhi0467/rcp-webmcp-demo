from __future__ import annotations

import io
from pathlib import Path

import pytest

import rcp.transfer.importer as transfer_importer
from rcp.agents import AgentLauncher
from rcp.server_runtime import ServerMetadata
from rcp.setup import ProjectSetupManager
from rcp.transfer.target import (
    TargetTransferActivationCoordinator,
    TargetTransferUploadCoordinator,
    TargetTransferUploadError,
    target_transfer_archive_path,
    upload_target_transfer_archive,
)

from .test_transfer_import import _archive_fixture


def _coordinator_fixture(manifest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fixture = _archive_fixture(
        manifest,
        tmp_path,
        monkeypatch,
        seal_archive=True,
    )
    sealed = fixture["sealed_archive_path"]
    assert sealed is not None
    target = fixture["target"]
    archive = fixture["archive"]
    envelope = fixture["envelope"]
    fixture["catalog"].data_dir.chmod(0o700)
    upload = target.begin_target_project_transfer_upload(archive.target_request_id)
    uploaded = upload_target_transfer_archive(
        fixture["catalog"].data_dir,
        archive.target_request_id,
        archive_sha256=envelope.archive_sha256,
        archive_size_bytes=envelope.archive_size_bytes,
        source=io.BytesIO(sealed.read_bytes()),
    )
    assert uploaded.archive_path == target_transfer_archive_path(
        fixture["catalog"].data_dir,
        archive.target_request_id,
    )
    target.complete_target_project_transfer_upload(
        archive.target_request_id,
        lease_boundary_sha256=upload.lease_boundary_sha256,
    )
    setup = ProjectSetupManager(
        fixture["catalog"].data_dir,
        fixture["catalog"],
        AgentLauncher(),
    )
    coordinator = TargetTransferActivationCoordinator(
        target,
        fixture["catalog"],
        setup,
        fixture["catalog"].data_dir,
    )
    return fixture, coordinator


def _patch_payloads(state_root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(state_root).as_posix(): path.read_bytes()
        for path in sorted((state_root / "patches").rglob("*.json"))
    }


def _activate(coordinator: TargetTransferActivationCoordinator, fixture):
    request_id = fixture["archive"].target_request_id
    upload = fixture["target"].target_project_transfer_upload(request_id)
    assert upload is not None
    return coordinator.activate(
        request_id,
        lease_boundary_sha256=upload.lease_boundary_sha256,
    )


def test_sealed_target_archive_activates_once_and_retry_returns_same_receipt(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, coordinator = _coordinator_fixture(manifest, tmp_path, monkeypatch)
    archive = fixture["archive"]
    target = fixture["target"]
    data_dir = fixture["catalog"].data_dir

    with pytest.raises(TargetTransferUploadError, match="lease boundary changed"):
        coordinator.activate(
            archive.target_request_id,
            lease_boundary_sha256="0" * 64,
        )
    assert target.project(archive.project_id) is None
    assert target_transfer_archive_path(data_dir, archive.target_request_id).is_file()

    first = _activate(coordinator, fixture)

    assert first.target_request_id == archive.target_request_id
    assert first.project_id == archive.project_id
    assert target.project(archive.project_id) is not None
    assert target.target_project_transfer_activation(archive.target_request_id) == first
    upload = target.target_project_transfer_upload(archive.target_request_id)
    assert upload is not None and upload.status == "consumed"
    assert not target_transfer_archive_path(data_dir, archive.target_request_id).exists()
    assert not (
        data_dir / "run-stage" / "project-transfer-activation" / archive.target_request_id
    ).exists()
    patch_payloads = _patch_payloads(fixture["target_state_root"])
    with target.connection() as connection:
        project_count = connection.execute(
            "SELECT COUNT(*) FROM projects WHERE project_id = ?",
            (archive.project_id,),
        ).fetchone()[0]
        member_count = connection.execute(
            "SELECT COUNT(*) FROM project_members WHERE project_id = ?",
            (archive.project_id,),
        ).fetchone()[0]
    assert project_count == 1
    assert member_count == 1

    upload_control = TargetTransferUploadCoordinator(
        target,
        data_dir,
        ServerMetadata.create(
            data_dir,
            host="127.0.0.1",
            port=8421,
            owner_kind="cli",
        ),
    )
    consumed_plan = upload_control.plan(archive.target_request_id)
    assert consumed_plan.state == "consumed"
    consumed_completion = upload_control.complete(
        archive.target_request_id,
        lease_boundary_sha256=consumed_plan.lease_boundary_sha256,
    )
    assert consumed_completion.state == "consumed"

    repeated = _activate(coordinator, fixture)

    assert repeated == first
    assert _patch_payloads(fixture["target_state_root"]) == patch_payloads
    with target.connection() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM projects WHERE project_id = ?",
                (archive.project_id,),
            ).fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM project_members WHERE project_id = ?",
                (archive.project_id,),
            ).fetchone()[0]
            == 1
        )


def test_activation_failure_keeps_exact_inbox_and_never_registers_project(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, coordinator = _coordinator_fixture(manifest, tmp_path, monkeypatch)
    archive = fixture["archive"]
    target = fixture["target"]
    data_dir = fixture["catalog"].data_dir
    original_activate = target.activate_target_project_transfer

    def fail_before_activation(*_args, **_kwargs):
        raise RuntimeError("injected failure before activation")

    monkeypatch.setattr(target, "activate_target_project_transfer", fail_before_activation)
    with pytest.raises(RuntimeError, match="injected failure"):
        _activate(coordinator, fixture)

    assert target.project(archive.project_id) is None
    assert target.target_project_transfer_activation(archive.target_request_id) is None
    assert target_transfer_archive_path(data_dir, archive.target_request_id).is_file()
    assert not (
        data_dir / "run-stage" / "project-transfer-activation" / archive.target_request_id
    ).exists()
    imported = target.project_transfer_import(archive.target_request_id)
    assert imported is not None and imported.status == "complete"

    monkeypatch.setattr(target, "activate_target_project_transfer", original_activate)
    receipt = _activate(coordinator, fixture)
    assert receipt.project_id == archive.project_id
    assert not target_transfer_archive_path(data_dir, archive.target_request_id).exists()


@pytest.mark.parametrize("boundary", ("canonical", "project_files"))
def test_activation_retry_reuses_prepublication_configuration_after_partial_publication(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    fixture, coordinator = _coordinator_fixture(manifest, tmp_path, monkeypatch)
    archive = fixture["archive"]
    target = fixture["target"]
    attribute = {
        "canonical": "_publish_canonical",
        "project_files": "_publish_project_files",
    }[boundary]
    original = getattr(transfer_importer, attribute)

    def fail_after_publication(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError(f"injected {boundary} publication crash")

    monkeypatch.setattr(transfer_importer, attribute, fail_after_publication)
    with pytest.raises(RuntimeError, match=f"injected {boundary}"):
        _activate(coordinator, fixture)

    pending = target.project_transfer_import(archive.target_request_id)
    assert pending is not None and pending.status == "database_imported"
    assert target.project(archive.project_id) is None
    persisted = target.project_transfer_import_configuration_receipt_json(archive.target_request_id)
    assert persisted is not None

    monkeypatch.setattr(transfer_importer, attribute, original)
    completed = _activate(coordinator, fixture)

    assert completed.project_id == archive.project_id
    assert target.project(archive.project_id) is not None
