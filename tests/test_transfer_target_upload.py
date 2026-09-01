from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import uuid
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

import rcp.transfer.target as target
from rcp.server_ops.cli import CallerIdentity, ServerEventEmitter
from rcp.server_ops.control import (
    ServerControlProjectTransferActivationResult,
    ServerControlProjectTransferUploadResult,
)
from rcp.server_ops.models import ServerCommandRequest
from rcp.server_runtime import ServerMetadata
from tests.test_project_transfer_request_storage import _archive_bound_pair

REQUEST_ID = "11111111-1111-4111-8111-111111111111"
OTHER_REQUEST_ID = "22222222-2222-4222-8222-222222222222"


def _data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    root.mkdir(mode=0o700)
    return root


def _payload() -> tuple[bytes, str]:
    payload = b"sealed target transfer bytes\n"
    return payload, hashlib.sha256(payload).hexdigest()


def _upload(root: Path, request_id: str = REQUEST_ID, payload: bytes | None = None):
    payload = payload if payload is not None else _payload()[0]
    return target.upload_target_transfer_archive(
        root,
        request_id,
        archive_sha256=hashlib.sha256(payload).hexdigest(),
        archive_size_bytes=len(payload),
        source=io.BytesIO(payload),
    )


def test_paths_are_request_and_digest_derived(tmp_path: Path) -> None:
    root = _data_root(tmp_path)
    _, digest = _payload()

    archive = root / "transfer-inbox" / f"{REQUEST_ID}.rcp-transfer"
    assert target.target_transfer_archive_path(root, REQUEST_ID) == archive
    assert target.target_transfer_partial_path(root, REQUEST_ID, digest) == archive.parent / (
        f".{REQUEST_ID}.{digest}.partial"
    )

    with pytest.raises(ValueError, match="canonical UUID4"):
        target.target_transfer_archive_path(root, "A1111111-1111-4111-8111-111111111111")
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        target.target_transfer_partial_path(root, REQUEST_ID, "A" * 64)
    with pytest.raises(ValueError, match="positive integer"):
        target.acquire_target_transfer_upload_lease(
            root,
            REQUEST_ID,
            archive_sha256=digest,
            archive_size_bytes=0,
        )


def test_upload_streams_to_private_final_and_is_idempotent(tmp_path: Path) -> None:
    root = _data_root(tmp_path)
    payload, digest = _payload()

    first = target.upload_target_transfer_archive(
        root,
        REQUEST_ID,
        archive_sha256=digest,
        archive_size_bytes=len(payload),
        source=io.BytesIO(payload),
    )
    assert first.reused_existing is False
    assert first.archive_path.read_bytes() == payload
    assert stat.S_IMODE(first.archive_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(first.archive_path.parent.stat().st_mode) == 0o700
    assert list(first.archive_path.parent.iterdir()) == [first.archive_path]

    second = target.upload_target_transfer_archive(
        root,
        REQUEST_ID,
        archive_sha256=digest,
        archive_size_bytes=len(payload),
        source=io.BytesIO(payload),
    )
    assert second.reused_existing is True
    assert second == target.verify_target_transfer_archive(
        root,
        REQUEST_ID,
        expected_sha256=digest,
        expected_size_bytes=len(payload),
    )
    assert first.archive_path.read_bytes() == payload


def test_wrong_bytes_never_publish_and_known_partial_is_recoverable(tmp_path: Path) -> None:
    root = _data_root(tmp_path)
    payload, digest = _payload()
    wrong = b"wrong target bytes"
    partial = target.target_transfer_partial_path(root, REQUEST_ID, digest)
    final = target.target_transfer_archive_path(root, REQUEST_ID)

    with pytest.raises(target.TargetTransferUploadError, match="expected bytes"):
        target.upload_target_transfer_archive(
            root,
            REQUEST_ID,
            archive_sha256=digest,
            archive_size_bytes=len(payload),
            source=io.BytesIO(wrong),
        )
    assert partial.read_bytes() == wrong
    assert stat.S_IMODE(partial.stat().st_mode) == 0o600
    assert not final.exists()

    completed = target.upload_target_transfer_archive(
        root,
        REQUEST_ID,
        archive_sha256=digest,
        archive_size_bytes=len(payload),
        source=io.BytesIO(payload),
    )
    assert completed.archive_path.read_bytes() == payload
    assert not partial.exists()


def test_partial_recovery_does_not_touch_another_request(tmp_path: Path) -> None:
    root = _data_root(tmp_path)
    payload, digest = _payload()
    other_partial = target.target_transfer_partial_path(root, OTHER_REQUEST_ID, digest)
    other_partial.parent.mkdir(parents=True, mode=0o700)
    other_partial.write_bytes(b"other request remains")
    other_partial.chmod(0o600)

    with pytest.raises(target.TargetTransferUploadError):
        target.upload_target_transfer_archive(
            root,
            REQUEST_ID,
            archive_sha256=digest,
            archive_size_bytes=len(payload),
            source=io.BytesIO(b"incomplete"),
        )
    assert other_partial.read_bytes() == b"other request remains"


def test_active_request_lease_rejects_concurrent_owner(tmp_path: Path) -> None:
    root = _data_root(tmp_path)
    _, digest = _payload()
    lease = target.acquire_target_transfer_upload_lease(
        root,
        REQUEST_ID,
        archive_sha256=digest,
        archive_size_bytes=len(_payload()[0]),
    )
    try:
        with pytest.raises(target.TargetTransferUploadBusy, match="already owns"):
            target.acquire_target_transfer_upload_lease(
                root,
                REQUEST_ID,
                archive_sha256=digest,
                archive_size_bytes=len(_payload()[0]),
            )
    finally:
        lease.release()
    assert lease.active is False


def test_crash_before_publication_leaves_only_exact_partial(tmp_path: Path, monkeypatch) -> None:
    root = _data_root(tmp_path)
    payload, digest = _payload()

    def crash(_partial: Path, _final: Path) -> None:
        raise RuntimeError("simulated crash before final publication")

    monkeypatch.setattr(target, "_publish_no_overwrite", crash)
    with pytest.raises(RuntimeError, match="before final publication"):
        target.upload_target_transfer_archive(
            root,
            REQUEST_ID,
            archive_sha256=digest,
            archive_size_bytes=len(payload),
            source=io.BytesIO(payload),
        )
    partial = target.target_transfer_partial_path(root, REQUEST_ID, digest)
    assert partial.read_bytes() == payload
    assert not target.target_transfer_archive_path(root, REQUEST_ID).exists()

    monkeypatch.undo()
    completed = _upload(root)
    assert completed.reused_existing is False
    assert not partial.exists()


def test_crash_after_link_is_recovered_without_replacing_final(tmp_path: Path, monkeypatch) -> None:
    root = _data_root(tmp_path)
    payload, digest = _payload()
    original = target._publish_no_overwrite

    def crash_after_link(partial: Path, final: Path) -> None:
        os.link(partial, final, follow_symlinks=False)
        raise RuntimeError("simulated crash after final publication")

    monkeypatch.setattr(target, "_publish_no_overwrite", crash_after_link)
    with pytest.raises(RuntimeError, match="after final publication"):
        target.upload_target_transfer_archive(
            root,
            REQUEST_ID,
            archive_sha256=digest,
            archive_size_bytes=len(payload),
            source=io.BytesIO(payload),
        )
    final = target.target_transfer_archive_path(root, REQUEST_ID)
    partial = target.target_transfer_partial_path(root, REQUEST_ID, digest)
    assert final.read_bytes() == payload
    assert partial.read_bytes() == payload

    monkeypatch.setattr(target, "_publish_no_overwrite", original)
    retry = target.upload_target_transfer_archive(
        root,
        REQUEST_ID,
        archive_sha256=digest,
        archive_size_bytes=len(payload),
        source=io.BytesIO(payload),
    )
    assert retry.reused_existing is True
    assert final.read_bytes() == payload
    assert not partial.exists()


@pytest.mark.parametrize("kind", ["final", "partial"])
def test_unsafe_existing_target_entry_fails_closed(tmp_path: Path, kind: str) -> None:
    root = _data_root(tmp_path)
    payload, digest = _payload()
    if kind == "final":
        path = target.target_transfer_archive_path(root, REQUEST_ID)
    else:
        path = target.target_transfer_partial_path(root, REQUEST_ID, digest)
    path.parent.mkdir(parents=True, mode=0o700)
    path.parent.parent.chmod(0o700)
    outside = tmp_path / "outside"
    outside.write_bytes(b"must remain")
    path.symlink_to(outside)

    with pytest.raises(target.TargetTransferUploadError, match="symlink"):
        target.upload_target_transfer_archive(
            root,
            REQUEST_ID,
            archive_sha256=digest,
            archive_size_bytes=len(payload),
            source=io.BytesIO(payload),
        )
    assert outside.read_bytes() == b"must remain"


def test_mismatched_final_is_never_overwritten(tmp_path: Path) -> None:
    root = _data_root(tmp_path)
    payload, digest = _payload()
    final = target.target_transfer_archive_path(root, REQUEST_ID)
    final.parent.mkdir(parents=True, mode=0o700)
    final.parent.parent.chmod(0o700)
    original = b"old final bytes"
    final.write_bytes(original)
    final.chmod(0o600)

    with pytest.raises(target.TargetTransferUploadError, match="differs"):
        target.upload_target_transfer_archive(
            root,
            REQUEST_ID,
            archive_sha256=digest,
            archive_size_bytes=len(payload),
            source=io.BytesIO(payload),
        )
    assert final.read_bytes() == original


def test_binary_input_and_exact_size_are_required(tmp_path: Path) -> None:
    root = _data_root(tmp_path)
    payload, digest = _payload()

    with pytest.raises(target.TargetTransferUploadError, match="yield bytes"):
        target.upload_target_transfer_archive(
            root,
            REQUEST_ID,
            archive_sha256=digest,
            archive_size_bytes=len(payload),
            source=io.StringIO(payload.decode()),  # type: ignore[arg-type]
        )
    with pytest.raises(target.TargetTransferUploadError, match="exceeds"):
        target.upload_target_transfer_archive(
            root,
            REQUEST_ID,
            archive_sha256=digest,
            archive_size_bytes=len(payload) - 1,
            source=io.BytesIO(payload),
        )


def test_coordinator_binds_the_durable_lease_to_verified_bytes(tmp_path: Path) -> None:
    _source, store, _source_request, request = _archive_bound_pair(tmp_path)
    data_dir = store.path.parent
    data_dir.chmod(0o700)
    metadata = ServerMetadata.create(
        data_dir,
        host="127.0.0.1",
        port=8421,
        owner_kind="cli",
        control_socket=data_dir / "control.sock",
    )
    coordinator = target.TargetTransferUploadCoordinator(store, data_dir, metadata)

    plan = coordinator.plan(request.request_id)
    assert plan.state == "active"
    assert not coordinator.uploads_idle()
    payload = b"one sealed transfer archive"
    assert hashlib.sha256(payload).hexdigest() == plan.archive_sha256
    target.upload_target_transfer_archive(
        data_dir,
        request.request_id,
        archive_sha256=plan.archive_sha256,
        archive_size_bytes=plan.archive_size_bytes,
        source=io.BytesIO(payload),
    )
    completed = coordinator.complete(
        request.request_id,
        lease_boundary_sha256=plan.lease_boundary_sha256,
    )

    assert completed.state == "complete"
    stored = store.target_project_transfer_upload(request.request_id)
    assert stored is not None and stored.status == "complete"
    assert coordinator.uploads_idle()
    target.target_transfer_archive_path(data_dir, request.request_id).unlink()
    with pytest.raises(target.TargetTransferUploadError, match="unavailable"):
        coordinator.complete(
            request.request_id,
            lease_boundary_sha256=plan.lease_boundary_sha256,
        )


class _FakeUploadControl:
    def __init__(
        self,
        plan: ServerControlProjectTransferUploadResult,
    ) -> None:
        self.plan = plan
        self.completions: list[tuple[str, str]] = []
        self.activations: list[tuple[str, str]] = []

    def project_transfer_upload_plan(
        self,
        *,
        request_id: str,
    ) -> ServerControlProjectTransferUploadResult:
        assert request_id == self.plan.request_id
        return self.plan

    def complete_project_transfer_upload(
        self,
        *,
        request_id: str,
        lease_boundary_sha256: str,
    ) -> ServerControlProjectTransferUploadResult:
        self.completions.append((request_id, lease_boundary_sha256))
        return ServerControlProjectTransferUploadResult(
            **self.plan.model_dump(exclude={"state"}),
            state="complete",
        )

    def activate_project_transfer(
        self,
        *,
        request_id: str,
        lease_boundary_sha256: str,
    ) -> ServerControlProjectTransferActivationResult:
        self.activations.append((request_id, lease_boundary_sha256))
        return ServerControlProjectTransferActivationResult(
            instance_id=self.plan.instance_id,
            pid=self.plan.pid,
            data_dir_id=self.plan.data_dir_id,
            space_id=self.plan.space_id,
            target_request_id=self.plan.request_id,
            source_request_id=str(uuid.uuid4()),
            project_id=self.plan.project_id,
            archive_sha256=self.plan.archive_sha256,
            upload_lease_boundary_sha256=self.plan.lease_boundary_sha256,
            archive_manifest_sha256="c" * 64,
            target_manifest_sha256="d" * 64,
            publication_sha256="e" * 64,
            activated_at="2026-08-31T20:00:00+00:00",
        )


def test_cli_owner_streams_stdin_without_opening_sqlite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _data_root(tmp_path)
    payload, digest = _payload()
    plan = ServerControlProjectTransferUploadResult(
        instance_id=str(uuid.uuid4()),
        pid=os.getpid(),
        data_dir_id="d" * 64,
        space_id=str(uuid.uuid4()),
        request_id=REQUEST_ID,
        project_id=str(uuid.uuid4()),
        archive_sha256=digest,
        archive_size_bytes=len(payload),
        lease_boundary_sha256="b" * 64,
        state="active",
    )
    control = _FakeUploadControl(plan)
    monkeypatch.setattr(
        target,
        "AppStore",
        lambda *_args, **_kwargs: pytest.fail("the upload CLI opened SQLite"),
    )
    prepared = target.prepare_transfer_import_command(
        ServerCommandRequest(
            command="server project transfer-import",
            request_id=REQUEST_ID,
        ),
        CallerIdentity(uid=os.geteuid(), username="rcp", host="lab"),
        control=control,
        layout=SimpleNamespace(data_dir=root),
    )
    output = StringIO()
    emitter = ServerEventEmitter(prepared.plan, machine_readable=True, stream=output)

    prepared.execute(emitter, io.BytesIO(payload))
    execution = emitter.finish(failed_exit_code=prepared.failed_exit_code)

    assert execution.exit_code == 0
    assert [step.phase for step in prepared.plan.steps] == [
        "transfer_upload",
        "transfer_activation",
    ]
    assert json.loads(output.getvalue().splitlines()[-1])["step"]["state"] == "succeeded"
    assert control.completions == [(REQUEST_ID, "b" * 64)]
    assert control.activations == [(REQUEST_ID, "b" * 64)]
    assert target.target_transfer_archive_path(root, REQUEST_ID).read_bytes() == payload
