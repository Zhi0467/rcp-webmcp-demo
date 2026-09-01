from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

from rcp.api import create_app
from rcp.project_transfer import capture_project_transfer_source
from rcp.storage import AppStore
from rcp.transfer.source import (
    advance_source_project_transfer,
    complete_source_project_transfer,
    read_transfer_archive,
    source_transfer_export_path,
)

from .helpers import create_named_app
from .test_project_transfer_request_api import _set_origin, _source_project
from .test_project_transfer_request_storage import (
    _activate_target,
    _actor,
    _complete_target_boundaries,
    _incoming_request,
    _ready_incoming,
)


def _released_source(tmp_path: Path):
    source_data = tmp_path / "personal"
    source_app = create_named_app(data_dir=source_data)
    source = source_app.state.background_tasks.store
    source_actor = _actor(source, "Z")
    project_id = _source_project(source_app, source_data)
    service = source_app.state.catalog.open(project_id)
    configuration, source_head = capture_project_transfer_source(service)

    target = AppStore(tmp_path / "team" / "rcp.sqlite3", space_kind="team")
    target_actor = _actor(target, "Alice")
    source_request = source.create_source_project_transfer_request(
        project_id=project_id,
        target_space_id=target.space_id,
        initiated_by=source_actor,
        source_configuration=configuration,
    )
    incoming = _incoming_request(target, target_actor, project_id)
    target_request = target.create_target_project_transfer_request(
        provisioning_request_id=incoming.request_id,
        source_request_id=source_request.request_id,
        source_project_id=project_id,
        source_space_id=source.space_id,
        initiated_by=target_actor,
        source_configuration=configuration,
        source_configuration_sha256=source_request.source_configuration_sha256,
        source_release_proof_sha256=source_request.source_release_proof_sha256,
        accepted_schema_generation=configuration.source_schema_generation,
        accepted_archive_codec="rcp-transfer-v1",
    )
    assert target_request.link_receipt is not None
    source_request = source.link_source_project_transfer_request(
        source_request.request_id,
        receipt=target_request.link_receipt,
    )
    _ready_incoming(target, target_request.request_id)
    target_request = target.record_target_project_transfer_admission(
        target_request.request_id,
        admitted_by=target_actor,
    )
    assert target_request.target_admission_receipt is not None
    source.accept_target_project_transfer_admission(
        source_request.request_id,
        receipt=target_request.target_admission_receipt,
    )
    source_request = source.record_source_project_transfer_release(
        source_request.request_id,
        released_by=source_actor,
        revalidated_configuration=configuration,
        source_head=source_head,
    )
    return (
        source_data,
        source_app,
        source,
        target,
        source_request,
        target_request,
        source_head,
    )


def _target_activation_proof(target, target_request, source_request) -> bytes:
    assert source_request.source_release_receipt is not None
    target_request = target.accept_source_project_transfer_release(
        target_request.request_id,
        receipt=source_request.source_release_receipt,
    )
    target_request = target.bind_project_transfer_archive(
        target_request.request_id,
        archive_sha256=source_request.archive_sha256,
        archive_size_bytes=source_request.archive_size_bytes,
        source_fence_head=source_request.source_fence_head,
    )
    _complete_target_boundaries(target, target_request.request_id)
    _activate_target(target, target_request.request_id)
    return target.expose_project_transfer_proof(target_request.request_id)


def test_source_transfer_reuses_exact_archive_after_publication_before_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_data, source_app, source, _target, request, _target_request, source_head = (
        _released_source(tmp_path)
    )
    original_bind = source.bind_project_transfer_archive

    def interrupt_before_receipt(*args, **kwargs):
        raise RuntimeError("simulated stop before archive receipt")

    monkeypatch.setattr(source, "bind_project_transfer_archive", interrupt_before_receipt)
    with pytest.raises(RuntimeError, match="simulated stop"):
        advance_source_project_transfer(source, source_app.state.catalog, request.request_id)

    archive_path = source_transfer_export_path(source_data, request.request_id)
    first_bytes = archive_path.read_bytes()
    fenced = source.project_transfer_request(request.request_id)
    assert fenced is not None
    assert fenced.phase == "source_fenced"
    assert fenced.source_fence_head is not None
    assert fenced.source_fence_head.revision == source_head.revision + 1

    monkeypatch.setattr(source, "bind_project_transfer_archive", original_bind)
    completed = advance_source_project_transfer(
        source, source_app.state.catalog, request.request_id
    )
    assert completed.phase == "archive_bound"
    assert archive_path.read_bytes() == first_bytes
    assert (
        advance_source_project_transfer(
            source,
            source_app.state.catalog,
            request.request_id,
        )
        == completed
    )
    assert archive_path.read_bytes() == first_bytes


def test_source_transfer_recovers_after_home_patch_before_sqlite_fence(tmp_path: Path) -> None:
    source_data, source_app, source, _target, request, _target_request, source_head = (
        _released_source(tmp_path)
    )
    service = source_app.state.catalog.open(request.project_id)
    assert request.source_release_receipt is not None
    assert request.target_admission_receipt is not None
    service.history.transfer_project_home(
        project_id=request.project_id,
        previous_home_space_id=request.source_space_id,
        new_home_space_id=request.target_space_id,
        source_released_by=request.source_release_receipt.released_by,
        target_admitted_by=request.target_admission_receipt.admitted_by,
    )

    restarted = create_app(data_dir=source_data)
    restarted_store = restarted.state.background_tasks.store
    completed = advance_source_project_transfer(
        restarted_store,
        restarted.state.catalog,
        request.request_id,
    )
    assert completed.phase == "archive_bound"
    assert completed.source_fence_head is not None
    assert completed.source_fence_head.revision == source_head.revision + 1


def test_source_capture_holds_the_canonical_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source_data, source_app, source, _target, request, _target_request, _source_head = (
        _released_source(tmp_path)
    )
    workspace = source_app.state.catalog.open(request.project_id).history.workspace
    original_transaction = workspace.transaction
    original_backup = workspace.backup_source_root
    transaction_depth = 0
    guarded_backups = 0

    @contextmanager
    def tracked_transaction():
        nonlocal transaction_depth
        with original_transaction():
            transaction_depth += 1
            try:
                yield
            finally:
                transaction_depth -= 1

    def guarded_backup(destination: Path) -> Path:
        nonlocal guarded_backups
        assert transaction_depth == 1
        guarded_backups += 1
        return original_backup(destination)

    monkeypatch.setattr(workspace, "transaction", tracked_transaction)
    monkeypatch.setattr(workspace, "backup_source_root", guarded_backup)

    advance_source_project_transfer(source, source_app.state.catalog, request.request_id)

    assert guarded_backups == 2


def test_source_transfer_refuses_configuration_drift_before_home_patch(tmp_path: Path) -> None:
    _source_data, source_app, source, _target, request, _target_request, source_head = (
        _released_source(tmp_path)
    )
    project = source.project(request.project_id)
    assert project is not None
    _set_origin(
        Path(project.state_location).parent,
        "https://github.com/openai/another.git",
    )

    with pytest.raises(ValueError, match="configuration changed after"):
        advance_source_project_transfer(source, source_app.state.catalog, request.request_id)
    retained = source.project_transfer_request(request.request_id)
    assert retained is not None
    assert retained.phase == "source_released"
    assert source_app.state.catalog.open(request.project_id).history.head_ref() == source_head


def test_bound_source_archive_missing_or_corrupt_fails_without_regeneration(tmp_path: Path) -> None:
    source_data, source_app, source, _target, request, _target_request, _source_head = (
        _released_source(tmp_path)
    )
    completed = advance_source_project_transfer(
        source, source_app.state.catalog, request.request_id
    )
    archive_path = source_transfer_export_path(source_data, request.request_id)
    envelope = read_transfer_archive(archive_path).envelope
    assert completed.archive_sha256 == envelope.archive_sha256

    archive_path.write_bytes(b"corrupt")
    archive_path.chmod(0o600)
    with pytest.raises(ValueError, match="missing, malformed, or unreadable"):
        advance_source_project_transfer(source, source_app.state.catalog, request.request_id)
    assert archive_path.read_bytes() == b"corrupt"

    archive_path.unlink()
    with pytest.raises(ValueError, match="sealed transfer archive is missing"):
        advance_source_project_transfer(source, source_app.state.catalog, request.request_id)
    assert not archive_path.exists()


@pytest.mark.parametrize("archive_state", ["missing", "corrupt"])
def test_source_cleanup_refuses_to_retire_without_the_exact_recovery_archive(
    tmp_path: Path,
    archive_state: str,
) -> None:
    source_data, source_app, source, target, request, target_request, _source_head = (
        _released_source(tmp_path)
    )
    completed_export = advance_source_project_transfer(
        source,
        source_app.state.catalog,
        request.request_id,
    )
    proof = _target_activation_proof(target, target_request, completed_export)
    archive_path = source_transfer_export_path(source_data, request.request_id)
    if archive_state == "missing":
        archive_path.unlink()
    else:
        archive_path.write_bytes(b"corrupt")
        archive_path.chmod(0o600)

    with pytest.raises(ValueError):
        complete_source_project_transfer(
            source,
            source_app.state.catalog,
            request.request_id,
            target_activation_proof=proof,
        )

    assert source.project(request.project_id) is not None
    assert source.retired_project(request.project_id) is None
    assert (
        source_app.state.catalog.open(request.project_id).history.project_id == request.project_id
    )


def test_source_cleanup_recovers_after_retirement_and_archive_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_data, source_app, source, target, request, target_request, _source_head = (
        _released_source(tmp_path)
    )
    completed_export = advance_source_project_transfer(
        source,
        source_app.state.catalog,
        request.request_id,
    )
    proof = _target_activation_proof(target, target_request, completed_export)
    archive_path = source_transfer_export_path(source_data, request.request_id)

    original_discard = source_app.state.catalog.discard_retired_transfer_source

    def interrupt_after_retirement(*args, **kwargs):
        raise RuntimeError("simulated stop after source retirement")

    monkeypatch.setattr(
        source_app.state.catalog,
        "discard_retired_transfer_source",
        interrupt_after_retirement,
    )
    with pytest.raises(RuntimeError, match="after source retirement"):
        complete_source_project_transfer(
            source,
            source_app.state.catalog,
            request.request_id,
            target_activation_proof=proof,
        )
    assert source.project(request.project_id) is None
    assert archive_path.exists()

    monkeypatch.setattr(
        source_app.state.catalog,
        "discard_retired_transfer_source",
        original_discard,
    )
    original_complete = source.complete_project_transfer_request

    def interrupt_after_unlink(*args, **kwargs):
        raise RuntimeError("simulated stop after archive unlink")

    monkeypatch.setattr(source, "complete_project_transfer_request", interrupt_after_unlink)
    with pytest.raises(RuntimeError, match="after archive unlink"):
        complete_source_project_transfer(
            source,
            source_app.state.catalog,
            request.request_id,
            target_activation_proof=proof,
        )
    assert not archive_path.exists()

    monkeypatch.setattr(source, "complete_project_transfer_request", original_complete)
    acknowledgment = complete_source_project_transfer(
        source,
        source_app.state.catalog,
        request.request_id,
        target_activation_proof=proof,
    )
    assert (
        complete_source_project_transfer(
            source,
            source_app.state.catalog,
            request.request_id,
            target_activation_proof=proof,
        )
        == acknowledgment
    )
    final = source.project_transfer_request(request.request_id)
    assert final is not None
    assert final.phase == "completed"
    assert final.proof_state == "consumed"
