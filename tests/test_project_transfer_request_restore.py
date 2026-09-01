from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rcp.core.transition_models import GraphHeadRef
from rcp.server_ops.restore import detach_restore_database
from rcp.storage import AppStore, ProjectTransferPhase, ProjectTransferRequestRecord
from tests.test_project_transfer_request_storage import (
    _activate_target,
    _archive_bound_pair,
    _complete_target_boundaries,
    _linked_pair,
    _ready_incoming,
)

TARGET_NONTERMINAL_PHASES: tuple[ProjectTransferPhase, ...] = (
    "linked",
    "target_admitted",
    "source_released",
    "archive_bound",
    "target_activated",
    "cleanup_acknowledged",
)


@pytest.fixture
def restored_at() -> datetime:
    return datetime.now(UTC) + timedelta(minutes=1)


def _protected_proof(store: AppStore, request_id: str) -> dict[str, object]:
    with sqlite3.connect(store.path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM project_transfer_proofs WHERE request_id = ?",
            (request_id,),
        ).fetchone()
    assert row is not None
    return dict(row)


def _target_at_phase(
    tmp_path: Path,
    phase: ProjectTransferPhase,
) -> tuple[AppStore, AppStore, ProjectTransferRequestRecord, ProjectTransferRequestRecord]:
    (
        source,
        target,
        source_actor,
        target_actor,
        configuration,
        source_request,
        target_request,
    ) = _linked_pair(tmp_path)
    if phase == "linked":
        return source, target, source_request, target_request

    _ready_incoming(target, target_request.request_id)
    target_request = target.record_target_project_transfer_admission(
        target_request.request_id,
        admitted_by=target_actor,
    )
    if phase == "target_admitted":
        return source, target, source_request, target_request

    assert target_request.target_admission_receipt is not None
    source_request = source.accept_target_project_transfer_admission(
        source_request.request_id,
        receipt=target_request.target_admission_receipt,
    )
    source_request = source.record_source_project_transfer_release(
        source_request.request_id,
        released_by=source_actor,
        revalidated_configuration=configuration,
        source_head=GraphHeadRef(revision=7, transition_id="c" * 64),
    )
    assert source_request.source_release_receipt is not None
    target_request = target.accept_source_project_transfer_release(
        target_request.request_id,
        receipt=source_request.source_release_receipt,
    )
    if phase == "source_released":
        return source, target, source_request, target_request

    fenced_head = GraphHeadRef(revision=8, transition_id="d" * 64)
    source_request = source.mark_source_project_transfer_fenced(
        source_request.request_id,
        source_head=fenced_head,
    )
    source.expose_project_transfer_proof(source_request.request_id)
    archive_sha256 = hashlib.sha256(b"one sealed transfer archive").hexdigest()
    source_request = source.bind_project_transfer_archive(
        source_request.request_id,
        archive_sha256=archive_sha256,
        archive_size_bytes=27,
    )
    target_request = target.bind_project_transfer_archive(
        target_request.request_id,
        archive_sha256=archive_sha256,
        archive_size_bytes=27,
        source_fence_head=fenced_head,
    )
    if phase == "archive_bound":
        return source, target, source_request, target_request

    _complete_target_boundaries(target, target_request.request_id)
    _activation, _project_record = _activate_target(target, target_request.request_id)
    target_request = target.project_transfer_request(target_request.request_id)
    assert target_request is not None
    if phase == "target_activated":
        return source, target, source_request, target_request

    target.expose_project_transfer_proof(target_request.request_id)
    target_request = target.acknowledge_project_transfer_proof(
        target_request.request_id,
        acknowledgement_sha256=hashlib.sha256(b"source accepted target proof").hexdigest(),
    )
    target_request = target.acknowledge_project_transfer_cleanup(target_request.request_id)
    assert phase == "cleanup_acknowledged"
    return source, target, source_request, target_request


@pytest.mark.parametrize("phase", TARGET_NONTERMINAL_PHASES)
def test_restore_freezes_each_nonterminal_target_phase_without_moving_its_boundary(
    tmp_path: Path,
    phase: ProjectTransferPhase,
    restored_at: datetime,
) -> None:
    source, target, source_request, target_request = _target_at_phase(tmp_path, phase)
    source_before = source.project_transfer_request(source_request.request_id)
    target_before = target_request.model_dump(
        mode="json",
        exclude={"phase", "restore_resume_phase", "restore_diagnostic", "revision", "updated_at"},
    )
    proof_before = _protected_proof(target, target_request.request_id)

    detach_restore_database(
        target.path,
        confirmed_by="root@lab uid=0",
        detached_at=restored_at,
    )

    restored_target = target.project_transfer_request(target_request.request_id)
    assert restored_target is not None
    assert restored_target.phase == "operator_action_needed"
    assert restored_target.restore_resume_phase == phase
    assert "replacement-server archive" in restored_target.restore_diagnostic
    assert restored_target.revision == target_request.revision + 1
    assert restored_target.updated_at == restored_at.isoformat()
    assert (
        restored_target.model_dump(
            mode="json",
            exclude={
                "phase",
                "restore_resume_phase",
                "restore_diagnostic",
                "revision",
                "updated_at",
            },
        )
        == target_before
    )
    assert _protected_proof(target, target_request.request_id) == proof_before
    assert source.project_transfer_request(source_request.request_id) == source_before

    with pytest.raises(ValueError, match="not exposed"):
        target.expose_project_transfer_proof(target_request.request_id)

    detach_restore_database(
        target.path,
        confirmed_by="root@lab uid=0",
        detached_at=restored_at,
    )
    assert target.project_transfer_request(target_request.request_id) == restored_target
    assert _protected_proof(target, target_request.request_id) == proof_before


def test_restore_keeps_completed_target_and_fenced_source_records_unchanged(
    tmp_path: Path,
    restored_at: datetime,
) -> None:
    source, target, source_request, target_request = _target_at_phase(
        tmp_path,
        "cleanup_acknowledged",
    )
    assert source_request.phase == "archive_bound"
    acknowledgment = target_request.proof_acknowledgement_sha256
    assert acknowledgment is not None
    target_request = target.consume_project_transfer_proof(
        target_request.request_id,
        acknowledgement_sha256=acknowledgment,
    )
    target_request = target.complete_project_transfer_request(target_request.request_id)
    source_before = source.project_transfer_request(source_request.request_id)
    target_before = target.project_transfer_request(target_request.request_id)
    proof_before = _protected_proof(target, target_request.request_id)

    detach_restore_database(
        target.path,
        confirmed_by="root@lab uid=0",
        detached_at=restored_at,
    )

    assert target.project_transfer_request(target_request.request_id) == target_before
    assert _protected_proof(target, target_request.request_id) == proof_before
    assert source.project_transfer_request(source_request.request_id) == source_before


@pytest.mark.parametrize("complete", [False, True])
def test_restore_invalidates_target_upload_and_refuses_reuse(
    tmp_path: Path,
    complete: bool,
    restored_at: datetime,
) -> None:
    _source, target, _source_request, target_request = _archive_bound_pair(tmp_path)
    leased = target.begin_target_project_transfer_upload(target_request.request_id)
    if complete:
        target.complete_target_project_transfer_upload(
            target_request.request_id,
            lease_boundary_sha256=leased.lease_boundary_sha256,
        )

    detach_restore_database(
        target.path,
        confirmed_by="root@lab uid=0",
        detached_at=restored_at,
    )

    invalidated = target.target_project_transfer_upload(target_request.request_id)
    assert invalidated is not None
    assert invalidated.status == "invalidated"
    assert invalidated.lease_boundary_sha256 == leased.lease_boundary_sha256
    with pytest.raises(ValueError, match="restore re-entry"):
        target.begin_target_project_transfer_upload(target_request.request_id)
    with pytest.raises(ValueError, match="restore re-entry"):
        target.complete_target_project_transfer_upload(
            target_request.request_id,
            lease_boundary_sha256=leased.lease_boundary_sha256,
        )


def test_restore_reentry_revalidates_and_issues_only_a_fresh_upload_lease(
    tmp_path: Path,
    restored_at: datetime,
) -> None:
    _source, target, _source_request, target_request = _archive_bound_pair(tmp_path)
    original = target.begin_target_project_transfer_upload(target_request.request_id)
    proof_before = _protected_proof(target, target_request.request_id)
    assert target_request.target_admission_receipt is not None
    confirmer = target_request.target_admission_receipt.admitted_by
    detach_restore_database(
        target.path,
        confirmed_by="root@lab uid=0",
        detached_at=restored_at,
    )
    restored = target.project_transfer_request(target_request.request_id)
    assert restored is not None
    provisioning = _ready_incoming(target, target_request.request_id)
    assert provisioning.final_review_digest is not None
    invalidated = target.target_project_transfer_upload(target_request.request_id)
    assert invalidated is not None
    assert invalidated.status == "invalidated"

    resumed, replacement = target.reenter_restored_target_project_transfer(
        restored.request_id,
        expected_restored_revision=restored.revision,
        expected_resume_phase="archive_bound",
        expected_final_review_digest=provisioning.final_review_digest,
        confirmed_by=confirmer,
    )

    assert resumed.phase == "archive_bound"
    assert resumed.restore_resume_phase is None
    assert resumed.restore_diagnostic is None
    assert resumed.revision == restored.revision + 1
    assert replacement.status == "active"
    assert replacement.lease_boundary_sha256 != original.lease_boundary_sha256
    assert replacement.archive_sha256 == original.archive_sha256
    assert replacement.archive_size_bytes == original.archive_size_bytes
    assert _protected_proof(target, target_request.request_id) == proof_before
    assert target.target_project_transfer_activation(target_request.request_id) is None
    with sqlite3.connect(target.path) as connection:
        row = connection.execute(
            """
            SELECT receipt_json FROM project_transfer_restore_reentries
            WHERE target_request_id = ? AND restored_revision = ?
            """,
            (target_request.request_id, restored.revision),
        ).fetchone()
    assert row is not None
    assert '"confirmed_by"' in row[0]
    assert "secret" not in row[0]
    with pytest.raises(ValueError, match="not exposed"):
        target.expose_project_transfer_proof(target_request.request_id)
    assert target.reenter_restored_target_project_transfer(
        restored.request_id,
        expected_restored_revision=restored.revision,
        expected_resume_phase="archive_bound",
        expected_final_review_digest=provisioning.final_review_digest,
        confirmed_by=confirmer,
    ) == (resumed, replacement)


def test_restore_reentry_guards_leave_the_invalidated_boundary_unchanged(
    tmp_path: Path,
    restored_at: datetime,
) -> None:
    _source, target, _source_request, target_request = _archive_bound_pair(tmp_path)
    target.begin_target_project_transfer_upload(target_request.request_id)
    assert target_request.target_admission_receipt is not None
    confirmer = target_request.target_admission_receipt.admitted_by
    detach_restore_database(
        target.path,
        confirmed_by="root@lab uid=0",
        detached_at=restored_at,
    )
    restored = target.project_transfer_request(target_request.request_id)
    assert restored is not None
    provisioning = _ready_incoming(target, target_request.request_id)
    assert provisioning.final_review_digest is not None
    before_upload = target.target_project_transfer_upload(target_request.request_id)
    assert before_upload is not None

    cases = (
        {
            "expected_restored_revision": restored.revision + 1,
            "expected_resume_phase": "archive_bound",
            "expected_final_review_digest": provisioning.final_review_digest,
        },
        {
            "expected_restored_revision": restored.revision,
            "expected_resume_phase": "source_released",
            "expected_final_review_digest": provisioning.final_review_digest,
        },
        {
            "expected_restored_revision": restored.revision,
            "expected_resume_phase": "archive_bound",
            "expected_final_review_digest": "0" * 64,
        },
    )
    for values in cases:
        with pytest.raises(ValueError):
            target.reenter_restored_target_project_transfer(
                restored.request_id,
                confirmed_by=confirmer,
                **values,
            )
        assert target.project_transfer_request(restored.request_id) == restored
        assert target.target_project_transfer_upload(restored.request_id) == before_upload

    with sqlite3.connect(target.path) as connection:
        connection.execute(
            "UPDATE space_users SET removal_started_at = ? WHERE user_id = ?",
            (target.now(), confirmer.user_id),
        )
    with pytest.raises(ValueError, match="not current"):
        target.reenter_restored_target_project_transfer(
            restored.request_id,
            expected_restored_revision=restored.revision,
            expected_resume_phase="archive_bound",
            expected_final_review_digest=provisioning.final_review_digest,
            confirmed_by=confirmer,
        )
    assert target.project_transfer_request(restored.request_id) == restored
    assert target.target_project_transfer_upload(restored.request_id) == before_upload
