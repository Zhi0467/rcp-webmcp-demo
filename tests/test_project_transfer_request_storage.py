from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from pathlib import Path

import pytest

from rcp.core.models import AuthorizedHuman
from rcp.core.transition_models import GraphHeadRef
from rcp.server_ops.github import parse_github_repository_ref
from rcp.server_ops.layout import DEFAULT_SERVER_LAYOUT
from rcp.storage import (
    AgentTaskRecord,
    AppStore,
    ProjectProvisioningGitCheckRecord,
    ProjectProvisioningMachineIntent,
    ProjectProvisioningProviderCheckRecord,
    ProjectProvisioningProviderIntent,
    ProjectProvisioningRepositoryIntent,
    ProjectRecord,
    ProjectTransferActivationReceipt,
    ProjectTransferRepositorySource,
    ProjectTransferSourceConfiguration,
    ProjectTransferUploadCompleteReceipt,
)
from rcp.storage.provisioning import project_transfer_source_configuration_sha256


def _actor(store: AppStore, name: str) -> AuthorizedHuman:
    if store.space_kind == "personal":
        owner = store.local_owner
        assert owner is not None
        member = store.rename_space_user(owner.user_id, name)
    else:
        member = store.preprovision_team_member(name)
    return AuthorizedHuman(
        space_id=store.space_id,
        user_id=member.user_id,
        display_name=name,
    )


def _source_configuration(**changes: object) -> ProjectTransferSourceConfiguration:
    values: dict[str, object] = {
        "source_rcp_version": "0.1.0.dev0+main",
        "source_schema_generation": 1,
        "supported_archive_codecs": ("rcp-transfer-v1", "rcp-transfer-v2"),
        "machine_aliases": ("laptop",),
        "repositories": (
            ProjectTransferRepositorySource(
                alias="paper",
                repository=parse_github_repository_ref("git@github.com:OpenAI/RCP.git"),
                machine_alias="laptop",
            ),
        ),
        "state_repository": "paper",
        "project_truth_scope": ("paper",),
        "default_run_truth_scope": ("paper",),
        "source_manifest_sha256": "a" * 64,
    }
    values.update(changes)
    return ProjectTransferSourceConfiguration.model_validate(values)


def _project(store: AppStore, project_id: str) -> ProjectRecord:
    return store.upsert_project(
        ProjectRecord(
            project_id=project_id,
            home_space_id=store.space_id,
            locator=f"/tmp/{project_id}/research.yaml",
            name="Transfer project",
            state_location=f"/tmp/{project_id}/.research",
            state_remote=False,
            added_at=store.now(),
        )
    )


def _machine() -> ProjectProvisioningMachineIntent:
    return ProjectProvisioningMachineIntent(
        alias="server",
        location="local",
        os_account="rcp",
        central_root=str(DEFAULT_SERVER_LAYOUT.projects_root),
    )


def _repository() -> ProjectProvisioningRepositoryIntent:
    return ProjectProvisioningRepositoryIntent(
        alias="paper",
        repository=parse_github_repository_ref("https://github.com/openai/rcp.git"),
        machine_alias="server",
    )


def _provider() -> ProjectProvisioningProviderIntent:
    return ProjectProvisioningProviderIntent(
        profile="seed",
        provider="codex",
        runtime_id="codex:exec",
        model="gpt-5.6-luna",
        reasoning="medium",
        machine_alias="server",
    )


def _incoming_request(
    target: AppStore,
    target_actor: AuthorizedHuman,
    project_id: str,
):
    return target.create_project_provisioning_request(
        kind="incoming_transfer",
        authorized_by=target_actor,
        machines=[_machine()],
        repositories=[_repository()],
        provider_checks=[_provider()],
        source_project_id=project_id,
        name="Transfer project",
        state_repository="paper",
        project_truth_scope=["paper"],
        default_run_truth_scope=["paper"],
    )


def _ready_incoming(target: AppStore, request_id: str):
    request = target.project_provisioning_request(request_id)
    assert request is not None
    assert request.status in {"waiting_for_server_setup", "operator_action_needed"}
    restored = request.status == "operator_action_needed"
    running = target.transition_project_provisioning_request(
        request_id,
        receipt_id=(f"restore-setup-{request.revision}" if restored else "setup-started"),
        phase="restore_reentry" if restored else "setup_start",
        expected_revision=request.revision,
        expected_status=request.status,
        to_status="setup_in_progress",
        machines=request.machines,
        repositories=request.repositories,
        provider_checks=request.provider_checks,
    )
    checked_at = target.now()
    machines = [
        running.machines[0].model_copy(
            update={"resolved_central_root": running.machines[0].central_root}
        )
    ]
    repositories = [
        running.repositories[0].model_copy(
            update={
                "resolved_path": running.repositories[0].intended_path,
                "checkout_disposition": "request_created",
                "git_check": ProjectProvisioningGitCheckRecord(
                    status="ready",
                    commit="b" * 40,
                    write_verified=True,
                    deploy_key_label=(f"rcp:{target.space_id}:{request.proposed_project_id}:paper"),
                    public_key_fingerprint="SHA256:" + ("B" * 43),
                    checked_at=checked_at,
                ),
            }
        )
    ]
    providers = [
        ProjectProvisioningProviderCheckRecord(
            **_provider().model_dump(mode="json"),
            status="ready",
            checked_at=checked_at,
        )
    ]
    return target.transition_project_provisioning_request(
        request_id,
        receipt_id=(
            f"restore-preparation-ready-{running.revision}" if restored else "preparation-ready"
        ),
        phase="final_review",
        expected_revision=running.revision,
        expected_status="setup_in_progress",
        to_status="ready_for_review",
        machines=machines,
        repositories=repositories,
        provider_checks=providers,
    )


def _linked_pair(tmp_path: Path):
    source = AppStore(tmp_path / "personal" / "rcp.sqlite3", space_kind="personal")
    target = AppStore(tmp_path / "team" / "rcp.sqlite3", space_kind="team")
    source_actor = _actor(source, "Z")
    target_actor = _actor(target, "Alice")
    project_id = str(uuid.uuid4())
    _project(source, project_id)
    configuration = _source_configuration()
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
        source_project_id=source_request.project_id,
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
    return (
        source,
        target,
        source_actor,
        target_actor,
        configuration,
        source_request,
        target_request,
    )


def _released_pair(tmp_path: Path):
    (
        source,
        target,
        source_actor,
        target_actor,
        configuration,
        source_request,
        target_request,
    ) = _linked_pair(tmp_path)
    _ready_incoming(target, target_request.request_id)
    target_request = target.record_target_project_transfer_admission(
        target_request.request_id,
        admitted_by=target_actor,
    )
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
    return source, target, source_request, target_request


def _archive_bound_pair(tmp_path: Path):
    source, target, source_request, target_request = _released_pair(tmp_path)
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
    return source, target, source_request, target_request


def _activation_project(target: AppStore, project_id: str) -> ProjectRecord:
    return ProjectRecord(
        project_id=project_id,
        home_space_id=target.space_id,
        locator=f"/srv/rcp/projects/{project_id}/paper/.research/manifest.toml",
        name="Transfer project",
        state_location=f"/srv/rcp/projects/{project_id}/paper/.research",
        state_remote=False,
        added_at=target.now(),
        revision=8,
    )


def _complete_target_boundaries(target: AppStore, request_id: str) -> None:
    upload = target.begin_target_project_transfer_upload(request_id)
    target.complete_target_project_transfer_upload(
        request_id,
        lease_boundary_sha256=upload.lease_boundary_sha256,
    )
    target_request = target.project_transfer_request(request_id)
    assert target_request is not None
    now = target.now()
    with sqlite3.connect(target.path) as connection:
        connection.execute(
            """
            INSERT INTO project_transfer_imports (
                request_id, project_id, archive_manifest_sha256,
                target_manifest_sha256, operational_payload_sha256, status,
                event_id_map_json, receipt_id_map_json, publication_sha256,
                created_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, 'complete', '{}', '{}', ?, ?, ?)
            """,
            (
                request_id,
                target_request.project_id,
                "1" * 64,
                "2" * 64,
                "3" * 64,
                "4" * 64,
                now,
                now,
            ),
        )


def _activate_target(
    target: AppStore,
    target_request_id: str,
) -> tuple[ProjectTransferActivationReceipt, ProjectRecord]:
    provisioning = target.project_provisioning_request(target_request_id)
    assert provisioning is not None
    assert provisioning.final_review_digest is not None
    project = _activation_project(target, provisioning.proposed_project_id)
    receipt = target.activate_target_project_transfer(
        target_request_id,
        project=project,
        expected_provisioning_revision=provisioning.revision,
        expected_final_review_digest=provisioning.final_review_digest,
    )
    return receipt, project


def test_target_upload_is_bound_to_one_archive_and_replays_its_receipt(
    tmp_path: Path,
) -> None:
    source, target, source_request, target_request = _archive_bound_pair(tmp_path)

    with pytest.raises(ValueError, match="only a target"):
        source.begin_target_project_transfer_upload(source_request.request_id)

    leased = target.begin_target_project_transfer_upload(target_request.request_id)
    assert leased.status == "active"
    assert len(leased.lease_boundary_sha256) == 64
    assert leased.archive_sha256 == target_request.archive_sha256
    assert leased.archive_size_bytes == target_request.archive_size_bytes
    assert target.begin_target_project_transfer_upload(target_request.request_id) == leased
    assert target.target_project_transfer_upload(target_request.request_id) == leased
    assert target.target_project_transfer_uploads() == [leased]

    completed = target.complete_target_project_transfer_upload(
        target_request.request_id,
        lease_boundary_sha256=leased.lease_boundary_sha256,
    )
    assert completed.status == "complete"
    assert isinstance(completed.receipt, ProjectTransferUploadCompleteReceipt)
    assert completed.receipt.lease_boundary_sha256 == leased.lease_boundary_sha256
    assert (
        target.complete_target_project_transfer_upload(
            target_request.request_id,
            lease_boundary_sha256=leased.lease_boundary_sha256,
        )
        == completed
    )
    with pytest.raises(ValueError, match="another lease boundary"):
        target.complete_target_project_transfer_upload(
            target_request.request_id,
            lease_boundary_sha256="0" * 64,
        )


def test_target_upload_requires_the_archive_bound_phase(tmp_path: Path) -> None:
    _source, target, _source_request, target_request = _released_pair(tmp_path)

    with pytest.raises(ValueError, match="archive-bound"):
        target.begin_target_project_transfer_upload(target_request.request_id)


def test_existing_upload_table_is_migrated_to_retain_consumed_receipts() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(
            """
            CREATE TABLE project_transfer_uploads (
                request_id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                archive_sha256 TEXT NOT NULL,
                archive_size_bytes INTEGER NOT NULL CHECK(archive_size_bytes >= 1),
                lease_boundary_sha256 TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('active', 'complete', 'invalidated')),
                receipt_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                invalidated_at TEXT,
                CHECK(
                    (status = 'active' AND receipt_json IS NULL AND invalidated_at IS NULL)
                    OR (status = 'complete' AND receipt_json IS NOT NULL
                        AND invalidated_at IS NULL)
                    OR (status = 'invalidated' AND receipt_json IS NULL
                        AND invalidated_at IS NOT NULL)
                )
            );
            INSERT INTO project_transfer_uploads VALUES (
                'request', 'project', 'archive', 1, 'lease', 'complete',
                '{}', 'created', 'updated', NULL
            );
            """
        )

        AppStore._allow_consumed_project_transfer_uploads(connection)
        connection.execute(
            "UPDATE project_transfer_uploads SET status = 'consumed' WHERE request_id = 'request'"
        )

        row = connection.execute(
            "SELECT status, receipt_json FROM project_transfer_uploads WHERE request_id = 'request'"
        ).fetchone()
        assert row == ("consumed", "{}")
    finally:
        connection.close()


def test_target_activation_compound_commits_and_replays_its_exact_receipt(
    tmp_path: Path,
) -> None:
    _source, target, _source_request, target_request = _archive_bound_pair(tmp_path)
    _complete_target_boundaries(target, target_request.request_id)
    provisioning = target.project_provisioning_request(target_request.request_id)
    assert provisioning is not None
    assert provisioning.status == "ready_for_review"
    assert provisioning.final_review_digest is not None
    project = _activation_project(target, target_request.project_id)

    assert target.project(project.project_id) is None
    assert target.project_members(project.project_id) == []
    receipt = target.activate_target_project_transfer(
        target_request.request_id,
        project=project,
        expected_provisioning_revision=provisioning.revision,
        expected_final_review_digest=provisioning.final_review_digest,
    )

    assert receipt.source_request_id == target_request.linked_request_id
    assert receipt.archive_sha256 == target_request.archive_sha256
    assert receipt.source_fence_head == target_request.source_fence_head
    assert receipt.archive_manifest_sha256 == "1" * 64
    assert receipt.target_manifest_sha256 == "2" * 64
    assert receipt.operational_payload_sha256 == "3" * 64
    assert receipt.publication_sha256 == "4" * 64
    assert receipt.admitted_by == target_request.target_admission_receipt.admitted_by
    assert receipt.registered_project.project_id == project.project_id
    assert receipt.registered_project.home_space_id == target.space_id
    assert receipt.registered_project.registered_at == project.added_at
    assert receipt.first_member.user_id == receipt.admitted_by.user_id
    assert receipt.first_member.seated_by == receipt.admitted_by.user_id
    assert "secret" not in receipt.model_dump(mode="json")

    stored_project = target.project(project.project_id)
    assert stored_project is not None
    assert stored_project.home_space_id == target.space_id
    assert target.project_members(project.project_id) == [receipt.first_member]
    completed_provisioning = target.project_provisioning_request(target_request.request_id)
    assert completed_provisioning is not None
    assert completed_provisioning.status == "completed"
    assert completed_provisioning.final_review_digest == provisioning.final_review_digest
    assert completed_provisioning.revision == provisioning.revision + 1
    assert target.project_transfer_request(target_request.request_id).phase == "target_activated"
    assert target.target_project_transfer_activation(target_request.request_id) == receipt
    consumed = target.target_project_transfer_upload(target_request.request_id)
    assert consumed is not None
    assert consumed.status == "consumed"
    assert target.begin_target_project_transfer_upload(target_request.request_id) == consumed
    assert (
        target.complete_target_project_transfer_upload(
            target_request.request_id,
            lease_boundary_sha256=consumed.lease_boundary_sha256,
        )
        == consumed
    )

    assert (
        target.activate_target_project_transfer(
            target_request.request_id,
            project=project,
            expected_provisioning_revision=provisioning.revision,
            expected_final_review_digest=provisioning.final_review_digest,
        )
        == receipt
    )
    with pytest.raises(ValueError, match="retry does not match"):
        target.activate_target_project_transfer(
            target_request.request_id,
            project=project.model_copy(update={"name": "Another project"}),
            expected_provisioning_revision=provisioning.revision,
            expected_final_review_digest=provisioning.final_review_digest,
        )


def test_target_activation_requires_complete_machine_boundaries_atomically(
    tmp_path: Path,
) -> None:
    _source, target, _source_request, target_request = _archive_bound_pair(tmp_path)
    upload = target.begin_target_project_transfer_upload(target_request.request_id)
    target.complete_target_project_transfer_upload(
        target_request.request_id,
        lease_boundary_sha256=upload.lease_boundary_sha256,
    )
    provisioning = target.project_provisioning_request(target_request.request_id)
    assert provisioning is not None
    assert provisioning.final_review_digest is not None
    project = _activation_project(target, target_request.project_id)

    with pytest.raises(ValueError, match="completed import"):
        target.activate_target_project_transfer(
            target_request.request_id,
            project=project,
            expected_provisioning_revision=provisioning.revision,
            expected_final_review_digest=provisioning.final_review_digest,
        )

    assert target.project(project.project_id) is None
    assert target.project_members(project.project_id) == []
    assert target.target_project_transfer_activation(target_request.request_id) is None
    assert target.project_transfer_request(target_request.request_id).phase == "archive_bound"
    unchanged = target.project_provisioning_request(target_request.request_id)
    assert unchanged is not None
    assert unchanged.status == "ready_for_review"
    retained_upload = target.target_project_transfer_upload(target_request.request_id)
    assert retained_upload is not None
    assert retained_upload.status == "complete"


def test_corrupt_target_activation_receipt_fails_loudly(tmp_path: Path) -> None:
    _source, target, _source_request, target_request = _archive_bound_pair(tmp_path)
    _complete_target_boundaries(target, target_request.request_id)
    receipt, _project_record = _activate_target(target, target_request.request_id)
    corrupted = receipt.model_dump(mode="json")
    corrupted["publication_sha256"] = "9" * 64
    with sqlite3.connect(target.path) as connection:
        connection.execute(
            """
            UPDATE project_transfer_activations SET receipt_json = ?
            WHERE target_request_id = ?
            """,
            (
                json.dumps(corrupted, sort_keys=True, separators=(",", ":")),
                target_request.request_id,
            ),
        )

    with pytest.raises(RuntimeError, match="import receipt"):
        target.target_project_transfer_activation(target_request.request_id)


def test_source_release_atomically_fences_new_root_task_admission(tmp_path: Path) -> None:
    source, _target, source_request, _target_request = _released_pair(tmp_path)
    now = source.now()
    task = AgentTaskRecord(
        operation_id=str(uuid.uuid4()),
        project_id=source_request.project_id,
        kind="refresh",
        status="queued",
        request={},
        created_at=now,
        updated_at=now,
        status_message="Waiting to refresh.",
    )

    with pytest.raises(ValueError, match="moving to its admitted team space"):
        source.create_agent_task(task)
    assert source.agent_task(task.operation_id) is None


def test_linked_requests_keep_independent_raw_proofs_out_of_public_state(tmp_path: Path) -> None:
    (
        source,
        target,
        source_actor,
        target_actor,
        configuration,
        source_request,
        target_request,
    ) = _linked_pair(tmp_path)

    assert source_actor.user_id != target_actor.user_id
    assert source_request.phase == target_request.phase == "linked"
    assert source_request.linked_request_id == target_request.request_id
    assert target_request.linked_request_id == source_request.request_id
    assert source_request.source_configuration == configuration
    assert source_request.source_configuration_sha256 == (
        project_transfer_source_configuration_sha256(configuration)
    )
    assert source_request.source_release_proof_sha256 == (
        target_request.source_release_proof_sha256
    )
    assert source_request.target_activation_proof_sha256 == (
        target_request.target_activation_proof_sha256
    )
    assert source_request.source_release_proof_sha256 != (
        target_request.target_activation_proof_sha256
    )

    for store, request in ((source, source_request), (target, target_request)):
        with sqlite3.connect(store.path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT record_json FROM project_transfer_requests WHERE request_id = ?",
                (request.request_id,),
            ).fetchone()
            proof = connection.execute(
                "SELECT secret, commitment_sha256 FROM project_transfer_proofs "
                "WHERE request_id = ?",
                (request.request_id,),
            ).fetchone()
        assert row is not None and proof is not None
        public = json.loads(row["record_json"])
        assert "secret" not in public
        assert len(proof["secret"]) == 32
        assert hashlib.sha256(proof["secret"]).hexdigest() == proof["commitment_sha256"]
        with pytest.raises(ValueError, match="not exposed"):
            store.expose_project_transfer_proof(request.request_id)

    assert AppStore(source.path).project_transfer_request(source_request.request_id) == (
        source_request
    )
    assert AppStore(target.path, space_kind="team").project_transfer_requests(side="target") == [
        target_request
    ]


def test_link_creation_is_exactly_idempotent_after_the_request_advances(tmp_path: Path) -> None:
    (
        source,
        target,
        source_actor,
        target_actor,
        configuration,
        source_request,
        target_request,
    ) = _linked_pair(tmp_path)

    repeated_source = source.create_source_project_transfer_request(
        request_id=source_request.request_id,
        project_id=source_request.project_id,
        target_space_id=target.space_id,
        initiated_by=source_actor,
        source_configuration=configuration,
    )
    repeated_target = target.create_target_project_transfer_request(
        provisioning_request_id=target_request.request_id,
        source_request_id=source_request.request_id,
        source_project_id=source_request.project_id,
        source_space_id=source.space_id,
        initiated_by=target_actor,
        source_configuration=configuration,
        source_configuration_sha256=source_request.source_configuration_sha256,
        source_release_proof_sha256=source_request.source_release_proof_sha256,
        accepted_schema_generation=1,
        accepted_archive_codec="rcp-transfer-v1",
    )
    assert target_request.link_receipt is not None
    repeated_link = source.link_source_project_transfer_request(
        source_request.request_id,
        receipt=target_request.link_receipt,
    )

    assert repeated_source == source_request
    assert repeated_target == target_request
    assert repeated_link == source_request
    assert len(source.project_transfer_requests()) == 1
    assert len(target.project_transfer_requests()) == 1
    with pytest.raises(ValueError, match="does not match"):
        source.link_source_project_transfer_request(
            source_request.request_id,
            receipt=target_request.link_receipt.model_copy(
                update={"target_request_id": str(uuid.uuid4())}
            ),
        )


def test_no_common_codec_or_stale_source_identity_fails_before_linking(tmp_path: Path) -> None:
    source = AppStore(tmp_path / "personal" / "rcp.sqlite3", space_kind="personal")
    target = AppStore(tmp_path / "team" / "rcp.sqlite3", space_kind="team")
    source_actor = _actor(source, "Z")
    target_actor = _actor(target, "Alice")
    project_id = str(uuid.uuid4())
    _project(source, project_id)
    configuration = _source_configuration()
    source_request = source.create_source_project_transfer_request(
        project_id=project_id,
        target_space_id=target.space_id,
        initiated_by=source_actor,
        source_configuration=configuration,
    )
    incoming = _incoming_request(target, target_actor, project_id)

    with pytest.raises(ValueError, match="did not offer"):
        target.create_target_project_transfer_request(
            provisioning_request_id=incoming.request_id,
            source_request_id=source_request.request_id,
            source_project_id=source_request.project_id,
            source_space_id=source.space_id,
            initiated_by=target_actor,
            source_configuration=configuration,
            source_configuration_sha256=source_request.source_configuration_sha256,
            source_release_proof_sha256=source_request.source_release_proof_sha256,
            accepted_schema_generation=1,
            accepted_archive_codec="rcp-transfer-v9",
        )
    assert target.project_transfer_requests() == []

    with pytest.raises(ValueError, match="does not match its incoming"):
        target.create_target_project_transfer_request(
            provisioning_request_id=incoming.request_id,
            source_request_id=source_request.request_id,
            source_project_id=str(uuid.uuid4()),
            source_space_id=source.space_id,
            initiated_by=target_actor,
            source_configuration=configuration,
            source_configuration_sha256=source_request.source_configuration_sha256,
            source_release_proof_sha256=source_request.source_release_proof_sha256,
            accepted_schema_generation=1,
            accepted_archive_codec="rcp-transfer-v1",
        )

    mismatched_repository = ProjectProvisioningRepositoryIntent(
        alias="paper",
        repository=parse_github_repository_ref("https://github.com/openai/other.git"),
        machine_alias="server",
    )
    other_target = AppStore(tmp_path / "other-team" / "rcp.sqlite3", space_kind="team")
    other_target_actor = _actor(other_target, "Alice")
    mismatched_incoming = other_target.create_project_provisioning_request(
        kind="incoming_transfer",
        authorized_by=other_target_actor,
        machines=[_machine()],
        repositories=[mismatched_repository],
        provider_checks=[_provider()],
        source_project_id=project_id,
        name="Transfer project",
        state_repository="paper",
        project_truth_scope=["paper"],
        default_run_truth_scope=["paper"],
    )
    with pytest.raises(ValueError, match="repositories do not match"):
        other_target.create_target_project_transfer_request(
            provisioning_request_id=mismatched_incoming.request_id,
            source_request_id=source_request.request_id,
            source_project_id=source_request.project_id,
            source_space_id=source.space_id,
            initiated_by=other_target_actor,
            source_configuration=configuration,
            source_configuration_sha256=source_request.source_configuration_sha256,
            source_release_proof_sha256=source_request.source_release_proof_sha256,
            accepted_schema_generation=1,
            accepted_archive_codec="rcp-transfer-v1",
        )

    stale_id = str(uuid.uuid4())
    with pytest.raises(ValueError, match="stale or belongs"):
        source.create_source_project_transfer_request(
            project_id=stale_id,
            target_space_id=target.space_id,
            initiated_by=source_actor,
            source_configuration=configuration,
        )


def test_both_human_receipts_bind_the_exact_review_without_creating_target_authority(
    tmp_path: Path,
) -> None:
    (
        source,
        target,
        source_actor,
        target_actor,
        configuration,
        source_request,
        target_request,
    ) = _linked_pair(tmp_path)
    ready = _ready_incoming(target, target_request.request_id)

    target_request = target.record_target_project_transfer_admission(
        target_request.request_id,
        admitted_by=target_actor,
    )
    assert target_request.phase == "target_admitted"
    assert target_request.target_admission_receipt is not None
    assert target_request.target_admission_receipt.target_preparation_revision == ready.revision
    assert target_request.target_admission_receipt.target_preparation_sha256 == (
        ready.final_review_digest
    )
    assert target.project(target_request.project_id) is None
    assert (
        target.record_target_project_transfer_admission(
            target_request.request_id,
            admitted_by=target_actor,
        )
        == target_request
    )

    forged = target_request.target_admission_receipt.model_copy(
        update={"source_configuration_sha256": "d" * 64}
    )
    with pytest.raises(ValueError, match="does not match"):
        source.accept_target_project_transfer_admission(
            source_request.request_id,
            receipt=forged,
        )
    source_request = source.accept_target_project_transfer_admission(
        source_request.request_id,
        receipt=target_request.target_admission_receipt,
    )
    drifted = configuration.model_copy(update={"source_manifest_sha256": "e" * 64})
    with pytest.raises(ValueError, match="changed after"):
        source.record_source_project_transfer_release(
            source_request.request_id,
            released_by=source_actor,
            revalidated_configuration=drifted,
            source_head=GraphHeadRef(revision=7, transition_id="c" * 64),
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
    assert source_request.phase == target_request.phase == "source_released"
    assert source_request.source_release_receipt.released_by == source_actor
    assert target_request.target_admission_receipt.admitted_by == target_actor
    assert target.project(target_request.project_id) is None


def test_proofs_expose_only_at_their_boundaries_then_consume_to_receipts(
    tmp_path: Path,
) -> None:
    source, target, source_request, target_request = _released_pair(tmp_path)
    assert source_request.source_release_receipt is not None
    fenced_head = GraphHeadRef(revision=8, transition_id="d" * 64)
    source_request = source.mark_source_project_transfer_fenced(
        source_request.request_id,
        source_head=fenced_head,
    )
    source_secret = source.expose_project_transfer_proof(source_request.request_id)
    assert source.expose_project_transfer_proof(source_request.request_id) == source_secret
    assert hashlib.sha256(source_secret).hexdigest() == (source_request.source_release_proof_sha256)
    source_ack = hashlib.sha256(b"target verified source release proof").hexdigest()
    with pytest.raises(ValueError, match="before its boundary"):
        source.acknowledge_project_transfer_proof(
            source_request.request_id,
            acknowledgement_sha256=source_ack,
        )

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
    _complete_target_boundaries(target, target_request.request_id)
    _activation, _project_record = _activate_target(target, target_request.request_id)
    target_request = target.project_transfer_request(target_request.request_id)
    assert target_request is not None
    target_secret = target.expose_project_transfer_proof(target_request.request_id)
    assert target.expose_project_transfer_proof(target_request.request_id) == target_secret
    assert hashlib.sha256(target_secret).hexdigest() == (
        target_request.target_activation_proof_sha256
    )

    target_ack = hashlib.sha256(b"source verified target activation proof").hexdigest()
    source.acknowledge_project_transfer_proof(
        source_request.request_id,
        acknowledgement_sha256=source_ack,
    )
    with pytest.raises(ValueError, match="before cleanup"):
        source.consume_project_transfer_proof(
            source_request.request_id,
            acknowledgement_sha256=source_ack,
        )
    source.acknowledge_project_transfer_cleanup(source_request.request_id)
    source.consume_project_transfer_proof(
        source_request.request_id,
        acknowledgement_sha256=source_ack,
    )
    source_request = source.complete_project_transfer_request(source_request.request_id)

    target.acknowledge_project_transfer_proof(
        target_request.request_id,
        acknowledgement_sha256=target_ack,
    )
    target.acknowledge_project_transfer_cleanup(target_request.request_id)
    target.consume_project_transfer_proof(
        target_request.request_id,
        acknowledgement_sha256=target_ack,
    )
    target_request = target.complete_project_transfer_request(target_request.request_id)

    assert source_request.phase == target_request.phase == "completed"
    assert source_request.proof_state == target_request.proof_state == "consumed"
    for store, request, acknowledgment in (
        (source, source_request, source_ack),
        (target, target_request, target_ack),
    ):
        with sqlite3.connect(store.path) as connection:
            connection.row_factory = sqlite3.Row
            proof = connection.execute(
                "SELECT * FROM project_transfer_proofs WHERE request_id = ?",
                (request.request_id,),
            ).fetchone()
        assert proof is not None
        assert proof["secret"] is None
        assert proof["commitment_sha256"] in {
            request.source_release_proof_sha256,
            request.target_activation_proof_sha256,
        }
        assert proof["acknowledgement_sha256"] == acknowledgment
        with pytest.raises(ValueError, match="already consumed"):
            store.expose_project_transfer_proof(request.request_id)
        assert store.complete_project_transfer_request(request.request_id) == request


def test_archive_and_proof_retries_reject_different_boundaries(tmp_path: Path) -> None:
    source, target, source_request, target_request = _released_pair(tmp_path)
    assert source_request.source_release_receipt is not None
    fenced_head = GraphHeadRef(revision=8, transition_id="d" * 64)
    source.mark_source_project_transfer_fenced(
        source_request.request_id,
        source_head=fenced_head,
    )
    source.expose_project_transfer_proof(source_request.request_id)
    archive = "f" * 64
    bound = source.bind_project_transfer_archive(
        source_request.request_id,
        archive_sha256=archive,
        archive_size_bytes=100,
    )
    assert (
        source.bind_project_transfer_archive(
            source_request.request_id,
            archive_sha256=archive,
            archive_size_bytes=100,
        )
        == bound
    )
    with pytest.raises(ValueError, match="another archive"):
        source.bind_project_transfer_archive(
            source_request.request_id,
            archive_sha256="0" * 64,
            archive_size_bytes=100,
        )
    with pytest.raises(ValueError, match="compound activation receipt"):
        target.mark_target_project_transfer_activated(target_request.request_id)


def test_corrupt_public_or_protected_transfer_state_fails_loudly(tmp_path: Path) -> None:
    source, _target, _source_actor, _target_actor, _config, source_request, _target_request = (
        _linked_pair(tmp_path)
    )
    with sqlite3.connect(source.path) as connection:
        connection.execute(
            "UPDATE project_transfer_proofs SET commitment_sha256 = ? WHERE request_id = ?",
            ("0" * 64, source_request.request_id),
        )
    with pytest.raises(RuntimeError, match="does not match"):
        source.project_transfer_request(source_request.request_id)
