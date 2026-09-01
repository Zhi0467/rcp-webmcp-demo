from __future__ import annotations

import hashlib
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from rcp.agents import AgentLauncher
from rcp.config import AGENT_EXECUTION_PROFILES
from rcp.core.models import AuthorizedHuman
from rcp.projects import ProjectCatalog
from rcp.providers import configured_runtime_id
from rcp.server_ops.github import parse_github_repository_ref
from rcp.storage import (
    AppStore,
    ProjectProvisioningGitCheckRecord,
    ProjectProvisioningMachineIntent,
    ProjectProvisioningProviderCheckRecord,
    ProjectProvisioningProviderIntent,
    ProjectProvisioningRepositoryIntent,
    ProjectTransferRepositorySource,
    ProjectTransferSourceConfiguration,
)
from rcp.storage import models as storage_models
from rcp.storage.provisioning import project_transfer_source_configuration_sha256
from rcp.transfer import (
    TRANSFER_ARCHIVE_CODEC,
    TransferArchiveActor,
    TransferArchiveAttribution,
    TransferArchiveEntry,
    TransferArchiveEnvelope,
    TransferArchiveManifest,
    TransferGraphHead,
)
from rcp.transfer import importer as transfer_importer
from rcp.transfer.configuration import build_transfer_target_configuration
from rcp.transfer.importer import import_project_transfer
from rcp.transfer.project_files import (
    TRANSFER_OPERATIONAL_RECORDS_PATH,
    capture_project_transfer_files,
    transfer_project_file_payload,
)
from rcp.transfer.source import seal_transfer_archive

from .helpers import authorized_human
from .test_transfer_project_files import _finished_project, _write_canonical_sources


def _entry(root: Path, archive_path: str, group: str, payload: bytes) -> TransferArchiveEntry:
    path = root / archive_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return TransferArchiveEntry(
        archive_path=archive_path,
        group=group,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def _target_actor(store: AppStore) -> AuthorizedHuman:
    member = store.preprovision_team_member("Alice")
    return AuthorizedHuman(
        space_id=store.space_id,
        user_id=member.user_id,
        display_name=member.display_name,
    )


def _source_configuration(service) -> ProjectTransferSourceConfiguration:
    manifest = service.history.manifest
    repositories = tuple(
        ProjectTransferRepositorySource(
            alias=repository.alias,
            repository=parse_github_repository_ref(
                f"git@github.com:Example/{repository.alias}.git"
            ),
            machine_alias=repository.machine,
        )
        for repository in manifest.repositories
    )
    return ProjectTransferSourceConfiguration(
        source_rcp_version="0.1.0.dev0+main",
        source_schema_generation=1,
        supported_archive_codecs=(TRANSFER_ARCHIVE_CODEC,),
        machine_aliases=tuple(sorted(manifest.machine_map)),
        repositories=repositories,
        state_repository=manifest.state.repository,
        project_truth_scope=tuple(manifest.project.truth_scope),
        default_run_truth_scope=tuple(manifest.agent.default_run_truth_scope),
        source_manifest_sha256=hashlib.sha256(manifest.path.read_bytes()).hexdigest(),
    )


def _prepare_target_request(
    target: AppStore,
    *,
    actor: AuthorizedHuman,
    source,
    project_id: str,
    central_root: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        storage_models,
        "DEFAULT_SERVER_LAYOUT",
        SimpleNamespace(service_account="rcp", projects_root=central_root),
    )
    machines = [
        ProjectProvisioningMachineIntent(
            alias=alias,
            location="local",
            os_account="rcp",
            central_root=str(central_root),
        )
        for alias in source.machine_aliases
    ]
    repositories = [
        ProjectProvisioningRepositoryIntent(
            alias=item.alias,
            repository=item.repository,
            machine_alias=item.machine_alias,
        )
        for item in source.repositories
    ]
    provider_checks = [
        ProjectProvisioningProviderIntent(
            profile=profile,
            provider="codex",
            runtime_id="codex:exec",
            model="gpt-test",
            reasoning="medium",
            machine_alias=source.machine_aliases[0],
        )
        for profile in AGENT_EXECUTION_PROFILES
    ]
    request = target.create_project_provisioning_request(
        kind="incoming_transfer",
        authorized_by=actor,
        machines=machines,
        repositories=repositories,
        provider_checks=provider_checks,
        source_project_id=project_id,
        name="Transferred project",
        state_repository=source.state_repository,
        project_truth_scope=list(source.project_truth_scope),
        default_run_truth_scope=list(source.default_run_truth_scope),
    )
    running = target.transition_project_provisioning_request(
        request.request_id,
        receipt_id="setup-started",
        phase="setup_start",
        expected_revision=0,
        expected_status="waiting_for_server_setup",
        to_status="setup_in_progress",
        machines=request.machines,
        repositories=request.repositories,
        provider_checks=request.provider_checks,
    )
    checked_at = target.now()
    ready_machines = [
        item.model_copy(update={"resolved_central_root": item.central_root})
        for item in running.machines
    ]
    ready_repositories = []
    for item in running.repositories:
        assert item.intended_path is not None
        Path(item.intended_path).mkdir(parents=True)
        ready_repositories.append(
            item.model_copy(
                update={
                    "resolved_path": item.intended_path,
                    "checkout_disposition": "request_created",
                    "git_check": ProjectProvisioningGitCheckRecord(
                        status="ready",
                        commit="a" * 40,
                        write_verified=True,
                        deploy_key_label=(f"rcp:{target.space_id}:{project_id}:{item.alias}"),
                        public_key_fingerprint="SHA256:" + ("A" * 43),
                        checked_at=checked_at,
                    ),
                }
            )
        )
    ready_providers = [
        ProjectProvisioningProviderCheckRecord(
            **intent.model_dump(mode="json"),
            status="ready",
            binary_path="/usr/bin/true",
            version="fixture 1.0",
            resolved_runtime_id=configured_runtime_id("codex", "exec"),
            execution_account="rcp",
            checked_at=checked_at,
        )
        for intent in provider_checks
    ]
    return target.transition_project_provisioning_request(
        request.request_id,
        receipt_id="preparation-ready",
        phase="final_review",
        expected_revision=1,
        expected_status="setup_in_progress",
        to_status="ready_for_review",
        machines=ready_machines,
        repositories=ready_repositories,
        provider_checks=ready_providers,
    )


def _archive_fixture(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    seal_archive: bool = False,
):
    service, records, artifact, artifact_name, view, view_name = _finished_project(
        manifest,
        tmp_path / "source",
    )
    _write_canonical_sources(service, records.tasks[0].operation_id)
    source = service.paper.store
    source_actor = authorized_human(source)
    target_data = tmp_path / "target-data"
    target = AppStore(target_data / "rcp.sqlite3", space_kind="team")
    target_actor = _target_actor(target)
    source_configuration = _source_configuration(service)
    source_request = source.create_source_project_transfer_request(
        project_id=records.project_id,
        target_space_id=target.space_id,
        initiated_by=source_actor,
        source_configuration=source_configuration,
    )
    provisioning = _prepare_target_request(
        target,
        actor=target_actor,
        source=source_configuration,
        project_id=records.project_id,
        central_root=tmp_path / "central",
        monkeypatch=monkeypatch,
    )
    target_request = target.create_target_project_transfer_request(
        provisioning_request_id=provisioning.request_id,
        source_request_id=source_request.request_id,
        source_project_id=records.project_id,
        source_space_id=source.space_id,
        initiated_by=target_actor,
        source_configuration=source_configuration,
        source_configuration_sha256=project_transfer_source_configuration_sha256(
            source_configuration
        ),
        source_release_proof_sha256=source_request.source_release_proof_sha256,
        accepted_schema_generation=source_configuration.source_schema_generation,
        accepted_archive_codec=TRANSFER_ARCHIVE_CODEC,
    )
    assert target_request.link_receipt is not None
    source_request = source.link_source_project_transfer_request(
        source_request.request_id,
        receipt=target_request.link_receipt,
    )
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
        revalidated_configuration=source_configuration,
        source_head=service.history.head_ref(service.history.materialize(write_outputs=False)),
    )
    assert source_request.source_release_receipt is not None
    target_request = target.accept_source_project_transfer_release(
        target_request.request_id,
        receipt=source_request.source_release_receipt,
    )
    service.history.transfer_project_home(
        project_id=records.project_id,
        previous_home_space_id=source.space_id,
        new_home_space_id=target.space_id,
        source_released_by=source_actor,
        target_admitted_by=target_actor,
    )
    fence_head = service.history.head_ref(service.history.materialize(write_outputs=False))
    source_request = source.mark_source_project_transfer_fenced(
        source_request.request_id,
        source_head=fence_head,
    )
    source_proof = source.expose_project_transfer_proof(source_request.request_id)

    attribution = TransferArchiveAttribution(
        archive_actor_id=str(uuid.uuid4()),
        source_actor=TransferArchiveActor.capture(source_actor),
    )
    records = source.export_project_transfer_records(
        records.project_id,
        attributions=(attribution,),
    )
    file_capture_root = tmp_path / "project-file-capture"
    capture = capture_project_transfer_files(service, records, file_capture_root)
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    entries = [
        _entry(
            archive_root,
            "provenance/manifest.toml",
            "source_manifest_provenance",
            manifest.path.read_bytes(),
        )
    ]
    entries.append(
        _entry(
            archive_root,
            "canonical/scope-base.json",
            "canonical_history",
            (manifest.research_dir / "scope-base.json").read_bytes(),
        )
    )
    for path in sorted((manifest.research_dir / "patches").rglob("*.json")):
        entries.append(
            _entry(
                archive_root,
                f"canonical/{path.relative_to(manifest.research_dir).as_posix()}",
                "canonical_history",
                path.read_bytes(),
            )
        )
    for item in capture.entries:
        source_path = file_capture_root / item.archive_path
        destination = archive_root / item.archive_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)
        entries.append(item)
    entries.append(
        _entry(
            archive_root,
            TRANSFER_OPERATIONAL_RECORDS_PATH,
            "operational_records",
            transfer_project_file_payload(capture),
        )
    )
    provider_payload = b'{"cwd":"/source/repo-a","messages":["history"]}\n'
    provider_digest = hashlib.sha256(provider_payload).hexdigest()
    entries.append(
        _entry(
            archive_root,
            f"provider-history/codex/{provider_digest}",
            "provider_history",
            provider_payload,
        )
    )
    entries.append(
        _entry(
            archive_root,
            "control/source-release-proof.bin",
            "source_release_proof",
            source_proof,
        )
    )
    ordered = tuple(sorted(entries, key=lambda item: item.archive_path))
    archive = TransferArchiveManifest(
        project_id=records.project_id,
        source_space_id=source.space_id,
        target_space_id=target.space_id,
        source_request_id=source_request.request_id,
        target_request_id=target_request.request_id,
        source_rcp_version=source_configuration.source_rcp_version,
        source_schema_generation=source_configuration.source_schema_generation,
        source_configuration_sha256=project_transfer_source_configuration_sha256(
            source_configuration
        ),
        source_manifest_sha256=source_configuration.source_manifest_sha256,
        source_release_proof_sha256=hashlib.sha256(source_proof).hexdigest(),
        target_activation_proof_sha256=target_request.target_activation_proof_sha256,
        main_head=TransferGraphHead.capture(fence_head),
        branch_heads=(),
        attributions=(attribution,),
        diagnostics=(),
        entries=ordered,
        payload_size_bytes=sum(item.size_bytes for item in ordered),
        created_at=datetime.now(UTC),
    )
    sealed_archive_path = None
    if seal_archive:
        archive_root.chmod(0o700)
        for path in archive_root.rglob("*"):
            if path.is_file():
                path.chmod(0o400)
        sealed_root = tmp_path / "sealed"
        sealed_root.mkdir(mode=0o700)
        sealed = seal_transfer_archive(
            manifest=archive,
            capture_root=archive_root,
            destination=sealed_root / f"{archive.target_request_id}.rcp-transfer",
        )
        envelope = sealed.envelope
        sealed_archive_path = sealed.archive_path
    else:
        encoded = b"sealed transfer archive fixture"
        envelope = TransferArchiveEnvelope.bind(
            archive,
            archive_sha256=hashlib.sha256(encoded).hexdigest(),
            archive_size_bytes=len(encoded),
        )
    source.bind_project_transfer_archive(
        source_request.request_id,
        archive_sha256=envelope.archive_sha256,
        archive_size_bytes=envelope.archive_size_bytes,
    )
    target.bind_project_transfer_archive(
        target_request.request_id,
        archive_sha256=envelope.archive_sha256,
        archive_size_bytes=envelope.archive_size_bytes,
        source_fence_head=fence_head,
    )
    configured = build_transfer_target_configuration(
        provisioning,
        source_configuration,
        target_request.link_receipt,
        archive,
        archive_root,
    )
    catalog = ProjectCatalog(target_data, target, AgentLauncher())
    state_repository = next(
        item for item in provisioning.repositories if item.alias == provisioning.state_repository
    )
    assert state_repository.resolved_path is not None
    return {
        "source": source,
        "target": target,
        "catalog": catalog,
        "archive": archive,
        "envelope": envelope,
        "sealed_archive_path": sealed_archive_path,
        "archive_root": archive_root,
        "configuration": configured,
        "artifact": artifact,
        "artifact_name": artifact_name,
        "view": view,
        "view_name": view_name,
        "provider_digest": provider_digest,
        "target_state_root": Path(state_repository.resolved_path) / ".research",
    }


def test_target_import_publishes_exact_history_but_does_not_activate(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _archive_fixture(manifest, tmp_path, monkeypatch)

    receipt = import_project_transfer(
        fixture["catalog"],
        archive=fixture["archive"],
        envelope=fixture["envelope"],
        archive_root=fixture["archive_root"],
        target_configuration=fixture["configuration"],
    )

    target = fixture["target"]
    archive = fixture["archive"]
    assert receipt.status == "complete"
    assert target.project(archive.project_id) is None
    imported_tasks = target.agent_tasks(archive.project_id)
    assert len(imported_tasks) == 1
    assert imported_tasks[0].history_only is True
    assert imported_tasks[0].native_session_id is None
    assert imported_tasks[0].can_resume is False
    with target.connection() as connection:
        writing_session_count = connection.execute(
            "SELECT COUNT(*) FROM writing_sessions WHERE project_id = ?",
            (archive.project_id,),
        ).fetchone()[0]
    assert writing_session_count == 0
    imported = fixture["catalog"].data_dir / "project-sources" / archive.project_id
    assert (imported / "provider-history" / "codex" / fixture["provider_digest"]).is_file()
    target_state_root = fixture["target_state_root"]
    assert (target_state_root / "manifest.toml").read_text(encoding="utf-8") == (
        fixture["configuration"].manifest_content
    )
    assert fixture["configuration"].manifest_content != manifest.path.read_text(encoding="utf-8")
    assert (target_state_root / "paper/introduction.md").is_file()
    assert (target_state_root / "facts/methods/protocol.bin").read_bytes() == (b"opaque fact bytes")
    assert (target_state_root.parent / "artifacts" / fixture["artifact_name"]).read_bytes() == (
        fixture["artifact"]
    )
    assert (target_state_root.parent / "views" / fixture["view_name"]).read_bytes() == (
        fixture["view"]
    )
    stored = target.project_transfer_import(archive.target_request_id)
    assert stored == receipt

    repeated = import_project_transfer(
        fixture["catalog"],
        archive=archive,
        envelope=fixture["envelope"],
        archive_root=fixture["archive_root"],
        target_configuration=fixture["configuration"],
    )
    assert repeated == receipt
    assert target.project(archive.project_id) is None


def test_target_import_cleans_only_imported_sources_if_completion_crashes(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _archive_fixture(manifest, tmp_path, monkeypatch)
    target = fixture["target"]
    archive = fixture["archive"]
    original = target.complete_project_transfer_import

    def crash(*_args, **_kwargs):
        raise RuntimeError("injected completion crash")

    monkeypatch.setattr(target, "complete_project_transfer_import", crash)
    with pytest.raises(RuntimeError, match="injected"):
        import_project_transfer(
            fixture["catalog"],
            archive=archive,
            envelope=fixture["envelope"],
            archive_root=fixture["archive_root"],
            target_configuration=fixture["configuration"],
        )

    assert target.project(archive.project_id) is None
    pending = target.project_transfer_import(archive.target_request_id)
    assert pending is not None and pending.status == "database_imported"
    assert not (fixture["catalog"].data_dir / "project-sources" / archive.project_id).exists()

    monkeypatch.setattr(target, "complete_project_transfer_import", original)
    completed = import_project_transfer(
        fixture["catalog"],
        archive=archive,
        envelope=fixture["envelope"],
        archive_root=fixture["archive_root"],
        target_configuration=fixture["configuration"],
    )
    assert completed.status == "complete"
    assert target.project(archive.project_id) is None


@pytest.mark.parametrize(
    "boundary",
    ("database", "canonical", "project_files", "provider_history", "completion"),
)
def test_target_import_retries_the_same_archive_after_each_committed_boundary(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    fixture = _archive_fixture(manifest, tmp_path, monkeypatch)
    target = fixture["target"]
    archive = fixture["archive"]

    if boundary == "database":
        original = target._insert_transfer_watchers

        def fail_after_task_rows(*args, **kwargs):
            raise RuntimeError("injected database boundary crash")

        monkeypatch.setattr(target, "_insert_transfer_watchers", fail_after_task_rows)
    elif boundary == "completion":
        original = target.complete_project_transfer_import

        def fail_after_completion(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("injected completion boundary crash")

        monkeypatch.setattr(target, "complete_project_transfer_import", fail_after_completion)
    else:
        attribute = {
            "canonical": "_publish_canonical",
            "project_files": "_publish_project_files",
            "provider_history": "_publish_provider_history",
        }[boundary]
        original = getattr(transfer_importer, attribute)

        def fail_after_publication(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError(f"injected {boundary} boundary crash")

        monkeypatch.setattr(transfer_importer, attribute, fail_after_publication)

    with pytest.raises(RuntimeError, match="injected"):
        import_project_transfer(
            fixture["catalog"],
            archive=archive,
            envelope=fixture["envelope"],
            archive_root=fixture["archive_root"],
            target_configuration=fixture["configuration"],
        )

    assert target.project(archive.project_id) is None
    with target.connection() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM writing_sessions WHERE project_id = ?",
                (archive.project_id,),
            ).fetchone()[0]
            == 0
        )
        task_count = connection.execute(
            "SELECT COUNT(*) FROM graph_runs WHERE project_id = ?",
            (archive.project_id,),
        ).fetchone()[0]
    receipt = target.project_transfer_import(archive.target_request_id)
    if boundary == "database":
        assert receipt is None
        assert task_count == 0
        monkeypatch.setattr(target, "_insert_transfer_watchers", original)
    else:
        assert receipt is not None
        assert receipt.status == ("complete" if boundary == "completion" else "database_imported")
        assert task_count == 1
        if boundary == "completion":
            monkeypatch.setattr(target, "complete_project_transfer_import", original)
        else:
            monkeypatch.setattr(transfer_importer, attribute, original)

    completed = import_project_transfer(
        fixture["catalog"],
        archive=archive,
        envelope=fixture["envelope"],
        archive_root=fixture["archive_root"],
        target_configuration=fixture["configuration"],
    )
    assert completed.status == "complete"
    assert target.project(archive.project_id) is None
    assert len(target.agent_tasks(archive.project_id)) == 1


def test_target_import_rejects_an_undeclared_archive_entry_before_database_mutation(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _archive_fixture(manifest, tmp_path, monkeypatch)
    archive = fixture["archive"]
    (fixture["archive_root"] / "undeclared.bin").write_bytes(b"not in the manifest")

    with pytest.raises(ValueError, match="staging tree differs"):
        import_project_transfer(
            fixture["catalog"],
            archive=archive,
            envelope=fixture["envelope"],
            archive_root=fixture["archive_root"],
            target_configuration=fixture["configuration"],
        )

    assert fixture["target"].project_transfer_import(archive.target_request_id) is None
    assert fixture["target"].project(archive.project_id) is None
