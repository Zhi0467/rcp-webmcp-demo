from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from rcp.config import AGENT_EXECUTION_PROFILES, Manifest, load_manifest
from rcp.core.models import AuthorizedHuman, GraphBranchMetadata
from rcp.core.transition_models import GraphHeadRef, GraphTargetRef
from rcp.history import HistoryManager
from rcp.projects import (
    BackupProjectUnavailable,
    inspect_backup_project_registration,
)
from rcp.providers import configured_runtime_id
from rcp.server_ops.backup_models import (
    BACKUP_APP_DATA_CAPTURED,
    BACKUP_APP_DATA_DEFERRED,
    BACKUP_APP_DATA_EXCLUSIONS,
    BACKUP_MANIFEST_SCHEMA_VERSION,
    BACKUP_RESEARCH_CANONICAL_ROOTS,
    BACKUP_RESEARCH_DELEGATED_ROOTS,
    BACKUP_RESEARCH_EXCLUSIONS,
    BackupArchiveManifest,
    BackupCanonicalSourceFile,
    BackupCanonicalSourcePlan,
    BackupFileEntry,
    BackupImportedProviderSourceCapture,
    BackupImportedProviderSourceInventory,
    BackupManifestConfiguration,
    BackupProjectCapture,
    inspect_app_data_capture_plan,
)
from rcp.server_ops.github import parse_github_repository_ref
from rcp.setup import render_prepared_team_manifest
from rcp.sources import ImportedProviderSourceStore
from rcp.storage import (
    ProjectProvisioningGitCheckRecord,
    ProjectProvisioningMachineRecord,
    ProjectProvisioningProviderCheckRecord,
    ProjectProvisioningRepositoryRecord,
    ProjectProvisioningRequestRecord,
    ProjectRecord,
)
from rcp.storage.provisioning import project_provisioning_review_digest

SPACE_ID = "70994440-4c57-41b0-a2f6-8878856db969"
PROJECT_ID = "9c59550a-9787-466a-9435-1e59f0a9803f"
REQUEST_ID = "f0b1cd24-a735-43f8-a184-2f4ba933934b"
USER_ID = "cf4d29d0-a1bd-4d38-8620-242adf195bf6"
INSTALLATION_ID = "69726714-fee6-427f-8e1b-337350518beb"
CAPTURED_AT = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)
FINGERPRINT = "SHA256:" + ("A" * 43)


def test_backup_root_classification_is_an_exact_closed_policy() -> None:
    assert {"project-sources"} == BACKUP_APP_DATA_CAPTURED
    assert not BACKUP_APP_DATA_DEFERRED
    assert {
        "bootstrap-manifests",
        "chat-attachments",
        "paper-snapshots",
        "project-caches",
        "project-snapshots",
        "rcp-server.json",
        "rcp.lock",
        "rcp.sqlite3-journal",
        "rcp.sqlite3-shm",
        "rcp.sqlite3-wal",
        "run-stage",
        "session-slices",
        "source-cache",
        "state-cache",
        "transfer-exports",
        "transfer-inbox",
    } == BACKUP_APP_DATA_EXCLUSIONS
    assert {
        "branches",
        "manifest.toml",
        "patches",
        "scope-base.json",
    } == BACKUP_RESEARCH_CANONICAL_ROOTS
    assert {"chat", "facts", "paper"} == BACKUP_RESEARCH_DELEGATED_ROOTS
    assert {
        ".agent-run.lock",
        ".append.lock",
        ".chat.lock",
        ".publish",
        ".refresh.lock",
        "coverage.json",
        "cursors.json",
        "glossary.json",
        "graph.json",
        "proposals.json",
        "research.md",
    } == BACKUP_RESEARCH_EXCLUSIONS


def _completed_registration(
    tmp_path: Path,
) -> tuple[ProjectRecord, ProjectProvisioningRequestRecord]:
    central_root = tmp_path / "remote-central"
    repository_path = central_root / PROJECT_ID / "repositories" / "paper"
    checked_at = CAPTURED_AT.isoformat()
    machine = ProjectProvisioningMachineRecord(
        alias="worker",
        location="ssh",
        host="gpu.example",
        os_account="alice",
        central_root=str(central_root),
        resolved_central_root=str(central_root),
    )
    repository = ProjectProvisioningRepositoryRecord(
        alias="paper",
        repository=parse_github_repository_ref("git@github.com:OpenAI/RCP.git"),
        machine_alias="worker",
        intended_path=str(repository_path),
        resolved_path=str(repository_path),
        checkout_disposition="request_created",
        git_check=ProjectProvisioningGitCheckRecord(
            status="ready",
            commit="a" * 40,
            write_verified=True,
            deploy_key_label=f"rcp:{SPACE_ID}:{PROJECT_ID}:paper",
            public_key_fingerprint=FINGERPRINT,
            checked_at=checked_at,
        ),
    )
    providers = [
        ProjectProvisioningProviderCheckRecord(
            profile=profile,
            provider="codex",
            runtime_id="codex:exec",
            model="gpt-test",
            reasoning="medium",
            machine_alias="worker",
            status="ready",
            binary_path="/usr/local/bin/codex",
            version="codex-cli 1.2.3",
            resolved_runtime_id=configured_runtime_id("codex", "exec"),
            execution_account="alice",
            checked_at=checked_at,
        )
        for profile in AGENT_EXECUTION_PROFILES
    ]
    base_values = {
        "request_id": REQUEST_ID,
        "kind": "create_team_project",
        "status": "completed",
        "target_space_id": SPACE_ID,
        "authorized_by": AuthorizedHuman(
            space_id=SPACE_ID,
            user_id=USER_ID,
            display_name="Alice",
        ),
        "proposed_project_id": PROJECT_ID,
        "name": "Shared paper",
        "state_repository": "paper",
        "project_truth_scope": ["paper"],
        "default_run_truth_scope": ["paper"],
        "default_auto_research_invocation_ceiling": 10,
        "machines": [machine],
        "repositories": [repository],
        "provider_checks": providers,
        "final_review_digest": "0" * 64,
        "revision": 5,
        "created_at": checked_at,
        "updated_at": checked_at,
        "setup_started_at": checked_at,
        "ready_at": checked_at,
        "completed_at": checked_at,
    }
    draft = ProjectProvisioningRequestRecord.model_validate(base_values)
    request = ProjectProvisioningRequestRecord.model_validate(
        {
            **base_values,
            "final_review_digest": project_provisioning_review_digest(draft),
        }
    )
    locator = tmp_path / "bootstrap" / "manifest.toml"
    locator.parent.mkdir()
    locator.write_text(render_prepared_team_manifest(request), encoding="utf-8")
    record = ProjectRecord(
        project_id=PROJECT_ID,
        home_space_id=SPACE_ID,
        locator=str(locator),
        name="Shared paper",
        state_location=f"gpu.example:{repository_path}/.research",
        state_remote=True,
        added_at=checked_at,
    )
    return record, request


def _captured_project(tmp_path: Path) -> BackupProjectCapture:
    record, request = _completed_registration(tmp_path)
    registration = inspect_backup_project_registration(
        record,
        data_dir=tmp_path / "data",
        provisioning_requests=[request],
    )
    entry = BackupFileEntry(
        archive_path=f"projects/{PROJECT_ID}/canonical/manifest.toml",
        source_relative_path=".research/manifest.toml",
        group="canonical",
        sha256="b" * 64,
        size_bytes=17,
    )
    return BackupProjectCapture(
        project_id=PROJECT_ID,
        home_space_id=SPACE_ID,
        locator=record.locator,
        status="captured",
        main_head=GraphHeadRef(revision=0),
        files=(entry,),
        recovery=registration.recovery,
        total_bytes=17,
    )


def test_app_data_inventory_is_closed_and_never_follows_unknown_roots(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "rcp.sqlite3").write_bytes(b"sqlite")
    (data_dir / "rcp.sqlite3-wal").write_bytes(b"wal")
    (data_dir / "project-snapshots").mkdir()
    (data_dir / "project-sources").mkdir()
    (data_dir / "future-durable-root").mkdir()

    plan = inspect_app_data_capture_plan(data_dir)

    assert plan.database_path == str(data_dir / "rcp.sqlite3")
    assert plan.excluded_entries == ("project-snapshots", "rcp.sqlite3-wal")
    assert plan.captured_entries == ("project-sources",)
    assert plan.deferred_entries == ()
    assert plan.unclassified_entries == ("future-durable-root",)
    assert plan.complete is False


def test_unsafe_database_entry_is_not_accepted_as_snapshot_input(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    target = tmp_path / "outside.sqlite3"
    target.write_bytes(b"not the owned database")
    (data_dir / "rcp.sqlite3").symlink_to(target)

    plan = inspect_app_data_capture_plan(data_dir)

    assert plan.database_path is None
    assert plan.database_unavailable_reason == (
        "The application database is not a safe regular file."
    )
    assert plan.complete is False


def test_canonical_inventory_reuses_retained_inputs_and_excludes_materializations(
    manifest: Manifest,
) -> None:
    history = HistoryManager(manifest)
    history.initialize()
    root = manifest.research_dir
    for delegated in ("chat", "facts", "paper"):
        (root / delegated).mkdir(exist_ok=True)
    (root / ".publish").mkdir()
    (root / "patches" / ".unconfirmed-000001-test").write_text(
        "quarantine",
        encoding="utf-8",
    )
    hidden_branch = root / "branches" / f".unconfirmed-{uuid.uuid4()}-test"
    hidden_branch.mkdir(parents=True)
    branch_id = str(uuid.uuid4())
    branch_root = root / "branches" / branch_id
    (branch_root / "patches").mkdir(parents=True)
    (branch_root / "merges").mkdir()
    (branch_root / "patches" / ".unconfirmed-000001.json-test").write_text(
        "quarantine",
        encoding="utf-8",
    )
    metadata = GraphBranchMetadata(
        branch_id=branch_id,
        episode_id=branch_id,
        project_id=PROJECT_ID,
        base_head=GraphHeadRef(revision=0),
        head=GraphHeadRef(
            target=GraphTargetRef(kind="branch", branch_id=branch_id),
            revision=0,
        ),
        authorized_by=AuthorizedHuman(
            space_id=SPACE_ID,
            user_id=USER_ID,
            display_name="Alice",
        ),
    )
    (branch_root / "branch.json").write_text(
        metadata.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="malformed canonical name"):
        history.workspace.retained_history_fingerprint()

    plan = history.workspace.backup_canonical_source_plan()

    assert plan.main_observed_revision == 0
    assert [item.kind for item in plan.main_files] == ["manifest", "scope_base"]
    assert plan.delegated_roots == ("chat", "facts", "paper")
    assert ".publish" in plan.excluded_roots
    assert "graph.json" in plan.excluded_roots
    assert plan.excluded_canonical_paths == tuple(
        sorted(
            (
                hidden_branch.relative_to(root).as_posix(),
                f"branches/{branch_id}/patches/.unconfirmed-000001.json-test",
                "patches/.unconfirmed-000001-test",
            )
        )
    )
    assert plan.unclassified_roots == ()
    assert plan.complete is True
    assert len(plan.branches) == 1
    assert plan.branches[0].branch_id == branch_id
    assert [item.kind for item in plan.branches[0].files] == ["branch_metadata"]
    assert all(
        Path(item.relative_path).name not in {"graph.json", "research.md"}
        for item in (*plan.main_files, *plan.branches[0].files)
    )

    (root / "future-durable-root").mkdir()
    partial = history.workspace.backup_canonical_source_plan()
    assert partial.unclassified_roots == ("future-durable-root",)
    assert partial.complete is False


def test_branch_metadata_must_match_the_observed_patch_head(manifest: Manifest) -> None:
    history = HistoryManager(manifest)
    history.initialize()
    root = manifest.research_dir
    branch_id = str(uuid.uuid4())
    branch_root = root / "branches" / branch_id
    (branch_root / "patches").mkdir(parents=True)
    (branch_root / "merges").mkdir()
    metadata = GraphBranchMetadata(
        branch_id=branch_id,
        episode_id=branch_id,
        project_id=PROJECT_ID,
        base_head=GraphHeadRef(revision=0),
        head=GraphHeadRef(
            target=GraphTargetRef(kind="branch", branch_id=branch_id),
            revision=0,
        ),
        authorized_by=AuthorizedHuman(
            space_id=SPACE_ID,
            user_id=USER_ID,
            display_name="Alice",
        ),
    )
    (branch_root / "branch.json").write_text(metadata.model_dump_json(), encoding="utf-8")
    (branch_root / "patches" / "000001.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="metadata and retained Patch head disagree"):
        history.workspace.backup_canonical_source_plan()


def test_completed_provisioning_builds_one_secret_free_recovery_descriptor(
    tmp_path: Path,
) -> None:
    record, request = _completed_registration(tmp_path)

    registration = inspect_backup_project_registration(
        record,
        data_dir=tmp_path / "data",
        provisioning_requests=[request],
    )

    assert registration.record == record
    assert registration.recovery.project_id == PROJECT_ID
    assert registration.recovery.configuration == BackupManifestConfiguration.from_manifest(
        load_manifest(record.locator)
    )
    assert registration.recovery.configuration.state_repository == "paper"
    assert registration.recovery.repositories[0].repository.identity == "openai/rcp"
    assert registration.recovery.repositories[0].deploy_key_label == (
        f"rcp:{SPACE_ID}:{PROJECT_ID}:paper"
    )
    assert registration.recovery.repositories[0].public_key_fingerprint == FINGERPRINT
    payload = registration.recovery.model_dump_json()
    assert "private_key" not in payload
    assert "AGE-SECRET-KEY" not in payload
    assert "token" not in payload


def test_missing_or_stale_recovery_proof_makes_the_project_uncapturable(
    tmp_path: Path,
) -> None:
    record, request = _completed_registration(tmp_path)

    with pytest.raises(BackupProjectUnavailable, match="exactly one completed"):
        inspect_backup_project_registration(
            record,
            data_dir=tmp_path / "data",
            provisioning_requests=[],
        )

    manifest_path = Path(record.locator)
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8").replace(
            'name = "Shared paper"',
            'name = "Changed later"',
            1,
        ),
        encoding="utf-8",
    )
    changed_record = record.model_copy(update={"name": "Changed later"})
    with pytest.raises(BackupProjectUnavailable, match="manifest changed"):
        inspect_backup_project_registration(
            changed_record,
            data_dir=tmp_path / "data",
            provisioning_requests=[request],
        )


def test_archive_manifest_round_trips_and_calculates_complete_or_partial(
    tmp_path: Path,
) -> None:
    project = _captured_project(tmp_path)
    sqlite = BackupFileEntry(
        archive_path="database/rcp.sqlite3",
        source_relative_path="rcp.sqlite3",
        group="sqlite_snapshot",
        sha256="c" * 64,
        size_bytes=31,
    )
    manifest = BackupArchiveManifest(
        space_id=SPACE_ID,
        space_name="Research lab",
        rcp_source_commit="d" * 40,
        database_schema_sha256="e" * 64,
        captured_at=CAPTURED_AT,
        sqlite_snapshot=sqlite,
        encryption_recipient_fingerprint="f" * 64,
        installation_id=INSTALLATION_ID,
        source_deploy_key_label=f"rcp-source:{INSTALLATION_ID}",
        source_public_key_fingerprint=FINGERPRINT,
        excluded_app_data_entries=("rcp.lock", "rcp.sqlite3-wal"),
        uncaptured_app_data_entries=(),
        projects=(project,),
        status="complete",
        total_bytes=48,
    )

    assert manifest.schema_version == BACKUP_MANIFEST_SCHEMA_VERSION
    assert BackupArchiveManifest.model_validate_json(manifest.model_dump_json()) == manifest
    absent_inventory = ImportedProviderSourceStore(
        tmp_path / "data",
        project.project_id,
    ).inventory()
    with_absent_source = manifest.model_copy(
        update={
            "imported_sources": (
                BackupImportedProviderSourceCapture(
                    project_id=project.project_id,
                    inventory=BackupImportedProviderSourceInventory.model_validate(
                        absent_inventory.model_dump()
                    ),
                    present=False,
                    files=(),
                    total_bytes=0,
                ),
            )
        }
    )
    assert "project-sources" not in with_absent_source.captured_app_data_entries
    assert (
        BackupArchiveManifest.model_validate(with_absent_source.model_dump(mode="python"))
        == with_absent_source
    )

    uncaptured = BackupProjectCapture(
        project_id=str(uuid.uuid4()),
        home_space_id=str(uuid.uuid4()),
        locator="/tmp/unavailable/manifest.toml",
        status="uncaptured",
        unavailable_reason="The canonical host was unreachable.",
        unavailable_at=CAPTURED_AT,
        total_bytes=0,
    )
    partial = manifest.model_copy(
        update={
            "projects": (project, uncaptured),
            "status": "partial",
        }
    )
    assert BackupArchiveManifest.model_validate(partial.model_dump(mode="python")) == partial

    with pytest.raises(ValidationError, match="status does not match"):
        BackupArchiveManifest.model_validate(
            {**partial.model_dump(mode="python"), "status": "complete"}
        )


def test_manifest_refuses_newer_schema_materialized_outputs_and_extra_fields(
    tmp_path: Path,
) -> None:
    project = _captured_project(tmp_path)
    sqlite = BackupFileEntry(
        archive_path="database/rcp.sqlite3",
        source_relative_path="rcp.sqlite3",
        group="sqlite_snapshot",
        sha256="c" * 64,
        size_bytes=31,
    )
    values = {
        "space_id": SPACE_ID,
        "space_name": "Research lab",
        "rcp_source_commit": "d" * 40,
        "database_schema_sha256": "e" * 64,
        "captured_at": CAPTURED_AT,
        "sqlite_snapshot": sqlite,
        "encryption_recipient_fingerprint": "f" * 64,
        "installation_id": INSTALLATION_ID,
        "excluded_app_data_entries": (),
        "uncaptured_app_data_entries": (),
        "projects": (project,),
        "status": "complete",
        "total_bytes": 48,
    }
    with pytest.raises(ValidationError):
        BackupArchiveManifest.model_validate({**values, "schema_version": 2})
    with pytest.raises(ValidationError, match="extra_forbidden"):
        BackupArchiveManifest.model_validate({**values, "private_identity": "secret"})
    with pytest.raises(ValidationError, match="retained research inputs"):
        BackupFileEntry(
            archive_path="projects/bad/canonical/graph.json",
            source_relative_path=".research/graph.json",
            group="canonical",
            sha256="0" * 64,
            size_bytes=1,
        )
    with pytest.raises(ValidationError, match="kind does not match"):
        BackupCanonicalSourceFile(
            relative_path="graph.json",
            kind="manifest",
            observed_size_bytes=1,
        )
    canonical_manifest = BackupCanonicalSourceFile(
        relative_path="manifest.toml",
        kind="manifest",
        observed_size_bytes=1,
    )
    with pytest.raises(ValidationError, match="main head does not match"):
        BackupCanonicalSourcePlan(
            main_observed_revision=1,
            main_files=(canonical_manifest,),
            branches=(),
            delegated_roots=(),
            excluded_roots=(),
            excluded_canonical_paths=(),
            unclassified_roots=(),
            observed_canonical_bytes=1,
        )
    with pytest.raises(ValidationError, match="project main head does not match"):
        BackupProjectCapture.model_validate(
            {
                **project.model_dump(mode="python"),
                "main_head": GraphHeadRef(revision=1),
            }
        )
    orphan_branch = str(uuid.uuid4())
    orphan_merge = BackupFileEntry(
        archive_path=f"projects/{PROJECT_ID}/canonical/orphan-merge.json",
        source_relative_path=(f".research/branches/{orphan_branch}/merges/{'1' * 64}.json"),
        group="canonical",
        sha256="1" * 64,
        size_bytes=1,
    )
    with pytest.raises(ValidationError, match="merge receipts require"):
        BackupProjectCapture.model_validate(
            {
                **project.model_dump(mode="python"),
                "files": (*project.files, orphan_merge),
                "total_bytes": project.total_bytes + orphan_merge.size_bytes,
            }
        )


def test_locator_manifest_is_loaded_without_refreshing_remote_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record, request = _completed_registration(tmp_path)

    def fail_refresh(*_args, **_kwargs):
        raise AssertionError("backup registration inspection must not refresh canonical state")

    monkeypatch.setattr("rcp.transport.state.SSHStateWorkspace.refresh", fail_refresh)
    registration = inspect_backup_project_registration(
        record,
        data_dir=tmp_path / "data",
        provisioning_requests=[request],
    )

    assert load_manifest(record.locator).name == "Shared paper"
    assert registration.workspace.remote is True
