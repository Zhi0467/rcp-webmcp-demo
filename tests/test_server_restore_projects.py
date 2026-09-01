from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from rcp.config import AGENT_EXECUTION_PROFILES, load_manifest, permissions_for
from rcp.core.models import AuthorizedHuman, GraphBranchMetadata
from rcp.core.transition_models import GraphHeadRef, GraphTargetRef
from rcp.history import HistoryManager
from rcp.projects import _render_restored_manifest, rebind_restored_project_registration
from rcp.server_ops.backup_capture import _database_schema_sha256
from rcp.server_ops.backup_models import (
    BackupArchiveManifest,
    BackupCheckoutRecoveryDescriptor,
    BackupFileEntry,
    BackupImportedProviderSourceCapture,
    BackupImportedProviderSourceInventory,
    BackupManifestAgentProfile,
    BackupManifestConfiguration,
    BackupManifestMachine,
    BackupManifestRepository,
    BackupManifestSources,
    BackupProjectCapture,
    BackupRecoveryMachine,
    BackupRecoveryRepository,
)
from rcp.server_ops.github import GitHubRepositoryRef
from rcp.server_ops.layout import ServerLayout
from rcp.server_ops.restore import (
    LinuxRestoreMachine,
    RestoreConfirmation,
    RestoreOperationJournal,
    RestoreProjectRebind,
    RestoreRefused,
    RestoreRepositoryRecovery,
    write_restore_journal,
)
from rcp.skill_registry import SkillDefaults
from rcp.sources import ImportedProviderSourceInventory, ImportedProviderSourceStore
from rcp.storage import AppStore, ProjectRecord
from rcp.transfer import TransferArchiveEntry

NOW = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)
COMMIT = "c" * 40
OLD_FINGERPRINT = "SHA256:" + ("A" * 43)
NEW_FINGERPRINT = "SHA256:" + ("B" * 43)
IMPORTED_PAYLOAD = b'{"type":"assistant","text":"restored native history"}\n'


def _layout(root: Path) -> ServerLayout:
    home = root / "home" / "rcp"
    server = home / "rcp-server"
    return ServerLayout(
        service_account="rcp",
        service_home=home,
        server_root=server,
        source_checkout=server / "source",
        releases_root=server / "releases",
        data_dir=server / "data",
        projects_root=server / "projects",
        credentials_root=server / "credentials",
        update_checkpoints_root=server / "update-checkpoints",
        restore_operations_root=server / "restore-operations",
        codex_state_root=home / ".codex",
        claude_state_root=home / ".claude",
        ssh_state_root=home / ".ssh",
        config_path=root / "etc" / "rcp" / "server.toml",
        current_release=root / "etc" / "rcp" / "current",
        runtime_dir=root / "run" / "rcp",
        control_socket=root / "run" / "rcp" / "control.sock",
        cli_wrapper=root / "usr" / "local" / "bin" / "rcp",
        systemd_unit=root / "etc" / "systemd" / "system" / "rcp.service",
        service_unit_name="rcp.service",
    )


def _configuration(repository: Path, service_home: Path) -> BackupManifestConfiguration:
    return BackupManifestConfiguration(
        name="Restored project",
        machines=(
            BackupManifestMachine(
                alias="server",
                host="",
                os_account="rcp",
                provider_paths={},
            ),
        ),
        repositories=(
            BackupManifestRepository(alias="paper", machine="server", path=str(repository)),
        ),
        project_truth_scope=("paper",),
        state_repository="paper",
        default_run_truth_scope=("paper",),
        default_auto_research_invocation_ceiling=10,
        skill_defaults=SkillDefaults(),
        agent_profiles=tuple(
            BackupManifestAgentProfile(
                profile=profile,
                provider="codex",
                runtime="exec",
                model="gpt-test",
                reasoning="medium",
                run_on="server",
                permissions=permissions_for(profile),
            )
            for profile in AGENT_EXECUTION_PROFILES
        ),
        sources=BackupManifestSources(
            claude_roots=(str(service_home / ".claude" / "projects"),),
            codex_roots=(str(service_home / ".codex" / "sessions"),),
            remote_claude_roots=("~/.claude/projects",),
            remote_codex_roots=("~/.codex/sessions",),
        ),
    )


def _entry(
    project_id: str,
    source_relative_path: str,
    group: str,
    source: Path,
) -> BackupFileEntry:
    data = source.read_bytes()
    return BackupFileEntry(
        archive_path=(PurePosixPath("projects") / project_id / source_relative_path).as_posix(),
        source_relative_path=source_relative_path,
        group=group,
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
    )


def _copy_entry(payload: Path, entry: BackupFileEntry, source: Path) -> None:
    destination = payload.joinpath(*PurePosixPath(entry.archive_path).parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _captured_imported_source(
    root: Path,
    *,
    project_id: str,
) -> BackupImportedProviderSourceCapture:
    digest = hashlib.sha256(IMPORTED_PAYLOAD).hexdigest()
    source_data = root / "imported-source-data"
    source_data.mkdir(mode=0o700)
    source_capture = root / "imported-source-capture"
    source = source_capture / "provider-history" / "codex" / digest
    source.parent.mkdir(parents=True, mode=0o700)
    source.write_bytes(IMPORTED_PAYLOAD)
    owner = ImportedProviderSourceStore(source_data, project_id)
    inventory = owner.publish(
        source_capture,
        (
            TransferArchiveEntry(
                archive_path=f"provider-history/codex/{digest}",
                group="provider_history",
                sha256=digest,
                size_bytes=len(IMPORTED_PAYLOAD),
            ),
        ),
    )
    collection = root / "payload" / "project-sources"
    collection.mkdir(mode=0o700)
    project_root = collection / project_id
    project_root.mkdir(mode=0o700)
    snapshot = owner.capture_snapshot(
        project_root / "provider-history",
        expected_inventory=inventory,
    )
    files = tuple(
        BackupFileEntry(
            archive_path=(f"project-sources/{project_id}/provider-history/{item.relative_path}"),
            source_relative_path=f"provider-history/{item.relative_path}",
            group="imported_provider_history",
            sha256=item.sha256,
            size_bytes=item.size_bytes,
        )
        for item in snapshot.files
    )
    return BackupImportedProviderSourceCapture(
        project_id=project_id,
        inventory=BackupImportedProviderSourceInventory.model_validate(inventory.model_dump()),
        present=True,
        files=files,
        total_bytes=sum(item.size_bytes for item in files),
    )


def _captured_project(
    root: Path,
    *,
    store: AppStore,
    layout: ServerLayout,
) -> tuple[BackupProjectCapture, Path]:
    project_id = str(uuid.uuid4())
    repository = layout.projects_root / project_id / "repositories" / "paper"
    repository.mkdir(parents=True)
    configuration = _configuration(repository, layout.service_home)
    recovery = BackupCheckoutRecoveryDescriptor(
        request_id=str(uuid.uuid4()),
        project_id=project_id,
        home_space_id=store.space_id,
        completed_at=NOW,
        final_review_digest="d" * 64,
        configuration=configuration,
        configuration_sha256=configuration.sha256,
        machines=(
            BackupRecoveryMachine(
                alias="server",
                location="local",
                host="",
                os_account="rcp",
                resolved_central_root=str(layout.projects_root),
            ),
        ),
        repositories=(
            BackupRecoveryRepository(
                alias="paper",
                repository=GitHubRepositoryRef(identity="openai/rcp"),
                machine_alias="server",
                resolved_path=str(repository),
                git_commit=COMMIT,
                deploy_key_label=f"rcp:{store.space_id}:{project_id}:paper",
                public_key_fingerprint=OLD_FINGERPRINT,
            ),
        ),
    )
    authoring = root / "authoring" / project_id / ".research"
    authoring.mkdir(parents=True)
    manifest_path = authoring / "manifest.toml"
    manifest_content = _render_restored_manifest(configuration, {"paper": str(repository)})
    manifest_content = manifest_content.replace(
        f'claude_roots = ["{layout.service_home}/.claude/projects"]',
        'claude_roots = ["~/.claude/projects"]',
    ).replace(
        f'codex_roots = ["{layout.service_home}/.codex/sessions"]',
        'codex_roots = ["~/.codex/sessions"]',
    )
    manifest_path.write_text(manifest_content, encoding="utf-8")
    history = HistoryManager(
        load_manifest(manifest_path, local_home=layout.service_home),
        expected_space_id=store.space_id,
    )
    history.claim_project_identity("created", project_id=project_id)
    branch_id = str(uuid.uuid4())
    main_head = history.head_ref()
    branch_head = GraphHeadRef(
        target=GraphTargetRef(kind="branch", branch_id=branch_id),
        revision=main_head.revision,
        transition_id=main_head.transition_id,
    )
    branch_root = authoring / "branches" / branch_id
    (branch_root / "patches").mkdir(parents=True)
    (branch_root / "merges").mkdir()
    (branch_root / "branch.json").write_text(
        GraphBranchMetadata(
            branch_id=branch_id,
            episode_id=branch_id,
            project_id=project_id,
            base_head=main_head,
            head=branch_head,
            authorized_by=AuthorizedHuman(
                space_id=store.space_id,
                user_id=str(uuid.uuid4()),
                display_name="Restore owner",
            ),
        ).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )

    chat_id = str(uuid.uuid4())
    chat = authoring / "chat" / f"project-{chat_id}.jsonl"
    chat.parent.mkdir(exist_ok=True)
    chat.write_text(
        json.dumps(
            {
                "sessionId": chat_id,
                "nodeId": None,
                "chatScope": "project",
                "timestamp": NOW.isoformat(),
                "uuid": str(uuid.uuid4()),
                "type": "assistant",
                "role": "assistant",
                "text": "Restored answer",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    paper = authoring / "paper" / "introduction.md"
    paper.parent.mkdir(exist_ok=True)
    paper.write_text("# Restored introduction\n", encoding="utf-8")
    fact = authoring / "facts" / "nested" / "evidence.json"
    fact.parent.mkdir(parents=True)
    fact.write_text('{"result": true}\n', encoding="utf-8")
    kept_artifact = root / "kept-result.html"
    kept_artifact.write_text("<p>artifact</p>\n", encoding="utf-8")
    kept_view = root / "legacy-result.html"
    kept_view.write_text("<p>legacy</p>\n", encoding="utf-8")

    canonical_plan = history.workspace.backup_canonical_source_plan()
    sources: list[tuple[BackupFileEntry, Path]] = []
    for item in (
        *canonical_plan.main_files,
        *(file for branch in canonical_plan.branches for file in branch.files),
    ):
        source = authoring / item.relative_path
        sources.append(
            (_entry(project_id, f".research/{item.relative_path}", "canonical", source), source)
        )
    sources.extend(
        (
            (_entry(project_id, f".research/chat/{chat.name}", "chat", chat), chat),
            (
                _entry(
                    project_id,
                    ".research/paper/introduction.md",
                    "paper_introduction",
                    paper,
                ),
                paper,
            ),
            (
                _entry(
                    project_id,
                    ".research/facts/nested/evidence.json",
                    "fact",
                    fact,
                ),
                fact,
            ),
            (
                _entry(project_id, "artifacts/kept-result.html", "kept_artifact", kept_artifact),
                kept_artifact,
            ),
            (
                _entry(
                    project_id,
                    "views/legacy-result.html",
                    "legacy_kept_result_view",
                    kept_view,
                ),
                kept_view,
            ),
        )
    )
    payload = root / "payload"
    for entry, source in sources:
        _copy_entry(payload, entry, source)
    files = tuple(sorted((entry for entry, _source in sources), key=lambda item: item.archive_path))
    capture = BackupProjectCapture(
        project_id=project_id,
        home_space_id=store.space_id,
        locator=str(root / "old" / "manifest.toml"),
        status="captured",
        main_head=main_head,
        branch_heads=(branch_head,),
        files=files,
        recovery=recovery,
        total_bytes=sum(item.size_bytes for item in files),
    )
    store.upsert_project(
        ProjectRecord(
            project_id=project_id,
            home_space_id=store.space_id,
            locator=str(root / "old" / "manifest.toml"),
            name=configuration.name,
            state_location="old:state",
            state_remote=False,
            added_at=NOW.isoformat(),
            reachable=True,
        )
    )
    rebind_restored_project_registration(
        store,
        capture,
        repository_paths={"paper": str(repository)},
        data_dir=layout.data_dir,
        uid=os.getuid(),
        gid=os.getgid(),
    )
    return capture, repository


class _RestoreMachine(LinuxRestoreMachine):
    @contextmanager
    def admission(self):
        yield


def _restore_case(tmp_path: Path):
    layout = _layout(tmp_path)
    for path in (layout.data_dir, layout.projects_root, layout.restore_operations_root):
        path.mkdir(parents=True, mode=0o700)
    store, _bootstrap = AppStore.initialize_team_space(
        layout.data_dir / "rcp.sqlite3", "Restore lab"
    )
    candidate_root = layout.restore_operations_root / f"candidate-{'f' * 64}"
    capture, repository = _captured_project(
        candidate_root,
        store=store,
        layout=layout,
    )
    sqlite_source = layout.data_dir / "rcp.sqlite3"
    sqlite_sha = hashlib.sha256(sqlite_source.read_bytes()).hexdigest()
    sqlite_entry = BackupFileEntry(
        archive_path="database/rcp.sqlite3",
        source_relative_path="rcp.sqlite3",
        group="sqlite_snapshot",
        sha256=sqlite_sha,
        size_bytes=sqlite_source.stat().st_size,
    )
    imported = _captured_imported_source(
        candidate_root,
        project_id=capture.project_id,
    )
    manifest = BackupArchiveManifest(
        space_id=store.space_id,
        space_name=store.space_name,
        rcp_source_commit=COMMIT,
        database_schema_sha256=_database_schema_sha256(store),
        captured_at=NOW,
        sqlite_snapshot=sqlite_entry,
        encryption_recipient_fingerprint="e" * 64,
        installation_id=str(uuid.uuid4()),
        excluded_app_data_entries=(),
        captured_app_data_entries=("project-sources",),
        uncaptured_app_data_entries=(),
        projects=(capture,),
        imported_sources=(imported,),
        status="complete",
        total_bytes=sqlite_entry.size_bytes + capture.total_bytes + imported.total_bytes,
    )
    candidate_sqlite = candidate_root / "restored" / "rcp.sqlite3"
    candidate_sqlite.parent.mkdir(parents=True)
    shutil.copyfile(sqlite_source, candidate_sqlite)
    archive = tmp_path / "archive.age"
    archive.write_bytes(b"archive")
    record = store.project(capture.project_id)
    assert record is not None and capture.recovery is not None
    journal = RestoreOperationJournal(
        operation_id=str(uuid.uuid4()),
        archive_path=str(archive),
        archive_sha256="f" * 64,
        archive_size_bytes=archive.stat().st_size,
        manifest_sha256="1" * 64,
        configured_data_dir=str(layout.data_dir),
        candidate_root=str(candidate_root),
        candidate_sqlite_path=str(candidate_sqlite),
        candidate_sqlite_sha256=sqlite_sha,
        manifest=manifest,
        confirmation=RestoreConfirmation(
            confirmed_data_dir=str(layout.data_dir),
            confirmed_by="root@lab uid=0",
            confirmed_at=NOW,
        ),
        phase="checkouts_reconstructed",
        detached_at=NOW,
        restored_sqlite_sha256=sqlite_sha,
        repository_recoveries=(
            RestoreRepositoryRecovery(
                project_id=capture.project_id,
                repository_alias="paper",
                machine_alias="server",
                state="checkout_ready",
                deploy_key_label=f"rcp:{store.space_id}:{capture.project_id}:paper",
                deploy_public_key="ssh-ed25519 AAAA restored",
                public_key_fingerprint=NEW_FINGERPRINT,
                probed_commit=COMMIT,
                central_root=str(layout.projects_root),
                repository_path=str(repository),
                checkout_disposition="request_created",
                checkout_commit=COMMIT,
            ),
        ),
        project_rebinds=(
            RestoreProjectRebind(
                project_id=capture.project_id,
                locator=record.locator,
                state_location=record.state_location,
                state_remote=False,
            ),
        ),
        updated_at=NOW,
    )
    write_restore_journal(journal, layout, uid=os.getuid(), gid=os.getgid())
    machine = _RestoreMachine(
        layout,
        config_loader=lambda _path: SimpleNamespace(
            paths=SimpleNamespace(model_dump=layout.recorded_paths)
        ),
        service_control=SimpleNamespace(fence_stopped_disabled=lambda: None),
        service_identity=(os.getuid(), os.getgid()),
        root_identity=(os.getuid(), os.getgid()),
        clock=lambda: NOW,
    )
    return machine, journal, capture, repository, store


def test_restore_publishes_replays_and_reads_back_every_project_owner(tmp_path: Path) -> None:
    machine, journal, capture, repository, store = _restore_case(tmp_path)
    archived_manifest = next(
        item for item in capture.files if item.source_relative_path == ".research/manifest.toml"
    )
    archived_manifest_path = (
        Path(journal.candidate_root) / "payload" / archived_manifest.archive_path
    )
    assert "~/.codex/sessions" in archived_manifest_path.read_text(encoding="utf-8")

    completed = machine.publish_projects(journal)

    assert completed.phase == "projects_published"
    assert completed.project_publications[0].project_id == capture.project_id
    imported = journal.manifest.imported_sources[0]
    expected_inventory = ImportedProviderSourceInventory.model_validate(
        imported.inventory.model_dump()
    )
    imported_owner = ImportedProviderSourceStore(machine.layout.data_dir, capture.project_id)
    assert imported_owner.inventory() == expected_inventory
    imported_file = (
        imported_owner.root
        / expected_inventory.files[0].provider
        / expected_inventory.files[0].sha256
    )
    assert imported_file.read_bytes() == IMPORTED_PAYLOAD
    assert completed.project_publications[0].imported_files == len(imported.files)
    assert completed.project_publications[0].imported_bytes == imported.total_bytes
    record = store.project(capture.project_id)
    assert record is not None and record.reachable is True and record.error is None
    research = repository / ".research"
    assert (research / "paper" / "introduction.md").read_text() == "# Restored introduction\n"
    assert (research / "facts" / "nested" / "evidence.json").read_text() == ('{"result": true}\n')
    assert (repository / "artifacts" / "kept-result.html").read_text() == ("<p>artifact</p>\n")
    assert (repository / "views" / "legacy-result.html").read_text() == "<p>legacy</p>\n"
    restored_manifest = load_manifest(
        research / "manifest.toml",
        local_home=machine.layout.service_home,
    )
    assert restored_manifest.sources.codex_roots == [
        str(machine.layout.service_home / ".codex" / "sessions")
    ]
    assert "~/.codex/sessions" in (research / "manifest.toml").read_text(encoding="utf-8")
    restored = HistoryManager(restored_manifest)
    assert restored.head_ref().revision == capture.main_head.revision
    assert (
        restored.branch(capture.branch_heads[0].target.branch_id).head_ref()
        == (capture.branch_heads[0])
    )
    assert machine.publish_projects(completed) == completed


def test_restore_refuses_conflicting_imported_source_before_visibility(
    tmp_path: Path,
) -> None:
    machine, journal, capture, _repository, store = _restore_case(tmp_path)
    conflicting = b'{"type":"assistant","text":"different history"}\n'
    digest = hashlib.sha256(conflicting).hexdigest()
    capture_root = tmp_path / "conflicting-imported-source"
    source = capture_root / "provider-history" / "codex" / digest
    source.parent.mkdir(parents=True, mode=0o700)
    source.write_bytes(conflicting)
    owner = ImportedProviderSourceStore(machine.layout.data_dir, capture.project_id)
    owner.publish(
        capture_root,
        (
            TransferArchiveEntry(
                archive_path=f"provider-history/codex/{digest}",
                group="provider_history",
                sha256=digest,
                size_bytes=len(conflicting),
            ),
        ),
    )

    with pytest.raises(RestoreRefused, match="failed imported-source publication"):
        machine.publish_projects(journal)

    record = store.project(capture.project_id)
    assert record is not None and record.reachable is False
    assert (
        owner.inventory().fingerprint != journal.manifest.imported_sources[0].inventory.fingerprint
    )


def test_restore_refuses_conflicting_project_byte_before_visibility(tmp_path: Path) -> None:
    machine, journal, capture, repository, store = _restore_case(tmp_path)
    fact = repository / ".research" / "facts" / "nested" / "evidence.json"
    fact.parent.mkdir(parents=True)
    fact.write_text("conflict\n", encoding="utf-8")

    with pytest.raises(RestoreRefused, match="failed canonical publication"):
        machine.publish_projects(journal)

    record = store.project(capture.project_id)
    assert record is not None and record.reachable is False
    assert fact.read_text() == "conflict\n"


def test_restore_keeps_explicitly_uncaptured_project_visible_but_unavailable(
    tmp_path: Path,
) -> None:
    machine, journal, _capture, _repository, store = _restore_case(tmp_path)
    project_id = str(uuid.uuid4())
    store.upsert_project(
        ProjectRecord(
            project_id=project_id,
            home_space_id=store.space_id,
            locator=str(tmp_path / "missing" / "manifest.toml"),
            name="Unavailable project",
            state_location="missing:state",
            state_remote=True,
            added_at=NOW.isoformat(),
            reachable=True,
        )
    )
    unavailable = BackupProjectCapture(
        project_id=project_id,
        home_space_id=store.space_id,
        locator=str(tmp_path / "missing" / "manifest.toml"),
        status="uncaptured",
        unavailable_kind="inventory_failure",
        unavailable_reason="The project inventory could not be validated.",
        unavailable_at=NOW,
        total_bytes=0,
    )
    updated_manifest = journal.manifest.model_copy(
        update={
            "projects": (*journal.manifest.projects, unavailable),
            "status": "partial",
        }
    )
    updated = journal.model_copy(update={"manifest": updated_manifest})
    write_restore_journal(
        updated,
        machine.layout,
        uid=os.getuid(),
        gid=os.getgid(),
    )

    machine.publish_projects(updated)

    record = store.project(project_id)
    assert record is not None and record.reachable is False
    assert record.error == (
        "Not captured by the replacement archive: The project inventory could not be validated."
    )
