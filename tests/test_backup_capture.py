from __future__ import annotations

import hashlib
import json
import os
import pwd
import queue
import subprocess
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

import rcp.server_ops.backup_checkout as backup_checkout
import rcp.server_ops.backup_project_files as project_files
import rcp.server_ops.backup_project_io as project_io
import rcp.transport.state as state_module
from rcp.config import load_manifest
from rcp.core.models import AuthorizedHuman, GraphBranchMetadata
from rcp.core.transition_models import GraphHeadRef, GraphTargetRef
from rcp.history import HistoryManager
from rcp.server_ops.backup_capture import (
    BackupCaptureCoordinator,
    BackupKeptArtifactReference,
    BackupKeptResultViewReference,
    BackupSnapshotProjectInventory,
    write_immutable_backup_receipt,
)
from rcp.server_ops.backup_checkout import BackupCheckoutHostUnavailable
from rcp.server_ops.backup_models import (
    BackupCheckoutRecoveryDescriptor,
    BackupManifestConfiguration,
    BackupRecoveryMachine,
    BackupRecoveryRepository,
)
from rcp.server_ops.backup_project_files import (
    BackupProjectFileCaptureCoordinator,
    read_backup_project_file_capture_receipt,
)
from rcp.server_ops.github import parse_github_repository_ref
from rcp.server_runtime import ServerMetadata
from rcp.service import canonical_chat_backup_sources, iter_canonical_chat_backup_prefix
from rcp.storage import AppStore
from rcp.transport import SSHStateWorkspace, StateUnavailable
from rcp.transport.remote_backup_checkout import CheckoutInspectionError, inspect_checkout
from rcp.transport.remote_backup_inventory import inspect_direct_root

SOURCE_COMMIT = "a" * 40
WEB_BUILD_ID = "sha256:" + ("b" * 64)
FINGERPRINT = "SHA256:" + ("A" * 43)


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _metadata(data_dir: Path) -> ServerMetadata:
    return ServerMetadata.create(
        data_dir,
        host="127.0.0.1",
        port=8421,
        owner_kind="cli",
        control_socket=data_dir / "control.sock",
        running_commit=SOURCE_COMMIT,
        web_build_id=WEB_BUILD_ID,
    )


def _initialize_git_repository(path: Path) -> str:
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.name", "RCP Backup Test")
    _git(path, "config", "user.email", "backup@example.test")
    (path / "README.md").write_text("backup source\n", encoding="utf-8")
    _git(path, "add", "README.md")
    _git(path, "commit", "-q", "-m", "initial")
    _git(path, "remote", "add", "origin", "git@github.com:openai/rcp.git")
    _git(path, "config", "remote.origin.pushurl", "git@github.com:openai/rcp.git")
    return _git(path, "rev-parse", "HEAD")


def _manifest_document(
    repository: Path,
    *,
    account: str,
    host: str = "",
) -> str:
    claude_root = repository.parent / "claude-history"
    codex_root = repository.parent / "codex-history"
    return f'''name = "Backup project"

[[machines]]
alias = "worker"
host = "{host}"
os_account = "{account}"

[[repositories]]
alias = "paper"
machine = "worker"
path = "{repository}"

[project]
truth_scope = ["paper"]

[state]
repository = "paper"

[agent]
default_run_truth_scope = ["paper"]

[sources]
claude_roots = ["{claude_root}"]
codex_roots = ["{codex_root}"]

[execution]
run_on = "worker"

[paper.coach]
default_provider = "codex"
default_model = ""
default_reasoning = "medium"
'''


def _project_inventory(
    tmp_path: Path,
    *,
    project_id: str,
    task_id: str,
    with_files: bool,
    host: str = "",
) -> tuple[BackupSnapshotProjectInventory, Path]:
    account = pwd.getpwuid(os.geteuid()).pw_name
    central_root = tmp_path / f"central-{project_id}"
    repository = central_root / project_id / "repositories" / "paper"
    if not host:
        commit = _initialize_git_repository(repository)
        manifest_path = repository / ".research" / "manifest.toml"
    else:
        commit = "c" * 40
        manifest_path = tmp_path / f"bootstrap-{project_id}" / "manifest.toml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        _manifest_document(repository, account=account, host=host),
        encoding="utf-8",
    )
    manifest = load_manifest(manifest_path)
    if not host:
        HistoryManager(manifest).initialize()
    configuration = BackupManifestConfiguration.from_manifest(manifest)
    recovery = BackupCheckoutRecoveryDescriptor(
        request_id=str(uuid.uuid4()),
        project_id=project_id,
        home_space_id="00000000-0000-4000-8000-000000000001",
        completed_at=datetime(2026, 8, 29, 12, 0, tzinfo=UTC),
        final_review_digest="d" * 64,
        configuration=configuration,
        configuration_sha256=configuration.sha256,
        machines=(
            BackupRecoveryMachine(
                alias="worker",
                location="ssh" if host else "local",
                host=host,
                os_account=account,
                resolved_central_root=str(central_root),
            ),
        ),
        repositories=(
            BackupRecoveryRepository(
                alias="paper",
                repository=parse_github_repository_ref("git@github.com:openai/rcp.git"),
                machine_alias="worker",
                resolved_path=str(repository),
                git_commit=commit,
                deploy_key_label=(f"rcp:00000000-0000-4000-8000-000000000001:{project_id}:paper"),
                public_key_fingerprint=FINGERPRINT,
            ),
        ),
    )
    kept_artifacts = ()
    kept_views = ()
    if with_files:
        research = repository / ".research"
        branch_id = str(uuid.uuid4())
        branch_root = research / "branches" / branch_id
        (branch_root / "patches").mkdir(parents=True)
        (branch_root / "merges").mkdir()
        branch = GraphBranchMetadata(
            branch_id=branch_id,
            episode_id=branch_id,
            project_id=project_id,
            base_head=GraphHeadRef(revision=0),
            head=GraphHeadRef(
                target=GraphTargetRef(kind="branch", branch_id=branch_id),
                revision=0,
            ),
            authorized_by=AuthorizedHuman(
                space_id="00000000-0000-4000-8000-000000000001",
                user_id="00000000-0000-4000-8000-000000000002",
                display_name="Backup test",
            ),
        )
        (branch_root / "branch.json").write_text(
            branch.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        (branch_root / "merges" / f"{'a' * 64}.json").write_text(
            "{}\n",
            encoding="utf-8",
        )
        (branch_root / "graph.json").write_text("derived branch graph\n", encoding="utf-8")
        chat_id = str(uuid.uuid4())
        later_operation = str(uuid.uuid4())
        first = {
            "sessionId": chat_id,
            "nodeId": None,
            "chatScope": "project",
            "timestamp": "2026-08-29T12:00:00+00:00",
            "uuid": str(uuid.uuid4()),
            "operationId": task_id,
            "type": "user",
            "role": "user",
            "text": "captured turn",
        }
        later = {
            **first,
            "timestamp": "2026-08-29T12:01:00+00:00",
            "uuid": str(uuid.uuid4()),
            "operationId": later_operation,
            "type": "assistant",
            "role": "assistant",
            "text": "created after SQLite snapshot",
        }
        chat_root = research / "chat"
        chat_root.mkdir(exist_ok=True)
        (chat_root / f"project-{chat_id}.jsonl").write_text(
            json.dumps(first, separators=(",", ":"))
            + "\n"
            + json.dumps(later, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        paper_root = research / "paper"
        paper_root.mkdir(exist_ok=True)
        (paper_root / "introduction.md").write_text("# Captured paper\n", encoding="utf-8")
        facts_root = research / "facts" / "methods"
        facts_root.mkdir(parents=True)
        (facts_root / "protocol.txt").write_bytes(b"fact-bytes")
        artifacts = repository / "artifacts"
        artifacts.mkdir()
        artifact = b"kept artifact"
        (artifacts / "kept-figure.png").write_bytes(artifact)
        (artifacts / "unrelated.png").write_bytes(b"do not capture")
        views = repository / "views"
        views.mkdir()
        view = b"<html>kept view</html>"
        (views / "kept-result.html").write_bytes(view)
        (views / "unrelated.html").write_bytes(b"do not capture")
        kept_artifacts = (
            BackupKeptArtifactReference(
                operation_id=task_id,
                artifact_id="e" * 24,
                source_name="figure.png",
                media_type="image/png",
                expected_size_bytes=len(artifact),
                kept_filename="kept-figure.png",
                kept_at="2026-08-29T12:00:00+00:00",
            ),
        )
        kept_views = (
            BackupKeptResultViewReference(
                view_id="f" * 24,
                origin_operation_id=task_id,
                latest_operation_id=task_id,
                kept_filename="kept-result.html",
                content_sha256=hashlib.sha256(view).hexdigest(),
                size_bytes=len(view),
                kept_at="2026-08-29T12:00:00+00:00",
            ),
        )
    return (
        BackupSnapshotProjectInventory(
            project_id=project_id,
            home_space_id="00000000-0000-4000-8000-000000000001",
            locator=str(manifest_path),
            status="capturable",
            recovery=recovery,
            task_operation_ids=(task_id,),
            kept_artifacts=kept_artifacts,
            kept_result_views=kept_views,
        ),
        repository,
    )


def _sqlite_capture_with_projects(
    data_dir: Path,
    inventories: tuple[BackupSnapshotProjectInventory, ...],
) -> tuple[Path, str]:
    store, _ = AppStore.initialize_team_space(data_dir / "rcp.sqlite3", "File capture lab")
    publication = BackupCaptureCoordinator(store, data_dir, _metadata(data_dir)).capture_sqlite()
    projects = tuple(
        sorted(
            (_inventory_for_space(inventory, space_id=store.space_id) for inventory in inventories),
            key=lambda item: item.project_id,
        )
    )
    receipt = publication.receipt.model_copy(update={"projects": projects})
    publication.receipt_path.chmod(0o600)
    publication.receipt_path.unlink()
    digest = write_immutable_backup_receipt(publication.receipt_path, receipt)
    return publication.receipt_path, digest


def _inventory_for_space(
    inventory: BackupSnapshotProjectInventory,
    *,
    space_id: str,
) -> BackupSnapshotProjectInventory:
    recovery = inventory.recovery
    assert recovery is not None
    recovery_document = recovery.model_dump(mode="python")
    recovery_document["home_space_id"] = space_id
    recovery_document["repositories"] = tuple(
        {
            **repository.model_dump(mode="python"),
            "deploy_key_label": (f"rcp:{space_id}:{inventory.project_id}:{repository.alias}"),
        }
        for repository in recovery.repositories
    )
    inventory_document = inventory.model_dump(mode="python")
    inventory_document["home_space_id"] = space_id
    inventory_document["recovery"] = BackupCheckoutRecoveryDescriptor.model_validate(
        recovery_document
    )
    return BackupSnapshotProjectInventory.model_validate(inventory_document)


def test_project_file_capture_selects_only_typed_sources_and_chat_snapshot_prefix(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    project_id = str(uuid.uuid4())
    task_id = str(uuid.uuid4())
    inventory, _ = _project_inventory(
        tmp_path,
        project_id=project_id,
        task_id=task_id,
        with_files=True,
    )
    receipt_path, receipt_sha256 = _sqlite_capture_with_projects(data_dir, (inventory,))

    publication = BackupProjectFileCaptureCoordinator(data_dir).capture(
        receipt_path,
        expected_sha256=receipt_sha256,
    )
    receipt = read_backup_project_file_capture_receipt(
        publication.receipt_path,
        expected_sha256=publication.receipt_sha256,
    )

    assert receipt.status == "complete"
    project = receipt.projects[0]
    assert project.status == "captured"
    assert len(project.branch_heads) == 1
    groups = {entry.group for entry in project.files}
    assert groups == {
        "canonical",
        "chat",
        "paper_introduction",
        "fact",
        "kept_artifact",
        "legacy_kept_result_view",
    }
    relative_paths = {entry.source_relative_path for entry in project.files}
    assert "artifacts/unrelated.png" not in relative_paths
    assert "views/unrelated.html" not in relative_paths
    assert ".research/graph.json" not in relative_paths
    assert not any(path.endswith("/graph.json") for path in relative_paths)
    assert any(path.endswith(f"/merges/{'a' * 64}.json") for path in relative_paths)
    chat_entry = next(entry for entry in project.files if entry.group == "chat")
    chat_bytes = (receipt_path.parent / chat_entry.archive_path).read_bytes()
    assert b"captured turn" in chat_bytes
    assert b"created after SQLite snapshot" not in chat_bytes
    for entry in project.files:
        captured = receipt_path.parent / entry.archive_path
        assert captured.is_file()
        assert hashlib.sha256(captured.read_bytes()).hexdigest() == entry.sha256


def test_one_unavailable_ssh_project_does_not_spoil_a_healthy_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    healthy_id = str(uuid.uuid4())
    remote_id = str(uuid.uuid4())
    healthy, _ = _project_inventory(
        tmp_path,
        project_id=healthy_id,
        task_id=str(uuid.uuid4()),
        with_files=False,
    )
    remote, _ = _project_inventory(
        tmp_path,
        project_id=remote_id,
        task_id=str(uuid.uuid4()),
        with_files=False,
        host="unreachable.example",
    )
    receipt_path, receipt_sha256 = _sqlite_capture_with_projects(
        data_dir,
        (healthy, remote),
    )
    original = project_files.verify_checkout_identities

    def fail_remote(recovery: BackupCheckoutRecoveryDescriptor) -> None:
        if recovery.machines[0].location == "ssh":
            raise BackupCheckoutHostUnavailable(
                "host unavailable",
                machine_alias=recovery.machines[0].alias,
            )
        original(recovery)

    monkeypatch.setattr(project_files, "verify_checkout_identities", fail_remote)

    receipt = (
        BackupProjectFileCaptureCoordinator(data_dir)
        .capture(
            receipt_path,
            expected_sha256=receipt_sha256,
        )
        .receipt
    )

    projects = {project.project_id: project for project in receipt.projects}
    assert {project_id: project.status for project_id, project in projects.items()} == {
        healthy_id: "captured",
        remote_id: "uncaptured",
    }
    assert projects[remote_id].unavailable_kind == "remote_unreachable"
    assert projects[remote_id].recovery is not None
    assert projects[remote_id].recovery.project_id == remote_id
    assert receipt.status == "partial"


def test_only_an_ssh_transport_failure_is_classified_as_host_unreachable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory, _ = _project_inventory(
        tmp_path,
        project_id=str(uuid.uuid4()),
        task_id=str(uuid.uuid4()),
        with_files=False,
        host="unreachable.example",
    )
    recovery = inventory.recovery
    assert recovery is not None
    results = iter(
        (
            subprocess.CompletedProcess(("ssh",), 255, "", "route unavailable"),
            subprocess.CompletedProcess(("ssh",), 1, "", "checkout identity invalid"),
        )
    )
    monkeypatch.setattr(backup_checkout.subprocess, "run", lambda *_args, **_kwargs: next(results))

    with pytest.raises(BackupCheckoutHostUnavailable):
        backup_checkout.verify_checkout_identities(recovery)
    with pytest.raises(CheckoutInspectionError) as invalid:
        backup_checkout.verify_checkout_identities(recovery)
    assert not isinstance(invalid.value, BackupCheckoutHostUnavailable)


def test_continued_fact_replacement_marks_only_that_project_uncaptured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    project_id = str(uuid.uuid4())
    inventory, repository = _project_inventory(
        tmp_path,
        project_id=project_id,
        task_id=str(uuid.uuid4()),
        with_files=False,
    )
    facts = repository / ".research" / "facts"
    facts.mkdir(exist_ok=True)
    fact = facts / "moving.txt"
    fact.write_bytes(b"fact-churn-0")
    receipt_path, receipt_sha256 = _sqlite_capture_with_projects(data_dir, (inventory,))

    requests: queue.Queue[None] = queue.Queue()
    completions: queue.Queue[None] = queue.Queue()
    writer_error: list[BaseException] = []

    def writer() -> None:
        try:
            for index in range(1, 4):
                requests.get(timeout=5)
                replacement = fact.with_name(f".moving-{index}.tmp")
                replacement.write_bytes(f"fact-churn-{index}".encode())
                os.replace(replacement, fact)
                completions.put(None)
        except BaseException as exc:  # pragma: no cover - surfaced below
            writer_error.append(exc)

    thread = threading.Thread(target=writer, name="backup-fact-writer")
    thread.start()
    original_write = project_io._write_all

    def replace_during_copy(descriptor: int, data: bytes) -> None:
        if data.startswith(b"fact-churn"):
            requests.put(None)
            completions.get(timeout=5)
        original_write(descriptor, data)

    monkeypatch.setattr(project_io, "_write_all", replace_during_copy)
    try:
        receipt = (
            BackupProjectFileCaptureCoordinator(data_dir)
            .capture(
                receipt_path,
                expected_sha256=receipt_sha256,
            )
            .receipt
        )
    finally:
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert writer_error == []
    assert receipt.projects[0].status == "uncaptured"
    assert receipt.status == "partial"
    assert not (receipt_path.parent / "projects" / project_id).exists()


def test_forbidden_fact_subtree_marks_that_project_uncaptured(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    project_id = str(uuid.uuid4())
    inventory, repository = _project_inventory(
        tmp_path,
        project_id=project_id,
        task_id=str(uuid.uuid4()),
        with_files=False,
    )
    (repository / ".research" / "facts" / "credentials").mkdir()
    receipt_path, receipt_sha256 = _sqlite_capture_with_projects(data_dir, (inventory,))

    receipt = (
        BackupProjectFileCaptureCoordinator(data_dir)
        .capture(
            receipt_path,
            expected_sha256=receipt_sha256,
        )
        .receipt
    )

    assert receipt.projects[0].status == "uncaptured"
    assert receipt.status == "partial"


def test_checkout_inspection_rejects_changed_origin_and_push_url(tmp_path: Path) -> None:
    repository = tmp_path / "checkout"
    commit = _initialize_git_repository(repository)
    account = pwd.getpwuid(os.geteuid()).pw_name

    proof = inspect_checkout(
        os_account=account,
        repository_path=str(repository),
        expected_origin="git@github.com:openai/rcp.git",
        recorded_commit=commit,
    )
    assert proof["head"] == commit

    _git(repository, "remote", "set-url", "origin", "git@github.com:openai/other.git")
    with pytest.raises(CheckoutInspectionError):
        inspect_checkout(
            os_account=account,
            repository_path=str(repository),
            expected_origin="git@github.com:openai/rcp.git",
            recorded_commit=commit,
        )

    _git(repository, "remote", "set-url", "origin", "git@github.com:openai/rcp.git")
    _git(
        repository,
        "config",
        "remote.origin.pushurl",
        "git@github.com:openai/other.git",
    )
    with pytest.raises(CheckoutInspectionError):
        inspect_checkout(
            os_account=account,
            repository_path=str(repository),
            expected_origin="git@github.com:openai/rcp.git",
            recorded_commit=commit,
        )


def test_chat_append_after_the_observed_byte_boundary_is_absent(tmp_path: Path) -> None:
    project_id = str(uuid.uuid4())
    task_id = str(uuid.uuid4())
    chat_id = str(uuid.uuid4())
    research = tmp_path / ".research"
    chat_root = research / "chat"
    chat_root.mkdir(parents=True)
    path = chat_root / f"project-{chat_id}.jsonl"
    record = {
        "sessionId": chat_id,
        "nodeId": None,
        "chatScope": "project",
        "timestamp": "2026-08-29T12:00:00+00:00",
        "uuid": str(uuid.uuid4()),
        "operationId": task_id,
        "type": "user",
        "role": "user",
        "text": "inside boundary",
    }
    path.write_text(json.dumps(record, separators=(",", ":")) + "\n", encoding="utf-8")
    source = canonical_chat_backup_sources(research)[0]
    appended = {
        **record,
        "timestamp": "2026-08-29T12:01:00+00:00",
        "uuid": str(uuid.uuid4()),
        "type": "assistant",
        "role": "assistant",
        "text": "after boundary",
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(appended, separators=(",", ":")) + "\n")

    captured = b"".join(
        iter_canonical_chat_backup_prefix(
            source,
            project_id=project_id,
            operation_projects={task_id: project_id},
        )
    )

    assert b"inside boundary" in captured
    assert b"after boundary" not in captured


def test_remote_backup_export_is_filtered_lock_free_and_root_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "export"
    destination.mkdir(mode=0o700)
    workspace = SSHStateWorkspace(
        tmp_path / "cache" / ".research",
        "research.example",
        "/srv/rcp/project/repositories/paper",
    )
    inventory = json.dumps(
        [
            {"kind": "directory", "name": "branches"},
            {"kind": "directory", "name": "chat"},
            {"kind": "file", "name": "graph.json"},
            {"kind": "file", "name": "manifest.toml"},
            {"kind": "directory", "name": "patches"},
        ],
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    remote_calls: list[list[str]] = []

    def remote(arguments: list[str], *, timeout: float = 30):
        remote_calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, inventory, b"")

    rsync_calls: list[list[str]] = []

    def rsync(arguments: list[str], **_kwargs):
        rsync_calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(workspace, "_ssh", remote)
    monkeypatch.setattr(state_module.subprocess, "run", rsync)

    assert workspace.backup_source_root(destination) == destination
    assert len(remote_calls) == 2
    assert all(call[:2] == ["python3", "-c"] for call in remote_calls)
    assert len(rsync_calls) == 1
    command = rsync_calls[0]
    assert "--include=/patches/***" in command
    assert "--include=/branches/***" in command
    assert "--include=/chat/***" in command
    assert "--exclude=/branches/*/graph.json" in command
    assert "--exclude=.refresh.lock" in command
    assert not any("lock" in argument and argument.startswith("mkdir") for argument in command)


def test_remote_backup_export_treats_a_proven_missing_root_as_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "export"
    destination.mkdir(mode=0o700)
    workspace = SSHStateWorkspace(
        tmp_path / "cache" / ".research",
        "research.example",
        "/srv/rcp/project/repositories/paper",
    )
    remote_calls: list[list[str]] = []

    def remote(arguments: list[str], *, timeout: float = 30):
        remote_calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, b"[]", b"")

    monkeypatch.setattr(workspace, "_ssh", remote)
    monkeypatch.setattr(
        state_module.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("an empty remote root must not invoke rsync"),
    )

    assert workspace.backup_source_root(destination) == destination
    assert len(remote_calls) == 2
    assert list(destination.iterdir()) == []
    assert inspect_direct_root(str(tmp_path / "missing" / ".research")) == []


def test_remote_backup_export_rejects_an_unknown_direct_root_before_rsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "export"
    destination.mkdir(mode=0o700)
    workspace = SSHStateWorkspace(
        tmp_path / "cache" / ".research",
        "research.example",
        "/srv/rcp/project/repositories/paper",
    )
    inventory = b'[{"kind":"directory","name":"future-durable-data"}]'
    monkeypatch.setattr(
        workspace,
        "_ssh",
        lambda arguments, timeout=30: subprocess.CompletedProcess(
            arguments,
            0,
            inventory,
            b"",
        ),
    )

    def unexpected_rsync(*_args, **_kwargs):
        raise AssertionError("rsync must not run for an unclassified root")

    monkeypatch.setattr(state_module.subprocess, "run", unexpected_rsync)
    with pytest.raises(StateUnavailable, match="unclassified"):
        workspace.backup_source_root(destination)
