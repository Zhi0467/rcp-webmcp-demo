from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

import pytest

from rcp.limits import REMOTE_SOURCE_OPERATION_TIMEOUT_SECONDS
from rcp.runs.tasks.graph import (
    _continuation_graph_context,
    _stage_graph_context,
    _stage_prepared_graph_context,
    _try_reuse_graph_context,
)
from rcp.service import RunRequest
from rcp.storage import AgentTaskRecord
from rcp.transfer import TransferArchiveEntry
from rcp.transport import RemoteRunStage, StateUnavailable

from .helpers import create_named_app as create_app


def _publish_imported_source(service, tmp_path: Path) -> tuple[object, bytes, str]:
    assert service.imported_sources is not None
    content = b'{"type":"assistant","message":"transferred evidence"}\n'
    digest = hashlib.sha256(content).hexdigest()
    capture_root = tmp_path / "capture"
    source = capture_root / "provider-history" / "codex" / digest
    source.parent.mkdir(parents=True)
    source.write_bytes(content)
    service.imported_sources.publish(
        capture_root,
        (
            TransferArchiveEntry(
                archive_path=f"provider-history/codex/{digest}",
                group="provider_history",
                sha256=digest,
                size_bytes=len(content),
            ),
        ),
    )
    inventory = service.imported_sources.inventory()
    return inventory, content, digest


def _local_remote_stage(monkeypatch: pytest.MonkeyPatch) -> RemoteRunStage:
    root = Path(tempfile.mkdtemp(prefix="rcp-run.", dir="/tmp"))
    (root / "inputs").mkdir()
    (root / "workspace").mkdir()
    stage = RemoteRunStage("research.example")
    stage.root = PurePosixPath(str(root))
    real_run = subprocess.run

    def fake_run(arguments, **_kwargs):
        if arguments[0] == "rsync":
            source = Path(arguments[-2].rstrip("/"))
            destination = Path(arguments[-1].split(":", 1)[1].rstrip("/"))
            shutil.copytree(source, destination)
            return subprocess.CompletedProcess(arguments, 0, "", "")
        return real_run(arguments, capture_output=True, text=True, check=False)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        RemoteRunStage,
        "_ssh",
        lambda _self, arguments: real_run(
            arguments,
            capture_output=True,
            text=True,
            check=False,
        ),
    )
    monkeypatch.setattr(
        RemoteRunStage,
        "_ssh_bytes",
        lambda _self, arguments, *, input_data=None, timeout_seconds=None: real_run(
            arguments,
            capture_output=True,
            input=input_data,
            check=False,
        ),
    )
    return stage


def test_remote_graph_context_stages_only_project_owned_imports(
    manifest,
    tmp_path: Path,
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    inventory, content, digest = _publish_imported_source(service, tmp_path)
    context = service.assemble_run(
        RunRequest(run_truth_scope=["repo-a"]),
        surface="refresh",
        imported_source_inventory=inventory,
    )
    native_marker = Path(context.source_roots["codex"][0]) / "native-only.jsonl"
    native_marker.write_text("native provider home\n", encoding="utf-8")
    stage = RemoteRunStage("research.example")
    stage.root = PurePosixPath("/tmp/rcp-run.imported-context")

    try:
        staged = _stage_graph_context(
            context,
            service,
            stage,
            "laptop",
            imported_source_inventory=inventory,
        )

        expected_root = "/tmp/rcp-run.imported-context/inputs/imported-provider-history/codex"
        assert staged.source_roots == context.source_roots
        assert staged.imported_source_roots == {"codex": [expected_root]}
        assert staged.imported_source_fingerprint == inventory.fingerprint
        assert stage._pending_inputs is not None
        copied = stage._pending_inputs / "imported-provider-history" / "codex" / digest
        assert copied.read_bytes() == content
        assert not any(path.name == native_marker.name for path in stage._pending_inputs.rglob("*"))
    finally:
        stage._clear_pending_inputs()


def test_remote_stage_reads_back_exact_immutable_imported_inventory(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    inventory, _content, digest = _publish_imported_source(service, tmp_path)
    assert service.imported_sources is not None
    stage = _local_remote_stage(monkeypatch)

    try:
        stage.put_imported_provider_sources(
            service.imported_sources,
            inventory,
            "imported-provider-history",
        )
        stage.finalize_inputs()

        readback = stage.verify_imported_provider_sources(
            inventory,
            "imported-provider-history",
        )

        assert readback.fingerprint == inventory.fingerprint
        assert readback.file_count == 1
        assert readback.payload_size_bytes == inventory.payload_size_bytes
        assert stage.root is not None
        staged_file = (
            Path(str(stage.root)) / "inputs" / "imported-provider-history" / "codex" / digest
        )
        staged_file.chmod(0o600)
        with pytest.raises(ValueError, match="immutable regular file"):
            stage.verify_imported_provider_sources(
                inventory,
                "imported-provider-history",
            )
    finally:
        stage.close()


def test_remote_stage_sends_imported_inventory_over_bounded_stdin(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    inventory, _content, _digest = _publish_imported_source(app.state.service, tmp_path)
    stage = RemoteRunStage("research.example")
    stage.root = PurePosixPath("/tmp/rcp-run.imported-stdin")
    captured = {}

    def fake_ssh_bytes(arguments, *, input_data=None, timeout_seconds=None):
        captured.update(
            {
                "arguments": arguments,
                "input_data": input_data,
                "timeout_seconds": timeout_seconds,
            }
        )
        payload = {
            "fingerprint": inventory.fingerprint,
            "file_count": len(inventory.files),
            "payload_size_bytes": inventory.payload_size_bytes,
        }
        return subprocess.CompletedProcess(arguments, 0, json.dumps(payload).encode(), b"")

    monkeypatch.setattr(stage, "_ssh_bytes", fake_ssh_bytes)

    readback = stage.verify_imported_provider_sources(
        inventory,
        "imported-provider-history",
    )

    assert readback.fingerprint == inventory.fingerprint
    assert (
        captured["input_data"]
        == json.dumps(
            [item.model_dump(mode="json") for item in inventory.files],
            separators=(",", ":"),
        ).encode()
    )
    assert inventory.files[0].sha256 not in captured["arguments"]
    assert captured["timeout_seconds"] == REMOTE_SOURCE_OPERATION_TIMEOUT_SECONDS


def test_remote_prepared_context_keeps_ssh_outage_distinct_from_checkpoint_drift(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    project_id = app.state.default_project_id
    assert project_id is not None
    now = "2026-08-31T12:00:00+00:00"
    record = AgentTaskRecord(
        operation_id="parent",
        project_id=project_id,
        kind="refresh",
        status="failed",
        request={"provider": "codex"},
        created_at=now,
        updated_at=now,
        status_message="failed",
        native_session_id="native-session",
        stage_host="research.example",
        stage_root="/tmp/rcp-run.unreachable",
    )

    class Store:
        def agent_task(self, _operation_id):
            return record

        def record_agent_task_receipt(self, *_args, **_kwargs):
            raise AssertionError("an SSH outage must not invalidate the checkpoint")

    execution = type(
        "Execution",
        (),
        {
            "operation_id": "parent",
            "store": Store(),
            "reuses_native_checkpoint": True,
        },
    )()
    monkeypatch.setattr(RemoteRunStage, "directory_exists", lambda *_args: None)

    with pytest.raises(StateUnavailable, match="could not reach"):
        _continuation_graph_context(
            service,
            execution,
            kind="refresh",
            request=RunRequest(run_truth_scope=["repo-a"]),
            execution_host="research.example",
            imported_source_inventory=service.imported_source_inventory(
                "refresh",
                manifest.machine_map["laptop"],
            ),
        )


@pytest.mark.parametrize("corruption", ["changed", "missing"])
def test_remote_resume_verifies_imported_checkpoint_and_clean_retry_rebuilds_on_drift(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    project_id = app.state.default_project_id
    assert project_id is not None
    inventory, _content, digest = _publish_imported_source(service, tmp_path)
    context = service.assemble_run(
        RunRequest(run_truth_scope=["repo-a"]),
        surface="refresh",
        imported_source_inventory=inventory,
    )
    stage = _local_remote_stage(monkeypatch)
    assert stage.root is not None
    staged = _stage_graph_context(
        context,
        service,
        stage,
        "laptop",
        imported_source_inventory=inventory,
    )
    stage.finalize_inputs()
    _stage_prepared_graph_context(
        None,
        stage,
        project_id=project_id,
        kind="refresh",
        graph_revision=staged.graph_revision,
        execution_host="research.example",
        original_contract_path=str(stage.root / "inputs" / "task.md"),
        context=staged,
    )
    stage.finalize_inputs()
    now = "2026-08-31T12:00:00+00:00"
    record = AgentTaskRecord(
        operation_id="parent",
        project_id=project_id,
        kind="refresh",
        status="failed",
        request={"provider": "codex"},
        created_at=now,
        updated_at=now,
        status_message="failed",
        native_session_id="native-session",
        stage_host="research.example",
        stage_root=str(stage.root),
    )

    class Store:
        def __init__(self) -> None:
            self.receipts = []
            self.retry = AgentTaskRecord(
                operation_id="retry",
                project_id=project_id,
                kind="refresh",
                status="running",
                request={"provider": "codex"},
                created_at=now,
                updated_at=now,
                status_message="running",
                parent_operation_id="parent",
            )

        def agent_task(self, operation_id):
            return self.retry if operation_id == "retry" else record

        def agent_task_patch_output(self, _operation_id):
            return None

        def record_agent_task_receipt(self, operation_id, category, payload, **_kwargs):
            self.receipts.append((operation_id, category, payload))

        def record_agent_task_event(self, *_args, **_kwargs):
            pass

    store = Store()
    resume_execution = type(
        "Execution",
        (),
        {
            "operation_id": "parent",
            "store": store,
            "reuses_native_checkpoint": True,
        },
    )()
    retry_execution = type(
        "Execution",
        (),
        {
            "operation_id": "retry",
            "store": store,
            "reuses_native_checkpoint": False,
        },
    )()
    retry_stage: RemoteRunStage | None = None

    try:
        resumed = _continuation_graph_context(
            service,
            resume_execution,
            kind="refresh",
            request=RunRequest(run_truth_scope=["repo-a"]),
            execution_host="research.example",
            imported_source_inventory=inventory,
        )

        assert resumed.context.imported_source_roots == staged.imported_source_roots
        assert any(
            category == "imported_source_stage_verified" and payload["reused_checkpoint"]
            for _operation_id, category, payload in store.receipts
        )

        clean_retry = _try_reuse_graph_context(
            service,
            retry_execution,
            kind="refresh",
            request=RunRequest(run_truth_scope=["repo-a"]),
            execution_host="research.example",
            imported_source_inventory=inventory,
        )
        assert clean_retry is not None
        assert clean_retry.prepared is not None
        retry_root = Path(tempfile.mkdtemp(prefix="rcp-run.", dir="/tmp"))
        (retry_root / "inputs").mkdir()
        (retry_root / "workspace").mkdir()
        retry_stage = RemoteRunStage("research.example")
        retry_stage.root = PurePosixPath(str(retry_root))
        retry_context = _stage_graph_context(
            clean_retry.prepared.context,
            service,
            retry_stage,
            "laptop",
            imported_source_inventory=inventory,
        )
        retry_stage.finalize_inputs()
        _stage_prepared_graph_context(
            None,
            retry_stage,
            project_id=project_id,
            kind="refresh",
            graph_revision=retry_context.graph_revision,
            execution_host="research.example",
            original_contract_path=str(retry_stage.root / "inputs" / "task.md"),
            context=retry_context,
        )
        retry_stage.finalize_inputs()
        store.retry = store.retry.model_copy(
            update={
                "stage_host": "research.example",
                "stage_root": str(retry_stage.root),
                "native_session_id": "retry-native-session",
            }
        )
        retry_resume_execution = type(
            "Execution",
            (),
            {
                "operation_id": "retry",
                "store": store,
                "reuses_native_checkpoint": True,
            },
        )()
        resumed_retry = _continuation_graph_context(
            service,
            retry_resume_execution,
            kind="refresh",
            request=RunRequest(run_truth_scope=["repo-a"]),
            execution_host="research.example",
            imported_source_inventory=inventory,
        )
        assert resumed_retry.context.imported_source_roots == {
            "codex": [f"{retry_stage.root}/inputs/imported-provider-history/codex"]
        }

        staged_file = (
            Path(str(stage.root)) / "inputs" / "imported-provider-history" / "codex" / digest
        )
        if corruption == "changed":
            staged_file.chmod(0o600)
            staged_file.write_bytes(b"changed")
            staged_file.chmod(0o400)
        else:
            staged_file.parent.chmod(0o700)
            staged_file.unlink()
            staged_file.parent.chmod(0o500)

        with pytest.raises(ValueError, match="Retry this task"):
            _continuation_graph_context(
                service,
                resume_execution,
                kind="refresh",
                request=RunRequest(run_truth_scope=["repo-a"]),
                execution_host="research.example",
                imported_source_inventory=inventory,
            )

        retry = _try_reuse_graph_context(
            service,
            retry_execution,
            kind="refresh",
            request=RunRequest(run_truth_scope=["repo-a"]),
            execution_host="research.example",
            imported_source_inventory=inventory,
        )
        assert retry is not None
        assert retry.prepared is None
        assert "inventory" in (retry.context_reason or "")
    finally:
        if retry_stage is not None:
            retry_stage.close()
        stage.close()
