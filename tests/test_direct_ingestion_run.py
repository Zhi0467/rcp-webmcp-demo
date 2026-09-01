from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import pytest
from fastapi.testclient import TestClient

import rcp.runs.tasks.graph as graph_run
from rcp.agents import AgentEvent
from rcp.runs.experiment_loop import experiment_loop_semantic_ending
from rcp.runs.tasks.graph import (
    _agent_read_dirs,
    _continuation_graph_context,
    _record_context_receipt,
    _stage_graph_context,
    _stage_prepared_graph_context,
    _try_reuse_graph_context,
    stream_graph_run,
)
from rcp.service import RunRequest
from rcp.storage import AgentTaskRecord
from rcp.transfer import TransferArchiveEntry

from .helpers import (
    agent_patch_json,
    append_fixture_patch,
    refresh_patch,
    seed_patch,
    wait_for_task_response,
)
from .helpers import create_named_app as create_app


class _FailingLauncher:
    def __init__(self) -> None:
        self.workspace: Path | None = None

    async def stream(self, *_args, **kwargs):
        self.workspace = kwargs["cwd"]
        yield AgentEvent(event="error", text="provider failed before writing a patch")


class _SuccessfulIngestLauncher:
    def __init__(self, patch_text: str, answer: str) -> None:
        self.patch_text = patch_text
        self.answer = answer

    async def stream(self, *_args, **kwargs):
        workspace = Path(kwargs["cwd"])
        (workspace / "patch.json").write_text(self.patch_text, encoding="utf-8")
        yield AgentEvent(event="answer", text=self.answer)
        yield AgentEvent(event="done")


@pytest.mark.parametrize("kind", ["seed", "refresh"])
def test_successful_ingest_answer_is_persisted_and_readable(
    manifest,
    tmp_path,
    kind: str,
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    patch = seed_patch() if kind == "seed" else refresh_patch()
    if kind == "refresh":
        append_fixture_patch(service, seed_patch())
    answer = (
        "Wrote the graph Patch. Hypothesis scope remains empty because no cited excerpt states "
        "the exact boundary."
    )
    launcher = _SuccessfulIngestLauncher(agent_patch_json(patch), answer)

    async def stream(_project_id, task_kind, request, execution):
        async for frame in stream_graph_run(
            service,
            launcher,
            task_kind,
            request,
            tmp_path / "data",
            execution=execution,
        ):
            yield frame

    app.state.background_tasks.stream = stream
    project_id = app.state.default_project_id
    assert project_id is not None
    with TestClient(app) as client:
        started = client.post(
            f"/api/projects/{project_id}/tasks/{kind}",
            json={"run_truth_scope": ["repo-a"]},
        )
        assert started.status_code == 202
        completed = wait_for_task_response(client, project_id, started.json()["operation_id"])

    assert completed["status"] == "succeeded"
    assert completed["result"] == {"messages": [answer]}
    assert completed["applied_revision"] == (2 if kind == "seed" else 3)


def test_failed_ingest_keeps_its_independent_answer(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    answer = "Wrote the attempted Patch, but the active ontology lacks a needed relation."
    invalid_patch = json.dumps(
        {
            "summary": "Used an operation outside the agent schema.",
            "ops": [{"op": "invent_nodes", "nodes": []}],
        }
    )
    launcher = _SuccessfulIngestLauncher(invalid_patch, answer)

    async def stream(_project_id, task_kind, request, execution):
        async for frame in stream_graph_run(
            service,
            launcher,
            task_kind,
            request,
            tmp_path / "data",
            execution=execution,
        ):
            yield frame

    app.state.background_tasks.stream = stream
    project_id = app.state.default_project_id
    assert project_id is not None
    with TestClient(app) as client:
        started = client.post(
            f"/api/projects/{project_id}/tasks/seed",
            json={"run_truth_scope": ["repo-a"]},
        )
        assert started.status_code == 202
        failed = wait_for_task_response(client, project_id, started.json()["operation_id"])

    assert failed["status"] == "failed"
    assert failed["result"] == {"messages": [answer]}
    assert failed["applied_revision"] is None


def test_queued_decision_is_an_explicit_experiment_loop_exit() -> None:
    base = {
        "summary": "Queued a research choice for the human.",
        "repositories_read": ["repo-a"],
        "change_summary": ["Queued the evaluation budget choice."],
    }
    for status in ("ready", "revisit"):
        queued = {
            **base,
            "ops": [
                {
                    "op": "update_nodes",
                    "nodes": [
                        {
                            "id": "dec/evaluation-budget",
                            "changes": {"status": status},
                        }
                    ],
                }
            ],
        }

        ending = experiment_loop_semantic_ending(json.dumps(queued), "exp/evaluation")
        assert ending is not None
        assert ending.ending == "human_pause"


def test_remote_graph_context_rebinds_metadata_without_staging_provider_logs(
    manifest, tmp_path
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    context = service.assemble_run(RunRequest(run_truth_scope=["repo-a"]), surface="seed")

    class NoTransferStage:
        root = PurePosixPath("/tmp/rcp-run.test")

        def put_file(self, *_args, **_kwargs):
            raise AssertionError("graph context must not stage files")

        def put_directory(self, *_args, **_kwargs):
            raise AssertionError("graph context must not stage directories")

    staged = _stage_graph_context(context, service, NoTransferStage(), "laptop")

    assert staged.source_roots == context.source_roots
    assert staged.ingestion_watermark == context.ingestion_watermark
    assert staged.graph_path == str(
        Path(manifest.repository_map["repo-a"].path) / ".research/graph.json"
    )


def test_graph_read_dirs_expose_each_provider_root_exactly(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    context = service.assemble_run(RunRequest(run_truth_scope=["repo-a"]), surface="seed")
    extra_root = tmp_path / "claude-extra"
    extra_root.mkdir()
    roots = {
        "claude": [manifest.sources.claude_roots[0], str(extra_root)],
        "codex": [manifest.sources.codex_roots[0]],
    }
    context = context.model_copy(update={"source_roots": roots})

    read_dirs = _agent_read_dirs(context, None, service, "laptop")

    for root in roots["claude"] + roots["codex"]:
        assert Path(root).expanduser() in read_dirs
    assert all("; " not in str(path) for path in read_dirs)


def test_prepared_context_and_receipt_preserve_project_watermark(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    watermark = datetime(2026, 7, 31, 14, 0, tzinfo=UTC)
    context = service.assemble_run(RunRequest(run_truth_scope=["repo-a"]), surface="refresh")
    context = context.model_copy(update={"ingestion_watermark": watermark})
    stage = tmp_path / "stage"
    stage.mkdir()

    _stage_prepared_graph_context(
        stage,
        None,
        project_id="project",
        kind="refresh",
        graph_revision=context.graph_revision,
        execution_host="",
        original_contract_path=str(stage / "inputs/task.md"),
        context=context,
    )

    payload = json.loads((stage / "inputs/prepared-context.json").read_text(encoding="utf-8"))
    assert payload["version"] == 2
    assert (
        datetime.fromisoformat(payload["context"]["ingestion_watermark"].replace("Z", "+00:00"))
        == watermark
    )
    assert payload["context"]["source_roots"] == context.source_roots

    class ReceiptStore:
        def __init__(self) -> None:
            self.payload = None

        def record_agent_task_receipt(self, _operation_id, category, payload, **_kwargs):
            assert category == "context_assembled"
            self.payload = payload

    store = ReceiptStore()
    execution = type("Execution", (), {"operation_id": "operation", "store": store})()
    _record_context_receipt(execution, context, surface="refresh")

    assert store.payload["source_root_count"] == sum(
        len(roots) for roots in context.source_roots.values()
    )
    assert store.payload["source_warnings"] == context.source_errors
    assert store.payload["graph_revision"] == context.graph_revision
    assert store.payload["ingestion_watermark"] == watermark.isoformat()


def test_clean_retry_reuses_prepared_metadata_without_inspecting_provider_logs(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    assert service.imported_sources is not None
    imported_bytes = b'{"type":"assistant","message":"imported evidence"}\n'
    imported_sha256 = hashlib.sha256(imported_bytes).hexdigest()
    capture_root = tmp_path / "capture"
    capture_path = capture_root / "provider-history" / "codex" / imported_sha256
    capture_path.parent.mkdir(parents=True)
    capture_path.write_bytes(imported_bytes)
    service.imported_sources.publish(
        capture_root,
        (
            TransferArchiveEntry(
                archive_path=f"provider-history/codex/{imported_sha256}",
                group="provider_history",
                sha256=imported_sha256,
                size_bytes=len(imported_bytes),
            ),
        ),
    )
    imported_inventory = service.imported_source_inventory(
        "refresh",
        manifest.machine_map["laptop"],
    )
    assert imported_inventory is not None
    context = service.assemble_run(RunRequest(run_truth_scope=["repo-a"]), surface="refresh")
    parent_stage = tmp_path / "parent-stage"
    parent_stage.mkdir()
    _stage_prepared_graph_context(
        parent_stage,
        None,
        project_id="project",
        kind="refresh",
        graph_revision=context.graph_revision,
        execution_host="",
        original_contract_path=str(parent_stage / "inputs/task.md"),
        context=context,
    )
    now = "2026-08-04T12:00:00+00:00"
    records = {
        "retry": AgentTaskRecord(
            operation_id="retry",
            project_id="project",
            kind="refresh",
            status="running",
            request={"provider": "codex"},
            created_at=now,
            updated_at=now,
            status_message="running",
            parent_operation_id="parent",
        ),
        "parent": AgentTaskRecord(
            operation_id="parent",
            project_id="project",
            kind="refresh",
            status="failed",
            request={"provider": "codex"},
            created_at=now,
            updated_at=now,
            status_message="failed",
            attempt=1,
            native_session_id="provider-native-session",
            stage_root=str(parent_stage),
        ),
    }

    class RetryStore:
        def __init__(self) -> None:
            self.receipts = []

        def agent_task(self, operation_id):
            return records.get(operation_id)

        def agent_task_patch_output(self, operation_id):
            assert operation_id == "parent"
            return '{"summary":"retained","ops":[]}'

        def record_agent_task_receipt(self, operation_id, category, payload, **_kwargs):
            self.receipts.append((operation_id, category, payload))

    retry_store = RetryStore()
    execution = type(
        "Execution",
        (),
        {
            "operation_id": "retry",
            "store": retry_store,
            "reuses_native_checkpoint": False,
        },
    )()
    monkeypatch.setattr(
        service,
        "index_snapshot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("retry must not index provider logs")
        ),
    )

    retry = _try_reuse_graph_context(
        service,
        execution,
        kind="refresh",
        request=RunRequest(run_truth_scope=["repo-a"]),
        execution_host="",
        imported_source_inventory=imported_inventory,
    )

    assert retry is not None
    assert retry.prepared is not None
    assert retry.prepared.context.source_roots == context.source_roots
    assert retry.prepared.context.imported_source_roots == context.imported_source_roots
    assert retry.prepared.context.ingestion_watermark == context.ingestion_watermark
    assert retry.retained_patch_text == '{"summary":"retained","ops":[]}'

    resume_execution = type(
        "Execution",
        (),
        {
            "operation_id": "parent",
            "store": retry_store,
            "reuses_native_checkpoint": True,
        },
    )()
    resumed = _continuation_graph_context(
        service,
        resume_execution,
        kind="refresh",
        request=RunRequest(run_truth_scope=["repo-a"]),
        execution_host="",
        imported_source_inventory=imported_inventory,
    )
    assert resumed.context.imported_source_fingerprint == imported_inventory.fingerprint

    assert service.imported_sources.discard(expected_inventory=imported_inventory)
    empty_inventory = service.imported_source_inventory(
        "refresh",
        manifest.machine_map["laptop"],
    )
    changed_retry = _try_reuse_graph_context(
        service,
        execution,
        kind="refresh",
        request=RunRequest(run_truth_scope=["repo-a"]),
        execution_host="",
        imported_source_inventory=empty_inventory,
    )
    assert changed_retry is not None
    assert changed_retry.prepared is None
    assert "imported provider history changed" in (changed_retry.context_reason or "")

    with pytest.raises(ValueError, match="imported provider history changed"):
        _continuation_graph_context(
            service,
            resume_execution,
            kind="refresh",
            request=RunRequest(run_truth_scope=["repo-a"]),
            execution_host="",
            imported_source_inventory=empty_inventory,
        )


@pytest.mark.asyncio
async def test_failed_graph_run_stages_no_conversation_bytes_and_keeps_watermark(
    manifest, tmp_path
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    watermark = service.history.state().last_refresh_at
    provider_marker = b"provider-owned-native-transcript-marker"
    provider_log = Path(manifest.sources.codex_roots[0]) / "native-session.jsonl"
    provider_log.write_bytes(provider_marker)
    launcher = _FailingLauncher()

    frames = [
        frame
        async for frame in stream_graph_run(
            service,
            launcher,
            "refresh",
            RunRequest(run_truth_scope=["repo-a"]),
            tmp_path / "data",
        )
    ]

    assert any('"event":"error"' in frame for frame in frames)
    assert service.history.state().last_refresh_at == watermark
    assert launcher.workspace is not None
    staged_files = [path for path in launcher.workspace.rglob("*") if path.is_file()]
    assert all("conversation" not in part for path in staged_files for part in path.parts)
    assert all(provider_marker not in path.read_bytes() for path in staged_files)
    assert not hasattr(graph_run, "_project_native_transcripts")
