from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from rcp.agents import AgentEvent
from rcp.runs.tasks.coach import stream_coach
from rcp.runs.tasks.discuss import stream_discuss_run
from rcp.runs.tasks.work import stream_work_run

from .helpers import (
    agent_patch_json,
    append_fixture_patch,
    refresh_patch,
    seed_patch,
    wait_for_task_response,
)
from .helpers import create_named_app as create_app


class _FailThenSucceedLauncher:
    def __init__(
        self,
        failure: str,
        *,
        first_files: dict[str, str] | None = None,
        retry_files: dict[str, str] | None = None,
    ) -> None:
        self.failure = failure
        self.first_files = first_files or {}
        self.retry_files = retry_files or {}
        self.native_session_id = str(uuid.uuid4())
        self.contract_paths: list[Path] = []
        self.contracts: list[str] = []
        self.input_snapshots: list[dict[str, str]] = []
        self.sessions: list[str | None] = []
        self.workspaces: list[Path] = []

    async def stream(self, _provider, prompt, **kwargs):
        attempt = len(self.contracts)
        contract_path = Path(prompt.splitlines()[1])
        inputs = contract_path.parent
        workspace = Path(kwargs["cwd"])
        self.contract_paths.append(contract_path)
        self.contracts.append(contract_path.read_text(encoding="utf-8"))
        self.input_snapshots.append(
            {
                item.name: item.read_text(encoding="utf-8")
                for item in inputs.iterdir()
                if item.is_file()
            }
        )
        self.sessions.append(kwargs.get("session_id"))
        self.workspaces.append(workspace)
        yield AgentEvent(event="session", session_id=self.native_session_id)
        if attempt == 0:
            for name, content in self.first_files.items():
                (workspace / name).write_text(content, encoding="utf-8")
            yield AgentEvent(event="error", text=self.failure)
            return
        for name, content in self.retry_files.items():
            (workspace / name).write_text(content, encoding="utf-8")
        yield AgentEvent(event="answer", text="The Retry completed.")
        yield AgentEvent(event="done")


def _retry_task(
    client: TestClient,
    project_id: str,
    kind: str,
    body: dict[str, object],
    *,
    retry_body: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    started = client.post(f"/api/projects/{project_id}/tasks/{kind}", json=body)
    assert started.status_code == 202
    failed = wait_for_task_response(client, project_id, started.json()["operation_id"])
    assert failed["status"] == "failed"
    retried_response = client.post(
        f"/api/projects/{project_id}/tasks/{failed['operation_id']}/retry",
        json=retry_body or {},
    )
    assert retried_response.status_code == 202
    retried = wait_for_task_response(client, project_id, retried_response.json()["operation_id"])
    assert retried["status"] == "succeeded"
    return failed, retried


def _assert_retry_contract(
    launcher: _FailThenSucceedLauncher,
    *,
    expected_failure: str,
) -> str:
    """A retry that still holds its native session gets a follow-up, not a rebuilt contract."""

    assert launcher.sessions == [None, launcher.native_session_id]
    retry_contract_path = launcher.contract_paths[1]
    retry_contract = launcher.contracts[1]
    prefix = retry_contract_path.name.removesuffix("-retry.md")
    stage_inputs = retry_contract_path.parent
    diagnostics_path = stage_inputs / f"{prefix}-retry-diagnostics.json"
    expected_diagnostics = [f"Attempt 1 (failed) failed with: {expected_failure}"]

    assert retry_contract.startswith("# RCP retry contract")
    assert "Exact failure diagnostics" in retry_contract
    assert str(diagnostics_path) in retry_contract
    assert str(launcher.contract_paths[0]) in retry_contract
    assert "same native session that ran the previous attempt" in retry_contract
    assert "Retry authority and side-effect safety" in retry_contract
    assert "inspect the authoritative external state" in retry_contract
    assert "# RCP resume contract" not in retry_contract
    assert "Patch-only correction authority" not in retry_contract
    # Nothing is rebuilt or restated for this attempt: no base contract, no human request.
    assert not (stage_inputs / f"{prefix}-base.md").exists()
    assert f"{prefix}-base.md" not in launcher.input_snapshots[1]
    assert f"{prefix}-human-request.txt" not in launcher.input_snapshots[1]
    assert "Retry context:" not in retry_contract
    assert json.loads(launcher.input_snapshots[1][diagnostics_path.name]) == {
        "prior_attempt_diagnostics": expected_diagnostics
    }
    return launcher.contracts[0]


def _assert_retry_receipt(app, operation_id: str) -> None:
    receipts = app.state.background_tasks.store.agent_task_receipts(operation_id)
    launches = [item for item in receipts if item.category == "agent_launch"]
    prompts = [item for item in receipts if item.category == "agent_prompt"]
    assert len(launches) == 1
    assert launches[0].payload["launch_kind"] == "retry"
    assert launches[0].payload["continuation_cause"] == "retry"
    assert len(prompts) == 1
    assert prompts[0].payload["launch_kind"] == "retry"
    assert prompts[0].payload["continuation_cause"] == "retry"


def test_same_provider_discuss_retry_receives_exact_failure(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    failure = "Discuss provider lost its response stream after reading the node."
    objective = "Explain why this hypothesis is still proposed."
    launcher = _FailThenSucceedLauncher(failure)

    async def stream(_project_id, kind, request, execution):
        assert kind == "node_chat"
        async for frame in stream_discuss_run(
            service,
            launcher,
            request,
            tmp_path / "data",
            execution=execution,
        ):
            yield frame

    app.state.background_tasks.stream = stream
    client = TestClient(app)
    project_id = app.state.default_project_id
    _, retried = _retry_task(
        client,
        project_id,
        "node_chat",
        {
            "node_id": "hyp/replanning-restores-plasticity",
            "chat_id": str(uuid.uuid4()),
            "message": objective,
            "run_truth_scope": ["repo-a"],
            "mode": "discuss",
        },
    )

    original_contract = _assert_retry_contract(launcher, expected_failure=failure)
    assert "This turn has no graph-change channel" in original_contract
    assert launcher.workspaces[0] == launcher.workspaces[1]
    _assert_retry_receipt(app, str(retried["operation_id"]))


def test_same_provider_work_retry_ignores_unchanged_predecessor_outputs(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    store = app.state.background_tasks.store
    failure = "Work provider disconnected after an external submission may have completed."
    objective = "Submit the bounded run once and report its result."
    stale_patch = refresh_patch("rq/stale-retry-deliverable").model_copy(update={"kind": "work"})
    stale_watch = json.dumps(
        {
            "external": [
                {
                    "check_command": "exit 1",
                    "log_path": str(tmp_path / "detached.log"),
                    "cwd": str(tmp_path),
                }
            ],
            "graph": [],
        }
    )
    launcher = _FailThenSucceedLauncher(
        failure,
        first_files={
            "patch.json": agent_patch_json(stale_patch),
            "watch.json": stale_watch,
        },
    )

    async def stream(_project_id, kind, request, execution):
        assert kind == "project_chat"
        async for frame in stream_work_run(
            service,
            launcher,
            request,
            tmp_path / "data",
            execution=execution,
        ):
            yield frame

    app.state.background_tasks.stream = stream
    client = TestClient(app)
    project_id = app.state.default_project_id
    _, retried = _retry_task(
        client,
        project_id,
        "project_chat",
        {
            "chat_id": str(uuid.uuid4()),
            "message": objective,
            "run_truth_scope": ["repo-a"],
            "mode": "work",
        },
    )

    original_contract = _assert_retry_contract(launcher, expected_failure=failure)
    assert "Operational authority" in original_contract
    assert launcher.workspaces[0] == launcher.workspaces[1]
    workspace = launcher.workspaces[1]
    assert (workspace / "patch.json").read_text(encoding="utf-8") == agent_patch_json(stale_patch)
    assert (workspace / "watch.json").read_text(encoding="utf-8") == stale_watch
    assert "rq/stale-retry-deliverable" not in service.history.state().nodes
    assert retried["result"]["graph_update"]["status"] == "none"
    assert store.watchers(project_id) == []
    comparisons = [
        item.payload
        for item in store.agent_task_receipts(str(retried["operation_id"]))
        if item.category == "retry_deliverable_comparison"
    ]
    assert {(item["filename"], item["unchanged"], item["consumed"]) for item in comparisons} == {
        ("patch.json", True, False),
        ("watch.json", True, False),
    }
    _assert_retry_receipt(app, str(retried["operation_id"]))


def test_same_provider_work_retry_applies_semantically_valid_patch_to_live_state(
    manifest, tmp_path
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    failure = "Work provider failed before returning its graph reflection."
    retried_patch = refresh_patch("rq/retry-applied-live").model_copy(update={"kind": "work"})
    launcher = _FailThenSucceedLauncher(
        failure,
        retry_files={"patch.json": agent_patch_json(retried_patch)},
    )

    async def stream(_project_id, kind, request, execution):
        assert kind == "project_chat"
        async for frame in stream_work_run(
            service,
            launcher,
            request,
            tmp_path / "data",
            execution=execution,
        ):
            yield frame

    app.state.background_tasks.stream = stream
    client = TestClient(app)
    project_id = app.state.default_project_id
    started = client.post(
        f"/api/projects/{project_id}/tasks/project_chat",
        json={
            "chat_id": str(uuid.uuid4()),
            "message": "Complete the original Work turn.",
            "run_truth_scope": ["repo-a"],
            "mode": "work",
        },
    )
    assert started.status_code == 202
    failed = wait_for_task_response(client, project_id, started.json()["operation_id"])
    assert failed["status"] == "failed"

    append_fixture_patch(service, refresh_patch("rq/landed-before-work-retry"))
    retried_response = client.post(
        f"/api/projects/{project_id}/tasks/{failed['operation_id']}/retry"
    )
    assert retried_response.status_code == 202
    retried = wait_for_task_response(client, project_id, retried_response.json()["operation_id"])

    assert retried["status"] == "succeeded"
    graph_update = retried["result"]["graph_update"]
    assert graph_update["status"] == "applied"
    assert graph_update["applied_revision"] == 4
    assert "rq/landed-before-work-retry" in service.history.state().nodes
    assert "rq/retry-applied-live" in service.history.state().nodes


def test_cross_provider_work_retry_uses_a_fresh_retry_contract(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    failure = "The first provider disconnected after a submission may have completed."
    objective = "Check whether the bounded run landed before taking any further action."
    launcher = _FailThenSucceedLauncher(failure)

    async def stream(_project_id, kind, request, execution):
        assert kind == "project_chat"
        async for frame in stream_work_run(
            service,
            launcher,
            request,
            tmp_path / "data",
            execution=execution,
        ):
            yield frame

    app.state.background_tasks.stream = stream
    client = TestClient(app)
    project_id = app.state.default_project_id
    _, retried = _retry_task(
        client,
        project_id,
        "project_chat",
        {
            "chat_id": str(uuid.uuid4()),
            "message": objective,
            "run_truth_scope": ["repo-a"],
            "mode": "work",
        },
        retry_body={"provider": "claude"},
    )

    assert launcher.sessions == [None, None]
    assert launcher.contract_paths[1].name.endswith("-base.md")
    retry_contract = launcher.contracts[1]
    assert retry_contract.startswith("# RCP Work task contract")
    assert "Retry context:" in retry_contract
    assert "inspect the authoritative external state" in " ".join(retry_contract.split())
    assert (
        objective
        == launcher.input_snapshots[1][
            next(name for name in launcher.input_snapshots[1] if name.endswith("human-request.txt"))
        ]
    )
    diagnostics_name = next(
        name for name in launcher.input_snapshots[1] if name.endswith("retry-diagnostics.json")
    )
    assert json.loads(launcher.input_snapshots[1][diagnostics_name]) == {
        "prior_attempt_diagnostics": [f"Attempt 1 (failed) failed with: {failure}"]
    }
    receipts = app.state.background_tasks.store.agent_task_receipts(str(retried["operation_id"]))
    launch = next(item for item in receipts if item.category == "agent_launch")
    assert launch.payload["launch_kind"] == "retry"
    assert launch.payload["continuation_cause"] == "handoff"


def test_same_provider_paper_coach_retry_receives_exact_failure(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    service.paper.create()
    failure = "Paper coach provider failed while examining the introduction."
    objective = "Review the introduction's causal argument."
    launcher = _FailThenSucceedLauncher(failure)

    async def stream(_project_id, kind, request, execution):
        assert kind == "paper_coach"
        async for frame in stream_coach(
            service,
            launcher,
            service.paper,
            request,
            tmp_path / "data",
            execution=execution,
        ):
            yield frame

    app.state.background_tasks.stream = stream
    client = TestClient(app)
    project_id = app.state.default_project_id
    _, retried = _retry_task(
        client,
        project_id,
        "paper_coach",
        {"message": objective},
    )

    original_contract = _assert_retry_contract(launcher, expected_failure=failure)
    assert "Authorship contract" in original_contract
    assert launcher.contract_paths[0].parent == launcher.contract_paths[1].parent
    _assert_retry_receipt(app, str(retried["operation_id"]))
