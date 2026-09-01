from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from rcp.agents import AgentLauncher
from rcp.agents.prompts import PromptFactory
from rcp.agents.write_scope import ProjectWriteScope
from rcp.artifacts import validate_artifact_bytes
from rcp.core.models import ExperimentDecisionPin
from rcp.core.validation import validate_patch
from rcp.demo_provider import resolve_contract
from rcp.runs.tasks.experiment_loop import _prepare_work_patch_candidate
from tests.helpers import create_named_app

FIXTURE = Path(__file__).resolve().parents[1] / "examples" / "demo-project"


def _demo_binary(tmp_path: Path) -> Path:
    binary = tmp_path / "rcp-demo"
    binary.write_text(
        f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -m rcp.demo_provider "$@"\n',
        encoding="utf-8",
    )
    binary.chmod(0o755)
    return binary


def test_demo_contract_resolves_only_rcp_operation_metadata(tmp_path: Path) -> None:
    patch_path = tmp_path / "patch.json"
    watch_path = tmp_path / "watch.json"
    artifact_path = tmp_path / "artifacts"
    graph_path = tmp_path / "graph.json"
    contract_path = tmp_path / "contract.md"
    contract_path.write_text(
        "\n".join(
            (
                "# RCP Experiment-loop task contract",
                "",
                "The human message can say anything and is not interpreted.",
                f"- Current graph, including the Experiment's attempts: `{graph_path}`",
                "- Focused Experiment id: `exp/two-update-matched-trajectory`",
                f"- Optional semantic graph Patch: `{patch_path}`",
                "- Required watcher handoff that continues this Experiment's bounded loop: "
                f"`{watch_path}`",
                f"- Optional preview artifact directory: `{artifact_path}`",
            )
        ),
        encoding="utf-8",
    )

    resolved = resolve_contract(
        PromptFactory.launch_prompt(str(contract_path)),
        capability="work_auto",
    )

    assert resolved.operation == "experiment"
    assert resolved.path == contract_path
    assert resolved.patch_path == patch_path
    assert resolved.watch_path == watch_path
    assert resolved.artifact_path == artifact_path
    assert resolved.graph_path == graph_path
    assert resolved.experiment_id == "exp/two-update-matched-trajectory"


def test_demo_discuss_output_does_not_depend_on_message(tmp_path: Path) -> None:
    binary = _demo_binary(tmp_path)

    first = subprocess.run(
        [str(binary), "turn", "--capability", "discuss"],
        input="first message",
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
    )
    second = subprocess.run(
        [str(binary), "turn", "--capability", "discuss"],
        input="completely different message and /skill tokens",
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
    )

    assert first.stdout == second.stdout
    events = [json.loads(line) for line in first.stdout.splitlines()]
    assert [event["type"] for event in events] == ["session.started", "message", "result"]
    assert "broader plasticity claim remains open" in events[-1]["result"]


def test_demo_episode_report_writes_only_the_frozen_html_output(tmp_path: Path) -> None:
    output_path = tmp_path / "episode-report.html"
    contract_path = tmp_path / "report-contract.md"
    contract_path.write_text(
        "\n".join(
            (
                "# RCP episode report contract",
                "",
                "Required read-only inputs:",
                f"- immutable compact episode receipt: `{tmp_path / 'receipt.json'}`",
                "- expected receipt SHA-256: `fixed-demo-digest`",
                f"- exact official `episode-report` SKILL.md: `{tmp_path / 'SKILL.md'}`",
                "Only permitted output:",
                f"- self-contained sandbox-safe HTML report: `{output_path}`",
            )
        ),
        encoding="utf-8",
    )
    contract = resolve_contract(
        PromptFactory.launch_prompt(str(contract_path)),
        capability="work_auto",
    )
    assert contract.operation == "report"
    assert contract.report_path == output_path
    binary = _demo_binary(tmp_path)

    run = subprocess.run(
        [str(binary), "turn", "--capability", "work_auto", "--session", "demo-session"],
        input=PromptFactory.launch_prompt(str(contract_path)),
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
    )

    events = [json.loads(line) for line in run.stdout.splitlines()]
    assert [event["type"] for event in events] == ["session.started", "message", "result"]
    assert events[0]["session_id"] == "demo-session"
    assert events[-1]["result"] == "RCP Demo wrote the fixed synthetic episode report."
    report = output_path.read_bytes()
    assert validate_artifact_bytes(output_path.name, report) == "text/html"
    assert b"Qualified, inconclusive" in report
    assert not (tmp_path / "patch.json").exists()
    assert not (tmp_path / "watch.json").exists()


@pytest.mark.asyncio
async def test_demo_provider_uses_configured_local_path_and_shared_launcher(tmp_path: Path) -> None:
    binary = _demo_binary(tmp_path)
    launcher = AgentLauncher()

    readiness = launcher.readiness("rcp-demo", binary=str(binary))
    events = [
        event
        async for event in launcher.stream(
            "rcp-demo",
            "message wording is ignored",
            cwd=tmp_path,
            binary=str(binary),
            capability="discuss",
        )
    ]

    assert readiness.installed is True
    assert readiness.authenticated is True
    assert readiness.binary_path == str(binary)
    assert readiness.version == "rcp-demo 1.0"
    assert [event.event for event in events] == [
        "runtime",
        "session",
        "message",
        "answer",
        "provider_exit",
        "done",
    ]


def test_demo_provider_keeps_the_standard_ssh_path_contract() -> None:
    scope = ProjectWriteScope.create(
        project_id="project-1",
        execution_machine="remote",
        execution_host="gpu.example",
        capability="work_auto",
        stage_root="/srv/rcp/task-1",
        workspace_root="/srv/rcp/task-1",
        repositories=[],
        protected_write_paths=[],
    )
    provider_command = AgentLauncher._command(
        "rcp-demo",
        PromptFactory.launch_prompt("/srv/rcp/task-1/contract.md"),
        binary="/opt/rcp/bin/rcp-demo",
        cwd=Path("/srv/rcp/task-1"),
        model=None,
        reasoning=None,
        session_id="demo-session",
        read_dirs=[],
        write_dirs=[],
        write_scope=scope,
        capability="work_auto",
        provider_version="rcp-demo 1.0",
    )
    command = AgentLauncher._remote_login_command(
        provider_command,
        cwd=Path("/srv/rcp/task-1"),
    )

    assert provider_command == [
        "/opt/rcp/bin/rcp-demo",
        "turn",
        "--capability",
        "work_auto",
        "--session",
        "demo-session",
    ]
    assert "cd /srv/rcp/task-1" in shlex.split(command)[2]
    assert "/opt/rcp/bin/rcp-demo turn" in shlex.split(command)[2]


def test_demo_experiment_writes_one_valid_scoped_terminal_result(tmp_path: Path) -> None:
    project = tmp_path / "demo-project"
    shutil.copytree(FIXTURE, project)
    state_repository = project / "state-repo"
    app = create_named_app(
        str(state_repository / ".research" / "manifest.toml"),
        data_dir=tmp_path / "data",
    )
    state = app.state.service.history.state()
    experiment = state.nodes["exp/two-update-matched-trajectory"]
    decision = state.nodes["dec/match-endpoint-or-training-path"]
    stage = tmp_path / "stage"
    artifacts = stage / "artifacts"
    stage.mkdir()
    contract_path = stage / "contract.md"
    patch_path = stage / "patch.json"
    watch_path = stage / "watch.json"
    contract_path.write_text(
        "\n".join(
            (
                "# RCP Experiment-loop task contract",
                "",
                "The human objective is deliberately irrelevant to the fixed output.",
                "- Current graph, including the Experiment's attempts: "
                f"`{state_repository / '.research' / 'graph.json'}`",
                f"- Focused Experiment id: `{experiment.id}`",
                f"- Optional semantic graph Patch: `{patch_path}`",
                "- Required watcher handoff that continues this Experiment's bounded loop: "
                f"`{watch_path}`",
                f"- Optional preview artifact directory: `{artifacts}`",
            )
        ),
        encoding="utf-8",
    )
    binary = _demo_binary(tmp_path)

    run = subprocess.run(
        [str(binary), "turn", "--capability", "work_auto"],
        input=PromptFactory.launch_prompt(str(contract_path)),
        capture_output=True,
        text=True,
        check=True,
        cwd=stage,
    )

    assert [json.loads(line)["type"] for line in run.stdout.splitlines()] == [
        "session.started",
        "message",
        "result",
    ]
    assert json.loads(watch_path.read_text(encoding="utf-8")) == {
        "external": [],
        "graph": [],
    }
    artifact = artifacts / "held-out-plasticity-replicate.html"
    assert "Future learning separates" in artifact.read_text(encoding="utf-8")
    patch = json.loads(patch_path.read_text(encoding="utf-8"))
    assert patch["ops"][0]["nodes"][0]["changes"]["status"] == "completed"
    assert patch["ops"][2]["edges"][1]["relation"] == "inconclusive"
    bundle = [
        ExperimentDecisionPin(
            decision_id=decision.id,
            decision_revision=decision.updated_rev,
            selected_option=decision.selected_option,
        )
    ]
    candidate = _prepare_work_patch_candidate(
        app.state.service,
        patch_path.read_text(encoding="utf-8"),
        run_truth_scope=["crlp-demo-state"],
        control_node_id=experiment.id,
        control_decision_bundle=bundle,
        source_operation_id="demo-operation",
    )
    report = validate_patch(
        state,
        candidate.patch,
        app.state.service.manifest.project.truth_scope,
        experiment_control_node_id=experiment.id,
        experiment_decision_bundle=bundle,
    )
    assert not report.rejected, [message.message for message in report.messages]
