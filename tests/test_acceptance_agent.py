from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import pytest
from fastapi.testclient import TestClient

import rcp.__main__ as main_module
from rcp.agents.acceptance import (
    ACCEPTANCE_CAMPAIGN_FAIL_MARKER,
    ACCEPTANCE_CAMPAIGN_FINISH_MARKER,
    ACCEPTANCE_CAMPAIGN_SPAWN_THEN_FINISH_MARKER,
    ACCEPTANCE_CAMPAIGN_STOP_MARKER,
    ACCEPTANCE_GENERIC_WATCHER_MARKER,
    AcceptanceAgentLauncher,
)
from rcp.agents.command_mailbox import (
    cleanup_command_mailbox,
    serve_command_mailbox,
    stage_command_mailbox,
)
from rcp.agents.command_protocol import CommandResponse
from rcp.agents.episode_report_prompt import episode_report_task_contract
from rcp.agents.launcher import AgentProcessControl
from rcp.agents.prompts import PromptFactory
from rcp.agents.schema import parse_agent_patch_json
from rcp.runs.auto_research_recovery import AutoResearchOrchestratorTerminalFailure
from tests.helpers import create_named_app as create_app


async def _events(launcher: AcceptanceAgentLauncher, prompt: str, cwd: Path, **kwargs):
    return [
        event
        async for event in launcher.stream(
            "codex",
            prompt,
            cwd=cwd,
            capability="work_auto",
            **kwargs,
        )
    ]


def _prompt(tmp_path: Path, contract: str) -> str:
    path = tmp_path / f"contract-{len(list(tmp_path.glob('contract-*')))}.md"
    path.write_text(contract, encoding="utf-8")
    return f"Open and follow the immutable RCP task contract at:\n{path}\nRead it first."


def _experiment_contract(graph_path: Path) -> str:
    return f"""# RCP Experiment-loop task contract

Required current inputs:
- Current graph, including the Experiment's attempts: `{graph_path}`
- Focused Experiment id: `exp/acceptance-control`
"""


def _campaign_contract(
    role: Literal["orchestrator", "worker"],
    *,
    continuation: bool = False,
) -> str:
    suffix = " continuation" if continuation else " contract"
    return f"# RCP auto-research {role}{suffix}\n\nAcceptance fixture campaign turn.\n"


def _result_view_contract(
    tmp_path: Path,
    *,
    action: Literal["create", "revise"],
    path: Path,
    master_context_path: Path | None = None,
) -> str:
    return PromptFactory.work_turn_prompt(
        artifact_path=str(tmp_path / "turns" / action / "artifacts"),
        human_message=(
            "Show the loss curves by seed."
            if action == "create"
            else "Boxed selection in loss-curves-by-seed.html: late spike. Why?"
        ),
        master_context_path=str(master_context_path) if master_context_path else None,
        result_view_action=action,
        result_view_path=str(path),
    )


def test_acceptance_app_mode_is_explicit_and_visible(tmp_path) -> None:
    provider_app = create_app(data_dir=tmp_path / "provider-data")
    acceptance_app = create_app(data_dir=tmp_path / "acceptance-data", acceptance_agent=True)

    assert provider_app.state.agent_mode == "provider"
    assert acceptance_app.state.agent_mode == "acceptance"
    assert isinstance(acceptance_app.state.launcher, AcceptanceAgentLauncher)
    with TestClient(provider_app) as client:
        assert client.get("/api/health").json()["agent_mode"] == "provider"
    with TestClient(acceptance_app) as client:
        assert client.get("/api/health").json()["agent_mode"] == "acceptance"


def test_acceptance_agent_cli_flag_is_explicit_and_survives_reload(monkeypatch) -> None:
    parsed = main_module.build_parser().parse_args(["serve", "--acceptance-agent"])
    captured: dict[str, object] = {}
    expected = object()

    def fake_create_app(project, *, instance_metadata, acceptance_agent):
        captured.update(
            project=project,
            instance_metadata=instance_metadata,
            acceptance_agent=acceptance_agent,
        )
        return expected

    monkeypatch.setattr(main_module, "create_app", fake_create_app)
    monkeypatch.delenv(main_module.RELOAD_PROJECT_ENV, raising=False)
    monkeypatch.delenv(main_module.RELOAD_METADATA_ENV, raising=False)
    monkeypatch.setenv(main_module.RELOAD_ACCEPTANCE_AGENT_ENV, "1")

    assert parsed.acceptance_agent is True
    assert main_module.reload_app() is expected
    assert captured == {
        "project": None,
        "instance_metadata": None,
        "acceptance_agent": True,
    }


def test_acceptance_agent_reuse_refuses_a_provider_mode_owner(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    metadata = main_module.ServerMetadata.create(
        tmp_path,
        host="127.0.0.1",
        port=8421,
        owner_kind="cli",
    )
    monkeypatch.setattr(
        main_module,
        "_probe_owner",
        lambda _data_dir: (metadata, {"agent_mode": "provider"}),
    )

    @contextmanager
    def held_lock(_data_dir):
        raise main_module.InstanceLockHeld("held")
        yield

    monkeypatch.setattr(main_module, "instance_lock", held_lock)

    with pytest.raises(SystemExit) as stopped:
        main_module._launch_automatically(
            main_module.build_parser().parse_args(
                ["serve", "--acceptance-agent", "--reuse-existing"]
            ),
            tmp_path,
        )
    assert stopped.value.code == main_module.EXIT_REFUSED_UNAVAILABLE
    assert "requested 'acceptance'" in capsys.readouterr().err


def test_acceptance_agent_reuse_refuses_an_owner_without_an_explicit_mode(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    metadata = main_module.ServerMetadata.create(
        tmp_path,
        host="127.0.0.1",
        port=8421,
        owner_kind="cli",
    )
    monkeypatch.setattr(main_module, "_probe_owner", lambda _data_dir: (metadata, {}))

    @contextmanager
    def held_lock(_data_dir):
        raise main_module.InstanceLockHeld("held")
        yield

    monkeypatch.setattr(main_module, "instance_lock", held_lock)

    with pytest.raises(SystemExit) as stopped:
        main_module._launch_automatically(
            main_module.build_parser().parse_args(["serve", "--reuse-existing"]),
            tmp_path,
        )
    assert stopped.value.code == main_module.EXIT_REFUSED_UNAVAILABLE
    assert "does not report a recognized agent mode" in capsys.readouterr().err


def test_acceptance_launcher_refuses_remote_execution(tmp_path) -> None:
    launcher = AcceptanceAgentLauncher()

    readiness = launcher.readiness("codex", host="fixture.invalid")
    events = asyncio.run(
        _events(
            launcher,
            "not used",
            tmp_path,
            host="fixture.invalid",
        )
    )

    assert readiness.installed is False
    assert readiness.authenticated is False
    assert [event.event for event in events] == ["error"]
    assert not (tmp_path / "watch.json").exists()
    assert launcher.launch_records[0].action == "remote_rejected"


@pytest.mark.parametrize("role", ["orchestrator", "worker"])
def test_acceptance_campaign_actor_contracts_keep_one_session_and_report_usage(
    tmp_path: Path,
    role: Literal["orchestrator", "worker"],
) -> None:
    stage = tmp_path / role
    stage.mkdir()
    fresh_launcher = AcceptanceAgentLauncher()

    fresh = asyncio.run(
        _events(
            fresh_launcher,
            _prompt(stage, _campaign_contract(role)),
            stage,
        )
    )
    session_id = fresh[0].session_id
    assert session_id is not None

    continuation_launcher = AcceptanceAgentLauncher()
    continuation = asyncio.run(
        _events(
            continuation_launcher,
            _prompt(stage, _campaign_contract(role, continuation=True)),
            stage,
            session_id=session_id,
        )
    )

    for events in (fresh, continuation):
        assert [event.event for event in events] == [
            "session",
            "answer",
            "raw",
            "provider_exit",
            "done",
        ]
        assert all(event.event != "error" for event in events)
        assert events[0].session_id == session_id
        assert f"campaign {role} turn" in events[1].text
        assert events[2].usage is not None
        assert events[2].usage.processed_input_tokens == 256
        assert events[2].usage.generated_tokens == 32

    assert fresh_launcher.launch_records[0].scenario == "campaign"
    assert fresh_launcher.launch_records[0].action == "turn"
    assert continuation_launcher.launch_records[0].scenario == "campaign"
    assert continuation_launcher.launch_records[0].action == "turn"
    assert {
        fresh_launcher.launch_records[0].cwd,
        continuation_launcher.launch_records[0].cwd,
    } == {str(stage.resolve())}
    assert {
        fresh_launcher.launch_records[0].session_id,
        continuation_launcher.launch_records[0].session_id,
    } == {session_id}
    state = json.loads((stage / ".rcp-acceptance-agent.json").read_text(encoding="utf-8"))
    assert state["campaign_actor"] == {
        "cwd": str(stage.resolve()),
        "role": role,
        "session_id": session_id,
    }
    assert not (stage / "patch.json").exists()
    assert not (stage / "watch.json").exists()
    assert not (stage / "messages.json").exists()

    with pytest.raises(ValueError, match="changed its native session"):
        asyncio.run(
            _events(
                continuation_launcher,
                _prompt(stage, _campaign_contract(role, continuation=True)),
                stage,
                session_id="different-acceptance-session",
            )
        )


@pytest.mark.parametrize(
    ("marker", "expected_verb"),
    [
        (ACCEPTANCE_CAMPAIGN_FINISH_MARKER, "finish"),
        (ACCEPTANCE_CAMPAIGN_SPAWN_THEN_FINISH_MARKER, "spawn"),
    ],
)
def test_acceptance_campaign_fixture_invokes_real_staged_client_and_deduplicates_key(
    tmp_path: Path,
    marker: str,
    expected_verb: str,
) -> None:
    stage = tmp_path / "campaign-stage"
    stage.mkdir()
    instruction = stage / "instruction.md"
    instruction.write_text(marker, encoding="utf-8")
    graph = stage / "graph.json"
    graph.write_text(
        json.dumps(
            {
                "nodes": {
                    "exp/acceptance": {
                        "id": "exp/acceptance",
                        "type": "experiment",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    staged = stage_command_mailbox(
        local_stage=stage,
        remote_stage=None,
        episode_id="episode-acceptance",
        task_id="task-acceptance",
        turn_id="turn-acceptance",
        timeout_seconds=2,
    )
    requests = []

    async def run() -> list:
        stop = asyncio.Event()

        def handle(request, _identity):
            requests.append(request)
            result = (
                {
                    "worker_id": "worker-acceptance",
                    "status": "queued",
                    "disposition": "created",
                }
                if request.verb == "spawn"
                else {"episode_id": "episode-acceptance", "ending": "completed"}
            )
            return CommandResponse(request_id=request.request_id, status="ok", result=result)

        server = asyncio.create_task(
            serve_command_mailbox(
                staged=staged,
                handler=handle,
                stop=stop,
                poll_seconds=0.01,
                invocation_gate=staged.invocation_gate,
            )
        )
        try:
            contract = f"""# RCP auto-research orchestrator contract

- starting instruction: `{instruction}`
- graph: `{graph}`
- Command prefix for this turn: `{staged.client_command()}`
"""
            return await _events(
                AcceptanceAgentLauncher(),
                _prompt(stage, contract),
                stage,
                invocation_gate=staged.invocation_gate,
            )
        finally:
            stop.set()
            await server
            cleanup_command_mailbox(mailbox=staged.mailbox, credential=staged.credential)

    events = asyncio.run(run())

    assert events[-1].event == "done"
    assert [request.verb for request in requests] == [expected_verb, expected_verb]
    assert len({request.idempotency_key for request in requests}) == 1
    state = json.loads((stage / ".rcp-acceptance-agent.json").read_text(encoding="utf-8"))
    assert state["campaign_fixture"]["directive"] == (
        "finish" if expected_verb == "finish" else "spawn_then_finish"
    )


def test_acceptance_campaign_held_turn_honors_human_pause(tmp_path: Path) -> None:
    stage = tmp_path / "campaign-stage"
    stage.mkdir()
    instruction = stage / "instruction.md"
    instruction.write_text(ACCEPTANCE_CAMPAIGN_STOP_MARKER, encoding="utf-8")
    control = AgentProcessControl()

    async def run():
        events = asyncio.create_task(
            _events(
                AcceptanceAgentLauncher(),
                _prompt(
                    stage,
                    f"""# RCP auto-research orchestrator contract

- starting instruction: `{instruction}`
""",
                ),
                stage,
                control=control,
            )
        )
        active = stage / ".rcp-acceptance-campaign-active"
        for _ in range(200):
            if active.is_file():
                break
            await asyncio.sleep(0.01)
        assert active.is_file()
        control.request_pause()
        return await asyncio.wait_for(events, timeout=2)

    events = asyncio.run(run())

    assert [event.event for event in events] == ["session", "paused"]
    assert events[-1].text == "Paused during acceptance fixture work."
    assert not (stage / ".rcp-acceptance-campaign-active").exists()


def test_acceptance_campaign_failure_is_an_internal_typed_exception_after_session(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "campaign-stage"
    stage.mkdir()
    instruction = stage / "instruction.md"
    instruction.write_text(ACCEPTANCE_CAMPAIGN_FAIL_MARKER, encoding="utf-8")
    graph = stage / "graph.json"
    graph.write_text(
        json.dumps(
            {
                "nodes": {
                    "exp/acceptance": {
                        "id": "exp/acceptance",
                        "type": "experiment",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (stage / ".rcp-acceptance-campaign-failure-release").write_text(
        "release after session\n",
        encoding="utf-8",
    )
    staged = stage_command_mailbox(
        local_stage=stage,
        remote_stage=None,
        episode_id="episode-acceptance",
        task_id="task-acceptance",
        turn_id="turn-acceptance",
        timeout_seconds=2,
    )
    requests = []

    async def run():
        stop = asyncio.Event()

        def handle(request, _identity):
            requests.append(request)
            return CommandResponse(
                request_id=request.request_id,
                status="ok",
                result={
                    "worker_id": "worker-acceptance",
                    "status": "queued",
                    "disposition": "created",
                },
            )

        server = asyncio.create_task(
            serve_command_mailbox(
                staged=staged,
                handler=handle,
                stop=stop,
                poll_seconds=0.01,
                invocation_gate=staged.invocation_gate,
            )
        )
        events = []
        try:
            contract = f"""# RCP auto-research orchestrator contract

- starting instruction: `{instruction}`
- graph: `{graph}`
- Command prefix for this turn: `{staged.client_command()}`
"""
            with pytest.raises(
                AutoResearchOrchestratorTerminalFailure,
                match="unrecoverable structural failure",
            ):
                async for event in AcceptanceAgentLauncher().stream(
                    "codex",
                    _prompt(stage, contract),
                    cwd=stage,
                    invocation_gate=staged.invocation_gate,
                    capability="orchestrate",
                ):
                    events.append(event)
            return events
        finally:
            stop.set()
            await server
            cleanup_command_mailbox(mailbox=staged.mailbox, credential=staged.credential)

    events = asyncio.run(run())

    assert [event.event for event in events] == ["session"]
    assert events[0].session_id is not None
    assert [request.verb for request in requests] == ["spawn", "spawn"]
    assert len({request.idempotency_key for request in requests}) == 1
    state = json.loads((stage / ".rcp-acceptance-agent.json").read_text(encoding="utf-8"))
    assert state["campaign_actor"]["session_id"] == events[0].session_id
    assert state["campaign_fixture"]["directive"] == "fail"
    assert not (stage / ".rcp-acceptance-campaign-failure-active").exists()
    assert not (stage / ".rcp-acceptance-campaign-failure-release").exists()


def test_acceptance_episode_report_requires_one_same_session_correction(tmp_path: Path) -> None:
    stage = tmp_path / "report-stage"
    stage.mkdir()
    launcher = AcceptanceAgentLauncher()
    actor_events = asyncio.run(
        _events(
            launcher,
            _prompt(stage, _campaign_contract("orchestrator")),
            stage,
        )
    )
    session_id = actor_events[0].session_id
    assert session_id is not None
    receipt_path = stage / "episode-receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "episode_id": "episode-acceptance",
                "mode": "auto_research",
                "ending": "completed",
                "receipt": {"starting_instruction": "Investigate the accepted route."},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    skill_path = stage / "episode-report-SKILL.md"
    skill_path.write_text(
        "# Episode report\n\nProduce an inherently visual, evidence-calibrated report.\n",
        encoding="utf-8",
    )
    report_path = stage / "episode-report.html"
    receipt_digest = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    report_contract = episode_report_task_contract(
        project_name="Acceptance project",
        ending="completed",
        partial=False,
        receipt_path=str(receipt_path),
        receipt_sha256=receipt_digest,
        report_skill_path=str(skill_path),
        report_output_path=str(report_path),
    )
    missing = asyncio.run(
        _events(
            launcher,
            _prompt(stage, report_contract),
            stage,
            session_id=session_id,
        )
    )
    assert (
        missing[1].text
        == "Left the first acceptance episode report attempt missing for correction."
    )
    assert not report_path.exists()

    diagnostic = stage / "report-diagnostic.md"
    diagnostic.write_text("Episode report is missing.", encoding="utf-8")
    correction_contract = episode_report_task_contract(
        project_name="Acceptance project",
        ending="completed",
        partial=False,
        receipt_path=str(receipt_path),
        receipt_sha256=receipt_digest,
        report_skill_path=str(skill_path),
        report_output_path=str(report_path),
        correction_diagnostic_path=str(diagnostic),
    )
    corrected = asyncio.run(
        _events(
            launcher,
            _prompt(
                stage,
                correction_contract,
            ),
            stage,
            session_id=session_id,
        )
    )

    assert corrected[0].session_id == session_id
    assert corrected[1].text == "Wrote the corrected deterministic acceptance episode report."
    assert "Acceptance episode conclusion" in report_path.read_text(encoding="utf-8")
    assert [record.action for record in launcher.launch_records[-2:]] == [
        "report",
        "report_correction",
    ]


def test_acceptance_result_view_create_and_revise_keep_one_stage_session_and_path(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "conversation-stage"
    slot = stage / "views" / ("a" * 24)
    slot.mkdir(parents=True)
    state_path = stage / ".rcp-acceptance-agent.json"
    state_path.write_text(
        json.dumps(
            {
                "scenario": "experiment_loop",
                "focused_experiment_id": "exp/acceptance-control",
                "jobs_started": True,
                "watch_corrected": True,
            }
        ),
        encoding="utf-8",
    )
    launcher = AcceptanceAgentLauncher()
    master_context_path = stage / "inputs" / "chat-master.md"
    master_context_path.parent.mkdir()
    master_context_path.write_text(
        _experiment_contract(stage / "graph.json"),
        encoding="utf-8",
    )

    created_events = asyncio.run(
        _events(
            launcher,
            _result_view_contract(
                stage,
                action="create",
                path=slot,
                master_context_path=master_context_path,
            ),
            stage,
        )
    )

    target = slot / "loss-curves-by-seed.html"
    created_html = target.read_text(encoding="utf-8")
    fixed_gesture = (
        "window.parent.postMessage({type:'rcp-result-view-gesture',version:1,"
        "gesture:'box',description:'late spike across steps 8,000–9,000 for seed 3'}, '*');"
    )
    assert [event.event for event in created_events] == [
        "session",
        "answer",
        "provider_exit",
        "done",
    ]
    assert launcher.launch_records[0].scenario == "result_view"
    assert launcher.launch_records[0].action == "create"
    assert "Loss curves by seed" in created_html
    assert "Revision 1 — initial curves" in created_html
    assert all(
        f"addEventListener('{event}'" in created_html
        for event in ("pointerdown", "pointermove", "pointerup")
    )
    assert created_html.count("postMessage") == 1
    assert fixed_gesture in created_html
    assert all(token not in created_html for token in ("fetch(", "XMLHttpRequest", "<form"))
    assert list(slot.iterdir()) == [target]
    assert list((stage / "views").iterdir()) == [slot]
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["scenario"] == "experiment_loop"
    assert persisted["focused_experiment_id"] == "exp/acceptance-control"
    assert persisted["result_view"] == {
        "cwd": str(stage.resolve()),
        "path": str(target.resolve()),
        "revision": 1,
        "session_id": created_events[0].session_id,
    }

    revise_contract = _result_view_contract(stage, action="revise", path=target)
    with pytest.raises(ValueError, match="changed the native session"):
        asyncio.run(
            _events(
                launcher,
                _prompt(stage, revise_contract),
                stage,
                session_id="different-acceptance-session",
            )
        )
    assert target.read_text(encoding="utf-8") == created_html

    revised_events = asyncio.run(
        _events(
            launcher,
            revise_contract,
            stage,
            session_id=created_events[0].session_id,
        )
    )

    revised_html = target.read_text(encoding="utf-8")
    assert [event.event for event in revised_events] == [
        "session",
        "answer",
        "provider_exit",
        "done",
    ]
    assert [record.action for record in launcher.launch_records] == ["create", "revise"]
    assert {record.cwd for record in launcher.launch_records} == {str(stage.resolve())}
    assert {record.session_id for record in launcher.launch_records} == {
        created_events[0].session_id
    }
    assert "Revision 2 — late spike annotated" in revised_html
    assert "reviewed late spike" in revised_html
    assert revised_html != created_html
    assert revised_html.count("postMessage") == 1
    assert fixed_gesture in revised_html
    assert list(slot.iterdir()) == [target]
    assert list((stage / "views").iterdir()) == [slot]
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["scenario"] == "experiment_loop"
    assert persisted["result_view"]["path"] == str(target.resolve())
    assert persisted["result_view"]["revision"] == 2


def test_acceptance_result_view_revision_requires_an_existing_path_in_the_same_stage(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "conversation-stage"
    stage.mkdir()
    outside = tmp_path / "outside" / "loss-curves-by-seed.html"
    outside.parent.mkdir()
    outside.write_text("<html>outside</html>", encoding="utf-8")
    launcher = AcceptanceAgentLauncher()

    with pytest.raises(ValueError, match="left the conversation cwd"):
        asyncio.run(
            _events(
                launcher,
                _prompt(
                    stage,
                    _result_view_contract(stage, action="revise", path=outside),
                ),
                stage,
            )
        )

    missing = stage / "views" / ("b" * 24) / "loss-curves-by-seed.html"
    with pytest.raises(ValueError, match="revision target is unavailable"):
        asyncio.run(
            _events(
                launcher,
                _prompt(
                    stage,
                    _result_view_contract(stage, action="revise", path=missing),
                ),
                stage,
            )
        )


def test_acceptance_experiment_corrects_watchers_then_completes_with_authority_item(
    tmp_path,
) -> None:
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "edges": {
                    "edge/acceptance-tests": {
                        "source": "exp/acceptance-control",
                        "target": "hyp/acceptance-sequence",
                        "relation": "tests",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    launcher = AcceptanceAgentLauncher()

    initial = asyncio.run(
        _events(launcher, _prompt(tmp_path, _experiment_contract(graph_path)), tmp_path)
    )
    assert [event.event for event in initial] == ["session", "answer", "provider_exit", "done"]
    assert json.loads((tmp_path / "watch.json").read_text(encoding="utf-8")) == {
        "invalid": "correction required"
    }
    jobs = tmp_path / "acceptance-agent-jobs"
    assert sorted(path.name for path in jobs.glob("*.status")) == [
        "job-one.status",
        "job-two.status",
    ]

    correction = asyncio.run(
        _events(
            launcher,
            _prompt(tmp_path, "# RCP Experiment-loop watcher correction"),
            tmp_path,
            session_id=initial[0].session_id,
        )
    )
    specs = json.loads((tmp_path / "watch.json").read_text(encoding="utf-8"))
    assert [event.event for event in correction] == [
        "session",
        "answer",
        "provider_exit",
        "done",
    ]
    assert len(specs["external"]) == 2
    assert specs["graph"] == []
    assert launcher.launch_records[-1].action == "watch_correction"
    assert launcher.launch_records[-1].watcher_count == 2
    assert len(list(jobs.glob("*.status"))) == 2

    for name in ("job-one", "job-two"):
        (jobs / f"{name}.done").write_text("done\n", encoding="utf-8")
    compact_wake = f"""The watched work is ready for another look.

Read the fresh state before acting:
- current graph: `{graph_path}`
"""
    wake = asyncio.run(_events(launcher, _prompt(tmp_path, compact_wake), tmp_path))

    assert [event.event for event in wake] == ["session", "answer", "provider_exit", "done"]
    assert json.loads((tmp_path / "watch.json").read_text(encoding="utf-8")) == {
        "external": [],
        "graph": [],
    }
    patch = parse_agent_patch_json((tmp_path / "patch.json").read_text(encoding="utf-8"))
    payload = patch.model_dump(mode="json")
    assert payload["ops"][0]["nodes"] == [
        {"id": "exp/acceptance-control", "changes": {"status": "completed"}, "cause": None}
    ]
    assert payload["ops"][2]["edges"][1]["id"] == "edge/acceptance-supports"
    proposal_update = payload["ops"][3]["proposals"][0]["ops"][0]["nodes"][0]
    assert proposal_update["id"] == "hyp/acceptance-sequence"
    assert proposal_update["cause"]["ref_id"] == "edge/acceptance-supports"


def test_acceptance_state_survives_a_fresh_launcher_instance(tmp_path) -> None:
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "edges": {
                    "edge/acceptance-tests": {
                        "source": "exp/acceptance-control",
                        "target": "hyp/acceptance-sequence",
                        "relation": "tests",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    asyncio.run(
        _events(
            AcceptanceAgentLauncher(),
            _prompt(tmp_path, _experiment_contract(graph_path)),
            tmp_path,
        )
    )

    fresh = AcceptanceAgentLauncher()
    asyncio.run(
        _events(
            fresh,
            _prompt(tmp_path, "# RCP Experiment-loop watcher correction"),
            tmp_path,
            session_id="persisted-native-session",
        )
    )

    assert fresh.launch_records == (
        fresh.launch_records[0].__class__(
            scenario="experiment_loop",
            action="watch_correction",
            cwd=str(tmp_path.resolve()),
            session_id="persisted-native-session",
            watcher_count=2,
        ),
    )


def test_acceptance_generic_marker_arms_two_watchers_without_a_patch(tmp_path) -> None:
    launcher = AcceptanceAgentLauncher()

    asyncio.run(
        _events(
            launcher,
            _prompt(
                tmp_path,
                f"# RCP ordinary Work contract\n\n{ACCEPTANCE_GENERIC_WATCHER_MARKER}",
            ),
            tmp_path,
        )
    )
    asyncio.run(
        _events(
            launcher,
            _prompt(tmp_path, "# RCP watch correction contract"),
            tmp_path,
            session_id=launcher.launch_records[0].session_id,
        )
    )

    specs = json.loads((tmp_path / "watch.json").read_text(encoding="utf-8"))
    assert len(specs["external"]) == 2
    assert specs["graph"] == []
    assert [record.action for record in launcher.launch_records] == [
        "initial",
        "watch_correction",
    ]
    assert not (tmp_path / "patch.json").exists()
