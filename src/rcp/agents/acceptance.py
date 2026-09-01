from __future__ import annotations

import asyncio
import hashlib
import json
import shlex
import subprocess
import sys
import threading
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import NAMESPACE_URL, uuid5

from rcp.agents.invocation_broker import ProviderInvocationGate
from rcp.agents.launcher import (
    AgentEvent,
    AgentLauncher,
    AgentProcessControl,
    ProviderReadiness,
)
from rcp.agents.write_scope import ProjectWriteScope
from rcp.limits import ACCEPTANCE_AGENT_JOB_SECONDS
from rcp.providers import AgentCapability, ProviderUsage, profile_for

ACCEPTANCE_GENERIC_WATCHER_MARKER = "[RCP acceptance: generic watchers]"
ACCEPTANCE_CAMPAIGN_FINISH_MARKER = "[RCP acceptance: campaign finish]"
ACCEPTANCE_CAMPAIGN_SPAWN_THEN_FINISH_MARKER = "[RCP acceptance: campaign spawn then finish]"
ACCEPTANCE_CAMPAIGN_SPAWN_THEN_INTERRUPT_MARKER = "[RCP acceptance: campaign spawn then interrupt]"
ACCEPTANCE_CAMPAIGN_EXHAUST_MARKER = "[RCP acceptance: campaign exhaust]"
ACCEPTANCE_CAMPAIGN_STOP_MARKER = "[RCP acceptance: campaign stop while active]"
ACCEPTANCE_CAMPAIGN_FAIL_MARKER = "[RCP acceptance: campaign unrecoverable failure]"
ACCEPTANCE_CAMPAIGN_INTERRUPT_ACTIVE_FILE = ".rcp-acceptance-campaign-interrupt-active"
ACCEPTANCE_CAMPAIGN_REAUTHORIZED_ACTIVE_FILE = ".rcp-acceptance-campaign-reauthorized-active"
ACCEPTANCE_CAMPAIGN_REAUTHORIZED_RELEASE_FILE = ".rcp-acceptance-campaign-reauthorized-release"

_STATE_FILE = ".rcp-acceptance-agent.json"
_JOBS_DIRECTORY = "acceptance-agent-jobs"
_RESULT_VIEW_AUTHORING_MARKER = "RCP result-view authoring contract:"
_RESULT_VIEW_CREATE_PREFIX = (
    "- Create exactly one bounded, self-contained, descriptively named HTML file directly inside `"
)
_RESULT_VIEW_CREATE_SUFFIX = "`."
_RESULT_VIEW_REVISE_PREFIX = "- Edit the existing HTML file `"
_RESULT_VIEW_REVISE_SUFFIX = (
    "` in place. Keep its exact path and name; atomic replacement at that path is allowed."
)
_RESULT_VIEW_NAME = "loss-curves-by-seed.html"
_RESULT_VIEW_STATE_KEY = "result_view"
_CAMPAIGN_STATE_KEY = "campaign_actor"
_CAMPAIGN_FIXTURE_STATE_KEY = "campaign_fixture"
_CAMPAIGN_WORKER_REPLY_MARKER = "[RCP acceptance: campaign worker reply]"
_CAMPAIGN_FAILURE_WORKER_MARKER = "[RCP acceptance: hold admitted worker for failure]"
_CAMPAIGN_STOP_ACTIVE_FILE = ".rcp-acceptance-campaign-active"
_CAMPAIGN_STOP_RELEASE_FILE = ".rcp-acceptance-campaign-release"
_CAMPAIGN_FAILURE_ACTIVE_FILE = ".rcp-acceptance-campaign-failure-active"
_CAMPAIGN_FAILURE_RELEASE_FILE = ".rcp-acceptance-campaign-failure-release"
_CAMPAIGN_FAILURE_WORKER_ACTIVE_FILE = ".rcp-acceptance-campaign-worker-active"
_CAMPAIGN_FAILURE_WORKER_RELEASE_FILE = ".rcp-acceptance-campaign-worker-release"
_CAMPAIGN_ORDINARY_CHILD_MARKER = "## Auto-research child Work boundary"
_CAMPAIGN_CONTRACTS: dict[
    str,
    tuple[
        Literal["orchestrator", "worker", "report"],
        Literal["fresh", "continuation", "report"],
    ],
] = {
    "# RCP auto-research orchestrator contract": ("orchestrator", "fresh"),
    "# RCP auto-research orchestrator continuation": ("orchestrator", "continuation"),
    "# RCP auto-research worker contract": ("worker", "fresh"),
    "# RCP auto-research worker continuation": ("worker", "continuation"),
    "# RCP episode report contract": ("report", "report"),
}


@dataclass(frozen=True)
class AcceptanceLaunchRecord:
    scenario: Literal[
        "experiment_loop",
        "generic_watchers",
        "result_view",
        "campaign",
        "unsupported",
    ]
    action: Literal[
        "initial",
        "watch_correction",
        "wake",
        "create",
        "revise",
        "turn",
        "report",
        "report_correction",
        "unsupported",
        "remote_rejected",
    ]
    cwd: str
    session_id: str
    watcher_count: int


class AcceptanceAgentLauncher(AgentLauncher):
    """Explicit, local-only provider double for served acceptance scenarios.

    This launcher is selected only by ``create_app(acceptance_agent=True)``. It
    never calls a provider, network service, scheduler, or GPU. Its state and
    detached job artifacts live in the persistent chat scratch directory so a
    server restart exercises the same recovery path as a real provider session.
    """

    def __init__(self) -> None:
        super().__init__()
        self._records_lock = threading.Lock()
        self._launch_records: list[AcceptanceLaunchRecord] = []

    @property
    def launch_records(self) -> tuple[AcceptanceLaunchRecord, ...]:
        with self._records_lock:
            return tuple(self._launch_records)

    def readiness(
        self,
        provider: str,
        *,
        host: str = "",
        binary: str | None = None,
        refresh: bool = False,
    ) -> ProviderReadiness:
        del binary, refresh
        profile = profile_for(provider)
        if host:
            return ProviderReadiness(
                provider=provider,
                label=f"Acceptance {profile.label}",
                installed=False,
                authenticated=False,
                version="acceptance-local-only",
                path_state="unreachable",
                reason=(
                    "Acceptance-agent mode is local-only and refuses to impersonate "
                    f"a provider on {host}."
                ),
                models=list(profile.declared),
            )
        return ProviderReadiness(
            provider=provider,
            label=f"Acceptance {profile.label}",
            installed=True,
            authenticated=True,
            version="acceptance-1",
            path_state="resolved",
            models=list(profile.declared),
        )

    async def stream(
        self,
        provider: str,
        prompt: str,
        *,
        cwd: Path,
        model: str | None = None,
        reasoning: str | None = None,
        session_id: str | None = None,
        read_dirs: list[Path] | None = None,
        write_dirs: list[Path] | None = None,
        write_scope: ProjectWriteScope | None = None,
        host: str = "",
        control: AgentProcessControl | None = None,
        remote_pid_file: str | None = None,
        invocation_gate: ProviderInvocationGate | None = None,
        capability: AgentCapability,
        binary: str | None = None,
        runtime_id: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        if invocation_gate is not None:
            async with invocation_gate.serve_current_session():
                async for event in self.stream(
                    provider,
                    prompt,
                    cwd=cwd,
                    model=model,
                    reasoning=reasoning,
                    session_id=session_id,
                    read_dirs=read_dirs,
                    write_dirs=write_dirs,
                    write_scope=write_scope,
                    host=host,
                    control=control,
                    remote_pid_file=remote_pid_file,
                    invocation_gate=None,
                    capability=capability,
                    binary=binary,
                    runtime_id=runtime_id,
                ):
                    yield event
            return
        del (
            model,
            reasoning,
            read_dirs,
            write_dirs,
            write_scope,
            remote_pid_file,
            capability,
            binary,
        )
        resolved_cwd = cwd.resolve()
        stable_session = session_id or str(
            uuid5(NAMESPACE_URL, f"rcp-acceptance-session:{resolved_cwd}")
        )
        if host:
            self._record(
                AcceptanceLaunchRecord(
                    scenario="unsupported",
                    action="remote_rejected",
                    cwd=str(resolved_cwd),
                    session_id=stable_session,
                    watcher_count=0,
                )
            )
            yield AgentEvent(
                event="error",
                text=(
                    f"Acceptance-agent mode is local-only and cannot run fixture work on {host}."
                ),
            )
            return
        if control is not None and control.pause_requested.is_set():
            yield AgentEvent(event="paused", text="Paused before acceptance fixture work started.")
            return

        state = _read_state(resolved_cwd)
        contract = (
            prompt if _RESULT_VIEW_AUTHORING_MARKER in prompt else _read_launch_contract(prompt)
        )
        scenario = _scenario(prompt, contract, state)
        active_contract = prompt if _RESULT_VIEW_AUTHORING_MARKER in prompt else contract
        campaign_contract = _campaign_contract(contract)
        if scenario == "result_view":
            action = _result_view_action(active_contract)
        elif scenario == "campaign":
            action = (
                "report_correction"
                if campaign_contract == ("report", "report")
                and "- exact report correction diagnostic: `" in contract
                else "report"
                if campaign_contract == ("report", "report")
                else "turn"
            )
        else:
            action = _action(contract, state)
        watcher_count = 0

        if _holds_reauthorized_exhaustion(state, campaign_contract):
            # ``session`` is the provider boundary that changes the durable task message
            # from preparation to "Agent task is running."  Create the held-turn marker
            # first so that visible message is a reliable browser synchronization point.
            _prepare_campaign_fixture_active(
                resolved_cwd,
                active_name=ACCEPTANCE_CAMPAIGN_REAUTHORIZED_ACTIVE_FILE,
                label="reauthorized exhaustion",
            )
        if runtime_id is not None:
            yield AgentEvent(
                event="runtime",
                text=profile_for(provider).runtime_candidates(runtime_id)[0].id,
            )
        yield AgentEvent(event="session", session_id=stable_session)
        if scenario == "unsupported":
            self._record(
                AcceptanceLaunchRecord(
                    scenario=scenario,
                    action="unsupported",
                    cwd=str(resolved_cwd),
                    session_id=stable_session,
                    watcher_count=0,
                )
            )
            yield AgentEvent(
                event="error",
                text=(
                    "The acceptance agent only runs an Experiment-loop invocation, a result-view "
                    "authoring Work turn, an auto-research campaign actor turn, or an ordinary "
                    "Work turn containing "
                    f"{ACCEPTANCE_GENERIC_WATCHER_MARKER!r}."
                ),
            )
            return

        if scenario == "campaign":
            if campaign_contract is None:
                raise ValueError("Acceptance campaign contract is not recognized.")
            campaign_role, campaign_phase = campaign_contract
            try:
                answer = await _accept_campaign_turn(
                    resolved_cwd,
                    state,
                    stable_session,
                    prompt=prompt,
                    contract=contract,
                    role=campaign_role,
                    phase=campaign_phase,
                    control=control,
                )
            except _AcceptancePauseRequested:
                yield AgentEvent(event="paused", text="Paused during acceptance fixture work.")
                return
        elif scenario == "result_view":
            answer = _author_result_view(
                resolved_cwd,
                active_contract,
                state,
                stable_session,
                action,
            )
        elif action == "initial":
            focused_experiment_id = _focused_experiment_id(contract)
            _start_fixture_jobs(resolved_cwd)
            state = {
                "scenario": scenario,
                "focused_experiment_id": focused_experiment_id,
                "jobs_started": True,
                "watch_corrected": False,
            }
            _write_state(resolved_cwd, state)
            # Deliberately invalid. Production orchestration must retain this
            # native session and request exactly one watcher-only correction.
            _write_json(resolved_cwd / "watch.json", {"invalid": "correction required"})
            answer = "Started two deterministic CPU-only fixture jobs."
        elif action == "watch_correction":
            specs = _watch_specs(resolved_cwd)
            watcher_count = len(specs)
            _write_json(
                resolved_cwd / "watch.json",
                {"external": specs, "graph": []},
            )
            state["watch_corrected"] = True
            _write_state(resolved_cwd, state)
            answer = "Corrected the watcher handoff without resubmitting either fixture job."
        else:
            if not (
                _fixture_jobs_complete(resolved_cwd)
                or _reauthorized_fixture_jobs_complete(contract)
            ):
                yield AgentEvent(
                    event="error",
                    text="Acceptance fixture watcher woke before both detached jobs completed.",
                )
                return
            if scenario == "experiment_loop":
                focused_experiment_id = state.get(
                    "focused_experiment_id"
                ) or _focused_experiment_id(contract)
                if not isinstance(focused_experiment_id, str) or not focused_experiment_id:
                    raise ValueError(
                        "Acceptance Experiment state has no persisted focused Experiment id."
                    )
                tested_hypothesis_id = _tested_hypothesis_id(
                    contract,
                    focused_experiment_id,
                )
                _write_json(
                    resolved_cwd / "watch.json",
                    {"external": [], "graph": []},
                )
                _write_json(
                    resolved_cwd / "patch.json",
                    _completion_patch(focused_experiment_id, tested_hypothesis_id),
                )
                answer = "Inspected both fixture jobs and completed the control Experiment."
            else:
                answer = "Inspected both completed fixture jobs; no graph Patch was needed."
            state["completed"] = True
            _write_state(resolved_cwd, state)

        self._record(
            AcceptanceLaunchRecord(
                scenario=scenario,
                action=action,
                cwd=str(resolved_cwd),
                session_id=stable_session,
                watcher_count=watcher_count,
            )
        )
        yield AgentEvent(event="answer", text=answer)
        if scenario == "campaign":
            yield AgentEvent(
                event="raw",
                text='{"type":"acceptance.campaign.completed"}',
                usage=_campaign_usage(
                    provider,
                    resolved_cwd,
                    stable_session,
                    contract,
                ),
            )
        yield AgentEvent(
            event="provider_exit",
            text=json.dumps(
                {
                    "return_code": 0,
                    "event_counts": {
                        "session": 1,
                        "answer": 1,
                        **({"raw": 1} if scenario == "campaign" else {}),
                    },
                    "explicit_terminal_event": True,
                    "acceptance_agent": True,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        yield AgentEvent(event="done")

    def _record(self, record: AcceptanceLaunchRecord) -> None:
        with self._records_lock:
            self._launch_records.append(record)


def _read_launch_contract(prompt: str) -> str:
    lines = prompt.splitlines()
    if len(lines) < 2:
        raise ValueError("Acceptance-agent launch text has no contract path.")
    path = Path(lines[1].strip())
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Acceptance-agent contract is unreadable: {exc}") from exc


def _read_state(cwd: Path) -> dict[str, object]:
    path = cwd / _STATE_FILE
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Acceptance fixture state is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Acceptance fixture state must be a JSON object.")
    return value


def _scenario(
    prompt: str,
    contract: str,
    state: dict[str, object],
) -> Literal[
    "experiment_loop",
    "generic_watchers",
    "result_view",
    "campaign",
    "unsupported",
]:
    # Result-view Work turns can reuse a cwd that carries older acceptance
    # fixture state. The explicit current contract must win over that receipt.
    if _RESULT_VIEW_AUTHORING_MARKER in prompt or _RESULT_VIEW_AUTHORING_MARKER in contract:
        return "result_view"
    if _campaign_contract(contract) is not None:
        return "campaign"
    persisted = state.get("scenario")
    if persisted in {"experiment_loop", "generic_watchers"}:
        return persisted
    if "# RCP Experiment-loop task contract" in contract:
        return "experiment_loop"
    if ACCEPTANCE_GENERIC_WATCHER_MARKER in prompt or ACCEPTANCE_GENERIC_WATCHER_MARKER in contract:
        return "generic_watchers"
    return "unsupported"


def _campaign_contract(
    contract: str,
) -> (
    tuple[
        Literal["orchestrator", "worker", "report"],
        Literal["fresh", "continuation", "report"],
    ]
    | None
):
    if _CAMPAIGN_ORDINARY_CHILD_MARKER in contract:
        phase = (
            "continuation"
            if "Continue the exact Auto-research child Work assignment" in contract
            else "fresh"
        )
        return "worker", phase
    return _CAMPAIGN_CONTRACTS.get(contract.partition("\n")[0])


def _holds_reauthorized_exhaustion(
    state: dict[str, object],
    contract: tuple[
        Literal["orchestrator", "worker", "report"],
        Literal["fresh", "continuation", "report"],
    ]
    | None,
) -> bool:
    fixture = state.get(_CAMPAIGN_FIXTURE_STATE_KEY)
    return bool(
        contract == ("orchestrator", "continuation")
        and isinstance(fixture, dict)
        and fixture.get("directive") == "exhaust"
        and fixture.get("exhausted_once") is True
    )


async def _accept_campaign_turn(
    cwd: Path,
    state: dict[str, object],
    session_id: str,
    *,
    prompt: str,
    contract: str,
    role: Literal["orchestrator", "worker", "report"],
    phase: Literal["fresh", "continuation", "report"],
    control: AgentProcessControl | None,
) -> str:
    receipt = state.get(_CAMPAIGN_STATE_KEY)
    expected_role = "orchestrator" if role == "report" else role
    if receipt is None:
        if phase != "fresh" or role == "report":
            raise ValueError("Acceptance campaign continuation has no persisted actor session.")
        receipt = {
            "cwd": str(cwd),
            "role": expected_role,
            "session_id": session_id,
        }
    elif not isinstance(receipt, dict):
        raise ValueError("Acceptance campaign actor state must be a JSON object.")
    elif receipt.get("cwd") != str(cwd):
        raise ValueError("Acceptance campaign continuation changed its actor-owned cwd.")
    elif receipt.get("role") != expected_role:
        raise ValueError("Acceptance campaign continuation changed its actor role.")
    elif receipt.get("session_id") != session_id:
        raise ValueError("Acceptance campaign continuation changed its native session.")

    updated_state = dict(state)
    updated_state[_CAMPAIGN_STATE_KEY] = receipt

    if role == "report":
        report_attempts = updated_state.get("campaign_report_attempts", 0)
        if not isinstance(report_attempts, int) or report_attempts < 0:
            raise ValueError("Acceptance episode report state is malformed.")
        updated_state["campaign_report_attempts"] = report_attempts + 1
        if "- exact report correction diagnostic: `" not in contract:
            _write_state(cwd, updated_state)
            return "Left the first acceptance episode report attempt missing for correction."
        output_path = _campaign_contract_path(
            contract,
            "- self-contained sandbox-safe HTML report: `",
        )
        if output_path.parent.resolve() != cwd or output_path.name != "episode-report.html":
            raise ValueError("Acceptance episode report output left its exact actor stage.")
        _write_text_atomically(output_path, _campaign_report_html(contract))
        _write_state(cwd, updated_state)
        return "Wrote the corrected deterministic acceptance episode report."

    fixture = updated_state.get(_CAMPAIGN_FIXTURE_STATE_KEY)
    if fixture is not None and not isinstance(fixture, dict):
        raise ValueError("Acceptance campaign fixture state must be a JSON object.")
    fixture_state = dict(fixture) if isinstance(fixture, dict) else {}

    if role == "worker":
        reply_prefix = _campaign_command_prefix(contract, "- Reply command prefix: `")
        reply_template = _campaign_ordinary_child_reply_template(contract)
        instruction_path = _campaign_optional_path(contract, "- worker instruction: `")
        worker_fixture = False
        held_failure_worker = False
        if instruction_path is not None:
            try:
                instruction = instruction_path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise ValueError(f"Acceptance worker instruction is unreadable: {exc}") from exc
            worker_fixture = _CAMPAIGN_WORKER_REPLY_MARKER in instruction
            held_failure_worker = _CAMPAIGN_FAILURE_WORKER_MARKER in instruction
        elif _CAMPAIGN_ORDINARY_CHILD_MARKER in contract:
            worker_fixture = _CAMPAIGN_WORKER_REPLY_MARKER in prompt
            held_failure_worker = _CAMPAIGN_FAILURE_WORKER_MARKER in prompt
        if held_failure_worker:
            _write_state(cwd, updated_state)
            await _wait_for_campaign_fixture_release(
                cwd,
                active_name=_CAMPAIGN_FAILURE_WORKER_ACTIVE_FILE,
                release_name=_CAMPAIGN_FAILURE_WORKER_RELEASE_FILE,
                label="terminal-failure worker",
                control=control,
            )
            return "Settled the admitted acceptance worker before the failed episode report."
        if worker_fixture and (reply_prefix is not None or reply_template is not None):
            body = "Acceptance worker completed its bounded assignment."
            if reply_prefix is not None:
                first = await _run_campaign_client(reply_prefix, body)
                second = await _run_campaign_client(reply_prefix, body)
            else:
                assert reply_template is not None
                first = await _run_campaign_reply_template(reply_template, body)
                second = await _run_campaign_reply_template(reply_template, body)
            if _command_effect_id(first, "message") != _command_effect_id(second, "message"):
                raise ValueError("Acceptance worker reply key did not deduplicate.")
        _write_state(cwd, updated_state)
        return "Completed the deterministic acceptance campaign worker turn without a graph Patch."

    directive = _campaign_fixture_directive(contract)
    if directive is None:
        persisted_directive = fixture_state.get("directive")
        directive = persisted_directive if isinstance(persisted_directive, str) else None
    if directive is not None:
        fixture_state["directive"] = directive
        if directive == "exhaust":
            if fixture_state.get("exhausted_once"):
                updated_state[_CAMPAIGN_FIXTURE_STATE_KEY] = fixture_state
                _write_state(cwd, updated_state)
                await _wait_for_campaign_fixture_release(
                    cwd,
                    active_name=ACCEPTANCE_CAMPAIGN_REAUTHORIZED_ACTIVE_FILE,
                    release_name=ACCEPTANCE_CAMPAIGN_REAUTHORIZED_RELEASE_FILE,
                    label="reauthorized exhaustion",
                    control=control,
                )
                return "Finished the reauthorized acceptance turn after human Stop."
            command_prefix = _campaign_command_prefix_for_orchestrator(contract)
            if command_prefix is None:
                raise ValueError("Acceptance campaign exhaustion fixture has no command prefix.")
            seat_node_id = _campaign_seat_node(contract)
            instruction_path = _campaign_worker_instruction(
                cwd,
                "exhaustion-probe.md",
                "This worker must not be admitted after the research pot is empty.",
            )
            exhausted = await _run_campaign_client(
                command_prefix,
                "spawn",
                "--key",
                "acceptance-exhaustion-probe",
                "--seat-node",
                seat_node_id,
                "--instruction-file",
                str(instruction_path),
                allow_invalid=True,
            )
            if exhausted.get("status") != "invalid":
                raise ValueError("Acceptance exhaustion probe did not hit the one-pot fence.")
            fixture_state["exhausted_once"] = True
            updated_state[_CAMPAIGN_FIXTURE_STATE_KEY] = fixture_state
            _write_state(cwd, updated_state)
            return "Completed the sole acceptance research allocation and exhausted the one pot."
        if directive == "stop":
            updated_state[_CAMPAIGN_FIXTURE_STATE_KEY] = fixture_state
            _write_state(cwd, updated_state)
            await _wait_for_campaign_stop_release(cwd, control=control)
            return "Finished the already-authorized acceptance turn after human Stop."
        command_prefix = _campaign_command_prefix_for_orchestrator(contract)
        if command_prefix is None:
            raise ValueError("Acceptance campaign fixture has no staged command prefix.")
        if directive == "fail":
            if phase == "fresh" and not fixture_state.get("spawned"):
                seat_node_id = _campaign_seat_node(contract)
                instruction_path = _campaign_worker_instruction(
                    cwd,
                    "terminal-failure-worker.md",
                    _CAMPAIGN_FAILURE_WORKER_MARKER,
                )
                result = await _run_deduplicated_campaign_command(
                    command_prefix,
                    "spawn",
                    "--key",
                    "acceptance-failure-worker",
                    "--seat-node",
                    seat_node_id,
                    "--instruction-file",
                    str(instruction_path),
                )
                fixture_state.update(spawned=True, spawn_result=result)
            updated_state[_CAMPAIGN_FIXTURE_STATE_KEY] = fixture_state
            _write_state(cwd, updated_state)
            await _wait_for_campaign_fixture_release(
                cwd,
                active_name=_CAMPAIGN_FAILURE_ACTIVE_FILE,
                release_name=_CAMPAIGN_FAILURE_RELEASE_FILE,
                label="terminal-failure orchestrator",
                control=control,
            )
            # This import is intentionally local. AcceptanceAgentLauncher is part of the
            # provider layer, while the typed verdict belongs to campaign orchestration.
            from rcp.runs.auto_research_recovery import (
                AutoResearchOrchestratorTerminalFailure,
            )

            raise AutoResearchOrchestratorTerminalFailure(
                "The deterministic acceptance orchestrator reached an unrecoverable "
                "structural failure after checkpointing its native session and stage."
            )
        if directive == "spawn_then_interrupt":
            persisted_seat_node_id = fixture_state.get("seat_node_id")
            seat_node_id = (
                persisted_seat_node_id
                if isinstance(persisted_seat_node_id, str) and persisted_seat_node_id
                else _campaign_seat_node(contract)
            )
            instruction_path = _campaign_worker_instruction(
                cwd,
                "interrupted-worker.md",
                f"{_CAMPAIGN_WORKER_REPLY_MARKER}\n"
                "Complete this worker allocation exactly once across orchestrator recovery.",
            )
            spawn_arguments = (
                "spawn",
                "--key",
                "acceptance-interrupt-spawn",
                "--seat-node",
                seat_node_id,
                "--instruction-file",
                str(instruction_path),
            )
            if not fixture_state.get("spawned"):
                result = await _run_deduplicated_campaign_command(
                    command_prefix,
                    *spawn_arguments,
                )
                fixture_state.update(
                    spawned=True,
                    seat_node_id=seat_node_id,
                    worker_id=_campaign_outcome_identity(result, "worker_id"),
                )
                updated_state[_CAMPAIGN_FIXTURE_STATE_KEY] = fixture_state
                _write_state(cwd, updated_state)
                await _wait_for_campaign_restart(cwd, control=control)
            _clear_campaign_fixture_paths(
                cwd,
                ACCEPTANCE_CAMPAIGN_INTERRUPT_ACTIVE_FILE,
            )
            result = await _run_deduplicated_campaign_command(
                command_prefix,
                *spawn_arguments,
            )
            if _campaign_outcome_identity(result, "worker_id") != fixture_state.get("worker_id"):
                raise ValueError("Acceptance interrupted spawn did not return its durable worker.")
            harvested = await _run_campaign_client(
                command_prefix,
                "inbox",
                "--key",
                "acceptance-harvest-after-interrupt",
                "--harvest",
            )
            if harvested.get("status") != "ok":
                raise ValueError("Acceptance interrupted recovery did not harvest child lifecycle.")
            await _run_deduplicated_campaign_command(
                command_prefix,
                "finish",
                "--key",
                "acceptance-finish-after-interrupt",
            )
            fixture_state["finished"] = True
        elif directive == "finish":
            await _run_deduplicated_campaign_command(
                command_prefix,
                "finish",
                "--key",
                "acceptance-finish",
            )
            fixture_state["finished"] = True
        elif phase == "fresh" and not fixture_state.get("spawned"):
            seat_node_id = _campaign_seat_node(contract)
            instruction_path = _campaign_worker_instruction(
                cwd,
                "bounded-worker.md",
                f"{_CAMPAIGN_WORKER_REPLY_MARKER}\n"
                "Complete the bounded acceptance assignment and reply to the orchestrator.",
            )
            result = await _run_deduplicated_campaign_command(
                command_prefix,
                "spawn",
                "--key",
                "acceptance-spawn",
                "--seat-node",
                seat_node_id,
                "--instruction-file",
                str(instruction_path),
            )
            fixture_state.update(spawned=True, spawn_result=result)
        else:
            await _run_deduplicated_campaign_command(
                command_prefix,
                "finish",
                "--key",
                "acceptance-finish-after-worker",
            )
            fixture_state["finished"] = True
        updated_state[_CAMPAIGN_FIXTURE_STATE_KEY] = fixture_state
    _write_state(cwd, updated_state)
    return f"Completed the deterministic acceptance campaign {role} turn without a graph Patch."


def _campaign_worker_instruction(cwd: Path, filename: str, text: str) -> Path:
    """Write one stable direct file for the real file-backed Spawn client."""

    path = cwd / filename
    _write_text_atomically(path, text.rstrip() + "\n")
    return path


def _campaign_fixture_directive(
    contract: str,
) -> (
    Literal[
        "finish",
        "spawn_then_finish",
        "spawn_then_interrupt",
        "exhaust",
        "stop",
        "fail",
    ]
    | None
):
    instruction_path = _campaign_optional_path(contract, "- starting instruction: `")
    if instruction_path is None:
        return None
    try:
        instruction = instruction_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Acceptance campaign instruction is unreadable: {exc}") from exc
    if ACCEPTANCE_CAMPAIGN_SPAWN_THEN_INTERRUPT_MARKER in instruction:
        return "spawn_then_interrupt"
    if ACCEPTANCE_CAMPAIGN_SPAWN_THEN_FINISH_MARKER in instruction:
        return "spawn_then_finish"
    if ACCEPTANCE_CAMPAIGN_EXHAUST_MARKER in instruction:
        return "exhaust"
    if ACCEPTANCE_CAMPAIGN_STOP_MARKER in instruction:
        return "stop"
    if ACCEPTANCE_CAMPAIGN_FAIL_MARKER in instruction:
        return "fail"
    if ACCEPTANCE_CAMPAIGN_FINISH_MARKER in instruction:
        return "finish"
    return None


async def _wait_for_campaign_stop_release(
    cwd: Path,
    *,
    control: AgentProcessControl | None,
) -> None:
    await _wait_for_campaign_fixture_release(
        cwd,
        active_name=_CAMPAIGN_STOP_ACTIVE_FILE,
        release_name=_CAMPAIGN_STOP_RELEASE_FILE,
        label="Stop",
        control=control,
    )


async def _wait_for_campaign_restart(
    cwd: Path,
    *,
    control: AgentProcessControl | None,
) -> None:
    """Leave a durable active marker until a real app shutdown abandons this stream."""

    if control is None:
        raise ValueError("Acceptance interrupted campaign fixture requires process control.")
    active_path = cwd / ACCEPTANCE_CAMPAIGN_INTERRUPT_ACTIVE_FILE
    _write_text_atomically(active_path, "acceptance campaign awaits an RCP restart\n")
    deadline = asyncio.get_running_loop().time() + 20
    while asyncio.get_running_loop().time() < deadline:
        if control.pause_requested.is_set():
            # App shutdown has already moved the durable task to ``pausing``. Exiting the
            # provider thread without a terminal event leaves that row for the next app
            # instance's ordinary restart sweep to classify as ``interrupted``.
            raise SystemExit("acceptance campaign provider stream ended with the RCP process")
        await asyncio.sleep(0.01)
    raise ValueError("Acceptance interrupted campaign fixture did not observe an RCP restart.")


def _clear_campaign_fixture_paths(cwd: Path, *names: str) -> None:
    for name in names:
        with suppress(FileNotFoundError):
            (cwd / name).unlink()


async def _wait_for_campaign_fixture_release(
    cwd: Path,
    *,
    active_name: str,
    release_name: str,
    label: str,
    control: AgentProcessControl | None = None,
) -> None:
    active_path = cwd / active_name
    release_path = cwd / release_name
    _prepare_campaign_fixture_active(cwd, active_name=active_name, label=label)
    try:
        deadline = asyncio.get_running_loop().time() + 120
        while asyncio.get_running_loop().time() < deadline:
            if control is not None and control.pause_requested.is_set():
                raise _AcceptancePauseRequested
            if release_path.is_file():
                return
            await asyncio.sleep(0.01)
        raise ValueError(
            f"Acceptance campaign {label} fixture was not released before its deadline."
        )
    finally:
        for path in (active_path, release_path):
            with suppress(FileNotFoundError):
                path.unlink()


class _AcceptancePauseRequested(Exception):
    """Translate an in-process acceptance hold into the provider Pause contract."""


def _prepare_campaign_fixture_active(cwd: Path, *, active_name: str, label: str) -> None:
    _write_text_atomically(
        cwd / active_name,
        f"acceptance campaign {label} turn is active\n",
    )


def _campaign_command_prefix_for_orchestrator(contract: str) -> str | None:
    return _campaign_command_prefix(
        contract,
        "- Command prefix for this turn: `",
    ) or _campaign_command_prefix(contract, "- Command prefix: `")


def _campaign_command_prefix(contract: str, prefix: str) -> str | None:
    lines = [line for line in contract.splitlines() if line.startswith(prefix)]
    if not lines:
        return None
    if len(lines) != 1 or not lines[0].endswith("`"):
        raise ValueError("Acceptance campaign contract has a malformed command prefix.")
    value = lines[0][len(prefix) : -1]
    if not value or "`" in value:
        raise ValueError("Acceptance campaign contract has a malformed command prefix.")
    return value


def _campaign_ordinary_child_reply_template(contract: str) -> str | None:
    marker = "optional reply to your orchestrator:"
    if marker not in contract:
        return None
    tail = contract.partition(marker)[2]
    line = next((item.strip() for item in tail.splitlines() if item.strip()), "")
    if not line.startswith("`") or not line.endswith("`") or "`" in line[1:-1]:
        raise ValueError("Acceptance ordinary child contract has a malformed reply command.")
    value = line[1:-1]
    try:
        argv = shlex.split(value)
    except ValueError as exc:
        raise ValueError("Acceptance ordinary child reply command is invalid.") from exc
    if argv.count("<idempotency-key>") != 1 or argv.count("<reply-body>") != 1:
        raise ValueError("Acceptance ordinary child reply command lost its placeholders.")
    return value


async def _run_deduplicated_campaign_command(
    command_prefix: str,
    *arguments: str,
) -> dict[str, object]:
    first = await _run_campaign_client(command_prefix, *arguments)
    second = await _run_campaign_client(command_prefix, *arguments)
    first_outcome = _command_outcome(first)
    verb = arguments[0] if arguments else ""
    if _command_effect_id(first, verb) != _command_effect_id(second, verb):
        raise ValueError("Acceptance campaign command key did not return its durable result.")
    return first_outcome


def _command_outcome(response: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in response.items() if key != "request_id"}


def _campaign_outcome_identity(outcome: dict[str, object], key: str) -> object:
    result = outcome.get("result")
    if not isinstance(result, dict) or result.get(key) is None:
        raise ValueError("Acceptance campaign command outcome has no durable effect identity.")
    return result[key]


def _command_effect_id(response: dict[str, object], verb: str) -> object:
    if response.get("status") != "ok":
        raise ValueError("Acceptance campaign staged command did not return an ok outcome.")
    result = response.get("result")
    if not isinstance(result, dict):
        raise ValueError("Acceptance campaign staged command has no result object.")
    key = {"spawn": "worker_id", "message": "message_id", "finish": "episode_id"}.get(verb)
    if key is None or result.get(key) is None:
        raise ValueError("Acceptance campaign staged command has no durable effect identity.")
    return result[key]


async def _run_campaign_client(
    command_prefix: str,
    *arguments: str,
    allow_invalid: bool = False,
) -> dict[str, object]:
    try:
        argv = [*shlex.split(command_prefix), *arguments]
    except ValueError as exc:
        raise ValueError(f"Acceptance campaign command prefix is invalid: {exc}") from exc
    if not argv:
        raise ValueError("Acceptance campaign command prefix is empty.")
    return await _run_campaign_argv(argv, allow_invalid=allow_invalid)


async def _run_campaign_reply_template(
    command: str,
    body: str,
) -> dict[str, object]:
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        raise ValueError("Acceptance ordinary child reply command is invalid.") from exc
    argv[argv.index("<idempotency-key>")] = "acceptance-worker-reply"
    argv[argv.index("<reply-body>")] = body
    return await _run_campaign_argv(argv)


async def _run_campaign_argv(
    argv: list[str],
    *,
    allow_invalid: bool = False,
) -> dict[str, object]:
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    output = stdout.decode("utf-8", errors="replace")
    diagnostic = stderr.decode("utf-8", errors="replace")
    if process.returncode != 0 and not (allow_invalid and process.returncode == 1):
        raise ValueError(
            "Acceptance campaign staged command failed "
            f"with exit {process.returncode}: {(diagnostic or output).strip()}"
        )
    try:
        result = json.loads(output)
    except json.JSONDecodeError as exc:
        raise ValueError("Acceptance campaign staged command returned invalid JSON.") from exc
    if not isinstance(result, dict):
        raise ValueError("Acceptance campaign staged command returned a non-object result.")
    return result


def _campaign_seat_node(contract: str) -> str:
    graph_path = _campaign_contract_path(contract, "- graph: `")
    try:
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Acceptance campaign graph is unreadable: {exc}") from exc
    nodes = graph.get("nodes") if isinstance(graph, dict) else None
    if not isinstance(nodes, dict):
        raise ValueError("Acceptance campaign graph has no node map.")
    candidates = sorted(
        node_id
        for node_id, node in nodes.items()
        if isinstance(node_id, str)
        and isinstance(node, dict)
        and node.get("type") in {"experiment", "blocker"}
    )
    if not candidates:
        raise ValueError("Acceptance campaign fixture needs an existing Experiment or Blocker.")
    return candidates[0]


def _campaign_optional_path(contract: str, prefix: str) -> Path | None:
    lines = [line for line in contract.splitlines() if line.startswith(prefix)]
    if not lines:
        return None
    if len(lines) != 1 or not lines[0].endswith("`"):
        raise ValueError("Acceptance campaign contract has a malformed path.")
    value = lines[0][len(prefix) : -1]
    if not value or "`" in value:
        raise ValueError("Acceptance campaign contract has a malformed path.")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("Acceptance campaign contract path must be absolute.")
    return path


def _campaign_contract_path(contract: str, prefix: str) -> Path:
    path = _campaign_optional_path(contract, prefix)
    if path is None:
        raise ValueError("Acceptance campaign contract is missing a required path.")
    return path


def _campaign_report_html(contract: str) -> str:
    ending = _campaign_report_ending(contract)
    outcome = {
        "completed": (
            "The orchestrator seated one bounded worker, received its result, and concluded "
            "the episode."
        ),
        "exhausted": (
            "The sole research allocation settled normally, then the shared invocation pot "
            "reached its operational ceiling."
        ),
        "failed": (
            "The orchestrator reached an explicitly typed unrecoverable structural failure; "
            "this is a partial report produced only after its admitted worker settled."
        ),
    }.get(ending, f"The episode ended as {ending} after its admitted work settled.")
    return f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Acceptance episode report</title>
<article>
  <h1>Acceptance episode conclusion</h1>
  <p>{outcome}</p>
  <figure aria-label="Episode progression">
    <svg viewBox="0 0 320 80" role="img" aria-label="Operational work flowed into report wrap-up">
      <rect x="8" y="20" width="120" height="40" rx="8" fill="#d8e7ff" />
      <path d="M136 40 H190" stroke="#345" stroke-width="4" />
      <rect x="198" y="20" width="114" height="40" rx="8" fill="#d9f4df" />
      <text x="68" y="45" text-anchor="middle">Operational work</text>
      <text x="255" y="45" text-anchor="middle">Visual report</text>
    </svg>
    <figcaption>The hidden report continuation reused the exact episode session.</figcaption>
  </figure>
  <p>The first report attempt was deliberately incomplete; this corrected document reused the
  exact orchestrator session and stage without repeating operational work.</p>
  <p>Human review remains authoritative for any protected belief change.</p>
</article>
</html>
"""


def _campaign_report_ending(contract: str) -> str:
    marker = " at ending `"
    for line in contract.splitlines():
        if marker in line and line.endswith("`."):
            ending = line.rsplit(marker, 1)[1][:-2]
            if ending:
                return ending
    return "completed"


def _campaign_usage(
    provider: str,
    cwd: Path,
    session_id: str,
    contract: str,
) -> ProviderUsage:
    digest = hashlib.sha256(f"{provider}\0{cwd}\0{session_id}\0{contract}".encode()).hexdigest()
    return ProviderUsage(
        provider_profile=f"acceptance.{provider}.campaign.v1",
        provider_event_type="acceptance.campaign.completed",
        dedupe_key=digest,
        processed_input_tokens=256,
        generated_tokens=32,
        reported_input_tokens=256,
        reported_output_tokens=32,
        reported_total_tokens=288,
        provider_fields={"acceptance_agent": True, "scenario": "campaign"},
    )


def _result_view_action(contract: str) -> Literal["create", "revise"]:
    section = _result_view_section(contract)
    action_lines = [
        line
        for line in section.splitlines()
        if line.startswith((_RESULT_VIEW_CREATE_PREFIX, _RESULT_VIEW_REVISE_PREFIX))
    ]
    if len(action_lines) != 1:
        raise ValueError("Acceptance result-view contract must name exactly one authoring action.")
    return "create" if action_lines[0].startswith(_RESULT_VIEW_CREATE_PREFIX) else "revise"


def _result_view_section(contract: str) -> str:
    sections = [
        section
        for section in contract.split("\n\n")
        if section.startswith(f"{_RESULT_VIEW_AUTHORING_MARKER}\n")
    ]
    if len(sections) != 1:
        raise ValueError("Acceptance result-view contract must contain one authoring section.")
    return sections[0]


def _author_result_view(
    cwd: Path,
    contract: str,
    state: dict[str, object],
    session_id: str,
    action: Literal["create", "revise"],
) -> str:
    if action == "create":
        slot = _result_view_slot(cwd, contract)
        try:
            entries = list(slot.iterdir())
        except OSError as exc:
            raise ValueError(f"Acceptance result-view slot is unreadable: {exc}") from exc
        if entries:
            raise ValueError("Acceptance result-view create slot must be empty.")
        target = slot / _RESULT_VIEW_NAME
        _write_text_atomically(target, _result_view_html(revision=1))
        updated_state = dict(state)
        updated_state[_RESULT_VIEW_STATE_KEY] = {
            "cwd": str(cwd),
            "path": str(target),
            "revision": 1,
            "session_id": session_id,
        }
        _write_state(cwd, updated_state)
        return "Created the deterministic acceptance loss-curves result view."

    target = _existing_result_view_path(cwd, contract)
    receipt = state.get(_RESULT_VIEW_STATE_KEY)
    if not isinstance(receipt, dict):
        raise ValueError("Acceptance result-view revision has no persisted create receipt.")
    if receipt.get("cwd") != str(cwd):
        raise ValueError("Acceptance result-view revision changed the conversation cwd.")
    if receipt.get("session_id") != session_id:
        raise ValueError("Acceptance result-view revision changed the native session.")
    if receipt.get("path") != str(target):
        raise ValueError("Acceptance result-view revision changed the stable file path.")
    if receipt.get("revision") != 1:
        raise ValueError("Acceptance result-view fixture supports exactly one revision.")
    try:
        entries = list(target.parent.iterdir())
    except OSError as exc:
        raise ValueError(f"Acceptance result-view slot is unreadable: {exc}") from exc
    if len(entries) != 1 or entries[0].resolve() != target:
        raise ValueError("Acceptance result-view revision requires exactly the created view file.")

    _write_text_atomically(target, _result_view_html(revision=2))
    updated_state = dict(state)
    updated_state[_RESULT_VIEW_STATE_KEY] = {
        "cwd": str(cwd),
        "path": str(target),
        "revision": 2,
        "session_id": session_id,
    }
    _write_state(cwd, updated_state)
    return "Revised the existing acceptance loss-curves result view in place."


def _result_view_slot(cwd: Path, contract: str) -> Path:
    raw_path = _result_view_contract_path(
        contract,
        prefix=_RESULT_VIEW_CREATE_PREFIX,
        suffix=_RESULT_VIEW_CREATE_SUFFIX,
    )
    slot = _resolve_existing_result_view_path(raw_path, label="create slot")
    relative = _result_view_relative_path(cwd, slot)
    if len(relative.parts) != 2 or relative.parts[0] != "views" or not slot.is_dir():
        raise ValueError("Acceptance result-view create slot is not the stable stage slot.")
    return slot


def _existing_result_view_path(cwd: Path, contract: str) -> Path:
    raw_path = _result_view_contract_path(
        contract,
        prefix=_RESULT_VIEW_REVISE_PREFIX,
        suffix=_RESULT_VIEW_REVISE_SUFFIX,
    )
    if raw_path.is_symlink():
        raise ValueError("Acceptance result-view revision target cannot be a symlink.")
    target = _resolve_existing_result_view_path(raw_path, label="revision target")
    relative = _result_view_relative_path(cwd, target)
    if (
        len(relative.parts) != 3
        or relative.parts[0] != "views"
        or target.name != _RESULT_VIEW_NAME
        or not target.is_file()
    ):
        raise ValueError("Acceptance result-view revision target is not the stable HTML file.")
    return target


def _result_view_contract_path(contract: str, *, prefix: str, suffix: str) -> Path:
    matches = [
        line[len(prefix) : -len(suffix)]
        for line in _result_view_section(contract).splitlines()
        if line.startswith(prefix) and line.endswith(suffix)
    ]
    if len(matches) != 1 or not matches[0] or "`" in matches[0]:
        raise ValueError("Acceptance result-view contract has no exact stable path.")
    path = Path(matches[0])
    if not path.is_absolute():
        raise ValueError("Acceptance result-view contract path must be absolute.")
    return path


def _resolve_existing_result_view_path(path: Path, *, label: str) -> Path:
    try:
        return path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Acceptance result-view {label} is unavailable: {exc}") from exc


def _result_view_relative_path(cwd: Path, path: Path) -> Path:
    try:
        return path.relative_to(cwd)
    except ValueError as exc:
        raise ValueError("Acceptance result-view path left the conversation cwd.") from exc


def _action(
    contract: str,
    state: dict[str, object],
) -> Literal["initial", "watch_correction", "wake"]:
    first_line = contract.partition("\n")[0].lower()
    if "watch correction" in first_line or "watcher correction" in first_line:
        return "watch_correction"
    if _experiment_loop_phase(contract) == "human_reauthorization":
        return "wake"
    if state.get("jobs_started"):
        return "wake"
    return "initial"


def _experiment_loop_phase(contract: str) -> str | None:
    prefix = "- Loop control for this invocation: `"
    for line in contract.splitlines():
        if not (line.startswith(prefix) and line.endswith("`")):
            continue
        try:
            value = json.loads(Path(line[len(prefix) : -1]).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Acceptance Experiment loop control is unreadable: {exc}") from exc
        return value.get("phase") if isinstance(value, dict) else None
    return None


def _focused_experiment_id(contract: str) -> str | None:
    prefix = "- Focused Experiment id: `"
    for line in contract.splitlines():
        if line.startswith(prefix) and line.endswith("`"):
            return line[len(prefix) : -1]
    return None


def _tested_hypothesis_id(contract: str, focused_experiment_id: str) -> str:
    prefixes = (
        "- Current graph, including the Experiment's attempts: `",
        "- current graph: `",
    )
    graph_path: Path | None = None
    for line in contract.splitlines():
        for prefix in prefixes:
            if line.startswith(prefix) and line.endswith("`"):
                graph_path = Path(line[len(prefix) : -1])
                break
        if graph_path is not None:
            break
    if graph_path is None:
        raise ValueError("Acceptance Experiment wake contract has no current graph path.")
    try:
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Acceptance Experiment graph is unreadable: {exc}") from exc
    if not isinstance(graph, dict) or not isinstance(graph.get("edges"), dict):
        raise ValueError("Acceptance Experiment graph has no edge map.")
    matches = [
        edge.get("target")
        for edge in graph["edges"].values()
        if isinstance(edge, dict)
        and edge.get("source") == focused_experiment_id
        and edge.get("relation") == "tests"
        and isinstance(edge.get("target"), str)
    ]
    if len(matches) != 1:
        raise ValueError("Acceptance Experiment fixture must have exactly one tested Hypothesis.")
    return matches[0]


def _start_fixture_jobs(cwd: Path) -> None:
    jobs = cwd / _JOBS_DIRECTORY
    jobs.mkdir(parents=True, exist_ok=True)
    for name in ("job-one", "job-two"):
        done_path = jobs / f"{name}.done"
        status_path = jobs / f"{name}.status"
        log_path = jobs / f"{name}.log"
        if done_path.exists() or status_path.exists() or log_path.exists():
            raise ValueError(f"Acceptance fixture job {name} already has persistent artifacts.")
        status_path.write_text("running\n", encoding="utf-8")
        log_path.write_text(f"{name}: started\n", encoding="utf-8")
        code = (
            "import pathlib,sys,time\n"
            "done,status,log,name,delay=sys.argv[1:]\n"
            "time.sleep(float(delay))\n"
            "pathlib.Path(log).open('a', encoding='utf-8').write(f'{name}: completed\\n')\n"
            "pathlib.Path(status).write_text('completed\\n', encoding='utf-8')\n"
            "pathlib.Path(done).write_text('done\\n', encoding='utf-8')\n"
        )
        subprocess.Popen(  # noqa: S603 - fixed interpreter and internal fixture payload
            [
                sys.executable,
                "-c",
                code,
                str(done_path),
                str(status_path),
                str(log_path),
                name,
                str(ACCEPTANCE_AGENT_JOB_SECONDS),
            ],
            cwd=jobs,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )


def _watch_specs(cwd: Path) -> list[dict[str, str]]:
    jobs = cwd / _JOBS_DIRECTORY
    return [
        {
            "check_command": f"test -f {str(jobs / f'{name}.done')!r}",
            "log_path": str(jobs / f"{name}.log"),
            "cwd": str(jobs),
        }
        for name in ("job-one", "job-two")
    ]


def _fixture_jobs_complete(cwd: Path) -> bool:
    jobs = cwd / _JOBS_DIRECTORY
    return all((jobs / f"{name}.done").is_file() for name in ("job-one", "job-two"))


def _reauthorized_fixture_jobs_complete(contract: str) -> bool:
    """Inspect delivered watcher evidence when a human Run starts in a fresh chat stage."""

    prefix = "- Current watcher state for this Experiment: `"
    for line in contract.splitlines():
        if not (line.startswith(prefix) and line.endswith("`")):
            continue
        try:
            value = json.loads(Path(line[len(prefix) : -1]).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Acceptance Experiment watcher state is unreadable: {exc}") from exc
        if not isinstance(value, list) or not value:
            return False
        delivered = [item for item in value if isinstance(item, dict) and item.get("notified")]
        return bool(delivered) and all(
            item.get("status") == "completed"
            and isinstance(item.get("log_path"), str)
            and Path(item["log_path"]).is_file()
            for item in delivered
        )
    return False


def _completion_patch(
    focused_experiment_id: str,
    tested_hypothesis_id: str,
) -> dict[str, object]:
    evidence_id = "ev/acceptance-result"
    evidence_edge_id = "edge/acceptance-supports"
    return {
        "summary": "Completed the deterministic acceptance Experiment with supporting evidence.",
        "ops": [
            {
                "op": "update_nodes",
                "nodes": [
                    {
                        "id": focused_experiment_id,
                        "changes": {"status": "completed"},
                    }
                ],
            },
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": evidence_id,
                        "type": "evidence",
                        "title": "Acceptance jobs completed",
                        "observation": (
                            "Both deterministic CPU-only acceptance jobs reached their "
                            "completed status and wrote their expected logs."
                        ),
                        "interpretation": (
                            "The bounded control loop delivered and inspected both watcher "
                            "completions."
                        ),
                        "role": "result",
                        "validity": "valid",
                        "origin": "internal_run",
                    }
                ],
            },
            {
                "op": "create_edges",
                "edges": [
                    {
                        "id": "edge/acceptance-produces",
                        "source": focused_experiment_id,
                        "target": evidence_id,
                        "relation": "produces",
                        "explanation": "The acceptance Experiment produced the fixture result.",
                    },
                    {
                        "id": evidence_edge_id,
                        "source": evidence_id,
                        "target": tested_hypothesis_id,
                        "relation": "supports",
                        "explanation": (
                            "The completed watcher sequence supports the fixture Hypothesis."
                        ),
                        "assessment": {
                            "relevance": "direct",
                            "weight": "strong",
                            "scope": "The deterministic CPU-only acceptance control loop.",
                            "qualifications": [],
                        },
                    },
                ],
            },
            {
                "op": "create_proposals",
                "proposals": [
                    {
                        "id": "prop/acceptance-result",
                        "title": "Accept the acceptance-loop result",
                        "card": {
                            "situation_cold": (
                                "Both deterministic acceptance jobs completed and their "
                                "watchers were delivered."
                            ),
                            "why_human_now": (
                                "Only a human may accept the resulting Hypothesis status change."
                            ),
                            "consequences": (
                                "Accepting marks the tested fixture Hypothesis as supported."
                            ),
                            "decision_needed": "Approve or reject the supported status.",
                        },
                        "ops": [
                            {
                                "op": "update_nodes",
                                "intent": "status_change",
                                "nodes": [
                                    {
                                        "id": tested_hypothesis_id,
                                        "changes": {"status": "supported"},
                                        "cause": {
                                            "kind": "evidence_edge",
                                            "ref_id": evidence_edge_id,
                                        },
                                    }
                                ],
                            }
                        ],
                    }
                ],
            },
        ],
        "repositories_read": [],
        "change_summary": [
            "Completed the control Experiment after both acceptance jobs finished.",
            "Recorded the fixture result as supporting evidence for the tested Hypothesis.",
            "Proposed marking the tested fixture Hypothesis as supported.",
        ],
    }


def _result_view_html(*, revision: Literal[1, 2]) -> str:
    revision_label = (
        "Revision 1 — initial curves" if revision == 1 else "Revision 2 — late spike annotated"
    )
    annotation = (
        ""
        if revision == 1
        else """
          <circle cx="653" cy="91" r="10" class="annotation-dot" />
          <text x="518" y="72" class="annotation-label">reviewed late spike</text>"""
    )
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Acceptance loss curves by seed</title>
  <style>
    :root { color-scheme: light; font-family: ui-sans-serif, system-ui, sans-serif; }
    body { margin: 0; background: #f4f1ea; color: #18251f; }
    main { max-width: 860px; margin: 0 auto; padding: 28px; }
    header { display: flex; align-items: end; justify-content: space-between; gap: 20px; }
    h1 { margin: 0; font-family: Georgia, serif; font-size: 30px; font-weight: 600; }
    .revision { color: #315f4c; font-size: 14px; font-weight: 700; }
    .plot-card { margin-top: 18px; padding: 16px; background: #fffdf8; border: 1px solid #d4cec1;
      border-radius: 14px; box-shadow: 0 12px 30px rgb(40 48 43 / 9%); }
    svg { display: block; width: 100%; height: auto; user-select: none; }
    .grid { stroke: #e8e2d7; stroke-width: 1; }
    .axis { stroke: #69756f; stroke-width: 1.5; }
    .axis-label { fill: #5c6862; font-size: 13px; }
    .seed-one { fill: none; stroke: #26745b; stroke-width: 4; }
    .seed-two { fill: none; stroke: #d0693d; stroke-width: 4; }
    .selection { fill: rgb(63 113 181 / 15%); stroke: #315f9a; stroke-width: 2;
      stroke-dasharray: 7 5; }
    .annotation-dot { fill: #fffdf8; stroke: #a53d2d; stroke-width: 4; }
    .annotation-label { fill: #873326; font-size: 14px; font-weight: 700; }
    #gesture-surface { cursor: crosshair; touch-action: none; }
    .legend { display: flex; gap: 20px; margin: 12px 0 0; font-size: 14px; }
    .legend span::before { display: inline-block; width: 22px; height: 4px; margin: 0 7px 3px 0;
      border-radius: 2px; content: ""; }
    .legend .one::before { background: #26745b; }
    .legend .two::before { background: #d0693d; }
    .instruction { margin: 16px 0 0; color: #45534c; font-size: 15px; }
    #selection-summary { min-height: 24px; margin: 8px 0 0; color: #315f9a; font-weight: 700; }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Loss curves by seed</h1>
        <div class="instruction">Drag a box over the plot to point at a region.</div>
      </div>
      <div class="revision" data-revision="__REVISION__">__REVISION_LABEL__</div>
    </header>
    <section class="plot-card" aria-label="Overlaid loss curves">
      <svg id="curve-plot" viewBox="0 0 760 380" role="img"
           aria-label="Training loss by step for seed two and seed three">
        <line x1="64" y1="42" x2="64" y2="322" class="axis" />
        <line x1="64" y1="322" x2="724" y2="322" class="axis" />
        <line x1="64" y1="112" x2="724" y2="112" class="grid" />
        <line x1="64" y1="182" x2="724" y2="182" class="grid" />
        <line x1="64" y1="252" x2="724" y2="252" class="grid" />
        <text x="64" y="348" class="axis-label">0</text>
        <text x="361" y="348" class="axis-label">5k steps</text>
        <text x="686" y="348" class="axis-label">10k</text>
        <text x="15" y="188" class="axis-label" transform="rotate(-90 15 188)">loss</text>
        <polyline class="seed-one"
          points="64,72 130,126 196,166 262,205 328,236 394,258 460,276 526,288 592,297 658,303 724,307" />
        <polyline class="seed-two"
          points="64,82 130,133 196,176 262,214 328,244 394,265 460,282 526,294 592,302 625,300 653,91 682,299 724,306" />
        __REVISION_ANNOTATION__
        <rect id="selection" class="selection" x="0" y="0" width="0" height="0" hidden />
        <rect id="gesture-surface" x="64" y="42" width="660" height="280" fill="transparent" />
      </svg>
      <div class="legend"><span class="one">seed 2</span><span class="two">seed 3</span></div>
      <p id="selection-summary" aria-live="polite"></p>
    </section>
  </main>
  <script>
    (() => {
      'use strict';
      const plot = document.getElementById('curve-plot');
      const surface = document.getElementById('gesture-surface');
      const selection = document.getElementById('selection');
      const summary = document.getElementById('selection-summary');
      let start = null;
      const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
      const pointFor = (event) => {
        const bounds = plot.getBoundingClientRect();
        return {
          x: clamp((event.clientX - bounds.left) * 760 / bounds.width, 64, 724),
          y: clamp((event.clientY - bounds.top) * 380 / bounds.height, 42, 322)
        };
      };
      const drawSelection = (first, last) => {
        selection.hidden = false;
        selection.setAttribute('x', String(Math.min(first.x, last.x)));
        selection.setAttribute('y', String(Math.min(first.y, last.y)));
        selection.setAttribute('width', String(Math.abs(first.x - last.x)));
        selection.setAttribute('height', String(Math.abs(first.y - last.y)));
      };
      surface.addEventListener('pointerdown', (event) => {
        event.preventDefault();
        start = pointFor(event);
        drawSelection(start, start);
      });
      surface.addEventListener('pointermove', (event) => {
        if (start === null) return;
        drawSelection(start, pointFor(event));
      });
      surface.addEventListener('pointerup', (event) => {
        if (start === null) return;
        const end = pointFor(event);
        drawSelection(start, end);
        const width = Math.abs(start.x - end.x);
        const height = Math.abs(start.y - end.y);
        start = null;
        if (width < 12 || height < 12) {
          selection.hidden = true;
          return;
        }
        summary.textContent = 'Boxed late spike across steps 8,000–9,000 for seed 3.';
        window.parent.postMessage({type:'rcp-result-view-gesture',version:1,gesture:'box',description:'late spike across steps 8,000–9,000 for seed 3'}, '*');
      });
      surface.addEventListener('pointercancel', () => {
        start = null;
        selection.hidden = true;
      });
    })();
  </script>
</body>
</html>
"""
    return (
        template.replace("__REVISION__", str(revision))
        .replace("__REVISION_LABEL__", revision_label)
        .replace("__REVISION_ANNOTATION__", annotation)
    )


def _write_text_atomically(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.acceptance.tmp")
    created = False
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            created = True
            handle.write(value)
        temporary.replace(path)
    finally:
        if created and temporary.exists():
            temporary.unlink()


def _write_state(cwd: Path, value: dict[str, object]) -> None:
    _write_json(cwd / _STATE_FILE, value)


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
