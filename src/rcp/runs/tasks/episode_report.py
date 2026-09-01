from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import AsyncIterator
from contextlib import aclosing, suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from rcp.agents import AgentEvent, AgentLauncher, PromptFactory
from rcp.agents.episode_report_prompt import episode_report_task_contract
from rcp.agents.write_scope import resolve_project_write_scope
from rcp.artifacts import validate_artifact_bytes
from rcp.limits import CHAT_ARTIFACT_MAX_FILE_BYTES
from rcp.providers import AgentCapability, ProviderId
from rcp.runs.shared import (
    _ProviderOutcome,
    _record_agent_launch_receipt,
    _sse,
    _stage_or_reuse_task_input,
    _stream_agent_events,
    _task_token,
)
from rcp.service import ProjectService, RunRequest
from rcp.skill_registry import SkillSelection, official_registry
from rcp.skills.staging import skill_bundle_label, stage_skill_selection
from rcp.storage import (
    AgentTaskRecord,
    EpisodeRecord,
    EpisodeReportAttemptRecord,
    EpisodeReportRecord,
    EpisodeWrapupRecord,
)
from rcp.transport import RemoteRunStage, RunStageMailbox, StateUnavailable

if TYPE_CHECKING:
    from rcp.background import AgentTaskExecution

_REPORT_SKILL_ID = "episode-report"
_REPORT_OUTPUT_DEFAULT = "episode-report.html"


class EpisodeReportRunRequest(BaseModel):
    """The complete, frozen provider binding for one hidden report allocation."""

    model_config = ConfigDict(extra="forbid", strict=True)

    episode_id: str = Field(min_length=1)
    provider: ProviderId
    model: str
    reasoning: str
    run_on: str = Field(min_length=1)
    execution_host: str
    session_id: str = Field(min_length=1)


@dataclass(frozen=True)
class _CanonicalReportTurn:
    task: AgentTaskRecord
    request: EpisodeReportRunRequest
    episode: EpisodeRecord
    wrapup: EpisodeWrapupRecord


@dataclass(frozen=True)
class _ReportStage:
    local: Path | None
    remote: RemoteRunStage | None
    workspace: Path
    execution_host: str
    provider_binary: str | None


async def stream_episode_report_run(
    service: ProjectService,
    launcher: AgentLauncher,
    request: EpisodeReportRunRequest,
    execution: AgentTaskExecution,
) -> AsyncIterator[str]:
    """Run at most three hidden report calls inside one exact episode continuation.

    Operational context is deliberately absent. The native provider session supplies its retained
    context; RCP adds only the immutable compact receipt, the official report skill, and an optional
    diagnostic produced by the immediately preceding report attempt.
    """

    try:
        turn = _canonical_report_turn(execution, request)
        existing = execution.store.episode_report(turn.episode.episode_id)
        if existing is not None:
            if turn.wrapup.state != "ready":
                raise ValueError("The episode report exists outside a ready wrap-up.")
            yield _report_ready_frame(existing)
            yield _sse(AgentEvent(event="done"))
            return
        if turn.wrapup.state == "failed":
            yield _sse(
                AgentEvent(
                    event="error",
                    text=turn.wrapup.diagnostic or "Episode report generation failed.",
                )
            )
            return

        stage = _open_exact_report_stage(service, turn)
        output_name, report_output_path = _exact_report_output(turn.wrapup, stage)
        mailbox = RunStageMailbox.for_stage(local_stage=stage.local, remote_stage=stage.remote)
        capability = _report_capability(execution, turn.wrapup)
        write_scope = resolve_project_write_scope(
            manifest=service.manifest,
            project_id=turn.task.project_id,
            execution_machine=turn.request.run_on,
            capability=capability,
            stage_root=(
                str(stage.remote.root)
                if stage.remote is not None and stage.remote.root is not None
                else str(stage.workspace)
            ),
            workspace_root=str(stage.workspace),
            admitted_aliases=[],
            repository_pointers=[],
            remote_stage=stage.remote,
            app_data_dir=None,
            repository_inventory=service.repository_ownership_inventory(
                project_id=turn.task.project_id
            ),
        )
        attempt = execution.store.current_episode_report_attempt(turn.episode.episode_id)
        if attempt is not None and attempt.status == "running":
            recovered = _reconcile_running_attempt(execution, turn, attempt, mailbox, output_name)
            if recovered is not None:
                yield _report_ready_frame(recovered)
                yield _sse(AgentEvent(event="done"))
                return
            episode = execution.store.episode(turn.episode.episode_id)
            assert episode is not None
            if episode.wrapup_state == "failed":
                yield _terminal_report_error(episode)
                return
            attempt = None

        # Finish every mechanical preflight before reserving a new provider-call attempt. A
        # queued attempt may already exist after a crash between its durable reservation and call.
        receipt_path = _stage_and_verify_receipt(turn.wrapup, stage)
        report_skill_path = _stage_and_verify_report_skill(turn.wrapup, stage)
        inputs_path = _inputs_path(stage)

        while True:
            attempt_number = (
                attempt.attempt_number
                if attempt is not None
                else _next_attempt_number(execution, turn)
            )
            diagnostic = _prior_attempt_diagnostic(execution, turn, attempt_number)
            diagnostic_path = (
                _stage_correction_diagnostic(turn, attempt_number, diagnostic, stage)
                if diagnostic is not None
                else None
            )
            contract = episode_report_task_contract(
                project_name=service.manifest.name,
                ending=cast(str, turn.wrapup.ending),
                partial=turn.wrapup.partial,
                receipt_path=receipt_path,
                receipt_sha256=turn.wrapup.receipt_sha256,
                report_skill_path=report_skill_path,
                report_output_path=report_output_path,
                correction_diagnostic_path=diagnostic_path,
            )
            contract_path, prompt = _stage_attempt_contract(
                execution,
                attempt_number,
                stage,
                contract,
            )

            # A retained output from an older operational turn or failed report attempt must never
            # be accepted as this provider call's deliverable.
            mailbox.remove(output_name)
            if stage.remote is not None:
                # Finish the entire report-only input batch while no attempt is active. Transport
                # or exact-stage loss here is an unlaunchable wrap-up, not a provider call.
                stage.remote.finalize_inputs()

            if attempt is None:
                attempt = execution.store.allocate_episode_report_attempt(turn.episode.episode_id)
                if attempt.attempt_number != attempt_number:
                    raise RuntimeError("Episode report attempt allocation changed under launch.")
            execution.store.mark_episode_report_attempt_running(attempt.attempt_id)
            _record_agent_launch_receipt(
                execution,
                cast(RunRequest, request),
                prompt=prompt,
                contract_path=contract_path,
                remote=bool(stage.execution_host),
                resumed=True,
                write_scope=write_scope,
                continuation="episode_report",
                extra={
                    "surface": "episode_report",
                    "role": "report",
                    "episode_id": turn.episode.episode_id,
                    "allocation_operation_id": turn.task.operation_id,
                    "attempt_id": attempt.attempt_id,
                    "attempt_number": attempt.attempt_number,
                    "capability": capability,
                    "graph_authority": "none",
                    "dispatch_authority": "none",
                    "report_output_path": report_output_path,
                    "report_skill_id": _REPORT_SKILL_ID,
                    "report_skill_version": turn.wrapup.skill_version,
                },
            )

            outcome = _ProviderOutcome(session_id=turn.wrapup.native_session_id)
            provider_errors: list[str] = []
            try:
                async with aclosing(
                    _stream_agent_events(
                        launcher,
                        cast(RunRequest, request),
                        prompt,
                        workspace=stage.workspace,
                        session_id=turn.wrapup.native_session_id,
                        read_dirs=[inputs_path],
                        write_dirs=[],
                        write_scope=write_scope,
                        execution_host=stage.execution_host,
                        execution=execution,
                        remote_stage=stage.remote,
                        capability=capability,
                        outcome=outcome,
                        binary=stage.provider_binary,
                    )
                ) as stream:
                    async for frame in stream:
                        event = _event_from_frame(frame)
                        if event.event == "error" and event.text.strip():
                            provider_errors.append(" ".join(event.text.split())[:1000])
                        if event.usage is not None:
                            # Usage is still real accounting even though intermediate report
                            # failures and progress stay off the user-visible task stream.
                            yield _sse(AgentEvent(event="raw", usage=event.usage))
            except (OSError, RuntimeError, StateUnavailable, ValueError) as exc:
                outcome.failed = True
                provider_errors.append(" ".join(str(exc).split())[:1000])

            if outcome.session_id != turn.wrapup.native_session_id:
                diagnostic = (
                    "Episode report continuation changed the frozen native provider session."
                )
                execution.store.finish_episode_report_error(attempt.attempt_id, diagnostic)
                yield _sse(AgentEvent(event="error", text=diagnostic))
                return

            if outcome.paused:
                diagnostic = "Episode report provider call was interrupted before completion."
                episode, _failed = execution.store.record_episode_report_attempt_error(
                    attempt.attempt_id,
                    diagnostic,
                )
                if episode.wrapup_state == "failed":
                    yield _terminal_report_error(episode)
                    return
                if execution.control.pause_requested.is_set():
                    yield _sse(
                        AgentEvent(
                            event="paused",
                            text=(
                                "Episode report generation was interrupted and will be "
                                "reconciled automatically."
                            ),
                        )
                    )
                    return
                _record_retryable_error(execution, attempt, diagnostic)
                attempt = None
                continue

            if outcome.failed or not outcome.completed:
                diagnostic = (
                    provider_errors[-1]
                    if provider_errors
                    else f"{request.provider} produced no completed episode report result."
                )
                episode, _failed = execution.store.record_episode_report_attempt_error(
                    attempt.attempt_id,
                    diagnostic,
                )
                if episode.wrapup_state == "failed":
                    yield _terminal_report_error(episode)
                    return
                _record_retryable_error(execution, attempt, diagnostic)
                attempt = None
                continue

            try:
                html = _read_valid_report(mailbox, output_name)
            except (FileNotFoundError, ValueError) as exc:
                diagnostic = _report_validation_diagnostic(exc)
                episode, _failed = execution.store.record_episode_report_attempt_error(
                    attempt.attempt_id,
                    diagnostic,
                )
                if episode.wrapup_state == "failed":
                    yield _terminal_report_error(episode)
                    return
                _record_retryable_error(execution, attempt, diagnostic)
                attempt = None
                continue

            report = _finish_report(execution, turn, attempt, html)
            yield _report_ready_frame(report)
            yield _sse(AgentEvent(event="done"))
            return
    except (KeyError, OSError, StateUnavailable, ValueError) as exc:
        diagnostic = " ".join(str(exc).split()) or "Episode report generation failed."
        _settle_unlaunchable_existing_wrapup(execution, request.episode_id, diagnostic)
        yield _sse(AgentEvent(event="error", text=diagnostic))


def _canonical_report_turn(
    execution: AgentTaskExecution,
    supplied: EpisodeReportRunRequest,
) -> _CanonicalReportTurn:
    task = execution.store.agent_task(execution.operation_id)
    if task is None or task.kind != "episode_report" or task.visible:
        raise ValueError("Episode report execution requires its one durable hidden task.")
    durable = EpisodeReportRunRequest.model_validate(task.request)
    if durable != supplied:
        raise ValueError("Episode report launch request differs from its durable hidden task.")
    if task.episode_id != durable.episode_id:
        raise ValueError("The hidden report task changed its episode lineage.")
    if task.dispatch_authority is not None:
        raise ValueError("Episode report generation cannot carry dispatch or graph authority.")

    episode = execution.store.episode(durable.episode_id)
    wrapup = execution.store.episode_wrapup(durable.episode_id)
    if episode is None or wrapup is None:
        raise ValueError("Episode report execution lost its durable episode wrap-up.")
    if episode.project_id != task.project_id or wrapup.episode_id != episode.episode_id:
        raise ValueError("Episode report execution crossed its project or episode boundary.")
    if episode.ending is None or wrapup.ending != episode.ending or episode.ending == "stopped":
        raise ValueError("Episode report execution does not match a reportable semantic ending.")
    if episode.wrapup_state not in {"pending", "running", "ready", "failed"}:
        raise ValueError("Episode report execution is outside its durable wrap-up lifecycle.")
    if wrapup.state != episode.wrapup_state:
        raise ValueError("Episode and wrap-up report states disagree.")
    if wrapup.allocation_operation_id != task.operation_id:
        raise ValueError("Episode report execution changed its hidden allocation.")
    if (
        wrapup.concluding_operation_id is None
        or task.parent_operation_id != wrapup.concluding_operation_id
    ):
        raise ValueError("Episode report execution changed its concluding task lineage.")
    frozen = {
        "provider": wrapup.provider,
        "run_on": wrapup.run_on,
        "execution_host": wrapup.execution_host,
        "session_id": wrapup.native_session_id,
    }
    changed = [field for field, value in frozen.items() if getattr(durable, field) != value]
    if changed:
        raise ValueError("Episode report execution changed its frozen " + ", ".join(changed) + ".")
    if (
        task.native_session_id != wrapup.native_session_id
        or (task.stage_host or "") != (wrapup.stage_host or "")
        or task.stage_root != wrapup.stage_root
        or (execution.stage_host or "") != (wrapup.stage_host or "")
        or execution.stage_root != wrapup.stage_root
    ):
        raise ValueError("Episode report execution changed its frozen native session or stage.")
    if not wrapup.native_session_id or not wrapup.stage_root:
        raise ValueError("Episode report execution has no complete native session and stage.")
    return _CanonicalReportTurn(task=task, request=durable, episode=episode, wrapup=wrapup)


def _open_exact_report_stage(
    service: ProjectService,
    turn: _CanonicalReportTurn,
) -> _ReportStage:
    machine = service.manifest.machine_map.get(turn.request.run_on)
    if machine is None:
        raise ValueError(f"Unknown episode report execution machine: {turn.request.run_on}")
    if machine.host != turn.request.execution_host:
        raise ValueError("Episode report execution host differs from its frozen machine.")
    stage_root = cast(str, turn.wrapup.stage_root)
    if machine.host:
        if turn.wrapup.stage_host != machine.host:
            raise ValueError("Episode report remote stage host differs from its frozen machine.")
        remote = RemoteRunStage(machine.host).attach(stage_root)
        return _ReportStage(
            local=None,
            remote=remote,
            workspace=Path(str(remote.workspace)),
            execution_host=machine.host,
            provider_binary=machine.provider_paths.get(turn.request.provider),
        )

    local = Path(stage_root)
    if turn.wrapup.stage_host not in {None, ""} or not local.is_absolute():
        raise ValueError("Episode report local stage differs from its frozen machine.")
    if local.is_symlink() or not local.is_dir():
        raise ValueError("Episode report saved local stage is unavailable or unsafe.")
    return _ReportStage(
        local=local,
        remote=None,
        workspace=local,
        execution_host="",
        provider_binary=machine.provider_paths.get(turn.request.provider),
    )


def _stage_and_verify_receipt(wrapup: EpisodeWrapupRecord, stage: _ReportStage) -> str:
    digest = hashlib.sha256(wrapup.receipt_json.encode("utf-8")).hexdigest()
    if digest != wrapup.receipt_sha256:
        raise ValueError("Episode report receipt differs from its frozen digest.")
    label = f"episode-report-receipt-{wrapup.receipt_sha256[:20]}.json"
    path = _stage_or_reuse_task_input(stage.local, stage.remote, label, wrapup.receipt_json)
    if stage.remote is not None:
        # A new remote immutable input is queued locally until its batch is committed. Commit the
        # receipt before reading it back so verification never mistakes pending local bytes for
        # execution-host state. Later skill/contract inputs form their own finalized batch.
        stage.remote.finalize_inputs()
    staged = (
        stage.remote.read_input_text(label)
        if stage.remote is not None
        else Path(path).read_text(encoding="utf-8")
    )
    if (
        staged != wrapup.receipt_json
        or hashlib.sha256(staged.encode("utf-8")).hexdigest() != wrapup.receipt_sha256
    ):
        raise ValueError("Staged episode report receipt differs from its frozen bytes.")
    return path


def _stage_and_verify_report_skill(wrapup: EpisodeWrapupRecord, stage: _ReportStage) -> str:
    reference = official_registry().package("skill", _REPORT_SKILL_ID).reference()
    if wrapup.skill_id != reference.id or wrapup.skill_version != reference.version:
        raise ValueError("Episode wrap-up does not pin the current official episode-report skill.")
    selection = SkillSelection(
        workflow_ids=[],
        skill_ids=[_REPORT_SKILL_ID],
        resolved_skill_packages=[reference],
    )
    pointers = stage_skill_selection(
        selection,
        local_stage=stage.local,
        remote_stage=stage.remote,
        label=skill_bundle_label(selection),
        reuse_existing=True,
    )
    if (
        len(pointers) != 1
        or pointers[0].get("kind") != "skill"
        or pointers[0].get("id") != _REPORT_SKILL_ID
        or pointers[0].get("version") != reference.version
    ):
        raise ValueError("Episode report staging did not resolve its one official skill.")
    return str(PurePosixPath(str(pointers[0]["path"])) / "SKILL.md")


def _exact_report_output(
    wrapup: EpisodeWrapupRecord,
    stage: _ReportStage,
) -> tuple[str, str]:
    output_name = wrapup.output_name or _REPORT_OUTPUT_DEFAULT
    output_path = wrapup.output_path
    if output_path is None:
        raise ValueError("Episode report wrap-up has no frozen output path.")
    expected = (
        PurePosixPath(str(stage.remote.workspace)) / output_name
        if stage.remote is not None
        else Path(stage.workspace) / output_name
    )
    if stage.remote is not None:
        matches = PurePosixPath(output_path) == expected
    else:
        matches = Path(output_path).absolute() == cast(Path, expected).absolute()
    if not matches:
        raise ValueError("Episode report output path is outside its exact saved workspace.")
    return output_name, output_path


def _inputs_path(stage: _ReportStage) -> Path:
    if stage.remote is not None:
        assert stage.remote.root is not None
        return Path(str(stage.remote.root / "inputs"))
    assert stage.local is not None
    return stage.local / "inputs"


def _report_capability(
    execution: AgentTaskExecution,
    wrapup: EpisodeWrapupRecord,
) -> AgentCapability:
    profile: Literal["ordinary", "orchestrator"] = "ordinary"
    if wrapup.concluding_operation_id is not None:
        profile = execution.store.agent_task_profile(wrapup.concluding_operation_id)
    return "orchestrate" if profile == "orchestrator" else "work_auto"


def _prior_attempt_diagnostic(
    execution: AgentTaskExecution,
    turn: _CanonicalReportTurn,
    attempt_number: int,
) -> str | None:
    if attempt_number <= 1:
        return None
    attempts = execution.store.episode_report_attempts(turn.episode.episode_id)
    prior = next(
        (item for item in reversed(attempts) if item.attempt_number < attempt_number),
        None,
    )
    if prior is None or prior.status != "failed" or not prior.error:
        raise ValueError("Episode report retry lost its preceding correction diagnostic.")
    return prior.error


def _next_attempt_number(
    execution: AgentTaskExecution,
    turn: _CanonicalReportTurn,
) -> int:
    attempts = execution.store.episode_report_attempts(turn.episode.episode_id)
    if any(item.status in {"queued", "running", "succeeded"} for item in attempts):
        raise ValueError("Episode report attempt ledger has an unexpected active result.")
    return (attempts[-1].attempt_number if attempts else 0) + 1


def _stage_correction_diagnostic(
    turn: _CanonicalReportTurn,
    attempt_number: int,
    diagnostic: str,
    stage: _ReportStage,
) -> str:
    content = json.dumps(
        {
            "attempt_number": attempt_number - 1,
            "diagnostic": diagnostic,
            "episode_id": turn.episode.episode_id,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return _stage_or_reuse_task_input(
        stage.local,
        stage.remote,
        f"episode-report-diagnostic-{attempt_number}-{digest[:20]}.json",
        content,
    )


def _stage_attempt_contract(
    execution: AgentTaskExecution,
    attempt_number: int,
    stage: _ReportStage,
    contract: str,
) -> tuple[str, str]:
    role = f"episode_report_attempt_{attempt_number}"
    digest = hashlib.sha256(contract.encode("utf-8")).hexdigest()
    execution.store.record_agent_task_contract(
        execution.operation_id,
        role,
        contract,
        digest,
    )
    path = _stage_or_reuse_task_input(
        stage.local,
        stage.remote,
        f"task-{_task_token(execution)}-episode-report-{attempt_number}.md",
        contract,
    )
    return path, PromptFactory.launch_prompt(path)


def _reconcile_running_attempt(
    execution: AgentTaskExecution,
    turn: _CanonicalReportTurn,
    attempt: EpisodeReportAttemptRecord,
    mailbox: RunStageMailbox,
    output_name: str,
) -> EpisodeReportRecord | None:
    try:
        html = _read_valid_report(mailbox, output_name)
    except (FileNotFoundError, ValueError) as exc:
        diagnostic = (
            "A previously started episode report call was interrupted without a valid retained "
            f"report. {_report_validation_diagnostic(exc)}"
        )
        execution.store.record_episode_report_attempt_error(attempt.attempt_id, diagnostic)
        _record_retryable_error(execution, attempt, diagnostic)
        return None
    return _finish_report(execution, turn, attempt, html)


def _read_valid_report(mailbox: RunStageMailbox, output_name: str) -> str:
    text = mailbox.read_text(output_name, max_bytes=CHAT_ARTIFACT_MAX_FILE_BYTES)
    data = text.encode("utf-8")
    if validate_artifact_bytes(output_name, data) != "text/html":
        raise ValueError("episode report output is not HTML")
    if not text.strip():
        raise ValueError("episode report output is empty")
    return text


def _finish_report(
    execution: AgentTaskExecution,
    turn: _CanonicalReportTurn,
    attempt: EpisodeReportAttemptRecord,
    html: str,
) -> EpisodeReportRecord:
    ending = turn.wrapup.ending
    assert ending is not None
    report = EpisodeReportRecord(
        report_id=str(uuid.uuid4()),
        episode_id=turn.episode.episode_id,
        attempt_id=attempt.attempt_id,
        allocation_operation_id=turn.task.operation_id,
        ending=ending,
        sha256=hashlib.sha256(html.encode("utf-8")).hexdigest(),
        html=html,
        created_at=execution.store.now(),
    )
    _episode, stored = execution.store.finish_episode_report_ready(attempt.attempt_id, report)
    return stored


def _record_retryable_error(
    execution: AgentTaskExecution,
    attempt: EpisodeReportAttemptRecord,
    diagnostic: str,
) -> None:
    execution.store.record_agent_task_event(
        execution.operation_id,
        f"Episode report attempt {attempt.attempt_number} failed: {diagnostic}",
        level="warning",
    )


def _settle_unlaunchable_existing_wrapup(
    execution: AgentTaskExecution,
    supplied_episode_id: str,
    diagnostic: str,
) -> None:
    """Close an allocated but mechanically unusable fence before any provider call."""

    with suppress(KeyError, RuntimeError, ValueError):
        task = execution.store.agent_task(execution.operation_id)
        episode_id = (
            task.episode_id if task is not None and task.episode_id else supplied_episode_id
        )
        execution.store.fail_episode_report_allocation_unlaunchable(episode_id, diagnostic)


def _report_validation_diagnostic(exc: BaseException) -> str:
    if isinstance(exc, FileNotFoundError):
        return "Episode report output is missing."
    detail = " ".join(str(exc).split())
    return f"Episode report output is invalid: {detail}."


def _event_from_frame(frame: str) -> AgentEvent:
    if not frame.startswith("data: "):
        raise ValueError("Episode report provider stream returned a malformed frame.")
    return AgentEvent.model_validate_json(frame.removeprefix("data: ").strip())


def _report_ready_frame(report: EpisodeReportRecord) -> str:
    return _sse(
        AgentEvent(
            event="message",
            text=json.dumps(
                {
                    "episode_report": {
                        "ending": report.ending,
                        "episode_id": report.episode_id,
                        "report_id": report.report_id,
                        "sha256": report.sha256,
                    }
                },
                separators=(",", ":"),
            ),
        )
    )


def _terminal_report_error(episode: EpisodeRecord) -> str:
    diagnostic = episode.wrapup_error or "Episode report generation failed."
    return _sse(
        AgentEvent(
            event="error",
            text=(
                f"Episode report generation failed after {episode.report_attempts_used} "
                f"attempts: {diagnostic}"
            ),
        )
    )
