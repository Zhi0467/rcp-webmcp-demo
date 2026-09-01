from __future__ import annotations

import asyncio
import hashlib
from contextlib import aclosing
from pathlib import Path, PurePosixPath

from rcp.agents import AgentEvent, AgentLauncher
from rcp.agents.experiment_loop_prompt import experiment_watcher_maintenance_correction_contract
from rcp.agents.write_scope import ProjectWriteScope
from rcp.background import AgentTaskExecution
from rcp.history import ReplayHalted
from rcp.limits import PATCH_CORRECTION_MAX_ROUNDS
from rcp.runs.experiment_loop import (
    StagedExperimentWatcherResource,
    experiment_watcher_output_name,
    persist_experiment_watchers_idempotently,
    read_experiment_watcher_outputs,
)
from rcp.runs.shared import (
    _ProviderOutcome,
    _record_agent_launch_receipt,
    _retry_deliverable_is_unchanged,
    _stage_json_task_input,
    _stage_task_contract,
    _stream_agent_events,
)
from rcp.service import ProjectService, RunRequest
from rcp.transport import RemoteRunStage, StateUnavailable
from rcp.watchers import (
    WatcherBinding,
    WatcherInitialCheckError,
    parse_experiment_watch_json,
    validate_graph_conditions,
    validate_watch_specs,
)


def _experiment_maintenance_binding(
    execution: AgentTaskExecution,
    staged: StagedExperimentWatcherResource,
) -> WatcherBinding:
    """Bind maintenance authority from the durable actor and staged node resource."""

    task = execution.store.agent_task(execution.operation_id)
    if task is None:
        raise ValueError("Experiment watcher maintenance actor is no longer available.")
    if task.kind not in {"node_chat", "project_chat"}:
        raise ValueError("Experiment watcher maintenance requires a durable chat actor.")
    chat_id = task.request.get("chat_id")
    if not isinstance(chat_id, str) or not chat_id:
        raise ValueError("Experiment watcher maintenance actor has no durable conversation.")
    resource = staged.resource
    return WatcherBinding(
        project_id=task.project_id,
        origin_operation_id=execution.operation_id,
        origin_task_kind=task.kind,
        chat_id=chat_id,
        node_id=resource.control_node_id,
        episode_id=resource.continuation.control_episode_id,
        graph_target=task.graph_target,
        execution_host=resource.execution_host,
        continuation=resource.continuation,
    )


async def _process_experiment_watcher_maintenance(
    *,
    service: ProjectService,
    launcher: AgentLauncher,
    request: RunRequest,
    execution: AgentTaskExecution | None,
    staged_resources: list[StagedExperimentWatcherResource],
    workspace: Path,
    remote_stage: RemoteRunStage | None,
    local_stage: Path | None,
    base_contract_path: str,
    token: str,
    native_session_id: str | None,
    read_dirs: list[Path | PurePosixPath],
    write_dirs: list[Path | PurePosixPath],
    write_scope: ProjectWriteScope,
    execution_host: str,
    provider_binary: str | None,
    retry_output_digests: dict[str, str],
) -> tuple[list[str], str | None, bool]:
    """Admit, validate, and atomically persist each physical Experiment watcher file."""

    if execution is None:
        return [], native_session_id, False
    frames: list[str] = []
    staged_by_name = {
        experiment_watcher_output_name(
            item.resource.control_node_id,
            item.resource.graph_target,
        ): item
        for item in staged_resources
    }
    try:
        outputs = read_experiment_watcher_outputs(workspace, remote_stage)
    except (OSError, StateUnavailable, ValueError) as exc:
        execution.store.record_agent_task_event(
            execution.operation_id,
            f"Experiment watcher maintenance output could not be inspected: {exc}",
            level="warning",
        )
        return frames, native_session_id, False

    for name, initial_text in sorted(outputs.items()):
        # A Retry reuses the conversation's folder without clearing it, so a
        # previous attempt's maintenance file is still sitting there. Applying it
        # would commit that attempt's handoff under this attempt's authorization
        # (invariant 10c), so an unchanged survivor counts as nothing written.
        if _retry_deliverable_is_unchanged(
            execution,
            filename=name,
            predecessor_digest=retry_output_digests.get(name),
            current_text=initial_text,
        ):
            continue
        staged = staged_by_name.get(name)
        if staged is None:
            problem = (
                "Experiment watcher maintenance permission denied: the physical output path was "
                "not staged for this actor's resolved resource scope."
            )
            execution.store.record_agent_task_receipt(
                execution.operation_id,
                "experiment_watcher_maintenance_rejected",
                {"path": str(workspace / name), "problem": problem},
                tier="diagnostic",
            )
            execution.store.record_agent_task_event(
                execution.operation_id,
                problem,
                level="warning",
            )
            continue

        text = initial_text
        correction_round = 0
        target_digest = hashlib.sha256(staged.resource.control_node_id.encode("utf-8")).hexdigest()[
            :16
        ]

        def reject_maintenance(
            problem: str,
            target: StagedExperimentWatcherResource,
            rounds: int,
        ) -> None:
            execution.store.record_agent_task_receipt(
                execution.operation_id,
                "experiment_watcher_maintenance_rejected",
                {
                    "control_node_id": target.resource.control_node_id,
                    "episode_id": target.resource.episode_id,
                    "path": target.watch_path,
                    "problem": problem[:1600],
                    "correction_rounds": rounds,
                },
                tier="diagnostic",
            )
            execution.store.record_agent_task_event(
                execution.operation_id,
                f"Experiment watcher maintenance was not applied: {problem}",
                level="warning",
            )

        while True:
            binding = _experiment_maintenance_binding(execution, staged)
            try:
                execution.store.admit_experiment_watcher_maintenance(binding)
            except ValueError as exc:
                problem = str(exc)
                correctable = False
            else:
                try:
                    handoff = parse_experiment_watch_json(text)
                    graph_state = (
                        await asyncio.to_thread(service.history.state)
                        if handoff.graph_conditions
                        else None
                    )
                    graph_armed_revision = graph_state.revision if graph_state is not None else None
                    if graph_state is not None:
                        await asyncio.to_thread(
                            validate_graph_conditions,
                            handoff.graph_conditions,
                            graph_state,
                        )
                    check_results = (
                        await asyncio.to_thread(
                            validate_watch_specs,
                            handoff.observers,
                            staged.resource.execution_host,
                        )
                        if handoff.observers
                        else []
                    )
                except (WatcherInitialCheckError, ValueError) as exc:
                    problem = str(exc)
                    correctable = True
                except (OSError, ReplayHalted, StateUnavailable) as exc:
                    problem = str(exc)
                    correctable = False
                else:
                    try:
                        fresh_graph_state = (
                            await asyncio.to_thread(service.history.state)
                            if handoff.graph_conditions
                            else None
                        )
                        if handoff.graph_conditions:
                            # Mark the settlement before the insert attempt. If
                            # cancellation lands after SQLite commits but before
                            # this await resumes, ordered reconciliation must
                            # still catch canonical movement after validation.
                            execution.armed_graph_watchers = True
                        armed = await asyncio.to_thread(
                            persist_experiment_watchers_idempotently,
                            execution,
                            handoff.observers,
                            check_results,
                            binding,
                            handoff.stops,
                            graph_conditions=handoff.graph_conditions,
                            graph_state=fresh_graph_state,
                            armed_revision=graph_armed_revision,
                            expected_watcher_snapshot_token=(
                                staged.resource.watcher_snapshot_token
                            ),
                        )
                    except (OSError, ReplayHalted, StateUnavailable, ValueError) as exc:
                        problem = str(exc)
                        correctable = False
                    else:
                        execution.store.record_agent_task_receipt(
                            execution.operation_id,
                            "experiment_watchers_maintained",
                            {
                                "control_node_id": staged.resource.control_node_id,
                                "episode_id": staged.resource.episode_id,
                                "watcher_ids": [item.watcher_id for item in armed],
                                "stopped_watcher_ids": [
                                    item.stop_watcher_id for item in handoff.stops
                                ],
                                "correction_rounds": correction_round,
                            },
                        )
                        break

            if (
                not correctable
                or correction_round >= PATCH_CORRECTION_MAX_ROUNDS
                or not native_session_id
            ):
                reject_maintenance(problem, staged, correction_round)
                break

            correction_round += 1
            diagnostics_path = _stage_json_task_input(
                local_stage,
                remote_stage,
                f"task-{token}-experiment-watch-correction-{target_digest}-{correction_round}.json",
                {
                    "control_node_id": staged.resource.control_node_id,
                    "problem": problem,
                },
            )
            correction_contract = experiment_watcher_maintenance_correction_contract(
                original_contract_path=base_contract_path,
                diagnostics_path=diagnostics_path,
                watch_path=staged.watch_path,
            )
            correction_path, correction_prompt = _stage_task_contract(
                local_stage,
                remote_stage,
                f"task-{token}-experiment-watch-correction-{target_digest}-{correction_round}.md",
                correction_contract,
                execution=execution,
                role=f"experiment_watch_correction_{target_digest}_{correction_round}",
            )
            before_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            _record_agent_launch_receipt(
                execution,
                request,
                prompt=correction_prompt,
                contract_path=correction_path,
                remote=bool(execution_host),
                resumed=True,
                write_scope=write_scope,
                continuation="watch_correction",
                extra={
                    "surface": binding.origin_task_kind,
                    "mode": "work",
                    "capability": "work_auto",
                    "network_access": True,
                    "launch_kind": "experiment_watch_correction",
                    "correction_round": correction_round,
                    "control_node_id": staged.resource.control_node_id,
                    "write_directory_count": len(write_dirs),
                    "canonical_state_boundary": "prompt_only",
                },
            )
            correction_outcome = _ProviderOutcome(session_id=native_session_id)
            correction_error: str | None = None
            async with aclosing(
                _stream_agent_events(
                    launcher,
                    request,
                    correction_prompt,
                    workspace=workspace,
                    session_id=native_session_id,
                    read_dirs=read_dirs,
                    write_dirs=write_dirs,
                    write_scope=write_scope,
                    execution_host=execution_host,
                    execution=execution,
                    remote_stage=remote_stage,
                    capability="work_auto",
                    outcome=correction_outcome,
                    binary=provider_binary,
                )
            ) as stream:
                async for frame in stream:
                    event = AgentEvent.model_validate_json(frame.removeprefix("data: ").strip())
                    if event.event == "error":
                        correction_error = event.text or "Watcher maintenance correction failed."
                    elif event.event not in {"answer", "done"}:
                        frames.append(frame)
            native_session_id = correction_outcome.session_id or native_session_id
            if correction_outcome.paused:
                return frames, native_session_id, True
            if correction_error or not correction_outcome.completed:
                problem = correction_error or (
                    f"{request.provider} produced no watcher maintenance correction result."
                )
                reject_maintenance(problem, staged, correction_round)
                break
            try:
                corrected_outputs = read_experiment_watcher_outputs(workspace, remote_stage)
            except (OSError, StateUnavailable, ValueError) as exc:
                problem = f"The corrected watcher maintenance output could not be read: {exc}"
                reject_maintenance(problem, staged, correction_round)
                break
            corrected = corrected_outputs.get(name)
            if corrected is None:
                problem = "The correction completed without rewriting the Experiment watcher file."
                reject_maintenance(problem, staged, correction_round)
                break
            if hashlib.sha256(corrected.encode("utf-8")).hexdigest() == before_digest:
                problem = (
                    f"{problem} The correction left the Experiment watcher file byte-identical."
                )
                reject_maintenance(problem, staged, correction_round)
                break
            text = corrected

    return frames, native_session_id, False
