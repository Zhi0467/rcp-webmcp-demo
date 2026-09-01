from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from collections.abc import AsyncIterator
from contextlib import aclosing, suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from rcp.agents import (
    AgentEvent,
    AgentLauncher,
    PromptFactory,
    parse_agent_patch_json,
    prepare_agent_patch,
    validate_agent_patch_shape,
    validate_work_patch,
)
from rcp.agents.command_mailbox import (
    StagedCommandMailbox,
)
from rcp.agents.experiment_loop_prompt import (
    experiment_loop_continuation_contract,
    experiment_loop_patch_correction_contract,
    experiment_loop_task_contract,
    experiment_loop_wake_message,
    experiment_loop_watcher_correction_contract,
)
from rcp.agents.prompts import invoked_package_pointers, invoked_provider_skill_section
from rcp.attachments import ChatAttachmentStore
from rcp.background import AgentTaskExecution
from rcp.config import AgentSurface
from rcp.core.authority import AgentProfile
from rcp.core.models import ExperimentDecisionPin, Patch
from rcp.history import PatchRejected, ReplayHalted
from rcp.limits import (
    PATCH_CORRECTION_MAX_ROUNDS,
    PATCH_SELF_CHECK_TIMEOUT_SECONDS,
)
from rcp.runs.chat import (
    _append_chat_graph_receipt,
    _chat_context_delta,
    _chat_read_dirs,
    _chat_stage_name,
    _ChatPatchInputs,
    _clear_stale_turn_handoffs,
    _discover_chat_artifacts,
    _logical_chat_turn_operation_id,
    _prepare_local_artifact_directory,
    _project_write_scope,
    _read_chat_patch,
    _read_watch_request,
    _record_artifact_discovery_receipt,
    _record_chat_context_receipt,
    _stage_chat_patch_inputs,
    _validated_local_chat_resume_stage,
    _validated_remote_chat_resume_stage,
)
from rcp.runs.experiment_loop import (
    commit_experiment_episode_binding,
    commit_experiment_episode_handoff,
    experiment_episode_context_values,
    experiment_exit_problem,
    experiment_graph_result_summary,
    experiment_loop_ending_signal,
    experiment_loop_semantic_ending,
    prepare_experiment_episode_context_candidate,
    prepare_experiment_watcher_records,
    root_experiment_loop_operation_id,
    stage_experiment_loop_context,
    validate_experiment_completion,
)
from rcp.runs.patch_validator import (
    PatchValidationBudget,
    PatchValidationResult,
    serve_patch_validation_mailbox,
    stage_patch_validation_mailbox,
)
from rcp.runs.shared import (
    _existing_patch_digest,
    _parent_task_contract_path,
    _pinned_to_profile,
    _ProviderOutcome,
    _record_agent_launch_receipt,
    _record_patch_applied_receipt,
    _record_patch_receipt,
    _retry_deliverable_is_unchanged,
    _sse,
    _stage_context_paths,
    _stage_json_task_input,
    _stage_task_contract,
    _stage_task_input,
    _swept_stage_root,
    _task_token,
)
from rcp.runs.tasks.work import (
    WorkTurn,
    _AppliedWorkTurn,
    _bounded_graph_messages,
    _capture_retry_deliverable_baseline,
    _close_work_validator_mailbox,
    _ComposedWorkPrompt,
    _DeliverableFailure,
    _DeliverableRead,
    _DeliverableStep,
    _finalize_work_turn,
    _read_correction_patch,
    _record_work_graph_rejection,
    _record_work_lock_lost,
    _record_work_lock_wait,
    _rejected_graph_update_for_repair,
    _resolve_work_execution,
    _ResolvedWorkExecution,
    _SettledWorkDeliverables,
    _stage_retry_diagnostics,
    _StagedWorkInputs,
    _stream_turn_agent_events,
    _work_graph_repairable,
    _work_patch_proposal_ids,
    _WorkValidatorMailboxLifecycle,
)
from rcp.service import GraphUpdateResult, ProjectService, RunRequest
from rcp.skills.staging import skill_bundle_label, stage_skill_selection
from rcp.storage import EpisodeRecord, WatcherContinuation
from rcp.transport import RemoteRunStage, RunLockCancelled, StateUnavailable
from rcp.watchers import (
    WatcherBinding,
    WatcherInitialCheckError,
    parse_experiment_watch_json,
    validate_graph_conditions,
    validate_watch_specs,
)


@dataclass(frozen=True)
class _PreparedWorkPatch:
    patch: Patch
    change_summary: tuple[str, ...]
    proposal_ids: tuple[str, ...]


@dataclass(frozen=True)
class _WorkPromptContext:
    episode_context_baseline: dict[str, object] | None
    experiment_control_snapshot: dict[str, object] | None
    wake_episode: EpisodeRecord | None
    context_replacement: dict[str, object] | None
    loop_control_path: str | None
    watcher_state_path: str | None
    provider_switch_recovery: bool


@dataclass
class _SettledExperimentDeliverables(_SettledWorkDeliverables):
    loop_watch_empty: bool = False
    loop_watch_text: str | None = None
    pending_loop_handoff: tuple[object, ...] | None = None


def _work_patch_source_operation_id(
    execution: AgentTaskExecution | None,
) -> str | None:
    if execution is None:
        return None
    if execution.continuation != "graph_repair":
        return root_experiment_loop_operation_id(execution)
    return execution.operation_id


def _experiment_loop_retry_handoff_authorized(
    turn: WorkTurn,
    watch_text: str | None,
) -> bool:
    """Authorize one exact retained loop watcher handoff from an ancestor receipt."""

    execution = turn.execution
    if (
        execution is None
        or execution.continuation != "retry"
        or turn.request.patch_kind != "experiment_loop"
        or watch_text is None
        or not turn.request.control_episode_id
        or turn.request.control_invocation is None
    ):
        return False
    try:
        root_id = root_experiment_loop_operation_id(execution)
        patch_text = _read_chat_patch(turn.workspace, turn.remote_stage)
    except (OSError, StateUnavailable, ValueError):
        return False
    patch_digest = (
        hashlib.sha256(patch_text.encode("utf-8")).hexdigest() if patch_text is not None else None
    )
    watch_digest = hashlib.sha256(watch_text.encode("utf-8")).hexdigest()
    current = execution.store.agent_task(execution.operation_id)
    if current is None:
        return False

    matched = False
    ancestor_id = current.parent_operation_id
    seen = {current.operation_id}
    while ancestor_id is not None:
        if ancestor_id in seen:
            return False
        seen.add(ancestor_id)
        ancestor = execution.store.agent_task(ancestor_id)
        if ancestor is None:
            return False
        if ancestor.request.get("patch_kind") == "experiment_loop":
            for receipt in execution.store.agent_task_receipts(ancestor.operation_id):
                if receipt.category != "experiment_loop_handoff_prepared":
                    continue
                payload = receipt.payload
                if (
                    payload.get("root_operation_id") == root_id
                    and payload.get("episode_id") == turn.request.control_episode_id
                    and payload.get("invocation") == turn.request.control_invocation
                    and payload.get("patch_sha256") == patch_digest
                    and payload.get("watch_sha256") == watch_digest
                ):
                    matched = True
                    break
        ancestor_id = ancestor.parent_operation_id
    return matched


async def _stage_work_turn(
    service: ProjectService,
    resolved: _ResolvedWorkExecution,
    data_dir: Path,
    execution: AgentTaskExecution | None,
) -> tuple[WorkTurn, _StagedWorkInputs]:
    request = resolved.request
    continuation = execution.continuation if execution is not None else "fresh"
    reusing_checkpoint = bool(execution is not None and execution.reuses_native_checkpoint)
    resuming = continuation == "resume"
    waking = continuation == "watcher_wake"
    local_stage: Path | None = None
    remote_stage: RemoteRunStage | None = None
    patch_inputs: _ChatPatchInputs | None = None
    validator_lifecycle: _WorkValidatorMailboxLifecycle | None = None
    validator_budget = PatchValidationBudget()
    outcome = _ProviderOutcome(session_id=request.session_id)
    try:
        context = service.assemble_chat(request)
        surface: AgentSurface = "project_chat" if request.chat_scope == "project" else "node_chat"
        _record_chat_context_receipt(execution, context, surface=surface)
        stage_name = _chat_stage_name(service, request, execution)
        saved_stage = execution is not None and execution.stage_root is not None
        if resolved.execution_host:
            if saved_stage:
                stage_root = _validated_remote_chat_resume_stage(
                    execution,
                    resolved.execution_host,
                    stage_name,
                )
                remote_stage = RemoteRunStage(resolved.execution_host).attach(stage_root)
            else:
                remote_stage = RemoteRunStage(resolved.execution_host).open(
                    stage_name,
                    reuse=True,
                )
            assert remote_stage.root is not None
            if execution is not None:
                execution.checkpoint_stage(resolved.execution_host, str(remote_stage.root))
            context = context.model_copy(
                update=_stage_context_paths(
                    context,
                    service,
                    remote_stage,
                    resolved.execution_machine_alias,
                )
            )
            workspace = Path(str(remote_stage.workspace))
        else:
            stage_root = _swept_stage_root(data_dir)
            expected_stage = stage_root / stage_name
            if saved_stage:
                local_stage = _validated_local_chat_resume_stage(execution, expected_stage)
            else:
                local_stage = expected_stage
                local_stage.mkdir(parents=True, exist_ok=True)
            if execution is not None:
                execution.checkpoint_stage("", str(local_stage))
            workspace = local_stage
        token = _task_token(execution)
        patch_inputs = _stage_chat_patch_inputs(
            local_stage,
            remote_stage,
            workspace=workspace,
            stage_name=stage_name,
            task_id=execution.operation_id if execution is not None else token,
            turn_id=f"{token}:work",
        )
        validator_lifecycle = _start_work_validator_mailbox(
            service,
            patch_inputs.validator_staged,
            execution=execution,
            budget=validator_budget,
            run_truth_scope=context.run_truth_scope,
            control_node_id=request.control_node_id,
            control_decision_bundle=request.control_decision_bundle,
        )
        if not reusing_checkpoint or waking:
            _clear_stale_turn_handoffs(workspace, remote_stage)
        artifact_scope_id = (
            _logical_chat_turn_operation_id(execution.store, execution.operation_id)
            if execution is not None and resuming
            else execution.operation_id
            if execution is not None
            else str(uuid.uuid4())
        )
        if remote_stage is not None:
            artifact_directory: Path | PurePosixPath = remote_stage.prepare_artifact_directory(
                artifact_scope_id,
                reuse=resuming,
            )
        else:
            assert local_stage is not None
            artifact_directory = _prepare_local_artifact_directory(
                local_stage,
                artifact_scope_id,
                reuse=resuming,
            )
        read_dirs = _chat_read_dirs(
            context,
            remote_stage,
            service,
            resolved.execution_machine_alias,
        )
        write_scope = _project_write_scope(
            context,
            service,
            resolved.execution_machine_alias,
            workspace=workspace,
            remote_stage=remote_stage,
            data_dir=data_dir,
            execution=execution,
            capability="work_auto",
        )
        write_dirs = [Path(item) for item in write_scope.repository_roots]
        experiment_resources = []
        experiment_resource_pointers = []
        skill_selection = service.resolve_skill_selection(request)
        skill_pointers = stage_skill_selection(
            skill_selection,
            local_stage=local_stage,
            remote_stage=remote_stage,
            label=skill_bundle_label(skill_selection),
            reuse_existing=True,
        )
        if bool(request.attachment_batch_id) != bool(request.attachments):
            raise ValueError("The chat task has incomplete attachment batch metadata.")
        attachment_pointers = (
            ChatAttachmentStore(data_dir / "chat-attachments").stage(
                request.attachment_batch_id,
                request.attachments,
                local_stage=local_stage,
                remote_stage=remote_stage,
            )
            if request.attachment_batch_id
            else []
        )
        read_dirs.extend(
            path
            for path in dict.fromkeys(
                Path(str(item["path"])).parent for item in attachment_pointers
            )
            if path not in read_dirs
        )
        repositories = [
            {"alias": item.alias, "host": item.host, "path": item.path}
            for item in context.repositories
        ]
        turn = WorkTurn(
            service=service,
            request=request,
            execution=execution,
            context=context,
            workspace=workspace,
            local_stage=local_stage,
            remote_stage=remote_stage,
            execution_host=resolved.execution_host,
            provider_binary=resolved.provider_binary,
            read_dirs=read_dirs,
            write_dirs=write_dirs,
            write_scope=write_scope,
            patch_inputs=patch_inputs,
            validator_lifecycle=validator_lifecycle,
            validator_budget=validator_budget,
            outcome=outcome,
        )
        return turn, _StagedWorkInputs(
            token=token,
            artifact_scope_id=artifact_scope_id,
            artifact_directory=artifact_directory,
            prepared_result_view=None,
            experiment_resources=experiment_resources,
            experiment_resource_pointers=experiment_resource_pointers,
            skill_selection=skill_selection,
            skill_pointers=skill_pointers,
            attachment_pointers=attachment_pointers,
            repositories=repositories,
        )
    except BaseException as exc:
        if validator_lifecycle is not None:
            await validator_lifecycle.close(primary_error=exc)
        elif patch_inputs is not None and not patch_inputs.validator_staged.credential.expired:
            await _close_work_validator_mailbox(
                patch_inputs.validator_staged,
                stop=None,
                task=None,
                execution=execution,
                primary_error=exc,
            )
        raise


async def _prepare_work_prompt_context(
    turn: WorkTurn,
    staged: _StagedWorkInputs,
) -> _WorkPromptContext:
    control_node = turn.context.node
    if (
        control_node is None
        or control_node.get("id") != turn.request.control_node_id
        or control_node.get("type") != "experiment"
    ):
        raise ValueError("Experiment-loop work no longer resolves to its Experiment.")
    experiment_control_snapshot = dict(control_node)
    loop_control_path, watcher_state_path = await stage_experiment_loop_context(
        turn.service,
        turn.request,
        turn.execution,
        turn.local_stage,
        turn.remote_stage,
        token=staged.token,
        continuation=turn.continuation,
    )
    assert turn.execution is not None
    episode = (
        turn.execution.store.experiment_episode(turn.request.control_episode_id)
        if turn.request.control_episode_id
        else None
    )
    provider_switch_recovery = bool(
        turn.continuation == "handoff" and episode is not None and episode.session_bound
    )
    ontology = turn.service.history.state().ontology.model_dump(mode="json")
    episode_context_baseline = prepare_experiment_episode_context_candidate(
        turn.execution,
        experiment_episode_context_values(
            ontology_extensions=turn.context.ontology_extensions,
            ontology=ontology,
            repositories=staged.repositories,
            skill_pointers=staged.skill_pointers,
        ),
    )
    wake_episode: EpisodeRecord | None = None
    context_replacement: dict[str, object] | None = None
    if turn.waking:
        if not turn.request.control_episode_id or turn.request.control_invocation is None:
            raise ValueError("Experiment-loop wake is missing its episode invocation.")
        wake_episode = turn.execution.store.experiment_episode(turn.request.control_episode_id)
        if wake_episode is None or not wake_episode.session_bound:
            raise ValueError("Experiment-loop wake has no committed episode session to continue.")
        if (
            wake_episode.native_session_id != turn.request.session_id
            or wake_episode.stage_host != turn.execution.stage_host
            or wake_episode.stage_root != turn.execution.stage_root
        ):
            raise ValueError(
                "Experiment-loop wake does not match its committed native session and exact stage."
            )
        if wake_episode.last_turn_invocation != turn.request.control_invocation - 1:
            raise ValueError(
                "Experiment-loop wake does not immediately follow the episode's last "
                "successful turn."
            )
        if not wake_episode.last_graph_result:
            raise ValueError("Experiment-loop wake cannot confirm the preceding graph handoff.")
        context_replacement = _chat_context_delta(
            wake_episode.context_baseline,
            episode_context_baseline,
        )
    if turn.reusing_checkpoint and not turn.request.session_id:
        raise ValueError(
            "The continued Work turn has no native agent session; retry it from a clean attempt "
            "instead."
        )
    return _WorkPromptContext(
        episode_context_baseline=episode_context_baseline,
        experiment_control_snapshot=experiment_control_snapshot,
        wake_episode=wake_episode,
        context_replacement=context_replacement,
        loop_control_path=loop_control_path,
        watcher_state_path=watcher_state_path,
        provider_switch_recovery=provider_switch_recovery,
    )


def _compose_resume_prompt(
    turn: WorkTurn,
    staged: _StagedWorkInputs,
    prepared: _WorkPromptContext,
) -> _ComposedWorkPrompt:
    assert turn.execution is not None
    original_contract_path = _parent_task_contract_path(
        turn.execution,
        turn.local_stage,
        turn.remote_stage,
    )
    if not prepared.loop_control_path:
        raise ValueError("Experiment-loop Resume is missing fresh loop control.")
    contract = experiment_loop_continuation_contract(
        original_contract_path=original_contract_path,
        mode="resume",
        loop_control_path=prepared.loop_control_path,
        patch_path=turn.patch_inputs.patch_path,
        watch_path=turn.patch_inputs.watch_path,
        output_schema_path=turn.patch_inputs.schema_path,
        validator_command=turn.patch_inputs.validator_command,
        invoked_skill_pointers=invoked_package_pointers(
            staged.skill_pointers,
            workflow_ids=turn.request.invoked_workflow_ids,
            skill_ids=turn.request.invoked_skill_ids,
        ),
    )
    contract += invoked_provider_skill_section(turn.request.resolved_provider_skills)
    contract_path, prompt = _stage_task_contract(
        turn.local_stage,
        turn.remote_stage,
        f"task-{staged.token}-resume.md",
        contract,
        execution=turn.execution,
        role="work_resume",
    )
    return _ComposedWorkPrompt(
        contract_path=contract_path,
        prompt=prompt,
        base_contract_path=original_contract_path,
    )


def _compose_wake_prompt(
    turn: WorkTurn,
    staged: _StagedWorkInputs,
    prepared: _WorkPromptContext,
) -> _ComposedWorkPrompt:
    if (
        prepared.wake_episode is None
        or not turn.request.control_node_id
        or turn.request.control_invocation is None
        or turn.request.control_invocation_ceiling is None
        or not prepared.loop_control_path
        or not prepared.watcher_state_path
    ):
        raise ValueError("Experiment-loop wake inputs are incomplete after staging.")
    contract = experiment_loop_wake_message(
        focused_experiment_id=turn.request.control_node_id,
        invocation=turn.request.control_invocation,
        invocation_ceiling=turn.request.control_invocation_ceiling,
        previous_graph_result=prepared.wake_episode.last_graph_result or "",
        previous_watcher_ids=prepared.wake_episode.last_watcher_ids,
        delivered_watcher_ids=turn.request.watcher_ids,
        loop_control_path=prepared.loop_control_path,
        watcher_state_path=prepared.watcher_state_path,
        graph_path=turn.context.graph_path,
        research_path=turn.context.research_md_path,
        patch_path=turn.patch_inputs.patch_path,
        watch_path=turn.patch_inputs.watch_path,
        output_schema_path=turn.patch_inputs.schema_path,
        validator_command=turn.patch_inputs.validator_command,
        execution_host=turn.execution_host,
        context_replacement=prepared.context_replacement,
        invoked_skill_pointers=invoked_package_pointers(
            staged.skill_pointers,
            workflow_ids=turn.request.invoked_workflow_ids,
            skill_ids=turn.request.invoked_skill_ids,
        ),
    )
    contract_path, prompt = _stage_task_contract(
        turn.local_stage,
        turn.remote_stage,
        f"task-{staged.token}-watcher-wake.md",
        contract,
        execution=turn.execution,
        role="experiment_loop_wake",
    )
    return _ComposedWorkPrompt(
        contract_path=contract_path,
        prompt=prompt,
        base_contract_path=contract_path,
    )


def _compose_fresh_prompt(
    turn: WorkTurn,
    staged: _StagedWorkInputs,
    prepared: _WorkPromptContext,
    *,
    retry_diagnostics_path: str | None = None,
) -> _ComposedWorkPrompt:
    assert turn.request.message is not None
    if (
        not turn.request.control_node_id
        or not prepared.loop_control_path
        or not prepared.watcher_state_path
    ):
        raise ValueError("Experiment-loop contract inputs are incomplete after staging.")
    human_request_path = _stage_task_input(
        turn.local_stage,
        turn.remote_stage,
        f"task-{staged.token}-human-request.txt",
        turn.request.message,
    )
    contract = experiment_loop_task_contract(
        project_name=turn.context.project_name,
        ontology_path=f"{turn.context.graph_path}#ontology",
        ontology_extensions=turn.context.ontology_extensions,
        graph_path=turn.context.graph_path,
        research_path=turn.context.research_md_path,
        focused_experiment_id=turn.request.control_node_id,
        repositories=staged.repositories,
        introduction_path=turn.context.introduction_path,
        human_request_path=human_request_path,
        loop_control_path=prepared.loop_control_path,
        watcher_state_path=prepared.watcher_state_path,
        patch_path=turn.patch_inputs.patch_path,
        watch_path=turn.patch_inputs.watch_path,
        artifact_path=str(staged.artifact_directory),
        output_schema_path=turn.patch_inputs.schema_path,
        validator_command=turn.patch_inputs.validator_command,
        write_scope=turn.write_scope,
        execution_host=turn.execution_host,
        recovery_diagnostics_path=(
            retry_diagnostics_path if prepared.provider_switch_recovery else None
        ),
        skill_pointers=staged.skill_pointers,
        invoked_skill_pointers=invoked_package_pointers(
            staged.skill_pointers,
            workflow_ids=turn.request.invoked_workflow_ids,
            skill_ids=turn.request.invoked_skill_ids,
        ),
    )
    contract += invoked_provider_skill_section(turn.request.resolved_provider_skills)
    contract_path, prompt = _stage_task_contract(
        turn.local_stage,
        turn.remote_stage,
        f"task-{staged.token}-{'base' if turn.retry_attempt else 'initial'}.md",
        contract,
        execution=turn.execution,
        role="work_retry_base" if turn.retry_attempt else "work",
    )
    return _ComposedWorkPrompt(
        contract_path=contract_path,
        prompt=prompt,
        base_contract_path=contract_path,
    )


def _compose_retry_prompt(
    turn: WorkTurn,
    staged: _StagedWorkInputs,
    prepared: _WorkPromptContext,
) -> _ComposedWorkPrompt:
    assert turn.execution is not None
    retry_diagnostics_path = _stage_retry_diagnostics(turn, staged)
    if not prepared.loop_control_path or not retry_diagnostics_path:
        raise ValueError("Experiment-loop Retry is missing fresh control or diagnostics.")
    original_contract_path = _parent_task_contract_path(
        turn.execution,
        turn.local_stage,
        turn.remote_stage,
    )
    retry_contract = experiment_loop_continuation_contract(
        original_contract_path=original_contract_path,
        mode="retry",
        loop_control_path=prepared.loop_control_path,
        patch_path=turn.patch_inputs.patch_path,
        watch_path=turn.patch_inputs.watch_path,
        output_schema_path=turn.patch_inputs.schema_path,
        validator_command=turn.patch_inputs.validator_command,
        diagnostics_path=retry_diagnostics_path,
        invoked_skill_pointers=invoked_package_pointers(
            staged.skill_pointers,
            workflow_ids=turn.request.invoked_workflow_ids,
            skill_ids=turn.request.invoked_skill_ids,
        ),
    )
    retry_contract += invoked_provider_skill_section(turn.request.resolved_provider_skills)
    contract_path, prompt = _stage_task_contract(
        turn.local_stage,
        turn.remote_stage,
        f"task-{staged.token}-retry.md",
        retry_contract,
        execution=turn.execution,
        role="work_retry",
    )
    return _ComposedWorkPrompt(
        contract_path=contract_path,
        prompt=prompt,
        base_contract_path=original_contract_path,
    )


def _read_initial_patch_deliverable(
    turn: WorkTurn,
    predecessor_digest: str | None,
    settled: _SettledExperimentDeliverables,
) -> _DeliverableRead:
    try:
        text = _read_chat_patch(turn.workspace, turn.remote_stage)
    except (OSError, StateUnavailable, ValueError) as exc:
        text = None
        failure = _DeliverableFailure(
            f"The agent wrote a patch file that could not be read: {exc}",
            correctable=False,
        )
    else:
        failure = None
    if _retry_deliverable_is_unchanged(
        turn.execution,
        filename="patch.json",
        predecessor_digest=predecessor_digest,
        current_text=text,
    ):
        text = None
    if text is not None and turn.execution is not None:
        turn.execution.store.record_agent_task_patch_output(
            turn.execution.operation_id,
            text,
        )
        turn.execution.store.record_agent_task_receipt(
            turn.execution.operation_id,
            "patch_retained",
            {
                "byte_length": len(text.encode("utf-8")),
                "file_name": "patch.json",
            },
            tier="diagnostic",
        )
        text = None
    # Loop graph admission is a joint Patch/watch handoff. Nothing in the
    # pre-handoff path may validate-correct-and-apply it.
    failure = None
    if text is None and failure is None:
        settled.graph_update = GraphUpdateResult(status="none")
    return _DeliverableRead(text=text, failure=failure)


def _read_initial_watch_deliverable(
    turn: WorkTurn,
    predecessor_digest: str | None,
) -> _DeliverableRead:
    try:
        text = _read_watch_request(turn.workspace, turn.remote_stage)
    except (OSError, StateUnavailable, ValueError) as exc:
        text = None
        failure = _DeliverableFailure(
            f"The watcher request could not be read: {exc}",
            correctable=isinstance(exc, ValueError),
        )
    else:
        failure = None
    allow_unchanged = _experiment_loop_retry_handoff_authorized(turn, text)
    if _retry_deliverable_is_unchanged(
        turn.execution,
        filename="watch.json",
        predecessor_digest=predecessor_digest,
        current_text=text,
        allow_unchanged=allow_unchanged,
    ):
        text = None
    if text is None and failure is None:
        failure = _DeliverableFailure(
            "Experiment-loop work must write watch.json as an object with external and graph "
            "lists; leave both lists empty only after confirming nothing remains to watch.",
            correctable=True,
        )
    return _DeliverableRead(text=text, failure=failure)


def _watcher_continuation(
    turn: WorkTurn,
    staged: _StagedWorkInputs,
) -> WatcherContinuation:
    request_values = turn.request.model_dump(mode="json")
    values = {
        name: request_values[name]
        for name in WatcherContinuation.model_fields
        if name in request_values
    }
    values.update(
        provider=turn.request.provider or "",
        run_on=turn.request.run_on or "",
        run_truth_scope=turn.context.run_truth_scope,
        workflow_ids=staged.skill_selection.workflow_ids,
        skill_ids=staged.skill_selection.skill_ids,
        resolved_skill_packages=staged.skill_selection.resolved_skill_packages,
    )
    return WatcherContinuation.model_validate(values)


def _read_corrected_watch_deliverable(turn: WorkTurn) -> _DeliverableRead:
    try:
        corrected_watch = _read_watch_request(turn.workspace, turn.remote_stage)
    except (OSError, StateUnavailable, ValueError) as exc:
        return _DeliverableRead(
            text=None,
            failure=_DeliverableFailure(
                f"The corrected watcher request could not be read: {exc}",
                correctable=True,
            ),
        )
    if corrected_watch is None:
        return _DeliverableRead(
            text=None,
            failure=_DeliverableFailure(
                "The correction completed without writing watch.json.",
                correctable=True,
            ),
        )
    # Correction validates the resulting Patch/watch handoff, not an output
    # delta. An empty handoff may already be correct while only patch.json changes.
    return _DeliverableRead(text=corrected_watch)


async def _validate_watch_deliverable(
    turn: WorkTurn,
    staged: _StagedWorkInputs,
    watch_text: str,
    correction_rounds: int,
    settled: _SettledExperimentDeliverables,
) -> _DeliverableStep:
    try:
        if turn.execution is None:
            raise ValueError("Watcher arming requires a durable originating operation.")
        origin_task = turn.execution.store.agent_task(turn.execution.operation_id)
        if origin_task is None:
            raise ValueError("The originating Experiment-loop operation is no longer available.")
        handoff = parse_experiment_watch_json(watch_text)
        settled.loop_watch_text = watch_text
        if handoff.is_empty:
            exit_patch_text = _read_chat_patch(turn.workspace, turn.remote_stage)
            if (
                not turn.request.control_node_id
                or experiment_loop_semantic_ending(
                    exit_patch_text,
                    turn.request.control_node_id,
                )
                is None
            ):
                completion_problem = (
                    experiment_exit_problem(
                        exit_patch_text,
                        turn.request.control_node_id,
                    )
                    if turn.request.control_node_id
                    else None
                )
                raise ValueError(
                    completion_problem
                    or "An Experiment-loop watch.json with both lists empty requires "
                    "patch.json to explicitly record success, a Proposal, or a "
                    "same-Patch Blocker."
                )
        binding = WatcherBinding(
            project_id=origin_task.project_id,
            origin_operation_id=root_experiment_loop_operation_id(turn.execution),
            origin_task_kind=turn.surface,
            chat_id=turn.request.chat_id or "",
            node_id=turn.request.node_id,
            episode_id=turn.request.control_episode_id,
            graph_target=origin_task.graph_target,
            execution_host=turn.execution_host,
            continuation=_watcher_continuation(turn, staged),
        )
        if handoff.stops:
            turn.execution.store.validate_experiment_agent_watcher_stops(
                binding,
                handoff.stops,
            )
        graph_armed_revision = None
        if handoff.graph_conditions:
            graph_state = await asyncio.to_thread(turn.service.history.state)
            await asyncio.to_thread(
                validate_graph_conditions,
                handoff.graph_conditions,
                graph_state,
            )
            graph_armed_revision = graph_state.revision
        check_results = (
            await asyncio.to_thread(
                validate_watch_specs,
                handoff.observers,
                turn.execution_host,
            )
            if handoff.observers
            else []
        )
        settled.pending_loop_handoff = (
            handoff.observers,
            check_results,
            handoff.graph_conditions,
            graph_armed_revision,
            binding,
            handoff.stops,
        )
    except WatcherInitialCheckError as exc:
        return _DeliverableStep(failure=_DeliverableFailure(str(exc), correctable=True))
    except ValueError as exc:
        return _DeliverableStep(failure=_DeliverableFailure(str(exc), correctable=True))
    except (OSError, ReplayHalted, StateUnavailable) as exc:
        return _DeliverableStep(failure=_DeliverableFailure(str(exc), correctable=False))

    settled.loop_watch_empty = (
        not handoff.observers
        and not handoff.graph_conditions
        and (
            not handoff.stops
            or not turn.execution.store.experiment_handoff_has_live_watcher_after_stops(
                binding,
                [item.stop_watcher_id for item in handoff.stops],
            )
        )
    )
    settled.watch_correction_rounds = correction_rounds
    return _DeliverableStep()


def _watch_correction_contract(
    turn: WorkTurn,
    composed: _ComposedWorkPrompt,
    diagnostics_path: str,
    validator_command: str,
) -> str:
    return experiment_loop_watcher_correction_contract(
        original_contract_path=composed.base_contract_path,
        diagnostics_path=diagnostics_path,
        watch_path=turn.patch_inputs.watch_path,
        patch_path=turn.patch_inputs.patch_path,
        output_schema_path=turn.patch_inputs.schema_path,
        validator_command=validator_command,
    )


def _reject_watch_deliverable(
    turn: WorkTurn,
    settled: _SettledExperimentDeliverables,
    failure: _DeliverableFailure,
    correction_rounds: int,
) -> _DeliverableStep:
    if turn.execution is not None:
        turn.execution.store.record_agent_task_receipt(
            turn.execution.operation_id,
            "watcher_handoff_rejected",
            {
                "problem": failure.message[:1600],
                "correction_rounds": correction_rounds,
            },
            tier="diagnostic",
        )
        turn.execution.store.record_agent_task_event(
            turn.execution.operation_id,
            f"Watcher handoff was not armed: {failure.message}",
            level="warning",
        )
    settled.watch_correction_rounds = correction_rounds
    return _DeliverableStep(
        frames=(
            _sse(
                AgentEvent(
                    event="error",
                    text=f"Experiment-loop watcher handoff failed: {failure.message}",
                )
            ),
        ),
        stop=True,
    )


async def _settle_watch_deliverable(
    turn: WorkTurn,
    launcher: AgentLauncher,
    staged: _StagedWorkInputs,
    composed: _ComposedWorkPrompt,
    predecessor_digest: str | None,
    settled: _SettledExperimentDeliverables,
) -> AsyncIterator[str]:
    initial = _read_initial_watch_deliverable(turn, predecessor_digest)
    text = initial.text
    failure = initial.failure
    settled.loop_watch_text = text
    if text is None and failure is None:
        return

    maximum_corrections = 1
    correction_rounds = 0
    while True:
        if text is not None:
            step = await _validate_watch_deliverable(
                turn,
                staged,
                text,
                correction_rounds,
                settled,
            )
            for frame in step.frames:
                yield frame
            if step.stop:
                settled.stop = True
                return
            failure = step.failure
            if failure is None:
                return
        assert failure is not None
        if (
            not failure.correctable
            or correction_rounds >= maximum_corrections
            or not settled.native_session_id
        ):
            step = _reject_watch_deliverable(turn, settled, failure, correction_rounds)
            for frame in step.frames:
                yield frame
            if step.stop:
                settled.stop = True
            return

        correction_rounds += 1
        assert turn.execution is not None
        turn.execution.store.record_agent_task_receipt(
            turn.execution.operation_id,
            "watcher_correction_requested",
            {"round": correction_rounds, "problem": failure.message[:400]},
            tier="diagnostic",
        )
        turn.execution.store.update_agent_task_message(
            turn.execution.operation_id,
            "Correcting watcher handoff.",
            phase="correcting",
            event=True,
        )
        diagnostics_path = _stage_json_task_input(
            turn.local_stage,
            turn.remote_stage,
            f"task-{staged.token}-watch-correction-{correction_rounds}.json",
            {"problem": failure.message},
        )
        correction_validator: StagedCommandMailbox | None = None
        correction_lifecycle: _WorkValidatorMailboxLifecycle | None = None
        try:
            validator_command = turn.patch_inputs.validator_command
            correction_validator = stage_patch_validation_mailbox(
                local_stage=turn.local_stage,
                remote_stage=turn.remote_stage,
                task_id=turn.execution.operation_id,
                turn_id=f"{staged.token}:watch-correction:{correction_rounds}",
                timeout_seconds=PATCH_SELF_CHECK_TIMEOUT_SECONDS,
            )
            correction_lifecycle = _start_work_validator_mailbox(
                turn.service,
                correction_validator,
                execution=turn.execution,
                budget=turn.validator_budget,
                run_truth_scope=turn.context.run_truth_scope,
                control_node_id=turn.request.control_node_id,
                control_decision_bundle=turn.request.control_decision_bundle,
            )
            validator_command = correction_validator.client_command(
                "validate",
                turn.patch_inputs.patch_path,
            )
            correction_contract = _watch_correction_contract(
                turn,
                composed,
                diagnostics_path,
                validator_command,
            )
            correction_path, correction_prompt = _stage_task_contract(
                turn.local_stage,
                turn.remote_stage,
                f"task-{staged.token}-watch-correction-{correction_rounds}.md",
                correction_contract,
                execution=turn.execution,
                role=f"watch_correction_{correction_rounds}",
            )
            _record_agent_launch_receipt(
                turn.execution,
                turn.request,
                prompt=correction_prompt,
                contract_path=correction_path,
                remote=bool(turn.execution_host),
                resumed=True,
                write_scope=turn.write_scope,
                continuation="watch_correction",
                extra={
                    "surface": turn.surface,
                    "mode": "work",
                    "capability": "work_auto",
                    "network_access": True,
                    "launch_kind": "watch_correction",
                    "correction_round": correction_rounds,
                    "write_directory_count": len(turn.write_dirs),
                    "canonical_state_boundary": "prompt_only",
                },
            )
            correction_outcome = _ProviderOutcome(session_id=settled.native_session_id)
            correction_error: str | None = None
            correction_stream = _stream_turn_agent_events(
                turn,
                launcher,
                correction_prompt,
                session_id=settled.native_session_id,
                required_session_id=settled.native_session_id,
                outcome=correction_outcome,
                validator_staged=correction_validator,
                validator_lifecycle=correction_lifecycle,
            )
        except BaseException as exc:
            if correction_lifecycle is not None:
                await correction_lifecycle.close(primary_error=exc)
            elif correction_validator is not None and not correction_validator.credential.expired:
                await _close_work_validator_mailbox(
                    correction_validator,
                    stop=None,
                    task=None,
                    execution=turn.execution,
                    primary_error=exc,
                )
            raise
        async with aclosing(correction_stream) as stream:
            async for frame in stream:
                event = AgentEvent.model_validate_json(frame.removeprefix("data: ").strip())
                if event.event == "error":
                    correction_error = event.text or "Watcher correction failed."
                    continue
                if event.event not in {"answer", "done"}:
                    yield frame
        settled.native_session_id = correction_outcome.session_id or settled.native_session_id
        if correction_outcome.paused:
            settled.stop = True
            return
        if correction_error or not correction_outcome.completed:
            detail = correction_error or (
                f"{turn.request.provider} produced no watcher correction result."
            )
            failure = _DeliverableFailure(
                detail,
                correctable=True,
                change_summary=failure.change_summary,
                proposal_ids=failure.proposal_ids,
            )
            text = None
            correction_rounds = maximum_corrections
            continue
        corrected = _read_corrected_watch_deliverable(turn)
        text = corrected.text
        failure = corrected.failure
        settled.loop_watch_text = text


async def _apply_experiment_loop_turn(
    turn: WorkTurn,
    launcher: AgentLauncher,
    staged: _StagedWorkInputs,
    composed: _ComposedWorkPrompt,
    prompt_context: _WorkPromptContext,
    settled: _SettledExperimentDeliverables,
    applied: _AppliedWorkTurn,
) -> AsyncIterator[str]:
    try:
        final_patch_text = _read_chat_patch(turn.workspace, turn.remote_stage)
    except (OSError, StateUnavailable, ValueError) as exc:
        yield _sse(
            AgentEvent(
                event="error",
                text=f"The final Experiment-loop patch could not be read: {exc}",
            )
        )
        applied.stop = True
        return
    if final_patch_text is None:
        applied.graph_update = GraphUpdateResult(status="none")
    else:
        loop_patch_correction_rounds = 0
        while True:
            # A correction round that produced nothing new leaves this None so
            # its own diagnostic reaches the agent instead of being overwritten.
            if final_patch_text is not None:
                if settled.loop_watch_empty and (
                    not turn.request.control_node_id
                    or experiment_loop_semantic_ending(
                        final_patch_text,
                        turn.request.control_node_id,
                    )
                    is None
                ):
                    final_failure = _DeliverableFailure(
                        "A watch.json with both lists empty requires this Patch to retain an "
                        "explicit success, Proposal, or same-Patch Blocker.",
                        correctable=True,
                    )
                    final_result = None
                else:
                    try:
                        final_result, final_failure = _apply_work_patch(
                            turn.service,
                            turn.execution,
                            final_patch_text,
                            run_truth_scope=turn.context.run_truth_scope,
                            control_node_id=turn.request.control_node_id,
                            control_decision_bundle=(turn.request.control_decision_bundle),
                        )
                    except RunLockCancelled:
                        yield _sse(
                            AgentEvent(
                                event="paused",
                                text=(
                                    "Paused while waiting for canonical state. The operational "
                                    "answer and retained patch are preserved."
                                ),
                            )
                        )
                        applied.stop = True
                        return
                if final_result is not None:
                    applied.graph_update = final_result.model_copy(
                        update={"correction_rounds": loop_patch_correction_rounds}
                    )
                    break
            assert final_failure is not None
            if (
                not final_failure.correctable
                or loop_patch_correction_rounds >= PATCH_CORRECTION_MAX_ROUNDS
                or not applied.native_session_id
            ):
                if settled.loop_watch_empty:
                    yield _sse(
                        AgentEvent(
                            event="error",
                            text=(
                                "Experiment-loop Patch could not be validated after its watcher "
                                f"handoff: {final_failure.message}"
                            ),
                        )
                    )
                    applied.stop = True
                    return
                repairable = _work_graph_repairable(
                    turn.execution,
                    applied.native_session_id,
                    final_failure,
                )
                applied.graph_update = GraphUpdateResult(
                    status="rejected",
                    change_summary=list(final_failure.change_summary),
                    proposal_ids=list(final_failure.proposal_ids),
                    validation_messages=_bounded_graph_messages(final_failure.message),
                    correction_rounds=loop_patch_correction_rounds,
                    repairable=repairable,
                )
                _record_work_graph_rejection(turn.execution, applied.graph_update)
                break

            loop_patch_correction_rounds += 1
            assert turn.execution is not None
            turn.execution.store.record_agent_task_receipt(
                turn.execution.operation_id,
                "patch_correction_requested",
                {
                    "round": loop_patch_correction_rounds,
                    "problem": final_failure.message[:400],
                },
                tier="diagnostic",
            )
            diagnostics_path = _stage_json_task_input(
                turn.local_stage,
                turn.remote_stage,
                f"task-{staged.token}-loop-patch-correction-{loop_patch_correction_rounds}.json",
                {"kind": "experiment_loop", "problem": final_failure.message},
            )
            loop_validator = stage_patch_validation_mailbox(
                local_stage=turn.local_stage,
                remote_stage=turn.remote_stage,
                task_id=turn.execution.operation_id,
                turn_id=(f"{staged.token}:loop-patch-correction:{loop_patch_correction_rounds}"),
                timeout_seconds=PATCH_SELF_CHECK_TIMEOUT_SECONDS,
            )
            loop_validator_lifecycle = _start_work_validator_mailbox(
                turn.service,
                loop_validator,
                execution=turn.execution,
                budget=turn.validator_budget,
                run_truth_scope=turn.context.run_truth_scope,
                control_node_id=turn.request.control_node_id,
                control_decision_bundle=turn.request.control_decision_bundle,
            )
            try:
                loop_validator_command = loop_validator.client_command(
                    "validate",
                    turn.patch_inputs.patch_path,
                )
                correction_contract = experiment_loop_patch_correction_contract(
                    original_contract_path=composed.base_contract_path,
                    diagnostics_path=diagnostics_path,
                    patch_path=turn.patch_inputs.patch_path,
                    watch_path=turn.patch_inputs.watch_path,
                    validator_command=loop_validator_command,
                )
                correction_path, correction_prompt = _stage_task_contract(
                    turn.local_stage,
                    turn.remote_stage,
                    f"task-{staged.token}-loop-patch-correction-{loop_patch_correction_rounds}.md",
                    correction_contract,
                    execution=turn.execution,
                    role=(f"experiment_loop_patch_correction_{loop_patch_correction_rounds}"),
                )
                pre_launch_digest = _existing_patch_digest(
                    turn.workspace,
                    turn.remote_stage,
                )
                _record_agent_launch_receipt(
                    turn.execution,
                    turn.request,
                    prompt=correction_prompt,
                    contract_path=correction_path,
                    remote=bool(turn.execution_host),
                    resumed=True,
                    write_scope=turn.write_scope,
                    continuation="graph_correction",
                    extra={
                        "surface": turn.surface,
                        "mode": "work",
                        "capability": "work_auto",
                        "network_access": True,
                        "launch_kind": "graph_correction",
                        "correction_round": loop_patch_correction_rounds,
                        "write_directory_count": len(turn.write_dirs),
                        "canonical_state_boundary": "prompt_only",
                    },
                )
                correction_outcome = _ProviderOutcome(session_id=applied.native_session_id)
            except BaseException as exc:
                await loop_validator_lifecycle.close(primary_error=exc)
                raise
            async with aclosing(
                _stream_turn_agent_events(
                    turn,
                    launcher,
                    correction_prompt,
                    session_id=applied.native_session_id,
                    required_session_id=applied.native_session_id,
                    outcome=correction_outcome,
                    validator_staged=loop_validator,
                    validator_lifecycle=loop_validator_lifecycle,
                )
            ) as stream:
                async for frame in stream:
                    yield frame
            applied.native_session_id = correction_outcome.session_id or applied.native_session_id
            if correction_outcome.paused:
                applied.stop = True
                return
            if not correction_outcome.completed:
                final_failure = _DeliverableFailure(
                    f"{turn.request.provider} produced no Patch correction result.",
                    correctable=True,
                )
                final_patch_text = None
                loop_patch_correction_rounds = PATCH_CORRECTION_MAX_ROUNDS
                continue
            corrected = _read_correction_patch(
                turn.workspace,
                turn.remote_stage,
                pre_launch_digest=pre_launch_digest,
            )
            if corrected.problem == "unreadable":
                final_failure = _DeliverableFailure(
                    f"The corrected loop Patch could not be read: {corrected.detail}",
                    correctable=True,
                )
                final_patch_text = None
                continue
            if corrected.problem in {"missing", "unchanged"}:
                final_failure = _DeliverableFailure(
                    "The loop Patch correction did not rewrite patch.json.",
                    correctable=True,
                )
                final_patch_text = None
                continue
            assert corrected.text is not None
            final_patch_text = corrected.text

    if (
        turn.execution is None
        or prompt_context.episode_context_baseline is None
        or prompt_context.experiment_control_snapshot is None
    ):
        raise ValueError("Experiment-loop handoff lost its durable episode context.")
    execution = turn.execution
    if settled.pending_loop_handoff is not None:
        (
            specs,
            check_results,
            graph_conditions,
            graph_armed_revision,
            binding,
            stop_requests,
        ) = settled.pending_loop_handoff
    else:
        origin_task = execution.store.agent_task(execution.operation_id)
        if origin_task is None:
            raise ValueError("The originating Experiment-loop operation is no longer available.")
        specs = []
        check_results = []
        graph_conditions = []
        graph_armed_revision = None
        stop_requests = []
        binding = WatcherBinding(
            project_id=origin_task.project_id,
            origin_operation_id=root_experiment_loop_operation_id(execution),
            origin_task_kind=turn.surface,
            chat_id=turn.request.chat_id or "",
            node_id=turn.request.node_id,
            episode_id=turn.request.control_episode_id,
            graph_target=origin_task.graph_target,
            execution_host=turn.execution_host,
            continuation=_watcher_continuation(turn, staged),
        )

    try:
        graph_state = (
            await asyncio.to_thread(turn.service.history.state) if graph_conditions else None
        )
        prepared = await asyncio.to_thread(
            prepare_experiment_watcher_records,
            execution,
            specs,
            check_results,
            binding,
            graph_conditions=graph_conditions,
            graph_state=graph_state,
            armed_revision=graph_armed_revision,
        )
        prepared_watcher_ids = [item.watcher_id for item in prepared]
        prepared_stopped_watcher_ids = [item.stop_watcher_id for item in stop_requests]
    except (OSError, ReplayHalted, StateUnavailable, ValueError) as exc:
        yield _sse(
            AgentEvent(
                event="error",
                text=f"Experiment-loop watcher handoff preparation failed: {exc}",
            )
        )
        applied.stop = True
        return

    accepted_loop_watcher_ids = prepared_watcher_ids
    accepted_loop_stopped_watcher_ids = prepared_stopped_watcher_ids
    ending_signal = None
    if applied.graph_update.status == "applied" and turn.request.control_node_id:
        semantic_ending = experiment_loop_semantic_ending(
            final_patch_text,
            turn.request.control_node_id,
        )
        if semantic_ending is not None:
            if (
                final_patch_text is None
                or not turn.request.control_episode_id
                or turn.request.control_invocation is None
                or turn.request.control_invocation_ceiling is None
            ):
                raise ValueError("Experiment-loop ending lost its compact episode receipt inputs.")
            ending_signal = experiment_loop_ending_signal(
                semantic_ending=semantic_ending,
                episode_id=turn.request.control_episode_id,
                control_node_id=turn.request.control_node_id,
                invocation=turn.request.control_invocation,
                invocation_ceiling=turn.request.control_invocation_ceiling,
                control_snapshot=prompt_context.experiment_control_snapshot,
                patch_text=final_patch_text,
                graph_update=applied.graph_update,
                watcher_ids=accepted_loop_watcher_ids,
                stopped_watcher_ids=accepted_loop_stopped_watcher_ids,
                decision_bundle=turn.request.control_decision_bundle,
            )
    patch_digest = (
        hashlib.sha256(final_patch_text.encode("utf-8")).hexdigest()
        if final_patch_text is not None
        else None
    )
    watch_text = settled.loop_watch_text
    watch_digest = (
        hashlib.sha256(watch_text.encode("utf-8")).hexdigest() if watch_text is not None else None
    )
    root_id = root_experiment_loop_operation_id(execution)
    execution.store.record_agent_task_receipt(
        execution.operation_id,
        "experiment_loop_handoff_prepared",
        {
            "episode_id": turn.request.control_episode_id,
            "invocation": turn.request.control_invocation,
            "root_operation_id": root_id,
            "patch_sha256": patch_digest,
            "watch_sha256": watch_digest,
            "graph_status": applied.graph_update.status,
            "applied_revision": applied.graph_update.applied_revision,
            "watcher_ids": prepared_watcher_ids,
            "requested_stop_ids": [item.stop_watcher_id for item in stop_requests],
        },
    )
    try:
        armed = await asyncio.to_thread(
            commit_experiment_episode_handoff,
            execution,
            turn.request,
            prepared,
            binding,
            native_session_id=applied.native_session_id,
            execution_host=turn.execution_host,
            stage_host=execution.stage_host,
            stage_root=execution.stage_root,
            graph_result=experiment_graph_result_summary(applied.graph_update),
            context_baseline=prompt_context.episode_context_baseline,
            stops=stop_requests,
            ending_signal=ending_signal,
        )
    except (OSError, ReplayHalted, StateUnavailable, ValueError) as exc:
        yield _sse(
            AgentEvent(
                event="error",
                text=f"Experiment-loop watcher handoff failed: {exc}",
            )
        )
        applied.stop = True
        return
    if graph_conditions:
        execution.armed_graph_watchers = True
    execution.store.record_agent_task_receipt(
        root_id,
        "watchers_armed",
        {
            "watcher_ids": [item.watcher_id for item in armed],
            "stopped_watcher_ids": [item.stop_watcher_id for item in stop_requests],
            "count": len(armed),
            "correction_rounds": settled.watch_correction_rounds,
        },
    )


async def _launch_and_stream_work_turn(
    turn: WorkTurn,
    launcher: AgentLauncher,
    prompt: str,
    contract_path: str,
    staged: _StagedWorkInputs,
    wake_episode: EpisodeRecord | None,
    required_session_id: str | None = None,
) -> AsyncIterator[str]:
    try:
        _record_agent_launch_receipt(
            turn.execution,
            turn.request,
            prompt=prompt,
            contract_path=contract_path,
            remote=bool(turn.execution_host),
            resumed=turn.reusing_checkpoint,
            write_scope=turn.write_scope,
            continuation=turn.continuation,
            extra={
                "surface": turn.surface,
                "mode": "work",
                "capability": "work_auto",
                "network_access": True,
                "launch_kind": (
                    "retry"
                    if turn.retry_attempt
                    else "resume"
                    if turn.resuming
                    else "message_wake"
                    if turn.continuation == "message_wake"
                    else "watcher_wake"
                    if turn.waking
                    else "initial"
                ),
                "write_directory_count": len(turn.write_dirs),
                "canonical_state_boundary": "prompt_only",
            },
        )
    except BaseException as exc:
        await turn.validator_lifecycle.close(primary_error=exc)
        raise
    try:
        try:
            async with aclosing(
                _stream_turn_agent_events(
                    turn,
                    launcher,
                    prompt,
                    session_id=turn.request.session_id,
                    required_session_id=(
                        required_session_id
                        if required_session_id is not None
                        else _required_work_continuation_session_id(
                            turn.request,
                            turn.execution,
                            session_id=turn.request.session_id,
                        )
                    ),
                    outcome=turn.outcome,
                )
            ) as stream:
                async for frame in stream:
                    yield frame
        except Exception:
            turn.outcome.failed = True
            raise

        answer = "\n\n".join(item.strip() for item in turn.outcome.answers if item.strip()).strip()
        if not turn.outcome.completed:
            if turn.outcome.failed or turn.outcome.paused:
                return
            turn.outcome.failed = True
            yield _sse(
                AgentEvent(event="error", text=f"{turn.request.provider} produced no result.")
            )
            return
        if not answer:
            yield _sse(
                AgentEvent(
                    event="error",
                    text=f"{turn.request.provider} finished without answering.",
                )
            )
            return
        if turn.waking and (
            wake_episode is None or turn.outcome.session_id != wake_episode.native_session_id
        ):
            yield _sse(
                AgentEvent(
                    event="error",
                    text=(
                        "The automatic Experiment wake did not continue its committed native "
                        "provider session. The watcher handoff was not accepted."
                    ),
                )
            )
            return

        try:
            artifacts = _discover_chat_artifacts(
                turn.execution,
                staged.artifact_scope_id,
                Path(str(staged.artifact_directory)),
                turn.remote_stage,
            )
        except Exception as exc:
            with suppress(Exception):
                _record_artifact_discovery_receipt(
                    turn.execution,
                    attached=0,
                    candidates=0,
                    ignored={"unexpected_error": 1},
                    detail=str(exc),
                )
            artifacts = []
    except BaseException:
        raise
    turn.answer = answer
    yield _sse(AgentEvent(event="answer", text=answer))
    for artifact in artifacts:
        yield _sse(AgentEvent(event="artifact", artifact=artifact))


async def stream_experiment_loop_task(
    service: ProjectService,
    launcher: AgentLauncher,
    request: RunRequest,
    data_dir: Path,
    execution: AgentTaskExecution | None = None,
) -> AsyncIterator[str]:
    """Run one already-admitted Experiment-loop invocation end to end."""

    if request.mode != "work" or request.patch_kind != "experiment_loop":
        yield _sse(
            AgentEvent(
                event="error",
                text="The Experiment-loop task owner received a non-Experiment Work request.",
            )
        )
        return

    if execution is not None and execution.continuation == "graph_repair":
        async with aclosing(
            _stream_work_graph_repair(
                service,
                launcher,
                request,
                data_dir,
                execution=execution,
            )
        ) as stream:
            async for frame in stream:
                yield frame
        return

    try:
        resolved = _resolve_work_execution(service, request, execution)
    except ValueError as exc:
        yield _sse(AgentEvent(event="error", text=str(exc)))
        return
    request = resolved.request
    turn: WorkTurn | None = None
    patch_inputs = None
    validator_lifecycle: _WorkValidatorMailboxLifecycle | None = None
    try:
        turn, staged = await _stage_work_turn(service, resolved, data_dir, execution)
        patch_inputs = turn.patch_inputs
        validator_lifecycle = turn.validator_lifecycle
        outcome = turn.outcome
        resuming = turn.resuming
        # An Experiment-loop watcher wake resumes the episode's native session, but it
        # is a new turn at the next invocation -- never task Resume, never a retry, and
        # never a rebuilt master contract.
        waking = turn.waking
        prompt_context = await _prepare_work_prompt_context(turn, staged)
        wake_episode = prompt_context.wake_episode
        if resuming:
            composed_prompt = _compose_resume_prompt(turn, staged, prompt_context)
        elif waking:
            composed_prompt = _compose_wake_prompt(turn, staged, prompt_context)
        else:
            if turn.retrying:
                composed_prompt = _compose_retry_prompt(turn, staged, prompt_context)
            else:
                retry_diagnostics_path = _stage_retry_diagnostics(turn, staged)
                composed_prompt = _compose_fresh_prompt(
                    turn,
                    staged,
                    prompt_context,
                    retry_diagnostics_path=retry_diagnostics_path,
                )
        contract_path = composed_prompt.contract_path
        prompt = composed_prompt.prompt
        retry_baseline = _capture_retry_deliverable_baseline(turn)
        retry_patch_digest = retry_baseline.patch_digest
        retry_watch_digest = retry_baseline.watch_digest
    except BaseException as exc:
        if validator_lifecycle is not None:
            await validator_lifecycle.close(primary_error=exc)
        elif patch_inputs is not None and not patch_inputs.validator_staged.credential.expired:
            await _close_work_validator_mailbox(
                patch_inputs.validator_staged,
                stop=None,
                task=None,
                execution=execution,
                primary_error=exc,
            )
        if isinstance(exc, (OSError, ReplayHalted, StateUnavailable, ValueError)):
            yield _sse(AgentEvent(event="error", text=str(exc)))
            return
        raise

    assert turn is not None
    async with aclosing(
        _launch_and_stream_work_turn(
            turn,
            launcher,
            prompt,
            contract_path,
            staged,
            wake_episode,
        )
    ) as stream:
        async for frame in stream:
            yield frame
    if turn.answer is None:
        return
    answer = turn.answer

    settled = _SettledExperimentDeliverables(native_session_id=outcome.session_id)
    _read_initial_patch_deliverable(turn, retry_patch_digest, settled)
    if settled.stop:
        return
    async with aclosing(
        _settle_watch_deliverable(
            turn,
            launcher,
            staged,
            composed_prompt,
            retry_watch_digest,
            settled,
        )
    ) as stream:
        async for frame in stream:
            yield frame
    if settled.stop:
        return
    applied = _AppliedWorkTurn(
        graph_update=settled.graph_update,
        native_session_id=settled.native_session_id,
    )
    apply_stream = _apply_experiment_loop_turn(
        turn,
        launcher,
        staged,
        composed_prompt,
        prompt_context,
        settled,
        applied,
    )
    async with aclosing(apply_stream) as stream:
        async for frame in stream:
            yield frame
    if applied.stop:
        return
    graph_update = applied.graph_update

    for frame in _finalize_work_turn(turn, answer, graph_update):
        yield frame


async def _stream_work_graph_repair(
    service: ProjectService,
    launcher: AgentLauncher,
    request: RunRequest,
    data_dir: Path,
    *,
    execution: AgentTaskExecution,
) -> AsyncIterator[str]:
    """Repair only a retained Experiment-loop Patch; never repeat the operational turn."""

    surface: AgentSurface = "project_chat" if request.chat_scope == "project" else "node_chat"
    patch_inputs = None
    validator_lifecycle: _WorkValidatorMailboxLifecycle | None = None
    validator_budget = PatchValidationBudget()
    try:
        profile = service.resolve_agent_profile(
            surface,
            provider=request.provider,
            model=request.model,
            reasoning=request.reasoning,
            run_on=request.run_on,
        )
        request = _pinned_to_profile(request, profile)
        execution_machine = service.manifest.machine_map[profile.run_on]
        execution_host = execution_machine.host
        provider_binary = execution_machine.provider_paths.get(profile.provider)
        context = service.assemble_chat(request)
        stage_name = _chat_stage_name(service, request, execution)
        local_stage: Path | None = None
        remote_stage: RemoteRunStage | None = None
        if execution_host:
            stage_root = _validated_remote_chat_resume_stage(execution, execution_host, stage_name)
            remote_stage = RemoteRunStage(execution_host).attach(stage_root)
            context = context.model_copy(
                update=_stage_context_paths(
                    context,
                    service,
                    remote_stage,
                    execution_machine.alias,
                )
            )
            workspace = Path(str(remote_stage.workspace))
        else:
            expected_stage = _swept_stage_root(data_dir) / stage_name
            local_stage = _validated_local_chat_resume_stage(execution, expected_stage)
            workspace = local_stage
        token = _task_token(execution)
        patch_inputs = _stage_chat_patch_inputs(
            local_stage,
            remote_stage,
            workspace=workspace,
            stage_name=stage_name,
            task_id=execution.operation_id,
            turn_id=f"{token}:work-graph-repair",
        )
        validator_lifecycle = _start_work_validator_mailbox(
            service,
            patch_inputs.validator_staged,
            execution=execution,
            budget=validator_budget,
            run_truth_scope=context.run_truth_scope,
            control_node_id=request.control_node_id,
            control_decision_bundle=request.control_decision_bundle,
        )
        patch_path = patch_inputs.patch_path
        read_dirs = _chat_read_dirs(
            context,
            remote_stage,
            service,
            execution_machine.alias,
        )
        write_scope = _project_write_scope(
            context,
            service,
            execution_machine.alias,
            workspace=workspace,
            remote_stage=remote_stage,
            data_dir=data_dir,
            execution=execution,
            capability="work_auto",
        )
        write_dirs = [Path(item) for item in write_scope.repository_roots]
        assert validator_lifecycle is not None
        outcome = _ProviderOutcome(session_id=request.session_id)
        turn = WorkTurn(
            service=service,
            request=request,
            execution=execution,
            context=context,
            workspace=workspace,
            local_stage=local_stage,
            remote_stage=remote_stage,
            execution_host=execution_host,
            provider_binary=provider_binary,
            read_dirs=read_dirs,
            write_dirs=write_dirs,
            write_scope=write_scope,
            patch_inputs=patch_inputs,
            validator_lifecycle=validator_lifecycle,
            validator_budget=validator_budget,
            outcome=outcome,
        )
        previous = _rejected_graph_update_for_repair(execution)
        original_contract_path = _parent_task_contract_path(execution, local_stage, remote_stage)
        validator_command = patch_inputs.validator_command
        diagnostics_path = _stage_json_task_input(
            local_stage,
            remote_stage,
            f"task-{token}-manual-graph-repair.json",
            {
                "kind": "work",
                "problems": previous.validation_messages,
                "prior_correction_rounds": previous.correction_rounds,
            },
        )
        contract = PromptFactory.continuation_task_contract(
            original_contract_path=original_contract_path,
            mode="work_patch_correction",
            patch_path=patch_path,
            diagnostics_path=diagnostics_path,
            validator_command=validator_command,
        )
        contract_path, prompt = _stage_task_contract(
            local_stage,
            remote_stage,
            f"task-{token}-manual-graph-repair.md",
            contract,
            execution=execution,
            role="work_patch_repair",
        )
        pre_launch_digest = _existing_patch_digest(workspace, remote_stage)
    except BaseException as exc:
        if validator_lifecycle is not None:
            await validator_lifecycle.close(primary_error=exc)
        elif patch_inputs is not None and not patch_inputs.validator_staged.credential.expired:
            await _close_work_validator_mailbox(
                patch_inputs.validator_staged,
                stop=None,
                task=None,
                execution=execution,
                primary_error=exc,
            )
        if isinstance(exc, (OSError, ReplayHalted, StateUnavailable, ValueError)):
            yield _sse(AgentEvent(event="error", text=str(exc)))
            return
        raise

    assert validator_lifecycle is not None
    try:
        _record_agent_launch_receipt(
            execution,
            request,
            prompt=prompt,
            contract_path=contract_path,
            remote=bool(execution_host),
            resumed=True,
            write_scope=write_scope,
            continuation="graph_repair",
            extra={
                "surface": surface,
                "mode": "work",
                "capability": "work_auto",
                "network_access": True,
                "launch_kind": "graph_repair",
                "write_directory_count": len(write_dirs),
                "canonical_state_boundary": "prompt_only",
            },
        )
    except BaseException as exc:
        await validator_lifecycle.close(primary_error=exc)
        raise
    async with aclosing(
        _stream_turn_agent_events(
            turn,
            launcher,
            prompt,
            session_id=request.session_id,
            required_session_id=request.session_id,
            outcome=outcome,
        )
    ) as stream:
        async for frame in stream:
            yield frame
    if not outcome.completed:
        if outcome.failed or outcome.paused:
            return
        yield _sse(AgentEvent(event="error", text=f"{request.provider} produced no result."))
        return
    try:
        patch_text = _read_chat_patch(workspace, remote_stage)
    except (OSError, StateUnavailable, ValueError) as exc:
        yield _sse(AgentEvent(event="error", text=f"The repaired patch could not be read: {exc}"))
        return
    if patch_text is None:
        yield _sse(AgentEvent(event="error", text="The repair did not write patch.json."))
        return
    if (
        pre_launch_digest is not None
        and hashlib.sha256(patch_text.encode("utf-8")).hexdigest() == pre_launch_digest
    ):
        yield _sse(
            AgentEvent(
                event="error",
                text="The repair left patch.json byte-identical to the rejected patch.",
            )
        )
        return
    try:
        graph_update, failure = _apply_work_patch(
            service,
            execution,
            patch_text,
            run_truth_scope=context.run_truth_scope,
            control_node_id=request.control_node_id,
            control_decision_bundle=request.control_decision_bundle,
        )
    except RunLockCancelled:
        yield _sse(
            AgentEvent(
                event="paused",
                text="Paused while waiting for canonical state. The retained patch is preserved.",
            )
        )
        return
    if graph_update is None:
        assert failure is not None
        graph_update = GraphUpdateResult(
            status="rejected",
            change_summary=list(failure.change_summary),
            proposal_ids=list(failure.proposal_ids),
            validation_messages=_bounded_graph_messages(failure.message),
            correction_rounds=1,
        )
        _record_work_graph_rejection(execution, graph_update)
    if (
        not request.control_episode_id
        or not request.control_node_id
        or request.control_invocation is None
        or request.control_invocation_ceiling is None
    ):
        yield _sse(AgentEvent(event="error", text="The graph repair lost its Experiment episode."))
        return
    episode = execution.store.experiment_episode(request.control_episode_id)
    if episode is None or not episode.session_bound:
        yield _sse(
            AgentEvent(
                event="error",
                text="The graph repair has no bound Experiment episode to update.",
            )
        )
        return
    control_node = context.node
    if (
        not isinstance(control_node, dict)
        or control_node.get("id") != request.control_node_id
        or control_node.get("type") != "experiment"
    ):
        yield _sse(
            AgentEvent(
                event="error",
                text="The graph repair lost its Experiment control snapshot.",
            )
        )
        return
    ending_signal = None
    if graph_update.status == "applied":
        semantic_ending = experiment_loop_semantic_ending(
            patch_text,
            request.control_node_id,
        )
        if semantic_ending is not None:
            ending_signal = experiment_loop_ending_signal(
                semantic_ending=semantic_ending,
                episode_id=request.control_episode_id,
                control_node_id=request.control_node_id,
                invocation=request.control_invocation,
                invocation_ceiling=request.control_invocation_ceiling,
                control_snapshot=dict(control_node),
                patch_text=patch_text,
                graph_update=graph_update,
                watcher_ids=episode.last_watcher_ids,
                stopped_watcher_ids=[],
                decision_bundle=request.control_decision_bundle,
            )
    try:
        commit_experiment_episode_binding(
            execution,
            request,
            native_session_id=outcome.session_id,
            execution_host=execution_host,
            stage_host=episode.stage_host,
            stage_root=episode.stage_root,
            graph_result=experiment_graph_result_summary(graph_update),
            watcher_ids=episode.last_watcher_ids,
            context_baseline=episode.context_baseline,
            ending_signal=ending_signal,
        )
    except ValueError as exc:
        yield _sse(
            AgentEvent(
                event="error",
                text=f"The graph repair could not update its Experiment handoff: {exc}",
            )
        )
        return
    try:
        _append_chat_graph_receipt(
            service,
            request,
            outcome.session_id,
            graph_update,
            execution,
        )
    except (OSError, StateUnavailable, ValueError) as exc:
        execution.store.record_agent_task_event(
            execution.operation_id,
            f"The graph repair completed but its chat receipt could not be written: {exc}",
            level="warning",
        )
    payload: dict[str, object] = {
        "graph_update": graph_update.model_dump(mode="json"),
    }
    if graph_update.applied_revision is not None:
        payload["applied_revision"] = graph_update.applied_revision
    yield _sse(AgentEvent(event="message", text=json.dumps(payload, separators=(",", ":"))))
    yield _sse(AgentEvent(event="done"))


def _start_work_validator_mailbox(
    service: ProjectService,
    staged: StagedCommandMailbox,
    *,
    execution: AgentTaskExecution | None,
    budget: PatchValidationBudget,
    run_truth_scope: list[str],
    control_node_id: str | None,
    control_decision_bundle: list[ExperimentDecisionPin],
) -> _WorkValidatorMailboxLifecycle:
    stop = asyncio.Event()
    try:
        task = asyncio.create_task(
            serve_patch_validation_mailbox(
                staged=staged,
                execution=execution,
                validate=lambda text: _validate_work_patch_live(
                    service,
                    text,
                    run_truth_scope=run_truth_scope,
                    control_node_id=control_node_id,
                    control_decision_bundle=control_decision_bundle,
                    source_operation_id=_work_patch_source_operation_id(execution),
                ),
                stop=stop,
                budget=budget,
            )
        )
    except BaseException:
        with suppress(BaseException):
            staged.cleanup()
        raise
    return _WorkValidatorMailboxLifecycle(
        staged=staged,
        execution=execution,
        stop=stop,
        task=task,
    )


def _required_work_continuation_session_id(
    _request: RunRequest,
    execution: AgentTaskExecution | None,
    *,
    session_id: str | None,
) -> str | None:
    """Pin Experiment continuations to their saved native provider session."""

    if execution is None or not execution.reuses_native_checkpoint:
        return None
    return session_id


def _prepare_work_patch_candidate(
    service: ProjectService,
    patch_text: str,
    *,
    run_truth_scope: list[str],
    control_node_id: str | None,
    control_decision_bundle: list[ExperimentDecisionPin] | None,
    source_operation_id: str | None = None,
    source_effect_id: str | None = None,
    profile: AgentProfile = "ordinary",
) -> _PreparedWorkPatch:
    if profile == "ordinary":
        draft, _ = service.parse_patch_output([patch_text])
    else:
        draft = parse_agent_patch_json(patch_text, profile=profile)
    validate_agent_patch_shape(draft, profile=profile)
    patch = prepare_agent_patch(
        draft,
        kind="experiment_loop",
        run_truth_scope=run_truth_scope,
        repository_paths=service.manifest.repository_paths,
        source_operation_id=source_operation_id,
        source_effect_id=source_effect_id,
        source_effect_sha256=(
            hashlib.sha256(patch_text.encode("utf-8")).hexdigest()
            if source_effect_id is not None
            else None
        ),
        profile=profile,
    )
    patch = patch.model_copy(
        update={
            "experiment_control_node_id": control_node_id,
            "experiment_decision_bundle": list(control_decision_bundle or ()),
        }
    )
    if not control_node_id:
        raise ValueError("Experiment-loop Patch validation requires its focused Experiment.")
    validate_experiment_completion(patch, control_node_id)
    validate_work_patch(patch)
    return _PreparedWorkPatch(
        patch=patch,
        change_summary=tuple(draft.change_summary),
        proposal_ids=tuple(_work_patch_proposal_ids(patch)),
    )


def _validate_work_patch_live(
    service: ProjectService,
    patch_text: str,
    *,
    run_truth_scope: list[str],
    control_node_id: str | None,
    control_decision_bundle: list[ExperimentDecisionPin] | None,
    source_operation_id: str | None = None,
    source_effect_id: str | None = None,
    profile: AgentProfile = "ordinary",
) -> PatchValidationResult:
    try:
        candidate = _prepare_work_patch_candidate(
            service,
            patch_text,
            run_truth_scope=run_truth_scope,
            control_node_id=control_node_id,
            control_decision_bundle=control_decision_bundle,
            source_operation_id=source_operation_id,
            source_effect_id=source_effect_id,
            profile=profile,
        )
        prepared, report, state = service.history.validate_candidate(candidate.patch)
    except (ReplayHalted, StateUnavailable, OSError) as exc:
        return PatchValidationResult(status="unavailable", messages=[str(exc)])
    except ValueError as exc:
        return PatchValidationResult(status="invalid", messages=[str(exc)])
    rejects = [item.message for item in report.messages if item.level == "reject"]
    if rejects:
        return PatchValidationResult(
            status="invalid",
            messages=_bounded_graph_messages(*rejects),
            live_revision=state.revision,
            candidate_revision=prepared.revision,
        )
    return PatchValidationResult(
        status="valid",
        messages=_bounded_graph_messages(*(item.message for item in report.flags)),
        live_revision=state.revision,
        candidate_revision=prepared.revision,
    )


def _apply_work_patch(
    service: ProjectService,
    execution: AgentTaskExecution | None,
    patch_text: str,
    *,
    run_truth_scope: list[str],
    control_node_id: str | None = None,
    control_decision_bundle: list[ExperimentDecisionPin] | None = None,
    profile: AgentProfile = "ordinary",
    source_operation_id: str | None = None,
    source_effect_id: str | None = None,
) -> tuple[GraphUpdateResult | None, _DeliverableFailure | None]:
    """Validate and atomically apply one Experiment-loop Patch candidate."""

    if execution is not None:
        execution.store.record_agent_task_patch_output(execution.operation_id, patch_text)
        execution.store.record_agent_task_receipt(
            execution.operation_id,
            "patch_retained",
            {"byte_length": len(patch_text.encode("utf-8")), "file_name": "patch.json"},
            tier="diagnostic",
        )
    change_summary: tuple[str, ...] = ()
    proposal_ids: tuple[str, ...] = ()
    source_operation_id = source_operation_id or _work_patch_source_operation_id(execution)
    canonical_patch: Patch | None = None
    try:
        candidate = _prepare_work_patch_candidate(
            service,
            patch_text,
            run_truth_scope=run_truth_scope,
            control_node_id=control_node_id,
            control_decision_bundle=control_decision_bundle,
            source_operation_id=source_operation_id,
            source_effect_id=source_effect_id,
            profile=profile,
        )
        patch = candidate.patch
        change_summary = candidate.change_summary
        proposal_ids = candidate.proposal_ids
        _record_patch_receipt(
            execution,
            patch,
            byte_length=len(patch_text.encode("utf-8")),
        )
        if not patch.ops:
            return GraphUpdateResult(status="none"), None
        workspace = service.history.workspace
        with workspace.run_lock(
            on_wait=(lambda message: _record_work_lock_wait(execution, message, workspace.location))
            if execution is not None
            else None,
            on_lost=(lambda message: _record_work_lock_lost(execution, message, workspace.location))
            if execution is not None
            else None,
            cancelled=(execution.control.pause_requested.is_set if execution is not None else None),
        ) as lease:
            lease.assert_owned()
            if source_operation_id:
                matches = [
                    item
                    for item in service.history.load_patches()
                    if (
                        item.source_effect_id == source_effect_id
                        if source_effect_id is not None
                        else item.source_operation_id == source_operation_id
                    )
                    and item.admission == "accepted"
                ]
                if len(matches) > 1:
                    raise ValueError("One agent effect has multiple canonical Patch commits.")
                if matches:
                    canonical_patch = matches[0]
                    if (
                        canonical_patch.source_operation_id != source_operation_id
                        or canonical_patch.source_effect_sha256 != patch.source_effect_sha256
                        or canonical_patch.kind != "experiment_loop"
                        or canonical_patch.experiment_control_node_id != control_node_id
                    ):
                        raise ValueError(
                            "Experiment-loop invocation source is bound to a different canonical "
                            "Patch."
                        )
                    result = service.history.current_materialization()
                    appended = canonical_patch
                elif not patch.ops:
                    return GraphUpdateResult(status="none"), None
                else:
                    appended, result = service.history.append(
                        patch,
                        discard_on_reject=True,
                    )
            else:
                appended, result = service.history.append(
                    patch,
                    discard_on_reject=True,
                )
    except PatchRejected as exc:
        messages = [item.message for item in exc.report.messages if item.level == "reject"]
        detail = "; ".join(messages) or str(exc) or "The graph rejected the Experiment-loop Patch."
        if execution is not None:
            execution.store.record_agent_task_receipt(
                execution.operation_id,
                "patch_rejected",
                {"messages": [item.model_dump(mode="json") for item in exc.report.messages[:16]]},
                tier="diagnostic",
            )
        return None, _DeliverableFailure(
            detail,
            correctable=True,
            change_summary=change_summary,
            proposal_ids=proposal_ids,
        )
    except (ReplayHalted, StateUnavailable) as exc:
        return None, _DeliverableFailure(
            str(exc),
            correctable=False,
            change_summary=change_summary,
            proposal_ids=proposal_ids,
        )
    except ValueError as exc:
        return None, _DeliverableFailure(
            str(exc),
            correctable=True,
            change_summary=change_summary,
            proposal_ids=proposal_ids,
        )

    if canonical_patch is not None:
        change_summary = tuple(canonical_patch.change_summary)
        proposal_ids = tuple(_work_patch_proposal_ids(canonical_patch))
    report = result.reports[appended.revision]
    _record_patch_applied_receipt(execution, result.state)
    return (
        GraphUpdateResult(
            status="applied",
            applied_revision=appended.revision,
            change_summary=list(change_summary),
            proposal_ids=list(proposal_ids),
            validation_messages=_bounded_graph_messages(*(item.message for item in report.flags)),
        ),
        None,
    )
