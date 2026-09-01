from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import aclosing, suppress
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Literal

from rcp.agents import AgentEvent, AgentLauncher, PromptFactory
from rcp.agents.command_mailbox import (
    CommandTurnIdentity,
    StagedCommandMailbox,
    serve_command_mailbox,
    stage_command_mailbox,
)
from rcp.agents.command_protocol import (
    CommandRequest,
    CommandResponse,
    MessageCommandRequest,
    ValidateCommandRequest,
)
from rcp.agents.prompts import invoked_package_pointers
from rcp.attachments import ChatAttachmentStore
from rcp.background import AgentTaskContinuation, AgentTaskExecution
from rcp.history import ReplayHalted
from rcp.limits import (
    AUTO_RESEARCH_MAIL_MAX_BYTES,
    PATCH_SELF_CHECK_MAX_COUNT,
    PATCH_SELF_CHECK_POLL_SECONDS,
    PATCH_SELF_CHECK_TIMEOUT_SECONDS,
)
from rcp.runs.auto_research_mail import (
    AUTO_RESEARCH_MAIL_HANDOFF_FILE,
    auto_research_mail_delivery,
    parse_auto_research_mail_delivery,
    stage_auto_research_mail_delivery,
)
from rcp.runs.chat import (
    _chat_read_dirs,
    _chat_stage_name,
    _ChatPatchInputs,
    _clear_stale_turn_handoffs,
    _logical_chat_turn_operation_id,
    _prepare_local_artifact_directory,
    _project_write_scope,
    _record_chat_context_receipt,
    _stage_chat_patch_inputs,
    _validated_local_chat_resume_stage,
    _validated_remote_chat_resume_stage,
)
from rcp.runs.patch_validator import PatchValidationBudget, PatchValidationResult
from rcp.runs.shared import (
    _parent_task_contract_path,
    _ProviderOutcome,
    _sse,
    _stage_context_paths,
    _stage_task_contract,
    _stage_task_input,
    _swept_stage_root,
    _task_token,
)
from rcp.runs.tasks.experiment_watcher_maintenance import _process_experiment_watcher_maintenance
from rcp.runs.tasks.result_views import _prepare_result_view_turn, _roll_result_view_retention
from rcp.runs.tasks.work import (
    WorkTurn,
    _capture_retry_deliverable_baseline,
    _close_work_validator_mailbox,
    _ComposedWorkPrompt,
    _finalize_work_turn,
    _launch_and_stream_work_turn,
    _prepare_work_chat_prompt,
    _resolve_work_execution,
    _ResolvedWorkExecution,
    _settle_patch_deliverable,
    _SettledWorkDeliverables,
    _stage_retry_diagnostics,
    _StagedWorkInputs,
    _stream_work_graph_repair,
    _validate_work_patch_live,
    _WorkValidatorMailboxLifecycle,
)
from rcp.service import ProjectService, RunRequest
from rcp.skills.staging import skill_bundle_label, stage_skill_selection
from rcp.storage import (
    AgentCommandInvocationRecord,
    AutoResearchChildWorkRecord,
    AutoResearchMessageRecord,
)
from rcp.transport import RemoteRunStage, RunStageMailbox, StateUnavailable

_CHILD_WORK_HANDOFFS_CLEARED_RECEIPT = "auto_research_child_work_handoffs_cleared"


def _prepare_auto_research_child_work_handoffs(
    execution: AgentTaskExecution,
    *,
    workspace: Path,
    remote_stage: RemoteRunStage | None,
) -> None:
    """Clear a new child turn once while preserving an interrupted turn's outputs."""

    if any(
        receipt.category == _CHILD_WORK_HANDOFFS_CLEARED_RECEIPT
        for receipt in execution.store.agent_task_receipts(execution.operation_id)
    ):
        return
    _clear_stale_turn_handoffs(workspace, remote_stage)
    execution.store.record_agent_task_receipt(
        execution.operation_id,
        _CHILD_WORK_HANDOFFS_CLEARED_RECEIPT,
        {
            "version": 1,
            "files": ["patch.json", "watch.json", "messages.json", "lifecycle.json"],
        },
    )


def _stage_auto_research_child_work_mail(
    execution: AgentTaskExecution,
    route: AutoResearchChildWorkRecord,
    *,
    local_stage: Path | None,
    remote_stage: RemoteRunStage | None,
    continuation: AgentTaskContinuation,
) -> str | None:
    """Stage only the batch durably claimed by this exact ordinary Work turn."""

    mailbox = RunStageMailbox.for_stage(local_stage=local_stage, remote_stage=remote_stage)
    delivery_operation_id = _auto_research_child_mail_allocation_id(
        execution,
        route,
        continuation=continuation,
    )
    if delivery_operation_id is None:
        return None
    claimed = [
        message
        for message in execution.store.auto_research_messages(route.episode_id)
        if message.delivery_operation_id == delivery_operation_id
    ]
    if not claimed:
        raise ValueError(
            "Auto-research child Work message wake has no mail claimed by this allocation."
        )
    delivery = auto_research_mail_delivery(
        episode_id=route.episode_id,
        recipient_task_id=route.worker_id,
        delivery_operation_id=delivery_operation_id,
        messages=claimed,
    )
    if AUTO_RESEARCH_MAIL_HANDOFF_FILE in mailbox.entry_names():
        retained = parse_auto_research_mail_delivery(
            mailbox.read_text(
                AUTO_RESEARCH_MAIL_HANDOFF_FILE,
                max_bytes=AUTO_RESEARCH_MAIL_MAX_BYTES,
            )
        )
        if retained != delivery:
            raise ValueError("Retained child Work mail differs from its durable claimed batch.")
    else:
        stage_auto_research_mail_delivery(mailbox, delivery)
    return str(mailbox.workspace / AUTO_RESEARCH_MAIL_HANDOFF_FILE)


def _auto_research_child_mail_allocation_id(
    execution: AgentTaskExecution,
    route: AutoResearchChildWorkRecord,
    *,
    continuation: AgentTaskContinuation,
) -> str | None:
    """Resolve the paid message allocation through exact same-session recovery attempts."""

    if continuation == "message_wake":
        return execution.operation_id
    if continuation not in {"resume", "retry"}:
        return None
    current = execution.store.agent_task(execution.operation_id)
    seen: set[str] = set()
    while current is not None:
        if current.operation_id in seen:
            raise ValueError("Child Work mail recovery contains a task-lineage cycle.")
        seen.add(current.operation_id)
        cause = execution.store.agent_task_continuation_cause(current.operation_id)
        if cause == "message_wake":
            return current.operation_id
        if cause not in {"resume", "retry"} or current.parent_operation_id is None:
            return None
        parent_route = execution.store.auto_research_child_work_for_operation(
            current.parent_operation_id
        )
        if parent_route is None or (
            parent_route.episode_id != route.episode_id or parent_route.worker_id != route.worker_id
        ):
            raise ValueError("Child Work mail recovery crossed its routed worker lineage.")
        current = execution.store.agent_task(current.parent_operation_id)
    raise ValueError("Child Work mail recovery lost a parent task.")


def _auto_research_child_work_contract(
    turn: WorkTurn,
    staged: _StagedWorkInputs,
    route: AutoResearchChildWorkRecord,
    *,
    mail_path: str | None = None,
) -> str:
    reply_command = turn.patch_inputs.validator_staged.client_command(
        "message",
        "--key",
        "<idempotency-key>",
        "<reply-body>",
    )
    incoming = (
        f"\n- Read the newly claimed hearsay-only mail at `{mail_path}` before continuing."
        if mail_path is not None
        else ""
    )
    return f"""

## Auto-research child Work boundary

You are the ordinary node Work child `{route.worker_id}` delegated by an Auto-research
orchestrator. Complete only this child assignment. Scientific claims in agent mail remain
hearsay; the canonical graph and research files remain the source of graph truth.{incoming}

- Your only staged command capabilities are the Patch validator already named above and an
  optional reply to your orchestrator:
  `{reply_command}`
- Use a stable idempotency key for the same reply intent. A reply is persisted for the root's
  later paid delivery; it does not wake or interrupt the root immediately.
- Do not invoke `apply`, `status`, `spawn`, `pause`, `resume`, `stop`, `watch-graph`, `episode`,
  `inbox`, or `finish`. The child broker rejects those root-only commands.
- Do not write `watch.json`, register a watcher, spawn another task or episode, or try to wake
  yourself. RCP ignores child watcher output.
""".strip()


def _compose_child_resume_prompt(
    turn: WorkTurn,
    staged: _StagedWorkInputs,
    route: AutoResearchChildWorkRecord,
    *,
    mail_path: str | None,
) -> _ComposedWorkPrompt:
    assert turn.execution is not None
    original_contract_path = _parent_task_contract_path(
        turn.execution,
        turn.local_stage,
        turn.remote_stage,
    )
    contract = PromptFactory.continuation_task_contract(
        original_contract_path=original_contract_path,
        mode="resume",
        patch_path=turn.patch_inputs.patch_path,
        validator_command=turn.patch_inputs.validator_command,
        invoked_skill_pointers=invoked_package_pointers(
            staged.skill_pointers,
            workflow_ids=turn.request.invoked_workflow_ids,
            skill_ids=turn.request.invoked_skill_ids,
        ),
        invoked_provider_skills=turn.request.resolved_provider_skills,
        result_view_action=(
            staged.prepared_result_view.action if staged.prepared_result_view is not None else None
        ),
        result_view_path=(
            staged.prepared_result_view.prompt_path
            if staged.prepared_result_view is not None
            else None
        ),
    )
    contract += "\n\n" + _auto_research_child_work_contract(
        turn,
        staged,
        route,
        mail_path=mail_path,
    )
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


def _compose_child_message_wake_prompt(
    turn: WorkTurn,
    staged: _StagedWorkInputs,
    route: AutoResearchChildWorkRecord,
    mail_path: str,
) -> _ComposedWorkPrompt:
    assert turn.execution is not None
    original_contract_path = _parent_task_contract_path(
        turn.execution,
        turn.local_stage,
        turn.remote_stage,
    )
    contract = f"""
Continue the exact Auto-research child Work assignment from `{original_contract_path}` in the
same native provider session. The newly claimed agent mail is staged separately from the task
contract. Read it, continue the bounded assignment, and reply only if useful.

{_auto_research_child_work_contract(turn, staged, route, mail_path=mail_path)}
""".strip()
    contract_path, prompt = _stage_task_contract(
        turn.local_stage,
        turn.remote_stage,
        f"task-{staged.token}-auto-research-child-mail.md",
        contract,
        execution=turn.execution,
        role="auto_research_child_message_wake",
    )
    return _ComposedWorkPrompt(
        contract_path=contract_path,
        prompt=prompt,
        base_contract_path=original_contract_path,
    )


async def _stage_auto_research_child_work_turn(
    service: ProjectService,
    resolved: _ResolvedWorkExecution,
    data_dir: Path,
    execution: AgentTaskExecution,
    route: AutoResearchChildWorkRecord,
) -> tuple[WorkTurn, _StagedWorkInputs, str | None]:
    request = resolved.request
    continuation = execution.continuation
    reusing_checkpoint = execution.reuses_native_checkpoint
    resuming = continuation == "resume"
    if reusing_checkpoint and not request.session_id:
        raise ValueError(
            "The continued Work turn has no native agent session; retry it from a clean attempt "
            "instead."
        )
    local_stage: Path | None = None
    remote_stage: RemoteRunStage | None = None
    patch_inputs: _ChatPatchInputs | None = None
    validator_lifecycle: _WorkValidatorMailboxLifecycle | None = None
    validator_budget = PatchValidationBudget()
    outcome = _ProviderOutcome(session_id=request.session_id)
    try:
        context = service.assemble_chat(request)
        surface = "project_chat" if request.chat_scope == "project" else "node_chat"
        _record_chat_context_receipt(execution, context, surface=surface)
        stage_name = _chat_stage_name(service, request, execution)
        saved_stage = execution.stage_root is not None
        if resolved.execution_host:
            if saved_stage:
                stage_root = _validated_remote_chat_resume_stage(
                    execution,
                    resolved.execution_host,
                    stage_name,
                )
                remote_stage = RemoteRunStage(resolved.execution_host).attach(stage_root)
            else:
                remote_stage = RemoteRunStage(resolved.execution_host).open(stage_name, reuse=True)
            assert remote_stage.root is not None
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
            execution.checkpoint_stage("", str(local_stage))
            workspace = local_stage
        token = _task_token(execution)
        _roll_result_view_retention(request, execution, local_stage, remote_stage)
        patch_inputs = _stage_chat_patch_inputs(
            local_stage,
            remote_stage,
            workspace=workspace,
            stage_name=stage_name,
            task_id=execution.operation_id,
            turn_id=f"{token}:work",
        )
        patch_inputs.validator_staged.cleanup()
        child_staged = stage_command_mailbox(
            local_stage=local_stage,
            remote_stage=remote_stage,
            episode_id=route.episode_id,
            task_id=execution.operation_id,
            turn_id=f"{token}:auto-research-child-work",
            timeout_seconds=PATCH_SELF_CHECK_TIMEOUT_SECONDS,
        )
        patch_inputs = _ChatPatchInputs(
            patch_path=patch_inputs.patch_path,
            watch_path=patch_inputs.watch_path,
            schema_path=patch_inputs.schema_path,
            validator_command=child_staged.client_command("validate", patch_inputs.patch_path),
            validator_mailbox_id=child_staged.credential.mailbox_id,
            validator_staged=child_staged,
        )
        validator_lifecycle = _start_auto_research_child_validator_mailbox(
            service,
            child_staged,
            execution=execution,
            route=route,
            budget=validator_budget,
            run_truth_scope=context.run_truth_scope,
        )
        if not reusing_checkpoint or continuation == "message_wake":
            _prepare_auto_research_child_work_handoffs(
                execution,
                workspace=workspace,
                remote_stage=remote_stage,
            )
        mail_path = _stage_auto_research_child_work_mail(
            execution,
            route,
            local_stage=local_stage,
            remote_stage=remote_stage,
            continuation=continuation,
        )
        artifact_scope_id = (
            _logical_chat_turn_operation_id(execution.store, execution.operation_id)
            if resuming
            else execution.operation_id
        )
        prepared_result_view = _prepare_result_view_turn(
            request,
            execution,
            local_stage,
            remote_stage,
            focused_node=context.node,
            logical_operation_id=artifact_scope_id,
            revision_preflight=resolved.revision_preflight,
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
            write_dirs=[Path(item) for item in write_scope.repository_roots],
            write_scope=write_scope,
            patch_inputs=patch_inputs,
            validator_lifecycle=validator_lifecycle,
            validator_budget=validator_budget,
            outcome=outcome,
        )
        return (
            turn,
            _StagedWorkInputs(
                token=token,
                artifact_scope_id=artifact_scope_id,
                artifact_directory=artifact_directory,
                prepared_result_view=prepared_result_view,
                experiment_resources=[],
                experiment_resource_pointers=[],
                skill_selection=skill_selection,
                skill_pointers=skill_pointers,
                attachment_pointers=attachment_pointers,
                repositories=repositories,
            ),
            mail_path,
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


def _compose_child_fresh_prompt(
    turn: WorkTurn,
    staged: _StagedWorkInputs,
    route: AutoResearchChildWorkRecord,
    *,
    mail_path: str | None,
    retry_diagnostics_path: str | None = None,
) -> _ComposedWorkPrompt:
    assert turn.request.message is not None
    focused_node_id = str(turn.context.node["id"]) if turn.context.node else None
    if not turn.uses_master_protocol:
        human_request_path = _stage_task_input(
            turn.local_stage,
            turn.remote_stage,
            f"task-{staged.token}-human-request.txt",
            turn.request.message,
        )
        contract = PromptFactory.work_task_contract(
            project_name=turn.context.project_name,
            ontology_path=f"{turn.context.graph_path}#ontology",
            ontology_extensions=turn.context.ontology_extensions,
            graph_path=turn.context.graph_path,
            research_path=turn.context.research_md_path,
            focused_node_id=focused_node_id,
            repositories=staged.repositories,
            introduction_path=turn.context.introduction_path,
            human_request_path=human_request_path,
            patch_path=turn.patch_inputs.patch_path,
            artifact_path=str(staged.artifact_directory),
            output_schema_path=turn.patch_inputs.schema_path,
            retry_diagnostics_path=retry_diagnostics_path,
            watch_path=turn.patch_inputs.watch_path,
            execution_host=turn.execution_host,
            experiment_watcher_resources=[],
            validator_command=turn.patch_inputs.validator_command,
            write_scope=turn.write_scope,
            skill_pointers=staged.skill_pointers,
            invoked_skill_pointers=invoked_package_pointers(
                staged.skill_pointers,
                workflow_ids=turn.request.invoked_workflow_ids,
                skill_ids=turn.request.invoked_skill_ids,
            ),
            invoked_provider_skills=turn.request.resolved_provider_skills,
            attachments=staged.attachment_pointers,
        )
        contract_path, prompt = _stage_task_contract(
            turn.local_stage,
            turn.remote_stage,
            f"task-{staged.token}-{'base' if turn.retry_attempt else 'initial'}.md",
            contract,
            execution=turn.execution,
            role="work_retry_base" if turn.retry_attempt else "work",
        )
        return _ComposedWorkPrompt(contract_path, prompt, contract_path)

    master_context = PromptFactory.chat_master_context(
        project_name=turn.context.project_name,
        ontology_path=f"{turn.context.graph_path}#ontology",
        ontology_extensions=turn.context.ontology_extensions,
        graph_path=turn.context.graph_path,
        research_path=turn.context.research_md_path,
        graph_revision=turn.context.graph_revision,
        focused_node_id=focused_node_id,
        focused_node=turn.context.node,
        focused_relations=[item.model_dump(mode="json") for item in turn.context.relations],
        repositories=staged.repositories,
        introduction_path=turn.context.introduction_path,
        patch_path=turn.patch_inputs.patch_path,
        workspace_path=str(turn.workspace),
        output_schema_path=turn.patch_inputs.schema_path,
        validator_command=turn.patch_inputs.validator_command,
        watch_path=turn.patch_inputs.watch_path,
        execution_host=turn.execution_host,
        experiment_watcher_resources=[],
        skill_pointers=staged.skill_pointers,
    )
    master_context += (
        "\n\n"
        + _auto_research_child_work_contract(
            turn,
            staged,
            route,
            mail_path=mail_path,
        )
        + "\n"
    )
    stable_prompt_values: dict[str, object] = {
        "project": {"name": turn.context.project_name},
        "settings": {
            "provider": turn.request.provider,
            "model": turn.request.model,
            "reasoning": turn.request.reasoning,
            "run_on": turn.request.run_on,
        },
        "current": {
            "ontology_path": f"{turn.context.graph_path}#ontology",
            "graph_revision": turn.context.graph_revision,
            "graph_path": turn.context.graph_path,
            "research_path": turn.context.research_md_path,
            "focused_node_id": focused_node_id,
            "introduction_path": turn.context.introduction_path,
            "experiment_watcher_resources": [],
        },
        "repositories": staged.repositories,
        "skills": {"pointers": staged.skill_pointers},
        "patch": {
            "path": turn.patch_inputs.patch_path,
            "watch_path": turn.patch_inputs.watch_path,
            "schema_path": turn.patch_inputs.schema_path,
            "validator_command": turn.patch_inputs.validator_command,
            "validator_mailbox_id": turn.patch_inputs.validator_mailbox_id,
        },
        "workspace": {"path": str(turn.workspace)},
        "auto_research_child": {
            "episode_id": route.episode_id,
            "worker_id": route.worker_id,
            "control_node_id": route.control_node_id,
            "allowed_staged_commands": ["validate", "message"],
        },
    }
    prompt, retained_master_path = _prepare_work_chat_prompt(
        turn.execution,
        turn.request,
        local_stage=turn.local_stage,
        remote_stage=turn.remote_stage,
        artifact_path=str(staged.artifact_directory),
        master_context=master_context,
        stable_values=stable_prompt_values,
        skill_pointers=staged.skill_pointers,
        attachment_pointers=staged.attachment_pointers,
        result_view=staged.prepared_result_view,
        write_scope=turn.write_scope,
    )
    return _ComposedWorkPrompt(retained_master_path, prompt, retained_master_path)


def _compose_child_retry_prompt(
    turn: WorkTurn,
    staged: _StagedWorkInputs,
    route: AutoResearchChildWorkRecord,
    *,
    mail_path: str | None,
) -> _ComposedWorkPrompt:
    assert turn.execution is not None
    retry_diagnostics_path = _stage_retry_diagnostics(turn, staged)
    resumed_retry = turn.retrying and turn.reusing_checkpoint
    explicit_contract = not turn.uses_master_protocol and not resumed_retry
    current: _ComposedWorkPrompt | None = None
    if explicit_contract:
        current = _compose_child_fresh_prompt(
            turn,
            staged,
            route,
            mail_path=mail_path,
            retry_diagnostics_path=retry_diagnostics_path,
        )
    result_view_handoff = bool(
        turn.continuation == "handoff" and staged.prepared_result_view is not None
    )
    if result_view_handoff:
        if current is None:
            raise ValueError("The result view create handoff lost its current Work contract.")
        original_contract_path = current.contract_path
        continuation_contract_path = None
    else:
        original_contract_path = _parent_task_contract_path(
            turn.execution,
            turn.local_stage,
            turn.remote_stage,
        )
        continuation_contract_path = current.contract_path if current is not None else None
    base_contract_path = (
        original_contract_path if resumed_retry or current is None else current.base_contract_path
    )
    retry_contract = PromptFactory.continuation_task_contract(
        original_contract_path=original_contract_path,
        current_contract_path=continuation_contract_path,
        diagnostics_path=retry_diagnostics_path,
        patch_path=turn.patch_inputs.patch_path,
        watch_path=turn.patch_inputs.watch_path,
        mode="retry",
        validator_command=turn.patch_inputs.validator_command,
        output_schema_path=turn.patch_inputs.schema_path if resumed_retry else None,
        skill_pointers=staged.skill_pointers if resumed_retry else None,
        invoked_skill_pointers=invoked_package_pointers(
            staged.skill_pointers,
            workflow_ids=turn.request.invoked_workflow_ids,
            skill_ids=turn.request.invoked_skill_ids,
        ),
        invoked_provider_skills=turn.request.resolved_provider_skills,
        result_view_action=(
            staged.prepared_result_view.action if staged.prepared_result_view is not None else None
        ),
        result_view_path=(
            staged.prepared_result_view.prompt_path
            if staged.prepared_result_view is not None
            else None
        ),
    )
    retry_contract += "\n\n" + _auto_research_child_work_contract(
        turn,
        staged,
        route,
        mail_path=mail_path,
    )
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
        base_contract_path=base_contract_path,
    )


def _compose_child_prompt(
    turn: WorkTurn,
    staged: _StagedWorkInputs,
    route: AutoResearchChildWorkRecord,
    *,
    mail_path: str | None,
) -> _ComposedWorkPrompt:
    if turn.resuming:
        return _compose_child_resume_prompt(
            turn,
            staged,
            route,
            mail_path=mail_path,
        )
    if turn.continuation == "message_wake":
        if mail_path is None:
            raise ValueError("Child Work message wake is missing its routed mail handoff.")
        return _compose_child_message_wake_prompt(turn, staged, route, mail_path)
    if turn.retrying or turn.continuation == "handoff":
        return _compose_child_retry_prompt(
            turn,
            staged,
            route,
            mail_path=mail_path,
        )
    return _compose_child_fresh_prompt(
        turn,
        staged,
        route,
        mail_path=mail_path,
        retry_diagnostics_path=_stage_retry_diagnostics(turn, staged),
    )


def _start_auto_research_child_validator_mailbox(
    service: ProjectService,
    staged: StagedCommandMailbox,
    *,
    execution: AgentTaskExecution,
    route: AutoResearchChildWorkRecord,
    budget: PatchValidationBudget,
    run_truth_scope: list[str],
) -> _WorkValidatorMailboxLifecycle:
    stop = asyncio.Event()
    try:
        task = asyncio.create_task(
            _serve_auto_research_child_work_mailbox(
                service,
                staged=staged,
                execution=execution,
                route=route,
                stop=stop,
                budget=budget,
                run_truth_scope=run_truth_scope,
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


async def stream_auto_research_child_work_run(
    service: ProjectService,
    launcher: AgentLauncher,
    request: RunRequest,
    data_dir: Path,
    execution: AgentTaskExecution,
    *,
    route: AutoResearchChildWorkRecord,
) -> AsyncIterator[str]:
    """Run one durably admitted Auto-research child Work task."""

    if execution.continuation == "graph_repair":
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

    turn: WorkTurn | None = None
    try:
        resolved = _resolve_work_execution(service, request, execution)
        turn, staged, mail_path = await _stage_auto_research_child_work_turn(
            service,
            resolved,
            data_dir,
            execution,
            route,
        )
        composed = _compose_child_prompt(turn, staged, route, mail_path=mail_path)
        retry_baseline = _capture_retry_deliverable_baseline(turn)
    except BaseException as exc:
        if turn is not None:
            await turn.validator_lifecycle.close(primary_error=exc)
        if isinstance(exc, (OSError, ReplayHalted, StateUnavailable, ValueError)):
            yield _sse(AgentEvent(event="error", text=str(exc)))
            return
        raise

    assert turn is not None
    required_session_id = turn.request.session_id if execution.reuses_native_checkpoint else None
    async with aclosing(
        _launch_and_stream_work_turn(
            turn,
            launcher,
            composed.prompt,
            composed.contract_path,
            staged,
            None,
            required_session_id=required_session_id,
        )
    ) as stream:
        async for frame in stream:
            yield frame
    if turn.answer is None:
        return

    settled = _SettledWorkDeliverables(native_session_id=turn.outcome.session_id)
    async with aclosing(
        _settle_patch_deliverable(
            turn,
            launcher,
            staged,
            composed,
            retry_baseline.patch_digest,
            settled,
            required_session_id=required_session_id,
        )
    ) as stream:
        async for frame in stream:
            yield frame
    if settled.stop:
        return

    (
        maintenance_frames,
        native_session_id,
        maintenance_paused,
    ) = await _process_experiment_watcher_maintenance(
        service=turn.service,
        launcher=launcher,
        request=turn.request,
        execution=turn.execution,
        staged_resources=[],
        workspace=turn.workspace,
        remote_stage=turn.remote_stage,
        local_stage=turn.local_stage,
        base_contract_path=composed.base_contract_path,
        token=staged.token,
        native_session_id=settled.native_session_id,
        read_dirs=turn.read_dirs,
        write_dirs=turn.write_dirs,
        write_scope=turn.write_scope,
        execution_host=turn.execution_host,
        provider_binary=turn.provider_binary,
        retry_output_digests=retry_baseline.experiment_watch_digests,
    )
    settled.native_session_id = native_session_id
    for frame in maintenance_frames:
        yield frame
    if maintenance_paused:
        return
    mailbox = RunStageMailbox.for_stage(
        local_stage=turn.local_stage, remote_stage=turn.remote_stage
    )
    mailbox.remove("watch.json")
    execution.store.record_agent_task_receipt(
        execution.operation_id,
        "auto_research_child_watcher_output_discarded",
        {
            "watcher_authority": "none",
            "reason": "ordinary Auto-research child Work cannot arm a watcher",
        },
        tier="diagnostic",
    )
    final_turn = turn
    if turn.continuation == "message_wake" and turn.request.message is None:
        final_turn = replace(
            turn,
            request=turn.request.model_copy(
                update={
                    "message": (
                        "RCP delivered a claimed Auto-research mail batch in the separate "
                        "messages.json handoff."
                    )
                }
            ),
        )
    for frame in _finalize_work_turn(final_turn, turn.answer, settled.graph_update):
        yield frame


def _child_reply_message_id(episode_id: str, idempotency_key: str) -> str:
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rcp:auto_research:{episode_id}:message:{idempotency_key}",
        )
    )


def _unused_child_command_id(execution: AgentTaskExecution, preferred: str) -> str:
    if execution.store.agent_command(preferred) is None:
        return preferred
    while True:
        candidate = uuid.uuid4().hex
        if execution.store.agent_command(candidate) is None:
            return candidate


def _child_reply_result(
    message: AutoResearchMessageRecord,
    *,
    disposition: Literal["created", "existing"],
) -> dict[str, object]:
    return {
        "message_id": message.message_id,
        "recipient_task_id": message.recipient_task_id,
        "delivery_operation_id": message.delivery_operation_id,
        "delivery": "started" if message.delivery_operation_id is not None else "pending",
        "graph_authority": "none",
        "epistemic_status": "hearsay",
        "disposition": disposition,
    }


def _finish_child_reply_command(
    execution: AgentTaskExecution,
    *,
    command_id: str,
    request_id: str,
    status: Literal["ok", "invalid", "unavailable"],
    message: str,
    result: dict[str, object] | None = None,
) -> CommandResponse:
    payload: dict[str, object] = {"result": result or {}, "diagnostic": message}
    try:
        stored = execution.store.finish_agent_command(
            command_id,
            status=status,
            payload=payload,
            message=message,
        )
    except ValueError:
        stored = execution.store.agent_command(command_id)
        if stored is None or stored.exited_at is None:
            raise
        return _recorded_child_reply_response(stored, request_id=request_id)
    return CommandResponse(
        request_id=request_id,
        status=stored.status or status,
        message=message,
        result=result or {},
    )


def _recorded_child_reply_response(
    invocation: AgentCommandInvocationRecord,
    *,
    request_id: str,
) -> CommandResponse:
    if invocation.status not in {"ok", "invalid", "unavailable"} or not isinstance(
        invocation.exit_payload, dict
    ):
        raise ValueError("Recorded child Work reply command exit is incomplete.")
    recorded_result = invocation.exit_payload.get("result")
    result = dict(recorded_result) if isinstance(recorded_result, dict) else {}
    diagnostic = invocation.exit_payload.get("diagnostic")
    message = diagnostic if isinstance(diagnostic, str) else None
    if invocation.status != "ok" and not message:
        message = "Recorded child Work reply command did not complete successfully."
    return CommandResponse(
        request_id=request_id,
        status=invocation.status,
        message=message,
        result=result,
    )


def _finish_child_reply_with_retry_attempt(
    execution: AgentTaskExecution,
    *,
    command_id: str,
    retry_command_id: str | None,
    request_id: str,
    status: Literal["ok", "invalid", "unavailable"],
    message: str,
    result: dict[str, object] | None = None,
) -> CommandResponse:
    response = _finish_child_reply_command(
        execution,
        command_id=command_id,
        request_id=request_id,
        status=status,
        message=message,
        result=result,
    )
    if retry_command_id is None:
        return response
    return _finish_child_reply_command(
        execution,
        command_id=retry_command_id,
        request_id=request_id,
        status=response.status,
        message=response.message or message,
        result=response.result,
    )


def _child_reply_matches(
    execution: AgentTaskExecution,
    route: AutoResearchChildWorkRecord,
    request: MessageCommandRequest,
    saved: AutoResearchMessageRecord,
) -> bool:
    episode = execution.store.episode(route.episode_id)
    sender_route = (
        execution.store.auto_research_child_work_for_operation(saved.sender_task_id)
        if saved.sender_task_id is not None
        else None
    )
    return bool(
        episode is not None
        and episode.root_operation_id is not None
        and sender_route is not None
        and sender_route.worker_id == route.worker_id
        and saved.episode_id == route.episode_id
        and saved.sender_role == "worker"
        and saved.authorized_by is None
        and saved.recipient_task_id == episode.root_operation_id
        and saved.control_node_id == route.control_node_id
        and saved.body == request.arguments.body
    )


def _dispatch_auto_research_child_reply(
    execution: AgentTaskExecution,
    route: AutoResearchChildWorkRecord,
    request: MessageCommandRequest,
) -> CommandResponse:
    """Persist one child-to-root reply without starting a concurrent root turn."""

    episode = execution.store.episode(route.episode_id)
    if episode is None or episode.mode != "auto_research" or episode.root_operation_id is None:
        return CommandResponse(
            request_id=request.request_id,
            status="unavailable",
            message="The Auto-research orchestrator recipient is unavailable.",
        )
    if request.arguments.recipient_task_id not in {None, episode.root_operation_id}:
        return CommandResponse(
            request_id=request.request_id,
            status="invalid",
            message="This child Work task may reply only to its Auto-research orchestrator.",
        )
    if request.idempotency_key is None:
        return CommandResponse(
            request_id=request.request_id,
            status="invalid",
            message="A child Work reply requires an idempotency key.",
        )
    start_payload = {
        "request_id": request.request_id,
        "arguments": request.arguments.model_dump(mode="json"),
        "planned_message_id": _child_reply_message_id(
            route.episode_id,
            request.idempotency_key,
        ),
    }
    prior = execution.store.agent_command_by_key(route.episode_id, request.idempotency_key)
    command_id = _unused_child_command_id(execution, request.request_id)
    retry_command_id: str | None = None
    if prior is None:
        try:
            invocation = execution.store.start_agent_command(
                operation_id=execution.operation_id,
                command_id=command_id,
                episode_id=route.episode_id,
                verb="message",
                idempotency_key=request.idempotency_key,
                payload=start_payload,
            )
        except ValueError:
            raced = execution.store.agent_command_by_key(
                route.episode_id,
                request.idempotency_key,
            )
            if raced is None:
                raise
            invocation = raced
        if invocation.command_id != command_id:
            prior = invocation
    if prior is not None:
        attempt = execution.store.start_agent_command(
            operation_id=execution.operation_id,
            command_id=command_id,
            episode_id=route.episode_id,
            verb="message",
            idempotency_key=None,
            payload={
                **start_payload,
                "idempotency_key": request.idempotency_key,
                "deduplicates_command_id": prior.command_id,
            },
        )
        prior_route = execution.store.auto_research_child_work_for_operation(prior.operation_id)
        if (
            prior.verb != "message"
            or prior.start_payload.get("arguments") != start_payload["arguments"]
            or prior.start_payload.get("planned_message_id") != start_payload["planned_message_id"]
            or prior_route is None
            or prior_route.worker_id != route.worker_id
        ):
            return _finish_child_reply_command(
                execution,
                command_id=attempt.command_id,
                request_id=request.request_id,
                status="invalid",
                message=(
                    "This idempotency key was already used by another actor or with different "
                    "reply arguments."
                ),
            )
        if prior.exited_at is not None:
            recorded = _recorded_child_reply_response(prior, request_id=request.request_id)
            return _finish_child_reply_command(
                execution,
                command_id=attempt.command_id,
                request_id=request.request_id,
                status=recorded.status,
                message=recorded.message or "The existing child Work reply was returned.",
                result=recorded.result,
            )
        invocation = prior
        retry_command_id = attempt.command_id

    planned_message_id = str(start_payload["planned_message_id"])
    saved = execution.store.auto_research_message(planned_message_id)
    disposition: Literal["created", "existing"] = "existing"
    if saved is None:
        saved = execution.store.record_auto_research_message(
            AutoResearchMessageRecord(
                message_id=planned_message_id,
                episode_id=route.episode_id,
                sender_role="worker",
                sender_task_id=execution.operation_id,
                recipient_task_id=episode.root_operation_id,
                control_node_id=route.control_node_id,
                body=request.arguments.body,
                created_at=execution.store.now(),
            )
        )
        disposition = "created"
    if not _child_reply_matches(execution, route, request, saved):
        return _finish_child_reply_with_retry_attempt(
            execution,
            command_id=invocation.command_id,
            retry_command_id=retry_command_id,
            request_id=request.request_id,
            status="unavailable",
            message="The durable child Work reply does not match this command intent.",
        )
    return _finish_child_reply_with_retry_attempt(
        execution,
        command_id=invocation.command_id,
        retry_command_id=retry_command_id,
        request_id=request.request_id,
        status="ok",
        message=(
            "Reply persisted for the Auto-research orchestrator's paid delivery."
            if saved.delivery_operation_id is not None
            else "Reply persisted for the Auto-research orchestrator's next paid delivery."
        ),
        result=_child_reply_result(saved, disposition=disposition),
    )


async def _serve_auto_research_child_work_mailbox(
    service: ProjectService,
    *,
    staged: StagedCommandMailbox,
    execution: AgentTaskExecution,
    route: AutoResearchChildWorkRecord,
    stop: asyncio.Event,
    budget: PatchValidationBudget,
    run_truth_scope: list[str],
) -> None:
    async def handle(
        request: CommandRequest,
        identity: CommandTurnIdentity,
    ) -> CommandResponse:
        if identity.episode_id != route.episode_id or identity.task_id != execution.operation_id:
            return CommandResponse(
                request_id=request.request_id,
                status="invalid",
                message="This child Work command credential is bound to another turn.",
            )
        if isinstance(request, ValidateCommandRequest):
            budget.count += 1
            if budget.count > PATCH_SELF_CHECK_MAX_COUNT:
                result = PatchValidationResult(
                    status="unavailable",
                    messages=["This task has reached its bounded RCP validator self-check limit."],
                )
            else:
                result = await asyncio.to_thread(
                    _validate_work_patch_live,
                    service,
                    request.arguments.patch,
                    run_truth_scope=run_truth_scope,
                    source_operation_id=execution.operation_id,
                )
            execution.store.record_agent_task_event(
                execution.operation_id,
                f"Patch self-check {budget.count}/{PATCH_SELF_CHECK_MAX_COUNT}: {result.status}.",
                level="info" if result.status == "valid" else "warning",
            )
            execution.store.record_agent_task_receipt(
                execution.operation_id,
                "patch_self_check",
                {
                    "count": budget.count,
                    "limit": PATCH_SELF_CHECK_MAX_COUNT,
                    **result.model_dump(mode="json"),
                },
                tier="diagnostic",
            )
            status = {
                "valid": "ok",
                "invalid": "invalid",
                "unavailable": "unavailable",
            }[result.status]
            diagnostic = " ".join(message.strip() for message in result.messages if message.strip())
            return CommandResponse(
                request_id=request.request_id,
                status=status,
                message=(diagnostic[:2_000] or None)
                if status == "ok"
                else (diagnostic[:2_000] or f"Patch validation was {result.status}."),
                result=result.model_dump(mode="json"),
            )
        if isinstance(request, MessageCommandRequest):
            return _dispatch_auto_research_child_reply(execution, route, request)
        return CommandResponse(
            request_id=request.request_id,
            status="invalid",
            message=(
                "This child Work credential authorizes only Patch validation and a reply to its "
                "Auto-research orchestrator."
            ),
        )

    try:
        await serve_command_mailbox(
            staged=staged,
            handler=handle,
            stop=stop,
            poll_seconds=PATCH_SELF_CHECK_POLL_SECONDS,
            invocation_gate=staged.invocation_gate,
        )
    except (OSError, StateUnavailable, ValueError) as exc:
        execution.store.record_agent_task_event(
            execution.operation_id,
            f"Child Work command broker became unavailable: {' '.join(str(exc).split())[:400]}",
            level="warning",
        )
