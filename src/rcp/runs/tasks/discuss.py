from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import aclosing, suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath

from rcp.agents import AgentEvent, AgentLauncher, PromptFactory
from rcp.agents.prompts import CHAT_MASTER_CONTEXT_VERSION, invoked_package_pointers
from rcp.attachments import ChatAttachmentStore
from rcp.background import AgentTaskExecution
from rcp.config import AgentSurface
from rcp.history import ReplayHalted
from rcp.limits import RUN_STAGE_RETENTION_DAYS
from rcp.runs.chat import (
    _append_chat_exchange,
    _chat_read_dirs,
    _chat_stage_name,
    _clear_stale_turn_handoffs,
    _commit_chat_prompt_state,
    _discover_chat_artifacts,
    _logical_chat_turn_operation_id,
    _prepare_chat_prompt_state,
    _prepare_local_artifact_directory,
    _read_chat_patch,
    _record_artifact_discovery_receipt,
    _record_chat_context_receipt,
    _retained_chat_patch_values,
    _stage_chat_patch_inputs,
    _validated_local_chat_resume_stage,
    _validated_remote_chat_resume_stage,
    stage_artifact_context,
)
from rcp.runs.experiment_loop import stage_chat_experiment_watcher_resources
from rcp.runs.patch_validator import cleanup_patch_validation_mailbox
from rcp.runs.shared import (
    _parent_task_contract_path,
    _pinned_to_profile,
    _ProviderOutcome,
    _record_agent_launch_receipt,
    _sse,
    _stage_context_paths,
    _stage_json_task_input,
    _stage_task_contract,
    _stage_task_input,
    _stream_agent_events,
    _swept_stage_root,
    _task_token,
)
from rcp.runs.tasks.result_views import touch_conversation_stage, touch_saved_conversation_stages
from rcp.service import ProjectService, RunRequest
from rcp.skills.staging import skill_bundle_label, stage_skill_selection
from rcp.transport import RemoteRunStage, StateUnavailable


def _refresh_result_view_retention(
    execution: AgentTaskExecution | None,
    request: RunRequest,
    *,
    local_stage: Path | None,
    remote_stage: RemoteRunStage | None,
) -> None:
    """Roll one conversation workspace and its unkept views forward together."""

    current_binding = touch_conversation_stage(local_stage, remote_stage)
    if execution is None or not request.chat_id:
        return
    task = execution.store.agent_task(execution.operation_id)
    if task is None:
        return
    try:
        now = datetime.fromisoformat(execution.store.now()).astimezone(UTC)
        views = execution.store.list_result_views(
            task.project_id,
            chat_id=request.chat_id,
            as_of=now,
        )
        touch_saved_conversation_stages(
            ((view.stage_host, view.stage_root) for view in views if view.kept_filename is None),
            current_binding=current_binding,
        )
        expires_at = (now + timedelta(days=RUN_STAGE_RETENTION_DAYS)).isoformat()
        execution.store.refresh_result_view_expiry(
            task.project_id,
            request.chat_id,
            expires_at=expires_at,
            as_of=now,
        )
    except Exception as exc:
        with suppress(Exception):
            execution.store.record_agent_task_event(
                execution.operation_id,
                f"Result-view retention could not be refreshed: {exc}",
                level="warning",
            )


def _prepare_discuss_chat_prompt(
    execution: AgentTaskExecution | None,
    request: RunRequest,
    *,
    local_stage: Path | None,
    remote_stage: RemoteRunStage | None,
    artifact_path: str,
    master_context: str,
    stable_values: dict[str, object],
    skill_pointers: list[dict[str, object]],
    attachment_pointers: list[dict[str, object]],
) -> tuple[str, str]:
    """Prepare the session baseline behind one Discuss-local seam."""

    if request.message is None:
        raise ValueError("An ordinary Discuss turn requires a human message.")
    bootstrap_path, context_delta, retained_master_path = _prepare_chat_prompt_state(
        execution,
        request,
        local_stage=local_stage,
        remote_stage=remote_stage,
        master_context=master_context,
        contract_key=f"chat-master-v{CHAT_MASTER_CONTEXT_VERSION}",
        values=stable_values,
    )
    prompt = PromptFactory.discuss_turn_prompt(
        artifact_path=artifact_path,
        human_message=request.message,
        master_context_path=bootstrap_path,
        context_delta=context_delta,
        invoked_skill_pointers=invoked_package_pointers(
            skill_pointers,
            workflow_ids=request.invoked_workflow_ids,
            skill_ids=request.invoked_skill_ids,
        ),
        invoked_provider_skills=request.resolved_provider_skills,
        attachments=attachment_pointers,
    )
    return prompt, retained_master_path


async def stream_discuss_run(
    service: ProjectService,
    launcher: AgentLauncher,
    request: RunRequest,
    data_dir: Path,
    execution: AgentTaskExecution | None = None,
) -> AsyncIterator[str]:
    """Run one Discuss turn over graph, node, request, and repository context."""
    continuation = execution.continuation if execution is not None else "fresh"
    reusing_checkpoint = bool(execution is not None and execution.reuses_native_checkpoint)
    resuming = continuation == "resume"
    retrying = continuation == "retry"
    retry_attempt = continuation in {"retry", "handoff"}
    surface: AgentSurface = "project_chat" if request.chat_scope == "project" else "node_chat"
    try:
        profile = service.resolve_agent_profile(
            surface,
            provider=request.provider,
            model=request.model,
            reasoning=request.reasoning,
            run_on=request.run_on,
        )
    except ValueError as exc:
        yield _sse(AgentEvent(event="error", text=str(exc)))
        return
    request = _pinned_to_profile(request, profile)
    local_stage: Path | None = None
    execution_machine = service.manifest.machine_map[profile.run_on]
    execution_host = execution_machine.host
    provider_binary = execution_machine.provider_paths.get(profile.provider)
    remote_stage: RemoteRunStage | None = None
    artifact_scope_id: str | None = None
    artifact_directory: Path | PurePosixPath | None = None
    patch_inputs = None
    outcome = _ProviderOutcome(session_id=request.session_id)
    try:
        try:
            context = service.assemble_chat(request)
            _record_chat_context_receipt(execution, context, surface=surface)
            # One scratch folder per conversation, not per turn. Resuming a native
            # session means resuming it in the directory it was given — Claude keys
            # its sessions by that directory — so every turn of a chat, local or
            # remote, reuses the same folder and _sweep_stale_stages ages it out.
            stage_name = _chat_stage_name(service, request, execution)
            saved_stage = execution is not None and execution.stage_root is not None
            if execution_host:
                if saved_stage:
                    stage_root = _validated_remote_chat_resume_stage(
                        execution, execution_host, stage_name
                    )
                    remote_stage = RemoteRunStage(execution_host).attach(stage_root)
                else:
                    remote_stage = RemoteRunStage(execution_host).open(stage_name, reuse=True)
                assert remote_stage.root is not None
                if execution is not None:
                    execution.checkpoint_stage(execution_host, str(remote_stage.root))
                if not reusing_checkpoint or retrying:
                    context = context.model_copy(
                        update=_stage_context_paths(
                            context, service, remote_stage, execution_machine.alias
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
            _refresh_result_view_retention(
                execution,
                request,
                local_stage=local_stage,
                remote_stage=remote_stage,
            )
            if not reusing_checkpoint:
                # A reused folder must not hand this turn any previous turn output.
                _clear_stale_turn_handoffs(workspace, remote_stage)
            artifact_scope_id = (
                _logical_chat_turn_operation_id(execution.store, execution.operation_id)
                if execution is not None and resuming
                else execution.operation_id
                if execution is not None
                else str(uuid.uuid4())
            )
            if remote_stage is not None:
                artifact_directory = remote_stage.prepare_artifact_directory(
                    artifact_scope_id, reuse=resuming
                )
            else:
                assert local_stage is not None
                artifact_directory = _prepare_local_artifact_directory(
                    local_stage, artifact_scope_id, reuse=resuming
                )

            token = _task_token(execution)
            experiment_resources = await stage_chat_experiment_watcher_resources(
                request,
                execution,
                local_stage,
                remote_stage,
                workspace=workspace,
                token=token,
                clear_stale=not reusing_checkpoint,
            )
            experiment_resource_pointers = [item.prompt_value() for item in experiment_resources]
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
            artifact_context_pointer = stage_artifact_context(
                service,
                request,
                execution,
                local_stage=local_stage,
                remote_stage=remote_stage,
                artifact_path=str(artifact_directory),
            )
            if artifact_context_pointer is not None:
                attachment_pointers.append(artifact_context_pointer)
            read_dirs = _chat_read_dirs(
                context,
                remote_stage,
                service,
                execution_machine.alias,
            )
            read_dirs.extend(
                path
                for path in dict.fromkeys(
                    Path(str(item["path"])).parent for item in attachment_pointers
                )
                if path not in read_dirs
            )
            if reusing_checkpoint and not request.session_id:
                raise ValueError(
                    "The continued chat has no native agent session; retry it from a clean "
                    "attempt instead."
                )
            if resuming:
                assert execution is not None
                original_contract_path = _parent_task_contract_path(
                    execution, local_stage, remote_stage
                )
                contract = PromptFactory.continuation_task_contract(
                    original_contract_path=original_contract_path,
                    mode="resume",
                    patch_path=None,
                    invoked_skill_pointers=invoked_package_pointers(
                        skill_pointers,
                        workflow_ids=request.invoked_workflow_ids,
                        skill_ids=request.invoked_skill_ids,
                    ),
                    invoked_provider_skills=request.resolved_provider_skills,
                )
                contract_path, prompt = _stage_task_contract(
                    local_stage,
                    remote_stage,
                    f"task-{token}-resume.md",
                    contract,
                    execution=execution,
                    role="discuss_resume",
                )
            elif retry_attempt:
                assert request.message is not None
                retry_diagnostics_path = (
                    _stage_json_task_input(
                        local_stage,
                        remote_stage,
                        f"task-{token}-retry-diagnostics.json",
                        {"prior_attempt_diagnostics": list(execution.retry_feedback)},
                    )
                    if execution is not None and (execution.retry_feedback or retry_attempt)
                    else None
                )
                # A retry that still holds its native session already has the contract in the
                # conversation; it gets a follow-up naming what changed, not a rebuilt contract.
                resumed_retry = retrying and reusing_checkpoint
                current_contract_path = None
                current_prompt = None
                if not resumed_retry:
                    human_request_path = _stage_task_input(
                        local_stage,
                        remote_stage,
                        f"task-{token}-human-request.txt",
                        request.message,
                    )
                    contract = PromptFactory.discuss_task_contract(
                        project_name=context.project_name,
                        ontology_path=f"{context.graph_path}#ontology",
                        ontology_extensions=context.ontology_extensions,
                        graph_path=context.graph_path,
                        research_path=context.research_md_path,
                        focused_node_id=str(context.node["id"]) if context.node else None,
                        repositories=[
                            {"alias": item.alias, "host": item.host, "path": item.path}
                            for item in context.repositories
                        ],
                        introduction_path=context.introduction_path,
                        human_request_path=human_request_path,
                        artifact_path=str(artifact_directory),
                        retry_diagnostics_path=retry_diagnostics_path,
                        experiment_watcher_resources=experiment_resource_pointers,
                        skill_pointers=skill_pointers,
                        invoked_skill_pointers=invoked_package_pointers(
                            skill_pointers,
                            workflow_ids=request.invoked_workflow_ids,
                            skill_ids=request.invoked_skill_ids,
                        ),
                        invoked_provider_skills=request.resolved_provider_skills,
                        attachments=attachment_pointers,
                    )
                    current_contract_path, current_prompt = _stage_task_contract(
                        local_stage,
                        remote_stage,
                        f"task-{token}-{'base' if retry_attempt else 'initial'}.md",
                        contract,
                        execution=execution,
                        role="discuss_retry_base" if retry_attempt else "discuss",
                    )
                if retrying:
                    assert execution is not None
                    assert retry_diagnostics_path is not None
                    original_contract_path = _parent_task_contract_path(
                        execution, local_stage, remote_stage
                    )
                    retry_contract = PromptFactory.continuation_task_contract(
                        original_contract_path=original_contract_path,
                        current_contract_path=current_contract_path,
                        diagnostics_path=retry_diagnostics_path,
                        mode="retry",
                        skill_pointers=skill_pointers if resumed_retry else None,
                        invoked_skill_pointers=invoked_package_pointers(
                            skill_pointers,
                            workflow_ids=request.invoked_workflow_ids,
                            skill_ids=request.invoked_skill_ids,
                        ),
                        invoked_provider_skills=request.resolved_provider_skills,
                    )
                    contract_path, prompt = _stage_task_contract(
                        local_stage,
                        remote_stage,
                        f"task-{token}-retry.md",
                        retry_contract,
                        execution=execution,
                        role="discuss_retry",
                    )
                else:
                    contract_path, prompt = current_contract_path, current_prompt
            else:
                assert request.message is not None
                assert artifact_scope_id is not None
                patch_values = _retained_chat_patch_values(execution, request)
                if patch_values is None:
                    patch_inputs = _stage_chat_patch_inputs(
                        local_stage,
                        remote_stage,
                        workspace=workspace,
                        stage_name=stage_name,
                        task_id=execution.operation_id if execution is not None else token,
                        turn_id=f"{token}:discuss",
                    )
                    patch_values = {
                        "path": patch_inputs.patch_path,
                        "watch_path": patch_inputs.watch_path,
                        "schema_path": patch_inputs.schema_path,
                        "validator_command": patch_inputs.validator_command,
                        "validator_mailbox_id": patch_inputs.validator_mailbox_id,
                    }
                repositories = [
                    {"alias": item.alias, "host": item.host, "path": item.path}
                    for item in context.repositories
                ]
                focused_node_id = str(context.node["id"]) if context.node else None
                stable_prompt_values: dict[str, object] = {
                    "project": {"name": context.project_name},
                    "settings": {
                        "provider": request.provider,
                        "model": request.model,
                        "reasoning": request.reasoning,
                        "run_on": request.run_on,
                    },
                    "current": {
                        "ontology_path": f"{context.graph_path}#ontology",
                        "graph_revision": context.graph_revision,
                        "graph_path": context.graph_path,
                        "research_path": context.research_md_path,
                        "focused_node_id": focused_node_id,
                        "introduction_path": context.introduction_path,
                    },
                    "repositories": repositories,
                    "skills": {"pointers": skill_pointers},
                    "patch": patch_values,
                    "workspace": {"path": str(workspace)},
                }
                master_context = PromptFactory.chat_master_context(
                    project_name=context.project_name,
                    ontology_path=f"{context.graph_path}#ontology",
                    ontology_extensions=context.ontology_extensions,
                    graph_path=context.graph_path,
                    research_path=context.research_md_path,
                    graph_revision=context.graph_revision,
                    focused_node_id=focused_node_id,
                    focused_node=context.node,
                    focused_relations=[item.model_dump(mode="json") for item in context.relations],
                    repositories=repositories,
                    introduction_path=context.introduction_path,
                    patch_path=patch_values["path"],
                    workspace_path=str(workspace),
                    output_schema_path=patch_values["schema_path"],
                    validator_command=patch_values["validator_command"],
                    watch_path=patch_values["watch_path"],
                    execution_host=execution_host,
                    experiment_watcher_resources=experiment_resource_pointers,
                    skill_pointers=skill_pointers,
                )
                prompt, retained_master_path = _prepare_discuss_chat_prompt(
                    execution,
                    request,
                    local_stage=local_stage,
                    remote_stage=remote_stage,
                    artifact_path=str(artifact_directory),
                    master_context=master_context,
                    stable_values=stable_prompt_values,
                    skill_pointers=skill_pointers,
                    attachment_pointers=attachment_pointers,
                )
                contract_path = retained_master_path
        except (OSError, ReplayHalted, StateUnavailable, ValueError) as exc:
            yield _sse(AgentEvent(event="error", text=str(exc)))
            return

        _record_agent_launch_receipt(
            execution,
            request,
            prompt=prompt,
            contract_path=contract_path,
            remote=bool(execution_host),
            resumed=reusing_checkpoint,
            continuation=continuation,
            extra={
                "surface": surface,
                "mode": "discuss",
                "capability": "discuss",
                "network_access": True,
                "launch_kind": "retry" if retry_attempt else "resume" if resuming else "initial",
                "write_directory_count": 0,
            },
        )
        try:
            async with aclosing(
                _stream_agent_events(
                    launcher,
                    request,
                    prompt,
                    workspace=workspace,
                    session_id=request.session_id,
                    read_dirs=read_dirs,
                    write_dirs=[],
                    write_scope=None,
                    execution_host=execution_host,
                    execution=execution,
                    remote_stage=remote_stage,
                    capability="discuss",
                    outcome=outcome,
                    binary=provider_binary,
                )
            ) as stream:
                async for frame in stream:
                    yield frame
        except Exception:
            # Provider launch/runtime exceptions are terminal and Background will
            # offer Retry. Cancellation and process shutdown use BaseException
            # paths and retain the reusable native-session stage for Resume.
            outcome.failed = True
            raise

        # Only a labelled final assistant message is the reply. A provider that
        # emitted none has not answered, and promoting its last trace would show
        # reasoning or tool output to the human as if it were the answer.
        answer = "\n\n".join(item.strip() for item in outcome.answers if item.strip()).strip()
        if not outcome.completed:
            if outcome.failed or outcome.paused:
                return
            outcome.failed = True
            yield _sse(AgentEvent(event="error", text=f"{request.provider} produced no result."))
            return
        if not answer:
            yield _sse(
                AgentEvent(
                    event="error",
                    text=f"{request.provider} finished without answering.",
                )
            )
            return

        _commit_chat_prompt_state(execution, request, outcome.session_id)

        assert artifact_scope_id is not None
        assert artifact_directory is not None
        try:
            artifacts = _discover_chat_artifacts(
                execution,
                artifact_scope_id,
                Path(str(artifact_directory)),
                remote_stage,
            )
        except Exception as exc:
            # Preview attachments are optional. Even a programming or storage
            # error in this branch must not take down a labelled chat answer.
            with suppress(Exception):
                _record_artifact_discovery_receipt(
                    execution,
                    attached=0,
                    candidates=0,
                    ignored={"unexpected_error": 1},
                    detail=str(exc),
                )
            artifacts = []
        yield _sse(AgentEvent(event="answer", text=answer))
        for artifact in artifacts:
            yield _sse(AgentEvent(event="artifact", artifact=artifact))

        # Authority to change the graph rides on the human's request. An agent
        # cannot grant it to itself by writing the file, so a stray patch is kept
        # as a receipt and discarded.
        if execution is not None:
            try:
                patch_text = _read_chat_patch(workspace, remote_stage)
            except (OSError, StateUnavailable, ValueError) as exc:
                execution.store.record_agent_task_receipt(
                    execution.operation_id,
                    "discuss_patch_discarded",
                    {
                        "reason": "unreadable",
                        "detail": f"The agent wrote a patch file that could not be read: {exc}"[
                            :400
                        ],
                    },
                    tier="diagnostic",
                )
                execution.store.record_agent_task_event(
                    execution.operation_id,
                    "Discuss wrote an unreadable patch.json; RCP discarded it without "
                    "changing the graph.",
                    level="warning",
                )
            else:
                if patch_text is not None:
                    execution.store.record_agent_task_patch_output(
                        execution.operation_id, patch_text
                    )
                    execution.store.record_agent_task_event(
                        execution.operation_id,
                        "Discuss has no graph authority, so the patch the agent wrote was "
                        "discarded. Switch to Work for a deliberate graph update.",
                        level="warning",
                    )
                    execution.store.record_agent_task_receipt(
                        execution.operation_id,
                        "discuss_patch_discarded",
                        {
                            "reason": "no_graph_authority",
                            "byte_length": len(patch_text.encode("utf-8")),
                        },
                        tier="diagnostic",
                    )

        try:
            _append_chat_exchange(
                service,
                request,
                answer,
                outcome.session_id,
                None,
                execution=execution,
            )
        except (OSError, StateUnavailable, ValueError) as exc:
            if execution is not None:
                execution.store.record_agent_task_event(
                    execution.operation_id,
                    f"The reply was delivered but could not be written to the chat "
                    f"transcript: {exc}",
                    level="warning",
                )
        yield _sse(AgentEvent(event="done"))
    finally:
        # There is no per-turn source cleanup; the reusable native-session stage
        # remains available to the normal stage sweeper.
        if patch_inputs is not None:
            await asyncio.to_thread(
                cleanup_patch_validation_mailbox,
                staged=patch_inputs.validator_staged,
                execution=execution,
            )
