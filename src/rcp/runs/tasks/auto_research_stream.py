from __future__ import annotations

import asyncio
import hashlib
import json
import os
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import aclosing, asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, cast

from rcp.agents import (
    AgentEvent,
    AgentLauncher,
    ChatContext,
    ContextAssembler,
    agent_output_schema,
)
from rcp.agents.auto_research_prompt import (
    auto_research_orchestrator_continuation_contract,
    auto_research_orchestrator_task_contract,
    auto_research_worker_continuation_contract,
    auto_research_worker_task_contract,
)
from rcp.agents.command_mailbox import (
    CommandTurnIdentity,
    StagedCommandMailbox,
    serve_command_mailbox,
    stage_command_mailbox,
)
from rcp.agents.command_protocol import CommandRequest, CommandResponse, ValidateCommandRequest
from rcp.agents.prompts import PromptFactory
from rcp.agents.write_scope import ProjectWriteScope
from rcp.background import AgentTaskExecution
from rcp.core.research_md import render_research_md
from rcp.limits import AUTO_RESEARCH_LIFECYCLE_MAX_BYTES, PATCH_CORRECTION_MAX_ROUNDS
from rcp.providers import classify_terminal_error
from rcp.runs.auto_research import (
    AutoResearchCommandContext,
    AutoResearchCommandDispatcher,
    AutoResearchCommandEffectResult,
    AutoResearchCommandInvalid,
    AutoResearchRunRequest,
)
from rcp.runs.auto_research_lifecycle import (
    AUTO_RESEARCH_LIFECYCLE_HANDOFF_FILE,
    auto_research_lifecycle_delivery,
    parse_auto_research_lifecycle_delivery,
    stage_auto_research_lifecycle_delivery,
)
from rcp.runs.auto_research_mail import (
    AUTO_RESEARCH_MAIL_HANDOFF_FILE,
    AUTO_RESEARCH_MAIL_MAX_BYTES,
    auto_research_mail_delivery,
    parse_auto_research_mail_delivery,
    stage_auto_research_mail_delivery,
)
from rcp.runs.chat import (
    _chat_read_dirs,
    _clear_stale_turn_handoffs,
    _project_write_scope,
    _read_chat_patch,
)
from rcp.runs.shared import (
    _existing_patch_digest,
    _parent_task_contract_path,
    _ProviderOutcome,
    _record_agent_launch_receipt,
    _retry_deliverable_is_unchanged,
    _sse,
    _stage_context_paths,
    _stage_json_task_input,
    _stage_or_reuse_task_input,
    _stage_task_contract,
    _stream_agent_events,
    _swept_stage_root,
    _task_token,
)
from rcp.runs.tasks.work import (
    _apply_work_patch,
    _bounded_graph_messages,
    _CorrectionPatchRead,
    _read_correction_patch,
    _record_work_graph_rejection,
    _WorkPatchFailure,
)
from rcp.service import GraphUpdateResult, ProjectService, RunRequest
from rcp.skills.staging import skill_bundle_label, stage_skill_selection
from rcp.storage import (
    AgentTaskRecord,
    AutoResearchActorBinding,
    AutoResearchMessageRecord,
)
from rcp.transport import (
    RemoteRunStage,
    RunLockCancelled,
    RunStageMailbox,
    StateUnavailable,
    repository_access,
)

_SAME_ALLOCATION_RECOVERY = frozenset({"resume", "retry"})
_HANDOFFS_CLEARED_RECEIPT = "auto_research_worker_handoffs_cleared"
_WORKER_CONTINUATIONS = frozenset(
    {
        "fresh",
        "resume",
        "retry",
        "watcher_wake",
        "graph_condition_wake",
        "message_wake",
        "auto_research_continuation",
    }
)
_ORCHESTRATOR_CONTINUATIONS = _WORKER_CONTINUATIONS | {"lifecycle_wake"}


@dataclass(frozen=True)
class _CanonicalWorkerTurn:
    task: AgentTaskRecord
    request: AutoResearchRunRequest
    binding: AutoResearchActorBinding
    allocation_operation_id: str
    recovering_allocation: bool


@dataclass(frozen=True)
class _CanonicalOrchestratorTurn:
    task: AgentTaskRecord
    request: AutoResearchRunRequest
    binding: AutoResearchActorBinding
    allocation_operation_id: str
    recovering_allocation: bool
    clean_session_retry: bool


@dataclass(frozen=True)
class _WorkerStage:
    local: Path | None
    remote: RemoteRunStage | None
    workspace: Path
    execution_host: str
    provider_binary: str | None


@dataclass(frozen=True)
class _PatchSettlement:
    graph_update: GraphUpdateResult | None
    frames: tuple[str, ...] = ()
    had_patch: bool = False


async def stream_auto_research_orchestrator_run(
    service: ProjectService,
    launcher: AgentLauncher,
    request: AutoResearchRunRequest,
    data_dir: Path,
    execution: AgentTaskExecution,
    *,
    command_dispatcher: AutoResearchCommandDispatcher,
) -> AsyncIterator[str]:
    """Run one paid turn of the sole project-owned auto_research orchestrator."""

    try:
        turn = _canonical_orchestrator_turn(execution, request)
        if command_dispatcher.store is not execution.store:
            raise ValueError(
                "auto_research orchestrator stream and command dispatcher must share one store"
            )
        stage = _open_orchestrator_stage(service, data_dir, execution, turn)
        command_files = RunStageMailbox.for_stage(
            local_stage=stage.local,
            remote_stage=stage.remote,
        )
        command_dispatcher = command_dispatcher.with_command_files(
            reader=lambda name, max_bytes: command_files.read_text(name, max_bytes=max_bytes),
            consumer=command_files.remove_if_sha256,
            refresher=lambda: _refreshed_orchestrator_state_paths(
                service,
                turn.request,
                stage,
            ),
        )
        context = _auto_research_context(service, turn.request, stage)
        _prepare_orchestrator_handoffs(execution, turn, stage)

        messages_path = _stage_claimed_mail(execution, turn, stage)
        lifecycle_path = _stage_claimed_lifecycle(execution, turn, stage)
        token = _task_token(execution)
        schema = (
            json.dumps(
                agent_output_schema(profile="orchestrator"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        schema_digest = hashlib.sha256(schema.encode("utf-8")).hexdigest()[:16]
        schema_path = _stage_or_reuse_task_input(
            stage.local,
            stage.remote,
            f"auto_research-orchestrator-patch-schema-{schema_digest}.json",
            schema,
        )
        patch_path = str(stage.workspace / "patch.json")
        # The orchestrator gets the project's enabled packages on the same terms as any
        # other Work agent: resolved from Settings at launch, never from the task receipt.
        orchestrator_selection = service.resolve_skill_selection(turn.request)
        orchestrator_skill_pointers = stage_skill_selection(
            orchestrator_selection,
            local_stage=stage.local,
            remote_stage=stage.remote,
            label=skill_bundle_label(orchestrator_selection),
            reuse_existing=True,
        )
        expected_turn_id = f"{execution.operation_id}:orchestrator"
        staged_commands = stage_command_mailbox(
            local_stage=stage.local,
            remote_stage=stage.remote,
            episode_id=turn.request.episode_id,
            task_id=execution.operation_id,
            turn_id=expected_turn_id,
        )
        async with _worker_mailbox_lifecycle(
            staged_commands,
            execution=execution,
            start=lambda stop: _serve_auto_research_commands(
                staged_commands,
                execution=execution,
                turn=turn,
                dispatcher=command_dispatcher,
                stop=stop,
                expected_turn_id=expected_turn_id,
            ),
        ):
            validator_command = staged_commands.client_command("validate", patch_path)
            contract_path, prompt = _orchestrator_prompt(
                execution,
                turn,
                context=context,
                local_stage=stage.local,
                remote_stage=stage.remote,
                token=token,
                patch_path=patch_path,
                schema_path=schema_path,
                validator_command=validator_command,
                command_client=staged_commands.client_command(),
                messages_path=messages_path,
                lifecycle_path=lifecycle_path,
                skill_pointers=orchestrator_skill_pointers,
            )
            read_dirs = _chat_read_dirs(
                context,
                stage.remote,
                service,
                turn.request.run_on or "",
            )
            write_scope = _project_write_scope(
                context,
                service,
                turn.request.run_on or "",
                workspace=stage.workspace,
                remote_stage=stage.remote,
                data_dir=data_dir,
                execution=execution,
                capability="orchestrate",
            )
            write_dirs = [Path(item) for item in write_scope.repository_roots]
            retry_patch_digest = (
                _existing_patch_digest(stage.workspace, stage.remote)
                if execution.continuation == "retry"
                else None
            )
            _record_agent_launch_receipt(
                execution,
                cast(RunRequest, turn.request),
                prompt=prompt,
                contract_path=contract_path,
                remote=bool(stage.execution_host),
                resumed=turn.binding.native_session_id is not None,
                write_scope=write_scope,
                continuation=execution.continuation,
                extra={
                    "surface": "auto_research",
                    "role": "orchestrator",
                    "profile": "orchestrator",
                    "actor_operation_id": turn.binding.actor_operation_id,
                    "allocation_operation_id": turn.allocation_operation_id,
                    "capability": "orchestrate",
                    "network_access": True,
                    "write_directory_count": len(write_dirs),
                    "canonical_state_boundary": "prompt_only",
                },
            )
            outcome = _ProviderOutcome(session_id=turn.binding.native_session_id)
            async with aclosing(
                _stream_agent_events(
                    launcher,
                    cast(RunRequest, turn.request),
                    prompt,
                    workspace=stage.workspace,
                    session_id=turn.binding.native_session_id,
                    read_dirs=read_dirs,
                    write_dirs=write_dirs,
                    write_scope=write_scope,
                    execution_host=stage.execution_host,
                    execution=execution,
                    remote_stage=stage.remote,
                    capability="orchestrate",
                    outcome=outcome,
                    binary=stage.provider_binary,
                    invocation_gate=staged_commands.invocation_gate,
                )
            ) as stream:
                async for frame in stream:
                    yield frame
        if outcome.paused or outcome.failed:
            return
        if not outcome.completed:
            yield _sse(
                AgentEvent(
                    event="error",
                    text=f"{turn.request.provider} produced no auto_research orchestrator result.",
                )
            )
            return
        if not outcome.session_id:
            yield _sse(
                AgentEvent(
                    event="error", text="AutoResearch orchestrator returned no native session id."
                )
            )
            return
        if (
            turn.binding.native_session_id is not None
            and outcome.session_id != turn.binding.native_session_id
        ):
            yield _sse(
                AgentEvent(
                    event="error",
                    text="AutoResearch orchestrator continuation changed its canonical native session.",
                )
            )
            return
        answer = "\n\n".join(item.strip() for item in outcome.answers if item.strip())
        if not answer:
            yield _sse(
                AgentEvent(
                    event="error", text="AutoResearch orchestrator finished without an answer."
                )
            )
            return
        yield _sse(AgentEvent(event="answer", text=answer))

        settlement = await _settle_orchestrator_patch(
            service,
            launcher,
            execution,
            turn,
            stage,
            contract_path=contract_path,
            patch_path=patch_path,
            schema_path=schema_path,
            read_dirs=read_dirs,
            write_dirs=write_dirs,
            write_scope=write_scope,
            provider_binary=stage.provider_binary,
            native_session_id=outcome.session_id,
            retry_patch_digest=retry_patch_digest,
            command_dispatcher=command_dispatcher,
        )
        for frame in settlement.frames:
            yield frame
        graph_updates = _ordered_orchestrator_graph_updates(execution)
        graph_update = settlement.graph_update
        if graph_update is not None and (settlement.had_patch or not graph_updates):
            graph_updates.append(graph_update)
        if not graph_updates:
            return
        graph_update = graph_updates[-1]
        payload: dict[str, object] = {
            "graph_update": graph_update.model_dump(mode="json"),
            "graph_updates": [item.model_dump(mode="json") for item in graph_updates],
        }
        latest_applied_revision = next(
            (
                item.applied_revision
                for item in reversed(graph_updates)
                if item.applied_revision is not None
            ),
            None,
        )
        if latest_applied_revision is not None:
            payload["applied_revision"] = latest_applied_revision
        yield _sse(AgentEvent(event="message", text=json.dumps(payload, separators=(",", ":"))))
        yield _sse(AgentEvent(event="done"))
    except (KeyError, OSError, StateUnavailable, ValueError) as exc:
        yield _sse(AgentEvent(event="error", text=str(exc)))


def _ordered_orchestrator_graph_updates(
    execution: AgentTaskExecution,
) -> list[GraphUpdateResult]:
    """Rebuild this turn's in-turn Apply history from its durable commit ledger."""

    graph_updates: list[GraphUpdateResult] = []
    for record in execution.store.auto_research_apply_results(execution.operation_id):
        effect_result = record.result.get("result")
        if not isinstance(effect_result, dict):
            continue
        raw_graph_update = effect_result.get("graph_update")
        if raw_graph_update is None:
            continue
        try:
            graph_updates.append(GraphUpdateResult.model_validate(raw_graph_update))
        except (TypeError, ValueError):
            execution.store.record_agent_task_event(
                execution.operation_id,
                f"Ignored an invalid durable Apply result: {record.apply_id}.",
                level="warning",
            )
    return graph_updates


async def stream_auto_research_worker_run(
    service: ProjectService,
    launcher: AgentLauncher,
    request: AutoResearchRunRequest,
    data_dir: Path,
    execution: AgentTaskExecution,
    *,
    command_dispatcher: AutoResearchCommandDispatcher,
) -> AsyncIterator[str]:
    """Run one ordinary auto_research worker on its canonical actor-owned stage."""

    try:
        turn = _canonical_worker_turn(execution, request)
        if command_dispatcher.store is not execution.store:
            raise ValueError(
                "auto_research worker stream and command dispatcher must share one store"
            )
        stage = _open_worker_stage(service, data_dir, execution, turn)
        context = _auto_research_context(service, turn.request, stage)
        _prepare_worker_handoffs(execution, turn, stage)

        messages_path = _stage_claimed_mail(execution, turn, stage)
        token = _task_token(execution)
        schema = (
            json.dumps(agent_output_schema(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        schema_digest = hashlib.sha256(schema.encode("utf-8")).hexdigest()[:16]
        schema_path = _stage_or_reuse_task_input(
            stage.local,
            stage.remote,
            f"auto_research-patch-schema-{schema_digest}.json",
            schema,
        )
        patch_path = str(stage.workspace / "patch.json")
        staged_commands = stage_command_mailbox(
            local_stage=stage.local,
            remote_stage=stage.remote,
            episode_id=turn.request.episode_id,
            task_id=execution.operation_id,
            turn_id=f"{execution.operation_id}:worker",
        )
        async with _worker_mailbox_lifecycle(
            staged_commands,
            execution=execution,
            start=lambda stop: _serve_worker_commands(
                staged_commands,
                execution=execution,
                turn=turn,
                dispatcher=command_dispatcher,
                stop=stop,
                expected_turn_id=f"{execution.operation_id}:worker",
            ),
        ):
            validator_command = staged_commands.client_command("validate", patch_path)
            reply_command = staged_commands.client_command(
                "message",
                "--key",
                _worker_reply_key(turn),
            )
            contract_path, prompt = _worker_prompt(
                service,
                execution,
                turn,
                context=context,
                local_stage=stage.local,
                remote_stage=stage.remote,
                token=token,
                patch_path=patch_path,
                schema_path=schema_path,
                validator_command=validator_command,
                reply_command=reply_command,
                messages_path=messages_path,
            )
            read_dirs = _chat_read_dirs(
                context,
                stage.remote,
                service,
                turn.request.run_on or "",
            )
            write_scope = _project_write_scope(
                context,
                service,
                turn.request.run_on or "",
                workspace=stage.workspace,
                remote_stage=stage.remote,
                data_dir=data_dir,
                execution=execution,
                capability="work_auto",
            )
            write_dirs = [Path(item) for item in write_scope.repository_roots]
            retry_patch_digest = (
                _existing_patch_digest(stage.workspace, stage.remote)
                if execution.continuation == "retry"
                else None
            )
            _record_agent_launch_receipt(
                execution,
                cast(RunRequest, turn.request),
                prompt=prompt,
                contract_path=contract_path,
                remote=bool(stage.execution_host),
                resumed=turn.binding.native_session_id is not None,
                write_scope=write_scope,
                continuation=execution.continuation,
                extra={
                    "surface": "auto_research",
                    "role": "worker",
                    "profile": "ordinary",
                    "actor_operation_id": turn.binding.actor_operation_id,
                    "allocation_operation_id": turn.allocation_operation_id,
                    "control_node_id": turn.binding.control_node_id,
                    "capability": "work_auto",
                    "network_access": True,
                    "write_directory_count": len(write_dirs),
                    "canonical_state_boundary": "prompt_only",
                },
            )

            outcome = _ProviderOutcome(session_id=turn.binding.native_session_id)
            async with aclosing(
                _stream_agent_events(
                    launcher,
                    cast(RunRequest, turn.request),
                    prompt,
                    workspace=stage.workspace,
                    session_id=turn.binding.native_session_id,
                    read_dirs=read_dirs,
                    write_dirs=write_dirs,
                    write_scope=write_scope,
                    execution_host=stage.execution_host,
                    execution=execution,
                    remote_stage=stage.remote,
                    capability="work_auto",
                    outcome=outcome,
                    binary=stage.provider_binary,
                    invocation_gate=staged_commands.invocation_gate,
                )
            ) as stream:
                async for frame in stream:
                    yield frame
        if outcome.paused or outcome.failed:
            return
        if not outcome.completed:
            yield _sse(
                AgentEvent(
                    event="error",
                    text=f"{turn.request.provider} produced no auto_research worker result.",
                )
            )
            return
        if not outcome.session_id:
            yield _sse(
                AgentEvent(event="error", text="AutoResearch worker returned no native session id.")
            )
            return
        if (
            turn.binding.native_session_id is not None
            and outcome.session_id != turn.binding.native_session_id
        ):
            yield _sse(
                AgentEvent(
                    event="error",
                    text="AutoResearch worker continuation changed its canonical native session.",
                )
            )
            return
        answer = "\n\n".join(item.strip() for item in outcome.answers if item.strip())
        if not answer:
            yield _sse(
                AgentEvent(event="error", text="AutoResearch worker finished without an answer.")
            )
            return
        yield _sse(AgentEvent(event="answer", text=answer))

        settlement = await _settle_worker_patch(
            service,
            launcher,
            execution,
            turn,
            stage,
            contract_path=contract_path,
            patch_path=patch_path,
            schema_path=schema_path,
            read_dirs=read_dirs,
            write_dirs=write_dirs,
            write_scope=write_scope,
            provider_binary=stage.provider_binary,
            native_session_id=outcome.session_id,
            retry_patch_digest=retry_patch_digest,
            command_dispatcher=command_dispatcher,
        )
        for frame in settlement.frames:
            yield frame
        graph_update = settlement.graph_update
        if graph_update is None:
            return
        payload: dict[str, object] = {"graph_update": graph_update.model_dump(mode="json")}
        if graph_update.applied_revision is not None:
            payload["applied_revision"] = graph_update.applied_revision
        yield _sse(AgentEvent(event="message", text=json.dumps(payload, separators=(",", ":"))))
        yield _sse(AgentEvent(event="done"))
    except (KeyError, OSError, StateUnavailable, ValueError) as exc:
        yield _sse(AgentEvent(event="error", text=str(exc)))


def _canonical_worker_turn(
    execution: AgentTaskExecution,
    supplied: AutoResearchRunRequest,
) -> _CanonicalWorkerTurn:
    if execution.continuation not in _WORKER_CONTINUATIONS:
        raise ValueError("AutoResearch worker continuation is not supported by this stream.")
    task = execution.store.agent_task(execution.operation_id)
    if task is None or task.kind != "auto_research" or task.episode_id is None:
        raise ValueError("AutoResearch worker execution requires one durable auto_research task.")
    durable = AutoResearchRunRequest.model_validate(task.request)
    if durable != supplied:
        raise ValueError("AutoResearch worker launch request differs from its durable task record.")
    binding = execution.store.auto_research_actor_binding(execution.operation_id)
    if execution.store.agent_task_profile(execution.operation_id) != "ordinary":
        raise ValueError("AutoResearch worker requires the canonical ordinary semantic profile.")
    if binding.role != "worker" or durable.role != "worker":
        raise ValueError("This stream executes ordinary auto_research workers only.")
    if (
        binding.episode_id != task.episode_id
        or durable.episode_id != task.episode_id
        or durable.actor_operation_id != binding.actor_operation_id
        or durable.control_node_id != binding.control_node_id
        or binding.current_operation_id != execution.operation_id
    ):
        raise ValueError(
            "AutoResearch worker role, actor, or seat conflicts with its durable binding."
        )
    if (execution.stage_host or "") != (binding.stage_host or ""):
        raise ValueError(
            "AutoResearch worker execution host conflicts with its durable actor binding."
        )
    if execution.stage_root != binding.stage_root:
        raise ValueError("AutoResearch worker stage conflicts with its durable actor binding.")
    cause = execution.store.agent_task_continuation_cause(execution.operation_id)
    if cause != execution.continuation:
        raise ValueError(
            "AutoResearch worker continuation conflicts with its durable launch cause."
        )
    recovering = execution.continuation in _SAME_ALLOCATION_RECOVERY
    if execution.continuation != "fresh" and (
        not binding.native_session_id
        or not binding.stage_root
        or durable.session_id != binding.native_session_id
    ):
        raise ValueError("AutoResearch worker recovery requires its exact saved session and stage.")
    if execution.continuation == "fresh" and binding.actor_operation_id != execution.operation_id:
        raise ValueError("A fresh auto_research worker cannot continue another actor.")
    allocation_operation_id = _paid_allocation_operation_id(execution, task)
    return _CanonicalWorkerTurn(
        task=task,
        request=durable.model_copy(update={"session_id": binding.native_session_id}),
        binding=binding,
        allocation_operation_id=allocation_operation_id,
        recovering_allocation=recovering,
    )


def _canonical_orchestrator_turn(
    execution: AgentTaskExecution,
    supplied: AutoResearchRunRequest,
) -> _CanonicalOrchestratorTurn:
    if execution.continuation not in _ORCHESTRATOR_CONTINUATIONS:
        raise ValueError("AutoResearch orchestrator continuation is not supported by this stream.")
    task = execution.store.agent_task(execution.operation_id)
    if task is None or task.kind != "auto_research" or task.episode_id is None:
        raise ValueError(
            "AutoResearch orchestrator execution requires one durable auto_research task."
        )
    durable = AutoResearchRunRequest.model_validate(task.request)
    if durable != supplied:
        raise ValueError(
            "AutoResearch orchestrator launch request differs from its durable task record."
        )
    binding = execution.store.auto_research_actor_binding(execution.operation_id)
    if execution.store.agent_task_profile(execution.operation_id) != "orchestrator":
        raise ValueError("AutoResearch orchestrator requires its sole elevated semantic profile.")
    if binding.role != "orchestrator" or durable.role != "orchestrator":
        raise ValueError("This stream executes the auto_research orchestrator only.")
    if (
        binding.episode_id != task.episode_id
        or durable.episode_id != task.episode_id
        or durable.actor_operation_id != binding.actor_operation_id
        or durable.control_node_id is not None
        or binding.control_node_id is not None
        or binding.current_operation_id != execution.operation_id
    ):
        raise ValueError(
            "AutoResearch orchestrator role, actor, or scope conflicts with its durable binding."
        )
    if (execution.stage_host or "") != (binding.stage_host or ""):
        raise ValueError(
            "AutoResearch orchestrator execution host conflicts with its durable actor binding."
        )
    if execution.stage_root != binding.stage_root:
        raise ValueError(
            "AutoResearch orchestrator stage conflicts with its durable actor binding."
        )
    cause = execution.store.agent_task_continuation_cause(execution.operation_id)
    if cause != execution.continuation:
        raise ValueError(
            "AutoResearch orchestrator continuation conflicts with its durable launch cause."
        )
    recovering = execution.continuation in _SAME_ALLOCATION_RECOVERY
    clean_retry = _is_authorized_clean_orchestrator_retry(
        execution,
        task=task,
        request=durable,
        binding=binding,
    )
    if (
        execution.continuation != "fresh"
        and not clean_retry
        and (
            not binding.native_session_id
            or not binding.stage_root
            or durable.session_id != binding.native_session_id
        )
    ):
        raise ValueError(
            "AutoResearch orchestrator continuation requires its exact session and stage."
        )
    if execution.continuation == "fresh" and (
        binding.actor_operation_id != execution.operation_id or task.parent_operation_id is not None
    ):
        raise ValueError("A fresh auto_research orchestrator must be the sole root actor.")
    allocation_operation_id = _paid_allocation_operation_id(execution, task)
    return _CanonicalOrchestratorTurn(
        task=task,
        request=durable.model_copy(update={"session_id": binding.native_session_id}),
        binding=binding,
        allocation_operation_id=allocation_operation_id,
        recovering_allocation=recovering,
        clean_session_retry=clean_retry,
    )


def _is_authorized_clean_orchestrator_retry(
    execution: AgentTaskExecution,
    *,
    task: AgentTaskRecord,
    request: AutoResearchRunRequest,
    binding: AutoResearchActorBinding,
) -> bool:
    """Recognize the storage-authorized same-allocation clean-session retry."""

    if (
        execution.continuation != "retry"
        or task.parent_operation_id is None
        or task.native_session_id is not None
        or request.session_id is not None
        or binding.native_session_id is not None
    ):
        return False
    parent = execution.store.agent_task(task.parent_operation_id)
    if (
        parent is None
        or parent.episode_id != task.episode_id
        or parent.kind != "auto_research"
        or parent.status not in {"paused", "interrupted", "failed"}
        or task.attempt != parent.attempt + 1
    ):
        return False
    parent_request = AutoResearchRunRequest.model_validate(parent.request)
    if (
        parent_request.role != "orchestrator"
        or (parent_request.actor_operation_id or parent.operation_id) != binding.actor_operation_id
        or (parent.stage_host or "") != (binding.stage_host or "")
        or parent.stage_root != binding.stage_root
    ):
        return False
    if not parent.native_session_id:
        return True
    receipts = execution.store.agent_task_receipts(parent.operation_id)
    session_limit = any(
        receipt.category == "provider_terminal_error"
        and receipt.payload.get("classification") == "session_limit"
        for receipt in receipts
    ) or (bool(parent.error) and classify_terminal_error(parent.error or "") == "session_limit")
    continuation_unavailable = any(
        receipt.category == "continuation_context_unavailable"
        and receipt.payload.get("retry_required") is True
        for receipt in receipts
    )
    if not session_limit and not continuation_unavailable:
        return False

    # A clean retry deliberately leaves the new task's session NULL so the
    # provider can bind its replacement with the existing checkpoint CAS. Walk
    # only this same-allocation recovery chain to prove that the actor previously
    # held the native session/stage pair being retired.
    prior = parent
    seen: set[str] = set()
    while True:
        if prior.operation_id in seen:
            return False
        seen.add(prior.operation_id)
        prior_request = AutoResearchRunRequest.model_validate(prior.request)
        if (
            prior.episode_id != task.episode_id
            or prior.kind != "auto_research"
            or prior_request.role != "orchestrator"
            or (prior_request.actor_operation_id or prior.operation_id)
            != binding.actor_operation_id
            or (prior.stage_host or "") != (binding.stage_host or "")
            or prior.stage_root != binding.stage_root
        ):
            return False
        if prior.native_session_id and prior.stage_root:
            return True
        if (
            prior.parent_operation_id is None
            or execution.store.agent_task_continuation_cause(prior.operation_id)
            not in _SAME_ALLOCATION_RECOVERY
        ):
            return False
        ancestor = execution.store.agent_task(prior.parent_operation_id)
        if ancestor is None:
            return False
        prior = ancestor


def _paid_allocation_operation_id(
    execution: AgentTaskExecution,
    task: AgentTaskRecord,
) -> str:
    current = task
    seen: set[str] = set()
    while execution.store.agent_task_continuation_cause(current.operation_id) in {
        "resume",
        "retry",
    }:
        if current.operation_id in seen or current.parent_operation_id is None:
            raise ValueError("AutoResearch actor recovery lost its paid allocation lineage.")
        seen.add(current.operation_id)
        parent = execution.store.agent_task(current.parent_operation_id)
        if parent is None or parent.episode_id != task.episode_id or parent.kind != "auto_research":
            raise ValueError("AutoResearch actor recovery crossed its paid allocation lineage.")
        current = parent
    return current.operation_id


def _worker_stage_name(project_id: str, actor_operation_id: str) -> str:
    project = hashlib.sha256(project_id.encode("utf-8")).hexdigest()[:16]
    actor = hashlib.sha256(actor_operation_id.encode("utf-8")).hexdigest()[:16]
    return f"auto_research-worker-{project}-{actor}"


def _orchestrator_stage_name(project_id: str, actor_operation_id: str) -> str:
    project = hashlib.sha256(project_id.encode("utf-8")).hexdigest()[:16]
    actor = hashlib.sha256(actor_operation_id.encode("utf-8")).hexdigest()[:16]
    return f"auto_research-orchestrator-{project}-{actor}"


def _open_worker_stage(
    service: ProjectService,
    data_dir: Path,
    execution: AgentTaskExecution,
    turn: _CanonicalWorkerTurn,
) -> _WorkerStage:
    return _open_auto_research_actor_stage(
        service,
        data_dir,
        execution,
        turn,
        stage_name=_worker_stage_name(turn.task.project_id, turn.binding.actor_operation_id),
        actor_label="worker",
    )


def _open_orchestrator_stage(
    service: ProjectService,
    data_dir: Path,
    execution: AgentTaskExecution,
    turn: _CanonicalOrchestratorTurn,
) -> _WorkerStage:
    return _open_auto_research_actor_stage(
        service,
        data_dir,
        execution,
        turn,
        stage_name=_orchestrator_stage_name(
            turn.task.project_id,
            turn.binding.actor_operation_id,
        ),
        actor_label="orchestrator",
        allow_new_stage=turn.clean_session_retry,
    )


def _open_auto_research_actor_stage(
    service: ProjectService,
    data_dir: Path,
    execution: AgentTaskExecution,
    turn: _CanonicalWorkerTurn | _CanonicalOrchestratorTurn,
    *,
    stage_name: str,
    actor_label: Literal["worker", "orchestrator"],
    allow_new_stage: bool = False,
) -> _WorkerStage:
    request = turn.request
    if (
        not request.provider
        or request.model is None
        or request.reasoning is None
        or not request.run_on
    ):
        raise ValueError(f"AutoResearch {actor_label} has no exact pinned provider profile.")
    machine = service.manifest.machine_map.get(request.run_on)
    if machine is None:
        raise ValueError(f"unknown auto_research {actor_label} execution machine: {request.run_on}")
    expected_remote = str(PurePosixPath("/tmp") / f"rcp-run.{stage_name}")
    if turn.binding.stage_root is not None:
        if machine.host:
            if (
                turn.binding.stage_host != machine.host
                or turn.binding.stage_root != expected_remote
            ):
                raise ValueError(
                    f"AutoResearch {actor_label} saved remote stage has a different actor binding."
                )
            remote = RemoteRunStage(machine.host).attach(turn.binding.stage_root)
            return _WorkerStage(
                local=None,
                remote=remote,
                workspace=Path(str(remote.workspace)),
                execution_host=machine.host,
                provider_binary=machine.provider_paths.get(request.provider),
            )
        expected_local = _swept_stage_root(data_dir) / stage_name
        saved = Path(turn.binding.stage_root)
        if (
            turn.binding.stage_host is not None
            or saved.absolute() != expected_local.absolute()
            or saved.is_symlink()
            or not saved.is_dir()
        ):
            raise ValueError(
                f"AutoResearch {actor_label} saved local stage has a different actor binding."
            )
        return _WorkerStage(
            local=saved,
            remote=None,
            workspace=saved,
            execution_host="",
            provider_binary=machine.provider_paths.get(request.provider),
        )

    if turn.binding.native_session_id is not None or (
        execution.continuation != "fresh" and not allow_new_stage
    ):
        raise ValueError(
            f"AutoResearch {actor_label} continuation cannot start a fresh execution stage."
        )
    if machine.host:
        remote = RemoteRunStage(machine.host).open(stage_name, reuse=True)
        assert remote.root is not None
        execution.checkpoint_stage(machine.host, str(remote.root))
        return _WorkerStage(
            local=None,
            remote=remote,
            workspace=Path(str(remote.workspace)),
            execution_host=machine.host,
            provider_binary=machine.provider_paths.get(request.provider),
        )
    local = _swept_stage_root(data_dir) / stage_name
    if os.path.lexists(local):
        if local.is_symlink() or not local.is_dir():
            raise ValueError(f"AutoResearch {actor_label} local stage is unsafe.")
    else:
        local.mkdir(mode=0o700, parents=True)
    execution.checkpoint_stage("", str(local))
    return _WorkerStage(
        local=local,
        remote=None,
        workspace=local,
        execution_host="",
        provider_binary=machine.provider_paths.get(request.provider),
    )


def _auto_research_context(
    service: ProjectService,
    request: AutoResearchRunRequest,
    stage: _WorkerStage,
) -> ChatContext:
    state = service.history.state()
    selected = request.run_truth_scope or service.manifest.agent.default_run_truth_scope
    access = {
        alias: repository_access(
            service.manifest.repository_map[alias],
            service.manifest.machine_map[service.manifest.repository_map[alias].machine],
        )
        for alias in selected
        if alias in service.manifest.repository_map
    }
    context = ContextAssembler(service.manifest).chat_context(
        state,
        node_id=None,
        run_truth_scope=request.run_truth_scope,
        repository_access=access,
    )
    state_machine = service.manifest.repository_map[service.manifest.state.repository].machine
    if state_machine != request.run_on:
        repositories = []
        for item in context.repositories:
            if item.machine == request.run_on:
                repositories.append(item.model_copy(update={"host": ""}))
            elif item.host:
                repositories.append(item)
            else:
                raise StateUnavailable(
                    f"Repository {item.alias!r} has no SSH host reachable from auto_research "
                    f"execution machine {request.run_on!r}."
                )
        graph = (
            json.dumps(
                state.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        research = render_research_md(state)
        graph_digest = hashlib.sha256(graph.encode("utf-8")).hexdigest()[:16]
        research_digest = hashlib.sha256(research.encode("utf-8")).hexdigest()[:16]
        updates = {
            "repositories": repositories,
            "graph_path": _stage_or_reuse_task_input(
                stage.local,
                stage.remote,
                f"auto_research-graph-r{state.revision}-{graph_digest}.json",
                graph,
            ),
            "research_md_path": _stage_or_reuse_task_input(
                stage.local,
                stage.remote,
                f"auto_research-research-r{state.revision}-{research_digest}.md",
                research,
            ),
        }
        context = context.model_copy(update=updates)
    elif stage.remote is not None:
        context = context.model_copy(
            update=_stage_context_paths(context, service, stage.remote, request.run_on or "")
        )
    return context


def _refreshed_orchestrator_state_paths(
    service: ProjectService,
    request: AutoResearchRunRequest,
    stage: _WorkerStage,
) -> tuple[int, str, str]:
    context = _auto_research_context(service, request, stage)
    return context.graph_revision, context.graph_path, context.research_md_path


def _claimed_messages(
    execution: AgentTaskExecution,
    turn: _CanonicalWorkerTurn | _CanonicalOrchestratorTurn,
) -> list[AutoResearchMessageRecord]:
    return [
        message
        for message in execution.store.auto_research_messages(turn.request.episode_id)
        if message.delivery_operation_id == turn.allocation_operation_id
    ]


def _prepare_worker_handoffs(
    execution: AgentTaskExecution,
    turn: _CanonicalWorkerTurn,
    stage: _WorkerStage,
) -> None:
    cleared = execution.store.auto_research_handoffs_cleared(turn.allocation_operation_id)
    if turn.recovering_allocation and cleared:
        return
    _clear_stale_turn_handoffs(stage.workspace, stage.remote)
    execution.store.mark_auto_research_handoffs_cleared(turn.allocation_operation_id)
    execution.store.record_agent_task_receipt(
        turn.allocation_operation_id,
        _HANDOFFS_CLEARED_RECEIPT,
        {
            "version": 1,
            "files": [
                "patch.json",
                "watch.json",
                AUTO_RESEARCH_MAIL_HANDOFF_FILE,
                AUTO_RESEARCH_LIFECYCLE_HANDOFF_FILE,
            ],
        },
    )


def _prepare_orchestrator_handoffs(
    execution: AgentTaskExecution,
    turn: _CanonicalOrchestratorTurn,
    stage: _WorkerStage,
) -> None:
    cleared = execution.store.auto_research_handoffs_cleared(turn.allocation_operation_id)
    if turn.recovering_allocation and cleared:
        return
    _clear_stale_turn_handoffs(stage.workspace, stage.remote)
    execution.store.mark_auto_research_handoffs_cleared(turn.allocation_operation_id)
    execution.store.record_agent_task_receipt(
        turn.allocation_operation_id,
        _HANDOFFS_CLEARED_RECEIPT,
        {
            "version": 1,
            "files": [
                "patch.json",
                "watch.json",
                AUTO_RESEARCH_MAIL_HANDOFF_FILE,
                AUTO_RESEARCH_LIFECYCLE_HANDOFF_FILE,
            ],
        },
    )


def _stage_claimed_mail(
    execution: AgentTaskExecution,
    turn: _CanonicalWorkerTurn | _CanonicalOrchestratorTurn,
    stage: _WorkerStage,
) -> str | None:
    messages = _claimed_messages(execution, turn)
    mailbox = RunStageMailbox.for_stage(local_stage=stage.local, remote_stage=stage.remote)
    if messages:
        recipient = messages[0].recipient_task_id
        if recipient != turn.binding.actor_operation_id:
            raise ValueError("Claimed auto_research mail targets another auto_research actor.")
        delivery = auto_research_mail_delivery(
            episode_id=turn.request.episode_id,
            recipient_task_id=recipient,
            delivery_operation_id=turn.allocation_operation_id,
            messages=messages,
        )
        if turn.recovering_allocation and AUTO_RESEARCH_MAIL_HANDOFF_FILE in mailbox.entry_names():
            retained = parse_auto_research_mail_delivery(
                mailbox.read_text(
                    AUTO_RESEARCH_MAIL_HANDOFF_FILE,
                    max_bytes=AUTO_RESEARCH_MAIL_MAX_BYTES,
                )
            )
            if retained != delivery:
                raise ValueError(
                    "Retained auto_research mail differs from its durable claimed batch."
                )
        else:
            stage_auto_research_mail_delivery(mailbox, delivery)
        return str(stage.workspace / AUTO_RESEARCH_MAIL_HANDOFF_FILE)
    mailbox.remove(AUTO_RESEARCH_MAIL_HANDOFF_FILE)
    if turn.request.wake_cause == "message" and not turn.recovering_allocation:
        raise ValueError("AutoResearch message wake has no mail claimed by this paid allocation.")
    return None


def _stage_claimed_lifecycle(
    execution: AgentTaskExecution,
    turn: _CanonicalOrchestratorTurn,
    stage: _WorkerStage,
) -> str | None:
    notices = execution.store.auto_research_lifecycle_delivery(turn.allocation_operation_id)
    mailbox = RunStageMailbox.for_stage(local_stage=stage.local, remote_stage=stage.remote)
    if notices:
        if turn.binding.role != "orchestrator":
            raise ValueError("Lifecycle notices may wake only the Auto-research orchestrator.")
        delivery = auto_research_lifecycle_delivery(
            episode_id=turn.request.episode_id,
            recipient_task_id=turn.binding.actor_operation_id,
            delivery_operation_id=turn.allocation_operation_id,
            notices=notices,
        )
        if (
            turn.recovering_allocation
            and AUTO_RESEARCH_LIFECYCLE_HANDOFF_FILE in mailbox.entry_names()
        ):
            retained = parse_auto_research_lifecycle_delivery(
                mailbox.read_text(
                    AUTO_RESEARCH_LIFECYCLE_HANDOFF_FILE,
                    max_bytes=AUTO_RESEARCH_LIFECYCLE_MAX_BYTES,
                )
            )
            if retained != delivery:
                raise ValueError(
                    "Retained Auto-research lifecycle input differs from its durable claim."
                )
        else:
            stage_auto_research_lifecycle_delivery(mailbox, delivery)
        return str(stage.workspace / AUTO_RESEARCH_LIFECYCLE_HANDOFF_FILE)
    mailbox.remove(AUTO_RESEARCH_LIFECYCLE_HANDOFF_FILE)
    if turn.request.wake_cause == "lifecycle":
        raise ValueError(
            "AutoResearch lifecycle wake has no lifecycle facts claimed by this paid allocation."
        )
    return None


def _worker_reply_key(turn: _CanonicalWorkerTurn) -> str:
    digest = hashlib.sha256(
        (
            "auto_research-worker-reply\0"
            + turn.request.episode_id
            + "\0"
            + turn.allocation_operation_id
        ).encode("utf-8")
    ).hexdigest()
    return f"worker-reply-{digest[:32]}"


def _orchestrator_prompt(
    execution: AgentTaskExecution,
    turn: _CanonicalOrchestratorTurn,
    *,
    context: ChatContext,
    local_stage: Path | None,
    remote_stage: RemoteRunStage | None,
    token: str,
    patch_path: str,
    schema_path: str,
    validator_command: str,
    command_client: str,
    messages_path: str | None,
    lifecycle_path: str | None,
    skill_pointers: list[dict[str, object]],
) -> tuple[str, str]:
    repositories = [
        {"alias": item.alias, "host": item.host, "path": item.path} for item in context.repositories
    ]
    # Without a saved stage the failed allocation never reached its first provider launch, so
    # its clean retry needs the full base contract rather than a reference to a contract it
    # never got.
    first_launch = execution.continuation == "fresh" or (
        turn.clean_session_retry and turn.binding.stage_root is None
    )
    if first_launch:
        instruction_path = None
        if turn.request.instruction:
            instruction_digest = hashlib.sha256(
                turn.request.instruction.encode("utf-8")
            ).hexdigest()[:16]
            instruction_path = _stage_or_reuse_task_input(
                local_stage,
                remote_stage,
                f"auto_research-starting-instruction-{instruction_digest}.txt",
                turn.request.instruction + "\n",
            )
        contract = auto_research_orchestrator_task_contract(
            project_name=context.project_name,
            graph_path=context.graph_path,
            research_path=context.research_md_path,
            repositories=repositories,
            patch_path=patch_path,
            output_schema_path=schema_path,
            validator_command=validator_command,
            command_client=command_client,
            instruction_path=instruction_path,
            messages_path=messages_path,
            lifecycle_path=lifecycle_path,
            skill_pointers=skill_pointers,
        )
        role = (
            "auto_research_orchestrator"
            if execution.continuation == "fresh"
            else "auto_research_orchestrator_retry"
        )
    else:
        retry_diagnostics_path = (
            _stage_json_task_input(
                local_stage,
                remote_stage,
                f"task-{token}-retry-diagnostics.json",
                {"prior_attempt_diagnostics": list(execution.retry_feedback)},
            )
            if execution.continuation == "retry"
            else None
        )
        contract = auto_research_orchestrator_continuation_contract(
            original_contract_path=_parent_task_contract_path(
                execution,
                local_stage,
                remote_stage,
            ),
            mode=(
                "resume"
                if execution.continuation == "resume"
                else "retry"
                if execution.continuation == "retry"
                else "continuation"
            ),
            graph_path=context.graph_path,
            research_path=context.research_md_path,
            repositories=repositories,
            patch_path=patch_path,
            output_schema_path=schema_path,
            validator_command=validator_command,
            command_client=command_client,
            messages_path=messages_path,
            lifecycle_path=lifecycle_path,
            retry_diagnostics_path=retry_diagnostics_path,
            skill_pointers=skill_pointers,
        )
        role = f"auto_research_orchestrator_{execution.continuation}"
    return _stage_task_contract(
        local_stage,
        remote_stage,
        f"task-{token}-auto_research-orchestrator.md",
        contract,
        execution=execution,
        role=role,
    )


def _worker_prompt(
    service: ProjectService,
    execution: AgentTaskExecution,
    turn: _CanonicalWorkerTurn,
    *,
    context: ChatContext,
    local_stage: Path | None,
    remote_stage: RemoteRunStage | None,
    token: str,
    patch_path: str,
    schema_path: str,
    validator_command: str,
    reply_command: str,
    messages_path: str | None,
) -> tuple[str, str]:
    actor = execution.store.agent_task(turn.binding.actor_operation_id)
    if actor is None:
        raise ValueError("AutoResearch worker origin task is missing.")
    actor_request = AutoResearchRunRequest.model_validate(actor.request)
    if not actor_request.instruction:
        raise ValueError("AutoResearch worker origin has no durable instruction.")
    instruction_digest = hashlib.sha256(actor_request.instruction.encode("utf-8")).hexdigest()[:16]
    instruction_path = _stage_or_reuse_task_input(
        local_stage,
        remote_stage,
        f"auto_research-worker-instruction-{instruction_digest}.txt",
        actor_request.instruction + "\n",
    )
    node = service.history.state().nodes.get(turn.binding.control_node_id or "")
    if node is None or node.type not in {"experiment", "blocker"}:
        raise ValueError("AutoResearch worker seat is no longer an Experiment or Blocker.")
    repositories = [
        {"alias": item.alias, "host": item.host, "path": item.path} for item in context.repositories
    ]
    if execution.continuation == "fresh":
        contract = auto_research_worker_task_contract(
            project_name=context.project_name,
            seat_node_type="Experiment" if node.type == "experiment" else "Blocker",
            seat_node_id=node.id,
            seat_difficulty=json.dumps(node.model_dump(mode="json"), ensure_ascii=False, indent=2),
            instruction_path=instruction_path,
            graph_path=context.graph_path,
            research_path=context.research_md_path,
            repositories=repositories,
            patch_path=patch_path,
            output_schema_path=schema_path,
            validator_command=validator_command,
            reply_command=reply_command,
            messages_path=messages_path,
        )
        role = "auto_research_worker"
    else:
        retry_diagnostics_path = (
            _stage_json_task_input(
                local_stage,
                remote_stage,
                f"task-{token}-retry-diagnostics.json",
                {"prior_attempt_diagnostics": list(execution.retry_feedback)},
            )
            if execution.continuation == "retry"
            else None
        )
        contract = auto_research_worker_continuation_contract(
            original_contract_path=_parent_task_contract_path(execution, local_stage, remote_stage),
            mode=(
                "resume"
                if execution.continuation == "resume"
                else "retry"
                if execution.continuation == "retry"
                else "continuation"
            ),
            graph_path=context.graph_path,
            research_path=context.research_md_path,
            repositories=repositories,
            patch_path=patch_path,
            output_schema_path=schema_path,
            validator_command=validator_command,
            reply_command=reply_command,
            messages_path=messages_path,
            retry_diagnostics_path=retry_diagnostics_path,
        )
        role = f"auto_research_worker_{execution.continuation}"
    return _stage_task_contract(
        local_stage,
        remote_stage,
        f"task-{token}-auto_research-worker.md",
        contract,
        execution=execution,
        role=role,
    )


async def _serve_worker_commands(
    staged: StagedCommandMailbox,
    *,
    execution: AgentTaskExecution,
    turn: _CanonicalWorkerTurn,
    dispatcher: AutoResearchCommandDispatcher,
    stop: asyncio.Event,
    expected_turn_id: str,
) -> None:
    await _serve_auto_research_commands(
        staged,
        execution=execution,
        turn=turn,
        dispatcher=dispatcher,
        stop=stop,
        expected_turn_id=expected_turn_id,
    )


async def _serve_auto_research_commands(
    staged: StagedCommandMailbox,
    *,
    execution: AgentTaskExecution,
    turn: _CanonicalWorkerTurn | _CanonicalOrchestratorTurn,
    dispatcher: AutoResearchCommandDispatcher,
    stop: asyncio.Event,
    expected_turn_id: str,
) -> None:
    async def handle(
        request: CommandRequest,
        identity: CommandTurnIdentity,
    ) -> CommandResponse:
        if (
            identity.episode_id != turn.request.episode_id
            or identity.task_id != execution.operation_id
            or identity.turn_id != expected_turn_id
        ):
            return CommandResponse(
                request_id=request.request_id,
                status="invalid",
                message="AutoResearch command credential does not match this actor turn.",
            )
        return await asyncio.to_thread(dispatcher.dispatch, execution.operation_id, request)

    await serve_command_mailbox(
        staged=staged,
        handler=handle,
        stop=stop,
        invocation_gate=staged.invocation_gate,
    )


class _ValidateOnlyAutoResearchCommandDispatcher(AutoResearchCommandDispatcher):
    """Keep dispatcher ledger semantics while denying every correction side effect."""

    def dispatch(self, operation_id: str, request: CommandRequest) -> CommandResponse:
        if not isinstance(request, ValidateCommandRequest):
            context = self._context(operation_id)
            invocation = self.store.start_agent_command(
                operation_id=operation_id,
                command_id=self._unused_command_id(request.request_id),
                episode_id=context.episode.episode_id,
                verb=request.verb,
                idempotency_key=None,
                payload={
                    "request_id": request.request_id,
                    "arguments": request.arguments.model_dump(mode="json"),
                    "supplied_idempotency_key": request.idempotency_key,
                    "denied_by": "auto_research_patch_correction_validate_only",
                },
            )
            return self._finish(
                invocation.command_id,
                request.request_id,
                AutoResearchCommandEffectResult(
                    status="invalid",
                    message=(
                        "AutoResearch graph-correction credentials authorize Patch validation only."
                    ),
                ),
            )
        return super().dispatch(operation_id, request)

    def _execute(
        self,
        context: AutoResearchCommandContext,
        request: CommandRequest,
        *,
        planned_worker_id: str | None,
        planned_message_id: str | None,
        planned_watcher_id: str | None,
        planned_apply_id: str | None,
        planned_resume_operation_id: str | None,
        planned_episode_effect_id: str | None,
        planned_inbox_effect_id: str | None,
        planned_finish_effect_id: str | None,
    ) -> AutoResearchCommandEffectResult:
        if not isinstance(request, ValidateCommandRequest):
            raise AutoResearchCommandInvalid(
                "AutoResearch graph-correction credentials authorize Patch validation only."
            )
        return super()._execute(
            context,
            request,
            planned_worker_id=planned_worker_id,
            planned_message_id=planned_message_id,
            planned_watcher_id=planned_watcher_id,
            planned_apply_id=planned_apply_id,
            planned_resume_operation_id=planned_resume_operation_id,
            planned_episode_effect_id=planned_episode_effect_id,
            planned_inbox_effect_id=planned_inbox_effect_id,
            planned_finish_effect_id=planned_finish_effect_id,
        )


@asynccontextmanager
async def _worker_mailbox_lifecycle(
    staged: StagedCommandMailbox,
    *,
    execution: AgentTaskExecution,
    start: Callable[[asyncio.Event], Awaitable[None]],
) -> AsyncIterator[None]:
    """Own one staged mailbox from server setup through fail-closed cleanup."""

    stop: asyncio.Event | None = None
    task: asyncio.Task[None] | None = None
    primary_error: BaseException | None = None
    try:
        stop = asyncio.Event()
        task = asyncio.create_task(start(stop))
        yield
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        await _close_worker_mailbox(
            staged,
            stop=stop,
            task=task,
            execution=execution,
            primary_error=primary_error,
        )


async def _wait_for_owned_task(
    task: asyncio.Task[None],
) -> tuple[BaseException | None, asyncio.CancelledError | None]:
    """Wait without allowing caller cancellation to abandon an owned task."""

    caller_cancelled: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if caller_cancelled is None:
                caller_cancelled = exc
        except BaseException:
            break
    try:
        task.result()
    except BaseException as exc:
        return exc, caller_cancelled
    return None, caller_cancelled


async def _close_worker_mailbox(
    staged: StagedCommandMailbox,
    *,
    stop: asyncio.Event | None,
    task: asyncio.Task[None] | None,
    execution: AgentTaskExecution,
    primary_error: BaseException | None = None,
) -> None:
    if stop is not None:
        stop.set()

    serve_error: BaseException | None = None
    caller_cancelled: asyncio.CancelledError | None = None
    if task is not None:
        serve_error, caller_cancelled = await _wait_for_owned_task(task)

    cleanup_task = asyncio.create_task(asyncio.to_thread(staged.cleanup))
    cleanup_error, cleanup_cancelled = await _wait_for_owned_task(cleanup_task)
    if caller_cancelled is None:
        caller_cancelled = cleanup_cancelled

    def warning(message: str) -> None:
        with suppress(Exception):
            execution.store.record_agent_task_event(
                execution.operation_id,
                message,
                level="warning",
            )

    expected_errors = (OSError, StateUnavailable, ValueError)
    if primary_error is not None:
        if serve_error is not None:
            warning(f"AutoResearch command mailbox became unavailable: {serve_error}")
        if cleanup_error is not None:
            warning(f"AutoResearch command mailbox cleanup failed: {cleanup_error}")
        return

    if caller_cancelled is not None:
        if serve_error is not None and not isinstance(serve_error, asyncio.CancelledError):
            warning(f"AutoResearch command mailbox became unavailable: {serve_error}")
        if cleanup_error is not None and not isinstance(cleanup_error, asyncio.CancelledError):
            warning(f"AutoResearch command mailbox cleanup failed: {cleanup_error}")
        raise caller_cancelled

    if serve_error is not None:
        if isinstance(serve_error, expected_errors):
            warning(f"AutoResearch command mailbox became unavailable: {serve_error}")
        else:
            if cleanup_error is not None:
                warning(f"AutoResearch command mailbox cleanup failed: {cleanup_error}")
            raise serve_error
    if cleanup_error is not None:
        if isinstance(cleanup_error, expected_errors):
            warning(f"AutoResearch command mailbox cleanup failed: {cleanup_error}")
        else:
            raise cleanup_error


async def _settle_worker_patch(
    service: ProjectService,
    launcher: AgentLauncher,
    execution: AgentTaskExecution,
    turn: _CanonicalWorkerTurn | _CanonicalOrchestratorTurn,
    stage: _WorkerStage,
    *,
    contract_path: str,
    patch_path: str,
    schema_path: str,
    read_dirs: list[Path],
    write_dirs: list[Path],
    write_scope: ProjectWriteScope,
    provider_binary: str | None,
    native_session_id: str,
    retry_patch_digest: str | None,
    command_dispatcher: AutoResearchCommandDispatcher,
    _actor_role: Literal["worker", "orchestrator"] = "worker",
    _profile: Literal["ordinary", "orchestrator"] = "ordinary",
    _capability: Literal["work_auto", "orchestrate"] = "work_auto",
) -> _PatchSettlement:
    try:
        patch_text = _read_chat_patch(stage.workspace, stage.remote)
    except (OSError, StateUnavailable, ValueError) as exc:
        patch_text = None
        failure: _WorkPatchFailure | None = _WorkPatchFailure(
            f"The auto_research {_actor_role} wrote a patch file that could not be read: {exc}",
            correctable=False,
        )
    else:
        failure = None
    if _retry_deliverable_is_unchanged(
        execution,
        filename="patch.json",
        predecessor_digest=retry_patch_digest,
        current_text=patch_text,
    ):
        patch_text = None
    had_patch = patch_text is not None or failure is not None
    if patch_text is None and failure is None:
        return _PatchSettlement(GraphUpdateResult(status="none"))

    correction_rounds = 0
    correction_frames: list[str] = []
    while True:
        if patch_text is not None:
            source_effect_id: str | None = None
            recovered_apply = False
            patch_digest = hashlib.sha256(patch_text.encode("utf-8")).hexdigest()
            if _profile == "orchestrator":
                source_effect_id, recovered_apply = _orchestrator_final_source_effect_id(
                    service,
                    execution,
                    patch_digest=patch_digest,
                )
            try:
                result, failure = _apply_work_patch(
                    service,
                    execution,
                    patch_text,
                    run_truth_scope=turn.request.run_truth_scope
                    or service.manifest.agent.default_run_truth_scope,
                    profile=_profile,
                    source_effect_id=source_effect_id,
                )
                if result is not None and recovered_apply:
                    consumer = command_dispatcher.command_file_consumer
                    if consumer is None:
                        raise ValueError(
                            "The retained in-turn Apply Patch could not be consumed during "
                            "final settlement."
                        )
                    if not consumer("patch.json", patch_digest):
                        raise ValueError(
                            "The retained in-turn Apply Patch changed before final settlement."
                        )
            except RunLockCancelled:
                correction_frames.append(
                    _sse(
                        AgentEvent(
                            event="paused",
                            text=(
                                f"Paused while waiting for canonical state. The auto_research "
                                f"{_actor_role} "
                                "answer and retained patch are preserved."
                            ),
                        )
                    )
                )
                return _PatchSettlement(None, tuple(correction_frames), had_patch=had_patch)
            if result is not None:
                return _PatchSettlement(
                    result.model_copy(update={"correction_rounds": correction_rounds}),
                    tuple(correction_frames),
                    had_patch=had_patch,
                )
        assert failure is not None
        if (
            not failure.correctable
            or correction_rounds >= PATCH_CORRECTION_MAX_ROUNDS
            or not native_session_id
        ):
            rejected = GraphUpdateResult(
                status="rejected",
                change_summary=list(failure.change_summary),
                proposal_ids=list(failure.proposal_ids),
                validation_messages=_bounded_graph_messages(failure.message),
                correction_rounds=correction_rounds,
                repairable=False,
            )
            _record_work_graph_rejection(execution, rejected)
            return _PatchSettlement(rejected, tuple(correction_frames), had_patch=had_patch)

        correction_rounds += 1
        execution.store.record_agent_task_receipt(
            execution.operation_id,
            "patch_correction_requested",
            {"round": correction_rounds, "problem": failure.message[:400]},
            tier="diagnostic",
        )
        execution.store.update_agent_task_message(
            execution.operation_id,
            f"Correcting auto_research {_actor_role} graph reflection.",
            phase="correcting",
            event=True,
        )
        token = _task_token(execution)
        diagnostics_path = _stage_json_task_input(
            stage.local,
            stage.remote,
            f"task-{token}-auto_research-work-correction-{correction_rounds}.json",
            {"kind": "work", "problem": failure.message},
        )
        correction_mailbox = stage_command_mailbox(
            local_stage=stage.local,
            remote_stage=stage.remote,
            episode_id=turn.request.episode_id,
            task_id=execution.operation_id,
            turn_id=(
                f"{execution.operation_id}:{_actor_role}-patch-correction:{correction_rounds}"
            ),
        )
        async with _worker_mailbox_lifecycle(
            correction_mailbox,
            execution=execution,
            start=lambda stop, mailbox=correction_mailbox, round_number=correction_rounds: (
                _serve_worker_commands if _actor_role == "worker" else _serve_auto_research_commands
            )(
                mailbox,
                execution=execution,
                turn=turn,
                dispatcher=_ValidateOnlyAutoResearchCommandDispatcher(
                    command_dispatcher.store,
                    command_dispatcher.effects,
                ),
                stop=stop,
                expected_turn_id=(
                    f"{execution.operation_id}:{_actor_role}-patch-correction:{round_number}"
                ),
            ),
        ):
            correction_validator_command = correction_mailbox.client_command("validate", patch_path)
            correction_contract = PromptFactory.continuation_task_contract(
                original_contract_path=contract_path,
                mode="work_patch_correction",
                patch_path=patch_path,
                diagnostics_path=diagnostics_path,
                validator_command=correction_validator_command,
                output_schema_path=schema_path,
            )
            correction_path, correction_prompt = _stage_task_contract(
                stage.local,
                stage.remote,
                f"task-{token}-auto_research-work-correction-{correction_rounds}.md",
                correction_contract,
                execution=execution,
                role=f"auto_research_{_actor_role}_patch_correction_{correction_rounds}",
            )
            pre_launch_digest = _existing_patch_digest(stage.workspace, stage.remote)
            _record_agent_launch_receipt(
                execution,
                cast(RunRequest, turn.request),
                prompt=correction_prompt,
                contract_path=correction_path,
                remote=bool(stage.execution_host),
                resumed=True,
                write_scope=write_scope,
                continuation="graph_correction",
                extra={
                    "surface": "auto_research",
                    "role": _actor_role,
                    "profile": _profile,
                    "capability": _capability,
                    "network_access": True,
                    "launch_kind": "graph_correction",
                    "correction_round": correction_rounds,
                    "write_directory_count": len(write_dirs),
                    "canonical_state_boundary": "prompt_only",
                    "repeat_operational_work": False,
                },
            )
            correction_outcome = _ProviderOutcome(session_id=native_session_id)
            correction_error: str | None = None
            async with aclosing(
                _stream_agent_events(
                    launcher,
                    cast(RunRequest, turn.request),
                    correction_prompt,
                    workspace=stage.workspace,
                    session_id=native_session_id,
                    read_dirs=read_dirs,
                    write_dirs=write_dirs,
                    write_scope=write_scope,
                    execution_host=stage.execution_host,
                    execution=execution,
                    remote_stage=stage.remote,
                    capability=_capability,
                    outcome=correction_outcome,
                    binary=provider_binary,
                    invocation_gate=correction_mailbox.invocation_gate,
                )
            ) as stream:
                async for frame in stream:
                    event = AgentEvent.model_validate_json(frame.removeprefix("data: ").strip())
                    if event.event == "error":
                        correction_error = (
                            event.text or f"AutoResearch {_actor_role} Patch correction failed."
                        )
                    else:
                        correction_frames.append(frame)
        if correction_outcome.paused:
            return _PatchSettlement(None, tuple(correction_frames), had_patch=had_patch)
        if (
            correction_error is not None
            or correction_outcome.failed
            or not correction_outcome.completed
            or correction_outcome.session_id != native_session_id
        ):
            failure = _WorkPatchFailure(
                correction_error
                or (
                    f"AutoResearch {_actor_role} Patch correction did not complete in its saved "
                    "session."
                ),
                correctable=True,
                change_summary=failure.change_summary,
                proposal_ids=failure.proposal_ids,
            )
            patch_text = None
            correction_rounds = PATCH_CORRECTION_MAX_ROUNDS
            continue
        corrected: _CorrectionPatchRead = _read_correction_patch(
            stage.workspace,
            stage.remote,
            pre_launch_digest=pre_launch_digest,
        )
        if corrected.problem == "unreadable":
            failure = _WorkPatchFailure(
                f"The corrected patch could not be read: {corrected.detail}",
                correctable=True,
                change_summary=failure.change_summary,
                proposal_ids=failure.proposal_ids,
            )
            patch_text = None
        elif corrected.problem == "missing":
            failure = _WorkPatchFailure(
                "The correction completed without writing patch.json.",
                correctable=True,
                change_summary=failure.change_summary,
                proposal_ids=failure.proposal_ids,
            )
            patch_text = None
        elif corrected.problem == "unchanged":
            failure = _WorkPatchFailure(
                f"{failure.message} The correction left patch.json byte-identical.",
                correctable=True,
                change_summary=failure.change_summary,
                proposal_ids=failure.proposal_ids,
            )
            patch_text = None
        else:
            assert corrected.text is not None
            patch_text = corrected.text


def _orchestrator_final_source_effect_id(
    service: ProjectService,
    execution: AgentTaskExecution,
    *,
    patch_digest: str,
) -> tuple[str, bool]:
    """Reuse an in-turn Apply commit when its exact handoff survives to settlement."""

    matches = [
        patch
        for patch in service.history.load_patches()
        if patch.admission == "accepted"
        and patch.kind == "work"
        and patch.source_operation_id == execution.operation_id
        and patch.source_effect_sha256 == patch_digest
    ]
    if len(matches) > 1:
        raise ValueError(
            "The retained orchestrator Patch matches multiple canonical in-turn Apply commits."
        )
    if matches:
        source_effect_id = matches[0].source_effect_id
        if source_effect_id is None:
            raise ValueError(
                "The retained orchestrator Patch matches a canonical commit without an effect id."
            )
        return source_effect_id, True
    return (
        str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                "rcp:auto-research-final-apply:" + execution.operation_id,
            )
        ),
        False,
    )


async def _settle_orchestrator_patch(
    service: ProjectService,
    launcher: AgentLauncher,
    execution: AgentTaskExecution,
    turn: _CanonicalOrchestratorTurn,
    stage: _WorkerStage,
    *,
    contract_path: str,
    patch_path: str,
    schema_path: str,
    read_dirs: list[Path],
    write_dirs: list[Path],
    write_scope: ProjectWriteScope,
    provider_binary: str | None,
    native_session_id: str,
    retry_patch_digest: str | None,
    command_dispatcher: AutoResearchCommandDispatcher,
) -> _PatchSettlement:
    return await _settle_worker_patch(
        service,
        launcher,
        execution,
        turn,
        stage,
        contract_path=contract_path,
        patch_path=patch_path,
        schema_path=schema_path,
        read_dirs=read_dirs,
        write_dirs=write_dirs,
        write_scope=write_scope,
        provider_binary=provider_binary,
        native_session_id=native_session_id,
        retry_patch_digest=retry_patch_digest,
        command_dispatcher=command_dispatcher,
        _actor_role="orchestrator",
        _profile="orchestrator",
        _capability="orchestrate",
    )
