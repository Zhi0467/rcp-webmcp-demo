from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import aclosing, suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel

from rcp.agents import (
    AgentEvent,
    AgentLauncher,
    PromptFactory,
    RunContext,
    agent_output_schema,
    prepare_agent_patch,
    validate_agent_patch_shape,
)
from rcp.agents.command_mailbox import StagedCommandMailbox
from rcp.background import AgentTaskExecution
from rcp.config import AgentSurface
from rcp.history import PatchRejected, ReplayHalted
from rcp.limits import PATCH_CORRECTION_MAX_ROUNDS, PATCH_SELF_CHECK_TIMEOUT_SECONDS
from rcp.providers import classify_terminal_error
from rcp.runs.patch_validator import (
    PatchValidationBudget,
    PatchValidationResult,
    cleanup_patch_validation_mailbox,
    serve_patch_validation_mailbox,
    stage_patch_validation_mailbox,
)
from rcp.runs.shared import (
    AgentOutputProblem,
    _collect_patch_text,
    _existing_patch_digest,
    _parent_task_contract_path,
    _pinned_to_profile,
    _ProviderOutcome,
    _record_agent_launch_receipt,
    _record_patch_applied_receipt,
    _record_patch_receipt,
    _record_provider_exit,
    _remove_local_tree,
    _safe_stage_name,
    _sse,
    _stage_context_paths,
    _stage_json_task_input,
    _stage_task_contract,
    _stage_task_input,
    _stream_agent_events,
    _swept_stage_root,
    _task_token,
)
from rcp.service import ProjectService, RunRequest
from rcp.skills.staging import stage_skill_selection
from rcp.sources import ImportedProviderSourceInventory
from rcp.storage import AgentTaskRecord
from rcp.transport import (
    ImportedProviderSourceReadback,
    RemoteRunStage,
    RunLockCancelled,
    RunLockLease,
    RunLockOwnershipLost,
    StateUnavailable,
)

logger = logging.getLogger(__name__)
_PREPARED_GRAPH_CONTEXT_FILE = "prepared-context.json"
_IMPORTED_PROVIDER_SOURCES_LABEL = "imported-provider-history"


class _PreparedGraphContext(BaseModel):
    version: Literal[2] = 2
    project_id: str
    kind: Literal["seed", "refresh"]
    graph_revision: int
    run_truth_scope: list[str]
    execution_host: str
    original_contract_path: str | None = None
    context: RunContext


@dataclass(frozen=True)
class _GraphRetryState:
    lineage: tuple[AgentTaskRecord, ...]
    prepared: _PreparedGraphContext | None
    prepared_parent: AgentTaskRecord | None
    progress_parent: AgentTaskRecord | None
    progress: dict[str, object]
    retained_patch_text: str | None = None
    context_reason: str | None = None
    progress_reason: str | None = None
    imported_source_readback: ImportedProviderSourceReadback | None = None


def _remote_imported_source_roots(
    inventory: ImportedProviderSourceInventory | None,
    staged_root: str,
) -> dict[str, list[str]]:
    if inventory is None or not inventory.files:
        return {}
    return inventory.roots(Path(staged_root))


def _verify_prepared_imported_sources(
    service: ProjectService,
    context: RunContext,
    inventory: ImportedProviderSourceInventory | None,
    parent: AgentTaskRecord,
) -> ImportedProviderSourceReadback | None:
    if not parent.stage_host:
        service.validate_imported_source_context(context, inventory)
        return None
    if inventory is None or not inventory.files:
        service.validate_imported_source_context(context, inventory, expected_roots={})
        return None
    if not parent.stage_root:
        raise ValueError("the prior attempt has no retained imported-source stage")
    staged_root = str(
        PurePosixPath(parent.stage_root) / "inputs" / _IMPORTED_PROVIDER_SOURCES_LABEL
    )
    service.validate_imported_source_context(
        context,
        inventory,
        expected_roots=_remote_imported_source_roots(inventory, staged_root),
    )
    stage = RemoteRunStage(parent.stage_host)
    stage_available = stage.directory_exists(parent.stage_root)
    if stage_available is None:
        raise StateUnavailable("could not reach the saved remote staging directory")
    if not stage_available:
        raise ValueError("the saved remote staging directory is unavailable")
    stage.attach(parent.stage_root)
    return stage.verify_imported_provider_sources(
        inventory,
        _IMPORTED_PROVIDER_SOURCES_LABEL,
    )


def _read_prepared_graph_context(parent: AgentTaskRecord) -> _PreparedGraphContext:
    if not parent.stage_root:
        raise ValueError("the prior attempt has no retained stage")
    if parent.stage_host:
        stage = RemoteRunStage(parent.stage_host)
        stage_available = stage.directory_exists(parent.stage_root)
        if stage_available is None:
            raise StateUnavailable("could not reach the saved remote staging directory")
        if not stage_available:
            raise ValueError("the prior attempt has no retained stage")
        stage.attach(parent.stage_root)
        assert stage.root is not None
        raw = stage.read_input_text(_PREPARED_GRAPH_CONTEXT_FILE)
    else:
        root = Path(parent.stage_root).resolve()
        path = (root / "inputs" / _PREPARED_GRAPH_CONTEXT_FILE).resolve()
        if path.parent != (root / "inputs").resolve() or not path.is_file():
            raise ValueError("the prior attempt has no prepared context metadata")
        raw = path.read_text(encoding="utf-8")
    return _PreparedGraphContext.model_validate_json(raw)


def _retry_lineage(execution: AgentTaskExecution | None) -> list[AgentTaskRecord]:
    if execution is None or execution.reuses_native_checkpoint:
        return []
    current = execution.store.agent_task(execution.operation_id)
    if current is None or current.parent_operation_id is None:
        return []
    lineage: list[AgentTaskRecord] = []
    seen = {current.operation_id}
    parent_id = current.parent_operation_id
    while parent_id and parent_id not in seen:
        seen.add(parent_id)
        parent = execution.store.agent_task(parent_id)
        if parent is None or parent.project_id != current.project_id or parent.kind != current.kind:
            break
        lineage.append(parent)
        parent_id = parent.parent_operation_id
    return lineage


def _continuation_graph_context(
    service: ProjectService,
    execution: AgentTaskExecution,
    *,
    kind: str,
    request: RunRequest,
    execution_host: str,
    imported_source_inventory: ImportedProviderSourceInventory | None,
) -> _PreparedGraphContext:
    """Load the immutable context owned by a native-session continuation.

    Resume and same-provider Retry continue a provider process in its
    original stage. Reassembling here would silently give that process a
    different graph and different evidence than the contract it is continuing.
    """
    record = execution.store.agent_task(execution.operation_id)
    if record is None:
        raise ValueError("The saved continuation task is unavailable. Retry this task.")
    try:
        prepared = _read_prepared_graph_context(record)
    except StateUnavailable:
        raise
    except (OSError, ValueError) as exc:
        reason = " ".join(str(exc).split())[:400]
        execution.store.record_agent_task_receipt(
            execution.operation_id,
            "continuation_context_unavailable",
            {"reason": reason, "retry_required": True},
            tier="diagnostic",
        )
        raise ValueError(
            f"The saved prepared context is unavailable ({exc}). Retry this task."
        ) from exc
    expected_scope = sorted(
        request.run_truth_scope or service.manifest.agent.default_run_truth_scope
    )
    current_revision = int(service.graph_snapshot()["revision"])
    problems: list[str] = []
    if prepared.project_id != record.project_id:
        problems.append("project identity changed")
    if prepared.kind != kind:
        problems.append("task kind changed")
    if sorted(prepared.run_truth_scope) != expected_scope:
        problems.append("run truth scope changed")
    if prepared.execution_host != execution_host or record.stage_host != (execution_host or None):
        problems.append("execution host changed")
    if prepared.graph_revision != current_revision:
        problems.append(
            f"graph revision moved from {prepared.graph_revision} to {current_revision}"
        )
    imported_source_readback = None
    try:
        imported_source_readback = _verify_prepared_imported_sources(
            service,
            prepared.context,
            imported_source_inventory,
            record,
        )
    except ValueError as exc:
        problems.append(str(exc))
    if problems:
        reason = "; ".join(problems)
        execution.store.record_agent_task_receipt(
            execution.operation_id,
            "continuation_context_unavailable",
            {"reason": reason, "retry_required": True},
            tier="diagnostic",
        )
        raise ValueError(
            f"The saved prepared context no longer matches ({reason}). Retry this task."
        )
    if imported_source_readback is not None:
        _record_imported_source_stage_receipt(
            execution,
            imported_source_readback,
            reused=True,
        )
    return prepared


def _try_reuse_graph_context(
    service: ProjectService,
    execution: AgentTaskExecution | None,
    *,
    kind: str,
    request: RunRequest,
    execution_host: str,
    imported_source_inventory: ImportedProviderSourceInventory | None,
) -> _GraphRetryState | None:
    lineage = _retry_lineage(execution)
    if not lineage or execution is None:
        return None
    expected_scope = sorted(
        request.run_truth_scope or service.manifest.agent.default_run_truth_scope
    )
    graph_revision = int(service.graph_snapshot()["revision"])
    prepared = None
    prepared_parent = None
    imported_source_readback = None
    context_errors: list[str] = []
    for candidate in lineage:
        try:
            value = _read_prepared_graph_context(candidate)
            if kind not in {"seed", "refresh"} or value.kind != kind:
                raise ValueError("task kind changed")
            if value.project_id != candidate.project_id:
                raise ValueError("project identity changed")
            if sorted(value.run_truth_scope) != expected_scope:
                raise ValueError("run truth scope changed")
            if value.execution_host != execution_host or candidate.stage_host != (
                execution_host or None
            ):
                raise ValueError("execution host changed")
            if value.graph_revision != graph_revision:
                raise ValueError("graph revision changed")
            imported_source_readback = _verify_prepared_imported_sources(
                service,
                value.context,
                imported_source_inventory,
                candidate,
            )
            prepared = value
            prepared_parent = candidate
            break
        except (OSError, ValueError) as exc:
            context_errors.append(f"attempt {candidate.attempt}: {exc}")

    progress_parent = None
    progress: dict[str, object] = {}
    retained_patch_text = None
    progress_errors: list[str] = []
    for candidate in lineage:
        retained_patch_text = execution.store.agent_task_patch_output(candidate.operation_id)
        if retained_patch_text:
            progress_parent = candidate
            progress = {
                "prior_operation_id": candidate.operation_id,
                "prior_attempt": candidate.attempt,
                "prior_provider": candidate.request.get("provider"),
                "prior_error": candidate.error,
            }
            if candidate.native_session_id:
                progress["native_session_id"] = candidate.native_session_id
            break
        progress_errors.append(f"attempt {candidate.attempt}: no retained provider progress")

    if imported_source_readback is not None:
        _record_imported_source_stage_receipt(
            execution,
            imported_source_readback,
            reused=True,
        )
    return _GraphRetryState(
        lineage=tuple(lineage),
        prepared=prepared,
        prepared_parent=prepared_parent,
        progress_parent=progress_parent,
        progress=progress,
        retained_patch_text=retained_patch_text if progress_parent else None,
        context_reason="; ".join(context_errors)[:1200] if prepared is None else None,
        progress_reason="; ".join(progress_errors)[:1200] if progress_parent is None else None,
        imported_source_readback=imported_source_readback,
    )


def _record_context_reuse(
    execution: AgentTaskExecution | None,
    *,
    reused: bool,
    reason: str | None = None,
) -> None:
    if execution is None:
        return
    category = "context_reused" if reused else "context_reuse_unavailable"
    payload = {"reused": reused}
    if reason:
        payload["reason"] = " ".join(reason.split())[:400]
    execution.store.record_agent_task_receipt(
        execution.operation_id, category, payload, tier="diagnostic"
    )
    execution.store.record_agent_task_event(
        execution.operation_id,
        (
            "Reusing the prior attempt's prepared context."
            if reused
            else (
                "Prepared context could not be reused; rebuilding it. "
                f"Reason: {' '.join(reason.split())[:400]}"
                if reason
                else "Prepared context could not be reused; rebuilding it."
            )
        ),
        level="info" if reused else "warning",
    )


def _record_progress_handoff(
    execution: AgentTaskExecution | None,
    *,
    handed_off: bool,
    source: AgentTaskRecord | None = None,
    reason: str | None = None,
) -> None:
    if execution is None:
        return
    payload: dict[str, object] = {"handed_off": handed_off}
    if source is not None:
        payload.update(
            {
                "source_operation_id": source.operation_id,
                "source_attempt": source.attempt,
                "source_provider": source.request.get("provider"),
            }
        )
    if reason:
        payload["reason"] = " ".join(reason.split())[:400]
    execution.store.record_agent_task_receipt(
        execution.operation_id,
        "progress_handed_off" if handed_off else "progress_handoff_unavailable",
        payload,
        tier="diagnostic",
    )
    execution.store.record_agent_task_event(
        execution.operation_id,
        (
            f"Handing off provider progress from attempt {source.attempt}."
            if handed_off and source is not None
            else (
                "No prior provider progress was handed off. "
                f"Reason: {' '.join(reason.split())[:400]}"
                if reason
                else "No prior provider progress was handed off."
            )
        ),
        level="info" if handed_off else "warning",
    )


def _stage_prepared_graph_context(
    local_stage: Path | None,
    remote_stage: RemoteRunStage | None,
    *,
    project_id: str,
    kind: str,
    graph_revision: int,
    execution_host: str,
    original_contract_path: str,
    context: RunContext,
) -> None:
    prepared = _PreparedGraphContext(
        project_id=project_id,
        kind=kind,
        graph_revision=graph_revision,
        run_truth_scope=context.run_truth_scope,
        execution_host=execution_host,
        original_contract_path=original_contract_path,
        context=context,
    )
    _stage_json_task_input(
        local_stage,
        remote_stage,
        _PREPARED_GRAPH_CONTEXT_FILE,
        prepared.model_dump(mode="json"),
    )


async def stream_graph_run(
    service: ProjectService,
    launcher: AgentLauncher,
    kind: str,
    request: RunRequest,
    data_dir: Path,
    execution: AgentTaskExecution | None = None,
) -> AsyncIterator[str]:
    continuation = execution.continuation if execution is not None else "fresh"
    reuses_native_checkpoint = bool(execution is not None and execution.reuses_native_checkpoint)
    if request.session_id and not reuses_native_checkpoint:
        yield _sse(
            AgentEvent(
                event="error",
                text=(
                    "Seed and refresh sessions can only be resumed from an RCP background "
                    "task checkpoint."
                ),
            )
        )
        return
    surface: AgentSurface = "seed" if kind == "seed" else "refresh"
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
    try:
        imported_source_inventory = await asyncio.to_thread(
            service.imported_source_inventory,
            surface,
            execution_machine,
        )
    except (OSError, ValueError) as exc:
        yield _sse(AgentEvent(event="error", text=str(exc)))
        return
    workspace = service.history.workspace
    canonical_state_location = workspace.location
    run_lock = workspace.run_lock(
        on_wait=(
            lambda message: _record_run_lock_wait(execution, message, canonical_state_location)
        )
        if execution is not None
        else None,
        cancelled=(execution.control.pause_requested.is_set if execution is not None else None),
        on_lost=(
            lambda message: _record_run_lock_lost(execution, message, canonical_state_location)
        )
        if execution is not None
        else None,
    )
    run_lock_lease: RunLockLease | None = None
    run_lock_acquired = False
    applied = False
    retry_state: _GraphRetryState | None = None
    graph_revision = 0
    imported_stage_verified = False
    validator_budget = PatchValidationBudget()
    validator_staged: StagedCommandMailbox | None = None
    try:
        try:
            run_lock_lease = run_lock.__enter__()
            run_lock_acquired = True
            run_lock_lease.assert_owned()
        except RunLockCancelled:
            yield _sse(AgentEvent(event="paused", text="Paused while waiting for canonical state."))
            return
        except RunLockOwnershipLost as exc:
            yield _sse(
                AgentEvent(
                    event="paused",
                    text=f"{exc} The task paused before applying further graph changes.",
                )
            )
            return
        except StateUnavailable as exc:
            if execution is not None and execution.control.pause_requested.is_set():
                yield _sse(
                    AgentEvent(event="paused", text="Paused while waiting for canonical state.")
                )
            else:
                yield _sse(AgentEvent(event="error", text=str(exc)))
            return
        try:
            continuation_prepared = (
                _continuation_graph_context(
                    service,
                    execution,
                    kind=kind,
                    request=request,
                    execution_host=execution_host,
                    imported_source_inventory=imported_source_inventory,
                )
                if execution is not None and reuses_native_checkpoint
                else None
            )
            retry_state = (
                None
                if continuation_prepared is not None
                else _try_reuse_graph_context(
                    service,
                    execution,
                    kind=kind,
                    request=request,
                    execution_host=execution_host,
                    imported_source_inventory=imported_source_inventory,
                )
            )
            if continuation_prepared is not None:
                context = continuation_prepared.context
                graph_revision = continuation_prepared.graph_revision
                _record_context_reuse(execution, reused=True)
                imported_stage_verified = bool(
                    imported_source_inventory is not None and imported_source_inventory.files
                )
            elif retry_state is not None and retry_state.prepared is not None:
                context = retry_state.prepared.context
                graph_revision = retry_state.prepared.graph_revision
                _record_context_reuse(execution, reused=True)
                imported_stage_verified = retry_state.imported_source_readback is not None
            else:
                if retry_state is not None:
                    _record_context_reuse(
                        execution, reused=False, reason=retry_state.context_reason
                    )
                context = service.assemble_run(
                    request,
                    surface,
                    imported_source_inventory=imported_source_inventory,
                )
                _record_context_receipt(execution, context, surface=surface)
                _report_source_errors(execution, context.source_errors)
                graph_revision = context.graph_revision
            # One scratch folder per operation, reused by every rung of the recovery
            # ladder so a resumed native session still points at the directory it was
            # originally given. It is never deleted on failure; _sweep_stale_stages
            # ages it out instead.
            if execution_host:
                if request.session_id and not reuses_native_checkpoint:
                    raise ValueError(
                        "Remote native-session resume needs persistent run staging; "
                        "start this chat on the local execution machine."
                    )
                if reuses_native_checkpoint and execution is not None and execution.stage_root:
                    remote_stage = RemoteRunStage(execution_host).attach(execution.stage_root)
                elif reuses_native_checkpoint:
                    raise ValueError(
                        "The interrupted remote operation has no staging checkpoint; retry it."
                    )
                else:
                    remote_stage = RemoteRunStage(execution_host).open(
                        execution.operation_id if execution is not None else None
                    )
                    if execution is not None:
                        assert remote_stage.root is not None
                        execution.checkpoint_stage(execution_host, str(remote_stage.root))
                    context = await asyncio.to_thread(
                        _stage_graph_context,
                        context,
                        service,
                        remote_stage,
                        execution_machine.alias,
                        imported_source_inventory=imported_source_inventory,
                    )
                    if imported_source_inventory is not None and imported_source_inventory.files:
                        imported_stage_verified = False
                workspace = Path(str(remote_stage.workspace))
                patch_path = str(remote_stage.workspace / "patch.json")
            else:
                stage_root = _swept_stage_root(data_dir)
                if reuses_native_checkpoint and execution is not None and execution.stage_root:
                    local_stage = Path(execution.stage_root).resolve()
                    if local_stage.parent != stage_root.resolve() or not local_stage.is_dir():
                        raise ValueError(
                            "The interrupted local operation has no valid staging checkpoint; "
                            "retry it instead."
                        )
                elif reuses_native_checkpoint:
                    raise ValueError(
                        "The interrupted local operation has no staging checkpoint; retry it."
                    )
                else:
                    name = execution.operation_id if execution is not None else uuid.uuid4().hex
                    local_stage = stage_root / _safe_stage_name(name)
                    local_stage.mkdir(parents=True, exist_ok=True)
                    if execution is not None:
                        execution.checkpoint_stage("", str(local_stage))
                workspace = local_stage
                patch_path = str(local_stage / "patch.json")

            skill_pointers = stage_skill_selection(
                service.resolve_skill_selection(request),
                local_stage=local_stage,
                remote_stage=remote_stage,
                label=f"rcp-skills-{_task_token(execution)}",
            )
            token = _task_token(execution)
            validator_staged = stage_patch_validation_mailbox(
                local_stage=local_stage,
                remote_stage=remote_stage,
                task_id=execution.operation_id if execution is not None else token,
                turn_id=f"{token}:graph-pass:0",
                timeout_seconds=PATCH_SELF_CHECK_TIMEOUT_SECONDS,
            )
            validator_command = validator_staged.client_command("validate", patch_path)
            read_dirs = _agent_read_dirs(context, remote_stage, service, execution_machine.alias)
            if (
                retry_state is not None
                and retry_state.prepared is not None
                and retry_state.prepared_parent is not None
                and retry_state.prepared_parent.stage_root
            ):
                parent_inputs = (
                    PurePosixPath(retry_state.prepared_parent.stage_root) / "inputs"
                    if execution_host
                    else Path(retry_state.prepared_parent.stage_root) / "inputs"
                )
                read_dirs.append(Path(str(parent_inputs)))
            if reuses_native_checkpoint and continuation == "resume":
                if not request.session_id:
                    raise ValueError(
                        "The interrupted operation has no native agent session; retry it instead."
                    )
                assert execution is not None
                original_contract_path = _parent_task_contract_path(
                    execution, local_stage, remote_stage
                )
                base_contract_path = original_contract_path
                contract = PromptFactory.continuation_task_contract(
                    original_contract_path=original_contract_path,
                    mode="resume",
                    patch_path=patch_path,
                    validator_command=validator_command,
                )
                contract_path, prompt = _stage_task_contract(
                    local_stage,
                    remote_stage,
                    f"task-{token}-resume.md",
                    contract,
                    execution=execution,
                    role="resume",
                )
            else:
                if reuses_native_checkpoint and continuation != "retry":
                    raise ValueError(f"Unsupported graph continuation: {continuation}")
                schema_path = _stage_json_task_input(
                    local_stage,
                    remote_stage,
                    f"task-{token}-patch-schema.json",
                    agent_output_schema(),
                )
                retry_diagnostics_path = (
                    _stage_json_task_input(
                        local_stage,
                        remote_stage,
                        f"task-{token}-retry-diagnostics.json",
                        {"prior_attempt_diagnostics": list(execution.retry_feedback)},
                    )
                    if execution is not None and execution.retry_feedback
                    else None
                )
            if reuses_native_checkpoint and continuation == "retry":
                # The live session still holds the original contract, so the retry is a
                # follow-up naming only what changed for this attempt. Rebuilding and
                # restating the whole contract would hand the agent its retry framing twice.
                if not request.session_id:
                    raise ValueError(
                        "The failed operation has no native agent session; retry it cleanly."
                    )
                assert execution is not None
                assert retry_diagnostics_path is not None
                base_contract_path = _parent_task_contract_path(
                    execution, local_stage, remote_stage
                )
                contract = PromptFactory.continuation_task_contract(
                    original_contract_path=base_contract_path,
                    mode="retry",
                    patch_path=patch_path,
                    diagnostics_path=retry_diagnostics_path,
                    output_schema_path=schema_path,
                    validator_command=validator_command,
                    skill_pointers=skill_pointers,
                )
                contract_path, prompt = _stage_task_contract(
                    local_stage,
                    remote_stage,
                    f"task-{token}-retry.md",
                    contract,
                    execution=execution,
                    role="retry",
                )
            elif continuation != "resume":
                human_request_path = (
                    _stage_task_input(
                        local_stage,
                        remote_stage,
                        f"task-{token}-human-request.txt",
                        request.message,
                    )
                    if request.message
                    else None
                )
                base_contract_content = service.graph_task_contract(
                    kind,
                    project_name=context.project_name,
                    ontology_path=f"{context.graph_path}#ontology",
                    ontology_extensions=context.ontology_extensions,
                    graph_path=context.graph_path,
                    research_path=context.research_md_path,
                    provider_log_roots=context.all_source_roots(),
                    ingestion_watermark=context.ingestion_watermark,
                    repositories=[
                        {"alias": item.alias, "host": item.host, "path": item.path}
                        for item in context.repositories
                    ],
                    patch_path=patch_path,
                    output_schema_path=schema_path,
                    human_request_path=human_request_path,
                    retry_diagnostics_path=retry_diagnostics_path,
                    source_errors=context.source_errors,
                    validator_command=validator_command,
                    skill_pointers=skill_pointers,
                )
                base_label = (
                    f"task-{token}-initial.md"
                    if continuation == "fresh"
                    else f"task-{token}-base.md"
                )
                base_contract_path, base_prompt = _stage_task_contract(
                    local_stage,
                    remote_stage,
                    base_label,
                    base_contract_content,
                    execution=execution,
                    role="base",
                )

                if retry_state is not None and retry_state.progress_parent is not None:
                    handoff = dict(retry_state.progress)
                    if retry_state.retained_patch_text:
                        handoff["retained_patch_path"] = _stage_task_input(
                            local_stage,
                            remote_stage,
                            f"task-{token}-prior-patch.json",
                            retry_state.retained_patch_text,
                        )
                    handoff_path = _stage_json_task_input(
                        local_stage,
                        remote_stage,
                        f"task-{token}-handoff.json",
                        handoff,
                    )
                    contract = PromptFactory.retry_handoff_task_contract(
                        kind=kind,
                        handoff_path=handoff_path,
                        original_contract_path=base_contract_path,
                        patch_path=patch_path,
                        validator_command=validator_command,
                    )
                    contract_path, prompt = _stage_task_contract(
                        local_stage,
                        remote_stage,
                        f"task-{token}-retry.md",
                        contract,
                        execution=execution,
                        role="retry",
                    )
                    _record_progress_handoff(
                        execution,
                        handed_off=True,
                        source=retry_state.progress_parent,
                    )
                else:
                    contract_path, prompt = base_contract_path, base_prompt
                    if retry_state is not None:
                        _record_progress_handoff(
                            execution,
                            handed_off=False,
                            reason=retry_state.progress_reason,
                        )
            if not reuses_native_checkpoint and execution is not None:
                execution_record = execution.store.agent_task(execution.operation_id)
                if execution_record is not None:
                    _stage_prepared_graph_context(
                        local_stage,
                        remote_stage,
                        project_id=execution_record.project_id,
                        kind=kind,
                        graph_revision=graph_revision,
                        execution_host=execution_host,
                        original_contract_path=base_contract_path,
                        context=context,
                    )
        except (ReplayHalted, StateUnavailable, ValueError) as exc:
            yield _sse(AgentEvent(event="error", text=str(exc)))
            return

        native_session_id = request.session_id
        session_id = request.session_id if reuses_native_checkpoint else None
        rounds = 0
        last_problem = (
            execution.retry_feedback[0]
            if execution is not None and continuation == "retry" and execution.retry_feedback
            else None
        )
        while True:
            assert run_lock_lease is not None
            run_lock_lease.assert_owned()
            # Resume and Retry reuse their predecessor's stage, including any
            # retained patch. Fingerprint it rather than deleting it: invariant 9
            # says failed work remains inspectable, but a Retry may not claim an
            # inherited file as output it produced. In-process correction rounds
            # have the same safeguard.
            requires_new_patch = bool(rounds) or continuation == "retry"
            pre_launch_patch_digest = (
                _existing_patch_digest(workspace, remote_stage)
                if reuses_native_checkpoint or rounds
                else None
            )
            if (
                remote_stage is not None
                and imported_source_inventory is not None
                and imported_source_inventory.files
                and not imported_stage_verified
            ):
                try:
                    await asyncio.to_thread(remote_stage.finalize_inputs)
                    imported_readback = await asyncio.to_thread(
                        remote_stage.verify_imported_provider_sources,
                        imported_source_inventory,
                        _IMPORTED_PROVIDER_SOURCES_LABEL,
                    )
                except (OSError, StateUnavailable, ValueError) as exc:
                    yield _sse(AgentEvent(event="error", text=str(exc)))
                    return
                _record_imported_source_stage_receipt(
                    execution,
                    imported_readback,
                    reused=False,
                )
                imported_stage_verified = True
            _record_agent_launch_receipt(
                execution,
                request,
                prompt=prompt,
                contract_path=contract_path,
                remote=bool(execution_host),
                resumed=reuses_native_checkpoint,
                continuation=("correction" if rounds else continuation),
                extra={
                    "surface": surface,
                    "capability": "scratch_patch",
                    "network_access": True,
                    "launch_kind": ("correction" if rounds else continuation),
                    "correction_round": rounds,
                },
            )
            # Hold the labelled final answer until the initial provider invocation
            # completes, and hold `done` until the Patch applies. Answer and Patch
            # verdict remain independent.
            outcome = _ProviderOutcome(session_id=native_session_id)
            async with aclosing(
                _stream_graph_agent_events(
                    service,
                    launcher,
                    request,
                    prompt,
                    workspace=workspace,
                    session_id=session_id,
                    read_dirs=read_dirs,
                    execution_host=execution_host,
                    execution=execution,
                    remote_stage=remote_stage,
                    outcome=outcome,
                    binary=provider_binary,
                    validator_staged=validator_staged,
                    validator_budget=validator_budget,
                    kind=kind,
                    run_truth_scope=context.run_truth_scope,
                )
            ) as stream:
                async for frame in stream:
                    if execution is not None:
                        streamed = AgentEvent.model_validate_json(
                            frame.removeprefix("data: ").strip()
                        )
                        execution_record = execution.store.agent_task(execution.operation_id)
                        if streamed.event == "error" and execution_record is not None:
                            execution.store.record_agent_task_receipt(
                                execution.operation_id,
                                "provider_terminal_error",
                                {
                                    "provider": request.provider,
                                    "classification": classify_terminal_error(streamed.text),
                                },
                                tier="diagnostic",
                            )
                            execution.store.record_agent_task_receipt(
                                execution.operation_id,
                                "patch_collection_skipped",
                                {
                                    "reason": "provider_terminal_error",
                                    "patch_availability_evaluated": False,
                                },
                                tier="diagnostic",
                            )
                    yield frame
            _record_provider_exit(
                execution,
                outcome,
                workspace=workspace,
                remote_stage=remote_stage,
            )
            native_session_id = outcome.session_id
            run_lock_lease.assert_owned()
            if not outcome.completed:
                if outcome.failed or outcome.paused:
                    return
                yield _sse(
                    AgentEvent(event="error", text=f"{request.provider} produced no result.")
                )
                return

            if rounds == 0:
                for answer in outcome.answers:
                    yield _sse(AgentEvent(event="answer", text=answer))

            if execution is not None:
                execution.store.update_agent_task_message(
                    execution.operation_id,
                    "Validating and applying the graph update.",
                    phase="applying",
                    event=True,
                )
            stale_patch = False
            try:
                patch_text, output_name = _collect_patch_text(workspace, remote_stage)
                unchanged = (
                    pre_launch_patch_digest is not None
                    and hashlib.sha256(patch_text.encode("utf-8")).hexdigest()
                    == pre_launch_patch_digest
                )
                if unchanged and execution is not None:
                    execution.store.record_agent_task_receipt(
                        execution.operation_id,
                        "patch_predates_launch",
                        {"correction_round": rounds, "accepted": not requires_new_patch},
                        tier="diagnostic",
                    )
                if unchanged and requires_new_patch:
                    # Applying it would attribute inherited output to this launch.
                    # The substantive diagnostic still leads; the fingerprint only
                    # proves that this launch did not rewrite the deliverable.
                    stale_patch = True
                    raise AgentOutputProblem(
                        (f"{last_problem} " if last_problem else "")
                        + "The patch file is byte-identical to the one this launch "
                        "inherited, so this launch did not write a new patch. Rewrite "
                        "patch.json before RCP can accept it as this attempt's output."
                    )
            except AgentOutputProblem as exc:
                problem = str(exc)
                if not stale_patch:
                    last_problem = problem
            else:
                if execution is not None:
                    # Persisted before validation: a patch that fails validation is
                    # still the run's work product and must survive the failure.
                    execution.store.record_agent_task_patch_output(
                        execution.operation_id, patch_text
                    )
                    execution.store.record_agent_task_receipt(
                        execution.operation_id,
                        "patch_retained",
                        {
                            "byte_length": len(patch_text.encode("utf-8")),
                            "file_name": output_name,
                        },
                        tier="diagnostic",
                    )
                    if output_name != "patch.json":
                        execution.store.record_agent_task_event(
                            execution.operation_id,
                            f"Recovered the patch from {output_name}.",
                            level="warning",
                        )
                try:
                    patch = _prepare_graph_patch_candidate(
                        service,
                        patch_text,
                        kind=kind,
                        run_truth_scope=context.run_truth_scope,
                        source_operation_id=execution.operation_id if execution else None,
                    )
                    _record_patch_receipt(
                        execution,
                        patch,
                        byte_length=len(patch_text.encode("utf-8")),
                    )
                except ValueError as exc:
                    problem = str(exc)
                    last_problem = problem
                else:
                    try:
                        run_lock_lease.assert_owned()
                        _appended, result = service.history.append(
                            patch,
                            discard_on_reject=True,
                        )
                    except PatchRejected as exc:
                        problem = str(exc)
                        last_problem = problem
                        if execution is not None:
                            execution.store.record_agent_task_receipt(
                                execution.operation_id,
                                "patch_rejected",
                                {
                                    "round": rounds,
                                    "messages": [
                                        item.model_dump(mode="json") for item in exc.report.messages
                                    ],
                                },
                                tier="diagnostic",
                            )
                    except (ReplayHalted, StateUnavailable) as exc:
                        yield _sse(AgentEvent(event="error", text=str(exc)))
                        return
                    else:
                        _record_patch_applied_receipt(execution, result.state)
                        applied = True
                        yield _sse(
                            AgentEvent(
                                event="message",
                                text=json.dumps(
                                    {"applied_revision": result.state.revision},
                                    separators=(",", ":"),
                                ),
                            )
                        )
                        yield _sse(AgentEvent(event="done"))
                        return

            # Rungs 2 and 3: hand the concrete problem back to the agent that is still
            # holding the analysis, rather than discarding the run and asking a human.
            if rounds >= PATCH_CORRECTION_MAX_ROUNDS or not native_session_id:
                yield _sse(AgentEvent(event="error", text=problem))
                return
            rounds += 1
            if execution is not None:
                execution.store.record_agent_task_receipt(
                    execution.operation_id,
                    "patch_correction_requested",
                    {"round": rounds, "problem": problem[:400]},
                    tier="diagnostic",
                )
                execution.store.record_agent_task_event(
                    execution.operation_id,
                    f"Asking the agent to correct its patch (round {rounds}).",
                    level="info",
                )
                execution.store.update_agent_task_message(
                    execution.operation_id,
                    "Asking the agent to correct its patch.",
                    phase="agent",
                    event=True,
                )
            diagnostics_path = _stage_json_task_input(
                local_stage,
                remote_stage,
                f"task-{token}-correction-{rounds}.json",
                {"kind": kind, "problem": problem},
            )
            validator_staged = stage_patch_validation_mailbox(
                local_stage=local_stage,
                remote_stage=remote_stage,
                task_id=execution.operation_id if execution is not None else token,
                turn_id=f"{token}:graph-pass:{rounds}",
                timeout_seconds=PATCH_SELF_CHECK_TIMEOUT_SECONDS,
            )
            validator_command = validator_staged.client_command("validate", patch_path)
            correction_contract = PromptFactory.continuation_task_contract(
                original_contract_path=base_contract_path,
                mode="patch_correction",
                patch_path=patch_path,
                diagnostics_path=diagnostics_path,
                validator_command=validator_command,
            )
            contract_path, prompt = _stage_task_contract(
                local_stage,
                remote_stage,
                f"task-{token}-correction-{rounds}.md",
                correction_contract,
                execution=execution,
                role=f"graph_patch_correction_{rounds}",
            )
            session_id = native_session_id
    except RunLockOwnershipLost as exc:
        yield _sse(
            AgentEvent(
                event="paused",
                text=f"{exc} The task paused before applying further graph changes.",
            )
        )
    finally:
        if validator_staged is not None and not validator_staged.credential.expired:
            await asyncio.to_thread(
                cleanup_patch_validation_mailbox,
                staged=validator_staged,
                execution=execution,
            )
        if applied:
            if local_stage is not None:
                with suppress(OSError, ValueError):
                    _remove_local_tree(local_stage, local_stage.parent)
            if remote_stage is not None:
                remote_stage.close()
            if execution is not None and (local_stage is not None or remote_stage is not None):
                execution.store.clear_agent_task_stage(execution.operation_id)
        if run_lock_acquired:
            run_lock.__exit__(None, None, None)


async def _stream_graph_agent_events(
    service: ProjectService,
    launcher: AgentLauncher,
    request: RunRequest,
    prompt: str,
    *,
    workspace: Path,
    session_id: str | None,
    read_dirs: list[Path],
    execution_host: str,
    execution: AgentTaskExecution | None,
    remote_stage: RemoteRunStage | None,
    outcome: _ProviderOutcome,
    binary: str | None,
    validator_staged: StagedCommandMailbox,
    validator_budget: PatchValidationBudget,
    kind: str,
    run_truth_scope: list[str],
) -> AsyncIterator[str]:
    stop = asyncio.Event()
    mailbox = asyncio.create_task(
        serve_patch_validation_mailbox(
            staged=validator_staged,
            execution=execution,
            validate=lambda text: _validate_graph_patch_live(
                service,
                text,
                kind=kind,
                run_truth_scope=run_truth_scope,
                source_operation_id=execution.operation_id if execution else None,
            ),
            stop=stop,
            budget=validator_budget,
        )
    )
    try:
        async with aclosing(
            _stream_agent_events(
                launcher,
                request,
                prompt,
                workspace=workspace,
                session_id=session_id,
                read_dirs=read_dirs,
                write_dirs=[],
                write_scope=None,
                execution_host=execution_host,
                execution=execution,
                remote_stage=remote_stage,
                capability="scratch_patch",
                outcome=outcome,
                binary=binary,
            )
        ) as stream:
            async for frame in stream:
                yield frame
    finally:
        stop.set()
        try:
            await mailbox
        finally:
            await asyncio.to_thread(
                cleanup_patch_validation_mailbox,
                staged=validator_staged,
                execution=execution,
            )


def _prepare_graph_patch_candidate(
    service: ProjectService,
    patch_text: str,
    *,
    kind: str,
    run_truth_scope: list[str],
    source_operation_id: str | None = None,
):
    draft, _ = service.parse_patch_output([patch_text])
    validate_agent_patch_shape(draft)
    return prepare_agent_patch(
        draft,
        kind=kind,
        run_truth_scope=run_truth_scope,
        repository_paths=service.manifest.repository_paths,
        source_operation_id=source_operation_id,
    )


def _validate_graph_patch_live(
    service: ProjectService,
    patch_text: str,
    *,
    kind: str,
    run_truth_scope: list[str],
    source_operation_id: str | None = None,
) -> PatchValidationResult:
    try:
        patch = _prepare_graph_patch_candidate(
            service,
            patch_text,
            kind=kind,
            run_truth_scope=run_truth_scope,
            source_operation_id=source_operation_id,
        )
        prepared, report, state = service.history.validate_candidate(patch)
    except (ReplayHalted, StateUnavailable, OSError) as exc:
        return PatchValidationResult(status="unavailable", messages=[str(exc)])
    except ValueError as exc:
        return PatchValidationResult(status="invalid", messages=[str(exc)])
    rejects = [item.message for item in report.messages if item.level == "reject"]
    if rejects:
        return PatchValidationResult(
            status="invalid",
            messages=rejects,
            live_revision=state.revision,
            candidate_revision=prepared.revision,
        )
    return PatchValidationResult(
        status="valid",
        messages=[item.message for item in report.flags],
        live_revision=state.revision,
        candidate_revision=prepared.revision,
    )


def _record_run_lock_wait(
    execution: AgentTaskExecution,
    message: str,
    location: str,
) -> None:
    detail = f"{message} Location: {location}"
    execution.store.update_agent_task_message(
        execution.operation_id,
        detail,
        phase="waiting",
        event=True,
    )
    execution.store.record_agent_task_receipt(
        execution.operation_id,
        "canonical_state_lock_wait",
        {"location": location},
        tier="diagnostic",
    )


def _record_run_lock_lost(
    execution: AgentTaskExecution,
    message: str,
    location: str,
) -> None:
    detail = f"{message} The task is pausing before further graph changes. Location: {location}"
    execution.store.update_agent_task_message(
        execution.operation_id,
        detail,
        phase="pausing",
    )
    execution.store.record_agent_task_event(
        execution.operation_id,
        detail,
        level="warning",
    )
    execution.store.record_agent_task_receipt(
        execution.operation_id,
        "canonical_state_lock_lost",
        {"location": location},
        tier="diagnostic",
    )
    execution.control.request_pause()


def _record_context_receipt(
    execution: AgentTaskExecution | None,
    context: RunContext,
    *,
    surface: AgentSurface,
) -> None:
    if execution is None:
        return
    native_source_root_count = sum(len(roots) for roots in context.source_roots.values())
    imported_source_root_count = sum(len(roots) for roots in context.imported_source_roots.values())
    execution.store.record_agent_task_receipt(
        execution.operation_id,
        "context_assembled",
        {
            "surface": surface,
            "repository_count": len(context.repositories),
            "source_root_count": native_source_root_count + imported_source_root_count,
            "imported_source_root_count": imported_source_root_count,
            "imported_source_fingerprint": context.imported_source_fingerprint,
            "source_warnings": list(context.source_errors),
            "source_error_count": len(context.source_errors),
            "graph_revision": context.graph_revision,
            "ingestion_watermark": (
                context.ingestion_watermark.isoformat()
                if context.ingestion_watermark is not None
                else None
            ),
        },
    )


def _record_imported_source_stage_receipt(
    execution: AgentTaskExecution | None,
    readback: ImportedProviderSourceReadback,
    *,
    reused: bool,
) -> None:
    if execution is None:
        return
    execution.store.record_agent_task_receipt(
        execution.operation_id,
        "imported_source_stage_verified",
        {
            "fingerprint": readback.fingerprint,
            "file_count": readback.file_count,
            "payload_size_bytes": readback.payload_size_bytes,
            "reused_checkpoint": reused,
        },
    )


def _report_source_errors(
    execution: AgentTaskExecution | None,
    source_errors: list[str],
) -> None:
    """Surface root-preflight warnings without turning them into launch authority."""

    if execution is None:
        return
    for detail in source_errors[:16]:
        execution.store.record_agent_task_event(
            execution.operation_id,
            f"Provider log root preflight warning; launch continues: {detail}",
            level="warning",
        )


def _agent_read_dirs(
    context: RunContext,
    remote_stage: RemoteRunStage | None,
    service: ProjectService,
    execution_machine: str,
) -> list[Path]:
    """Directories the agent may need to read from outside its scratch folder.

    Only Claude consumes these (as `--add-dir`); Codex reads are unrestricted in
    every sandbox mode. Repositories on another machine are deliberately absent —
    those are reached over ssh from the pointers in the prompt, never copied.
    """
    read_dirs = [
        Path(item.path) for item in context.repositories if item.machine == execution_machine
    ]
    if remote_stage is not None:
        # Derived from the manifest, not from the context: on a resumed run the
        # context still carries local paths because it is never re-staged.
        assert remote_stage.root is not None
        read_dirs.append(Path(str(remote_stage.root / "inputs")))
        state_repository = service.manifest.repository_map[service.manifest.state.repository]
        if state_repository.machine == execution_machine:
            state_root = Path(state_repository.path) / ".research"
            if str(state_root) not in {str(item) for item in read_dirs}:
                read_dirs.append(state_root)
        for roots in context.all_source_roots().values():
            read_dirs.extend(Path(item) for item in roots)
        return _deduplicate_paths(read_dirs)
    read_dirs = [item for item in read_dirs if item.exists()]
    read_dirs.append(service.manifest.research_dir)
    for roots in context.all_source_roots().values():
        for item in roots:
            read_dirs.append(Path(item).expanduser())
    return _deduplicate_paths(read_dirs)


def _deduplicate_paths(paths: list[Path]) -> list[Path]:
    return [Path(value) for value in dict.fromkeys(str(path) for path in paths)]


def _stage_graph_context(
    context: RunContext,
    service: ProjectService,
    stage: RemoteRunStage,
    execution_machine: str,
    *,
    imported_source_inventory: ImportedProviderSourceInventory | None = None,
) -> RunContext:
    """Stage project-owned imports and rebind remote-readable context pointers."""

    local_imported_roots = (
        imported_source_inventory.roots(service.imported_sources.root)
        if imported_source_inventory is not None
        and imported_source_inventory.files
        and service.imported_sources is not None
        else {}
    )
    local_context = context.model_copy(update={"imported_source_roots": local_imported_roots})
    service.validate_imported_source_context(local_context, imported_source_inventory)
    updates = _stage_context_paths(local_context, service, stage, execution_machine)
    if imported_source_inventory is not None and imported_source_inventory.files:
        if service.imported_sources is None:
            raise ValueError("imported provider source store is unavailable")
        staged_root = stage.put_imported_provider_sources(
            service.imported_sources,
            imported_source_inventory,
            _IMPORTED_PROVIDER_SOURCES_LABEL,
        )
        updates["imported_source_roots"] = _remote_imported_source_roots(
            imported_source_inventory,
            staged_root,
        )
    else:
        updates["imported_source_roots"] = {}
    return local_context.model_copy(
        update=updates,
    )
