from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing, suppress
from pathlib import Path
from typing import TYPE_CHECKING

from rcp.agents import AgentEvent, AgentLauncher
from rcp.agents.write_scope import resolve_project_write_scope
from rcp.core.models import BranchMergeReceipt
from rcp.limits import PATCH_SELF_CHECK_TIMEOUT_SECONDS
from rcp.runs.branch_merge import (
    BranchMergeCandidateProblem,
    BranchMergeContext,
    BranchMergeEligibility,
    BranchMergeRunOutcome,
    BranchMergeSemanticConflict,
    BranchMergeStage,
    BranchPatchSummary,
    branch_merge_can_resolve_without_patch,
    branch_merge_id,
    parse_branch_merge_candidate,
    prepare_branch_merge_with_history,
    stream_branch_merge_run,
)
from rcp.runs.patch_validator import (
    PatchValidationBudget,
    PatchValidationResult,
    cleanup_patch_validation_mailbox,
    serve_patch_validation_mailbox,
    stage_patch_validation_mailbox,
)
from rcp.runs.shared import _remove_local_tree, _safe_stage_name, _sse, _swept_stage_root
from rcp.runs.task_policy import task_graph_capable
from rcp.service import ProjectService, RunRequest
from rcp.storage import ACTIVE_AGENT_TASK_STATUSES, AgentTaskRecord, EpisodeRecord
from rcp.transport import RemoteRunStage, StateUnavailable

if TYPE_CHECKING:
    from rcp.background import AgentTaskExecution


async def stream_branch_merge_task(
    service: ProjectService,
    launcher: AgentLauncher,
    request: RunRequest,
    data_dir: Path,
    *,
    episode: EpisodeRecord,
    task: AgentTaskRecord,
    execution: AgentTaskExecution,
) -> AsyncIterator[str]:
    """Run one human-dispatched branch merge through its contained scratch stage."""

    if (
        task.kind != "branch_merge"
        or task.episode_id != episode.episode_id
        or task.project_id != episode.project_id
        or task.graph_target != episode.graph_target
        or episode.graph_target.kind != "branch"
        or episode.graph_target.branch_id != episode.episode_id
        or task.authorized_by is None
    ):
        yield _sse(
            AgentEvent(event="error", text="The branch merge task lost its episode binding.")
        )
        return
    if request.provider is None or request.run_on is None:
        yield _sse(
            AgentEvent(event="error", text="The branch merge task has no pinned provider target.")
        )
        return
    machine = service.manifest.machine_map.get(request.run_on)
    if machine is None:
        yield _sse(AgentEvent(event="error", text=f"unknown execution machine: {request.run_on}"))
        return

    local_stage: Path | None = None
    remote_stage: RemoteRunStage | None = None
    staged_validator = None
    completed = False
    try:
        if machine.host:
            remote_stage = RemoteRunStage(machine.host).open(task.operation_id)
            assert remote_stage.root is not None
            execution.checkpoint_stage(machine.host, str(remote_stage.root))
            workspace = Path(str(remote_stage.workspace))
            stage_root = str(remote_stage.root)
        else:
            parent = _swept_stage_root(data_dir)
            local_stage = parent / _safe_stage_name(task.operation_id)
            local_stage.mkdir(mode=0o700, parents=True, exist_ok=False)
            workspace = local_stage / "workspace"
            workspace.mkdir(mode=0o700)
            execution.checkpoint_stage("", str(local_stage))
            stage_root = str(local_stage)

        stage = BranchMergeStage(
            local_stage=local_stage,
            remote_stage=remote_stage,
            workspace=workspace,
        )
        write_scope = resolve_project_write_scope(
            manifest=service.manifest,
            project_id=task.project_id,
            execution_machine=request.run_on,
            capability="orchestrate",
            stage_root=stage_root,
            workspace_root=str(workspace),
            admitted_aliases=[],
            repository_pointers=[],
            remote_stage=remote_stage,
            app_data_dir=data_dir,
            repository_inventory=service.repository_ownership_inventory(project_id=task.project_id),
        )
        execution.bind_write_scope(write_scope)

        branch = service.history.branch(
            episode.episode_id,
            expected_episode_id=episode.episode_id,
            expected_project_id=task.project_id,
        )

        def load_context() -> BranchMergeContext:
            current_episode = execution.store.episode(episode.episode_id)
            if (
                current_episode is None
                or current_episode.project_id != task.project_id
                or current_episode.graph_target != task.graph_target
                or current_episode.ending is None
                or not execution.store.auto_research_is_quiescent(episode.episode_id)
            ):
                raise ValueError("The Auto-research branch is no longer quiescent or ended.")
            branch_result = branch.current_materialization()
            metadata = branch.branch_metadata()
            branch_head = branch.head_ref(branch_result)
            if metadata.head != branch_head:
                raise StateUnavailable("The graph branch changed while its merge context loaded.")
            active_writers = _active_branch_writer_task_ids(
                execution,
                episode.episode_id,
                exclude_operation_id=task.operation_id,
            )
            eligibility = BranchMergeEligibility(
                branch_head=branch_head,
                episode_ending=current_episode.ending,
                active_branch_writer_task_ids=active_writers,
            )
            main = service.history.current_materialization()
            main_head = service.history.head_ref(main)
            patches = [
                patch
                for patch in branch.load_patches()
                if metadata.base_head.revision < patch.revision <= branch_head.revision
            ]
            return BranchMergeContext.create(
                merge_task_id=task.operation_id,
                authorized_by=task.authorized_by,
                metadata=metadata,
                eligibility=eligibility,
                base_graph=branch.base_state(),
                branch_graph=branch_result.state,
                main_head=main_head,
                main_graph=main.state,
                run_truth_scope=sorted(set(request.run_truth_scope or ())),
                branch_patch_summaries=[
                    BranchPatchSummary(
                        revision=patch.revision,
                        transition_id=(
                            patch.transition.transition_id if patch.transition is not None else None
                        ),
                        summary=patch.summary,
                        change_summary=list(patch.change_summary),
                        task_id=patch.task_id,
                        profile=patch.profile,
                    )
                    for patch in patches
                ],
            )

        initial_context = load_context()
        merge_id = branch_merge_id(initial_context.metadata)
        existing_receipt = branch.reconcile_merge_receipt(merge_id)
        if existing_receipt is not None:
            _apply_receipt_to_execution(service, execution, existing_receipt)
            execution.store.record_agent_task_receipt(
                task.operation_id,
                "branch_merge_reconciled",
                existing_receipt.model_dump(mode="json"),
            )
            completed = True
            yield _sse(
                AgentEvent(
                    event="answer",
                    text=(
                        "This branch head was already merged into main revision "
                        f"{existing_receipt.result_main_head.revision}."
                    ),
                )
            )
            if existing_receipt.outcome == "committed":
                yield _applied_revision_event(existing_receipt.result_main_head.revision)
            yield _sse(AgentEvent(event="done"))
            return

        staged_validator = stage_patch_validation_mailbox(
            local_stage=workspace if remote_stage is None else None,
            remote_stage=remote_stage,
            task_id=task.operation_id,
            turn_id=f"{task.operation_id}:branch-merge",
            timeout_seconds=PATCH_SELF_CHECK_TIMEOUT_SECONDS,
        )
        validator_stop = asyncio.Event()
        validator_budget = PatchValidationBudget()
        validator_task = asyncio.create_task(
            serve_patch_validation_mailbox(
                staged=staged_validator,
                execution=execution,
                validate=lambda text: _validate_candidate(service, load_context, text),
                stop=validator_stop,
                budget=validator_budget,
            )
        )
        outcome = BranchMergeRunOutcome()
        held_done: str | None = None
        try:
            async with aclosing(
                stream_branch_merge_run(
                    request,
                    launcher,
                    load_context=load_context,
                    main_history=service.history,
                    stage=stage,
                    write_scope=write_scope,
                    validator_command=staged_validator.client_command(
                        "validate",
                        str(workspace / "patch.json"),
                    ),
                    outcome=outcome,
                    execution=execution,
                    binary=machine.provider_paths.get(request.provider),
                )
            ) as stream:
                async for frame in stream:
                    event = _event(frame)
                    if event.event == "done":
                        held_done = frame
                    else:
                        yield frame
        finally:
            validator_stop.set()
            await validator_task

        if outcome.receipt is None:
            if outcome.status in {"rejected", "retryable"} and outcome.diagnostic:
                yield _sse(AgentEvent(event="error", text=outcome.diagnostic))
            return
        receipt = branch.write_merge_receipt(outcome.receipt)
        _apply_receipt_to_execution(service, execution, receipt)
        execution.store.record_agent_task_receipt(
            task.operation_id,
            "branch_merge_outcome",
            receipt.model_dump(mode="json"),
        )
        completed = True
        if receipt.outcome == "committed":
            yield _applied_revision_event(receipt.result_main_head.revision)
        yield held_done or _sse(AgentEvent(event="done"))
    except (OSError, StateUnavailable, ValueError) as exc:
        yield _sse(AgentEvent(event="error", text=str(exc)))
    finally:
        if staged_validator is not None and not staged_validator.credential.expired:
            await asyncio.to_thread(
                cleanup_patch_validation_mailbox,
                staged=staged_validator,
                execution=execution,
            )
        if completed:
            if local_stage is not None:
                with suppress(OSError, ValueError):
                    _remove_local_tree(local_stage, local_stage.parent)
            if remote_stage is not None:
                remote_stage.close()
            execution.store.clear_agent_task_stage(task.operation_id)


def _active_branch_writer_task_ids(
    execution: AgentTaskExecution,
    episode_id: str,
    *,
    exclude_operation_id: str,
) -> list[str]:
    target = execution.store.episode(episode_id)
    if target is None:
        raise ValueError("The Auto-research branch lost its episode binding.")
    return [
        item.operation_id
        for item in execution.store.graph_target_tasks(
            target.project_id,
            target.graph_target,
            include_hidden=True,
        )
        if item.operation_id != exclude_operation_id
        and item.kind != "branch_merge"
        and item.status in {*ACTIVE_AGENT_TASK_STATUSES, "paused"}
        and task_graph_capable(item.kind, item.request)
    ]


def _validate_candidate(
    service: ProjectService,
    load_context: Callable[[], BranchMergeContext],
    text: str,
) -> PatchValidationResult:
    try:
        context = load_context()
        patch = parse_branch_merge_candidate(text, context)
        if not patch.ops:
            if not branch_merge_can_resolve_without_patch(context):
                raise BranchMergeCandidateProblem(
                    "An empty merge candidate omits non-conflicting source branch changes."
                )
            return PatchValidationResult(
                status="valid",
                live_revision=context.main_head.revision,
                candidate_revision=context.main_head.revision,
            )
        prepared = prepare_branch_merge_with_history(
            service.history,
            patch,
            expected_main_head=context.main_head,
            context=context,
        )
    except (
        BranchMergeCandidateProblem,
        BranchMergeSemanticConflict,
        StateUnavailable,
        ValueError,
    ) as exc:
        return PatchValidationResult(
            status="unavailable" if isinstance(exc, StateUnavailable) else "invalid",
            messages=[str(exc)],
        )
    return PatchValidationResult(
        status="valid",
        live_revision=context.main_head.revision,
        candidate_revision=prepared.patch.revision,
    )


def _apply_receipt_to_execution(
    service: ProjectService,
    execution: AgentTaskExecution,
    receipt: BranchMergeReceipt,
) -> None:
    if receipt.outcome != "committed":
        return
    result = service.history.current_materialization()
    if result.state.revision < receipt.result_main_head.revision:
        raise StateUnavailable("The committed branch merge main state is not yet coherent.")
    matching = [
        patch for patch in result.patches if patch.revision == receipt.result_main_head.revision
    ]
    if len(matching) != 1:
        raise StateUnavailable("The committed branch merge revision is not uniquely readable.")
    patch = matching[0]
    if (
        patch.admission != "accepted"
        or patch.branch_merge != receipt.provenance
        or patch.transition is None
        or patch.transition.transition_id != receipt.result_main_head.transition_id
        or patch.authorized_by != receipt.authorized_by
    ):
        raise StateUnavailable("The committed branch merge receipt disagrees with main history.")
    execution.applied_revision = receipt.result_main_head.revision
    execution.applied_graph_state = result.state


def _applied_revision_event(revision: int) -> str:
    return _sse(
        AgentEvent(
            event="message",
            text=json.dumps({"applied_revision": revision}, separators=(",", ":")),
        )
    )


def _event(frame: str) -> AgentEvent:
    return AgentEvent.model_validate_json(frame.removeprefix("data: ").strip())
