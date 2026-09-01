"""Admission for one graph-only Auto-research branch merge.

Branch-merge policy, not engine plumbing: the checks below are about what an
ended Auto-research branch is, and none of them generalise to any other task
kind.  It takes the engine because launching needs the engine's launch gate.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from rcp.core.models import AuthorizedHuman
from rcp.runs.branch_merge_request import BranchMergeRunRequest
from rcp.runs.task_policy import resolved_dispatch_authority, task_graph_capable
from rcp.storage import ACTIVE_AGENT_TASK_STATUSES, AgentTaskRecord

if TYPE_CHECKING:
    from rcp.background import BackgroundAgentTasks


def start_branch_merge(
    tasks: BackgroundAgentTasks,
    project_id: str,
    request: BranchMergeRunRequest,
    *,
    authorized_by: AuthorizedHuman,
    operation_id: str | None = None,
) -> AgentTaskRecord:
    """Dispatch one graph-only merge without reopening or spending the episode."""

    if not isinstance(request, BranchMergeRunRequest):
        raise TypeError("branch_merge requires a BranchMergeRunRequest")
    episode = tasks.store.episode(request.episode_id)
    if (
        episode is None
        or episode.project_id != project_id
        or episode.mode != "auto_research"
        or episode.graph_target.kind != "branch"
        or episode.graph_target.branch_id != episode.episode_id
    ):
        raise ValueError("branch merge requires its exact Auto-research episode branch")
    if episode.ending is None or not tasks.store.auto_research_is_quiescent(episode.episode_id):
        raise ValueError("the Auto-research branch is not ended and quiescent")
    active_branch_writers = [
        item
        for item in tasks.store.graph_target_tasks(
            project_id,
            episode.graph_target,
            include_hidden=True,
        )
        if item.kind != "branch_merge"
        and item.status in {*ACTIVE_AGENT_TASK_STATUSES, "paused"}
        and task_graph_capable(item.kind, item.request)
    ]
    if active_branch_writers:
        raise ValueError("the Auto-research branch still has an active graph writer")
    if not authorized_by.display_name.strip():
        raise ValueError("branch merge requires a named human authorizer snapshot")

    operation_id = operation_id or str(uuid.uuid4())
    authority = resolved_dispatch_authority(
        tasks.store,
        tasks.dispatch_authority_resolver,
        "branch_merge",
        request,
        project_id=project_id,
        operation_id=operation_id,
    )
    assert authority is not None
    request_data = request.model_dump(mode="json")
    estimate, samples = tasks.store.agent_task_estimate(
        project_id,
        "branch_merge",
        request_data,
    )
    now = tasks.store.now()
    record = AgentTaskRecord(
        operation_id=operation_id,
        project_id=project_id,
        episode_id=episode.episode_id,
        graph_target=episode.graph_target,
        kind="branch_merge",
        status="queued",
        request=request_data,
        created_at=now,
        updated_at=now,
        status_message="Waiting for the graph branch merge agent to start.",
        estimate_seconds=estimate,
        estimate_samples=samples,
        phase="queued",
        last_activity_at=now,
        authorized_by=authorized_by,
        dispatch_authority=authority,
    )
    stored = tasks.store.create_branch_merge_task(record)
    return tasks.launch_admitted(stored.operation_id)
