from __future__ import annotations

from typing import Literal

from rcp.api.dependencies import get_project_service
from rcp.core.models import BranchMergeReceipt, GraphBranchMetadata, GraphBranchSummary
from rcp.core.transition_models import GraphHeadRef
from rcp.projects import ProjectCatalog
from rcp.runs.task_policy import task_graph_capable
from rcp.storage import ACTIVE_AGENT_TASK_STATUSES, AppStore, EpisodeRecord


def ensure_auto_research_graph_target(
    episode: EpisodeRecord,
    *,
    catalog: ProjectCatalog,
) -> None:
    if (
        episode.mode != "auto_research"
        or episode.graph_target.kind != "branch"
        or episode.graph_target.branch_id != episode.episode_id
        or episode.graph_base_head is None
        or episode.authorized_by is None
    ):
        raise ValueError("Auto-research reservation lost its exact graph branch identity.")
    service = get_project_service(catalog, episode.project_id)
    service.history.create_auto_research_branch(
        GraphBranchMetadata(
            branch_id=episode.episode_id,
            episode_id=episode.episode_id,
            project_id=episode.project_id,
            base_head=episode.graph_base_head,
            head=GraphHeadRef(
                target=episode.graph_target,
                revision=episode.graph_base_head.revision,
                transition_id=episode.graph_base_head.transition_id,
            ),
            created_at=episode.created_at,
            authorized_by=episode.authorized_by,
        )
    )


def graph_branch_summaries(
    episodes: list[EpisodeRecord],
    *,
    store: AppStore,
    catalog: ProjectCatalog,
) -> dict[str, GraphBranchSummary]:
    grouped: dict[str, list[EpisodeRecord]] = {}
    for episode in episodes:
        if (
            episode.mode != "auto_research"
            or episode.graph_target.kind != "branch"
            or episode.graph_target.branch_id != episode.episode_id
        ):
            raise ValueError("only an Auto-research branch has a graph branch summary")
        grouped.setdefault(episode.project_id, []).append(episode)

    summaries: dict[str, GraphBranchSummary] = {}
    for project_id, project_episodes in grouped.items():
        service = get_project_service(catalog, project_id)
        snapshots = service.history.branch_read_snapshots(
            [
                (episode.episode_id, episode.episode_id, episode.project_id)
                for episode in project_episodes
            ]
        )
        for episode in project_episodes:
            snapshot = snapshots[episode.episode_id]
            summaries[episode.episode_id] = (
                missing_graph_branch_summary(episode, store=store)
                if snapshot is None
                else graph_branch_summary_from_snapshot(
                    episode,
                    snapshot.metadata,
                    list(snapshot.receipts),
                    store=store,
                )
            )
    return summaries


def missing_graph_branch_summary(
    episode: EpisodeRecord,
    *,
    store: AppStore,
) -> GraphBranchSummary:
    if episode.status not in {"queued", "failed"} or episode.graph_base_head is None:
        raise KeyError(episode.episode_id)
    root = (
        store.agent_task(episode.root_operation_id)
        if episode.root_operation_id is not None
        else None
    )
    return GraphBranchSummary(
        branch_id=episode.episode_id,
        episode_id=episode.episode_id,
        base_head=episode.graph_base_head,
        head=GraphHeadRef(
            target=episode.graph_target,
            revision=episode.graph_base_head.revision,
            transition_id=episode.graph_base_head.transition_id,
        ),
        merge_eligible=False,
        merge_state="failed" if episode.status == "failed" else "unmerged",
        merge_diagnostic=(
            root.error
            if root is not None and root.error
            else episode.ending_diagnostic
            if episode.status == "failed"
            else "Establishing the episode graph branch before provider launch."
        ),
    )


def graph_branch_summary_from_snapshot(
    episode: EpisodeRecord,
    metadata: GraphBranchMetadata,
    receipts: list[BranchMergeReceipt],
    *,
    store: AppStore,
) -> GraphBranchSummary:
    current_receipt = next(
        (item for item in reversed(receipts) if item.provenance.branch_head == metadata.head),
        None,
    )
    merge_tasks = [
        item
        for item in store.episode_tasks(episode.episode_id)
        if item.kind == "branch_merge"
        and item.project_id == episode.project_id
        and item.graph_target == episode.graph_target
    ]
    active_task = next(
        (item for item in reversed(merge_tasks) if item.status in ACTIVE_AGENT_TASK_STATUSES),
        None,
    )
    latest_task = merge_tasks[-1] if merge_tasks else None
    active_branch_writers = [
        item
        for item in store.graph_target_tasks(
            episode.project_id,
            episode.graph_target,
            include_hidden=True,
        )
        if item.kind != "branch_merge"
        and item.status in {*ACTIVE_AGENT_TASK_STATUSES, "paused"}
        and task_graph_capable(item.kind, item.request)
    ]
    if active_task is not None:
        merge_state: Literal["unmerged", "running", "merged", "needs_action", "failed"] = "running"
    elif current_receipt is not None:
        merge_state = "merged"
    elif latest_task is not None and latest_task.status in {"paused", "interrupted"}:
        merge_state = "needs_action"
    elif latest_task is not None and latest_task.status == "failed":
        merge_state = "failed"
    else:
        merge_state = "unmerged"
    diagnostic = (
        (latest_task.error or latest_task.status_message)
        if merge_state in {"needs_action", "failed"} and latest_task is not None
        else None
    )
    merge_eligible = (
        episode.ending is not None
        and metadata.head.revision > metadata.base_head.revision
        and store.auto_research_is_quiescent(episode.episode_id)
        and active_task is None
        and not active_branch_writers
        and current_receipt is None
    )
    return GraphBranchSummary(
        branch_id=metadata.branch_id,
        episode_id=metadata.episode_id,
        base_head=metadata.base_head,
        head=metadata.head,
        merge_eligible=merge_eligible,
        merge_state=merge_state,
        latest_successful_merge=receipts[-1] if receipts else None,
        active_merge_task_id=(active_task.operation_id if active_task is not None else None),
        merge_diagnostic=diagnostic,
    )


def graph_branch_summary(
    episode: EpisodeRecord,
    *,
    store: AppStore,
    catalog: ProjectCatalog,
) -> GraphBranchSummary:
    return graph_branch_summaries([episode], store=store, catalog=catalog)[episode.episode_id]


__all__ = [
    "ensure_auto_research_graph_target",
    "graph_branch_summaries",
    "graph_branch_summary",
    "graph_branch_summary_from_snapshot",
    "missing_graph_branch_summary",
]
