"""Admission for an episode's one durable hidden report allocation.

This is episode policy, not engine plumbing, so it lives beside the other
episode owners rather than on ``BackgroundAgentTasks``.  Relocating it does not
decouple it: launching still needs the engine's in-process worker registry and
its launch gate, so the engine is passed in, exactly as ``EpisodeReconciler``
already takes it.  The gain is an address, not reduced coupling.

That is why this module reaches into the engine's private worker registry and
launch gate rather than calling a public facade: inventing one here would imply
a supported boundary that does not exist.
"""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

from rcp.runs.tasks.episode_report import EpisodeReportRunRequest
from rcp.storage import AgentTaskRecord

if TYPE_CHECKING:
    from rcp.background import BackgroundAgentTasks


def start_episode_report(
    tasks: BackgroundAgentTasks,
    episode_id: str,
) -> AgentTaskRecord | None:
    """Launch or restart the one durable hidden allocation for an episode."""

    store = tasks.store
    episode = store.episode(episode_id)
    if episode is None:
        raise KeyError(episode_id)
    if episode.status != "wrapping_up" or episode.wrapup_state not in {"pending", "running"}:
        return None
    wrapup = store.episode_wrapup(episode_id)
    if wrapup is None or wrapup.allocation_operation_id is None:
        raise ValueError("The episode report lost its durable allocation fence.")
    existing = store.agent_task(wrapup.allocation_operation_id)
    if existing is not None and existing.status in {"queued", "running", "pausing"}:
        with tasks._controls_lock:
            worker = tasks._workers.get(existing.operation_id)
            if worker is not None:
                return existing
        if existing.status != "queued":
            return None
    task = store.requeue_interrupted_episode_report_allocation(episode_id)
    if task.status != "queued":
        return None
    if task.kind != "episode_report" or task.visible or task.episode_id != episode_id:
        raise ValueError("The episode report allocation lost its hidden task boundary.")
    request = EpisodeReportRunRequest.model_validate(task.request)
    if request.episode_id != episode_id:
        raise ValueError("The episode report request changed its parent episode.")
    tasks._require_operation(task.parent_operation_id or "")
    return tasks.launch_admitted(task.operation_id)


def restart_interrupted_episode_reports(tasks: BackgroundAgentTasks) -> None:
    """Restart every report a previous process left mid-flight."""

    for episode in tasks.store.episodes_awaiting_report():
        with suppress(KeyError, RuntimeError, ValueError):
            start_episode_report(tasks, episode.episode_id)
