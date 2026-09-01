"""S122 — fence an episode whose authorizer left the project.

Losing membership fences new work exactly the way **Stop loop** does under
invariant 10g. This module adds no second fence: it presses the existing one.
The turn running now finishes normally, no further watcher wake is claimed, and
the whole thing is durable across a restart because the Stop request is
persisted before any unclaimed watcher can win a claim.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rcp.runs.auto_research import settle_auto_research_stop
from rcp.runs.auto_research_admission import stop_auto_research
from rcp.storage import AppStore, EpisodeNotRunning, EpisodeRecord

if TYPE_CHECKING:
    from rcp.background import BackgroundAgentTasks

logger = logging.getLogger(__name__)

FENCE_DIAGNOSTIC = "The authorizing member left this project."
REMOVAL_FENCE_DIAGNOSTIC = "The authorizing member was removed from this team space."


def fence_episodes_for_departed_member(
    store: AppStore,
    background_tasks: BackgroundAgentTasks,
    project_id: str,
    user_id: str,
    *,
    diagnostic: str = FENCE_DIAGNOSTIC,
) -> list[str]:
    """Stop every live episode in ``project_id`` authorized by ``user_id``.

    Returns the episode ids that were fenced. Membership has already been
    given up by the time this runs, so a failure here must not resurrect it —
    each episode is fenced independently and a failure is logged rather than
    raised.
    """

    fenced: list[str] = []
    for episode in store.episodes(project_id, limit=500):
        authorizer = episode.authorized_by
        if authorizer is None or authorizer.user_id != user_id:
            continue
        if episode.status not in {"queued", "running", "stopping"}:
            continue
        try:
            _fence_one(store, background_tasks, project_id, episode, diagnostic=diagnostic)
        except (EpisodeNotRunning, KeyError, ValueError) as exc:
            # An episode that settled between the read and the press is already
            # fenced, which is the outcome this wanted.
            logger.info("Episode %s needed no membership fence: %s", episode.episode_id, exc)
            continue
        fenced.append(episode.episode_id)
    return fenced


def fence_episodes_for_removed_member(
    store: AppStore,
    background_tasks: BackgroundAgentTasks,
    user_id: str,
) -> list[str]:
    """Press the same per-project Stop owner after space-wide member removal."""

    fenced: list[str] = []
    preview = store.member_removal_preview(user_id)
    for episode_id in preview.active_episode_ids:
        episode = store.episode(episode_id)
        if episode is None:
            continue
        try:
            _fence_one(
                store,
                background_tasks,
                episode.project_id,
                episode,
                diagnostic=REMOVAL_FENCE_DIAGNOSTIC,
            )
        except (EpisodeNotRunning, KeyError, ValueError) as exc:
            logger.info("Episode %s needed no removal fence: %s", episode_id, exc)
            continue
        fenced.append(episode_id)
    return fenced


def _fence_one(
    store: AppStore,
    background_tasks: BackgroundAgentTasks,
    project_id: str,
    episode: EpisodeRecord,
    *,
    diagnostic: str,
) -> None:
    episode_id = episode.episode_id
    if episode.mode == "auto_research":
        stop_auto_research(background_tasks, episode_id)
        settle_auto_research_stop(store, episode_id, diagnostic=diagnostic)
        return
    node_id = _experiment_control_node(store, project_id, episode_id)
    if node_id is None:
        # Its Stop is still durable: the episode-level request is what fences
        # new admissions, and the loop reads it before claiming a wake.
        store.request_episode_stop(episode_id)
        return
    store.request_experiment_loop_stop(
        project_id,
        node_id,
        episode_id=episode.episode_id,
        graph_target=episode.graph_target,
    )


def _experiment_control_node(store: AppStore, project_id: str, episode_id: str) -> str | None:
    record = store.experiment_episode(episode_id)
    if record is None or record.project_id != project_id:
        return None
    return record.control_node_id
