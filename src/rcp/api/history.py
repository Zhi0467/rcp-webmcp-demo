from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict

from rcp.api.dependencies import (
    get_catalog,
    get_project_service,
    get_store,
    require_project_membership,
)
from rcp.api.episodes import EpisodeReportSummary
from rcp.core.transitions import transition_trigger_manifest
from rcp.projects import ProjectCatalog
from rcp.storage import AppStore
from rcp.storage.models import (
    EpisodeEnding,
    EpisodeMode,
    EpisodeRecord,
    EpisodeWrapupState,
)

router = APIRouter(dependencies=[Depends(require_project_membership)])

CatalogDependency = Annotated[ProjectCatalog, Depends(get_catalog)]
StoreDependency = Annotated[AppStore, Depends(get_store)]


@router.get("/api/projects/{project_id}/history")
def history(
    project_id: str,
    from_revision: int = 1,
    to_revision: int | None = None,
    *,
    catalog: CatalogDependency,
):
    service = get_project_service(catalog, project_id)
    return service.history.slice(from_revision, to_revision)


@router.get("/api/projects/{project_id}/history/summaries")
def history_summaries(
    project_id: str,
    from_revision: int = 1,
    to_revision: int | None = None,
    *,
    catalog: CatalogDependency,
    store: StoreDependency,
):
    service = get_project_service(catalog, project_id)
    summaries = service.history.revision_summaries(from_revision, to_revision)
    episode_ids = {
        episode_id
        for summary in summaries
        if isinstance(episode_id := summary.get("episode_id"), str)
    }
    episodes = {
        episode_id: _history_episode_decoration(store, project_id, episode_id)
        for episode_id in episode_ids
    }
    return [
        {
            **summary,
            "episode": (
                decoration.model_dump(mode="json")
                if (decoration := episodes.get(summary.get("episode_id"))) is not None
                else None
            ),
        }
        for summary in summaries
    ]


@router.get("/api/projects/{project_id}/transition-manifest")
def graph_transition_manifest(
    project_id: str,
    *,
    catalog: CatalogDependency,
):
    get_project_service(catalog, project_id)
    return transition_trigger_manifest().model_dump(mode="json")


class HistoryEpisodeDecoration(BaseModel):
    """The episode facts a history row renders, declared like every other response.

    This was a bare dict, so its keys were checked by nothing. A response the web
    layer consumes without deriving anything from it still has to be a contract,
    or the client is trusting a shape the server never promised.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    mode: EpisodeMode
    state_label: str
    ending: EpisodeEnding | None
    wrapup_state: EpisodeWrapupState
    report: EpisodeReportSummary | None


def _episode_state_label(episode: EpisodeRecord) -> str:
    if episode.ending == "human_pause":
        return "Human-authority pause"
    state = episode.ending or episode.status
    return state.replace("_", " ").capitalize()


def _history_episode_decoration(
    store: AppStore,
    project_id: str,
    episode_id: str,
) -> HistoryEpisodeDecoration | None:
    episode = store.episode(episode_id)
    if episode is None or episode.project_id != project_id:
        return None
    report = None if episode.ending == "stopped" else store.episode_report(episode_id)
    return HistoryEpisodeDecoration(
        mode=episode.mode,
        # The row shows what state this episode reached. That is a rendered name,
        # so the projection supplies it rather than exporting the enum for a
        # surface to capitalize into one.
        state_label=_episode_state_label(episode),
        ending=episode.ending,
        wrapup_state=episode.wrapup_state,
        report=(
            EpisodeReportSummary(
                report_id=report.report_id,
                ending=report.ending,
                created_at=report.created_at,
            )
            if report is not None
            else None
        ),
    )


__all__ = [
    "graph_transition_manifest",
    "history",
    "history_summaries",
    "router",
]
