from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rcp.runs.tasks.episode_report import EpisodeReportRunRequest
from rcp.skill_registry import official_registry
from rcp.storage import (
    AgentTaskRecord,
    AppStore,
    EpisodeEnding,
    EpisodeRecord,
    EpisodeWrapupRecord,
)
from rcp.storage.episodes import compact_episode_receipt

_REPORT_SKILL_ID = "episode-report"
_REPORT_OUTPUT_NAME = "episode-report.html"


class EpisodeWrapupSpec(BaseModel):
    """Mode-owned ending facts handed to the shared report admission seam."""

    model_config = ConfigDict(extra="forbid", strict=True)

    episode_id: str = Field(min_length=1)
    ending: EpisodeEnding
    partial: bool
    continuation_operation_id: str = Field(min_length=1)
    receipt: dict[str, object]
    diagnostic: str | None = None


@dataclass(frozen=True)
class EpisodeWrapupAdmission:
    episode: EpisodeRecord
    #: ``None`` when the ending had no session to report from, so no wrap-up began.
    wrapup: EpisodeWrapupRecord | None
    task: AgentTaskRecord | None
    request: EpisodeReportRunRequest | None

    @property
    def launchable(self) -> bool:
        return self.task is not None and self.request is not None


def begin_episode_report_wrapup(
    store: AppStore,
    spec: EpisodeWrapupSpec,
) -> EpisodeWrapupAdmission:
    """Fence one non-Stop ending and admit its single hidden report allocation.

    The mode adapter chooses the exact actor task whose native session owns the
    retrospective and supplies only compact ending facts. This shared seam does
    not inspect graph, research, transcript, or mode-specific history.
    """

    if spec.ending == "stopped":
        raise ValueError("Stop skips report generation instead of entering wrap-up.")
    episode = store.episode(spec.episode_id)
    if episode is None:
        raise KeyError(spec.episode_id)
    if store.episode_wrapup(spec.episode_id) is None and _never_bound_a_session(
        store.agent_task(spec.continuation_operation_id)
    ):
        # There is no episode session to resume, so this ending has no report to
        # generate rather than a report that failed. Settle it outright: fencing
        # into wrap-up first and discovering that here would park the episode on
        # the live `wrapping_up` status and post a report error for work that
        # never ran.
        ended = store.end_episode_without_report(
            spec.episode_id,
            ending=spec.ending,
            diagnostic=spec.diagnostic,
        )
        return EpisodeWrapupAdmission(ended, None, None, None)
    episode = store.fence_episode_ending(
        spec.episode_id,
        spec.ending,
        diagnostic=spec.diagnostic,
    )
    receipt_json, receipt_sha256 = compact_episode_receipt(
        {
            **spec.receipt,
            **({"diagnostic": spec.diagnostic} if spec.diagnostic is not None else {}),
            "ending": spec.ending,
            "episode_id": spec.episode_id,
            "mode": episode.mode,
            "partial": spec.partial,
        }
    )
    existing = store.episode_wrapup(spec.episode_id)
    if existing is not None:
        if (
            existing.ending != spec.ending
            or existing.partial != spec.partial
            or existing.concluding_operation_id != spec.continuation_operation_id
            or existing.receipt_json != receipt_json
            or existing.receipt_sha256 != receipt_sha256
        ):
            raise ValueError("The episode already has a different immutable wrap-up fence.")
        return _existing_admission(store, episode, existing)

    continuation = store.agent_task(spec.continuation_operation_id)
    now = store.now()
    diagnostic = _binding_diagnostic(episode, continuation)
    if diagnostic is not None:
        return _fail_unlaunchable(
            store,
            spec,
            receipt_json=receipt_json,
            receipt_sha256=receipt_sha256,
            diagnostic=diagnostic,
            now=now,
        )

    assert continuation is not None
    assert continuation.native_session_id is not None
    assert continuation.stage_root is not None
    try:
        request = EpisodeReportRunRequest.model_validate(
            {
                "episode_id": spec.episode_id,
                "provider": continuation.request.get("provider"),
                "model": continuation.request.get("model"),
                "reasoning": continuation.request.get("reasoning"),
                "run_on": continuation.request.get("run_on"),
                "execution_host": continuation.stage_host or "",
                "session_id": continuation.native_session_id,
            }
        )
        output_path = _report_output_path(
            stage_root=continuation.stage_root,
            stage_host=continuation.stage_host,
        )
    except (ValidationError, ValueError):
        return _fail_unlaunchable(
            store,
            spec,
            receipt_json=receipt_json,
            receipt_sha256=receipt_sha256,
            diagnostic="Episode report generation lost its frozen provider profile or stage.",
            now=now,
        )
    skill = official_registry().package("skill", _REPORT_SKILL_ID).reference()
    operation_id = str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"rcp:episode-report-allocation:{spec.episode_id}")
    )
    wrapup = EpisodeWrapupRecord(
        episode_id=spec.episode_id,
        ending=spec.ending,
        partial=spec.partial,
        concluding_operation_id=continuation.operation_id,
        allocation_operation_id=operation_id,
        provider=request.provider,
        run_on=request.run_on,
        execution_host=request.execution_host,
        native_session_id=request.session_id,
        stage_host=continuation.stage_host,
        stage_root=continuation.stage_root,
        skill_id=skill.id,
        skill_version=skill.version,
        output_name=_REPORT_OUTPUT_NAME,
        output_path=output_path,
        receipt_json=receipt_json,
        receipt_sha256=receipt_sha256,
        state="pending",
        diagnostic=spec.diagnostic,
        created_at=now,
        updated_at=now,
    )
    hidden = AgentTaskRecord(
        operation_id=operation_id,
        project_id=episode.project_id,
        episode_id=episode.episode_id,
        graph_target=episode.graph_target,
        kind="episode_report",
        status="queued",
        request=request.model_dump(mode="json"),
        created_at=now,
        updated_at=now,
        status_message="Wrapping up visualization and report",
        parent_operation_id=continuation.operation_id,
        native_session_id=continuation.native_session_id,
        stage_host=continuation.stage_host,
        stage_root=continuation.stage_root,
        phase="queued",
        authorized_by=episode.authorized_by,
        visible=False,
    )
    stored_episode, stored_wrapup, stored_task = store.begin_episode_wrapup(
        episode.episode_id,
        wrapup,
        hidden,
    )
    return EpisodeWrapupAdmission(stored_episode, stored_wrapup, stored_task, request)


def _existing_admission(
    store: AppStore,
    episode: EpisodeRecord,
    wrapup: EpisodeWrapupRecord,
) -> EpisodeWrapupAdmission:
    operation_id = wrapup.allocation_operation_id
    if operation_id is None:
        return EpisodeWrapupAdmission(episode, wrapup, None, None)
    task = store.agent_task(operation_id)
    if task is None:
        raise RuntimeError("The episode wrap-up lost its hidden report allocation.")
    request = EpisodeReportRunRequest.model_validate(task.request)
    return EpisodeWrapupAdmission(episode, wrapup, task, request)


def _fail_unlaunchable(
    store: AppStore,
    spec: EpisodeWrapupSpec,
    *,
    receipt_json: str,
    receipt_sha256: str,
    diagnostic: str,
    now: str,
) -> EpisodeWrapupAdmission:
    failed = EpisodeWrapupRecord(
        episode_id=spec.episode_id,
        ending=spec.ending,
        partial=spec.partial,
        concluding_operation_id=spec.continuation_operation_id,
        receipt_json=receipt_json,
        receipt_sha256=receipt_sha256,
        state="failed",
        diagnostic=diagnostic,
        created_at=now,
        updated_at=now,
        finished_at=now,
    )
    ended, stored = store.fail_episode_wrapup_unlaunchable(
        spec.episode_id,
        failed,
        ending_diagnostic=spec.diagnostic,
    )
    return EpisodeWrapupAdmission(ended, stored, None, None)


def _never_bound_a_session(continuation: AgentTaskRecord | None) -> bool:
    """Report the one unlaunchable cause that is an absence rather than a defect."""

    return continuation is not None and not (
        continuation.native_session_id and continuation.stage_root
    )


def _binding_diagnostic(
    episode: EpisodeRecord,
    continuation: AgentTaskRecord | None,
) -> str | None:
    if continuation is None:
        return "Episode report generation cannot find its exact continuation task."
    if (
        continuation.project_id != episode.project_id
        or continuation.episode_id != episode.episode_id
    ):
        return "Episode report generation cannot prove its continuation task lineage."
    if continuation.visible is False or continuation.kind == "episode_report":
        return "Episode report generation cannot continue from an internal task."
    if not continuation.native_session_id or not continuation.stage_root:
        return "Episode report generation has no exact saved native session and stage."
    required = ("provider", "model", "reasoning", "run_on")
    if any(not isinstance(continuation.request.get(field), str) for field in required):
        return "Episode report generation lost its frozen provider profile."
    return None


def _report_output_path(*, stage_root: str, stage_host: str | None) -> str:
    if stage_host:
        return str(PurePosixPath(stage_root) / "workspace" / _REPORT_OUTPUT_NAME)
    root = Path(stage_root)
    if not root.is_absolute():
        raise ValueError("Episode report local stage must be an absolute path.")
    return str(root / _REPORT_OUTPUT_NAME)
