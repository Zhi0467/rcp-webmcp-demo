from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from rcp.runs.auto_research import AutoResearchRunRequest
from rcp.runs.task_policy import (
    load_stored_request,
    task_experiment_episode_id,
    task_graph_capable,
)
from rcp.runs.tasks.episode_report import EpisodeReportRunRequest
from rcp.service import CoachRequest, RunRequest


@pytest.mark.parametrize("kind", ["seed", "refresh"])
def test_ingest_tasks_are_graph_capable(kind: str) -> None:
    assert task_graph_capable(kind, RunRequest())
    assert task_graph_capable(kind, {})


@pytest.mark.parametrize("kind", ["node_chat", "project_chat"])
def test_only_work_chats_are_graph_capable(kind: str) -> None:
    assert task_graph_capable(kind, RunRequest(mode="work"))
    assert task_graph_capable(kind, {"mode": "work"})
    assert not task_graph_capable(kind, RunRequest(mode="discuss"))
    assert not task_graph_capable(kind, {"mode": "discuss"})


@pytest.mark.parametrize("role", ["orchestrator", "worker"])
def test_auto_research_graph_capability_uses_explicit_actor_allowlist(role: str) -> None:
    values = {
        "episode_id": "episode-1",
        "role": role,
        **({"control_node_id": "experiment-1"} if role == "worker" else {}),
    }
    request = AutoResearchRunRequest.model_validate(values)

    assert task_graph_capable("auto_research", request)
    assert task_graph_capable("auto_research", values)


@pytest.mark.parametrize(
    "candidate",
    [
        {"episode_id": "episode-1", "role": "report"},
        {"episode_id": "episode-1", "role": "unknown"},
        {"episode_id": "episode-1", "role": []},
        {"episode_id": "episode-1", "role": {}},
        {"role": "orchestrator"},
        object(),
    ],
)
def test_auto_research_unknown_or_invalid_request_shapes_default_deny(candidate: object) -> None:
    assert not task_graph_capable("auto_research", candidate)


def test_episode_report_and_unknown_task_kinds_are_not_graph_capable() -> None:
    report = EpisodeReportRunRequest(
        episode_id="episode-1",
        provider="codex",
        model="gpt-5",
        reasoning="high",
        run_on="local",
        execution_host="local",
        session_id="session-1",
    )

    assert not task_graph_capable("episode_report", report)
    assert not task_graph_capable("auto_research", report)
    assert not task_graph_capable("unknown", {"mode": "work"})
    assert not task_graph_capable("refresh", CoachRequest(message="hello"))


def test_experiment_episode_id_is_selected_only_for_live_experiment_requests() -> None:
    assert (
        task_experiment_episode_id(
            RunRequest(patch_kind="experiment_loop", control_episode_id="episode-1")
        )
        == "episode-1"
    )
    assert task_experiment_episode_id(RunRequest(patch_kind="experiment_loop")) == ""
    assert (
        task_experiment_episode_id(RunRequest(patch_kind="work", control_episode_id="episode-1"))
        is None
    )
    assert (
        task_experiment_episode_id(
            {"patch_kind": "experiment_loop", "control_episode_id": "episode-1"}
        )
        is None
    )
    assert task_experiment_episode_id(CoachRequest(message="hello")) is None


def test_stored_request_keeps_every_declared_field_strict() -> None:
    """A field the model still declares is validated exactly as before."""

    with pytest.raises(ValidationError):
        load_stored_request(RunRequest, {"mode": "not-a-mode"})


def test_stored_request_drops_only_fields_this_build_removed() -> None:
    """RCP must stay able to read a task it wrote before a field was deleted.

    Regression: every stored ``auto_research`` request carried an ``ending`` key
    that the model later dropped, so Retry answered a raw validation dump and
    the task was permanently unrecoverable.
    """

    stored = {"episode_id": "episode-1", "role": "orchestrator", "ending": None}
    request = load_stored_request(AutoResearchRunRequest, stored, operation_id="op-1")
    assert request.episode_id == "episode-1"
    assert request.role == "orchestrator"
    assert not hasattr(request, "ending")


def test_dropping_a_stored_field_is_logged_rather_than_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The compatibility read stays observable, so drift is visible in the log."""

    with caplog.at_level(logging.WARNING, logger="rcp.storage.request_compat"):
        load_stored_request(
            AutoResearchRunRequest,
            {"episode_id": "episode-1", "role": "orchestrator", "ending": None},
            operation_id="op-1",
        )
    assert any(
        "ending" in record.getMessage() and "op-1" in record.getMessage()
        for record in caplog.records
    )


def test_a_live_request_still_cannot_smuggle_an_unknown_field() -> None:
    """Tolerance is for RCP's own records only; the model itself stays strict."""

    with pytest.raises(ValidationError):
        AutoResearchRunRequest.model_validate(
            {"episode_id": "episode-1", "role": "orchestrator", "ending": None}
        )
