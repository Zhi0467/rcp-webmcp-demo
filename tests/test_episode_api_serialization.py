from __future__ import annotations

import hashlib
import uuid

import pytest

from rcp.api.episodes import (
    EpisodeMessageBody,
    ReauthorizeEpisodeBody,
    StartEpisodeBody,
    episode_for_project,
    serialize_episode,
    serialize_episodes,
)
from rcp.core.authority import AgentDispatchAuthority, AgentDispatchScope
from rcp.core.models import AuthorizedHuman, GraphBranchSummary
from rcp.core.transition_models import GraphHeadRef, GraphTargetRef
from rcp.storage import (
    AgentTaskRecord,
    AppStore,
    AutoResearchStateRecord,
    EpisodeRecord,
    EpisodeReportRecord,
    EpisodeWrapupRecord,
    ProjectRecord,
)
from rcp.storage.episodes import compact_episode_receipt


def _project(store: AppStore, project_id: str = "project") -> None:
    store.upsert_project(
        ProjectRecord(
            project_id=project_id,
            locator=f"/tmp/{project_id}/research.yaml",
            name=project_id,
            state_location=f"/tmp/{project_id}/.research",
            state_remote=False,
            added_at=store.now(),
        )
    )


def _authorizer(store: AppStore) -> AuthorizedHuman:
    owner = store.local_owner
    assert owner is not None
    if owner.display_name is None:
        owner = store.rename_space_user(owner.user_id, "Researcher")
    return AuthorizedHuman(
        space_id=store.space_id,
        user_id=owner.user_id,
        display_name=owner.display_name,
    )


def _auto_episode(
    store: AppStore,
    episode_id: str,
    *,
    root_status: str = "succeeded",
    ending: str | None = None,
) -> tuple[EpisodeRecord, AgentTaskRecord]:
    episode_id = str(
        uuid.UUID(bytes=hashlib.sha256(episode_id.encode("utf-8")).digest()[:16], version=4)
    )
    now = store.now()
    graph_target = GraphTargetRef(kind="branch", branch_id=episode_id)
    episode = EpisodeRecord(
        episode_id=episode_id,
        project_id="project",
        mode="auto_research",
        graph_target=graph_target,
        graph_base_head=GraphHeadRef(revision=0),
        status="queued",
        invocation_ceiling=3,
        authorized_by=_authorizer(store),
        created_at=now,
        updated_at=now,
    )
    root = AgentTaskRecord(
        operation_id=f"{episode_id}-root",
        project_id="project",
        episode_id=episode_id,
        graph_target=graph_target,
        kind="auto_research",
        status="queued",
        request={
            "episode_id": episode_id,
            "role": "orchestrator",
            "actor_operation_id": f"{episode_id}-root",
            "run_truth_scope": ["repo"],
        },
        created_at=now,
        updated_at=now,
        status_message="queued",
        authorized_by=episode.authorized_by,
        dispatch_authority=AgentDispatchAuthority(
            profile="orchestrator",
            task_contract="orchestrate",
            scope=AgentDispatchScope(
                run_truth_scope=["repo"],
                episode_id=episode_id,
                patch_kind="work",
            ),
        ),
    )
    stored_episode, stored_root = store.create_auto_research_episode_with_root_task(
        episode,
        AutoResearchStateRecord(
            episode_id=episode_id,
            starting_instruction="Trace the strongest evidence.",
            created_at=now,
            updated_at=now,
        ),
        root,
    )
    if root_status == "succeeded":
        store.complete_agent_task(stored_root.operation_id, applied_revision=None, result={})
    elif root_status == "failed":
        store.fail_agent_task(stored_root.operation_id, "provider failed")
    elif root_status != "queued":
        raise ValueError(f"unsupported fixture root status: {root_status}")
    stored_root = store.agent_task(stored_root.operation_id)
    assert stored_root is not None
    if ending is not None:
        stored_episode = store.fence_episode_ending(episode_id, ending)
    return stored_episode, stored_root


def _branch_summary(episode: EpisodeRecord) -> GraphBranchSummary:
    assert episode.graph_base_head is not None
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
        merge_state="unmerged",
    )


def _begin_report(
    store: AppStore,
    episode: EpisodeRecord,
    root: AgentTaskRecord,
    *,
    ending: str,
) -> tuple[str, str]:
    now = store.now()
    allocation_operation_id = f"{episode.episode_id}-report"
    receipt_json, receipt_sha256 = compact_episode_receipt(
        {
            "ending": ending,
            "episode_id": episode.episode_id,
            "source_operation_id": root.operation_id,
        }
    )
    wrapup = EpisodeWrapupRecord(
        episode_id=episode.episode_id,
        ending=ending,
        partial=ending != "completed",
        concluding_operation_id=root.operation_id,
        allocation_operation_id=allocation_operation_id,
        provider="codex",
        run_on="local",
        execution_host="",
        native_session_id="native-session",
        stage_root="/tmp/episode-stage",
        skill_id="episode-report",
        skill_version="1",
        output_name="episode-report.html",
        output_path="/tmp/episode-stage/episode-report.html",
        receipt_json=receipt_json,
        receipt_sha256=receipt_sha256,
        state="pending",
        created_at=now,
        updated_at=now,
    )
    hidden_task = AgentTaskRecord(
        operation_id=allocation_operation_id,
        project_id=episode.project_id,
        episode_id=episode.episode_id,
        graph_target=episode.graph_target,
        kind="episode_report",
        status="queued",
        request={"provider": "codex", "run_on": "local", "execution_host": ""},
        created_at=now,
        updated_at=now,
        status_message="Wrapping up visualization and report",
        parent_operation_id=root.operation_id,
        native_session_id="native-session",
        stage_root="/tmp/episode-stage",
        visible=False,
    )
    store.begin_episode_wrapup(episode.episode_id, wrapup, hidden_task)
    attempt = store.allocate_episode_report_attempt(episode.episode_id)
    return allocation_operation_id, attempt.attempt_id


def test_auto_episode_projection_includes_mode_state_and_exact_recovery(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    episode, root = _auto_episode(store, "auto", root_status="failed")
    store.schedule_auto_research_task_recovery(
        root.operation_id,
        failure_kind="transport",
        retry_mode="exact",
        diagnostic="The exact host is temporarily unreachable.",
    )

    response = serialize_episode(
        store,
        "project",
        store.episode(episode.episode_id) or episode,
        branch_summary=_branch_summary,
    )

    assert response.starting_instruction == "Trace the strongest evidence."
    assert response.graph_target == episode.graph_target
    assert response.graph_base_head == episode.graph_base_head
    assert response.graph_branch is not None
    assert response.graph_branch.branch_id == episode.episode_id
    assert response.tasks[0].graph_target == episode.graph_target
    assert response.current_operation_id == root.operation_id
    assert response.current_orchestrator_task_id == root.operation_id
    assert response.current_control_task_id == root.operation_id
    assert response.recovery is not None
    assert response.recovery.model_dump() == {
        "purpose": "task",
        "status": "pending",
        "retry_mode": "exact",
        "operation_id": root.operation_id,
        "attempts": 0,
        "max_attempts": 3,
        "next_attempt_at": response.recovery.next_attempt_at,
    }
    assert response.budget.invocations_used == 1
    assert response.can_stop


def test_episode_route_bodies_are_strict_and_normalize_only_text() -> None:
    assert (
        StartEpisodeBody.model_validate(
            {
                "mode": "auto_research",
                "invocation_ceiling": 1,
                "starting_instruction": "  Investigate this.  ",
            }
        ).starting_instruction
        == "Investigate this."
    )
    assert (
        StartEpisodeBody.model_validate(
            {
                "mode": "auto_research",
                "invocation_ceiling": 1,
                "starting_instruction": "   ",
            }
        ).starting_instruction
        is None
    )
    assert EpisodeMessageBody.model_validate({"body": "  Status?  "}).body == "Status?"
    assert ReauthorizeEpisodeBody.model_validate({"invocation_ceiling": 1}).invocation_ceiling == 1

    with pytest.raises(ValueError):
        StartEpisodeBody.model_validate(
            {"mode": "auto_research", "invocation_ceiling": 1, "campaign_id": "legacy"}
        )
    with pytest.raises(ValueError):
        ReauthorizeEpisodeBody.model_validate({"additional_invocations": 2})
    with pytest.raises(ValueError):
        ReauthorizeEpisodeBody.model_validate({"invocation_ceiling": "2"})
    with pytest.raises(ValueError):
        EpisodeMessageBody.model_validate({"body": " \n "})


def test_ready_report_is_singular_and_hidden_report_work_is_not_public(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    episode, root = _auto_episode(store, "ready")
    allocation_id, attempt_id = _begin_report(store, episode, root, ending="exhausted")
    html = "<html><body><figure>Result</figure></body></html>"
    report = EpisodeReportRecord(
        report_id="report",
        episode_id=episode.episode_id,
        attempt_id=attempt_id,
        allocation_operation_id=allocation_id,
        ending="exhausted",
        sha256=hashlib.sha256(html.encode("utf-8")).hexdigest(),
        html=html,
        created_at=store.now(),
    )
    store.finish_episode_report_ready(attempt_id, report)

    stored = store.episode(episode.episode_id)
    assert stored is not None
    response = serialize_episode(store, "project", stored, branch_summary=_branch_summary)
    payload = response.model_dump(mode="json")

    assert response.report is not None
    assert response.report.model_dump() == {
        "report_id": "report",
        "ending": "exhausted",
        "created_at": report.created_at,
    }
    assert [task.operation_id for task in response.tasks] == [root.operation_id]
    assert response.budget.invocations_used == 1
    assert response.can_reauthorize
    assert response.run_section == "needs_action"
    assert "report_attempts_used" not in payload
    assert "stop_settled_at" not in payload
    assert "reports" not in payload
    assert "html" not in payload["report"]
    assert "visible" not in payload["tasks"][0]
    assert "dispatch_authority" not in payload["tasks"][0]
    assert all(task["kind"] != "episode_report" for task in payload["tasks"])


def test_failed_report_is_terminal_without_a_report_recovery_surface(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    episode, root = _auto_episode(store, "failed-report", root_status="failed")
    _, attempt_id = _begin_report(store, episode, root, ending="failed")
    store.finish_episode_report_error(attempt_id, "The report output was invalid.")

    stored = store.episode(episode.episode_id)
    assert stored is not None
    response = serialize_episode(store, "project", stored, branch_summary=_branch_summary)

    assert response.status == "failed"
    assert response.wrapup_state == "failed"
    assert response.wrapup_error == "The report output was invalid."
    assert response.report is None
    assert [task.operation_id for task in response.tasks] == [root.operation_id]
    assert response.tasks[0].can_retry is False
    assert response.tasks[0].can_resume is False
    assert not response.can_stop
    assert not response.can_reauthorize
    assert response.run_section == "needs_action"
    assert not {"report_retry", "report_resume"} & type(response).model_fields.keys()


def test_project_ownership_and_mode_filtered_lists_fail_closed(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    auto, _ = _auto_episode(store, "stopped")
    store.request_episode_stop(auto.episode_id)
    stopped = store.mark_episode_stop_skipped(auto.episode_id)

    now = store.now()
    experiment_id = "11111111-1111-4111-8111-111111111111"
    experiment = EpisodeRecord(
        episode_id=experiment_id,
        project_id="project",
        mode="experiment_loop",
        control_node_id="exp/one",
        status="queued",
        invocation_ceiling=2,
        authorized_by=_authorizer(store),
        created_at=now,
        updated_at=now,
    )
    experiment_task = AgentTaskRecord(
        operation_id="experiment-root",
        project_id="project",
        episode_id=experiment_id,
        kind="node_chat",
        status="queued",
        request={
            "chat_scope": "node",
            "chat_id": "experiment-chat",
            "node_id": "exp/one",
            "mode": "work",
            "trigger": "experiment_run",
            "run_truth_scope": ["repo"],
            "patch_kind": "experiment_loop",
            "control_node_id": "exp/one",
            "control_revision": 0,
            "control_episode_id": experiment_id,
            "control_invocation": 1,
            "control_invocation_ceiling": 2,
            "control_decision_bundle": [],
            "control_completion_criteria": [],
        },
        created_at=now,
        updated_at=now,
        status_message="Queued",
        authorized_by=experiment.authorized_by,
        dispatch_authority=AgentDispatchAuthority(
            profile="ordinary",
            task_contract="work_auto",
            scope=AgentDispatchScope(
                run_truth_scope=["repo"],
                chat_scope="node",
                chat_id="experiment-chat",
                node_id="exp/one",
                patch_kind="experiment_loop",
                control_node_id="exp/one",
                control_episode_id=experiment_id,
            ),
        ),
    )
    store.create_episode_with_invocation(experiment, experiment_task)

    auto_responses = serialize_episodes(
        store,
        "project",
        mode="auto_research",
        branch_summary=_branch_summary,
    )
    experiment_responses = serialize_episodes(store, "project", mode="experiment_loop")

    assert [item.episode_id for item in auto_responses] == [stopped.episode_id]
    assert auto_responses[0].report is None
    assert auto_responses[0].wrapup_error is None
    assert [item.episode_id for item in experiment_responses] == [experiment_id]
    assert experiment_responses[0].starting_instruction is None
    assert experiment_responses[0].recovery is None
    with pytest.raises(KeyError, match=stopped.episode_id):
        episode_for_project(store, "another-project", stopped.episode_id)
    with pytest.raises(KeyError, match=stopped.episode_id):
        serialize_episode(
            store,
            "another-project",
            stopped,
            branch_summary=_branch_summary,
        )


def test_the_projection_decides_lifecycle_state_so_no_surface_has_to() -> None:
    """Health, next step, and control come from one place, on backend inputs only.

    Four different situations reach `needs_action`, which is why the
    recommendation travels beside the health rather than being re-derived from
    `status` wherever it is needed.
    """

    from rcp.api.episodes import _episode_projection

    def episode(**fields: object) -> EpisodeRecord:
        base = {
            "episode_id": str(uuid.uuid4()),
            "project_id": "project",
            "mode": "auto_research",
            "status": "running",
            "invocation_ceiling": 4,
            "invocations_used": 1,
            "created_at": "2026-08-24T00:00:00Z",
            "updated_at": "2026-08-24T00:00:00Z",
        }
        return EpisodeRecord.model_validate({**base, **fields})

    def project(record: EpisodeRecord, **kwargs: object) -> tuple[str, str, str | None]:
        return _episode_projection(
            record,
            [],
            control_task_id=None,
            recovery=None,
            has_report=False,
            can_reauthorize=False,
            **kwargs,  # type: ignore[arg-type]
        )

    assert project(episode(wrapup_state="running")) == ("wrapping_up", "wait", None)
    assert project(episode(status="stopped", ending="stopped", wrapup_state="skipped")) == (
        "stopped",
        "none",
        None,
    )
    assert project(episode(status="failed", ending="failed")) == ("failed", "review", None)
    assert project(episode(status="queued")) == ("starting", "wait", None)
    assert project(episode()) == ("active", "continue", None)

    reauthorizable = _episode_projection(
        episode(status="needs_action", ending="exhausted", wrapup_state="ready"),
        [],
        control_task_id=None,
        recovery=None,
        has_report=False,
        can_reauthorize=True,
    )
    assert reauthorizable == ("needs_action", "reauthorize", None)


def test_a_stopped_episode_arrives_with_nothing_left_to_suppress(tmp_path) -> None:
    """Withholding beats guarding: what a stopped episode must not show is absent."""

    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    auto, _ = _auto_episode(store, "stopped-diagnostic")
    store.request_episode_stop(auto.episode_id)
    store.mark_episode_stop_skipped(auto.episode_id)
    with store.connection() as connection:
        connection.execute(
            "UPDATE episodes SET ending_diagnostic = ? WHERE episode_id = ?",
            ("must not reach the client", auto.episode_id),
        )

    stored = store.episode(auto.episode_id)
    assert stored is not None and stored.ending == "stopped"
    assert stored.ending_diagnostic == "must not reach the client"

    response = serialize_episode(store, "project", stored, branch_summary=_branch_summary)
    assert response.run_section == "completed"

    assert response.ending_diagnostic is None
    assert response.wrapup_error is None
    assert response.report is None
    assert response.can_message is False
    assert (response.health, response.recommendation) == ("stopped", "none")
