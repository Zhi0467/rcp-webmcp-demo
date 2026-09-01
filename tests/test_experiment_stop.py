from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from pathlib import Path
from threading import Event

import pytest
from fastapi.testclient import TestClient

from rcp.agents import AgentEvent, AgentProcessControl
from rcp.api.task_requests import _resolved_graph_request
from rcp.background import AgentTaskExecution, BackgroundAgentTasks
from rcp.core.models import Patch
from rcp.core.transition_models import GraphTargetRef
from rcp.runs.episodes.reconcile import EpisodeReconciler
from rcp.runs.experiment_loop import commit_experiment_episode_binding
from rcp.runs.shared import _sse
from rcp.runs.watcher_admission import start_watcher_notification
from rcp.service import RunRequest, resolve_dispatch_authority
from rcp.skill_registry import SkillReference
from rcp.storage import AgentTaskRecord, AppStore, WatcherContinuation, WatcherRecord
from rcp.watchers import WatcherBinding

from .helpers import append_fixture_patch, authorized_human, seed_patch, wait_for_task
from .helpers import create_named_app as create_app

EXPERIMENT_ID = "exp/bounded-loop"
NODE_PATH = "exp%2Fbounded-loop"


def _task_authority(request: RunRequest, *, kind: str = "node_chat"):
    authority = resolve_dispatch_authority(kind, request)
    assert authority is not None
    return authority


def _experiment_patch(*, invocation_ceiling: int = 2) -> Patch:
    return Patch(
        kind="refresh",
        author="agent",
        summary="Added an experiment for graceful-stop tests.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": EXPERIMENT_ID,
                        "type": "experiment",
                        "title": "Bounded loop",
                        "objective": "Exercise the graceful-stop contract.",
                        "completion_criteria": ["The detached fixture exits cleanly."],
                        "invocation_ceiling": invocation_ceiling,
                    }
                ],
            }
        ],
    )


def _experiment_status_patch(status: str) -> Patch:
    return Patch(
        kind="refresh",
        author="agent",
        summary=f"Set the graceful-stop Experiment to {status}.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "update_nodes",
                "nodes": [{"id": EXPERIMENT_ID, "changes": {"status": status}}],
            }
        ],
    )


class _Loop:
    """One project holding a bounded Experiment with a persisted loop episode."""

    def __init__(self, app, *, invocation_ceiling: int = 2) -> None:
        self.app = app
        self.service = app.state.service
        append_fixture_patch(self.service, seed_patch())
        append_fixture_patch(self.service, _experiment_patch(invocation_ceiling=invocation_ceiling))
        self.control_revision = self.service.history.state().revision
        self.project_id = app.state.default_project_id
        self.store: AppStore = app.state.background_tasks.store
        self.authorizer = authorized_human(self.store)
        self.client = TestClient(app)
        self.chat_id = str(uuid.uuid4())
        self.episode_id = str(uuid.uuid4())
        self.invocation_ceiling = invocation_ceiling

    def root_request(self, *, invocation: int = 1) -> RunRequest:
        return RunRequest(
            provider="codex",
            model="gpt-5",
            reasoning="medium",
            run_on="laptop",
            run_truth_scope=["repo-a"],
            chat_id=self.chat_id,
            chat_scope="node",
            node_id=EXPERIMENT_ID,
            message="Begin a bounded Experiment-loop episode.",
            mode="work",
            trigger="experiment_run",
            patch_kind="experiment_loop",
            control_node_id=EXPERIMENT_ID,
            control_revision=self.control_revision,
            control_episode_id=self.episode_id,
            control_invocation=invocation,
            control_invocation_ceiling=self.invocation_ceiling,
            control_decision_bundle=[],
            control_completion_criteria=["The detached fixture exits cleanly."],
        )

    def _set_task_status(self, record: AgentTaskRecord) -> AgentTaskRecord:
        if record.status == "running":
            self.store.mark_agent_task_running(record.operation_id)
            self.store.update_agent_task_message(
                record.operation_id,
                record.status_message,
                phase=record.phase,
            )
        elif record.status == "succeeded":
            self.store.complete_agent_task(
                record.operation_id,
                applied_revision=record.applied_revision,
                result=record.result or {},
            )
        elif record.status == "paused":
            self.store.pause_agent_task(
                record.operation_id,
                detail=record.status_message,
                result=record.result,
            )
        elif record.status in {"failed", "interrupted"}:
            self.store.fail_agent_task(
                record.operation_id,
                record.error or record.status_message,
                status=record.status,
                result=record.result,
            )
        elif record.status != "queued":
            raise AssertionError(f"Unsupported fixture task status: {record.status}")
        stored = self.store.agent_task(record.operation_id)
        assert stored is not None
        return stored

    def start_episode(
        self,
        *,
        status: str = "succeeded",
        operation_id: str = "loop-root",
        request: RunRequest | None = None,
    ) -> str:
        now = self.store.now()
        request = request or self.root_request()
        dispatch_authority = _task_authority(request)
        record = AgentTaskRecord(
            operation_id=operation_id,
            project_id=self.project_id,
            episode_id=request.control_episode_id,
            kind="node_chat",
            status=status,
            request=request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="Working on the bounded loop.",
            phase="agent",
            last_activity_at=now,
            authorized_by=self.authorizer,
            dispatch_authority=dispatch_authority,
        )
        self.store.create_experiment_episode_with_invocation(
            record.model_copy(update={"status": "queued", "phase": "queued"})
        )
        self._set_task_status(record)
        return operation_id

    def create_watcher_invocation(
        self,
        record: AgentTaskRecord,
        watcher_ids: list[str],
    ) -> AgentTaskRecord:
        queued = record.model_copy(
            update={
                "episode_id": self.episode_id,
                "status": "queued",
                "error": None,
                "result": None,
                "phase": "queued",
            }
        )
        stored = self.store.create_experiment_watcher_invocation(queued, watcher_ids)
        assert stored is not None
        return self._set_task_status(record)

    def create_recovery_task(
        self,
        record: AgentTaskRecord,
        *,
        continuation_cause: str = "resume",
    ) -> AgentTaskRecord:
        queued = record.model_copy(
            update={
                "episode_id": self.episode_id,
                "status": "queued",
                "error": None,
                "result": None,
                "phase": "queued",
            }
        )
        self.store.create_experiment_recovery_task(
            queued,
            continuation_cause=continuation_cause,
        )
        return self._set_task_status(record)

    def bind_session(
        self,
        stage_root: Path,
        *,
        provider: str = "codex",
        native_session_id: str = "native-session-abc",
        operation_id: str = "loop-root",
    ) -> None:
        stage_root.mkdir(parents=True, exist_ok=True)
        self.store.commit_experiment_episode_turn(
            episode_id=self.episode_id,
            project_id=self.project_id,
            control_node_id=EXPERIMENT_ID,
            provider=provider,
            execution_machine="laptop",
            execution_host="",
            native_session_id=native_session_id,
            stage_host=None,
            stage_root=str(stage_root),
            chat_id=self.chat_id,
            operation_id=operation_id,
            invocation=1,
            graph_result="applied",
            watcher_ids=[],
            context_baseline={},
        )

    def continuation(self, **overrides: object) -> WatcherContinuation:
        base: dict[str, object] = {
            "provider": "codex",
            "model": "gpt-5",
            "reasoning": "medium",
            "run_on": "laptop",
            "run_truth_scope": ["repo-a"],
            "patch_kind": "experiment_loop",
            "control_node_id": EXPERIMENT_ID,
            "control_revision": self.control_revision,
            "control_episode_id": self.episode_id,
            "control_invocation": 1,
            "control_invocation_ceiling": self.invocation_ceiling,
            "control_decision_bundle": [],
            "control_completion_criteria": ["The detached fixture exits cleanly."],
        }
        base.update(overrides)
        return WatcherContinuation.model_validate(base)

    def arm_watcher(
        self,
        watcher_id: str,
        *,
        status: str = "active",
        continuation: WatcherContinuation | None = None,
        origin_operation_id: str = "loop-root",
        execution_host: str = "",
    ) -> WatcherRecord:
        now = self.store.now()
        record = WatcherRecord(
            watcher_id=watcher_id,
            project_id=self.project_id,
            origin_operation_id=origin_operation_id,
            origin_task_kind="node_chat",
            chat_id=self.chat_id,
            node_id=EXPERIMENT_ID,
            execution_host=execution_host,
            check_command="true",
            log_path=f"/tmp/{watcher_id}.log",
            cwd="/tmp",
            continuation=continuation or self.continuation(),
            status="active",
            created_at=now,
        )
        self.store.create_watchers([record])
        if status != "active":
            self.store.record_watcher_check(
                watcher_id,
                status=status,
                exit_code=0,
                error=None,
            )
        stored = self.store.watcher(watcher_id)
        assert stored is not None
        return stored

    def stop(self) -> dict[str, object]:
        response = self.client.post(f"/api/projects/{self.project_id}/experiments/{NODE_PATH}/stop")
        assert response.status_code == 200, response.text
        return response.json()

    def control(self) -> dict[str, object]:
        snapshot = self.client.get(f"/api/projects/{self.project_id}").json()
        return snapshot["experiment_control"][EXPERIMENT_ID]

    def deliver(self, *watcher_ids: str) -> None:
        deliver = self.app.state.watcher_poller.on_completed
        assert deliver is not None
        group = []
        for watcher_id in watcher_ids:
            record = self.store.watcher(watcher_id)
            assert record is not None
            group.append(record)
        deliver(group)

    def loop_task_ids(self) -> set[str]:
        return {
            record.operation_id
            for record in self.store.agent_tasks(self.project_id)
            if record.request.get("patch_kind") == "experiment_loop"
        }

    def record_answers(self) -> list[RunRequest]:
        """Replace agent execution with a stub that only records its request."""

        seen: list[RunRequest] = []

        async def stream(_project_id, _kind, request, _execution):
            seen.append(request)
            yield _sse(AgentEvent(event="answer", text="Stub turn."))
            yield _sse(AgentEvent(event="done"))

        self.app.state.background_tasks.stream = stream
        return seen


def test_stop_while_a_turn_runs_leaves_the_task_alone_and_blocks_a_fresh_run(
    manifest, tmp_path
) -> None:
    loop = _Loop(create_app(str(manifest.path), data_dir=tmp_path / "data"))
    loop.start_episode(status="running")
    watcher = loop.arm_watcher("live-watcher")

    control = loop.stop()

    operational = control["operational"]
    assert operational["stop_requested"] is True
    assert operational["stop_settled"] is False
    assert operational["task_active"] is True
    assert control["ready"] is False
    assert "A graceful stop is finishing the current loop turn." in control["reasons"]
    projected = loop.control()
    assert {
        field: projected[field]
        for field in (
            "health",
            "recommendation",
            "run_section",
            "live",
            "can_start",
            "can_stop",
            "stop_pending",
            "task_control",
        )
    } == {
        "health": "stopping",
        "recommendation": "wait",
        "run_section": "running",
        "live": True,
        "can_start": False,
        "can_stop": False,
        "stop_pending": True,
        "task_control": None,
    }
    # The authorized turn is untouched, and its observers stay live until it ends.
    task = loop.store.agent_task("loop-root")
    assert task is not None and task.status == "running"
    assert loop.store.watcher(watcher.watcher_id).status == "active"
    assert watcher.watcher_id not in {
        record.watcher_id for record in loop.store.pollable_watchers()
    }

    refused = loop.client.post(
        f"/api/projects/{loop.project_id}/experiments/{NODE_PATH}/run",
        json={"chat_id": str(uuid.uuid4()), "run_truth_scope": ["repo-a"]},
    )
    assert refused.status_code == 409
    assert "graceful stop" in refused.json()["detail"]

    loop.store.complete_agent_task("loop-root", applied_revision=None, result={})
    assert loop.store.settle_ready_experiment_loop_stops() == 1
    assert loop.store.watcher(watcher.watcher_id).status == "stopped"


def test_stop_with_only_watchers_left_terminalizes_them_and_settles_at_once(
    manifest, tmp_path
) -> None:
    loop = _Loop(create_app(str(manifest.path), data_dir=tmp_path / "data"))
    loop.start_episode()
    loop.arm_watcher("still-running")
    loop.arm_watcher("finished-unclaimed", status="completed")

    control = loop.stop()

    assert control["operational"]["stop_requested"] is True
    assert control["operational"]["stop_settled"] is True
    for watcher_id in ("still-running", "finished-unclaimed"):
        record = loop.store.watcher(watcher_id)
        # Retained as inspectable history, never deleted and never deliverable.
        assert record is not None
        assert record.status == "stopped"
        assert record.notified is True
        assert record.notification_operation_id is None
    assert control["ready"] is True
    assert control["reasons"] == []


def test_stop_is_idempotent(manifest, tmp_path) -> None:
    loop = _Loop(create_app(str(manifest.path), data_dir=tmp_path / "data"))
    loop.start_episode()
    loop.arm_watcher("finished-unclaimed", status="completed")

    first = loop.stop()
    episode_after_first = loop.store.experiment_episode(loop.episode_id)
    watcher_after_first = loop.store.watcher("finished-unclaimed")
    tasks_after_first = loop.loop_task_ids()

    second = loop.stop()

    assert second == first
    episode_after_second = loop.store.experiment_episode(loop.episode_id)
    assert episode_after_second is not None and episode_after_first is not None
    assert episode_after_second.stop_requested_at == episode_after_first.stop_requested_at
    assert episode_after_second.stop_settled_at == episode_after_first.stop_settled_at
    assert loop.store.watcher("finished-unclaimed") == watcher_after_first
    assert loop.loop_task_ids() == tasks_after_first


def test_closed_experiment_outranks_a_stopped_episode_until_the_node_is_reopened(
    manifest, tmp_path
) -> None:
    loop = _Loop(create_app(str(manifest.path), data_dir=tmp_path / "data"))
    append_fixture_patch(loop.service, _experiment_status_patch("completed"))
    loop.start_episode(status="paused")

    loop.stop()

    closed = loop.control()
    assert closed["node_closed"] is True
    assert closed["health"] == "completed"
    assert closed["recommendation"] == "none"
    assert closed["run_section"] == "completed"
    assert closed["can_start"] is False
    assert closed["reasons"] == [
        "This Experiment is completed. Edit its status before starting a new episode."
    ]

    refused = loop.client.post(
        f"/api/projects/{loop.project_id}/experiments/{NODE_PATH}/run",
        json={"chat_id": str(uuid.uuid4()), "run_truth_scope": ["repo-a"]},
    )
    assert refused.status_code == 409
    assert refused.json()["detail"] == closed["reasons"][0]

    append_fixture_patch(loop.service, _experiment_status_patch("running"))
    assert loop.service.history.state().nodes[EXPERIMENT_ID].status == "running"

    _, snapshot = loop.app.state.services.project_display_cache.open_snapshot(loop.project_id)
    reopened = snapshot["experiment_control"][EXPERIMENT_ID]
    assert reopened["node_closed"] is False
    assert reopened["health"] == "human_stopped"
    assert reopened["recommendation"] == "start_episode"
    assert reopened["run_section"] == "actionable"
    assert reopened["can_start"] is True
    assert reopened["reasons"] == []


def test_experiment_control_projection_reads_runtime_and_episode_from_one_snapshot(
    manifest,
    tmp_path,
    monkeypatch,
) -> None:
    loop = _Loop(create_app(str(manifest.path), data_dir=tmp_path / "data"))
    loop.start_episode()
    original = loop.store._experiment_episode_projection_snapshot_in_connection
    stop_written = False

    def write_stop_between_runtime_and_episode_reads(
        connection,
        project_id,
        control_node_id,
        episode_id,
    ):
        nonlocal stop_written
        if not stop_written:
            stop_written = True
            loop.store.request_experiment_loop_stop(loop.project_id, EXPERIMENT_ID)
        return original(connection, project_id, control_node_id, episode_id)

    monkeypatch.setattr(
        loop.store,
        "_experiment_episode_projection_snapshot_in_connection",
        write_stop_between_runtime_and_episode_reads,
    )

    read_model = loop.store.experiment_control_projection_snapshots(
        loop.project_id,
        [EXPERIMENT_ID],
        graph_target=GraphTargetRef(),
    )[EXPERIMENT_ID]

    assert stop_written is True
    assert read_model.runtime.stop_requested is False
    assert read_model.episode is not None
    assert read_model.episode.episode.stop_requested_at is None
    current = loop.store.episode(loop.episode_id)
    assert current is not None and current.stop_requested_at is not None


@pytest.mark.parametrize(
    ("status", "expected_continuation"),
    [("running", "resume"), ("paused", "resume"), ("failed", "retry")],
)
def test_restart_recovers_a_healthy_authorized_turn_behind_the_stop_fence(
    manifest,
    tmp_path,
    status,
    expected_continuation,
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    loop = _Loop(app, invocation_ceiling=3)
    loop.start_episode(status=status)
    stage = tmp_path / f"restart-{status}-stage"
    stage.mkdir()
    loop.store.checkpoint_agent_task(
        "loop-root",
        native_session_id=f"restart-{status}-session",
        stage_root=str(stage),
    )
    candidate = "{}"
    loop.store.record_agent_task_contract(
        "loop-root",
        "experiment_episode_context_candidate",
        candidate,
        hashlib.sha256(candidate.encode()).hexdigest(),
    )
    stopping = loop.store.request_experiment_loop_stop(loop.project_id, EXPERIMENT_ID)
    assert stopping is not None and stopping.stop_settled_at is None
    observed = Event()
    captured: dict[str, object] = {}

    async def stream(_project_id, _kind, request, execution):
        captured.update(request=request, continuation=execution.continuation)
        observed.set()
        yield _sse(AgentEvent(event="done"))

    BackgroundAgentTasks(loop.store, stream).recover_at_startup()

    assert observed.wait(timeout=2)
    recoveries = [
        task
        for task in loop.store.episode_tasks(loop.episode_id)
        if task.parent_operation_id == "loop-root"
    ]
    assert len(recoveries) == 1
    recovered = wait_for_task(loop.store, recoveries[0].operation_id, expect="succeeded")
    request = captured["request"]
    assert isinstance(request, RunRequest)
    assert request.session_id == f"restart-{status}-session"
    assert captured["continuation"] == expected_continuation
    assert recovered.stage_root == str(stage)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        settled_episode = loop.store.episode(loop.episode_id)
        if settled_episode is not None and settled_episode.status == "stopped":
            break
        time.sleep(0.01)
    else:
        raise AssertionError("the recovered turn did not settle the durable Stop fence")
    assert loop.store.episode(loop.episode_id).invocations_used == 1  # type: ignore[union-attr]
    assert "experiment_stop_recovery" in {
        receipt.category for receipt in loop.store.agent_task_receipts(recovered.operation_id)
    }


def test_restart_keeps_stop_recovery_pending_when_remote_stage_probe_is_uncertain(
    manifest,
    tmp_path,
    monkeypatch,
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    loop = _Loop(app, invocation_ceiling=3)
    loop.start_episode(status="running")
    loop.store.checkpoint_agent_task(
        "loop-root",
        native_session_id="remote-stop-session",
        stage_host="worker.example",
        stage_root="/remote/rcp-stage",
    )
    candidate = "{}"
    loop.store.record_agent_task_contract(
        "loop-root",
        "experiment_episode_context_candidate",
        candidate,
        hashlib.sha256(candidate.encode()).hexdigest(),
    )
    stopping = loop.store.request_experiment_loop_stop(loop.project_id, EXPERIMENT_ID)
    assert stopping is not None and stopping.stop_settled_at is None
    monkeypatch.setattr(
        "rcp.background.RemoteRunStage.directory_exists",
        lambda _stage, _path: None,
    )

    async def stream(*_args, **_kwargs):
        raise AssertionError("transient remote uncertainty must not launch recovery")
        yield  # pragma: no cover

    BackgroundAgentTasks(loop.store, stream).recover_at_startup()

    tasks = loop.store.episode_tasks(loop.episode_id)
    assert [task.operation_id for task in tasks] == ["loop-root"]
    episode = loop.store.episode(loop.episode_id)
    assert episode is not None and episode.status == "stopping"
    assert episode.stop_settled_at is None


def test_run_after_a_settled_stop_starts_a_fresh_episode_with_no_delivery(
    manifest, tmp_path
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    loop = _Loop(app)
    loop.start_episode()
    loop.arm_watcher("finished-unclaimed", status="completed")
    loop.stop()

    loop.record_answers()
    response = loop.client.post(
        f"/api/projects/{loop.project_id}/experiments/{NODE_PATH}/run",
        json={"chat_id": str(uuid.uuid4()), "run_truth_scope": ["repo-a"]},
    )

    assert response.status_code == 202, response.text
    request = response.json()["request"]
    assert request["trigger"] == "experiment_run"
    assert request["control_invocation"] == 1
    assert request["control_episode_id"] != loop.episode_id
    # A settled stop leaves no unnotified completion, so this is an ordinary
    # fresh Run rather than a human reauthorization of pending watcher state.
    assert request["watcher_ids"] == []
    assert loop.store.watcher("finished-unclaimed").notification_operation_id is None


def test_an_unclaimed_completion_cannot_win_a_wake_after_a_persisted_stop(
    manifest, tmp_path
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    loop = _Loop(app)
    loop.start_episode()
    loop.bind_session(tmp_path / "stage")
    loop.arm_watcher("finished-unclaimed", status="completed")

    loop.store.request_experiment_loop_stop(loop.project_id, EXPERIMENT_ID)
    before = loop.loop_task_ids()
    loop.deliver("finished-unclaimed")

    assert loop.loop_task_ids() == before
    assert loop.store.watcher("finished-unclaimed").notification_operation_id is None
    runtime = loop.store.experiment_loop_runtime(loop.project_id, EXPERIMENT_ID)
    # The session was usable; only the stop refused the wake, and no budget moved.
    assert runtime.session_bound is True
    assert runtime.invocations_used == 1


def test_a_stop_settles_on_the_next_derivation_after_its_turn_ends(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    loop = _Loop(app)
    loop.start_episode(status="running")
    loop.arm_watcher("still-running")
    assert loop.stop()["operational"]["stop_settled"] is False

    # The authorized turn finishes on its own. Nothing replays the stop; the
    # next derivation reconciles it, which is what survives a restart.
    loop.store.complete_agent_task("loop-root", applied_revision=None, result={})

    control = loop.control()
    assert control["operational"]["stop_requested"] is True
    assert control["operational"]["stop_settled"] is True
    assert loop.store.watcher("still-running").status == "stopped"
    assert control["ready"] is True


def test_a_wake_without_a_committed_binding_never_claims_or_spends_budget(
    manifest, tmp_path
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    loop = _Loop(app)
    loop.start_episode()
    loop.arm_watcher("finished-unclaimed", status="completed")

    before = loop.loop_task_ids()
    loop.deliver("finished-unclaimed")

    assert loop.loop_task_ids() == before
    record = loop.store.watcher("finished-unclaimed")
    assert record.status == "completed"
    assert record.notified is False
    control = loop.control()
    assert control["invocations_used"] == 1
    assert control["operational"]["session"]["native_session_bound"] is False
    assert "no validated native provider session" in control["operational"]["session"]["diagnostic"]


def test_a_vanished_episode_stage_becomes_a_durable_diagnostic(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    loop = _Loop(app)
    loop.start_episode()
    stage = tmp_path / "stage"
    loop.bind_session(stage)
    stage.rmdir()
    loop.arm_watcher("finished-unclaimed", status="completed")

    before = loop.loop_task_ids()
    loop.deliver("finished-unclaimed")

    assert loop.loop_task_ids() == before
    record = loop.store.watcher("finished-unclaimed")
    assert record.status == "completed"
    assert record.notified is False
    diagnostic = loop.control()["operational"]["session"]["diagnostic"]
    assert diagnostic is not None
    assert "saved provider workspace is gone" in diagnostic
    assert loop.store.experiment_loop_runtime(loop.project_id, EXPERIMENT_ID).invocations_used == 1


def test_provider_provenance_does_not_block_current_episode_delivery(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    loop = _Loop(app)
    loop.start_episode()
    loop.bind_session(tmp_path / "stage")
    loop.record_answers()
    loop.arm_watcher(
        "other-provider",
        status="completed",
        continuation=loop.continuation(provider="claude"),
    )

    loop.deliver("other-provider")

    record = loop.store.watcher("other-provider")
    assert record.status == "completed"
    assert record.notified is True
    woken = [
        item
        for item in loop.store.agent_tasks(loop.project_id)
        if item.request.get("trigger") == "watcher"
    ]
    assert len(woken) == 1
    assert woken[0].request["provider"] == "codex"
    assert woken[0].request["chat_id"] == loop.chat_id
    assert loop.control()["operational"]["session"]["diagnostic"] is None


def test_a_ready_wake_resumes_the_episode_session_at_the_next_invocation(
    manifest, tmp_path
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    loop = _Loop(app)
    loop.start_episode()
    loop.bind_session(tmp_path / "stage")
    loop.arm_watcher("finished-unclaimed", status="completed")

    loop.record_answers()
    loop.deliver("finished-unclaimed")

    woken = [
        record
        for record in loop.store.agent_tasks(loop.project_id)
        if record.request.get("trigger") == "watcher"
    ]
    assert len(woken) == 1
    task = woken[0]
    assert task.request["control_invocation"] == 2
    assert task.request["control_episode_id"] == loop.episode_id
    assert task.request["session_id"] == "native-session-abc"
    assert task.request["watcher_ids"] == ["finished-unclaimed"]
    assert task.stage_root == str(tmp_path / "stage")
    assert task.stage_host is None
    # A wake is a new task at the next invocation, never task Resume.
    assert task.parent_operation_id is None
    causes = [
        receipt.payload.get("continuation_cause")
        for receipt in loop.store.agent_task_receipts(task.operation_id)
        if receipt.category == "operation_admitted"
    ]
    assert causes == ["watcher_wake"]
    admission = next(
        item
        for item in loop.store.agent_task_receipts(task.operation_id)
        if item.category == "operation_admitted"
    )
    assert admission.payload["parent_operation_id"] is None
    assert loop.store.watcher("finished-unclaimed").notified is True


def test_control_state_exposes_the_operational_block_without_the_session_id(
    manifest, tmp_path
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    loop = _Loop(app)
    loop.start_episode(status="running")
    loop.bind_session(tmp_path / "stage")

    control = loop.control()

    operational = control["operational"]
    assert operational["current_operation_id"] == "loop-root"
    assert operational["current_status"] == "running"
    assert operational["current_phase"] == "agent"
    assert operational["current_invocation"] == 1
    assert operational["chat_id"] == loop.chat_id
    assert operational["current_status_message"] == "Working on the bounded loop."
    assert operational["current_last_activity_at"] is not None
    session = operational["session"]
    assert session == {
        "provider": "codex",
        "model": "gpt-5",
        "reasoning": "medium",
        "run_on": "laptop",
        "execution_host": "",
        "run_truth_scope": ["repo-a"],
        "native_session_bound": True,
        "diagnostic": None,
    }
    assert "native-session-abc" not in json.dumps(control)


def test_stop_on_a_node_that_is_not_an_experiment_is_not_found(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    loop = _Loop(app)

    response = loop.client.post(
        f"/api/projects/{loop.project_id}/experiments/rq%2Flearning-after-shift/stop"
    )

    assert response.status_code == 404


def test_legacy_experiment_watcher_stop_requires_graceful_stop_loop(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    loop = _Loop(app)
    loop.start_episode()
    loop.arm_watcher("still-running")

    response = loop.client.post(
        f"/api/projects/{loop.project_id}/experiments/{NODE_PATH}/watchers/stop"
    )

    assert response.status_code == 409, response.text
    assert "Use Stop loop" in response.json()["detail"]
    assert loop.store.watcher("still-running").status == "active"
    episode = loop.store.experiment_episode(loop.episode_id)
    assert episode is not None
    assert episode.session_bound is False


def test_individual_stop_rejects_experiment_watcher(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    loop = _Loop(app)
    loop.start_episode()
    loop.arm_watcher("loop-watcher")

    response = loop.client.post(f"/api/projects/{loop.project_id}/watchers/loop-watcher/stop")

    assert response.status_code == 409
    assert "Use Stop loop" in response.json()["detail"]
    assert loop.store.watcher("loop-watcher").status == "active"


def test_episode_binding_is_immutable_after_first_success(manifest, tmp_path) -> None:
    loop = _Loop(create_app(str(manifest.path), data_dir=tmp_path / "data"))
    loop.start_episode()
    loop.bind_session(tmp_path / "stage")

    with pytest.raises(ValueError, match="cannot change its native-session binding"):
        loop.store.commit_experiment_episode_turn(
            episode_id=loop.episode_id,
            project_id=loop.project_id,
            control_node_id=EXPERIMENT_ID,
            provider="claude",
            execution_machine="laptop",
            execution_host="",
            native_session_id="different-session",
            stage_host=None,
            stage_root=str(tmp_path / "other-stage"),
            chat_id=loop.chat_id,
            operation_id="loop-root",
            invocation=1,
            graph_result="none",
            watcher_ids=[],
            context_baseline={},
        )


def test_explicit_recovery_atomically_replaces_binding_and_runtime_profile(
    manifest, tmp_path
) -> None:
    loop = _Loop(create_app(str(manifest.path), data_dir=tmp_path / "data"), invocation_ceiling=3)
    loop.start_episode()
    old_stage = tmp_path / "old-stage"
    loop.bind_session(old_stage)
    new_stage = tmp_path / "new-stage"
    new_stage.mkdir()
    request = loop.root_request(invocation=2).model_copy(
        update={
            "provider": "claude",
            "model": "sonnet",
            "reasoning": "high",
            "trigger": "watcher",
            "session_id": None,
        }
    )
    now = loop.store.now()
    loop.arm_watcher("failed-switch-watcher", status="completed")
    failed_request = loop.root_request(invocation=2).model_copy(
        update={
            "trigger": "watcher",
            "session_id": "native-session-abc",
            "watcher_ids": ["failed-switch-watcher"],
        }
    )
    loop.create_watcher_invocation(
        AgentTaskRecord(
            operation_id="failed-wake-for-switch",
            project_id=loop.project_id,
            kind="node_chat",
            status="failed",
            request=failed_request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="Provider limit reached.",
            error="Usage limit exceeded",
            native_session_id="native-session-abc",
            stage_root=str(old_stage),
            dispatch_authority=_task_authority(failed_request),
        ),
        ["failed-switch-watcher"],
    )
    loop.create_recovery_task(
        AgentTaskRecord(
            operation_id="successful-provider-switch",
            project_id=loop.project_id,
            kind="node_chat",
            status="running",
            request=request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="Switching provider.",
            attempt=2,
            parent_operation_id="failed-wake-for-switch",
            native_session_id="new-claude-session",
            stage_root=str(new_stage),
            dispatch_authority=_task_authority(request),
        ),
        continuation_cause="handoff",
    )
    execution = AgentTaskExecution(
        operation_id="successful-provider-switch",
        store=loop.store,
        control=AgentProcessControl(),
        stage_root=str(new_stage),
        continuation="handoff",
    )

    commit_experiment_episode_binding(
        execution,
        request,
        native_session_id="new-claude-session",
        execution_host="",
        stage_host=None,
        stage_root=str(new_stage),
        graph_result="no graph change",
        watcher_ids=["next-observer"],
        context_baseline={"revision": 2},
    )

    episode = loop.store.experiment_episode(loop.episode_id)
    assert episode is not None
    assert episode.provider == "claude"
    assert episode.native_session_id == "new-claude-session"
    assert episode.stage_root == str(new_stage)
    assert episode.last_turn_invocation == 2
    runtime = loop.store.experiment_loop_runtime(loop.project_id, EXPERIMENT_ID)
    assert runtime.episode_id == loop.episode_id
    assert runtime.provider == "claude"
    assert runtime.model == "sonnet"
    assert runtime.reasoning == "high"
    assert runtime.run_on == "laptop"
    receipts = loop.store.agent_task_receipts("successful-provider-switch")
    replacement = next(
        item for item in receipts if item.category == "experiment_episode_binding_replaced"
    )
    assert replacement.payload["previous"]["provider"] == "codex"
    assert replacement.payload["replacement"]["provider"] == "claude"

    with pytest.raises(ValueError, match="pinned identity"):
        loop.store.commit_experiment_episode_turn(
            episode_id=loop.episode_id,
            project_id=loop.project_id,
            control_node_id=EXPERIMENT_ID,
            provider="codex",
            execution_machine="gpu",
            execution_host="gpu.example",
            native_session_id="third-session",
            stage_host="gpu.example",
            stage_root="/tmp/third-stage",
            chat_id=loop.chat_id,
            operation_id="successful-provider-switch",
            invocation=2,
            graph_result="none",
            watcher_ids=[],
            context_baseline={},
            replace_binding=True,
            replacement_provenance={"reason": "test"},
        )


def test_same_episode_roots_cannot_change_provider_configuration(manifest, tmp_path) -> None:
    loop = _Loop(create_app(str(manifest.path), data_dir=tmp_path / "data"))
    loop.start_episode()
    stage = tmp_path / "stage"
    loop.bind_session(stage)
    loop.arm_watcher("changed-config-watcher", status="completed")
    changed = loop.root_request(invocation=2).model_copy(
        update={
            "trigger": "watcher",
            "model": "different-model",
            "session_id": "native-session-abc",
            "watcher_ids": ["changed-config-watcher"],
        }
    )
    now = loop.store.now()

    with pytest.raises(ValueError, match="episode binding: model"):
        loop.store.create_experiment_watcher_invocation(
            AgentTaskRecord(
                operation_id="changed-config",
                project_id=loop.project_id,
                episode_id=loop.episode_id,
                kind="node_chat",
                status="queued",
                request=changed.model_dump(mode="json"),
                created_at=now,
                updated_at=now,
                status_message="Should not start.",
                native_session_id="native-session-abc",
                stage_root=str(stage),
                dispatch_authority=_task_authority(changed),
            ),
            ["changed-config-watcher"],
        )


def test_automatic_wake_requires_session_and_exact_episode_stage(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    loop = _Loop(app)
    loop.start_episode()
    stage = tmp_path / "stage"
    loop.bind_session(stage)
    loop.arm_watcher("ready", status="completed")
    request = loop.root_request(invocation=2).model_copy(
        update={
            "trigger": "watcher",
            "session_id": "wrong-session",
            "watcher_ids": ["ready"],
        }
    )
    now = loop.store.now()
    record = AgentTaskRecord(
        operation_id="wrong-binding",
        project_id=loop.project_id,
        episode_id=loop.episode_id,
        kind="node_chat",
        status="queued",
        request=request.model_dump(mode="json"),
        created_at=now,
        updated_at=now,
        status_message="Should not start.",
        native_session_id="wrong-session",
        stage_root=str(stage),
        dispatch_authority=_task_authority(request),
    )

    with pytest.raises(ValueError, match="episode binding"):
        loop.store.create_experiment_watcher_invocation(record, ["ready"])
    assert loop.store.watcher("ready").notified is False

    no_session = request.model_copy(update={"session_id": None})
    with pytest.raises(ValueError, match="session and exact stage"):
        start_watcher_notification(
            app.state.background_tasks,
            loop.project_id,
            "node_chat",
            no_session,
            ["ready"],
            authorized_by=loop.authorizer,
            episode_stage_root=str(stage),
        )


def test_automatic_compatibility_ignores_model_reasoning_and_packages(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    loop = _Loop(app)
    loop.start_episode()
    loop.bind_session(tmp_path / "stage")
    loop.arm_watcher(
        "old-config",
        status="completed",
        continuation=loop.continuation(
            model="old-model",
            reasoning="low",
            resolved_skill_packages=[
                SkillReference(id="old-package", kind="skill", version="1.0.0")
            ],
        ),
    )
    loop.arm_watcher(
        "new-config",
        status="completed",
        continuation=loop.continuation(
            model="new-model",
            reasoning="high",
            resolved_skill_packages=[
                SkillReference(id="new-package", kind="skill", version="2.0.0")
            ],
        ),
    )

    groups = app.state.background_tasks.store.completed_watcher_groups()
    experiment_group = next(group for group in groups if group[0].watcher_id == "old-config")
    assert {item.watcher_id for item in experiment_group} == {"old-config", "new-config"}

    loop.record_answers()
    loop.deliver("old-config", "new-config")
    task = next(
        task
        for task in loop.store.agent_tasks(loop.project_id)
        if task.request.get("trigger") == "watcher"
    )
    assert task.request["model"] == "gpt-5"
    assert task.request["reasoning"] == "medium"


def test_provider_default_model_stays_pinned_after_settings_change(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    loop = _Loop(app)
    root_request = loop.root_request().model_copy(update={"model": ""})
    loop.start_episode(request=root_request)
    loop.bind_session(tmp_path / "stage")
    loop.arm_watcher(
        "provider-default-result",
        status="completed",
        continuation=loop.continuation(model=""),
    )
    project = loop.client.get(f"/api/projects/{loop.project_id}").json()
    profiles = {
        surface: {key: profile[key] for key in ("provider", "model", "reasoning", "run_on")}
        for surface, profile in project["agent_profiles"].items()
    }
    profiles["node_chat"]["model"] = "new-explicit-model"
    changed = loop.client.put(
        f"/api/projects/{loop.project_id}/settings",
        json={
            "default_run_truth_scope": ["repo-a"],
            "agent_profiles": profiles,
        },
    )
    assert changed.status_code == 200, changed.text

    loop.record_answers()
    loop.deliver("provider-default-result")
    task = next(
        task
        for task in loop.store.agent_tasks(loop.project_id)
        if task.request.get("trigger") == "watcher"
    )
    assert task.request["model"] == ""


@pytest.mark.parametrize("requested_scope", [None, []])
def test_default_truth_scope_is_pinned_before_watcher_completion(
    manifest, tmp_path, requested_scope
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    loop = _Loop(app)
    resolved = _resolved_graph_request(
        loop.service,
        "node_chat",
        loop.root_request().model_copy(update={"run_truth_scope": requested_scope}),
    )
    assert resolved.run_truth_scope == ["repo-a"]
    loop.start_episode(request=resolved)
    loop.bind_session(tmp_path / "scope-stage")
    loop.arm_watcher(
        "default-scope-result",
        status="completed",
        continuation=loop.continuation(run_truth_scope=["repo-a"]),
    )
    loop.record_answers()

    loop.deliver("default-scope-result")

    wake = next(
        task
        for task in loop.store.agent_tasks(loop.project_id)
        if task.request.get("trigger") == "watcher"
    )
    assert wake.request["run_truth_scope"] == ["repo-a"]


def test_old_experiment_graph_repair_is_rejected_after_progress_or_new_episode(
    manifest, tmp_path
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    loop = _Loop(app, invocation_ceiling=3)
    loop.start_episode(status="running")
    rejected = {
        "status": "rejected",
        "validation_messages": ["The graph update needs repair."],
        "repairable": True,
    }
    loop.store.complete_agent_task(
        "loop-root",
        applied_revision=None,
        result={"messages": ["Operational work completed."], "graph_update": rejected},
    )
    stage = tmp_path / "stage"
    loop.bind_session(stage)
    loop.store.checkpoint_agent_task(
        "loop-root",
        native_session_id="native-session-abc",
        stage_root=str(stage),
    )
    loop.arm_watcher("second-invocation-watcher", status="completed")
    second_request = loop.root_request(invocation=2).model_copy(
        update={
            "trigger": "watcher",
            "session_id": "native-session-abc",
            "watcher_ids": ["second-invocation-watcher"],
        }
    )
    now = loop.store.now()
    loop.create_watcher_invocation(
        AgentTaskRecord(
            operation_id="loop-second",
            project_id=loop.project_id,
            kind="node_chat",
            status="succeeded",
            request=second_request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="Second invocation completed.",
            native_session_id="native-session-abc",
            stage_root=str(stage),
            dispatch_authority=_task_authority(second_request),
        ),
        ["second-invocation-watcher"],
    )

    with pytest.raises(ValueError, match="newest Experiment invocation"):
        loop.store.claim_agent_task_graph_repair("loop-root")

    loop.stop()
    loop.episode_id = str(uuid.uuid4())
    loop.chat_id = str(uuid.uuid4())
    loop.start_episode(operation_id="fresh-loop-root")
    with pytest.raises(ValueError, match="newest Experiment episode"):
        loop.store.claim_agent_task_graph_repair("loop-root")


def test_stopped_experiment_episode_cannot_start_an_old_graph_repair(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    loop = _Loop(app)
    loop.start_episode(status="running")
    loop.store.complete_agent_task(
        "loop-root",
        applied_revision=None,
        result={
            "messages": ["Operational work completed."],
            "graph_update": {
                "status": "rejected",
                "validation_messages": ["Repair the Patch."],
                "repairable": True,
            },
        },
    )
    stage = tmp_path / "stopped-repair-stage"
    loop.bind_session(stage)
    loop.store.checkpoint_agent_task(
        "loop-root",
        native_session_id="native-session-abc",
        stage_root=str(stage),
    )

    stopped = loop.stop()

    assert stopped["operational"]["stop_settled"] is True
    with pytest.raises(ValueError, match="stopped Experiment episode"):
        loop.store.claim_agent_task_graph_repair("loop-root")


def test_experiment_graph_repair_admission_rolls_back_claim_child_and_receipt(
    manifest, tmp_path, monkeypatch
) -> None:
    loop = _Loop(create_app(str(manifest.path), data_dir=tmp_path / "data"))
    loop.start_episode(status="running")
    loop.store.complete_agent_task(
        "loop-root",
        applied_revision=None,
        result={
            "messages": ["Operational work completed."],
            "graph_update": {
                "status": "rejected",
                "validation_messages": ["Repair the Patch."],
                "repairable": True,
            },
        },
    )
    stage = tmp_path / "repair-stage"
    loop.bind_session(stage)
    loop.store.checkpoint_agent_task(
        "loop-root",
        native_session_id="native-session-abc",
        stage_root=str(stage),
    )
    episode = loop.store.episode(loop.episode_id)
    assert episode is not None
    repair_request = loop.root_request().model_copy(
        update={"message": None, "session_id": "native-session-abc"}
    )
    now = loop.store.now()

    def child(operation_id: str) -> AgentTaskRecord:
        return AgentTaskRecord(
            operation_id=operation_id,
            project_id=loop.project_id,
            episode_id=loop.episode_id,
            kind="node_chat",
            status="queued",
            request=repair_request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="Waiting to repair the graph update.",
            attempt=2,
            parent_operation_id="loop-root",
            native_session_id="native-session-abc",
            stage_root=str(stage),
            graph_target=episode.graph_target,
            authorized_by=loop.authorizer,
            dispatch_authority=_task_authority(repair_request),
        )

    original_insert = loop.store._insert_agent_task

    def fail_after_child_insert(connection, record, **kwargs) -> None:
        original_insert(connection, record, **kwargs)
        raise RuntimeError("simulated Experiment graph repair insert failure")

    monkeypatch.setattr(loop.store, "_insert_agent_task", fail_after_child_insert)
    with pytest.raises(RuntimeError, match="simulated Experiment graph repair insert failure"):
        loop.store.create_experiment_graph_repair_task(
            "loop-root", child("experiment-repair-failed")
        )

    parent = loop.store.agent_task("loop-root")
    assert parent is not None and parent.result is not None
    assert parent.result["graph_update"]["repairable"] is True
    assert loop.store.agent_task("experiment-repair-failed") is None
    assert [task.operation_id for task in loop.store.episode_tasks(loop.episode_id)] == [
        "loop-root"
    ]

    monkeypatch.setattr(loop.store, "_insert_agent_task", original_insert)
    admitted = loop.store.create_experiment_graph_repair_task(
        "loop-root", child("experiment-repair")
    )

    assert admitted.operation_id == "experiment-repair"
    parent = loop.store.agent_task("loop-root")
    assert parent is not None and parent.result is not None
    assert parent.result["graph_update"]["repairable"] is False
    children = [
        task
        for task in loop.store.episode_tasks(loop.episode_id)
        if task.parent_operation_id == "loop-root"
    ]
    assert [task.operation_id for task in children] == ["experiment-repair"]
    assert loop.store.agent_task_continuation_cause("experiment-repair") == "graph_repair"
    receipt = next(
        item
        for item in loop.store.agent_task_receipts("experiment-repair")
        if item.category == "operation_admitted"
    )
    assert receipt.payload["continuation_cause"] == "graph_repair"
    assert receipt.payload["parent_operation_id"] == "loop-root"
    assert receipt.payload["admission_committed"] is True


@pytest.mark.parametrize(
    ("status", "action"),
    [("paused", "resume"), ("failed", "retry")],
)
def test_graph_repair_recovery_remains_patch_only(manifest, tmp_path, status, action) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    store: AppStore = app.state.background_tasks.store
    project_id = app.state.default_project_id
    assert project_id is not None
    stage = tmp_path / "repair-stage"
    stage.mkdir()
    request = RunRequest(
        provider="codex",
        reasoning="medium",
        run_on="laptop",
        run_truth_scope=["repo-a"],
        chat_scope="project",
        chat_id=str(uuid.uuid4()),
        message="Complete operational work.",
        mode="work",
    )
    now = store.now()
    store.create_agent_task(
        AgentTaskRecord(
            operation_id="rejected-work",
            project_id=project_id,
            kind="project_chat",
            status="succeeded",
            request=request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="Graph update rejected.",
            native_session_id="repair-session",
            stage_root=str(stage),
            result={
                "messages": ["Operational work completed."],
                "graph_update": {
                    "status": "rejected",
                    "validation_messages": ["Repair the Patch."],
                    "repairable": False,
                },
            },
            dispatch_authority=_task_authority(request, kind="project_chat"),
        )
    )
    repair_request = request.model_copy(update={"message": None, "session_id": "repair-session"})
    store.create_agent_task(
        AgentTaskRecord(
            operation_id="repair-attempt",
            project_id=project_id,
            kind="project_chat",
            status=status,
            request=repair_request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message=f"Repair {status}.",
            error="provider connection dropped" if status == "failed" else None,
            attempt=2,
            parent_operation_id="rejected-work",
            native_session_id="repair-session",
            stage_root=str(stage),
            dispatch_authority=_task_authority(repair_request, kind="project_chat"),
        ),
        continuation_cause="graph_repair",
    )
    observed = Event()
    continuations: list[str] = []

    async def stream(_project_id, _kind, _request, execution):
        continuations.append(execution.continuation)
        observed.set()
        yield _sse(AgentEvent(event="done"))

    app.state.background_tasks.stream = stream
    recovered = getattr(app.state.background_tasks, action)(
        "repair-attempt",
        authorized_by=authorized_human(store),
    )

    assert observed.wait(timeout=2)
    assert continuations == ["graph_repair"]
    assert store.agent_task_continuation_cause(recovered.operation_id) == "graph_repair"
    assert recovered.request["message"] is None
    assert recovered.native_session_id == "repair-session"


@pytest.mark.parametrize("task_status", ["failed", "paused"])
def test_watcher_wake_retry_never_falls_back_to_a_fresh_session(
    manifest, tmp_path, task_status
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    loop = _Loop(app, invocation_ceiling=3)
    loop.start_episode()
    stage = tmp_path / "wake-stage"
    loop.bind_session(stage)
    loop.arm_watcher("failed-wake-watcher", status="completed")
    wake_request = loop.root_request(invocation=2).model_copy(
        update={
            "trigger": "watcher",
            "session_id": "native-session-abc",
            "watcher_ids": ["failed-wake-watcher"],
        }
    )
    now = loop.store.now()
    loop.create_watcher_invocation(
        AgentTaskRecord(
            operation_id="failed-wake",
            project_id=loop.project_id,
            kind="node_chat",
            status=task_status,
            request=wake_request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="Automatic wake failed.",
            error=None if task_status == "paused" else "provider connection dropped",
            native_session_id="native-session-abc",
            stage_root=str(stage),
            dispatch_authority=_task_authority(wake_request),
        ),
        ["failed-wake-watcher"],
    )
    candidate = "{}"
    loop.store.record_agent_task_contract(
        "failed-wake",
        "experiment_episode_context_candidate",
        candidate,
        hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
    )
    stage.rmdir()

    stopping = loop.stop()
    assert stopping["operational"]["stop_settled"] is False
    with pytest.raises(ValueError, match="cannot start a fresh provider session"):
        app.state.background_tasks.retry("failed-wake")

    assert not [
        task
        for task in loop.store.agent_tasks(loop.project_id)
        if task.parent_operation_id == "failed-wake"
    ]
    episode = loop.store.experiment_episode(loop.episode_id)
    assert episode is not None
    assert episode.session_diagnostic is not None
    assert "Switch provider" in episode.session_diagnostic
    settled = loop.control()
    assert settled["operational"]["stop_settled"] is True
    assert settled["ready"] is True
    failed = loop.store.agent_task("failed-wake")
    assert failed is not None
    assert failed.can_resume is False
    assert failed.can_retry is False
    assert "experiment_recovery_abandoned" in {
        receipt.category for receipt in loop.store.agent_task_receipts("failed-wake")
    }
    if task_status == "paused":
        assert not loop.store.has_resumable_paused_chat_task(
            loop.project_id,
            "node_chat",
            loop.chat_id,
        )
        loop.record_answers()
        ordinary = loop.client.post(
            f"/api/projects/{loop.project_id}/tasks/node_chat",
            json={
                "chat_id": loop.chat_id,
                "node_id": EXPERIMENT_ID,
                "message": "Discuss the preserved stopped-loop history.",
                "mode": "discuss",
            },
        )
        assert ordinary.status_code == 202, ordinary.text


def test_provider_limit_retry_rechecks_exact_episode_session(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    loop = _Loop(app, invocation_ceiling=3)
    loop.start_episode()
    stage = tmp_path / "wake-stage"
    loop.bind_session(stage)
    loop.arm_watcher("limited-wake-watcher", status="completed")
    wake_request = loop.root_request(invocation=2).model_copy(
        update={
            "trigger": "watcher",
            "session_id": "native-session-abc",
            "watcher_ids": ["limited-wake-watcher"],
        }
    )
    now = loop.store.now()
    loop.create_watcher_invocation(
        AgentTaskRecord(
            operation_id="limited-wake",
            project_id=loop.project_id,
            kind="node_chat",
            status="failed",
            request=wake_request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="Provider limit reached.",
            error="You've hit your session limit",
            native_session_id="native-session-abc",
            stage_root=str(stage),
            dispatch_authority=_task_authority(wake_request),
        ),
        ["limited-wake-watcher"],
    )
    candidate = "{}"
    loop.store.record_agent_task_contract(
        "limited-wake",
        "experiment_episode_context_candidate",
        candidate,
        hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
    )
    control = loop.control()
    assert {
        field: control[field]
        for field in (
            "health",
            "recommendation",
            "run_section",
            "live",
            "can_start",
            "can_stop",
            "stop_pending",
            "task_control",
            "can_switch_provider",
        )
    } == {
        "health": "needs_action",
        "recommendation": "retry",
        "run_section": "actionable",
        "live": True,
        "can_start": False,
        "can_stop": True,
        "stop_pending": False,
        "task_control": "retry",
        "can_switch_provider": True,
    }
    observed = Event()
    captured: dict[str, object] = {}

    async def stream(_project_id, _kind, request, execution):
        captured.update(request=request, continuation=execution.continuation)
        observed.set()
        yield _sse(AgentEvent(event="done"))

    app.state.background_tasks.stream = stream
    retried = app.state.background_tasks.retry(
        "limited-wake",
        authorized_by=loop.authorizer,
    )

    assert observed.wait(timeout=2)
    request = captured["request"]
    assert isinstance(request, RunRequest)
    assert captured["continuation"] == "retry"
    assert request.session_id == "native-session-abc"
    assert request.control_episode_id == loop.episode_id
    assert request.control_invocation == 2
    assert retried.stage_root == str(stage)
    assert retried.parent_operation_id == "limited-wake"


def test_provider_switch_is_provisional_until_successful_episode_handoff(
    manifest, tmp_path
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    loop = _Loop(app, invocation_ceiling=3)
    loop.start_episode()
    old_stage = tmp_path / "old-stage"
    loop.bind_session(old_stage)
    loop.arm_watcher("provider-switch-watcher", status="completed")
    failed_request = loop.root_request(invocation=2).model_copy(
        update={
            "trigger": "watcher",
            "session_id": "native-session-abc",
            "watcher_ids": ["provider-switch-watcher"],
        }
    )
    now = loop.store.now()
    loop.create_watcher_invocation(
        AgentTaskRecord(
            operation_id="failed-wake",
            project_id=loop.project_id,
            kind="node_chat",
            status="failed",
            request=failed_request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="Provider limit reached.",
            error="Quota exceeded",
            native_session_id="native-session-abc",
            stage_root=str(old_stage),
            dispatch_authority=_task_authority(failed_request),
        ),
        ["provider-switch-watcher"],
    )
    candidate = "{}"
    loop.store.record_agent_task_contract(
        "failed-wake",
        "experiment_episode_context_candidate",
        candidate,
        hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
    )
    observed = Event()
    captured: dict[str, object] = {}

    async def fail_before_handoff(_project_id, _kind, request, execution):
        captured.update(request=request, continuation=execution.continuation)
        observed.set()
        yield _sse(AgentEvent(event="error", text="provider unavailable"))

    app.state.background_tasks.stream = fail_before_handoff
    retried = app.state.background_tasks.retry(
        "failed-wake",
        provider="claude",
        model="sonnet",
        reasoning="high",
        authorized_by=loop.authorizer,
    )

    assert observed.wait(timeout=2)
    request = captured["request"]
    assert isinstance(request, RunRequest)
    assert captured["continuation"] == "handoff"
    assert request.provider == "claude"
    assert request.model == "sonnet"
    assert request.reasoning == "high"
    assert request.run_on == "laptop"
    assert request.session_id is None
    assert request.control_episode_id == loop.episode_id
    assert request.control_invocation == 2
    assert retried.parent_operation_id == "failed-wake"
    episode = loop.store.experiment_episode(loop.episode_id)
    assert episode is not None
    assert episode.provider == "codex"
    assert episode.native_session_id == "native-session-abc"
    assert episode.stage_root == str(old_stage)


def test_retry_of_failed_provisional_switch_keeps_its_provider_and_can_commit(
    manifest, tmp_path
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    loop = _Loop(app, invocation_ceiling=3)
    loop.start_episode()
    old_stage = tmp_path / "old-stage"
    loop.bind_session(old_stage)
    candidate = "{}"
    now = loop.store.now()
    loop.arm_watcher("provisional-switch-watcher", status="completed")
    failed_wake_request = loop.root_request(invocation=2).model_copy(
        update={
            "trigger": "watcher",
            "session_id": "native-session-abc",
            "watcher_ids": ["provisional-switch-watcher"],
        }
    )
    loop.create_watcher_invocation(
        AgentTaskRecord(
            operation_id="failed-wake-before-switch",
            project_id=loop.project_id,
            kind="node_chat",
            status="failed",
            request=failed_wake_request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="Provider limit reached.",
            error="Usage limit exceeded",
            native_session_id="native-session-abc",
            stage_root=str(old_stage),
            dispatch_authority=_task_authority(failed_wake_request),
        ),
        ["provisional-switch-watcher"],
    )
    loop.store.record_agent_task_contract(
        "failed-wake-before-switch",
        "experiment_episode_context_candidate",
        candidate,
        hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
    )
    provisional_stage = tmp_path / "provisional-stage"
    provisional_stage.mkdir()
    provisional_request = failed_wake_request.model_copy(
        update={
            "provider": "claude",
            "model": "sonnet",
            "reasoning": "high",
            "session_id": None,
        }
    )
    loop.create_recovery_task(
        AgentTaskRecord(
            operation_id="failed-provisional-switch",
            project_id=loop.project_id,
            kind="node_chat",
            status="failed",
            request=provisional_request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="Provisional provider failed.",
            error="provider temporarily unavailable",
            attempt=2,
            parent_operation_id="failed-wake-before-switch",
            native_session_id="provisional-claude-session",
            stage_root=str(provisional_stage),
            dispatch_authority=_task_authority(provisional_request),
        ),
        continuation_cause="handoff",
    )
    observed = Event()
    captured: dict[str, object] = {}

    async def stream(_project_id, _kind, request, execution):
        captured.update(request=request, continuation=execution.continuation)
        observed.set()
        yield _sse(AgentEvent(event="done"))

    app.state.background_tasks.stream = stream
    retried = app.state.background_tasks.retry(
        "failed-provisional-switch",
        authorized_by=loop.authorizer,
    )

    assert observed.wait(timeout=2)
    retried_request = captured["request"]
    assert isinstance(retried_request, RunRequest)
    assert captured["continuation"] == "retry"
    assert retried_request.provider == "claude"
    assert retried_request.model == "sonnet"
    assert retried_request.reasoning == "high"
    assert retried_request.session_id == "provisional-claude-session"
    assert retried.stage_root == str(provisional_stage)
    episode = loop.store.experiment_episode(loop.episode_id)
    assert episode is not None and episode.provider == "codex"

    commit_experiment_episode_binding(
        AgentTaskExecution(
            operation_id=retried.operation_id,
            store=loop.store,
            control=AgentProcessControl(),
            stage_root=str(provisional_stage),
            continuation="retry",
        ),
        retried_request,
        native_session_id="provisional-claude-session",
        execution_host="",
        stage_host=None,
        stage_root=str(provisional_stage),
        graph_result="no graph change",
        watcher_ids=["next-observer"],
        context_baseline={},
    )

    episode = loop.store.experiment_episode(loop.episode_id)
    assert episode is not None
    assert episode.provider == "claude"
    assert episode.native_session_id == "provisional-claude-session"


@pytest.mark.parametrize(
    ("status", "action"),
    [("paused", "resume"), ("failed", "retry")],
)
def test_legacy_missing_context_candidate_refuses_recovery_before_provider_launch(
    manifest, tmp_path, status, action
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    loop = _Loop(app)
    loop.start_episode(status=status)
    stage = tmp_path / "legacy-stage"
    stage.mkdir()
    loop.store.checkpoint_agent_task(
        "loop-root",
        native_session_id="legacy-session",
        stage_root=str(stage),
    )
    launched = Event()

    async def stream(*_args, **_kwargs):
        launched.set()
        yield _sse(AgentEvent(event="done"))

    app.state.background_tasks.stream = stream

    with pytest.raises(ValueError, match="no retained episode context candidate"):
        getattr(app.state.background_tasks, action)("loop-root")

    assert not launched.is_set()
    assert not [
        task
        for task in loop.store.agent_tasks(loop.project_id)
        if task.parent_operation_id == "loop-root"
    ]
    episode = loop.store.experiment_episode(loop.episode_id)
    assert episode is not None
    assert episode.session_diagnostic is not None
    assert "pre-migration root" in episode.session_diagnostic

    stopped = loop.stop()
    assert stopped["operational"]["stop_settled"] is True
    preserved = loop.store.agent_task("loop-root")
    assert preserved is not None and preserved.status == status
    assert preserved.can_resume is False
    assert preserved.can_retry is False


def test_restart_settles_an_already_stuck_legacy_recovery_and_enables_fresh_run(
    manifest, tmp_path
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    loop = _Loop(app)
    loop.start_episode(status="paused")
    stage = tmp_path / "legacy-stage"
    stage.mkdir()
    loop.store.checkpoint_agent_task(
        "loop-root",
        native_session_id="legacy-session",
        stage_root=str(stage),
    )
    retry_request = loop.root_request().model_copy(update={"session_id": "legacy-session"})
    now = loop.store.now()
    loop.store.request_episode_stop(loop.episode_id)
    with loop.store.connection() as connection:
        # This row predates dispatch-authority admission. Current task creation
        # correctly refuses it; raw SQL keeps restart compatibility covered.
        connection.execute(
            """
            INSERT INTO graph_runs (
                operation_id, project_id, episode_id, kind, status, request_json,
                created_at, updated_at, status_message, error, attempt,
                parent_operation_id, native_session_id, stage_root
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "doomed-retry",
                loop.project_id,
                loop.episode_id,
                "node_chat",
                "failed",
                json.dumps(retry_request.model_dump(mode="json"), separators=(",", ":")),
                now,
                now,
                "The retained episode context candidate was unavailable.",
                "The continued Experiment-loop turn has no retained episode context candidate.",
                2,
                "loop-root",
                "legacy-session",
                str(stage),
            ),
        )

    before = loop.loop_task_ids()
    BackgroundAgentTasks(loop.store, app.state.background_tasks.stream).recover_at_startup()

    episode = loop.store.experiment_episode(loop.episode_id)
    assert episode is not None
    assert episode.stop_settled_at is not None
    assert episode.session_diagnostic is not None
    assert "pre-migration root" in episode.session_diagnostic
    assert loop.loop_task_ids() == before
    root = loop.store.agent_task("loop-root")
    retry = loop.store.agent_task("doomed-retry")
    assert root is not None and root.status == "paused"
    assert retry is not None and retry.status == "failed"
    assert retry.can_retry is False
    assert "experiment_recovery_abandoned" in {
        receipt.category for receipt in loop.store.agent_task_receipts("doomed-retry")
    }

    loop.record_answers()
    response = loop.client.post(
        f"/api/projects/{loop.project_id}/experiments/{NODE_PATH}/run",
        json={"chat_id": str(uuid.uuid4()), "run_truth_scope": ["repo-a"]},
    )
    assert response.status_code == 202, response.text
    assert response.json()["request"]["control_episode_id"] != loop.episode_id


@pytest.mark.parametrize(
    "provider_error",
    [
        "You've hit your session limit",
        "Usage limit exceeded",
        "Quota exceeded",
        "Out of credits",
    ],
)
def test_bound_provider_limit_records_diagnostic_before_direct_stop(
    manifest, tmp_path, provider_error
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    loop = _Loop(app, invocation_ceiling=3)
    loop.start_episode()
    stage = tmp_path / "bound-stage"
    loop.bind_session(stage)
    loop.arm_watcher("bound-limit-watcher", status="completed")
    request = loop.root_request(invocation=2).model_copy(
        update={
            "trigger": "watcher",
            "session_id": "native-session-abc",
            "watcher_ids": ["bound-limit-watcher"],
        }
    )
    now = loop.store.now()
    task = loop.store.create_experiment_watcher_invocation(
        AgentTaskRecord(
            operation_id="limited-wake",
            project_id=loop.project_id,
            episode_id=loop.episode_id,
            kind="node_chat",
            status="queued",
            request=request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="Waiting for the provider.",
            native_session_id="native-session-abc",
            stage_root=str(stage),
            dispatch_authority=_task_authority(request),
        ),
        ["bound-limit-watcher"],
    )
    assert task is not None
    candidate = "{}"
    loop.store.record_agent_task_contract(
        task.operation_id,
        "experiment_episode_context_candidate",
        candidate,
        hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
    )

    async def stream(*_args, **_kwargs):
        yield _sse(AgentEvent(event="error", text=provider_error))

    app.state.background_tasks.stream = stream
    app.state.background_tasks._run(
        task,
        request,
        AgentProcessControl(),
        "watcher_wake",
    )

    failed = loop.store.agent_task(task.operation_id)
    assert failed is not None and failed.status == "failed"
    assert failed.can_retry is True
    episode = loop.store.experiment_episode(loop.episode_id)
    assert episode is not None
    assert episode.stop_requested_at is None
    assert episode.session_diagnostic is not None
    assert "Retry the same provider" in episode.session_diagnostic
    assert "switch provider" in episode.session_diagnostic
    assert not [
        item
        for item in loop.store.agent_tasks(loop.project_id)
        if item.parent_operation_id == task.operation_id
    ]

    stopped = loop.stop()
    assert stopped["operational"]["stop_settled"] is True
    preserved = loop.store.agent_task(task.operation_id)
    assert preserved is not None and preserved.status == "failed"
    assert preserved.can_retry is False
    assert "experiment_recovery_abandoned" in {
        receipt.category for receipt in loop.store.agent_task_receipts(task.operation_id)
    }

    loop.record_answers()
    response = loop.client.post(
        f"/api/projects/{loop.project_id}/experiments/{NODE_PATH}/run",
        json={"chat_id": str(uuid.uuid4()), "run_truth_scope": ["repo-a"]},
    )
    assert response.status_code == 202, response.text
    assert response.json()["request"]["control_episode_id"] != loop.episode_id


def test_unbound_initial_provider_limit_remains_clean_retry_eligible(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    loop = _Loop(app)
    loop.start_episode(status="queued")
    task = loop.store.agent_task("loop-root")
    assert task is not None
    candidate = "{}"
    loop.store.record_agent_task_contract(
        task.operation_id,
        "experiment_episode_context_candidate",
        candidate,
        hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
    )

    async def limited_stream(*_args, **_kwargs):
        yield _sse(AgentEvent(event="session", session_id="unbound-session"))
        yield _sse(AgentEvent(event="error", text="Quota exceeded"))

    app.state.background_tasks.stream = limited_stream
    app.state.background_tasks._run(
        task,
        loop.root_request(),
        AgentProcessControl(),
        "fresh",
    )

    failed = loop.store.agent_task(task.operation_id)
    assert failed is not None and failed.status == "failed"
    assert failed.can_retry is True
    episode = loop.store.experiment_episode(loop.episode_id)
    assert episode is not None
    assert episode.session_bound is False

    retried = Event()

    async def clean_retry_stream(*_args, **_kwargs):
        retried.set()
        yield _sse(AgentEvent(event="answer", text="Clean retry started."))
        yield _sse(AgentEvent(event="done"))

    app.state.background_tasks.stream = clean_retry_stream
    child = app.state.background_tasks.retry(
        task.operation_id,
        authorized_by=loop.authorizer,
    )

    assert retried.wait(timeout=2)
    assert child.parent_operation_id == task.operation_id
    assert child.request["session_id"] is None
    assert child.native_session_id is None
    assert loop.store.agent_task_continuation_cause(child.operation_id) == "handoff"


def test_initial_run_uses_current_node_chat_profile_not_client_overrides(
    manifest, tmp_path
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    loop = _Loop(app)
    loop.record_answers()

    response = loop.client.post(
        f"/api/projects/{loop.project_id}/experiments/{NODE_PATH}/run",
        json={
            "chat_id": str(uuid.uuid4()),
            "run_truth_scope": ["repo-a"],
            "provider": "claude",
            "model": "client-model",
            "reasoning": "high",
        },
    )

    assert response.status_code == 202, response.text
    request = response.json()["request"]
    assert request["provider"] == "codex"
    assert request["model"] == ""
    assert request["reasoning"] == "medium"
    assert request["run_on"] == "laptop"


def test_human_reauthorization_uses_current_node_profile_and_new_chat(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    loop = _Loop(app, invocation_ceiling=1)
    loop.start_episode()
    loop.arm_watcher(
        "other-provider-result",
        status="completed",
        continuation=loop.continuation(
            provider="claude",
            model="sonnet",
            reasoning="high",
        ),
    )
    loop.record_answers()
    new_chat_id = str(uuid.uuid4())
    reconcile = app.state.watcher_poller.on_poll_completed
    assert reconcile is not None
    reconcile()
    exhausted = loop.store.episode(loop.episode_id)
    assert exhausted is not None
    assert exhausted.ending == "exhausted"
    assert exhausted.wrapup_state == "not_started"
    assert exhausted.wrapup_error is None

    response = loop.client.post(
        f"/api/projects/{loop.project_id}/experiments/{NODE_PATH}/run",
        json={
            "chat_id": new_chat_id,
            "run_truth_scope": [],
            "provider": "codex",
        },
    )

    assert response.status_code == 202, response.text
    request = response.json()["request"]
    assert request["trigger"] == "experiment_run"
    assert request["control_invocation"] == 1
    assert request["provider"] == "codex"
    assert request["model"] == ""
    assert request["reasoning"] == "medium"
    assert request["run_on"] == "laptop"
    assert request["run_truth_scope"] == ["repo-a"]
    assert request["chat_id"] == new_chat_id
    assert request["session_id"] is None


def test_experiment_retry_allows_provider_overrides_but_rejects_run_on(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    loop = _Loop(app)
    loop.start_episode(status="failed")
    candidate = "{}"
    loop.store.record_agent_task_contract(
        "loop-root",
        "experiment_episode_context_candidate",
        candidate,
        hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
    )
    seen = loop.record_answers()

    response = loop.client.post(
        f"/api/projects/{loop.project_id}/tasks/loop-root/retry",
        json={"provider": "claude", "model": "sonnet", "reasoning": "high"},
    )

    assert response.status_code == 202, response.text
    assert response.json()["request"]["provider"] == "claude"
    for _ in range(100):
        if seen:
            break
        Event().wait(0.01)
    assert seen and seen[0].run_on == "laptop"

    pinned = loop.client.post(
        f"/api/projects/{loop.project_id}/tasks/loop-root/retry",
        json={"run_on": "gpu"},
    )
    assert pinned.status_code == 409
    assert "pinned execution machine" in pinned.json()["detail"]


def test_stop_preserves_compatible_stopped_watcher_history_across_episodes(
    manifest, tmp_path
) -> None:
    loop = _Loop(create_app(str(manifest.path), data_dir=tmp_path / "data"))
    loop.start_episode(operation_id="old-root")
    loop.arm_watcher("old-watcher", origin_operation_id="old-root")
    loop.stop()
    loop.episode_id = str(uuid.uuid4())
    loop.chat_id = str(uuid.uuid4())
    loop.start_episode(operation_id="current-root")
    loop.arm_watcher("current-watcher", origin_operation_id="current-root")

    loop.stop()

    assert loop.store.watcher("current-watcher").status == "stopped"
    assert loop.store.watcher("old-watcher").status == "stopped"


def test_reconciling_stopped_episode_does_not_retire_new_episode_watcher(
    manifest, tmp_path
) -> None:
    loop = _Loop(create_app(str(manifest.path), data_dir=tmp_path / "data"))
    loop.start_episode(operation_id="old-root")
    loop.arm_watcher("old-watcher", origin_operation_id="old-root")
    old_episode_id = loop.episode_id
    loop.stop()

    loop.episode_id = str(uuid.uuid4())
    loop.chat_id = str(uuid.uuid4())
    loop.start_episode(operation_id="current-root")
    loop.arm_watcher("current-watcher", origin_operation_id="current-root")

    # Reading or otherwise reconciling the already-stopped prior episode must
    # never sweep a compatible watcher that belongs to the replacement episode.
    loop.store.settle_experiment_loop_stop(
        loop.project_id,
        EXPERIMENT_ID,
        episode_id=old_episode_id,
        graph_target=GraphTargetRef(),
    )

    current = loop.store.watcher("current-watcher")
    assert current is not None
    assert current.status == "active"
    assert current.notified is False


def test_stop_fences_current_turn_and_preserves_compatible_prior_watcher(
    manifest, tmp_path
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    loop = _Loop(app)
    loop.start_episode(operation_id="old-root")
    old_episode = loop.episode_id
    loop.arm_watcher("adopted-watcher", origin_operation_id="old-root")
    loop.stop()
    loop.episode_id = str(uuid.uuid4())
    loop.chat_id = str(uuid.uuid4())
    loop.start_episode(status="running", operation_id="current-root")

    control = loop.stop()

    assert control["operational"]["stop_settled"] is False
    assert loop.store.watcher("adopted-watcher").status == "stopped"
    assert "adopted-watcher" not in {record.watcher_id for record in loop.store.pollable_watchers()}
    loop.store.complete_agent_task("current-root", applied_revision=None, result={})
    assert loop.store.settle_ready_experiment_loop_stops() == 1
    adopted = loop.store.watcher("adopted-watcher")
    assert adopted is not None and adopted.status == "stopped" and adopted.notified is True
    assert loop.store.experiment_watcher_compatible_with_episode(
        "adopted-watcher",
        loop.episode_id,
    )
    assert adopted.continuation.control_episode_id == old_episode

    loop.record_answers()
    response = loop.client.post(
        f"/api/projects/{loop.project_id}/experiments/{NODE_PATH}/run",
        json={"chat_id": str(uuid.uuid4()), "run_truth_scope": ["repo-a"]},
    )
    assert response.status_code == 202, response.text
    assert response.json()["request"]["watcher_ids"] == []


def test_final_handoff_is_born_stopped_when_stop_wins_transaction(manifest, tmp_path) -> None:
    loop = _Loop(create_app(str(manifest.path), data_dir=tmp_path / "data"))
    loop.start_episode(status="running")
    loop.store.request_experiment_loop_stop(loop.project_id, EXPERIMENT_ID)
    now = loop.store.now()
    desired = WatcherRecord(
        watcher_id="final-handoff",
        project_id=loop.project_id,
        origin_operation_id="loop-root",
        origin_task_kind="node_chat",
        chat_id=loop.chat_id,
        node_id=EXPERIMENT_ID,
        execution_host="",
        check_command="true",
        log_path="/tmp/final-handoff.log",
        cwd="/tmp",
        continuation=loop.continuation(),
        created_at=now,
    )

    stored = loop.store.persist_experiment_watchers_idempotently(
        [desired],
        binding=WatcherBinding(
            project_id=loop.project_id,
            origin_operation_id="loop-root",
            origin_task_kind="node_chat",
            chat_id=loop.chat_id,
            node_id=EXPERIMENT_ID,
            execution_host="",
            continuation=loop.continuation(),
        ),
    )

    assert stored[0].status == "stopped"
    assert stored[0].notified is True


def test_a_turn_failing_before_its_session_ends_the_episode_without_a_report(
    manifest, tmp_path
) -> None:
    """The reported deadlock: a launch failure looked like a report error.

    The turn dies before it binds a provider session, so nothing can resume it and
    nothing can report on it. The episode must terminalize with its own reason and
    leave the Experiment free to start again.
    """

    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    loop = _Loop(app, invocation_ceiling=5)
    loop.start_episode(status="queued")
    # The real failure: write-scope resolution rejected the repository long before
    # any provider session existed.
    loop.store.fail_agent_task(
        "loop-root",
        "repository 'vista' does not match its project execution host",
    )
    root = loop.store.agent_task("loop-root")
    assert root is not None and not root.native_session_id
    reconciler = EpisodeReconciler(
        loop.store,
        app.state.background_tasks,
        logger=logging.getLogger(__name__),
    )

    reconciler.reconcile_experiment_episode(
        loop.episode_id,
        source="test",
        operation_id="loop-root",
    )

    episode = loop.store.episode(loop.episode_id)
    assert episode is not None
    assert episode.status == "failed"
    assert episode.ending == "failed"
    # No wrap-up ran, so there is no report error competing with the real reason.
    assert episode.wrapup_state == "not_started"
    assert episode.wrapup_error is None
    assert loop.store.episode_wrapup(loop.episode_id) is None
    assert loop.store.episode_report(loop.episode_id) is None
    diagnostic = episode.ending_diagnostic or ""
    assert "before it started its agent session" in diagnostic
    assert "repository 'vista' does not match its project execution host" in diagnostic
    # The old text blamed a pre-migration lineage and sent the human to a control
    # the ending fence had already retired.
    assert "pre-migration" not in diagnostic
    assert "Stop loop" not in diagnostic

    # The Experiment is restartable: a terminal episode is not a live one.
    response = loop.client.post(
        f"/api/projects/{loop.project_id}/experiments/{EXPERIMENT_ID.replace('/', '%2F')}/run",
        json={"chat_id": str(uuid.uuid4())},
    )
    assert response.status_code == 202, response.text
    assert response.json()["episode_id"] != loop.episode_id
