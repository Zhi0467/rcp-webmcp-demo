from __future__ import annotations

import asyncio
import hashlib
import threading
import time
import uuid
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from pathlib import Path

import pytest

from rcp.agents import AgentEvent
from rcp.background import AgentTaskExecution, BackgroundAgentTasks
from rcp.config import Manifest, write_agent_settings
from rcp.core.models import Experiment, Patch
from rcp.core.transition_models import GraphHeadRef
from rcp.history import HistoryManager
from rcp.paper import PaperService
from rcp.runs.auto_research import AutoResearchStartRequest
from rcp.runs.auto_research_admission import (
    start_auto_research,
    start_auto_research_child_experiment,
)
from rcp.runs.auto_research_experiments import (
    AutoResearchExperimentCoordinator,
    AutoResearchExperimentLimitInvalid,
)
from rcp.runs.experiment_admission import experiment_start_message, fresh_experiment_run_request
from rcp.service import ProjectService, RunRequest
from rcp.storage import (
    AppStore,
    AutoResearchChildAdmissionRecord,
    AutoResearchChildExperimentRecord,
    AutoResearchExperimentAllowanceReached,
    ProjectRecord,
)

from .helpers import fabricated_authorizer, wait_for_task

EXPERIMENT_ID = "exp/orchestrated-loop"
PROJECT_ID = "project"
CHILD_EXPLICIT = "00000000-0000-4000-8000-000000000401"
CHILD_FALLBACK = "00000000-0000-4000-8000-000000000402"
CHILD_OVER_LIMIT = "00000000-0000-4000-8000-000000000403"
CHILD_REPLAY = "00000000-0000-4000-8000-000000000404"
HUMAN_PREDECESSOR = "00000000-0000-4000-8000-000000000405"
REPLACEMENT_TO_CANCEL = "00000000-0000-4000-8000-000000000406"
READINESS_PREDECESSOR = "00000000-0000-4000-8000-000000000407"
READINESS_REPLACEMENT = "00000000-0000-4000-8000-000000000408"
RESTART_PREDECESSOR = "00000000-0000-4000-8000-000000000409"
RESTART_REPLACEMENT = "00000000-0000-4000-8000-000000000410"
IDLE_PREDECESSOR = "00000000-0000-4000-8000-000000000411"
IDLE_REPLACEMENT = "00000000-0000-4000-8000-000000000412"
STOP_RECOVERY_PREDECESSOR = "00000000-0000-4000-8000-000000000413"
STOP_RECOVERY_REPLACEMENT = "00000000-0000-4000-8000-000000000414"
EXECUTION_PROFILES = (
    "seed",
    "refresh",
    "node_chat",
    "project_chat",
    "paper_coach",
    "orchestrator",
)


def _sse(event: AgentEvent) -> str:
    return f"data: {event.model_dump_json()}\n\n"


def _experiment_patch() -> Patch:
    return Patch(
        kind="seed",
        author="agent",
        summary="Added the orchestrated Experiment fixture.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": EXPERIMENT_ID,
                        "type": "experiment",
                        "title": "Orchestrated loop",
                        "objective": "Exercise child Experiment admission.",
                        "completion_criteria": ["The bounded comparison finishes."],
                        "invocation_ceiling": 4,
                    }
                ],
            }
        ],
    )


def _blocking_patch() -> Patch:
    return Patch(
        kind="refresh",
        author="agent",
        summary="Recorded a new Experiment blocker.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": "blk/new-readiness-failure",
                        "type": "blocker",
                        "title": "New readiness failure",
                        "description": "The replacement cannot start until this is resolved.",
                        "status": "open",
                    }
                ],
            },
            {
                "op": "create_edges",
                "edges": [
                    {
                        "source": EXPERIMENT_ID,
                        "target": "blk/new-readiness-failure",
                        "relation": "blocked_by",
                    }
                ],
            },
        ],
    )


def _service(manifest: Manifest, tmp_path: Path) -> ProjectService:
    history = HistoryManager(manifest)
    history.append(_experiment_patch())
    paper_store = AppStore(tmp_path / "paper.sqlite3")
    return ProjectService(
        manifest,
        history,
        PaperService(manifest, paper_store, history.workspace, project_id=PROJECT_ID),
        data_dir=tmp_path / "service-data",
        project_id=PROJECT_ID,
    )


def _auto_start(*, ceiling: int) -> AutoResearchStartRequest:
    return AutoResearchStartRequest(
        invocation_ceiling=ceiling,
        provider="codex",
        model="",
        reasoning="medium",
        run_on="laptop",
        run_truth_scope=["repo-a"],
    )


def _admit(
    store: AppStore,
    *,
    parent_episode_id: str,
    child_episode_id: str,
    admission_id: str,
) -> None:
    parent = store.episode(parent_episode_id)
    assert parent is not None
    now = store.now()
    store.record_auto_research_child_admission(
        AutoResearchChildAdmissionRecord(
            admission_id=admission_id,
            episode_id=parent_episode_id,
            project_id=parent.project_id,
            child_kind="experiment",
            child_id=child_episode_id,
            state="accepted",
            created_at=now,
            updated_at=now,
        )
    )


def _spend_child_experiment_allowance(
    store: AppStore,
    background: BackgroundAgentTasks,
    *,
    parent_episode_id: str,
    parent_operation_id: str,
    child_episode_id: str,
    node_id: str,
) -> None:
    now = store.now()
    route = AutoResearchChildExperimentRecord(
        child_episode_id=child_episode_id,
        auto_research_episode_id=parent_episode_id,
        project_id=PROJECT_ID,
        control_node_id=node_id,
        state="running",
        request={"goal": None, "invocation_limit": None},
        parent_operation_id=parent_operation_id,
        created_at=now,
        updated_at=now,
    )
    request = RunRequest(
        provider="codex",
        model="",
        reasoning="medium",
        run_on="laptop",
        run_truth_scope=["repo-a"],
        chat_id=child_episode_id,
        chat_scope="node",
        node_id=node_id,
        message=experiment_start_message(None, node_id),
        mode="work",
        trigger="orchestrator",
        patch_kind="experiment_loop",
        control_node_id=node_id,
        control_revision=1,
        control_episode_id=child_episode_id,
        control_invocation=1,
        control_invocation_ceiling=1,
        control_decision_bundle=[],
        control_completion_criteria=["The bounded allowance probe is analyzed."],
    )
    spent = start_auto_research_child_experiment(background, route, request)
    wait_for_task(store, spent.operation_id, expect="succeeded")


def _setup(
    manifest: Manifest,
    tmp_path: Path,
    *,
    ceiling: int = 2,
    child_stream=None,
) -> tuple[
    ProjectService,
    AppStore,
    BackgroundAgentTasks,
    AutoResearchExperimentCoordinator,
    str,
    str,
]:
    service = _service(manifest, tmp_path)
    store = AppStore(tmp_path / "app.sqlite3")
    store.upsert_project(
        ProjectRecord(
            project_id=PROJECT_ID,
            locator=str(manifest.path),
            name="Project",
            state_location=str(manifest.research_dir),
            state_remote=False,
            added_at=store.now(),
        )
    )

    async def stream(
        project_id: str,
        kind: str,
        request: object,
        execution: AgentTaskExecution,
    ) -> AsyncIterator[str]:
        if kind == "auto_research":
            yield _sse(AgentEvent(event="done"))
            return
        if child_stream is None:
            yield _sse(AgentEvent(event="done"))
            return
        async for frame in child_stream(project_id, kind, request, execution):
            yield frame

    background = BackgroundAgentTasks(store, stream)
    parent, root = start_auto_research(
        background,
        PROJECT_ID,
        _auto_start(ceiling=ceiling),
        authorized_by=fabricated_authorizer("Researcher"),
        graph_base_head=GraphHeadRef(revision=0),
        ensure_graph_target=lambda _episode: None,
        episode_id=f"parent-{tmp_path.name}",
        operation_id=f"root-{tmp_path.name}",
    )
    wait_for_task(store, root.operation_id, expect="succeeded")
    assert parent.graph_target.kind == "branch"
    assert parent.graph_target.branch_id == parent.episode_id
    assert parent.graph_base_head == GraphHeadRef(revision=0)
    assert root.graph_target == parent.graph_target
    coordinator = AutoResearchExperimentCoordinator(
        store,
        background,
        project_service=lambda project_id, _episode_id: (
            service if project_id == PROJECT_ID else None
        ),  # type: ignore[return-value]
        operation_lock=lambda _project_id: nullcontext(),
    )
    return service, store, background, coordinator, parent.episode_id, root.operation_id


@pytest.mark.parametrize(
    ("goal", "invocation_limit", "expected_message", "expected_ceiling"),
    [
        (
            "Measure whether the repaired runtime survives a concise probe.",
            6,
            "Measure whether the repaired runtime survives a concise probe.",
            6,
        ),
        (None, None, f"Begin a bounded Experiment-loop episode for {EXPERIMENT_ID}.", 4),
    ],
)
def test_kickoff_preserves_optional_goal_and_uses_current_node_work_profile(
    manifest: Manifest,
    tmp_path: Path,
    goal: str | None,
    invocation_limit: int | None,
    expected_message: str,
    expected_ceiling: int,
) -> None:
    profiles = {
        name: manifest.agent_profile(name).model_copy(deep=True) for name in EXECUTION_PROFILES
    }
    profiles["node_chat"] = profiles["node_chat"].model_copy(
        update={
            "provider": "claude",
            "runtime": "stream-json",
            "model": "current-node-work",
            "reasoning": "low",
        }
    )
    profiles["orchestrator"] = profiles["orchestrator"].model_copy(
        update={"provider": "codex", "model": "orchestrator-only", "reasoning": "high"}
    )
    manifest = write_agent_settings(
        manifest,
        list(manifest.agent.default_run_truth_scope),
        profiles,
    )
    _, store, _, coordinator, parent_id, root_id = _setup(
        manifest,
        tmp_path,
        ceiling=2,
    )
    child_id = CHILD_EXPLICIT if goal is not None else CHILD_FALLBACK
    admission_id = f"admission-{child_id}"
    _admit(
        store,
        parent_episode_id=parent_id,
        child_episode_id=child_id,
        admission_id=admission_id,
    )

    action = coordinator.kick_off(
        auto_research_episode_id=parent_id,
        parent_operation_id=root_id,
        child_episode_id=child_id,
        node_id=EXPERIMENT_ID,
        goal=goal,
        goal_sha256=hashlib.sha256(goal.encode()).hexdigest() if goal is not None else None,
        invocation_limit=invocation_limit,
        admission_id=admission_id,
    )

    assert action.disposition == "created"
    task = store.agent_task(action.operation_id or "")
    assert task is not None
    child = store.episode(child_id)
    assert child is not None
    assert child.graph_target == task.graph_target
    assert child.graph_target.kind == "branch"
    assert child.graph_target.branch_id == parent_id
    assert child.graph_base_head == GraphHeadRef(revision=0)
    assert task.request["message"] == expected_message
    assert task.request["control_invocation_ceiling"] == expected_ceiling
    assert (task.request["provider"], task.request["model"], task.request["reasoning"]) == (
        "claude",
        "current-node-work",
        "low",
    )
    assert task.request["model"] != "orchestrator-only"
    assert task.request["trigger"] == "orchestrator"
    assert action.allowance.model_dump() == {"total": 10, "used": 1, "remaining": 9}


def test_explicit_invocation_limit_over_total_allowance_is_pre_admission(
    manifest: Manifest,
    tmp_path: Path,
) -> None:
    _, store, _, coordinator, parent_id, root_id = _setup(manifest, tmp_path, ceiling=1)
    _admit(
        store,
        parent_episode_id=parent_id,
        child_episode_id=CHILD_OVER_LIMIT,
        admission_id="admission-over-limit",
    )

    with pytest.raises(AutoResearchExperimentLimitInvalid) as caught:
        coordinator.kick_off(
            auto_research_episode_id=parent_id,
            parent_operation_id=root_id,
            child_episode_id=CHILD_OVER_LIMIT,
            node_id=EXPERIMENT_ID,
            goal=None,
            goal_sha256=None,
            invocation_limit=6,
            admission_id="admission-over-limit",
        )

    assert str(caught.value) == (
        "The requested Experiment invocation limit exceeds the Auto-research allowance of 5; "
        "lower --invocation-limit to 5 or less."
    )
    assert caught.value.allowance.model_dump() == {"total": 5, "used": 0, "remaining": 5}
    assert store.auto_research_child_experiment(CHILD_OVER_LIMIT) is None
    admission = store.auto_research_child_admission("admission-over-limit")
    assert admission is not None and admission.state == "accepted"


def test_exhausted_allowance_does_not_reserve_or_stop_an_active_predecessor(
    manifest: Manifest,
    tmp_path: Path,
) -> None:
    predecessor_started = threading.Event()
    release_predecessor = threading.Event()

    async def child_stream(_project_id, _kind, request, _execution):
        if request.control_episode_id == HUMAN_PREDECESSOR:
            predecessor_started.set()
            while not release_predecessor.is_set():
                await asyncio.sleep(0.01)
        yield _sse(AgentEvent(event="done"))

    service, store, background, coordinator, parent_id, root_id = _setup(
        manifest,
        tmp_path,
        ceiling=1,
        child_stream=child_stream,
    )
    parent = store.episode(parent_id)
    assert parent is not None and parent.authorized_by is not None

    for index in range(5):
        _spend_child_experiment_allowance(
            store,
            background,
            parent_episode_id=parent_id,
            parent_operation_id=root_id,
            child_episode_id=f"00000000-0000-4000-8000-00000000042{index}",
            node_id=f"exp/allowance-{index}",
        )

    exhausted = store.auto_research_experiment_allowance(parent_id)
    assert exhausted.model_dump() == {"total": 5, "used": 5, "remaining": 0}
    predecessor = background.start(
        PROJECT_ID,
        "node_chat",
        _human_experiment_request(service, HUMAN_PREDECESSOR),
        authorized_by=parent.authorized_by,
    )
    assert predecessor_started.wait(timeout=2)
    child_id = "00000000-0000-4000-8000-000000000429"
    admission_id = "admission-exhausted-with-predecessor"
    _admit(
        store,
        parent_episode_id=parent_id,
        child_episode_id=child_id,
        admission_id=admission_id,
    )

    with pytest.raises(AutoResearchExperimentAllowanceReached) as caught:
        coordinator.kick_off(
            auto_research_episode_id=parent_id,
            parent_operation_id=root_id,
            child_episode_id=child_id,
            node_id=EXPERIMENT_ID,
            goal=None,
            goal_sha256=None,
            invocation_limit=None,
            admission_id=admission_id,
        )

    assert caught.value.allowance == exhausted
    assert store.auto_research_child_experiment(child_id) is None
    predecessor_episode = store.episode(HUMAN_PREDECESSOR)
    assert predecessor_episode is not None
    assert predecessor_episode.stop_requested_at is None
    admission = store.auto_research_child_admission(admission_id)
    assert admission is not None and admission.state == "accepted"

    release_predecessor.set()
    wait_for_task(store, predecessor.operation_id, expect="succeeded")


def test_allowance_is_rechecked_after_waiting_for_experiment_operation_lock(
    manifest: Manifest,
    tmp_path: Path,
) -> None:
    predecessor_started = threading.Event()
    release_predecessor = threading.Event()

    async def child_stream(_project_id, _kind, request, _execution):
        if request.control_episode_id == HUMAN_PREDECESSOR:
            predecessor_started.set()
            while not release_predecessor.is_set():
                await asyncio.sleep(0.01)
        yield _sse(AgentEvent(event="done"))

    service, store, background, _, parent_id, root_id = _setup(
        manifest,
        tmp_path,
        ceiling=1,
        child_stream=child_stream,
    )
    parent = store.episode(parent_id)
    assert parent is not None and parent.authorized_by is not None
    for index in range(4):
        _spend_child_experiment_allowance(
            store,
            background,
            parent_episode_id=parent_id,
            parent_operation_id=root_id,
            child_episode_id=f"00000000-0000-4000-8000-00000000043{index}",
            node_id=f"exp/race-allowance-{index}",
        )
    assert store.auto_research_experiment_allowance(parent_id).remaining == 1

    predecessor = background.start(
        PROJECT_ID,
        "node_chat",
        _human_experiment_request(service, HUMAN_PREDECESSOR),
        authorized_by=parent.authorized_by,
    )
    assert predecessor_started.wait(timeout=2)
    replacement_id = "00000000-0000-4000-8000-000000000435"
    admission_id = "admission-exhausted-while-waiting"
    _admit(
        store,
        parent_episode_id=parent_id,
        child_episode_id=replacement_id,
        admission_id=admission_id,
    )

    operation_gate = threading.Lock()
    operation_gate.acquire()
    waiting_for_lock = threading.Event()

    @contextmanager
    def locked_operation():
        waiting_for_lock.set()
        with operation_gate:
            yield

    coordinator = AutoResearchExperimentCoordinator(
        store,
        background,
        project_service=lambda project_id, _episode_id: (
            service if project_id == PROJECT_ID else None
        ),  # type: ignore[return-value]
        operation_lock=lambda _project_id: locked_operation(),
    )
    arguments = {
        "auto_research_episode_id": parent_id,
        "parent_operation_id": root_id,
        "child_episode_id": replacement_id,
        "node_id": EXPERIMENT_ID,
        "goal": None,
        "goal_sha256": None,
        "invocation_limit": None,
        "admission_id": admission_id,
    }

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(coordinator.kick_off, **arguments)
            assert waiting_for_lock.wait(timeout=2)
            _spend_child_experiment_allowance(
                store,
                background,
                parent_episode_id=parent_id,
                parent_operation_id=root_id,
                child_episode_id="00000000-0000-4000-8000-000000000434",
                node_id="exp/race-allowance-final",
            )
            operation_gate.release()
            with pytest.raises(AutoResearchExperimentAllowanceReached) as caught:
                future.result(timeout=2)

        assert caught.value.allowance.model_dump() == {
            "total": 5,
            "used": 5,
            "remaining": 0,
        }
        assert store.auto_research_child_experiment(replacement_id) is None
        predecessor_episode = store.episode(HUMAN_PREDECESSOR)
        assert predecessor_episode is not None
        assert predecessor_episode.stop_requested_at is None
    finally:
        if operation_gate.locked():
            operation_gate.release()
        release_predecessor.set()
        wait_for_task(store, predecessor.operation_id, expect="succeeded")


def test_kickoff_replay_is_deterministic_and_exact_resume_does_not_respend_e(
    manifest: Manifest,
    tmp_path: Path,
) -> None:
    stage = tmp_path / "experiment-stage"
    stage.mkdir()
    resumed_requests: list[RunRequest] = []

    async def child_stream(_project_id, _kind, request, execution):
        assert isinstance(request, RunRequest)
        if execution.continuation == "fresh":
            candidate = "{}"
            execution.store.record_agent_task_contract(
                execution.operation_id,
                "experiment_episode_context_candidate",
                candidate,
                hashlib.sha256(candidate.encode()).hexdigest(),
            )
            execution.checkpoint_stage("", str(stage))
            yield _sse(AgentEvent(event="session", session_id="child-experiment-session"))
            yield _sse(AgentEvent(event="error", text="Transient network failure."))
            return
        resumed_requests.append(request)
        yield _sse(AgentEvent(event="done"))

    _, store, _, coordinator, parent_id, root_id = _setup(
        manifest,
        tmp_path,
        child_stream=child_stream,
    )
    goal = "Verify the bounded runtime repair."
    digest = hashlib.sha256(goal.encode()).hexdigest()
    child_id = CHILD_REPLAY
    admission_id = "admission-replay"
    _admit(
        store,
        parent_episode_id=parent_id,
        child_episode_id=child_id,
        admission_id=admission_id,
    )
    parameters = {
        "auto_research_episode_id": parent_id,
        "parent_operation_id": root_id,
        "child_episode_id": child_id,
        "node_id": EXPERIMENT_ID,
        "goal": goal,
        "goal_sha256": digest,
        "invocation_limit": None,
        "admission_id": admission_id,
    }

    created = coordinator.kick_off(**parameters)
    assert created.operation_id is not None
    failed = wait_for_task(store, created.operation_id, expect="failed")
    replayed = coordinator.kick_off(**parameters)

    assert replayed.disposition == "existing"
    assert replayed.operation_id == created.operation_id
    assert store.auto_research_experiment_allowance(parent_id).used == 1

    resume_operation_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rcp:auto_research:{parent_id}:episode:resume-child-replay",
        )
    )
    resumed = coordinator.resume(
        parent_id,
        child_id,
        operation_id=resume_operation_id,
    )
    assert resumed.disposition == "resumed"
    assert resumed.operation_id is not None
    recovered = wait_for_task(store, resumed.operation_id, expect="succeeded")
    assert recovered.parent_operation_id == failed.operation_id
    assert recovered.native_session_id == "child-experiment-session"
    assert resumed_requests[0].session_id == "child-experiment-session"
    assert store.auto_research_experiment_allowance(parent_id).used == 1

    # A same-key retry after the recovery row committed returns that exact row;
    # it neither creates another attempt nor spends another E unit.
    replayed_resume = coordinator.resume(
        parent_id,
        child_id,
        operation_id=resume_operation_id,
    )
    assert replayed_resume.operation_id == resume_operation_id
    assert len(resumed_requests) == 1
    assert store.auto_research_experiment_allowance(parent_id).used == 1
    assert [
        task.operation_id
        for task in store.episode_tasks(child_id)
        if task.parent_operation_id == failed.operation_id
    ] == [resume_operation_id]

    # A crash can happen after the monotonic episode Stop intent commits but
    # before the command exit is recorded. Reissuing Stop settles that same
    # episode and remains safe on another same-key replay.
    store.request_episode_stop(child_id)
    stopped = coordinator.stop(parent_id, child_id)
    replayed_stop = coordinator.stop(parent_id, child_id)
    assert stopped.disposition in {"stopping", "stopped"}
    assert replayed_stop.disposition in {"stopping", "stopped"}
    child_episode = store.episode(child_id)
    assert child_episode is not None and child_episode.stop_requested_at is not None


def test_experiment_resume_recovers_a_row_committed_before_process_spawn(
    manifest: Manifest,
    tmp_path: Path,
    monkeypatch,
) -> None:
    stage = tmp_path / "experiment-stage"
    stage.mkdir()
    resume_executions = 0

    async def child_stream(_project_id, _kind, _request, execution):
        nonlocal resume_executions
        if execution.continuation == "fresh":
            candidate = "{}"
            execution.store.record_agent_task_contract(
                execution.operation_id,
                "experiment_episode_context_candidate",
                candidate,
                hashlib.sha256(candidate.encode()).hexdigest(),
            )
            execution.checkpoint_stage("", str(stage))
            yield _sse(AgentEvent(event="session", session_id="child-experiment-session"))
            yield _sse(AgentEvent(event="error", text="Transient network failure."))
            return
        resume_executions += 1
        yield _sse(AgentEvent(event="done"))

    _, store, background, coordinator, parent_id, root_id = _setup(
        manifest,
        tmp_path,
        child_stream=child_stream,
    )
    child_id = CHILD_REPLAY
    admission_id = "admission-resume-crash"
    _admit(
        store,
        parent_episode_id=parent_id,
        child_episode_id=child_id,
        admission_id=admission_id,
    )
    created = coordinator.kick_off(
        auto_research_episode_id=parent_id,
        parent_operation_id=root_id,
        child_episode_id=child_id,
        node_id=EXPERIMENT_ID,
        goal=None,
        goal_sha256=None,
        invocation_limit=None,
        admission_id=admission_id,
    )
    assert created.operation_id is not None
    failed = wait_for_task(store, created.operation_id, expect="failed")
    resume_operation_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rcp:auto_research:{parent_id}:episode:resume-before-spawn",
        )
    )
    real_spawn_record = background._spawn_record

    def crash_before_spawn(*_args, **_kwargs):
        raise RuntimeError("simulated crash after Experiment recovery commit")

    monkeypatch.setattr(background, "_spawn_record", crash_before_spawn)
    with pytest.raises(RuntimeError, match="after Experiment recovery commit"):
        coordinator.resume(
            parent_id,
            child_id,
            operation_id=resume_operation_id,
        )

    committed = store.agent_task(resume_operation_id)
    assert committed is not None and committed.status == "queued"
    assert committed.parent_operation_id == failed.operation_id
    assert store.agent_task_continuation_cause(resume_operation_id) == "resume"
    assert store.auto_research_experiment_allowance(parent_id).used == 1
    monkeypatch.setattr(background, "_spawn_record", real_spawn_record)

    replayed = coordinator.resume(
        parent_id,
        child_id,
        operation_id=resume_operation_id,
    )
    assert replayed.operation_id == resume_operation_id
    wait_for_task(store, resume_operation_id, expect="succeeded")
    replayed_again = coordinator.resume(
        parent_id,
        child_id,
        operation_id=resume_operation_id,
    )
    assert replayed_again.operation_id == resume_operation_id
    assert resume_executions == 1
    assert store.auto_research_experiment_allowance(parent_id).used == 1
    assert [
        task.operation_id
        for task in store.episode_tasks(child_id)
        if task.parent_operation_id == failed.operation_id
    ] == [resume_operation_id]


def test_experiment_kickoff_replay_dispatches_invocation_one_committed_before_launch(
    manifest: Manifest,
    tmp_path: Path,
    monkeypatch,
) -> None:
    fresh_executions = 0

    async def child_stream(_project_id, _kind, _request, execution):
        nonlocal fresh_executions
        assert execution.continuation == "fresh"
        fresh_executions += 1
        yield _sse(AgentEvent(event="done"))

    _, store, background, coordinator, parent_id, root_id = _setup(
        manifest,
        tmp_path,
        child_stream=child_stream,
    )
    child_id = "00000000-0000-4000-8000-000000000415"
    admission_id = "admission-fresh-experiment-crash"
    _admit(
        store,
        parent_episode_id=parent_id,
        child_episode_id=child_id,
        admission_id=admission_id,
    )
    parameters = {
        "auto_research_episode_id": parent_id,
        "parent_operation_id": root_id,
        "child_episode_id": child_id,
        "node_id": EXPERIMENT_ID,
        "goal": None,
        "goal_sha256": None,
        "invocation_limit": None,
        "admission_id": admission_id,
    }
    real_spawn_record = background._spawn_record

    def crash_before_spawn(*_args, **_kwargs):
        raise RuntimeError("simulated crash after fresh Experiment commit")

    monkeypatch.setattr(background, "_spawn_record", crash_before_spawn)
    with pytest.raises(RuntimeError, match="after fresh Experiment commit"):
        coordinator.kick_off(**parameters)

    child = store.episode(child_id)
    assert child is not None and child.root_operation_id is not None
    committed = store.agent_task(child.root_operation_id)
    assert committed is not None and committed.status == "queued"
    assert fresh_executions == 0
    monkeypatch.setattr(background, "_spawn_record", real_spawn_record)

    replayed = coordinator.kick_off(**parameters)
    assert replayed.disposition == "existing"
    assert replayed.operation_id == committed.operation_id
    wait_for_task(store, committed.operation_id, expect="succeeded")
    replayed_again = coordinator.kick_off(**parameters)
    assert replayed_again.operation_id == committed.operation_id
    assert fresh_executions == 1
    assert store.auto_research_experiment_allowance(parent_id).used == 1


def test_transient_experiment_kickoff_failure_keeps_admission_for_exact_recovery(
    manifest: Manifest,
    tmp_path: Path,
    monkeypatch,
) -> None:
    fresh_executions = 0

    async def child_stream(_project_id, _kind, _request, execution):
        nonlocal fresh_executions
        assert execution.continuation == "fresh"
        fresh_executions += 1
        yield _sse(AgentEvent(event="done"))

    _, store, background, coordinator, parent_id, root_id = _setup(
        manifest,
        tmp_path,
        child_stream=child_stream,
    )
    child_id = "00000000-0000-4000-8000-000000000417"
    admission_id = "admission-transient-experiment-kickoff"
    _admit(
        store,
        parent_episode_id=parent_id,
        child_episode_id=child_id,
        admission_id=admission_id,
    )
    parameters = {
        "auto_research_episode_id": parent_id,
        "parent_operation_id": root_id,
        "child_episode_id": child_id,
        "node_id": EXPERIMENT_ID,
        "goal": "Run the exact bounded goal after infrastructure recovers.",
        "goal_sha256": hashlib.sha256(
            b"Run the exact bounded goal after infrastructure recovers."
        ).hexdigest(),
        "invocation_limit": None,
        "admission_id": admission_id,
    }
    real_start = start_auto_research_child_experiment

    def unavailable(*_args, **_kwargs):
        raise OSError("canonical state is temporarily unavailable")

    before = store.auto_research_experiment_allowance(parent_id)
    monkeypatch.setattr(
        "rcp.runs.auto_research_experiments.start_auto_research_child_experiment", unavailable
    )
    with pytest.raises(OSError, match="temporarily unavailable"):
        coordinator.kick_off(**parameters)

    admission = store.auto_research_child_admission(admission_id)
    assert admission is not None and admission.state == "accepted"
    assert store.auto_research_child_experiment(child_id) is None
    assert store.auto_research_experiment_allowance(parent_id) == before

    monkeypatch.setattr(
        "rcp.runs.auto_research_experiments.start_auto_research_child_experiment", real_start
    )
    recovered = coordinator.kick_off(**parameters)

    assert recovered.disposition == "created"
    assert recovered.operation_id is not None
    wait_for_task(store, recovered.operation_id, expect="succeeded")
    admission = store.auto_research_child_admission(admission_id)
    assert admission is not None and admission.state == "reflected"
    assert store.auto_research_experiment_allowance(parent_id).used == before.used + 1
    assert fresh_executions == 1


def test_terminal_experiment_kickoff_replay_returns_existing_without_redispatch(
    manifest: Manifest,
    tmp_path: Path,
) -> None:
    fresh_executions = 0

    async def child_stream(_project_id, _kind, _request, execution):
        nonlocal fresh_executions
        assert execution.continuation == "fresh"
        fresh_executions += 1
        yield _sse(AgentEvent(event="done"))

    _, store, _, coordinator, parent_id, root_id = _setup(
        manifest,
        tmp_path,
        child_stream=child_stream,
    )
    child_id = "00000000-0000-4000-8000-000000000416"
    admission_id = "admission-terminal-experiment-replay"
    _admit(
        store,
        parent_episode_id=parent_id,
        child_episode_id=child_id,
        admission_id=admission_id,
    )
    parameters = {
        "auto_research_episode_id": parent_id,
        "parent_operation_id": root_id,
        "child_episode_id": child_id,
        "node_id": EXPERIMENT_ID,
        "goal": None,
        "goal_sha256": None,
        "invocation_limit": None,
        "admission_id": admission_id,
    }
    created = coordinator.kick_off(**parameters)
    assert created.operation_id is not None
    wait_for_task(store, created.operation_id, expect="succeeded")
    terminal = store.terminalize_auto_research_child_experiment(
        child_id,
        diagnostic="The bounded child Experiment completed.",
    )
    assert terminal.state == "terminal"

    replayed = coordinator.kick_off(**parameters)

    assert replayed.disposition == "existing"
    assert replayed.operation_id == created.operation_id
    assert replayed.status == "terminal"
    assert fresh_executions == 1
    assert store.auto_research_experiment_allowance(parent_id).used == 1


def _human_experiment_request(service: ProjectService, episode_id: str) -> RunRequest:
    state = service.history.state()
    node = state.nodes[EXPERIMENT_ID]
    assert isinstance(node, Experiment)
    from rcp.control import derive_experiment_control_state

    return fresh_experiment_run_request(
        service,
        RunRequest(
            chat_scope="node",
            chat_id=episode_id,
            node_id=EXPERIMENT_ID,
            message="Run the predecessor Experiment.",
            mode="work",
            trigger="experiment_run",
            patch_kind="experiment_loop",
        ),
        node=node,
        state_revision=state.revision,
        control=derive_experiment_control_state(state, EXPERIMENT_ID),
        episode_id=episode_id,
        trigger="experiment_run",
    )


def test_active_predecessor_is_gracefully_stopped_and_pending_replacement_can_cancel(
    manifest: Manifest,
    tmp_path: Path,
) -> None:
    release = threading.Event()
    predecessor_started = threading.Event()

    async def child_stream(_project_id, _kind, request, _execution):
        if request.control_episode_id == HUMAN_PREDECESSOR:
            predecessor_started.set()
            while not release.is_set():
                await asyncio.sleep(0.01)
        yield _sse(AgentEvent(event="done"))

    service, store, background, coordinator, parent_id, root_id = _setup(
        manifest,
        tmp_path,
        child_stream=child_stream,
    )
    parent = store.episode(parent_id)
    assert parent is not None and parent.authorized_by is not None
    predecessor = background.start(
        PROJECT_ID,
        "node_chat",
        _human_experiment_request(service, HUMAN_PREDECESSOR),
        authorized_by=parent.authorized_by,
    )
    assert predecessor_started.wait(timeout=2)
    _admit(
        store,
        parent_episode_id=parent_id,
        child_episode_id=REPLACEMENT_TO_CANCEL,
        admission_id="admission-to-cancel",
    )

    pending = coordinator.kick_off(
        auto_research_episode_id=parent_id,
        parent_operation_id=root_id,
        child_episode_id=REPLACEMENT_TO_CANCEL,
        node_id=EXPERIMENT_ID,
        goal=None,
        goal_sha256=None,
        invocation_limit=None,
        admission_id="admission-to-cancel",
    )

    assert pending.disposition == "replacement_pending"
    assert pending.status == "pending"
    predecessor_episode = store.episode(HUMAN_PREDECESSOR)
    assert predecessor_episode is not None
    assert predecessor_episode.stop_requested_at is not None
    assert predecessor_episode.stop_settled_at is None

    cancelled = coordinator.stop(parent_id, REPLACEMENT_TO_CANCEL)
    assert cancelled.disposition == "cancelled"
    route = store.auto_research_child_experiment(REPLACEMENT_TO_CANCEL)
    assert route is not None and route.state == "cancelled"
    notices = store.auto_research_lifecycle_notices(parent_id)
    assert [(item.source_kind, item.source_id, item.source_event) for item in notices] == [
        ("experiment_replacement", REPLACEMENT_TO_CANCEL, "cancelled")
    ]

    release.set()
    wait_for_task(store, predecessor.operation_id, expect="succeeded")


def test_restart_reconciliation_reissues_stop_after_durable_replacement_reservation(
    manifest: Manifest,
    tmp_path: Path,
) -> None:
    release = threading.Event()
    predecessor_started = threading.Event()

    async def child_stream(_project_id, _kind, request, _execution):
        if request.control_episode_id == RESTART_PREDECESSOR:
            predecessor_started.set()
            while not release.is_set():
                await asyncio.sleep(0.01)
        yield _sse(AgentEvent(event="done"))

    service, store, background, coordinator, parent_id, root_id = _setup(
        manifest,
        tmp_path,
        child_stream=child_stream,
    )
    parent = store.episode(parent_id)
    assert parent is not None and parent.authorized_by is not None
    predecessor = background.start(
        PROJECT_ID,
        "node_chat",
        _human_experiment_request(service, RESTART_PREDECESSOR),
        authorized_by=parent.authorized_by,
    )
    assert predecessor_started.wait(timeout=2)
    admission_id = "admission-restart-replacement"
    _admit(
        store,
        parent_episode_id=parent_id,
        child_episode_id=RESTART_REPLACEMENT,
        admission_id=admission_id,
    )
    now = store.now()
    store.reserve_auto_research_experiment_replacement(
        AutoResearchChildExperimentRecord(
            child_episode_id=RESTART_REPLACEMENT,
            auto_research_episode_id=parent_id,
            project_id=PROJECT_ID,
            control_node_id=EXPERIMENT_ID,
            state="pending",
            replaces_episode_id=RESTART_PREDECESSOR,
            request={"goal": None, "invocation_limit": None},
            parent_operation_id=root_id,
            created_at=now,
            updated_at=now,
        ),
        admission_id=admission_id,
    )
    assert store.episode(RESTART_PREDECESSOR).stop_requested_at is None  # type: ignore[union-attr]

    assert coordinator.reconcile(parent_id) == 0

    stopping = store.episode(RESTART_PREDECESSOR)
    assert stopping is not None and stopping.stop_requested_at is not None
    route = store.auto_research_child_experiment(RESTART_REPLACEMENT)
    assert route is not None and route.state == "pending"

    release.set()
    wait_for_task(store, predecessor.operation_id, expect="succeeded")
    assert coordinator.reconcile(parent_id) == 1
    route = store.auto_research_child_experiment(RESTART_REPLACEMENT)
    assert route is not None and route.state == "running"
    assert store.auto_research_experiment_allowance(parent_id).used == 1


def test_restart_recovers_the_stopped_predecessor_before_starting_its_replacement(
    manifest: Manifest,
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stopped-predecessor-stage"
    stage.mkdir()
    predecessor_started = threading.Event()
    pause_predecessor = threading.Event()
    recovery_started = threading.Event()
    finish_recovery = threading.Event()
    replacement_started = threading.Event()
    overlap_observed = threading.Event()

    async def child_stream(_project_id, _kind, request, execution):
        assert isinstance(request, RunRequest)
        if request.control_episode_id == STOP_RECOVERY_PREDECESSOR:
            if execution.continuation == "fresh":
                candidate = "{}"
                execution.store.record_agent_task_contract(
                    execution.operation_id,
                    "experiment_episode_context_candidate",
                    candidate,
                    hashlib.sha256(candidate.encode()).hexdigest(),
                )
                execution.checkpoint_stage("", str(stage))
                yield _sse(AgentEvent(event="session", session_id="stopped-predecessor-session"))
                predecessor_started.set()
                while not pause_predecessor.is_set():
                    await asyncio.sleep(0.01)
                yield _sse(AgentEvent(event="paused", text="Provider paused during Stop."))
                return
            assert request.session_id == "stopped-predecessor-session"
            recovery_started.set()
            while not finish_recovery.is_set():
                await asyncio.sleep(0.01)
            yield _sse(AgentEvent(event="done"))
            return
        if request.control_episode_id == STOP_RECOVERY_REPLACEMENT:
            if not finish_recovery.is_set():
                overlap_observed.set()
            replacement_started.set()
        yield _sse(AgentEvent(event="done"))

    service, store, background, coordinator, parent_id, root_id = _setup(
        manifest,
        tmp_path,
        child_stream=child_stream,
    )
    parent = store.episode(parent_id)
    assert parent is not None and parent.authorized_by is not None
    predecessor = background.start(
        PROJECT_ID,
        "node_chat",
        _human_experiment_request(service, STOP_RECOVERY_PREDECESSOR),
        authorized_by=parent.authorized_by,
    )
    assert predecessor_started.wait(timeout=2)
    admission_id = "admission-stopped-predecessor-recovery"
    _admit(
        store,
        parent_episode_id=parent_id,
        child_episode_id=STOP_RECOVERY_REPLACEMENT,
        admission_id=admission_id,
    )
    pending = coordinator.kick_off(
        auto_research_episode_id=parent_id,
        parent_operation_id=root_id,
        child_episode_id=STOP_RECOVERY_REPLACEMENT,
        node_id=EXPERIMENT_ID,
        goal=None,
        goal_sha256=None,
        invocation_limit=None,
        admission_id=admission_id,
    )
    assert pending.disposition == "replacement_pending"

    pause_predecessor.set()
    wait_for_task(store, predecessor.operation_id, expect="paused")
    stopped_predecessor = store.episode(STOP_RECOVERY_PREDECESSOR)
    assert stopped_predecessor is not None
    assert stopped_predecessor.status == "stopping"
    assert stopped_predecessor.stop_settled_at is None

    restarted = BackgroundAgentTasks(store, child_stream)
    restarted.recover_at_startup()
    restarted_coordinator = AutoResearchExperimentCoordinator(
        store,
        restarted,
        project_service=lambda project_id, _episode_id: (
            service if project_id == PROJECT_ID else None
        ),  # type: ignore[return-value]
        operation_lock=lambda _project_id: nullcontext(),
    )
    assert recovery_started.wait(timeout=2)
    recovery = next(
        task
        for task in store.episode_tasks(STOP_RECOVERY_PREDECESSOR)
        if task.parent_operation_id == predecessor.operation_id
    )
    assert recovery.request["session_id"] == "stopped-predecessor-session"
    assert store.episode(STOP_RECOVERY_PREDECESSOR).invocations_used == 1  # type: ignore[union-attr]
    assert restarted_coordinator.reconcile(parent_id) == 0
    assert not replacement_started.is_set()

    finish_recovery.set()
    wait_for_task(store, recovery.operation_id, expect="succeeded")
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        stopped_predecessor = store.episode(STOP_RECOVERY_PREDECESSOR)
        if stopped_predecessor is not None and stopped_predecessor.status == "stopped":
            break
        time.sleep(0.01)
    stopped_predecessor = store.episode(STOP_RECOVERY_PREDECESSOR)
    assert stopped_predecessor is not None and stopped_predecessor.status == "stopped"
    assert "experiment_stop_recovery" in {
        receipt.category for receipt in store.agent_task_receipts(recovery.operation_id)
    }

    assert restarted_coordinator.reconcile(parent_id) == 1
    assert replacement_started.wait(timeout=2)
    assert not overlap_observed.is_set()
    route = store.auto_research_child_experiment(STOP_RECOVERY_REPLACEMENT)
    assert route is not None and route.state == "running"
    assert store.auto_research_experiment_allowance(parent_id).used == 1


def test_kickoff_replaces_a_live_predecessor_even_when_its_runtime_is_idle(
    manifest: Manifest,
    tmp_path: Path,
) -> None:
    service, store, background, coordinator, parent_id, root_id = _setup(
        manifest,
        tmp_path,
    )
    parent = store.episode(parent_id)
    assert parent is not None and parent.authorized_by is not None
    predecessor = background.start(
        PROJECT_ID,
        "node_chat",
        _human_experiment_request(service, IDLE_PREDECESSOR),
        authorized_by=parent.authorized_by,
    )
    wait_for_task(store, predecessor.operation_id, expect="succeeded")
    runtime = store.experiment_loop_runtime(PROJECT_ID, EXPERIMENT_ID)
    assert runtime.episode_id == IDLE_PREDECESSOR
    assert runtime.active is False
    assert store.episode(IDLE_PREDECESSOR).status == "running"  # type: ignore[union-attr]
    admission_id = "admission-idle-replacement"
    _admit(
        store,
        parent_episode_id=parent_id,
        child_episode_id=IDLE_REPLACEMENT,
        admission_id=admission_id,
    )

    action = coordinator.kick_off(
        auto_research_episode_id=parent_id,
        parent_operation_id=root_id,
        child_episode_id=IDLE_REPLACEMENT,
        node_id=EXPERIMENT_ID,
        goal=None,
        goal_sha256=None,
        invocation_limit=None,
        admission_id=admission_id,
    )

    assert action.disposition == "created"
    assert store.episode(IDLE_PREDECESSOR).status == "stopped"  # type: ignore[union-attr]
    route = store.auto_research_child_experiment(IDLE_REPLACEMENT)
    assert route is not None and route.state == "running"


def test_pending_replacement_rechecks_readiness_and_emits_terminal_failure_notice(
    manifest: Manifest,
    tmp_path: Path,
) -> None:
    release = threading.Event()
    predecessor_started = threading.Event()

    async def child_stream(_project_id, _kind, request, _execution):
        if request.control_episode_id == READINESS_PREDECESSOR:
            predecessor_started.set()
            while not release.is_set():
                await asyncio.sleep(0.01)
        yield _sse(AgentEvent(event="done"))

    service, store, background, coordinator, parent_id, root_id = _setup(
        manifest,
        tmp_path,
        child_stream=child_stream,
    )
    parent = store.episode(parent_id)
    assert parent is not None and parent.authorized_by is not None
    predecessor = background.start(
        PROJECT_ID,
        "node_chat",
        _human_experiment_request(service, READINESS_PREDECESSOR),
        authorized_by=parent.authorized_by,
    )
    assert predecessor_started.wait(timeout=2)
    _admit(
        store,
        parent_episode_id=parent_id,
        child_episode_id=READINESS_REPLACEMENT,
        admission_id="admission-readiness",
    )
    pending = coordinator.kick_off(
        auto_research_episode_id=parent_id,
        parent_operation_id=root_id,
        child_episode_id=READINESS_REPLACEMENT,
        node_id=EXPERIMENT_ID,
        goal="Run after the predecessor stops.",
        goal_sha256=hashlib.sha256(b"Run after the predecessor stops.").hexdigest(),
        invocation_limit=None,
        admission_id="admission-readiness",
    )
    assert pending.disposition == "replacement_pending"

    service.history.append(_blocking_patch())
    release.set()
    wait_for_task(store, predecessor.operation_id, expect="succeeded")

    assert coordinator.reconcile(parent_id) == 1
    route = store.auto_research_child_experiment(READINESS_REPLACEMENT)
    assert route is not None
    assert route.state == "cancelled"
    assert route.terminal_diagnostic is not None
    assert "Blocker blk/new-readiness-failure is open." in route.terminal_diagnostic
    notices = store.auto_research_lifecycle_notices(parent_id)
    failed = [item for item in notices if item.source_id == READINESS_REPLACEMENT]
    assert len(failed) == 1
    assert failed[0].source_kind == "experiment_replacement"
    assert failed[0].source_event == "failed"
    assert failed[0].payload["status"] == "failed"
    assert "Blocker blk/new-readiness-failure is open." in str(failed[0].payload["diagnostic"])
