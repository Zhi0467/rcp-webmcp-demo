from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path

from rcp.agents import AgentEvent
from rcp.background import BackgroundAgentTasks
from rcp.core.transition_models import GraphHeadRef
from rcp.runs.auto_research import (
    AutoResearchRunRequest,
    AutoResearchStartRequest,
    settle_auto_research_stop,
)
from rcp.runs.auto_research_admission import (
    start_auto_research,
    start_auto_research_turn,
)
from rcp.runs.auto_research_recovery import (
    AutoResearchOrchestratorTerminalFailure,
    reconcile_auto_research_task_settlement,
    reconcile_due_auto_research_recoveries,
)
from rcp.storage import AppStore, ProjectRecord

from .helpers import fabricated_authorizer, wait_for_task


def _sse(event: AgentEvent) -> str:
    return f"data: {event.model_dump_json()}\n\n"


def _store(tmp_path: Path) -> AppStore:
    store = AppStore(tmp_path / "rcp.sqlite3")
    store.upsert_project(
        ProjectRecord(
            project_id="project",
            locator="/tmp/project/research.yaml",
            name="project",
            state_location="/tmp/project/.research",
            state_remote=False,
            added_at=store.now(),
        )
    )
    return store


def _start(tasks: BackgroundAgentTasks, *, operation_id: str = "root"):
    episode, root = start_auto_research(
        tasks,
        "project",
        AutoResearchStartRequest(invocation_ceiling=4, run_truth_scope=["repo"]),
        authorized_by=fabricated_authorizer(),
        graph_base_head=GraphHeadRef(revision=0),
        ensure_graph_target=lambda _episode: None,
        episode_id="auto_research",
        operation_id=operation_id,
    )
    assert episode.graph_target.kind == "branch"
    assert episode.graph_target.branch_id == episode.episode_id
    assert episode.graph_base_head == GraphHeadRef(revision=0)
    assert root.graph_target == episode.graph_target
    return episode, root


def _install_recovery_callback(tasks: BackgroundAgentTasks) -> None:
    """Stand in for the app's one settlement callback, Auto-research half only."""

    def settled(_project_id, _kind, request, execution) -> None:
        if not isinstance(request, AutoResearchRunRequest):
            return
        episode = tasks.store.episode(request.episode_id)
        if episode is None:
            return
        if episode.stop_requested_at is not None:
            episode = settle_auto_research_stop(tasks.store, episode.episode_id) or episode
        reconcile_auto_research_task_settlement(tasks, episode, request, execution)

    tasks.on_task_settled = settled


def _wait_for_recovery(store: AppStore, recovery_id: str):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        recovery = store.auto_research_recovery(recovery_id)
        if recovery is not None:
            return recovery
        time.sleep(0.01)
    raise AssertionError(f"auto_research recovery did not appear: {recovery_id}")


def _recovery_delay_seconds(recovery) -> int:
    assert recovery.next_attempt_at is not None
    return round(
        (
            datetime.fromisoformat(recovery.next_attempt_at)
            - datetime.fromisoformat(recovery.updated_at)
        ).total_seconds()
    )


def test_transient_orchestrator_failure_retries_exact_session_without_spend(tmp_path: Path) -> None:
    store = _store(tmp_path)
    stage = tmp_path / "orchestrator-stage"
    stage.mkdir()
    observed: list[tuple[str, str | None]] = []

    async def stream(_project_id, _kind, request, execution):
        observed.append((execution.continuation, request.session_id))
        if execution.continuation == "fresh":
            execution.checkpoint_stage("", str(stage))
            yield _sse(AgentEvent(event="session", session_id="session-1"))
            yield _sse(AgentEvent(event="error", text="temporary provider failure"))
            return
        yield _sse(AgentEvent(event="session", session_id="session-1"))
        yield _sse(AgentEvent(event="done"))

    tasks = BackgroundAgentTasks(store, stream)
    _install_recovery_callback(tasks)
    auto_research, root = _start(tasks)
    root = wait_for_task(store, root.operation_id, expect="failed")
    recovery = _wait_for_recovery(store, "task:root")
    assert recovery.retry_mode == "exact"
    assert store.episode_budget_meter(auto_research.episode_id).invocations_used == 1

    assert (
        reconcile_due_auto_research_recoveries(
            tasks,
            as_of=recovery.next_attempt_at,
        )
        == 1
    )
    admitted = store.auto_research_recovery("task:root")
    assert admitted is not None and admitted.admitted_operation_id is not None
    child = wait_for_task(store, admitted.admitted_operation_id, expect="succeeded")
    assert child.native_session_id == root.native_session_id == "session-1"
    assert child.stage_root == root.stage_root == str(stage)
    assert store.episode_budget_meter(auto_research.episode_id).invocations_used == 1
    assert observed == [("fresh", None), ("retry", "session-1")]


def test_due_recovery_adopts_existing_human_retry_without_spawning_or_deferring(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = _store(tmp_path)
    stage = tmp_path / "orchestrator-stage"
    stage.mkdir()

    async def stream(_project_id, _kind, _request, execution):
        execution.checkpoint_stage("", str(stage))
        yield _sse(AgentEvent(event="session", session_id="session-1"))
        if execution.continuation == "fresh":
            yield _sse(AgentEvent(event="error", text="provider unavailable"))
        else:
            yield _sse(AgentEvent(event="done"))

    tasks = BackgroundAgentTasks(store, stream)
    _install_recovery_callback(tasks)
    _, root = _start(tasks)
    wait_for_task(store, root.operation_id, expect="failed")
    recovery = _wait_for_recovery(store, "task:root")
    human_child = tasks.retry(root.operation_id)
    wait_for_task(store, human_child.operation_id, expect="succeeded")

    def unexpected_retry(_operation_id):
        raise AssertionError("automatic reconciliation must adopt the existing child")

    monkeypatch.setattr(tasks, "retry", unexpected_retry)
    assert (
        reconcile_due_auto_research_recoveries(
            tasks,
            as_of=recovery.next_attempt_at,
        )
        == 1
    )
    admitted = store.auto_research_recovery("task:root")
    assert admitted is not None
    assert admitted.status == "admitted"
    assert admitted.attempts == 1
    assert admitted.admitted_operation_id == human_child.operation_id
    assert admitted.next_attempt_at is None


def test_automatic_recovery_adopts_human_child_that_wins_admission_race(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = _store(tmp_path)
    stage = tmp_path / "orchestrator-stage"
    stage.mkdir()

    async def stream(_project_id, _kind, _request, execution):
        execution.checkpoint_stage("", str(stage))
        yield _sse(AgentEvent(event="session", session_id="session-1"))
        if execution.continuation == "fresh":
            yield _sse(AgentEvent(event="error", text="provider unavailable"))
        else:
            yield _sse(AgentEvent(event="done"))

    tasks = BackgroundAgentTasks(store, stream)
    _install_recovery_callback(tasks)
    _, root = _start(tasks)
    wait_for_task(store, root.operation_id, expect="failed")
    recovery = _wait_for_recovery(store, "task:root")
    real_retry = tasks.retry
    automatic_entered = threading.Event()
    human_admitted = threading.Event()

    def racing_automatic_retry(operation_id):
        automatic_entered.set()
        assert human_admitted.wait(5)
        return real_retry(operation_id)

    monkeypatch.setattr(tasks, "retry", racing_automatic_retry)
    errors: list[BaseException] = []

    def reconcile() -> None:
        try:
            reconcile_due_auto_research_recoveries(
                tasks,
                as_of=recovery.next_attempt_at,
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=reconcile)
    thread.start()
    assert automatic_entered.wait(5)
    human_child = real_retry(root.operation_id)
    human_admitted.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert errors == []
    wait_for_task(store, human_child.operation_id, expect="succeeded")

    admitted = store.auto_research_recovery("task:root")
    assert admitted is not None
    assert admitted.status == "admitted"
    assert admitted.attempts == 1
    assert admitted.admitted_operation_id == human_child.operation_id
    assert admitted.next_attempt_at is None


def test_precheckpoint_failure_retries_clean_orchestrator_session_and_survives_restart(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    stage = tmp_path / "orchestrator-stage"
    stage.mkdir()
    observed: list[str | None] = []

    async def stream(_project_id, _kind, request, execution):
        observed.append(request.session_id)
        if execution.continuation == "fresh":
            execution.checkpoint_stage("", str(stage))
            yield _sse(AgentEvent(event="error", text="network unavailable"))
            return
        yield _sse(AgentEvent(event="session", session_id="clean-session"))
        yield _sse(AgentEvent(event="done"))

    tasks = BackgroundAgentTasks(store, stream)
    _install_recovery_callback(tasks)
    auto_research, root = _start(tasks)
    wait_for_task(store, root.operation_id, expect="failed")
    pending = _wait_for_recovery(store, "task:root")
    assert pending.retry_mode == "clean"

    restarted = BackgroundAgentTasks(AppStore(store.path), stream)
    assert len(restarted.store.due_auto_research_recoveries(as_of=pending.next_attempt_at)) == 1
    assert (
        reconcile_due_auto_research_recoveries(
            restarted,
            as_of=pending.next_attempt_at,
        )
        == 1
    )
    admitted = restarted.store.auto_research_recovery("task:root")
    assert admitted is not None and admitted.admitted_operation_id is not None
    child = wait_for_task(restarted.store, admitted.admitted_operation_id, expect="succeeded")
    assert child.native_session_id == "clean-session"
    assert observed == [None, None]
    assert restarted.store.episode_budget_meter(auto_research.episode_id).invocations_used == 1


def test_worker_failure_never_becomes_auto_research_verdict(tmp_path: Path) -> None:
    store = _store(tmp_path)

    async def stream(_project_id, _kind, request, execution):
        execution.checkpoint_stage("", str(tmp_path))
        yield _sse(AgentEvent(event="session", session_id=f"session-{request.role}"))
        if request.role == "worker":
            yield _sse(AgentEvent(event="error", text="worker failed"))
        else:
            yield _sse(AgentEvent(event="done"))

    tasks = BackgroundAgentTasks(store, stream)
    _install_recovery_callback(tasks)
    auto_research, root = _start(tasks)
    root = wait_for_task(store, root.operation_id, expect="succeeded")
    worker = start_auto_research_turn(
        tasks,
        auto_research.episode_id,
        AutoResearchRunRequest(
            episode_id=auto_research.episode_id,
            role="worker",
            control_node_id="exp/check",
        ),
        parent_operation_id=root.operation_id,
        operation_id="worker",
    )
    wait_for_task(store, worker.operation_id, expect="failed")
    current = store.episode(auto_research.episode_id)
    assert current is not None and current.status == "running" and current.ending is None
    assert store.auto_research_recovery("task:worker") is None


def test_session_limit_uses_clean_orchestrator_retry_even_after_checkpoint(tmp_path: Path) -> None:
    store = _store(tmp_path)
    stage = tmp_path / "orchestrator-stage"
    stage.mkdir()

    async def stream(_project_id, _kind, _request, execution):
        execution.checkpoint_stage("", str(stage))
        yield _sse(AgentEvent(event="session", session_id="limited-session"))
        yield _sse(AgentEvent(event="error", text="provider session limit reached"))

    tasks = BackgroundAgentTasks(store, stream)
    _install_recovery_callback(tasks)
    _, root = _start(tasks)
    wait_for_task(store, root.operation_id, expect="failed")
    recovery = _wait_for_recovery(store, "task:root")
    assert recovery.failure_kind == "session_limit"
    assert recovery.retry_mode == "clean"


def test_repeated_provider_failures_share_one_bounded_allocation_recovery(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = _store(tmp_path)
    stage = tmp_path / "orchestrator-stage"
    stage.mkdir()

    async def stream(_project_id, _kind, _request, execution):
        if execution.continuation == "fresh":
            execution.checkpoint_stage("", str(stage))
        yield _sse(AgentEvent(event="session", session_id="session-1"))
        yield _sse(AgentEvent(event="error", text="provider unavailable"))

    tasks = BackgroundAgentTasks(store, stream)
    _install_recovery_callback(tasks)
    auto_research, root = _start(tasks)
    wait_for_task(store, root.operation_id, expect="failed")
    recovery = _wait_for_recovery(store, "task:root")
    assert recovery.attempts == 0
    assert _recovery_delay_seconds(recovery) == 120

    complete_recovery = store.complete_auto_research_recovery

    def complete_after_child_settles(
        recovery_id: str,
        *,
        admitted_operation_id: str | None = None,
        expected_operation_id: str | None = None,
    ):
        if admitted_operation_id is not None:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                child = store.agent_task(admitted_operation_id)
                current = store.auto_research_recovery(recovery_id)
                if (
                    child is not None
                    and child.status == "failed"
                    and current is not None
                    and current.operation_id == admitted_operation_id
                ):
                    break
                time.sleep(0.01)
            else:
                raise AssertionError("recovery child did not settle before admission checkpoint")
        return complete_recovery(
            recovery_id,
            admitted_operation_id=admitted_operation_id,
            expected_operation_id=expected_operation_id,
        )

    monkeypatch.setattr(store, "complete_auto_research_recovery", complete_after_child_settles)

    for expected_attempt, expected_consumed, expected_delay in (
        (2, 1, 240),
        (3, 2, 480),
        (4, 3, None),
    ):
        expected_status = "exhausted" if expected_delay is None else "pending"
        reconcile_due_auto_research_recoveries(
            tasks,
            as_of=recovery.next_attempt_at,
        )
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            recovery = store.auto_research_recovery("task:root")
            assert recovery is not None
            current = store.agent_task(recovery.operation_id or "")
            if (
                recovery.attempts == expected_consumed
                and recovery.status == expected_status
                and current is not None
                and current.attempt == expected_attempt
                and current.status == "failed"
            ):
                break
            time.sleep(0.01)
        else:
            raise AssertionError(f"auto_research recovery attempt {expected_attempt} did not fail")
        assert recovery.attempts == expected_consumed
        if expected_delay is None:
            assert recovery.status == "exhausted"
            assert recovery.next_attempt_at is None
        else:
            assert recovery.status == "pending"
            assert _recovery_delay_seconds(recovery) == expected_delay

    assert recovery.recovery_id == "task:root"
    assert recovery.attempts == 3
    assert recovery.status == "exhausted"
    current_auto_research = store.episode(auto_research.episode_id)
    assert current_auto_research is not None
    assert current_auto_research.status == "running"
    assert current_auto_research.ending is None
    assert store.episode_budget_meter(auto_research.episode_id).invocations_used == 1


def test_admission_and_provider_failures_share_durable_allocation_attempt_cap(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = _store(tmp_path)
    stage = tmp_path / "orchestrator-stage"
    stage.mkdir()

    async def stream(_project_id, _kind, _request, execution):
        if execution.continuation == "fresh":
            execution.checkpoint_stage("", str(stage))
        yield _sse(AgentEvent(event="session", session_id="session-1"))
        yield _sse(AgentEvent(event="error", text="provider unavailable"))

    tasks = BackgroundAgentTasks(store, stream)
    _install_recovery_callback(tasks)
    _, root = _start(tasks)
    wait_for_task(store, root.operation_id, expect="failed")
    recovery = _wait_for_recovery(store, "task:root")

    retry = tasks.retry

    def fail_admission(_operation_id):
        raise RuntimeError("recovery admission unavailable")

    monkeypatch.setattr(tasks, "retry", fail_admission)
    reconcile_due_auto_research_recoveries(
        tasks,
        as_of=recovery.next_attempt_at,
    )
    recovery = store.auto_research_recovery("task:root")
    assert recovery is not None
    assert recovery.attempts == 1
    assert recovery.status == "pending"
    assert _recovery_delay_seconds(recovery) == 240

    monkeypatch.setattr(tasks, "retry", retry)
    restarted = BackgroundAgentTasks(AppStore(store.path), stream)
    _install_recovery_callback(restarted)
    assert len(restarted.store.due_auto_research_recoveries(as_of=recovery.next_attempt_at)) == 1
    reconcile_due_auto_research_recoveries(
        restarted,
        as_of=recovery.next_attempt_at,
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        recovery = restarted.store.auto_research_recovery("task:root")
        assert recovery is not None
        current = restarted.store.agent_task(recovery.operation_id or "")
        if (
            recovery.attempts == 2
            and recovery.status == "pending"
            and current is not None
            and current.status == "failed"
        ):
            break
        time.sleep(0.01)
    else:
        raise AssertionError("mixed recovery provider attempt did not fail")
    assert recovery.status == "pending"
    assert _recovery_delay_seconds(recovery) == 480

    durable_store = AppStore(store.path)
    durable = durable_store.auto_research_recovery("task:root")
    assert durable is not None
    assert durable.attempts == 2
    assert durable.status == "pending"
    assert durable.next_attempt_at == recovery.next_attempt_at

    durable_tasks = BackgroundAgentTasks(durable_store, stream)
    monkeypatch.setattr(durable_tasks, "retry", fail_admission)
    reconcile_due_auto_research_recoveries(
        durable_tasks,
        as_of=durable.next_attempt_at,
    )
    exhausted = AppStore(store.path).auto_research_recovery("task:root")
    assert exhausted is not None
    assert exhausted.attempts == 3
    assert exhausted.status == "exhausted"
    assert exhausted.next_attempt_at is None


def test_typed_structural_orchestrator_failure_fences_atomically(tmp_path: Path) -> None:
    store = _store(tmp_path)
    stage = tmp_path / "orchestrator-stage"
    stage.mkdir()

    async def stream(_project_id, _kind, _request, execution):
        execution.checkpoint_stage("", str(stage))
        yield _sse(AgentEvent(event="session", session_id="orchestrator-session"))
        raise AutoResearchOrchestratorTerminalFailure(
            "typed structural failure",
        )

    tasks = BackgroundAgentTasks(store, stream)
    _install_recovery_callback(tasks)
    auto_research, root = _start(tasks)
    wait_for_task(store, root.operation_id, expect="failed")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        fenced = store.episode(auto_research.episode_id)
        if fenced is not None and fenced.status == "wrapping_up":
            break
        time.sleep(0.01)
    else:
        raise AssertionError("typed failure did not fence the auto_research")
    assert fenced.status == "wrapping_up"
    assert fenced.ending == "failed"
    assert fenced.ending_diagnostic == "typed structural failure"
    assert store.auto_research_recovery("task:root") is None

    receipts = store.agent_task_receipts(root.operation_id)
    typed = [item for item in receipts if item.category == "auto_research_orchestrator_failure"]
    assert len(typed) == 1
    assert typed[0].payload["classification"] == "structural_unrecoverable"
    assert typed[0].payload["recoverable"] is False
