from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from rcp.agents import AgentEvent
from rcp.background import BackgroundAgentTasks
from rcp.core.transition_models import GraphHeadRef
from rcp.runs.auto_research import AutoResearchRunRequest, AutoResearchStartRequest
from rcp.runs.auto_research_admission import (
    auto_research_child_work_task,
    ensure_auto_research_child_work_spawned,
    pause_auto_research_child_work,
    reconcile_committed_auto_research_dispatches,
    reconcile_reserved_auto_research_roots,
    reserve_auto_research,
    resume_auto_research_child_experiment,
    resume_auto_research_child_work,
    start_auto_research,
    start_auto_research_child_experiment,
    start_auto_research_child_work,
    start_auto_research_child_work_message_wake,
    start_auto_research_turn,
    stop_auto_research,
    stop_auto_research_child_work,
)
from rcp.runs.episodes.report import start_episode_report
from rcp.runs.episodes.wrapup import EpisodeWrapupSpec, begin_episode_report_wrapup
from rcp.runs.tasks.episode_report import EpisodeReportRunRequest
from rcp.runs.watcher_admission import start_watcher_notification
from rcp.service import RunRequest, resolve_dispatch_authority
from rcp.storage import (
    AgentTaskRecord,
    AppStore,
    AutoResearchChildExperimentRecord,
    AutoResearchMessageRecord,
    EpisodeInvocationCeilingReached,
    EpisodeRecord,
    EpisodeReportRecord,
    ProjectRecord,
    WatcherContinuation,
    WatcherRecord,
)

from .helpers import fabricated_authorizer, wait_for_task

_EXPERIMENT_ID = "exp/background-admission"
_EXPERIMENT_EPISODE_ID = "00000000-0000-4000-8000-000000000101"


def _sse(event: AgentEvent) -> str:
    return f"data: {event.model_dump_json()}\n\n"


def _store(tmp_path: Path) -> AppStore:
    store = AppStore(tmp_path / "rcp.sqlite3")
    store.upsert_project(
        ProjectRecord(
            project_id="project",
            locator=str(tmp_path / "research.yaml"),
            name="Project",
            state_location=str(tmp_path / ".research"),
            state_remote=False,
            added_at=store.now(),
        )
    )
    return store


def _auto_start(**updates: object) -> AutoResearchStartRequest:
    return AutoResearchStartRequest.model_validate(
        {
            "invocation_ceiling": 3,
            "provider": "codex",
            "model": "",
            "reasoning": "medium",
            "run_on": "laptop",
            "run_truth_scope": ["repo"],
            **updates,
        }
    )


def _experiment_request(
    *,
    trigger: str = "experiment_run",
    invocation: int = 1,
    watcher_ids: list[str] | None = None,
    session_id: str | None = None,
) -> RunRequest:
    return RunRequest(
        provider="codex",
        model="",
        reasoning="medium",
        run_on="laptop",
        run_truth_scope=["repo"],
        chat_id="experiment-chat",
        chat_scope="node",
        node_id=_EXPERIMENT_ID,
        message="Continue the bounded experiment.",
        mode="work",
        trigger=trigger,
        patch_kind="experiment_loop",
        control_node_id=_EXPERIMENT_ID,
        control_revision=1,
        control_episode_id=_EXPERIMENT_EPISODE_ID,
        control_invocation=invocation,
        control_invocation_ceiling=3,
        control_decision_bundle=[],
        control_completion_criteria=["The bounded comparison is analyzed."],
        watcher_ids=list(watcher_ids or []),
        session_id=session_id,
    )


def _spawned_work_request(worker_id: str, instruction: str) -> RunRequest:
    return RunRequest(
        provider="codex",
        model="",
        reasoning="medium",
        run_on="laptop",
        run_truth_scope=["repo"],
        chat_id=worker_id,
        chat_scope="node",
        node_id="blk/spawned-work",
        message=instruction,
        mode="work",
        trigger="orchestrator",
        patch_kind="work",
    )


def _child_experiment_request(episode_id: str, goal: str) -> RunRequest:
    return RunRequest(
        provider="codex",
        model="",
        reasoning="medium",
        run_on="laptop",
        run_truth_scope=["repo"],
        chat_id="00000000-0000-4000-8000-000000000302",
        chat_scope="node",
        node_id="exp/child",
        message=goal,
        mode="work",
        trigger="orchestrator",
        patch_kind="experiment_loop",
        control_node_id="exp/child",
        control_revision=1,
        control_episode_id=episode_id,
        control_invocation=1,
        control_invocation_ceiling=2,
        control_completion_criteria=["The bounded child comparison is analyzed."],
    )


def _admitted_launch_task(
    store: AppStore,
    *,
    operation_id: str,
    request: RunRequest | None = None,
    parent_operation_id: str | None = None,
    record_updates: dict[str, object] | None = None,
) -> AgentTaskRecord:
    request = request or RunRequest(
        provider="codex",
        model="",
        reasoning="medium",
        run_on="laptop",
        run_truth_scope=["repo"],
        chat_scope="project",
        chat_id=f"launch-{operation_id}",
        message="Exercise the admitted launch boundary.",
        mode="work",
        patch_kind="work",
    )
    authority = resolve_dispatch_authority("project_chat", request)
    assert authority is not None
    now = store.now()
    record = AgentTaskRecord(
        operation_id=operation_id,
        project_id="project",
        kind="project_chat",
        status="queued",
        request=request.model_dump(mode="json"),
        created_at=now,
        updated_at=now,
        status_message="Queued",
        attempt=2 if parent_operation_id is not None else 1,
        parent_operation_id=parent_operation_id,
        phase="queued",
        last_activity_at=now,
        authorized_by=fabricated_authorizer("Researcher"),
        dispatch_authority=authority,
    )
    return store.create_agent_task(record.model_copy(update=record_updates or {}))


async def _done_stream(_project_id, _kind, _request, _execution):
    yield _sse(AgentEvent(event="done"))


def test_construction_leaves_recovery_undone_until_startup_asks_for_it(
    tmp_path: Path,
) -> None:
    """Constructing the engine must not write to the store.

    Startup recovery is an explicit call so that constructing this object — which
    358 sites do, most of them tests — cannot silently interrupt live work.
    """

    store = _store(tmp_path)
    task = _admitted_launch_task(store, operation_id="survives-construction")
    store.mark_agent_task_running(task.operation_id)

    tasks = BackgroundAgentTasks(store, _done_stream)

    untouched = store.agent_task(task.operation_id)
    assert untouched is not None and untouched.status == "running"

    tasks.recover_at_startup()

    interrupted = store.agent_task(task.operation_id)
    assert interrupted is not None and interrupted.status == "interrupted"


def test_launch_admitted_missing_operation_is_read_only(tmp_path: Path) -> None:
    store = _store(tmp_path)
    tasks = BackgroundAgentTasks(store, _done_stream)

    with pytest.raises(KeyError, match="missing-launch"):
        tasks.launch_admitted("missing-launch")

    assert store.agent_task("missing-launch") is None
    assert store.agent_task_receipts("missing-launch") == []


def test_launch_admitted_valid_task_uses_persisted_cause_and_receipt_order(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    tasks = BackgroundAgentTasks(store, _done_stream)
    task = _admitted_launch_task(store, operation_id="valid-launch")

    launched = tasks.launch_admitted(task.operation_id)
    finished = wait_for_task(store, launched.operation_id, expect="succeeded")

    assert finished.operation_id == task.operation_id
    assert [
        item.category
        for item in store.agent_task_receipts(task.operation_id)
        if item.category
        in {
            "operation_admitted",
            "operation_dispatch_attempt",
            "operation_created",
            "operation_dispatch_started",
        }
    ] == [
        "operation_admitted",
        "operation_dispatch_attempt",
        "operation_created",
        "operation_dispatch_started",
    ]


def test_runtime_is_checkpointed_before_the_provider_session(tmp_path: Path) -> None:
    store = _store(tmp_path)

    async def stream(_project_id, _kind, _request, _execution):
        yield _sse(AgentEvent(event="runtime", text="codex.app-server-stdio.v1"))
        yield _sse(AgentEvent(event="session", session_id="app-server-thread"))
        yield _sse(AgentEvent(event="done"))

    tasks = BackgroundAgentTasks(store, stream)
    task = _admitted_launch_task(store, operation_id="runtime-checkpoint")
    launched = tasks.launch_admitted(task.operation_id)
    finished = wait_for_task(store, launched.operation_id, expect="succeeded")

    assert finished.runtime_id == "codex.app-server-stdio.v1"
    receipt = next(
        item
        for item in store.agent_task_receipts(task.operation_id)
        if item.category == "provider_runtime_selected"
    )
    assert receipt.payload == {
        "provider": "codex",
        "runtime_id": "codex.app-server-stdio.v1",
    }


def test_a_passed_over_runtime_is_recorded_as_a_diagnostic(tmp_path: Path) -> None:
    """The fallback is silent to the human, so the reason must survive somewhere."""

    store = _store(tmp_path)

    async def stream(_project_id, _kind, _request, _execution):
        yield _sse(
            AgentEvent(
                event="runtime_fallback",
                text=json.dumps(
                    {"runtime_id": "codex.app-server-stdio.v1", "detail": "codex 0.140.0 is old"}
                ),
            )
        )
        yield _sse(AgentEvent(event="runtime", text="codex.exec-json.v1"))
        yield _sse(AgentEvent(event="session", session_id="exec-thread"))
        yield _sse(AgentEvent(event="done"))

    tasks = BackgroundAgentTasks(store, stream)
    task = _admitted_launch_task(store, operation_id="runtime-fallback")
    launched = tasks.launch_admitted(task.operation_id)
    finished = wait_for_task(store, launched.operation_id, expect="succeeded")

    assert finished.runtime_id == "codex.exec-json.v1"
    receipt = next(
        item
        for item in store.agent_task_receipts(task.operation_id)
        if item.category == "provider_runtime_fallback"
    )
    assert receipt.payload == {
        "runtime_id": "codex.app-server-stdio.v1",
        "detail": "codex 0.140.0 is old",
    }


def test_launch_admitted_is_idempotent_for_live_and_terminal_duplicates(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    stage = tmp_path / "duplicate-launch-stage"
    stage.mkdir()

    async def stream(_project_id, _kind, _request, execution):
        execution.checkpoint_stage("", str(stage))
        yield _sse(AgentEvent(event="session", session_id="duplicate-launch-session"))
        entered.set()
        while not release.is_set():
            await asyncio.sleep(0.01)
        yield _sse(AgentEvent(event="done"))

    tasks = BackgroundAgentTasks(store, stream)
    task = _admitted_launch_task(store, operation_id="duplicate-launch")
    tasks.launch_admitted(task.operation_id)
    assert entered.wait(timeout=2)

    live_duplicate = tasks.launch_admitted(task.operation_id)
    assert live_duplicate.operation_id == task.operation_id
    assert live_duplicate.native_session_id == "duplicate-launch-session"
    release.set()
    wait_for_task(store, task.operation_id, expect="succeeded")

    terminal_duplicate = tasks.launch_admitted(task.operation_id)
    assert terminal_duplicate.status == "succeeded"
    receipts = store.agent_task_receipts(task.operation_id)
    assert sum(item.category == "operation_created" for item in receipts) == 1
    assert sum(item.category == "operation_dispatch_attempt" for item in receipts) == 1


@pytest.mark.parametrize("intent_state", ["missing", "malformed"])
def test_launch_admitted_rejects_missing_or_malformed_intent_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    intent_state: str,
) -> None:
    store = _store(tmp_path)
    tasks = BackgroundAgentTasks(store, _done_stream)
    task = _admitted_launch_task(store, operation_id=f"bad-intent-{intent_state}")
    before = store.agent_task_receipts(task.operation_id)
    if intent_state == "missing":
        monkeypatch.setattr(store, "agent_task_admission_intent", lambda _operation_id: None)
    else:
        monkeypatch.setattr(
            store,
            "agent_task_admission_intent",
            lambda _operation_id: (_ for _ in ()).throw(ValueError("malformed intent")),
        )

    with pytest.raises(ValueError, match="intent"):
        tasks.launch_admitted(task.operation_id)

    assert store.agent_task(task.operation_id).status == "queued"  # type: ignore[union-attr]
    assert store.agent_task_receipts(task.operation_id) == before
    assert task.operation_id not in tasks._workers


def test_launch_admitted_rejects_unknown_dispatch_attempt_before_new_receipts(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    tasks = BackgroundAgentTasks(store, _done_stream)
    task = _admitted_launch_task(store, operation_id="unknown-dispatch-attempt")
    store.record_agent_task_receipt(
        task.operation_id,
        "operation_dispatch_attempt",
        {"dispatch_attempt_id": "unknown-attempt"},
        tier="diagnostic",
    )
    before = store.agent_task_receipts(task.operation_id)

    with pytest.raises(ValueError, match="ambiguous|already-started"):
        tasks.launch_admitted(task.operation_id)

    assert store.agent_task_receipts(task.operation_id) == before
    assert task.operation_id not in tasks._workers


@pytest.mark.parametrize(
    ("record_updates", "message"),
    [
        ({"dispatch_authority": None}, "dispatch authority"),
        ({"native_session_id": "changed-session"}, "native session"),
        ({"stage_host": "remote"}, "stage binding"),
        ({"write_scope_fingerprint": "a" * 64}, "write-scope"),
    ],
)
def test_launch_admitted_rejects_invalid_launch_bindings_before_dispatch(
    tmp_path: Path,
    record_updates: dict[str, object],
    message: str,
) -> None:
    store = _store(tmp_path)
    tasks = BackgroundAgentTasks(store, _done_stream)
    task = _admitted_launch_task(
        store,
        operation_id=f"invalid-binding-{message.replace(' ', '-')}",
        record_updates=record_updates,
    )
    before = store.agent_task_receipts(task.operation_id)

    with pytest.raises(ValueError, match=message):
        tasks.launch_admitted(task.operation_id)

    assert store.agent_task_receipts(task.operation_id) == before
    assert task.operation_id not in tasks._workers


def test_launch_admitted_retries_proven_prestart_failure_without_duplicate_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    tasks = BackgroundAgentTasks(store, _done_stream)
    task = _admitted_launch_task(store, operation_id="prestart-retry")
    original_start = threading.Thread.start
    failed = True

    def fail_once(worker: threading.Thread) -> None:
        nonlocal failed
        if failed:
            failed = False
            raise RuntimeError("thread start failed before provider launch")
        original_start(worker)

    monkeypatch.setattr(threading.Thread, "start", fail_once)
    with pytest.raises(RuntimeError, match="before provider launch"):
        tasks.launch_admitted(task.operation_id)

    first = store.agent_task_receipts(task.operation_id)
    assert sum(item.category == "operation_created" for item in first) == 1
    assert sum(item.category == "operation_dispatch_attempt" for item in first) == 1
    assert sum(item.category == "operation_dispatch_failed_before_start" for item in first) == 1

    launched = tasks.launch_admitted(task.operation_id)
    wait_for_task(store, launched.operation_id, expect="succeeded")
    second = store.agent_task_receipts(task.operation_id)
    assert sum(item.category == "operation_created" for item in second) == 1
    assert sum(item.category == "operation_dispatch_attempt" for item in second) == 2
    assert sum(item.category == "operation_dispatch_failed_before_start" for item in second) == 1


def test_validated_spawn_record_rejects_both_parent_presence_directions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    parent = _admitted_launch_task(store, operation_id="parent-binding")
    store.mark_agent_task_running(parent.operation_id)
    store.complete_agent_task(parent.operation_id, applied_revision=None, result={})
    child = _admitted_launch_task(
        store,
        operation_id="child-binding",
        request=RunRequest.model_validate(parent.request),
        parent_operation_id=parent.operation_id,
    )
    tasks = BackgroundAgentTasks(store, _done_stream)
    request = BackgroundAgentTasks._request_from_record(child)

    monkeypatch.setattr(
        store,
        "agent_task",
        lambda operation_id: (
            child.model_copy(update={"parent_operation_id": parent.operation_id})
            if operation_id == child.operation_id
            else parent
        ),
    )
    with pytest.raises(ValueError, match="changed before background dispatch"):
        tasks._validated_spawn_record(
            child.model_copy(update={"parent_operation_id": None}), request, parent=None
        )

    monkeypatch.setattr(
        store,
        "agent_task",
        lambda operation_id: (
            child.model_copy(update={"parent_operation_id": None})
            if operation_id == child.operation_id
            else parent
        ),
    )
    with pytest.raises(ValueError, match="changed before background dispatch"):
        tasks._validated_spawn_record(child, request, parent=parent)


def test_auto_research_root_uses_episode_lineage_and_strict_request_decode(tmp_path: Path) -> None:
    store = _store(tmp_path)
    stage = tmp_path / "auto-stage"
    stage.mkdir()

    async def stream(_project_id, kind, request, execution):
        assert kind == "auto_research"
        assert isinstance(request, AutoResearchRunRequest)
        execution.checkpoint_stage("", str(stage))
        yield _sse(AgentEvent(event="session", session_id="auto-session"))
        yield _sse(AgentEvent(event="done"))

    tasks = BackgroundAgentTasks(store, stream)
    episode, root = start_auto_research(
        tasks,
        "project",
        _auto_start(starting_instruction="  Begin with the disputed claim.  "),
        authorized_by=fabricated_authorizer("Researcher"),
        graph_base_head=GraphHeadRef(revision=0),
        ensure_graph_target=lambda _episode: None,
        episode_id="auto-episode",
        operation_id="auto-root",
    )
    root = wait_for_task(store, root.operation_id, expect="succeeded")

    assert episode.mode == "auto_research"
    assert episode.graph_target.kind == "branch"
    assert episode.graph_target.branch_id == episode.episode_id
    assert episode.graph_base_head == GraphHeadRef(revision=0)
    assert root.kind == "auto_research"
    assert root.graph_target == episode.graph_target
    assert root.episode_id == episode.episode_id
    assert root.request["episode_id"] == episode.episode_id
    assert root.request["instruction"] == "Begin with the disputed claim."
    assert "campaign_id" not in root.request
    assert isinstance(tasks._request_from_record(root), AutoResearchRunRequest)
    assert store.episode_budget_meter(episode.episode_id).invocations_used == 1
    assert store.episode_tasks(episode.episode_id) == [root]


def test_reserved_auto_research_root_reconciles_branch_before_provider_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)

    async def forbidden_stream(*_args, **_kwargs):
        raise AssertionError("the reservation must not launch before branch reconciliation")
        yield  # pragma: no cover

    tasks = BackgroundAgentTasks(store, forbidden_stream)
    episode, root, _request = reserve_auto_research(
        tasks,
        "project",
        _auto_start(),
        authorized_by=fabricated_authorizer("Researcher"),
        graph_base_head=GraphHeadRef(revision=0),
        episode_id="reserved-auto-episode",
        operation_id="reserved-auto-root",
    )
    assert store.agent_task(root.operation_id).status == "queued"  # type: ignore[union-attr]

    restarted = BackgroundAgentTasks(store, forbidden_stream)
    order: list[str] = []

    def held_spawn(record, _request, **_kwargs):
        order.append("spawn")
        return record

    monkeypatch.setattr(restarted, "_spawn_record", held_spawn)
    started = reconcile_reserved_auto_research_roots(
        restarted, lambda reserved: order.append(f"branch:{reserved.episode_id}")
    )

    assert started == [root.operation_id]
    assert order == [f"branch:{episode.episode_id}", "spawn"]
    assert store.agent_task(root.operation_id).status == "queued"  # type: ignore[union-attr]


def test_spawned_child_uses_ordinary_node_work_and_atomically_spends_b(tmp_path: Path) -> None:
    store = _store(tmp_path)
    seen: list[tuple[str, object]] = []

    async def stream(_project_id, kind, request, _execution):
        seen.append((kind, request))
        yield _sse(AgentEvent(event="done"))

    background = BackgroundAgentTasks(store, stream)
    episode, root = start_auto_research(
        background,
        "project",
        _auto_start(),
        authorized_by=fabricated_authorizer("Researcher"),
        graph_base_head=GraphHeadRef(revision=0),
        ensure_graph_target=lambda _episode: None,
        episode_id="auto-child-work",
        operation_id="auto-child-work-root",
    )
    wait_for_task(store, root.operation_id, expect="succeeded")
    worker_id = "00000000-0000-4000-8000-000000000301"
    instruction = "Resolve the runtime blocker, then report the bounded evidence."
    request = _spawned_work_request(worker_id, instruction)

    child = start_auto_research_child_work(
        background,
        episode.episode_id,
        request,
        admitted_by_operation_id=root.operation_id,
        worker_id=worker_id,
        instruction=instruction,
        instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
    )
    child = wait_for_task(store, child.operation_id, expect="succeeded")

    route, current = auto_research_child_work_task(
        background,
        episode.episode_id,
        worker_id,
    )
    assert route.current_operation_id == child.operation_id == worker_id
    assert route.instruction == instruction
    assert current.kind == "node_chat"
    assert current.graph_target == episode.graph_target
    assert current.request["mode"] == "work"
    assert current.request["trigger"] == "orchestrator"
    assert current.dispatch_authority is not None
    assert current.dispatch_authority.profile == "ordinary"
    assert current.dispatch_authority.task_contract == "work_auto"
    assert store.episode_budget_meter(episode.episode_id).invocations_used == 2
    assert isinstance(seen[-1][1], RunRequest)
    assert seen[-1][0] == "node_chat"


def test_committed_child_dispatch_is_claimed_once_under_concurrent_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    executions = 0

    async def stream(_project_id, kind, _request, _execution):
        nonlocal executions
        if kind == "auto_research":
            yield _sse(AgentEvent(event="done"))
            return
        executions += 1
        entered.set()
        while not release.is_set():
            await asyncio.sleep(0.01)
        yield _sse(AgentEvent(event="done"))

    background = BackgroundAgentTasks(store, stream)
    episode, root = start_auto_research(
        background,
        "project",
        _auto_start(),
        authorized_by=fabricated_authorizer("Researcher"),
        graph_base_head=GraphHeadRef(revision=0),
        ensure_graph_target=lambda _episode: None,
        episode_id="auto-child-dispatch-claim",
        operation_id="auto-child-dispatch-root",
    )
    wait_for_task(store, root.operation_id, expect="succeeded")
    worker_id = "00000000-0000-4000-8000-000000000319"
    instruction = "Dispatch this committed worker once."
    real_spawn_record = background._spawn_record

    def crash_before_spawn(*_args, **_kwargs):
        raise RuntimeError("simulated crash after child commit")

    monkeypatch.setattr(background, "_spawn_record", crash_before_spawn)
    with pytest.raises(RuntimeError, match="after child commit"):
        start_auto_research_child_work(
            background,
            episode.episode_id,
            _spawned_work_request(worker_id, instruction),
            admitted_by_operation_id=root.operation_id,
            worker_id=worker_id,
            instruction=instruction,
            instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
        )
    monkeypatch.setattr(background, "_spawn_record", real_spawn_record)

    def ensure() -> str:
        return ensure_auto_research_child_work_spawned(
            background,
            episode.episode_id,
            worker_id,
            operation_id=worker_id,
            continuation="fresh",
        ).operation_id

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: ensure(), range(2)))

    assert results == [worker_id, worker_id]
    assert entered.wait(timeout=2)
    assert executions == 1
    release.set()
    wait_for_task(store, worker_id, expect="succeeded")


def test_restart_dispatches_committed_fresh_child_work_without_respending_b(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    executions: list[str] = []

    async def stream(_project_id, kind, _request, execution):
        if kind != "auto_research":
            executions.append(execution.continuation)
        yield _sse(AgentEvent(event="done"))

    background = BackgroundAgentTasks(store, stream)
    episode, root = start_auto_research(
        background,
        "project",
        _auto_start(),
        authorized_by=fabricated_authorizer("Researcher"),
        graph_base_head=GraphHeadRef(revision=0),
        ensure_graph_target=lambda _episode: None,
        episode_id="auto-child-fresh-restart",
        operation_id="auto-child-fresh-restart-root",
    )
    wait_for_task(store, root.operation_id, expect="succeeded")
    worker_id = "00000000-0000-4000-8000-000000000351"
    instruction = "Run the exact committed fresh Work allocation after restart."

    def crash_after_commit(*_args, **_kwargs):
        raise RuntimeError("simulated process loss before Work dispatch")

    monkeypatch.setattr(background, "_spawn_record", crash_after_commit)
    with pytest.raises(RuntimeError, match="before Work dispatch"):
        start_auto_research_child_work(
            background,
            episode.episode_id,
            _spawned_work_request(worker_id, instruction),
            admitted_by_operation_id=root.operation_id,
            worker_id=worker_id,
            instruction=instruction,
            instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
        )
    before = store.episode_budget_meter(episode.episode_id)

    restarted = BackgroundAgentTasks(store, stream)
    queued = store.agent_task(worker_id)
    assert queued is not None and queued.status == "queued"
    assert reconcile_committed_auto_research_dispatches(
        restarted,
    ) == [worker_id]
    wait_for_task(store, worker_id, expect="succeeded")

    assert executions == ["fresh"]
    assert store.episode_budget_meter(episode.episode_id) == before


def test_restart_dispatches_committed_child_work_resume_without_respending_b(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    stage = tmp_path / "restart-work-resume-stage"
    stage.mkdir()
    resumed: list[str] = []

    async def stream(_project_id, kind, _request, execution):
        if kind == "auto_research":
            yield _sse(AgentEvent(event="done"))
            return
        if execution.continuation == "fresh":
            execution.checkpoint_stage("", str(stage))
            yield _sse(AgentEvent(event="session", session_id="restart-work-session"))
            yield _sse(AgentEvent(event="error", text="Transient network failure."))
            return
        resumed.append(execution.continuation)
        yield _sse(AgentEvent(event="done"))

    background = BackgroundAgentTasks(store, stream)
    episode, root = start_auto_research(
        background,
        "project",
        _auto_start(),
        authorized_by=fabricated_authorizer("Researcher"),
        graph_base_head=GraphHeadRef(revision=0),
        ensure_graph_target=lambda _episode: None,
        episode_id="auto-child-resume-restart",
        operation_id="auto-child-resume-restart-root",
    )
    wait_for_task(store, root.operation_id, expect="succeeded")
    worker_id = "00000000-0000-4000-8000-000000000352"
    instruction = "Resume this exact paid child allocation after restart."
    failed = start_auto_research_child_work(
        background,
        episode.episode_id,
        _spawned_work_request(worker_id, instruction),
        admitted_by_operation_id=root.operation_id,
        worker_id=worker_id,
        instruction=instruction,
        instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
    )
    wait_for_task(store, failed.operation_id, expect="failed")
    resume_id = "00000000-0000-4000-8000-000000000353"
    store.start_agent_command(
        operation_id=root.operation_id,
        command_id="restart-work-resume-command",
        episode_id=episode.episode_id,
        verb="resume",
        idempotency_key="restart-work-resume",
        payload={
            "request_id": "1" * 32,
            "arguments": {"worker_id": worker_id},
            "planned_resume_operation_id": resume_id,
        },
    )

    def crash_after_commit(*_args, **_kwargs):
        raise RuntimeError("simulated process loss before Work Resume dispatch")

    monkeypatch.setattr(background, "_spawn_record", crash_after_commit)
    with pytest.raises(RuntimeError, match="before Work Resume dispatch"):
        resume_auto_research_child_work(
            background,
            episode.episode_id,
            worker_id,
            operation_id=resume_id,
        )
    before = store.episode_budget_meter(episode.episode_id)

    restarted = BackgroundAgentTasks(store, stream)
    queued = store.agent_task(resume_id)
    assert queued is not None and queued.status == "queued"
    assert reconcile_committed_auto_research_dispatches(
        restarted,
    ) == [resume_id]
    completed = wait_for_task(store, resume_id, expect="succeeded")

    assert completed.native_session_id == "restart-work-session"
    assert resumed == ["resume"]
    assert store.episode_budget_meter(episode.episode_id) == before


def test_spawned_child_message_wake_reuses_exact_work_session_and_spends_b(tmp_path: Path) -> None:
    store = _store(tmp_path)
    stage = tmp_path / "spawned-work-mail-stage"
    stage.mkdir()
    seen_continuations: list[str] = []

    async def stream(_project_id, kind, request, execution):
        if kind == "auto_research":
            yield _sse(AgentEvent(event="done"))
            return
        assert isinstance(request, RunRequest)
        seen_continuations.append(execution.continuation)
        if execution.continuation == "fresh":
            execution.checkpoint_stage("", str(stage))
            yield _sse(AgentEvent(event="session", session_id="spawned-work-mail-session"))
        else:
            assert execution.continuation == "message_wake"
            assert request.message is None
            assert request.session_id == "spawned-work-mail-session"
        yield _sse(AgentEvent(event="done"))

    background = BackgroundAgentTasks(store, stream)
    episode, root = start_auto_research(
        background,
        "project",
        _auto_start(),
        authorized_by=fabricated_authorizer("Researcher"),
        graph_base_head=GraphHeadRef(revision=0),
        ensure_graph_target=lambda _episode: None,
        episode_id="auto-child-mail",
        operation_id="auto-child-mail-root",
    )
    wait_for_task(store, root.operation_id, expect="succeeded")
    worker_id = "00000000-0000-4000-8000-000000000321"
    instruction = "Inspect the bounded result and report it."
    child = start_auto_research_child_work(
        background,
        episode.episode_id,
        _spawned_work_request(worker_id, instruction),
        admitted_by_operation_id=root.operation_id,
        worker_id=worker_id,
        instruction=instruction,
        instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
    )
    child = wait_for_task(store, child.operation_id, expect="succeeded")
    message_id = "00000000-0000-4000-8000-000000000322"
    store.record_auto_research_message(
        AutoResearchMessageRecord(
            message_id=message_id,
            episode_id=episode.episode_id,
            sender_role="orchestrator",
            sender_task_id=root.operation_id,
            recipient_task_id=worker_id,
            control_node_id="blk/spawned-work",
            body="Recheck the new observation.",
            created_at=store.now(),
        )
    )
    wake_id = "00000000-0000-4000-8000-000000000323"

    wake = start_auto_research_child_work_message_wake(
        background,
        episode.episode_id,
        worker_id,
        [message_id],
        operation_id=wake_id,
    )

    assert wake is not None
    wake = wait_for_task(store, wake.operation_id, expect="succeeded")
    route = store.auto_research_child_work(worker_id)
    delivered = store.auto_research_message(message_id)
    assert route is not None and route.current_operation_id == wake_id
    assert wake.parent_operation_id == child.operation_id
    assert wake.attempt == child.attempt + 1 == 2
    assert wake.native_session_id == child.native_session_id == "spawned-work-mail-session"
    assert wake.stage_root == child.stage_root == str(stage)
    assert wake.request["message"] is None
    assert delivered is not None and delivered.delivery_operation_id == wake_id
    assert store.episode_budget_meter(episode.episode_id).invocations_used == 3
    assert seen_continuations == ["fresh", "message_wake"]


def test_failed_spawned_child_exact_resume_reuses_checkpoint_and_b_allocation(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    stage = tmp_path / "spawned-work-stage"
    stage.mkdir()
    resumed_requests: list[RunRequest] = []

    async def stream(_project_id, kind, request, execution):
        if kind == "auto_research":
            yield _sse(AgentEvent(event="done"))
            return
        assert isinstance(request, RunRequest)
        if execution.continuation == "fresh":
            execution.checkpoint_stage("", str(stage))
            yield _sse(AgentEvent(event="session", session_id="spawned-work-session"))
            yield _sse(AgentEvent(event="error", text="Transient network failure."))
            return
        resumed_requests.append(request)
        yield _sse(AgentEvent(event="done"))

    background = BackgroundAgentTasks(store, stream)
    episode, root = start_auto_research(
        background,
        "project",
        _auto_start(),
        authorized_by=fabricated_authorizer("Researcher"),
        graph_base_head=GraphHeadRef(revision=0),
        ensure_graph_target=lambda _episode: None,
        episode_id="auto-child-resume",
        operation_id="auto-child-resume-root",
    )
    wait_for_task(store, root.operation_id, expect="succeeded")
    worker_id = "00000000-0000-4000-8000-000000000311"
    instruction = "Diagnose the transient runtime failure."
    failed = start_auto_research_child_work(
        background,
        episode.episode_id,
        _spawned_work_request(worker_id, instruction),
        admitted_by_operation_id=root.operation_id,
        worker_id=worker_id,
        instruction=instruction,
        instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
    )
    wait_for_task(store, failed.operation_id, expect="failed")
    before = store.episode_budget_meter(episode.episode_id).invocations_used

    outcome = resume_auto_research_child_work(
        background,
        episode.episode_id,
        worker_id,
        operation_id="00000000-0000-4000-8000-000000000312",
    )
    assert outcome.disposition == "resumed"
    assert outcome.task is not None
    resumed = wait_for_task(store, outcome.task.operation_id, expect="succeeded")

    assert resumed.parent_operation_id == worker_id
    assert resumed.native_session_id == "spawned-work-session"
    assert resumed.stage_root == str(stage)
    assert resumed_requests[0].session_id == "spawned-work-session"
    assert store.episode_budget_meter(episode.episode_id).invocations_used == before
    route = store.auto_research_child_work(worker_id)
    assert route is not None
    assert route.current_operation_id == resumed.operation_id


def test_spawned_child_resume_preserves_recovery_when_remote_stage_probe_is_uncertain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)

    async def stream(_project_id, kind, _request, execution):
        if kind == "auto_research":
            yield _sse(AgentEvent(event="done"))
            return
        execution.checkpoint_stage("worker-host", "/tmp/rcp-run.remote-work")
        yield _sse(AgentEvent(event="session", session_id="remote-work-session"))
        yield _sse(AgentEvent(event="error", text="Transient network failure."))

    background = BackgroundAgentTasks(store, stream)
    episode, root = start_auto_research(
        background,
        "project",
        _auto_start(),
        authorized_by=fabricated_authorizer("Researcher"),
        graph_base_head=GraphHeadRef(revision=0),
        ensure_graph_target=lambda _episode: None,
        episode_id="auto-child-remote-resume",
        operation_id="auto-child-remote-resume-root",
    )
    wait_for_task(store, root.operation_id, expect="succeeded")
    worker_id = "00000000-0000-4000-8000-000000000313"
    instruction = "Resume only if the exact remote workspace can be checked."
    failed = start_auto_research_child_work(
        background,
        episode.episode_id,
        _spawned_work_request(worker_id, instruction),
        admitted_by_operation_id=root.operation_id,
        worker_id=worker_id,
        instruction=instruction,
        instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
    )
    wait_for_task(store, failed.operation_id, expect="failed")
    before = store.episode_budget_meter(episode.episode_id).invocations_used
    monkeypatch.setattr(
        "rcp.background.RemoteRunStage.directory_exists",
        lambda _stage, _root: None,
    )

    with pytest.raises(OSError, match="remote infrastructure is unavailable"):
        resume_auto_research_child_work(
            background,
            episode.episode_id,
            worker_id,
            operation_id="00000000-0000-4000-8000-000000000314",
        )

    route = store.auto_research_child_work(worker_id)
    assert route is not None and route.current_operation_id == failed.operation_id
    assert store.agent_task("00000000-0000-4000-8000-000000000314") is None
    assert store.episode_budget_meter(episode.episode_id).invocations_used == before

    monkeypatch.setattr(
        "rcp.background.RemoteRunStage.directory_exists",
        lambda _stage, _root: False,
    )
    unavailable = resume_auto_research_child_work(
        background,
        episode.episode_id,
        worker_id,
        operation_id="00000000-0000-4000-8000-000000000314",
    )
    assert unavailable.disposition == "resume_unavailable"
    assert unavailable.reason == "the saved provider workspace is unavailable"


def test_unusable_spawned_child_resume_names_fresh_spawn(tmp_path: Path) -> None:
    store = _store(tmp_path)
    stage = tmp_path / "limited-work-stage"
    stage.mkdir()

    async def stream(_project_id, kind, _request, execution):
        if kind == "auto_research":
            yield _sse(AgentEvent(event="done"))
            return
        execution.checkpoint_stage("", str(stage))
        yield _sse(AgentEvent(event="session", session_id="limited-session"))
        yield _sse(AgentEvent(event="error", text="You've hit your session limit"))

    background = BackgroundAgentTasks(store, stream)
    episode, root = start_auto_research(
        background,
        "project",
        _auto_start(),
        authorized_by=fabricated_authorizer("Researcher"),
        graph_base_head=GraphHeadRef(revision=0),
        ensure_graph_target=lambda _episode: None,
        episode_id="auto-child-unavailable",
        operation_id="auto-child-unavailable-root",
    )
    wait_for_task(store, root.operation_id, expect="succeeded")
    worker_id = "00000000-0000-4000-8000-000000000321"
    instruction = "Continue the bounded probe."
    failed = start_auto_research_child_work(
        background,
        episode.episode_id,
        _spawned_work_request(worker_id, instruction),
        admitted_by_operation_id=root.operation_id,
        worker_id=worker_id,
        instruction=instruction,
        instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
    )
    wait_for_task(store, failed.operation_id, expect="failed")
    before = store.episode_budget_meter(episode.episode_id).invocations_used

    outcome = resume_auto_research_child_work(background, episode.episode_id, worker_id)

    assert outcome.disposition == "resume_unavailable"
    assert outcome.task is None
    assert outcome.replacement_command == "spawn"
    assert outcome.reason == "the saved provider session reached its limit"
    assert store.episode_budget_meter(episode.episode_id).invocations_used == before


def test_routed_worker_pause_and_stop_target_only_its_current_attempt(tmp_path: Path) -> None:
    store = _store(tmp_path)
    child_started = threading.Event()

    async def stream(_project_id, kind, _request, execution):
        if kind == "auto_research":
            yield _sse(AgentEvent(event="done"))
            return
        child_started.set()
        while not execution.control.pause_requested.is_set():
            await asyncio.sleep(0.01)
        yield _sse(AgentEvent(event="paused", text="Paused at the exact child checkpoint."))

    background = BackgroundAgentTasks(store, stream)
    episode, root = start_auto_research(
        background,
        "project",
        _auto_start(),
        authorized_by=fabricated_authorizer("Researcher"),
        graph_base_head=GraphHeadRef(revision=0),
        ensure_graph_target=lambda _episode: None,
        episode_id="auto-child-control",
        operation_id="auto-child-control-root",
    )
    wait_for_task(store, root.operation_id, expect="succeeded")
    worker_id = "00000000-0000-4000-8000-000000000325"
    instruction = "Pause this bounded diagnostic when asked."
    child = start_auto_research_child_work(
        background,
        episode.episode_id,
        _spawned_work_request(worker_id, instruction),
        admitted_by_operation_id=root.operation_id,
        worker_id=worker_id,
        instruction=instruction,
        instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
    )
    assert child_started.wait(timeout=2)

    pausing = pause_auto_research_child_work(background, episode.episode_id, worker_id)
    assert pausing.operation_id == child.operation_id
    paused = wait_for_task(store, child.operation_id, expect="paused")
    stopped_attempt = stop_auto_research_child_work(
        background,
        episode.episode_id,
        worker_id,
    )

    assert stopped_attempt.operation_id == paused.operation_id
    route = store.auto_research_child_work(worker_id)
    assert route is not None
    assert route.stop_requested_at is not None
    unavailable = resume_auto_research_child_work(background, episode.episode_id, worker_id)
    assert unavailable.disposition == "resume_unavailable"
    assert unavailable.reason == "the worker was stopped"
    assert unavailable.replacement_command == "spawn"


def test_child_experiment_start_and_exact_resume_spend_e_only_once(tmp_path: Path) -> None:
    store = _store(tmp_path)
    stage = tmp_path / "child-experiment-stage"
    stage.mkdir()
    resumed: list[RunRequest] = []

    async def stream(_project_id, kind, request, execution):
        if kind == "auto_research":
            yield _sse(AgentEvent(event="done"))
            return
        assert isinstance(request, RunRequest)
        if execution.continuation == "fresh":
            candidate = "{}"
            store.record_agent_task_contract(
                execution.operation_id,
                "experiment_episode_context_candidate",
                candidate,
                hashlib.sha256(candidate.encode()).hexdigest(),
            )
            execution.checkpoint_stage("", str(stage))
            yield _sse(AgentEvent(event="session", session_id="child-experiment-session"))
            yield _sse(AgentEvent(event="error", text="Transient network failure."))
            return
        resumed.append(request)
        yield _sse(AgentEvent(event="done"))

    background = BackgroundAgentTasks(store, stream)
    parent, root = start_auto_research(
        background,
        "project",
        _auto_start(),
        authorized_by=fabricated_authorizer("Researcher"),
        graph_base_head=GraphHeadRef(revision=0),
        ensure_graph_target=lambda _episode: None,
        episode_id="auto-child-experiment-parent",
        operation_id="auto-child-experiment-root",
    )
    wait_for_task(store, root.operation_id, expect="succeeded")
    child_episode_id = "00000000-0000-4000-8000-000000000331"
    goal = "Determine whether the repaired runtime survives the bounded probe."
    request = _child_experiment_request(child_episode_id, goal)
    now = store.now()
    route = AutoResearchChildExperimentRecord(
        child_episode_id=child_episode_id,
        auto_research_episode_id=parent.episode_id,
        project_id="project",
        control_node_id="exp/child",
        state="running",
        request={"goal": goal, "invocation_limit": 2},
        goal_sha256=hashlib.sha256(goal.encode()).hexdigest(),
        parent_operation_id=root.operation_id,
        created_at=now,
        updated_at=now,
    )

    failed = start_auto_research_child_experiment(background, route, request)
    wait_for_task(store, failed.operation_id, expect="failed")
    child_episode = store.episode(child_episode_id)
    assert child_episode is not None
    assert child_episode.graph_target == parent.graph_target
    assert child_episode.graph_base_head == parent.graph_base_head
    assert failed.graph_target == parent.graph_target
    allowance = store.auto_research_experiment_allowance(parent.episode_id)
    assert allowance.used == 1

    outcome = resume_auto_research_child_experiment(
        background,
        parent.episode_id,
        child_episode_id,
        operation_id="00000000-0000-4000-8000-000000000332",
    )
    assert outcome.disposition == "resumed"
    assert outcome.task is not None
    resumed_task = wait_for_task(store, outcome.task.operation_id, expect="succeeded")

    assert resumed_task.parent_operation_id == failed.operation_id
    assert resumed_task.graph_target == parent.graph_target
    assert resumed_task.native_session_id == "child-experiment-session"
    assert resumed[0].session_id == "child-experiment-session"
    assert store.auto_research_experiment_allowance(parent.episode_id).used == 1


def test_restart_dispatches_committed_fresh_child_experiment_without_respending_e(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    executions: list[str] = []

    async def stream(_project_id, kind, _request, execution):
        if kind != "auto_research":
            executions.append(execution.continuation)
        yield _sse(AgentEvent(event="done"))

    background = BackgroundAgentTasks(store, stream)
    parent, root = start_auto_research(
        background,
        "project",
        _auto_start(),
        authorized_by=fabricated_authorizer("Researcher"),
        graph_base_head=GraphHeadRef(revision=0),
        ensure_graph_target=lambda _episode: None,
        episode_id="auto-child-experiment-fresh-restart",
        operation_id="auto-child-experiment-fresh-restart-root",
    )
    wait_for_task(store, root.operation_id, expect="succeeded")
    child_id = "00000000-0000-4000-8000-000000000354"
    goal = "Run the exact committed child Experiment after restart."
    now = store.now()
    route = AutoResearchChildExperimentRecord(
        child_episode_id=child_id,
        auto_research_episode_id=parent.episode_id,
        project_id="project",
        control_node_id="exp/child",
        state="running",
        request={"goal": goal, "invocation_limit": 2},
        goal_sha256=hashlib.sha256(goal.encode()).hexdigest(),
        parent_operation_id=root.operation_id,
        created_at=now,
        updated_at=now,
    )

    def crash_after_commit(*_args, **_kwargs):
        raise RuntimeError("simulated process loss before Experiment dispatch")

    monkeypatch.setattr(background, "_spawn_record", crash_after_commit)
    with pytest.raises(RuntimeError, match="before Experiment dispatch"):
        start_auto_research_child_experiment(
            background,
            route,
            _child_experiment_request(child_id, goal),
        )
    child = store.episode(child_id)
    assert child is not None and child.root_operation_id is not None
    operation_id = child.root_operation_id
    before = store.auto_research_experiment_allowance(parent.episode_id)

    restarted = BackgroundAgentTasks(store, stream)
    queued = store.agent_task(operation_id)
    assert queued is not None and queued.status == "queued"
    assert reconcile_committed_auto_research_dispatches(
        restarted,
    ) == [operation_id]
    wait_for_task(store, operation_id, expect="succeeded")

    assert executions == ["fresh"]
    assert store.auto_research_experiment_allowance(parent.episode_id) == before


def test_restart_dispatches_committed_child_experiment_resume_without_respending_e(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    stage = tmp_path / "restart-experiment-resume-stage"
    stage.mkdir()
    resumed: list[str] = []

    async def stream(_project_id, kind, _request, execution):
        if kind == "auto_research":
            yield _sse(AgentEvent(event="done"))
            return
        if execution.continuation == "fresh":
            candidate = "{}"
            store.record_agent_task_contract(
                execution.operation_id,
                "experiment_episode_context_candidate",
                candidate,
                hashlib.sha256(candidate.encode()).hexdigest(),
            )
            execution.checkpoint_stage("", str(stage))
            yield _sse(AgentEvent(event="session", session_id="restart-experiment-session"))
            yield _sse(AgentEvent(event="error", text="Transient network failure."))
            return
        resumed.append(execution.continuation)
        yield _sse(AgentEvent(event="done"))

    background = BackgroundAgentTasks(store, stream)
    parent, root = start_auto_research(
        background,
        "project",
        _auto_start(),
        authorized_by=fabricated_authorizer("Researcher"),
        graph_base_head=GraphHeadRef(revision=0),
        ensure_graph_target=lambda _episode: None,
        episode_id="auto-child-experiment-resume-restart",
        operation_id="auto-child-experiment-resume-restart-root",
    )
    wait_for_task(store, root.operation_id, expect="succeeded")
    child_id = "00000000-0000-4000-8000-000000000355"
    goal = "Resume the exact child Experiment allocation after restart."
    now = store.now()
    route = AutoResearchChildExperimentRecord(
        child_episode_id=child_id,
        auto_research_episode_id=parent.episode_id,
        project_id="project",
        control_node_id="exp/child",
        state="running",
        request={"goal": goal, "invocation_limit": 2},
        goal_sha256=hashlib.sha256(goal.encode()).hexdigest(),
        parent_operation_id=root.operation_id,
        created_at=now,
        updated_at=now,
    )
    failed = start_auto_research_child_experiment(
        background,
        route,
        _child_experiment_request(child_id, goal),
    )
    wait_for_task(store, failed.operation_id, expect="failed")
    resume_id = "00000000-0000-4000-8000-000000000356"
    store.start_agent_command(
        operation_id=root.operation_id,
        command_id="restart-experiment-resume-command",
        episode_id=parent.episode_id,
        verb="episode",
        idempotency_key="restart-experiment-resume",
        payload={
            "request_id": "2" * 32,
            "arguments": {"action": "resume", "episode_id": child_id},
            "planned_episode_effect_id": resume_id,
        },
    )

    def crash_after_commit(*_args, **_kwargs):
        raise RuntimeError("simulated process loss before Experiment Resume dispatch")

    monkeypatch.setattr(background, "_spawn_record", crash_after_commit)
    with pytest.raises(RuntimeError, match="before Experiment Resume dispatch"):
        resume_auto_research_child_experiment(
            background,
            parent.episode_id,
            child_id,
            operation_id=resume_id,
        )
    before = store.auto_research_experiment_allowance(parent.episode_id)

    restarted = BackgroundAgentTasks(store, stream)
    queued = store.agent_task(resume_id)
    assert queued is not None and queued.status == "queued"
    assert reconcile_committed_auto_research_dispatches(
        restarted,
    ) == [resume_id]
    completed = wait_for_task(store, resume_id, expect="succeeded")

    assert completed.native_session_id == "restart-experiment-session"
    assert resumed == ["resume"]
    assert store.auto_research_experiment_allowance(parent.episode_id) == before


def test_restart_redispatches_committed_child_experiment_graph_repair_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    stage = tmp_path / "restart-experiment-graph-repair-stage"
    stage.mkdir()
    continuations: list[str] = []

    async def stream(_project_id, kind, request, execution):
        if kind == "auto_research":
            yield _sse(AgentEvent(event="done"))
            return
        continuations.append(execution.continuation)
        if execution.continuation == "fresh":
            execution.checkpoint_stage("", str(stage))
            yield _sse(AgentEvent(event="session", session_id="graph-repair-session"))
            yield _sse(
                AgentEvent(
                    event="message",
                    text=json.dumps(
                        {
                            "graph_update": {
                                "status": "rejected",
                                "validation_messages": ["Patch requires correction."],
                                "repairable": True,
                            }
                        }
                    ),
                )
            )
        else:
            assert execution.continuation == "graph_repair"
            assert isinstance(request, RunRequest) and request.message is None
        yield _sse(AgentEvent(event="done"))

    background = BackgroundAgentTasks(store, stream)
    parent, root = start_auto_research(
        background,
        "project",
        _auto_start(),
        authorized_by=fabricated_authorizer("Researcher"),
        graph_base_head=GraphHeadRef(revision=0),
        ensure_graph_target=lambda _episode: None,
        episode_id="auto-child-experiment-graph-repair-restart",
        operation_id="auto-child-experiment-graph-repair-restart-root",
    )
    wait_for_task(store, root.operation_id, expect="succeeded")
    child_id = "00000000-0000-4000-8000-000000000358"
    goal = "Do not turn a graph repair into a full CLI Resume."
    child_request = _child_experiment_request(child_id, goal)
    now = store.now()
    route = AutoResearchChildExperimentRecord(
        child_episode_id=child_id,
        auto_research_episode_id=parent.episode_id,
        project_id="project",
        control_node_id="exp/child",
        state="running",
        request={"goal": goal, "invocation_limit": 2},
        goal_sha256=hashlib.sha256(goal.encode()).hexdigest(),
        parent_operation_id=root.operation_id,
        created_at=now,
        updated_at=now,
    )
    rejected = start_auto_research_child_experiment(background, route, child_request)
    rejected = wait_for_task(store, rejected.operation_id, expect="succeeded")
    assert rejected.native_session_id == "graph-repair-session"
    assert rejected.result is not None
    assert rejected.result["graph_update"]["repairable"] is True
    allowance_before = store.auto_research_experiment_allowance(parent.episode_id)

    def crash_after_commit(*_args, **_kwargs):
        raise RuntimeError("simulated process loss before graph-repair dispatch")

    monkeypatch.setattr(background, "_spawn_record", crash_after_commit)
    with pytest.raises(RuntimeError, match="before graph-repair dispatch"):
        background.repair_graph_update(rejected.operation_id)
    repair_tasks = [
        task
        for task in store.episode_tasks(child_id)
        if task.parent_operation_id == rejected.operation_id
    ]
    assert len(repair_tasks) == 1
    repair_id = repair_tasks[0].operation_id
    assert store.agent_task_continuation_cause(repair_id) == "graph_repair"

    restarted = BackgroundAgentTasks(store, stream)
    queued = store.agent_task(repair_id)
    assert queued is not None and queued.status == "queued"
    assert reconcile_committed_auto_research_dispatches(
        restarted,
    ) == [repair_id]
    wait_for_task(store, repair_id, expect="succeeded")

    assert continuations == ["fresh", "graph_repair"]
    assert store.auto_research_experiment_allowance(parent.episode_id) == allowance_before
    assert [task.operation_id for task in store.episode_tasks(child_id)].count(repair_id) == 1
    created = [
        receipt
        for receipt in store.agent_task_receipts(repair_id)
        if receipt.category == "operation_created"
    ]
    assert len(created) == 1
    assert created[0].payload["continuation_cause"] == "graph_repair"


def test_child_experiment_resume_preserves_recovery_when_remote_stage_probe_is_uncertain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)

    async def stream(_project_id, kind, _request, execution):
        if kind == "auto_research":
            yield _sse(AgentEvent(event="done"))
            return
        candidate = "{}"
        store.record_agent_task_contract(
            execution.operation_id,
            "experiment_episode_context_candidate",
            candidate,
            hashlib.sha256(candidate.encode()).hexdigest(),
        )
        execution.checkpoint_stage("experiment-host", "/tmp/rcp-run.remote-experiment")
        yield _sse(AgentEvent(event="session", session_id="remote-experiment-session"))
        yield _sse(AgentEvent(event="error", text="Transient network failure."))

    background = BackgroundAgentTasks(store, stream)
    parent, root = start_auto_research(
        background,
        "project",
        _auto_start(),
        authorized_by=fabricated_authorizer("Researcher"),
        graph_base_head=GraphHeadRef(revision=0),
        ensure_graph_target=lambda _episode: None,
        episode_id="auto-child-remote-experiment-parent",
        operation_id="auto-child-remote-experiment-root",
    )
    wait_for_task(store, root.operation_id, expect="succeeded")
    child_episode_id = "00000000-0000-4000-8000-000000000333"
    goal = "Resume only if the exact remote Experiment workspace can be checked."
    now = store.now()
    route = AutoResearchChildExperimentRecord(
        child_episode_id=child_episode_id,
        auto_research_episode_id=parent.episode_id,
        project_id="project",
        control_node_id="exp/child",
        state="running",
        request={"goal": goal, "invocation_limit": 2},
        goal_sha256=hashlib.sha256(goal.encode()).hexdigest(),
        parent_operation_id=root.operation_id,
        created_at=now,
        updated_at=now,
    )
    failed = start_auto_research_child_experiment(
        background,
        route,
        _child_experiment_request(child_episode_id, goal),
    )
    wait_for_task(store, failed.operation_id, expect="failed")
    allowance_before = store.auto_research_experiment_allowance(parent.episode_id)
    monkeypatch.setattr(
        "rcp.background.RemoteRunStage.directory_exists",
        lambda _stage, _root: None,
    )

    with pytest.raises(OSError, match="remote infrastructure is unavailable"):
        resume_auto_research_child_experiment(
            background,
            parent.episode_id,
            child_episode_id,
            operation_id="00000000-0000-4000-8000-000000000334",
        )

    tasks = store.episode_tasks(child_episode_id)
    assert tasks[-1].operation_id == failed.operation_id
    assert store.agent_task("00000000-0000-4000-8000-000000000334") is None
    assert store.auto_research_experiment_allowance(parent.episode_id) == allowance_before

    monkeypatch.setattr(
        "rcp.background.RemoteRunStage.directory_exists",
        lambda _stage, _root: False,
    )
    unavailable = resume_auto_research_child_experiment(
        background,
        parent.episode_id,
        child_episode_id,
        operation_id="00000000-0000-4000-8000-000000000334",
    )
    assert unavailable.disposition == "resume_unavailable"
    assert unavailable.reason == "the saved provider workspace is unavailable"


def test_task_result_keeps_ordered_graph_updates_and_latest_compatibility_projection(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    in_turn_updates = [
        {
            "status": "applied",
            "applied_revision": revision,
            "change_summary": [f"Applied in-turn revision {revision}."],
        }
        for revision in (2, 3)
    ]
    final_update = {
        "status": "applied",
        "applied_revision": 4,
        "change_summary": ["Applied the final unconsumed Patch."],
    }

    async def stream(_project_id, _kind, _request, _execution):
        yield _sse(
            AgentEvent(
                event="message",
                text=json.dumps(
                    {
                        "applied_revision": 4,
                        "graph_update": final_update,
                        "graph_updates": in_turn_updates,
                    }
                ),
            )
        )
        yield _sse(AgentEvent(event="done"))

    tasks = BackgroundAgentTasks(store, stream)
    task = tasks.start(
        "project",
        "node_chat",
        RunRequest(
            provider="codex",
            model="",
            reasoning="medium",
            run_on="laptop",
            run_truth_scope=["repo"],
            chat_id="graph-update-chat",
            chat_scope="node",
            node_id="exp/result-compatibility",
            message="Apply the bounded updates.",
            mode="work",
        ),
        operation_id="graph-update-task",
        authorized_by=fabricated_authorizer("Researcher"),
    )
    completed = wait_for_task(store, task.operation_id, expect="succeeded")

    assert completed.applied_revision == 4
    assert completed.result is not None
    assert completed.result["graph_update"]["applied_revision"] == 4
    assert [update["applied_revision"] for update in completed.result["graph_updates"]] == [2, 3]
    assert completed.result["messages"] == []


def test_auto_research_clean_orchestrator_retry_keeps_paid_allocation(tmp_path: Path) -> None:
    store = _store(tmp_path)
    stage = tmp_path / "replacement-stage"
    stage.mkdir()

    async def stream(_project_id, _kind, request, execution):
        if execution.continuation == "fresh":
            yield _sse(AgentEvent(event="error", text="Network connection failed."))
            return
        assert execution.continuation == "retry"
        assert request.session_id is None
        execution.checkpoint_stage("", str(stage))
        yield _sse(AgentEvent(event="session", session_id="replacement-session"))
        yield _sse(AgentEvent(event="done"))

    tasks = BackgroundAgentTasks(store, stream)
    episode, root = start_auto_research(
        tasks,
        "project",
        _auto_start(),
        authorized_by=fabricated_authorizer("Researcher"),
        graph_base_head=GraphHeadRef(revision=0),
        ensure_graph_target=lambda _episode: None,
        episode_id="auto-episode",
        operation_id="auto-root",
    )
    root = wait_for_task(store, root.operation_id, expect="failed")

    retried = wait_for_task(store, tasks.retry(root.operation_id).operation_id, expect="succeeded")

    assert retried.parent_operation_id == root.operation_id
    assert retried.episode_id == episode.episode_id
    assert retried.native_session_id == "replacement-session"
    assert retried.stage_root == str(stage)
    assert store.episode_budget_meter(episode.episode_id).invocations_used == 1
    clean = next(
        receipt
        for receipt in store.agent_task_receipts(retried.operation_id)
        if receipt.category == "auto_research_orchestrator_clean_retry"
    )
    assert clean.payload["same_allocation"] is True
    assert clean.payload["classification"] == "checkpoint_missing"


def test_auto_research_stop_skips_report_generation(tmp_path: Path) -> None:
    store = _store(tmp_path)

    async def stream(_project_id, _kind, _request, _execution):
        yield _sse(AgentEvent(event="done"))

    tasks = BackgroundAgentTasks(store, stream)
    episode, root = start_auto_research(
        tasks,
        "project",
        _auto_start(),
        authorized_by=fabricated_authorizer("Researcher"),
        graph_base_head=GraphHeadRef(revision=0),
        ensure_graph_target=lambda _episode: None,
        episode_id="auto-episode",
        operation_id="auto-root",
    )
    wait_for_task(store, root.operation_id, expect="succeeded")

    stopped = stop_auto_research(tasks, episode.episode_id)

    assert stopped.status == "stopped"
    assert stopped.ending == "stopped"
    assert stopped.wrapup_state == "skipped"
    assert store.episode_report(stopped.episode_id) is None
    assert all(
        task.kind != "episode_report"
        for task in store.episode_tasks(stopped.episode_id, include_hidden=True)
    )


def test_over_ceiling_admission_does_not_fence_an_active_paid_turn(tmp_path: Path) -> None:
    store = _store(tmp_path)
    started = threading.Event()
    release = threading.Event()

    async def stream(_project_id, _kind, _request, _execution):
        started.set()
        while not release.is_set():
            await asyncio.sleep(0.01)
        yield _sse(AgentEvent(event="done"))

    tasks = BackgroundAgentTasks(store, stream)
    episode, root = start_auto_research(
        tasks,
        "project",
        _auto_start(invocation_ceiling=1),
        authorized_by=fabricated_authorizer("Researcher"),
        graph_base_head=GraphHeadRef(revision=0),
        ensure_graph_target=lambda _episode: None,
        episode_id="auto-episode",
        operation_id="auto-root",
    )
    assert started.wait(timeout=2)
    root_request = AutoResearchRunRequest.model_validate(root.request)
    worker_request = root_request.model_copy(
        update={
            "role": "worker",
            "actor_operation_id": None,
            "instruction": "Check the bounded claim.",
            "control_node_id": "exp/check",
        }
    )

    try:
        with pytest.raises(EpisodeInvocationCeilingReached):
            start_auto_research_turn(
                tasks,
                episode.episode_id,
                worker_request,
                parent_operation_id=root.operation_id,
                operation_id="over-ceiling-worker",
            )
        current = store.episode(episode.episode_id)
        assert current is not None
        assert current.status == "running"
        assert current.ending is None
    finally:
        release.set()
        wait_for_task(store, root.operation_id, expect="succeeded")


def test_experiment_root_and_recovery_use_atomic_episode_admission(tmp_path: Path) -> None:
    store = _store(tmp_path)
    stage = tmp_path / "experiment-recovery-stage"
    stage.mkdir()

    async def stream(_project_id, _kind, request, execution):
        execution.checkpoint_stage("", str(stage))
        if execution.continuation == "fresh":
            candidate = "{}"
            store.record_agent_task_contract(
                execution.operation_id,
                "experiment_episode_context_candidate",
                candidate,
                hashlib.sha256(candidate.encode()).hexdigest(),
            )
        yield _sse(AgentEvent(event="session", session_id="experiment-session"))
        if execution.continuation == "fresh":
            yield _sse(AgentEvent(event="error", text="Transient provider failure."))
            return
        yield _sse(AgentEvent(event="done"))

    tasks = BackgroundAgentTasks(store, stream)
    root = tasks.start(
        "project",
        "node_chat",
        _experiment_request(),
        operation_id="experiment-root",
        authorized_by=fabricated_authorizer("Researcher"),
    )
    root = wait_for_task(store, root.operation_id, expect="failed")

    episode = store.episode(_EXPERIMENT_EPISODE_ID)
    assert episode is not None
    assert episode.mode == "experiment_loop"
    assert episode.root_operation_id == root.operation_id
    assert root.episode_id == episode.episode_id
    assert episode.invocations_used == 1

    recovered = wait_for_task(
        store, tasks.retry(root.operation_id).operation_id, expect="succeeded"
    )
    assert recovered.parent_operation_id == root.operation_id
    assert recovered.episode_id == episode.episode_id
    assert store.episode_budget_meter(episode.episode_id).invocations_used == 1


def test_experiment_watcher_wake_uses_atomic_episode_invocation(tmp_path: Path) -> None:
    store = _store(tmp_path)
    stage = tmp_path / "experiment-stage"
    stage.mkdir()
    authorizer = fabricated_authorizer("Researcher")

    async def stream(_project_id, _kind, request, execution):
        if request.trigger == "experiment_run":
            execution.checkpoint_stage("", str(stage))
        yield _sse(AgentEvent(event="session", session_id="experiment-session"))
        yield _sse(AgentEvent(event="done"))

    tasks = BackgroundAgentTasks(store, stream)
    root = wait_for_task(
        store,
        tasks.start(
            "project",
            "node_chat",
            _experiment_request(),
            operation_id="experiment-root",
            authorized_by=authorizer,
        ).operation_id,
        expect="succeeded",
    )
    store.commit_experiment_episode_turn(
        episode_id=_EXPERIMENT_EPISODE_ID,
        project_id="project",
        control_node_id=_EXPERIMENT_ID,
        provider="codex",
        execution_machine="laptop",
        execution_host="",
        native_session_id="experiment-session",
        stage_host=None,
        stage_root=str(stage),
        chat_id="experiment-chat",
        operation_id=root.operation_id,
        invocation=1,
        graph_result="applied",
        watcher_ids=[],
        context_baseline={},
    )
    now = store.now()
    watcher = WatcherRecord(
        watcher_id="completed-watcher",
        project_id="project",
        origin_operation_id=root.operation_id,
        origin_task_kind="node_chat",
        chat_id="experiment-chat",
        node_id=_EXPERIMENT_ID,
        execution_host="",
        check_command="true",
        log_path="/tmp/completed-watcher.log",
        cwd="/tmp",
        continuation=WatcherContinuation(
            provider="codex",
            model="",
            reasoning="medium",
            run_on="laptop",
            run_truth_scope=["repo"],
            patch_kind="experiment_loop",
            control_node_id=_EXPERIMENT_ID,
            control_revision=1,
            control_episode_id=_EXPERIMENT_EPISODE_ID,
            control_invocation=1,
            control_invocation_ceiling=3,
            control_decision_bundle=[],
            control_completion_criteria=["The bounded comparison is analyzed."],
        ),
        status="active",
        created_at=now,
    )
    store.create_watchers([watcher])
    store.record_watcher_check(
        watcher.watcher_id,
        status="completed",
        exit_code=0,
        error=None,
    )

    wake = start_watcher_notification(
        tasks,
        "project",
        "node_chat",
        _experiment_request(
            trigger="watcher",
            invocation=2,
            watcher_ids=[watcher.watcher_id],
            session_id="experiment-session",
        ),
        [watcher.watcher_id],
        authorized_by=authorizer,
        episode_stage_root=str(stage),
    )
    assert wake is not None
    wake = wait_for_task(store, wake.operation_id, expect="succeeded")

    assert wake.episode_id == _EXPERIMENT_EPISODE_ID
    assert store.episode_budget_meter(_EXPERIMENT_EPISODE_ID).invocations_used == 2
    claimed = store.watcher(watcher.watcher_id)
    assert claimed is not None
    assert claimed.notified is True
    assert claimed.notification_operation_id == wake.operation_id


def test_restart_dispatches_committed_child_experiment_watcher_wake_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    stage = tmp_path / "restart-child-experiment-watcher-stage"
    stage.mkdir()
    authorizer = fabricated_authorizer("Researcher")
    watcher_executions: list[str] = []

    async def stream(_project_id, kind, request, execution):
        if kind == "auto_research":
            yield _sse(AgentEvent(event="done"))
            return
        assert isinstance(request, RunRequest)
        if execution.continuation == "fresh":
            execution.checkpoint_stage("", str(stage))
            yield _sse(AgentEvent(event="session", session_id="child-watcher-session"))
        else:
            watcher_executions.append(execution.continuation)
        yield _sse(AgentEvent(event="done"))

    background = BackgroundAgentTasks(store, stream)
    parent, root = start_auto_research(
        background,
        "project",
        _auto_start(),
        authorized_by=authorizer,
        graph_base_head=GraphHeadRef(revision=0),
        ensure_graph_target=lambda _episode: None,
        episode_id="auto-child-experiment-watcher-restart",
        operation_id="auto-child-experiment-watcher-restart-root",
    )
    wait_for_task(store, root.operation_id, expect="succeeded")
    child_id = "00000000-0000-4000-8000-000000000357"
    goal = "Continue the child Experiment when its watcher completes."
    child_request = _child_experiment_request(child_id, goal)
    now = store.now()
    route = AutoResearchChildExperimentRecord(
        child_episode_id=child_id,
        auto_research_episode_id=parent.episode_id,
        project_id="project",
        control_node_id="exp/child",
        state="running",
        request={"goal": goal, "invocation_limit": 2},
        goal_sha256=hashlib.sha256(goal.encode()).hexdigest(),
        parent_operation_id=root.operation_id,
        created_at=now,
        updated_at=now,
    )
    child_root = start_auto_research_child_experiment(background, route, child_request)
    child_root = wait_for_task(store, child_root.operation_id, expect="succeeded")
    store.commit_experiment_episode_turn(
        episode_id=child_id,
        project_id="project",
        control_node_id="exp/child",
        provider="codex",
        execution_machine="laptop",
        execution_host="",
        native_session_id="child-watcher-session",
        stage_host=None,
        stage_root=str(stage),
        chat_id=child_request.chat_id,
        operation_id=child_root.operation_id,
        invocation=1,
        graph_result="applied",
        watcher_ids=[],
        context_baseline={},
    )
    watcher = WatcherRecord(
        watcher_id="child-experiment-completed-watcher",
        project_id="project",
        origin_operation_id=child_root.operation_id,
        origin_task_kind="node_chat",
        chat_id=child_request.chat_id,
        node_id="exp/child",
        episode_id=child_id,
        graph_target=parent.graph_target,
        execution_host="",
        check_command="true",
        log_path="/tmp/child-experiment-completed-watcher.log",
        cwd="/tmp",
        continuation=WatcherContinuation(
            provider="codex",
            model="",
            reasoning="medium",
            run_on="laptop",
            run_truth_scope=["repo"],
            patch_kind="experiment_loop",
            control_node_id="exp/child",
            control_revision=1,
            control_episode_id=child_id,
            control_invocation=1,
            control_invocation_ceiling=2,
            control_decision_bundle=[],
            control_completion_criteria=["The bounded child comparison is analyzed."],
        ),
        status="active",
        created_at=store.now(),
    )
    store.create_watchers([watcher])
    store.record_watcher_check(watcher.watcher_id, status="completed", exit_code=0, error=None)
    wake_request = child_request.model_copy(
        update={
            "trigger": "watcher",
            "control_invocation": 2,
            "watcher_ids": [watcher.watcher_id],
            "session_id": "child-watcher-session",
        }
    )
    real_thread_start = threading.Thread.start

    def fail_before_thread_start(_thread):
        raise RuntimeError("simulated process loss before watcher thread start")

    monkeypatch.setattr(threading.Thread, "start", fail_before_thread_start)
    with pytest.raises(RuntimeError, match="before watcher thread start"):
        start_watcher_notification(
            background,
            "project",
            "node_chat",
            wake_request,
            [watcher.watcher_id],
            authorized_by=authorizer,
            episode_stage_root=str(stage),
        )
    monkeypatch.setattr(threading.Thread, "start", real_thread_start)
    claimed = store.watcher(watcher.watcher_id)
    assert claimed is not None and claimed.notification_operation_id is not None
    operation_id = claimed.notification_operation_id
    allowance_before = store.auto_research_experiment_allowance(parent.episode_id)
    meter_before = store.episode_budget_meter(child_id)

    restarted = BackgroundAgentTasks(store, stream)
    queued = store.agent_task(operation_id)
    assert queued is not None and queued.status == "queued"
    assert reconcile_committed_auto_research_dispatches(
        restarted,
    ) == [operation_id]
    completed = wait_for_task(store, operation_id, expect="succeeded")

    claimed_after = store.watcher(watcher.watcher_id)
    assert completed.native_session_id == "child-watcher-session"
    assert completed.graph_target == parent.graph_target
    assert claimed_after is not None and claimed_after.graph_target == parent.graph_target
    assert watcher_executions == ["watcher_wake"]
    assert claimed_after.notification_operation_id == operation_id
    assert store.auto_research_experiment_allowance(parent.episode_id) == allowance_before
    assert store.episode_budget_meter(child_id) == meter_before


def _report_allocation(store: AppStore, tmp_path: Path) -> AgentTaskRecord:
    now = store.now()
    store.create_episode(
        EpisodeRecord(
            episode_id="report-episode",
            project_id="project",
            mode="experiment_loop",
            control_node_id="exp/report",
            status="queued",
            invocation_ceiling=1,
            authorized_by=fabricated_authorizer("Researcher"),
            created_at=now,
            updated_at=now,
        )
    )
    stage = tmp_path / "report-stage"
    stage.mkdir()
    operational = AgentTaskRecord(
        operation_id="operational",
        project_id="project",
        episode_id="report-episode",
        kind="node_chat",
        status="queued",
        request={
            "provider": "codex",
            "model": "",
            "reasoning": "medium",
            "run_on": "laptop",
        },
        created_at=now,
        updated_at=now,
        status_message="Queued",
        native_session_id="report-session",
        stage_root=str(stage),
    )
    store.allocate_episode_invocation("report-episode", operational)
    store.complete_agent_task(operational.operation_id, applied_revision=None, result={})
    admission = begin_episode_report_wrapup(
        store,
        EpisodeWrapupSpec(
            episode_id="report-episode",
            ending="completed",
            partial=False,
            continuation_operation_id=operational.operation_id,
            receipt={"observations": ["One bounded result."]},
        ),
    )
    assert admission.task is not None
    return admission.task


@pytest.mark.parametrize("prior_status", ["interrupted", "paused"])
def test_interrupted_hidden_report_restarts_once_and_runner_owns_success(
    tmp_path: Path,
    prior_status: str,
) -> None:
    store = _store(tmp_path)
    hidden = _report_allocation(store, tmp_path)
    store.mark_agent_task_running(hidden.operation_id)
    store.bind_agent_task_write_scope(
        hidden.operation_id,
        project_id="project",
        stage_host="",
        stage_root=str(tmp_path / "report-stage"),
        fingerprint="a" * 64,
    )
    store.record_agent_task_receipt(
        hidden.operation_id,
        "operation_dispatch_attempt",
        {"dispatch_attempt_id": "previous-report-dispatch"},
        tier="diagnostic",
    )
    store.record_agent_task_receipt(
        hidden.operation_id,
        "operation_dispatch_started",
        {"dispatch_attempt_id": "previous-report-dispatch"},
        tier="diagnostic",
    )
    if prior_status == "interrupted":
        store.interrupt_active_agent_tasks()
    else:
        store.pause_agent_task(hidden.operation_id, detail="Paused for shutdown")
    store.record_agent_task_receipt(
        hidden.operation_id,
        "operation_created",
        {
            "kind": "episode_report",
            "attempt": 1,
            "has_parent": True,
            "continuation_cause": "episode_report",
            "resumed": True,
        },
    )
    generic_settlements: list[str] = []

    async def stream(_project_id, kind, request, execution):
        assert kind == "episode_report"
        assert isinstance(request, EpisodeReportRunRequest)
        attempt = store.allocate_episode_report_attempt(request.episode_id)
        store.mark_episode_report_attempt_running(attempt.attempt_id)
        html = "<html><body><figure>Evidence map</figure></body></html>"
        store.finish_episode_report_ready(
            attempt.attempt_id,
            EpisodeReportRecord(
                report_id="report",
                episode_id=request.episode_id,
                attempt_id=attempt.attempt_id,
                allocation_operation_id=execution.operation_id,
                ending="completed",
                sha256=hashlib.sha256(html.encode()).hexdigest(),
                html=html,
                created_at=store.now(),
            ),
        )
        yield _sse(AgentEvent(event="done"))

    tasks = BackgroundAgentTasks(
        store,
        stream,
        on_task_settled=lambda _project, _kind, _request, execution: generic_settlements.append(
            execution.operation_id
        ),
    )
    tasks.recover_at_startup()
    finished = wait_for_task(store, hidden.operation_id, expect="succeeded")

    assert finished.operation_id == hidden.operation_id
    assert store.episode("report-episode").wrapup_state == "ready"  # type: ignore[union-attr]
    assert store.episode_report("report-episode") is not None
    assert store.episode_tasks("report-episode") == [store.agent_task("operational")]
    assert [task.kind for task in store.episode_tasks("report-episode", include_hidden=True)] == [
        "node_chat",
        "episode_report",
    ]
    receipts = store.agent_task_receipts(hidden.operation_id)
    assert sum(item.category == "operation_created" for item in receipts) == 1
    assert not any(item.category == "operation_completed" for item in receipts)
    assert generic_settlements == []
    assert start_episode_report(tasks, "report-episode") is None
    with pytest.raises(ValueError, match="no Retry control"):
        tasks.retry(hidden.operation_id)
    with pytest.raises(ValueError, match="no Resume control"):
        tasks.resume(hidden.operation_id)
    with pytest.raises(ValueError, match="no manual Pause control"):
        tasks.pause(hidden.operation_id)


def test_report_runner_terminal_error_is_not_generically_retried_or_resettled(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    hidden = _report_allocation(store, tmp_path)
    generic_settlements: list[str] = []
    entered = threading.Event()
    release = threading.Event()

    async def stream(_project_id, kind, request, _execution):
        assert kind == "episode_report"
        assert isinstance(request, EpisodeReportRunRequest)
        entered.set()
        while not release.is_set():
            await asyncio.sleep(0.01)
        store.fail_episode_report_allocation_unlaunchable(
            request.episode_id,
            "The exact report continuation is unavailable.",
        )
        yield _sse(AgentEvent(event="error", text="Report runner already settled the error."))

    tasks = BackgroundAgentTasks(
        store,
        stream,
        on_task_settled=lambda _project, _kind, _request, execution: generic_settlements.append(
            execution.operation_id
        ),
    )
    tasks.recover_at_startup()
    assert entered.wait(timeout=2)
    duplicate = start_episode_report(tasks, "report-episode")
    assert duplicate is not None and duplicate.operation_id == hidden.operation_id
    release.set()
    failed = wait_for_task(store, hidden.operation_id, expect="failed")

    episode = store.episode("report-episode")
    assert episode is not None
    assert episode.status == "completed"
    assert episode.wrapup_state == "failed"
    assert episode.wrapup_error == "The exact report continuation is unavailable."
    assert failed.error == episode.wrapup_error
    assert store.episode_report(episode.episode_id) is None
    assert not any(
        item.category == "operation_failed"
        for item in store.agent_task_receipts(hidden.operation_id)
    )
    assert (
        sum(
            item.category == "operation_created"
            for item in store.agent_task_receipts(hidden.operation_id)
        )
        == 1
    )
    assert generic_settlements == []
    with pytest.raises(ValueError, match="no Retry control"):
        tasks.retry(hidden.operation_id)


def test_report_request_decode_never_accepts_an_auto_research_task_shape(tmp_path: Path) -> None:
    store = _store(tmp_path)
    hidden = _report_allocation(store, tmp_path)
    decoded = BackgroundAgentTasks._request_from_record(hidden)

    assert isinstance(decoded, EpisodeReportRunRequest)
    assert decoded.episode_id == "report-episode"
    assert not hasattr(decoded, "role")
    assert not hasattr(decoded, "campaign_id")


def test_legacy_experiment_episode_without_authorizer_names_the_fresh_run(tmp_path: Path) -> None:
    """A recorded episode authorizer is required, so say what the human can do.

    Regression: an Experiment episode written before the authorizer snapshot
    existed refused Retry with "A patch-capable agent task requires a human
    authorizer snapshot." That is true and useless. The episode's own human is
    the authority for every turn in it, so a current human cannot stand in --
    but pressing Run starts a fresh episode, and the message must say so.
    """

    store = _store(tmp_path)
    stage = tmp_path / "legacy-experiment-stage"
    stage.mkdir()

    async def stream(_project_id, _kind, request, execution):
        execution.checkpoint_stage("", str(stage))
        candidate = "{}"
        store.record_agent_task_contract(
            execution.operation_id,
            "experiment_episode_context_candidate",
            candidate,
            hashlib.sha256(candidate.encode()).hexdigest(),
        )
        yield _sse(AgentEvent(event="session", session_id="experiment-session"))
        yield _sse(AgentEvent(event="error", text="Transient provider failure."))

    tasks = BackgroundAgentTasks(store, stream)
    root = tasks.start(
        "project",
        "node_chat",
        _experiment_request(),
        operation_id="legacy-experiment-root",
        authorized_by=fabricated_authorizer("Researcher"),
    )
    wait_for_task(store, root.operation_id, expect="failed")

    # Age the episode back to before the authorizer snapshot was recorded.
    with store.connection() as connection:
        connection.execute(
            "UPDATE episodes SET authorized_space_id = NULL, authorized_user_id = NULL, "
            "authorized_display_name = NULL WHERE episode_id = ?",
            (_EXPERIMENT_EPISODE_ID,),
        )
    assert store.episode(_EXPERIMENT_EPISODE_ID).authorized_by is None

    with pytest.raises(ValueError) as refusal:
        tasks.retry(root.operation_id, authorized_by=fabricated_authorizer("Someone else"))
    message = str(refusal.value)
    assert "predates the recorded human authorizer" in message
    assert "Press Run on the Experiment to start a fresh episode." in message
    assert store.agent_task(root.operation_id).status == "failed"
