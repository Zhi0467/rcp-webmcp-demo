from __future__ import annotations

import asyncio
import hashlib
import threading
import time

import pytest

from rcp.agents import AgentEvent, AgentProcessControl
from rcp.agents.command_protocol import MessageArguments, MessageCommandRequest, WatchGraphArguments
from rcp.background import AgentTaskExecution, BackgroundAgentTasks
from rcp.core.models import Blocker, GraphState
from rcp.core.transition_models import GraphHeadRef
from rcp.runs.auto_research import (
    AutoResearchCommandContext,
    AutoResearchRunRequest,
    AutoResearchStartRequest,
)
from rcp.runs.auto_research_admission import (
    start_auto_research,
    start_auto_research_child_work,
    start_auto_research_turn,
)
from rcp.runs.auto_research_delivery import (
    arm_auto_research_graph_condition,
    deliver_auto_research_watcher_group,
    deliver_pending_auto_research_lifecycle,
    deliver_pending_auto_research_mail,
    pending_auto_research_lifecycle_episodes,
    pending_auto_research_mail_recipients,
    reconcile_auto_research_graph_condition,
    reconcile_pending_auto_research_lifecycle,
    reconcile_pending_auto_research_mail,
    record_auto_research_message,
)
from rcp.runs.tasks.auto_research_child_work import _dispatch_auto_research_child_reply
from rcp.service import RunRequest
from rcp.storage import (
    AgentTaskRecord,
    AppStore,
    AutoResearchLifecycleNoticeRecord,
    EpisodeRecord,
    GraphWatcherRecord,
    NodeStatusGraphCondition,
    ProjectRecord,
)

from .helpers import fabricated_authorizer, wait_for_task


def _sse(event: AgentEvent) -> str:
    return f"data: {event.model_dump_json()}\n\n"


def _store(tmp_path) -> AppStore:
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


def _start_auto_research(
    tasks: BackgroundAgentTasks,
    *,
    invocation_ceiling: int = 6,
) -> tuple[EpisodeRecord, AgentTaskRecord]:
    auto_research, root = start_auto_research(
        tasks,
        "project",
        AutoResearchStartRequest(
            invocation_ceiling=invocation_ceiling,
            provider="codex",
            run_on="local",
            run_truth_scope=["repo-a"],
        ),
        authorized_by=fabricated_authorizer(),
        graph_base_head=GraphHeadRef(revision=0),
        ensure_graph_target=lambda _episode: None,
        episode_id="auto_research",
        operation_id="root",
    )
    assert auto_research.graph_target.kind == "branch"
    assert auto_research.graph_target.branch_id == auto_research.episode_id
    assert auto_research.graph_base_head == GraphHeadRef(revision=0)
    assert root.graph_target == auto_research.graph_target
    return auto_research, wait_for_task(tasks.store, root.operation_id, expect="succeeded")


def _arm_completed_graph_condition(
    store: AppStore,
    auto_research: EpisodeRecord,
    origin: AgentTaskRecord,
    *,
    watcher_id: str = "auto_research-watcher",
) -> GraphWatcherRecord:
    current = store.episode(auto_research.episode_id)
    assert current is not None
    condition = NodeStatusGraphCondition(node_id="blk/result", status_in=["resolved"])
    blocker = Blocker(
        id="blk/result",
        type="blocker",
        title="Wait for the result",
        description="The auto_research continues after this blocker resolves.",
        status="resolved",
    )
    watcher = arm_auto_research_graph_condition(
        store,
        AutoResearchCommandContext(
            episode=current,
            task=origin,
            request=AutoResearchRunRequest.model_validate(origin.request),
        ),
        WatchGraphArguments(
            condition=condition,
            reason="Continue after the canonical result is available.",
        ),
        watcher_id=watcher_id,
        state=GraphState(revision=2, nodes={blocker.id: blocker}),
        execution_host=origin.stage_host or "",
    )
    assert watcher.watcher_id == watcher_id
    assert watcher.status == "completed"
    assert watcher.origin_task_kind == "auto_research"
    assert watcher.origin_operation_id == origin.operation_id
    assert watcher.graph_target == auto_research.graph_target
    assert watcher.chat_id == origin.operation_id
    assert watcher.notified is False
    return watcher


def test_auto_research_effect_ids_are_exact_and_graph_reconciliation_is_read_only(tmp_path) -> None:
    store = _store(tmp_path)
    stage = tmp_path / "auto_research-stage"
    stage.mkdir()

    async def stream(_project_id, _kind, request, execution):
        if execution.continuation == "fresh":
            execution.checkpoint_stage("execution-host", str(stage))
        yield _sse(
            AgentEvent(
                event="session",
                session_id=request.session_id or "orchestrator-session",
            )
        )
        yield _sse(AgentEvent(event="done"))

    tasks = BackgroundAgentTasks(store, stream)
    auto_research, root = _start_auto_research(tasks)
    message = record_auto_research_message(
        store,
        message_id="planned-message",
        episode_id=auto_research.episode_id,
        sender_role="human",
        sender_task_id=None,
        authorized_by=auto_research.authorized_by,
        recipient_task_id=root.operation_id,
        body="Persist this under the command's planned identity.",
    )
    assert message.message_id == "planned-message"
    assert store.auto_research_message("planned-message") == message

    watcher = _arm_completed_graph_condition(
        store,
        auto_research,
        root,
        watcher_id="planned-watcher",
    )
    context = AutoResearchCommandContext(
        episode=auto_research,
        task=root,
        request=AutoResearchRunRequest.model_validate(root.request),
    )
    arguments = WatchGraphArguments(
        condition=watcher.condition,
        reason="Continue after the canonical result is available.",
    )
    events_before = store.agent_task_events(root.operation_id)
    receipts_before = store.agent_task_receipts(root.operation_id)

    assert (
        reconcile_auto_research_graph_condition(
            store,
            context,
            arguments,
            watcher_id="planned-watcher",
            execution_host=root.stage_host or "",
        )
        == watcher
    )
    assert (
        reconcile_auto_research_graph_condition(
            store,
            context,
            arguments,
            watcher_id="missing-watcher",
            execution_host=root.stage_host or "",
        )
        is None
    )
    mismatched_arguments = WatchGraphArguments(
        condition=NodeStatusGraphCondition(node_id="blk/other", status_in=["resolved"]),
        reason=arguments.reason,
    )
    assert (
        reconcile_auto_research_graph_condition(
            store,
            context,
            mismatched_arguments,
            watcher_id="planned-watcher",
            execution_host=root.stage_host or "",
        )
        is None
    )
    assert (
        reconcile_auto_research_graph_condition(
            store,
            context,
            arguments,
            watcher_id="planned-watcher",
            execution_host="different-host",
        )
        is None
    )
    assert store.agent_task_events(root.operation_id) == events_before
    assert store.agent_task_receipts(root.operation_id) == receipts_before
    assert store.watcher("planned-watcher") == watcher


def test_auto_research_graph_watcher_wake_is_one_atomic_paid_actor_continuation(tmp_path) -> None:
    store = _store(tmp_path)
    stage = tmp_path / "auto_research-stage"
    stage.mkdir()
    observed: list[tuple[str, str, str | None]] = []

    async def stream(_project_id, _kind, request, execution):
        observed.append((execution.operation_id, execution.continuation, request.session_id))
        if execution.continuation == "fresh":
            execution.checkpoint_stage("execution-host", str(stage))
        yield _sse(
            AgentEvent(
                event="session",
                session_id=request.session_id or "orchestrator-session",
            )
        )
        yield _sse(AgentEvent(event="done"))

    tasks = BackgroundAgentTasks(store, stream)
    auto_research, root = _start_auto_research(tasks)
    watcher = _arm_completed_graph_condition(store, auto_research, root)
    before = store.episode_budget_meter(auto_research.episode_id)

    wake_id = deliver_auto_research_watcher_group(tasks, [watcher])

    assert wake_id is not None
    wake = wait_for_task(store, wake_id, expect="succeeded")
    delivered = store.watcher(watcher.watcher_id)
    assert isinstance(delivered, GraphWatcherRecord)
    assert delivered.notified is True
    assert delivered.notification_operation_id == wake.operation_id
    assert store.episode_budget_meter(auto_research.episode_id).invocations_used == (
        before.invocations_used + 1
    )
    assert store.auto_research_invocation_role(wake.operation_id) == "orchestrator"
    assert wake.request["actor_operation_id"] == root.operation_id
    assert wake.request["role"] == "orchestrator"
    assert wake.request["control_node_id"] is None
    assert wake.request["wake_cause"] == "graph_condition"
    assert wake.request["watcher_ids"] == [watcher.watcher_id]
    assert wake.graph_target == auto_research.graph_target
    assert wake.native_session_id == root.native_session_id == "orchestrator-session"
    assert wake.stage_host == root.stage_host == "execution-host"
    assert wake.stage_root == root.stage_root == str(stage)
    assert store.agent_task_continuation_cause(wake.operation_id) == "graph_condition_wake"
    assert observed == [
        (root.operation_id, "fresh", None),
        (wake.operation_id, "graph_condition_wake", "orchestrator-session"),
    ]


def test_committed_watcher_wake_reconciles_after_thread_start_failure(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    stage = tmp_path / "auto_research-stage"
    stage.mkdir()
    executions: list[str] = []

    async def stream(_project_id, _kind, request, execution):
        executions.append(execution.operation_id)
        if execution.continuation == "fresh":
            execution.checkpoint_stage("execution-host", str(stage))
        yield _sse(
            AgentEvent(
                event="session",
                session_id=request.session_id or "orchestrator-session",
            )
        )
        yield _sse(AgentEvent(event="done"))

    tasks = BackgroundAgentTasks(store, stream)
    auto_research, root = _start_auto_research(tasks)
    watcher = _arm_completed_graph_condition(store, auto_research, root)
    before = store.episode_budget_meter(auto_research.episode_id)
    real_start = threading.Thread.start
    failed = False

    def fail_once(thread: threading.Thread) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("simulated Thread.start failure")
        real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_once)
    with pytest.raises(RuntimeError, match="Thread.start failure"):
        deliver_auto_research_watcher_group(tasks, [watcher])
    monkeypatch.setattr(threading.Thread, "start", real_start)

    claimed = store.watcher(watcher.watcher_id)
    assert isinstance(claimed, GraphWatcherRecord)
    wake_id = claimed.notification_operation_id
    assert wake_id is not None
    queued = store.agent_task(wake_id)
    assert queued is not None and queued.status == "queued"
    after_claim = store.episode_budget_meter(auto_research.episode_id)
    assert after_claim.invocations_used == before.invocations_used + 1

    assert reconcile_pending_auto_research_lifecycle(
        tasks,
        episode_id=auto_research.episode_id,
    ) == [wake_id]
    wait_for_task(store, wake_id, expect="succeeded")

    delivered = store.watcher(watcher.watcher_id)
    assert isinstance(delivered, GraphWatcherRecord)
    assert delivered.notification_operation_id == wake_id
    assert store.episode_budget_meter(auto_research.episode_id) == after_claim
    assert executions.count(wake_id) == 1
    assert (
        sum(
            receipt.category == "operation_created"
            for receipt in store.agent_task_receipts(wake_id)
        )
        == 1
    )


def test_busy_auto_research_actor_leaves_completed_watcher_unclaimed_and_unspent(tmp_path) -> None:
    store = _store(tmp_path)
    stage = tmp_path / "auto_research-stage"
    stage.mkdir()
    busy_entered = threading.Event()
    release_busy = threading.Event()

    async def stream(_project_id, _kind, request, execution):
        if execution.continuation == "fresh":
            execution.checkpoint_stage("execution-host", str(stage))
        yield _sse(
            AgentEvent(
                event="session",
                session_id=request.session_id or "orchestrator-session",
            )
        )
        if execution.operation_id == "busy-turn":
            busy_entered.set()
            while not release_busy.is_set():
                await asyncio.sleep(0.01)
        yield _sse(AgentEvent(event="done"))

    tasks = BackgroundAgentTasks(store, stream)
    auto_research, root = _start_auto_research(tasks)
    watcher = _arm_completed_graph_condition(store, auto_research, root)
    busy = start_auto_research_turn(
        tasks,
        auto_research.episode_id,
        AutoResearchRunRequest(
            episode_id=auto_research.episode_id,
            role="orchestrator",
            provider="codex",
            run_on="local",
            instruction="Keep the actor occupied while delivery races.",
        ),
        parent_operation_id=root.operation_id,
        operation_id="busy-turn",
    )
    assert busy is not None
    assert busy_entered.wait(timeout=2)
    before = store.episode_budget_meter(auto_research.episode_id)
    task_ids_before = [
        task.operation_id for task in store.auto_research_tasks(auto_research.episode_id)
    ]

    try:
        assert deliver_auto_research_watcher_group(tasks, [watcher]) is None
        unchanged = store.watcher(watcher.watcher_id)
        assert isinstance(unchanged, GraphWatcherRecord)
        assert unchanged.notified is False
        assert unchanged.notification_operation_id is None
        assert store.episode_budget_meter(auto_research.episode_id) == before
        assert [
            task.operation_id for task in store.auto_research_tasks(auto_research.episode_id)
        ] == task_ids_before
    finally:
        release_busy.set()
        wait_for_task(store, busy.operation_id, expect="succeeded")


def test_pending_auto_research_mail_coalesces_into_one_paid_message_wake(tmp_path) -> None:
    store = _store(tmp_path)
    stage = tmp_path / "auto_research-stage"
    stage.mkdir()

    async def stream(_project_id, _kind, request, execution):
        if execution.continuation == "fresh":
            execution.checkpoint_stage("execution-host", str(stage))
        yield _sse(
            AgentEvent(
                event="session",
                session_id=request.session_id or "orchestrator-session",
            )
        )
        yield _sse(AgentEvent(event="done"))

    tasks = BackgroundAgentTasks(store, stream)
    auto_research, root = _start_auto_research(tasks)
    messages = [
        record_auto_research_message(
            store,
            episode_id=auto_research.episode_id,
            sender_role="human",
            sender_task_id=None,
            authorized_by=auto_research.authorized_by,
            recipient_task_id=root.operation_id,
            body=body,
        )
        for body in ("Review the new evidence.", "Also resolve the blocker.")
    ]
    before = store.episode_budget_meter(auto_research.episode_id)

    wake_id = deliver_pending_auto_research_mail(
        tasks,
        episode_id=auto_research.episode_id,
        recipient_task_id=root.operation_id,
    )

    assert wake_id is not None
    wake = wait_for_task(store, wake_id, expect="succeeded")
    claimed = {
        message.message_id: message
        for message in store.auto_research_messages(auto_research.episode_id)
    }
    assert {claimed[message.message_id].delivery_operation_id for message in messages} == {
        wake.operation_id
    }
    assert all(claimed[message.message_id].delivered_at is not None for message in messages)
    assert store.pending_auto_research_messages(auto_research.episode_id, root.operation_id) == []
    assert store.episode_budget_meter(auto_research.episode_id).invocations_used == (
        before.invocations_used + 1
    )
    assert store.auto_research_invocation_role(wake.operation_id) == "orchestrator"
    assert wake.request["wake_cause"] == "message"
    assert wake.request["actor_operation_id"] == root.operation_id
    assert wake.native_session_id == root.native_session_id == "orchestrator-session"
    assert wake.stage_host == root.stage_host == "execution-host"
    assert wake.stage_root == root.stage_root == str(stage)
    assert store.agent_task_continuation_cause(wake.operation_id) == "message_wake"
    assert len(store.auto_research_tasks(auto_research.episode_id)) == 2


def test_committed_root_mail_wake_reconciles_without_reclaim_or_respend(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    stage = tmp_path / "auto_research-stage"
    stage.mkdir()
    executions: list[str] = []

    async def stream(_project_id, _kind, request, execution):
        executions.append(execution.operation_id)
        if execution.continuation == "fresh":
            execution.checkpoint_stage("execution-host", str(stage))
        yield _sse(
            AgentEvent(
                event="session",
                session_id=request.session_id or "orchestrator-session",
            )
        )
        yield _sse(AgentEvent(event="done"))

    tasks = BackgroundAgentTasks(store, stream)
    auto_research, root = _start_auto_research(tasks)
    message = record_auto_research_message(
        store,
        episode_id=auto_research.episode_id,
        sender_role="human",
        sender_task_id=None,
        authorized_by=auto_research.authorized_by,
        recipient_task_id=root.operation_id,
        body="Continue from this exact mail allocation.",
    )
    before = store.episode_budget_meter(auto_research.episode_id)
    real_spawn_record = tasks._spawn_record

    def crash_after_commit(*_args, **_kwargs):
        raise RuntimeError("simulated crash after root mail commit")

    monkeypatch.setattr(tasks, "_spawn_record", crash_after_commit)
    with pytest.raises(RuntimeError, match="after root mail commit"):
        deliver_pending_auto_research_mail(
            tasks,
            episode_id=auto_research.episode_id,
            recipient_task_id=root.operation_id,
        )
    monkeypatch.setattr(tasks, "_spawn_record", real_spawn_record)

    claimed = store.auto_research_message(message.message_id)
    assert claimed is not None
    wake_id = claimed.delivery_operation_id
    assert wake_id is not None
    assert store.agent_task(wake_id).status == "queued"  # type: ignore[union-attr]
    after_claim = store.episode_budget_meter(auto_research.episode_id)
    assert after_claim.invocations_used == before.invocations_used + 1

    assert reconcile_pending_auto_research_mail(
        tasks,
        episode_id=auto_research.episode_id,
    ) == [wake_id]
    wait_for_task(store, wake_id, expect="succeeded")

    delivered = store.auto_research_message(message.message_id)
    assert delivered is not None and delivered.delivery_operation_id == wake_id
    assert store.episode_budget_meter(auto_research.episode_id) == after_claim
    assert executions.count(wake_id) == 1


def test_root_mail_wakes_the_exact_routed_ordinary_child_work_session(tmp_path) -> None:
    store = _store(tmp_path)
    root_stage = tmp_path / "auto_research-stage"
    child_stage = tmp_path / "child-work-stage"
    root_stage.mkdir()
    child_stage.mkdir()
    observed: list[tuple[str, str, str | None]] = []

    async def stream(_project_id, kind, request, execution):
        observed.append((kind, execution.continuation, request.session_id))
        if kind == "auto_research" and execution.continuation == "fresh":
            execution.checkpoint_stage("", str(root_stage))
        elif kind == "node_chat" and execution.continuation == "fresh":
            execution.checkpoint_stage("", str(child_stage))
        yield _sse(
            AgentEvent(
                event="session",
                session_id=request.session_id
                or ("child-session" if kind == "node_chat" else "orchestrator-session"),
            )
        )
        yield _sse(AgentEvent(event="done"))

    tasks = BackgroundAgentTasks(store, stream)
    auto_research, root = _start_auto_research(tasks)
    worker_id = "00000000-0000-4000-8000-000000000411"
    instruction = "Recheck the bounded runtime evidence and report back."
    child = start_auto_research_child_work(
        tasks,
        auto_research.episode_id,
        RunRequest(
            provider="codex",
            run_on="local",
            run_truth_scope=["repo-a"],
            chat_scope="node",
            node_id="blk/result",
            chat_id=worker_id,
            message=instruction,
            mode="work",
            trigger="orchestrator",
            patch_kind="work",
        ),
        admitted_by_operation_id=root.operation_id,
        worker_id=worker_id,
        instruction=instruction,
        instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
    )
    child = wait_for_task(store, child.operation_id, expect="succeeded")
    message = record_auto_research_message(
        store,
        episode_id=auto_research.episode_id,
        sender_role="orchestrator",
        sender_task_id=root.operation_id,
        authorized_by=None,
        recipient_task_id=worker_id,
        control_node_id="blk/result",
        body="The graph moved; check the new observation.",
    )
    before = store.episode_budget_meter(auto_research.episode_id)

    assert pending_auto_research_mail_recipients(
        store,
        episode_id=auto_research.episode_id,
    ) == [(auto_research.episode_id, worker_id)]
    wake_id = deliver_pending_auto_research_mail(
        tasks,
        episode_id=auto_research.episode_id,
        recipient_task_id=worker_id,
    )

    assert wake_id is not None
    wake = wait_for_task(store, wake_id, expect="succeeded")
    delivered = store.auto_research_message(message.message_id)
    assert delivered is not None and delivered.delivery_operation_id == wake.operation_id
    assert wake.kind == "node_chat"
    assert wake.parent_operation_id == child.operation_id
    assert wake.native_session_id == child.native_session_id == "child-session"
    assert wake.stage_root == child.stage_root == str(child_stage)
    assert wake.request["message"] is None
    assert store.agent_task_continuation_cause(wake.operation_id) == "message_wake"
    assert store.episode_budget_meter(auto_research.episode_id).invocations_used == (
        before.invocations_used + 1
    )
    assert observed[-1] == ("node_chat", "message_wake", "child-session")

    route = store.auto_research_child_work(worker_id)
    assert route is not None
    reply_request = MessageCommandRequest(
        verb="message",
        mailbox_id="a" * 32,
        request_id="b" * 32,
        credential="c" * 64,
        idempotency_key="reply-after-recheck",
        arguments=MessageArguments(body="The recheck is complete; inspect the task result."),
    )
    task_count = len(store.episode_tasks(auto_research.episode_id))
    reply = _dispatch_auto_research_child_reply(
        AgentTaskExecution(
            operation_id=wake.operation_id,
            store=store,
            control=AgentProcessControl(),
        ),
        route,
        reply_request,
    )
    replay = _dispatch_auto_research_child_reply(
        AgentTaskExecution(
            operation_id=wake.operation_id,
            store=store,
            control=AgentProcessControl(),
        ),
        route,
        reply_request.model_copy(update={"request_id": "d" * 32}),
    )

    assert reply.status == replay.status == "ok"
    assert reply.result == replay.result
    assert reply.result["delivery"] == "pending"
    pending_root_mail = store.pending_auto_research_messages(
        auto_research.episode_id,
        root.operation_id,
    )
    assert [item.message_id for item in pending_root_mail] == [reply.result["message_id"]]
    assert pending_root_mail[0].sender_role == "worker"
    assert pending_root_mail[0].sender_task_id == wake.operation_id
    assert len(store.episode_tasks(auto_research.episode_id)) == task_count


def test_committed_child_work_mail_wake_reconciles_the_same_operation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    root_stage = tmp_path / "auto_research-stage"
    child_stage = tmp_path / "child-work-stage"
    root_stage.mkdir()
    child_stage.mkdir()
    executions: list[tuple[str, str]] = []

    async def stream(_project_id, kind, request, execution):
        executions.append((execution.operation_id, execution.continuation))
        if kind == "auto_research" and execution.continuation == "fresh":
            execution.checkpoint_stage("", str(root_stage))
        elif kind == "node_chat" and execution.continuation == "fresh":
            execution.checkpoint_stage("", str(child_stage))
        yield _sse(
            AgentEvent(
                event="session",
                session_id=request.session_id
                or ("child-session" if kind == "node_chat" else "orchestrator-session"),
            )
        )
        yield _sse(AgentEvent(event="done"))

    tasks = BackgroundAgentTasks(store, stream)
    auto_research, root = _start_auto_research(tasks)
    worker_id = "00000000-0000-4000-8000-000000000419"
    instruction = "Recheck the bounded runtime evidence and report back."
    child = start_auto_research_child_work(
        tasks,
        auto_research.episode_id,
        RunRequest(
            provider="codex",
            run_on="local",
            run_truth_scope=["repo-a"],
            chat_scope="node",
            node_id="blk/result",
            chat_id=worker_id,
            message=instruction,
            mode="work",
            trigger="orchestrator",
            patch_kind="work",
        ),
        admitted_by_operation_id=root.operation_id,
        worker_id=worker_id,
        instruction=instruction,
        instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
    )
    child = wait_for_task(store, child.operation_id, expect="succeeded")
    message = record_auto_research_message(
        store,
        episode_id=auto_research.episode_id,
        sender_role="orchestrator",
        sender_task_id=root.operation_id,
        authorized_by=None,
        recipient_task_id=worker_id,
        control_node_id="blk/result",
        body="Continue from this exact child mail allocation.",
    )
    before = store.episode_budget_meter(auto_research.episode_id)
    real_spawn_record = tasks._spawn_record

    def crash_after_commit(*_args, **_kwargs):
        raise RuntimeError("simulated crash after child mail commit")

    monkeypatch.setattr(tasks, "_spawn_record", crash_after_commit)
    with pytest.raises(RuntimeError, match="after child mail commit"):
        deliver_pending_auto_research_mail(
            tasks,
            episode_id=auto_research.episode_id,
            recipient_task_id=worker_id,
        )
    monkeypatch.setattr(tasks, "_spawn_record", real_spawn_record)

    claimed = store.auto_research_message(message.message_id)
    assert claimed is not None
    wake_id = claimed.delivery_operation_id
    assert wake_id is not None
    route = store.auto_research_child_work(worker_id)
    assert route is not None and route.current_operation_id == wake_id
    after_claim = store.episode_budget_meter(auto_research.episode_id)
    assert after_claim.invocations_used == before.invocations_used + 1

    assert reconcile_pending_auto_research_mail(
        tasks,
        episode_id=auto_research.episode_id,
    ) == [wake_id]
    wake = wait_for_task(store, wake_id, expect="succeeded")

    delivered = store.auto_research_message(message.message_id)
    assert delivered is not None and delivered.delivery_operation_id == wake_id
    assert wake.parent_operation_id == child.operation_id
    assert wake.native_session_id == child.native_session_id == "child-session"
    assert store.episode_budget_meter(auto_research.episode_id) == after_claim
    assert executions.count((wake_id, "message_wake")) == 1


def test_child_work_mail_claims_only_the_bounded_wire_prefix(tmp_path) -> None:
    store = _store(tmp_path)
    root_stage = tmp_path / "auto_research-stage"
    child_stage = tmp_path / "child-work-stage"
    root_stage.mkdir()
    child_stage.mkdir()

    async def stream(_project_id, kind, request, execution):
        if kind == "auto_research" and execution.continuation == "fresh":
            execution.checkpoint_stage("", str(root_stage))
        elif kind == "node_chat" and execution.continuation == "fresh":
            execution.checkpoint_stage("", str(child_stage))
        yield _sse(
            AgentEvent(
                event="session",
                session_id=request.session_id
                or ("child-session" if kind == "node_chat" else "orchestrator-session"),
            )
        )
        yield _sse(AgentEvent(event="done"))

    tasks = BackgroundAgentTasks(store, stream)
    auto_research, root = _start_auto_research(tasks)
    worker_id = "00000000-0000-4000-8000-000000000412"
    instruction = "Review the bounded mail batch."
    child = start_auto_research_child_work(
        tasks,
        auto_research.episode_id,
        RunRequest(
            provider="codex",
            run_on="local",
            run_truth_scope=["repo-a"],
            chat_scope="node",
            node_id="blk/result",
            chat_id=worker_id,
            message=instruction,
            mode="work",
            trigger="orchestrator",
            patch_kind="work",
        ),
        admitted_by_operation_id=root.operation_id,
        worker_id=worker_id,
        instruction=instruction,
        instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
    )
    wait_for_task(store, child.operation_id, expect="succeeded")
    messages = [
        record_auto_research_message(
            store,
            message_id=f"large-{index:03d}-{'x' * 256}",
            episode_id=auto_research.episode_id,
            sender_role="orchestrator",
            sender_task_id=root.operation_id,
            authorized_by=None,
            recipient_task_id=worker_id,
            control_node_id="blk/result",
            body="y" * 16_000,
        )
        for index in range(64)
    ]
    before = store.episode_budget_meter(auto_research.episode_id)

    wake_id = deliver_pending_auto_research_mail(
        tasks,
        episode_id=auto_research.episode_id,
        recipient_task_id=worker_id,
    )

    assert wake_id is not None
    wait_for_task(store, wake_id, expect="succeeded")
    claimed = [
        message
        for message in store.auto_research_messages(auto_research.episode_id)
        if message.delivery_operation_id == wake_id
    ]
    pending = store.pending_auto_research_messages(auto_research.episode_id, worker_id)
    assert 0 < len(claimed) < len(messages)
    assert [message.message_id for message in claimed] == [
        message.message_id for message in messages[: len(claimed)]
    ]
    assert [message.message_id for message in pending] == [
        message.message_id for message in messages[len(claimed) :]
    ]
    assert store.episode_budget_meter(auto_research.episode_id).invocations_used == (
        before.invocations_used + 1
    )


def test_lifecycle_notice_and_root_mail_share_one_paid_wake(tmp_path) -> None:
    store = _store(tmp_path)
    stage = tmp_path / "auto_research-stage"
    stage.mkdir()
    observed: list[tuple[str, str, str | None]] = []

    async def stream(_project_id, _kind, request, execution):
        observed.append((execution.operation_id, execution.continuation, request.wake_cause))
        if execution.continuation == "fresh":
            execution.checkpoint_stage("execution-host", str(stage))
        yield _sse(
            AgentEvent(
                event="session",
                session_id=request.session_id or "orchestrator-session",
            )
        )
        yield _sse(AgentEvent(event="done"))

    tasks = BackgroundAgentTasks(store, stream)
    auto_research, root = _start_auto_research(tasks)
    notice = store.record_auto_research_lifecycle_notice(
        AutoResearchLifecycleNoticeRecord(
            notice_id="worker-finished",
            episode_id=auto_research.episode_id,
            source_kind="worker",
            source_id="worker-one",
            source_event="succeeded",
            payload={"kind": "work", "status": "succeeded"},
            created_at=store.now(),
        )
    )
    message = record_auto_research_message(
        store,
        episode_id=auto_research.episode_id,
        sender_role="human",
        sender_task_id=None,
        authorized_by=auto_research.authorized_by,
        recipient_task_id=root.operation_id,
        body="Also inspect the human's pending note.",
    )
    before = store.episode_budget_meter(auto_research.episode_id)

    assert pending_auto_research_lifecycle_episodes(
        store,
        episode_id=auto_research.episode_id,
    ) == [auto_research.episode_id]
    wake_ids = reconcile_pending_auto_research_lifecycle(
        tasks,
        episode_id=auto_research.episode_id,
    )

    assert len(wake_ids) == 1
    wake = wait_for_task(store, wake_ids[0], expect="succeeded")
    assert [
        item.notice_id for item in store.auto_research_lifecycle_delivery(wake.operation_id)
    ] == [notice.notice_id]
    claimed_message = store.auto_research_message(message.message_id)
    assert claimed_message is not None
    assert claimed_message.delivery_operation_id == wake.operation_id
    assert store.pending_auto_research_lifecycle_notices(auto_research.episode_id) == []
    assert store.pending_auto_research_messages(auto_research.episode_id, root.operation_id) == []
    assert store.episode_budget_meter(auto_research.episode_id).invocations_used == (
        before.invocations_used + 1
    )
    assert wake.request["wake_cause"] == "lifecycle"
    assert wake.request["role"] == "orchestrator"
    assert wake.request["actor_operation_id"] == root.operation_id
    assert store.agent_task_continuation_cause(wake.operation_id) == "lifecycle_wake"
    assert observed == [
        (root.operation_id, "fresh", None),
        (wake.operation_id, "lifecycle_wake", "lifecycle"),
    ]


def test_terminal_parent_keeps_pending_lifecycle_notice_without_repolling(tmp_path) -> None:
    store = _store(tmp_path)
    stage = tmp_path / "auto_research-stage"
    stage.mkdir()

    async def stream(_project_id, _kind, request, execution):
        if execution.continuation == "fresh":
            execution.checkpoint_stage("execution-host", str(stage))
        yield _sse(
            AgentEvent(
                event="session",
                session_id=request.session_id or "orchestrator-session",
            )
        )
        yield _sse(AgentEvent(event="done"))

    tasks = BackgroundAgentTasks(store, stream)
    auto_research, _root = _start_auto_research(tasks)
    notice = store.record_auto_research_lifecycle_notice(
        AutoResearchLifecycleNoticeRecord(
            notice_id="terminal-parent-notice",
            episode_id=auto_research.episode_id,
            source_kind="worker",
            source_id="worker-one",
            source_event="succeeded",
            payload={"kind": "work", "status": "succeeded"},
            created_at=store.now(),
        )
    )

    assert pending_auto_research_lifecycle_episodes(store) == [auto_research.episode_id]

    store.fence_episode_ending(auto_research.episode_id, "completed")

    assert pending_auto_research_lifecycle_episodes(store) == []
    assert store.pending_auto_research_lifecycle_notices(auto_research.episode_id) == [notice]


def test_committed_lifecycle_wake_reconciles_after_dispatch_preparation_failure(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    stage = tmp_path / "auto_research-stage"
    stage.mkdir()
    executions: list[str] = []

    async def stream(_project_id, _kind, request, execution):
        executions.append(execution.operation_id)
        if execution.continuation == "fresh":
            execution.checkpoint_stage("", str(stage))
        yield _sse(
            AgentEvent(
                event="session",
                session_id=request.session_id or "orchestrator-session",
            )
        )
        yield _sse(AgentEvent(event="done"))

    tasks = BackgroundAgentTasks(store, stream)
    auto_research, _root = _start_auto_research(tasks)
    notice = store.record_auto_research_lifecycle_notice(
        AutoResearchLifecycleNoticeRecord(
            notice_id="committed-lifecycle-notice",
            episode_id=auto_research.episode_id,
            source_kind="worker",
            source_id="worker-one",
            source_event="succeeded",
            payload={"kind": "work", "status": "succeeded"},
            created_at=store.now(),
        )
    )
    before = store.episode_budget_meter(auto_research.episode_id)
    real_prepare = tasks._record_spawn_dispatch
    failed = False

    def fail_once(record, request, *, continuation, parent):
        nonlocal failed
        if not failed and isinstance(request, AutoResearchRunRequest):
            failed = True
            raise RuntimeError("simulated dispatch preparation failure")
        real_prepare(
            record,
            request,
            continuation=continuation,
            parent=parent,
        )

    monkeypatch.setattr(tasks, "_record_spawn_dispatch", fail_once)
    with pytest.raises(RuntimeError, match="dispatch preparation failure"):
        deliver_pending_auto_research_lifecycle(
            tasks,
            episode_id=auto_research.episode_id,
        )
    monkeypatch.setattr(tasks, "_record_spawn_dispatch", real_prepare)

    claimed = store.auto_research_lifecycle_notices(auto_research.episode_id)[0]
    wake_id = claimed.delivery_operation_id
    assert claimed.notice_id == notice.notice_id
    assert wake_id is not None
    assert store.agent_task(wake_id).status == "queued"  # type: ignore[union-attr]
    after_claim = store.episode_budget_meter(auto_research.episode_id)
    assert after_claim.invocations_used == before.invocations_used + 1

    assert reconcile_pending_auto_research_lifecycle(
        tasks,
        episode_id=auto_research.episode_id,
    ) == [wake_id]
    wait_for_task(store, wake_id, expect="succeeded")

    delivered = store.auto_research_lifecycle_delivery(wake_id)
    assert [item.notice_id for item in delivered] == [notice.notice_id]
    assert store.episode_budget_meter(auto_research.episode_id) == after_claim
    assert executions.count(wake_id) == 1


def test_active_child_reply_waits_and_coalesces_with_its_lifecycle_notice(tmp_path) -> None:
    store = _store(tmp_path)
    root_stage = tmp_path / "auto_research-stage"
    child_stage = tmp_path / "child-work-stage"
    root_stage.mkdir()
    child_stage.mkdir()
    reply_written = threading.Event()
    release_child = threading.Event()
    message_id = "reply-before-child-settlement"

    async def stream(_project_id, kind, request, execution):
        if kind == "auto_research" and execution.continuation == "fresh":
            execution.checkpoint_stage("", str(root_stage))
        elif kind == "node_chat" and execution.continuation == "fresh":
            execution.checkpoint_stage("", str(child_stage))
        yield _sse(
            AgentEvent(
                event="session",
                session_id=request.session_id
                or ("child-session" if kind == "node_chat" else "orchestrator-session"),
            )
        )
        if kind == "node_chat":
            route = store.auto_research_child_work_for_operation(execution.operation_id)
            assert route is not None
            record_auto_research_message(
                store,
                message_id=message_id,
                episode_id=route.episode_id,
                sender_role="worker",
                sender_task_id=execution.operation_id,
                authorized_by=None,
                recipient_task_id=root.operation_id,
                control_node_id=route.control_node_id,
                body="The delegated result is ready.",
            )
            reply_written.set()
            while not release_child.is_set():
                await asyncio.sleep(0.01)
        yield _sse(AgentEvent(event="done"))

    tasks = BackgroundAgentTasks(store, stream)
    auto_research, root = _start_auto_research(tasks)
    worker_id = "00000000-0000-4000-8000-000000000413"
    instruction = "Produce one bounded result and reply."
    child = start_auto_research_child_work(
        tasks,
        auto_research.episode_id,
        RunRequest(
            provider="codex",
            run_on="local",
            run_truth_scope=["repo-a"],
            chat_scope="node",
            node_id="blk/result",
            chat_id=worker_id,
            message=instruction,
            mode="work",
            trigger="orchestrator",
            patch_kind="work",
        ),
        admitted_by_operation_id=root.operation_id,
        worker_id=worker_id,
        instruction=instruction,
        instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
    )
    deadline = time.monotonic() + 5
    while not reply_written.wait(timeout=0.05) and time.monotonic() < deadline:
        current = store.agent_task(child.operation_id)
        if current is not None and current.status in {"failed", "paused", "interrupted"}:
            raise AssertionError(current.error or current.status_message)
    assert reply_written.is_set()
    before = store.episode_budget_meter(auto_research.episode_id)

    assert (
        pending_auto_research_mail_recipients(
            store,
            episode_id=auto_research.episode_id,
        )
        == []
    )
    assert (
        reconcile_pending_auto_research_mail(
            tasks,
            episode_id=auto_research.episode_id,
        )
        == []
    )
    assert store.episode_budget_meter(auto_research.episode_id) == before
    assert store.auto_research_message(message_id).delivered_at is None  # type: ignore[union-attr]

    release_child.set()
    wait_for_task(store, child.operation_id, expect="succeeded")
    wake_ids = reconcile_pending_auto_research_lifecycle(
        tasks,
        episode_id=auto_research.episode_id,
    )

    assert len(wake_ids) == 1
    wake = wait_for_task(store, wake_ids[0], expect="succeeded")
    delivered_message = store.auto_research_message(message_id)
    assert delivered_message is not None
    assert delivered_message.delivery_operation_id == wake.operation_id
    lifecycle = store.auto_research_lifecycle_delivery(wake.operation_id)
    assert len(lifecycle) == 1
    assert lifecycle[0].source_id == worker_id
    assert store.episode_budget_meter(auto_research.episode_id).invocations_used == (
        before.invocations_used + 1
    )


def test_busy_root_leaves_lifecycle_and_mail_unclaimed(tmp_path) -> None:
    store = _store(tmp_path)
    stage = tmp_path / "auto_research-stage"
    stage.mkdir()
    busy_entered = threading.Event()
    release_busy = threading.Event()

    async def stream(_project_id, _kind, request, execution):
        if execution.continuation == "fresh":
            execution.checkpoint_stage("execution-host", str(stage))
        yield _sse(
            AgentEvent(
                event="session",
                session_id=request.session_id or "orchestrator-session",
            )
        )
        if execution.operation_id == "busy-turn":
            busy_entered.set()
            while not release_busy.is_set():
                await asyncio.sleep(0.01)
        yield _sse(AgentEvent(event="done"))

    tasks = BackgroundAgentTasks(store, stream)
    auto_research, root = _start_auto_research(tasks)
    busy = start_auto_research_turn(
        tasks,
        auto_research.episode_id,
        AutoResearchRunRequest(
            episode_id=auto_research.episode_id,
            role="orchestrator",
            provider="codex",
            run_on="local",
            instruction="Keep the root actor busy.",
        ),
        parent_operation_id=root.operation_id,
        operation_id="busy-turn",
    )
    assert busy is not None
    assert busy_entered.wait(timeout=2)
    notice = store.record_auto_research_lifecycle_notice(
        AutoResearchLifecycleNoticeRecord(
            notice_id="busy-worker-finished",
            episode_id=auto_research.episode_id,
            source_kind="worker",
            source_id="worker-two",
            source_event="succeeded",
            payload={"kind": "work", "status": "succeeded"},
            created_at=store.now(),
        )
    )
    message = record_auto_research_message(
        store,
        episode_id=auto_research.episode_id,
        sender_role="human",
        sender_task_id=None,
        authorized_by=auto_research.authorized_by,
        recipient_task_id=root.operation_id,
        body="Keep this mail pending with the notice.",
    )
    before = store.episode_budget_meter(auto_research.episode_id)

    try:
        assert (
            deliver_pending_auto_research_lifecycle(
                tasks,
                episode_id=auto_research.episode_id,
            )
            is None
        )
        assert store.episode_budget_meter(auto_research.episode_id) == before
        assert store.pending_auto_research_lifecycle_notices(auto_research.episode_id) == [notice]
        assert store.pending_auto_research_messages(
            auto_research.episode_id,
            root.operation_id,
        ) == [message]
    finally:
        release_busy.set()
        wait_for_task(store, busy.operation_id, expect="succeeded")


def test_uncheckpointed_root_leaves_lifecycle_notice_unclaimed(tmp_path) -> None:
    store = _store(tmp_path)

    async def stream(_project_id, _kind, request, _execution):
        yield _sse(
            AgentEvent(
                event="session",
                session_id=request.session_id or "orchestrator-session",
            )
        )
        yield _sse(AgentEvent(event="done"))

    tasks = BackgroundAgentTasks(store, stream)
    auto_research, root = _start_auto_research(tasks)
    assert root.native_session_id == "orchestrator-session"
    assert root.stage_root is None
    notice = store.record_auto_research_lifecycle_notice(
        AutoResearchLifecycleNoticeRecord(
            notice_id="uncheckpointed-worker-finished",
            episode_id=auto_research.episode_id,
            source_kind="worker",
            source_id="worker-three",
            source_event="succeeded",
            payload={"kind": "work", "status": "succeeded"},
            created_at=store.now(),
        )
    )
    before = store.episode_budget_meter(auto_research.episode_id)

    assert (
        deliver_pending_auto_research_lifecycle(
            tasks,
            episode_id=auto_research.episode_id,
        )
        is None
    )
    assert store.episode_budget_meter(auto_research.episode_id) == before
    assert store.pending_auto_research_lifecycle_notices(auto_research.episode_id) == [notice]


def test_reconciliation_retries_every_pending_canonical_actor(tmp_path) -> None:
    store = _store(tmp_path)
    stage = tmp_path / "auto_research-stage"
    stage.mkdir()

    async def stream(_project_id, _kind, request, execution):
        if execution.continuation == "fresh":
            execution.checkpoint_stage("execution-host", str(stage))
        yield _sse(
            AgentEvent(
                event="session",
                session_id=request.session_id or f"{execution.operation_id}-session",
            )
        )
        yield _sse(AgentEvent(event="done"))

    tasks = BackgroundAgentTasks(store, stream)
    auto_research, root = _start_auto_research(tasks)
    worker = start_auto_research_turn(
        tasks,
        auto_research.episode_id,
        AutoResearchRunRequest(
            episode_id=auto_research.episode_id,
            role="worker",
            control_node_id="blk/check-result",
            provider="codex",
            run_on="local",
            instruction="Check the result and report back.",
        ),
        parent_operation_id=root.operation_id,
        operation_id="worker",
    )
    assert worker is not None
    worker = wait_for_task(store, worker.operation_id, expect="succeeded")
    root_message = record_auto_research_message(
        store,
        episode_id=auto_research.episode_id,
        sender_role="human",
        sender_task_id=None,
        authorized_by=auto_research.authorized_by,
        recipient_task_id=root.operation_id,
        body="Review the worker result.",
    )
    worker_message = record_auto_research_message(
        store,
        episode_id=auto_research.episode_id,
        sender_role="orchestrator",
        sender_task_id=root.operation_id,
        authorized_by=None,
        recipient_task_id=worker.operation_id,
        body="Re-check the canonical blocker.",
    )

    assert pending_auto_research_mail_recipients(
        store,
        episode_id=auto_research.episode_id,
    ) == [
        (auto_research.episode_id, root.operation_id),
        (auto_research.episode_id, worker.operation_id),
    ]
    wake_ids = reconcile_pending_auto_research_mail(
        tasks,
        episode_id=auto_research.episode_id,
    )

    assert len(wake_ids) == 2
    for wake_id in wake_ids:
        wait_for_task(store, wake_id, expect="succeeded")
    claimed = {
        message.message_id: message
        for message in store.auto_research_messages(auto_research.episode_id)
    }
    assert claimed[root_message.message_id].delivery_operation_id in wake_ids
    assert claimed[worker_message.message_id].delivery_operation_id in wake_ids
    assert (
        claimed[root_message.message_id].delivery_operation_id
        != claimed[worker_message.message_id].delivery_operation_id
    )
    assert (
        pending_auto_research_mail_recipients(
            store,
            episode_id=auto_research.episode_id,
        )
        == []
    )


def test_bounded_mail_overflow_stays_pending_for_the_next_paid_retry(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("rcp.runs.auto_research_mail.AUTO_RESEARCH_MAIL_MAX_MESSAGES", 2)
    store = _store(tmp_path)
    stage = tmp_path / "auto_research-stage"
    stage.mkdir()

    async def stream(_project_id, _kind, request, execution):
        if execution.continuation == "fresh":
            execution.checkpoint_stage("execution-host", str(stage))
        yield _sse(
            AgentEvent(
                event="session",
                session_id=request.session_id or "orchestrator-session",
            )
        )
        yield _sse(AgentEvent(event="done"))

    tasks = BackgroundAgentTasks(store, stream)
    auto_research, root = _start_auto_research(tasks)
    messages = [
        record_auto_research_message(
            store,
            episode_id=auto_research.episode_id,
            sender_role="human",
            sender_task_id=None,
            authorized_by=auto_research.authorized_by,
            recipient_task_id=root.operation_id,
            body=f"Bounded message {index}.",
        )
        for index in range(3)
    ]

    first_wake_id = deliver_pending_auto_research_mail(
        tasks,
        episode_id=auto_research.episode_id,
        recipient_task_id=root.operation_id,
    )

    assert first_wake_id is not None
    wait_for_task(store, first_wake_id, expect="succeeded")
    first_batch = [
        message
        for message in store.auto_research_messages(auto_research.episode_id)
        if message.delivery_operation_id == first_wake_id
    ]
    assert [message.message_id for message in first_batch] == [
        messages[0].message_id,
        messages[1].message_id,
    ]
    assert store.pending_auto_research_messages(auto_research.episode_id, root.operation_id) == [
        messages[2]
    ]

    retry_wake_id = deliver_pending_auto_research_mail(
        tasks,
        episode_id=auto_research.episode_id,
        recipient_task_id=root.operation_id,
    )

    assert retry_wake_id is not None
    wait_for_task(store, retry_wake_id, expect="succeeded")
    retried = store.auto_research_message(messages[2].message_id)
    assert retried is not None
    assert retried.delivery_operation_id == retry_wake_id
    assert store.pending_auto_research_messages(auto_research.episode_id, root.operation_id) == []


def test_busy_auto_research_actor_leaves_pending_mail_unclaimed_and_unspent(tmp_path) -> None:
    store = _store(tmp_path)
    stage = tmp_path / "auto_research-stage"
    stage.mkdir()
    busy_entered = threading.Event()
    release_busy = threading.Event()

    async def stream(_project_id, _kind, request, execution):
        if execution.continuation == "fresh":
            execution.checkpoint_stage("execution-host", str(stage))
        yield _sse(
            AgentEvent(
                event="session",
                session_id=request.session_id or "orchestrator-session",
            )
        )
        if execution.operation_id == "busy-turn":
            busy_entered.set()
            while not release_busy.is_set():
                await asyncio.sleep(0.01)
        yield _sse(AgentEvent(event="done"))

    tasks = BackgroundAgentTasks(store, stream)
    auto_research, root = _start_auto_research(tasks)
    busy = start_auto_research_turn(
        tasks,
        auto_research.episode_id,
        AutoResearchRunRequest(
            episode_id=auto_research.episode_id,
            role="orchestrator",
            provider="codex",
            run_on="local",
            instruction="Keep the actor occupied while mail arrives.",
        ),
        parent_operation_id=root.operation_id,
        operation_id="busy-turn",
    )
    assert busy is not None
    assert busy_entered.wait(timeout=2)
    message = record_auto_research_message(
        store,
        episode_id=auto_research.episode_id,
        sender_role="human",
        sender_task_id=None,
        authorized_by=auto_research.authorized_by,
        recipient_task_id=root.operation_id,
        body="Wait until the current turn settles.",
    )
    before = store.episode_budget_meter(auto_research.episode_id)

    try:
        assert (
            deliver_pending_auto_research_mail(
                tasks,
                episode_id=auto_research.episode_id,
                recipient_task_id=root.operation_id,
            )
            is None
        )
        assert store.episode_budget_meter(auto_research.episode_id) == before
        assert store.pending_auto_research_messages(
            auto_research.episode_id, root.operation_id
        ) == [message]
        assert store.auto_research_messages(auto_research.episode_id) == [message]
    finally:
        release_busy.set()
        wait_for_task(store, busy.operation_id, expect="succeeded")


def test_not_yet_checkpointed_auto_research_actor_leaves_mail_pending(tmp_path) -> None:
    store = _store(tmp_path)

    async def stream(_project_id, _kind, request, _execution):
        yield _sse(
            AgentEvent(
                event="session",
                session_id=request.session_id or "orchestrator-session",
            )
        )
        yield _sse(AgentEvent(event="done"))

    tasks = BackgroundAgentTasks(store, stream)
    auto_research, root = _start_auto_research(tasks)
    assert root.native_session_id == "orchestrator-session"
    assert root.stage_root is None
    message = record_auto_research_message(
        store,
        episode_id=auto_research.episode_id,
        sender_role="human",
        sender_task_id=None,
        authorized_by=auto_research.authorized_by,
        recipient_task_id=root.operation_id,
        body="Deliver this after a stage checkpoint exists.",
    )
    before = store.episode_budget_meter(auto_research.episode_id)

    assert (
        deliver_pending_auto_research_mail(
            tasks,
            episode_id=auto_research.episode_id,
            recipient_task_id=root.operation_id,
        )
        is None
    )
    assert store.episode_budget_meter(auto_research.episode_id) == before
    assert store.pending_auto_research_messages(auto_research.episode_id, root.operation_id) == [
        message
    ]
    assert store.auto_research_messages(auto_research.episode_id) == [message]
