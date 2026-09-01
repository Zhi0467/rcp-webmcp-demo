from __future__ import annotations

import hashlib
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

import pytest

from rcp.agents import AgentEvent
from rcp.agents.command_protocol import (
    ApplyArguments,
    EpisodeCommandRequest,
    ExperimentKickoffArguments,
    FinishCommandRequest,
    InboxClearArguments,
    InboxCommandRequest,
    InboxHarvestArguments,
    MessageArguments,
    MessageCommandRequest,
    ResumeCommandRequest,
    SpawnArguments,
    SpawnCommandRequest,
    StatusArguments,
    StopCommandRequest,
    WatchGraphArguments,
    WatchGraphCommandRequest,
)
from rcp.background import BackgroundAgentTasks
from rcp.core.authority import AgentDispatchAuthority, AgentDispatchScope
from rcp.core.models import Blocker, GraphState
from rcp.core.transition_models import GraphHeadRef, GraphTargetRef
from rcp.limits import AGENT_COMMAND_EVENT_MAX_BYTES
from rcp.runs.auto_research import (
    AutoResearchCommandContext,
    AutoResearchCommandDispatcher,
    AutoResearchCommandEffectResult,
    AutoResearchCommandFile,
    AutoResearchRunRequest,
    request_auto_research_stop,
)
from rcp.runs.auto_research_admission import (
    AutoResearchChildResumeResult,
    pause_auto_research_child_work,
    start_auto_research_child_work,
    stop_auto_research_child_work,
)
from rcp.runs.auto_research_delivery import record_auto_research_message
from rcp.runs.auto_research_effects import auto_research_command_effects
from rcp.service import GraphUpdateResult, RunRequest, resolve_dispatch_authority
from rcp.storage import (
    AgentTaskRecord,
    AppStore,
    AutoResearchChildAdmissionRecord,
    AutoResearchChildWorkRecord,
    AutoResearchExperimentAllowance,
    AutoResearchExperimentAllowanceReached,
    AutoResearchLifecycleNoticeRecord,
    AutoResearchStateRecord,
    EpisodeRecord,
    GraphWatcherRecord,
    NodeStatusGraphCondition,
    ProjectRecord,
)
from rcp.storage.auto_research_children import auto_research_inbox_projection

from .helpers import fabricated_authorizer, wait_for_task

MAILBOX_ID = "a" * 32
CREDENTIAL = "b" * 64
_RUN_TRUTH_SCOPE = ["repo-a"]


def _insert_unbounded_legacy_lifecycle_notice(
    store: AppStore,
    record: AutoResearchLifecycleNoticeRecord,
) -> AutoResearchLifecycleNoticeRecord:
    """Simulate a notice persisted before per-command response bounds existed."""

    with store.connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            INSERT INTO auto_research_lifecycle_notices (
                notice_id, episode_id, source_kind, source_id, source_event,
                source_attempt, state, payload_json, created_at, delivered_at,
                delivery_operation_id, acknowledged_at, acknowledged_by
            ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, NULL, NULL, NULL, NULL)
            """,
            (
                record.notice_id,
                record.episode_id,
                record.source_kind,
                record.source_id,
                record.source_event,
                record.source_attempt,
                json.dumps(record.payload, sort_keys=True, separators=(",", ":")),
                record.created_at,
            ),
        )
    return record


def _auto_research_authority(episode_id: str, role: str) -> AgentDispatchAuthority:
    return AgentDispatchAuthority(
        profile="orchestrator" if role == "orchestrator" else "ordinary",
        task_contract="orchestrate" if role == "orchestrator" else "work_auto",
        scope=AgentDispatchScope(
            run_truth_scope=_RUN_TRUTH_SCOPE,
            episode_id=episode_id,
            patch_kind="work",
        ),
    )


def _sse(event: AgentEvent) -> str:
    return f"data: {event.model_dump_json()}\n\n"


async def _successful_stream(_project_id, _kind, _request, _execution):
    yield _sse(AgentEvent(event="done"))


def _setup_auto_research(tmp_path) -> tuple[AppStore, EpisodeRecord, AgentTaskRecord]:
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
    now = store.now()
    authorizer = fabricated_authorizer()
    graph_target = GraphTargetRef(kind="branch", branch_id="auto_research")
    request = AutoResearchRunRequest(
        episode_id="auto_research",
        role="orchestrator",
        actor_operation_id="root",
        provider="codex",
        run_on="local",
        run_truth_scope=_RUN_TRUTH_SCOPE,
    )
    episode, root = store.create_auto_research_episode_with_root_task(
        EpisodeRecord(
            episode_id="auto_research",
            project_id="project",
            mode="auto_research",
            graph_target=graph_target,
            graph_base_head=GraphHeadRef(revision=0),
            status="queued",
            invocation_ceiling=8,
            authorized_by=authorizer,
            created_at=now,
            updated_at=now,
        ),
        AutoResearchStateRecord(
            episode_id="auto_research",
            starting_instruction=None,
            created_at=now,
            updated_at=now,
        ),
        AgentTaskRecord(
            operation_id="root",
            project_id="project",
            episode_id="auto_research",
            graph_target=graph_target,
            kind="auto_research",
            status="queued",
            request=request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="orchestrator ready",
            authorized_by=authorizer,
            dispatch_authority=_auto_research_authority("auto_research", "orchestrator"),
        ),
    )
    store.complete_agent_task(root.operation_id, applied_revision=None, result={})
    stored_root = store.agent_task(root.operation_id)
    assert stored_root is not None
    return store, episode, stored_root


def _context(
    store: AppStore,
    auto_research: EpisodeRecord,
    task: AgentTaskRecord,
    *,
    command_file: AutoResearchCommandFile | None = None,
) -> AutoResearchCommandContext:
    current = store.episode(auto_research.episode_id)
    stored_task = store.agent_task(task.operation_id)
    assert current is not None and stored_task is not None
    return AutoResearchCommandContext(
        episode=current,
        task=stored_task,
        request=AutoResearchRunRequest.model_validate(stored_task.request),
        command_file=command_file,
    )


def _worker_request(
    _context: AutoResearchCommandContext,
    arguments: SpawnArguments,
    instruction: str,
    worker_id: str,
) -> RunRequest:
    return RunRequest(
        provider="codex",
        run_on="local",
        run_truth_scope=_RUN_TRUTH_SCOPE,
        chat_id=worker_id,
        chat_scope="node",
        node_id=arguments.seat_node_id,
        message=instruction,
        mode="work",
        trigger="orchestrator",
        patch_kind="work",
    )


def _create_worker(
    store: AppStore,
    auto_research: EpisodeRecord,
    parent: AgentTaskRecord,
    *,
    operation_id: str = "worker",
    status: str = "succeeded",
    native_session_id: str | None = None,
    stage_root: str | None = None,
) -> AgentTaskRecord:
    request = AutoResearchRunRequest(
        episode_id=auto_research.episode_id,
        role="worker",
        actor_operation_id=operation_id,
        provider="codex",
        run_on="local",
        run_truth_scope=_RUN_TRUTH_SCOPE,
        control_node_id="blk/check",
        instruction="Resolve the blocker.",
    )
    now = store.now()
    return store.create_auto_research_agent_task(
        AgentTaskRecord(
            operation_id=operation_id,
            project_id=auto_research.project_id,
            episode_id=auto_research.episode_id,
            graph_target=auto_research.graph_target,
            kind="auto_research",
            status=status,
            request=request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message=f"{operation_id} {status}",
            parent_operation_id=parent.operation_id,
            native_session_id=native_session_id,
            stage_root=stage_root,
            authorized_by=auto_research.authorized_by,
            dispatch_authority=_auto_research_authority(auto_research.episode_id, "worker"),
        ),
        role="worker",
    )


def _create_routed_worker(
    store: AppStore,
    auto_research: EpisodeRecord,
    parent: AgentTaskRecord,
    *,
    operation_id: str = "worker",
    status: str = "succeeded",
    native_session_id: str | None = None,
    stage_root: str | None = None,
) -> AgentTaskRecord:
    instruction = "Resolve the blocker."
    request = RunRequest(
        provider="codex",
        run_on="local",
        run_truth_scope=_RUN_TRUTH_SCOPE,
        chat_id=operation_id,
        chat_scope="node",
        node_id="blk/check",
        message=instruction,
        mode="work",
        trigger="orchestrator",
        patch_kind="work",
    )
    now = store.now()
    authority = resolve_dispatch_authority("node_chat", request)
    assert authority is not None
    route, task = store.create_auto_research_child_work(
        AutoResearchChildWorkRecord(
            worker_id=operation_id,
            episode_id=auto_research.episode_id,
            project_id=auto_research.project_id,
            control_node_id="blk/check",
            root_operation_id=operation_id,
            current_operation_id=operation_id,
            admitted_by_operation_id=parent.operation_id,
            instruction=instruction,
            instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
            created_at=now,
            updated_at=now,
        ),
        AgentTaskRecord(
            operation_id=operation_id,
            project_id=auto_research.project_id,
            episode_id=auto_research.episode_id,
            graph_target=auto_research.graph_target,
            kind="node_chat",
            status="queued",
            request=request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message=f"{operation_id} queued",
            native_session_id=native_session_id,
            stage_root=stage_root,
            authorized_by=auto_research.authorized_by,
            dispatch_authority=authority,
        ),
    )
    assert route.current_operation_id == task.operation_id
    if status == "paused":
        store.pause_agent_task(task.operation_id, detail=f"{operation_id} paused")
    elif status == "failed":
        store.fail_agent_task(task.operation_id, f"{operation_id} failed")
    elif status == "succeeded":
        store.complete_agent_task(task.operation_id, applied_revision=None, result={})
    elif status != "queued":
        raise AssertionError(f"unsupported test worker status: {status}")
    stored = store.agent_task(task.operation_id)
    assert stored is not None
    return stored


def _create_worker_recovery(
    store: AppStore,
    auto_research: EpisodeRecord,
    parent: AgentTaskRecord,
    *,
    operation_id: str = "worker-recovery",
) -> AgentTaskRecord:
    request = RunRequest.model_validate(parent.request).model_copy(
        update={"session_id": parent.native_session_id}
    )
    now = store.now()
    task = AgentTaskRecord(
        operation_id=operation_id,
        project_id=auto_research.project_id,
        episode_id=auto_research.episode_id,
        graph_target=auto_research.graph_target,
        kind="node_chat",
        status="queued",
        request=request.model_dump(mode="json"),
        created_at=now,
        updated_at=now,
        status_message="latest queued attempt",
        attempt=parent.attempt + 1,
        parent_operation_id=parent.operation_id,
        native_session_id=parent.native_session_id,
        stage_host=parent.stage_host,
        stage_root=parent.stage_root,
        authorized_by=auto_research.authorized_by,
        dispatch_authority=parent.dispatch_authority,
    )
    _, stored = store.create_auto_research_child_work_recovery("worker", task)
    store.pause_agent_task(stored.operation_id, detail="latest paused attempt")
    latest = store.agent_task(stored.operation_id)
    assert latest is not None
    return latest


def _blocker_state(*, status: str = "open", revision: int = 1) -> GraphState:
    blocker = Blocker(
        id="blk/check",
        type="blocker",
        title="Check the result",
        description="Resolve this after checking the external result.",
        status=status,
    )
    return GraphState(revision=revision, nodes={blocker.id: blocker})


def _effects(
    store,
    background,
    *,
    state=None,
    on_watcher_ready=None,
    apply_patch=None,
    on_graph_applied=None,
):
    return auto_research_command_effects(
        store=store,
        background=background,
        validate=lambda _context, _arguments: AutoResearchCommandEffectResult(),
        worker_request_factory=_worker_request,
        graph_state=lambda: state or _blocker_state(),
        execution_host="execution.example",
        apply_patch=apply_patch,
        on_graph_applied=on_graph_applied,
        on_watcher_ready=on_watcher_ready,
    )


@dataclass
class _RecordingBackground:
    store: AppStore
    paused: list[str] = field(default_factory=list)
    resumed: list[str] = field(default_factory=list)
    stopped: list[str] = field(default_factory=list)

    def pause_auto_research_child_work(
        self,
        episode_id: str,
        worker_id: str,
    ) -> AgentTaskRecord:
        route = self.store.auto_research_child_work(worker_id)
        assert route is not None and route.episode_id == episode_id
        self.paused.append(route.current_operation_id)
        task = self.store.agent_task(route.current_operation_id)
        assert task is not None
        return task

    def resume_auto_research_child_work(
        self,
        episode_id: str,
        worker_id: str,
        *,
        operation_id: str | None = None,
    ) -> AutoResearchChildResumeResult:
        route = self.store.auto_research_child_work(worker_id)
        assert route is not None and route.episode_id == episode_id
        self.resumed.append(route.current_operation_id)
        task = self.store.agent_task(route.current_operation_id)
        assert task is not None
        assert operation_id is not None
        return AutoResearchChildResumeResult(
            disposition="resumed",
            child_kind="work",
            child_id=worker_id,
            current_operation_id=task.operation_id,
            task=task,
        )

    def stop_auto_research_child_work(
        self,
        episode_id: str,
        worker_id: str,
    ) -> AgentTaskRecord:
        route = self.store.auto_research_child_work(worker_id)
        assert route is not None and route.episode_id == episode_id
        self.stopped.append(route.current_operation_id)
        self.store.request_auto_research_child_work_stop(worker_id)
        task = self.store.agent_task(route.current_operation_id)
        assert task is not None
        return task


def _record_child_controls(monkeypatch, background: _RecordingBackground) -> None:
    """Route the child-work controls to the recorder.

    They are module functions taking the engine now, so a stub object with those
    method names no longer intercepts them.
    """

    for name in (
        "pause_auto_research_child_work",
        "resume_auto_research_child_work",
        "stop_auto_research_child_work",
    ):
        monkeypatch.setattr(
            f"rcp.runs.auto_research_effects.{name}",
            lambda _tasks, *args, _name=name, **kwargs: getattr(background, _name)(*args, **kwargs),
        )


def test_apply_records_valid_invalid_and_empty_graph_dispositions(tmp_path) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    consumed: list[tuple[str, str]] = []
    watcher_boundaries: list[str] = []

    def apply_patch(_context, _text, apply_id):
        if apply_id == "apply-applied":
            return GraphUpdateResult(status="applied", applied_revision=2), None, False
        if apply_id == "apply-invalid":
            return None, "The corrected node does not exist.", True
        if apply_id == "apply-empty":
            return GraphUpdateResult(status="none"), None, False
        if apply_id == "apply-unavailable":
            return None, "The canonical state host is temporarily unavailable.", False
        raise AssertionError(apply_id)

    effects = _effects(
        store,
        _RecordingBackground(store),
        state=_blocker_state(revision=2),
        apply_patch=apply_patch,
        on_graph_applied=lambda: watcher_boundaries.append("applied"),
    )

    outcomes = []
    for apply_id, text in (
        ("apply-applied", '{"ops":[{"op":"update_nodes"}]}'),
        ("apply-invalid", '{"ops":[{"op":"invalid"}]}'),
        ("apply-empty", '{"ops":[]}'),
        ("apply-unavailable", '{"ops":[{"op":"unavailable"}]}'),
    ):
        digest = hashlib.sha256(text.encode()).hexdigest()
        context = replace(
            _context(
                store,
                auto_research,
                root,
                command_file=AutoResearchCommandFile(
                    kind="apply",
                    filename="patch.json",
                    text=text,
                    sha256=digest,
                ),
            ),
            consume_command_file=lambda filename, sha256: (
                consumed.append((filename, sha256)) or True
            ),
            refresh_command_state=lambda: (2, "/state/graph.json", "/state/research.md"),
        )
        outcomes.append(
            effects.apply(
                context,
                ApplyArguments(patch_file="patch.json"),
                apply_id,
            )
        )

    assert [outcome.status for outcome in outcomes] == ["ok", "invalid", "ok", "unavailable"]
    assert len(consumed) == 2
    assert watcher_boundaries == ["applied"]
    records = store.auto_research_apply_results(root.operation_id)
    assert [record.apply_id for record in records] == [
        "apply-applied",
        "apply-invalid",
        "apply-empty",
    ]
    assert [record.result["result"]["graph_update"]["status"] for record in records] == [
        "applied",
        "rejected",
        "none",
    ]
    invalid_result = records[1].result["result"]
    assert invalid_result["disposition"] == "invalid"
    assert invalid_result["graph_update"]["repairable"] is True


def test_concurrent_same_apply_identity_returns_one_canonical_response(tmp_path) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    barrier = threading.Barrier(2)

    def apply_patch(_context, _text, _apply_id):
        barrier.wait()
        return GraphUpdateResult(status="none"), None, False

    effects = _effects(
        store,
        _RecordingBackground(store),
        apply_patch=apply_patch,
    )
    text = '{"ops":[]}'
    digest = hashlib.sha256(text.encode()).hexdigest()

    def invoke(consumed: bool) -> AutoResearchCommandEffectResult:
        context = replace(
            _context(
                store,
                auto_research,
                root,
                command_file=AutoResearchCommandFile(
                    kind="apply",
                    filename="patch.json",
                    text=text,
                    sha256=digest,
                ),
            ),
            consume_command_file=lambda _filename, _sha256: consumed,
        )
        return effects.apply(
            context,
            ApplyArguments(patch_file="patch.json"),
            "concurrent-apply",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(invoke, True)
        second = executor.submit(invoke, False)
        outcomes = [first.result(), second.result()]

    assert outcomes[0] == outcomes[1]
    assert outcomes[0].status == "ok"
    records = store.auto_research_apply_results(root.operation_id)
    assert len(records) == 1
    assert records[0].apply_id == "concurrent-apply"
    assert records[0].result == outcomes[0].model_dump(mode="json")


def test_status_and_controls_resolve_the_latest_canonical_worker_leaf(
    tmp_path, monkeypatch
) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    stage = tmp_path / "worker-stage"
    stage.mkdir()
    worker = _create_routed_worker(
        store,
        auto_research,
        root,
        status="paused",
        native_session_id="worker-session",
        stage_root=str(stage),
    )
    latest = _create_worker_recovery(store, auto_research, worker)
    background = _RecordingBackground(store)
    _record_child_controls(monkeypatch, background)
    effects = _effects(store, background)
    context = _context(store, auto_research, root)

    status = effects.status(context, StatusArguments(worker_id=worker.operation_id))
    paused = effects.pause(context, worker.operation_id)

    assert status.message == "Auto-research status is current."
    assert status.result["episode"]["status"] == "running"  # type: ignore[index]
    assert status.result["budget"]["invocations_used"] == 2  # type: ignore[index]
    assert "report_units_reserved" not in status.result["budget"]  # type: ignore[operator]
    assert status.result["worker"] == {
        "worker_id": worker.operation_id,
        "current_operation_id": latest.operation_id,
        "control_node_id": "blk/check",
        "status": "paused",
        "status_message": "latest paused attempt",
        "stop_requested": False,
        "can_pause": False,
        "can_resume": True,
    }
    assert background.paused == [latest.operation_id]
    assert paused.result["current_operation_id"] == latest.operation_id
    assert "current worker task attempt" in (paused.message or "")


def test_worker_pause_resignals_pausing_and_accepts_an_already_paused_attempt(
    tmp_path,
    monkeypatch,
) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    background = BackgroundAgentTasks(store, _successful_stream)
    worker = _create_routed_worker(store, auto_research, root, status="queued")
    signals: list[str] = []
    monkeypatch.setattr(background, "_signal_agent_task_pause", signals.append)

    first = pause_auto_research_child_work(
        background,
        auto_research.episode_id,
        worker.operation_id,
    )
    second = pause_auto_research_child_work(
        background,
        auto_research.episode_id,
        worker.operation_id,
    )
    store.pause_agent_task(worker.operation_id, detail="paused at the checkpoint")
    third = pause_auto_research_child_work(
        background,
        auto_research.episode_id,
        worker.operation_id,
    )

    assert first.status == "pausing"
    assert second.status == "pausing"
    assert third.status == "paused"
    assert signals == [worker.operation_id, worker.operation_id]


def test_worker_stop_recovers_the_split_after_durable_route_stop(
    tmp_path,
    monkeypatch,
) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    background = BackgroundAgentTasks(store, _successful_stream)
    worker = _create_routed_worker(store, auto_research, root, status="queued")
    signals: list[str] = []
    monkeypatch.setattr(background, "_signal_agent_task_pause", signals.append)

    # This is the crash boundary: Stop intent committed, process signal not sent.
    store.request_auto_research_child_work_stop(worker.operation_id)
    recovered = stop_auto_research_child_work(
        background,
        auto_research.episode_id,
        worker.operation_id,
    )
    replayed = stop_auto_research_child_work(
        background,
        auto_research.episode_id,
        worker.operation_id,
    )
    store.pause_agent_task(worker.operation_id, detail="stopped at the checkpoint")
    settled = stop_auto_research_child_work(
        background,
        auto_research.episode_id,
        worker.operation_id,
    )

    route = store.auto_research_child_work(worker.operation_id)
    assert route is not None and route.stop_requested_at is not None
    assert recovered.status == "pausing"
    assert replayed.status == "pausing"
    assert settled.status == "paused"
    assert signals == [worker.operation_id, worker.operation_id]


def test_resume_uses_the_latest_leaf_exact_session_without_spending_another_unit(
    tmp_path,
) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    stage = tmp_path / "worker-stage"
    stage.mkdir()
    worker = _create_routed_worker(
        store,
        auto_research,
        root,
        status="paused",
        native_session_id="worker-session",
        stage_root=str(stage),
    )
    background = BackgroundAgentTasks(store, _successful_stream)
    effects = _effects(store, background)
    before = store.episode_budget_meter(auto_research.episode_id)

    outcome = effects.resume(
        _context(store, auto_research, root),
        worker.operation_id,
        str(uuid.uuid4()),
    )

    resumed = store.agent_task(str(outcome.result["current_operation_id"]))
    assert resumed is not None
    assert resumed.parent_operation_id == worker.operation_id
    assert resumed.native_session_id == "worker-session"
    assert resumed.stage_root == str(stage)
    assert resumed.request["session_id"] == "worker-session"
    assert store.episode_budget_meter(auto_research.episode_id) == before
    wait_for_task(store, resumed.operation_id)


def test_resume_without_a_usable_checkpoint_returns_spawn_replacement_without_spend(
    tmp_path,
) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    worker = _create_routed_worker(store, auto_research, root, status="failed")
    background = BackgroundAgentTasks(store, _successful_stream)
    effects = _effects(store, background)
    before = store.episode_budget_meter(auto_research.episode_id)

    outcome = effects.resume(
        _context(store, auto_research, root),
        worker.operation_id,
        str(uuid.uuid4()),
    )

    assert outcome.status == "invalid"
    assert outcome.result == {
        "disposition": "resume_unavailable",
        "worker_id": worker.operation_id,
        "current_operation_id": worker.operation_id,
        "reason": "the attempt has no complete RCP-owned session and stage",
        "replacement_command": "spawn",
    }
    assert store.episode_budget_meter(auto_research.episode_id) == before


def test_same_key_worker_resume_recovers_a_committed_task_without_creating_another(
    tmp_path,
    monkeypatch,
) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    stage = tmp_path / "worker-stage"
    stage.mkdir()
    worker = _create_routed_worker(
        store,
        auto_research,
        root,
        status="paused",
        native_session_id="worker-session",
        stage_root=str(stage),
    )
    executions: list[str] = []

    async def stream(_project_id, _kind, _request, execution):
        executions.append(execution.continuation)
        yield _sse(AgentEvent(event="done"))

    background = BackgroundAgentTasks(store, stream)
    effects = _effects(store, background)
    key = "worker-resume-crash-boundary"
    expected_operation_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rcp:auto_research:{auto_research.episode_id}:resume:{key}",
        )
    )
    request = ResumeCommandRequest(
        mailbox_id=MAILBOX_ID,
        request_id="1" * 32,
        credential=CREDENTIAL,
        verb="resume",
        idempotency_key=key,
        arguments={"worker_id": worker.operation_id},
    )
    real_spawn_record = background._spawn_record

    def crash_before_spawn(*_args, **_kwargs):
        raise RuntimeError("simulated crash after recovery commit")

    before = store.episode_budget_meter(auto_research.episode_id)
    monkeypatch.setattr(background, "_spawn_record", crash_before_spawn)
    with pytest.raises(RuntimeError, match="simulated crash"):
        AutoResearchCommandDispatcher(store, effects).dispatch(root.operation_id, request)

    committed = store.agent_task(expected_operation_id)
    assert committed is not None and committed.status == "queued"
    assert store.agent_task_continuation_cause(expected_operation_id) == "resume"
    admitted = [
        receipt
        for receipt in store.agent_task_receipts(expected_operation_id)
        if receipt.category == "operation_admitted"
    ]
    assert len(admitted) == 1
    assert admitted[0].payload["continuation_cause"] == "resume"
    assert admitted[0].payload["admission_committed"] is True
    monkeypatch.setattr(background, "_spawn_record", real_spawn_record)
    replay = AutoResearchCommandDispatcher(store, effects).dispatch(
        root.operation_id,
        request.model_copy(update={"request_id": "2" * 32}),
    )

    assert replay.status == "ok"
    assert replay.result["current_operation_id"] == expected_operation_id
    assert store.episode_budget_meter(auto_research.episode_id) == before
    recoveries = [
        task
        for task in store.episode_tasks(auto_research.episode_id)
        if task.parent_operation_id == worker.operation_id
    ]
    assert [task.operation_id for task in recoveries] == [expected_operation_id]
    wait_for_task(store, expected_operation_id, expect="succeeded")
    replay_again = AutoResearchCommandDispatcher(store, effects).dispatch(
        root.operation_id,
        request.model_copy(update={"request_id": "3" * 32}),
    )
    assert replay_again.status == "ok"
    assert executions == ["resume"]


def test_same_key_spawn_dispatches_a_worker_committed_before_process_launch(
    tmp_path,
    monkeypatch,
) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    executions: list[str] = []

    async def stream(_project_id, _kind, _request, execution):
        executions.append(execution.continuation)
        yield _sse(AgentEvent(event="done"))

    background = BackgroundAgentTasks(store, stream)
    effects = _effects(store, background)
    instruction = "Resolve the blocker through the committed worker route."
    key = "worker-fresh-crash-boundary"
    request = SpawnCommandRequest(
        mailbox_id=MAILBOX_ID,
        request_id="4" * 32,
        credential=CREDENTIAL,
        verb="spawn",
        idempotency_key=key,
        arguments={
            "seat_node_id": "blk/check",
            "instruction_file": "worker.md",
        },
    )
    dispatcher = AutoResearchCommandDispatcher(
        store,
        effects,
        command_file_reader=lambda _filename, _max_bytes: instruction,
    )
    expected_worker_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rcp:auto_research:{auto_research.episode_id}:spawn:{key}",
        )
    )
    real_spawn_record = background._spawn_record

    def crash_before_spawn(*_args, **_kwargs):
        raise RuntimeError("simulated crash after fresh Work commit")

    monkeypatch.setattr(background, "_spawn_record", crash_before_spawn)
    with pytest.raises(RuntimeError, match="after fresh Work commit"):
        dispatcher.dispatch(root.operation_id, request)

    committed = store.agent_task(expected_worker_id)
    assert committed is not None and committed.status == "queued"
    assert executions == []
    monkeypatch.setattr(background, "_spawn_record", real_spawn_record)

    replay = dispatcher.dispatch(
        root.operation_id,
        request.model_copy(update={"request_id": "5" * 32}),
    )
    assert replay.status == "ok"
    assert replay.result["worker_id"] == expected_worker_id
    wait_for_task(store, expected_worker_id, expect="succeeded")
    replay_again = dispatcher.dispatch(
        root.operation_id,
        request.model_copy(update={"request_id": "6" * 32}),
    )
    assert replay_again.status == "ok"
    assert executions == ["fresh"]


def test_transient_spawn_failure_keeps_admission_for_same_key_recovery(
    tmp_path,
    monkeypatch,
) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    executions: list[str] = []

    async def stream(_project_id, _kind, _request, execution):
        executions.append(execution.continuation)
        yield _sse(AgentEvent(event="done"))

    background = BackgroundAgentTasks(store, stream)
    effects = _effects(store, background)
    instruction = "Recover this exact Spawn after transient canonical-state unavailability."
    request = SpawnCommandRequest(
        mailbox_id=MAILBOX_ID,
        request_id="7" * 32,
        credential=CREDENTIAL,
        verb="spawn",
        idempotency_key="transient-spawn-same-key",
        arguments={
            "seat_node_id": "blk/check",
            "instruction_file": "worker.md",
        },
    )
    dispatcher = AutoResearchCommandDispatcher(
        store,
        effects,
        command_file_reader=lambda _filename, _max_bytes: instruction,
    )
    real_start = start_auto_research_child_work

    def unavailable(*_args, **_kwargs):
        raise OSError("canonical state is temporarily unavailable")

    before = store.episode_budget_meter(auto_research.episode_id)
    monkeypatch.setattr(
        "rcp.runs.auto_research_effects.start_auto_research_child_work", unavailable
    )
    first = dispatcher.dispatch(root.operation_id, request)
    assert first.status == "unavailable"
    command = store.agent_command_by_key(
        auto_research.episode_id,
        request.idempotency_key,
    )
    assert command is not None
    worker_id = str(command.start_payload["planned_worker_id"])
    admission = store.auto_research_child_admission(worker_id)
    assert admission is not None and admission.state == "accepted"
    assert store.auto_research_child_work(worker_id) is None
    assert store.episode_budget_meter(auto_research.episode_id) == before

    monkeypatch.setattr("rcp.runs.auto_research_effects.start_auto_research_child_work", real_start)
    replay = dispatcher.dispatch(
        root.operation_id,
        request.model_copy(update={"request_id": "8" * 32}),
    )

    assert replay.status == "ok"
    assert replay.result["worker_id"] == worker_id
    wait_for_task(store, worker_id, expect="succeeded")
    admission = store.auto_research_child_admission(worker_id)
    assert admission is not None and admission.state == "reflected"
    assert store.episode_budget_meter(auto_research.episode_id).invocations_used == (
        before.invocations_used + 1
    )
    assert executions == ["fresh"]


def test_spawn_uses_the_planned_id_and_ordinary_work_route(tmp_path) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    background = BackgroundAgentTasks(store, _successful_stream)
    effects = _effects(store, background)
    arguments = SpawnArguments(
        seat_node_id="blk/check",
        instruction_file="worker-task.md",
    )
    instruction = "Resolve the blocker."
    planned_worker_id = str(uuid.uuid4())
    digest = hashlib.sha256(instruction.encode()).hexdigest()
    now = store.now()
    store.record_auto_research_child_admission(
        AutoResearchChildAdmissionRecord(
            admission_id=planned_worker_id,
            episode_id=auto_research.episode_id,
            project_id=auto_research.project_id,
            child_kind="work",
            child_id=planned_worker_id,
            state="accepted",
            created_at=now,
            updated_at=now,
        )
    )
    context = _context(
        store,
        auto_research,
        root,
        command_file=AutoResearchCommandFile(
            kind="instruction",
            filename=arguments.instruction_file,
            text=instruction,
            sha256=digest,
        ),
    )
    before = store.episode_budget_meter(auto_research.episode_id)

    outcome = effects.spawn(context, arguments, planned_worker_id)

    worker = store.agent_task(planned_worker_id)
    assert worker is not None
    assert outcome.result["worker_id"] == planned_worker_id
    assert outcome.result["current_operation_id"] == worker.operation_id
    assert worker.kind == "node_chat"
    assert worker.parent_operation_id is None
    assert worker.request["chat_id"] == planned_worker_id
    assert worker.request["message"] == instruction
    assert worker.request["mode"] == "work"
    assert worker.request["trigger"] == "orchestrator"
    assert worker.request["provider"] == "codex"
    assert worker.request["run_on"] == "local"
    route = store.auto_research_child_work(planned_worker_id)
    assert route is not None
    assert route.admitted_by_operation_id == root.operation_id
    assert route.instruction_sha256 == digest
    assert store.auto_research_invocation_role(worker.operation_id) is None
    assert store.episode_budget_meter(auto_research.episode_id).invocations_used == (
        before.invocations_used + 1
    )
    wait_for_task(store, worker.operation_id)


def test_message_is_durable_hearsay_when_the_recipient_cannot_wake_yet(tmp_path) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    worker = _create_worker(store, auto_research, root)
    background = BackgroundAgentTasks(store, _successful_stream)
    effects = _effects(store, background)
    before = store.episode_budget_meter(auto_research.episode_id)
    planned_message_id = str(uuid.uuid4())

    outcome = effects.message(
        _context(store, auto_research, root),
        MessageArguments(
            recipient_task_id=worker.operation_id,
            body="Check the new result, but treat this note as hearsay.",
        ),
        planned_message_id,
    )

    assert outcome.result["message_id"] == planned_message_id
    assert outcome.result["disposition"] == "created"
    assert outcome.result["delivery"] == "pending"
    assert outcome.result["graph_authority"] == "none"
    assert outcome.result["epistemic_status"] == "hearsay"
    pending = store.pending_auto_research_messages(auto_research.episode_id, worker.operation_id)
    assert [message.message_id for message in pending] == [outcome.result["message_id"]]
    assert pending[0].body == "Check the new result, but treat this note as hearsay."
    assert pending[0].sender_task_id == root.operation_id
    assert pending[0].authorized_by is None
    assert store.episode_budget_meter(auto_research.episode_id) == before


def test_new_message_stays_pending_when_an_older_bounded_batch_starts(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("rcp.runs.auto_research_mail.AUTO_RESEARCH_MAIL_MAX_MESSAGES", 2)
    store, auto_research, root = _setup_auto_research(tmp_path)
    stage = tmp_path / "worker-stage"
    stage.mkdir()
    worker = _create_worker(
        store,
        auto_research,
        root,
        native_session_id="worker-session",
        stage_root=str(stage),
    )
    older = [
        record_auto_research_message(
            store,
            episode_id=auto_research.episode_id,
            sender_role="orchestrator",
            sender_task_id=root.operation_id,
            authorized_by=None,
            recipient_task_id=worker.operation_id,
            body=f"Older pending message {index}.",
        )
        for index in range(2)
    ]
    background = BackgroundAgentTasks(store, _successful_stream)
    effects = _effects(store, background)
    planned_message_id = str(uuid.uuid4())

    outcome = effects.message(
        _context(store, auto_research, root),
        MessageArguments(
            recipient_task_id=worker.operation_id,
            body="New message behind the bounded prefix.",
        ),
        planned_message_id,
    )

    canonical = store.auto_research_message(planned_message_id)
    assert canonical is not None
    assert canonical.delivery_operation_id is None
    assert outcome.result["message_id"] == planned_message_id
    assert outcome.result["delivery"] == "pending"
    assert outcome.result["delivery_operation_id"] is None
    assert outcome.message == "The message was queued behind an older pending delivery."
    claimed = [store.auto_research_message(message.message_id) for message in older]
    assert all(message is not None for message in claimed)
    delivery_ids = {message.delivery_operation_id for message in claimed if message is not None}
    assert len(delivery_ids) == 1
    delivery_operation_id = delivery_ids.pop()
    assert delivery_operation_id is not None
    wait_for_task(store, delivery_operation_id)


def test_watch_graph_uses_live_state_and_the_explicit_execution_host(tmp_path) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    background = BackgroundAgentTasks(store, _successful_stream)
    ready: list[str] = []
    calls = 0

    def live_state() -> GraphState:
        nonlocal calls
        calls += 1
        return _blocker_state(status="resolved", revision=2)

    effects = auto_research_command_effects(
        store=store,
        background=background,
        validate=lambda _context, _arguments: AutoResearchCommandEffectResult(),
        worker_request_factory=_worker_request,
        graph_state=live_state,
        execution_host="ssh.execution.example",
        on_watcher_ready=ready.append,
    )
    context = _context(store, auto_research, root)
    planned_watcher_id = str(uuid.uuid4())

    assert effects.seat_node_type(auto_research.project_id, "blk/check") == "blocker"
    outcome = effects.watch_graph(
        context,
        WatchGraphArguments(
            condition=NodeStatusGraphCondition(
                node_id="blk/check",
                status_in=["resolved"],
            ),
            reason="Continue after the canonical blocker is resolved.",
        ),
        planned_watcher_id,
    )

    assert outcome.result["watcher_id"] == planned_watcher_id
    assert outcome.result["disposition"] == "created"
    watcher = store.watcher(str(outcome.result["watcher_id"]))
    assert isinstance(watcher, GraphWatcherRecord)
    assert watcher.execution_host == "ssh.execution.example"
    assert watcher.status == "completed"
    assert outcome.result["completed_immediately"] is True
    assert ready == [auto_research.project_id]
    assert calls == 2


def test_graph_watcher_born_after_stop_reports_its_durable_stopped_status(tmp_path) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    background = BackgroundAgentTasks(store, _successful_stream)
    effects = _effects(store, background, state=_blocker_state(status="resolved"))
    request_auto_research_stop(store, auto_research.episode_id)

    outcome = effects.watch_graph(
        _context(store, auto_research, root),
        WatchGraphArguments(
            condition=NodeStatusGraphCondition(
                node_id="blk/check",
                status_in=["resolved"],
            ),
            reason="Retain this condition under the existing Stop intent.",
        ),
        str(uuid.uuid4()),
    )

    watcher = store.watcher(str(outcome.result["watcher_id"]))
    assert isinstance(watcher, GraphWatcherRecord)
    assert watcher.status == "stopped"
    assert outcome.result["status"] == "stopped"
    assert outcome.result["completed_immediately"] is False


def test_finish_fences_completed_and_unknown_reconciliation_never_reexecutes(tmp_path) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    background = BackgroundAgentTasks(store, _successful_stream)
    effects = _effects(store, background)
    context = _context(store, auto_research, root)
    request = FinishCommandRequest(
        mailbox_id="a" * 32,
        request_id="f" * 32,
        credential="b" * 64,
        verb="finish",
        idempotency_key="finish-once",
    )

    effect_id = str(uuid.uuid4())
    finished = effects.finish(context, effect_id)
    reconciled = effects.reconcile_unknown(context, request, effect_id)

    assert finished.status == "ok"
    assert reconciled is not None
    assert reconciled.status == "ok"
    assert reconciled.result == finished.result
    assert reconciled.result["disposition"] == "completed"
    fenced = store.episode(auto_research.episode_id)
    assert fenced is not None
    assert (fenced.status, fenced.ending) == ("wrapping_up", "completed")


def test_finish_receipt_commit_before_command_exit_reconciles_without_refencing(
    tmp_path,
    monkeypatch,
) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    background = BackgroundAgentTasks(store, _successful_stream)
    effects = _effects(store, background)
    key = "finish-crash-window"
    effect_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rcp:auto_research:{auto_research.episode_id}:finish:{key}",
        )
    )
    request = FinishCommandRequest(
        mailbox_id=MAILBOX_ID,
        request_id="c" * 32,
        credential=CREDENTIAL,
        verb="finish",
        idempotency_key=key,
    )
    store.start_agent_command(
        operation_id=root.operation_id,
        command_id="interrupted-finish",
        episode_id=auto_research.episode_id,
        verb="finish",
        idempotency_key=key,
        payload={
            "request_id": request.request_id,
            "arguments": {},
            "planned_finish_effect_id": effect_id,
        },
    )
    committed = effects.finish(_context(store, auto_research, root), effect_id)

    def fail_if_refenced(*_args, **_kwargs):
        raise AssertionError("reconciliation must hydrate the committed receipt")

    monkeypatch.setattr(store, "guard_auto_research_finish", fail_if_refenced)
    response = AutoResearchCommandDispatcher(store, effects).dispatch(
        root.operation_id,
        request.model_copy(update={"request_id": "b" * 32}),
    )

    original = store.agent_command("interrupted-finish")
    assert original is not None and original.status == "ok"
    assert original.exit_payload is not None
    assert original.exit_payload["result"] == committed.result
    assert response.status == "ok"
    assert response.result == {
        "episode_id": auto_research.episode_id,
        "status": "wrapping_up",
        "ending": "completed",
    }


def test_finish_returns_complete_large_blocker_snapshot_with_compact_durable_exit(
    tmp_path,
) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    background = BackgroundAgentTasks(store, _successful_stream)
    dispatcher = AutoResearchCommandDispatcher(store, _effects(store, background))
    for index in range(500):
        store.record_auto_research_lifecycle_notice(
            AutoResearchLifecycleNoticeRecord(
                notice_id=f"notice-{index:04d}",
                episode_id=auto_research.episode_id,
                source_kind="worker",
                source_id=f"worker-{index:04d}",
                source_event="settled",
                payload={"status": "failed"},
                created_at=store.now(),
            )
        )
    request = FinishCommandRequest(
        mailbox_id=MAILBOX_ID,
        request_id="f" * 32,
        credential=CREDENTIAL,
        verb="finish",
        idempotency_key="finish-large-snapshot",
    )

    first = dispatcher.dispatch(root.operation_id, request)
    invocation = store.agent_command_by_key(
        auto_research.episode_id,
        "finish-large-snapshot",
    )
    assert invocation is not None and invocation.exit_payload is not None
    original_result = first.result
    store.clear_auto_research_lifecycle_notices(
        auto_research.episode_id,
        acknowledged_by=root.operation_id,
    )
    replay = dispatcher.dispatch(
        root.operation_id,
        request.model_copy(update={"request_id": "e" * 32}),
    )
    completed = dispatcher.dispatch(
        root.operation_id,
        request.model_copy(
            update={
                "request_id": "d" * 32,
                "idempotency_key": "finish-after-clear",
            }
        ),
    )

    assert first.status == replay.status == "invalid"
    assert first.message == (
        "Auto-research has 500 unsettled obligations; settle them, then call finish with a new key."
    )
    assert len(first.result["blockers"]) == 500
    assert replay.result == original_result
    assert len(first.model_dump_json().encode("utf-8")) > 64 * 1024
    assert (
        len(
            json.dumps(
                invocation.exit_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        <= AGENT_COMMAND_EVENT_MAX_BYTES
    )
    assert "blockers" not in invocation.exit_payload["result"]
    assert completed.status == "ok"
    assert completed.result == {
        "episode_id": auto_research.episode_id,
        "status": "wrapping_up",
        "ending": "completed",
    }


def test_inbox_receipts_reconcile_exact_snapshots_without_acknowledging_later_notices(
    tmp_path,
) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    background = BackgroundAgentTasks(store, _successful_stream)
    effects = _effects(store, background)
    context = _context(store, auto_research, root)

    def notice(notice_id: str, attempt: int) -> AutoResearchLifecycleNoticeRecord:
        return store.record_auto_research_lifecycle_notice(
            AutoResearchLifecycleNoticeRecord(
                notice_id=notice_id,
                episode_id=auto_research.episode_id,
                source_kind="work",
                source_id="worker",
                source_event="settled",
                source_attempt=attempt,
                payload={"attempt": attempt},
                created_at=store.now(),
            )
        )

    first = notice("notice-first", 1)
    effect_id = str(uuid.uuid4())
    request = InboxCommandRequest(
        mailbox_id=MAILBOX_ID,
        request_id="1" * 32,
        credential=CREDENTIAL,
        verb="inbox",
        idempotency_key="harvest-once",
        arguments=InboxHarvestArguments(action="harvest"),
    )

    harvested = effects.inbox(context, request.arguments, effect_id)
    second = notice("notice-second", 2)
    reconciled = effects.reconcile_unknown(context, request, effect_id)

    assert harvested.result == {
        "action": "harvest",
        "count": 1,
        "notices": [
            {
                "notice_id": first.notice_id,
                "source_kind": first.source_kind,
                "source_id": first.source_id,
                "source_event": first.source_event,
                "source_attempt": first.source_attempt,
                "payload": first.payload,
                "created_at": first.created_at,
            }
        ],
    }
    assert reconciled == harvested
    pending = store.pending_auto_research_lifecycle_notices(auto_research.episode_id)
    assert [item.notice_id for item in pending] == [second.notice_id]

    clear_effect_id = str(uuid.uuid4())
    cleared = effects.inbox(
        context,
        InboxCommandRequest(
            mailbox_id=MAILBOX_ID,
            request_id="2" * 32,
            credential=CREDENTIAL,
            verb="inbox",
            idempotency_key="clear-once",
            arguments={"action": "clear"},
        ).arguments,
        clear_effect_id,
    )
    third = notice("notice-third", 3)

    assert cleared.result == {
        "action": "clear",
        "count": 1,
        "notice_ids": [second.notice_id],
    }
    assert [
        item.notice_id
        for item in store.pending_auto_research_lifecycle_notices(auto_research.episode_id)
    ] == [third.notice_id]


def test_inbox_harvest_leaves_a_body_that_cannot_fit_the_command_response_pending(
    tmp_path,
) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    background = BackgroundAgentTasks(store, _successful_stream)
    effects = _effects(store, background)
    context = _context(store, auto_research, root)
    oversized = _insert_unbounded_legacy_lifecycle_notice(
        store,
        AutoResearchLifecycleNoticeRecord(
            notice_id="oversized-lifecycle-body",
            episode_id=auto_research.episode_id,
            source_kind="work",
            source_id="worker-large",
            source_event="failed",
            payload={"diagnostic": "x" * 40_000},
            created_at=store.now(),
        ),
    )

    harvest_effect_id = str(uuid.uuid4())
    harvested = effects.inbox(
        context,
        InboxHarvestArguments(action="harvest"),
        harvest_effect_id,
    )

    assert harvested.status == "invalid"
    assert harvested.message == (
        "Harvest could not acknowledge the oldest lifecycle notice because its body exceeds "
        "the durable command response limit; run inbox --key <new-key> --clear to acknowledge "
        "it without returning the body."
    )
    assert harvested.result == {
        "action": "harvest",
        "disposition": "notice_too_large",
        "replacement_command": "inbox --key <new-key> --clear",
    }
    assert store.auto_research_inbox_receipt(harvest_effect_id) is None
    assert store.pending_auto_research_lifecycle_notices(auto_research.episode_id) == [oversized]

    cleared = effects.inbox(
        context,
        InboxClearArguments(action="clear"),
        str(uuid.uuid4()),
    )
    assert cleared.result == {
        "action": "clear",
        "count": 1,
        "notice_ids": [oversized.notice_id],
    }
    assert store.pending_auto_research_lifecycle_notices(auto_research.episode_id) == []


def test_inbox_clear_refuses_before_mutation_then_clears_after_bounded_harvest(
    tmp_path,
) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    background = BackgroundAgentTasks(store, _successful_stream)
    effects = _effects(store, background)
    all_ids: list[str] = []
    for index in range(48):
        notice_id = f"{index:03d}-" + ("n" * 1_024)
        all_ids.append(notice_id)
        store.record_auto_research_lifecycle_notice(
            AutoResearchLifecycleNoticeRecord(
                notice_id=notice_id,
                episode_id=auto_research.episode_id,
                source_kind="work",
                source_id=f"worker-{index}",
                source_event="settled",
                payload={"status": "succeeded"},
                created_at=store.now(),
            )
        )
    dispatcher = AutoResearchCommandDispatcher(store, effects)
    refused_request = InboxCommandRequest(
        mailbox_id=MAILBOX_ID,
        request_id="3" * 32,
        credential=CREDENTIAL,
        verb="inbox",
        idempotency_key="clear-too-large",
        arguments=InboxClearArguments(action="clear"),
    )
    refused = dispatcher.dispatch(root.operation_id, refused_request)

    assert refused.status == "invalid"
    assert refused.message == (
        "Clear would exceed the durable command response limit, so no lifecycle "
        "notices were acknowledged; run inbox --harvest with a new key before "
        "running inbox --clear with another new key."
    )
    assert refused.result == {
        "action": "clear",
        "disposition": "response_too_large",
    }
    assert [
        notice.notice_id
        for notice in store.pending_auto_research_lifecycle_notices(
            auto_research.episode_id,
            limit=len(all_ids),
        )
    ] == all_ids
    refused_command = store.agent_command(refused_request.request_id)
    assert refused_command is not None
    refused_effect_id = str(refused_command.start_payload["planned_inbox_effect_id"])
    assert store.auto_research_inbox_receipt(refused_effect_id) is None
    assert all(
        notice.acknowledged_at is None
        for notice in store.auto_research_lifecycle_notices(auto_research.episode_id)
    )

    harvested = dispatcher.dispatch(
        root.operation_id,
        InboxCommandRequest(
            mailbox_id=MAILBOX_ID,
            request_id="4" * 32,
            credential=CREDENTIAL,
            verb="inbox",
            idempotency_key="harvest-before-clear",
            arguments=InboxHarvestArguments(action="harvest"),
        ),
    )
    harvested_notices = harvested.result["notices"]
    assert isinstance(harvested_notices, list)
    assert 0 < len(harvested_notices) < len(all_ids)
    harvested_ids = [str(item["notice_id"]) for item in harvested_notices]
    assert harvested_ids == all_ids[: len(harvested_ids)]

    remaining_ids = all_ids[len(harvested_ids) :]
    cleared = dispatcher.dispatch(
        root.operation_id,
        InboxCommandRequest(
            mailbox_id=MAILBOX_ID,
            request_id="5" * 32,
            credential=CREDENTIAL,
            verb="inbox",
            idempotency_key="clear-after-harvest",
            arguments=InboxClearArguments(action="clear"),
        ),
    )

    assert cleared.status == "ok"
    assert cleared.result == {
        "action": "clear",
        "count": len(remaining_ids),
        "notice_ids": remaining_ids,
    }
    assert [
        notice.notice_id
        for notice in store.pending_auto_research_lifecycle_notices(
            auto_research.episode_id,
            limit=len(all_ids),
        )
    ] == []


def test_the_inbox_size_bound_measures_the_payload_the_orchestrator_receives(
    tmp_path,
) -> None:
    """Clear is all-or-nothing, so its bound must weigh the emitted projection.

    The storage layer refuses a Clear it cannot fit and the effect layer emits
    the result. While those were two copies of one dict, either could change
    shape and leave Clear measuring bytes the client never sees.
    """

    store, auto_research, root = _setup_auto_research(tmp_path)
    background = BackgroundAgentTasks(store, _successful_stream)
    effects = _effects(store, background)
    for index in range(3):
        store.record_auto_research_lifecycle_notice(
            AutoResearchLifecycleNoticeRecord(
                notice_id=f"notice-{index}",
                episode_id=auto_research.episode_id,
                source_kind="work",
                source_id=f"worker-{index}",
                source_event="settled",
                payload={"status": "succeeded"},
                created_at=store.now(),
            )
        )
    context = _context(store, auto_research, root)
    harvest_effect_id = str(uuid.uuid4())
    harvested = effects.inbox(
        context,
        InboxHarvestArguments(action="harvest"),
        harvest_effect_id,
    )
    harvest_receipt = store.auto_research_inbox_receipt(harvest_effect_id)
    assert harvest_receipt is not None and harvest_receipt.count == 3

    store.record_auto_research_lifecycle_notice(
        AutoResearchLifecycleNoticeRecord(
            notice_id="notice-later",
            episode_id=auto_research.episode_id,
            source_kind="work",
            source_id="worker-later",
            source_event="settled",
            payload={"status": "succeeded"},
            created_at=store.now(),
        )
    )
    clear_effect_id = str(uuid.uuid4())
    cleared = effects.inbox(
        context,
        InboxClearArguments(action="clear"),
        clear_effect_id,
    )
    clear_receipt = store.auto_research_inbox_receipt(clear_effect_id)
    assert clear_receipt is not None and clear_receipt.notice_ids == ["notice-later"]

    for effect, receipt in ((harvested, harvest_receipt), (cleared, clear_receipt)):
        measured, message = auto_research_inbox_projection(
            receipt.mode,
            notice_ids=receipt.notice_ids,
            notices=receipt.notices,
        )
        assert effect.result == measured
        assert effect.message == message


def test_experiment_kickoff_reports_an_exhausted_shared_allowance(tmp_path) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    background = BackgroundAgentTasks(store, _successful_stream)
    exhausted = AutoResearchExperimentAllowance(total=40, used=40, remaining=0)

    class ExhaustedCoordinator:
        @staticmethod
        def kick_off(**_arguments):
            raise AutoResearchExperimentAllowanceReached(exhausted)

    effects = auto_research_command_effects(
        store=store,
        background=background,
        validate=lambda _context, _arguments: AutoResearchCommandEffectResult(),
        worker_request_factory=_worker_request,
        graph_state=_blocker_state,
        execution_host="execution.example",
        experiment_coordinator=ExhaustedCoordinator(),  # type: ignore[arg-type]
    )
    arguments = ExperimentKickoffArguments(
        action="kick_off_experiment",
        node_id="exp/exhausted",
    )
    request = EpisodeCommandRequest(
        mailbox_id=MAILBOX_ID,
        request_id="0" * 32,
        credential=CREDENTIAL,
        verb="episode",
        idempotency_key="experiment-exhausted",
        arguments=arguments,
    )

    response = AutoResearchCommandDispatcher(store, effects).dispatch(
        root.operation_id,
        request,
    )

    assert response.status == "invalid"
    assert response.message == (
        "The Auto-research child Experiment allowance is exhausted; no Experiment was started."
    )
    assert response.result == {
        "disposition": "allowance_exhausted",
        "experiment_allowance": {"total": 40, "used": 40, "remaining": 0},
    }
    command = store.agent_command(request.request_id)
    assert command is not None
    planned_episode_id = str(command.start_payload["planned_episode_effect_id"])
    admission = store.auto_research_child_admission(planned_episode_id)
    assert admission is not None and admission.state == "cancelled"


def test_unknown_message_reconciliation_returns_the_exact_row_without_redelivery(
    tmp_path,
) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    stage = tmp_path / "worker-stage"
    stage.mkdir()
    worker = _create_worker(
        store,
        auto_research,
        root,
        native_session_id="worker-session",
        stage_root=str(stage),
    )
    background = BackgroundAgentTasks(store, _successful_stream)
    effects = _effects(store, background)
    context = _context(store, auto_research, root)
    planned_message_id = str(uuid.uuid4())
    request = MessageCommandRequest(
        mailbox_id=MAILBOX_ID,
        request_id="d" * 32,
        credential=CREDENTIAL,
        verb="message",
        idempotency_key="message-once",
        arguments={
            "recipient_task_id": worker.operation_id,
            "body": "Deliver this instruction exactly once.",
        },
    )

    saved = record_auto_research_message(
        store,
        message_id=planned_message_id,
        episode_id=auto_research.episode_id,
        sender_role="orchestrator",
        sender_task_id=root.operation_id,
        authorized_by=None,
        recipient_task_id=worker.operation_id,
        body=request.arguments.body,
    )
    tasks_before = store.auto_research_tasks(auto_research.episode_id)
    budget_before = store.episode_budget_meter(auto_research.episode_id)

    reconciled = effects.reconcile_unknown(context, request, planned_message_id)

    assert reconciled is not None
    assert reconciled.result == {
        "message_id": saved.message_id,
        "recipient_task_id": worker.operation_id,
        "delivery_operation_id": None,
        "delivery": "pending",
        "graph_authority": "none",
        "epistemic_status": "hearsay",
        "disposition": "existing",
    }
    assert store.auto_research_tasks(auto_research.episode_id) == tasks_before
    assert store.episode_budget_meter(auto_research.episode_id) == budget_before
    assert store.pending_auto_research_messages(auto_research.episode_id, worker.operation_id) == [
        saved
    ]
    messages = store.auto_research_messages(auto_research.episode_id)
    assert [message.message_id for message in messages] == [planned_message_id]
    assert effects.reconcile_unknown(context, request, str(uuid.uuid4())) is None
    assert effects.reconcile_unknown(context, request, "not-a-uuid") is None
    mismatched = request.model_copy(
        update={
            "arguments": request.arguments.model_copy(
                update={"body": "A different instruction must not match."}
            )
        }
    )
    assert effects.reconcile_unknown(context, mismatched, planned_message_id) is None
    assert store.auto_research_tasks(auto_research.episode_id) == tasks_before
    assert store.episode_budget_meter(auto_research.episode_id) == budget_before
    assert [
        message.message_id for message in store.auto_research_messages(auto_research.episode_id)
    ] == [planned_message_id]


def test_unknown_graph_watch_reconciliation_is_read_only_and_fail_closed(tmp_path) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    background = BackgroundAgentTasks(store, _successful_stream)
    ready: list[str] = []
    graph_reads = 0

    def live_state() -> GraphState:
        nonlocal graph_reads
        graph_reads += 1
        return _blocker_state(status="resolved", revision=2)

    effects = auto_research_command_effects(
        store=store,
        background=background,
        validate=lambda _context, _arguments: AutoResearchCommandEffectResult(),
        worker_request_factory=_worker_request,
        graph_state=live_state,
        execution_host="ssh.execution.example",
        on_watcher_ready=ready.append,
    )
    context = _context(store, auto_research, root)
    planned_watcher_id = str(uuid.uuid4())
    request = WatchGraphCommandRequest(
        mailbox_id=MAILBOX_ID,
        request_id="e" * 32,
        credential=CREDENTIAL,
        verb="watch_graph",
        idempotency_key="watch-once",
        arguments={
            "condition": {"node_id": "blk/check", "status_in": ["resolved"]},
            "reason": "Continue after the durable condition is satisfied.",
        },
    )

    created = effects.watch_graph(context, request.arguments, planned_watcher_id)
    assert graph_reads == 1
    assert ready == [auto_research.project_id]

    reconciled = effects.reconcile_unknown(context, request, planned_watcher_id)

    assert reconciled is not None
    assert reconciled.result == {
        **created.result,
        "disposition": "existing",
    }
    assert graph_reads == 1
    assert ready == [auto_research.project_id]
    assert effects.reconcile_unknown(context, request, str(uuid.uuid4())) is None
    assert effects.reconcile_unknown(context, request, "not-a-uuid") is None
    mismatched = request.model_copy(
        update={
            "arguments": request.arguments.model_copy(
                update={
                    "condition": NodeStatusGraphCondition(
                        node_id="blk/check",
                        status_in=["open"],
                    )
                }
            )
        }
    )
    assert effects.reconcile_unknown(context, mismatched, planned_watcher_id) is None
    assert graph_reads == 1
    assert ready == [auto_research.project_id]


def test_individual_worker_stop_routes_to_the_current_attempt_and_never_stops_the_parent(
    tmp_path,
    monkeypatch,
) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    worker = _create_routed_worker(store, auto_research, root, status="queued")
    background = _RecordingBackground(store)
    _record_child_controls(monkeypatch, background)
    effects = _effects(store, background)
    dispatcher = AutoResearchCommandDispatcher(store, effects)

    response = dispatcher.dispatch(
        root.operation_id,
        StopCommandRequest(
            mailbox_id=MAILBOX_ID,
            request_id="c" * 32,
            credential=CREDENTIAL,
            verb="stop",
            idempotency_key="stop-worker",
            arguments={"worker_id": worker.operation_id},
        ),
    )

    assert response.status == "ok"
    assert response.result == {
        "worker_id": worker.operation_id,
        "current_operation_id": worker.operation_id,
        "status": "queued",
        "disposition": "stopped",
    }
    assert background.stopped == [worker.operation_id]
    route = store.auto_research_child_work(worker.operation_id)
    assert route is not None and route.stop_requested_at is not None
    assert store.episode(auto_research.episode_id).stop_requested_at is None  # type: ignore[union-attr]
    assert background.paused == []
    assert background.resumed == []
