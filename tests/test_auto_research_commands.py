from __future__ import annotations

import asyncio
import hashlib
import json
import socket
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

import pytest

from rcp.agents.command_mailbox import serve_command_mailbox, stage_command_mailbox
from rcp.agents.command_protocol import (
    ApplyArguments,
    ApplyCommandRequest,
    CommandResponse,
    EpisodeCommandRequest,
    FinishCommandRequest,
    InboxCommandRequest,
    MessageArguments,
    MessageCommandRequest,
    PauseCommandRequest,
    ResumeCommandRequest,
    SpawnArguments,
    SpawnCommandRequest,
    StatusCommandRequest,
    StopCommandRequest,
    ValidateCommandRequest,
    WatchGraphCommandRequest,
)
from rcp.core.authority import AgentDispatchAuthority, AgentDispatchScope
from rcp.core.models import AuthorizedHuman
from rcp.core.transition_models import GraphHeadRef, GraphTargetRef
from rcp.limits import AUTO_RESEARCH_APPLY_MAX_PER_TURN
from rcp.runs.auto_research import (
    AutoResearchCommandDispatcher,
    AutoResearchCommandEffectResult,
    AutoResearchCommandEffects,
    AutoResearchCommandUnavailable,
    AutoResearchRunRequest,
    auto_research_completion_signal,
    request_auto_research_stop,
)
from rcp.service import RunRequest, resolve_dispatch_authority
from rcp.storage import (
    AgentTaskRecord,
    AppStore,
    AutoResearchChildAdmissionRecord,
    AutoResearchChildExperimentRecord,
    AutoResearchChildWorkRecord,
    AutoResearchCommandFileRecord,
    AutoResearchStateRecord,
    EpisodeRecord,
    ProjectRecord,
)

MAILBOX_ID = "a" * 32
CREDENTIAL = "b" * 64
_RUN_TRUTH_SCOPE = ["repo-a"]
_SPAWN_INSTRUCTION_FILE = "worker-task.md"
_SPAWN_INSTRUCTION = "Inspect everything needed to settle the seat."


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
    authorizer = AuthorizedHuman(
        space_id=str(uuid.uuid4()),
        user_id=str(uuid.uuid4()),
        display_name="AutoResearch owner",
    )
    now = store.now()
    graph_target = GraphTargetRef(kind="branch", branch_id="auto_research")
    root_request = AutoResearchRunRequest(
        episode_id="auto_research",
        role="orchestrator",
        actor_operation_id="root",
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
            request=root_request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="done",
            authorized_by=authorizer,
            dispatch_authority=_auto_research_authority("auto_research", "orchestrator"),
        ),
    )
    store.complete_agent_task(root.operation_id, applied_revision=None, result={})
    stored_root = store.agent_task(root.operation_id)
    assert stored_root is not None
    return store, episode, stored_root


def _worker(
    store: AppStore,
    auto_research: EpisodeRecord,
    root: AgentTaskRecord,
    operation_id: str,
    *,
    seat_node_id: str = "exp/check",
    instruction: str = "Run the check.",
) -> AgentTaskRecord:
    request = AutoResearchRunRequest(
        episode_id=auto_research.episode_id,
        role="worker",
        actor_operation_id=operation_id,
        run_truth_scope=_RUN_TRUTH_SCOPE,
        control_node_id=seat_node_id,
        instruction=instruction,
    )
    now = store.now()
    return store.create_auto_research_agent_task(
        AgentTaskRecord(
            operation_id=operation_id,
            project_id=auto_research.project_id,
            episode_id=auto_research.episode_id,
            graph_target=auto_research.graph_target,
            kind="auto_research",
            status="queued",
            request=request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="queued",
            parent_operation_id=root.operation_id,
            authorized_by=auto_research.authorized_by,
            dispatch_authority=_auto_research_authority(auto_research.episode_id, "worker"),
        ),
        role="worker",
    )


def _orchestrator_turn(
    store: AppStore,
    auto_research: EpisodeRecord,
    root: AgentTaskRecord,
    *,
    operation_id: str,
) -> AgentTaskRecord:
    request = AutoResearchRunRequest(
        episode_id=auto_research.episode_id,
        role="orchestrator",
        actor_operation_id=root.operation_id,
        run_truth_scope=_RUN_TRUTH_SCOPE,
    )
    now = store.now()
    return store.create_auto_research_agent_task(
        AgentTaskRecord(
            operation_id=operation_id,
            project_id=auto_research.project_id,
            episode_id=auto_research.episode_id,
            graph_target=auto_research.graph_target,
            kind="auto_research",
            status="succeeded",
            request=request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="done",
            parent_operation_id=root.operation_id,
            authorized_by=auto_research.authorized_by,
            dispatch_authority=_auto_research_authority(auto_research.episode_id, "orchestrator"),
        ),
        role="orchestrator",
    )


@dataclass
class _Effects:
    store: AppStore
    auto_research: EpisodeRecord
    root: AgentTaskRecord
    seat_type: str | None = "experiment"
    spawn_calls: list[tuple[SpawnArguments, str]] = field(default_factory=list)
    message_calls: list[MessageArguments] = field(default_factory=list)
    planned_message_ids: list[str] = field(default_factory=list)
    planned_watcher_ids: list[str] = field(default_factory=list)
    pause_worker_ids: list[str] = field(default_factory=list)
    resume_operation_ids: list[str] = field(default_factory=list)
    stop_worker_ids: list[str] = field(default_factory=list)
    episode_effect_ids: list[str] = field(default_factory=list)
    reconcile_calls: list[str] = field(default_factory=list)
    reconcile_planned_effect_ids: list[str | None] = field(default_factory=list)
    reconcile_result: AutoResearchCommandEffectResult | None = None
    finish_calls: int = 0

    def bundle(self) -> AutoResearchCommandEffects:
        return AutoResearchCommandEffects(
            validate=lambda _context, _arguments: AutoResearchCommandEffectResult(),
            status=lambda _context, _arguments: AutoResearchCommandEffectResult(),
            spawn=self.spawn,
            pause=self.pause,
            resume=self.resume,
            stop=self.stop,
            message=self.message,
            watch_graph=self.watch_graph,
            episode=self.episode,
            finish=self.finish,
            seat_node_type=lambda _project_id, _node_id: self.seat_type,
            reconcile_unknown=self.reconcile_unknown,
            worker_lookup=self.worker_lookup,
            verify_spawn=self.verify_spawn,
        )

    def spawn(
        self,
        context,
        arguments: SpawnArguments,
        planned_worker_id: str,
    ) -> AutoResearchCommandEffectResult:
        self.spawn_calls.append((arguments, planned_worker_id))
        snapshot = context.command_file
        assert snapshot is not None
        worker = _routed_worker(
            self.store,
            self.auto_research,
            admitted_by=context.task,
            worker_id=planned_worker_id,
            seat_node_id=arguments.seat_node_id,
            instruction=snapshot.text,
            instruction_sha256=snapshot.sha256,
            admission_id=planned_worker_id,
        )
        return AutoResearchCommandEffectResult(
            result={"worker_id": worker.operation_id, "disposition": "created"}
        )

    def worker_lookup(self, context, worker_id: str) -> AgentTaskRecord:
        route = self.store.auto_research_child_work(worker_id)
        if route is not None and route.episode_id == context.episode.episode_id:
            task = self.store.agent_task(route.current_operation_id)
            assert task is not None
            return task
        worker = self.store.agent_task(worker_id)
        if worker is None or worker.episode_id != context.episode.episode_id:
            raise ValueError("Worker control target is outside this auto_research.")
        binding = self.store.auto_research_actor_binding(worker.operation_id)
        if binding.actor_operation_id != worker_id:
            raise ValueError("The orchestrator must address a worker by its stable worker id.")
        return worker

    def verify_spawn(self, context, arguments, planned_worker_id) -> AgentTaskRecord:
        route = self.store.auto_research_child_work(planned_worker_id)
        snapshot = context.command_file
        if (
            route is None
            or snapshot is None
            or route.episode_id != context.episode.episode_id
            or route.control_node_id != arguments.seat_node_id
            or route.admitted_by_operation_id != context.task.operation_id
            or route.instruction != snapshot.text
            or route.instruction_sha256 != snapshot.sha256
        ):
            raise AutoResearchCommandUnavailable(
                "Spawn created an incorrect ordinary Work route, seat, parent, or instruction."
            )
        worker = self.store.agent_task(route.root_operation_id)
        if worker is None or worker.kind != "node_chat" or worker.parent_operation_id is not None:
            raise AutoResearchCommandUnavailable(
                "Spawn created an incorrect ordinary Work route, seat, parent, or instruction."
            )
        return worker

    def message(
        self,
        _context,
        arguments: MessageArguments,
        planned_message_id: str,
    ) -> AutoResearchCommandEffectResult:
        self.message_calls.append(arguments)
        self.planned_message_ids.append(planned_message_id)
        return AutoResearchCommandEffectResult(result={"delivered": True})

    def watch_graph(
        self,
        _context,
        _arguments,
        planned_watcher_id: str,
    ) -> AutoResearchCommandEffectResult:
        self.planned_watcher_ids.append(planned_watcher_id)
        return AutoResearchCommandEffectResult(result={"armed": True})

    def resume(
        self,
        _context,
        worker_id: str,
        planned_operation_id: str,
    ) -> AutoResearchCommandEffectResult:
        self.resume_operation_ids.append(planned_operation_id)
        return AutoResearchCommandEffectResult(
            result={
                "disposition": "resumed",
                "worker_id": worker_id,
                "current_operation_id": planned_operation_id,
            }
        )

    def pause(self, _context, worker_id: str) -> AutoResearchCommandEffectResult:
        self.pause_worker_ids.append(worker_id)
        return AutoResearchCommandEffectResult(result={"worker_id": worker_id})

    def stop(self, _context, worker_id: str) -> AutoResearchCommandEffectResult:
        self.stop_worker_ids.append(worker_id)
        return AutoResearchCommandEffectResult(result={"worker_id": worker_id})

    def episode(self, _context, arguments, planned_effect_id: str):
        self.episode_effect_ids.append(planned_effect_id)
        return AutoResearchCommandEffectResult(
            result={
                "disposition": arguments.action,
                "episode_id": arguments.episode_id,
                "operation_id": planned_effect_id,
            }
        )

    def reconcile_unknown(
        self,
        _context,
        request,
        planned_effect_id: str | None,
    ) -> AutoResearchCommandEffectResult | None:
        self.reconcile_calls.append(request.verb)
        self.reconcile_planned_effect_ids.append(planned_effect_id)
        return self.reconcile_result

    def finish(self, _context, _planned_effect_id: str) -> AutoResearchCommandEffectResult:
        self.finish_calls += 1
        auto_research_completion_signal(self.store, self.auto_research.episode_id)
        auto_research = self.store.episode(self.auto_research.episode_id)
        assert auto_research is not None
        return AutoResearchCommandEffectResult(
            result={"episode_id": auto_research.episode_id, "ending": auto_research.ending}
        )


def _spawn_request(
    request_id: str,
    *,
    key: str | None,
    seat_node_id: str = "exp/check",
) -> SpawnCommandRequest:
    return SpawnCommandRequest(
        mailbox_id=MAILBOX_ID,
        request_id=request_id,
        credential=CREDENTIAL,
        verb="spawn",
        idempotency_key=key,
        arguments={
            "seat_node_id": seat_node_id,
            "instruction_file": _SPAWN_INSTRUCTION_FILE,
        },
    )


def _remaining_idempotent_request(
    kind: str,
    *,
    request_id: str,
    key: str,
    worker_id: str,
) -> MessageCommandRequest | WatchGraphCommandRequest | InboxCommandRequest:
    if kind == "message":
        return MessageCommandRequest(
            mailbox_id=MAILBOX_ID,
            request_id=request_id,
            credential=CREDENTIAL,
            verb="message",
            idempotency_key=key,
            arguments={
                "recipient_task_id": worker_id,
                "body": "Continue the bounded child task.",
            },
        )
    if kind == "watch_graph":
        return WatchGraphCommandRequest(
            mailbox_id=MAILBOX_ID,
            request_id=request_id,
            credential=CREDENTIAL,
            verb="watch_graph",
            idempotency_key=key,
            arguments={
                "condition": {"node_id": "hyp/result", "status_in": ["active"]},
                "reason": "Continue when the graph condition settles.",
            },
        )
    assert kind == "inbox"
    return InboxCommandRequest(
        mailbox_id=MAILBOX_ID,
        request_id=request_id,
        credential=CREDENTIAL,
        verb="inbox",
        idempotency_key=key,
        arguments={"action": "harvest"},
    )


def _dispatcher(
    store: AppStore,
    effects: AutoResearchCommandEffects,
    *,
    instruction: str = _SPAWN_INSTRUCTION,
) -> AutoResearchCommandDispatcher:
    return AutoResearchCommandDispatcher(
        store,
        effects,
        command_file_reader=lambda _filename, _max_bytes: instruction,
    )


def _routed_worker(
    store: AppStore,
    auto_research: EpisodeRecord,
    *,
    admitted_by: AgentTaskRecord,
    worker_id: str,
    seat_node_id: str,
    instruction: str,
    instruction_sha256: str | None = None,
    admission_id: str | None = None,
) -> AgentTaskRecord:
    request = RunRequest(
        provider="codex",
        run_on="local",
        run_truth_scope=_RUN_TRUTH_SCOPE,
        chat_id=worker_id,
        chat_scope="node",
        node_id=seat_node_id,
        message=instruction,
        mode="work",
        trigger="orchestrator",
        patch_kind="work",
    )
    authority = resolve_dispatch_authority("node_chat", request)
    assert authority is not None
    now = store.now()
    _, worker = store.create_auto_research_child_work(
        AutoResearchChildWorkRecord(
            worker_id=worker_id,
            episode_id=auto_research.episode_id,
            project_id=auto_research.project_id,
            control_node_id=seat_node_id,
            root_operation_id=worker_id,
            current_operation_id=worker_id,
            admitted_by_operation_id=admitted_by.operation_id,
            instruction=instruction,
            instruction_sha256=(
                instruction_sha256 or hashlib.sha256(instruction.encode()).hexdigest()
            ),
            created_at=now,
            updated_at=now,
        ),
        AgentTaskRecord(
            operation_id=worker_id,
            project_id=auto_research.project_id,
            episode_id=auto_research.episode_id,
            graph_target=auto_research.graph_target,
            kind="node_chat",
            status="queued",
            request=request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="queued",
            authorized_by=auto_research.authorized_by,
            dispatch_authority=authority,
        ),
        admission_id=admission_id,
    )
    return worker


def _record_interrupted_spawn(
    store: AppStore,
    auto_research: EpisodeRecord,
    root: AgentTaskRecord,
    *,
    command_id: str,
    key: str,
    arguments: SpawnArguments,
    instruction: str,
) -> str:
    planned_worker_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rcp:auto_research:{auto_research.episode_id}:spawn:{key}",
        )
    )
    now = store.now()
    digest = hashlib.sha256(instruction.encode()).hexdigest()
    store.start_agent_command(
        operation_id=root.operation_id,
        command_id=command_id,
        episode_id=auto_research.episode_id,
        verb="spawn",
        idempotency_key=key,
        payload={
            "request_id": command_id,
            "arguments": arguments.model_dump(mode="json"),
            "planned_worker_id": planned_worker_id,
            "command_file": {
                "filename": arguments.instruction_file,
                "byte_length": len(instruction.encode()),
                "sha256": digest,
            },
        },
        file_snapshot=AutoResearchCommandFileRecord(
            command_id=command_id,
            episode_id=auto_research.episode_id,
            operation_id=root.operation_id,
            kind="instruction",
            filename=arguments.instruction_file,
            sha256=digest,
            content=instruction,
            created_at=now,
        ),
        child_admission=AutoResearchChildAdmissionRecord(
            admission_id=planned_worker_id,
            episode_id=auto_research.episode_id,
            project_id=auto_research.project_id,
            child_kind="work",
            child_id=planned_worker_id,
            state="accepted",
            created_at=now,
            updated_at=now,
        ),
    )
    return planned_worker_id


def test_mutating_command_requires_caller_idempotency_key_and_records_the_exit(tmp_path) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    effects = _Effects(store, auto_research, root)
    dispatcher = _dispatcher(store, effects.bundle())

    response = dispatcher.dispatch(root.operation_id, _spawn_request("1" * 32, key=None))

    assert response.status == "invalid"
    assert response.exit_code == 1
    assert "idempotency key" in (response.message or "")
    assert effects.spawn_calls == []
    invocation = store.agent_command("1" * 32)
    assert invocation is not None
    assert invocation.started_at
    assert invocation.exited_at is not None
    assert invocation.status == "invalid"


def test_transient_apply_snapshot_read_leaves_key_and_apply_slot_for_exact_retry(
    tmp_path,
) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    effects = _Effects(store, auto_research, root)
    read_attempts = 0
    successful_reads: list[str] = []
    apply_calls: list[tuple[str, str]] = []
    patch = '{"summary":"repair runtime","ops":[]}'

    def read_patch(filename: str, _max_bytes: int) -> str:
        nonlocal read_attempts
        read_attempts += 1
        if read_attempts == 1:
            raise OSError("staged workspace is temporarily unreachable")
        successful_reads.append(filename)
        return patch

    def apply(context, _arguments, planned_apply_id):
        assert context.command_file is not None
        apply_calls.append((planned_apply_id, context.command_file.sha256))
        return AutoResearchCommandEffectResult(
            result={"apply_id": planned_apply_id, "disposition": "applied"}
        )

    dispatcher = AutoResearchCommandDispatcher(
        store,
        replace(effects.bundle(), apply=apply),
        command_file_reader=read_patch,
    )
    key = "apply-after-snapshot-read-recovers"
    request = ApplyCommandRequest(
        mailbox_id=MAILBOX_ID,
        request_id="4" * 32,
        credential=CREDENTIAL,
        verb="apply",
        idempotency_key=key,
        arguments={"patch_file": "patch.json"},
    )

    unavailable = dispatcher.dispatch(root.operation_id, request)

    assert unavailable.status == "unavailable"
    audit = store.agent_command(request.request_id)
    assert audit is not None and audit.idempotency_key is None
    assert audit.start_payload["attempted_idempotency_key"] == key
    assert audit.start_payload["pre_admission_unavailable"] is True
    assert store.agent_command_by_key(auto_research.episode_id, key) is None
    assert store.auto_research_apply_admission_count(root.operation_id) == 0

    recovered = dispatcher.dispatch(
        root.operation_id,
        request.model_copy(update={"request_id": "5" * 32}),
    )
    replayed = dispatcher.dispatch(
        root.operation_id,
        request.model_copy(update={"request_id": "6" * 32}),
    )
    mismatched = dispatcher.dispatch(
        root.operation_id,
        request.model_copy(
            update={
                "request_id": "7" * 32,
                "arguments": request.arguments.model_copy(update={"patch_file": "different.json"}),
            }
        ),
    )

    expected_apply_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rcp:auto_research:{auto_research.episode_id}:apply:{key}",
        )
    )
    assert recovered.status == replayed.status == "ok"
    assert mismatched.status == "invalid"
    assert "different command arguments" in (mismatched.message or "")
    assert read_attempts == 2
    assert successful_reads == ["patch.json"]
    assert apply_calls == [(expected_apply_id, hashlib.sha256(patch.encode()).hexdigest())]
    assert store.auto_research_apply_admission_count(root.operation_id) == 1
    canonical = store.agent_command_by_key(auto_research.episode_id, key)
    assert canonical is not None and canonical.command_id == recovered.request_id
    assert store.auto_research_command_file(canonical.command_id) is not None


def test_transient_spawn_snapshot_read_leaves_key_for_one_successful_admission(
    tmp_path,
) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    effects = _Effects(store, auto_research, root)
    read_attempts = 0
    successful_reads: list[str] = []

    def read_instruction(filename: str, _max_bytes: int) -> str:
        nonlocal read_attempts
        read_attempts += 1
        if read_attempts == 1:
            raise OSError("staged workspace is temporarily unreachable")
        successful_reads.append(filename)
        return _SPAWN_INSTRUCTION

    dispatcher = AutoResearchCommandDispatcher(
        store,
        effects.bundle(),
        command_file_reader=read_instruction,
    )
    key = "spawn-after-snapshot-read-recovers"
    request = _spawn_request("8" * 32, key=key)
    expected_worker_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rcp:auto_research:{auto_research.episode_id}:spawn:{key}",
        )
    )

    unavailable = dispatcher.dispatch(root.operation_id, request)

    assert unavailable.status == "unavailable"
    audit = store.agent_command(request.request_id)
    assert audit is not None and audit.idempotency_key is None
    assert audit.start_payload["attempted_idempotency_key"] == key
    assert store.agent_command_by_key(auto_research.episode_id, key) is None
    assert store.auto_research_child_admission(expected_worker_id) is None

    recovered = dispatcher.dispatch(
        root.operation_id,
        request.model_copy(update={"request_id": "9" * 32}),
    )
    replayed = dispatcher.dispatch(
        root.operation_id,
        request.model_copy(update={"request_id": "a" * 32}),
    )
    mismatched = dispatcher.dispatch(
        root.operation_id,
        request.model_copy(
            update={
                "request_id": "b" * 32,
                "arguments": request.arguments.model_copy(update={"seat_node_id": "exp/different"}),
            }
        ),
    )

    assert recovered.status == replayed.status == "ok"
    assert recovered.result["worker_id"] == expected_worker_id
    assert mismatched.status == "invalid"
    assert "different command arguments" in (mismatched.message or "")
    assert read_attempts == 2
    assert successful_reads == [_SPAWN_INSTRUCTION_FILE]
    assert len(effects.spawn_calls) == 1
    admission = store.auto_research_child_admission(expected_worker_id)
    assert admission is not None and admission.state == "reflected"
    canonical = store.agent_command_by_key(auto_research.episode_id, key)
    assert canonical is not None and canonical.command_id == recovered.request_id


def test_transient_goal_snapshot_read_leaves_kickoff_key_for_exact_retry(tmp_path) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    effects = _Effects(store, auto_research, root)
    read_attempts = 0
    successful_reads: list[str] = []
    episode_calls: list[str] = []
    goal = "Compare the repaired runtime against the bounded baseline."

    def read_goal(filename: str, _max_bytes: int) -> str:
        nonlocal read_attempts
        read_attempts += 1
        if read_attempts == 1:
            raise OSError("staged workspace is temporarily unreachable")
        successful_reads.append(filename)
        return goal

    def episode(context, arguments, planned_effect_id):
        assert context.command_file is not None
        assert context.command_file.text == goal
        episode_calls.append(planned_effect_id)
        now = store.now()
        route = store.reserve_auto_research_experiment_replacement(
            AutoResearchChildExperimentRecord(
                child_episode_id=planned_effect_id,
                auto_research_episode_id=auto_research.episode_id,
                project_id=auto_research.project_id,
                control_node_id=arguments.node_id,
                state="pending",
                replaces_episode_id="predecessor",
                request={"goal": goal, "invocation_limit": arguments.invocation_limit},
                goal_sha256=context.command_file.sha256,
                parent_operation_id=context.task.operation_id,
                created_at=now,
                updated_at=now,
            ),
            admission_id=planned_effect_id,
        )
        return AutoResearchCommandEffectResult(
            result={
                "disposition": "replacement_pending",
                "episode_id": route.child_episode_id,
            }
        )

    dispatcher = AutoResearchCommandDispatcher(
        store,
        replace(effects.bundle(), episode=episode),
        command_file_reader=read_goal,
    )
    key = "experiment-after-goal-read-recovers"
    request = EpisodeCommandRequest(
        mailbox_id=MAILBOX_ID,
        request_id="c" * 32,
        credential=CREDENTIAL,
        verb="episode",
        idempotency_key=key,
        arguments={
            "action": "kick_off_experiment",
            "node_id": "exp/check",
            "goal_file": "goal.md",
            "invocation_limit": 3,
        },
    )
    expected_episode_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rcp:auto_research:{auto_research.episode_id}:episode:{key}",
        )
    )

    unavailable = dispatcher.dispatch(root.operation_id, request)

    assert unavailable.status == "unavailable"
    audit = store.agent_command(request.request_id)
    assert audit is not None and audit.idempotency_key is None
    assert audit.start_payload["attempted_idempotency_key"] == key
    assert store.agent_command_by_key(auto_research.episode_id, key) is None
    assert store.auto_research_child_admission(expected_episode_id) is None

    recovered = dispatcher.dispatch(
        root.operation_id,
        request.model_copy(update={"request_id": "d" * 32}),
    )
    replayed = dispatcher.dispatch(
        root.operation_id,
        request.model_copy(update={"request_id": "e" * 32}),
    )
    mismatched = dispatcher.dispatch(
        root.operation_id,
        request.model_copy(
            update={
                "request_id": "f" * 32,
                "arguments": request.arguments.model_copy(update={"node_id": "exp/different"}),
            }
        ),
    )

    assert recovered.status == replayed.status == "ok"
    assert recovered.result["episode_id"] == expected_episode_id
    assert mismatched.status == "invalid"
    assert "different command arguments" in (mismatched.message or "")
    assert read_attempts == 2
    assert successful_reads == ["goal.md"]
    assert episode_calls == [expected_episode_id]
    admission = store.auto_research_child_admission(expected_episode_id)
    assert admission is not None and admission.state == "reflected"
    canonical = store.agent_command_by_key(auto_research.episode_id, key)
    assert canonical is not None and canonical.command_id == recovered.request_id


def test_apply_limit_refuses_before_reading_another_patch_file(tmp_path) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    reads: list[tuple[str, int]] = []
    effects = _Effects(store, auto_research, root)
    dispatcher = AutoResearchCommandDispatcher(
        store,
        effects.bundle(),
        command_file_reader=lambda filename, max_bytes: (
            reads.append((filename, max_bytes)) or '{"ops":[]}'
        ),
    )
    admitted = [
        dispatcher.dispatch(
            root.operation_id,
            ApplyCommandRequest(
                verb="apply",
                mailbox_id=MAILBOX_ID,
                request_id=uuid.uuid4().hex,
                credential=CREDENTIAL,
                idempotency_key=f"unavailable-apply-{index}",
                arguments=ApplyArguments(patch_file="patch.json"),
            ),
        )
        for index in range(AUTO_RESEARCH_APPLY_MAX_PER_TURN)
    ]
    request = ApplyCommandRequest(
        verb="apply",
        mailbox_id=MAILBOX_ID,
        request_id="f" * 32,
        credential=CREDENTIAL,
        idempotency_key="apply-over-limit",
        arguments=ApplyArguments(patch_file="patch.json"),
    )
    response = dispatcher.dispatch(root.operation_id, request)
    replay = dispatcher.dispatch(
        root.operation_id,
        request.model_copy(update={"request_id": "e" * 32}),
    )

    assert all(item.status == "unavailable" for item in admitted)
    assert response.status == "invalid"
    assert replay.status == "invalid"
    assert replay.message == response.message
    assert f"{AUTO_RESEARCH_APPLY_MAX_PER_TURN}-Apply limit" in (response.message or "")
    assert len(reads) == AUTO_RESEARCH_APPLY_MAX_PER_TURN
    assert store.auto_research_apply_results(root.operation_id) == []
    assert (
        store.auto_research_apply_admission_count(root.operation_id)
        == AUTO_RESEARCH_APPLY_MAX_PER_TURN
    )
    invocation = store.agent_command(request.request_id)
    assert invocation is not None and invocation.exited_at is not None
    assert invocation.start_payload["apply_admitted"] is False
    assert store.auto_research_command_file(invocation.command_id) is None


def test_concurrent_apply_admission_reads_only_the_single_remaining_patch(tmp_path) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    for index in range(AUTO_RESEARCH_APPLY_MAX_PER_TURN - 1):
        store.start_agent_command(
            operation_id=root.operation_id,
            command_id=f"seed-apply-command-{index}",
            episode_id=auto_research.episode_id,
            verb="apply",
            idempotency_key=f"seed-apply-key-{index}",
            payload={"request_id": f"seed-apply-request-{index}"},
            apply_admission_limit=AUTO_RESEARCH_APPLY_MAX_PER_TURN,
        )

    effects = _Effects(store, auto_research, root)
    base_dispatcher = AutoResearchCommandDispatcher(store, effects.bundle())
    reads: list[str] = []

    def read_patch(filename: str, _max_bytes: int) -> str:
        reads.append(filename)
        return '{"ops":[]}'

    dispatchers = [
        base_dispatcher.with_command_files(
            reader=read_patch,
            consumer=lambda _filename, _sha256: True,
            refresher=lambda: (0, "", ""),
        )
        for _ in range(2)
    ]
    requests = [
        ApplyCommandRequest(
            verb="apply",
            mailbox_id=MAILBOX_ID,
            request_id=str(index + 1) * 32,
            credential=CREDENTIAL,
            idempotency_key=f"concurrent-apply-key-{index}",
            arguments=ApplyArguments(patch_file="patch.json"),
        )
        for index in range(2)
    ]
    barrier = threading.Barrier(2)

    def dispatch(index: int) -> CommandResponse:
        barrier.wait()
        return dispatchers[index].dispatch(root.operation_id, requests[index])

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(dispatch, index) for index in range(2)]
        responses = [future.result() for future in futures]

    assert sorted(response.status for response in responses) == ["invalid", "unavailable"]
    assert reads == ["patch.json"]
    assert (
        store.auto_research_apply_admission_count(root.operation_id)
        == AUTO_RESEARCH_APPLY_MAX_PER_TURN
    )
    invocations = [store.agent_command(request.request_id) for request in requests]
    assert all(invocation is not None for invocation in invocations)
    admitted = [
        invocation
        for invocation in invocations
        if invocation is not None and invocation.start_payload["apply_admitted"] is True
    ]
    refused = [
        invocation
        for invocation in invocations
        if invocation is not None and invocation.start_payload["apply_admitted"] is False
    ]
    assert len(admitted) == len(refused) == 1
    assert store.auto_research_command_file(admitted[0].command_id) is not None
    assert store.auto_research_command_file(refused[0].command_id) is None


def test_finish_is_orchestrator_only_idempotent_and_fences_later_work(tmp_path) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    effects = _Effects(store, auto_research, root)
    dispatcher = _dispatcher(store, effects.bundle())
    spawned = dispatcher.dispatch(
        root.operation_id,
        _spawn_request("a" * 32, key="spawn-before-finish"),
    )
    assert spawned.status == "ok"
    worker_id = str(spawned.result["worker_id"])
    unknown_key = "unknown-spawn-before-finish"
    unknown_request = _spawn_request("3" * 32, key=unknown_key)
    unknown_worker_id = _record_interrupted_spawn(
        store,
        auto_research,
        root,
        command_id=unknown_request.request_id,
        key=unknown_key,
        arguments=unknown_request.arguments,
        instruction=_SPAWN_INSTRUCTION,
    )
    request = FinishCommandRequest(
        mailbox_id=MAILBOX_ID,
        request_id="f" * 32,
        credential=CREDENTIAL,
        verb="finish",
        idempotency_key="finish-once",
    )

    first = dispatcher.dispatch(root.operation_id, request)
    replay = dispatcher.dispatch(
        root.operation_id,
        request.model_copy(update={"request_id": "0" * 32}),
    )
    replayed_spawn = dispatcher.dispatch(
        root.operation_id,
        _spawn_request("b" * 32, key="spawn-before-finish"),
    )
    unknown_spawn = dispatcher.dispatch(
        root.operation_id,
        unknown_request.model_copy(update={"request_id": "4" * 32}),
    )

    denied = [
        dispatcher.dispatch(
            root.operation_id,
            _spawn_request("c" * 32, key="spawn-after-finish"),
        ),
        dispatcher.dispatch(
            root.operation_id,
            MessageCommandRequest(
                mailbox_id=MAILBOX_ID,
                request_id="d" * 32,
                credential=CREDENTIAL,
                verb="message",
                idempotency_key="message-after-finish",
                arguments={
                    "recipient_task_id": worker_id,
                    "body": "This must not cross the ending fence.",
                },
            ),
        ),
        dispatcher.dispatch(
            root.operation_id,
            WatchGraphCommandRequest(
                mailbox_id=MAILBOX_ID,
                request_id="e" * 32,
                credential=CREDENTIAL,
                verb="watch_graph",
                idempotency_key="watch-after-finish",
                arguments={
                    "condition": {"node_id": "hyp/result", "status_in": ["active"]},
                    "reason": "This must not cross the ending fence.",
                },
            ),
        ),
    ]
    status = dispatcher.dispatch(
        root.operation_id,
        StatusCommandRequest(
            mailbox_id=MAILBOX_ID,
            request_id="1" * 32,
            credential=CREDENTIAL,
            verb="status",
        ),
    )
    validation = dispatcher.dispatch(
        root.operation_id,
        ValidateCommandRequest(
            mailbox_id=MAILBOX_ID,
            request_id="2" * 32,
            credential=CREDENTIAL,
            verb="validate",
            arguments={"patch": '{"summary":"read only","ops":[]}'},
        ),
    )

    assert first.status == replay.status == "ok"
    assert (
        first.result
        == replay.result
        == {
            "episode_id": auto_research.episode_id,
            "ending": "completed",
        }
    )
    assert effects.finish_calls == 1
    assert replayed_spawn.status == "ok"
    assert replayed_spawn.result == {
        "worker_id": worker_id,
        "status": "queued",
        "disposition": "existing",
    }
    assert unknown_spawn.status == "unavailable"
    assert (
        unknown_spawn.message
        == "The Auto-research episode is no longer accepting mutating commands."
    )
    assert store.agent_task(unknown_worker_id) is None
    assert all(response.status == "unavailable" for response in denied)
    assert all(
        response.message == "The Auto-research episode is no longer accepting mutating commands."
        for response in denied
    )
    assert len(effects.spawn_calls) == 1
    assert effects.message_calls == []
    assert effects.planned_watcher_ids == []
    assert status.status == validation.status == "ok"
    fenced = store.episode(auto_research.episode_id)
    assert fenced is not None
    assert (fenced.status, fenced.ending) == ("wrapping_up", "completed")
    with pytest.raises(ValueError, match="not accepting new work"):
        _orchestrator_turn(
            store,
            auto_research,
            root,
            operation_id="too-late",
        )
    invocation = store.agent_command_by_key(auto_research.episode_id, "finish-once")
    assert invocation is not None
    assert invocation.exited_at is not None
    assert invocation.status == "ok"
    for request_id, expected_status in {
        "b" * 32: "ok",
        "3" * 32: "unavailable",
        "4" * 32: "unavailable",
        "c" * 32: "unavailable",
        "d" * 32: "unavailable",
        "e" * 32: "unavailable",
        "1" * 32: "ok",
        "2" * 32: "ok",
    }.items():
        audited = store.agent_command(request_id)
        assert audited is not None
        assert audited.exited_at is not None
        assert audited.status == expected_status


def test_stop_intent_fences_new_mutating_commands_before_effect_execution(tmp_path) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    _worker(store, auto_research, root, "active-worker")
    effects = _Effects(store, auto_research, root)
    dispatcher = _dispatcher(store, effects.bundle())

    stopping = request_auto_research_stop(store, auto_research.episode_id)
    response = dispatcher.dispatch(
        root.operation_id,
        _spawn_request("5" * 32, key="spawn-after-stop"),
    )

    assert stopping.status == "stopping"
    assert stopping.stop_requested_at is not None
    assert response.status == "unavailable"
    assert response.message == "The Auto-research episode is no longer accepting mutating commands."
    assert effects.spawn_calls == []
    invocation = store.agent_command("5" * 32)
    assert invocation is not None
    assert invocation.exited_at is not None
    assert invocation.status == "unavailable"


@pytest.mark.asyncio
async def test_auto_research_mailbox_audits_authenticated_mutation_without_key(tmp_path) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    effects = _Effects(store, auto_research, root)
    dispatcher = _dispatcher(store, effects.bundle())
    workspace = tmp_path / "stage"
    workspace.mkdir()
    staged = stage_command_mailbox(
        local_stage=workspace,
        remote_stage=None,
        episode_id=auto_research.episode_id,
        task_id=root.operation_id,
        turn_id="turn",
        timeout_seconds=2,
    )
    request_id = "6" * 32
    request = _spawn_request(request_id, key=None).model_copy(
        update={
            "mailbox_id": staged.credential.mailbox_id,
            "credential": "0" * 64,
        }
    )
    handled = asyncio.Event()

    def handler(parsed, identity):
        assert identity.episode_id == auto_research.episode_id
        assert identity.task_id == root.operation_id
        handled.set()
        return dispatcher.dispatch(identity.task_id, parsed)

    stop = asyncio.Event()
    server = asyncio.create_task(
        serve_command_mailbox(
            staged=staged,
            handler=handler,
            stop=stop,
            poll_seconds=0.01,
            invocation_gate=staged.invocation_gate,
        )
    )
    assert staged.invocation_gate is not None
    async with staged.invocation_gate.serve_current_session():

        def send_request() -> dict[str, object]:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.connect(staged.invocation_gate.socket_path)
                client.sendall(request.model_dump_json().encode("utf-8") + b"\n")
                response = bytearray()
                while not response.endswith(b"\n"):
                    chunk = client.recv(4096)
                    if not chunk:
                        break
                    response.extend(chunk)
            return json.loads(response)

        response = await asyncio.to_thread(send_request)
        await asyncio.wait_for(handled.wait(), timeout=2)
    stop.set()
    await server

    assert response["status"] == "invalid"
    assert "idempotency key" in response["message"]
    assert effects.spawn_calls == []
    invocation = store.agent_command(request_id)
    assert invocation is not None
    assert invocation.idempotency_key is None
    assert invocation.status == "invalid"
    assert invocation.exited_at is not None
    staged.cleanup()


def test_large_validation_records_patch_identity_instead_of_patch_bytes(tmp_path) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    dispatcher = _dispatcher(store, _Effects(store, auto_research, root).bundle())
    patch = '{"summary":"large","ops":[],"padding":"' + ("x" * 64_000) + '"}'

    response = dispatcher.dispatch(
        root.operation_id,
        ValidateCommandRequest(
            mailbox_id=MAILBOX_ID,
            request_id="9" * 32,
            credential=CREDENTIAL,
            verb="validate",
            arguments={"patch": patch},
        ),
    )

    assert response.status == "ok"
    invocation = store.agent_command("9" * 32)
    assert invocation is not None
    assert invocation.start_payload["arguments"] == {
        "patch_byte_length": len(patch.encode("utf-8")),
        "patch_sha256": hashlib.sha256(patch.encode("utf-8")).hexdigest(),
    }


def test_command_result_must_fit_the_durable_event_ledger() -> None:
    with pytest.raises(ValueError, match="event ledger limit"):
        AutoResearchCommandEffectResult(result={"too_large": "x" * 40_000})


def test_status_worker_id_is_normalized_and_bounded_before_durable_start(tmp_path) -> None:
    request = StatusCommandRequest(
        mailbox_id=MAILBOX_ID,
        request_id="8" * 32,
        credential=CREDENTIAL,
        verb="status",
        arguments={"worker_id": "  worker  "},
    )
    assert request.arguments.worker_id == "worker"

    store, auto_research, root = _setup_auto_research(tmp_path)
    dispatcher = _dispatcher(store, _Effects(store, auto_research, root).bundle())
    with pytest.raises(ValueError, match="at most 200 characters"):
        oversized = StatusCommandRequest(
            mailbox_id=MAILBOX_ID,
            request_id="7" * 32,
            credential=CREDENTIAL,
            verb="status",
            arguments={"worker_id": "x" * 201},
        )
        dispatcher.dispatch(root.operation_id, oversized)
    assert store.agent_command("7" * 32) is None


def test_spawn_seat_is_bounded_but_worker_request_gets_no_mechanical_scope(tmp_path) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    effects = _Effects(store, auto_research, root, seat_type="evidence")
    dispatcher = _dispatcher(store, effects.bundle())

    refused = dispatcher.dispatch(
        root.operation_id,
        _spawn_request("2" * 32, key="seat-evidence", seat_node_id="ev/result"),
    )
    assert refused.status == "invalid"
    assert "Experiments and Blockers" in (refused.message or "")
    assert effects.spawn_calls == []

    effects.seat_type = "blocker"
    accepted = dispatcher.dispatch(
        root.operation_id,
        _spawn_request("3" * 32, key="seat-blocker", seat_node_id="blocker/input"),
    )
    assert accepted.status == "ok"
    assert len(effects.spawn_calls) == 1
    worker_id = str(accepted.result["worker_id"])
    worker = store.agent_task(worker_id)
    assert worker is not None
    assert worker.kind == "node_chat"
    assert worker.parent_operation_id is None
    assert worker.authorized_by == auto_research.authorized_by
    assert worker.request["node_id"] == "blocker/input"
    assert worker.request["mode"] == "work"
    assert worker.request["trigger"] == "orchestrator"
    assert "scope" not in worker.request


def test_interrupted_successful_spawn_reconciles_existing_worker_without_restart(tmp_path) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    effects = _Effects(store, auto_research, root)
    dispatcher = _dispatcher(store, effects.bundle())
    key = "dangerous-spawn"
    first_request_id = "4" * 32
    retry_request_id = "5" * 32
    instruction = "Run the check exactly once."
    arguments = SpawnArguments(
        seat_node_id="exp/check",
        instruction_file=_SPAWN_INSTRUCTION_FILE,
    )
    planned_worker_id = _record_interrupted_spawn(
        store,
        auto_research,
        root,
        command_id=first_request_id,
        key=key,
        arguments=arguments,
        instruction=instruction,
    )
    existing = _routed_worker(
        store,
        auto_research,
        admitted_by=root,
        worker_id=planned_worker_id,
        seat_node_id=arguments.seat_node_id,
        instruction=instruction,
        admission_id=planned_worker_id,
    )
    assert store.agent_command(first_request_id).exited_at is None  # type: ignore[union-attr]

    response = dispatcher.dispatch(
        root.operation_id,
        SpawnCommandRequest(
            mailbox_id=MAILBOX_ID,
            request_id=retry_request_id,
            credential=CREDENTIAL,
            verb="spawn",
            idempotency_key=key,
            arguments=arguments,
        ),
    )

    assert response == CommandResponse(
        request_id=retry_request_id,
        status="ok",
        message="The existing Auto-research worker was recovered after interrupted Spawn.",
        result={
            "worker_id": existing.operation_id,
            "status": existing.status,
            "disposition": "existing",
        },
    )
    assert effects.spawn_calls == []
    assert [
        route.worker_id for route in store.auto_research_child_works(auto_research.episode_id)
    ] == [planned_worker_id]
    reconciled = store.agent_command(first_request_id)
    assert reconciled is not None
    assert reconciled.exited_at is not None
    assert reconciled.status == "ok"


def test_interrupted_spawn_rejects_an_existing_worker_with_another_instruction(
    tmp_path,
) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    effects = _Effects(store, auto_research, root)
    dispatcher = _dispatcher(store, effects.bundle())
    key = "instruction-mismatch"
    request_id = "a" * 32
    retry_id = "b" * 32
    instruction = "Run the intended check exactly once."
    arguments = SpawnArguments(
        seat_node_id="exp/check",
        instruction_file=_SPAWN_INSTRUCTION_FILE,
    )
    planned_worker_id = _record_interrupted_spawn(
        store,
        auto_research,
        root,
        command_id=request_id,
        key=key,
        arguments=arguments,
        instruction=instruction,
    )
    _routed_worker(
        store,
        auto_research,
        admitted_by=root,
        worker_id=planned_worker_id,
        seat_node_id=arguments.seat_node_id,
        instruction="Run some different work.",
        admission_id=planned_worker_id,
    )

    response = dispatcher.dispatch(
        root.operation_id,
        SpawnCommandRequest(
            mailbox_id=MAILBOX_ID,
            request_id=retry_id,
            credential=CREDENTIAL,
            verb="spawn",
            idempotency_key=key,
            arguments=arguments,
        ),
    )

    assert response.status == "unavailable"
    assert "instruction" in (response.message or "").lower()
    assert effects.spawn_calls == []


def test_completed_spawn_key_returns_the_existing_worker_and_never_runs_effect_again(
    tmp_path,
) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    effects = _Effects(store, auto_research, root)
    dispatcher = _dispatcher(store, effects.bundle())

    created = dispatcher.dispatch(root.operation_id, _spawn_request("6" * 32, key="spawn-once"))
    replayed = dispatcher.dispatch(root.operation_id, _spawn_request("7" * 32, key="spawn-once"))

    assert created.status == "ok"
    assert replayed.status == "ok"
    assert replayed.request_id == "7" * 32
    assert replayed.result["worker_id"] == created.result["worker_id"]
    assert replayed.result["disposition"] == "existing"
    assert len(effects.spawn_calls) == 1
    assert len(store.auto_research_tasks(auto_research.episode_id)) == 1
    assert len(store.auto_research_child_works(auto_research.episode_id)) == 1


def test_reusing_a_key_with_different_arguments_is_rejected_instead_of_deduplicated(
    tmp_path,
) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    effects = _Effects(store, auto_research, root)
    dispatcher = _dispatcher(store, effects.bundle())
    created = dispatcher.dispatch(root.operation_id, _spawn_request("e" * 32, key="same-key"))

    mismatched = dispatcher.dispatch(
        root.operation_id,
        _spawn_request("f" * 32, key="same-key", seat_node_id="blocker/other"),
    )

    assert created.status == "ok"
    assert mismatched.status == "invalid"
    assert "idempotency" in (mismatched.message or "").lower()
    assert "arguments" in (mismatched.message or "").lower()
    assert len(effects.spawn_calls) == 1


def test_message_and_watch_graph_persist_and_pass_their_planned_effect_ids(tmp_path) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    worker = _worker(store, auto_research, root, "worker")
    effects = _Effects(store, auto_research, root)
    bundled = effects.bundle()
    message_key = "planned-message"
    watcher_key = "planned-watcher"
    message_command_id = "0" * 32
    watcher_command_id = "1" * 32

    def message_after_durable_start(context, arguments, planned_message_id):
        invocation = store.agent_command(message_command_id)
        assert invocation is not None
        assert invocation.exited_at is None
        assert invocation.start_payload["planned_message_id"] == planned_message_id
        return effects.message(context, arguments, planned_message_id)

    def watcher_after_durable_start(context, arguments, planned_watcher_id):
        invocation = store.agent_command(watcher_command_id)
        assert invocation is not None
        assert invocation.exited_at is None
        assert invocation.start_payload["planned_watcher_id"] == planned_watcher_id
        return effects.watch_graph(context, arguments, planned_watcher_id)

    dispatcher = _dispatcher(
        store,
        replace(
            bundled,
            message=message_after_durable_start,
            watch_graph=watcher_after_durable_start,
        ),
    )

    message = dispatcher.dispatch(
        root.operation_id,
        MessageCommandRequest(
            mailbox_id=MAILBOX_ID,
            request_id=message_command_id,
            credential=CREDENTIAL,
            verb="message",
            idempotency_key=message_key,
            arguments={
                "recipient_task_id": worker.operation_id,
                "body": "Use the durable message id.",
            },
        ),
    )
    watch = dispatcher.dispatch(
        root.operation_id,
        WatchGraphCommandRequest(
            mailbox_id=MAILBOX_ID,
            request_id=watcher_command_id,
            credential=CREDENTIAL,
            verb="watch_graph",
            idempotency_key=watcher_key,
            arguments={
                "condition": {"node_id": "hyp/result", "status_in": ["active"]},
                "reason": "Wait for the durable graph condition.",
            },
        ),
    )

    expected_message_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rcp:auto_research:{auto_research.episode_id}:message:{message_key}",
        )
    )
    expected_watcher_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rcp:auto_research:{auto_research.episode_id}:watch_graph:{watcher_key}",
        )
    )
    assert message.status == "ok"
    assert watch.status == "ok"
    assert effects.planned_message_ids == [expected_message_id]
    assert effects.planned_watcher_ids == [expected_watcher_id]
    message_invocation = store.agent_command(message_command_id)
    watcher_invocation = store.agent_command(watcher_command_id)
    assert message_invocation is not None
    assert watcher_invocation is not None
    assert message_invocation.start_payload["planned_message_id"] == expected_message_id
    assert watcher_invocation.start_payload["planned_watcher_id"] == expected_watcher_id


def test_unknown_message_reexecutes_with_its_original_deterministic_effect_id(
    tmp_path,
) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    worker = _worker(store, auto_research, root, "worker")
    effects = _Effects(store, auto_research, root)
    dispatcher = _dispatcher(store, effects.bundle())
    key = "message-once"
    first_request_id = "0" * 32
    arguments = MessageArguments(
        recipient_task_id=worker.operation_id,
        body="Carry this instruction exactly once.",
    )
    planned_message_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rcp:auto_research:{auto_research.episode_id}:message:{key}",
        )
    )
    store.start_agent_command(
        operation_id=root.operation_id,
        command_id=first_request_id,
        episode_id=auto_research.episode_id,
        verb="message",
        idempotency_key=key,
        payload={
            "request_id": first_request_id,
            "arguments": arguments.model_dump(mode="json"),
            "planned_message_id": planned_message_id,
        },
    )

    response = dispatcher.dispatch(
        root.operation_id,
        MessageCommandRequest(
            mailbox_id=MAILBOX_ID,
            request_id="1" * 32,
            credential=CREDENTIAL,
            verb="message",
            idempotency_key=key,
            arguments=arguments,
        ),
    )

    assert response.status == "ok"
    assert effects.reconcile_calls == ["message"]
    assert effects.reconcile_planned_effect_ids == [planned_message_id]
    assert effects.planned_message_ids == [planned_message_id]
    assert effects.message_calls == [arguments]


def test_completed_unavailable_apply_retries_the_original_snapshot_and_effect_id(tmp_path) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    effects = _Effects(store, auto_research, root)
    apply_calls: list[tuple[str, str]] = []
    reads: list[str] = []

    def apply(context, _arguments, planned_apply_id):
        assert context.command_file is not None
        apply_calls.append((planned_apply_id, context.command_file.sha256))
        if len(apply_calls) == 1:
            raise AutoResearchCommandUnavailable("Canonical state is temporarily unreachable.")
        return AutoResearchCommandEffectResult(
            message="The recorded Apply completed.",
            result={"apply_id": planned_apply_id, "disposition": "applied"},
        )

    patch = '{"summary":"settle blocker","ops":[]}'

    def read_patch(filename: str, _max_bytes: int) -> str:
        reads.append(filename)
        return patch

    dispatcher = AutoResearchCommandDispatcher(
        store,
        replace(effects.bundle(), apply=apply),
        command_file_reader=read_patch,
    )
    key = "apply-after-transient-unavailable"
    request = ApplyCommandRequest(
        mailbox_id=MAILBOX_ID,
        request_id="8" * 32,
        credential=CREDENTIAL,
        verb="apply",
        idempotency_key=key,
        arguments={"patch_file": "patch.json"},
    )

    unavailable = dispatcher.dispatch(root.operation_id, request)
    recovered = dispatcher.dispatch(
        root.operation_id,
        request.model_copy(update={"request_id": "9" * 32}),
    )

    expected_apply_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rcp:auto_research:{auto_research.episode_id}:apply:{key}",
        )
    )
    expected_digest = hashlib.sha256(patch.encode()).hexdigest()
    assert unavailable.status == "unavailable"
    assert recovered.status == "ok"
    assert recovered.result == {
        "apply_id": expected_apply_id,
        "disposition": "applied",
    }
    assert apply_calls == [
        (expected_apply_id, expected_digest),
        (expected_apply_id, expected_digest),
    ]
    assert reads == ["patch.json"]
    assert effects.reconcile_calls == ["apply"]
    assert effects.reconcile_planned_effect_ids == [expected_apply_id]
    assert store.auto_research_apply_admission_count(root.operation_id) == 1
    original = store.agent_command(request.request_id)
    assert original is not None
    assert original.status == "unavailable"
    assert original.exited_at is not None
    retry = store.agent_command(recovered.request_id)
    assert retry is not None and retry.status == "ok"


def test_completed_unavailable_spawn_keeps_and_reflects_its_original_admission(tmp_path) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    effects = _Effects(store, auto_research, root)
    spawn_attempts: list[str] = []

    def spawn(context, arguments, planned_worker_id):
        spawn_attempts.append(planned_worker_id)
        if len(spawn_attempts) == 1:
            raise AutoResearchCommandUnavailable("Worker launch transport is unavailable.")
        return effects.spawn(context, arguments, planned_worker_id)

    dispatcher = _dispatcher(store, replace(effects.bundle(), spawn=spawn))
    key = "spawn-after-transient-unavailable"
    request = _spawn_request("0" * 32, key=key)
    expected_worker_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rcp:auto_research:{auto_research.episode_id}:spawn:{key}",
        )
    )

    unavailable = dispatcher.dispatch(root.operation_id, request)
    accepted = store.auto_research_child_admission(expected_worker_id)
    recovered = dispatcher.dispatch(
        root.operation_id,
        request.model_copy(update={"request_id": "1" * 32}),
    )

    assert unavailable.status == "unavailable"
    assert accepted is not None and accepted.state == "accepted"
    assert recovered.status == "ok"
    assert recovered.result["worker_id"] == expected_worker_id
    assert spawn_attempts == [expected_worker_id, expected_worker_id]
    assert [
        route.worker_id for route in store.auto_research_child_works(auto_research.episode_id)
    ] == [expected_worker_id]
    reflected = store.auto_research_child_admission(expected_worker_id)
    assert reflected is not None and reflected.state == "reflected"


def test_completed_unavailable_experiment_kickoff_keeps_and_reflects_admission(
    tmp_path,
) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    effects = _Effects(store, auto_research, root)
    episode_attempts: list[str] = []

    def episode(context, arguments, planned_effect_id):
        episode_attempts.append(planned_effect_id)
        if len(episode_attempts) == 1:
            raise AutoResearchCommandUnavailable("Experiment launch transport is unavailable.")
        assert arguments.action == "kick_off_experiment"
        existing = store.auto_research_child_experiment(planned_effect_id)
        if existing is None:
            now = store.now()
            existing = store.reserve_auto_research_experiment_replacement(
                AutoResearchChildExperimentRecord(
                    child_episode_id=planned_effect_id,
                    auto_research_episode_id=auto_research.episode_id,
                    project_id=auto_research.project_id,
                    control_node_id=arguments.node_id,
                    state="pending",
                    replaces_episode_id="predecessor",
                    request={"goal": "bounded goal", "invocation_limit": None},
                    goal_sha256=hashlib.sha256(b"bounded goal").hexdigest(),
                    parent_operation_id=context.task.operation_id,
                    created_at=now,
                    updated_at=now,
                ),
                admission_id=planned_effect_id,
            )
        return AutoResearchCommandEffectResult(
            result={
                "disposition": "replacement_pending",
                "episode_id": existing.child_episode_id,
            }
        )

    dispatcher = _dispatcher(store, replace(effects.bundle(), episode=episode))
    key = "experiment-after-transient-unavailable"
    request = EpisodeCommandRequest(
        mailbox_id=MAILBOX_ID,
        request_id="2" * 32,
        credential=CREDENTIAL,
        verb="episode",
        idempotency_key=key,
        arguments={
            "action": "kick_off_experiment",
            "node_id": "exp/check",
        },
    )
    expected_episode_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rcp:auto_research:{auto_research.episode_id}:episode:{key}",
        )
    )

    unavailable = dispatcher.dispatch(root.operation_id, request)
    accepted = store.auto_research_child_admission(expected_episode_id)
    recovered = dispatcher.dispatch(
        root.operation_id,
        request.model_copy(update={"request_id": "3" * 32}),
    )

    assert unavailable.status == "unavailable"
    assert accepted is not None and accepted.state == "accepted"
    assert recovered.status == "ok"
    assert recovered.result["episode_id"] == expected_episode_id
    assert episode_attempts == [expected_episode_id, expected_episode_id]
    assert [
        route.child_episode_id
        for route in store.auto_research_child_experiments(auto_research.episode_id)
    ] == [expected_episode_id]
    reflected = store.auto_research_child_admission(expected_episode_id)
    assert reflected is not None and reflected.state == "reflected"


def test_semantically_invalid_spawn_cancels_its_child_admission(tmp_path) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    effects = _Effects(store, auto_research, root)

    def invalid_spawn(_context, _arguments, _planned_worker_id):
        return AutoResearchCommandEffectResult(
            status="invalid",
            message="The child request is semantically invalid.",
        )

    dispatcher = _dispatcher(store, replace(effects.bundle(), spawn=invalid_spawn))
    key = "invalid-spawn-cancels-admission"
    response = dispatcher.dispatch(
        root.operation_id,
        _spawn_request("4" * 32, key=key),
    )
    expected_worker_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rcp:auto_research:{auto_research.episode_id}:spawn:{key}",
        )
    )

    assert response.status == "invalid"
    admission = store.auto_research_child_admission(expected_worker_id)
    assert admission is not None and admission.state == "cancelled"


def test_completed_unavailable_worker_resume_reuses_the_planned_operation_id(tmp_path) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    worker = _worker(store, auto_research, root, "worker")
    other_worker = _worker(store, auto_research, root, "other-worker")
    effects = _Effects(store, auto_research, root)

    def resume(context, worker_id, planned_operation_id):
        effects.resume_operation_ids.append(planned_operation_id)
        if len(effects.resume_operation_ids) == 1:
            raise AutoResearchCommandUnavailable("Execution host is temporarily unreachable.")
        return AutoResearchCommandEffectResult(
            result={
                "disposition": "resumed",
                "worker_id": worker_id,
                "current_operation_id": planned_operation_id,
            }
        )

    dispatcher = _dispatcher(store, replace(effects.bundle(), resume=resume))
    key = "resume-worker-after-transient-unavailable"
    request = ResumeCommandRequest(
        mailbox_id=MAILBOX_ID,
        request_id="b" * 32,
        credential=CREDENTIAL,
        verb="resume",
        idempotency_key=key,
        arguments={"worker_id": worker.operation_id},
    )

    unavailable = dispatcher.dispatch(root.operation_id, request)
    mismatched = dispatcher.dispatch(
        root.operation_id,
        request.model_copy(
            update={
                "request_id": "c" * 32,
                "arguments": request.arguments.model_copy(
                    update={"worker_id": other_worker.operation_id}
                ),
            }
        ),
    )
    recovered = dispatcher.dispatch(
        root.operation_id,
        request.model_copy(update={"request_id": "a" * 32}),
    )

    expected_operation_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rcp:auto_research:{auto_research.episode_id}:resume:{key}",
        )
    )
    assert unavailable.status == "unavailable"
    assert mismatched.status == "invalid"
    assert "different command arguments" in (mismatched.message or "")
    assert recovered.status == "ok"
    assert recovered.result["current_operation_id"] == expected_operation_id
    assert effects.resume_operation_ids == [expected_operation_id, expected_operation_id]
    assert effects.reconcile_planned_effect_ids == [expected_operation_id]
    original = store.agent_command(request.request_id)
    assert original is not None and original.status == "unavailable"


def test_completed_unavailable_experiment_resume_reuses_the_planned_operation_id(
    tmp_path,
) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    effects = _Effects(store, auto_research, root)

    def episode(_context, arguments, planned_effect_id):
        effects.episode_effect_ids.append(planned_effect_id)
        if len(effects.episode_effect_ids) == 1:
            raise AutoResearchCommandUnavailable("Execution host is temporarily unreachable.")
        return AutoResearchCommandEffectResult(
            result={
                "disposition": arguments.action,
                "episode_id": arguments.episode_id,
                "operation_id": planned_effect_id,
            }
        )

    dispatcher = _dispatcher(store, replace(effects.bundle(), episode=episode))
    key = "resume-experiment-after-transient-unavailable"
    request = EpisodeCommandRequest(
        mailbox_id=MAILBOX_ID,
        request_id="d" * 32,
        credential=CREDENTIAL,
        verb="episode",
        idempotency_key=key,
        arguments={
            "action": "resume",
            "episode_id": "00000000-0000-4000-8000-000000000997",
        },
    )

    unavailable = dispatcher.dispatch(root.operation_id, request)
    recovered = dispatcher.dispatch(
        root.operation_id,
        request.model_copy(update={"request_id": "e" * 32}),
    )

    expected_operation_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rcp:auto_research:{auto_research.episode_id}:episode:{key}",
        )
    )
    assert unavailable.status == "unavailable"
    assert recovered.status == "ok"
    assert recovered.result["operation_id"] == expected_operation_id
    assert effects.episode_effect_ids == [expected_operation_id, expected_operation_id]
    assert effects.reconcile_planned_effect_ids == [expected_operation_id]
    original = store.agent_command(request.request_id)
    assert original is not None and original.status == "unavailable"


@pytest.mark.parametrize("effect_name", ["message", "watch_graph", "inbox"])
def test_completed_unavailable_idempotent_effect_reexecutes_with_recorded_id(
    tmp_path,
    effect_name,
) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    worker = _worker(store, auto_research, root, "worker")
    effects = _Effects(store, auto_research, root)
    effect_attempts: list[str] = []

    def transient_effect(_context, _arguments, planned_effect_id):
        effect_attempts.append(planned_effect_id)
        if len(effect_attempts) == 1:
            raise AutoResearchCommandUnavailable("The effect store is temporarily unavailable.")
        return AutoResearchCommandEffectResult(
            result={"effect_id": planned_effect_id, "disposition": "created"}
        )

    dispatcher = _dispatcher(
        store,
        replace(effects.bundle(), **{effect_name: transient_effect}),
    )
    key = f"{effect_name}-after-transient-unavailable"
    request = _remaining_idempotent_request(
        effect_name,
        request_id="5" * 32,
        key=key,
        worker_id=worker.operation_id,
    )

    unavailable = dispatcher.dispatch(root.operation_id, request)
    recovered = dispatcher.dispatch(
        root.operation_id,
        request.model_copy(update={"request_id": "6" * 32}),
    )

    expected_effect_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rcp:auto_research:{auto_research.episode_id}:{effect_name}:{key}",
        )
    )
    assert unavailable.status == "unavailable"
    assert recovered.status == "ok"
    assert recovered.result == {
        "effect_id": expected_effect_id,
        "disposition": "created",
    }
    assert effect_attempts == [expected_effect_id, expected_effect_id]
    assert effects.reconcile_calls == [effect_name]
    assert effects.reconcile_planned_effect_ids == [expected_effect_id]


@pytest.mark.parametrize("effect_name", ["message", "watch_graph", "inbox"])
def test_completed_unavailable_idempotent_effect_returns_reconciled_commit(
    tmp_path,
    effect_name,
) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    worker = _worker(store, auto_research, root, "worker")
    reconciled = AutoResearchCommandEffectResult(
        message="The durable effect already exists.",
        result={"disposition": "existing"},
    )
    effects = _Effects(store, auto_research, root, reconcile_result=reconciled)
    effect_attempts: list[str] = []

    def committed_then_unavailable(_context, _arguments, planned_effect_id):
        effect_attempts.append(planned_effect_id)
        raise AutoResearchCommandUnavailable("The response transport was interrupted.")

    dispatcher = _dispatcher(
        store,
        replace(effects.bundle(), **{effect_name: committed_then_unavailable}),
    )
    key = f"{effect_name}-committed-before-unavailable"
    request = _remaining_idempotent_request(
        effect_name,
        request_id="7" * 32,
        key=key,
        worker_id=worker.operation_id,
    )

    unavailable = dispatcher.dispatch(root.operation_id, request)
    recovered = dispatcher.dispatch(
        root.operation_id,
        request.model_copy(update={"request_id": "8" * 32}),
    )

    expected_effect_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rcp:auto_research:{auto_research.episode_id}:{effect_name}:{key}",
        )
    )
    assert unavailable.status == "unavailable"
    assert recovered.status == "ok"
    assert recovered.result == {"disposition": "existing"}
    assert effect_attempts == [expected_effect_id]
    assert effects.reconcile_calls == [effect_name]
    assert effects.reconcile_planned_effect_ids == [expected_effect_id]


def test_unknown_worker_resume_reexecutes_with_the_original_deterministic_operation_id(
    tmp_path,
) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    worker = _worker(store, auto_research, root, "worker")
    effects = _Effects(store, auto_research, root)
    dispatcher = _dispatcher(store, effects.bundle())
    key = "resume-worker-once"
    original_request_id = "2" * 32
    planned_operation_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rcp:auto_research:{auto_research.episode_id}:resume:{key}",
        )
    )
    arguments = {"worker_id": worker.operation_id}
    store.start_agent_command(
        operation_id=root.operation_id,
        command_id=original_request_id,
        episode_id=auto_research.episode_id,
        verb="resume",
        idempotency_key=key,
        payload={
            "request_id": original_request_id,
            "arguments": arguments,
            "planned_resume_operation_id": planned_operation_id,
        },
    )

    response = dispatcher.dispatch(
        root.operation_id,
        ResumeCommandRequest(
            mailbox_id=MAILBOX_ID,
            request_id="3" * 32,
            credential=CREDENTIAL,
            verb="resume",
            idempotency_key=key,
            arguments=arguments,
        ),
    )

    assert response.status == "ok"
    assert response.result["current_operation_id"] == planned_operation_id
    assert effects.reconcile_planned_effect_ids == [planned_operation_id]
    assert effects.resume_operation_ids == [planned_operation_id]
    original = store.agent_command(original_request_id)
    assert original is not None and original.exited_at is not None


def test_unknown_experiment_resume_reexecutes_with_the_original_deterministic_operation_id(
    tmp_path,
) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    effects = _Effects(store, auto_research, root)
    dispatcher = _dispatcher(store, effects.bundle())
    key = "resume-experiment-once"
    original_request_id = "4" * 32
    planned_operation_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rcp:auto_research:{auto_research.episode_id}:episode:{key}",
        )
    )
    arguments = {
        "action": "resume",
        "episode_id": "00000000-0000-4000-8000-000000000999",
    }
    store.start_agent_command(
        operation_id=root.operation_id,
        command_id=original_request_id,
        episode_id=auto_research.episode_id,
        verb="episode",
        idempotency_key=key,
        payload={
            "request_id": original_request_id,
            "arguments": arguments,
            "planned_episode_effect_id": planned_operation_id,
        },
    )

    response = dispatcher.dispatch(
        root.operation_id,
        EpisodeCommandRequest(
            mailbox_id=MAILBOX_ID,
            request_id="5" * 32,
            credential=CREDENTIAL,
            verb="episode",
            idempotency_key=key,
            arguments=arguments,
        ),
    )

    assert response.status == "ok"
    assert response.result["operation_id"] == planned_operation_id
    assert effects.reconcile_planned_effect_ids == [planned_operation_id]
    assert effects.episode_effect_ids == [planned_operation_id]
    original = store.agent_command(original_request_id)
    assert original is not None and original.exited_at is not None


@pytest.mark.parametrize(
    ("verb", "request_type", "call_field"),
    [
        ("pause", PauseCommandRequest, "pause_worker_ids"),
        ("stop", StopCommandRequest, "stop_worker_ids"),
    ],
)
def test_unknown_worker_control_reissues_only_the_idempotent_pause_or_stop(
    tmp_path,
    verb,
    request_type,
    call_field,
) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    worker = _worker(store, auto_research, root, "worker")
    effects = _Effects(store, auto_research, root)
    dispatcher = _dispatcher(store, effects.bundle())
    key = f"{verb}-worker-once"
    original_request_id = "6" * 32
    arguments = {"worker_id": worker.operation_id}
    store.start_agent_command(
        operation_id=root.operation_id,
        command_id=original_request_id,
        episode_id=auto_research.episode_id,
        verb=verb,
        idempotency_key=key,
        payload={"request_id": original_request_id, "arguments": arguments},
    )

    response = dispatcher.dispatch(
        root.operation_id,
        request_type(
            mailbox_id=MAILBOX_ID,
            request_id="7" * 32,
            credential=CREDENTIAL,
            verb=verb,
            idempotency_key=key,
            arguments=arguments,
        ),
    )

    assert response.status == "ok"
    assert effects.reconcile_calls == [verb]
    assert getattr(effects, call_field) == [worker.operation_id]


def test_unknown_experiment_stop_reissues_the_monotonic_stop_path(tmp_path) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    effects = _Effects(store, auto_research, root)
    dispatcher = _dispatcher(store, effects.bundle())
    key = "stop-experiment-once"
    original_request_id = "8" * 32
    arguments = {
        "action": "stop",
        "episode_id": "00000000-0000-4000-8000-000000000998",
    }
    planned_effect_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rcp:auto_research:{auto_research.episode_id}:episode:{key}",
        )
    )
    store.start_agent_command(
        operation_id=root.operation_id,
        command_id=original_request_id,
        episode_id=auto_research.episode_id,
        verb="episode",
        idempotency_key=key,
        payload={
            "request_id": original_request_id,
            "arguments": arguments,
            "planned_episode_effect_id": planned_effect_id,
        },
    )

    response = dispatcher.dispatch(
        root.operation_id,
        EpisodeCommandRequest(
            mailbox_id=MAILBOX_ID,
            request_id="9" * 32,
            credential=CREDENTIAL,
            verb="episode",
            idempotency_key=key,
            arguments=arguments,
        ),
    )

    assert response.status == "ok"
    assert effects.reconcile_calls == ["episode"]
    assert effects.episode_effect_ids == [planned_effect_id]


def test_unknown_watch_retry_uses_and_validates_the_original_planned_watcher_id(
    tmp_path,
) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    effects = _Effects(
        store,
        auto_research,
        root,
        reconcile_result=AutoResearchCommandEffectResult(result={"watcher_id": "existing"}),
    )
    dispatcher = _dispatcher(store, effects.bundle())
    key = "watch-once"
    original_request_id = "4" * 32
    arguments = WatchGraphCommandRequest(
        mailbox_id=MAILBOX_ID,
        request_id=original_request_id,
        credential=CREDENTIAL,
        verb="watch_graph",
        idempotency_key=key,
        arguments={
            "condition": {"node_id": "hyp/result", "status_in": ["active"]},
            "reason": "Wait for the original watcher.",
        },
    ).arguments
    planned_watcher_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rcp:auto_research:{auto_research.episode_id}:watch_graph:{key}",
        )
    )
    store.start_agent_command(
        operation_id=root.operation_id,
        command_id=original_request_id,
        episode_id=auto_research.episode_id,
        verb="watch_graph",
        idempotency_key=key,
        payload={
            "request_id": original_request_id,
            "arguments": arguments.model_dump(mode="json"),
            "planned_watcher_id": planned_watcher_id,
        },
    )

    response = dispatcher.dispatch(
        root.operation_id,
        WatchGraphCommandRequest(
            mailbox_id=MAILBOX_ID,
            request_id="5" * 32,
            credential=CREDENTIAL,
            verb="watch_graph",
            idempotency_key=key,
            arguments=arguments,
        ),
    )

    assert response.status == "ok"
    assert response.result == {"watcher_id": "existing"}
    assert effects.reconcile_calls == ["watch_graph"]
    assert effects.reconcile_planned_effect_ids == [planned_watcher_id]
    assert effects.planned_watcher_ids == []

    # A malformed durable planned id is unavailable and never reaches reconciliation.
    mismatch_path = tmp_path / "mismatch"
    mismatch_path.mkdir()
    store, auto_research, root = _setup_auto_research(mismatch_path)
    effects = _Effects(store, auto_research, root)
    dispatcher = _dispatcher(store, effects.bundle())
    store.start_agent_command(
        operation_id=root.operation_id,
        command_id="6" * 32,
        episode_id=auto_research.episode_id,
        verb="watch_graph",
        idempotency_key=key,
        payload={
            "request_id": "6" * 32,
            "arguments": arguments.model_dump(mode="json"),
            "planned_watcher_id": str(uuid.uuid4()),
        },
    )
    refused = dispatcher.dispatch(
        root.operation_id,
        WatchGraphCommandRequest(
            mailbox_id=MAILBOX_ID,
            request_id="7" * 32,
            credential=CREDENTIAL,
            verb="watch_graph",
            idempotency_key=key,
            arguments=arguments,
        ),
    )
    assert refused.status == "unavailable"
    assert "deterministic effect id" in (refused.message or "")
    assert effects.reconcile_calls == []


def test_orchestrator_message_requires_the_stable_worker_actor_id_before_effect_or_spend(
    tmp_path,
) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    worker = _worker(store, auto_research, root, "worker")
    effects = _Effects(store, auto_research, root)
    dispatcher = _dispatcher(store, effects.bundle())

    accepted = dispatcher.dispatch(
        root.operation_id,
        MessageCommandRequest(
            mailbox_id=MAILBOX_ID,
            request_id="2" * 32,
            credential=CREDENTIAL,
            verb="message",
            idempotency_key="message-stable-worker",
            arguments={
                "recipient_task_id": worker.operation_id,
                "body": "Continue the bounded check.",
            },
        ),
    )

    assert accepted.status == "ok"
    assert effects.message_calls == [
        MessageArguments(
            recipient_task_id=worker.operation_id,
            body="Continue the bounded check.",
        )
    ]

    store.complete_agent_task(worker.operation_id, applied_revision=None, result={})
    worker = store.agent_task(worker.operation_id)
    assert worker is not None
    worker_request = AutoResearchRunRequest.model_validate(worker.request)
    now = store.now()
    continuation = store.create_auto_research_agent_task(
        AgentTaskRecord(
            operation_id="worker-continuation",
            project_id=auto_research.project_id,
            episode_id=auto_research.episode_id,
            graph_target=auto_research.graph_target,
            kind="auto_research",
            status="queued",
            request=worker_request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="queued",
            parent_operation_id=worker.operation_id,
            authorized_by=auto_research.authorized_by,
            dispatch_authority=worker.dispatch_authority,
        ),
        role="worker",
    )
    effects.message_calls.clear()
    budget_before = store.episode_budget_meter(auto_research.episode_id)

    refused = dispatcher.dispatch(
        root.operation_id,
        MessageCommandRequest(
            mailbox_id=MAILBOX_ID,
            request_id="3" * 32,
            credential=CREDENTIAL,
            verb="message",
            idempotency_key="message-worker-continuation",
            arguments={
                "recipient_task_id": continuation.operation_id,
                "body": "Do not create another paid delivery.",
            },
        ),
    )

    assert refused.status == "invalid"
    assert "stable worker id" in (refused.message or "")
    assert effects.message_calls == []
    assert store.episode_budget_meter(auto_research.episode_id) == budget_before


@pytest.mark.parametrize(
    ("worker_role", "seat_node_id", "parent_matches_context"),
    [
        ("orchestrator", None, True),
        ("worker", "blocker/not-the-requested-seat", True),
        ("worker", "exp/check", False),
    ],
)
def test_spawn_verifies_worker_role_exact_seat_and_parent_before_reporting_success(
    tmp_path,
    worker_role,
    seat_node_id,
    parent_matches_context,
) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    effects = _Effects(store, auto_research, root)
    command_task = root
    if not parent_matches_context:
        command_task = _orchestrator_turn(
            store,
            auto_research,
            root,
            operation_id="later-orchestrator-turn",
        )

    def malformed_spawn(context, arguments, planned_worker_id):
        assert context.command_file is not None
        request = AutoResearchRunRequest(
            episode_id=auto_research.episode_id,
            role=worker_role,
            actor_operation_id=(
                root.operation_id if worker_role == "orchestrator" else planned_worker_id
            ),
            run_truth_scope=_RUN_TRUTH_SCOPE,
            control_node_id=seat_node_id,
            instruction=context.command_file.text,
        )
        now = store.now()
        store.create_auto_research_agent_task(
            AgentTaskRecord(
                operation_id=planned_worker_id,
                project_id=auto_research.project_id,
                episode_id=auto_research.episode_id,
                graph_target=auto_research.graph_target,
                kind="auto_research",
                status="queued",
                request=request.model_dump(mode="json"),
                created_at=now,
                updated_at=now,
                status_message="queued",
                parent_operation_id=root.operation_id,
                authorized_by=auto_research.authorized_by,
                dispatch_authority=_auto_research_authority(
                    auto_research.episode_id,
                    worker_role,
                ),
            ),
            role=worker_role,
        )
        return AutoResearchCommandEffectResult(result={"worker_id": planned_worker_id})

    dispatcher = _dispatcher(
        store,
        replace(effects.bundle(), spawn=malformed_spawn),
    )
    response = dispatcher.dispatch(
        command_task.operation_id,
        _spawn_request("2" * 32, key="verify-postconditions"),
    )

    assert response.status == "unavailable"
    assert any(word in (response.message or "").lower() for word in ("role", "seat", "parent"))


def test_deduplicated_client_attempt_is_audited_on_the_current_task(tmp_path) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    effects = _Effects(store, auto_research, root)
    dispatcher = _dispatcher(store, effects.bundle())
    first = dispatcher.dispatch(root.operation_id, _spawn_request("3" * 32, key="audit-replay"))
    current = store.create_auto_research_agent_task(
        AgentTaskRecord(
            operation_id="later-orchestrator-turn",
            project_id=auto_research.project_id,
            episode_id=auto_research.episode_id,
            graph_target=auto_research.graph_target,
            kind="auto_research",
            status="succeeded",
            request=AutoResearchRunRequest(
                episode_id=auto_research.episode_id,
                role="orchestrator",
                actor_operation_id=root.operation_id,
                run_truth_scope=_RUN_TRUTH_SCOPE,
            ).model_dump(mode="json"),
            created_at=store.now(),
            updated_at=store.now(),
            status_message="done",
            parent_operation_id=root.operation_id,
            authorized_by=auto_research.authorized_by,
            dispatch_authority=_auto_research_authority(
                auto_research.episode_id,
                "orchestrator",
            ),
        ),
        role="orchestrator",
    )

    replayed = dispatcher.dispatch(
        current.operation_id,
        _spawn_request("4" * 32, key="audit-replay"),
    )

    assert replayed.status == "ok"
    assert replayed.result["worker_id"] == first.result["worker_id"]
    assert len(effects.spawn_calls) == 1
    current_events = store.agent_task_events(current.operation_id)
    assert any(
        "spawn" in event.message.lower()
        and any(
            marker in event.message.lower()
            for marker in ("idempotency", "existing", "reused", "duplicate")
        )
        for event in current_events
    )


@pytest.mark.parametrize("original_completed", [False, True])
def test_worker_cannot_replay_an_orchestrator_idempotency_key(
    tmp_path,
    original_completed: bool,
) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    effects = _Effects(store, auto_research, root)
    dispatcher = _dispatcher(store, effects.bundle())
    key = "orchestrator-only-key"
    first_request_id = "c" * 32
    request = _spawn_request(first_request_id, key=key)

    if original_completed:
        original = dispatcher.dispatch(root.operation_id, request)
        assert original.status == "ok"
    else:
        planned_worker_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"rcp:auto_research:{auto_research.episode_id}:spawn:{key}",
            )
        )
        store.start_agent_command(
            operation_id=root.operation_id,
            command_id=first_request_id,
            episode_id=auto_research.episode_id,
            verb="spawn",
            idempotency_key=key,
            payload={
                "request_id": first_request_id,
                "arguments": request.arguments.model_dump(mode="json"),
                "planned_worker_id": planned_worker_id,
            },
        )
    caller = _worker(store, auto_research, root, "caller-worker")

    refused = dispatcher.dispatch(
        caller.operation_id,
        request.model_copy(update={"request_id": "d" * 32}),
    )

    assert refused.status == "invalid"
    assert "same canonical Auto-research actor and role" in (refused.message or "")
    original_invocation = store.agent_command(first_request_id)
    assert original_invocation is not None
    assert (original_invocation.exited_at is not None) is original_completed
    assert len(effects.spawn_calls) == int(original_completed)
    retry_attempt = store.agent_command("d" * 32)
    assert retry_attempt is not None and retry_attempt.status == "invalid"


def test_worker_may_reply_only_by_message_while_other_mutations_remain_orchestrator_only(
    tmp_path,
) -> None:
    store, auto_research, root = _setup_auto_research(tmp_path)
    worker = _worker(store, auto_research, root, "worker")
    effects = _Effects(store, auto_research, root)
    dispatcher = _dispatcher(store, effects.bundle())

    reply = dispatcher.dispatch(
        worker.operation_id,
        MessageCommandRequest(
            mailbox_id=MAILBOX_ID,
            request_id="8" * 32,
            credential=CREDENTIAL,
            verb="message",
            idempotency_key="worker-reply",
            arguments={
                "recipient_task_id": root.operation_id,
                "body": "The bounded check finished.",
            },
        ),
    )
    assert reply.status == "ok"
    assert effects.message_calls == [
        MessageArguments(
            recipient_task_id=root.operation_id,
            body="The bounded check finished.",
        )
    ]

    forbidden = [
        SpawnCommandRequest(
            mailbox_id=MAILBOX_ID,
            request_id="9" * 32,
            credential=CREDENTIAL,
            verb="spawn",
            idempotency_key="worker-spawn",
            arguments={
                "seat_node_id": "exp/other",
                "instruction_file": _SPAWN_INSTRUCTION_FILE,
            },
        ),
        PauseCommandRequest(
            mailbox_id=MAILBOX_ID,
            request_id="a" * 32,
            credential=CREDENTIAL,
            verb="pause",
            idempotency_key="worker-pause",
            arguments={"worker_id": worker.operation_id},
        ),
        ResumeCommandRequest(
            mailbox_id=MAILBOX_ID,
            request_id="b" * 32,
            credential=CREDENTIAL,
            verb="resume",
            idempotency_key="worker-resume",
            arguments={"worker_id": worker.operation_id},
        ),
        StopCommandRequest(
            mailbox_id=MAILBOX_ID,
            request_id="c" * 32,
            credential=CREDENTIAL,
            verb="stop",
            idempotency_key="worker-stop",
            arguments={"worker_id": worker.operation_id},
        ),
        WatchGraphCommandRequest(
            mailbox_id=MAILBOX_ID,
            request_id="d" * 32,
            credential=CREDENTIAL,
            verb="watch_graph",
            idempotency_key="worker-watch",
            arguments={
                "condition": {"node_id": "hyp/result", "status_in": ["active"]},
                "reason": "Wait for the belief transition.",
            },
        ),
        FinishCommandRequest(
            mailbox_id=MAILBOX_ID,
            request_id="e" * 32,
            credential=CREDENTIAL,
            verb="finish",
            idempotency_key="worker-finish",
        ),
    ]
    for request in forbidden:
        response = dispatcher.dispatch(worker.operation_id, request)
        assert response.status == "invalid"
        assert "Only the Auto-research orchestrator" in (response.message or "")
    assert effects.spawn_calls == []
