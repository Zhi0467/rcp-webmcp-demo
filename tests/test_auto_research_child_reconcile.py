from __future__ import annotations

import hashlib
import uuid

from fastapi.testclient import TestClient

from rcp.agents import AgentEvent, AgentProcessControl
from rcp.agents.command_protocol import SpawnArguments
from rcp.background import AgentTaskExecution, BackgroundAgentTasks
from rcp.core.authority import AgentDispatchAuthority, AgentDispatchScope
from rcp.core.models import Patch
from rcp.core.transition_models import GraphHeadRef, GraphTargetRef
from rcp.runs.auto_research import (
    AutoResearchCommandContext,
    AutoResearchRunRequest,
)
from rcp.runs.auto_research_admission import (
    start_auto_research_child_work,
)
from rcp.runs.auto_research_child_reconcile import (
    reconcile_pending_auto_research_child_admissions,
)
from rcp.runs.auto_research_experiments import AutoResearchExperimentAction
from rcp.service import RunRequest
from rcp.storage import (
    AgentTaskRecord,
    AppStore,
    AutoResearchChildAdmissionRecord,
    AutoResearchChildExperimentRecord,
    AutoResearchCommandFileRecord,
    AutoResearchStateRecord,
    EpisodeRecord,
    ProjectRecord,
)

from .helpers import create_named_app, fabricated_authorizer, wait_for_task


async def _successful_stream(_project_id, _kind, _request, _execution):
    yield f"data: {AgentEvent(event='done').model_dump_json()}\n\n"


def _setup(tmp_path) -> tuple[AppStore, EpisodeRecord, AgentTaskRecord]:
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
        run_truth_scope=["repo-a"],
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
            dispatch_authority=AgentDispatchAuthority(
                profile="orchestrator",
                task_contract="orchestrate",
                scope=AgentDispatchScope(
                    run_truth_scope=["repo-a"],
                    episode_id="auto_research",
                    patch_kind="work",
                ),
            ),
        ),
    )
    store.complete_agent_task(root.operation_id, applied_revision=None, result={})
    stored_episode = store.episode(episode.episode_id)
    stored_root = store.agent_task(root.operation_id)
    assert stored_episode is not None and stored_root is not None
    return store, stored_episode, stored_root


def _worker_request(
    _context: AutoResearchCommandContext,
    arguments: SpawnArguments,
    instruction: str,
    worker_id: str,
) -> RunRequest:
    return RunRequest(
        provider="codex",
        run_on="local",
        run_truth_scope=["repo-a"],
        chat_id=worker_id,
        chat_scope="node",
        node_id=arguments.seat_node_id,
        message=instruction,
        mode="work",
        trigger="orchestrator",
        patch_kind="work",
    )


def _admit_command(
    store: AppStore,
    episode: EpisodeRecord,
    root: AgentTaskRecord,
    *,
    command_id: str,
    key: str,
    child_id: str,
    child_kind: str,
    arguments: dict[str, object],
    file_kind: str | None,
    filename: str | None,
    content: str | None,
) -> None:
    now = store.now()
    command_file = None
    command_file_metadata = None
    if file_kind is not None and filename is not None and content is not None:
        digest = hashlib.sha256(content.encode()).hexdigest()
        command_file = AutoResearchCommandFileRecord(
            command_id=command_id,
            episode_id=episode.episode_id,
            operation_id=root.operation_id,
            kind=file_kind,
            filename=filename,
            sha256=digest,
            content=content,
            created_at=now,
        )
        command_file_metadata = {
            "filename": filename,
            "byte_length": len(content.encode()),
            "sha256": digest,
        }
    planned_field = "planned_worker_id" if child_kind == "work" else "planned_episode_effect_id"
    payload: dict[str, object] = {
        "request_id": uuid.uuid4().hex,
        "arguments": arguments,
        planned_field: child_id,
    }
    if command_file_metadata is not None:
        payload["command_file"] = command_file_metadata
    store.start_agent_command(
        operation_id=root.operation_id,
        command_id=command_id,
        episode_id=episode.episode_id,
        verb="spawn" if child_kind == "work" else "episode",
        idempotency_key=key,
        payload=payload,
        file_snapshot=command_file,
        child_admission=AutoResearchChildAdmissionRecord(
            admission_id=child_id,
            episode_id=episode.episode_id,
            project_id=episode.project_id,
            child_kind=child_kind,
            child_id=child_id,
            state="accepted",
            created_at=now,
            updated_at=now,
        ),
    )


class _UnusedExperimentCoordinator:
    def kick_off(self, **_kwargs):
        raise AssertionError("Experiment coordinator was not expected")


def test_restart_reconciles_spawn_from_immutable_snapshot_once(tmp_path) -> None:
    store, episode, root = _setup(tmp_path)
    key = "spawn-after-runtime-repair"
    child_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rcp:auto_research:{episode.episode_id}:spawn:{key}",
        )
    )
    instruction = "Inspect the failed runtime and continue the bounded work."
    _admit_command(
        store,
        episode,
        root,
        command_id="spawn-command",
        key=key,
        child_id=child_id,
        child_kind="work",
        arguments={"seat_node_id": "blk/runtime", "instruction_file": "worker.md"},
        file_kind="instruction",
        filename="worker.md",
        content=instruction,
    )
    command = store.auto_research_child_admission_command(child_id)
    assert command is not None and command.command_id == "spawn-command"
    before = store.episode_budget_meter(episode.episode_id)
    background = BackgroundAgentTasks(store, _successful_stream)

    first = reconcile_pending_auto_research_child_admissions(
        store,
        background,
        _UnusedExperimentCoordinator(),  # type: ignore[arg-type]
        worker_request_factory=_worker_request,
        seat_node_type=lambda _project_id, _episode_id, _node_id: "blocker",
    )
    second = reconcile_pending_auto_research_child_admissions(
        store,
        background,
        _UnusedExperimentCoordinator(),  # type: ignore[arg-type]
        worker_request_factory=_worker_request,
        seat_node_type=lambda _project_id, _episode_id, _node_id: "blocker",
    )

    route = store.auto_research_child_work(child_id)
    task = store.agent_task(child_id)
    admission = store.auto_research_child_admission(child_id)
    invocation = store.agent_command("spawn-command")
    assert first.examined == 1 and first.reflected == 1 and first.cancelled == 0
    assert second.examined == 0
    assert route is not None and route.instruction == instruction
    assert task is not None and task.request["message"] == instruction
    assert admission is not None and admission.state == "reflected"
    assert invocation is not None and invocation.status == "ok"
    assert store.episode_budget_meter(episode.episode_id).invocations_used == (
        before.invocations_used + 1
    )
    wait_for_task(store, child_id)


def test_transient_recovery_failure_keeps_admission_for_exact_later_retry(tmp_path) -> None:
    store, episode, root = _setup(tmp_path)
    key = "transient-state-unavailable"
    child_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rcp:auto_research:{episode.episode_id}:spawn:{key}",
        )
    )
    instruction = "Use the original durable instruction after state recovers."
    _admit_command(
        store,
        episode,
        root,
        command_id="transient-spawn-command",
        key=key,
        child_id=child_id,
        child_kind="work",
        arguments={"seat_node_id": "blk/runtime", "instruction_file": "worker.md"},
        file_kind="instruction",
        filename="worker.md",
        content=instruction,
    )
    background = BackgroundAgentTasks(store, _successful_stream)
    before = store.episode_budget_meter(episode.episode_id)
    attempts = 0
    received_instructions: list[str] = []

    def unavailable_then_ready(
        context: AutoResearchCommandContext,
        arguments: SpawnArguments,
        recovered_instruction: str,
        worker_id: str,
    ) -> RunRequest:
        nonlocal attempts
        attempts += 1
        received_instructions.append(recovered_instruction)
        if attempts == 1:
            raise OSError("canonical state is temporarily unavailable")
        return _worker_request(context, arguments, recovered_instruction, worker_id)

    first = reconcile_pending_auto_research_child_admissions(
        store,
        background,
        _UnusedExperimentCoordinator(),  # type: ignore[arg-type]
        worker_request_factory=unavailable_then_ready,
        seat_node_type=lambda _project_id, _episode_id, _node_id: "blocker",
    )

    admission = store.auto_research_child_admission(child_id)
    command = store.agent_command("transient-spawn-command")
    snapshot = store.auto_research_command_file("transient-spawn-command")
    assert first.deferred == 1 and first.cancelled == 0 and first.reflected == 0
    # A deferral blocks Finish, so it must never be silent: the reason travels
    # out for the caller to report rather than dying inside the except branch.
    assert [(item.admission_id, item.child_kind, item.reason) for item in first.deferrals] == [
        (
            child_id,
            "work",
            "OSError: canonical state is temporarily unavailable",
        )
    ]
    assert admission is not None and admission.state == "accepted"
    assert command is not None and command.exited_at is None
    assert snapshot is not None and snapshot.content == instruction
    assert store.auto_research_child_work(child_id) is None
    assert store.episode_budget_meter(episode.episode_id) == before
    assert [item.kind for item in store.auto_research_finish_blockers(episode.episode_id)] == [
        "child_admission"
    ]

    second = reconcile_pending_auto_research_child_admissions(
        store,
        background,
        _UnusedExperimentCoordinator(),  # type: ignore[arg-type]
        worker_request_factory=unavailable_then_ready,
        seat_node_type=lambda _project_id, _episode_id, _node_id: "blocker",
    )

    route = store.auto_research_child_work(child_id)
    command = store.agent_command("transient-spawn-command")
    assert second.reflected == 1 and second.deferred == 0 and second.deferrals == ()
    assert received_instructions == [instruction, instruction]
    assert route is not None and route.instruction == instruction
    assert command is not None and command.status == "ok"
    assert store.episode_budget_meter(episode.episode_id).invocations_used == (
        before.invocations_used + 1
    )
    wait_for_task(store, child_id)


def test_restart_recovers_completed_unavailable_spawn_owner_without_rewriting_it(
    tmp_path,
) -> None:
    store, episode, root = _setup(tmp_path)
    key = "completed-unavailable-spawn"
    child_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rcp:auto_research:{episode.episode_id}:spawn:{key}",
        )
    )
    instruction = "Recover the exact accepted Spawn whose first effect was unavailable."
    command_id = "completed-unavailable-spawn-command"
    _admit_command(
        store,
        episode,
        root,
        command_id=command_id,
        key=key,
        child_id=child_id,
        child_kind="work",
        arguments={"seat_node_id": "blk/runtime", "instruction_file": "worker.md"},
        file_kind="instruction",
        filename="worker.md",
        content=instruction,
    )
    store.finish_agent_command(
        command_id,
        status="unavailable",
        payload={
            "result": {"disposition": "unavailable"},
            "diagnostic": "The first dispatch boundary was unavailable.",
        },
        message="The first dispatch boundary was unavailable.",
    )
    before = store.episode_budget_meter(episode.episode_id)

    background = BackgroundAgentTasks(store, _successful_stream)
    result = reconcile_pending_auto_research_child_admissions(
        store,
        background,
        _UnusedExperimentCoordinator(),  # type: ignore[arg-type]
        worker_request_factory=_worker_request,
        seat_node_type=lambda _project_id, _episode_id, _node_id: "blocker",
    )

    command = store.agent_command(command_id)
    admission = store.auto_research_child_admission(child_id)
    assert result.reflected == 1 and result.cancelled == 0
    assert command is not None and command.status == "unavailable"
    assert command.exit_payload["diagnostic"] == "The first dispatch boundary was unavailable."
    assert admission is not None and admission.state == "reflected"
    assert store.episode_budget_meter(episode.episode_id).invocations_used == (
        before.invocations_used + 1
    )
    wait_for_task(store, child_id, expect="succeeded")


class _RecordingExperimentCoordinator:
    def __init__(self, store: AppStore) -> None:
        self.store = store
        self.goal: str | None = None

    def kick_off(self, **kwargs) -> AutoResearchExperimentAction:
        self.goal = kwargs["goal"]
        now = self.store.now()
        child_id = kwargs["child_episode_id"]
        parent_id = kwargs["auto_research_episode_id"]
        route = AutoResearchChildExperimentRecord(
            child_episode_id=child_id,
            auto_research_episode_id=parent_id,
            project_id="project",
            control_node_id=kwargs["node_id"],
            state="pending",
            replaces_episode_id="prior-experiment",
            request={
                "goal": kwargs["goal"],
                "invocation_limit": kwargs["invocation_limit"],
            },
            goal_sha256=kwargs["goal_sha256"],
            parent_operation_id=kwargs["parent_operation_id"],
            created_at=now,
            updated_at=now,
        )
        self.store.reserve_auto_research_experiment_replacement(
            route,
            admission_id=kwargs["admission_id"],
        )
        return AutoResearchExperimentAction(
            disposition="replacement_pending",
            episode_id=child_id,
            status="pending",
            allowance=self.store.auto_research_experiment_allowance(parent_id),
        )


def test_restart_reconciles_experiment_goal_from_durable_snapshot(tmp_path) -> None:
    store, episode, root = _setup(tmp_path)
    key = "experiment-after-runtime-repair"
    child_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rcp:auto_research:{episode.episode_id}:episode:{key}",
        )
    )
    goal = "Compare the repaired runtime against the bounded baseline."
    _admit_command(
        store,
        episode,
        root,
        command_id="episode-command",
        key=key,
        child_id=child_id,
        child_kind="experiment",
        arguments={
            "action": "kick_off_experiment",
            "node_id": "exp/runtime",
            "goal_file": "goal.md",
            "invocation_limit": 4,
        },
        file_kind="goal",
        filename="goal.md",
        content=goal,
    )
    coordinator = _RecordingExperimentCoordinator(store)
    background = BackgroundAgentTasks(store, _successful_stream)

    result = reconcile_pending_auto_research_child_admissions(
        store,
        background,
        coordinator,  # type: ignore[arg-type]
        worker_request_factory=_worker_request,
        seat_node_type=lambda _project_id, _episode_id, _node_id: "experiment",
    )

    admission = store.auto_research_child_admission(child_id)
    command = store.agent_command("episode-command")
    assert result.reflected == 1 and result.cancelled == 0
    assert coordinator.goal == goal
    assert admission is not None and admission.state == "reflected"
    assert command is not None and command.status == "ok"


def test_restart_recovers_completed_unavailable_experiment_owner_from_goal_snapshot(
    tmp_path,
) -> None:
    store, episode, root = _setup(tmp_path)
    key = "completed-unavailable-experiment"
    child_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rcp:auto_research:{episode.episode_id}:episode:{key}",
        )
    )
    goal = "Recover this exact bounded goal after the first unavailable kickoff."
    command_id = "completed-unavailable-experiment-command"
    _admit_command(
        store,
        episode,
        root,
        command_id=command_id,
        key=key,
        child_id=child_id,
        child_kind="experiment",
        arguments={
            "action": "kick_off_experiment",
            "node_id": "exp/runtime",
            "goal_file": "goal.md",
            "invocation_limit": 4,
        },
        file_kind="goal",
        filename="goal.md",
        content=goal,
    )
    store.finish_agent_command(
        command_id,
        status="unavailable",
        payload={
            "result": {"disposition": "unavailable"},
            "diagnostic": "The first Experiment dispatch boundary was unavailable.",
        },
        message="The first Experiment dispatch boundary was unavailable.",
    )
    coordinator = _RecordingExperimentCoordinator(store)

    background = BackgroundAgentTasks(store, _successful_stream)
    result = reconcile_pending_auto_research_child_admissions(
        store,
        background,
        coordinator,  # type: ignore[arg-type]
        worker_request_factory=_worker_request,
        seat_node_type=lambda _project_id, _episode_id, _node_id: "experiment",
    )

    command = store.agent_command(command_id)
    admission = store.auto_research_child_admission(child_id)
    assert result.reflected == 1 and result.cancelled == 0
    assert coordinator.goal == goal
    assert command is not None and command.status == "unavailable"
    assert command.exit_payload["diagnostic"] == (
        "The first Experiment dispatch boundary was unavailable."
    )
    assert admission is not None and admission.state == "reflected"


def test_restart_cancels_malformed_admission_and_finish_no_longer_blocks(tmp_path) -> None:
    store, episode, root = _setup(tmp_path)
    key = "missing-snapshot"
    child_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rcp:auto_research:{episode.episode_id}:spawn:{key}",
        )
    )
    _admit_command(
        store,
        episode,
        root,
        command_id="malformed-spawn",
        key=key,
        child_id=child_id,
        child_kind="work",
        arguments={"seat_node_id": "blk/runtime", "instruction_file": "missing.md"},
        file_kind=None,
        filename=None,
        content=None,
    )
    assert [item.kind for item in store.auto_research_finish_blockers(episode.episode_id)] == [
        "child_admission"
    ]
    background = BackgroundAgentTasks(store, _successful_stream)

    result = reconcile_pending_auto_research_child_admissions(
        store,
        background,
        _UnusedExperimentCoordinator(),  # type: ignore[arg-type]
        worker_request_factory=_worker_request,
        seat_node_type=lambda _project_id, _episode_id, _node_id: "blocker",
    )

    admission = store.auto_research_child_admission(child_id)
    command = store.agent_command("malformed-spawn")
    assert result.cancelled == 1 and result.reflected == 0
    assert admission is not None and admission.state == "cancelled"
    assert command is not None and command.status == "invalid"
    receipt = store.guard_auto_research_finish(
        episode.episode_id,
        effect_id=str(uuid.uuid4()),
        actor_operation_id=root.operation_id,
    )
    blockers = receipt.result.get("blockers", [])
    assert blockers == []


def test_app_startup_reconciles_the_crash_window(manifest, tmp_path) -> None:
    data_dir = tmp_path / "data"
    first_app = create_named_app(str(manifest.path), data_dir=data_dir)
    project_id = first_app.state.default_project_id
    assert project_id is not None
    first_app.state.background_tasks.stream = _successful_stream
    with TestClient(first_app) as client:
        started = client.post(
            f"/api/projects/{project_id}/episodes",
            json={"mode": "auto_research", "invocation_ceiling": 4},
        )
        assert started.status_code == 202
        episode_id = started.json()["episode_id"]
        root_id = started.json()["root_operation_id"]
        store = first_app.state.background_tasks.store
        root = wait_for_task(store, root_id)
        episode = store.episode(episode_id)
        assert episode is not None
        first_app.state.catalog.open(project_id).for_graph_target(
            episode.graph_target
        ).history.append(
            Patch(
                kind="work",
                author="agent",
                summary="Added a restart reconciliation seat.",
                source_operation_id=root_id,
                run_truth_scope=["repo-a"],
                repositories_read=["repo-a"],
                ops=[
                    {
                        "op": "create_nodes",
                        "nodes": [
                            {
                                "id": "blk/restart",
                                "type": "blocker",
                                "title": "Recover child admission",
                                "description": "Re-drive the admitted child after restart.",
                                "status": "open",
                            }
                        ],
                    }
                ],
            )
        )
        key = "startup-crash-window"
        child_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"rcp:auto_research:{episode_id}:spawn:{key}",
            )
        )
        _admit_command(
            store,
            episode,
            root,
            command_id="startup-spawn-command",
            key=key,
            child_id=child_id,
            child_kind="work",
            arguments={
                "seat_node_id": "blk/restart",
                "instruction_file": "startup-worker.md",
            },
            file_kind="instruction",
            filename="startup-worker.md",
            content="Continue the exact child work admitted before restart.",
        )
        assert store.auto_research_child_work(child_id) is None

    restarted = create_named_app(str(manifest.path), data_dir=data_dir)
    restarted.state.background_tasks.stream = _successful_stream
    with TestClient(restarted):
        restarted_store = restarted.state.background_tasks.store
        route = restarted_store.auto_research_child_work(child_id)
        admission = restarted_store.auto_research_child_admission(child_id)
        command = restarted_store.agent_command("startup-spawn-command")
        assert route is not None
        assert admission is not None and admission.state == "reflected"
        assert command is not None and command.status == "ok"
        wait_for_task(restarted_store, child_id)


def test_normal_live_task_settlement_retries_a_deferred_child_admission(
    manifest,
    tmp_path,
    monkeypatch,
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    assert project_id is not None
    background = app.state.background_tasks
    background.stream = _successful_stream
    with TestClient(app) as client:
        started = client.post(
            f"/api/projects/{project_id}/episodes",
            json={"mode": "auto_research", "invocation_ceiling": 4},
        )
        assert started.status_code == 202
        episode_id = started.json()["episode_id"]
        root_id = started.json()["root_operation_id"]
        store = background.store
        root = wait_for_task(store, root_id, expect="succeeded")
        episode = store.episode(episode_id)
        assert episode is not None
        app.state.catalog.open(project_id).for_graph_target(episode.graph_target).history.append(
            Patch(
                kind="work",
                author="agent",
                summary="Added a live child-reconciliation seat.",
                source_operation_id=root_id,
                run_truth_scope=["repo-a"],
                repositories_read=["repo-a"],
                ops=[
                    {
                        "op": "create_nodes",
                        "nodes": [
                            {
                                "id": "blk/live-reconcile",
                                "type": "blocker",
                                "title": "Retry accepted child live",
                                "description": "Retry without restarting the app.",
                                "status": "open",
                            }
                        ],
                    }
                ],
            )
        )
        key = "live-deferred-child"
        child_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"rcp:auto_research:{episode_id}:spawn:{key}",
            )
        )
        _admit_command(
            store,
            episode,
            root,
            command_id="live-deferred-child-command",
            key=key,
            child_id=child_id,
            child_kind="work",
            arguments={
                "seat_node_id": "blk/live-reconcile",
                "instruction_file": "worker.md",
            },
            file_kind="instruction",
            filename="worker.md",
            content="Continue this exact accepted child when live infrastructure recovers.",
        )
        callback = background.on_task_settled
        assert callback is not None
        request = AutoResearchRunRequest.model_validate(root.request)
        execution = AgentTaskExecution(
            operation_id=root.operation_id,
            store=store,
            control=AgentProcessControl(),
        )
        real_start = start_auto_research_child_work

        def unavailable(*_args, **_kwargs):
            raise OSError("canonical state is temporarily unavailable")

        monkeypatch.setattr(
            "rcp.runs.auto_research_child_reconcile.start_auto_research_child_work", unavailable
        )
        callback(project_id, "auto_research", request, execution)
        admission = store.auto_research_child_admission(child_id)
        assert admission is not None and admission.state == "accepted"
        assert store.auto_research_child_work(child_id) is None

        monkeypatch.setattr(
            "rcp.runs.auto_research_child_reconcile.start_auto_research_child_work", real_start
        )
        callback(project_id, "auto_research", request, execution)

        admission = store.auto_research_child_admission(child_id)
        assert admission is not None and admission.state == "reflected"
        assert store.auto_research_child_work(child_id) is not None
        wait_for_task(store, child_id, expect="succeeded")
