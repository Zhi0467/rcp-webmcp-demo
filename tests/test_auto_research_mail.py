from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import rcp.api.app as api_app_module
import rcp.runs.tasks.auto_research_child_work as child_work_module
from rcp.agents import AgentEvent, AgentProcessControl
from rcp.background import AgentTaskExecution, BackgroundAgentTasks
from rcp.core.authority import AgentDispatchAuthority, AgentDispatchScope
from rcp.core.models import AuthorizedHuman
from rcp.core.transition_models import GraphHeadRef, GraphTargetRef
from rcp.runs.auto_research import AutoResearchRunRequest, AutoResearchStartRequest
from rcp.runs.auto_research_admission import (
    resume_auto_research_child_work,
    start_auto_research,
    start_auto_research_child_work,
)
from rcp.runs.auto_research_delivery import (
    deliver_pending_auto_research_mail,
    record_auto_research_message,
)
from rcp.runs.auto_research_mail import (
    AUTO_RESEARCH_MAIL_MAX_BYTES,
    AutoResearchMailDelivery,
    auto_research_mail_claim_prefix,
    auto_research_mail_delivery,
    parse_auto_research_mail_delivery,
    stage_auto_research_mail_delivery,
)
from rcp.runs.experiment_loop import experiment_watcher_output_name
from rcp.runs.tasks.auto_research_child_work import (
    _stage_auto_research_child_work_mail,
    stream_auto_research_child_work_run,
)
from rcp.runs.tasks.work import stream_work_run
from rcp.service import RunRequest
from rcp.storage import (
    AgentTaskRecord,
    AppStore,
    AutoResearchChildWorkRecord,
    AutoResearchMessageRecord,
    AutoResearchStateRecord,
    EpisodeRecord,
    ProjectRecord,
    WatcherContinuation,
    WatcherRecord,
)
from rcp.transport.workspace_mailbox import RunStageMailbox, clear_turn_handoff_files

from .helpers import (
    append_fixture_patch,
    create_named_app,
    fabricated_authorizer,
    seed_patch,
    wait_for_task,
)
from .test_api import ScriptedLauncher, _experiment_fixture_patch

_RUN_TRUTH_SCOPE = ["repo-a"]


@pytest.mark.asyncio
async def test_child_prelaunch_failure_closes_validator_lifecycle(
    tmp_path,
    monkeypatch,
) -> None:
    closed_with: list[BaseException | None] = []

    class Lifecycle:
        async def close(self, *, primary_error=None):
            closed_with.append(primary_error)

    turn = SimpleNamespace(validator_lifecycle=Lifecycle())

    async def stage(*_args, **_kwargs):
        return turn, object(), None

    def fail_prompt(*_args, **_kwargs):
        raise ValueError("child prompt failed")

    monkeypatch.setattr(child_work_module, "_resolve_work_execution", lambda *_args: object())
    monkeypatch.setattr(child_work_module, "_stage_auto_research_child_work_turn", stage)
    monkeypatch.setattr(child_work_module, "_compose_child_prompt", fail_prompt)

    frames = [
        frame
        async for frame in child_work_module.stream_auto_research_child_work_run(
            object(),
            object(),
            object(),
            tmp_path,
            SimpleNamespace(continuation="fresh"),
            route=object(),
        )
    ]

    assert "child prompt failed" in "".join(frames)
    assert len(closed_with) == 1
    assert isinstance(closed_with[0], ValueError)


def test_api_dispatches_routed_child_work_without_falling_back_to_ordinary_work(
    manifest,
    tmp_path,
    monkeypatch,
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    project_id = app.state.default_project_id
    store = app.state.background_tasks.store
    owner = store.local_owner
    assert owner is not None
    authorizer = AuthorizedHuman(
        space_id=store.space_id,
        user_id=owner.user_id,
        display_name=owner.display_name,
    )
    child_calls: list[str] = []
    ordinary_calls: list[str] = []

    async def fake_auto_research(*_args, **_kwargs):
        yield 'data: {"event":"session","session_id":"root-session"}\n\n'
        yield 'data: {"event":"done"}\n\n'

    async def child_entry(_service, _launcher, _request, _data_dir, execution, *, route):
        child_calls.append(route.worker_id)
        assert execution.operation_id == route.current_operation_id
        if False:
            yield ""
        raise RuntimeError("child entry failed")

    async def ordinary_entry(_service, _launcher, _request, _data_dir, execution):
        ordinary_calls.append(execution.operation_id)
        if False:
            yield ""
        raise AssertionError("routed child fell back to ordinary Work")

    monkeypatch.setattr(api_app_module, "stream_auto_research_orchestrator_run", fake_auto_research)
    monkeypatch.setattr(api_app_module, "stream_auto_research_child_work_run", child_entry)
    monkeypatch.setattr(api_app_module, "stream_work_run", ordinary_entry)

    episode, root = start_auto_research(
        app.state.background_tasks,
        project_id,
        AutoResearchStartRequest(
            invocation_ceiling=5,
            provider="codex",
            run_on="laptop",
            run_truth_scope=_RUN_TRUTH_SCOPE,
        ),
        authorized_by=authorizer,
        graph_base_head=service.history.head_ref(),
        ensure_graph_target=lambda episode: api_app_module._ensure_auto_research_graph_target(
            episode,
            catalog=app.state.catalog,
        ),
        episode_id="00000000-0000-4000-8000-0000000004a0",
        operation_id="00000000-0000-4000-8000-0000000004a1",
    )
    root = wait_for_task(store, root.operation_id, expect="succeeded")
    worker_id = "00000000-0000-4000-8000-0000000004a2"
    instruction = "Inspect the focused hypothesis and report the bounded evidence."
    child = start_auto_research_child_work(
        app.state.background_tasks,
        episode.episode_id,
        RunRequest(
            provider="codex",
            run_on="laptop",
            run_truth_scope=_RUN_TRUTH_SCOPE,
            chat_scope="node",
            node_id="hyp/replanning-restores-plasticity",
            chat_id=worker_id,
            message=instruction,
            mode="work",
            trigger="orchestrator",
            patch_kind="work",
        ),
        admitted_by_operation_id=root.operation_id,
        worker_id=worker_id,
        instruction=instruction,
        instruction_sha256=hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
    )
    child = wait_for_task(store, child.operation_id, expect="failed")

    assert child_calls == [worker_id]
    assert ordinary_calls == []
    assert child.error is not None and "child entry failed" in child.error


class _SessionSequenceLauncher(ScriptedLauncher):
    def __init__(
        self,
        turns: list[dict[str, str]],
        *,
        sessions: list[str],
        fail_on: set[int] | None = None,
        message: str = "",
    ) -> None:
        super().__init__(turns, message=message)
        self.sessions = sessions
        self.fail_on = fail_on or set()
        self.native_session_id = sessions[0]

    async def stream(self, provider, prompt, **kwargs):
        async for event in super().stream(provider, prompt, **kwargs):
            call_index = self.calls - 1
            if event.event == "session":
                event = event.model_copy(
                    update={"session_id": self.sessions[min(call_index, len(self.sessions) - 1)]}
                )
            if event.event == "done" and call_index in self.fail_on:
                yield AgentEvent(event="error", text="network failed")
                return
            yield event


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


def _claimed_delivery(tmp_path):
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
    auto_research, root = store.create_auto_research_episode_with_root_task(
        EpisodeRecord(
            episode_id="auto_research",
            project_id="project",
            mode="auto_research",
            graph_target=graph_target,
            graph_base_head=GraphHeadRef(revision=0),
            status="queued",
            invocation_ceiling=5,
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
            request=AutoResearchRunRequest(
                episode_id="auto_research",
                role="orchestrator",
                actor_operation_id="root",
                run_truth_scope=_RUN_TRUTH_SCOPE,
            ).model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="done",
            authorized_by=authorizer,
            dispatch_authority=_auto_research_authority("auto_research", "orchestrator"),
        ),
    )
    store.complete_agent_task(root.operation_id, applied_revision=None, result={})
    root = store.agent_task(root.operation_id)
    assert root is not None
    assert auto_research.graph_target == root.graph_target == graph_target
    assert auto_research.graph_base_head == GraphHeadRef(revision=0)
    worker = store.create_auto_research_agent_task(
        AgentTaskRecord(
            operation_id="worker",
            project_id="project",
            episode_id="auto_research",
            graph_target=graph_target,
            kind="auto_research",
            status="queued",
            request=AutoResearchRunRequest(
                episode_id="auto_research",
                role="worker",
                actor_operation_id="worker",
                run_truth_scope=_RUN_TRUTH_SCOPE,
                control_node_id="exp/check",
            ).model_dump(mode="json"),
            created_at=store.now(),
            updated_at=store.now(),
            status_message="done",
            parent_operation_id=root.operation_id,
            authorized_by=authorizer,
            dispatch_authority=_auto_research_authority("auto_research", "worker"),
        ),
        role="worker",
    )
    assert worker.graph_target == graph_target
    store.complete_agent_task(worker.operation_id, applied_revision=None, result={})
    messages = [
        AutoResearchMessageRecord(
            message_id="message-one",
            episode_id=auto_research.episode_id,
            sender_role="human",
            authorized_by=authorizer,
            recipient_task_id=root.operation_id,
            body="First result",
            created_at=store.now(),
        ),
        AutoResearchMessageRecord(
            message_id="message-two",
            episode_id=auto_research.episode_id,
            sender_role="worker",
            sender_task_id=worker.operation_id,
            recipient_task_id=root.operation_id,
            control_node_id="exp/check",
            body="Second result",
            created_at=store.now(),
        ),
    ]
    for message in messages:
        store.record_auto_research_message(message)
    store.mark_auto_research_messages_delivered(
        ["message-one", "message-two"],
        operation_id="mail-wake",
    )
    claimed = store.auto_research_messages(auto_research.episode_id)
    return auto_research_mail_delivery(
        episode_id=auto_research.episode_id,
        recipient_task_id=root.operation_id,
        delivery_operation_id="mail-wake",
        messages=claimed,
    )


def test_claimed_inbound_mail_is_staged_as_one_exact_hearsay_only_batch(tmp_path) -> None:
    delivery = _claimed_delivery(tmp_path)
    workspace = tmp_path / "stage"
    workspace.mkdir()
    for name in ("patch.json", "watch.json", "messages.json"):
        (workspace / name).write_text("stale", encoding="utf-8")
    mailbox = RunStageMailbox.for_stage(local_stage=workspace, remote_stage=None)

    clear_turn_handoff_files(mailbox)
    stage_auto_research_mail_delivery(mailbox, delivery)
    parsed = parse_auto_research_mail_delivery(
        (workspace / "messages.json").read_text(encoding="utf-8")
    )

    assert parsed == delivery
    assert parsed.graph_authority == "none"
    assert parsed.epistemic_status == "hearsay"
    assert parsed.delivery_operation_id == "mail-wake"
    assert parsed.message_ids == ["message-one", "message-two"]
    assert [message.body for message in parsed.messages] == ["First result", "Second result"]
    assert parsed.messages[0].sender_role == "human"
    assert parsed.messages[0].authorized_by is not None
    assert parsed.messages[0].authorized_by.display_name == "AutoResearch owner"
    assert parsed.messages[1].sender_role == "worker"
    assert parsed.messages[1].authorized_by is None
    assert not (workspace / "patch.json").exists()
    assert not (workspace / "watch.json").exists()


def test_ordinary_child_work_stages_and_reuses_only_its_exact_claimed_mail(tmp_path) -> None:
    workspace = tmp_path / "ordinary-child-stage"
    workspace.mkdir()
    message = AutoResearchMessageRecord(
        message_id="ordinary-child-message",
        episode_id="auto_research",
        sender_role="orchestrator",
        sender_task_id="root",
        recipient_task_id="worker",
        control_node_id="exp/check",
        body="Recheck the bounded observation.",
        created_at="2026-08-16T00:00:00+00:00",
        delivered_at="2026-08-16T00:01:00+00:00",
        delivery_operation_id="child-mail-wake",
    )
    route = AutoResearchChildWorkRecord(
        worker_id="worker",
        episode_id="auto_research",
        project_id="project",
        control_node_id="exp/check",
        root_operation_id="worker",
        current_operation_id="child-mail-wake",
        admitted_by_operation_id="root",
        instruction="Check the result.",
        instruction_sha256=hashlib.sha256(b"Check the result.").hexdigest(),
        created_at="2026-08-16T00:00:00+00:00",
        updated_at="2026-08-16T00:01:00+00:00",
    )

    class _Store:
        @staticmethod
        def auto_research_messages(_episode_id):
            return [message]

        @staticmethod
        def agent_task(operation_id):
            tasks = {
                "child-mail-resume": SimpleNamespace(
                    operation_id="child-mail-resume",
                    parent_operation_id="child-mail-wake",
                ),
                "child-mail-wake": SimpleNamespace(
                    operation_id="child-mail-wake",
                    parent_operation_id="child-initial",
                ),
            }
            return tasks.get(operation_id)

        @staticmethod
        def agent_task_continuation_cause(operation_id):
            return {
                "child-mail-resume": "resume",
                "child-mail-wake": "message_wake",
            }.get(operation_id)

        @staticmethod
        def auto_research_child_work_for_operation(_operation_id):
            return route

    execution = AgentTaskExecution(
        operation_id="child-mail-wake",
        store=_Store(),  # type: ignore[arg-type]
        control=AgentProcessControl(),
        continuation="message_wake",
    )

    assert (
        _stage_auto_research_child_work_mail(
            AgentTaskExecution(
                operation_id="child-initial",
                store=_Store(),  # type: ignore[arg-type]
                control=AgentProcessControl(),
                continuation="fresh",
            ),
            route.model_copy(update={"current_operation_id": "child-initial"}),
            local_stage=workspace,
            remote_stage=None,
            continuation="fresh",
        )
        is None
    )
    assert not (workspace / "messages.json").exists()
    first = _stage_auto_research_child_work_mail(
        execution,
        route,
        local_stage=workspace,
        remote_stage=None,
        continuation="message_wake",
    )
    second = _stage_auto_research_child_work_mail(
        execution,
        route,
        local_stage=workspace,
        remote_stage=None,
        continuation="message_wake",
    )
    resumed = _stage_auto_research_child_work_mail(
        AgentTaskExecution(
            operation_id="child-mail-resume",
            store=_Store(),  # type: ignore[arg-type]
            control=AgentProcessControl(),
            continuation="resume",
        ),
        route.model_copy(update={"current_operation_id": "child-mail-resume"}),
        local_stage=workspace,
        remote_stage=None,
        continuation="resume",
    )

    assert first == second == resumed == str(workspace / "messages.json")
    retained = parse_auto_research_mail_delivery(
        (workspace / "messages.json").read_text(encoding="utf-8")
    )
    assert retained.recipient_task_id == route.worker_id
    assert retained.delivery_operation_id == execution.operation_id
    assert retained.message_ids == [message.message_id]
    assert retained.epistemic_status == "hearsay"


def test_legacy_v1_human_mail_without_sender_snapshot_still_parses(tmp_path) -> None:
    delivery = _claimed_delivery(tmp_path)
    payload = delivery.model_dump(mode="json")
    human = next(message for message in payload["messages"] if message["sender_role"] == "human")
    human.pop("authorized_by")

    parsed = parse_auto_research_mail_delivery(json.dumps(payload))

    parsed_human = next(message for message in parsed.messages if message.sender_role == "human")
    assert parsed_human.authorized_by is None


def test_auto_research_mail_batch_validation_is_all_or_none(tmp_path) -> None:
    delivery = _claimed_delivery(tmp_path)
    payload = delivery.model_dump(mode="json")
    payload["messages"][1]["delivery_operation_id"] = "another-wake"
    with pytest.raises(ValidationError, match="crosses claimed wake operations"):
        parse_auto_research_mail_delivery(json.dumps(payload))

    payload = delivery.model_dump(mode="json")
    payload["messages"].append(payload["messages"][0])
    with pytest.raises(ValidationError, match="duplicate message"):
        AutoResearchMailDelivery.model_validate(payload)

    unclaimed = delivery.messages[0].model_dump(mode="json")
    unclaimed["delivered_at"] = None
    with pytest.raises(ValidationError):
        AutoResearchMailDelivery.model_validate(
            {
                **delivery.model_dump(mode="json"),
                "messages": [unclaimed],
            }
        )


def test_claim_prefix_stops_at_the_shared_count_and_exact_wire_boundary(
    monkeypatch,
) -> None:
    messages = [
        AutoResearchMessageRecord(
            message_id=f"message-{index}",
            episode_id="auto_research",
            sender_role="human",
            recipient_task_id="root",
            body=f"Result {index}",
            created_at=f"2026-08-12T00:00:0{index}+00:00",
        )
        for index in range(3)
    ]
    claimed = [
        message.model_copy(
            update={
                "delivered_at": "2026-08-12T00:01:00+00:00",
                "delivery_operation_id": "mail-wake",
            }
        )
        for message in messages[:2]
    ]
    two_message_delivery = auto_research_mail_delivery(
        episode_id="auto_research",
        recipient_task_id="root",
        delivery_operation_id="mail-wake",
        messages=claimed,
    )
    exact_boundary = len((two_message_delivery.model_dump_json() + "\n").encode("utf-8"))
    assert exact_boundary < AUTO_RESEARCH_MAIL_MAX_BYTES
    monkeypatch.setattr("rcp.runs.auto_research_mail.AUTO_RESEARCH_MAIL_MAX_MESSAGES", 2)
    monkeypatch.setattr(
        "rcp.runs.auto_research_mail.AUTO_RESEARCH_MAIL_MAX_BYTES",
        exact_boundary,
    )

    selected = auto_research_mail_claim_prefix(
        episode_id="auto_research",
        recipient_task_id="root",
        delivery_operation_id="mail-wake",
        delivered_at="2026-08-12T00:01:00+00:00",
        messages=messages,
    )

    assert [message.message_id for message in selected] == ["message-0", "message-1"]
    rendered = two_message_delivery.model_dump_json() + "\n"
    assert parse_auto_research_mail_delivery(rendered) == two_message_delivery
    with pytest.raises(ValueError, match="exceeds .* bytes"):
        parse_auto_research_mail_delivery(rendered + " ")
    with pytest.raises(ValidationError, match="exceeds 2 messages"):
        auto_research_mail_delivery(
            episode_id="auto_research",
            recipient_task_id="root",
            delivery_operation_id="mail-wake",
            messages=[
                message.model_copy(
                    update={
                        "delivered_at": "2026-08-12T00:01:00+00:00",
                        "delivery_operation_id": "mail-wake",
                    }
                )
                for message in messages
            ],
        )

    monkeypatch.setattr(
        "rcp.runs.auto_research_mail.AUTO_RESEARCH_MAIL_MAX_BYTES",
        exact_boundary - 1,
    )
    smaller = auto_research_mail_claim_prefix(
        episode_id="auto_research",
        recipient_task_id="root",
        delivery_operation_id="mail-wake",
        delivered_at="2026-08-12T00:01:00+00:00",
        messages=messages,
    )
    assert [message.message_id for message in smaller] == ["message-0"]


@pytest.mark.asyncio
async def test_ordinary_child_work_prompt_and_mail_continuation_keep_narrow_authority(
    manifest,
    tmp_path,
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    store = app.state.background_tasks.store
    launcher = _SessionSequenceLauncher(
        [
            {"watch.json": "child watcher output must not be parsed"},
            {},
            {},
            {},
        ],
        sessions=["child-session", "forked-session", "resume-session", "resume-forked"],
        fail_on={2},
        message="The delegated check completed.",
    )

    async def stream(project_id, kind, request, execution):
        if kind == "auto_research":
            yield 'data: {"event":"session","session_id":"root-session"}\n\n'
            yield 'data: {"event":"done"}\n\n'
            return
        route = store.auto_research_child_work_for_operation(execution.operation_id)
        run = stream_auto_research_child_work_run if route is not None else stream_work_run
        kwargs = {"route": route} if route is not None else {}
        async for frame in run(
            service,
            launcher,
            request,
            tmp_path / "data",
            execution=execution,
            **kwargs,
        ):
            yield frame

    background = BackgroundAgentTasks(store, stream)
    episode, root = start_auto_research(
        background,
        app.state.default_project_id,
        AutoResearchStartRequest(
            invocation_ceiling=5,
            provider="codex",
            run_on="laptop",
            run_truth_scope=["repo-a"],
        ),
        authorized_by=fabricated_authorizer(),
        graph_base_head=GraphHeadRef(revision=0),
        ensure_graph_target=lambda _episode: None,
        episode_id="ordinary-child-prompt-episode",
        operation_id="ordinary-child-prompt-root",
    )
    root = wait_for_task(store, root.operation_id, expect="succeeded")
    worker_id = "00000000-0000-4000-8000-000000000451"
    instruction = "Check the focused hypothesis and report the bounded evidence."
    child = start_auto_research_child_work(
        background,
        episode.episode_id,
        RunRequest(
            provider="codex",
            run_on="laptop",
            run_truth_scope=["repo-a"],
            chat_scope="node",
            node_id="hyp/replanning-restores-plasticity",
            chat_id=worker_id,
            message=instruction,
            mode="work",
            trigger="orchestrator",
            patch_kind="work",
        ),
        admitted_by_operation_id=root.operation_id,
        worker_id=worker_id,
        instruction=instruction,
        instruction_sha256=hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
    )
    child = wait_for_task(store, child.operation_id, expect="succeeded")
    assert child.graph_target == episode.graph_target == root.graph_target

    initial_master_path = Path(launcher.prompts[0].splitlines()[1])
    initial_master = initial_master_path.read_text(encoding="utf-8")
    assert "## Auto-research child Work boundary" in initial_master
    assert "only staged command capabilities" in initial_master
    assert "Do not invoke `apply`, `status`, `spawn`" in initial_master
    assert "Do not write `watch.json`" in initial_master
    assert launcher.launch_kwargs[0]["invocation_gate"] is not None
    assert not (launcher.workspaces[0] / "watch.json").exists()
    assert any(
        receipt.category == "auto_research_child_watcher_output_discarded"
        for receipt in store.agent_task_receipts(child.operation_id)
    )

    message = record_auto_research_message(
        store,
        episode_id=episode.episode_id,
        sender_role="orchestrator",
        sender_task_id=root.operation_id,
        authorized_by=None,
        recipient_task_id=worker_id,
        control_node_id="hyp/replanning-restores-plasticity",
        body="Recheck after the graph update.",
    )
    wake_id = deliver_pending_auto_research_mail(
        background,
        episode_id=episode.episode_id,
        recipient_task_id=worker_id,
    )
    assert wake_id is not None
    wake = wait_for_task(store, wake_id, expect="failed")

    continuation_contract_path = Path(launcher.prompts[1].splitlines()[1])
    continuation_contract = continuation_contract_path.read_text(encoding="utf-8")
    assert "same native provider session" in continuation_contract
    assert "newly claimed agent mail is staged separately" in continuation_contract
    assert "messages.json" in continuation_contract
    assert "Do not invoke `apply`, `status`, `spawn`" in continuation_contract
    assert launcher.resumed_sessions == [None, launcher.native_session_id]
    assert launcher.launch_kwargs[1]["invocation_gate"] is not None
    staged_mail = parse_auto_research_mail_delivery(
        (launcher.workspaces[1] / "messages.json").read_text(encoding="utf-8")
    )
    assert staged_mail.delivery_operation_id == wake.operation_id
    assert staged_mail.message_ids == [message.message_id]
    assert not (launcher.workspaces[1] / "watch.json").exists()
    assert store.agent_task(wake.operation_id).native_session_id == "child-session"  # type: ignore[union-attr]
    assert any(
        receipt.category == "continuation_context_unavailable"
        and receipt.payload.get("reason") == "native_session_mismatch"
        for receipt in store.agent_task_receipts(wake.operation_id)
    )

    resume_worker_id = "00000000-0000-4000-8000-000000000452"
    resume_instruction = "Check a second focused hypothesis."
    interrupted = start_auto_research_child_work(
        background,
        episode.episode_id,
        RunRequest(
            provider="codex",
            run_on="laptop",
            run_truth_scope=["repo-a"],
            chat_scope="node",
            node_id="hyp/replanning-restores-plasticity",
            chat_id=resume_worker_id,
            message=resume_instruction,
            mode="work",
            trigger="orchestrator",
            patch_kind="work",
        ),
        admitted_by_operation_id=root.operation_id,
        worker_id=resume_worker_id,
        instruction=resume_instruction,
        instruction_sha256=hashlib.sha256(resume_instruction.encode("utf-8")).hexdigest(),
    )
    interrupted = wait_for_task(store, interrupted.operation_id, expect="failed")
    assert interrupted.native_session_id == "resume-session"

    resumed = resume_auto_research_child_work(
        background,
        episode.episode_id,
        resume_worker_id,
    )
    assert resumed.disposition == "resumed"
    assert resumed.task is not None
    resumed_task = wait_for_task(store, resumed.task.operation_id, expect="failed")
    assert resumed_task.native_session_id == "resume-session"
    assert any(
        receipt.category == "continuation_context_unavailable"
        and receipt.payload.get("reason") == "native_session_mismatch"
        for receipt in store.agent_task_receipts(resumed_task.operation_id)
    )


@pytest.mark.asyncio
async def test_ordinary_child_work_cannot_see_or_maintain_active_experiment_watchers(
    manifest,
    tmp_path,
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    append_fixture_patch(service, _experiment_fixture_patch())
    project_id = app.state.default_project_id
    store = app.state.background_tasks.store
    experiment_id = "exp/bounded-loop"
    loop_episode_id = "00000000-0000-4000-8000-000000000461"
    loop_operation_id = "00000000-0000-4000-8000-000000000462"
    loop_chat_id = "00000000-0000-4000-8000-000000000463"
    watcher_id = "active-experiment-watcher"
    graph_revision = service.history.state().revision
    loop_request = RunRequest(
        provider="codex",
        run_on="laptop",
        run_truth_scope=["repo-a"],
        chat_scope="node",
        node_id=experiment_id,
        chat_id=loop_chat_id,
        message="Begin the bounded Experiment loop.",
        mode="work",
        trigger="experiment_run",
        patch_kind="experiment_loop",
        control_node_id=experiment_id,
        control_revision=graph_revision,
        control_episode_id=loop_episode_id,
        control_invocation=1,
        control_invocation_ceiling=2,
    )
    now = store.now()
    store.create_experiment_episode_with_invocation(
        AgentTaskRecord(
            operation_id=loop_operation_id,
            project_id=project_id,
            episode_id=loop_episode_id,
            kind="node_chat",
            status="queued",
            request=loop_request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="queued",
            authorized_by=fabricated_authorizer(),
        )
    )
    store.complete_agent_task(loop_operation_id, applied_revision=None, result={})
    continuation = WatcherContinuation(
        provider="codex",
        run_on="laptop",
        run_truth_scope=["repo-a"],
        patch_kind="experiment_loop",
        control_node_id=experiment_id,
        control_revision=graph_revision,
        control_episode_id=loop_episode_id,
        control_invocation=1,
        control_invocation_ceiling=2,
    )
    store.create_watchers(
        [
            WatcherRecord(
                watcher_id=watcher_id,
                project_id=project_id,
                origin_operation_id=loop_operation_id,
                origin_task_kind="node_chat",
                chat_id=loop_chat_id,
                node_id=experiment_id,
                episode_id=loop_episode_id,
                continuation=continuation,
                check_command="false",
                log_path="/tmp/active-experiment-watcher.log",
                cwd="/tmp",
                created_at=store.now(),
            )
        ]
    )
    loop_stage = tmp_path / "loop-stage"
    loop_stage.mkdir()
    store.commit_experiment_episode_turn(
        episode_id=loop_episode_id,
        project_id=project_id,
        control_node_id=experiment_id,
        provider="codex",
        execution_machine="laptop",
        execution_host="",
        native_session_id="loop-session",
        stage_host=None,
        stage_root=str(loop_stage),
        chat_id=loop_chat_id,
        operation_id=loop_operation_id,
        invocation=1,
        graph_result="no graph change",
        watcher_ids=[watcher_id],
        context_baseline={},
    )
    assert [item.control_node_id for item in store.experiment_watcher_resources(project_id)] == [
        experiment_id
    ]

    output_name = experiment_watcher_output_name(experiment_id)
    launcher = ScriptedLauncher(
        [
            {
                output_name: json.dumps(
                    {
                        "external": [
                            {
                                "stop_watcher_id": watcher_id,
                                "reason": "Child Work must not have this authority.",
                            }
                        ],
                        "graph": [],
                    }
                )
            }
        ],
        message="The delegated Experiment check completed.",
    )

    async def stream(project_id, kind, request, execution):
        if kind == "auto_research":
            yield 'data: {"event":"session","session_id":"root-session"}\n\n'
            yield 'data: {"event":"done"}\n\n'
            return
        route = store.auto_research_child_work_for_operation(execution.operation_id)
        run = stream_auto_research_child_work_run if route is not None else stream_work_run
        kwargs = {"route": route} if route is not None else {}
        async for frame in run(
            service,
            launcher,
            request,
            tmp_path / "data",
            execution=execution,
            **kwargs,
        ):
            yield frame

    background = BackgroundAgentTasks(store, stream)
    episode, root = start_auto_research(
        background,
        project_id,
        AutoResearchStartRequest(
            invocation_ceiling=5,
            provider="codex",
            run_on="laptop",
            run_truth_scope=["repo-a"],
        ),
        authorized_by=fabricated_authorizer(),
        graph_base_head=GraphHeadRef(revision=0),
        ensure_graph_target=lambda _episode: None,
        episode_id="ordinary-child-experiment-authority-episode",
        operation_id="ordinary-child-experiment-authority-root",
    )
    root = wait_for_task(store, root.operation_id, expect="succeeded")
    worker_id = "00000000-0000-4000-8000-000000000464"
    instruction = "Inspect the focused Experiment without changing its watcher lifecycle."
    child = start_auto_research_child_work(
        background,
        episode.episode_id,
        RunRequest(
            provider="codex",
            run_on="laptop",
            run_truth_scope=["repo-a"],
            chat_scope="node",
            node_id=experiment_id,
            chat_id=worker_id,
            message=instruction,
            mode="work",
            trigger="orchestrator",
            patch_kind="work",
        ),
        admitted_by_operation_id=root.operation_id,
        worker_id=worker_id,
        instruction=instruction,
        instruction_sha256=hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
    )
    child = wait_for_task(store, child.operation_id, expect="succeeded")
    assert child.graph_target == episode.graph_target == root.graph_target

    master_path = Path(launcher.prompts[0].splitlines()[1])
    master = master_path.read_text(encoding="utf-8")
    assert output_name not in master
    assert loop_episode_id not in master
    assert not [name for name in launcher.input_snapshots[0] if "experiment-watchers" in name]
    watcher = store.watcher(watcher_id)
    assert watcher is not None and watcher.status == "active"
    receipts = store.agent_task_receipts(child.operation_id)
    rejection = next(
        item for item in receipts if item.category == "experiment_watcher_maintenance_rejected"
    )
    assert "not staged" in str(rejection.payload["problem"])
    assert not [item for item in receipts if item.category == "experiment_watchers_maintained"]
    assert child.result is not None
    assert child.result["messages"][-1] == "The delegated Experiment check completed."
    assert service.history.state().revision == graph_revision
