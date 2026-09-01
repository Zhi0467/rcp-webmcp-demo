from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shlex
import shutil
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from rcp.agents import AgentEvent, AgentProcessControl
from rcp.agents.auto_research_prompt import (
    auto_research_orchestrator_continuation_contract,
    auto_research_orchestrator_task_contract,
    auto_research_worker_continuation_contract,
    auto_research_worker_task_contract,
)
from rcp.agents.command_mailbox import StagedCommandMailbox
from rcp.agents.command_mailbox import stage_command_mailbox as _stage_command_mailbox
from rcp.agents.command_protocol import MessageCommandRequest
from rcp.agents.invocation_broker import ProviderInvocationGate
from rcp.background import AgentTaskExecution, BackgroundAgentTasks
from rcp.config import load_manifest
from rcp.core.authority import AgentDispatchAuthority, AgentDispatchScope
from rcp.core.models import GraphBranchMetadata, Patch
from rcp.core.transition_models import GraphHeadRef, GraphTargetRef
from rcp.history import HistoryManager
from rcp.limits import AGENT_TASK_RECEIPT_RETENTION_COUNTS
from rcp.paper import PaperService
from rcp.runs import auto_research_mail as auto_research_mail_module
from rcp.runs.auto_research import (
    AutoResearchCommandDispatcher,
    AutoResearchCommandEffectResult,
    AutoResearchCommandEffects,
    AutoResearchRunRequest,
    request_auto_research_stop,
)
from rcp.runs.auto_research_delivery import record_auto_research_message
from rcp.runs.auto_research_effects import auto_research_command_effects
from rcp.runs.auto_research_lifecycle import parse_auto_research_lifecycle_delivery
from rcp.runs.auto_research_mail import (
    auto_research_mail_delivery,
    parse_auto_research_mail_delivery,
)
from rcp.runs.tasks import auto_research_stream as auto_research_stream_module
from rcp.runs.tasks.auto_research_stream import (
    _HANDOFFS_CLEARED_RECEIPT,
    _close_worker_mailbox,
    _orchestrator_final_source_effect_id,
    _orchestrator_stage_name,
    _ValidateOnlyAutoResearchCommandDispatcher,
    _worker_stage_name,
    stream_auto_research_orchestrator_run,
    stream_auto_research_worker_run,
)
from rcp.runs.tasks.work import _apply_work_patch, _validate_work_patch_live
from rcp.service import GraphUpdateResult, ProjectService
from rcp.storage import (
    AgentTaskRecord,
    AppStore,
    AutoResearchApplyResultRecord,
    AutoResearchLifecycleNoticeRecord,
    AutoResearchStateRecord,
    EpisodeRecord,
    ProjectRecord,
)
from rcp.transport.workspace_mailbox import RunStageMailbox

from .helpers import (
    agent_patch_json,
    fabricated_authorizer,
    seated_on_every_project,
    wait_for_task,
)


def test_orchestrator_contract_assigns_clear_work_and_requests_prose_difficulty() -> None:
    contract = auto_research_orchestrator_task_contract(
        project_name="project",
        graph_path="/stage/graph.json",
        research_path="/stage/research.md",
        repositories=[],
        patch_path="/stage/patch.json",
        output_schema_path="/stage/schema.json",
        validator_command="/stage/rcp-agent validate",
        command_client="/stage/rcp-agent",
    )

    assert "Give every worker a clear, executable assignment" in contract
    assert "report in prose" in contract
    assert "without changing an existing ResearchQuestion or Hypothesis" in contract
    assert "treating a Proposal as completed work" in contract
    # The orchestrator's only way to wait is a graph condition, so the contract has to
    # say what one looks like rather than just naming the verb.
    assert "`watch-graph --key <key> --condition-json <json> --reason <text>`" in contract
    assert '`{"node_id": "<id>", "status_in": ["<status>", ...]}`' in contract
    assert '`{"node_id": "<id>", "proposal_resolved": true}`' in contract
    assert "their order does not matter" in contract
    assert "A wake spends one invocation" in contract
    assert "finish --key <key>" in contract
    assert "Sleeping on a watcher or mail is not completion" in contract
    assert "structured `result` as the authoritative\n  disposition" in contract
    assert "Exit 0 / `ok` means RCP recorded a durable disposition" in contract
    assert "A completed `unavailable` attempt is not an effect verdict" in contract
    assert "it acknowledges nothing" in contract
    assert "including\n  effects that return `unavailable`" in contract
    assert "the refused key replays its exact snapshot" in contract
    # Every node type it may touch is defined, not just fenced.
    for node_type in (
        "ResearchQuestion",
        "Hypothesis",
        "Experiment",
        "Evidence",
        "Decision",
        "Blocker",
    ):
        assert f"- {node_type} \u2014" in contract


def test_refreshed_orchestrator_paths_return_the_staged_context_revision(monkeypatch) -> None:
    staged_context = SimpleNamespace(
        graph_revision=17,
        graph_path="/stage/graph-r17.json",
        research_md_path="/stage/research-r17.md",
    )
    monkeypatch.setattr(
        auto_research_stream_module,
        "_auto_research_context",
        lambda *_args: staged_context,
    )
    service = SimpleNamespace(
        history=SimpleNamespace(
            state=lambda: (_ for _ in ()).throw(
                AssertionError("the live revision must not be reread after staging")
            )
        )
    )

    refreshed = auto_research_stream_module._refreshed_orchestrator_state_paths(
        service,
        SimpleNamespace(),
        SimpleNamespace(),
    )

    assert refreshed == (17, "/stage/graph-r17.json", "/stage/research-r17.md")


def test_root_prompts_expose_the_same_exact_callable_surface_and_lifecycle_boundary() -> None:
    common = dict(
        graph_path="/stage/graph.json",
        research_path="/stage/research.md",
        repositories=[],
        patch_path="/stage/patch.json",
        output_schema_path="/stage/schema.json",
        validator_command="/stage/rcp-agent validate",
        command_client="/stage/rcp-agent",
        lifecycle_path="/stage/lifecycle.json",
    )
    contracts = (
        auto_research_orchestrator_task_contract(project_name="project", **common),
        auto_research_orchestrator_continuation_contract(
            original_contract_path="/stage/original.md",
            mode="continuation",
            **common,
        ),
    )
    expected = [
        "- `validate patch.json`",
        "- `apply --key <key> patch.json`",
        "- `status [--worker-id <worker-id> | --episode-id <episode-id>]`",
        "- `spawn --key <key> --seat-node <node-id> --instruction-file <filename>`",
        "- `pause --key <key> <worker-id>`",
        "- `resume --key <key> <worker-id>`",
        "- `stop --key <key> <worker-id>`",
        "- `message --key <key> --recipient <worker-id> <body>`",
        "- `watch-graph --key <key> --condition-json <json> --reason <text>`",
        (
            "- `episode --key <key> --kick-off-experiment --node <node-id> "
            "[--goal-file <filename>] [--invocation-limit <positive-int>]`"
        ),
        "- `episode --key <key> --stop <episode-id>`",
        "- `episode --key <key> --resume <episode-id>`",
        "- `inbox --key <key> --harvest`",
        "- `inbox --key <key> --clear`",
        "- `finish --key <key>`",
    ]

    for contract in contracts:
        command_block = contract.split("- Exact invocations, all prefixed by that command:\n", 1)[
            1
        ].split("  Each response is one JSON object.", 1)[0]
        assert [line.strip() for line in command_block.splitlines()] == expected
        assert "- RCP lifecycle facts: `/stage/lifecycle.json`" in contract
        assert "authoritative only about the child task and\nepisode transitions" in contract
        assert "There is no Retry command" in contract
        assert "--provider" not in contract
        assert "--model" not in contract
        assert "--effort" not in contract
        assert "--host" not in contract
        assert "--instruction <text>" not in contract


def test_auto_research_workers_cannot_orchestrate_or_wake_themselves() -> None:
    common = dict(
        graph_path="/stage/graph.json",
        research_path="/stage/research.md",
        repositories=[],
        patch_path="/stage/patch.json",
        output_schema_path="/stage/schema.json",
        validator_command="/stage/rcp-agent validate",
        reply_command="/stage/rcp-agent reply --key reply-once",
    )
    contracts = (
        auto_research_worker_task_contract(
            project_name="project",
            seat_node_type="Experiment",
            seat_node_id="exp/worker",
            seat_difficulty="Run the bounded check.",
            instruction_path="/stage/instruction.md",
            **common,
        ),
        auto_research_worker_continuation_contract(
            original_contract_path="/stage/original.md",
            mode="continuation",
            **common,
        ),
    )

    for contract in contracts:
        normalized = " ".join(contract.split())
        assert "start an episode" in normalized
        assert "register a watcher" in normalized
        assert "wake yourself" in normalized
        assert "Staged command client:" not in contract


def test_the_orchestrator_prefers_apply_while_the_worker_is_never_told_to_apply() -> None:
    """Settlement costs the episode a wake it can spend on work instead.

    Only the orchestrator holds an Apply credential, so the preference belongs in
    its command block and must not reach a worker's validate-only contract.
    """

    common = dict(
        graph_path="/stage/graph.json",
        research_path="/stage/research.md",
        repositories=[],
        patch_path="/stage/patch.json",
        output_schema_path="/stage/schema.json",
        validator_command="/stage/rcp-agent validate",
    )
    orchestrator = auto_research_orchestrator_task_contract(
        project_name="project",
        command_client="/stage/rcp-agent",
        **common,
    )
    worker = auto_research_worker_task_contract(
        project_name="project",
        seat_node_type="Experiment",
        seat_node_id="exp/worker",
        seat_difficulty="Run the bounded check.",
        instruction_path="/stage/instruction.md",
        reply_command="/stage/rcp-agent reply --key reply-once",
        **common,
    )

    preference = "Prefer `apply` over leaving the Patch for turn settlement"
    assert preference in orchestrator
    assert "another invocation wakes you" in orchestrator
    assert preference not in worker
    assert "apply --key" not in worker


def test_agent_resolvable_blockers_and_temporary_capacity_do_not_finish_the_episode() -> None:
    common = dict(
        graph_path="/stage/graph.json",
        research_path="/stage/research.md",
        repositories=[],
        patch_path="/stage/patch.json",
        output_schema_path="/stage/schema.json",
        validator_command="/stage/rcp-agent validate",
        command_client="/stage/rcp-agent",
    )
    contracts = (
        auto_research_orchestrator_task_contract(project_name="project", **common),
        auto_research_orchestrator_continuation_contract(
            original_contract_path="/stage/original.md",
            mode="continuation",
            **common,
        ),
    )

    for contract in contracts:
        assert (
            "Settled children are a prerequisite for finish, not a\n  reason to finish" in contract
        )
        assert "because a\n  Blocker remains" in contract
        # Capacity contention states its own remedy instead of riding along with two
        # conditions that have none. An orchestrator finished an episode with most of
        # its budget unspent because the cluster was full, having already identified
        # the scheduler as the way around it.
        assert "Capacity contention is never a reason to finish" in contract
        assert "seat a worker or kick off the Experiment to\n  submit the queued work" in contract
        assert "temporary resource or capacity contention" not in contract
        assert (
            "without new human judgment, credentials,\n  approval, privileged action, or "
            "coordination with another person" in contract
        )
        assert (
            "Act directly, delegate\n  executable work, or arrange an observable continuation"
            in contract
        )
        assert "Do not turn a self-service step into a recommended human next step" in contract
        assert (
            "complete all self-service diagnosis, preparation,\n  and prerequisites first"
            in contract
        )
        assert "only at that true human-only\n  boundary" in contract


def test_profile_aware_live_validator_uses_orchestrator_schema_and_authority(
    manifest,
    tmp_path,
) -> None:
    main_service = _service(manifest, tmp_path)
    main_service.history.append(
        _orchestrator_state_patch(
            {
                "id": "dec/validator",
                "type": "decision",
                "title": "Validation route",
                "question": "Which route?",
                "options": ["small", "large"],
                "status": "ready",
            }
        )
    )
    service, store, _auto_research, root, _worker = _setup_branch_auto_research(
        main_service,
        tmp_path / "store",
    )
    candidate = json.dumps(
        {
            "summary": "Selected the validated route.",
            "agent_action": "decision_choice",
            "ops": [
                {
                    "op": "update_nodes",
                    "nodes": [
                        {
                            "id": "dec/validator",
                            "changes": {"selected_option": "large", "status": "decided"},
                        }
                    ],
                }
            ],
        }
    )
    elevated = _validate_work_patch_live(
        service,
        candidate,
        run_truth_scope=["repo-a"],
        source_operation_id=root.operation_id,
        profile="orchestrator",
    )
    ordinary = _validate_work_patch_live(
        service,
        candidate,
        run_truth_scope=["repo-a"],
        source_operation_id=root.operation_id,
    )

    assert elevated.status == "valid"
    assert ordinary.status == "invalid"


def _seat_patch() -> Patch:
    return Patch(
        kind="refresh",
        author="agent",
        summary="Added the auto_research worker seat.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": "blk/check-result",
                        "type": "blocker",
                        "title": "Check the result",
                        "description": "Inspect the run and resolve the discrepancy.",
                        "status": "open",
                    }
                ],
            }
        ],
    )


def _service(manifest, tmp_path: Path) -> ProjectService:
    history = HistoryManager(manifest)
    history.append(_seat_patch())
    return ProjectService(
        manifest,
        history,
        PaperService(manifest, AppStore(tmp_path / "paper.sqlite3")),
        data_dir=tmp_path / "service-data",
        project_id="project",
    )


def _setup_auto_research(
    tmp_path: Path,
    *,
    root_status: str = "succeeded",
    worker_status: str = "running",
    native_session_id: str | None = None,
    stage_root: str | None = None,
    run_on: str = "laptop",
    invocation_ceiling: int = 12,
    episode_id: str = "auto_research",
    graph_base_head: GraphHeadRef | None = None,
) -> tuple[AppStore, EpisodeRecord, AgentTaskRecord, AgentTaskRecord]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    store = AppStore(tmp_path / "app.sqlite3")
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
    authorizer = fabricated_authorizer()
    now = store.now()
    graph_target = GraphTargetRef(kind="branch", branch_id=episode_id)
    graph_base_head = graph_base_head or GraphHeadRef(revision=0)
    root_request = AutoResearchRunRequest(
        episode_id=episode_id,
        role="orchestrator",
        actor_operation_id="root",
        provider="codex",
        model="",
        reasoning="medium",
        run_on="laptop",
        run_truth_scope=["repo-a"],
    )
    orchestrator_authority = AgentDispatchAuthority(
        profile="orchestrator",
        task_contract="orchestrate",
        scope=AgentDispatchScope(
            run_truth_scope=["repo-a"],
            episode_id=episode_id,
            patch_kind="work",
        ),
    )
    auto_research, root = store.create_auto_research_episode_with_root_task(
        EpisodeRecord(
            episode_id=episode_id,
            project_id="project",
            mode="auto_research",
            graph_target=graph_target,
            graph_base_head=graph_base_head,
            status="queued",
            invocation_ceiling=invocation_ceiling,
            authorized_by=authorizer,
            created_at=now,
            updated_at=now,
        ),
        AutoResearchStateRecord(
            episode_id=episode_id,
            starting_instruction=None,
            created_at=now,
            updated_at=now,
        ),
        AgentTaskRecord(
            operation_id="root",
            project_id="project",
            episode_id=episode_id,
            graph_target=graph_target,
            kind="auto_research",
            status="queued",
            request=root_request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message=f"root {root_status}",
            authorized_by=authorizer,
            dispatch_authority=orchestrator_authority,
        ),
    )
    if root_status == "succeeded":
        store.complete_agent_task(root.operation_id, applied_revision=None, result={})
    elif root_status == "running":
        store.mark_agent_task_running(root.operation_id)
    elif root_status != "queued":
        raise ValueError(f"unsupported root test status: {root_status}")
    root = store.agent_task(root.operation_id)
    assert root is not None
    assert auto_research.graph_target == root.graph_target == graph_target
    assert auto_research.graph_base_head == graph_base_head
    worker_request = AutoResearchRunRequest(
        episode_id=episode_id,
        role="worker",
        actor_operation_id="worker",
        provider="codex",
        model="",
        reasoning="medium",
        run_on=run_on,
        run_truth_scope=["repo-a"],
        session_id=native_session_id,
        control_node_id="blk/check-result",
        instruction="Inspect the discrepancy and report the concrete result.",
    )
    worker_authority = AgentDispatchAuthority(
        profile="ordinary",
        task_contract="work_auto",
        scope=AgentDispatchScope(
            run_truth_scope=["repo-a"],
            episode_id=episode_id,
            patch_kind="work",
        ),
    )
    worker = store.create_auto_research_agent_task(
        AgentTaskRecord(
            operation_id="worker",
            project_id="project",
            episode_id=episode_id,
            graph_target=graph_target,
            kind="auto_research",
            status=worker_status,
            request=worker_request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message=f"worker {worker_status}",
            parent_operation_id=root.operation_id,
            native_session_id=native_session_id,
            stage_root=stage_root,
            authorized_by=authorizer,
            dispatch_authority=worker_authority,
        ),
        role="worker",
    )
    assert worker.graph_target == graph_target
    return store, auto_research, root, worker


def _execution(
    store: AppStore,
    task: AgentTaskRecord,
    *,
    continuation="fresh",
) -> AgentTaskExecution:
    return AgentTaskExecution(
        operation_id=task.operation_id,
        store=store,
        control=AgentProcessControl(),
        stage_host=task.stage_host,
        stage_root=task.stage_root,
        continuation=continuation,
        retry_feedback=("provider disconnected after doing the work",)
        if continuation == "retry"
        else (),
    )


def _dispatcher(store: AppStore, replies: list[str] | None = None) -> AutoResearchCommandDispatcher:
    replies = replies if replies is not None else []

    def message(_context, arguments, _planned_message_id):
        replies.append(arguments.body)
        return AutoResearchCommandEffectResult(
            message="reply recorded",
            result={"graph_authority": "none", "epistemic_status": "hearsay"},
        )

    unavailable = lambda *_args: AutoResearchCommandEffectResult(  # noqa: E731
        status="unavailable", message="not used in this worker test"
    )
    return AutoResearchCommandDispatcher(
        store,
        AutoResearchCommandEffects(
            validate=lambda *_args: AutoResearchCommandEffectResult(
                result={"status": "valid", "messages": []}
            ),
            status=lambda *_args: AutoResearchCommandEffectResult(),
            spawn=unavailable,
            pause=unavailable,
            resume=unavailable,
            stop=unavailable,
            message=message,
            watch_graph=unavailable,
            finish=lambda _context, _planned_effect_id: AutoResearchCommandEffectResult(),
            seat_node_type=lambda _project_id, _node_id: "blocker",
            reconcile_unknown=lambda _context, _request, _planned_effect_id: None,
        ),
    )


def _contract(prompt: str) -> str:
    match = re.search(r"Open and follow the immutable RCP task contract at:\s*([^\n]+)", prompt)
    assert match is not None
    return Path(match.group(1)).read_text(encoding="utf-8")


async def _events(stream) -> list[AgentEvent]:
    return [
        AgentEvent.model_validate_json(frame.removeprefix("data: ").strip())
        async for frame in stream
    ]


def _capture_served_invocation_gates(monkeypatch) -> list[ProviderInvocationGate | None]:
    served: list[ProviderInvocationGate | None] = []
    original = auto_research_stream_module.serve_command_mailbox

    async def capture(*, invocation_gate=None, **kwargs):
        served.append(invocation_gate)
        await original(invocation_gate=invocation_gate, **kwargs)

    monkeypatch.setattr(auto_research_stream_module, "serve_command_mailbox", capture)
    return served


def _assert_fresh_matching_invocation_gates(
    served: list[ProviderInvocationGate | None],
    launched: list[ProviderInvocationGate | None],
) -> None:
    assert len(served) == len(launched)
    assert all(
        server is launcher and server is not None
        for server, launcher in zip(served, launched, strict=True)
    )
    assert len({id(gate) for gate in served}) == len(served)


class _WorkerLauncher:
    def __init__(self, *, session_id: str = "worker-session", writer=None) -> None:
        self.session_id = session_id
        self.writer = writer
        self.calls = 0
        self.contracts: list[str] = []
        self.requested_session_ids: list[str | None] = []
        self.read_dirs: list[list[Path]] = []
        self.invocation_gates: list[ProviderInvocationGate | None] = []

    async def stream(self, _provider, prompt, **kwargs):
        self.calls += 1
        self.requested_session_ids.append(kwargs["session_id"])
        self.read_dirs.append(list(kwargs["read_dirs"]))
        invocation_gate = kwargs.get("invocation_gate")
        self.invocation_gates.append(invocation_gate)

        async def events():
            workspace = Path(kwargs["cwd"])
            contract = _contract(prompt)
            self.contracts.append(contract)
            if self.writer is not None:
                result = self.writer(contract, workspace)
                if asyncio.iscoroutine(result):
                    await result
            yield AgentEvent(event="session", session_id=self.session_id)
            yield AgentEvent(event="answer", text="Worker completed the useful operation.")
            yield AgentEvent(event="done")

        if invocation_gate is None:
            async for event in events():
                yield event
            return
        async with invocation_gate.serve_current_session():
            async for event in events():
                yield event


def _enable_task_attribution(
    service: ProjectService,
    store: AppStore,
) -> None:
    service.history.project_id = "project"
    service.history.require_attribution = True
    service.history.agent_authority_resolver = store.agent_task_authority
    service.history.project_membership_check = seated_on_every_project


def _setup_branch_auto_research(
    main_service: ProjectService,
    tmp_path: Path,
) -> tuple[
    ProjectService,
    AppStore,
    EpisodeRecord,
    AgentTaskRecord,
    AgentTaskRecord,
]:
    """Create the real episode branch used by exact-target validation and Apply tests."""

    episode_id = str(uuid.uuid4())
    base_head = main_service.history.head_ref()
    store, episode, root, worker = _setup_auto_research(
        tmp_path,
        episode_id=episode_id,
        graph_base_head=base_head,
    )
    _enable_task_attribution(main_service, store)
    assert episode.authorized_by is not None
    main_service.history.create_auto_research_branch(
        GraphBranchMetadata(
            branch_id=episode_id,
            episode_id=episode_id,
            project_id=episode.project_id,
            base_head=base_head,
            head=GraphHeadRef(
                target=episode.graph_target,
                revision=base_head.revision,
                transition_id=base_head.transition_id,
            ),
            authorized_by=episode.authorized_by,
        )
    )
    branch_service = main_service.for_graph_target(
        episode.graph_target,
        expected_episode_id=episode_id,
    )
    return branch_service, store, episode, root, worker


def _orchestrator_state_patch(*nodes: dict[str, object]) -> Patch:
    return Patch(
        kind="refresh",
        author="agent",
        summary="Prepared orchestrator test state.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[{"op": "create_nodes", "nodes": list(nodes)}],
    )


class _LocalBackedRemoteStage:
    base: Path

    def __init__(self, host: str) -> None:
        self.host = host
        self.root: Path | None = None

    @property
    def workspace(self) -> Path:
        assert self.root is not None
        return self.root / "workspace"

    def open(self, operation_id: str, *, reuse: bool = False):
        del reuse
        self.root = self.base / operation_id
        (self.root / "inputs").mkdir(parents=True, exist_ok=True)
        self.workspace.mkdir(exist_ok=True)
        return self

    def attach(self, root: str):
        self.root = Path(root)
        return self

    def put_file(self, source: Path, label: str) -> str:
        assert self.root is not None
        target = self.root / "inputs" / label
        if target.exists():
            raise ValueError(f"immutable remote task input already exists: {label}")
        shutil.copyfile(source, target)
        return str(target)

    def read_input_text(self, label: str) -> str:
        assert self.root is not None
        try:
            return (self.root / "inputs" / label).read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ValueError("remote input is absent") from exc

    def finalize_inputs(self) -> None:
        return None

    def list_workspace_entries(self) -> list[str]:
        return sorted(item.name for item in self.workspace.iterdir())

    def list_workspace_files(self) -> list[str]:
        return sorted(item.name for item in self.workspace.iterdir() if item.is_file())

    def read_workspace_text(self, name: str, *, max_bytes: int | None = None) -> str:
        data = (self.workspace / name).read_bytes()
        if max_bytes is not None and len(data) > max_bytes:
            raise ValueError("mailbox file exceeds limit")
        return data.decode("utf-8")

    def read_text(self, path: Path) -> str:
        return Path(path).read_text(encoding="utf-8")

    def write_workspace_text(self, name: str, content: str) -> None:
        (self.workspace / name).write_text(content, encoding="utf-8")

    def remove_workspace_file(self, name: str) -> None:
        path = self.workspace / name
        if path.exists():
            path.unlink()

    def canonical_directories(
        self,
        paths: list[str],
        *,
        require_writable: bool,
    ) -> tuple[dict[str, str], str]:
        canonical: dict[str, str] = {}
        for raw in paths:
            resolved = Path(raw).resolve(strict=True)
            if not resolved.is_dir():
                raise ValueError(f"remote path is not a directory: {raw}")
            if require_writable and not os.access(resolved, os.W_OK):
                raise ValueError(f"remote path is not writable: {raw}")
            canonical[raw] = str(resolved)
        return canonical, str(Path.home().resolve())


def _record_original_contract(
    store: AppStore,
    worker: AgentTaskRecord,
    stage: Path,
    *,
    content: str = "# original worker contract\n",
) -> None:
    inputs = stage / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    original = inputs / "original-worker-contract.md"
    original.write_text(content, encoding="utf-8")
    store.record_agent_task_contract(
        worker.operation_id,
        "auto_research_worker",
        content,
        hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )
    store.record_agent_task_receipt(
        worker.operation_id,
        "agent_prompt",
        {"contract_path": str(original)},
        tier="diagnostic",
    )


def _recovery_task(
    store: AppStore,
    auto_research: EpisodeRecord,
    worker: AgentTaskRecord,
    *,
    operation_id: str = "worker-retry",
    continuation_cause: str = "retry",
) -> AgentTaskRecord:
    parent_request = AutoResearchRunRequest.model_validate(worker.request)
    request = parent_request.model_copy(
        update={
            "actor_operation_id": parent_request.actor_operation_id or worker.operation_id,
            "session_id": worker.native_session_id,
        }
    )
    now = store.now()
    recovery = store.create_auto_research_recovery_task(
        AgentTaskRecord(
            operation_id=operation_id,
            project_id=worker.project_id,
            episode_id=auto_research.episode_id,
            graph_target=auto_research.graph_target,
            kind="auto_research",
            status="running",
            request=request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="retry running",
            attempt=worker.attempt + 1,
            parent_operation_id=worker.operation_id,
            native_session_id=worker.native_session_id,
            stage_host=worker.stage_host,
            stage_root=worker.stage_root,
            authorized_by=auto_research.authorized_by,
            dispatch_authority=worker.dispatch_authority,
        ),
        continuation_cause=continuation_cause,
    )
    return recovery


def _claimed_mail_recovery(tmp_path: Path):
    stage = tmp_path / "data" / "run-stage" / _worker_stage_name("project", "worker")
    stage.mkdir(parents=True)
    store, auto_research, root, worker = _setup_auto_research(
        tmp_path,
        worker_status="succeeded",
        native_session_id="worker-session",
        stage_root=str(stage),
    )
    message = record_auto_research_message(
        store,
        episode_id=auto_research.episode_id,
        sender_role="orchestrator",
        sender_task_id=root.operation_id,
        authorized_by=None,
        recipient_task_id=worker.operation_id,
        control_node_id="blk/check-result",
        body="Retain this exact claimed message across recovery.",
    )
    request = AutoResearchRunRequest.model_validate(worker.request).model_copy(
        update={"instruction": None, "wake_cause": "message"}
    )
    now = store.now()
    wake = store.create_auto_research_message_wake_task(
        AgentTaskRecord(
            operation_id="mail-wake",
            project_id=worker.project_id,
            episode_id=auto_research.episode_id,
            graph_target=auto_research.graph_target,
            kind="auto_research",
            status="running",
            request=request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="mail wake running",
            parent_operation_id=worker.operation_id,
            native_session_id=worker.native_session_id,
            stage_root=worker.stage_root,
            authorized_by=auto_research.authorized_by,
            dispatch_authority=worker.dispatch_authority,
        ),
        role="worker",
        recipient_task_id=worker.operation_id,
        message_ids=[message.message_id],
    )
    assert wake is not None
    store.fail_agent_task(wake.operation_id, "provider connection interrupted")
    retry = _recovery_task(store, auto_research, wake, operation_id="mail-wake-retry")
    execution = _execution(store, retry, continuation="retry")
    turn = auto_research_stream_module._canonical_worker_turn(
        execution,
        AutoResearchRunRequest.model_validate(retry.request),
    )
    worker_stage = auto_research_stream_module._WorkerStage(
        local=stage,
        remote=None,
        workspace=stage,
        execution_host="",
        provider_binary=None,
    )
    delivery = auto_research_mail_delivery(
        episode_id=auto_research.episode_id,
        recipient_task_id=worker.operation_id,
        delivery_operation_id=wake.operation_id,
        messages=store.auto_research_messages(auto_research.episode_id),
    )
    return execution, turn, worker_stage, delivery


@pytest.mark.asyncio
async def test_canonical_worker_binding_rejects_requested_role_mismatch(manifest, tmp_path) -> None:
    store, _auto_research, _root, worker = _setup_auto_research(tmp_path)
    launcher = _WorkerLauncher()
    supplied = AutoResearchRunRequest.model_validate(worker.request).model_copy(
        update={"role": "orchestrator", "control_node_id": None}
    )

    events = await _events(
        stream_auto_research_worker_run(
            _service(manifest, tmp_path),
            launcher,
            supplied,
            tmp_path / "data",
            _execution(store, worker),
            command_dispatcher=_dispatcher(store),
        )
    )

    assert launcher.calls == 0
    assert events[-1].event == "error"
    assert "differs from its durable task record" in events[-1].text


@pytest.mark.asyncio
async def test_orchestrator_stream_uses_elevated_profile_commands_and_work_apply(
    manifest,
    tmp_path,
    monkeypatch,
) -> None:
    main_service = _service(manifest, tmp_path)
    main_service.history.append(
        _orchestrator_state_patch(
            {
                "id": "dec/budget",
                "type": "decision",
                "title": "Evaluation budget",
                "question": "Which budget?",
                "options": ["small", "large"],
                "status": "ready",
            },
            {
                "id": "ev/result",
                "type": "evidence",
                "title": "Evaluation result",
                "observation": "The larger budget exposed the failure mode.",
                "origin": "internal_run",
            },
        )
    )
    service, store, auto_research, root, _worker = _setup_branch_auto_research(
        main_service,
        tmp_path / "store",
    )
    execution = _execution(store, root)

    def apply_patch(_context, patch_text, source_effect_id):
        result, failure = _apply_work_patch(
            service,
            execution,
            patch_text,
            run_truth_scope=["repo-a"],
            profile="orchestrator",
            source_operation_id=root.operation_id,
            source_effect_id=source_effect_id,
        )
        if failure is not None:
            return None, failure.message, failure.correctable
        return result, None, False

    def worker_request_factory(*_args):
        raise AssertionError("this test does not spawn ordinary Work")

    dispatcher = AutoResearchCommandDispatcher(
        store,
        auto_research_command_effects(
            store=store,
            background=SimpleNamespace(store=store),
            validate=lambda *_args: AutoResearchCommandEffectResult(),
            worker_request_factory=worker_request_factory,
            graph_state=service.history.state,
            execution_host="",
            apply_patch=apply_patch,
        ),
    )
    command_results: list[dict[str, object]] = []
    observed: dict[str, object] = {}
    original_consume = RunStageMailbox.remove_if_sha256
    consume_attempts = 0

    def transiently_unavailable_consume(
        mailbox: RunStageMailbox,
        name: str,
        expected_sha256: str,
    ) -> bool:
        nonlocal consume_attempts
        consume_attempts += 1
        if consume_attempts == 1:
            raise OSError("the stage is temporarily unavailable")
        return original_consume(mailbox, name, expected_sha256)

    monkeypatch.setattr(
        RunStageMailbox,
        "remove_if_sha256",
        transiently_unavailable_consume,
    )
    candidate = json.dumps(
        {
            "summary": "Chose the diagnostic budget and accepted its evidence.",
            "agent_action": "decision_choice",
            "repositories_read": ["repo-a"],
            "change_summary": ["Selected the large evaluation budget."],
            "ops": [
                {
                    "op": "update_nodes",
                    "nodes": [
                        {
                            "id": "dec/budget",
                            "changes": {"selected_option": "large", "status": "decided"},
                        }
                    ],
                },
                {"op": "set_standing", "node_id": "ev/result", "standing": "accepted"},
            ],
        }
    )
    second_candidate = json.dumps(
        {
            "summary": "Recorded a second in-turn observation.",
            "repositories_read": ["repo-a"],
            "change_summary": ["Added the second in-turn observation."],
            "ops": [
                {
                    "op": "create_nodes",
                    "nodes": [
                        {
                            "id": "ev/in-turn-two",
                            "type": "evidence",
                            "title": "Second in-turn result",
                            "observation": "The second Apply committed independently.",
                            "origin": "internal_run",
                        }
                    ],
                }
            ],
        }
    )
    fallback_candidate = json.dumps(
        {
            "summary": "Recorded the final unconsumed observation.",
            "repositories_read": ["repo-a"],
            "change_summary": ["Added the end-of-turn fallback observation."],
            "ops": [
                {
                    "op": "create_nodes",
                    "nodes": [
                        {
                            "id": "ev/final-fallback",
                            "type": "evidence",
                            "title": "Final fallback result",
                            "observation": "The final Patch settled after two in-turn Applies.",
                            "origin": "internal_run",
                        }
                    ],
                }
            ],
        }
    )

    async def writer(contract_text: str, workspace: Path) -> None:
        assert "one project-owned auto-research orchestrator profile" in contract_text
        prefix = re.search(r"Command prefix(?: for this turn)?: `([^`]+)`", contract_text)
        assert prefix is not None
        process = await asyncio.create_subprocess_exec(
            *shlex.split(prefix.group(1)),
            "status",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        assert process.returncode == 0, stderr.decode()
        command_results.append(json.loads(stdout))
        (workspace / "patch.json").write_text(candidate, encoding="utf-8")
        process = await asyncio.create_subprocess_exec(
            *shlex.split(prefix.group(1)),
            "apply",
            "--key",
            "apply-diagnostic-budget",
            str(workspace / "patch.json"),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        assert process.returncode == 2, stdout.decode() + stderr.decode()
        command_results.append(json.loads(stdout))
        assert (workspace / "patch.json").is_file()
        process = await asyncio.create_subprocess_exec(
            *shlex.split(prefix.group(1)),
            "apply",
            "--key",
            "apply-diagnostic-budget",
            str(workspace / "patch.json"),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        assert process.returncode == 0, stdout.decode() + stderr.decode()
        command_results.append(json.loads(stdout))
        (workspace / "patch.json").write_text(second_candidate, encoding="utf-8")
        process = await asyncio.create_subprocess_exec(
            *shlex.split(prefix.group(1)),
            "apply",
            "--key",
            "apply-second-observation",
            str(workspace / "patch.json"),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        assert process.returncode == 0, stdout.decode() + stderr.decode()
        command_results.append(json.loads(stdout))
        (workspace / "patch.json").write_text(fallback_candidate, encoding="utf-8")

    class Launcher(_WorkerLauncher):
        async def stream(self, provider, prompt, **kwargs):
            observed["capability"] = kwargs["capability"]
            observed["session_id"] = kwargs["session_id"]
            async for event in super().stream(provider, prompt, **kwargs):
                yield event

    events = await _events(
        stream_auto_research_orchestrator_run(
            service,
            Launcher(session_id="orchestrator-session", writer=writer),
            AutoResearchRunRequest.model_validate(root.request),
            tmp_path / "data",
            execution,
            command_dispatcher=dispatcher,
        )
    )

    assert events[-1].event == "done", [(event.event, event.text) for event in events]
    assert observed == {"capability": "orchestrate", "session_id": None}
    assert len(command_results) == 4
    assert command_results[0]["status"] == "ok"
    assert command_results[0]["result"]["episode"]["episode_id"] == auto_research.episode_id
    assert command_results[1]["status"] == "unavailable"
    assert command_results[2]["status"] == "ok"
    assert command_results[2]["result"]["disposition"] == "applied"
    assert command_results[2]["result"]["patch_consumed"] is True
    assert command_results[3]["status"] == "ok"
    assert command_results[3]["result"]["disposition"] == "applied"
    assert command_results[3]["result"]["patch_consumed"] is True
    state = service.history.state()
    assert state.nodes["dec/budget"].status == "decided"
    assert state.nodes["dec/budget"].selected_option == "large"
    assert state.nodes["ev/result"].standing == "accepted"
    assert "ev/in-turn-two" in state.nodes
    assert "ev/final-fallback" in state.nodes
    applied_patches = [
        patch
        for patch in service.history.load_patches()
        if patch.source_operation_id == root.operation_id
    ]
    assert len(applied_patches) == 3
    assert all(patch.profile == "orchestrator" for patch in applied_patches)
    assert applied_patches[0].agent_action is None
    assert applied_patches[0].transition is not None
    assert applied_patches[0].transition.initiating_groups[0].agent_action == "decision_choice"
    assert all(patch.source_effect_id is not None for patch in applied_patches)
    assert len({patch.source_effect_id for patch in applied_patches}) == 3
    assert all(patch.source_effect_sha256 is not None for patch in applied_patches)
    assert all(patch.task_id == root.operation_id for patch in applied_patches)
    assert all(patch.authorized_by == auto_research.authorized_by for patch in applied_patches)
    task_result = json.loads(next(event.text for event in events if event.event == "message"))
    assert [item["applied_revision"] for item in task_result["graph_updates"]] == [3, 4, 5]

    committed_revision = state.revision
    replayed, failure = _apply_work_patch(
        service,
        execution,
        candidate,
        run_truth_scope=["repo-a"],
        profile="orchestrator",
        source_operation_id=root.operation_id,
        source_effect_id=applied_patches[0].source_effect_id,
    )
    assert failure is None
    assert replayed is not None
    assert replayed.applied_revision == applied_patches[0].revision
    assert service.history.state().revision == committed_revision
    assert (
        len(
            [
                patch
                for patch in service.history.load_patches()
                if patch.source_effect_id == applied_patches[0].source_effect_id
            ]
        )
        == 1
    )
    mismatched, mismatch_failure = _apply_work_patch(
        service,
        execution,
        fallback_candidate,
        run_truth_scope=["repo-a"],
        profile="orchestrator",
        source_operation_id=root.operation_id,
        source_effect_id=applied_patches[0].source_effect_id,
    )
    assert mismatched is None
    assert mismatch_failure is not None
    assert "different canonical Patch" in mismatch_failure.message
    replayed_fallback, fallback_failure = _apply_work_patch(
        service,
        execution,
        fallback_candidate,
        run_truth_scope=["repo-a"],
        profile="orchestrator",
        source_operation_id=root.operation_id,
        source_effect_id=applied_patches[-1].source_effect_id,
    )
    assert fallback_failure is None
    assert replayed_fallback is not None
    assert replayed_fallback.applied_revision == committed_revision
    assert service.history.state().revision == committed_revision


@pytest.mark.asyncio
async def test_orchestrator_final_settlement_recovers_apply_committed_before_consume(
    manifest,
    tmp_path,
    monkeypatch,
) -> None:
    main_service = _service(manifest, tmp_path)
    service, store, auto_research, root, _worker = _setup_branch_auto_research(
        main_service,
        tmp_path / "store",
    )
    execution = _execution(store, root)

    def apply_patch(_context, patch_text, source_effect_id):
        result, failure = _apply_work_patch(
            service,
            execution,
            patch_text,
            run_truth_scope=["repo-a"],
            profile="orchestrator",
            source_operation_id=root.operation_id,
            source_effect_id=source_effect_id,
        )
        if failure is not None:
            return None, failure.message, failure.correctable
        return result, None, False

    def worker_request_factory(*_args):
        raise AssertionError("this test does not spawn ordinary Work")

    dispatcher = AutoResearchCommandDispatcher(
        store,
        auto_research_command_effects(
            store=store,
            background=SimpleNamespace(store=store),
            validate=lambda *_args: AutoResearchCommandEffectResult(),
            worker_request_factory=worker_request_factory,
            graph_state=service.history.state,
            execution_host="",
            apply_patch=apply_patch,
        ),
    )
    candidate = json.dumps(
        {
            "summary": "Recorded the result whose Apply receipt was interrupted.",
            "repositories_read": ["repo-a"],
            "change_summary": ["Added the interrupted Apply result."],
            "ops": [
                {
                    "op": "create_nodes",
                    "nodes": [
                        {
                            "id": "ev/interrupted-apply",
                            "type": "evidence",
                            "title": "Interrupted Apply",
                            "observation": "The canonical commit preceded handoff consumption.",
                            "origin": "internal_run",
                        }
                    ],
                }
            ],
        }
    )
    original_consume = RunStageMailbox.remove_if_sha256
    consume_attempts = 0
    retained_workspace: Path | None = None
    command_result: dict[str, object] | None = None

    def fail_first_consume(
        mailbox: RunStageMailbox,
        name: str,
        expected_sha256: str,
    ) -> bool:
        nonlocal consume_attempts
        consume_attempts += 1
        if consume_attempts == 1:
            raise OSError("the stage disappeared after the canonical commit")
        return original_consume(mailbox, name, expected_sha256)

    monkeypatch.setattr(RunStageMailbox, "remove_if_sha256", fail_first_consume)

    async def writer(contract_text: str, workspace: Path) -> None:
        nonlocal retained_workspace, command_result
        retained_workspace = workspace
        prefix = re.search(r"Command prefix(?: for this turn)?: `([^`]+)`", contract_text)
        assert prefix is not None
        (workspace / "patch.json").write_text(candidate, encoding="utf-8")
        process = await asyncio.create_subprocess_exec(
            *shlex.split(prefix.group(1)),
            "apply",
            "--key",
            "apply-before-consume-failure",
            str(workspace / "patch.json"),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        assert process.returncode == 2, stdout.decode() + stderr.decode()
        command_result = json.loads(stdout)
        assert (workspace / "patch.json").read_text(encoding="utf-8") == candidate

    events = await _events(
        stream_auto_research_orchestrator_run(
            service,
            _WorkerLauncher(session_id="orchestrator-session", writer=writer),
            AutoResearchRunRequest.model_validate(root.request),
            tmp_path / "data",
            execution,
            command_dispatcher=dispatcher,
        )
    )

    assert events[-1].event == "done", [(event.event, event.text) for event in events]
    assert command_result is not None
    assert command_result["status"] == "unavailable"
    assert consume_attempts == 2
    assert retained_workspace is not None
    assert not (retained_workspace / "patch.json").exists()
    assert "ev/interrupted-apply" in service.history.state().nodes
    applied_patches = [
        patch
        for patch in service.history.load_patches()
        if patch.source_operation_id == root.operation_id
    ]
    assert len(applied_patches) == 1
    task_result = json.loads(next(event.text for event in events if event.event == "message"))
    assert task_result["applied_revision"] == applied_patches[0].revision
    assert [item["applied_revision"] for item in task_result["graph_updates"]] == [
        applied_patches[0].revision
    ]
    assert task_result["graph_update"]["applied_revision"] == applied_patches[0].revision
    assert store.auto_research_apply_results(root.operation_id) == []
    assert applied_patches[0].episode_id == auto_research.episode_id


def test_orchestrator_final_settlement_refuses_ambiguous_apply_commits(
    manifest,
    tmp_path,
    monkeypatch,
) -> None:
    service = _service(manifest, tmp_path)
    store, _auto_research, root, _worker = _setup_auto_research(tmp_path / "store")
    digest = hashlib.sha256(b"same retained patch").hexdigest()
    matches = [
        SimpleNamespace(
            admission="accepted",
            kind="work",
            source_operation_id=root.operation_id,
            source_effect_sha256=digest,
            source_effect_id="effect-one",
        ),
        SimpleNamespace(
            admission="accepted",
            kind="work",
            source_operation_id=root.operation_id,
            source_effect_sha256=digest,
            source_effect_id="effect-two",
        ),
    ]
    monkeypatch.setattr(service.history, "load_patches", lambda: matches)

    with pytest.raises(ValueError, match="multiple canonical in-turn Apply commits"):
        _orchestrator_final_source_effect_id(
            service,
            _execution(store, root),
            patch_digest=digest,
        )


@pytest.mark.asyncio
async def test_orchestrator_result_orders_applied_invalid_and_valid_empty_dispositions(
    manifest,
    tmp_path,
) -> None:
    service = _service(manifest, tmp_path)
    store, auto_research, root, _worker = _setup_auto_research(tmp_path / "store")
    updates = [
        GraphUpdateResult(status="applied", applied_revision=7),
        GraphUpdateResult(
            status="rejected",
            validation_messages=["The corrected node does not exist."],
            repairable=True,
        ),
        GraphUpdateResult(status="none"),
    ]
    for index, update in enumerate(updates):
        outcome = AutoResearchCommandEffectResult(
            status="invalid" if update.status == "rejected" else "ok",
            message=(
                "The Patch is invalid."
                if update.status == "rejected"
                else "The Patch reached a durable disposition."
            ),
            result={
                "disposition": (
                    "applied"
                    if update.status == "applied"
                    else "invalid"
                    if update.status == "rejected"
                    else "valid_empty"
                ),
                "graph_update": update.model_dump(mode="json"),
            },
        )
        store.save_auto_research_apply_result(
            AutoResearchApplyResultRecord(
                apply_id=f"apply-{index}",
                episode_id=auto_research.episode_id,
                operation_id=root.operation_id,
                patch_sha256=hashlib.sha256(f"patch-{index}".encode()).hexdigest(),
                result=outcome.model_dump(mode="json"),
                created_at=f"2026-08-16T00:00:0{index}+00:00",
            )
        )

    events = await _events(
        stream_auto_research_orchestrator_run(
            service,
            _WorkerLauncher(session_id="orchestrator-session"),
            AutoResearchRunRequest.model_validate(root.request),
            tmp_path / "data",
            _execution(store, root),
            command_dispatcher=_dispatcher(store),
        )
    )

    payload = json.loads(next(event.text for event in events if event.event == "message"))
    assert [update["status"] for update in payload["graph_updates"]] == [
        "applied",
        "rejected",
        "none",
    ]
    assert payload["graph_update"]["status"] == "none"
    assert payload["applied_revision"] == 7
    assert events[-1].event == "done"


@pytest.mark.asyncio
async def test_orchestrator_stream_rejects_direct_existing_belief_change(
    manifest,
    tmp_path,
    monkeypatch,
) -> None:
    main_service = _service(manifest, tmp_path)
    main_service.history.append(
        _orchestrator_state_patch(
            {
                "id": "rq/existing",
                "type": "research_question",
                "title": "Existing question",
                "question": "What already exists?",
            }
        )
    )
    service, store, _auto_research, root, _worker = _setup_branch_auto_research(
        main_service,
        tmp_path / "store",
    )
    candidate = json.dumps(
        {
            "summary": "Tried to rewrite an existing protected belief.",
            "ops": [
                {
                    "op": "update_nodes",
                    "nodes": [
                        {
                            "id": "rq/existing",
                            "changes": {"question": "A changed question?"},
                        }
                    ],
                }
            ],
        }
    )

    def writer(_contract_text: str, workspace: Path) -> None:
        (workspace / "patch.json").write_text(candidate, encoding="utf-8")

    served_gates = _capture_served_invocation_gates(monkeypatch)
    launcher = _WorkerLauncher(session_id="orchestrator-session", writer=writer)
    events = await _events(
        stream_auto_research_orchestrator_run(
            service,
            launcher,
            AutoResearchRunRequest.model_validate(root.request),
            tmp_path / "data",
            _execution(store, root),
            command_dispatcher=_dispatcher(store),
        )
    )

    updates = [
        json.loads(event.text)["graph_update"] for event in events if event.event == "message"
    ]
    assert updates[-1]["status"] == "rejected"
    assert any("not permit" in message for message in updates[-1]["validation_messages"])
    assert service.history.state().nodes["rq/existing"].question == "What already exists?"
    assert launcher.calls == 3
    _assert_fresh_matching_invocation_gates(served_gates, launcher.invocation_gates)


@pytest.mark.asyncio
async def test_orchestrator_continuation_preserves_actor_session_stage_and_handoff_fence(
    manifest,
    tmp_path,
) -> None:
    service = _service(manifest, tmp_path)
    store, auto_research, root, _worker = _setup_auto_research(tmp_path / "store")
    data_dir = tmp_path / "data"
    fresh_launcher = _WorkerLauncher(session_id="orchestrator-session")

    fresh_events = await _events(
        stream_auto_research_orchestrator_run(
            service,
            fresh_launcher,
            AutoResearchRunRequest.model_validate(root.request),
            data_dir,
            _execution(store, root),
            command_dispatcher=_dispatcher(store),
        )
    )
    assert fresh_events[-1].event == "done"
    stage = data_dir / "run-stage" / _orchestrator_stage_name("project", root.operation_id)
    stored_root = store.agent_task(root.operation_id)
    assert stored_root is not None and stored_root.stage_root == str(stage)
    assert store.auto_research_handoffs_cleared(root.operation_id) is True
    store.checkpoint_agent_task(root.operation_id, native_session_id="orchestrator-session")

    root_request = AutoResearchRunRequest.model_validate(stored_root.request)
    continuation_request = root_request.model_copy(
        update={
            "actor_operation_id": root.operation_id,
            "instruction": None,
            "session_id": "orchestrator-session",
        }
    )
    now = store.now()
    continuation = store.create_auto_research_agent_task(
        AgentTaskRecord(
            operation_id="orchestrator-continuation",
            project_id=root.project_id,
            episode_id=auto_research.episode_id,
            graph_target=auto_research.graph_target,
            kind="auto_research",
            status="running",
            request=continuation_request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="orchestrator continuation running",
            parent_operation_id=root.operation_id,
            native_session_id="orchestrator-session",
            stage_root=str(stage),
            authorized_by=auto_research.authorized_by,
            dispatch_authority=root.dispatch_authority,
        ),
        role="orchestrator",
        continuation_cause="auto_research_continuation",
    )
    for name in ("patch.json", "watch.json", "messages.json"):
        (stage / name).write_text("stale", encoding="utf-8")
    continuation_launcher = _WorkerLauncher(session_id="orchestrator-session")
    continuation_events = await _events(
        stream_auto_research_orchestrator_run(
            service,
            continuation_launcher,
            AutoResearchRunRequest.model_validate(continuation.request),
            data_dir,
            _execution(store, continuation, continuation="auto_research_continuation"),
            command_dispatcher=_dispatcher(store),
        )
    )
    assert continuation_events[-1].event == "done"
    assert continuation_launcher.requested_session_ids == ["orchestrator-session"]
    assert all(
        not (stage / name).exists() for name in ("patch.json", "watch.json", "messages.json")
    )
    assert store.auto_research_handoffs_cleared(continuation.operation_id) is True

    store.fail_agent_task(continuation.operation_id, "provider disconnected")
    empty_patch = json.dumps({"summary": "No graph change.", "ops": [], "repositories_read": []})
    (stage / "patch.json").write_text(empty_patch, encoding="utf-8")
    (stage / "watch.json").write_text("retained watcher handoff", encoding="utf-8")
    (stage / "messages.json").write_text("unowned retained mail", encoding="utf-8")
    retry = _recovery_task(
        store,
        auto_research,
        continuation,
        operation_id="orchestrator-retry",
    )

    def inspect_recovery(contract_text: str, workspace: Path) -> None:
        assert "orchestrator continuation" in contract_text
        assert (workspace / "patch.json").read_text(encoding="utf-8") == empty_patch
        assert (workspace / "watch.json").read_text(encoding="utf-8") == (
            "retained watcher handoff"
        )
        assert not (workspace / "messages.json").exists()

    retry_launcher = _WorkerLauncher(
        session_id="orchestrator-session",
        writer=inspect_recovery,
    )
    retry_events = await _events(
        stream_auto_research_orchestrator_run(
            service,
            retry_launcher,
            AutoResearchRunRequest.model_validate(retry.request),
            data_dir,
            _execution(store, retry, continuation="retry"),
            command_dispatcher=_dispatcher(store),
        )
    )
    assert retry_events[-1].event == "done"
    assert retry_launcher.requested_session_ids == ["orchestrator-session"]
    reused_stage_gates = [
        fresh_launcher.invocation_gates[0],
        continuation_launcher.invocation_gates[0],
        retry_launcher.invocation_gates[0],
    ]
    assert all(gate is not None for gate in reused_stage_gates)
    assert len({id(gate) for gate in reused_stage_gates}) == 3
    assert store.auto_research_handoffs_cleared(continuation.operation_id) is True


@pytest.mark.parametrize(
    "failure_point",
    ["session-limit", "saved-stage", "pre-stage"],
)
def test_orchestrator_clean_retry_binds_replacement_session_in_production_stream(
    manifest,
    tmp_path,
    failure_point,
) -> None:
    service = _service(manifest, tmp_path)
    store, _auto_research, root, _worker = _setup_auto_research(
        tmp_path / "store",
        root_status="running",
    )
    data_dir = tmp_path / "data"
    stage = data_dir / "run-stage" / _orchestrator_stage_name("project", root.operation_id)
    if failure_point != "pre-stage":
        stage.mkdir(parents=True)
        store.checkpoint_agent_task(
            root.operation_id,
            native_session_id=("spent-session" if failure_point == "session-limit" else None),
            stage_host=None,
            stage_root=str(stage),
        )
        original_contract = "# original auto_research orchestrator contract\n"
        original_path = stage / "inputs" / "original-orchestrator-contract.md"
        original_path.parent.mkdir(parents=True)
        original_path.write_text(original_contract, encoding="utf-8")
        store.record_agent_task_contract(
            root.operation_id,
            "auto_research_orchestrator",
            original_contract,
            hashlib.sha256(original_contract.encode("utf-8")).hexdigest(),
        )
        store.record_agent_task_receipt(
            root.operation_id,
            "agent_prompt",
            {"contract_path": str(original_path)},
            tier="diagnostic",
        )
    store.fail_agent_task(
        root.operation_id,
        "provider session limit"
        if failure_point == "session-limit"
        else "Network connection failed before session creation.",
    )
    root = store.agent_task(root.operation_id)
    assert root is not None

    def writer(_contract_text: str, workspace: Path) -> None:
        workspace.joinpath("patch.json").write_text(
            json.dumps(
                {
                    "summary": "No graph change was required after recovery.",
                    "ops": [],
                    "repositories_read": [],
                }
            ),
            encoding="utf-8",
        )

    launcher = _WorkerLauncher(session_id="replacement-session", writer=writer)

    async def stream(_project_id, kind, request, execution):
        assert kind == "auto_research"
        async for frame in stream_auto_research_orchestrator_run(
            service,
            launcher,
            request,
            data_dir,
            execution,
            command_dispatcher=_dispatcher(store),
        ):
            yield frame

    tasks = BackgroundAgentTasks(store, stream)
    tasks.recover_at_startup()
    retry = tasks.retry(root.operation_id)
    retry = wait_for_task(store, retry.operation_id, expect="succeeded")

    assert launcher.requested_session_ids == [None]
    assert retry.native_session_id == "replacement-session"
    assert retry.stage_root == str(stage)
    clean_retry_receipt = next(
        receipt
        for receipt in store.agent_task_receipts(retry.operation_id)
        if receipt.category == "auto_research_orchestrator_clean_retry"
    )
    assert clean_retry_receipt.payload == {
        "classification": (
            "session_limit" if failure_point == "session-limit" else "checkpoint_missing"
        ),
        "same_allocation": True,
        "actor_operation_id": root.operation_id,
        "retry_mode": "clean_native_session",
    }
    if failure_point == "pre-stage":
        assert launcher.contracts[0].startswith("# RCP auto-research orchestrator contract")
    else:
        assert launcher.contracts[0].startswith("# RCP auto-research orchestrator continuation")
    binding = store.auto_research_actor_binding(retry.operation_id)
    assert binding.current_operation_id == retry.operation_id
    assert binding.native_session_id == "replacement-session"
    assert binding.stage_root == str(stage)


@pytest.mark.asyncio
async def test_orchestrator_null_session_resume_is_not_a_clean_retry(
    manifest,
    tmp_path,
) -> None:
    service = _service(manifest, tmp_path)
    store, auto_research, root, _worker = _setup_auto_research(
        tmp_path / "store",
        root_status="running",
    )
    data_dir = tmp_path / "data"
    stage = data_dir / "run-stage" / _orchestrator_stage_name("project", root.operation_id)
    stage.mkdir(parents=True)
    store.checkpoint_agent_task(
        root.operation_id,
        native_session_id="spent-session",
        stage_host=None,
        stage_root=str(stage),
    )
    store.fail_agent_task(root.operation_id, "provider session limit")
    root = store.agent_task(root.operation_id)
    assert root is not None
    root_request = AutoResearchRunRequest.model_validate(root.request)
    clean_request = root_request.model_copy(update={"session_id": None})
    now = store.now()
    resumed = store.create_auto_research_recovery_task(
        AgentTaskRecord(
            operation_id="orchestrator-null-resume",
            project_id=root.project_id,
            episode_id=auto_research.episode_id,
            graph_target=auto_research.graph_target,
            kind="auto_research",
            status="running",
            request=clean_request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="clean retry running",
            attempt=root.attempt + 1,
            parent_operation_id=root.operation_id,
            native_session_id=None,
            stage_host=root.stage_host,
            stage_root=root.stage_root,
            authorized_by=auto_research.authorized_by,
            dispatch_authority=root.dispatch_authority,
        ),
        continuation_cause="resume",
    )

    launcher = _WorkerLauncher(session_id="replacement-session")
    events = await _events(
        stream_auto_research_orchestrator_run(
            service,
            launcher,
            AutoResearchRunRequest.model_validate(resumed.request),
            data_dir,
            _execution(store, resumed, continuation="resume"),
            command_dispatcher=_dispatcher(store),
        )
    )

    assert events[-1].event == "error"
    assert "requires its exact session and stage" in events[-1].text
    assert launcher.calls == 0
    assert store.agent_task(resumed.operation_id).native_session_id is None


def test_validate_only_dispatch_audits_denial_without_keyed_lookup_or_replay(tmp_path) -> None:
    store, auto_research, root, worker = _setup_auto_research(tmp_path)
    completed_id = "1" * 32
    unknown_id = "2" * 32
    for command_id, key in (
        (completed_id, "completed-message"),
        (unknown_id, "unknown-message"),
    ):
        store.start_agent_command(
            operation_id=worker.operation_id,
            command_id=command_id,
            episode_id=auto_research.episode_id,
            verb="message",
            idempotency_key=key,
            payload={"request_id": command_id, "arguments": {"body": "original"}},
        )
    store.finish_agent_command(
        completed_id,
        status="ok",
        payload={"result": {"delivered": True}},
        message="Original message completed.",
    )
    replies: list[str] = []
    base = _dispatcher(store, replies)
    dispatcher = _ValidateOnlyAutoResearchCommandDispatcher(store, base.effects)
    with store.connection() as connection:
        before = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM graph_run_events ORDER BY event_id"
            ).fetchall()
        ]

    for request_id, key in (("3" * 32, "completed-message"), ("4" * 32, "unknown-message")):
        response = dispatcher.dispatch(
            worker.operation_id,
            MessageCommandRequest(
                mailbox_id="a" * 32,
                request_id=request_id,
                credential="b" * 64,
                verb="message",
                idempotency_key=key,
                arguments={
                    "recipient_task_id": root.operation_id,
                    "body": "must not replay",
                },
            ),
        )
        assert response.status == "invalid"
        assert "validation only" in (response.message or "")

    with store.connection() as connection:
        after = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM graph_run_events ORDER BY event_id"
            ).fetchall()
        ]
    assert [row for row in after if row["command_id"] in {completed_id, unknown_id}] == [
        row for row in before if row["command_id"] in {completed_id, unknown_id}
    ]
    audited = after[len(before) :]
    assert len(audited) == 4
    for offset, (request_id, key) in enumerate(
        (("3" * 32, "completed-message"), ("4" * 32, "unknown-message"))
    ):
        started, exited = audited[offset * 2 : offset * 2 + 2]
        assert started["command_phase"] == "start"
        assert exited["command_phase"] == "exit"
        assert started["command_id"] == exited["command_id"]
        assert started["command_id"] not in {completed_id, unknown_id}
        assert started["idempotency_key"] is None
        assert exited["idempotency_key"] is None
        start_payload = json.loads(started["payload_json"])
        exit_payload = json.loads(exited["payload_json"])
        assert start_payload["request_id"] == request_id
        assert start_payload["supplied_idempotency_key"] == key
        assert start_payload["denied_by"] == "auto_research_patch_correction_validate_only"
        assert exit_payload["status"] == "invalid"
    assert replies == []
    assert store.agent_command(completed_id).status == "ok"  # type: ignore[union-attr]
    assert store.agent_command(unknown_id).exited_at is None  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_main_mailbox_is_closed_when_prompt_build_fails_after_staging(
    manifest, tmp_path, monkeypatch
) -> None:
    service = _service(manifest, tmp_path)
    store, _auto_research, _root, worker = _setup_auto_research(tmp_path)
    staged_mailboxes = []
    started: list[str] = []
    finished: list[str] = []

    def capture_mailbox(**kwargs):
        staged = _stage_command_mailbox(**kwargs)
        assert staged.credential_path is None
        assert not any(name.endswith(".credential.json") for name in staged.mailbox.entry_names())
        staged_mailboxes.append(staged)
        return staged

    original_serve = auto_research_stream_module._serve_worker_commands

    async def tracked_serve(*args, **kwargs):
        turn_id = str(kwargs["expected_turn_id"])
        started.append(turn_id)
        try:
            await original_serve(*args, **kwargs)
        finally:
            finished.append(turn_id)

    def fail_prompt(*_args, **_kwargs):
        raise ValueError("prompt failed after command mailbox staging")

    monkeypatch.setattr(auto_research_stream_module, "stage_command_mailbox", capture_mailbox)
    monkeypatch.setattr(auto_research_stream_module, "_serve_worker_commands", tracked_serve)
    monkeypatch.setattr(auto_research_stream_module, "_worker_prompt", fail_prompt)

    events = await _events(
        stream_auto_research_worker_run(
            service,
            _WorkerLauncher(),
            AutoResearchRunRequest.model_validate(worker.request),
            tmp_path / "data",
            _execution(store, worker),
            command_dispatcher=_dispatcher(store),
        )
    )

    assert events[-1].event == "error"
    assert events[-1].text == "prompt failed after command mailbox staging"
    assert started == finished == [f"{worker.operation_id}:worker"]
    assert len(staged_mailboxes) == 1
    staged = staged_mailboxes[0]
    assert staged.credential_path is None
    assert staged.credential.expired
    assert not any(name.endswith(".credential.json") for name in staged.mailbox.entry_names())
    assert not any(
        name.startswith(("rcp-command-", ".rcp-command-", ".rcp-mailbox-"))
        for name in staged.mailbox.entry_names()
    )


@pytest.mark.asyncio
async def test_main_mailbox_preserves_prompt_failure_over_server_and_cleanup_failures(
    manifest, tmp_path, monkeypatch
) -> None:
    service = _service(manifest, tmp_path)
    store, _auto_research, _root, worker = _setup_auto_research(tmp_path)
    staged_mailboxes: list[StagedCommandMailbox] = []
    original_cleanup = StagedCommandMailbox.cleanup

    def capture_mailbox(**kwargs):
        staged = _stage_command_mailbox(**kwargs)
        staged_mailboxes.append(staged)
        return staged

    async def fail_serve(*_args, **_kwargs):
        raise RuntimeError("mailbox server also failed")

    def fail_cleanup(staged):
        original_cleanup(staged)
        raise RuntimeError("mailbox cleanup also failed")

    def fail_prompt(*_args, **_kwargs):
        raise ValueError("primary prompt failure")

    monkeypatch.setattr(auto_research_stream_module, "stage_command_mailbox", capture_mailbox)
    monkeypatch.setattr(auto_research_stream_module, "_serve_worker_commands", fail_serve)
    monkeypatch.setattr(auto_research_stream_module, "_worker_prompt", fail_prompt)
    monkeypatch.setattr(StagedCommandMailbox, "cleanup", fail_cleanup)

    events = await _events(
        stream_auto_research_worker_run(
            service,
            _WorkerLauncher(),
            AutoResearchRunRequest.model_validate(worker.request),
            tmp_path / "data",
            _execution(store, worker),
            command_dispatcher=_dispatcher(store),
        )
    )

    assert events[-1].event == "error"
    assert events[-1].text == "primary prompt failure"
    warnings = [event.message for event in store.agent_task_events(worker.operation_id)]
    assert any("mailbox server also failed" in message for message in warnings)
    assert any("mailbox cleanup also failed" in message for message in warnings)
    assert len(staged_mailboxes) == 1
    assert staged_mailboxes[0].credential_path is None
    assert staged_mailboxes[0].credential.expired


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ["expected", "unexpected", "cancelled"])
async def test_close_mailbox_cleans_after_every_serve_task_failure(tmp_path, failure_kind) -> None:
    store, auto_research, _root, worker = _setup_auto_research(tmp_path)
    stage = tmp_path / "mailbox" / failure_kind
    stage.mkdir(parents=True)
    staged = _stage_command_mailbox(
        local_stage=stage,
        remote_stage=None,
        episode_id=auto_research.episode_id,
        task_id=worker.operation_id,
        turn_id=f"{worker.operation_id}:{failure_kind}",
    )
    stop = asyncio.Event()

    async def fail_serve() -> None:
        if failure_kind == "expected":
            raise OSError("expected mailbox transport failure")
        if failure_kind == "unexpected":
            raise RuntimeError("unexpected mailbox server failure")
        raise asyncio.CancelledError

    task = asyncio.create_task(fail_serve())
    if failure_kind == "expected":
        await _close_worker_mailbox(
            staged,
            stop=stop,
            task=task,
            execution=_execution(store, worker),
        )
    else:
        error_type = RuntimeError if failure_kind == "unexpected" else asyncio.CancelledError
        with pytest.raises(error_type):
            await _close_worker_mailbox(
                staged,
                stop=stop,
                task=task,
                execution=_execution(store, worker),
            )

    assert stop.is_set()
    assert staged.credential_path is None
    assert staged.credential.expired
    assert not any(name.endswith(".credential.json") for name in staged.mailbox.entry_names())
    assert not any(
        name.startswith(("rcp-command-", ".rcp-command-", ".rcp-mailbox-"))
        for name in staged.mailbox.entry_names()
    )


@pytest.mark.asyncio
async def test_close_mailbox_finishes_shielded_cleanup_before_reraising_cancellation(
    tmp_path, monkeypatch
) -> None:
    store, auto_research, _root, worker = _setup_auto_research(tmp_path)
    stage = tmp_path / "mailbox" / "shielded-cleanup"
    stage.mkdir(parents=True)
    staged = _stage_command_mailbox(
        local_stage=stage,
        remote_stage=None,
        episode_id=auto_research.episode_id,
        task_id=worker.operation_id,
        turn_id=f"{worker.operation_id}:shielded-cleanup",
    )
    stop = asyncio.Event()
    cleanup_started = threading.Event()
    cleanup_release = threading.Event()
    cleanup_finished = threading.Event()
    original_cleanup = StagedCommandMailbox.cleanup

    def blocking_failed_cleanup(current):
        cleanup_started.set()
        if not cleanup_release.wait(timeout=5):
            raise AssertionError("test did not release mailbox cleanup")
        original_cleanup(current)
        cleanup_finished.set()
        raise RuntimeError("secondary cleanup failure")

    async def serve_until_stopped() -> None:
        await stop.wait()

    monkeypatch.setattr(StagedCommandMailbox, "cleanup", blocking_failed_cleanup)
    serve_task = asyncio.create_task(serve_until_stopped())
    close_task = asyncio.create_task(
        _close_worker_mailbox(
            staged,
            stop=stop,
            task=serve_task,
            execution=_execution(store, worker),
        )
    )
    assert await asyncio.to_thread(cleanup_started.wait, 2)
    close_task.cancel()
    await asyncio.sleep(0)
    assert not close_task.done()
    cleanup_release.set()

    with pytest.raises(asyncio.CancelledError):
        await close_task

    assert cleanup_finished.is_set()
    assert staged.credential_path is None
    assert staged.credential.expired
    warnings = [event.message for event in store.agent_task_events(worker.operation_id)]
    assert any("secondary cleanup failure" in message for message in warnings)


@pytest.mark.asyncio
async def test_fresh_turn_fences_clear_and_recovery_preserves_only_authorized_handoffs(
    manifest, tmp_path
) -> None:
    service = _service(manifest, tmp_path)
    store, auto_research, _root, worker = _setup_auto_research(tmp_path / "fresh")
    stage = (
        tmp_path
        / "fresh-data"
        / "run-stage"
        / _worker_stage_name(worker.project_id, worker.operation_id)
    )
    stage.mkdir(parents=True)
    for name in ("patch.json", "watch.json", "messages.json"):
        (stage / name).write_text("stale", encoding="utf-8")

    def fresh_writer(_contract_text, workspace):
        assert not any(
            (workspace / name).exists() for name in ("patch.json", "watch.json", "messages.json")
        )

    fresh_events = await _events(
        stream_auto_research_worker_run(
            service,
            _WorkerLauncher(writer=fresh_writer),
            AutoResearchRunRequest.model_validate(worker.request),
            tmp_path / "fresh-data",
            _execution(store, worker),
            command_dispatcher=_dispatcher(store),
        )
    )
    assert fresh_events[-1].event == "done"
    assert [
        receipt.category
        for receipt in store.agent_task_receipts(worker.operation_id)
        if receipt.category == _HANDOFFS_CLEARED_RECEIPT
    ] == [_HANDOFFS_CLEARED_RECEIPT]
    assert store.auto_research_handoffs_cleared(worker.operation_id) is True

    retry_root = tmp_path / "retry-data" / "run-stage" / _worker_stage_name("project", "worker")
    retry_root.mkdir(parents=True)
    retry_store, retry_auto_research, _retry_orchestrator, retry_worker = _setup_auto_research(
        tmp_path / "retry",
        worker_status="failed",
        native_session_id="worker-session",
        stage_root=str(retry_root),
    )
    _record_original_contract(retry_store, retry_worker, retry_root)
    retry_store.mark_auto_research_handoffs_cleared(retry_worker.operation_id)
    retry_store.record_agent_task_receipt(
        retry_worker.operation_id,
        _HANDOFFS_CLEARED_RECEIPT,
        {"version": 1, "files": ["patch.json", "watch.json", "messages.json"]},
    )
    for index in range(AGENT_TASK_RECEIPT_RETENTION_COUNTS["summary"] + 1):
        retry_store.record_agent_task_receipt(
            retry_worker.operation_id,
            f"retention-pressure-{index}",
            {"index": index},
        )
    assert not any(
        receipt.category == _HANDOFFS_CLEARED_RECEIPT
        for receipt in retry_store.agent_task_receipts(retry_worker.operation_id)
    )
    retained_patch = json.dumps(
        {"summary": "No change", "repositories_read": [], "change_summary": [], "ops": []}
    )
    retained = {
        "patch.json": retained_patch,
        "watch.json": "retained watcher",
        "messages.json": "retained messages",
    }
    for name, content in retained.items():
        (retry_root / name).write_text(content, encoding="utf-8")
    retry = _recovery_task(retry_store, retry_auto_research, retry_worker)

    def retry_writer(contract_text, workspace):
        assert (workspace / "patch.json").read_text(encoding="utf-8") == retained_patch
        assert (workspace / "watch.json").read_text(encoding="utf-8") == retained["watch.json"]
        assert not (workspace / "messages.json").exists()
        assert str(workspace / "messages.json") not in contract_text

    retry_events = await _events(
        stream_auto_research_worker_run(
            service,
            _WorkerLauncher(writer=retry_writer),
            AutoResearchRunRequest.model_validate(retry.request),
            tmp_path / "retry-data",
            _execution(retry_store, retry, continuation="retry"),
            command_dispatcher=_dispatcher(retry_store),
        )
    )
    assert retry_events[-1].event == "done"
    assert not (retry_root / "messages.json").exists()


@pytest.mark.asyncio
async def test_recovery_repeats_fail_closed_clear_when_interruption_left_no_fence(
    manifest, tmp_path, monkeypatch
) -> None:
    service = _service(manifest, tmp_path)
    stage = tmp_path / "data" / "run-stage" / _worker_stage_name("project", "worker")
    stage.mkdir(parents=True)
    store, auto_research, _root, worker = _setup_auto_research(
        tmp_path,
        worker_status="failed",
        native_session_id="worker-session",
        stage_root=str(stage),
    )
    _record_original_contract(store, worker, stage)
    for name in ("patch.json", "watch.json", "messages.json"):
        (stage / name).write_text("stale", encoding="utf-8")

    original_clear = auto_research_stream_module._clear_stale_turn_handoffs

    def interrupted_clear(workspace, _remote) -> None:
        (workspace / "patch.json").unlink()
        raise ValueError("interrupted before handoff clear completed")

    monkeypatch.setattr(
        auto_research_stream_module,
        "_clear_stale_turn_handoffs",
        interrupted_clear,
    )
    interrupted = await _events(
        stream_auto_research_worker_run(
            service,
            _WorkerLauncher(session_id="worker-session"),
            AutoResearchRunRequest.model_validate(worker.request),
            tmp_path / "data",
            _execution(store, worker),
            command_dispatcher=_dispatcher(store),
        )
    )
    assert interrupted[-1].event == "error"
    assert "interrupted before handoff clear completed" in interrupted[-1].text
    assert store.auto_research_handoffs_cleared(worker.operation_id) is False
    assert not any(
        receipt.category == _HANDOFFS_CLEARED_RECEIPT
        for receipt in store.agent_task_receipts(worker.operation_id)
    )

    monkeypatch.setattr(
        auto_research_stream_module,
        "_clear_stale_turn_handoffs",
        original_clear,
    )
    retry = _recovery_task(store, auto_research, worker)

    def retry_writer(_contract_text, workspace):
        assert not any(
            (workspace / name).exists() for name in ("patch.json", "watch.json", "messages.json")
        )

    recovered = await _events(
        stream_auto_research_worker_run(
            service,
            _WorkerLauncher(session_id="worker-session", writer=retry_writer),
            AutoResearchRunRequest.model_validate(retry.request),
            tmp_path / "data",
            _execution(store, retry, continuation="retry"),
            command_dispatcher=_dispatcher(store),
        )
    )
    assert recovered[-1].event == "done"
    assert store.auto_research_handoffs_cleared(worker.operation_id) is True
    assert any(
        receipt.category == _HANDOFFS_CLEARED_RECEIPT
        for receipt in store.agent_task_receipts(worker.operation_id)
    )


@pytest.mark.asyncio
async def test_worker_continuation_replaces_original_repository_pointers(
    manifest, tmp_path
) -> None:
    service = _service(manifest, tmp_path)
    stage = tmp_path / "data" / "run-stage" / _worker_stage_name("project", "worker")
    stage.mkdir(parents=True)
    store, auto_research, _root, worker = _setup_auto_research(
        tmp_path,
        worker_status="failed",
        native_session_id="worker-session",
        stage_root=str(stage),
    )
    _record_original_contract(
        store,
        worker,
        stage,
        content=(
            "# original worker contract\n- repo-a: host=`retired.example` path=`/retired/repo-a`\n"
        ),
    )
    retry = _recovery_task(store, auto_research, worker)
    current_path = manifest.repository_map["repo-a"].path

    def writer(contract_text, _workspace):
        assert "These replace every repository pointer in the original contract" in contract_text
        # A local repository says so, rather than rendering an empty host.
        assert f"- repo-a: path=`{current_path}` on this machine" in contract_text
        assert "retired.example" not in contract_text
        assert "/retired/repo-a" not in contract_text

    events = await _events(
        stream_auto_research_worker_run(
            service,
            _WorkerLauncher(session_id="worker-session", writer=writer),
            AutoResearchRunRequest.model_validate(retry.request),
            tmp_path / "data",
            _execution(store, retry, continuation="retry"),
            command_dispatcher=_dispatcher(store),
        )
    )
    assert events[-1].event == "done"


@pytest.mark.asyncio
async def test_claimed_mail_is_staged_exactly_for_worker_wake(manifest, tmp_path) -> None:
    service = _service(manifest, tmp_path)
    stage = tmp_path / "data" / "run-stage" / _worker_stage_name("project", "worker")
    stage.mkdir(parents=True)
    store, auto_research, root, worker = _setup_auto_research(
        tmp_path,
        worker_status="succeeded",
        native_session_id="worker-session",
        stage_root=str(stage),
    )
    _record_original_contract(store, worker, stage)
    message = record_auto_research_message(
        store,
        episode_id=auto_research.episode_id,
        sender_role="orchestrator",
        sender_task_id=root.operation_id,
        authorized_by=None,
        recipient_task_id=worker.operation_id,
        control_node_id="blk/check-result",
        body="The graph moved; re-check the discrepancy.",
    )
    request = AutoResearchRunRequest.model_validate(worker.request).model_copy(
        update={"instruction": None, "wake_cause": "message"}
    )
    now = store.now()
    wake = store.create_auto_research_message_wake_task(
        AgentTaskRecord(
            operation_id="mail-wake",
            project_id=worker.project_id,
            episode_id=auto_research.episode_id,
            graph_target=auto_research.graph_target,
            kind="auto_research",
            status="running",
            request=request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="mail wake running",
            parent_operation_id=worker.operation_id,
            native_session_id=worker.native_session_id,
            stage_root=worker.stage_root,
            authorized_by=auto_research.authorized_by,
            dispatch_authority=worker.dispatch_authority,
        ),
        role="worker",
        recipient_task_id=worker.operation_id,
        message_ids=[message.message_id],
    )
    assert wake is not None

    def writer(contract_text, workspace):
        delivery = parse_auto_research_mail_delivery(
            (workspace / "messages.json").read_text(encoding="utf-8")
        )
        assert delivery.delivery_operation_id == wake.operation_id
        assert delivery.message_ids == [message.message_id]
        assert delivery.messages[0].body == message.body
        assert delivery.graph_authority == "none"
        assert str(workspace / "messages.json") in contract_text

    events = await _events(
        stream_auto_research_worker_run(
            service,
            _WorkerLauncher(session_id="worker-session", writer=writer),
            AutoResearchRunRequest.model_validate(wake.request),
            tmp_path / "data",
            _execution(store, wake, continuation="message_wake"),
            command_dispatcher=_dispatcher(store),
        )
    )
    assert events[-1].event == "done"
    expected_handoff = (stage / "messages.json").read_text(encoding="utf-8")
    store.fail_agent_task(wake.operation_id, "provider connection interrupted")
    (stage / "messages.json").unlink()
    retry = _recovery_task(
        store,
        auto_research,
        wake,
        operation_id="mail-wake-retry",
    )

    def recovery_writer(contract_text, workspace):
        assert (workspace / "messages.json").read_text(encoding="utf-8") == expected_handoff
        assert str(workspace / "messages.json") in contract_text

    recovery_events = await _events(
        stream_auto_research_worker_run(
            service,
            _WorkerLauncher(session_id="worker-session", writer=recovery_writer),
            AutoResearchRunRequest.model_validate(retry.request),
            tmp_path / "data",
            _execution(store, retry, continuation="retry"),
            command_dispatcher=_dispatcher(store),
        )
    )
    assert recovery_events[-1].event == "done"


@pytest.mark.asyncio
async def test_claimed_lifecycle_and_mail_are_separate_and_reused_on_exact_recovery(
    manifest,
    tmp_path,
) -> None:
    service = _service(manifest, tmp_path)
    store, auto_research, root, _worker = _setup_auto_research(tmp_path / "store")
    data_dir = tmp_path / "data"
    fresh_events = await _events(
        stream_auto_research_orchestrator_run(
            service,
            _WorkerLauncher(session_id="orchestrator-session"),
            AutoResearchRunRequest.model_validate(root.request),
            data_dir,
            _execution(store, root),
            command_dispatcher=_dispatcher(store),
        )
    )
    assert fresh_events[-1].event == "done"
    stage = data_dir / "run-stage" / _orchestrator_stage_name("project", root.operation_id)
    root = store.agent_task(root.operation_id)
    assert root is not None
    store.checkpoint_agent_task(root.operation_id, native_session_id="orchestrator-session")
    root = store.agent_task(root.operation_id)
    assert root is not None
    assert root.native_session_id == "orchestrator-session"
    assert root.stage_root == str(stage)
    notice = store.record_auto_research_lifecycle_notice(
        AutoResearchLifecycleNoticeRecord(
            notice_id="worker-settled",
            episode_id=auto_research.episode_id,
            source_kind="worker",
            source_id="worker",
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
        body="This remains hearsay beside the lifecycle fact.",
    )
    request = AutoResearchRunRequest.model_validate(root.request).model_copy(
        update={
            "actor_operation_id": root.operation_id,
            "session_id": root.native_session_id,
            "instruction": None,
            "wake_cause": "lifecycle",
        }
    )
    now = store.now()
    wake = store.create_auto_research_lifecycle_wake_task(
        AgentTaskRecord(
            operation_id="lifecycle-wake",
            project_id=root.project_id,
            episode_id=auto_research.episode_id,
            graph_target=auto_research.graph_target,
            kind="auto_research",
            status="running",
            request=request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="lifecycle wake running",
            parent_operation_id=root.operation_id,
            native_session_id=root.native_session_id,
            stage_host=root.stage_host,
            stage_root=root.stage_root,
            authorized_by=auto_research.authorized_by,
            dispatch_authority=root.dispatch_authority,
        ),
        lifecycle_notice_ids=[notice.notice_id],
        message_ids=[message.message_id],
    )
    assert wake is not None

    def writer(_contract_text, workspace):
        lifecycle = parse_auto_research_lifecycle_delivery(
            (workspace / "lifecycle.json").read_text(encoding="utf-8")
        )
        mail = parse_auto_research_mail_delivery(
            (workspace / "messages.json").read_text(encoding="utf-8")
        )
        assert lifecycle.delivery_operation_id == wake.operation_id
        assert [item.notice_id for item in lifecycle.notices] == [notice.notice_id]
        assert lifecycle.authority == "rcp_lifecycle"
        assert lifecycle.graph_authority == "none"
        assert mail.delivery_operation_id == wake.operation_id
        assert mail.message_ids == [message.message_id]
        assert mail.epistemic_status == "hearsay"

    events = await _events(
        stream_auto_research_orchestrator_run(
            service,
            _WorkerLauncher(session_id="orchestrator-session", writer=writer),
            AutoResearchRunRequest.model_validate(wake.request),
            data_dir,
            _execution(store, wake, continuation="lifecycle_wake"),
            command_dispatcher=_dispatcher(store),
        )
    )
    assert events[-1].event == "done"
    retained_lifecycle = (stage / "lifecycle.json").read_text(encoding="utf-8")
    retained_mail = (stage / "messages.json").read_text(encoding="utf-8")
    store.fail_agent_task(wake.operation_id, "provider connection interrupted")
    retry = _recovery_task(
        store,
        auto_research,
        wake,
        operation_id="lifecycle-wake-retry",
    )

    def recovery_writer(_contract_text, workspace):
        assert (workspace / "lifecycle.json").read_text(encoding="utf-8") == (retained_lifecycle)
        assert (workspace / "messages.json").read_text(encoding="utf-8") == retained_mail

    recovery_events = await _events(
        stream_auto_research_orchestrator_run(
            service,
            _WorkerLauncher(session_id="orchestrator-session", writer=recovery_writer),
            AutoResearchRunRequest.model_validate(retry.request),
            data_dir,
            _execution(store, retry, continuation="retry"),
            command_dispatcher=_dispatcher(store),
        )
    )
    assert recovery_events[-1].event == "done"


@pytest.mark.parametrize(
    ("retained_kind", "error_match"),
    [
        ("valid", None),
        ("mismatch", "differs from its durable claimed batch"),
        ("oversize", "exceeds .* bytes"),
        ("non_regular", "not a regular file"),
    ],
)
def test_retained_claimed_mail_is_validated_directly_before_recovery(
    tmp_path,
    retained_kind,
    error_match,
) -> None:
    execution, turn, stage, delivery = _claimed_mail_recovery(tmp_path)
    retained_path = stage.workspace / "messages.json"
    if retained_kind == "valid":
        retained_path.write_text(delivery.model_dump_json() + "\n", encoding="utf-8")
    elif retained_kind == "mismatch":
        payload = delivery.model_dump(mode="json")
        payload["messages"][0]["body"] = "Different retained hearsay."
        retained_path.write_text(json.dumps(payload), encoding="utf-8")
    elif retained_kind == "oversize":
        retained_path.write_text(
            " " * (auto_research_mail_module.AUTO_RESEARCH_MAIL_MAX_BYTES + 1),
            encoding="utf-8",
        )
    else:
        retained_path.mkdir()

    if error_match is None:
        assert auto_research_stream_module._stage_claimed_mail(execution, turn, stage) == str(
            retained_path
        )
        assert (
            parse_auto_research_mail_delivery(retained_path.read_text(encoding="utf-8")) == delivery
        )
    else:
        with pytest.raises(ValueError, match=error_match):
            auto_research_stream_module._stage_claimed_mail(execution, turn, stage)


@pytest.mark.asyncio
async def test_worker_on_another_machine_gets_staged_current_graph_and_resolved_host(
    manifest, tmp_path, monkeypatch
) -> None:
    remote_manifest_path = tmp_path / "remote-manifest" / ".research" / "manifest.toml"
    remote_manifest_path.parent.mkdir(parents=True)
    manifest_text = manifest.path.read_text(encoding="utf-8")
    manifest_text = manifest_text.replace('host = ""', 'host = "state.example"', 1)
    manifest_text = manifest_text.replace(
        "[[repositories]]",
        '[[machines]]\nalias = "worker-host"\nhost = "worker.example"\n\n[[repositories]]',
        1,
    )
    remote_manifest_path.write_text(manifest_text, encoding="utf-8")
    service = _service(load_manifest(remote_manifest_path), tmp_path)
    _LocalBackedRemoteStage.base = tmp_path / "remote"
    monkeypatch.setattr(
        "rcp.runs.tasks.auto_research_stream.RemoteRunStage",
        _LocalBackedRemoteStage,
    )
    store, _auto_research, _root, worker = _setup_auto_research(
        tmp_path / "store",
        run_on="worker-host",
    )
    observed: dict[str, object] = {}

    class Launcher(_WorkerLauncher):
        async def stream(self, provider, prompt, **kwargs):
            observed["host"] = kwargs["host"]
            observed["capability"] = kwargs["capability"]
            contract = _contract(prompt)
            graph_match = re.search(r"- graph: `([^`]+)`", contract)
            research_match = re.search(r"- research rendering: `([^`]+)`", contract)
            assert graph_match is not None and research_match is not None
            graph = json.loads(Path(graph_match.group(1)).read_text(encoding="utf-8"))
            observed["revision"] = graph["revision"]
            observed["seat"] = graph["nodes"]["blk/check-result"]["type"]
            observed["research"] = Path(research_match.group(1)).read_text(encoding="utf-8")
            yield AgentEvent(event="session", session_id="remote-worker-session")
            yield AgentEvent(event="answer", text="Remote worker read current project state.")
            yield AgentEvent(event="done")

    events = await _events(
        stream_auto_research_worker_run(
            service,
            Launcher(),
            AutoResearchRunRequest.model_validate(worker.request),
            tmp_path / "data",
            _execution(store, worker),
            command_dispatcher=_dispatcher(store),
        )
    )

    assert events[-1].event == "done", [(event.event, event.text) for event in events]
    assert observed == {
        "host": "worker.example",
        "capability": "work_auto",
        "revision": 1,
        "seat": "blocker",
        "research": service.history.root.joinpath("research.md").read_text(encoding="utf-8"),
    }
    stored = store.agent_task(worker.operation_id)
    assert stored is not None
    assert stored.stage_host == "worker.example"


@pytest.mark.asyncio
async def test_worker_reply_command_uses_one_auto_research_mailbox_and_stable_allocation_key(
    manifest, tmp_path, monkeypatch
) -> None:
    service = _service(manifest, tmp_path)
    stage = tmp_path / "data" / "run-stage" / _worker_stage_name("project", "worker")
    stage.mkdir(parents=True)
    store, auto_research, _root, worker = _setup_auto_research(
        tmp_path,
        worker_status="failed",
        native_session_id="worker-session",
        stage_root=str(stage),
    )
    _record_original_contract(store, worker, stage)
    retry = _recovery_task(store, auto_research, worker)
    replies: list[str] = []
    command_outputs: list[dict[str, object]] = []
    staged_turn_ids: list[str] = []

    def capture_mailbox(**kwargs):
        staged_turn_ids.append(str(kwargs["turn_id"]))
        return _stage_command_mailbox(**kwargs)

    monkeypatch.setattr(
        "rcp.runs.tasks.auto_research_stream.stage_command_mailbox",
        capture_mailbox,
    )

    async def writer(contract_text, _workspace):
        match = re.search(r"Reply command prefix: `([^`]+)`", contract_text)
        assert match is not None
        argv = [*shlex.split(match.group(1)), "Recovered worker result"]
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        assert process.returncode == 0, stderr.decode()
        command_outputs.append(json.loads(stdout))

    events = await _events(
        stream_auto_research_worker_run(
            service,
            _WorkerLauncher(session_id="worker-session", writer=writer),
            AutoResearchRunRequest.model_validate(retry.request),
            tmp_path / "data",
            _execution(store, retry, continuation="retry"),
            command_dispatcher=_dispatcher(store, replies),
        )
    )

    expected_digest = hashlib.sha256(
        b"auto_research-worker-reply\0auto_research\0worker"
    ).hexdigest()[:32]
    assert replies == ["Recovered worker result"]
    assert command_outputs[0]["status"] == "ok"
    assert staged_turn_ids == [f"{retry.operation_id}:worker"]
    assert events[-1].event == "done"
    invocation = store.agent_command_by_key(
        auto_research.episode_id,
        f"worker-reply-{expected_digest}",
    )
    assert invocation is not None
    assert invocation.operation_id == retry.operation_id


@pytest.mark.asyncio
async def test_patch_correction_uses_fresh_validate_only_auto_research_gate(
    manifest, tmp_path, monkeypatch
) -> None:
    service = _service(manifest, tmp_path)
    stage = tmp_path / "data" / "run-stage" / _worker_stage_name("project", "worker")
    stage.mkdir(parents=True)
    store, auto_research, _root, worker = _setup_auto_research(
        tmp_path,
        worker_status="failed",
        native_session_id="worker-session",
        stage_root=str(stage),
    )
    _record_original_contract(store, worker, stage)
    retry = _recovery_task(store, auto_research, worker)
    replies: list[str] = []
    command_results: list[tuple[int, dict[str, object]]] = []
    valid_empty_patch = json.dumps(
        {
            "summary": "No graph change after correction.",
            "repositories_read": [],
            "change_summary": [],
            "ops": [],
        }
    )
    launch_count = 0

    async def writer(contract_text, workspace):
        nonlocal launch_count
        launch_count += 1
        if launch_count == 1:
            (workspace / "patch.json").write_text(
                '{"summary":"broken","ops":"not-a-list"}',
                encoding="utf-8",
            )
            return
        commands = re.findall(r"`([^`]*\svalidate\s[^`]*)`", contract_text)
        assert commands
        validate_argv = shlex.split(commands[-1])
        assert validate_argv[-2:] == ["validate", str(workspace / "patch.json")]
        prefix = validate_argv[:-2]
        message = await asyncio.create_subprocess_exec(
            *prefix,
            "message",
            "--key",
            "correction-must-not-message",
            "Forbidden repeated reply",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        message_stdout, message_stderr = await message.communicate()
        assert message.returncode == 1, message_stderr.decode()
        command_results.append((message.returncode, json.loads(message_stdout)))

        validation = await asyncio.create_subprocess_exec(
            *validate_argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        validation_stdout, validation_stderr = await validation.communicate()
        assert validation.returncode == 0, validation_stdout.decode() + validation_stderr.decode()
        command_results.append((validation.returncode, json.loads(validation_stdout)))
        (workspace / "patch.json").write_text(valid_empty_patch, encoding="utf-8")

    served_gates = _capture_served_invocation_gates(monkeypatch)
    launcher = _WorkerLauncher(session_id="worker-session", writer=writer)
    events = await _events(
        stream_auto_research_worker_run(
            service,
            launcher,
            AutoResearchRunRequest.model_validate(retry.request),
            tmp_path / "data",
            _execution(store, retry, continuation="retry"),
            command_dispatcher=_dispatcher(store, replies),
        )
    )

    assert launcher.requested_session_ids == ["worker-session", "worker-session"]
    _assert_fresh_matching_invocation_gates(served_gates, launcher.invocation_gates)
    assert replies == []
    assert command_results[0][1]["status"] == "invalid"
    assert "validation only" in str(command_results[0][1]["message"])
    assert command_results[1][1]["status"] == "valid"
    rejected = store.agent_command_by_key(
        "auto_research",
        "correction-must-not-message",
    )
    assert rejected is None
    with store.connection() as connection:
        validate_phases = connection.execute(
            """
            SELECT command_phase FROM graph_run_events
            WHERE operation_id = ? AND event_kind = 'command' AND command_verb = 'validate'
            ORDER BY event_id
            """,
            (retry.operation_id,),
        ).fetchall()
    assert [row["command_phase"] for row in validate_phases] == ["start", "exit"]
    assert events[-1].event == "done"


@pytest.mark.asyncio
async def test_patch_correction_mailbox_closes_when_post_stage_receipt_fails(
    manifest, tmp_path, monkeypatch
) -> None:
    service = _service(manifest, tmp_path)
    stage = tmp_path / "data" / "run-stage" / _worker_stage_name("project", "worker")
    stage.mkdir(parents=True)
    store, auto_research, _root, worker = _setup_auto_research(
        tmp_path,
        worker_status="failed",
        native_session_id="worker-session",
        stage_root=str(stage),
    )
    _record_original_contract(store, worker, stage)
    retry = _recovery_task(store, auto_research, worker)
    staged_mailboxes = []
    started: list[str] = []
    finished: list[str] = []

    def capture_mailbox(**kwargs):
        staged = _stage_command_mailbox(**kwargs)
        staged_mailboxes.append(staged)
        return staged

    original_serve = auto_research_stream_module._serve_worker_commands

    async def tracked_serve(*args, **kwargs):
        turn_id = str(kwargs["expected_turn_id"])
        started.append(turn_id)
        try:
            await original_serve(*args, **kwargs)
        finally:
            finished.append(turn_id)

    original_receipt = auto_research_stream_module._record_agent_launch_receipt

    def fail_correction_receipt(*args, **kwargs):
        if kwargs.get("continuation") == "graph_correction":
            raise ValueError("correction receipt failed after mailbox staging")
        return original_receipt(*args, **kwargs)

    def writer(_contract_text, workspace):
        (workspace / "patch.json").write_text(
            '{"summary":"broken","ops":"not-a-list"}',
            encoding="utf-8",
        )

    monkeypatch.setattr(auto_research_stream_module, "stage_command_mailbox", capture_mailbox)
    monkeypatch.setattr(auto_research_stream_module, "_serve_worker_commands", tracked_serve)
    monkeypatch.setattr(
        auto_research_stream_module,
        "_record_agent_launch_receipt",
        fail_correction_receipt,
    )
    events = await _events(
        stream_auto_research_worker_run(
            service,
            _WorkerLauncher(session_id="worker-session", writer=writer),
            AutoResearchRunRequest.model_validate(retry.request),
            tmp_path / "data",
            _execution(store, retry, continuation="retry"),
            command_dispatcher=_dispatcher(store),
        )
    )

    assert events[-1].event == "error"
    assert events[-1].text == "correction receipt failed after mailbox staging"
    assert (
        started
        == finished
        == [
            f"{retry.operation_id}:worker",
            f"{retry.operation_id}:worker-patch-correction:1",
        ]
    )
    assert len(staged_mailboxes) == 2
    for staged in staged_mailboxes:
        assert staged.credential_path is None
        assert staged.credential.expired
        assert not any(name.endswith(".credential.json") for name in staged.mailbox.entry_names())


@pytest.mark.asyncio
async def test_patch_correction_setup_failure_survives_secondary_cleanup_failure(
    manifest, tmp_path, monkeypatch
) -> None:
    service = _service(manifest, tmp_path)
    stage = tmp_path / "data" / "run-stage" / _worker_stage_name("project", "worker")
    stage.mkdir(parents=True)
    store, auto_research, _root, worker = _setup_auto_research(
        tmp_path,
        worker_status="failed",
        native_session_id="worker-session",
        stage_root=str(stage),
    )
    _record_original_contract(store, worker, stage)
    retry = _recovery_task(store, auto_research, worker)
    staged_mailboxes: list[StagedCommandMailbox] = []
    original_cleanup = StagedCommandMailbox.cleanup

    def capture_mailbox(**kwargs):
        staged = _stage_command_mailbox(**kwargs)
        staged_mailboxes.append(staged)
        return staged

    def fail_correction_dispatcher(*_args, **_kwargs):
        raise RuntimeError("primary correction setup failure")

    def fail_correction_cleanup(staged):
        original_cleanup(staged)
        if "worker-patch-correction" in staged.credential.identity.turn_id:
            raise RuntimeError("secondary correction cleanup failure")

    def writer(_contract_text, workspace):
        (workspace / "patch.json").write_text(
            '{"summary":"broken","ops":"not-a-list"}',
            encoding="utf-8",
        )

    monkeypatch.setattr(auto_research_stream_module, "stage_command_mailbox", capture_mailbox)
    monkeypatch.setattr(
        auto_research_stream_module,
        "_ValidateOnlyAutoResearchCommandDispatcher",
        fail_correction_dispatcher,
    )
    monkeypatch.setattr(StagedCommandMailbox, "cleanup", fail_correction_cleanup)

    with pytest.raises(RuntimeError, match="primary correction setup failure"):
        await _events(
            stream_auto_research_worker_run(
                service,
                _WorkerLauncher(session_id="worker-session", writer=writer),
                AutoResearchRunRequest.model_validate(retry.request),
                tmp_path / "data",
                _execution(store, retry, continuation="retry"),
                command_dispatcher=_dispatcher(store),
            )
        )

    assert len(staged_mailboxes) == 2
    assert all(staged.credential_path is None for staged in staged_mailboxes)
    assert all(staged.credential.expired for staged in staged_mailboxes)
    warnings = [event.message for event in store.agent_task_events(retry.operation_id)]
    assert any("secondary correction cleanup failure" in message for message in warnings)


@pytest.mark.asyncio
async def test_worker_patch_applies_with_ordinary_attribution_after_stop_intent(
    manifest, tmp_path
) -> None:
    service = _service(manifest, tmp_path)
    store, auto_research, _root, worker = _setup_auto_research(tmp_path)
    request_auto_research_stop(store, auto_research.episode_id)
    patch = Patch(
        kind="work",
        author="agent",
        summary="Recorded the worker's concrete blocker.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": "blk/worker-followup",
                        "type": "blocker",
                        "title": "Worker follow-up",
                        "description": "The measured discrepancy still needs a second run.",
                        "status": "open",
                    }
                ],
            }
        ],
        change_summary=["Recorded the worker follow-up blocker."],
    )

    def writer(_contract_text, workspace):
        (workspace / "patch.json").write_text(agent_patch_json(patch), encoding="utf-8")

    events = await _events(
        stream_auto_research_worker_run(
            service,
            _WorkerLauncher(writer=writer),
            AutoResearchRunRequest.model_validate(worker.request),
            tmp_path / "data",
            _execution(store, worker),
            command_dispatcher=_dispatcher(store),
        )
    )

    graph_messages = [event for event in events if event.event == "message"]
    assert graph_messages
    update = json.loads(graph_messages[-1].text)["graph_update"]
    assert update["status"] == "applied"
    assert service.history.state().nodes["blk/worker-followup"].type == "blocker"
    applied = service.history.load_patches()[-1]
    assert applied.kind == "work"
    assert applied.author == "agent"
    assert applied.source_operation_id == worker.operation_id
    assert store.episode(auto_research.episode_id).stop_requested_at is not None


def test_orchestrator_receives_the_project_settings_packages(manifest) -> None:
    """The orchestrator is a Work agent and gets Settings packages like any other.

    Only the auto_research report is restricted to one required skill; the orchestrator
    was previously given nothing at all.
    """

    pointers = [
        {
            "label": "Graph audit",
            "kind": "skill",
            "id": "graph-audit",
            "version": "1.0.0",
            "description": "Check the graph before asserting new claims.",
            "path": "/stage/inputs/bundle/graph-audit",
        }
    ]
    common = dict(
        graph_path="/s/graph.json",
        research_path="/s/research.md",
        patch_path="/s/patch.json",
        output_schema_path="/s/schema.json",
        validator_command="/stage/rcp-agent validate",
        command_client="/stage/rcp-agent",
    )
    contract = auto_research_orchestrator_task_contract(
        project_name="project", repositories=[], skill_pointers=pointers, **common
    )
    continuation = auto_research_orchestrator_continuation_contract(
        original_contract_path="/s/original.md",
        mode="continuation",
        repositories=[],
        skill_pointers=pointers,
        **common,
    )
    for text in (contract, continuation):
        assert "Skills and workflows staged for this run:" in text
        assert "Graph audit (skill graph-audit v1.0.0)" in text
        assert "folder: /stage/inputs/bundle/graph-audit" in text

    # No packages means no heading, not an empty one.
    bare = auto_research_orchestrator_task_contract(
        project_name="project", repositories=[], skill_pointers=[], **common
    )
    assert "Skills and workflows" not in bare
