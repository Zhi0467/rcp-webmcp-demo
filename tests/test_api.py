from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import rcp.projects as projects_module
import rcp.runs.tasks.work as work_module
from rcp.agents import AgentEvent, AgentPatch, AgentProcessControl, PromptFactory, ProviderReadiness
from rcp.agents.context import RepositoryPointer
from rcp.api.app import (
    _generic_watcher_delivery_request,
)
from rcp.api.tasks import _validate_stored_task_request, _validated_task_request
from rcp.artifacts import AgentArtifactDescriptor
from rcp.background import AgentTaskExecution
from rcp.config import MachineConfig
from rcp.core.attention import decision_awaits_choice
from rcp.core.models import AuthorizedHuman, Blocker, Decision, GraphState, Patch
from rcp.core.validation.constants import NODE_ADAPTER
from rcp.history import HistoryManager, ReplayHalted
from rcp.limits import PATCH_CORRECTION_MAX_ROUNDS
from rcp.paper import WritingSession
from rcp.providers import ProviderSkill
from rcp.runs.chat import (
    _chat_stage_name,
    _discover_chat_artifacts,
    _prepare_local_artifact_directory,
    _project_write_scope,
)
from rcp.runs.experiment_loop import persist_experiment_watchers_idempotently
from rcp.runs.shared import (
    AgentOutputProblem,
    _collect_patch_text,
    _sse,
    _sweep_stale_stages,
)
from rcp.runs.tasks.coach import _paper_snapshot_path, stream_coach
from rcp.runs.tasks.discuss import stream_discuss_run
from rcp.runs.tasks.experiment_loop import stream_experiment_loop_task
from rcp.runs.tasks.graph import (
    _record_context_reuse,
    _record_progress_handoff,
    _stage_graph_context,
    stream_graph_run,
)
from rcp.runs.tasks.work import stream_work_run
from rcp.service import (
    CoachRequest,
    ProposalDecisionRequest,
    ReviewRequest,
    RunRequest,
    resolve_dispatch_authority,
)
from rcp.skill_registry import SkillDefaults, official_registry
from rcp.sources import project_cache_roots
from rcp.storage import (
    AgentTaskRecord,
    AppStore,
    GraphWatcherRecord,
    ProjectRecord,
    WatcherContinuation,
    WatcherRecord,
)
from rcp.transport import RunLockCancelled, RunLockLease, SSHStateWorkspace, StateUnavailable
from rcp.watchers import WatcherBinding, WatcherCheckResult, WatchSpec

from .helpers import (
    agent_patch_json,
    append_fixture_patch,
    create_named_app,
    gated_patch,
    refresh_patch,
    seed_patch,
    shape_invalid_patch,
)
from .helpers import wait_for_task_response as _wait_for_run


def _named_test_authorizer(store: AppStore) -> AuthorizedHuman:
    owner = store.local_owner
    assert owner is not None
    assert owner.display_name == "Test researcher"
    return AuthorizedHuman(
        space_id=store.space_id,
        user_id=owner.user_id,
        display_name=owner.display_name,
    )


def _persist_skill_defaults(service, defaults: SkillDefaults) -> None:
    surfaces = ("seed", "refresh", "node_chat", "project_chat", "paper_coach")
    service.history.update_agent_settings(
        service.manifest.agent.default_run_truth_scope,
        {surface: service.manifest.agent_profile(surface) for surface in surfaces},
        skill_defaults=defaults,
    )


def test_generic_watcher_wake_keeps_packages_available_without_reinvoking_them(
    tmp_path: Path,
) -> None:
    continuation = WatcherContinuation(
        provider="codex",
        run_on="laptop",
        run_truth_scope=["repo-a"],
        patch_kind="work",
        skill_ids=["experiment-causality"],
        invoked_skill_ids=["experiment-causality"],
    )
    watcher = WatcherRecord(
        watcher_id="generic-turn-local-invocation",
        project_id="project-turn-local-invocation",
        origin_operation_id="origin-turn-local-invocation",
        origin_task_kind="project_chat",
        chat_id="chat-turn-local-invocation",
        check_command="false",
        log_path=str(tmp_path / "detached.log"),
        cwd=str(tmp_path),
        continuation=continuation,
        status="completed",
        created_at="2026-08-08T00:00:00Z",
    )

    request = _generic_watcher_delivery_request([watcher])

    assert request.skill_ids == ["experiment-causality"]
    assert request.invoked_skill_ids == []
    assert request.invoked_workflow_ids == []
    assert request.invoked_provider_skill_names == []
    assert request.resolved_provider_skills == []

    experiment_watcher = watcher.model_copy(
        update={"continuation": continuation.model_copy(update={"patch_kind": "experiment_loop"})}
    )
    with pytest.raises(ValueError, match="cannot carry Experiment-loop authority"):
        _generic_watcher_delivery_request([experiment_watcher])


def test_generic_watcher_delivery_wakes_its_own_project_chat(
    manifest, tmp_path: Path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    store = app.state.background_tasks.store
    authorized_by = _named_test_authorizer(store)
    now = store.now()
    store.create_agent_task(
        AgentTaskRecord(
            operation_id="generic-project-chat-origin",
            project_id=project_id,
            kind="project_chat",
            status="succeeded",
            request={},
            created_at=now,
            updated_at=now,
            status_message="Generic watcher origin fixture.",
            authorized_by=authorized_by,
        )
    )
    watcher = WatcherRecord(
        watcher_id="generic-project-chat-wake",
        project_id=project_id,
        origin_operation_id="generic-project-chat-origin",
        origin_task_kind="project_chat",
        chat_id="generic-project-chat",
        check_command="true",
        log_path=str(tmp_path / "generic.log"),
        cwd=str(tmp_path),
        continuation=WatcherContinuation(
            provider="claude",
            run_on="laptop",
            run_truth_scope=["repo-a"],
            patch_kind="work",
        ),
        status="completed",
        created_at="2026-08-08T00:00:00Z",
    )
    store.create_watchers([watcher])
    captured: dict[str, object] = {}

    def capture(_tasks, project, kind, request, watcher_ids, **kwargs):
        captured.update(
            project_id=project,
            kind=kind,
            request=request,
            watcher_ids=watcher_ids,
            authorized_by=kwargs["authorized_by"],
        )

    monkeypatch.setattr("rcp.api.app.start_watcher_notification", capture)

    assert app.state.watcher_poller.on_completed is not None
    app.state.watcher_poller.on_completed([watcher])

    request = captured["request"]
    assert isinstance(request, RunRequest)
    assert captured["project_id"] == project_id
    assert captured["kind"] == "project_chat"
    assert captured["watcher_ids"] == [watcher.watcher_id]
    assert captured["authorized_by"] == authorized_by
    assert request.chat_scope == "project"
    assert request.chat_id == watcher.chat_id
    assert request.node_id is None


class FakeLauncher:
    def __init__(self, events: list[AgentEvent]) -> None:
        self.events = events
        self.calls = 0
        self.last_args = ()
        self.last_kwargs = {}

    async def stream(self, *args, **kwargs):
        self.calls += 1
        self.last_args = args
        self.last_kwargs = kwargs
        for event in self.events:
            yield event


def test_create_app_canonicalizes_a_symlinked_data_directory(manifest, tmp_path) -> None:
    canonical = tmp_path / "canonical-data"
    canonical.mkdir()
    alias = tmp_path / "data-alias"
    alias.symlink_to(canonical, target_is_directory=True)

    app = create_named_app(str(manifest.path), data_dir=alias)

    assert app.state.data_dir == canonical.resolve()


def test_provider_warmup_starts_after_health_is_available(manifest, tmp_path, monkeypatch) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    entered = threading.Event()
    release = threading.Event()
    calls: list[tuple[str, str, str | None]] = []

    monkeypatch.setattr(
        app.state.catalog,
        "provider_targets",
        lambda: [("codex", "", "/opt/agents/codex")],
    )

    def readiness(provider: str, *, host: str = "", binary: str | None = None):
        calls.append((provider, host, binary))
        entered.set()
        assert release.wait(timeout=3)
        return ProviderReadiness(
            provider=provider,
            installed=True,
            authenticated=True,
            binary_path=binary,
            path_state="resolved",
        )

    monkeypatch.setattr(app.state.catalog.launcher, "readiness", readiness)
    with TestClient(app) as client:
        try:
            assert entered.wait(timeout=1)
            assert client.get("/api/health").status_code == 200
            assert calls == [("codex", "", "/opt/agents/codex")]
        finally:
            release.set()


def test_startup_marks_all_skill_targets_then_refreshes_each_once(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    targets = [
        ("codex", "", "/opt/agents/codex"),
        ("claude", "research.example", "/opt/agents/claude"),
    ]
    calls: list[tuple[str, str, str, str | None]] = []
    completed = threading.Event()

    monkeypatch.setattr(app.state.catalog, "provider_targets", lambda: targets)

    def mark(provider: str, host: str, binary: str | None):
        calls.append(("mark", provider, host, binary))

    def readiness(provider: str, *, host: str = "", binary: str | None = None):
        calls.append(("readiness", provider, host, binary))
        return ProviderReadiness(
            provider=provider,
            installed=True,
            authenticated=True,
            version="test-version",
            binary_path=binary,
            path_state="resolved",
        )

    def refresh(provider: str, host: str, binary: str | None, _readiness):
        calls.append(("refresh", provider, host, binary))
        if sum(call[0] == "refresh" for call in calls) == len(targets):
            completed.set()

    monkeypatch.setattr(app.state.provider_skills, "mark_refreshing", mark)
    monkeypatch.setattr(app.state.catalog.launcher, "readiness", readiness)
    monkeypatch.setattr(app.state.provider_skills, "refresh", refresh)

    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
        assert completed.wait(timeout=2)

    assert calls[:2] == [
        ("mark", "codex", "", "/opt/agents/codex"),
        ("mark", "claude", "research.example", "/opt/agents/claude"),
    ]
    assert sorted(call[1:] for call in calls if call[0] == "readiness") == sorted(targets)
    assert sorted(call[1:] for call in calls if call[0] == "refresh") == sorted(targets)


def test_project_snapshot_and_resolution_use_last_good_provider_skills(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    app.state.catalog.store.save_provider_skill_inventory_success(
        "codex",
        "",
        None,
        resolved_binary="/opt/agents/codex",
        provider_version="1.2.3",
        command=["/opt/agents/codex", "app-server"],
        protocol="jsonrpc",
        skills=[
            ProviderSkill(
                name="native-review",
                label="Native review",
                description="Review with the provider-native workflow.",
            )
        ],
        inventory_hash="inventory-123",
        refreshed_at="2026-08-08T00:00:00+00:00",
    )
    service = app.state.catalog.open(app.state.default_project_id)

    snapshot = service.project_snapshot()
    inventory = snapshot["provider_skill_inventories"]["laptop"]["codex"]
    assert inventory["status"] == "fresh"
    assert [skill["name"] for skill in inventory["skills"]] == ["native-review"]

    request = RunRequest(
        provider="codex",
        run_on="laptop",
        invoked_provider_skill_names=["native-review"],
        resolved_provider_skills=[
            {
                "provider": "wrong",
                "machine": "wrong",
                "provider_version": "old",
                "inventory_hash": "old",
                "name": "old",
                "label": "Old",
                "description": "Old receipt",
            }
        ],
    )
    resolved = service.resolve_skill_request(request)
    assert [skill.name for skill in resolved.resolved_provider_skills] == ["native-review"]
    assert resolved.resolved_provider_skills[0].machine == "laptop"
    assert not resolved.resolved_provider_skills[0].stale

    app.state.catalog.store.save_provider_skill_inventory_failure(
        "codex",
        "",
        None,
        diagnostic="remote probe timed out",
        updated_at="2026-08-08T00:01:00+00:00",
    )
    stale = service.project_snapshot()["provider_skill_inventories"]["laptop"]["codex"]
    assert stale["status"] == "stale"
    assert stale["diagnostic"] == "remote probe timed out"
    assert [skill["name"] for skill in stale["skills"]] == ["native-review"]
    stale_request = service.resolve_skill_request(request)
    assert stale_request.resolved_provider_skills[0].stale

    with pytest.raises(ValueError, match="missing-native"):
        service.resolve_skill_request(
            request.model_copy(update={"invoked_provider_skill_names": ["missing-native"]})
        )


def test_remote_stage_sweep_starts_after_health_is_available(
    manifest, tmp_path, monkeypatch
) -> None:
    entered = threading.Event()
    release = threading.Event()

    monkeypatch.setattr(
        "rcp.api.app.ProjectCatalog.state_host",
        lambda _catalog, _project_id: "research.example",
    )

    def blocked_sweep(_stage) -> None:
        entered.set()
        assert release.wait(timeout=3)

    monkeypatch.setattr("rcp.api.app.RemoteRunStage.sweep", blocked_sweep)
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    monkeypatch.setattr(app.state.catalog, "provider_targets", lambda: [])

    assert not entered.is_set()
    with TestClient(app) as client:
        try:
            assert entered.wait(timeout=1)
            assert client.get("/api/health").status_code == 200
        finally:
            release.set()


def test_stale_instance_guard_rejects_mutation_before_side_effect(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    client = TestClient(app)
    project_id = app.state.default_project_id

    rejected = client.delete(
        f"/api/projects/{project_id}",
        headers={"X-RCP-Instance-ID": "replaced-instance"},
    )

    assert rejected.status_code == 409
    assert "replaced by another backend instance" in rejected.json()["detail"]
    assert rejected.json()["instance_id"] == app.state.instance_metadata.instance_id
    assert any(item["id"] == project_id for item in client.get("/api/projects").json())


class ScriptedLauncher:
    """Provider stub that writes files into the run's scratch folder.

    One script entry per launch, mapping file name to content; the last entry
    repeats once the script is exhausted, so an agent that never gets it right is
    one entry long.
    """

    def __init__(self, turns: list[dict[str, str]], *, message: str = "") -> None:
        self.turns = turns
        self.message = message
        self.native_session_id = str(uuid.uuid4())
        self.prompts: list[str] = []
        self.resumed_sessions: list[str | None] = []
        self.workspaces: list[Path] = []
        self.launch_kwargs: list[dict[str, object]] = []
        self.input_snapshots: list[dict[str, str]] = []

    @property
    def calls(self) -> int:
        return len(self.prompts)

    async def stream(self, _provider, prompt, **kwargs):
        turn = self.turns[min(self.calls, len(self.turns) - 1)]
        self.prompts.append(prompt)
        self.resumed_sessions.append(kwargs.get("session_id"))
        self.launch_kwargs.append(kwargs)
        workspace = Path(kwargs["cwd"])
        self.workspaces.append(workspace)
        inputs = workspace / "inputs"
        self.input_snapshots.append(
            {
                item.name: item.read_text(encoding="utf-8")
                for item in inputs.iterdir()
                if item.is_file()
            }
            if inputs.is_dir()
            else {}
        )
        for name, content in turn.items():
            (workspace / name).write_text(content, encoding="utf-8")
        yield AgentEvent(event="session", session_id=self.native_session_id)
        if self.message:
            yield AgentEvent(event="answer", text=self.message)
        yield AgentEvent(event="done")


def _local_task_contract(prompt: str) -> str:
    lines = prompt.splitlines()
    assert len(lines) >= 2
    return Path(lines[1]).read_text(encoding="utf-8")


def _events(frames: list[str]) -> list[AgentEvent]:
    return [
        AgentEvent.model_validate_json(frame.removeprefix("data: ").strip()) for frame in frames
    ]


def _record_lineage_task(
    store: AppStore,
    operation_id: str,
    *,
    parent: str | None,
    resumed: bool,
    graph_revision: int | None = None,
    with_created_receipt: bool = True,
    attempt: int = 1,
) -> None:
    now = store.now()
    store.create_agent_task(
        AgentTaskRecord(
            operation_id=operation_id,
            project_id="lineage-test",
            kind="node_chat",
            status="failed",
            request={},
            created_at=now,
            updated_at=now,
            status_message="test fixture",
            parent_operation_id=parent,
            attempt=attempt,
        )
    )
    if with_created_receipt:
        store.record_agent_task_receipt(
            operation_id,
            "operation_created",
            {
                "kind": "node_chat",
                "attempt": attempt,
                "has_parent": parent is not None,
                "resumed": resumed,
            },
        )
    if graph_revision is not None:
        store.record_agent_task_receipt(
            operation_id,
            "chat_context_assembled",
            {"graph_revision": graph_revision},
        )


def _applied_revision(frames: list[str]) -> int | None:
    for event in _events(frames):
        if event.event != "message":
            continue
        try:
            value = json.loads(event.text)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "applied_revision" in value:
            return int(value["applied_revision"])
    return None


def _error_texts(frames: list[str]) -> list[str]:
    return [event.text for event in _events(frames) if event.event == "error"]


def _graph_update(frames: list[str]) -> dict[str, object] | None:
    for event in _events(frames):
        if event.event != "message":
            continue
        try:
            value = json.loads(event.text)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("graph_update"), dict):
            return value["graph_update"]
    return None


def test_project_endpoints(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    client = TestClient(app)

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["project"] == "test-paper"

    project_id = app.state.default_project_id
    project = client.get(f"/api/projects/{project_id}")
    assert project.status_code == 200
    assert project.json()["id"] == project_id
    assert project.json()["project_truth_scope"] == ["repo-a", "repo-b"]

    append_fixture_patch(app.state.service, seed_patch())
    generation = app.state.catalog.reserve_cached_snapshot_generation(project_id)
    _, snapshot = app.state.catalog.reconcile_snapshot(project_id)
    assert app.state.catalog.commit_cached_snapshot(
        project_id,
        snapshot,
        generation=generation,
    )
    seeded_project = client.get(f"/api/projects/{project_id}")
    assert seeded_project.status_code == 200
    assert seeded_project.json()["primary_question"]["type"] == "research_question"
    reviewed = client.post(
        f"/api/projects/{project_id}/sync",
        json={
            "base_revision": 2,
            "nodes": [
                {
                    "node_id": "hyp/replanning-restores-plasticity",
                    "base_updated_rev": 2,
                    "standing": "accepted",
                }
            ],
        },
    )
    assert reviewed.status_code == 200
    assert reviewed.json()["nodes"]["hyp/replanning-restores-plasticity"]["standing"] == "accepted"


def test_degraded_replay_is_visible_and_canonical_api_writes_are_blocked(
    manifest, tmp_path
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    project_id = app.state.default_project_id
    append_fixture_patch(service, seed_patch())
    append_fixture_patch(service, refresh_patch("rq/tampered"))
    append_fixture_patch(service, refresh_patch("rq/not-replayed"))
    patch_path = manifest.research_dir / "patches" / "000003.json"
    raw = json.loads(patch_path.read_text(encoding="utf-8"))
    raw["ops"][0]["nodes"][0]["type"] = "not-a-node-type"
    patch_path.write_text(json.dumps(raw), encoding="utf-8")
    client = TestClient(app)

    graph = client.get(f"/api/projects/{project_id}/graph")

    assert graph.status_code == 200, graph.json()
    assert graph.json()["replay_status"] == "degraded"
    assert graph.json()["revision"] == 2
    assert graph.json()["replay_failure"]["revision"] == 3
    assert "rq/not-replayed" not in graph.json()["nodes"]

    sync = client.post(
        f"/api/projects/{project_id}/sync",
        json={
            "base_revision": 2,
            "nodes": [
                {
                    "node_id": "rq/learning-after-shift",
                    "base_updated_rev": 2,
                    "standing": "accepted",
                }
            ],
        },
    )
    assert sync.status_code == 409
    assert sync.json()["failed_revision"] == 3

    refresh = client.post(f"/api/projects/{project_id}/tasks/refresh", json={})
    assert refresh.status_code == 409
    assert refresh.json()["code"] == "patch-schema-invalid"

    project = client.get(f"/api/projects/{project_id}").json()
    profiles = {
        surface: {key: profile[key] for key in ("provider", "model", "reasoning", "run_on")}
        for surface, profile in project["agent_profiles"].items()
    }
    settings = client.put(
        f"/api/projects/{project_id}/settings",
        json={
            "default_run_truth_scope": ["repo-a"],
            "agent_profiles": profiles,
        },
    )
    assert settings.status_code == 409
    assert settings.json()["coherent_revision"] == 2

    # Conversations stay usable: a degraded graph refuses the patch at append
    # time rather than blocking the turn that might not write one.
    for mode in ("discuss", "work"):
        accepted = _validated_task_request(
            service,
            "project_chat",
            {
                "message": "What is the current coherent graph?",
                "chat_id": str(uuid.uuid4()),
                "mode": mode,
            },
        )
        assert accepted.mode == mode
    with pytest.raises(ReplayHalted):
        _validated_task_request(service, "refresh", {})


def test_project_open_reuses_its_single_materialization(manifest, tmp_path, monkeypatch) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    client = TestClient(app)
    project_id = app.state.default_project_id
    app.state.catalog._services.clear()
    initialize = HistoryManager.initialize
    materializations = 0

    def counted_initialize(history):
        nonlocal materializations
        materializations += 1
        return initialize(history)

    monkeypatch.setattr(HistoryManager, "initialize", counted_initialize)
    monkeypatch.setattr(
        app.state.catalog.launcher,
        "readiness",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("project open must not probe provider readiness")
        ),
    )

    response = client.get(f"/api/projects/{project_id}")

    assert response.status_code == 200
    assert materializations == 1
    payload = response.json()
    assert payload["graph"]["revision"] == payload["revision"]
    assert payload["paper"] == client.get(f"/api/projects/{project_id}/paper").json()
    assert payload["provider_readiness"] == {}
    assert payload["providers"] == {}


def test_remote_probe_compares_with_display_snapshot_head_after_interrupted_reconcile(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    initial = TestClient(app).get(f"/api/projects/{project_id}").json()
    assert initial["revision"] == 1

    append_fixture_patch(app.state.service, seed_patch())
    workspace = SSHStateWorkspace(manifest.research_dir, "research.example", "/srv/project")
    monkeypatch.setattr(
        workspace,
        "_ssh",
        lambda arguments, **_kwargs: subprocess.CompletedProcess(
            arguments,
            0,
            "000002.json\n",
            "",
        ),
    )
    app.state.catalog._services.clear()
    monkeypatch.setattr(
        projects_module,
        "state_workspace_for_probe",
        lambda _manifest, _data_dir: workspace,
    )

    assert app.state.catalog.probe_remote_patch_log_head(project_id) == "moved"


def test_cached_project_survives_restart_without_opening_history(
    manifest, tmp_path, monkeypatch
) -> None:
    data_dir = tmp_path / "data"
    first_app = create_named_app(str(manifest.path), data_dir=data_dir)
    project_id = first_app.state.default_project_id
    authoritative = TestClient(first_app).get(f"/api/projects/{project_id}")
    assert authoritative.status_code == 200

    restarted = create_named_app(data_dir=data_dir)
    monkeypatch.setattr(
        restarted.state.catalog,
        "_open_service",
        lambda _project_id: (_ for _ in ()).throw(
            AssertionError("display and task history reads must not open the project")
        ),
    )
    monkeypatch.setattr(
        HistoryManager,
        "initialize",
        lambda _history: (_ for _ in ()).throw(
            AssertionError("display and task history reads must not materialize history")
        ),
    )
    client = TestClient(restarted)

    cached = client.get(f"/api/projects/{project_id}/cached")
    tasks = client.get(f"/api/projects/{project_id}/tasks")
    task_detail = client.get(f"/api/projects/{project_id}/tasks/missing")

    assert cached.status_code == 200
    assert cached.json() == authoritative.json()
    assert tasks.status_code == 200
    assert tasks.json() == []
    assert task_detail.status_code == 404
    assert project_id not in restarted.state.catalog._services


def test_normal_launch_exposes_health_and_cache_without_opening_canonical_state(
    manifest, tmp_path, monkeypatch
) -> None:
    data_dir = tmp_path / "data"
    first_app = create_named_app(str(manifest.path), data_dir=data_dir)
    project_id = first_app.state.default_project_id
    authoritative = TestClient(first_app).get(f"/api/projects/{project_id}")
    assert authoritative.status_code == 200

    app = create_named_app(str(manifest.path), data_dir=data_dir)
    assert project_id not in app.state.catalog._services

    def forbidden_open_service(requested_project_id):
        raise AssertionError(f"cached navigation opened {requested_project_id}")

    monkeypatch.setattr(app.state.catalog, "_open_service", forbidden_open_service)

    async def drive_concurrently():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            project = await asyncio.wait_for(
                client.get(f"/api/projects/{project_id}"),
                timeout=1,
            )
            health = await asyncio.wait_for(client.get("/api/health"), timeout=1)
            cached = await asyncio.wait_for(
                client.get(f"/api/projects/{project_id}/cached"),
                timeout=1,
            )
            return health, cached, project

    health, cached, project = asyncio.run(drive_concurrently())

    assert health.status_code == 200
    assert health.json()["project"] == manifest.name
    assert cached.status_code == 200
    assert cached.json() == authoritative.json()
    assert project.status_code == 200
    assert project.json() == authoritative.json()


def test_slow_project_open_does_not_block_concurrent_task_history(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    app.state.catalog._services.clear()
    original_open_service = app.state.catalog._open_service
    entered = threading.Event()
    release = threading.Event()
    open_calls = 0

    def slow_open_service(requested_project_id):
        nonlocal open_calls
        open_calls += 1
        entered.set()
        release.wait(timeout=3)
        return original_open_service(requested_project_id)

    monkeypatch.setattr(app.state.catalog, "_open_service", slow_open_service)

    async def drive_concurrently():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            started_at = time.perf_counter()
            authoritative = asyncio.create_task(client.get(f"/api/projects/{project_id}"))
            task_response = None
            try:
                assert await asyncio.to_thread(entered.wait, 1)
                entered_after = time.perf_counter() - started_at
                task_response = await asyncio.wait_for(
                    client.get(f"/api/projects/{project_id}/tasks"),
                    timeout=1,
                )
            finally:
                release.set()
            return entered_after, task_response, await authoritative

    entered_after, task_response, authoritative = asyncio.run(drive_concurrently())

    assert entered_after < 1
    assert task_response is not None
    assert task_response.status_code == 200
    assert authoritative.status_code == 200
    assert open_calls == 1


def test_blocking_project_source_read_does_not_stall_health(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    entered = threading.Event()
    release = threading.Event()
    original_index_snapshot = app.state.service.index_snapshot

    def slow_index_snapshot(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=3)
        return original_index_snapshot(*args, **kwargs)

    monkeypatch.setattr(app.state.service, "index_snapshot", slow_index_snapshot)

    async def drive_concurrently():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            sources = asyncio.create_task(client.get(f"/api/projects/{project_id}/sources"))
            try:
                assert await asyncio.to_thread(entered.wait, 1)
                health = await asyncio.wait_for(client.get("/api/health"), timeout=1)
            finally:
                release.set()
            return health, await sources

    health, sources = asyncio.run(drive_concurrently())

    assert health.status_code == 200
    assert sources.status_code == 200


def test_concurrent_project_calls_share_first_open_without_blocking_health(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    app.state.catalog._services.clear()
    original_open_service = app.state.catalog._open_service
    entered = threading.Event()
    release = threading.Event()
    open_calls = 0

    def slow_open_service(requested_project_id):
        nonlocal open_calls
        open_calls += 1
        entered.set()
        assert release.wait(timeout=3)
        return original_open_service(requested_project_id)

    monkeypatch.setattr(app.state.catalog, "_open_service", slow_open_service)

    async def drive_concurrently():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            chats = asyncio.create_task(client.get(f"/api/projects/{project_id}/chats"))
            try:
                assert await asyncio.to_thread(entered.wait, 1)
                project = asyncio.create_task(client.get(f"/api/projects/{project_id}"))
                health = await asyncio.wait_for(client.get("/api/health"), timeout=1)
            finally:
                release.set()
            return health, await chats, await project

    health, chats, project = asyncio.run(drive_concurrently())

    assert health.status_code == 200
    assert chats.status_code == 200
    assert project.status_code == 200
    assert open_calls == 1


def test_failed_singleflight_open_preserves_error_and_can_retry(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    catalog = app.state.catalog
    project_id = app.state.default_project_id
    catalog._services.clear()
    original_open_service = catalog._open_service
    calls = 0

    def flaky_open_service(requested_project_id):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise StateUnavailable("exact remote open failure")
        return original_open_service(requested_project_id)

    monkeypatch.setattr(catalog, "_open_service", flaky_open_service)

    with pytest.raises(StateUnavailable, match="exact remote open failure"):
        catalog.open(project_id)
    service = catalog.open(project_id)

    assert service.manifest.name == manifest.name
    assert calls == 2


def test_delete_tombstones_an_inflight_first_open(manifest, tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    app = create_named_app(str(manifest.path), data_dir=data_dir)
    catalog = app.state.catalog
    project_id = app.state.default_project_id
    original_open_service = catalog._open_service
    entered = threading.Event()
    release = threading.Event()

    def slow_open_service(requested_project_id):
        entered.set()
        assert release.wait(timeout=3)
        return original_open_service(requested_project_id)

    monkeypatch.setattr(catalog, "_open_service", slow_open_service)

    async def drive_concurrently():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            opening = asyncio.create_task(client.get(f"/api/projects/{project_id}"))
            assert await asyncio.to_thread(entered.wait, 1)
            deleting = asyncio.create_task(client.delete(f"/api/projects/{project_id}"))
            for _ in range(100):
                if project_id in catalog._deleting:
                    break
                await asyncio.sleep(0.01)
            assert project_id in catalog._deleting
            cached = await client.get(f"/api/projects/{project_id}/cached")
            release.set()
            return await opening, await deleting, cached

    opening, deleted, cached = asyncio.run(drive_concurrently())

    assert opening.status_code == 404
    assert deleted.status_code == 200
    assert cached.status_code == 404
    assert project_id not in catalog._services
    assert not catalog._cached_snapshot_path(project_id).exists()
    client = TestClient(app)
    assert client.get(f"/api/projects/{project_id}").status_code == 404
    assert client.get(f"/api/projects/{project_id}/cached").status_code == 404


def test_delete_serializes_against_display_snapshot_replacement(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    catalog = app.state.catalog
    project_id = app.state.default_project_id
    snapshot = TestClient(app).get(f"/api/projects/{project_id}").json()
    cache_path = catalog._cached_snapshot_path(project_id)
    entered = threading.Event()
    release = threading.Event()
    original_replace = projects_module.os.replace

    def blocked_replace(source, destination):
        if Path(destination) == cache_path:
            entered.set()
            assert release.wait(timeout=3)
        return original_replace(source, destination)

    monkeypatch.setattr(projects_module.os, "replace", blocked_replace)

    async def drive_concurrently():
        writer = asyncio.create_task(
            asyncio.to_thread(catalog.write_cached_snapshot, project_id, snapshot)
        )
        assert await asyncio.to_thread(entered.wait, 1)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            deleting = asyncio.create_task(client.delete(f"/api/projects/{project_id}"))
            for _ in range(100):
                if project_id in catalog._deleting:
                    break
                await asyncio.sleep(0.01)
            assert project_id in catalog._deleting
            release.set()
            await writer
            return await deleting

    deleted = asyncio.run(drive_concurrently())

    assert deleted.status_code == 200
    assert not cache_path.exists()
    assert catalog.cached_snapshot(project_id) is None
    assert project_id not in catalog._services
    client = TestClient(app)
    assert client.get(f"/api/projects/{project_id}").status_code == 404
    assert client.get(f"/api/projects/{project_id}/cached").status_code == 404


def test_catalog_summary_reuses_project_snapshot(manifest, tmp_path, monkeypatch) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    service = app.state.service
    snapshot = service.project_snapshot()
    snapshot["revision"] = 17
    snapshot["primary_question"] = {"question": "Which path remains plastic?"}
    snapshot["counts"] = {
        "pending_proposals": 2,
        "decisions_awaiting_choice": 3,
        "open_blockers": 4,
    }
    monkeypatch.setattr(
        service,
        "project_snapshot",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("summary update must reuse the supplied snapshot")
        ),
    )

    record = app.state.catalog.update_summary(project_id, snapshot)

    assert record.revision == 17
    assert record.primary_question == "Which path remains plastic?"
    assert record.attention_count == 9


def test_project_snapshot_counts_only_ripe_decisions_and_open_asserted_blockers(
    manifest, tmp_path
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    blockers = [
        Blocker(
            id="blk/asserted-open",
            type="blocker",
            title="Asserted open blocker",
            description="Needs human attention.",
            standing="asserted",
            status="open",
        ),
        Blocker(
            id="blk/accepted-open",
            type="blocker",
            title="Accepted open blocker",
            description="Remains operationally open.",
            standing="accepted",
            status="open",
        ),
        Blocker(
            id="blk/contested-open",
            type="blocker",
            title="Contested open blocker",
            description="Remains operationally open.",
            standing="contested",
            status="open",
        ),
        Blocker(
            id="blk/asserted-resolved",
            type="blocker",
            title="Resolved asserted blocker",
            description="No longer operationally open.",
            standing="asserted",
            status="resolved",
        ),
    ]
    decisions = [
        Decision(
            id=f"dec/{status}",
            type="decision",
            title=f"{status.title()} decision",
            question="Which option should be used?",
            options=["first", "second"],
            selected_option="first" if status in {"decided", "revisit"} else None,
            status=status,
        )
        for status in ("open", "ready", "decided", "revisit", "superseded")
    ]
    state = GraphState(
        nodes={item.id: item for item in [*blockers, *decisions]},
    )

    snapshot = app.state.service.project_snapshot(state=state)

    assert snapshot["counts"]["decisions_awaiting_choice"] == 2
    assert snapshot["counts"]["open_blockers"] == 1
    assert snapshot["attention"] == {
        "pending_proposal_ids": [],
        "decisions_awaiting_choice_ids": ["dec/ready", "dec/revisit"],
        "open_blocker_ids": ["blk/asserted-open"],
    }
    assert {
        node_id: (node["status"], node["standing"])
        for node_id, node in snapshot["graph"]["nodes"].items()
        if node["type"] == "blocker"
    } == {
        "blk/asserted-open": ("open", "asserted"),
        "blk/accepted-open": ("open", "accepted"),
        "blk/contested-open": ("open", "contested"),
        "blk/asserted-resolved": ("resolved", "asserted"),
    }


def test_cached_catalog_open_returns_service_without_building_snapshot(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    service = app.state.catalog.open(project_id)
    monkeypatch.setattr(
        service,
        "project_snapshot",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("catalog.open must not serialize the project")
        ),
    )

    assert app.state.catalog.open(project_id) is service


def test_legacy_direct_human_write_endpoints_are_not_exposed(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    client = TestClient(app)
    project_id = app.state.default_project_id
    append_fixture_patch(app.state.service, seed_patch())
    node_path = "hyp/replanning-restores-plasticity"

    responses = [
        client.post(
            f"/api/projects/{project_id}/nodes/{node_path}/review",
            json={"standing": "accepted"},
        ),
        client.put(
            f"/api/projects/{project_id}/nodes/{node_path}",
            json={"base_updated_rev": 1, "changes": {"title": "Bypass"}},
        ),
        client.post(
            f"/api/projects/{project_id}/proposals/missing/decide",
            json={"decision": "approved"},
        ),
    ]

    assert [response.status_code for response in responses] == [405, 405, 405]
    assert app.state.service.history.current_materialization().state.revision == 2


def test_cache_metrics_and_clear_endpoint_respect_active_task_boundary(manifest, tmp_path) -> None:
    data_dir = tmp_path / "data"
    app = create_named_app(str(manifest.path), data_dir=data_dir)
    client = TestClient(app)
    project_id = app.state.default_project_id
    source_root, slice_root = project_cache_roots(data_dir, project_id)
    assert app.state.service.indexer.cache_root == source_root
    assert app.state.service.indexer.session_artifact_root() == slice_root
    cached = source_root / "remote" / "codex" / "source.jsonl"
    cached_slice = slice_root / "slice-a" / "records.jsonl"
    cached.parent.mkdir(parents=True)
    cached_slice.parent.mkdir(parents=True)
    cached.write_text("project-a-source", encoding="utf-8")
    cached_slice.write_text("project-a-slice", encoding="utf-8")

    store = app.state.background_tasks.store
    project_b_id = str(uuid.uuid4())
    store.upsert_project(
        ProjectRecord(
            project_id=project_b_id,
            home_space_id=store.space_id,
            locator=str(tmp_path / "project-b" / ".research" / "manifest.toml"),
            name="project-b",
            state_location=str(tmp_path / "project-b" / ".research"),
            state_remote=False,
            added_at=store.now(),
        )
    )
    b_source_root, b_slice_root = project_cache_roots(data_dir, project_b_id)
    b_cached = b_source_root / "remote" / "codex" / "source.jsonl"
    b_cached_slice = b_slice_root / "slice-b" / "records.jsonl"
    b_cached.parent.mkdir(parents=True)
    b_cached_slice.parent.mkdir(parents=True)
    b_cached.write_text("project-b-source", encoding="utf-8")
    b_cached_slice.write_text("project-b-slice", encoding="utf-8")

    legacy_cached = data_dir / "source-cache" / "remote" / "legacy.jsonl"
    legacy_slice = data_dir / "session-slices" / "legacy" / "records.jsonl"
    legacy_cached.parent.mkdir(parents=True)
    legacy_slice.parent.mkdir(parents=True)
    legacy_cached.write_text("legacy-source", encoding="utf-8")
    legacy_slice.write_text("legacy-slice", encoding="utf-8")
    original = Path(manifest.sources.codex_roots[0]) / "provider-original.jsonl"
    original.write_text("provider-original", encoding="utf-8")
    manifest_before = manifest.path.read_bytes()

    snapshot = client.get(f"/api/projects/{project_id}")
    assert snapshot.status_code == 200
    assert snapshot.json()["cache_metrics"]["remote_sources"]["count"] == 1
    assert snapshot.json()["cache_metrics"]["session_slices"]["count"] == 1

    now = datetime.now(UTC).isoformat()
    store.create_agent_task(
        AgentTaskRecord(
            operation_id="other-project-cache-reader",
            project_id=project_b_id,
            kind="refresh",
            status="running",
            request={},
            created_at=now,
            updated_at=now,
            status_message="running",
        )
    )
    cleared = client.delete(f"/api/projects/{project_id}/caches")
    assert cleared.status_code == 200
    assert cleared.json()["remote_sources"]["count"] == 0
    assert cleared.json()["session_slices"]["count"] == 0
    assert not cached.exists()
    assert not cached_slice.exists()
    assert b_cached.read_text(encoding="utf-8") == "project-b-source"
    assert b_cached_slice.read_text(encoding="utf-8") == "project-b-slice"
    assert legacy_cached.read_text(encoding="utf-8") == "legacy-source"
    assert legacy_slice.read_text(encoding="utf-8") == "legacy-slice"

    cached.parent.mkdir(parents=True, exist_ok=True)
    cached_slice.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text("project-a-source-again", encoding="utf-8")
    cached_slice.write_text("project-a-slice-again", encoding="utf-8")
    store.fail_agent_task("other-project-cache-reader", "finished for test")
    store.create_agent_task(
        AgentTaskRecord(
            operation_id="this-project-cache-reader",
            project_id=project_id,
            kind="refresh",
            status="running",
            request={},
            created_at=now,
            updated_at=now,
            status_message="running",
        )
    )

    refused = client.delete(f"/api/projects/{project_id}/caches")
    assert refused.status_code == 409
    assert cached.read_text(encoding="utf-8") == "project-a-source-again"

    global_refused = client.delete(f"/api/caches?project_id={project_id}")
    assert global_refused.status_code == 409
    assert cached.exists()
    assert b_cached.exists()
    assert legacy_cached.exists()

    store.fail_agent_task("this-project-cache-reader", "finished for test")
    all_cleared = client.delete(f"/api/caches?project_id={project_id}")
    assert all_cleared.status_code == 200
    assert all_cleared.json()["remote_sources"]["count"] == 0
    assert all_cleared.json()["remote_sources"]["bytes"] == 0
    assert all_cleared.json()["session_slices"]["count"] == 0
    assert all_cleared.json()["session_slices"]["bytes"] == 0
    assert not cached.exists()
    assert not b_cached.exists()
    assert not b_cached_slice.exists()
    assert not legacy_cached.exists()
    assert not legacy_slice.exists()
    assert original.read_text(encoding="utf-8") == "provider-original"
    assert manifest.path.read_bytes() == manifest_before
    assert store.agent_task("this-project-cache-reader").status == "failed"


def test_project_registry_survives_hub_restart(manifest, tmp_path) -> None:
    data_dir = tmp_path / "data"
    registered = create_named_app(str(manifest.path), data_dir=data_dir)
    project_id = registered.state.default_project_id
    assert TestClient(registered).get(f"/api/projects/{project_id}").status_code == 200

    hub = create_named_app(data_dir=data_dir)
    client = TestClient(hub)
    cards = client.get("/api/projects")

    assert cards.status_code == 200
    assert cards.json()[0]["id"] == project_id
    assert cards.json()[0]["name"] == "test-paper"
    assert cards.json()[0]["revision"] == 1


def test_seed_runs_in_background_and_keeps_api_responsive(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id

    async def stream(*_args):
        yield _event_frame(AgentEvent(event="session", session_id=str(uuid.uuid4())))
        await asyncio.sleep(0.1)
        yield _event_frame(AgentEvent(event="message", text=json.dumps({"applied_revision": 1})))
        yield _event_frame(AgentEvent(event="done"))

    app.state.background_tasks.stream = stream
    client = TestClient(app)
    started = client.post(
        f"/api/projects/{project_id}/tasks/seed",
        json={"run_truth_scope": ["repo-a"]},
    )

    assert started.status_code == 202
    assert client.get("/api/health").status_code == 200
    operation_id = started.json()["operation_id"]
    completed = _wait_for_run(client, project_id, operation_id)
    assert completed["status"] == "succeeded"
    assert completed["applied_revision"] == 1


def test_seed_waits_for_live_canonical_owner_without_failing(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    workspace = app.state.service.history.workspace
    lock_waiting = threading.Event()
    release_lock = threading.Event()
    launcher = ScriptedLauncher([{"patch.json": agent_patch_json(seed_patch())}])

    @contextmanager
    def contended_lock(*, on_wait=None, cancelled=None, on_lost=None):
        del on_lost
        if on_wait is not None:
            on_wait("Waiting for another graph-writing run to release canonical state.")
        lock_waiting.set()
        while not release_lock.wait(timeout=0.01):
            if cancelled is not None and cancelled():
                raise RunLockCancelled("Run-lock acquisition was cancelled while waiting.")
        yield RunLockLease(workspace.location)

    monkeypatch.setattr(workspace, "run_lock", contended_lock)
    monkeypatch.setattr(app.state.catalog.launcher, "stream", launcher.stream)
    client = TestClient(app)
    started = client.post(
        f"/api/projects/{project_id}/tasks/seed",
        json={"run_truth_scope": ["repo-a"]},
    )
    operation_id = started.json()["operation_id"]

    try:
        assert lock_waiting.wait(timeout=2)
        waiting = client.get(f"/api/projects/{project_id}/tasks/{operation_id}").json()
        assert waiting["status"] == "running"
        assert waiting["phase"] == "waiting"
        assert "Waiting for another graph-writing run" in waiting["status_message"]
        assert workspace.location in waiting["status_message"]
        assert launcher.calls == 0
    finally:
        release_lock.set()

    completed = _wait_for_run(client, project_id, operation_id)
    assert completed["status"] == "succeeded"
    assert launcher.calls == 1
    assert "canonical_state_lock_wait" in {item["category"] for item in completed["debug_receipts"]}


def test_seed_can_pause_while_waiting_for_canonical_owner(manifest, tmp_path, monkeypatch) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    workspace = app.state.service.history.workspace
    lock_waiting = threading.Event()

    @contextmanager
    def contended_lock(*, on_wait=None, cancelled=None, on_lost=None):
        del on_lost
        if on_wait is not None:
            on_wait("Waiting for another graph-writing run to release canonical state.")
        lock_waiting.set()
        while cancelled is None or not cancelled():
            time.sleep(0.01)
        raise RunLockCancelled("Run-lock acquisition was cancelled while waiting.")
        yield  # pragma: no cover - makes this function a context manager

    monkeypatch.setattr(workspace, "run_lock", contended_lock)
    client = TestClient(app)
    started = client.post(
        f"/api/projects/{project_id}/tasks/seed",
        json={"run_truth_scope": ["repo-a"]},
    )
    operation_id = started.json()["operation_id"]
    assert lock_waiting.wait(timeout=2)

    paused_request = client.post(f"/api/projects/{project_id}/tasks/{operation_id}/pause")

    assert paused_request.status_code == 202
    paused = _wait_for_status(client, project_id, operation_id, {"paused"})
    assert paused["error"] is None
    assert paused["status_message"] == "Paused while waiting for canonical state."
    assert paused["can_resume"] is False


def test_seed_pauses_and_retains_its_patch_when_run_lock_ownership_is_lost(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    workspace = app.state.service.history.workspace
    provider_started = threading.Event()
    release_provider = threading.Event()
    acquired_lease: list[RunLockLease] = []

    class PausingLauncher(ScriptedLauncher):
        async def stream(self, *args, **kwargs):
            async for event in super().stream(*args, **kwargs):
                if event.event == "done":
                    provider_started.set()
                    while not release_provider.is_set():
                        await asyncio.sleep(0.01)
                yield event

    @contextmanager
    def observable_lock(*, on_wait=None, cancelled=None, on_lost=None):
        del on_wait, cancelled
        lease = RunLockLease(workspace.location, on_lost=on_lost)
        acquired_lease.append(lease)
        yield lease

    launcher = PausingLauncher([{"patch.json": agent_patch_json(seed_patch())}])
    monkeypatch.setattr(workspace, "run_lock", observable_lock)
    monkeypatch.setattr(app.state.catalog.launcher, "stream", launcher.stream)
    client = TestClient(app)
    started = client.post(
        f"/api/projects/{project_id}/tasks/seed",
        json={"run_truth_scope": ["repo-a"]},
    )
    operation_id = started.json()["operation_id"]

    assert provider_started.wait(timeout=2)
    acquired_lease[0]._mark_lost(
        f"Canonical-state lock holder for {workspace.location} exited unexpectedly."
    )
    release_provider.set()

    paused = _wait_for_status(client, project_id, operation_id, {"paused"})
    assert paused["error"] is None
    assert "lock holder" in paused["status_message"]
    assert "paused before applying further graph changes" in paused["status_message"]
    assert paused["stage_root"] is not None
    assert (Path(paused["stage_root"]) / "patch.json").is_file()
    assert client.get(f"/api/projects/{project_id}").json()["revision"] == 1
    categories = {item["category"] for item in paused["debug_receipts"]}
    assert "canonical_state_lock_lost" in categories


@pytest.mark.parametrize("kind", ["seed", "refresh"])
def test_seed_and_refresh_reject_caller_supplied_sessions(manifest, tmp_path, kind) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    client = TestClient(app)
    project_id = app.state.default_project_id
    session_id = str(uuid.uuid4())

    background = client.post(
        f"/api/projects/{project_id}/tasks/{kind}",
        json={"session_id": session_id},
    )

    assert background.status_code == 422
    assert client.get(f"/api/projects/{project_id}/tasks").json() == []


@pytest.mark.asyncio
async def test_graph_stream_rejects_uncheckpointed_session_before_launch(
    manifest, tmp_path
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    launcher = FakeLauncher([AgentEvent(event="done")])

    events = [
        item
        async for item in stream_graph_run(
            app.state.service,
            launcher,
            "seed",
            RunRequest(session_id=str(uuid.uuid4())),
            tmp_path / "data",
        )
    ]

    assert launcher.calls == 0
    assert "only be resumed from an RCP background task checkpoint" in events[0]


@pytest.mark.asyncio
async def test_graph_stream_launches_with_degraded_source_fallback(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())

    monkeypatch.setattr(
        "rcp.service.preflight_provider_roots",
        lambda *_args, **_kwargs: ["laptop/codex source root: provider source is unreadable"],
    )
    launcher = ScriptedLauncher([{"patch.json": agent_patch_json(refresh_patch())}])

    frames = [
        item
        async for item in stream_graph_run(
            service,
            launcher,
            "refresh",
            RunRequest(run_truth_scope=["repo-a"]),
            tmp_path / "data",
            execution=_agent_task_execution(
                app.state.background_tasks.store,
                "degraded-source-refresh",
            ),
        )
    ]

    assert launcher.calls == 1
    assert not _error_texts(frames)
    contract = next(
        value for name, value in launcher.input_snapshots[0].items() if name.endswith("-initial.md")
    )
    assert "did not respond to a readability check" in contract
    assert "provider source is unreadable" in contract
    assert "inspect them in place" in contract
    assert "Project ingestion watermark" in contract
    assert service.manifest.sources.codex_roots[0] in contract
    assert any(
        Path(path) == Path(service.manifest.sources.codex_roots[0])
        for path in launcher.launch_kwargs[0]["read_dirs"]
    )


@pytest.mark.asyncio
async def test_graph_stream_reuses_revision_from_assembled_context(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    context = service.assemble_run(
        RunRequest(run_truth_scope=["repo-a"]),
        surface="refresh",
    )
    assert context.graph_revision == 2
    monkeypatch.setattr(service, "assemble_run", lambda *_args, **_kwargs: context)
    monkeypatch.setattr(
        service,
        "graph_snapshot",
        lambda: (_ for _ in ()).throw(
            AssertionError("fresh graph run must reuse its materialized revision")
        ),
    )
    launcher = ScriptedLauncher([{"patch.json": agent_patch_json(refresh_patch())}])

    frames = [
        frame
        async for frame in stream_graph_run(
            service,
            launcher,
            "refresh",
            RunRequest(run_truth_scope=["repo-a"]),
            tmp_path / "data",
            execution=_agent_task_execution(
                app.state.background_tasks.store,
                "assembled-context-refresh",
            ),
        )
    ]

    assert _applied_revision(frames) == 3


def test_legacy_run_with_caller_session_cannot_resume(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    store = app.state.background_tasks.store
    project_id = app.state.default_project_id
    session_id = str(uuid.uuid4())
    now = store.now()
    legacy = store.create_agent_task(
        AgentTaskRecord(
            operation_id=str(uuid.uuid4()),
            project_id=project_id,
            kind="seed",
            status="paused",
            request=RunRequest(session_id=session_id).model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="Paused legacy run.",
            native_session_id=session_id,
        )
    )

    response = TestClient(app).post(
        f"/api/projects/{project_id}/tasks/{legacy.operation_id}/resume"
    )

    assert response.status_code == 409
    assert "not checkpointed or validated by RCP" in response.json()["detail"]


def test_background_seed_persists_exact_failure(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id

    async def stream(*_args):
        yield _event_frame(
            AgentEvent(
                event="error",
                text="Local execution would copy 3.8 GiB. Choose remote-1 in Run on.",
            )
        )

    app.state.background_tasks.stream = stream
    client = TestClient(app)
    started = client.post(
        f"/api/projects/{project_id}/tasks/seed",
        json={"run_truth_scope": ["repo-a"]},
    )

    failed = _wait_for_run(client, project_id, started.json()["operation_id"])
    assert failed["status"] == "failed"
    assert failed["error"] == ("Local execution would copy 3.8 GiB. Choose remote-1 in Run on.")
    receipt_categories = {receipt["category"] for receipt in failed["debug_receipts"]}
    assert {"operation_created", "operation_failed"}.issubset(receipt_categories)
    failure_receipt = next(
        receipt for receipt in failed["debug_receipts"] if receipt["category"] == "operation_failed"
    )
    assert failure_receipt["payload"] == {
        "status": "failed",
        "error_length": len(failed["error"]),
    }
    assert failed["error"] not in json.dumps(failure_receipt["payload"])


def test_rejected_refresh_is_corrected_without_burning_a_revision(manifest, tmp_path) -> None:
    """A validator rejection is a correctable authoring error, not history."""
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    launcher = ScriptedLauncher(
        [
            {"patch.json": agent_patch_json(gated_patch())},
            {"patch.json": agent_patch_json(refresh_patch())},
        ]
    )

    async def stream(_project_id, kind, request, execution):
        async for frame in stream_graph_run(
            service,
            launcher,
            kind,
            request,
            tmp_path / "data",
            execution=execution,
        ):
            yield frame

    app.state.background_tasks.stream = stream
    client = TestClient(app)
    started = client.post(
        f"/api/projects/{project_id}/tasks/refresh",
        json={
            "provider": "codex",
            "run_truth_scope": ["repo-a"],
            "run_on": "laptop",
        },
    )

    completed = _wait_for_run(client, project_id, started.json()["operation_id"])
    assert completed["status"] == "succeeded"
    assert completed["applied_revision"] == 3
    assert launcher.calls == 2
    assert len(launcher.prompts[0].splitlines()) < 200
    assert "Specialists remain\n  read-only" not in launcher.prompts[0]
    contract = next(
        value for name, value in launcher.input_snapshots[0].items() if name.endswith("initial.md")
    )
    assert "provider-owned fan-out into bounded read-only source-inspection subagents" in contract
    assert "sole writer of the final Patch" in contract
    assert "semantic Patch JSON object" in contract
    assert "authorized-session-keys.json" not in contract
    assert "authorized-session-keys.json" not in launcher.input_snapshots[0]
    rejection_diagnostic = next(
        value
        for name, value in launcher.input_snapshots[1].items()
        if name.endswith("correction-1.json")
    )
    assert "requires a Proposal and human approval" in rejection_diagnostic
    assert service.history.state().revision == 3
    assert "rq/transfer-after-shift" in service.history.state().nodes
    receipt_categories = {receipt["category"] for receipt in completed["debug_receipts"]}
    assert {
        "context_assembled",
        "stage_checkpoint",
        "agent_launch",
        "agent_prompt",
        "native_agent_checkpoint",
        "patch_retained",
        "patch_parsed",
        "patch_rejected",
        "patch_correction_requested",
        "operation_completed",
    }.issubset(receipt_categories)
    event_messages = [event["message"] for event in completed["events"]]
    assert "Agent task is running." not in event_messages
    assert "Validating and applying the graph update." in event_messages

    context_receipt = next(
        receipt
        for receipt in completed["debug_receipts"]
        if receipt["category"] == "context_assembled"
    )
    assert context_receipt["payload"]["repository_count"] == 1
    assert "source_errors" not in context_receipt["payload"]
    prompt_receipt = next(
        receipt for receipt in completed["debug_receipts"] if receipt["category"] == "agent_prompt"
    )
    exact_prompt = launcher.prompts[0]
    assert prompt_receipt["payload"]["prompt"] == exact_prompt
    assert prompt_receipt["payload"]["line_count"] == len(exact_prompt.splitlines())
    assert (
        prompt_receipt["payload"]["sha256"]
        == hashlib.sha256(exact_prompt.encode("utf-8")).hexdigest()
    )
    contracts = {item["role"]: item for item in completed["contracts"]}
    assert contracts["base"]["content"] == contract
    assert contracts["base"]["sha256"] == hashlib.sha256(contract.encode("utf-8")).hexdigest()
    assert "Patch-only correction authority" in contracts["graph_patch_correction_1"]["content"]


@pytest.mark.asyncio
async def test_graph_launch_passes_the_recorded_provider_binary(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    service.history.update_machine_provider_paths({"laptop": {"codex": "/opt/agents/codex"}})
    launcher = ScriptedLauncher([{"patch.json": agent_patch_json(refresh_patch())}])

    _ = [
        frame
        async for frame in stream_graph_run(
            service,
            launcher,
            "refresh",
            RunRequest(provider="codex", run_truth_scope=["repo-a"]),
            tmp_path / "data",
            execution=_agent_task_execution(
                app.state.background_tasks.store,
                "provider-binary-refresh",
            ),
        )
    ]

    assert launcher.launch_kwargs[0]["binary"] == "/opt/agents/codex"


@pytest.mark.parametrize("file_name", ["Patch.json", "output.json"])
@pytest.mark.asyncio
async def test_patch_under_an_unexpected_filename_is_still_applied(
    manifest, tmp_path, file_name
) -> None:
    """Rung 1: a filename mismatch must not throw away a whole run's work."""
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    launcher = ScriptedLauncher([{file_name: agent_patch_json(refresh_patch())}])

    frames = [
        item
        async for item in stream_graph_run(
            service,
            launcher,
            "refresh",
            RunRequest(run_truth_scope=["repo-a"]),
            tmp_path / "data",
            execution=_agent_task_execution(
                app.state.background_tasks.store,
                f"unexpected-filename-{file_name}",
            ),
        )
    ]

    assert launcher.calls == 1
    assert _applied_revision(frames) == 3
    assert _events(frames)[-1].event == "done"
    assert "rq/transfer-after-shift" in service.history.state().nodes


@pytest.mark.asyncio
async def test_invalid_patch_is_corrected_in_the_same_native_session(manifest, tmp_path) -> None:
    """Rung 2: hand the concrete problem back to the session holding the analysis."""
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    launcher = ScriptedLauncher(
        [
            {"patch.json": agent_patch_json(shape_invalid_patch())},
            {"patch.json": agent_patch_json(refresh_patch())},
        ]
    )

    frames = [
        item
        async for item in stream_graph_run(
            service,
            launcher,
            "refresh",
            RunRequest(run_truth_scope=["repo-a"]),
            tmp_path / "data",
            execution=_agent_task_execution(
                app.state.background_tasks.store,
                "invalid-patch-correction",
            ),
        )
    ]

    assert launcher.calls == 2
    assert _applied_revision(frames) == 3
    assert service.history.state().revision == 3
    # The second launch continues the first native session rather than starting over.
    assert launcher.resumed_sessions == [None, launcher.native_session_id]
    correction = launcher.prompts[1]
    assert len(correction.splitlines()) < 200
    assert "does not match the graph operation schema" not in correction
    assert "set_ontology" not in correction
    correction_inputs = launcher.input_snapshots[1]
    diagnostic = next(
        value for name, value in correction_inputs.items() if name.endswith("correction-1.json")
    )
    contract = next(
        value for name, value in correction_inputs.items() if name.endswith("correction-1.md")
    )
    assert "does not match the graph operation schema" in diagnostic
    assert "set_ontology" in diagnostic
    assert str(launcher.workspaces[0] / "patch.json") in contract


@pytest.mark.asyncio
async def test_correction_rounds_are_bounded_instead_of_looping(manifest, tmp_path) -> None:
    """Rung 3 plus the round limit: no patch at all, corrected only twice."""
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    launcher = ScriptedLauncher([{}])

    frames = [
        item
        async for item in stream_graph_run(
            service,
            launcher,
            "refresh",
            RunRequest(run_truth_scope=["repo-a"]),
            tmp_path / "data",
        )
    ]

    assert launcher.calls == PATCH_CORRECTION_MAX_ROUNDS + 1
    assert launcher.resumed_sessions == [None] + [launcher.native_session_id] * (
        PATCH_CORRECTION_MAX_ROUNDS
    )
    assert _applied_revision(frames) is None
    assert any("without writing any JSON file" in text for text in _error_texts(frames))
    assert service.history.state().revision == 2


def test_failed_run_retains_its_patch_and_scratch_folder(manifest, tmp_path) -> None:
    """Retention: a failure keeps its evidence; a success cleans up after itself."""
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    store = app.state.background_tasks.store
    rejected = agent_patch_json(gated_patch())

    failed_execution = _agent_task_execution(store, "failed-operation")
    failed_launcher = ScriptedLauncher([{"patch.json": rejected}])
    failed_frames = _drain(
        stream_graph_run(
            service,
            failed_launcher,
            "refresh",
            RunRequest(run_truth_scope=["repo-a"]),
            tmp_path / "data",
            execution=failed_execution,
        )
    )

    assert _error_texts(failed_frames)
    assert _applied_revision(failed_frames) is None
    failed_stage = failed_launcher.workspaces[0]
    assert failed_stage.is_dir()
    assert (failed_stage / "patch.json").is_file()
    # The patch is persisted before validation, so the run's work survives.
    assert store.agent_task_patch_output("failed-operation") == rejected
    assert store.agent_task("failed-operation").stage_root == str(failed_stage)
    store.fail_agent_task("failed-operation", _error_texts(failed_frames)[0])

    applied_execution = _agent_task_execution(store, "applied-operation")
    applied_launcher = ScriptedLauncher([{"patch.json": agent_patch_json(refresh_patch())}])
    applied_frames = _drain(
        stream_graph_run(
            service,
            applied_launcher,
            "refresh",
            RunRequest(run_truth_scope=["repo-a"]),
            tmp_path / "data",
            execution=applied_execution,
        )
    )

    # Rejected agent drafts never enter canonical history, so the next accepted
    # patch receives the next coherent revision.
    assert _applied_revision(applied_frames) == 3
    assert service.history.state().revision == 3
    assert "rq/transfer-after-shift" in service.history.state().nodes
    assert not applied_launcher.workspaces[0].exists()
    assert store.agent_task("applied-operation").stage_root is None
    # The failed run's folder is untouched by the successful one.
    assert failed_stage.is_dir()


def test_patch_collector_prefers_patch_json_and_refuses_ambiguity(tmp_path) -> None:
    patch = agent_patch_json(refresh_patch())
    (tmp_path / "notes.json").write_text('{"kind": "refresh"}', encoding="utf-8")
    (tmp_path / "output.json").write_text(patch, encoding="utf-8")
    (tmp_path / "patch.json").write_text(patch, encoding="utf-8")

    text, name = _collect_patch_text(tmp_path, None)

    assert name == "patch.json"
    assert text == patch

    # Without the canonical name, two equally plausible candidates are ambiguous
    # and must not be guessed at.
    (tmp_path / "patch.json").unlink()
    (tmp_path / "backup.json").write_text(patch, encoding="utf-8")
    with pytest.raises(AgentOutputProblem, match="more than one patch-shaped JSON file"):
        _collect_patch_text(tmp_path, None)


def test_local_state_repository_is_read_in_place_instead_of_copied(manifest, tmp_path) -> None:
    """The canonical `.research/` on the execution machine is pointed at, never staged."""
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    context = service.assemble_run(RunRequest(run_truth_scope=["repo-a"]), surface="refresh")

    class RecordingStage:
        def __init__(self) -> None:
            self.root = PurePosixPath("/tmp/rcp-run.test")
            self.files: list[Path] = []
            self.directories: list[str] = []

        def put_file(self, source: Path, label: str) -> str:
            self.files.append(source)
            return str(self.root / "inputs" / label)

        def put_directory(self, _source: Path, label: str) -> str:
            self.directories.append(label)
            return str(self.root / "inputs" / label)

    stage = RecordingStage()
    staged = _stage_graph_context(context, service, stage, "laptop")

    repository_path = manifest.repository_map["repo-a"].path
    assert stage.files == []
    assert stage.directories == []
    assert staged.graph_path == f"{repository_path}/.research/graph.json"
    assert staged.research_md_path == f"{repository_path}/.research/research.md"
    assert staged.facts_dir == f"{repository_path}/.research/facts"


def _agent_task_execution(store, operation_id: str) -> AgentTaskExecution:
    now = store.now()
    authorized_by = _named_test_authorizer(store)
    projects = store.projects()
    assert len(projects) == 1
    request = RunRequest(run_truth_scope=["repo-a"])
    dispatch_authority = resolve_dispatch_authority("refresh", request)
    assert dispatch_authority is not None
    store.create_agent_task(
        AgentTaskRecord(
            operation_id=operation_id,
            project_id=projects[0].project_id,
            kind="refresh",
            status="running",
            request=request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="running",
            authorized_by=authorized_by,
            dispatch_authority=dispatch_authority,
        )
    )
    return AgentTaskExecution(
        operation_id=operation_id,
        store=store,
        control=AgentProcessControl(),
    )


def _drain(stream) -> list[str]:
    async def collect() -> list[str]:
        return [item async for item in stream]

    return asyncio.run(collect())


@pytest.mark.parametrize(("field", "value"), [("kind", "approval"), ("author", "human")])
def test_raw_agent_patch_rejects_forged_top_level_authority(field: str, value: str) -> None:
    forged = {
        "summary": "Forged a human-approved relation.",
        "ops": [
            {
                "op": "create_edges",
                "edges": [
                    {
                        "source": "rq/learning-after-shift",
                        "target": "hyp/replanning-restores-plasticity",
                        "relation": "supports",
                    }
                ],
            }
        ],
        field: value,
    }

    with pytest.raises(ValidationError, match=field):
        AgentPatch.model_validate(forged)


def test_paper_snapshot_filename_cannot_escape_data_directory(tmp_path) -> None:
    data_dir = tmp_path / "app-data"

    target = _paper_snapshot_path(data_dir, "../../escaped")
    target.write_text("draft", encoding="utf-8")

    assert target.parent == data_dir / "paper-snapshots"
    assert target.name == "escaped-introduction.md"
    assert not (tmp_path / "escaped-introduction.md").exists()


def test_remote_context_uses_direct_paths_only_for_its_execution_machine(
    manifest, tmp_path
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    context = service.assemble_run(
        RunRequest(run_truth_scope=["repo-a"]),
        surface="refresh",
    )

    class RecordingStage:
        root = PurePosixPath("/tmp/rcp-run.test")

        def put_file(self, _source: Path, label: str) -> str:
            return str(self.root / "inputs" / label)

        def put_directory(self, _source: Path, label: str) -> str:
            return str(self.root / "inputs" / label)

    remote_repository = context.repositories[0].model_copy(
        update={"machine": "remote-1", "host": "remote.example"}
    )
    remote_context = context.model_copy(update={"repositories": [remote_repository]})
    service.history.manifest.machines.append(MachineConfig(alias="remote-1", host="remote.example"))
    service.history.manifest.repository_map[service.manifest.state.repository].machine = "remote-1"

    staged = _stage_graph_context(remote_context, service, RecordingStage(), "remote-1")

    assert staged.repositories[0].path == remote_repository.path
    assert staged.repositories[0].host == ""

    with pytest.raises(StateUnavailable, match="has no SSH host reachable"):
        _stage_graph_context(context, service, RecordingStage(), "remote-1")


@pytest.mark.asyncio
async def test_remote_stage_is_retained_after_failure_and_after_pause(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    context = service.assemble_run(
        RunRequest(run_truth_scope=["repo-a"]),
        surface="refresh",
    )
    context = context.model_copy(
        update={
            "repositories": [
                repository.model_copy(update={"host": "laptop.example"})
                for repository in context.repositories
            ]
        }
    )
    service.history.manifest.machines.append(
        MachineConfig(alias="remote-1", host="research.example")
    )
    service.history.manifest.repository_map[service.manifest.state.repository].machine = "remote-1"
    service.history.manifest.agent.refresh.run_on = "remote-1"
    monkeypatch.setattr(service.history, "_reload_manifest", lambda: None)
    monkeypatch.setattr(service, "assemble_run", lambda *_args, **_kwargs: context)

    class FakeRemoteStage:
        instances = []
        finalize_error = False

        def __init__(self, _host):
            self.root = None
            self.closed = False
            self.finalized = 0
            self.workspace_text: dict[str, str] = {}
            self.instances.append(self)

        @property
        def workspace(self):
            assert self.root is not None
            return self.root / "workspace"

        def open(self, operation_id=None):
            self.root = PurePosixPath(f"/tmp/rcp-run.{operation_id}")
            return self

        def put_file(self, _source, label):
            assert self.root is not None
            return str(self.root / "inputs" / label)

        def put_directory(self, _source, label, *, reuse=False):
            assert self.root is not None
            assert reuse is False
            return str(self.root / "inputs" / label)

        def finalize_inputs(self):
            self.finalized += 1
            if self.finalize_error:
                raise StateUnavailable("remote input batch was incomplete")

        def list_workspace_files(self):
            assert self.root is not None
            return []

        def list_workspace_entries(self):
            assert self.root is not None
            return sorted(self.workspace_text)

        def write_workspace_text(self, name, content):
            assert self.root is not None
            self.workspace_text[name] = content

        def read_workspace_text(self, name, *, max_bytes=None):
            assert self.root is not None
            content = self.workspace_text[name]
            if max_bytes is not None and len(content.encode("utf-8")) > max_bytes:
                raise ValueError("remote workspace file exceeds the test byte limit")
            return content

        def remove_workspace_file(self, name):
            assert self.root is not None
            self.workspace_text.pop(name, None)

        def close(self):
            self.closed = True
            self.root = None
            return True

    monkeypatch.setattr("rcp.runs.tasks.graph.RemoteRunStage", FakeRemoteStage)
    request = RunRequest(
        provider="codex",
        run_truth_scope=["repo-a"],
        run_on="remote-1",
    )

    failed_execution = _agent_task_execution(
        app.state.background_tasks.store,
        "remote-failure",
    )
    failed_launcher = FakeLauncher([AgentEvent(event="error", text="provider failed")])
    failed_events = [
        item
        async for item in stream_graph_run(
            service,
            failed_launcher,
            "refresh",
            request,
            tmp_path / "data",
            execution=failed_execution,
        )
    ]

    assert any("provider failed" in item for item in failed_events)
    # A failed remote run keeps its scratch folder for inspection and for a
    # Resume that points the native session back at the same directory.
    assert FakeRemoteStage.instances[0].closed is False
    assert FakeRemoteStage.instances[0].finalized == 2
    assert failed_launcher.calls == 1

    paused_execution = _agent_task_execution(
        app.state.background_tasks.store,
        "remote-pause",
    )
    paused_launcher = FakeLauncher([AgentEvent(event="paused", text="paused")])
    paused_events = [
        item
        async for item in stream_graph_run(
            service,
            paused_launcher,
            "refresh",
            request,
            tmp_path / "data",
            execution=paused_execution,
        )
    ]

    assert any('"event":"paused"' in item for item in paused_events)
    assert FakeRemoteStage.instances[1].closed is False
    assert FakeRemoteStage.instances[1].finalized == 2
    assert paused_launcher.calls == 1

    FakeRemoteStage.finalize_error = True
    blocked_launcher = FakeLauncher([AgentEvent(event="done")])
    blocked_events = [
        item
        async for item in stream_graph_run(
            service,
            blocked_launcher,
            "refresh",
            request,
            tmp_path / "data",
            execution=_agent_task_execution(
                app.state.background_tasks.store,
                "remote-incomplete-inputs",
            ),
        )
    ]

    assert any("remote input batch was incomplete" in item for item in blocked_events)
    assert FakeRemoteStage.instances[2].finalized == 1
    assert blocked_launcher.calls == 0


def test_background_seed_can_pause_inspect_and_resume(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    native_session_id = str(uuid.uuid4())
    resumed_requests: list[RunRequest] = []

    async def pausable_stream(_project_id, _kind, _request, execution):
        yield _event_frame(AgentEvent(event="session", session_id=native_session_id))
        while not execution.control.pause_requested.is_set():
            await asyncio.sleep(0.01)
        yield _event_frame(AgentEvent(event="paused", text="Provider process paused."))

    app.state.background_tasks.stream = pausable_stream
    client = TestClient(app)
    started = client.post(
        f"/api/projects/{project_id}/tasks/seed",
        json={"run_truth_scope": ["repo-a"]},
    )
    operation_id = started.json()["operation_id"]
    _wait_for_status(client, project_id, operation_id, {"running"})

    pause = client.post(f"/api/projects/{project_id}/tasks/{operation_id}/pause")

    assert pause.status_code == 202
    paused = _wait_for_status(client, project_id, operation_id, {"paused"})
    assert paused["can_resume"] is True
    assert paused["can_retry"] is True
    assert paused["native_session_id"] == native_session_id
    assert paused["progress"] < 1
    assert any("Pause requested" in event["message"] for event in paused["events"])

    async def resumed_stream(_project_id, _kind, request, execution):
        resumed_requests.append(request)
        assert execution.continuation == "resume"
        assert execution.reuses_native_checkpoint is True
        assert execution.retry_feedback == ()
        yield _event_frame(AgentEvent(event="message", text=json.dumps({"applied_revision": 1})))
        yield _event_frame(AgentEvent(event="done"))

    app.state.background_tasks.stream = resumed_stream
    resumed = client.post(f"/api/projects/{project_id}/tasks/{operation_id}/resume")

    assert resumed.status_code == 202
    resumed_id = resumed.json()["operation_id"]
    completed = _wait_for_run(client, project_id, resumed_id)
    assert completed["status"] == "succeeded"
    assert completed["parent_operation_id"] == operation_id
    assert completed["attempt"] == 2
    assert completed["progress"] == 1
    assert resumed_requests[0].session_id == native_session_id
    created = next(
        item for item in completed["debug_receipts"] if item["category"] == "operation_created"
    )
    assert created["payload"]["continuation_cause"] == "resume"
    assert any("Resuming task" in item["message"] for item in completed["events"])
    assert not any("prior failure diagnostics" in item["message"] for item in completed["events"])


def test_background_shutdown_records_reload_provenance_not_human_pause(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id

    async def pausable_stream(_project_id, _kind, _request, execution):
        yield _event_frame(AgentEvent(event="session", session_id="shutdown-session"))
        while not execution.control.pause_requested.is_set():
            await asyncio.sleep(0.01)
        yield _event_frame(AgentEvent(event="paused", text="Provider process paused."))

    app.state.background_tasks.stream = pausable_stream
    client = TestClient(app)
    started = client.post(f"/api/projects/{project_id}/tasks/seed", json={})
    operation_id = started.json()["operation_id"]
    _wait_for_status(client, project_id, operation_id, {"running"})

    app.state.background_tasks.shutdown(timeout=2)

    paused = _wait_for_status(client, project_id, operation_id, {"paused"})
    messages = [event["message"] for event in paused["events"]]
    assert "Paused for RCP shutdown or reload." in messages
    assert "Pause requested by the human." not in messages


@pytest.mark.asyncio
async def test_closing_paused_background_stream_awaits_launcher_cleanup(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    project_id = app.state.default_project_id
    request = RunRequest(
        chat_scope="project",
        chat_id=str(uuid.uuid4()),
        message="Pause while reading the project.",
        run_truth_scope=["repo-a"],
    )
    operation_id = "paused-stream-cleanup"
    store = app.state.background_tasks.store
    now = store.now()
    store.create_agent_task(
        AgentTaskRecord(
            operation_id=operation_id,
            project_id=project_id,
            kind="project_chat",
            status="running",
            request=request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="running",
        )
    )
    execution = AgentTaskExecution(
        operation_id=operation_id,
        store=store,
        control=AgentProcessControl(),
    )
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()
    provider_streams = []

    async def provider_stream():
        try:
            yield AgentEvent(event="paused", text="Provider process paused.")
        finally:
            cleanup_started.set()
            await allow_cleanup.wait()
            cleanup_finished.set()

    def launch(*_args, **_kwargs):
        stream = provider_stream()
        provider_streams.append(stream)
        return stream

    monkeypatch.setattr(app.state.catalog.launcher, "stream", launch)
    stream = app.state.background_tasks.stream(
        project_id,
        "project_chat",
        request,
        execution,
    )

    paused = AgentEvent.model_validate_json((await anext(stream)).removeprefix("data: ").strip())
    assert paused.event == "paused"

    close = asyncio.create_task(stream.aclose())
    try:
        await asyncio.sleep(0)
        assert cleanup_started.is_set()
        assert not close.done()
    finally:
        allow_cleanup.set()
        await close
        if provider_streams and not cleanup_finished.is_set():
            await provider_streams[0].aclose()

    assert cleanup_finished.is_set()


def test_failed_background_seed_can_retry_without_native_session(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id

    async def failed_stream(*_args):
        yield _event_frame(AgentEvent(event="error", text="provider connection dropped"))

    app.state.background_tasks.stream = failed_stream
    client = TestClient(app)
    started = client.post(f"/api/projects/{project_id}/tasks/seed", json={})
    failed = _wait_for_run(client, project_id, started.json()["operation_id"])
    assert failed["can_resume"] is False
    assert failed["can_retry"] is True

    async def retry_stream(_project_id, _kind, request, _execution):
        assert request.session_id is None
        assert request.provider == "codex"
        assert request.model == "gpt-5.6-codex"
        assert request.reasoning == "high"
        assert _execution.continuation == "handoff"
        assert _execution.reuses_native_checkpoint is False
        assert _execution.retry_feedback == (
            "Attempt 1 (failed) failed with: provider connection dropped",
        )
        yield _event_frame(AgentEvent(event="message", text=json.dumps({"applied_revision": 1})))
        yield _event_frame(AgentEvent(event="done"))

    app.state.background_tasks.stream = retry_stream
    retried = client.post(
        f"/api/projects/{project_id}/tasks/{failed['operation_id']}/retry",
        json={
            "provider": "codex",
            "model": "gpt-5.6-codex",
            "reasoning": "high",
        },
    )

    assert retried.status_code == 202
    completed = _wait_for_run(client, project_id, retried.json()["operation_id"])
    assert completed["status"] == "succeeded"
    assert completed["parent_operation_id"] == failed["operation_id"]
    assert "native_resume_unavailable" in {item["category"] for item in completed["debug_receipts"]}
    assert any("starting a clean retry" in item["message"] for item in completed["events"])


def test_same_provider_retry_resumes_owned_checkpoint(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    stage = tmp_path / "retained-stage"
    stage.mkdir()

    async def failed_stream(_project_id, _kind, _request, execution):
        execution.checkpoint_stage("", str(stage))
        yield _event_frame(AgentEvent(event="session", session_id="owned-session"))
        yield _event_frame(AgentEvent(event="error", text="provider connection dropped"))

    app.state.background_tasks.stream = failed_stream
    client = TestClient(app)
    started = client.post(f"/api/projects/{project_id}/tasks/seed", json={"provider": "codex"})
    failed = _wait_for_run(client, project_id, started.json()["operation_id"])

    async def resumed_stream(_project_id, _kind, request, execution):
        assert request.session_id == "owned-session"
        assert execution.continuation == "retry"
        assert execution.reuses_native_checkpoint is True
        assert execution.retry_feedback == (
            "Attempt 1 (failed) failed with: provider connection dropped",
        )
        assert execution.stage_root == str(stage)
        yield _event_frame(AgentEvent(event="message", text=json.dumps({"applied_revision": 1})))
        yield _event_frame(AgentEvent(event="done"))

    app.state.background_tasks.stream = resumed_stream
    retried = client.post(
        f"/api/projects/{project_id}/tasks/{failed['operation_id']}/retry", json={}
    )
    completed = _wait_for_run(client, project_id, retried.json()["operation_id"])

    assert completed["status"] == "succeeded"
    assert completed["native_session_id"] == "owned-session"
    created = next(
        item for item in completed["debug_receipts"] if item["category"] == "operation_created"
    )
    assert created["payload"]["continuation_cause"] == "retry"
    assert any("Retrying task" in item["message"] for item in completed["events"])
    assert any("prior failure diagnostics" in item["message"] for item in completed["events"])


def test_same_provider_session_limit_retry_starts_clean(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    stage = tmp_path / "exhausted-stage"
    stage.mkdir()

    async def failed_stream(_project_id, _kind, _request, execution):
        execution.checkpoint_stage("", str(stage))
        yield _event_frame(AgentEvent(event="session", session_id="exhausted-session"))
        yield _event_frame(AgentEvent(event="error", text="You've hit your limit"))

    app.state.background_tasks.stream = failed_stream
    client = TestClient(app)
    started = client.post(f"/api/projects/{project_id}/tasks/seed", json={"provider": "claude"})
    failed = _wait_for_run(client, project_id, started.json()["operation_id"])

    async def clean_stream(_project_id, _kind, request, execution):
        assert request.session_id is None
        assert execution.continuation == "handoff"
        assert execution.reuses_native_checkpoint is False
        assert execution.stage_root is None
        yield _event_frame(AgentEvent(event="message", text=json.dumps({"applied_revision": 1})))
        yield _event_frame(AgentEvent(event="done"))

    app.state.background_tasks.stream = clean_stream
    retried = client.post(
        f"/api/projects/{project_id}/tasks/{failed['operation_id']}/retry", json={}
    )
    completed = _wait_for_run(client, project_id, retried.json()["operation_id"])

    assert completed["status"] == "succeeded"
    assert "native_resume_skipped" in {item["category"] for item in completed["debug_receipts"]}
    assert any("session limit was exhausted" in item["message"] for item in completed["events"])


def test_retry_reuse_and_handoff_fallback_events_include_concrete_reasons(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    now = store.now()
    store.create_agent_task(
        AgentTaskRecord(
            operation_id="retry",
            project_id="project",
            kind="seed",
            status="running",
            request={},
            created_at=now,
            updated_at=now,
            status_message="running",
        )
    )
    execution = AgentTaskExecution(operation_id="retry", store=store, control=AgentProcessControl())

    _record_context_reuse(
        execution, reused=False, reason="attempt 2 has no prepared context metadata"
    )
    _record_progress_handoff(
        execution, handed_off=False, reason="attempt 2 left no retained provider progress"
    )

    messages = [item.message for item in store.agent_task_events("retry")]
    assert any("Reason: attempt 2 has no prepared context metadata" in item for item in messages)
    assert any("Reason: attempt 2 left no retained provider progress" in item for item in messages)


def test_seed_quota_failure_retries_with_new_provider_and_reuses_context(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    service = app.state.service
    original_assemble = service.assemble_run
    assemble_calls = 0

    def counted_assemble(*args, **kwargs):
        nonlocal assemble_calls
        assemble_calls += 1
        return original_assemble(*args, **kwargs)

    monkeypatch.setattr(service, "assemble_run", counted_assemble)

    class ProviderSwitchLauncher:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def stream(self, provider, prompt, **kwargs):
            workspace = Path(kwargs["cwd"])
            inputs = workspace / "inputs"
            self.calls.append(
                {
                    "provider": provider,
                    "prompt": prompt,
                    "session_id": kwargs.get("session_id"),
                    "workspace": str(workspace),
                    "inputs": {
                        item.name: item.read_text(encoding="utf-8")
                        for item in inputs.iterdir()
                        if item.is_file()
                    },
                }
            )
            if len(self.calls) == 1:
                transcript = Path(manifest.sources.claude_roots[0]) / "claude-native-session.jsonl"
                transcript.write_text(
                    json.dumps(
                        {
                            "uuid": "partial-work",
                            "sessionId": "claude-native-session",
                            "cwd": manifest.repository_map["repo-a"].path,
                            "timestamp": "2026-07-31T00:00:00Z",
                            "type": "assistant",
                            "message": {"content": "partial synthesis from provider A"},
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                yield AgentEvent(event="session", session_id="claude-native-session")
                yield AgentEvent(event="message", text="Indexed the corpus and formed a draft.")
                yield AgentEvent(
                    event="error",
                    text="You've hit your limit · resets 5pm (Asia/Shanghai)",
                )
                return
            (workspace / "patch.json").write_text(agent_patch_json(seed_patch()), encoding="utf-8")
            yield AgentEvent(event="session", session_id="codex-native-session")
            yield AgentEvent(event="done")

    launcher = ProviderSwitchLauncher()

    async def stream(_project_id, kind, request, execution):
        async for frame in stream_graph_run(
            service,
            launcher,
            kind,
            request,
            tmp_path / "data",
            execution=execution,
        ):
            yield frame

    app.state.background_tasks.stream = stream
    client = TestClient(app)
    started = client.post(
        f"/api/projects/{project_id}/tasks/seed",
        json={"provider": "claude", "run_truth_scope": ["repo-a"]},
    )
    failed = _wait_for_run(client, project_id, started.json()["operation_id"])

    assert failed["status"] == "failed"
    assert failed["error"] == "You've hit your limit · resets 5pm (Asia/Shanghai)"
    failed_categories = {item["category"] for item in failed["debug_receipts"]}
    assert "patch_collection_skipped" in failed_categories
    assert "patch_retained" not in failed_categories
    assert "patch_correction_requested" not in failed_categories

    # A paused legacy retry made no provider dispatch and has neither prepared
    # context nor a native session. It must not hide attempt 1's useful work.
    store = app.state.catalog.store
    now = store.now()
    empty_stage = tmp_path / "attempt-2-stage"
    (empty_stage / "inputs").mkdir(parents=True)
    failed_request = RunRequest.model_validate(failed["request"])
    failed_dispatch_authority = resolve_dispatch_authority("seed", failed_request)
    assert failed_dispatch_authority is not None
    attempt_2 = store.create_agent_task(
        AgentTaskRecord(
            operation_id="attempt-2-without-progress",
            project_id=project_id,
            kind="seed",
            status="paused",
            request=failed["request"],
            created_at=now,
            updated_at=now,
            status_message="Paused before provider launch.",
            attempt=2,
            parent_operation_id=failed["operation_id"],
            stage_root=str(empty_stage),
            dispatch_authority=failed_dispatch_authority,
        )
    )

    retried = client.post(
        f"/api/projects/{project_id}/tasks/{attempt_2.operation_id}/retry",
        json={"provider": "codex"},
    )
    assert retried.status_code == 202
    completed = _wait_for_run(client, project_id, retried.json()["operation_id"])

    assert completed["status"] == "succeeded"
    assert completed["applied_revision"] == 2
    assert assemble_calls == 1
    assert [item["provider"] for item in launcher.calls] == ["claude", "codex"]
    assert launcher.calls[1]["session_id"] is None
    retry_prompt = str(launcher.calls[1]["prompt"])
    retry_contract_path = Path(retry_prompt.splitlines()[1])
    assert retry_prompt == PromptFactory.launch_prompt(str(retry_contract_path))
    assert completed["parent_operation_id"] == attempt_2.operation_id
    assert not any(name.endswith("handoff.json") for name in launcher.calls[1]["inputs"])
    completed_categories = {item["category"] for item in completed["debug_receipts"]}
    assert "context_reused" in completed_categories
    assert "context_reuse_unavailable" not in completed_categories
    assert "progress_handoff_unavailable" in completed_categories
    first_base = store.agent_task_contract(failed["operation_id"], "base")
    retry_base = store.agent_task_contract(completed["operation_id"], "base")
    assert first_base is not None
    assert retry_base is not None
    assert retry_base != first_base
    retry_workspace = Path(str(launcher.calls[1]["workspace"]))
    assert str(retry_workspace / "patch.json") in retry_base
    assert str(Path(failed["stage_root"]) / "patch.json") not in retry_base
    assert str(retry_workspace / "inputs") in retry_base
    assert f"task-{completed['operation_id']}-patch-schema.json" in retry_base
    assert "authorized-session-keys.json" not in retry_base
    assert f"task-{completed['operation_id']}-retry-diagnostics.json" in retry_base
    retry_contract = store.agent_task_contract(completed["operation_id"], "retry")
    assert retry_contract is None


def test_clean_retry_without_progress_uses_reused_context_and_fresh_base(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    service = app.state.service
    original_assemble = service.assemble_run
    assemble_calls = 0

    def counted_assemble(*args, **kwargs):
        nonlocal assemble_calls
        assemble_calls += 1
        return original_assemble(*args, **kwargs)

    monkeypatch.setattr(service, "assemble_run", counted_assemble)

    class NoProgressLauncher:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def stream(self, _provider, prompt, **kwargs):
            workspace = Path(kwargs["cwd"])
            self.calls.append(
                {
                    "prompt": prompt,
                    "workspace": str(workspace),
                    "inputs": {
                        item.name: item.read_text(encoding="utf-8")
                        for item in (workspace / "inputs").iterdir()
                        if item.is_file()
                    },
                }
            )
            if len(self.calls) == 1:
                yield AgentEvent(event="error", text="provider connection dropped before work")
                return
            (workspace / "patch.json").write_text(agent_patch_json(seed_patch()), encoding="utf-8")
            yield AgentEvent(event="done")

    launcher = NoProgressLauncher()

    async def stream(_project_id, kind, request, execution):
        async for frame in stream_graph_run(
            service, launcher, kind, request, tmp_path / "data", execution=execution
        ):
            yield frame

    app.state.background_tasks.stream = stream
    client = TestClient(app)
    started = client.post(f"/api/projects/{project_id}/tasks/seed", json={})
    failed = _wait_for_run(client, project_id, started.json()["operation_id"])
    retried = client.post(
        f"/api/projects/{project_id}/tasks/{failed['operation_id']}/retry", json={}
    )
    completed = _wait_for_run(client, project_id, retried.json()["operation_id"])

    assert completed["status"] == "succeeded"
    assert assemble_calls == 1
    store = app.state.catalog.store
    first_base = store.agent_task_contract(failed["operation_id"], "base")
    retry_base = store.agent_task_contract(completed["operation_id"], "base")
    assert first_base is not None
    assert retry_base is not None
    assert retry_base != first_base
    assert store.agent_task_contract(completed["operation_id"], "retry") is None
    retry_prompt = str(launcher.calls[1]["prompt"])
    retry_base_path = Path(retry_prompt.splitlines()[1])
    assert retry_prompt == PromptFactory.launch_prompt(str(retry_base_path))
    assert retry_base_path.name == f"task-{completed['operation_id']}-base.md"
    retry_workspace = Path(str(launcher.calls[1]["workspace"]))
    assert str(retry_workspace / "patch.json") in retry_base
    assert str(Path(failed["stage_root"]) / "patch.json") not in retry_base
    diagnostics = json.loads(
        next(
            value
            for name, value in launcher.calls[1]["inputs"].items()
            if name.endswith("retry-diagnostics.json")
        )
    )
    assert diagnostics == {
        "prior_attempt_diagnostics": [
            "Attempt 1 (failed) failed with: provider connection dropped before work"
        ]
    }
    categories = {item["category"] for item in completed["debug_receipts"]}
    assert "context_reused" in categories
    assert "progress_handoff_unavailable" in categories


def test_same_provider_retry_reuses_its_session_with_a_followup_only(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    service = app.state.service
    original_assemble = service.assemble_run
    assemble_calls = 0

    def counted_assemble(*args, **kwargs):
        nonlocal assemble_calls
        assemble_calls += 1
        return original_assemble(*args, **kwargs)

    monkeypatch.setattr(service, "assemble_run", counted_assemble)

    class RetryLauncher:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def stream(self, _provider, prompt, **kwargs):
            workspace = Path(kwargs["cwd"])
            self.calls.append(
                {
                    "prompt": prompt,
                    "session_id": kwargs.get("session_id"),
                    "workspace": str(workspace),
                    "inputs": {
                        item.name: item.read_text(encoding="utf-8")
                        for item in (workspace / "inputs").iterdir()
                        if item.is_file()
                    },
                }
            )
            if len(self.calls) == 1:
                yield AgentEvent(event="session", session_id="owned-seed-session")
                yield AgentEvent(event="error", text="provider connection dropped")
                return
            (workspace / "patch.json").write_text(agent_patch_json(seed_patch()), encoding="utf-8")
            yield AgentEvent(event="session", session_id="owned-seed-session")
            yield AgentEvent(event="done")

    launcher = RetryLauncher()

    async def stream(_project_id, kind, request, execution):
        async for frame in stream_graph_run(
            service,
            launcher,
            kind,
            request,
            tmp_path / "data",
            execution=execution,
        ):
            yield frame

    app.state.background_tasks.stream = stream
    client = TestClient(app)
    started = client.post(
        f"/api/projects/{project_id}/tasks/seed",
        json={"provider": "codex", "run_truth_scope": ["repo-a"]},
    )
    failed = _wait_for_run(client, project_id, started.json()["operation_id"])
    retried = client.post(
        f"/api/projects/{project_id}/tasks/{failed['operation_id']}/retry", json={}
    )
    completed = _wait_for_run(client, project_id, retried.json()["operation_id"])

    assert completed["status"] == "succeeded"
    assert assemble_calls == 1
    assert launcher.calls[1]["session_id"] == "owned-seed-session"
    retry_prompt = str(launcher.calls[1]["prompt"])
    retry_contract_path = Path(retry_prompt.splitlines()[1])
    assert retry_prompt == PromptFactory.launch_prompt(str(retry_contract_path))
    retry_diagnostics = json.loads(
        next(
            value
            for name, value in launcher.calls[1]["inputs"].items()
            if name.endswith("retry-diagnostics.json")
        )
    )
    assert retry_diagnostics == {
        "prior_attempt_diagnostics": ["Attempt 1 (failed) failed with: provider connection dropped"]
    }
    store = app.state.catalog.store
    first_base = store.agent_task_contract(failed["operation_id"], "base")
    retry_base = store.agent_task_contract(completed["operation_id"], "base")
    retry_contract = store.agent_task_contract(completed["operation_id"], "retry")
    assert first_base is not None
    assert retry_contract is not None
    # The session still holds the original contract, so this attempt rebuilds nothing.
    assert retry_base is None
    assert "Retry context:" not in retry_contract
    assert "same native session that ran the previous attempt" in retry_contract
    assert f"task-{failed['operation_id']}-initial.md" in retry_contract
    # It names only what changed for this attempt.
    assert f"task-{completed['operation_id']}-patch-schema.json" in retry_contract
    assert f"task-{failed['operation_id']}-patch-schema.json" not in retry_contract
    assert f"task-{completed['operation_id']}-retry-diagnostics.json" in retry_contract
    assert "Patch-only correction authority" not in retry_contract
    assert "provider connection dropped" in next(
        value
        for name, value in launcher.calls[1]["inputs"].items()
        if name.endswith("retry-diagnostics.json")
    )
    launches = [item for item in completed["debug_receipts"] if item["category"] == "agent_launch"]
    assert launches[0]["payload"]["continuation_cause"] == "retry"
    assert launches[0]["payload"]["launch_kind"] == "retry"


def test_retry_launch_refuses_a_patch_it_did_not_write(manifest, tmp_path) -> None:
    """A Retry reuses its predecessor's stage, patch file included.

    A provider that writes nothing must not have that earlier file collected as
    its own work: applying it would attribute inherited output to the Retry.
    """
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    service = app.state.service

    class StaleLauncher:
        def __init__(self) -> None:
            self.workspaces: list[str] = []

        async def stream(self, _provider, _prompt, **kwargs):
            workspace = Path(kwargs["cwd"])
            self.workspaces.append(str(workspace))
            if not self.workspaces[:-1]:
                # The first attempt writes a patch, then loses its connection
                # before RCP ever collects it.
                (workspace / "patch.json").write_text(
                    agent_patch_json(seed_patch()), encoding="utf-8"
                )
                yield AgentEvent(event="session", session_id="owned-seed-session")
                yield AgentEvent(event="error", text="provider connection dropped")
                return
            # Every later launch does no work and writes nothing.
            yield AgentEvent(event="session", session_id="owned-seed-session")
            yield AgentEvent(event="done")

    launcher = StaleLauncher()

    async def stream(_project_id, kind, request, execution):
        async for frame in stream_graph_run(
            service, launcher, kind, request, tmp_path / "data", execution=execution
        ):
            yield frame

    app.state.background_tasks.stream = stream
    client = TestClient(app)
    started = client.post(
        f"/api/projects/{project_id}/tasks/seed",
        json={"provider": "codex", "run_truth_scope": ["repo-a"]},
    )
    failed = _wait_for_run(client, project_id, started.json()["operation_id"])
    assert failed["status"] == "failed"

    retried = client.post(
        f"/api/projects/{project_id}/tasks/{failed['operation_id']}/retry", json={}
    )
    completed = _wait_for_run(client, project_id, retried.json()["operation_id"])

    assert launcher.workspaces[0] == launcher.workspaces[1]
    assert completed["status"] == "failed"
    assert completed["applied_revision"] is None
    assert "provider connection dropped" in completed["error"]
    assert "did not write a new patch" in completed["error"]
    assert completed["error"].index("provider connection dropped") < completed["error"].index(
        "did not write a new patch"
    )
    assert service.history.state().revision == 1
    assert not service.history.state().nodes
    assert any(
        receipt["category"] == "patch_predates_launch" and receipt["payload"]["accepted"] is False
        for receipt in completed["debug_receipts"]
    )


def test_literal_resume_uses_saved_context_without_reassembly(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    service = app.state.service
    original_assemble = service.assemble_run
    assemble_calls = 0

    def counted_assemble(*args, **kwargs):
        nonlocal assemble_calls
        assemble_calls += 1
        return original_assemble(*args, **kwargs)

    monkeypatch.setattr(service, "assemble_run", counted_assemble)

    class ResumeLauncher:
        def __init__(self) -> None:
            self.prompts: list[str] = []
            self.sessions: list[str | None] = []
            self.contracts: list[str] = []

        async def stream(self, _provider, prompt, **kwargs):
            workspace = Path(kwargs["cwd"])
            self.prompts.append(prompt)
            self.sessions.append(kwargs.get("session_id"))
            self.contracts.append(Path(prompt.splitlines()[1]).read_text(encoding="utf-8"))
            if len(self.prompts) == 1:
                yield AgentEvent(event="session", session_id="paused-seed-session")
                yield AgentEvent(event="paused", text="Provider process paused.")
                return
            (workspace / "patch.json").write_text(agent_patch_json(seed_patch()), encoding="utf-8")
            yield AgentEvent(event="session", session_id="paused-seed-session")
            yield AgentEvent(event="done")

    launcher = ResumeLauncher()

    async def stream(_project_id, kind, request, execution):
        async for frame in stream_graph_run(
            service,
            launcher,
            kind,
            request,
            tmp_path / "data",
            execution=execution,
        ):
            yield frame

    app.state.background_tasks.stream = stream
    client = TestClient(app)
    started = client.post(f"/api/projects/{project_id}/tasks/seed", json={})
    paused = _wait_for_run(client, project_id, started.json()["operation_id"])
    assert paused["status"] == "paused"

    resumed = client.post(f"/api/projects/{project_id}/tasks/{paused['operation_id']}/resume")
    completed = _wait_for_run(client, project_id, resumed.json()["operation_id"])

    assert completed["status"] == "succeeded"
    assert assemble_calls == 1
    assert launcher.sessions == [None, "paused-seed-session"]
    resume_contract_path = Path(launcher.prompts[1].splitlines()[1])
    assert launcher.prompts[1] == PromptFactory.launch_prompt(str(resume_contract_path))
    assert launcher.prompts[1] != "Continue the interrupted task."
    resume_contract = launcher.contracts[1]
    assert "# RCP resume contract" in resume_contract
    assert f"task-{paused['operation_id']}-initial.md" in resume_contract
    assert "Prior-attempt diagnostics" not in resume_contract
    assert "Patch-only correction authority" not in resume_contract
    assert (
        app.state.catalog.store.agent_task_contract(completed["operation_id"], "resume")
        == resume_contract
    )
    launch = next(
        item for item in completed["debug_receipts"] if item["category"] == "agent_launch"
    )
    assert launch["payload"]["continuation_cause"] == "resume"


def test_provider_exit_receipt_survives_terminal_error(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    service = app.state.service

    class ExitLauncher:
        async def stream(self, _provider, _prompt, **_kwargs):
            yield AgentEvent(event="session", session_id="failed-session")
            yield AgentEvent(
                event="provider_exit",
                text=json.dumps(
                    {
                        "event_counts": {"session": 1},
                        "explicit_terminal_event": False,
                        "return_code": 7,
                    }
                ),
            )
            yield AgentEvent(event="error", text="provider exited with status 7")

    async def stream(_project_id, kind, request, execution):
        async for frame in stream_graph_run(
            service,
            ExitLauncher(),
            kind,
            request,
            tmp_path / "data",
            execution=execution,
        ):
            yield frame

    app.state.background_tasks.stream = stream
    client = TestClient(app)
    started = client.post(f"/api/projects/{project_id}/tasks/seed", json={})
    failed = _wait_for_run(client, project_id, started.json()["operation_id"])

    assert failed["status"] == "failed"
    receipt = next(item for item in failed["debug_receipts"] if item["category"] == "provider_exit")
    assert receipt["payload"] == {
        "event_counts": {"session": 1},
        "explicit_terminal_event": False,
        "return_code": 7,
        "patch_json_exists": False,
    }


def test_retry_escapes_a_moved_saved_context_instead_of_looping(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    service = app.state.service

    class FailingLauncher:
        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, _provider, _prompt, **_kwargs):
            self.calls += 1
            yield AgentEvent(event="session", session_id="moved-context-session")
            yield AgentEvent(event="error", text="provider connection dropped")

    launcher = FailingLauncher()

    async def graph_stream(_project_id, kind, request, execution):
        async for frame in stream_graph_run(
            service,
            launcher,
            kind,
            request,
            tmp_path / "data",
            execution=execution,
        ):
            yield frame

    app.state.background_tasks.stream = graph_stream
    client = TestClient(app)
    started = client.post(f"/api/projects/{project_id}/tasks/seed", json={})
    failed = _wait_for_run(client, project_id, started.json()["operation_id"])
    append_fixture_patch(service, seed_patch())

    correction = client.post(
        f"/api/projects/{project_id}/tasks/{failed['operation_id']}/retry", json={}
    )
    moved = _wait_for_run(client, project_id, correction.json()["operation_id"])
    assert moved["status"] == "failed"
    assert "graph revision moved from 1 to 2" in moved["error"]
    assert "Retry this task" in moved["error"]
    assert launcher.calls == 1
    assert any(
        item["category"] == "continuation_context_unavailable" for item in moved["debug_receipts"]
    )

    async def clean_stream(_project_id, _kind, request, execution):
        assert request.session_id is None
        assert execution.continuation == "handoff"
        assert execution.stage_root is None
        yield _event_frame(AgentEvent(event="message", text=json.dumps({"applied_revision": 2})))
        yield _event_frame(AgentEvent(event="done"))

    app.state.background_tasks.stream = clean_stream
    escaped = client.post(
        f"/api/projects/{project_id}/tasks/{moved['operation_id']}/retry", json={}
    )
    completed = _wait_for_run(client, project_id, escaped.json()["operation_id"])
    assert completed["status"] == "succeeded"


def test_failed_chat_task_retains_artifacts_emitted_before_the_error(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    append_fixture_patch(app.state.service, seed_patch())
    project_id = app.state.default_project_id
    descriptor = AgentArtifactDescriptor(
        artifact_id="b" * 24,
        name="preview.html",
        media_type="text/html",
    )

    async def failed_stream(*_args):
        yield _event_frame(AgentEvent(event="answer", text="The reply stands."))
        yield _event_frame(AgentEvent(event="artifact", artifact=descriptor))
        yield _event_frame(AgentEvent(event="error", text="graph change was rejected"))

    app.state.background_tasks.stream = failed_stream
    client = TestClient(app)
    started = client.post(
        f"/api/projects/{project_id}/tasks/project_chat",
        json={
            "chat_id": str(uuid.uuid4()),
            "message": "Answer and propose a change.",
            "run_truth_scope": ["repo-a"],
        },
    )
    failed = _wait_for_run(client, project_id, started.json()["operation_id"])

    assert failed["status"] == "failed"
    assert failed["result"] == {
        "messages": ["The reply stands."],
        "artifacts": [
            {
                **descriptor.model_dump(mode="json"),
                "available": False,
                "unavailable_reason": "Artifact bytes are no longer available.",
                "can_open": False,
                "can_download": False,
                "can_keep": False,
                "can_revise": False,
            }
        ],
    }


def test_server_shutdown_pauses_live_background_seed(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id

    async def pausable_stream(_project_id, _kind, _request, execution):
        yield _event_frame(AgentEvent(event="session", session_id=str(uuid.uuid4())))
        while not execution.control.pause_requested.is_set():
            await asyncio.sleep(0.01)
        yield _event_frame(AgentEvent(event="paused"))

    app.state.background_tasks.stream = pausable_stream
    with TestClient(app) as client:
        started = client.post(f"/api/projects/{project_id}/tasks/seed", json={})
        operation_id = started.json()["operation_id"]
        _wait_for_status(client, project_id, operation_id, {"running"})

    persisted = app.state.background_tasks.store.agent_task(operation_id)
    assert persisted is not None
    assert persisted.status == "paused"
    assert persisted.can_resume is True


def test_node_chat_returns_as_task_then_persists_result_and_transcript(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    service.history.update_machine_provider_paths({"laptop": {"codex": "/opt/agents/codex"}})
    append_fixture_patch(service, seed_patch())
    chat_id = str(uuid.uuid4())
    answer = "The node remains proposed because the matched forward test has not run."
    patch = Patch(
        kind="chat",
        author="agent",
        summary="Explained current hypothesis status.",
        ops=[],
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
    )
    launcher = ScriptedLauncher([{"patch.json": agent_patch_json(patch)}], message=answer)
    worker_started = threading.Event()
    release_worker = threading.Event()

    async def stream(_project_id, kind, request, execution):
        assert kind == "node_chat"
        worker_started.set()
        while not release_worker.is_set():
            await asyncio.sleep(0.01)
        async for frame in stream_discuss_run(
            service,
            launcher,
            request,
            tmp_path / "data",
            execution=execution,
        ):
            yield frame

    app.state.background_tasks.stream = stream
    client = TestClient(app)
    started = client.post(
        f"/api/projects/{app.state.default_project_id}/tasks/node_chat",
        json={
            "node_id": "hyp/replanning-restores-plasticity",
            "chat_id": chat_id,
            "message": "Why is this still proposed?",
            "run_truth_scope": ["repo-a"],
        },
    )

    assert started.status_code == 202
    assert worker_started.wait(timeout=1)
    operation_id = started.json()["operation_id"]
    active = client.get(f"/api/projects/{app.state.default_project_id}/tasks/{operation_id}").json()
    assert active["status"] in {"queued", "running"}
    assert client.get("/api/health").status_code == 200

    release_worker.set()
    completed = _wait_for_run(client, app.state.default_project_id, operation_id)
    assert completed["status"] == "succeeded"
    assert completed["kind"] == "node_chat"
    assert completed["result"] == {"messages": [answer]}
    assert launcher.launch_kwargs[0]["binary"] == "/opt/agents/codex"
    # A question costs no graph revision, even when the agent writes a patch file:
    # this turn carried no human authorization and the patch changed nothing anyway.
    assert completed["applied_revision"] is None
    assert service.history.state().revision == 2
    transcript = next((manifest.research_dir / "chat").glob("*.jsonl"))
    records = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines()]
    assert [record["role"] for record in records] == ["user", "assistant"]
    assert records[-1]["text"] == answer
    assert records[-1]["nativeSessionId"] == launcher.native_session_id


def test_new_chat_turn_refuses_resumable_paused_attempt(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    store = app.state.background_tasks.store
    project_id = app.state.default_project_id
    chat_id = str(uuid.uuid4())
    now = store.now()
    store.create_agent_task(
        AgentTaskRecord(
            operation_id="paused-project-chat",
            project_id=project_id,
            kind="project_chat",
            status="paused",
            request={
                "chat_scope": "project",
                "chat_id": chat_id,
                "message": "unfinished question",
            },
            created_at=now,
            updated_at=now,
            status_message="paused",
            native_session_id=str(uuid.uuid4()),
            stage_host="",
            stage_root=str(tmp_path / "paused-stage"),
        )
    )

    response = TestClient(app).post(
        f"/api/projects/{project_id}/tasks/project_chat",
        json={"chat_id": chat_id, "message": "start another turn"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "This conversation has a paused turn. Resume or retry it before starting a new turn."
    )


def test_chat_artifacts_are_bounded_sandboxed_and_independent(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    answer = "# Result\n\n```unknown-language\nkept as code\n```"
    html_source = (
        b"<!doctype html><button id='go'>Run</button>"
        b"<a href='https://google.com'>Reference</a>"
        b"<a href='javascript:alert(1)'>Unsafe</a>"
        b"<img src='https://example.com/tracker.png'>"
        b"<script>document.querySelector('#go').onclick=()=>{go.textContent='Done'}</script>"
    )
    png_source = b"\x89PNG\r\n\x1a\npreview-bytes"

    class ArtifactLauncher(FakeLauncher):
        async def stream(self, *args, **kwargs):
            workspace = Path(kwargs["cwd"])
            artifact_directory = next((workspace / "turns").glob("*/artifacts"))
            artifact_directory.joinpath("preview.html").write_bytes(html_source)
            artifact_directory.joinpath("plot.png").write_bytes(png_source)
            artifact_directory.joinpath("bad.jpg").write_bytes(b"not a jpeg")
            artifact_directory.joinpath("unsupported.svg").write_text("<svg/>")
            nested = artifact_directory / "nested"
            nested.mkdir()
            nested.joinpath("hidden.html").write_text("hidden")
            artifact_directory.joinpath("linked.html").symlink_to(
                artifact_directory / "preview.html"
            )
            workspace.joinpath("patch.json").write_text(agent_patch_json(refresh_patch()))
            artifact_template = str(workspace / "turns")
            assert artifact_template in _local_task_contract(args[1])
            async for event in super().stream(*args, **kwargs):
                yield event

    launcher = ArtifactLauncher([AgentEvent(event="answer", text=answer), AgentEvent(event="done")])

    async def stream(_project_id, _kind, request, execution):
        async for frame in stream_discuss_run(
            service, launcher, request, tmp_path / "data", execution=execution
        ):
            yield frame

    app.state.background_tasks.stream = stream
    client = TestClient(app)
    project_id = app.state.default_project_id
    started = client.post(
        f"/api/projects/{project_id}/tasks/project_chat",
        json={
            "chat_id": str(uuid.uuid4()),
            "message": "Explain this with a preview.",
            "run_truth_scope": ["repo-a"],
        },
    )
    completed = _wait_for_run(client, project_id, started.json()["operation_id"])

    assert completed["status"] == "succeeded"
    assert completed["result"]["messages"] == [answer]
    assert service.history.state().revision == 2
    artifacts = completed["result"]["artifacts"]
    assert [item["name"] for item in artifacts] == [
        "plot.png",
        "preview.html",
        "unsupported.svg",
    ]
    assert all("path" not in item and "host" not in item for item in artifacts)

    by_name = {item["name"]: item for item in artifacts}
    base = f"/api/projects/{project_id}/tasks/{completed['operation_id']}/artifacts"
    html_url = f"{base}/{by_name['preview.html']['artifact_id']}/content"
    html_head = client.head(html_url)
    html_preview = client.get(html_url)
    assert html_head.status_code == 200 and html_head.content == b""
    assert html_preview.status_code == 200
    assert 'sandbox="allow-scripts"' in html_preview.text
    assert "allow-top-navigation" not in html_preview.text
    assert "data-rcp-href=&quot;https://google.com&quot;" in html_preview.text
    assert "&lt;a href=&quot;" not in html_preview.text
    assert "javascript:alert" not in html_preview.text
    assert "https://example.com/tracker.png" not in html_preview.text
    assert "event.isTrusted" in html_preview.text
    assert "value.secret" in html_preview.text
    assert "window.open(target.href,'_blank','noopener,noreferrer')" in html_preview.text
    assert "window.location.assign" not in html_preview.text
    assert "connect-src &amp;#x27;none&amp;#x27;" in html_preview.text

    image_url = f"{base}/{by_name['plot.png']['artifact_id']}/content"
    image = client.get(image_url)
    assert image.status_code == 200 and image.content == png_source
    assert image.headers["content-type"].startswith("image/png")
    assert image.headers["x-content-type-options"] == "nosniff"

    legacy_image_url = f"{base}/{by_name['plot.png']['artifact_id']}/preview"
    legacy_image = client.get(legacy_image_url, headers={"Accept": "image/png,image/*"})
    assert legacy_image.status_code == 200 and legacy_image.content == png_source
    assert legacy_image.headers["content-type"].startswith("image/png")

    download_url = f"{base}/{by_name['preview.html']['artifact_id']}/download"
    download = client.get(download_url)
    assert download.status_code == 200 and download.content == html_source
    assert download.headers["content-disposition"].startswith("attachment;")
    assert client.head(download_url).content == b""
    assert client.get(f"{base}/000000000000000000000000/content").status_code == 404

    viewer_url = f"{base}/{by_name['preview.html']['artifact_id']}/viewer"
    viewer = client.get(viewer_url)
    assert viewer.status_code == 200
    assert "rcp-artifact-context" not in viewer.text
    assert 'id="keep"' in viewer.text

    legacy_preview_url = f"{base}/{by_name['preview.html']['artifact_id']}/preview"
    legacy_preview = client.get(legacy_preview_url)
    assert legacy_preview.status_code == 200
    assert "rcp-artifact-context" not in legacy_preview.text
    assert html_url in legacy_preview.text
    assert client.head(legacy_preview_url).content == b""

    kept = client.post(f"{base}/{by_name['preview.html']['artifact_id']}/keep")
    assert kept.status_code == 200
    kept_filename = kept.json()["kept_filename"]
    kept_path = service.history.workspace.root.parent / "artifacts" / kept_filename
    assert kept_path.read_bytes() == html_source

    externally_edited = b"<!doctype html><p>external edit</p>"
    kept_path.write_bytes(externally_edited)
    edited_preview = client.get(html_url)
    assert edited_preview.status_code == 200
    assert "external edit" in edited_preview.text
    assert client.get(download_url).content == externally_edited

    store = app.state.background_tasks.store
    store.checkpoint_agent_task(completed["operation_id"], native_session_id="artifact-session")
    origin = store.agent_task(completed["operation_id"])
    assert origin is not None and origin.native_session_id
    revisable_viewer = client.get(viewer_url)
    assert "rcp-artifact-context" in revisable_viewer.text
    assert "Added to the originating chat draft." in revisable_viewer.text
    admitted_requests: list[RunRequest] = []

    def capture_artifact_question(
        admitted_project_id,
        kind,
        request,
        *,
        operation_id=None,
        authorized_by=None,
        stage_host=None,
        stage_root=None,
    ):
        assert kind == "project_chat"
        assert stage_host is None and stage_root is None
        admitted_requests.append(request)
        now = store.now()
        return AgentTaskRecord(
            operation_id=operation_id or str(uuid.uuid4()),
            project_id=admitted_project_id,
            kind=kind,
            status="queued",
            request=request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="Queued.",
            authorized_by=authorized_by,
        )

    monkeypatch.setattr(app.state.background_tasks, "start", capture_artifact_question)
    asked = client.post(
        f"/api/projects/{project_id}/tasks/project_chat",
        json={
            "chat_id": origin.request["chat_id"],
            "message": "Why does this section jump?",
            "mode": "discuss",
            # Artifact-context admission ignores stale settings from the current
            # chat and resumes the exact profile recorded by the origin turn.
            "run_on": "stale-machine",
            "artifact_context": {
                "source": "task",
                "operation_id": origin.operation_id,
                "artifact_id": by_name["preview.html"]["artifact_id"],
                "selections": [{"kind": "text", "text": "external edit", "comment": "Why?"}],
            },
        },
    )
    assert asked.status_code == 202
    admitted = admitted_requests[-1]
    assert admitted.mode == "discuss"
    assert admitted.session_id == origin.native_session_id
    assert admitted.artifact_context is not None
    assert admitted.artifact_context.operation_id == origin.operation_id

    monkeypatch.setattr(
        "rcp.api.tasks.html_preview_document",
        lambda _data: (_ for _ in ()).throw(RuntimeError("preview renderer failed")),
    )
    assert client.head(html_url).status_code == 410
    assert client.get(html_url).status_code == 410

    persisted_before = app.state.background_tasks.store.agent_task(completed["operation_id"])
    assert persisted_before is not None and persisted_before.stage_root
    artifact_file = (
        Path(persisted_before.stage_root)
        / "turns"
        / completed["operation_id"]
        / "artifacts"
        / "preview.html"
    )
    artifact_file.unlink()
    assert client.get(html_url).status_code == 410
    persisted_after = app.state.background_tasks.store.agent_task(completed["operation_id"])
    assert persisted_after is not None
    assert persisted_after.result == persisted_before.result


def test_resumed_artifact_directory_rejects_a_symlinked_scope(tmp_path) -> None:
    stage = tmp_path / "stage"
    turns = stage / "turns"
    outside = tmp_path / "outside"
    turns.mkdir(parents=True)
    outside.joinpath("artifacts").mkdir(parents=True)
    turns.joinpath("original-turn").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="saved artifact directory"):
        _prepare_local_artifact_directory(
            stage,
            "original-turn",
            reuse=True,
        )


def test_chat_artifact_discovery_enforces_every_central_bound(tmp_path, monkeypatch) -> None:
    directory = tmp_path / "artifacts"
    directory.mkdir()
    directory.joinpath("a.html").write_text("a")
    directory.joinpath("b.html").write_text("bb")
    directory.joinpath("c.html").write_text("ccc")

    monkeypatch.setattr("rcp.runs.chat.CHAT_ARTIFACT_MAX_COUNT", 2)
    monkeypatch.setattr("rcp.runs.chat.CHAT_ARTIFACT_MAX_FILE_BYTES", 10)
    monkeypatch.setattr("rcp.runs.chat.CHAT_ARTIFACT_MAX_TOTAL_BYTES", 10)
    assert [item.name for item in _discover_chat_artifacts(None, "turn", directory, None)] == [
        "a.html",
        "b.html",
    ]

    monkeypatch.setattr("rcp.runs.chat.CHAT_ARTIFACT_MAX_COUNT", 8)
    monkeypatch.setattr("rcp.runs.chat.CHAT_ARTIFACT_MAX_FILE_BYTES", 2)
    assert [item.name for item in _discover_chat_artifacts(None, "turn", directory, None)] == [
        "a.html",
        "b.html",
    ]

    monkeypatch.setattr("rcp.runs.chat.CHAT_ARTIFACT_MAX_FILE_BYTES", 10)
    monkeypatch.setattr("rcp.runs.chat.CHAT_ARTIFACT_MAX_TOTAL_BYTES", 2)
    assert [item.name for item in _discover_chat_artifacts(None, "turn", directory, None)] == [
        "a.html"
    ]


@pytest.mark.asyncio
async def test_unexpected_artifact_discovery_error_does_not_fail_chat(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    answer = "The reply remains available."
    launcher = FakeLauncher([AgentEvent(event="answer", text=answer), AgentEvent(event="done")])
    monkeypatch.setattr(
        "rcp.runs.tasks.discuss._discover_chat_artifacts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("previewer bug")),
    )
    request = RunRequest(
        chat_scope="project",
        chat_id=str(uuid.uuid4()),
        message="Answer even if previewing fails.",
        run_truth_scope=["repo-a"],
    )

    frames = [
        frame async for frame in stream_discuss_run(service, launcher, request, tmp_path / "data")
    ]

    assert not _error_texts(frames)
    assert [event.text for event in _events(frames) if event.event == "answer"] == [answer]
    assert _events(frames)[-1].event == "done"


@pytest.mark.asyncio
async def test_chat_does_not_assemble_or_project_transcripts(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())

    def index_must_not_run(*_args, **_kwargs):
        raise AssertionError("chat must not assemble a source index")

    monkeypatch.setattr(service, "index_snapshot", index_must_not_run)
    request = RunRequest(
        chat_scope="project",
        message="What did we decide?",
        chat_id=str(uuid.uuid4()),
        run_truth_scope=["repo-a"],
        provider="claude",
    )

    class InspectingLauncher(FakeLauncher):
        async def stream(self, *args, **kwargs):
            self.read_dirs = list(kwargs["read_dirs"])
            async for event in super().stream(*args, **kwargs):
                yield event

    launcher = InspectingLauncher(
        [
            AgentEvent(event="answer", text="We kept the original boundary."),
            AgentEvent(event="done"),
        ]
    )

    frames = [
        frame async for frame in stream_discuss_run(service, launcher, request, tmp_path / "data")
    ]

    assert not _error_texts(frames)
    contract = _local_task_contract(launcher.last_args[1])
    assert Path(manifest.repository_map["repo-a"].path) in launcher.read_dirs
    assert not any(Path(path).name == "conversations" for path in launcher.read_dirs)
    assert "Conversations:" not in contract
    assert ".jsonl" not in contract
    assert launcher.last_kwargs["capability"] == "discuss"
    assert Path(launcher.last_kwargs["cwd"]).is_dir()


@pytest.mark.asyncio
async def test_chat_keeps_its_answer_when_transcript_persistence_rejects_a_path(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    launcher = FakeLauncher(
        [AgentEvent(event="answer", text="The answer survived."), AgentEvent(event="done")]
    )
    request = RunRequest(
        chat_scope="project",
        message="What is the question?",
        chat_id=str(uuid.uuid4()),
        run_truth_scope=["repo-a"],
    )

    def reject_path(*_args, **_kwargs) -> None:
        raise ValueError("cache path used a different canonical spelling")

    monkeypatch.setattr("rcp.runs.tasks.discuss._append_chat_exchange", reject_path)

    frames = [
        frame async for frame in stream_discuss_run(service, launcher, request, tmp_path / "data")
    ]

    assert [event.text for event in _events(frames) if event.event == "answer"] == [
        "The answer survived."
    ]
    assert any(event.event == "done" for event in _events(frames))
    assert not _error_texts(frames)


@pytest.mark.asyncio
async def test_same_chat_id_uses_distinct_stages_for_distinct_projects(manifest, tmp_path) -> None:
    shared_data = tmp_path / "data"
    first_app = create_named_app(str(manifest.path), data_dir=shared_data)
    first_service = first_app.state.service
    append_fixture_patch(first_service, seed_patch())

    second_repo = tmp_path / "second-repo-a"
    second_research = second_repo / ".research"
    second_research.mkdir(parents=True)
    second_manifest_path = second_research / "manifest.toml"
    first_repo = Path(manifest.repository_map["repo-a"].path)
    second_manifest_path.write_text(
        manifest.path.read_text(encoding="utf-8").replace(str(first_repo), str(second_repo)),
        encoding="utf-8",
    )
    second_app = create_named_app(str(second_manifest_path), data_dir=shared_data)
    second_service = second_app.state.service
    append_fixture_patch(second_service, seed_patch())

    store = first_app.state.background_tasks.store
    chat_id = str(uuid.uuid4())
    request = RunRequest(
        chat_scope="project",
        message="Keep these workspaces separate.",
        chat_id=chat_id,
        run_truth_scope=["repo-a"],
    )

    def execution(operation_id: str, project_id: str) -> AgentTaskExecution:
        now = store.now()
        store.create_agent_task(
            AgentTaskRecord(
                operation_id=operation_id,
                project_id=project_id,
                kind="project_chat",
                status="running",
                request=request.model_dump(mode="json"),
                created_at=now,
                updated_at=now,
                status_message="running",
            )
        )
        return AgentTaskExecution(
            operation_id=operation_id,
            store=store,
            control=AgentProcessControl(),
        )

    first_launcher = FakeLauncher(
        [AgentEvent(event="answer", text="First project."), AgentEvent(event="done")]
    )
    second_launcher = FakeLauncher(
        [AgentEvent(event="answer", text="Second project."), AgentEvent(event="done")]
    )
    first_execution = execution("first-project-chat", first_app.state.default_project_id)
    second_execution = execution("second-project-chat", second_app.state.default_project_id)

    first_frames = [
        frame
        async for frame in stream_discuss_run(
            first_service,
            first_launcher,
            request,
            shared_data,
            execution=first_execution,
        )
    ]
    second_frames = [
        frame
        async for frame in stream_discuss_run(
            second_service,
            second_launcher,
            request,
            shared_data,
            execution=second_execution,
        )
    ]

    assert not _error_texts(first_frames)
    assert not _error_texts(second_frames)
    first_workspace = Path(first_launcher.last_kwargs["cwd"])
    second_workspace = Path(second_launcher.last_kwargs["cwd"])
    assert first_workspace != second_workspace
    assert first_workspace.parent == second_workspace.parent == shared_data / "run-stage"
    assert chat_id in first_workspace.name
    assert chat_id in second_workspace.name


@pytest.mark.asyncio
async def test_pause_before_native_checkpoint_reclaims_claude_projection(
    manifest, tmp_path
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    codex_root = Path(next(iter(manifest.sources.codex_roots)))
    (codex_root / "source.jsonl").write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {"id": "source", "cwd": manifest.repository_map["repo-a"].path},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    request = RunRequest(
        chat_scope="project",
        message="Pause before initialization.",
        chat_id=str(uuid.uuid4()),
        run_truth_scope=["repo-a"],
        provider="claude",
    )
    store = app.state.background_tasks.store
    operation_id = "pause-without-native-checkpoint"
    now = store.now()
    store.create_agent_task(
        AgentTaskRecord(
            operation_id=operation_id,
            project_id=app.state.default_project_id,
            kind="project_chat",
            status="running",
            request=request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="running",
        )
    )
    execution = AgentTaskExecution(
        operation_id=operation_id,
        store=store,
        control=AgentProcessControl(),
    )
    launcher = FakeLauncher([AgentEvent(event="paused", text="Paused before startup.")])

    frames = [
        frame
        async for frame in stream_discuss_run(
            service,
            launcher,
            request,
            tmp_path / "data",
            execution=execution,
        )
    ]

    assert any(event.event == "paused" for event in _events(frames))
    workspace = Path(launcher.last_kwargs["cwd"])
    assert not (workspace / "inputs" / "conversations").exists()
    assert workspace.is_dir()


@pytest.mark.asyncio
async def test_authorized_chat_applies_its_patch_with_an_artifact_present(
    manifest, tmp_path
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    patch = refresh_patch("rq/artifact-backed-change").model_copy(update={"kind": "work"})

    class ArtifactPatchLauncher(ScriptedLauncher):
        async def stream(self, provider, prompt, **kwargs):
            workspace = Path(kwargs["cwd"])
            artifact_directory = next((workspace / "turns").glob("*/artifacts"))
            artifact_directory.joinpath("change.html").write_text(
                "<!doctype html><p>Graph change preview</p>", encoding="utf-8"
            )
            async for event in super().stream(provider, prompt, **kwargs):
                yield event

    launcher = ArtifactPatchLauncher(
        [{"patch.json": agent_patch_json(patch)}], message="Recorded with a preview."
    )
    request = RunRequest(
        chat_scope="project",
        message="Record this change and preview it.",
        chat_id=str(uuid.uuid4()),
        run_truth_scope=["repo-a"],
        mode="work",
    )

    execution = _chat_task_execution(
        app.state.background_tasks.store,
        operation_id="artifact-backed-work",
        project_id=app.state.default_project_id,
        request=request,
    )
    frames = [
        frame
        async for frame in stream_work_run(
            service,
            launcher,
            request,
            tmp_path / "data",
            execution=execution,
        )
    ]

    assert not _error_texts(frames)
    assert "rq/artifact-backed-change" in service.history.state().nodes
    assert [event.artifact.name for event in _events(frames) if event.event == "artifact"] == [
        "change.html"
    ]


@pytest.mark.asyncio
async def test_chat_launch_exception_keeps_workspace_without_transcript_projection(
    manifest, tmp_path
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())

    class ExplodingLauncher:
        async def stream(self, *_args, **kwargs):
            self.workspace = Path(kwargs["cwd"])
            self.projection = self.workspace / "inputs" / "conversations"
            assert not self.projection.exists()
            if False:
                yield AgentEvent(event="done")
            raise OSError("provider launch failed")

    launcher = ExplodingLauncher()
    request = RunRequest(
        chat_scope="project",
        message="What did we decide?",
        chat_id=str(uuid.uuid4()),
        run_truth_scope=["repo-a"],
        provider="claude",
    )

    with pytest.raises(OSError, match="provider launch failed"):
        async for _frame in stream_discuss_run(service, launcher, request, tmp_path / "data"):
            pass

    assert not launcher.projection.exists()
    assert launcher.workspace.is_dir()


def test_local_stage_sweeper_removes_stale_read_only_tree(tmp_path) -> None:
    root = tmp_path / "run-stage"
    stale_tree = root / "chat-expired" / "inputs" / "old-tree"
    stale_tree.mkdir(parents=True)
    copied = stale_tree / "old-input.txt"
    copied.write_text("old input", encoding="utf-8")
    copied.chmod(0o400)
    stale_tree.chmod(0o500)
    stage = root / "chat-expired"
    old_mtime = stage.stat().st_mtime

    _sweep_stale_stages(root, now=old_mtime + 8 * 86400)

    assert not stage.exists()


def test_failed_chat_task_keeps_the_answer_it_already_produced(manifest, tmp_path) -> None:
    """A failure after the reply must not take the reply down with it.

    The human asked a question and got an answer; only what came after it failed,
    and the answer is already in the transcript. Dropping it from the task result
    would leave the chat showing an error where its reply should be.
    """

    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    append_fixture_patch(app.state.service, seed_patch())
    answer = "Recorded — though staging the follow-up failed right afterwards."

    async def stream(_project_id, _kind, _request, _execution):
        yield _sse(AgentEvent(event="answer", text=answer))
        yield _sse(AgentEvent(event="error", text="The correction could not be staged."))

    app.state.background_tasks.stream = stream
    client = TestClient(app)
    started = client.post(
        f"/api/projects/{app.state.default_project_id}/tasks/node_chat",
        json={
            "node_id": "hyp/replanning-restores-plasticity",
            "chat_id": str(uuid.uuid4()),
            "message": "Record the transfer question.",
            "run_truth_scope": ["repo-a"],
            "mode": "work",
        },
    )

    assert started.status_code == 202
    record = _wait_for_run(client, app.state.default_project_id, started.json()["operation_id"])
    assert record["status"] == "failed"
    assert record["result"] == {"messages": [answer]}
    assert "could not be staged" in record["error"]


def test_paper_coach_uses_agent_task_manager_and_result_shape(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    service.paper.create()
    answer = "State the comparison before introducing the endpoint-KL terminology."
    session_id = str(uuid.uuid4())
    launcher = FakeLauncher(
        [
            AgentEvent(event="session", session_id=session_id),
            AgentEvent(event="message", text=answer),
            AgentEvent(event="done"),
        ]
    )

    async def stream(_project_id, kind, request, execution):
        assert kind == "paper_coach"
        async for frame in stream_coach(
            service,
            launcher,
            service.paper,
            request,
            tmp_path / "data",
            execution=execution,
        ):
            yield frame

    app.state.background_tasks.stream = stream
    client = TestClient(app)
    started = client.post(
        f"/api/projects/{app.state.default_project_id}/tasks/paper_coach",
        json={"message": "Review the argument."},
    )

    assert started.status_code == 202
    completed = _wait_for_run(
        client,
        app.state.default_project_id,
        started.json()["operation_id"],
    )
    assert completed["status"] == "succeeded"
    assert completed["kind"] == "paper_coach"
    assert completed["result"] == {"messages": [answer]}
    assert completed["applied_revision"] is None
    launch = next(
        receipt for receipt in completed["debug_receipts"] if receipt["category"] == "agent_launch"
    )
    assert launch["payload"]["capability"] == "paper_readonly"
    assert launch["payload"]["network_access"] is True
    assert isinstance(launcher.last_kwargs["control"], AgentProcessControl)
    assert launcher.last_kwargs["capability"] == "paper_readonly"
    assert service.paper.sessions()[0].native_session_id == session_id


def test_paused_paper_coach_resumes_from_task_checkpoint_before_session_record(
    manifest, tmp_path
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    service.paper.create()
    session_id = str(uuid.uuid4())
    answer = "Define the comparison in ordinary language before naming the metric."

    class PausingCoachLauncher:
        def __init__(self) -> None:
            self.calls = 0
            self.sessions: list[str | None] = []

        async def stream(self, *_args, **kwargs):
            self.calls += 1
            self.sessions.append(kwargs.get("session_id"))
            if self.calls == 1:
                yield AgentEvent(event="session", session_id=session_id)
                control = kwargs["control"]
                while not control.pause_requested.is_set():
                    await asyncio.sleep(0.01)
                yield AgentEvent(event="paused", text="Provider process paused.")
                return
            yield AgentEvent(event="session", session_id=session_id)
            yield AgentEvent(event="message", text=answer)
            yield AgentEvent(event="done")

    launcher = PausingCoachLauncher()

    async def stream(_project_id, kind, request, execution):
        assert kind == "paper_coach"
        async for frame in stream_coach(
            service,
            launcher,
            service.paper,
            request,
            tmp_path / "data",
            execution=execution,
        ):
            yield frame

    app.state.background_tasks.stream = stream
    client = TestClient(app)
    started = client.post(
        f"/api/projects/{app.state.default_project_id}/tasks/paper_coach",
        json={"message": "Review the framing."},
    )
    operation_id = started.json()["operation_id"]
    _wait_for_status(client, app.state.default_project_id, operation_id, {"running"})
    paused_response = client.post(
        f"/api/projects/{app.state.default_project_id}/tasks/{operation_id}/pause"
    )

    assert paused_response.status_code == 202
    paused = _wait_for_status(
        client,
        app.state.default_project_id,
        operation_id,
        {"paused"},
    )
    assert paused["native_session_id"] == session_id
    assert paused["can_resume"] is True
    assert service.paper.sessions() == []

    resumed_response = client.post(
        f"/api/projects/{app.state.default_project_id}/tasks/{operation_id}/resume"
    )

    assert resumed_response.status_code == 202
    resumed = _wait_for_run(
        client,
        app.state.default_project_id,
        resumed_response.json()["operation_id"],
    )
    assert resumed["status"] == "succeeded"
    assert resumed["result"] == {"messages": [answer]}
    assert launcher.sessions == [None, session_id]
    assert service.paper.sessions()[0].native_session_id == session_id


@pytest.mark.asyncio
async def test_node_chat_streams_answer_and_persists_transcript(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    chat_id = str(uuid.uuid4())
    answer = "Added the transfer question you described."
    patch = refresh_patch().model_copy(update={"kind": "work"})
    launcher = ScriptedLauncher(
        [{"patch.json": agent_patch_json(patch)}],
        message=answer,
    )
    request = RunRequest(
        node_id="hyp/replanning-restores-plasticity",
        message="Record the transfer question as its own node.",
        chat_id=chat_id,
        run_truth_scope=["repo-a"],
        mode="work",
    )
    execution = _chat_task_execution(
        app.state.background_tasks.store,
        operation_id="streamed-node-chat-work",
        project_id=app.state.default_project_id,
        request=request,
    )

    frames = [
        item
        async for item in stream_work_run(
            service,
            launcher,
            request,
            tmp_path / "data",
            execution=execution,
        )
    ]

    assert answer in [event.text for event in _events(frames) if event.event == "answer"]
    assert service.history.state().revision == 3
    transcript = next((manifest.research_dir / "chat").glob("*.jsonl"))
    records = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines()]
    assert [record["role"] for record in records] == ["user", "assistant"]
    assert records[-1]["nativeSessionId"] == launcher.native_session_id
    assert any(session.provider == "app_chat" for session in service.index_snapshot().sessions)


@pytest.mark.asyncio
async def test_node_chat_answers_without_writing_a_patch(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    revision_before = service.history.state().revision
    answer = "It is still proposed because no matched forward test has run yet."
    launcher = ScriptedLauncher([{}], message=answer)
    request = RunRequest(
        node_id="hyp/replanning-restores-plasticity",
        message="Why is this not accepted?",
        chat_id=str(uuid.uuid4()),
        run_truth_scope=["repo-a"],
    )

    frames = [
        item async for item in stream_discuss_run(service, launcher, request, tmp_path / "data")
    ]

    events = _events(frames)
    assert [event.text for event in events if event.event == "answer"] == [answer]
    assert not _error_texts(frames)
    assert events[-1].event == "done"
    # Answering a question is not a graph edit.
    assert _applied_revision(frames) is None
    assert service.history.state().revision == revision_before
    transcript = next((manifest.research_dir / "chat").glob("*.jsonl"))
    records = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines()]
    assert [record["role"] for record in records] == ["user", "assistant"]
    assert records[-1]["appliedRevision"] is None


@pytest.mark.asyncio
async def test_node_chat_survives_a_stale_ingest_cursor(manifest, tmp_path) -> None:
    """The reported failure: a corrupt ingest cursor must not block a question."""

    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    raw_path = Path(next(iter(manifest.sources.codex_roots))) / "stale.jsonl"
    raw_path.write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {"id": "stale-session", "cwd": manifest.repository_map["repo-a"].path},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    session = next(
        item
        for item in service.index_snapshot(refresh=True).sessions
        if item.session_id == "stale-session"
    )
    (manifest.research_dir / "cursors.json").write_text(
        json.dumps({session.key: "missing-record"}), encoding="utf-8"
    )
    answer = "This node records the runtime monitor contract decision."
    launcher = ScriptedLauncher([{}], message=answer)
    request = RunRequest(
        node_id="hyp/replanning-restores-plasticity",
        message="I do not understand the question in this node.",
        chat_id=str(uuid.uuid4()),
        run_truth_scope=["repo-a"],
    )

    frames = [
        item async for item in stream_discuss_run(service, launcher, request, tmp_path / "data")
    ]

    assert not _error_texts(frames)
    assert [event.text for event in _events(frames) if event.event == "answer"] == [answer]


@pytest.mark.asyncio
async def test_chat_prompt_carries_the_node_and_not_the_ingest_contract(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    launcher = ScriptedLauncher([{}], message="Answered.")
    request = RunRequest(
        node_id="hyp/replanning-restores-plasticity",
        message="What does this claim?",
        chat_id=str(uuid.uuid4()),
        run_truth_scope=["repo-a"],
    )

    async for _ in stream_discuss_run(service, launcher, request, tmp_path / "data"):
        pass

    prompt = launcher.prompts[0]
    assert len(prompt.splitlines()) < 200
    assert "Search-time replanning restores future learning ability." not in prompt
    assert "rq/learning-after-shift" not in prompt
    assert not any(name.endswith("context.json") for name in launcher.input_snapshots[0])
    contract = Path(prompt.splitlines()[1]).read_text(encoding="utf-8")
    assert "hyp/replanning-restores-plasticity" in contract
    assert str(manifest.research_dir / "graph.json") in contract
    # No project extension was ever defined, so the extension pointer and its
    # authoring rules stay out; the base vocabulary always ships.
    assert "graph.json#ontology" not in contract
    assert "Base node ids are" in contract
    assert "This is a Discuss turn." in prompt
    assert prompt.endswith("\n\nWhat does this claim?")
    assert not any(name.endswith("human-request.txt") for name in launcher.input_snapshots[0])
    # None of the ingest machinery: no evidence slices, no cursors, no coverage
    # bookkeeping, no multi-agent synthesis contract.
    assert "slice_sha256" not in prompt
    assert "cursor_note" not in prompt
    assert "coverage.sessions_read" not in prompt
    assert "Coordinator contract" not in prompt


@pytest.mark.asyncio
async def test_chat_patch_cannot_move_the_ingest_boundary(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    revision_before = service.history.state().revision
    patch = Patch(
        kind="work",
        author="agent",
        summary="Claimed coverage from a chat turn.",
        ops=[{"op": "set_coverage", "coverage": {"sessions_read": ["repo-a/laptop/codex/x"]}}],
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
    )
    answer = "Here is the explanation."
    launcher = ScriptedLauncher([{"patch.json": agent_patch_json(patch)}], message=answer)
    request = RunRequest(
        node_id="hyp/replanning-restores-plasticity",
        message="Explain and record what you read.",
        chat_id=str(uuid.uuid4()),
        run_truth_scope=["repo-a"],
        mode="work",
    )

    frames = [item async for item in stream_work_run(service, launcher, request, tmp_path / "data")]

    # The answer still reaches the human; only the graph change is refused.
    assert not _error_texts(frames)
    assert [event.text for event in _events(frames) if event.event == "answer"] == [answer]
    graph_update = _graph_update(frames)
    assert graph_update is not None
    assert graph_update["status"] == "rejected"
    assert any("set_coverage" in message for message in graph_update["validation_messages"])
    assert service.history.state().revision == revision_before


@pytest.mark.asyncio
async def test_project_chat_persists_project_scoped_transcript(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    chat_id = str(uuid.uuid4())
    answer = "The project is testing whether matched trajectories preserve future learning."
    patch = Patch(
        kind="chat",
        author="agent",
        summary="Explained the project-level research state.",
        ops=[],
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
    )
    launcher = ScriptedLauncher(
        [{"patch.json": agent_patch_json(patch)}],
        message=answer,
    )
    request = RunRequest(
        chat_scope="project",
        message="What is this project trying to establish?",
        chat_id=chat_id,
        run_truth_scope=["repo-a"],
    )

    frames = [
        item async for item in stream_discuss_run(service, launcher, request, tmp_path / "data")
    ]

    assert answer in [event.text for event in _events(frames) if event.event == "answer"]
    transcript = manifest.research_dir / "chat" / f"project-{chat_id}.jsonl"
    records = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines()]
    assert {record["chatScope"] for record in records} == {"project"}
    assert {record["nodeId"] for record in records} == {None}


@pytest.mark.asyncio
async def test_unauthorized_chat_patch_is_discarded_not_applied(manifest, tmp_path) -> None:
    """Writing the file does not grant the authority to change the graph."""

    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    revision_before = service.history.state().revision
    answer = "It is still proposed because no matched forward test has run."
    launcher = ScriptedLauncher(
        [{"patch.json": agent_patch_json(refresh_patch().model_copy(update={"kind": "chat"}))}],
        message=answer,
    )
    request = RunRequest(
        node_id="hyp/replanning-restores-plasticity",
        message="Why is this not accepted?",
        chat_id=str(uuid.uuid4()),
        run_truth_scope=["repo-a"],
    )

    frames = [
        item async for item in stream_discuss_run(service, launcher, request, tmp_path / "data")
    ]

    assert [event.text for event in _events(frames) if event.event == "answer"] == [answer]
    assert service.history.state().revision == revision_before
    prompt = launcher.prompts[0]
    contract = Path(prompt.splitlines()[1]).read_text(encoding="utf-8")
    assert "This turn has no graph-change channel" in contract
    assert "## Discuss contract" in contract
    assert "## Work contract" in contract
    assert "Patch JSON Schema" in contract


@pytest.mark.asyncio
async def test_chat_turns_share_one_scratch_folder_and_drop_the_last_patch(
    manifest, tmp_path
) -> None:
    """A conversation keeps its workspace so the next turn can resume in it.

    The same folder must not hand turn two the patch file turn one left behind.
    """

    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    chat_id = str(uuid.uuid4())
    patch = refresh_patch().model_copy(update={"kind": "work"})
    first = ScriptedLauncher([{"patch.json": agent_patch_json(patch)}], message="First answer.")
    request = RunRequest(
        node_id="hyp/replanning-restores-plasticity",
        message="Record the transfer question.",
        chat_id=chat_id,
        run_truth_scope=["repo-a"],
        mode="work",
    )
    first_execution = _chat_task_execution(
        app.state.background_tasks.store,
        operation_id="shared-chat-first-work",
        project_id=app.state.default_project_id,
        request=request,
    )
    async for _ in stream_work_run(
        service,
        first,
        request,
        tmp_path / "data",
        execution=first_execution,
    ):
        pass
    applied = service.history.state().revision
    app.state.background_tasks.store.complete_agent_task(
        first_execution.operation_id,
        applied_revision=applied,
        result={"messages": ["First answer."]},
    )
    workspace = first.workspaces[0]
    assert workspace.is_dir()

    # Turn two writes no patch at all, in the same folder, and must not inherit one.
    second = ScriptedLauncher([{}], message="Second answer.")
    second.native_session_id = first.native_session_id
    second_request = request.model_copy(
        update={
            "message": "Thanks — what does that node mean?",
            "session_id": first.native_session_id,
        }
    )
    second_execution = _chat_task_execution(
        app.state.background_tasks.store,
        operation_id="shared-chat-second-work",
        project_id=app.state.default_project_id,
        request=second_request,
    )
    frames = [
        item
        async for item in stream_work_run(
            service,
            second,
            second_request,
            tmp_path / "data",
            execution=second_execution,
        )
    ]

    assert second.workspaces[0] == workspace
    assert not (workspace / "patch.json").exists()
    assert [event.text for event in _events(frames) if event.event == "answer"] == [
        "Second answer."
    ]
    assert service.history.state().revision == applied


@pytest.mark.asyncio
async def test_chat_patch_is_applied_to_live_state_when_the_graph_moves(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    answer = "Recorded."
    patch = refresh_patch("rq/late-arrival").model_copy(update={"kind": "work"})

    class RacingLauncher(ScriptedLauncher):
        async def stream(self, provider, prompt, **kwargs):
            # A refresh lands between context assembly and the patch being applied.
            append_fixture_patch(service, refresh_patch("rq/landed-first"))
            async for event in super().stream(provider, prompt, **kwargs):
                yield event

    launcher = RacingLauncher([{"patch.json": agent_patch_json(patch)}], message=answer)
    request = RunRequest(
        node_id="hyp/replanning-restores-plasticity",
        message="Record the transfer question.",
        chat_id=str(uuid.uuid4()),
        run_truth_scope=["repo-a"],
        mode="work",
    )

    execution = _chat_task_execution(
        app.state.background_tasks.store,
        operation_id="live-state-work",
        project_id=app.state.default_project_id,
        request=request,
    )
    frames = [
        item
        async for item in stream_work_run(
            service,
            launcher,
            request,
            tmp_path / "data",
            execution=execution,
        )
    ]

    assert not _error_texts(frames)
    assert [event.text for event in _events(frames) if event.event == "answer"] == [answer]
    graph_update = _graph_update(frames)
    assert graph_update is not None
    assert graph_update["status"] == "applied"
    assert graph_update["applied_revision"] == 4
    assert graph_update["correction_rounds"] == 0
    assert "rq/landed-first" in service.history.state().nodes
    assert "rq/late-arrival" in service.history.state().nodes


def test_resumed_chat_patch_is_applied_to_live_current_state(manifest, tmp_path) -> None:
    """A semantically valid resumed patch is revalidated against current state."""

    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    session_id = str(uuid.uuid4())
    patch = refresh_patch("rq/written-before-the-pause").model_copy(update={"kind": "work"})

    class PausingChatLauncher:
        def __init__(self) -> None:
            self.sessions: list[str | None] = []
            self.capabilities: list[str] = []
            self.read_dirs: list[list[Path]] = []
            self.workspaces: list[Path] = []
            self.prompts: list[str] = []

        async def stream(self, _provider, prompt, **kwargs):
            self.sessions.append(kwargs.get("session_id"))
            self.capabilities.append(kwargs["capability"])
            self.workspaces.append(Path(kwargs["cwd"]))
            self.read_dirs.append([Path(path) for path in kwargs["read_dirs"]])
            self.prompts.append(prompt)
            yield AgentEvent(event="session", session_id=session_id)
            if len(self.sessions) == 1:
                control = kwargs["control"]
                while not control.pause_requested.is_set():
                    await asyncio.sleep(0.01)
                yield AgentEvent(event="paused", text="Provider process paused.")
                return
            (Path(kwargs["cwd"]) / "patch.json").write_text(
                agent_patch_json(patch), encoding="utf-8"
            )
            yield AgentEvent(event="answer", text="Recorded.")
            yield AgentEvent(event="done")

    launcher = PausingChatLauncher()

    async def stream(_project_id, _kind, request, execution):
        async for frame in stream_work_run(
            service, launcher, request, tmp_path / "data", execution=execution
        ):
            yield frame

    app.state.background_tasks.stream = stream
    client = TestClient(app)
    project_id = app.state.default_project_id
    started = client.post(
        f"/api/projects/{project_id}/tasks/node_chat",
        json={
            "node_id": "hyp/replanning-restores-plasticity",
            "chat_id": str(uuid.uuid4()),
            "message": "Record the transfer question.",
            "run_truth_scope": ["repo-a"],
            "mode": "work",
            "provider": "claude",
        },
    )
    operation_id = started.json()["operation_id"]
    _wait_for_status(client, project_id, operation_id, {"running"})
    original = client.get(f"/api/projects/{project_id}/tasks/{operation_id}").json()
    assert original["request"]["mode"] == "work"
    client.post(f"/api/projects/{project_id}/tasks/{operation_id}/pause")
    _wait_for_status(client, project_id, operation_id, {"paused"})
    assert not any(path.name == "conversations" for path in launcher.read_dirs[0])
    assert [item.name for item in (launcher.workspaces[0] / "turns").iterdir()] == [operation_id]

    # The human works on the graph while the turn is paused.
    append_fixture_patch(service, refresh_patch("rq/landed-while-paused"))
    resumed_response = client.post(f"/api/projects/{project_id}/tasks/{operation_id}/resume")

    assert resumed_response.status_code == 202
    resumed = _wait_for_run(client, project_id, resumed_response.json()["operation_id"])
    assert resumed["parent_operation_id"] == operation_id
    assert resumed["request"]["mode"] == "work"
    assert launcher.sessions == [None, session_id]
    assert launcher.capabilities == ["work_auto", "work_auto"]
    assert "This is a Work turn." in launcher.prompts[0]
    resume_contract = _local_task_contract(launcher.prompts[1])
    assert "# RCP resume contract" in resume_contract
    assert "This is a Work turn." not in launcher.prompts[1]
    assert not list((launcher.workspaces[1] / "inputs").glob("*human-request.txt"))
    assert not any(path.name == "conversations" for path in launcher.read_dirs[1])
    assert launcher.workspaces[1].is_dir()
    assert [item.name for item in (launcher.workspaces[1] / "turns").iterdir()] == [operation_id]
    assert resumed["status"] == "succeeded"
    assert resumed["result"]["graph_update"]["status"] == "applied"
    assert resumed["result"]["graph_update"]["applied_revision"] == 4
    assert "rq/landed-while-paused" in service.history.state().nodes
    assert "rq/written-before-the-pause" in service.history.state().nodes


def test_retried_chat_gets_a_new_artifact_scope_in_the_same_conversation_stage(
    manifest, tmp_path
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())

    class RetryLauncher:
        def __init__(self) -> None:
            self.calls = 0
            self.workspaces: list[Path] = []

        async def stream(self, *_args, **kwargs):
            self.calls += 1
            self.workspaces.append(Path(kwargs["cwd"]))
            if self.calls == 1:
                yield AgentEvent(event="error", text="provider failed")
                return
            yield AgentEvent(event="answer", text="Retry answered.")
            yield AgentEvent(event="done")

    launcher = RetryLauncher()

    async def stream(_project_id, _kind, request, execution):
        async for frame in stream_discuss_run(
            service, launcher, request, tmp_path / "data", execution=execution
        ):
            yield frame

    app.state.background_tasks.stream = stream
    client = TestClient(app)
    project_id = app.state.default_project_id
    started = client.post(
        f"/api/projects/{project_id}/tasks/project_chat",
        json={
            "chat_id": str(uuid.uuid4()),
            "message": "Try this preview.",
            "run_truth_scope": ["repo-a"],
        },
    )
    failed = _wait_for_run(client, project_id, started.json()["operation_id"])
    retried_response = client.post(
        f"/api/projects/{project_id}/tasks/{failed['operation_id']}/retry"
    )
    retried = _wait_for_run(client, project_id, retried_response.json()["operation_id"])

    assert retried["status"] == "succeeded"
    assert launcher.workspaces[0] == launcher.workspaces[1]
    assert {item.name for item in (launcher.workspaces[1] / "turns").iterdir()} == {
        failed["operation_id"],
        retried["operation_id"],
    }


@pytest.mark.parametrize("fault", ["wrong-root", "wrong-host"])
@pytest.mark.asyncio
async def test_resumed_chat_rejects_a_mismatched_saved_stage(
    manifest, tmp_path, fault: str
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    store = app.state.background_tasks.store
    _record_lineage_task(
        store,
        "original-stage-attempt",
        parent=None,
        resumed=False,
        graph_revision=1,
    )
    _record_lineage_task(
        store,
        "resumed-stage-attempt",
        parent="original-stage-attempt",
        resumed=True,
        graph_revision=1,
        attempt=2,
    )
    request = RunRequest(
        node_id="hyp/replanning-restores-plasticity",
        message="Continue the paused turn.",
        chat_id=str(uuid.uuid4()),
        session_id=str(uuid.uuid4()),
        run_truth_scope=["repo-a"],
    )
    execution = AgentTaskExecution(
        operation_id="resumed-stage-attempt",
        store=store,
        control=AgentProcessControl(),
        continuation="resume",
    )
    expected = tmp_path / "data" / "run-stage" / _chat_stage_name(service, request, execution)
    expected.mkdir(parents=True)
    stored_root = (
        str(expected.with_name("chat-from-another-project"))
        if fault == "wrong-root"
        else str(expected)
    )
    stored_host = "remote.example" if fault == "wrong-host" else None
    store.checkpoint_agent_task(
        execution.operation_id,
        stage_root=stored_root,
        stage_host=stored_host,
    )
    stored = store.agent_task(execution.operation_id)
    assert stored is not None
    execution.stage_root = stored.stage_root
    execution.stage_host = stored.stage_host
    launcher = FakeLauncher([AgentEvent(event="done")])

    frames = [
        frame
        async for frame in stream_discuss_run(
            service,
            launcher,
            request,
            tmp_path / "data",
            execution=execution,
        )
    ]

    assert launcher.calls == 0
    assert any("Cannot safely resume this chat" in text for text in _error_texts(frames))
    assert any("Retry the turn from the beginning" in text for text in _error_texts(frames))


@pytest.mark.asyncio
async def test_remote_chat_resume_attaches_its_validated_saved_stage(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    request = RunRequest(
        node_id="hyp/replanning-restores-plasticity",
        message="Continue remotely.",
        chat_id=str(uuid.uuid4()),
        session_id=str(uuid.uuid4()),
        run_truth_scope=["repo-a"],
        run_on="remote-1",
    )
    context = service.assemble_chat(request.model_copy(update={"run_on": "laptop"}))
    service.history.manifest.machines.append(MachineConfig(alias="remote-1", host="remote.example"))
    service.history.manifest.repository_map[service.manifest.state.repository].machine = "remote-1"
    service.history.manifest.agent.node_chat.run_on = "remote-1"
    assert "remote-1" in service.manifest.machine_map
    monkeypatch.setattr(service, "assemble_chat", lambda _request: context)
    store = app.state.background_tasks.store
    _record_lineage_task(
        store,
        "remote-original",
        parent=None,
        resumed=False,
        graph_revision=1,
    )
    _record_lineage_task(
        store,
        "remote-resume",
        parent="remote-original",
        resumed=True,
        graph_revision=1,
        attempt=2,
    )
    execution = AgentTaskExecution(
        operation_id="remote-resume",
        store=store,
        control=AgentProcessControl(),
        continuation="resume",
    )
    stage_name = _chat_stage_name(service, request, execution)
    saved_root = f"/tmp/rcp-run.{stage_name}"
    store.checkpoint_agent_task(
        execution.operation_id,
        stage_host="remote.example",
        stage_root=saved_root,
    )
    execution.stage_host = "remote.example"
    execution.stage_root = saved_root
    store.record_agent_task_receipt(
        "remote-original",
        "agent_prompt",
        {
            "prompt": "pointer",
            "contract_path": f"{saved_root}/inputs/task-original-initial.md",
        },
        tier="diagnostic",
    )

    class RecordingRemoteStage:
        attached: list[tuple[str, str]] = []
        finalized = 0
        touched = 0

        def __init__(self, host: str) -> None:
            self.host = host
            self.root = None

        @property
        def workspace(self):
            assert self.root is not None
            return self.root / "workspace"

        def attach(self, root: str):
            self.root = PurePosixPath(root)
            self.attached.append((self.host, root))
            return self

        def open(self, *_args, **_kwargs):
            raise AssertionError("Resume must attach the saved stage, not reopen by name")

        def list_workspace_files(self):
            return []

        def touch(self):
            type(self).touched += 1

        def prepare_artifact_directory(self, scope_id, *, reuse):
            assert reuse is True
            assert scope_id == "remote-original"
            return self.workspace / "turns" / scope_id / "artifacts"

        def put_file(self, _source, label):
            assert self.root is not None
            return str(self.root / "inputs" / label)

        def put_directory(self, _source, label, *, reuse=False):
            assert self.root is not None
            assert reuse is True
            return str(self.root / "inputs" / label)

        def finalize_inputs(self):
            type(self).finalized += 1

        def list_artifact_files(self, _scope_id):
            return []

    monkeypatch.setattr("rcp.runs.tasks.discuss.RemoteRunStage", RecordingRemoteStage)
    launcher = FakeLauncher(
        [AgentEvent(event="answer", text="Remote continuation."), AgentEvent(event="done")]
    )

    frames = [
        frame
        async for frame in stream_discuss_run(
            service,
            launcher,
            request,
            tmp_path / "data",
            execution=execution,
        )
    ]

    assert not _error_texts(frames)
    assert RecordingRemoteStage.attached == [("remote.example", saved_root)]
    assert RecordingRemoteStage.finalized == launcher.calls == 1
    assert RecordingRemoteStage.touched == 1
    assert launcher.last_kwargs["cwd"] == Path(saved_root) / "workspace"


@pytest.mark.asyncio
async def test_paper_resume_rejects_settings_change_before_launch(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    paper = service.paper
    session_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    paper.record_session(
        WritingSession(
            provider="codex",
            native_session_id=session_id,
            execution_machine="laptop",
            project_id=paper.project_id,
            title="Coach",
            model="gpt-pinned",
            reasoning="medium",
            created_at=now,
            last_resumed_at=now,
            introduction_hash_examined="intro",
            graph_revision_examined=0,
            research_md_hash_examined="research",
        )
    )
    launcher = FakeLauncher([AgentEvent(event="done")])

    events = [
        item
        async for item in stream_coach(
            service,
            launcher,
            paper,
            CoachRequest(
                message="Review the argument.",
                session_id=session_id,
                model="different-model",
            ),
            tmp_path / "data",
        )
    ]

    assert any("cannot change model" in item for item in events)
    assert launcher.calls == 0


@pytest.mark.asyncio
async def test_paper_coach_uses_its_read_only_launcher_contract(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    service.manifest.machine_map["laptop"].provider_paths["codex"] = "/opt/agents/codex"
    paper = service.paper
    paper.create()
    session_id = str(uuid.uuid4())
    launcher = FakeLauncher(
        [
            AgentEvent(event="session", session_id=session_id),
            AgentEvent(event="message", text="Review the claim boundary."),
            AgentEvent(event="done"),
        ]
    )

    events = [
        item
        async for item in stream_coach(
            service,
            launcher,
            paper,
            CoachRequest(message="Review the argument."),
            tmp_path / "data",
        )
    ]

    assert launcher.calls == 1
    assert launcher.last_kwargs["capability"] == "paper_readonly"
    assert launcher.last_kwargs["binary"] == "/opt/agents/codex"
    assert launcher.last_kwargs["cwd"] == manifest.research_dir
    assert any("Review the claim boundary." in item for item in events)
    assert paper.sessions()[0].native_session_id == session_id


def _event_frame(event: AgentEvent) -> str:
    return f"data: {event.model_dump_json()}\n\n"


def _wait_for_status(
    client: TestClient,
    project_id: str,
    operation_id: str,
    statuses: set[str],
) -> dict[str, object]:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        response = client.get(f"/api/projects/{project_id}/tasks/{operation_id}")
        assert response.status_code == 200
        record = response.json()
        if record["status"] in statuses:
            return record
        time.sleep(0.01)
    raise AssertionError(f"background run did not reach {sorted(statuses)}")


@pytest.mark.asyncio
async def test_work_without_patch_succeeds_without_spending_a_revision(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    launcher = ScriptedLauncher([{}], message="The requested check completed.")
    request = RunRequest(
        chat_scope="project",
        chat_id=str(uuid.uuid4()),
        message="Run the check.",
        run_truth_scope=["repo-a"],
        mode="work",
    )

    frames = [
        frame async for frame in stream_work_run(service, launcher, request, tmp_path / "data")
    ]

    assert not _error_texts(frames)
    assert _graph_update(frames) == {
        "status": "none",
        "applied_revision": None,
        "change_summary": [],
        "proposal_ids": [],
        "validation_messages": [],
        "correction_rounds": 0,
        "repairable": False,
    }
    assert service.history.state().revision == 2
    assert launcher.launch_kwargs[0]["capability"] == "work_auto"
    assert launcher.launch_kwargs[0]["write_dirs"] == [Path(manifest.repository_map["repo-a"].path)]
    transcript = service.chat_transcript(request.chat_id)
    assert transcript is not None
    assert [message.mode for message in transcript.messages] == ["work", "work"]


@pytest.mark.asyncio
async def test_ordinary_work_turns_retain_one_master_and_send_only_turn_envelopes(
    manifest, tmp_path
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    store = app.state.background_tasks.store
    project_id = app.state.default_project_id
    chat_id = str(uuid.uuid4())
    launcher = ScriptedLauncher([{}, {}, {}], message="Work answered.")

    async def stream(_project_id, kind, request, execution):
        assert kind == "project_chat"
        async for frame in stream_work_run(
            service,
            launcher,
            request,
            tmp_path / "data",
            execution=execution,
        ):
            yield frame

    app.state.background_tasks.stream = stream
    client = TestClient(app)
    first_message = "First  Work request.\nKeep this line."
    first_response = client.post(
        f"/api/projects/{project_id}/tasks/project_chat",
        json={
            "chat_id": chat_id,
            "message": first_message,
            "run_truth_scope": ["repo-a"],
            "mode": "work",
        },
    )
    assert first_response.status_code == 202, first_response.json()
    first_operation_id = first_response.json()["operation_id"]
    assert _wait_for_run(client, project_id, first_operation_id)["status"] == "succeeded"
    first_prompt = launcher.prompts[0]
    assert first_prompt.startswith("Open and retain the RCP chat master context at:\n")
    assert "This is a Work turn.\nArtifact directory for this turn: " in first_prompt
    assert f"\n\n{first_message}" in first_prompt
    master_path = Path(first_prompt.splitlines()[1])
    master = master_path.read_text(encoding="utf-8")
    assert "## Discuss contract" in master
    assert "## Work contract" in master
    assert "named in the envelope" in master
    workspace = launcher.workspaces[0]
    prompt_candidate = json.loads(
        store.agent_task_contract(first_operation_id, "chat_prompt_state") or "{}"
    )
    prompt_values = prompt_candidate["snapshot"]["values"]
    assert set(prompt_values) == {
        "project",
        "settings",
        "current",
        "repositories",
        "skills",
        "patch",
        "workspace",
    }
    assert isinstance(prompt_values["current"]["graph_revision"], int)
    inputs = master_path.parent
    assert len(list(inputs.glob("chat-master-v*.md"))) == 1
    assert len(list(inputs.glob("chat-patch-schema-*.json"))) == 1
    assert len(list(inputs.glob("rcp-agent-client-*.py"))) == 1
    assert not list(inputs.glob("*human-request.txt"))
    assert not list(inputs.glob("task-*-initial.md"))

    second_message = "/graph-audit Keep  this spacing.\nAnd this line."
    second_response = client.post(
        f"/api/projects/{project_id}/tasks/project_chat",
        json={
            "chat_id": chat_id,
            "message": second_message,
            "session_id": launcher.native_session_id,
            "run_truth_scope": ["repo-a"],
            "mode": "work",
        },
    )
    assert second_response.status_code == 202
    second_operation_id = second_response.json()["operation_id"]
    assert _wait_for_run(client, project_id, second_operation_id)["status"] == "succeeded"
    assert launcher.resumed_sessions == [None, launcher.native_session_id]
    assert launcher.workspaces == [workspace, workspace]
    second_artifacts = workspace / "turns" / second_operation_id / "artifacts"
    second_prompt = launcher.prompts[1]
    # A Work envelope is the marker, that turn's enforced write boundary, the unchanged human
    # bytes, and the delta — in that order and nothing else.
    assert second_prompt.startswith(
        f"This is a Work turn.\nArtifact directory for this turn: {second_artifacts}"
        f"\n\nEnforced write boundary on the machine this turn runs on:\n"
    )
    assert f"\n\n{second_message}\n\nRCP context update" in second_prompt
    assert second_prompt.index("Enforced write boundary") < second_prompt.index(second_message)
    second_delta = json.loads(second_prompt.split("RCP context update", 1)[1].split(":\n", 1)[1])
    assert set(second_delta) == {"patch"}
    assert set(second_delta["patch"]) == {
        "path",
        "watch_path",
        "schema_path",
        "validator_command",
        "validator_mailbox_id",
    }
    assert (
        second_delta["patch"]["validator_mailbox_id"]
        != prompt_values["patch"]["validator_mailbox_id"]
    )
    assert "rcp-agent-client-" in second_delta["patch"]["validator_command"]
    assert not (workspace / "current-turn.json").exists()
    assert len(list(inputs.glob("chat-master-v*.md"))) == 1
    assert len(list(inputs.glob("rcp-agent-client-*.py"))) == 2
    assert not list(inputs.glob("*human-request.txt"))
    launch_receipt = next(
        item
        for item in store.agent_task_receipts(second_operation_id)
        if item.category == "agent_prompt"
    )
    assert launch_receipt.payload["contract_path"] == str(master_path)

    third_message = "Use deeper reasoning for this turn."
    third_response = client.post(
        f"/api/projects/{project_id}/tasks/project_chat",
        json={
            "chat_id": chat_id,
            "message": third_message,
            "session_id": launcher.native_session_id,
            "run_truth_scope": ["repo-a"],
            "mode": "work",
            "reasoning": "high",
        },
    )
    assert third_response.status_code == 202
    third_operation_id = third_response.json()["operation_id"]
    assert _wait_for_run(client, project_id, third_operation_id)["status"] == "succeeded"
    third_prompt = launcher.prompts[2]
    third_artifacts = workspace / "turns" / third_operation_id / "artifacts"
    assert third_prompt.startswith(
        f"This is a Work turn.\nArtifact directory for this turn: {third_artifacts}"
        f"\n\nEnforced write boundary on the machine this turn runs on:\n"
    )
    assert f"\n\n{third_message}\n\nRCP context update" in third_prompt
    delta = json.loads(third_prompt.split("RCP context update", 1)[1].split(":\n", 1)[1])
    assert set(delta) == {"patch", "settings"}
    assert delta["settings"]["reasoning"] == "high"
    assert "rcp-agent-client-" in delta["patch"]["validator_command"]
    assert delta["patch"]["validator_mailbox_id"] != second_delta["patch"]["validator_mailbox_id"]
    assert len(list(inputs.glob("chat-master-v*.md"))) == 1
    assert len(list(inputs.glob("rcp-agent-client-*.py"))) == 3


@pytest.mark.parametrize(
    "provider",
    ["codex", "claude"],
)
def test_work_launch_receipt_names_the_canonical_state_boundary(
    manifest, tmp_path, provider: str
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    launcher = ScriptedLauncher([{}], message="Finished.")

    async def stream(_project_id, _kind, request, execution):
        async for frame in stream_work_run(
            service, launcher, request, tmp_path / "data", execution=execution
        ):
            yield frame

    app.state.background_tasks.stream = stream
    client = TestClient(app)
    project_id = app.state.default_project_id
    started = client.post(
        f"/api/projects/{project_id}/tasks/project_chat",
        json={
            "chat_id": str(uuid.uuid4()),
            "message": "Run it.",
            "run_truth_scope": ["repo-a"],
            "mode": "work",
            "provider": provider,
        },
    )
    completed = _wait_for_run(client, project_id, started.json()["operation_id"])

    assert completed["status"] == "succeeded"
    launch = next(
        receipt for receipt in completed["debug_receipts"] if receipt["category"] == "agent_launch"
    )
    repository_root = str(Path(manifest.repository_map["repo-a"].path).resolve())
    assert launch["payload"]["canonical_state_boundary"] == "provider_enforced"
    assert (
        launch["payload"]["provider_enforcement_mode"]
        == {
            "codex": "codex.permission-profile.v1",
            "claude": "claude.sandbox-allowlist.v1",
        }[provider]
    )
    assert launch["payload"]["canonical_repository_roots"] == [repository_root]
    assert launch["payload"]["canonical_write_roots"][1:] == [repository_root]
    assert str(Path(repository_root) / ".research") in launch["payload"]["protected_write_paths"]
    assert len(launch["payload"]["write_scope_fingerprint"]) == 64
    assert launch["payload"]["network_access"] is True


def test_work_write_scope_protects_and_rejects_canonical_research_pointer(
    manifest, tmp_path
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    request = RunRequest(
        chat_scope="project",
        chat_id=str(uuid.uuid4()),
        message="Run it.",
        run_truth_scope=["repo-a"],
        mode="work",
    )
    context = service.assemble_chat(request)
    workspace = tmp_path / "stage"
    workspace.mkdir()
    scope = _project_write_scope(
        context,
        service,
        "laptop",
        workspace=workspace,
        remote_stage=None,
        data_dir=tmp_path / "data",
        execution=None,
        capability="work_auto",
    )
    repository_root = Path(manifest.repository_map["repo-a"].path).resolve()
    canonical_research = repository_root / ".research"

    assert scope.repository_roots == [str(repository_root)]
    assert str(canonical_research) in scope.protected_write_paths

    unsafe_context = context.model_copy(
        update={
            "repositories": [
                RepositoryPointer(
                    alias="repo-a",
                    machine="laptop",
                    path=str(canonical_research),
                )
            ]
        }
    )

    with pytest.raises(ValueError, match="registered project root"):
        _project_write_scope(
            unsafe_context,
            service,
            "laptop",
            workspace=workspace,
            remote_stage=None,
            data_dir=tmp_path / "data",
            execution=None,
            capability="work_auto",
        )


def test_work_write_scope_rejects_pointer_with_parent_segments(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    context = service.assemble_chat(
        RunRequest(
            chat_scope="project",
            chat_id=str(uuid.uuid4()),
            message="Run it.",
            run_truth_scope=["repo-a"],
            mode="work",
        )
    )
    state_root = Path(manifest.repository_map["repo-a"].path)
    context = context.model_copy(
        update={
            "repositories": [
                RepositoryPointer(
                    alias="repo-a",
                    machine="laptop",
                    path=f"{state_root}/..",
                )
            ]
        }
    )
    workspace = tmp_path / "stage"
    workspace.mkdir()

    with pytest.raises(ValueError, match="registered project root"):
        _project_write_scope(
            context,
            service,
            "laptop",
            workspace=workspace,
            remote_stage=None,
            data_dir=tmp_path / "data",
            execution=None,
            capability="work_auto",
        )


@pytest.mark.parametrize("target", ["research", "ancestor"])
def test_local_work_write_scope_rejects_symlinked_pointer(manifest, tmp_path, target: str) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    context = service.assemble_chat(
        RunRequest(
            chat_scope="project",
            chat_id=str(uuid.uuid4()),
            message="Run it.",
            run_truth_scope=["repo-a"],
            mode="work",
        )
    )
    state_root = Path(manifest.repository_map["repo-a"].path)
    destination = state_root / ".research" if target == "research" else state_root.parent
    alias = tmp_path / f"canonical-{target}-alias"
    alias.symlink_to(destination, target_is_directory=True)
    context = context.model_copy(
        update={
            "repositories": [
                RepositoryPointer(
                    alias="repo-a",
                    machine="laptop",
                    path=str(alias),
                )
            ]
        }
    )
    workspace = tmp_path / "stage"
    workspace.mkdir()

    with pytest.raises(ValueError, match="registered project root"):
        _project_write_scope(
            context,
            service,
            "laptop",
            workspace=workspace,
            remote_stage=None,
            data_dir=tmp_path / "data",
            execution=None,
            capability="work_auto",
        )


@pytest.mark.asyncio
async def test_invalid_work_patch_is_corrected_without_repeating_operational_work(
    manifest, tmp_path
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    invalid = shape_invalid_patch().model_copy(update={"kind": "work"})
    valid = refresh_patch("rq/work-corrected").model_copy(update={"kind": "work"})

    class WorkLauncher(ScriptedLauncher):
        def __init__(self) -> None:
            super().__init__(
                [{"patch.json": agent_patch_json(invalid)}, {"patch.json": agent_patch_json(valid)}]
            )
            self.operational_effects = 0

        async def stream(self, provider, prompt, **kwargs):
            if self.calls == 0:
                self.operational_effects += 1
                self.message = "The experiment was submitted once."
            else:
                self.message = "The patch was corrected."
            async for event in super().stream(provider, prompt, **kwargs):
                yield event

    launcher = WorkLauncher()
    request = RunRequest(
        chat_scope="project",
        chat_id=str(uuid.uuid4()),
        message="Submit the experiment and record it.",
        run_truth_scope=["repo-a"],
        mode="work",
    )
    execution = _chat_task_execution(
        app.state.background_tasks.store,
        operation_id="corrected-work-patch",
        project_id=app.state.default_project_id,
        request=request,
    )

    frames = [
        frame
        async for frame in stream_work_run(
            service,
            launcher,
            request,
            tmp_path / "data",
            execution=execution,
        )
    ]

    assert not _error_texts(frames)
    graph_update = _graph_update(frames)
    assert graph_update is not None
    assert graph_update["status"] == "applied"
    assert graph_update["applied_revision"] == 3
    assert graph_update["correction_rounds"] == 1
    assert launcher.operational_effects == 1
    assert [item["capability"] for item in launcher.launch_kwargs] == [
        "work_auto",
        "work_auto",
    ]
    assert launcher.resumed_sessions == [None, launcher.native_session_id]
    assert launcher.launch_kwargs[1]["read_dirs"] == launcher.launch_kwargs[0]["read_dirs"]
    assert launcher.launch_kwargs[1]["write_dirs"] == launcher.launch_kwargs[0]["write_dirs"]
    correction_contract = next(
        content
        for name, content in launcher.input_snapshots[1].items()
        if name.endswith("work-correction-1.md")
    )
    assert "Work graph-correction instruction" in correction_contract
    assert "same native Work session" in correction_contract
    assert "Do not repeat a submission, experiment, message" in correction_contract
    answers = [event.text for event in _events(frames) if event.event == "answer"]
    assert answers == ["The experiment was submitted once."]


@pytest.mark.asyncio
async def test_exhausted_work_patch_correction_preserves_successful_answer(
    manifest, tmp_path
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    invalid = agent_patch_json(shape_invalid_patch().model_copy(update={"kind": "work"}))
    launcher = ScriptedLauncher([{"patch.json": invalid}], message="The run finished.")
    request = RunRequest(
        chat_scope="project",
        chat_id=str(uuid.uuid4()),
        message="Run it and reflect the result.",
        run_truth_scope=["repo-a"],
        mode="work",
    )

    frames = [
        frame async for frame in stream_work_run(service, launcher, request, tmp_path / "data")
    ]

    assert not _error_texts(frames)
    graph_update = _graph_update(frames)
    assert graph_update is not None
    assert graph_update["status"] == "rejected"
    assert graph_update["proposal_ids"] == []
    assert graph_update["correction_rounds"] == 2
    assert graph_update["repairable"] is False
    assert launcher.calls == 3
    assert service.history.state().revision == 2
    # The agent rewrote nothing, so the second correction must say so instead of
    # repeating the first diagnostic as though the file had changed.
    second_correction = next(
        content
        for name, content in launcher.input_snapshots[2].items()
        if name.endswith("work-correction-2.json")
    )
    assert "byte-identical" in second_correction


@pytest.mark.asyncio
async def test_unreadable_corrected_work_patch_reports_the_read_failure(manifest, tmp_path) -> None:
    """A correction that writes an undecodable patch must not be told it wrote nothing.

    The two diagnostics send the agent to different fixes, so collapsing the
    read failure into "you never wrote the file" makes it rewrite the same
    unreadable bytes and spend the whole correction budget on the wrong problem.
    """

    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    invalid = agent_patch_json(shape_invalid_patch().model_copy(update={"kind": "work"}))

    class UnreadableCorrectionLauncher(ScriptedLauncher):
        async def stream(self, provider, prompt, **kwargs):
            launch = self.calls
            async for event in super().stream(provider, prompt, **kwargs):
                yield event
            if launch > 0:
                (self.workspaces[-1] / "patch.json").write_bytes(b'{"summary": "\xff\xfe"}')

    launcher = UnreadableCorrectionLauncher([{"patch.json": invalid}], message="The run finished.")
    request = RunRequest(
        chat_scope="project",
        chat_id=str(uuid.uuid4()),
        message="Run it and reflect the result.",
        run_truth_scope=["repo-a"],
        mode="work",
    )

    frames = [
        frame async for frame in stream_work_run(service, launcher, request, tmp_path / "data")
    ]

    assert not _error_texts(frames)
    graph_update = _graph_update(frames)
    assert graph_update is not None
    assert graph_update["status"] == "rejected"
    assert graph_update["correction_rounds"] == 2
    assert service.history.state().revision == 2
    second_correction = next(
        content
        for name, content in launcher.input_snapshots[2].items()
        if name.endswith("work-correction-2.json")
    )
    assert "could not be read" in second_correction
    assert "without writing patch.json" not in second_correction
    assert any("could not be read" in item for item in graph_update["validation_messages"])


@pytest.mark.asyncio
async def test_work_patch_is_applied_to_live_state_without_correction(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    work_patch = refresh_patch("rq/live-work").model_copy(update={"kind": "work"})

    class MovingGraphLauncher(ScriptedLauncher):
        async def stream(self, provider, prompt, **kwargs):
            append_fixture_patch(service, refresh_patch("rq/concurrent-human-work"))
            async for event in super().stream(provider, prompt, **kwargs):
                yield event

    launcher = MovingGraphLauncher(
        [{"patch.json": agent_patch_json(work_patch)}], message="The operation completed."
    )
    request = RunRequest(
        chat_scope="project",
        chat_id=str(uuid.uuid4()),
        message="Do the work.",
        run_truth_scope=["repo-a"],
        mode="work",
    )
    execution = _chat_task_execution(
        app.state.background_tasks.store,
        operation_id="moving-live-work",
        project_id=app.state.default_project_id,
        request=request,
    )

    frames = [
        frame
        async for frame in stream_work_run(
            service,
            launcher,
            request,
            tmp_path / "data",
            execution=execution,
        )
    ]

    assert not _error_texts(frames)
    graph_update = _graph_update(frames)
    assert graph_update is not None
    assert graph_update["status"] == "applied"
    assert graph_update["applied_revision"] == 4
    assert graph_update["correction_rounds"] == 0
    assert launcher.calls == 1
    assert "rq/concurrent-human-work" in service.history.state().nodes
    assert "rq/live-work" in service.history.state().nodes


@pytest.mark.asyncio
async def test_work_apply_rechecks_authority_after_human_removes_proposal_target(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    hypothesis_id = "hyp/replanning-restores-plasticity"
    authorizer = _named_test_authorizer(app.state.background_tasks.store)
    service.review_node(
        hypothesis_id,
        ReviewRequest(standing="accepted"),
        authorized_by=authorizer,
    )
    removal_id = "prop/remove-hypothesis-while-work-runs"
    append_fixture_patch(
        service,
        Patch(
            kind="refresh",
            author="agent",
            summary="Proposed removing the accepted hypothesis.",
            run_truth_scope=["repo-a"],
            repositories_read=["repo-a"],
            ops=[
                {
                    "op": "create_proposals",
                    "proposals": [
                        {
                            "id": removal_id,
                            "title": "Remove the replanning hypothesis",
                            "card": {
                                "decision_needed": "Decide whether to remove this hypothesis."
                            },
                            "ops": [
                                {
                                    "op": "remove_nodes",
                                    "intent": "removal",
                                    "node_ids": [hypothesis_id],
                                }
                            ],
                        }
                    ],
                }
            ],
        ),
    )
    work_patch = Patch(
        kind="work",
        author="agent",
        summary="Proposed clarifying the accepted hypothesis.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "create_proposals",
                "proposals": [
                    {
                        "id": "prop/clarify-hypothesis-after-dispatch",
                        "title": "Clarify the replanning hypothesis",
                        "card": {
                            "decision_needed": "Decide whether to use the clarified statement."
                        },
                        "ops": [
                            {
                                "op": "update_nodes",
                                "intent": "content_change",
                                "nodes": [
                                    {
                                        "id": hypothesis_id,
                                        "changes": {
                                            "statement": (
                                                "Search-time replanning restores plasticity "
                                                "after repeated task shifts."
                                            )
                                        },
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    )
    effect_path = Path(manifest.repository_map["repo-a"].path) / "s100-effect.txt"

    class HeldWorkLauncher(ScriptedLauncher):
        async def stream(self, provider, prompt, **kwargs):
            if self.calls == 0:
                effect_path.write_text("completed before Apply\n", encoding="utf-8")
            async for event in super().stream(provider, prompt, **kwargs):
                yield event

    launcher = HeldWorkLauncher(
        [{"patch.json": agent_patch_json(work_patch)}],
        message="The operational work completed and the proposed wording is ready for review.",
    )
    request = RunRequest(
        chat_scope="project",
        chat_id=str(uuid.uuid4()),
        message="Do the work and propose the clarified hypothesis.",
        run_truth_scope=["repo-a"],
        mode="work",
    )
    execution = _chat_task_execution(
        app.state.background_tasks.store,
        operation_id="work-held-before-live-apply",
        project_id=app.state.default_project_id,
        request=request,
    )
    workspace = service.history.workspace
    original_run_lock = workspace.run_lock
    graph_moved = False

    @contextmanager
    def move_graph_before_apply(**kwargs):
        nonlocal graph_moved
        with original_run_lock(**kwargs) as lease:
            if not graph_moved:
                service.decide_proposal(
                    removal_id,
                    ProposalDecisionRequest(decision="approved"),
                    authorized_by=authorizer,
                )
                graph_moved = True
            yield lease

    monkeypatch.setattr(workspace, "run_lock", move_graph_before_apply)
    frames = [
        frame
        async for frame in stream_work_run(
            service,
            launcher,
            request,
            tmp_path / "data",
            execution=execution,
        )
    ]

    assert graph_moved is True
    assert effect_path.read_text(encoding="utf-8") == "completed before Apply\n"
    assert [event.text for event in _events(frames) if event.event == "answer"] == [
        "The operational work completed and the proposed wording is ready for review."
    ]
    graph_update = _graph_update(frames)
    assert graph_update is not None
    assert graph_update["status"] == "rejected"
    assert hypothesis_id not in service.history.state().nodes
    assert "prop/clarify-hypothesis-after-dispatch" not in service.history.state().proposals


@pytest.mark.asyncio
async def test_work_lock_ownership_loss_preserves_the_answer_and_skips_graph_apply(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    workspace = service.history.workspace
    work_patch = refresh_patch("rq/lost-lock-work").model_copy(update={"kind": "work"})

    @contextmanager
    def lost_lock(*, on_wait=None, cancelled=None, on_lost=None):
        del on_wait, cancelled
        lease = RunLockLease(workspace.location, on_lost=on_lost)
        lease._mark_lost(
            f"Canonical-state lock holder for {workspace.location} exited unexpectedly."
        )
        yield lease

    monkeypatch.setattr(workspace, "run_lock", lost_lock)
    launcher = ScriptedLauncher(
        [{"patch.json": agent_patch_json(work_patch)}],
        message="The operational work completed.",
    )
    request = RunRequest(
        chat_scope="project",
        chat_id=str(uuid.uuid4()),
        message="Do the work and reflect it.",
        run_truth_scope=["repo-a"],
        mode="work",
    )

    frames = [
        frame async for frame in stream_work_run(service, launcher, request, tmp_path / "data")
    ]

    assert [event.text for event in _events(frames) if event.event == "answer"] == [
        "The operational work completed."
    ]
    graph_update = _graph_update(frames)
    assert graph_update is not None
    assert graph_update["status"] == "rejected"
    assert graph_update["repairable"] is False
    assert "lock holder" in graph_update["validation_messages"][0]
    assert launcher.calls == 1
    assert service.history.state().revision == 2


@pytest.mark.asyncio
async def test_work_patch_adds_decision_edges_to_an_accepted_question_without_correction(
    manifest, tmp_path
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    service.review_node(
        "rq/learning-after-shift",
        ReviewRequest(standing="accepted"),
        authorized_by=_named_test_authorizer(app.state.background_tasks.store),
    )
    patch = Patch(
        kind="work",
        author="agent",
        summary="Added the decisions that operationalize the accepted question.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": "dec/evaluation-rule",
                        "type": "decision",
                        "title": "Evaluation rule",
                        "question": "Which evaluation rule should govern the experiment?",
                        "options": ["matched", "shifted"],
                    },
                    {
                        "id": "dec/intervention-budget",
                        "type": "decision",
                        "title": "Intervention budget",
                        "question": "Which intervention budget should the experiment use?",
                        "options": ["small", "large"],
                    },
                ],
            },
            {
                "op": "create_edges",
                "edges": [
                    {
                        "source": "rq/learning-after-shift",
                        "target": "dec/evaluation-rule",
                        "relation": "has_decision",
                    },
                    {
                        "source": "rq/learning-after-shift",
                        "target": "dec/intervention-budget",
                        "relation": "has_decision",
                    },
                ],
            },
        ],
    )
    launcher = ScriptedLauncher(
        [{"patch.json": agent_patch_json(patch)}],
        message="I added the experiment decisions.",
    )
    request = RunRequest(
        chat_scope="project",
        chat_id=str(uuid.uuid4()),
        message="Add the decision structure.",
        run_truth_scope=["repo-a"],
        mode="work",
    )
    execution = _chat_task_execution(
        app.state.background_tasks.store,
        operation_id="accepted-question-decisions",
        project_id=app.state.default_project_id,
        request=request,
    )

    frames = [
        frame
        async for frame in stream_work_run(
            service,
            launcher,
            request,
            tmp_path / "data",
            execution=execution,
        )
    ]

    graph_update = _graph_update(frames)
    assert graph_update is not None
    assert graph_update["status"] == "applied"
    assert graph_update["correction_rounds"] == 0
    assert launcher.calls == 1
    state = service.history.state()
    assert "rq/learning-after-shift::has_decision::dec/evaluation-rule" in state.edges
    assert "rq/learning-after-shift::has_decision::dec/intervention-budget" in state.edges
    assert state.nodes["rq/learning-after-shift"].standing == "accepted"


@pytest.mark.asyncio
async def test_work_proposal_is_applied_as_a_proposal_not_a_universal_gate(
    manifest, tmp_path
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    patch = Patch(
        kind="work",
        author="agent",
        summary="Proposed activating the hypothesis.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": "ev/work-activation-result",
                        "type": "evidence",
                        "title": "Work activation result",
                        "observation": "The completed work supports activating the hypothesis.",
                        "origin": "internal_run",
                    }
                ],
            },
            {
                "op": "create_edges",
                "edges": [
                    {
                        "id": "edge/work-activation-support",
                        "source": "ev/work-activation-result",
                        "target": "hyp/replanning-restores-plasticity",
                        "relation": "supports",
                        "assessment": {
                            "relevance": "direct",
                            "weight": "moderate",
                        },
                    }
                ],
            },
            {
                "op": "create_proposals",
                "proposals": [
                    {
                        "id": "prop/work-activate-hypothesis",
                        "title": "Activate the replanning hypothesis",
                        "card": {
                            "situation_cold": "The hypothesis remains proposed.",
                            "why_human_now": "Activation changes experiment interpretation.",
                            "consequences": "New evidence will be organized around it.",
                            "decision_needed": "Decide whether to activate it.",
                        },
                        "ops": [
                            {
                                "op": "update_nodes",
                                "intent": "status_change",
                                "nodes": [
                                    {
                                        "id": "hyp/replanning-restores-plasticity",
                                        "changes": {"status": "active"},
                                        "cause": {
                                            "kind": "evidence_edge",
                                            "ref_id": "edge/work-activation-support",
                                        },
                                    }
                                ],
                            }
                        ],
                        "related_node_ids": ["hyp/replanning-restores-plasticity"],
                        "base_rev": 2,
                    }
                ],
            },
        ],
        change_summary=["Proposed activating the hypothesis."],
    )
    launcher = ScriptedLauncher(
        [{"patch.json": agent_patch_json(patch)}], message="I sent the decision to Inbox."
    )
    request = RunRequest(
        chat_scope="project",
        chat_id=str(uuid.uuid4()),
        message="Prepare the decision.",
        run_truth_scope=["repo-a"],
        mode="work",
    )
    execution = _chat_task_execution(
        app.state.background_tasks.store,
        operation_id="work-proposal",
        project_id=app.state.default_project_id,
        request=request,
    )

    frames = [
        frame
        async for frame in stream_work_run(
            service,
            launcher,
            request,
            tmp_path / "data",
            execution=execution,
        )
    ]

    graph_update = _graph_update(frames)
    assert graph_update is not None
    assert graph_update["status"] == "applied"
    assert graph_update["proposal_ids"] == ["prop/work-activate-hypothesis"]
    state = service.history.state()
    assert state.proposals["prop/work-activate-hypothesis"].status == "pending"
    assert state.nodes["hyp/replanning-restores-plasticity"].status == "proposed"
    assert "ev/work-activation-result" in state.nodes
    assert "edge/work-activation-support" in state.edges


def test_background_work_can_pause_while_waiting_for_canonical_state(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    workspace = service.history.workspace
    waiting = threading.Event()
    answer = "The operational work completed."
    patch_text = agent_patch_json(
        refresh_patch("rq/paused-work-apply").model_copy(update={"kind": "work"})
    )
    launcher = ScriptedLauncher([{"patch.json": patch_text}], message=answer)

    @contextmanager
    def contended_lock(*, on_wait=None, cancelled=None, on_lost=None):
        del on_lost
        if on_wait is not None:
            on_wait("Waiting for another graph-writing run to release canonical state.")
        waiting.set()
        while cancelled is None or not cancelled():
            time.sleep(0.01)
        raise RunLockCancelled("Run-lock acquisition was cancelled while waiting.")
        yield  # pragma: no cover - makes this function a context manager

    async def stream(_project_id, _kind, request, execution):
        async for frame in stream_work_run(
            service, launcher, request, tmp_path / "data", execution=execution
        ):
            yield frame

    monkeypatch.setattr(workspace, "run_lock", contended_lock)
    app.state.background_tasks.stream = stream
    client = TestClient(app)
    project_id = app.state.default_project_id
    started = client.post(
        f"/api/projects/{project_id}/tasks/project_chat",
        json={
            "chat_id": str(uuid.uuid4()),
            "message": "Do the work and reflect it.",
            "run_truth_scope": ["repo-a"],
            "mode": "work",
        },
    )
    operation_id = started.json()["operation_id"]

    assert waiting.wait(timeout=2)
    active = client.get(f"/api/projects/{project_id}/tasks/{operation_id}").json()
    assert active["status"] == "running"
    assert active["phase"] == "waiting"
    assert "Waiting for another graph-writing run" in active["status_message"]
    paused_request = client.post(f"/api/projects/{project_id}/tasks/{operation_id}/pause")

    assert paused_request.status_code == 202
    paused = _wait_for_status(client, project_id, operation_id, {"paused"})
    assert paused["error"] is None
    assert paused["result"] == {"messages": [answer]}
    assert "answer and retained patch are preserved" in paused["status_message"]
    assert app.state.background_tasks.store.agent_task_patch_output(operation_id) == patch_text
    assert service.history.state().revision == 2
    categories = {item["category"] for item in paused["debug_receipts"]}
    assert "canonical_state_lock_wait" in categories
    assert "operation_paused" in categories


def test_background_work_rejection_succeeds_and_manual_repair_is_idempotent(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    invalid = agent_patch_json(shape_invalid_patch().model_copy(update={"kind": "work"}))
    valid = agent_patch_json(
        refresh_patch("rq/manually-repaired-work").model_copy(update={"kind": "work"})
    )
    launcher = ScriptedLauncher(
        [
            {"patch.json": invalid},
            {"patch.json": invalid},
            {"patch.json": invalid},
            {"patch.json": valid},
        ],
        message="The operational work completed.",
    )

    async def stream(_project_id, _kind, request, execution):
        async for frame in stream_work_run(
            service, launcher, request, tmp_path / "data", execution=execution
        ):
            yield frame

    app.state.background_tasks.stream = stream
    client = TestClient(app)
    project_id = app.state.default_project_id
    chat_id = str(uuid.uuid4())
    started = client.post(
        f"/api/projects/{project_id}/tasks/project_chat",
        json={
            "chat_id": chat_id,
            "message": "Run the operation and reflect it.",
            "run_truth_scope": ["repo-a"],
            "mode": "work",
        },
    )
    assert started.status_code == 202
    parent = _wait_for_run(client, project_id, started.json()["operation_id"])
    assert parent["status"] == "succeeded", parent
    assert parent["status_message"] == "Completed; graph update rejected."
    assert parent["result"]["messages"] == ["The operational work completed."]
    assert parent["result"]["graph_update"]["status"] == "rejected"
    assert parent["result"]["graph_update"]["proposal_ids"] == []
    assert parent["result"]["graph_update"]["repairable"] is True
    assert (
        app.state.background_tasks.store.agent_task_patch_output(parent["operation_id"]) == invalid
    )
    transcript = service.chat_transcript(chat_id)
    assert transcript is not None
    assert {message.operation_id for message in transcript.messages} == {parent["operation_id"]}

    ownership_check = app.state.background_tasks._session_is_rcp_owned
    monkeypatch.setattr(
        app.state.background_tasks,
        "_session_is_rcp_owned",
        lambda _record: False,
    )
    unowned = client.post(
        f"/api/projects/{project_id}/tasks/{parent['operation_id']}/repair-graph-update"
    )
    assert unowned.status_code == 409
    still_repairable = client.get(
        f"/api/projects/{project_id}/tasks/{parent['operation_id']}"
    ).json()
    assert still_repairable["result"]["graph_update"]["repairable"] is True
    monkeypatch.setattr(
        app.state.background_tasks,
        "_session_is_rcp_owned",
        ownership_check,
    )
    append_fixture_patch(service, refresh_patch("rq/landed-before-manual-repair"))

    repaired_response = client.post(
        f"/api/projects/{project_id}/tasks/{parent['operation_id']}/repair-graph-update"
    )
    assert repaired_response.status_code == 202
    duplicate = client.post(
        f"/api/projects/{project_id}/tasks/{parent['operation_id']}/repair-graph-update"
    )
    assert duplicate.status_code == 409
    repaired = _wait_for_run(client, project_id, repaired_response.json()["operation_id"])

    assert repaired["status"] == "succeeded"
    assert repaired["parent_operation_id"] == parent["operation_id"]
    assert repaired["request"]["message"] is None
    assert repaired["result"]["messages"] == []
    assert repaired["result"]["graph_update"]["status"] == "applied"
    assert repaired["result"]["graph_update"]["applied_revision"] == 4
    assert "rq/landed-before-manual-repair" in service.history.state().nodes
    assert "rq/manually-repaired-work" in service.history.state().nodes
    parent_after = client.get(f"/api/projects/{project_id}/tasks/{parent['operation_id']}").json()
    assert parent_after["result"]["graph_update"]["repairable"] is False
    transcript = service.chat_transcript(chat_id)
    assert transcript is not None
    assert transcript.messages[-1].operation_id == repaired["operation_id"]
    assert transcript.messages[-1].text == ""
    assert transcript.messages[-1].graph_update is not None
    assert transcript.messages[-1].graph_update.status == "applied"
    assert transcript.last_message_preview == "The operational work completed."


def _experiment_fixture_patch(
    experiment_id: str = "exp/bounded-loop",
    *,
    invocation_ceiling: int = 2,
) -> Patch:
    return Patch(
        kind="refresh",
        author="agent",
        summary="Added an experiment for control-loop tests.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": experiment_id,
                        "type": "experiment",
                        "title": "Bounded loop",
                        "objective": "Exercise the experiment control contract.",
                        "completion_criteria": ["The detached fixture exits cleanly."],
                        "invocation_ceiling": invocation_ceiling,
                    }
                ],
            }
        ],
    )


def test_experiment_watcher_delivery_uses_live_episode_not_maintenance_provenance(
    manifest, tmp_path: Path
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    append_fixture_patch(service, _experiment_fixture_patch(invocation_ceiling=3))
    project_id = app.state.default_project_id
    store = app.state.background_tasks.store
    episode_id = str(uuid.uuid4())
    loop_chat_id = str(uuid.uuid4())
    maintenance_chat_id = str(uuid.uuid4())
    stage_root = tmp_path / "episode-stage"
    stage_root.mkdir()
    now = store.now()
    authorized_by = _named_test_authorizer(store)
    root_request = RunRequest(
        provider="codex",
        model="episode-model",
        reasoning="medium",
        run_on="laptop",
        run_truth_scope=["repo-a"],
        chat_id=loop_chat_id,
        chat_scope="node",
        node_id="exp/bounded-loop",
        mode="work",
        trigger="experiment_run",
        patch_kind="experiment_loop",
        control_node_id="exp/bounded-loop",
        control_revision=3,
        control_episode_id=episode_id,
        control_invocation=1,
        control_invocation_ceiling=3,
        control_decision_bundle=[],
        control_completion_criteria=["The detached fixture exits cleanly."],
    )
    store.create_experiment_episode_with_invocation(
        AgentTaskRecord(
            operation_id="loop-root-for-maintenance-wake",
            project_id=project_id,
            episode_id=episode_id,
            kind="node_chat",
            status="queued",
            request=root_request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="Loop waiting on detached work.",
            authorized_by=authorized_by,
        )
    )
    store.complete_agent_task(
        "loop-root-for-maintenance-wake",
        applied_revision=None,
        result={},
    )
    store.create_agent_task(
        AgentTaskRecord(
            operation_id="maintenance-project-chat-turn",
            project_id=project_id,
            kind="project_chat",
            status="succeeded",
            request={},
            created_at=now,
            updated_at=now,
            status_message="Maintenance watcher origin fixture.",
            authorized_by=authorized_by,
        )
    )
    store.commit_experiment_episode_turn(
        episode_id=episode_id,
        project_id=project_id,
        control_node_id="exp/bounded-loop",
        provider="codex",
        execution_machine="laptop",
        execution_host="",
        native_session_id="episode-native-session",
        stage_host=None,
        stage_root=str(stage_root),
        chat_id=loop_chat_id,
        operation_id="loop-root-for-maintenance-wake",
        invocation=1,
        graph_result="no graph change",
        watcher_ids=["maintenance-origin-observer"],
        context_baseline={},
    )
    watcher = WatcherRecord(
        watcher_id="maintenance-origin-observer",
        project_id=project_id,
        origin_operation_id="maintenance-project-chat-turn",
        origin_task_kind="project_chat",
        chat_id=maintenance_chat_id,
        node_id="exp/bounded-loop",
        execution_host="",
        check_command="true",
        log_path=str(tmp_path / "maintenance-observer.log"),
        cwd=str(tmp_path),
        continuation=WatcherContinuation(
            provider="claude",
            model="maintenance-model",
            reasoning="high",
            run_on="maintenance-machine",
            run_truth_scope=["repo-b"],
            patch_kind="experiment_loop",
            control_node_id="exp/bounded-loop",
            control_revision=3,
            control_episode_id=episode_id,
            control_invocation=1,
            control_invocation_ceiling=3,
            control_decision_bundle=[],
            control_completion_criteria=["The detached fixture exits cleanly."],
        ),
        status="completed",
        created_at=now,
        completed_at=now,
    )
    store.create_watchers([watcher])
    captured: dict[str, object] = {}
    entered = threading.Event()

    async def stream(project, kind, request, _execution):
        captured.update(
            project_id=project,
            kind=kind,
            request=request,
        )
        entered.set()
        yield _sse(AgentEvent(event="answer", text="Continued the live loop."))
        yield _sse(AgentEvent(event="done"))

    app.state.background_tasks.stream = stream

    stored = store.watcher(watcher.watcher_id)
    assert stored is not None
    assert app.state.watcher_poller.on_completed is not None
    app.state.watcher_poller.on_completed([stored])
    assert entered.wait(timeout=1)

    request = captured["request"]
    assert isinstance(request, RunRequest)
    assert captured["project_id"] == project_id
    assert captured["kind"] == "node_chat"
    assert request.watcher_ids == [watcher.watcher_id]
    assert request.chat_scope == "node"
    assert request.node_id == "exp/bounded-loop"
    assert request.chat_id == loop_chat_id
    assert request.session_id == "episode-native-session"
    assert request.provider == "codex"
    assert request.model == "episode-model"
    assert request.reasoning == "medium"
    assert request.run_on == "laptop"
    assert request.run_truth_scope == ["repo-a"]
    assert request.chat_id != maintenance_chat_id
    delivered = store.watcher(watcher.watcher_id)
    assert delivered is not None
    assert delivered.notified is True
    assert delivered.notification_operation_id is not None
    notification = store.agent_task(delivered.notification_operation_id)
    assert notification is not None
    assert notification.kind == "node_chat"
    assert notification.authorized_by == authorized_by
    assert notification.stage_host is None
    assert notification.stage_root == str(stage_root)


def _chat_task_execution(
    store: AppStore,
    *,
    operation_id: str,
    project_id: str,
    request: RunRequest,
) -> AgentTaskExecution:
    now = store.now()
    authorized_by = _named_test_authorizer(store)
    kind = "node_chat" if request.chat_scope == "node" else "project_chat"
    dispatch_authority = resolve_dispatch_authority(kind, request)
    assert dispatch_authority is not None
    record = AgentTaskRecord(
        operation_id=operation_id,
        project_id=project_id,
        episode_id=(
            request.control_episode_id if request.patch_kind == "experiment_loop" else None
        ),
        kind=kind,
        status="queued" if request.patch_kind == "experiment_loop" else "running",
        request=request.model_dump(mode="json"),
        created_at=now,
        updated_at=now,
        status_message="running",
        authorized_by=authorized_by,
        dispatch_authority=dispatch_authority,
    )
    if request.patch_kind == "experiment_loop":
        store.create_experiment_episode_with_invocation(record)
        store.mark_agent_task_running(operation_id)
    else:
        store.create_agent_task(record)
    return AgentTaskExecution(
        operation_id=operation_id,
        store=store,
        control=AgentProcessControl(),
    )


def test_public_task_request_cannot_select_watcher_or_control_authority(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    request = _validated_task_request(
        service,
        "project_chat",
        {
            "chat_id": str(uuid.uuid4()),
            "message": "Ordinary Work.",
            "mode": "work",
            "trigger": "watcher",
            "patch_kind": "experiment_loop",
            "control_node_id": "experiment/forged",
            "watcher_ids": ["forged"],
        },
    )

    assert isinstance(request, RunRequest)
    assert request.trigger == "human"
    assert request.patch_kind == "work"
    assert request.control_node_id is None
    assert request.watcher_ids == []


def test_run_endpoint_pins_control_without_spending_an_attempt(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    append_fixture_patch(service, _experiment_fixture_patch())
    entered = threading.Event()
    release = threading.Event()

    async def stream(_project_id, kind, request, _execution):
        assert kind == "node_chat"
        assert request.trigger == "experiment_run"
        assert request.patch_kind == "experiment_loop"
        assert request.control_node_id == "exp/bounded-loop"
        assert request.control_revision == 3
        assert request.control_episode_id is not None
        uuid.UUID(request.control_episode_id)
        assert request.control_invocation == 1
        assert request.control_invocation_ceiling == 2
        assert request.control_completion_criteria == ["The detached fixture exits cleanly."]
        entered.set()
        while not release.is_set():
            await asyncio.sleep(0.01)
        yield _sse(AgentEvent(event="answer", text="Preflight stopped before launch."))
        yield _sse(AgentEvent(event="done"))

    app.state.background_tasks.stream = stream
    client = TestClient(app)
    project_id = app.state.default_project_id
    response = client.post(
        f"/api/projects/{project_id}/experiments/exp%2Fbounded-loop/run",
        json={"chat_id": str(uuid.uuid4()), "run_truth_scope": ["repo-a"]},
    )
    try:
        assert response.status_code == 202, response.text
        assert entered.wait(timeout=1)
        task = response.json()
        assert task["request"]["control_decision_bundle"] == []
        assert service.history.state().nodes["exp/bounded-loop"].attempts == []

        snapshot = client.get(f"/api/projects/{project_id}").json()
        control = snapshot["experiment_control"]["exp/bounded-loop"]
        assert control["active"] is True
        assert control["ready"] is False
        assert control["invocations_used"] == 1
        assert control["invocation_ceiling"] == 2
        assert control["invocations_remaining"] == 1
        assert control["reasons"] == ["An experiment loop is already active."]
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
                "can_open_report",
            )
        } == {
            "health": "agent_active",
            "recommendation": "wait",
            "run_section": "running",
            "live": True,
            "can_start": False,
            "can_stop": True,
            "stop_pending": False,
            "task_control": None,
            "can_switch_provider": False,
            "can_open_report": False,
        }

        duplicate = client.post(
            f"/api/projects/{project_id}/experiments/exp%2Fbounded-loop/run",
            json={"chat_id": str(uuid.uuid4()), "run_truth_scope": ["repo-a"]},
        )
        assert duplicate.status_code == 409
        assert "already active" in duplicate.json()["detail"]
    finally:
        release.set()

    completed = _wait_for_run(client, project_id, response.json()["operation_id"])
    assert completed["status"] == "succeeded"
    assert service.history.state().nodes["exp/bounded-loop"].attempts == []


def test_run_endpoint_preserves_a_nonblank_experiment_goal(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    append_fixture_patch(service, _experiment_fixture_patch())
    goal = "  Establish whether the bounded runner survives a graceful restart.  "
    seen = threading.Event()

    async def stream(_project_id, kind, request, _execution):
        assert kind == "node_chat"
        assert request.message == goal
        seen.set()
        yield _sse(AgentEvent(event="answer", text="Goal accepted."))
        yield _sse(AgentEvent(event="done"))

    app.state.background_tasks.stream = stream
    client = TestClient(app)
    project_id = app.state.default_project_id
    response = client.post(
        f"/api/projects/{project_id}/experiments/exp%2Fbounded-loop/run",
        json={"chat_id": str(uuid.uuid4()), "message": goal},
    )

    assert response.status_code == 202, response.text
    assert response.json()["request"]["message"] == goal
    assert seen.wait(timeout=1)


def test_human_run_claims_over_ceiling_completion_into_a_new_episode(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    append_fixture_patch(service, _experiment_fixture_patch(invocation_ceiling=1))
    project_id = app.state.default_project_id
    store = app.state.background_tasks.store
    old_episode = str(uuid.uuid4())
    loop_chat_id = str(uuid.uuid4())
    maintenance_chat_id = str(uuid.uuid4())
    now = store.now()
    authorized_by = _named_test_authorizer(store)
    old_request = RunRequest(
        provider="codex",
        run_on="laptop",
        run_truth_scope=["repo-a"],
        chat_id=loop_chat_id,
        chat_scope="node",
        node_id="exp/bounded-loop",
        mode="work",
        trigger="experiment_run",
        patch_kind="experiment_loop",
        control_node_id="exp/bounded-loop",
        control_revision=3,
        control_episode_id=old_episode,
        control_invocation=1,
        control_invocation_ceiling=1,
        control_decision_bundle=[],
        control_completion_criteria=["The detached fixture exits cleanly."],
    )
    store.create_experiment_episode_with_invocation(
        AgentTaskRecord(
            operation_id="old-loop-root",
            project_id=project_id,
            episode_id=old_episode,
            kind="node_chat",
            status="queued",
            request=old_request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="complete",
            authorized_by=authorized_by,
        )
    )
    store.complete_agent_task("old-loop-root", applied_revision=None, result={})
    store.create_watchers(
        [
            WatcherRecord(
                watcher_id="pending-loop-watcher",
                project_id=project_id,
                origin_operation_id="old-loop-root",
                origin_task_kind="project_chat",
                chat_id=maintenance_chat_id,
                node_id="exp/bounded-loop",
                check_command="true",
                log_path="/tmp/pending-loop.log",
                cwd="/tmp",
                continuation=WatcherContinuation(
                    provider="claude",
                    model="maintenance-model",
                    reasoning="high",
                    run_on="maintenance-machine",
                    run_truth_scope=["repo-b"],
                    patch_kind="experiment_loop",
                    control_node_id="exp/bounded-loop",
                    control_revision=3,
                    control_episode_id=old_episode,
                    control_invocation=1,
                    control_invocation_ceiling=1,
                    control_decision_bundle=[],
                    control_completion_criteria=["The detached fixture exits cleanly."],
                ),
                created_at=now,
            )
        ]
    )

    client = TestClient(app)
    still_running = client.post(
        f"/api/projects/{project_id}/experiments/exp%2Fbounded-loop/run",
        json={"chat_id": str(uuid.uuid4())},
    )
    assert still_running.status_code == 409
    assert still_running.json()["detail"] == "Detached Experiment work is still running."
    control = client.get(f"/api/projects/{project_id}").json()["experiment_control"][
        "exp/bounded-loop"
    ]
    assert control["active"] is False
    assert control["paused"] is True
    assert control["ready"] is False
    assert control["reasons"] == ["Detached Experiment work is still running."]
    # The projection ships its own split, so no client has to recognise which of
    # these sentences a human resolves in the graph.
    assert control["graph_reasons"] == []
    assert control["operational"]["episode_live"] is True

    store.record_watcher_check(
        "pending-loop-watcher",
        status="completed",
        exit_code=0,
        error=None,
    )
    assert app.state.watcher_poller.on_completed is not None
    app.state.watcher_poller.on_completed([store.watcher("pending-loop-watcher")])
    assert store.watcher("pending-loop-watcher").notified is False
    assert app.state.watcher_poller.on_poll_completed is not None
    app.state.watcher_poller.on_poll_completed()
    exhausted_parent = store.episode(old_episode)
    assert exhausted_parent is not None
    assert exhausted_parent.status == "needs_action"
    assert exhausted_parent.ending == "exhausted"
    # The fixture turn bound no native session, so no report was ever possible and
    # the episode terminalizes without one rather than posting a report error.
    assert exhausted_parent.wrapup_state == "not_started"
    assert exhausted_parent.wrapup_error is None
    assert store.episode_wrapup(old_episode) is None
    completed_control = client.get(f"/api/projects/{project_id}").json()["experiment_control"][
        "exp/bounded-loop"
    ]
    assert completed_control["paused"] is False
    assert completed_control["ready"] is True
    assert completed_control["reasons"] == []
    assert {
        field: completed_control[field]
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
            "can_open_report",
        )
    } == {
        "health": "paused_at_limit",
        "recommendation": "start_episode",
        "run_section": "actionable",
        "live": False,
        "can_start": True,
        "can_stop": False,
        "stop_pending": False,
        "task_control": None,
        "can_switch_provider": False,
        "can_open_report": False,
    }

    new_loop_chat_id = str(uuid.uuid4())

    async def stream(_project_id, kind, request, _execution):
        assert kind == "node_chat"
        assert request.chat_scope == "node"
        assert request.node_id == "exp/bounded-loop"
        assert request.chat_id == new_loop_chat_id
        assert request.chat_id != maintenance_chat_id
        assert request.provider == "codex"
        assert request.model == ""
        assert request.reasoning == "medium"
        assert request.run_on == "laptop"
        assert request.run_truth_scope == ["repo-a"]
        assert request.message == ("Begin a bounded Experiment-loop episode for exp/bounded-loop.")
        assert "/tmp/pending-loop.log" not in request.message
        yield _sse(AgentEvent(event="answer", text="Inspected the pending result."))
        yield _sse(AgentEvent(event="done"))

    app.state.background_tasks.stream = stream
    response = client.post(
        f"/api/projects/{project_id}/experiments/exp%2Fbounded-loop/run",
        json={"chat_id": new_loop_chat_id},
    )

    assert response.status_code == 202, response.text
    request = response.json()["request"]
    assert request["trigger"] == "experiment_run"
    assert request["control_invocation"] == 1
    assert request["control_invocation_ceiling"] == 1
    assert request["control_episode_id"] != old_episode
    assert request["watcher_ids"] == ["pending-loop-watcher"]
    assert request["chat_scope"] == "node"
    assert request["node_id"] == "exp/bounded-loop"
    assert request["chat_id"] == new_loop_chat_id
    assert request["provider"] == "codex"
    assert request["model"] == ""
    assert request["reasoning"] == "medium"
    assert request["run_on"] == "laptop"
    assert request["run_truth_scope"] == ["repo-a"]
    assert response.json()["kind"] == "node_chat"
    assert store.watcher("pending-loop-watcher").continuation.control_episode_id == old_episode
    assert store.watcher("pending-loop-watcher").notified is True
    operation_id = response.json()["operation_id"]
    _wait_for_run(client, project_id, operation_id)
    assert "watcher_notification" in {
        receipt.category for receipt in store.agent_task_receipts(operation_id)
    }
    assert any(
        "reauthorized by human Run" in event.message
        for event in store.agent_task_events(operation_id)
    )


def test_experiment_removal_and_run_admission_are_atomic_when_removal_wins(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    append_fixture_patch(service, _experiment_fixture_patch())
    project_id = app.state.default_project_id
    winner_barrier = threading.Barrier(2)
    release_winner = threading.Event()
    original_sync = service.sync_graph_transition

    def held_sync(*args, **kwargs):
        winner_barrier.wait(timeout=3)
        assert release_winner.wait(timeout=3)
        return original_sync(*args, **kwargs)

    monkeypatch.setattr(service, "sync_graph_transition", held_sync)

    async def drive_race():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            removal = asyncio.create_task(
                client.post(
                    f"/api/projects/{project_id}/sync",
                    json={"base_revision": 3, "removed_node_ids": ["exp/bounded-loop"]},
                )
            )
            try:
                await asyncio.to_thread(winner_barrier.wait, 3)
                admission = asyncio.create_task(
                    client.post(
                        f"/api/projects/{project_id}/experiments/exp%2Fbounded-loop/run",
                        json={"chat_id": str(uuid.uuid4())},
                    )
                )
                await asyncio.sleep(0)
            finally:
                release_winner.set()
            return await removal, await admission

    removal, admission = asyncio.run(drive_race())

    assert removal.status_code == 200
    assert admission.status_code == 404
    assert "exp/bounded-loop" not in service.history.state().nodes
    assert app.state.background_tasks.store.agent_tasks(project_id) == []


def test_experiment_removal_and_run_admission_are_atomic_when_admission_wins(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    append_fixture_patch(service, _experiment_fixture_patch())
    project_id = app.state.default_project_id
    winner_barrier = threading.Barrier(2)
    release_winner = threading.Event()
    release_stream = threading.Event()
    original_start = app.state.background_tasks.start

    async def held_stream(*_args):
        while not release_stream.is_set():
            await asyncio.sleep(0.01)
        yield _sse(AgentEvent(event="answer", text="Admission won the race."))
        yield _sse(AgentEvent(event="done"))

    def held_start(*args, **kwargs):
        record = original_start(*args, **kwargs)
        winner_barrier.wait(timeout=3)
        assert release_winner.wait(timeout=3)
        return record

    app.state.background_tasks.stream = held_stream
    monkeypatch.setattr(app.state.background_tasks, "start", held_start)

    async def drive_race():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            admission = asyncio.create_task(
                client.post(
                    f"/api/projects/{project_id}/experiments/exp%2Fbounded-loop/run",
                    json={"chat_id": str(uuid.uuid4())},
                )
            )
            try:
                await asyncio.to_thread(winner_barrier.wait, 3)
                removal = asyncio.create_task(
                    client.post(
                        f"/api/projects/{project_id}/sync",
                        json={"base_revision": 3, "removed_node_ids": ["exp/bounded-loop"]},
                    )
                )
                await asyncio.sleep(0)
            finally:
                release_winner.set()
            return await admission, await removal

    try:
        admission, removal = asyncio.run(drive_race())
        assert admission.status_code == 202
        assert removal.status_code == 409
        assert "bounded experiment loop is active" in removal.json()["detail"]
        assert "exp/bounded-loop" in service.history.state().nodes
    finally:
        release_stream.set()


def test_experiment_operation_lock_can_cross_fastapi_worker_threads(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    lock = app.state.services.experiment_operation_lock(project_id)
    failures: list[BaseException] = []

    def enter() -> None:
        try:
            lock.__enter__()
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    def exit() -> None:
        try:
            lock.__exit__(None, None, None)
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    entering = threading.Thread(target=enter)
    entering.start()
    entering.join(timeout=3)
    assert not entering.is_alive()

    exiting = threading.Thread(target=exit)
    exiting.start()
    exiting.join(timeout=3)
    assert not exiting.is_alive()
    assert failures == []


def test_removed_experiment_fails_closed_for_every_continuation_admission(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    append_fixture_patch(service, _experiment_fixture_patch())
    project_id = app.state.default_project_id
    client = TestClient(app)
    removed = client.post(
        f"/api/projects/{project_id}/sync",
        json={"base_revision": 3, "removed_node_ids": ["exp/bounded-loop"]},
    )
    assert removed.status_code == 200

    request = RunRequest(
        provider="codex",
        run_on="laptop",
        chat_id=str(uuid.uuid4()),
        chat_scope="node",
        node_id="exp/bounded-loop",
        mode="work",
        trigger="experiment_run",
        patch_kind="experiment_loop",
        control_node_id="exp/bounded-loop",
        control_revision=3,
        control_episode_id=str(uuid.uuid4()),
        control_invocation=1,
        control_invocation_ceiling=2,
    )
    store = app.state.background_tasks.store
    now = store.now()
    operation_id = "removed-experiment-task"
    episode_id = request.control_episode_id
    assert episode_id is not None
    store.create_experiment_episode_with_invocation(
        AgentTaskRecord(
            operation_id=operation_id,
            project_id=project_id,
            episode_id=episode_id,
            kind="node_chat",
            status="queued",
            request=request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="Paused before the Experiment was removed.",
            authorized_by=_named_test_authorizer(store),
        )
    )
    store.pause_agent_task(operation_id)
    operation_ids = {
        "resume": operation_id,
        "retry": operation_id,
        "repair-graph-update": operation_id,
    }

    def unexpected_admission(*_args, **_kwargs):
        raise AssertionError("removed Experiment continuation reached task admission")

    monkeypatch.setattr(app.state.background_tasks, "resume", unexpected_admission)
    monkeypatch.setattr(app.state.background_tasks, "retry", unexpected_admission)
    monkeypatch.setattr(app.state.background_tasks, "repair_graph_update", unexpected_admission)
    monkeypatch.setattr("rcp.api.app.start_watcher_notification", unexpected_admission)

    for endpoint, operation_id in operation_ids.items():
        response = client.post(f"/api/projects/{project_id}/tasks/{operation_id}/{endpoint}")
        assert response.status_code == 409
        assert response.json()["detail"] == (
            "Experiment exp/bounded-loop no longer exists; it cannot be continued."
        )

    continuation = WatcherContinuation(
        provider="codex",
        run_on="laptop",
        patch_kind="experiment_loop",
        control_node_id="exp/bounded-loop",
        control_revision=3,
        control_episode_id=str(uuid.uuid4()),
        control_invocation=1,
        control_invocation_ceiling=2,
    )
    watcher = WatcherRecord(
        watcher_id="removed-experiment-watcher",
        project_id=project_id,
        origin_operation_id="removed-experiment-origin",
        origin_task_kind="node_chat",
        chat_id=request.chat_id,
        node_id="exp/bounded-loop",
        check_command="true",
        log_path="/tmp/removed-experiment.log",
        cwd="/tmp",
        continuation=continuation,
        status="completed",
        created_at=now,
        completed_at=now,
    )
    store.create_agent_task(
        AgentTaskRecord(
            operation_id="removed-experiment-origin",
            project_id=project_id,
            kind="node_chat",
            status="succeeded",
            request={},
            created_at=now,
            updated_at=now,
            status_message="Removed Experiment watcher origin fixture.",
            authorized_by=_named_test_authorizer(store),
        )
    )
    store.create_watchers([watcher])
    assert app.state.watcher_poller.on_completed is not None
    app.state.watcher_poller.on_completed([store.watcher(watcher.watcher_id)])
    retired = store.watcher(watcher.watcher_id)
    assert retired is not None
    assert retired.status == "stopped"
    assert retired.notified is True


@pytest.mark.asyncio
async def test_experiment_work_stamps_and_applies_the_bound_control_patch(
    manifest, tmp_path
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    append_fixture_patch(service, _experiment_fixture_patch())
    attempt = {
        "id": "attempt-1",
        "sequence": 1,
        "purpose": "Launch the detached fixture.",
        "attempt_kind": "external_run",
        "decision_bundle": [],
        "status": "running",
        "job_refs": ["4471"],
    }
    patch = Patch(
        kind="experiment_loop",
        author="agent",
        summary="Recorded the launched attempt.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "update_nodes",
                "nodes": [
                    {
                        "id": "exp/bounded-loop",
                        "changes": {"status": "running", "attempts": [attempt]},
                    }
                ],
            }
        ],
    )
    request = RunRequest(
        node_id="exp/bounded-loop",
        message="Run the bounded loop.",
        chat_id=str(uuid.uuid4()),
        run_truth_scope=["repo-a"],
        mode="work",
        trigger="experiment_run",
        patch_kind="experiment_loop",
        control_node_id="exp/bounded-loop",
        control_revision=3,
        control_episode_id=str(uuid.uuid4()),
        control_invocation=1,
        control_invocation_ceiling=2,
        control_completion_criteria=["The detached fixture exits cleanly."],
    )
    execution = _chat_task_execution(
        app.state.background_tasks.store,
        operation_id="experiment-work",
        project_id=app.state.default_project_id,
        request=request,
    )
    watcher_now = execution.store.now()
    execution.store.create_watchers(
        [
            WatcherRecord(
                watcher_id="stopped-current-episode",
                project_id=app.state.default_project_id,
                origin_operation_id=execution.operation_id,
                origin_task_kind="node_chat",
                chat_id=request.chat_id,
                node_id=request.control_node_id,
                check_command="false",
                log_path="/tmp/stopped-current.log",
                cwd="/tmp",
                continuation=WatcherContinuation(
                    provider="codex",
                    run_on="laptop",
                    patch_kind="experiment_loop",
                    control_node_id=request.control_node_id,
                    control_revision=request.control_revision,
                    control_episode_id=request.control_episode_id,
                    control_invocation=request.control_invocation,
                    control_invocation_ceiling=request.control_invocation_ceiling,
                ),
                status="stopped",
                created_at=watcher_now,
            )
        ]
    )
    execution.store.create_watchers(
        [
            # A retained main-graph watcher may predate durable origin tasks; do not
            # forge the current episode root as provenance for this prior episode.
            WatcherRecord(
                watcher_id="stopped-prior-episode",
                project_id=app.state.default_project_id,
                origin_operation_id="legacy-prior-episode-operation",
                origin_task_kind="node_chat",
                chat_id=request.chat_id,
                node_id=request.control_node_id,
                check_command="false",
                log_path="/tmp/stopped-prior.log",
                cwd="/tmp",
                continuation=WatcherContinuation(
                    provider="codex",
                    run_on="laptop",
                    patch_kind="experiment_loop",
                    control_node_id=request.control_node_id,
                    control_revision=request.control_revision,
                    control_episode_id=str(uuid.uuid4()),
                    control_invocation=request.control_invocation,
                    control_invocation_ceiling=request.control_invocation_ceiling,
                ),
                status="stopped",
                created_at=watcher_now,
            )
        ]
    )
    launcher = ScriptedLauncher(
        [
            {
                "patch.json": agent_patch_json(patch),
                "watch.json": json.dumps(
                    {
                        "external": [
                            {
                                "check_command": "false",
                                "log_path": "/tmp/bounded-loop.log",
                                "cwd": "/tmp",
                            }
                        ],
                        "graph": [],
                    }
                ),
            }
        ],
        message="The fixture was launched once.",
    )

    frames = [
        frame
        async for frame in stream_experiment_loop_task(
            service,
            launcher,
            request,
            tmp_path / "data",
            execution=execution,
        )
    ]

    assert not _error_texts(frames)
    assert _graph_update(frames)["status"] == "applied"
    experiment = service.history.state().nodes["exp/bounded-loop"]
    assert [item.id for item in experiment.attempts] == ["attempt-1"]
    persisted = service.history.load_patches()[-1]
    assert persisted.kind == "experiment_loop"
    assert persisted.experiment_control_node_id == "exp/bounded-loop"
    assert persisted.experiment_decision_bundle == []
    contract = _local_task_contract(launcher.prompts[0])
    assert contract.startswith("# RCP Experiment-loop task contract")
    assert "Watcher handoff protocol" in contract
    control_name = next(
        name for name in launcher.input_snapshots[0] if "experiment-control-initial_run" in name
    )
    watcher_name = next(
        name for name in launcher.input_snapshots[0] if "experiment-watchers" in name
    )
    control = json.loads(launcher.input_snapshots[0][control_name])
    assert set(control) == {
        "phase",
        "episode_id",
        "invocation",
        "invocation_ceiling",
        "remaining_invocations",
        "decision_bundle",
        "decision_drift",
        "completion_criteria",
        "delivered_watcher_ids",
        "delivered_watcher_groups",
        "watcher_state_path",
    }
    assert control["phase"] == "initial_run"
    assert control["invocation"] == 1
    assert control["remaining_invocations"] == 1
    assert control["delivered_watcher_groups"] == []
    watcher_state = json.loads(launcher.input_snapshots[0][watcher_name])
    assert [item["watcher_id"] for item in watcher_state] == ["stopped-current-episode"]
    assert "remaining_invocations" not in launcher.prompts[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "initial_watch, expected_calls",
    [('{"external":[],"graph":[]}', 1), (None, 2)],
)
async def test_experiment_loop_accepts_empty_watch_only_with_explicit_exit(
    manifest, tmp_path, initial_watch, expected_calls
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    append_fixture_patch(service, _experiment_fixture_patch())
    patch = Patch(
        kind="experiment_loop",
        author="agent",
        summary="Finished the bounded experiment.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "update_nodes",
                "nodes": [{"id": "exp/bounded-loop", "changes": {"status": "completed"}}],
            }
        ],
    )
    request = RunRequest(
        node_id="exp/bounded-loop",
        message="Finish the bounded loop.",
        chat_id=str(uuid.uuid4()),
        run_truth_scope=["repo-a"],
        mode="work",
        trigger="experiment_run",
        patch_kind="experiment_loop",
        control_node_id="exp/bounded-loop",
        control_revision=3,
        control_episode_id=str(uuid.uuid4()),
        control_invocation=1,
        control_invocation_ceiling=2,
    )
    execution = _chat_task_execution(
        app.state.background_tasks.store,
        operation_id="experiment-empty-watch",
        project_id=app.state.default_project_id,
        request=request,
    )
    first = {"patch.json": agent_patch_json(patch)}
    if initial_watch is not None:
        first["watch.json"] = initial_watch
    launcher = ScriptedLauncher(
        [first, {"watch.json": '{"external":[],"graph":[]}'}],
        message="Finished.",
    )

    frames = [
        frame
        async for frame in stream_experiment_loop_task(
            service,
            launcher,
            request,
            tmp_path / "data",
            execution=execution,
        )
    ]

    assert not _error_texts(frames)
    assert _events(frames)[-1].event == "done"
    assert launcher.calls == expected_calls
    assert service.history.state().nodes["exp/bounded-loop"].status == "completed"
    assert any(
        receipt.category == "experiment_loop_exit"
        for receipt in execution.store.agent_task_receipts(execution.operation_id)
    )


@pytest.mark.asyncio
async def test_experiment_loop_missing_handoff_fails_without_done_after_one_correction(
    manifest, tmp_path
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    append_fixture_patch(service, _experiment_fixture_patch())
    request = RunRequest(
        node_id="exp/bounded-loop",
        message="Run the bounded loop.",
        chat_id=str(uuid.uuid4()),
        run_truth_scope=["repo-a"],
        mode="work",
        trigger="experiment_run",
        patch_kind="experiment_loop",
        control_node_id="exp/bounded-loop",
        control_revision=3,
        control_episode_id=str(uuid.uuid4()),
        control_invocation=1,
        control_invocation_ceiling=2,
    )
    execution = _chat_task_execution(
        app.state.background_tasks.store,
        operation_id="experiment-missing-watch",
        project_id=app.state.default_project_id,
        request=request,
    )
    launcher = ScriptedLauncher([{}], message="Could not establish handoff.")

    frames = [
        frame
        async for frame in stream_experiment_loop_task(
            service,
            launcher,
            request,
            tmp_path / "data",
            execution=execution,
        )
    ]

    assert launcher.calls == 2
    assert any("watcher handoff failed" in text for text in _error_texts(frames))
    assert all(event.event != "done" for event in _events(frames))
    assert service.history.state().revision == 3


@pytest.mark.asyncio
async def test_experiment_loop_patch_correction_rechecks_empty_watch_exit(
    manifest, tmp_path
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    append_fixture_patch(service, _experiment_fixture_patch())
    invalid_exit = Patch(
        kind="experiment_loop",
        author="agent",
        summary="Finished with an invalid extra update.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "update_nodes",
                "nodes": [
                    {
                        "id": "exp/bounded-loop",
                        "changes": {"status": "completed", "title": "Forbidden rewrite"},
                    }
                ],
            }
        ],
    )
    removed_exit = json.dumps({"summary": "Removed the exit.", "ops": [], "repositories_read": []})
    request = RunRequest(
        node_id="exp/bounded-loop",
        message="Finish the bounded loop.",
        chat_id=str(uuid.uuid4()),
        run_truth_scope=["repo-a"],
        mode="work",
        trigger="experiment_run",
        patch_kind="experiment_loop",
        control_node_id="exp/bounded-loop",
        control_revision=3,
        control_episode_id=str(uuid.uuid4()),
        control_invocation=1,
        control_invocation_ceiling=2,
    )
    execution = _chat_task_execution(
        app.state.background_tasks.store,
        operation_id="experiment-exit-recheck",
        project_id=app.state.default_project_id,
        request=request,
    )
    launcher = ScriptedLauncher(
        [
            {
                "patch.json": agent_patch_json(invalid_exit),
                "watch.json": '{"external":[],"graph":[]}',
            },
            {"patch.json": removed_exit},
        ],
        message="Finished.",
    )

    frames = [
        frame
        async for frame in stream_experiment_loop_task(
            service,
            launcher,
            request,
            tmp_path / "data",
            execution=execution,
        )
    ]

    assert any("Patch could not be validated" in text for text in _error_texts(frames))
    assert all(event.event != "done" for event in _events(frames))
    assert service.history.state().revision == 3
    assert not any(
        receipt.category == "experiment_loop_exit"
        for receipt in execution.store.agent_task_receipts(execution.operation_id)
    )


@pytest.mark.asyncio
async def test_unreadable_loop_patch_correction_stays_a_correction(manifest, tmp_path) -> None:
    """An undecodable loop Patch correction is a diagnostic, not an escaped exception.

    Letting the read raise out of the run aborts before the episode commits its
    session binding, so a transient read failure would cost the loop its native
    session instead of one correction round.
    """

    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    append_fixture_patch(service, _experiment_fixture_patch())
    invalid_exit = Patch(
        kind="experiment_loop",
        author="agent",
        summary="Finished with an invalid extra update.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "update_nodes",
                "nodes": [
                    {
                        "id": "exp/bounded-loop",
                        "changes": {"status": "completed", "title": "Forbidden rewrite"},
                    }
                ],
            }
        ],
    )
    request = RunRequest(
        node_id="exp/bounded-loop",
        message="Finish the bounded loop.",
        chat_id=str(uuid.uuid4()),
        run_truth_scope=["repo-a"],
        mode="work",
        trigger="experiment_run",
        patch_kind="experiment_loop",
        control_node_id="exp/bounded-loop",
        control_revision=3,
        control_episode_id=str(uuid.uuid4()),
        control_invocation=1,
        control_invocation_ceiling=2,
    )
    execution = _chat_task_execution(
        app.state.background_tasks.store,
        operation_id="experiment-unreadable-correction",
        project_id=app.state.default_project_id,
        request=request,
    )

    class UnreadableLoopCorrectionLauncher(ScriptedLauncher):
        async def stream(self, provider, prompt, **kwargs):
            launch = self.calls
            async for event in super().stream(provider, prompt, **kwargs):
                yield event
            if launch > 0:
                (self.workspaces[-1] / "patch.json").write_bytes(b'{"summary": "\xff\xfe"}')

    launcher = UnreadableLoopCorrectionLauncher(
        [
            {
                "patch.json": agent_patch_json(invalid_exit),
                "watch.json": '{"external":[],"graph":[]}',
            }
        ],
        message="Finished.",
    )

    frames = [
        frame
        async for frame in stream_experiment_loop_task(
            service,
            launcher,
            request,
            tmp_path / "data",
            execution=execution,
        )
    ]

    assert any("could not be read" in text for text in _error_texts(frames))
    assert service.history.state().revision == 3


@pytest.mark.asyncio
async def test_experiment_loop_keeps_one_watcher_when_graph_reflection_is_rejected(
    manifest, tmp_path
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    append_fixture_patch(service, _experiment_fixture_patch())
    invalid = Patch(
        kind="experiment_loop",
        author="agent",
        summary="Reflected running work with a forbidden rewrite.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "update_nodes",
                "nodes": [
                    {
                        "id": "exp/bounded-loop",
                        "changes": {"status": "running", "title": "Forbidden rewrite"},
                    }
                ],
            }
        ],
    )
    request = RunRequest(
        node_id="exp/bounded-loop",
        message="Run the bounded loop.",
        chat_id=str(uuid.uuid4()),
        run_truth_scope=["repo-a"],
        mode="work",
        trigger="experiment_run",
        patch_kind="experiment_loop",
        control_node_id="exp/bounded-loop",
        control_revision=3,
        control_episode_id=str(uuid.uuid4()),
        control_invocation=1,
        control_invocation_ceiling=2,
    )
    execution = _chat_task_execution(
        app.state.background_tasks.store,
        operation_id="experiment-rejected-reflection",
        project_id=app.state.default_project_id,
        request=request,
    )
    handoff = {
        "patch.json": agent_patch_json(invalid),
        "watch.json": json.dumps(
            {
                "external": [
                    {"check_command": "false", "log_path": "/tmp/loop.log", "cwd": "/tmp"}
                ],
                "graph": [],
            }
        ),
    }
    launcher = ScriptedLauncher([handoff], message="Detached work is still running.")

    frames = [
        frame
        async for frame in stream_experiment_loop_task(
            service,
            launcher,
            request,
            tmp_path / "data",
            execution=execution,
        )
    ]

    assert not _error_texts(frames)
    assert _events(frames)[-1].event == "done"
    assert _graph_update(frames)["status"] == "rejected"
    assert len(execution.store.watchers(app.state.default_project_id)) == 1
    assert service.history.state().revision == 3


@pytest.mark.asyncio
async def test_experiment_loop_retry_reuses_canonical_patch_and_watcher_handoff(
    manifest, tmp_path
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    append_fixture_patch(service, _experiment_fixture_patch())
    operation_id = "experiment-committed-before-receipt"
    exit_patch = Patch(
        kind="experiment_loop",
        author="agent",
        summary="Finished before the task receipt was written.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        source_operation_id=operation_id,
        experiment_control_node_id="exp/bounded-loop",
        ops=[
            {
                "op": "update_nodes",
                "nodes": [{"id": "exp/bounded-loop", "changes": {"status": "completed"}}],
            }
        ],
    )
    append_fixture_patch(service, exit_patch)
    append_fixture_patch(service, refresh_patch("rq/after-committed-loop"))
    request = RunRequest(
        node_id="exp/bounded-loop",
        message="Recover the committed invocation.",
        chat_id=str(uuid.uuid4()),
        run_truth_scope=["repo-a"],
        mode="work",
        trigger="experiment_run",
        patch_kind="experiment_loop",
        control_node_id="exp/bounded-loop",
        control_revision=3,
        control_episode_id=str(uuid.uuid4()),
        control_invocation=1,
        control_invocation_ceiling=2,
    )
    execution = _chat_task_execution(
        app.state.background_tasks.store,
        operation_id=operation_id,
        project_id=app.state.default_project_id,
        request=request,
    )
    launcher = ScriptedLauncher(
        [
            {
                "patch.json": agent_patch_json(exit_patch),
                "watch.json": json.dumps(
                    {
                        "external": [
                            {
                                "check_command": "false",
                                "log_path": "/tmp/recovered-loop.log",
                                "cwd": "/tmp",
                            }
                        ],
                        "graph": [],
                    }
                ),
            }
        ],
        message="Recovered the committed result.",
    )

    frames = [
        frame
        async for frame in stream_experiment_loop_task(
            service,
            launcher,
            request,
            tmp_path / "data",
            execution=execution,
        )
    ]

    assert not _error_texts(frames)
    assert _graph_update(frames)["applied_revision"] == 4
    assert service.history.state().revision == 5
    assert len(service.history.load_patches()) == 5
    watchers = execution.store.watchers(app.state.default_project_id)
    assert len(watchers) == 1
    stored = watchers[0]
    repeated = persist_experiment_watchers_idempotently(
        execution,
        [
            WatchSpec(
                check_command=stored.check_command,
                log_path=stored.log_path,
                cwd=stored.cwd,
            )
        ],
        [
            WatcherCheckResult(
                state="active",
                checked_at=stored.last_checked_at or stored.created_at,
                exit_code=1,
            )
        ],
        WatcherBinding(
            project_id=stored.project_id,
            origin_operation_id=stored.origin_operation_id,
            origin_task_kind=stored.origin_task_kind,
            chat_id=stored.chat_id,
            node_id=stored.node_id,
            execution_host=stored.execution_host,
            continuation=stored.continuation,
        ),
    )
    assert repeated[0].watcher_id == stored.watcher_id
    assert len(execution.store.watchers(app.state.default_project_id)) == 1
    assert any(
        receipt.category == "experiment_loop_exit"
        for receipt in execution.store.agent_task_receipts(operation_id)
    )


@pytest.mark.asyncio
async def test_readable_watch_value_error_gets_same_session_correction(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    request = RunRequest(
        chat_scope="project",
        message="Launch the detached fixture.",
        chat_id=str(uuid.uuid4()),
        run_truth_scope=["repo-a"],
        mode="work",
    )
    execution = _chat_task_execution(
        app.state.background_tasks.store,
        operation_id="watch-readable-value-error",
        project_id=app.state.default_project_id,
        request=request,
    )
    watch = json.dumps(
        {
            "external": [
                {
                    "check_command": "false",
                    "log_path": str(tmp_path / "fixture.log"),
                    "cwd": str(tmp_path),
                }
            ],
            "graph": [],
        }
    )
    launcher = ScriptedLauncher(
        [{"watch.json": watch}, {"watch.json": json.dumps(json.loads(watch), indent=2)}],
        message="The fixture was launched once.",
    )
    original_read = work_module._read_watch_request
    read_calls = 0

    def flaky_read(*args, **kwargs):
        nonlocal read_calls
        read_calls += 1
        if read_calls == 1:
            raise ValueError("watch.json is not a direct regular file")
        return original_read(*args, **kwargs)

    monkeypatch.setattr(work_module, "_read_watch_request", flaky_read)

    frames = [
        frame
        async for frame in stream_work_run(
            service,
            launcher,
            request,
            tmp_path / "data",
            execution=execution,
        )
    ]

    assert not _error_texts(frames)
    assert launcher.calls == 2
    assert launcher.resumed_sessions == [None, launcher.native_session_id]
    assert len(execution.store.watchers(app.state.default_project_id)) == 1


@pytest.mark.asyncio
async def test_watch_handoff_correction_arms_once_and_wake_is_not_a_user_turn(
    manifest, tmp_path
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    store = app.state.background_tasks.store
    chat_id = str(uuid.uuid4())
    request = RunRequest(
        chat_scope="project",
        message="Launch the detached fixture.",
        chat_id=chat_id,
        run_truth_scope=["repo-a"],
        mode="work",
    )
    execution = _chat_task_execution(
        store,
        operation_id="watch-origin",
        project_id=app.state.default_project_id,
        request=request,
    )
    common = {
        "log_path": str(tmp_path / "fixture.log"),
        "cwd": str(tmp_path),
    }
    graph = [
        {
            "node_id": "hyp/replanning-restores-plasticity",
            "proposal_resolved": True,
        }
    ]
    invalid = json.dumps({"external": [{**common, "check_command": "exit 2"}], "graph": graph})
    corrected = json.dumps({"external": [{**common, "check_command": "exit 1"}], "graph": graph})
    launcher = ScriptedLauncher(
        [{"watch.json": invalid}, {"watch.json": corrected}],
        message="The fixture was launched once.",
    )

    frames = [
        frame
        async for frame in stream_work_run(
            service,
            launcher,
            request,
            tmp_path / "data",
            execution=execution,
        )
    ]

    assert not _error_texts(frames)
    assert [event.text for event in _events(frames) if event.event == "answer"] == [
        "The fixture was launched once."
    ]
    assert [item["capability"] for item in launcher.launch_kwargs] == [
        "work_auto",
        "work_auto",
    ]
    assert launcher.launch_kwargs[1]["read_dirs"] == launcher.launch_kwargs[0]["read_dirs"]
    assert launcher.launch_kwargs[1]["write_dirs"] == launcher.launch_kwargs[0]["write_dirs"]
    armed = store.watchers(app.state.default_project_id)
    assert len(armed) == 2
    assert {type(record) for record in armed} == {WatcherRecord, GraphWatcherRecord}
    assert all(record.status == "active" for record in armed)
    assert all(record.origin_operation_id == "watch-origin" for record in armed)
    assert all(record.continuation.patch_kind == "work" for record in armed)
    assert any(
        receipt.category == "watchers_armed"
        for receipt in store.agent_task_receipts("watch-origin")
    )

    store.complete_agent_task("watch-origin", applied_revision=None, result={"messages": []})
    external = next(record for record in armed if isinstance(record, WatcherRecord))
    wake = request.model_copy(
        update={
            "message": "RCP watcher update for the completed fixture.",
            "trigger": "watcher",
            "watcher_ids": [external.watcher_id],
        }
    )
    wake_execution = _chat_task_execution(
        store,
        operation_id="watch-wake",
        project_id=app.state.default_project_id,
        request=wake,
    )
    wake_launcher = ScriptedLauncher([{}], message="The watched fixture completed cleanly.")
    wake_frames = [
        frame
        async for frame in stream_work_run(
            service,
            wake_launcher,
            wake,
            tmp_path / "data",
            execution=wake_execution,
        )
    ]

    assert not _error_texts(wake_frames)
    transcript = service.chat_transcript(chat_id)
    assert transcript is not None
    assert [message.role for message in transcript.messages] == ["user", "assistant", "assistant"]
    assert [message.trigger for message in transcript.messages] == ["human", "human", "watcher"]
    assert all(
        message.text != "RCP watcher update for the completed fixture."
        for message in transcript.messages
    )
    persisted = store.watchers(app.state.default_project_id)
    assert len(persisted) == 2
    assert any(isinstance(record, GraphWatcherRecord) for record in persisted)


def _stuck_experiment_patch() -> Patch:
    """An experiment whose only attempt is still open, as a dead watcher leaves it."""

    return Patch(
        kind="refresh",
        author="agent",
        summary="Record an experiment with an open attempt.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": "exp/stuck",
                        "type": "experiment",
                        "title": "Stuck run",
                        "objective": "Train the ablation.",
                        "status": "running",
                        "attempts": [
                            {
                                "id": "attempt-1",
                                "sequence": 1,
                                "purpose": "Train the ablation.",
                                "attempt_kind": "external_run",
                                "status": "running",
                                "job_refs": ["4471"],
                            }
                        ],
                    }
                ],
            }
        ],
    )


def test_a_human_may_release_an_attempt_without_it_gating_the_loop(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    append_fixture_patch(service, _stuck_experiment_patch())
    client = TestClient(app)

    control = client.get(f"/api/projects/{project_id}").json()["experiment_control"]["exp/stuck"]
    assert control["active"] is False
    assert control["ready"] is True
    assert control["reasons"] == []

    released = client.post(
        f"/api/projects/{project_id}/sync",
        json={
            "base_revision": 3,
            "nodes": [
                {
                    "node_id": "exp/stuck",
                    "base_updated_rev": 3,
                    "cancel_attempt_ids": ["attempt-1"],
                }
            ],
        },
    )

    assert released.status_code == 200, released.json()
    attempts = released.json()["nodes"]["exp/stuck"]["attempts"]
    assert [item["status"] for item in attempts] == ["cancelled"]
    assert attempts[0]["finished_at"] is not None
    after = client.get(f"/api/projects/{project_id}").json()["experiment_control"]["exp/stuck"]
    assert after["active"] is False
    assert after["ready"] is True
    # A finished attempt is a record, not something a later release may rewrite.
    again = client.post(
        f"/api/projects/{project_id}/sync",
        json={
            "base_revision": 4,
            "nodes": [
                {
                    "node_id": "exp/stuck",
                    "base_updated_rev": 4,
                    "cancel_attempt_ids": ["attempt-1"],
                }
            ],
        },
    )
    assert again.status_code == 422
    assert "no open attempt" in again.json()["detail"]


def test_seed_stages_its_selected_skills_and_records_what_it_ran(
    manifest, tmp_path, monkeypatch
) -> None:
    """S64: the selection is structured task metadata, not text the agent parses."""

    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    service = app.state.catalog.open(project_id)
    _persist_skill_defaults(
        service,
        SkillDefaults(workflow_ids=["research-graph-audit"], skill_ids=[]),
    )
    observed: dict[str, object] = {}

    class InspectingLauncher(ScriptedLauncher):
        async def stream(self, provider, prompt, **kwargs):
            # A succeeded run reclaims its scratch folder, so read it in place.
            bundle = next((Path(kwargs["cwd"]) / "inputs").glob("rcp-skills-*"))
            observed["contract"] = _local_task_contract(prompt)
            observed["staged"] = sorted(
                str(item.relative_to(bundle)) for item in bundle.rglob("*.md")
            )
            async for event in super().stream(provider, prompt, **kwargs):
                yield event

    launcher = InspectingLauncher([{"patch.json": agent_patch_json(seed_patch())}])
    monkeypatch.setattr(app.state.catalog.launcher, "stream", launcher.stream)
    client = TestClient(app)

    started = client.post(
        f"/api/projects/{project_id}/tasks/seed",
        json={
            "run_truth_scope": ["repo-a"],
        },
    )
    operation_id = started.json()["operation_id"]
    completed = _wait_for_run(client, project_id, operation_id)

    assert completed["status"] == "succeeded"
    # The workflow's dependency closure is resolved and recorded with exact versions.
    assert [
        (item["kind"], item["id"], item["version"])
        for item in completed["request"]["resolved_skill_packages"]
    ] == [
        (item.kind, item.id, item.version)
        for item in official_registry()
        .resolve(workflow_ids=["research-graph-audit"])
        .resolved_skill_packages
    ]
    # The workflow file and every declared dependency are staged together, alone.
    assert observed["staged"] == [
        "skill/evidence-triage/SKILL.md",
        "skill/evidence-triage/references/worked-examples.md",
        "skill/experiment-causality/SKILL.md",
        "skill/graph-audit/SKILL.md",
        "workflow/research-graph-audit/WORKFLOW.md",
    ]
    # The contract points at the staged folders; the bodies stay on disk.
    contract = observed["contract"]
    workflow_package = official_registry().package("workflow", "research-graph-audit")
    assert f"Research graph audit (workflow research-graph-audit v{workflow_package.version})" in (
        contract
    )
    # The description is wrapped into the contract, so compare on normalized whitespace.
    assert workflow_package.description in " ".join(contract.split())
    assert f"rcp-skills-{operation_id}" in contract
    assert "Run the structural review" not in contract
    # Nothing about the selection reaches canonical state.
    research_dir = service.manifest.research_dir
    assert not list(research_dir.rglob("*SKILL.md"))
    assert not list(research_dir.rglob("*WORKFLOW.md"))


def test_an_upgraded_package_never_makes_a_stored_task_un_retryable(manifest, tmp_path) -> None:
    """S64: the registry is authoritative; a recorded version is a receipt, not a pin."""

    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.catalog.open(app.state.default_project_id)
    _persist_skill_defaults(service, SkillDefaults(skill_ids=["graph-audit"]))
    stored_request = {
        "provider": "codex",
        "run_on": "laptop",
        "run_truth_scope": ["repo-a"],
        # What the earlier attempt ran with, before the package was upgraded.
        "resolved_skill_packages": [{"id": "graph-audit", "kind": "skill", "version": "0.0.1"}],
    }

    selection = _validate_stored_task_request(service, "seed", stored_request)

    assert [(item.id, item.version) for item in selection.resolved_skill_packages] == [
        ("graph-audit", official_registry().package("skill", "graph-audit").version)
    ]


def test_retrying_a_failed_seed_records_the_selection_it_will_stage(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    service = app.state.catalog.open(project_id)
    _persist_skill_defaults(
        service,
        SkillDefaults(workflow_ids=["research-graph-audit"], skill_ids=[]),
    )

    async def failing_stream(*_args, **_kwargs):
        yield _event_frame(AgentEvent(event="error", text="Provider exited before writing."))

    ordinary_stream = app.state.background_tasks.stream
    app.state.background_tasks.stream = failing_stream
    client = TestClient(app)
    started = client.post(
        f"/api/projects/{project_id}/tasks/seed",
        json={"run_truth_scope": ["repo-a"]},
    )
    operation_id = started.json()["operation_id"]
    assert _wait_for_run(client, project_id, operation_id)["status"] == "failed"

    launcher = ScriptedLauncher([{"patch.json": agent_patch_json(seed_patch())}])
    monkeypatch.setattr(app.state.catalog.launcher, "stream", launcher.stream)
    app.state.background_tasks.stream = ordinary_stream
    retried = client.post(f"/api/projects/{project_id}/tasks/{operation_id}/retry", json={})

    assert retried.status_code == 202
    child = _wait_for_run(client, project_id, retried.json()["operation_id"])
    assert child["status"] == "succeeded"
    assert [
        (item["id"], item["version"]) for item in child["request"]["resolved_skill_packages"]
    ] == [
        (item.id, item.version)
        for item in official_registry()
        .resolve(workflow_ids=["research-graph-audit"])
        .resolved_skill_packages
    ]


def test_decisions_awaiting_choice_matches_the_shared_frontend_fixture() -> None:
    """Bind the backend predicate to the fixture `web/tests` asserts against.

    The web client filters the graph itself to render the Inbox and its badge,
    so the rule genuinely exists twice. This is the only test that fails when
    the two copies drift apart.
    """

    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "decisions_awaiting_choice.json").read_text()
    )
    nodes = [NODE_ADAPTER.validate_python(raw) for raw in fixture["nodes"]]

    awaiting = [node.id for node in nodes if decision_awaits_choice(node)]

    assert awaiting == fixture["expected_awaiting_choice"]
