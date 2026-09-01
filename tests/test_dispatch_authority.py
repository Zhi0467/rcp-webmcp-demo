from __future__ import annotations

import threading
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from rcp.agents import AgentEvent
from rcp.background import AgentTaskExecution, AgentTaskRequest, BackgroundAgentTasks
from rcp.core.authority import (
    AgentDispatchAuthority,
    AgentDispatchScope,
    AgentTaskAuthority,
    require_apply,
    require_dispatch,
)
from rcp.core.models import AuthorizedHuman
from rcp.core.transition_models import GraphHeadRef, GraphTargetRef
from rcp.history import HistoryManager
from rcp.runs.auto_research import AutoResearchRunRequest
from rcp.runs.watcher_admission import start_watcher_notification
from rcp.service import CoachRequest, RunRequest, resolve_dispatch_authority
from rcp.storage import (
    AgentTaskKind,
    AgentTaskRecord,
    AppStore,
    AutoResearchStateRecord,
    EpisodeRecord,
    ProjectRecord,
    WatcherContinuation,
    WatcherRecord,
)
from tests.helpers import seated_on_every_project, seed_patch


def _authorizer(store: AppStore) -> AuthorizedHuman:
    owner = store.local_owner
    assert owner is not None
    owner = store.rename_space_user(owner.user_id, "Dispatch researcher")
    assert owner.display_name is not None
    return AuthorizedHuman(
        space_id=store.space_id,
        user_id=owner.user_id,
        display_name=owner.display_name,
    )


def _work_request(**updates: object) -> RunRequest:
    return RunRequest(
        provider="codex",
        model="",
        reasoning="high",
        run_on="local",
        run_truth_scope=["repo-a"],
        chat_scope="project",
        chat_id=str(uuid.uuid4()),
        message="Investigate the question.",
        mode="work",
        **updates,
    )


def _work_authority(request: RunRequest) -> AgentDispatchAuthority:
    authority = resolve_dispatch_authority("project_chat", request)
    assert authority is not None
    return authority


def _record(
    store: AppStore,
    *,
    operation_id: str,
    project_id: str = "project-one",
    request: RunRequest | None = None,
    status: str = "failed",
    authorized_by: AuthorizedHuman | None = None,
    dispatch_authority: AgentDispatchAuthority | None = None,
    native_session_id: str | None = None,
    stage_root: str | None = None,
) -> AgentTaskRecord:
    request = request or _work_request()
    now = store.now()
    return store.create_agent_task(
        AgentTaskRecord(
            operation_id=operation_id,
            project_id=project_id,
            kind="project_chat",
            status=status,
            request=request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            started_at=now,
            finished_at=now if status != "paused" else None,
            status_message=status,
            error="provider failed" if status == "failed" else None,
            native_session_id=native_session_id,
            stage_root=stage_root,
            phase=status,
            last_activity_at=now,
            authorized_by=authorized_by,
            dispatch_authority=dispatch_authority,
        )
    )


def _wait_for_terminal(store: AppStore, operation_id: str) -> AgentTaskRecord:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        record = store.agent_task(operation_id)
        assert record is not None
        if record.status in {"succeeded", "failed", "paused", "interrupted"}:
            return record
        time.sleep(0.01)
    raise AssertionError("background task did not settle")


def test_ordinary_resolver_maps_every_current_task_and_ignores_forged_fields() -> None:
    seed = resolve_dispatch_authority(
        "seed",
        RunRequest.model_validate(
            {
                "run_truth_scope": ["repo-b", "repo-a", "repo-a"],
                "profile": "orchestrator",
                "task_contract": "orchestrate",
            }
        ),
    )
    refresh = resolve_dispatch_authority("refresh", RunRequest(run_truth_scope=["repo-a"]))
    discuss = resolve_dispatch_authority(
        "node_chat",
        RunRequest(
            run_truth_scope=["repo-a"],
            chat_scope="node",
            chat_id="chat-one",
            node_id="rq/one",
            mode="discuss",
        ),
    )
    work_request = _work_request()
    work = resolve_dispatch_authority("project_chat", work_request)
    episode_id = str(uuid.uuid4())
    experiment = resolve_dispatch_authority(
        "node_chat",
        RunRequest(
            run_truth_scope=["repo-a"],
            chat_scope="node",
            chat_id="experiment-chat",
            node_id="exp/one",
            mode="work",
            patch_kind="experiment_loop",
            control_node_id="exp/one",
            control_episode_id=episode_id,
        ),
    )
    paper = resolve_dispatch_authority("paper_coach", CoachRequest(message="Review this."))

    assert seed == AgentDispatchAuthority(
        profile="ordinary",
        task_contract="scratch_patch",
        scope=AgentDispatchScope(
            run_truth_scope=["repo-a", "repo-b"],
            patch_kind="seed",
        ),
    )
    assert refresh is not None and refresh.task_contract == "scratch_patch"
    assert refresh.scope.patch_kind == "refresh"
    assert discuss is not None and discuss.task_contract == "discuss"
    assert discuss.scope.model_dump() == {
        "run_truth_scope": ["repo-a"],
        "episode_id": None,
        "chat_scope": "node",
        "chat_id": "chat-one",
        "node_id": "rq/one",
        "patch_kind": None,
        "control_node_id": None,
        "control_episode_id": None,
    }
    assert work is not None and work.task_contract == "work_auto"
    assert work.scope.patch_kind == "work"
    assert experiment is not None and experiment.scope.control_episode_id == episode_id
    assert paper is not None and paper.task_contract == "paper_readonly"
    orchestrator = resolve_dispatch_authority(
        "auto_research",
        AutoResearchRunRequest(
            episode_id="episode-one",
            role="orchestrator",
            run_truth_scope=["repo-b", "repo-a", "repo-a"],
        ),
    )
    worker = resolve_dispatch_authority(
        "auto_research",
        AutoResearchRunRequest(
            episode_id="episode-one",
            role="worker",
            run_truth_scope=["repo-a"],
            control_node_id="exp/seat",
        ),
    )
    assert orchestrator == AgentDispatchAuthority(
        profile="orchestrator",
        task_contract="orchestrate",
        scope=AgentDispatchScope(
            run_truth_scope=["repo-a", "repo-b"],
            episode_id="episode-one",
            patch_kind="work",
        ),
    )
    assert worker == AgentDispatchAuthority(
        profile="ordinary",
        task_contract="work_auto",
        scope=AgentDispatchScope(
            run_truth_scope=["repo-a"],
            episode_id="episode-one",
            patch_kind="work",
        ),
    )
    with pytest.raises(ValidationError, match="role"):
        AutoResearchRunRequest.model_validate(
            {"episode_id": "episode-one", "role": "report", "ending": "completed"}
        )
    with pytest.raises(TypeError, match="AutoResearchRunRequest"):
        resolve_dispatch_authority("auto_research", object())


def test_dispatch_binding_is_strict_and_normalized() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AgentDispatchAuthority.model_validate(
            {
                "profile": "ordinary",
                "task_contract": "work_auto",
                "scope": {"run_truth_scope": [], "patch_kind": "work"},
                "permission": "forged",
            }
        )
    with pytest.raises(ValidationError, match="sorted and unique"):
        AgentDispatchScope(run_truth_scope=["repo-b", "repo-a"], patch_kind="work")
    with pytest.raises(ValidationError, match="apply_target"):
        AgentTaskAuthority.model_validate(
            {
                "operation_id": "missing-apply-target",
                "project_id": "project-one",
                "authorized_by": None,
                "dispatch_authority": None,
            }
        )


def test_agent_task_authority_carries_episode_id_from_each_exact_task_row(
    tmp_path: Path,
) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    authorizer = _authorizer(store)
    episode_id = "episode-one"
    project_id = "project-one"
    root_operation_id = "auto-research-root"
    now = store.now()
    store.upsert_project(
        ProjectRecord(
            project_id=project_id,
            locator="/tmp/project-one/research.yaml",
            name="Project one",
            state_location="/tmp/project-one/.research",
            state_remote=False,
            added_at=now,
        )
    )
    root_request = AutoResearchRunRequest(
        episode_id=episode_id,
        role="orchestrator",
        actor_operation_id=root_operation_id,
        run_truth_scope=["repo-a"],
    )
    root_authority = resolve_dispatch_authority("auto_research", root_request)
    assert root_authority is not None
    _, root = store.create_auto_research_episode_with_root_task(
        EpisodeRecord(
            episode_id=episode_id,
            project_id=project_id,
            mode="auto_research",
            graph_target=GraphTargetRef(kind="branch", branch_id=episode_id),
            graph_base_head=GraphHeadRef(revision=0),
            status="queued",
            invocation_ceiling=4,
            authorized_by=authorizer,
            created_at=now,
            updated_at=now,
        ),
        AutoResearchStateRecord(
            episode_id=episode_id,
            starting_instruction="Investigate the question.",
            created_at=now,
            updated_at=now,
        ),
        AgentTaskRecord(
            operation_id=root_operation_id,
            project_id=project_id,
            episode_id=episode_id,
            graph_target=GraphTargetRef(kind="branch", branch_id=episode_id),
            kind="auto_research",
            status="queued",
            request=root_request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="queued",
            authorized_by=authorizer,
            dispatch_authority=root_authority,
        ),
    )
    store.complete_agent_task(root.operation_id, applied_revision=None, result={})
    root = store.agent_task(root.operation_id)
    assert root is not None
    worker_operation_id = "auto-research-worker"
    worker_request = AutoResearchRunRequest(
        episode_id=episode_id,
        role="worker",
        actor_operation_id=worker_operation_id,
        control_node_id="exp/seat",
        run_truth_scope=["repo-a"],
    )
    worker_authority = resolve_dispatch_authority("auto_research", worker_request)
    assert worker_authority is not None
    store.create_auto_research_agent_task(
        AgentTaskRecord(
            operation_id=worker_operation_id,
            project_id=project_id,
            episode_id=episode_id,
            graph_target=GraphTargetRef(kind="branch", branch_id=episode_id),
            kind="auto_research",
            status="queued",
            request=worker_request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="done",
            parent_operation_id=root.operation_id,
            authorized_by=authorizer,
            dispatch_authority=worker_authority,
        ),
        role="worker",
    )
    ordinary_operation_id = "ordinary-task"
    ordinary_request = _work_request()
    _record(
        store,
        operation_id=ordinary_operation_id,
        project_id=project_id,
        request=ordinary_request,
        authorized_by=authorizer,
        dispatch_authority=_work_authority(ordinary_request),
    )

    root_binding = store.agent_task_authority(project_id, root_operation_id)
    worker_binding = store.agent_task_authority(project_id, worker_operation_id)
    ordinary_binding = store.agent_task_authority(project_id, ordinary_operation_id)

    assert root_binding.episode_id == episode_id
    assert root_binding.apply_target == GraphTargetRef(kind="branch", branch_id=episode_id)
    assert root_binding.dispatch_authority is not None
    assert root_binding.dispatch_authority.profile == "orchestrator"
    assert worker_binding.episode_id == episode_id
    assert worker_binding.apply_target == GraphTargetRef(kind="branch", branch_id=episode_id)
    assert worker_binding.dispatch_authority is not None
    assert worker_binding.dispatch_authority.profile == "ordinary"
    assert ordinary_binding.episode_id is None
    assert ordinary_binding.apply_target == GraphTargetRef()


@pytest.mark.parametrize(
    ("authority", "message"),
    [
        (
            AgentDispatchAuthority(
                profile="ordinary",
                task_contract="discuss",
                scope=AgentDispatchScope(chat_scope="project"),
            ),
            "requires an exact chat scope and chat id",
        ),
        (
            AgentDispatchAuthority(
                profile="ordinary",
                task_contract="work_auto",
                scope=AgentDispatchScope(patch_kind="work"),
            ),
            "requires an exact chat scope and chat id",
        ),
        (
            AgentDispatchAuthority(
                profile="ordinary",
                task_contract="work_auto",
                scope=AgentDispatchScope(
                    chat_scope="node",
                    chat_id="chat-one",
                    patch_kind="work",
                ),
            ),
            "node chat scope requires an exact node id",
        ),
        (
            AgentDispatchAuthority(
                profile="ordinary",
                task_contract="scratch_patch",
                scope=AgentDispatchScope(
                    chat_scope="project",
                    chat_id="chat-one",
                    patch_kind="seed",
                ),
            ),
            "scratch_patch cannot carry chat identity",
        ),
        (
            AgentDispatchAuthority(
                profile="ordinary",
                task_contract="paper_readonly",
                scope=AgentDispatchScope(chat_scope="project", chat_id="chat-one"),
            ),
            "paper_readonly cannot carry chat, Patch, or control scope",
        ),
    ],
)
def test_incomplete_contract_scope_refuses_dispatch_and_apply(
    authority: AgentDispatchAuthority,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        require_dispatch(authority)

    task = AgentTaskAuthority(
        operation_id="malformed-scope-task",
        project_id="project-one",
        apply_target=GraphTargetRef(),
        authorized_by=AuthorizedHuman(
            space_id=str(uuid.uuid4()),
            user_id=str(uuid.uuid4()),
            display_name="Scope researcher",
        ),
        dispatch_authority=authority,
    )
    with pytest.raises(ValueError, match=message):
        require_apply(task, seed_patch(), is_project_member=seated_on_every_project)


def test_refused_dispatch_creates_no_task_and_never_enters_stream(tmp_path: Path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    entered = threading.Event()

    async def stream(
        _project_id: str,
        _kind: AgentTaskKind,
        _request: AgentTaskRequest,
        _execution: AgentTaskExecution,
    ) -> AsyncIterator[str]:
        entered.set()
        yield f"data: {AgentEvent(event='answer', text='unexpected').model_dump_json()}\n\n"

    orchestrate = AgentDispatchAuthority(
        profile="ordinary",
        task_contract="orchestrate",
        scope=AgentDispatchScope(run_truth_scope=["repo-a"], patch_kind="work"),
    )
    tasks = BackgroundAgentTasks(
        store,
        stream,
        dispatch_authority_resolver=lambda _kind, _request: orchestrate,
    )
    ordinary_tasks = BackgroundAgentTasks(store, stream)
    try:
        with pytest.raises(ValueError, match="action 'dispatch'.*orchestrate"):
            tasks.start(
                "project-one",
                "project_chat",
                _work_request(),
                authorized_by=_authorizer(store),
            )
        with pytest.raises(ValueError, match="human authorizer"):
            ordinary_tasks.start(
                "project-one",
                "project_chat",
                _work_request(),
            )
        with pytest.raises(ValueError, match="ordinary agent task.*human authorizer"):
            ordinary_tasks.start(
                "project-one",
                "project_chat",
                _work_request().model_copy(update={"mode": "discuss"}),
            )
        with pytest.raises(ValueError, match="ordinary agent task.*human authorizer"):
            ordinary_tasks.start(
                "project-one",
                "paper_coach",
                CoachRequest(message="Review this introduction."),
            )
    finally:
        tasks.shutdown()
        ordinary_tasks.shutdown()

    assert entered.is_set() is False
    assert store.agent_tasks("project-one") == []
    assert store.agent_usage_snapshot("project-one").records == []


def test_task_binding_is_durable_before_provider_execution(tmp_path: Path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    entered = threading.Event()
    observed: list[AgentDispatchAuthority | None] = []

    async def stream(
        _project_id: str,
        _kind: AgentTaskKind,
        _request: AgentTaskRequest,
        execution: AgentTaskExecution,
    ) -> AsyncIterator[str]:
        stored = store.agent_task(execution.operation_id)
        assert stored is not None
        observed.append(stored.dispatch_authority)
        entered.set()
        yield f"data: {AgentEvent(event='answer', text='done').model_dump_json()}\n\n"

    tasks = BackgroundAgentTasks(store, stream)
    request = _work_request()
    try:
        started = tasks.start(
            "project-one",
            "project_chat",
            request,
            authorized_by=_authorizer(store),
        )
        assert entered.wait(timeout=2)
        assert _wait_for_terminal(store, started.operation_id).status == "succeeded"
    finally:
        tasks.shutdown()

    assert observed == [_work_authority(request)]


@pytest.mark.parametrize(("action", "status"), [("resume", "paused"), ("retry", "failed")])
def test_recovery_readmits_and_preserves_parent_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    status: str,
) -> None:
    store = AppStore(tmp_path / f"{action}.sqlite3")
    authorizer = _authorizer(store)
    request = _work_request()
    authority = _work_authority(request)
    session_id = str(uuid.uuid4())
    stage = tmp_path / f"{action}-stage"
    stage.mkdir()
    previous = _record(
        store,
        operation_id=f"{action}-parent",
        request=request,
        status=status,
        authorized_by=authorizer,
        dispatch_authority=authority,
        native_session_id=session_id,
        stage_root=str(stage),
    )
    tasks = BackgroundAgentTasks(store, _unused_stream)
    monkeypatch.setattr(tasks, "_spawn_record", lambda record, *_args, **_kwargs: record)

    recovered = getattr(tasks, action)(previous.operation_id, authorized_by=authorizer)

    assert recovered.parent_operation_id == previous.operation_id
    assert recovered.dispatch_authority == authority
    assert store.agent_task(recovered.operation_id).dispatch_authority == authority  # type: ignore[union-attr]


@pytest.mark.parametrize(("action", "status"), [("resume", "paused"), ("retry", "failed")])
def test_recovery_of_a_legacy_parent_binds_the_continuations_own_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    status: str,
) -> None:
    """A task predating dispatch authority stays recoverable.

    Its binding cannot be invented after the fact, so the parent imposes no
    constraint. The continuation is still gated on the authority it resolves for
    itself, which is what keeps every dispatch checked.
    """

    store = AppStore(tmp_path / f"legacy-{action}.sqlite3")
    authorizer = _authorizer(store)
    request = _work_request()
    session_id = str(uuid.uuid4())
    stage = tmp_path / f"legacy-{action}-stage"
    stage.mkdir()
    previous = _record(
        store,
        operation_id=f"legacy-{action}-parent",
        request=request,
        status=status,
        authorized_by=authorizer,
        dispatch_authority=None,
        native_session_id=session_id,
        stage_root=str(stage),
    )
    tasks = BackgroundAgentTasks(store, _unused_stream)
    monkeypatch.setattr(tasks, "_spawn_record", lambda record, *_args, **_kwargs: record)

    recovered = getattr(tasks, action)(previous.operation_id, authorized_by=authorizer)

    assert recovered.parent_operation_id == previous.operation_id
    assert recovered.dispatch_authority == _work_authority(request)
    assert store.agent_task(recovered.operation_id).dispatch_authority is not None  # type: ignore[union-attr]


def test_continuation_refuses_a_missing_parent_before_insert_or_spawn(tmp_path: Path) -> None:
    store = AppStore(tmp_path / "missing-parent.sqlite3")
    authorizer = _authorizer(store)
    request = _work_request()
    authority = _work_authority(request)
    now = store.now()
    missing_parent = AgentTaskRecord(
        operation_id="missing-parent",
        project_id="project-one",
        kind="project_chat",
        status="failed",
        request=request.model_dump(mode="json"),
        created_at=now,
        updated_at=now,
        status_message="failed",
        authorized_by=authorizer,
        dispatch_authority=authority,
    )
    tasks = BackgroundAgentTasks(store, _unused_stream)

    with pytest.raises(ValueError, match="continuation parent is missing"):
        tasks._create_and_spawn(
            "project-one",
            "project_chat",
            request,
            parent=missing_parent,
            continuation="retry",
            estimate_seconds=1.0,
            estimate_samples=0,
            authorized_by=authorizer,
        )

    assert store.agent_tasks("project-one") == []


def _bound_child(
    store: AppStore,
    request: RunRequest,
    authorizer: AuthorizedHuman,
    *,
    operation_id: str,
    parent_id: str,
) -> AgentTaskRecord:
    now = store.now()
    return AgentTaskRecord(
        operation_id=operation_id,
        project_id="project-one",
        kind="project_chat",
        status="queued",
        request=request.model_dump(mode="json"),
        created_at=now,
        updated_at=now,
        status_message="queued",
        attempt=2,
        parent_operation_id=parent_id,
        authorized_by=authorizer,
        dispatch_authority=_work_authority(request),
    )


def test_storage_refuses_non_episode_child_without_an_existing_parent(tmp_path: Path) -> None:
    store = AppStore(tmp_path / "storage-missing.sqlite3")
    authorizer = _authorizer(store)
    request = _work_request()
    child = _bound_child(
        store,
        request,
        authorizer,
        operation_id="missing-child",
        parent_id="storage-parent",
    )

    with pytest.raises(ValueError, match="existing parent task"):
        store.create_agent_task(child)

    assert store.agent_task(child.operation_id) is None


def test_storage_admits_a_child_of_a_legacy_parent_that_has_no_binding(tmp_path: Path) -> None:
    store = AppStore(tmp_path / "storage-legacy.sqlite3")
    authorizer = _authorizer(store)
    request = _work_request()
    parent_id = "storage-parent"
    _record(
        store,
        operation_id=parent_id,
        request=request,
        authorized_by=authorizer,
        dispatch_authority=None,
    )
    child = _bound_child(
        store,
        request,
        authorizer,
        operation_id="legacy-child",
        parent_id=parent_id,
    )

    store.create_agent_task(child)

    stored = store.agent_task(child.operation_id)
    assert stored is not None
    assert stored.dispatch_authority == _work_authority(request)


def test_watcher_dispatch_binds_before_claim_and_refusal_leaves_claim_untouched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = AppStore(tmp_path / "watcher.sqlite3")
    authorizer = _authorizer(store)
    request = _work_request(trigger="watcher", watcher_ids=["watcher-one"])
    now = store.now()
    store.create_watchers(
        [
            WatcherRecord(
                watcher_id="watcher-one",
                project_id="project-one",
                origin_operation_id="origin-task",
                origin_task_kind="project_chat",
                chat_id=request.chat_id or "project-chat",
                continuation=WatcherContinuation(
                    provider=request.provider or "codex",
                    model=request.model,
                    reasoning=request.reasoning,
                    run_on=request.run_on or "local",
                    run_truth_scope=request.run_truth_scope,
                    patch_kind="work",
                ),
                status="completed",
                created_at=now,
                completed_at=now,
                check_command="true",
                log_path=str(tmp_path / "watcher.log"),
                cwd=str(tmp_path),
            )
        ]
    )
    claim_seen: list[AgentDispatchAuthority | None] = []
    stream_seen = threading.Event()

    def claim(
        record: AgentTaskRecord,
        _watcher_ids: list[str],
        **kwargs,
    ) -> AgentTaskRecord:
        claim_seen.append(record.dispatch_authority)
        return store.create_agent_task(record, **kwargs)

    async def stream(
        _project_id: str,
        _kind: AgentTaskKind,
        _request: AgentTaskRequest,
        execution: AgentTaskExecution,
    ) -> AsyncIterator[str]:
        stored = store.agent_task(execution.operation_id)
        assert stored is not None and stored.dispatch_authority == _work_authority(request)
        stream_seen.set()
        yield f"data: {AgentEvent(event='answer', text='done').model_dump_json()}\n\n"

    monkeypatch.setattr(store, "create_watcher_notification_task", claim)
    tasks = BackgroundAgentTasks(store, stream)
    try:
        started = start_watcher_notification(
            tasks,
            "project-one",
            "project_chat",
            request,
            ["watcher-one"],
            authorized_by=authorizer,
        )
        assert started is not None and stream_seen.wait(timeout=2)
        assert _wait_for_terminal(store, started.operation_id).status == "succeeded"
    finally:
        tasks.shutdown()

    assert claim_seen == [_work_authority(request)]

    refused_claim = False

    def must_not_claim(_record: AgentTaskRecord, _ids: list[str]) -> AgentTaskRecord:
        nonlocal refused_claim
        refused_claim = True
        raise AssertionError("refused watcher dispatch reached its claim")

    monkeypatch.setattr(store, "create_watcher_notification_task", must_not_claim)
    orchestrate = AgentDispatchAuthority(
        profile="ordinary",
        task_contract="orchestrate",
        scope=AgentDispatchScope(run_truth_scope=["repo-a"], patch_kind="work"),
    )
    refused = BackgroundAgentTasks(
        store,
        _unused_stream,
        dispatch_authority_resolver=lambda _kind, _request: orchestrate,
    )
    with pytest.raises(ValueError, match="action 'dispatch'"):
        start_watcher_notification(
            refused,
            "project-one",
            "project_chat",
            request,
            ["watcher-one"],
            authorized_by=authorizer,
        )
    assert refused_claim is False


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("cross-project", "unknown agent task"),
        ("missing-binding", "no dispatch authority binding"),
        ("discuss", "exposes no graph Patch channel"),
        ("paper", "exposes no graph Patch channel"),
        ("scope", "run_truth_scope does not match"),
        ("patch-kind", "Patch kind does not match"),
        ("control-node", "Experiment control node does not match"),
    ],
)
def test_live_apply_rejects_wrong_project_contract_or_scope_without_revision(
    manifest,
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    store = AppStore(tmp_path / f"{case}.sqlite3")
    authorizer = _authorizer(store)
    request = _work_request()
    authority: AgentDispatchAuthority | None = _work_authority(request)
    project_id = "other-project" if case == "cross-project" else "project-one"
    if case == "missing-binding":
        authority = None
    elif case == "discuss":
        authority = AgentDispatchAuthority(
            profile="ordinary",
            task_contract="discuss",
            scope=AgentDispatchScope(
                run_truth_scope=["repo-a"],
                chat_scope="project",
                chat_id=request.chat_id,
            ),
        )
    elif case == "paper":
        authority = AgentDispatchAuthority(
            profile="ordinary",
            task_contract="paper_readonly",
            scope=AgentDispatchScope(),
        )
    elif case == "scope":
        authority = AgentDispatchAuthority(
            profile="ordinary",
            task_contract="work_auto",
            scope=AgentDispatchScope(
                run_truth_scope=["repo-b"],
                chat_scope="project",
                chat_id=request.chat_id,
                patch_kind="work",
            ),
        )
    elif case == "patch-kind":
        authority = AgentDispatchAuthority(
            profile="ordinary",
            task_contract="scratch_patch",
            scope=AgentDispatchScope(
                run_truth_scope=["repo-a"],
                patch_kind="refresh",
            ),
        )
    elif case == "control-node":
        authority = AgentDispatchAuthority(
            profile="ordinary",
            task_contract="work_auto",
            scope=AgentDispatchScope(
                run_truth_scope=["repo-a"],
                chat_scope="project",
                chat_id=request.chat_id,
                patch_kind="experiment_loop",
                control_node_id="exp/bound",
                control_episode_id=str(uuid.uuid4()),
            ),
        )
    operation_id = f"{case}-task"
    _record(
        store,
        operation_id=operation_id,
        project_id=project_id,
        request=request,
        authorized_by=authorizer,
        dispatch_authority=authority,
    )
    history = HistoryManager(
        manifest,
        project_id="project-one",
        require_attribution=True,
        project_membership_check=seated_on_every_project,
        agent_authority_resolver=store.agent_task_authority,
    )
    patch = seed_patch().model_copy(
        update={
            "kind": "experiment_loop" if case == "control-node" else "work",
            "source_operation_id": operation_id,
            "experiment_control_node_id": ("exp/forged" if case == "control-node" else None),
        }
    )

    with pytest.raises(ValueError, match=message):
        history.append(patch)

    assert history.load_patches() == []
    assert history.state().revision == 0


def test_valid_work_apply_stamps_canonical_task_and_replay_needs_no_task_rows(
    manifest,
    tmp_path: Path,
) -> None:
    store = AppStore(tmp_path / "valid.sqlite3")
    authorizer = _authorizer(store)
    request = _work_request()
    operation_id = "valid-work"
    _record(
        store,
        operation_id=operation_id,
        request=request,
        authorized_by=authorizer,
        dispatch_authority=_work_authority(request),
    )
    history = HistoryManager(
        manifest,
        project_id="project-one",
        require_attribution=True,
        project_membership_check=seated_on_every_project,
        agent_authority_resolver=store.agent_task_authority,
    )
    patch = seed_patch().model_copy(
        update={
            "kind": "work",
            "source_operation_id": operation_id,
        }
    )

    appended, result = history.append(patch)

    assert result.state.revision == 1
    assert appended.authorized_by == authorizer
    assert appended.profile == "ordinary"
    assert appended.task_id == operation_id

    def resolver_must_not_run(_project_id: str, _operation_id: str):
        raise AssertionError("replay consulted operational authority")

    replay = HistoryManager(
        manifest,
        project_id="project-one",
        require_attribution=True,
        project_membership_check=seated_on_every_project,
        agent_authority_resolver=resolver_must_not_run,
    )
    assert replay.state().revision == 1


async def _unused_stream(
    _project_id: str,
    _kind: AgentTaskKind,
    _request: AgentTaskRequest,
    _execution: AgentTaskExecution,
) -> AsyncIterator[str]:
    raise AssertionError("provider stream should not run")
    yield ""
