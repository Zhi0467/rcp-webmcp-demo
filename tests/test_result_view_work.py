from __future__ import annotations

import hashlib
import re
from contextlib import aclosing
from datetime import datetime
from pathlib import Path

import pytest

from rcp.agents import AgentEvent, AgentProcessControl
from rcp.background import AgentTaskExecution, BackgroundAgentTasks
from rcp.core.models import AuthorizedHuman, Patch
from rcp.runs.tasks.work import stream_work_run
from rcp.service import RunRequest, resolve_dispatch_authority
from rcp.storage import AgentTaskRecord, AppStore

from .helpers import append_fixture_patch, authorized_human, seed_patch, wait_for_task
from .helpers import create_named_app as create_app

_EXPERIMENT_ID = "exp/result-view"


def _experiment_patch() -> Patch:
    return Patch(
        kind="refresh",
        author="agent",
        summary="Added an Experiment for result-view tests.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": _EXPERIMENT_ID,
                        "type": "experiment",
                        "title": "Compare learning curves",
                        "objective": "Compare loss curves and generated samples across seeds.",
                        "completion_criteria": ["The completed runs are compared."],
                        "invocation_ceiling": 2,
                    }
                ],
            }
        ],
    )


def _request(
    chat_id: str,
    message: str,
    *,
    result_view: dict[str, str] | None,
    session_id: str | None = None,
) -> RunRequest:
    return RunRequest.model_validate(
        {
            "provider": "codex",
            "reasoning": "medium",
            "run_on": "laptop",
            "run_truth_scope": ["repo-a"],
            "chat_scope": "node",
            "node_id": _EXPERIMENT_ID,
            "message": message,
            "chat_id": chat_id,
            "session_id": session_id,
            "mode": "work",
            "trigger": "human",
            "patch_kind": "work",
            "result_view": result_view,
        }
    )


def _execution(
    store: AppStore,
    project_id: str,
    operation_id: str,
    request: RunRequest,
    *,
    stage_root: str | None = None,
    parent_operation_id: str | None = None,
    continuation="fresh",
) -> AgentTaskExecution:
    now = store.now()
    owner = store.local_owner
    assert owner is not None and owner.display_name is not None
    dispatch_authority = resolve_dispatch_authority("node_chat", request)
    assert dispatch_authority is not None
    store.create_agent_task(
        AgentTaskRecord(
            operation_id=operation_id,
            project_id=project_id,
            kind="node_chat",
            status="running",
            request=request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="running",
            attempt=2 if parent_operation_id is not None else 1,
            parent_operation_id=parent_operation_id,
            native_session_id=request.session_id,
            stage_root=stage_root,
            dispatch_authority=dispatch_authority,
            authorized_by=AuthorizedHuman(
                space_id=store.space_id,
                user_id=owner.user_id,
                display_name=owner.display_name,
            ),
        )
    )
    store.record_agent_task_receipt(
        operation_id,
        "operation_created",
        {
            "kind": "node_chat",
            "attempt": 2 if parent_operation_id is not None else 1,
            "has_parent": parent_operation_id is not None,
            "resumed": continuation != "fresh",
            "continuation_cause": continuation,
        },
    )
    return AgentTaskExecution(
        operation_id=operation_id,
        store=store,
        control=AgentProcessControl(),
        stage_root=stage_root,
        continuation=continuation,
    )


class _ViewLauncher:
    def __init__(self, session_id: str, writer=None) -> None:
        self.session_id = session_id
        self.writer = writer
        self.prompts: list[str] = []
        self.workspaces: list[Path] = []
        self.sessions: list[str | None] = []

    async def stream(self, _provider, prompt, **kwargs):
        workspace = Path(kwargs["cwd"])
        self.prompts.append(prompt)
        self.workspaces.append(workspace)
        self.sessions.append(kwargs.get("session_id"))
        if self.writer is not None:
            self.writer(prompt, workspace)
        yield AgentEvent(event="session", session_id=self.session_id)
        yield AgentEvent(event="answer", text="The view is ready without changing the graph.")
        yield AgentEvent(event="done")


class _StoppingViewLauncher(_ViewLauncher):
    def __init__(self, session_id: str, event: str, writer) -> None:
        super().__init__(session_id, writer)
        self.event = event

    async def stream(self, _provider, prompt, **kwargs):
        workspace = Path(kwargs["cwd"])
        self.prompts.append(prompt)
        self.workspaces.append(workspace)
        self.sessions.append(kwargs.get("session_id"))
        self.writer(prompt, workspace)
        yield AgentEvent(event="session", session_id=self.session_id)
        yield AgentEvent(event=self.event, text=f"provider {self.event}")


class _RetryCreateLauncher(_ViewLauncher):
    async def stream(self, _provider, prompt, **kwargs):
        workspace = Path(kwargs["cwd"])
        self.prompts.append(prompt)
        self.workspaces.append(workspace)
        self.sessions.append(kwargs.get("session_id"))
        yield AgentEvent(event="session", session_id=self.session_id)
        if len(self.prompts) == 1:
            yield AgentEvent(event="error", text="provider disconnected")
            return
        _created_slot(prompt).joinpath("retry-loss-curves.html").write_text(
            "<html><body>retry succeeded</body></html>",
            encoding="utf-8",
        )
        yield AgentEvent(event="answer", text="The retried view is ready.")
        yield AgentEvent(event="done")


class _HardCrashViewLauncher(_ViewLauncher):
    async def stream(self, _provider, prompt, **kwargs):
        workspace = Path(kwargs["cwd"])
        self.prompts.append(prompt)
        self.workspaces.append(workspace)
        self.sessions.append(kwargs.get("session_id"))
        yield AgentEvent(event="session", session_id=self.session_id)
        assert self.writer is not None
        self.writer(prompt, workspace)
        raise RuntimeError("simulated hard process interruption")


async def _events(stream) -> list[AgentEvent]:
    return [
        AgentEvent.model_validate_json(frame.removeprefix("data: ").strip())
        async for frame in stream
    ]


def _created_slot(prompt: str) -> Path:
    prompt = _launch_contract_text(prompt)
    match = re.search(r"directly inside `([^`]+)`", prompt)
    if match is None:
        raise RuntimeError("result-view create prompt omitted its exact slot")
    return Path(match.group(1))


def _revised_path(prompt: str) -> Path:
    prompt = _launch_contract_text(prompt)
    match = re.search(r"existing HTML file `([^`]+)`", prompt)
    assert match is not None
    return Path(match.group(1))


def _launch_contract_text(prompt: str) -> str:
    contract = re.search(
        r"Open and follow the immutable RCP task contract at:\s*([^\n]+)",
        prompt,
    )
    return Path(contract.group(1)).read_text(encoding="utf-8") if contract is not None else prompt


def _receipts(store: AppStore, operation_id: str, category: str):
    return [
        receipt
        for receipt in store.agent_task_receipts(operation_id)
        if receipt.category == category
    ]


@pytest.mark.asyncio
async def test_create_and_revise_result_view_keep_one_cwd_session_and_stable_file(
    manifest,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    app = create_app(str(manifest.path), data_dir=data_dir)
    service = app.state.service
    store = app.state.background_tasks.store
    append_fixture_patch(service, seed_patch())
    append_fixture_patch(service, _experiment_patch())
    project_id = app.state.default_project_id
    assert project_id is not None
    chat_id = "result-view-chat"
    session_id = "result-view-native-session"
    create_message = "/show-results  keep  these\nexact human bytes"
    initial_revision = service.history.state().revision

    def create_writer(prompt: str, _workspace: Path) -> None:
        slot = _created_slot(prompt)
        assert re.fullmatch(r"[0-9a-f]{24}", slot.name)
        slot.joinpath("loss-curves-by-seed.html").write_text(
            "<html><body>loss curves v1</body></html>",
            encoding="utf-8",
        )

    create_request = _request(
        chat_id,
        create_message,
        result_view={"action": "create"},
    )
    create_execution = _execution(
        store,
        project_id,
        "result-view-create",
        create_request,
    )
    create_launcher = _ViewLauncher(session_id, create_writer)
    create_events = await _events(
        stream_work_run(
            service,
            create_launcher,
            create_request,
            data_dir,
            execution=create_execution,
        )
    )

    assert not [event for event in create_events if event.event == "error"]
    assert [event.event for event in create_events].count("artifact") == 0
    assert create_launcher.prompts[0].count(create_message) == 1
    assert "rcp-result-view-gesture" in create_launcher.prompts[0]
    assert "gesture:'box'|'underscore'" in create_launcher.prompts[0]
    assert "The page may omit gestures" in create_launcher.prompts[0]
    assert create_launcher.workspaces[0] == Path(create_execution.stage_root or "")
    records = store.list_result_views(project_id, chat_id=chat_id)
    assert len(records) == 1
    created = records[0]
    assert created.source_name == "loss-curves-by-seed.html"
    assert created.native_session_id == session_id
    assert created.stage_host == ""
    assert created.stage_root == create_execution.stage_root
    assert created.origin_operation_id == "result-view-create"
    assert created.latest_operation_id == "result-view-create"
    assert datetime.fromisoformat(created.created_at).utcoffset() is not None
    assert datetime.fromisoformat(created.expires_at).utcoffset() is not None
    assert len(_receipts(store, "result-view-create", "result_view_created")) == 1
    assert service.history.state().revision == initial_revision
    store.complete_agent_task("result-view-create", applied_revision=None, result={})

    revise_message = "Box the late spike, then explain  why.\nDo not normalize this."

    def revise_writer(prompt: str, workspace: Path) -> None:
        target = _revised_path(prompt)
        assert target.name == created.source_name
        assert target.parent.name == created.view_id
        assert workspace == create_launcher.workspaces[0]
        target.write_text("<html><body>loss curves v2</body></html>", encoding="utf-8")

    revise_request = _request(
        chat_id,
        revise_message,
        result_view={"action": "revise", "view_id": created.view_id},
        session_id=session_id,
    )
    revise_execution = _execution(
        store,
        project_id,
        "result-view-revise",
        revise_request,
        stage_root=created.stage_root,
    )
    revise_launcher = _ViewLauncher(session_id, revise_writer)
    revise_events = await _events(
        stream_work_run(
            service,
            revise_launcher,
            revise_request,
            data_dir,
            execution=revise_execution,
        )
    )

    assert not [event for event in revise_events if event.event == "error"]
    assert [event.event for event in revise_events].count("artifact") == 0
    assert revise_launcher.sessions == [session_id]
    assert revise_launcher.prompts[0].count(revise_message) == 1
    assert "atomic replacement at that path is allowed" in revise_launcher.prompts[0]
    revised = store.result_view_for_diagnostics(created.view_id)
    assert revised is not None
    assert revised.view_id == created.view_id
    assert revised.origin_operation_id == created.origin_operation_id
    assert revised.latest_operation_id == "result-view-revise"
    assert (
        revised.content_sha256
        == hashlib.sha256(b"<html><body>loss curves v2</body></html>").hexdigest()
    )
    assert revised.updated_at > created.updated_at
    assert revised.expires_at > created.expires_at
    assert (
        store.result_view_bytes(
            revised.view_id,
            expected_content_sha256=revised.content_sha256,
        )
        == b"<html><body>loss curves v2</body></html>"
    )
    assert len(_receipts(store, "result-view-revise", "result_view_revised")) == 1
    assert service.history.state().revision == initial_revision


@pytest.mark.asyncio
async def test_rejected_revision_leaves_stored_bytes_without_rejecting_answer_or_graph(
    manifest,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    app = create_app(str(manifest.path), data_dir=data_dir)
    service = app.state.service
    store = app.state.background_tasks.store
    append_fixture_patch(service, seed_patch())
    append_fixture_patch(service, _experiment_patch())
    project_id = app.state.default_project_id
    assert project_id is not None
    chat_id = "result-view-rejection-chat"
    session_id = "result-view-rejection-session"

    def create_writer(prompt: str, _workspace: Path) -> None:
        _created_slot(prompt).joinpath("sample-failures.html").write_text(
            "<html><body>original samples</body></html>",
            encoding="utf-8",
        )

    create_request = _request(
        chat_id,
        "Show the failure samples.",
        result_view={"action": "create"},
    )
    create_execution = _execution(store, project_id, "rejected-view-create", create_request)
    await _events(
        stream_work_run(
            service,
            _ViewLauncher(session_id, create_writer),
            create_request,
            data_dir,
            execution=create_execution,
        )
    )
    created = store.list_result_views(project_id, chat_id=chat_id)[0]
    store.complete_agent_task("rejected-view-create", applied_revision=None, result={})
    original = Path(created.stage_root) / "views" / created.view_id / created.source_name
    original_bytes = original.read_bytes()
    initial_revision = service.history.state().revision

    def invalid_revision_writer(prompt: str, _workspace: Path) -> None:
        target = _revised_path(prompt)
        target.write_text("<html><body>mutated samples</body></html>", encoding="utf-8")
        target.with_name("unexpected-second-view.html").write_text(
            "<html><body>extra</body></html>",
            encoding="utf-8",
        )

    request = _request(
        chat_id,
        "Underscore two samples and compare them.",
        result_view={"action": "revise", "view_id": created.view_id},
        session_id=session_id,
    )
    execution = _execution(
        store,
        project_id,
        "rejected-view-revise",
        request,
        stage_root=created.stage_root,
    )
    events = await _events(
        stream_work_run(
            service,
            _ViewLauncher(session_id, invalid_revision_writer),
            request,
            data_dir,
            execution=execution,
        )
    )

    assert not [event for event in events if event.event == "error"]
    assert any(event.event == "answer" for event in events)
    assert events[-1].event == "done"
    assert original.read_bytes() != original_bytes
    assert sorted(item.name for item in original.parent.iterdir()) == [
        created.source_name,
        "unexpected-second-view.html",
    ]
    unchanged = store.result_view_for_diagnostics(created.view_id)
    assert unchanged is not None
    assert unchanged.latest_operation_id == created.latest_operation_id
    assert unchanged.content_sha256 == created.content_sha256
    assert unchanged.size_bytes == created.size_bytes
    assert unchanged.source_name == created.source_name
    assert (
        store.result_view_bytes(
            unchanged.view_id,
            expected_content_sha256=unchanged.content_sha256,
        )
        == original_bytes
    )
    rejection = _receipts(store, "rejected-view-revise", "result_view_rejected")
    assert len(rejection) == 1
    assert set(rejection[0].payload) == {"action", "view_id", "problem"}
    assert "exactly one" in str(rejection[0].payload["problem"])
    assert service.history.state().revision == initial_revision


@pytest.mark.asyncio
async def test_revision_without_inherited_stage_fails_before_provider_launch(
    manifest,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    app = create_app(str(manifest.path), data_dir=data_dir)
    service = app.state.service
    store = app.state.background_tasks.store
    append_fixture_patch(service, seed_patch())
    append_fixture_patch(service, _experiment_patch())
    project_id = app.state.default_project_id
    assert project_id is not None
    chat_id = "result-view-missing-stage-chat"
    session_id = "result-view-missing-stage-session"

    def create_writer(prompt: str, _workspace: Path) -> None:
        _created_slot(prompt).joinpath("curves.html").write_text(
            "<html><body>curves</body></html>",
            encoding="utf-8",
        )

    create_request = _request(
        chat_id,
        "Draw the curves.",
        result_view={"action": "create"},
    )
    create_execution = _execution(store, project_id, "missing-stage-create", create_request)
    await _events(
        stream_work_run(
            service,
            _ViewLauncher(session_id, create_writer),
            create_request,
            data_dir,
            execution=create_execution,
        )
    )
    created = store.list_result_views(project_id, chat_id=chat_id)[0]
    store.complete_agent_task("missing-stage-create", applied_revision=None, result={})

    request = _request(
        chat_id,
        "Revise the curves.",
        result_view={"action": "revise", "view_id": created.view_id},
        session_id=session_id,
    )
    execution = _execution(store, project_id, "missing-stage-revise", request)
    launcher = _ViewLauncher(session_id)
    events = await _events(
        stream_work_run(
            service,
            launcher,
            request,
            data_dir,
            execution=execution,
        )
    )

    errors = [event.text for event in events if event.event == "error"]
    assert errors and "no inherited conversation stage" in (errors[0] or "")
    assert launcher.prompts == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("terminal_event", "expected_status"),
    [
        ("paused", "paused"),
        ("error", "failed"),
        ("post_provider_error", "failed"),
    ],
)
async def test_background_stream_close_leaves_stored_bytes_unchanged(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_event: str,
    expected_status: str,
) -> None:
    data_dir = tmp_path / "data"
    app = create_app(str(manifest.path), data_dir=data_dir)
    service = app.state.service
    store = app.state.background_tasks.store
    append_fixture_patch(service, seed_patch())
    append_fixture_patch(service, _experiment_patch())
    project_id = app.state.default_project_id
    assert project_id is not None
    chat_id = f"result-view-close-{terminal_event}"
    session_id = f"result-view-close-session-{terminal_event}"

    def create_writer(prompt: str, _workspace: Path) -> None:
        _created_slot(prompt).joinpath("close-safe-curves.html").write_text(
            "<html><body>original close-safe curves</body></html>",
            encoding="utf-8",
        )

    create_request = _request(
        chat_id,
        "Create the close-safe curves.",
        result_view={"action": "create"},
    )
    create_operation = f"result-view-close-create-{terminal_event}"
    create_execution = _execution(store, project_id, create_operation, create_request)
    await _events(
        stream_work_run(
            service,
            _ViewLauncher(session_id, create_writer),
            create_request,
            data_dir,
            execution=create_execution,
        )
    )
    created = store.list_result_views(project_id, chat_id=chat_id)[0]
    store.complete_agent_task(create_operation, applied_revision=None, result={})
    target = Path(created.stage_root) / "views" / created.view_id / created.source_name
    original_bytes = target.read_bytes()

    def revision_writer(prompt: str, _workspace: Path) -> None:
        _revised_path(prompt).write_text(
            "<html><body>unaccepted mutation</body></html>",
            encoding="utf-8",
        )

    request = _request(
        chat_id,
        "Revise the curves, but preserve them if this turn stops.",
        result_view={"action": "revise", "view_id": created.view_id},
        session_id=session_id,
    )
    if terminal_event == "post_provider_error":
        import rcp.runs.tasks.work as work_module

        def fail_after_provider(*_args, **_kwargs) -> None:
            raise RuntimeError("post-provider lifecycle failure")

        monkeypatch.setattr(work_module, "_commit_chat_prompt_state", fail_after_provider)
        launcher = _ViewLauncher(session_id, revision_writer)
    else:
        launcher = _StoppingViewLauncher(session_id, terminal_event, revision_writer)

    async def stream(_project_id, _kind, run_request, execution):
        async for frame in stream_work_run(
            service,
            launcher,
            run_request,
            data_dir,
            execution=execution,
        ):
            yield frame

    tasks = BackgroundAgentTasks(store, stream)
    started = tasks.start(
        project_id,
        "node_chat",
        request,
        authorized_by=authorized_human(store),
        stage_root=created.stage_root,
    )
    settled = wait_for_task(store, started.operation_id)

    assert settled.status == expected_status
    assert target.read_bytes() != original_bytes
    unchanged = store.result_view_for_diagnostics(created.view_id)
    assert unchanged is not None
    assert unchanged.latest_operation_id == created.latest_operation_id
    assert unchanged.content_sha256 == created.content_sha256
    assert unchanged.size_bytes == created.size_bytes
    assert (
        store.result_view_bytes(
            unchanged.view_id,
            expected_content_sha256=unchanged.content_sha256,
        )
        == original_bytes
    )


@pytest.mark.asyncio
async def test_hard_interrupted_revision_leaves_stored_bytes_unchanged(
    manifest,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    app = create_app(str(manifest.path), data_dir=data_dir)
    service = app.state.service
    store = app.state.background_tasks.store
    append_fixture_patch(service, seed_patch())
    append_fixture_patch(service, _experiment_patch())
    project_id = app.state.default_project_id
    assert project_id is not None
    chat_id = "result-view-crash"
    session_id = "result-view-crash-session"

    def create_writer(prompt: str, _workspace: Path) -> None:
        _created_slot(prompt).joinpath("crash-safe.html").write_text(
            "<html><body>durable original</body></html>",
            encoding="utf-8",
        )

    create_request = _request(
        chat_id,
        "Create the crash-safe view.",
        result_view={"action": "create"},
    )
    create_execution = _execution(store, project_id, "crash-create", create_request)
    await _events(
        stream_work_run(
            service,
            _ViewLauncher(session_id, create_writer),
            create_request,
            data_dir,
            execution=create_execution,
        )
    )
    created = store.list_result_views(project_id, chat_id=chat_id)[0]
    store.complete_agent_task(create_execution.operation_id, applied_revision=None, result={})
    target = Path(created.stage_root) / "views" / created.view_id / created.source_name
    original = target.read_bytes()

    revision_request = _request(
        chat_id,
        "Revise the view before a hard interruption.",
        result_view={"action": "revise", "view_id": created.view_id},
        session_id=session_id,
    )
    crashed_operation = "crash-revise"
    crashed_execution = _execution(
        store,
        project_id,
        crashed_operation,
        revision_request,
        stage_root=created.stage_root,
    )

    def interrupted_writer(prompt: str, _workspace: Path) -> None:
        interrupted_target = _revised_path(prompt)
        interrupted_target.write_text(
            "<html><body>uncommitted crash mutation</body></html>",
            encoding="utf-8",
        )

    with pytest.raises(RuntimeError, match="hard process interruption"):
        await _events(
            stream_work_run(
                service,
                _HardCrashViewLauncher(session_id, interrupted_writer),
                revision_request,
                data_dir,
                execution=crashed_execution,
            )
        )
    assert target.read_bytes() != original
    assert (
        store.result_view_bytes(
            created.view_id,
            expected_content_sha256=created.content_sha256,
        )
        == original
    )


def test_background_retry_creates_once_from_the_original_unbound_slot(
    manifest,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    app = create_app(str(manifest.path), data_dir=data_dir)
    service = app.state.service
    store = app.state.background_tasks.store
    append_fixture_patch(service, seed_patch())
    append_fixture_patch(service, _experiment_patch())
    project_id = app.state.default_project_id
    assert project_id is not None
    session_id = "result-view-create-retry-session"
    request = _request(
        "result-view-create-retry-chat",
        "Create the retryable view.",
        result_view={"action": "create"},
    )
    launcher = _RetryCreateLauncher(session_id)

    async def stream(_project_id, _kind, run_request, execution):
        async for frame in stream_work_run(
            service,
            launcher,
            run_request,
            data_dir,
            execution=execution,
        ):
            yield frame

    tasks = BackgroundAgentTasks(store, stream)
    authorized_by = authorized_human(store)
    started = tasks.start(
        project_id,
        "node_chat",
        request,
        authorized_by=authorized_by,
    )
    failed = wait_for_task(store, started.operation_id)
    assert failed.status == "failed"
    assert failed.native_session_id == session_id
    assert failed.stage_root is not None
    assert store.list_result_views(project_id, chat_id=request.chat_id) == []

    retried = tasks.retry(started.operation_id, authorized_by=authorized_by)
    completed = wait_for_task(store, retried.operation_id)

    assert completed.status == "succeeded", completed.error
    records = store.list_result_views(project_id, chat_id=request.chat_id)
    assert len(records) == 1
    assert records[0].origin_operation_id == started.operation_id
    assert records[0].latest_operation_id == retried.operation_id
    assert _created_slot(launcher.prompts[0]) == _created_slot(launcher.prompts[1])
    assert launcher.workspaces[0] == launcher.workspaces[1]


def test_background_retry_handoff_creates_after_pre_slot_failure(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rcp.runs.tasks.result_views as result_views_module

    data_dir = tmp_path / "data"
    app = create_app(str(manifest.path), data_dir=data_dir)
    service = app.state.service
    store = app.state.background_tasks.store
    append_fixture_patch(service, seed_patch())
    append_fixture_patch(service, _experiment_patch())
    project_id = app.state.default_project_id
    assert project_id is not None
    request = _request(
        "result-view-create-handoff-chat",
        "Create the view after setup recovers.",
        result_view={"action": "create"},
    )
    real_prepare_slot = result_views_module.prepare_result_view_slot
    prepare_reuse: list[bool] = []

    def fail_before_first_slot(*args, **kwargs):
        prepare_reuse.append(bool(kwargs["reuse"]))
        if len(prepare_reuse) == 1:
            raise ValueError("simulated failure before the deterministic slot existed")
        return real_prepare_slot(*args, **kwargs)

    monkeypatch.setattr(result_views_module, "prepare_result_view_slot", fail_before_first_slot)

    def create_writer(prompt: str, _workspace: Path) -> None:
        _created_slot(prompt).joinpath("handoff-recovery.html").write_text(
            "<html><body>handoff recovery</body></html>",
            encoding="utf-8",
        )

    launcher = _ViewLauncher("result-view-create-handoff-session", create_writer)

    async def stream(_project_id, _kind, run_request, execution):
        async for frame in stream_work_run(
            service,
            launcher,
            run_request,
            data_dir,
            execution=execution,
        ):
            yield frame

    tasks = BackgroundAgentTasks(store, stream)
    authorized_by = authorized_human(store)
    started = tasks.start(
        project_id,
        "node_chat",
        request,
        authorized_by=authorized_by,
    )
    failed = wait_for_task(store, started.operation_id)
    assert failed.status == "failed"
    assert failed.native_session_id is None
    assert failed.stage_root is not None
    expected_view_id = hashlib.sha256(f"result-view\0{started.operation_id}".encode()).hexdigest()[
        :24
    ]
    assert not Path(failed.stage_root).joinpath("views", expected_view_id).exists()

    retried = tasks.retry(started.operation_id, authorized_by=authorized_by)
    completed = wait_for_task(store, retried.operation_id)

    assert store.agent_task_continuation_cause(retried.operation_id) == "handoff"
    assert completed.status == "succeeded", completed.error
    assert launcher.sessions == [None]
    records = store.list_result_views(project_id, chat_id=request.chat_id)
    assert len(records) == 1
    assert records[0].view_id == expected_view_id
    assert records[0].origin_operation_id == started.operation_id
    assert records[0].latest_operation_id == retried.operation_id
    assert prepare_reuse == [False, True, False]


@pytest.mark.parametrize(
    ("retry_provider", "expected_continuation"),
    [(None, "retry"), ("claude", "handoff")],
)
def test_background_retry_recovers_without_reauthoring_an_already_bound_create(
    manifest,
    tmp_path: Path,
    retry_provider: str | None,
    expected_continuation: str,
) -> None:
    data_dir = tmp_path / "data"
    app = create_app(str(manifest.path), data_dir=data_dir)
    service = app.state.service
    store = app.state.background_tasks.store
    append_fixture_patch(service, seed_patch())
    append_fixture_patch(service, _experiment_patch())
    project_id = app.state.default_project_id
    assert project_id is not None
    session_id = "result-view-bound-create-session"
    request = _request(
        "result-view-bound-create-chat",
        "Create one view before the downstream task failure.",
        result_view={"action": "create"},
    )

    launcher: _ViewLauncher

    def create_writer(prompt: str, _workspace: Path) -> None:
        if len(launcher.prompts) == 1:
            _created_slot(prompt).joinpath("already-bound-view.html").write_text(
                "<html><body>already bound</body></html>",
                encoding="utf-8",
            )
            return
        assert "result-view authoring contract" not in _launch_contract_text(prompt)

    launcher = _ViewLauncher(session_id, create_writer)
    fail_after_bound = True

    async def stream(_project_id, _kind, run_request, execution):
        nonlocal fail_after_bound
        async with aclosing(
            stream_work_run(
                service,
                launcher,
                run_request,
                data_dir,
                execution=execution,
            )
        ) as work_stream:
            async for frame in work_stream:
                yield frame
                event = AgentEvent.model_validate_json(frame.removeprefix("data: ").strip())
                if event.event == "answer" and fail_after_bound:
                    fail_after_bound = False
                    error = AgentEvent(event="error", text="downstream delivery failed")
                    yield f"data: {error.model_dump_json()}\n\n"
                    return

    tasks = BackgroundAgentTasks(store, stream)
    authorized_by = authorized_human(store)
    started = tasks.start(
        project_id,
        "node_chat",
        request,
        authorized_by=authorized_by,
    )
    failed = wait_for_task(store, started.operation_id)
    assert failed.status == "failed"
    records = store.list_result_views(project_id, chat_id=request.chat_id)
    assert len(records) == 1
    created = records[0]
    target = Path(created.stage_root) / "views" / created.view_id / created.source_name
    created_bytes = target.read_bytes()

    if retry_provider is not None:
        launcher.session_id = "result-view-bound-create-handoff-session"
    retried = tasks.retry(
        started.operation_id,
        provider=retry_provider,
        authorized_by=authorized_by,
    )
    recovered = wait_for_task(store, retried.operation_id)

    assert store.agent_task_continuation_cause(retried.operation_id) == expected_continuation
    assert recovered.status == "succeeded", recovered.error
    assert len(launcher.prompts) == 2
    assert "result-view authoring contract" not in _launch_contract_text(launcher.prompts[1])
    unchanged = store.list_result_views(project_id, chat_id=request.chat_id)
    assert len(unchanged) == 1
    assert unchanged[0].view_id == created.view_id
    assert unchanged[0].latest_operation_id == created.latest_operation_id
    assert unchanged[0].content_sha256 == created.content_sha256
    assert unchanged[0].provider == created.provider
    assert unchanged[0].native_session_id == created.native_session_id
    assert unchanged[0].stage_root == created.stage_root
    assert target.read_bytes() == created_bytes


@pytest.mark.asyncio
@pytest.mark.parametrize("continuation", ["resume", "retry"])
async def test_accepted_create_recovery_continues_without_reauthoring_result_view(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    continuation: str,
) -> None:
    data_dir = tmp_path / "data"
    app = create_app(str(manifest.path), data_dir=data_dir)
    service = app.state.service
    store = app.state.background_tasks.store
    append_fixture_patch(service, seed_patch())
    append_fixture_patch(service, _experiment_patch())
    project_id = app.state.default_project_id
    assert project_id is not None
    chat_id = f"settled-create-{continuation}-chat"
    session_id = f"settled-create-{continuation}-session"
    original_operation = f"settled-create-{continuation}"

    def create_writer(prompt: str, _workspace: Path) -> None:
        _created_slot(prompt).joinpath("settled-create.html").write_text(
            "<html><body>settled create</body></html>",
            encoding="utf-8",
        )

    create_request = _request(
        chat_id,
        "Create the result view before downstream recovery.",
        result_view={"action": "create"},
    )
    create_execution = _execution(
        store,
        project_id,
        original_operation,
        create_request,
    )
    await _events(
        stream_work_run(
            service,
            _ViewLauncher(session_id, create_writer),
            create_request,
            data_dir,
            execution=create_execution,
        )
    )
    created = store.list_result_views(project_id, chat_id=chat_id)[0]
    store.checkpoint_agent_task(original_operation, native_session_id=session_id)
    store.fail_agent_task(
        original_operation,
        "Downstream task settlement was interrupted after accepting the view.",
        status="interrupted" if continuation == "resume" else "failed",
    )
    target = Path(created.stage_root) / "views" / created.view_id / created.source_name
    original_bytes = target.read_bytes()

    recovery_request = create_request.model_copy(update={"session_id": session_id})
    recovery_operation = f"settled-create-{continuation}-recovery"
    recovery_execution = _execution(
        store,
        project_id,
        recovery_operation,
        recovery_request,
        stage_root=created.stage_root,
        parent_operation_id=original_operation,
        continuation=continuation,
    )

    def recovery_writer(prompt: str, _workspace: Path) -> None:
        assert "result-view authoring contract" not in _launch_contract_text(prompt)

    launcher = _ViewLauncher(session_id, recovery_writer)
    with monkeypatch.context() as recovery_patch:
        if continuation == "resume":
            # Isolate result-view settlement from the independent prompt-baseline replay seam.
            recovery_patch.setattr(
                "rcp.runs.tasks.work._commit_chat_prompt_state",
                lambda *_args, **_kwargs: None,
            )
        events = await _events(
            stream_work_run(
                service,
                launcher,
                recovery_request,
                data_dir,
                execution=recovery_execution,
            )
        )

    assert not [event for event in events if event.event == "error"]
    assert any(event.event == "answer" for event in events)
    assert len(launcher.prompts) == 1
    assert target.read_bytes() == original_bytes
    unchanged = store.list_result_views(project_id, chat_id=chat_id)
    assert len(unchanged) == 1
    assert unchanged[0].latest_operation_id == original_operation
    assert unchanged[0].content_sha256 == created.content_sha256
    assert _receipts(store, recovery_operation, "result_view_created") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("continuation", ["resume", "retry"])
async def test_accepted_revision_recovery_continues_without_reauthoring_result_view(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    continuation: str,
) -> None:
    data_dir = tmp_path / "data"
    app = create_app(str(manifest.path), data_dir=data_dir)
    service = app.state.service
    store = app.state.background_tasks.store
    append_fixture_patch(service, seed_patch())
    append_fixture_patch(service, _experiment_patch())
    project_id = app.state.default_project_id
    assert project_id is not None
    chat_id = f"settled-revision-{continuation}-chat"
    session_id = f"settled-revision-{continuation}-session"

    def create_writer(prompt: str, _workspace: Path) -> None:
        _created_slot(prompt).joinpath("settled-revision.html").write_text(
            "<html><body>before revision</body></html>",
            encoding="utf-8",
        )

    create_request = _request(
        chat_id,
        "Create the view to revise.",
        result_view={"action": "create"},
    )
    create_operation = f"settled-revision-{continuation}-create"
    create_execution = _execution(
        store,
        project_id,
        create_operation,
        create_request,
    )
    await _events(
        stream_work_run(
            service,
            _ViewLauncher(session_id, create_writer),
            create_request,
            data_dir,
            execution=create_execution,
        )
    )
    created = store.list_result_views(project_id, chat_id=chat_id)[0]
    store.complete_agent_task(create_operation, applied_revision=None, result={})

    revision_request = _request(
        chat_id,
        "Revise the view before downstream recovery.",
        result_view={"action": "revise", "view_id": created.view_id},
        session_id=session_id,
    )
    revision_operation = f"settled-revision-{continuation}"
    revision_execution = _execution(
        store,
        project_id,
        revision_operation,
        revision_request,
        stage_root=created.stage_root,
    )

    def revision_writer(prompt: str, _workspace: Path) -> None:
        _revised_path(prompt).write_text(
            "<html><body>accepted revision</body></html>",
            encoding="utf-8",
        )

    await _events(
        stream_work_run(
            service,
            _ViewLauncher(session_id, revision_writer),
            revision_request,
            data_dir,
            execution=revision_execution,
        )
    )
    revised = store.result_view_for_diagnostics(created.view_id)
    assert revised is not None
    assert revised.latest_operation_id == revision_operation
    store.fail_agent_task(
        revision_operation,
        "Downstream task settlement was interrupted after accepting the revision.",
        status="interrupted" if continuation == "resume" else "failed",
    )
    target = Path(revised.stage_root) / "views" / revised.view_id / revised.source_name
    revised_bytes = target.read_bytes()

    recovery_operation = f"settled-revision-{continuation}-recovery"
    recovery_execution = _execution(
        store,
        project_id,
        recovery_operation,
        revision_request,
        stage_root=revised.stage_root,
        parent_operation_id=revision_operation,
        continuation=continuation,
    )

    def recovery_writer(prompt: str, _workspace: Path) -> None:
        assert "result-view authoring contract" not in _launch_contract_text(prompt)

    launcher = _ViewLauncher(session_id, recovery_writer)
    with monkeypatch.context() as recovery_patch:
        if continuation == "resume":
            # Isolate result-view settlement from the independent prompt-baseline replay seam.
            recovery_patch.setattr(
                "rcp.runs.tasks.work._commit_chat_prompt_state",
                lambda *_args, **_kwargs: None,
            )
        events = await _events(
            stream_work_run(
                service,
                launcher,
                revision_request,
                data_dir,
                execution=recovery_execution,
            )
        )

    assert not [event for event in events if event.event == "error"]
    assert any(event.event == "answer" for event in events)
    assert len(launcher.prompts) == 1
    assert target.read_bytes() == revised_bytes
    unchanged = store.result_view_for_diagnostics(created.view_id)
    assert unchanged is not None
    assert unchanged.latest_operation_id == revision_operation
    assert unchanged.content_sha256 == revised.content_sha256
    assert _receipts(store, recovery_operation, "result_view_revised") == []


@pytest.mark.parametrize("continuation", ["resume", "retry"])
def test_unbound_create_recovery_creates_its_missing_deterministic_slot(
    tmp_path: Path,
    continuation: str,
) -> None:
    from rcp.runs.tasks.result_views import (
        _prepare_result_view_create_slot,
        _prepare_result_view_turn,
    )

    store = AppStore(tmp_path / "rcp.sqlite3")
    owner = store.local_owner
    assert owner is not None
    store.rename_space_user(owner.user_id, "Result view owner")
    project_id = "result-view-slot-recovery-project"
    session_id = f"result-view-slot-{continuation}-session"
    request = _request(
        f"result-view-slot-{continuation}-chat",
        "Create the view after setup recovery.",
        result_view={"action": "create"},
        session_id=session_id,
    )
    stage = tmp_path / f"result-view-slot-{continuation}-stage"
    stage.mkdir()
    original_operation = f"result-view-slot-{continuation}-original"
    _execution(
        store,
        project_id,
        original_operation,
        request,
        stage_root=str(stage),
    )
    store.fail_agent_task(
        original_operation,
        "The task stopped before preparing its result-view slot.",
        status="interrupted",
    )
    recovery_execution = _execution(
        store,
        project_id,
        f"result-view-slot-{continuation}-recovery",
        request,
        stage_root=str(stage),
        parent_operation_id=original_operation,
        continuation=continuation,
    )

    prepared = _prepare_result_view_turn(
        request,
        recovery_execution,
        stage,
        None,
        focused_node={"id": _EXPERIMENT_ID, "type": "experiment"},
        logical_operation_id=original_operation,
        revision_preflight=None,
    )

    assert prepared is not None
    assert prepared.action == "create"
    assert Path(prepared.prompt_path).is_dir()
    assert Path(prepared.prompt_path).parent == stage / "views"

    outside = tmp_path / f"result-view-slot-{continuation}-outside"
    outside.mkdir()
    unsafe_view_id = "f" * 24
    (stage / "views" / unsafe_view_id).symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="slot is unsafe"):
        _prepare_result_view_create_slot(
            stage,
            None,
            unsafe_view_id,
            recovering=True,
        )


def test_result_view_prompt_is_short_private_and_preserves_human_bytes() -> None:
    from rcp.agents.prompts import PromptFactory

    message = "/show  keep  spacing\nand punctuation?!"
    prompt = PromptFactory.work_turn_prompt(
        artifact_path="/stage/turns/op/artifacts",
        human_message=message,
        result_view_action="create",
        result_view_path="/stage/views/0123456789abcdef01234567",
    )

    assert prompt.count(message) == 1
    assert len(prompt.splitlines()) < 30
    assert "/stage/views/0123456789abcdef01234567" in prompt
    assert "/stage/turns/op/artifacts" in prompt
    assert "independent of the turn artifact directory" in prompt
    assert "{type:'rcp-result-view-gesture',version:1" in prompt
    assert "The page may omit gestures" in prompt


@pytest.mark.asyncio
async def test_retention_bookkeeping_failure_warns_without_aborting_work_turn(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    app = create_app(str(manifest.path), data_dir=data_dir)
    service = app.state.service
    store = app.state.background_tasks.store
    append_fixture_patch(service, seed_patch())
    append_fixture_patch(service, _experiment_patch())
    project_id = app.state.default_project_id
    assert project_id is not None
    request = _request(
        "result-view-retention-warning-chat",
        "Continue the ordinary Work conversation.",
        result_view=None,
    )
    execution = _execution(store, project_id, "retention-warning-work", request)

    def fail_refresh(*_args, **_kwargs):
        raise RuntimeError("simulated expiry bookkeeping failure")

    monkeypatch.setattr(store, "refresh_result_view_expiry", fail_refresh)
    events = await _events(
        stream_work_run(
            service,
            _ViewLauncher("retention-warning-session"),
            request,
            data_dir,
            execution=execution,
        )
    )

    assert not [event for event in events if event.event == "error"]
    assert events[-1].event == "done"
    warnings = [
        event
        for event in store.agent_task_events("retention-warning-work")
        if event.level == "warning"
    ]
    assert len(warnings) == 1
    assert "Result-view retention could not be refreshed" in warnings[0].message
