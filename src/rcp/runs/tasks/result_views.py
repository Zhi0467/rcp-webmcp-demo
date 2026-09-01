from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Literal

from rcp.artifacts import validate_artifact_bytes, validate_result_view_id
from rcp.background import AgentTaskExecution
from rcp.limits import CHAT_ARTIFACT_MAX_FILE_BYTES, RUN_STAGE_RETENTION_DAYS
from rcp.service import RunRequest
from rcp.storage import ResultViewRecord
from rcp.transport.run_stage import RemoteRunStage
from rcp.transport.state import StateUnavailable


@dataclass(frozen=True)
class ResultViewSnapshot:
    name: str
    size: int
    sha256: str
    data: bytes


def result_view_slot_path(stage: Path, view_id: str) -> Path:
    """Reconstruct the stable local path without consulting a turn artifact scope."""
    return stage / "views" / validate_result_view_id(view_id)


def prepare_local_result_view_slot(stage: Path, view_id: str, *, reuse: bool) -> Path:
    """Create or reopen one exact result-view slot and roll stage retention forward."""
    view_id = validate_result_view_id(view_id)
    root_fd = _open_stage_root(stage)
    fds = [root_fd]
    try:
        try:
            views_fd = os.open("views", _DIRECTORY_FLAGS, dir_fd=root_fd)
        except FileNotFoundError:
            with suppress(FileExistsError):
                os.mkdir("views", mode=0o700, dir_fd=root_fd)
            try:
                views_fd = os.open("views", _DIRECTORY_FLAGS, dir_fd=root_fd)
            except OSError as exc:
                raise ValueError("result view parent is unsafe") from exc
        except OSError as exc:
            raise ValueError("result view parent is unsafe") from exc
        fds.append(views_fd)
        if reuse:
            try:
                slot_fd = os.open(view_id, _DIRECTORY_FLAGS, dir_fd=views_fd)
            except FileNotFoundError as exc:
                raise FileNotFoundError(f"result view slot is absent: {view_id}") from exc
            except OSError as exc:
                raise ValueError(f"result view slot is unsafe: {view_id}") from exc
        else:
            try:
                os.mkdir(view_id, mode=0o700, dir_fd=views_fd)
            except FileExistsError as exc:
                raise FileExistsError(f"result view slot already exists: {view_id}") from exc
            except OSError as exc:
                raise ValueError(f"result view slot is unsafe: {view_id}") from exc
            slot_fd = os.open(view_id, _DIRECTORY_FLAGS, dir_fd=views_fd)
        fds.append(slot_fd)
        os.utime(root_fd, None)
    finally:
        for descriptor in reversed(fds):
            os.close(descriptor)
    return result_view_slot_path(stage, view_id)


def touch_local_conversation_stage(stage: Path) -> None:
    """Refresh a reused local conversation stage without changing its cwd."""
    root_fd = _open_stage_root(stage)
    try:
        os.utime(root_fd, None)
    finally:
        os.close(root_fd)


def touch_conversation_stage(
    local_stage: Path | None,
    remote_stage: RemoteRunStage | None,
) -> tuple[str, str]:
    """Touch the current exact stage and return its durable host/root binding."""
    _require_one_stage(local_stage, remote_stage)
    if remote_stage is not None:
        remote_stage.touch()
        assert remote_stage.root is not None
        return remote_stage.host, str(remote_stage.root)
    assert local_stage is not None
    touch_local_conversation_stage(local_stage)
    return "", str(local_stage)


def touch_saved_conversation_stages(
    stage_bindings: Iterable[tuple[str, str]],
    *,
    current_binding: tuple[str, str],
) -> None:
    """Touch every distinct saved stage except the already-touched current stage."""
    for stage_host, stage_root in sorted(set(stage_bindings) - {current_binding}):
        if stage_host:
            RemoteRunStage(stage_host).attach(stage_root).touch()
        else:
            touch_local_conversation_stage(Path(stage_root))


def list_local_result_view_files(stage: Path, view_id: str) -> list[tuple[str, int]]:
    """Inspect at most two direct entries, enough to prove the one-file contract."""
    fds, slot_fd = _open_local_result_view_slot(stage, view_id)
    try:
        return sorted((name, info.st_size) for name, info in _bounded_local_slot_files(slot_fd))
    finally:
        _close_descriptors(fds)


def read_local_result_view_bytes(
    stage: Path,
    view_id: str,
    name: str,
    *,
    max_bytes: int,
) -> bytes:
    """Read one bounded direct child without following any component symlink."""
    name = _plain_name(name)
    if max_bytes < 0:
        raise ValueError("result view byte limit must be non-negative")
    fds, slot_fd = _open_local_result_view_slot(stage, view_id)
    try:
        try:
            file_fd = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
                dir_fd=slot_fd,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"result view file is absent: {view_id}/{name}") from exc
        except OSError as exc:
            raise ValueError(f"result view file is unsafe: {view_id}/{name}") from exc
        fds.append(file_fd)
        info = os.fstat(file_fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"result view file is unsafe: {view_id}/{name}")
        if info.st_size > max_bytes:
            raise ValueError(f"result view file exceeds its byte limit: {view_id}/{name}")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(file_fd, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > max_bytes:
            raise ValueError(f"result view file exceeds its byte limit: {view_id}/{name}")
        return data
    finally:
        _close_descriptors(fds)


def prepare_result_view_slot(
    local_stage: Path | None,
    remote_stage: RemoteRunStage | None,
    view_id: str,
    *,
    reuse: bool,
) -> Path | PurePosixPath:
    """Prepare the stable slot on exactly one execution host."""
    _require_one_stage(local_stage, remote_stage)
    if remote_stage is not None:
        return remote_stage.prepare_result_view_slot(view_id, reuse=reuse)
    assert local_stage is not None
    return prepare_local_result_view_slot(local_stage, view_id, reuse=reuse)


def discover_result_view(
    local_stage: Path | None,
    remote_stage: RemoteRunStage | None,
    view_id: str,
    *,
    expected_name: str | None = None,
    max_bytes: int = CHAT_ARTIFACT_MAX_FILE_BYTES,
) -> ResultViewSnapshot:
    """Validate and snapshot the one self-contained HTML file in a view slot."""
    _require_one_stage(local_stage, remote_stage)
    files = (
        remote_stage.list_result_view_files(view_id)
        if remote_stage is not None
        else list_local_result_view_files(_required_local_stage(local_stage), view_id)
    )
    if len(files) != 1:
        raise ValueError("result view slot must contain exactly one direct regular HTML file")
    name, advertised_size = files[0]
    if expected_name is not None and name != expected_name:
        raise ValueError("result view revision must update its existing exact file")
    if len(name) > 255 or Path(name).suffix.casefold() != ".html":
        raise ValueError("result view must be one descriptively named .html file")
    if advertised_size > max_bytes:
        raise ValueError("result view file exceeds its byte limit")
    data = (
        remote_stage.read_result_view_bytes(view_id, name, max_bytes=max_bytes)
        if remote_stage is not None
        else read_local_result_view_bytes(
            _required_local_stage(local_stage),
            view_id,
            name,
            max_bytes=max_bytes,
        )
    )
    if validate_artifact_bytes(name, data) != "text/html":
        raise ValueError("result view must be HTML")
    return ResultViewSnapshot(
        name=name,
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        data=data,
    )


def require_result_view_changed(before: ResultViewSnapshot, after: ResultViewSnapshot) -> None:
    """Accept atomic replacement at the same path, but reject a no-op revision."""
    if after.name != before.name:
        raise ValueError("result view revision must update its existing exact file")
    if after.sha256 == before.sha256:
        raise ValueError("result view revision did not change the existing file")


_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


def _plain_name(name: str) -> str:
    if not name or name in {".", ".."} or "/" in name or "\x00" in name:
        raise ValueError("result view file name must be a plain base name")
    return name


def _open_stage_root(stage: Path) -> int:
    if not stage.is_absolute():
        raise ValueError("result view stage must be absolute")
    try:
        return os.open(stage, _DIRECTORY_FLAGS)
    except OSError as exc:
        raise StateUnavailable(f"conversation stage is unavailable: {stage}") from exc


def _open_local_result_view_slot(stage: Path, view_id: str) -> tuple[list[int], int]:
    view_id = validate_result_view_id(view_id)
    root_fd = _open_stage_root(stage)
    fds = [root_fd]
    try:
        try:
            views_fd = os.open("views", _DIRECTORY_FLAGS, dir_fd=root_fd)
            fds.append(views_fd)
            slot_fd = os.open(view_id, _DIRECTORY_FLAGS, dir_fd=views_fd)
            fds.append(slot_fd)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"result view slot is absent: {view_id}") from exc
        except OSError as exc:
            raise ValueError(f"result view slot is unsafe: {view_id}") from exc
        return fds, slot_fd
    except BaseException:
        _close_descriptors(fds)
        raise


def _close_descriptors(descriptors: list[int]) -> None:
    for descriptor in reversed(descriptors):
        os.close(descriptor)


def _bounded_local_slot_files(slot_fd: int) -> list[tuple[str, os.stat_result]]:
    """Inspect no more than the two entries needed to disprove exact-one."""
    files: list[tuple[str, os.stat_result]] = []
    with os.scandir(slot_fd) as entries:
        for entry in entries:
            try:
                info = os.stat(entry.name, dir_fd=slot_fd, follow_symlinks=False)
            except OSError as exc:
                raise ValueError("result view slot contains an unsafe entry") from exc
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("result view slot contains an unsafe entry")
            files.append((entry.name, info))
            if len(files) == 2:
                break
    return files


def _require_one_stage(local: Path | None, remote: RemoteRunStage | None) -> None:
    if (local is None) == (remote is None):
        raise ValueError("exactly one result view stage must be selected")


def _required_local_stage(stage: Path | None) -> Path:
    if stage is None:
        raise ValueError("local result view stage is missing")
    return stage


@dataclass(frozen=True)
class _PreparedResultView:
    action: Literal["create", "revise"]
    view_id: str
    prompt_path: str
    origin_operation_id: str | None = None
    record: ResultViewRecord | None = None
    before: ResultViewSnapshot | None = None


def _result_view_expiry(now: datetime) -> str:
    return (now + timedelta(days=RUN_STAGE_RETENTION_DAYS)).isoformat()


def _result_view_task(execution: AgentTaskExecution | None):
    if execution is None:
        raise ValueError("A result view requires a durable RCP Work task.")
    task = execution.store.agent_task(execution.operation_id)
    if task is None or task.kind != "node_chat" or not task.project_id:
        raise ValueError("A result view requires a durable node conversation task.")
    return task


def _preflight_result_view_revision(
    request: RunRequest,
    execution: AgentTaskExecution | None,
) -> tuple[ResultViewRecord, ResultViewSnapshot] | None:
    """Load the stored bytes and require their durable binding before opening the stage."""

    result_view = request.result_view
    if result_view is None or result_view.action != "revise":
        return None
    task = _result_view_task(execution)
    assert execution is not None
    if not execution.stage_root:
        raise ValueError(
            "The result view revision has no inherited conversation stage; it cannot be redrawn "
            "from a fresh session."
        )
    record = execution.store.result_view(result_view.view_id)
    if record is None:
        raise ValueError("The result view is missing or expired and cannot be revised.")
    if record.kept_filename is not None:
        raise ValueError("A kept result view is immutable and cannot be revised.")
    expected_binding = {
        "project": (record.project_id, task.project_id),
        "Experiment": (record.experiment_id, request.node_id or ""),
        "conversation": (record.chat_id, request.chat_id or ""),
        "provider": (record.provider, request.provider or ""),
        "model": (record.model, request.model or ""),
        "reasoning": (record.reasoning, request.reasoning or ""),
        "execution machine": (record.run_on, request.run_on or ""),
        "native session": (record.native_session_id, request.session_id or ""),
        "stage host": (record.stage_host, execution.stage_host or ""),
        "stage root": (record.stage_root, execution.stage_root),
    }
    mismatched = [label for label, (saved, current) in expected_binding.items() if saved != current]
    if mismatched:
        raise ValueError(
            "The result view cannot be revised because its saved "
            + ", ".join(mismatched)
            + " binding does not match this turn."
        )
    stored = execution.store.result_view_bytes(
        record.view_id,
        expected_content_sha256=record.content_sha256,
    )
    return (
        record,
        ResultViewSnapshot(
            name=record.source_name,
            size=record.size_bytes,
            sha256=record.content_sha256,
            data=stored,
        ),
    )


def _roll_result_view_retention(
    request: RunRequest,
    execution: AgentTaskExecution | None,
    local_stage: Path | None,
    remote_stage: RemoteRunStage | None,
) -> None:
    """Keep one reused Work conversation and its unkept cards on the same rolling clock."""

    if request.trigger != "human" or request.patch_kind != "work":
        return
    current_binding = touch_conversation_stage(local_stage, remote_stage)
    if execution is None or not request.chat_id:
        return
    task = execution.store.agent_task(execution.operation_id)
    if task is None or not task.project_id:
        return
    try:
        now = datetime.fromisoformat(execution.store.now()).astimezone(UTC)
        views = execution.store.list_result_views(
            task.project_id,
            chat_id=request.chat_id,
            as_of=now,
        )
        touch_saved_conversation_stages(
            ((view.stage_host, view.stage_root) for view in views if view.kept_filename is None),
            current_binding=current_binding,
        )
        execution.store.refresh_result_view_expiry(
            task.project_id,
            request.chat_id,
            expires_at=_result_view_expiry(now),
            as_of=now,
        )
    except Exception as exc:
        with suppress(Exception):
            execution.store.record_agent_task_event(
                execution.operation_id,
                f"Result-view retention could not be refreshed: {exc}",
                level="warning",
            )


def _result_view_action_was_settled_by_ancestor(
    request: RunRequest,
    execution: AgentTaskExecution,
    record: ResultViewRecord,
) -> bool:
    """Recognize only this recovery lineage's exact already-committed view action."""

    if execution.continuation not in {"resume", "retry", "handoff"}:
        return False
    result_view = request.result_view
    if result_view is None:
        return False
    current = _result_view_task(execution)
    ancestor_id = current.parent_operation_id
    seen = {current.operation_id}
    while ancestor_id is not None:
        if ancestor_id in seen:
            raise ValueError("The result view recovery lineage contains a cycle.")
        seen.add(ancestor_id)
        ancestor = execution.store.agent_task(ancestor_id)
        if ancestor is None:
            raise ValueError("The result view recovery lineage lost its parent task.")
        if ancestor.project_id != current.project_id or ancestor.kind != current.kind:
            raise ValueError("The result view recovery lineage crossed a task boundary.")
        if ancestor.operation_id == record.latest_operation_id:
            try:
                ancestor_request = RunRequest.model_validate(ancestor.request)
            except ValueError as exc:
                raise ValueError(
                    "The result view recovery lineage has an invalid request."
                ) from exc
            ancestor_view = ancestor_request.result_view
            same_view = bool(
                ancestor_view is not None
                and ancestor_view.action == result_view.action
                and (result_view.action == "create" or ancestor_view.view_id == record.view_id)
            )
            ancestor_matches = bool(
                same_view
                and ancestor.project_id == record.project_id
                and ancestor.kind == "node_chat"
                and ancestor_request.trigger == "human"
                and ancestor_request.patch_kind == "work"
                and ancestor_request.mode == "work"
                and ancestor_request.chat_scope == "node"
                and ancestor_request.node_id == record.experiment_id == request.node_id
                and ancestor_request.chat_id == record.chat_id == request.chat_id
                and ancestor_request.provider in {None, record.provider}
                and ancestor_request.model in {None, "", record.model}
                and ancestor_request.reasoning in {None, record.reasoning}
                and ancestor_request.run_on in {None, record.run_on}
                and ancestor.native_session_id == record.native_session_id
                and (ancestor.stage_host or "") == record.stage_host
                and ancestor.stage_root == record.stage_root
            )
            if not ancestor_matches:
                return False
            if execution.continuation == "handoff":
                return result_view.action == "create"
            return bool(
                record.provider == request.provider
                and record.model == request.model
                and record.reasoning == request.reasoning
                and record.run_on == request.run_on
                and request.session_id == record.native_session_id
                and (execution.stage_host or "") == record.stage_host
                and execution.stage_root == record.stage_root
            )
        ancestor_id = ancestor.parent_operation_id
    return False


def _prepare_result_view_create_slot(
    local_stage: Path | None,
    remote_stage: RemoteRunStage | None,
    view_id: str,
    *,
    recovering: bool,
) -> Path | PurePosixPath:
    """Reuse a recovery slot, creating it only when its exact path is genuinely absent."""

    if not recovering:
        return prepare_result_view_slot(local_stage, remote_stage, view_id, reuse=False)
    try:
        return prepare_result_view_slot(local_stage, remote_stage, view_id, reuse=True)
    except FileNotFoundError:
        return prepare_result_view_slot(local_stage, remote_stage, view_id, reuse=False)


def _prepare_result_view_turn(
    request: RunRequest,
    execution: AgentTaskExecution | None,
    local_stage: Path | None,
    remote_stage: RemoteRunStage | None,
    *,
    focused_node: dict[str, object] | None,
    logical_operation_id: str,
    revision_preflight: tuple[ResultViewRecord, ResultViewSnapshot] | None,
) -> _PreparedResultView | None:
    result_view = request.result_view
    if result_view is None:
        return None
    if (
        request.trigger != "human"
        or request.patch_kind != "work"
        or request.mode != "work"
        or request.chat_scope != "node"
        or execution is None
        or (
            execution.continuation not in {"fresh", "resume"}
            and not (
                execution.continuation == "retry" and result_view.action in {"create", "revise"}
            )
            and not (execution.continuation == "handoff" and result_view.action == "create")
        )
    ):
        raise ValueError("A result view is available only on an ordinary human node Work turn.")
    if (
        focused_node is None
        or focused_node.get("type") != "experiment"
        or focused_node.get("id") != request.node_id
    ):
        raise ValueError("A result view must be scoped to an existing Experiment.")
    if not request.chat_id or not request.node_id:
        raise ValueError("A result view requires its exact Experiment conversation.")

    task = _result_view_task(execution)
    if result_view.action == "create":
        origin_operation_id = logical_operation_id
        if execution.continuation in {"resume", "retry", "handoff"}:
            current = execution.store.agent_task(execution.operation_id)
            seen: set[str] = set()
            while current is not None and current.parent_operation_id is not None:
                if current.operation_id in seen:
                    raise ValueError("The result view create lineage contains a cycle.")
                seen.add(current.operation_id)
                parent = execution.store.agent_task(current.parent_operation_id)
                if (
                    parent is None
                    or parent.project_id != current.project_id
                    or parent.kind != current.kind
                ):
                    raise ValueError("The result view create recovery lost its task lineage.")
                parent_request = RunRequest.model_validate(parent.request)
                if (
                    parent_request.result_view is None
                    or parent_request.result_view.action != "create"
                    or parent_request.chat_id != request.chat_id
                    or parent_request.node_id != request.node_id
                ):
                    raise ValueError("The result view create recovery crossed a task boundary.")
                origin_operation_id = parent.operation_id
                current = parent
        view_id = hashlib.sha256(f"result-view\0{origin_operation_id}".encode()).hexdigest()[:24]
        existing = execution.store.result_view_for_diagnostics(view_id)
        if existing is not None:
            if _result_view_action_was_settled_by_ancestor(request, execution, existing):
                return None
            expected_binding = {
                "project": (existing.project_id, task.project_id),
                "Experiment": (existing.experiment_id, request.node_id),
                "conversation": (existing.chat_id, request.chat_id),
                "origin": (existing.origin_operation_id, origin_operation_id),
                "provider": (existing.provider, request.provider or ""),
                "model": (existing.model, request.model or ""),
                "reasoning": (existing.reasoning, request.reasoning or ""),
                "execution machine": (existing.run_on, request.run_on or ""),
                "native session": (existing.native_session_id, request.session_id or ""),
                "stage host": (existing.stage_host, execution.stage_host or ""),
                "stage root": (existing.stage_root, execution.stage_root or ""),
            }
            mismatched = [
                label for label, (saved, current) in expected_binding.items() if saved != current
            ]
            if mismatched:
                raise ValueError(
                    "The created result view has a mismatched "
                    + ", ".join(mismatched)
                    + " binding."
                )
            raise ValueError("This result view was already created and cannot be created again.")
        slot = _prepare_result_view_create_slot(
            local_stage,
            remote_stage,
            view_id,
            recovering=execution.continuation in {"resume", "retry", "handoff"},
        )
        return _PreparedResultView(
            action="create",
            view_id=view_id,
            prompt_path=str(slot),
            origin_operation_id=origin_operation_id,
        )

    if revision_preflight is None:
        raise ValueError("The result view revision lost its durable preflight binding.")
    record, before = revision_preflight
    if record.view_id != result_view.view_id:
        raise ValueError("The result view revision lost its durable preflight binding.")
    if _result_view_action_was_settled_by_ancestor(request, execution, record):
        return None
    slot = prepare_result_view_slot(local_stage, remote_stage, record.view_id, reuse=True)
    return _PreparedResultView(
        action="revise",
        view_id=record.view_id,
        prompt_path=str(slot / record.source_name),
        record=record,
        before=before,
    )


def _record_result_view_rejection(
    execution: AgentTaskExecution,
    prepared: _PreparedResultView,
    problem: str,
) -> None:
    payload: dict[str, object] = {
        "action": prepared.action,
        "view_id": prepared.view_id,
        "problem": problem[:1600],
    }
    with suppress(Exception):
        execution.store.record_agent_task_receipt(
            execution.operation_id,
            "result_view_rejected",
            payload,
            tier="diagnostic",
        )
    detail = f"Result view was not updated: {problem}"
    with suppress(Exception):
        execution.store.record_agent_task_event(
            execution.operation_id,
            detail,
            level="warning",
        )


def _finalize_result_view_turn(
    request: RunRequest,
    execution: AgentTaskExecution | None,
    prepared: _PreparedResultView | None,
    local_stage: Path | None,
    remote_stage: RemoteRunStage | None,
    *,
    native_session_id: str | None,
) -> None:
    """Validate and bind a view without coupling its outcome to answer or graph delivery."""

    if prepared is None:
        return
    assert execution is not None
    try:
        snapshot = discover_result_view(
            local_stage,
            remote_stage,
            prepared.view_id,
            expected_name=(prepared.record.source_name if prepared.record is not None else None),
        )
        now = datetime.fromisoformat(execution.store.now()).astimezone(UTC)
        expires_at = _result_view_expiry(now)
        if prepared.action == "create":
            if not native_session_id:
                raise ValueError("the provider returned no native session for later revision")
            task = _result_view_task(execution)
            assert request.node_id is not None
            assert request.chat_id is not None
            record = execution.store.create_result_view(
                ResultViewRecord(
                    view_id=prepared.view_id,
                    project_id=task.project_id,
                    experiment_id=request.node_id,
                    chat_id=request.chat_id,
                    origin_operation_id=prepared.origin_operation_id or task.operation_id,
                    latest_operation_id=execution.operation_id,
                    provider=request.provider or "",
                    model=request.model or "",
                    reasoning=request.reasoning or "",
                    run_on=request.run_on or "",
                    native_session_id=native_session_id,
                    stage_host=execution.stage_host or "",
                    stage_root=execution.stage_root or "",
                    source_name=snapshot.name,
                    content_sha256=snapshot.sha256,
                    size_bytes=snapshot.size,
                    created_at=now.isoformat(),
                    updated_at=now.isoformat(),
                    expires_at=expires_at,
                ),
                html=snapshot.data,
            )
            category = "result_view_created"
        else:
            assert prepared.record is not None
            assert prepared.before is not None
            if native_session_id != prepared.record.native_session_id:
                raise ValueError(
                    "the provider did not resume the result view's exact native session"
                )
            require_result_view_changed(prepared.before, snapshot)
            record = execution.store.revise_result_view(
                prepared.view_id,
                expected_content_sha256=prepared.before.sha256,
                latest_operation_id=execution.operation_id,
                content_sha256=snapshot.sha256,
                size_bytes=snapshot.size,
                html=snapshot.data,
                updated_at=now.isoformat(),
                expires_at=expires_at,
            )
            category = "result_view_revised"
    except Exception as exc:
        _record_result_view_rejection(execution, prepared, str(exc))
        return

    payload = {
        "view_id": record.view_id,
        "experiment_id": record.experiment_id,
        "chat_id": record.chat_id,
        "source_name": record.source_name,
        "content_sha256": record.content_sha256,
        "size_bytes": record.size_bytes,
        "updated_at": record.updated_at,
        "expires_at": record.expires_at,
        "native_session_id": record.native_session_id,
        "stage_host": record.stage_host,
        "stage_root": record.stage_root,
    }
    with suppress(Exception):
        execution.store.record_agent_task_receipt(
            execution.operation_id,
            category,
            payload,
        )
    with suppress(Exception):
        execution.store.record_agent_task_event(
            execution.operation_id,
            "Result view created." if prepared.action == "create" else "Result view revised.",
        )
