from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
import tempfile
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import AbstractContextManager, aclosing
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Protocol, TypeVar

from pydantic import BaseModel

from rcp.agents import AgentEvent, AgentLauncher, ChatContext, PromptFactory, RunContext
from rcp.agents.invocation_broker import ProviderInvocationGate
from rcp.agents.write_scope import ProjectWriteScope
from rcp.config import AgentSurfaceConfig
from rcp.core.models import GraphState, Patch
from rcp.core.operations import CreateEdgesOperation, CreateNodesOperation
from rcp.limits import RUN_STAGE_RETENTION_DAYS
from rcp.providers import AgentCapability, project_write_enforcement_mode
from rcp.service import CoachRequest, ProjectService, RunRequest
from rcp.transport import RemoteRunStage, StateUnavailable

if TYPE_CHECKING:
    from rcp.background import AgentTaskExecution

_STAGE_RETENTION_SECONDS = RUN_STAGE_RETENTION_DAYS * 24 * 3600
_MAX_PATCH_CANDIDATES = 8
_STATE_PATH_FIELDS = (
    "graph_path",
    "research_md_path",
    "introduction_path",
    "glossary_path",
    "coverage_path",
)
_NON_PROMPT_CONTRACT_ROLES = {
    "chat_prompt_state",
    "experiment_episode_context_candidate",
}
_RequestT = TypeVar("_RequestT", bound=BaseModel)


class _RecoveryStageStore(Protocol):
    def connection(self) -> AbstractContextManager[sqlite3.Connection]: ...


@dataclass(frozen=True)
class LocalRecoveryStage:
    """One retained local task stage that startup recovery may still consume."""

    root: Path
    owner_refs: tuple[str, ...]


@dataclass
class _RecoveryStageBinding:
    owner_refs: set[str]
    required: bool = False


def checkpoint_local_recovery_stages(
    store: _RecoveryStageStore,
    data_dir: Path,
) -> tuple[LocalRecoveryStage, ...]:
    """Inventory exact local run stages without treating remote stages as local paths.

    The durable lifecycle ledgers own stage identity. Unreferenced retention debris is
    not a recovery input. A stage still needed by an active lifecycle must exist; an
    already-retired historical reference is copied when present and ignored after the
    normal retention sweep has removed it.
    """

    if not data_dir.is_absolute() or ".." in data_dir.parts:
        raise ValueError("recovery-stage inventory requires one absolute data directory")
    stage_root = data_dir / "run-stage"
    bindings: dict[Path, _RecoveryStageBinding] = {}
    stage_owners = (
        (
            "graph_runs",
            """
            SELECT operation_id AS owner, COALESCE(stage_host, '') AS host,
                   stage_root AS root,
                   status IN ('queued', 'running', 'pausing') AS required
            FROM graph_runs
            WHERE stage_root IS NOT NULL AND stage_root != ''
            """,
        ),
        (
            "experiment_episode_state",
            """
            SELECT state.episode_id AS owner, COALESCE(state.stage_host, '') AS host,
                   state.stage_root AS root,
                   episode.status IN ('queued', 'running', 'stopping', 'wrapping_up')
                       AS required
            FROM experiment_episode_state AS state
            JOIN episodes AS episode ON episode.episode_id = state.episode_id
            WHERE state.stage_root IS NOT NULL AND state.stage_root != ''
            """,
        ),
        (
            "episode_wrapups",
            """
            SELECT episode_id AS owner, COALESCE(stage_host, '') AS host,
                   stage_root AS root,
                   state IN ('pending', 'running') AS required
            FROM episode_wrapups
            WHERE stage_root IS NOT NULL AND stage_root != ''
            """,
        ),
        (
            "result_views",
            """
            SELECT view_id AS owner, COALESCE(stage_host, '') AS host,
                   stage_root AS root, 0 AS required
            FROM result_views
            WHERE stage_root IS NOT NULL AND stage_root != ''
            """,
        ),
    )
    with store.connection() as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        for table, query in stage_owners:
            if table not in tables:
                continue
            if table == "experiment_episode_state" and "episodes" not in tables:
                continue
            rows = connection.execute(query).fetchall()
            for row in rows:
                if row["host"]:
                    continue
                stage_value = str(row["root"])
                candidate = Path(stage_value)
                if (
                    not candidate.is_absolute()
                    or ".." in candidate.parts
                    or str(candidate) != stage_value
                    or candidate.parent != stage_root
                ):
                    raise ValueError("a saved local run stage escaped its exact data boundary")
                binding = bindings.setdefault(candidate, _RecoveryStageBinding(owner_refs=set()))
                binding.owner_refs.add(f"{table}:{row['owner']}")
                binding.required = binding.required or bool(row["required"])

    if not bindings:
        return ()

    inventory: list[LocalRecoveryStage] = []
    for root, binding in sorted(bindings.items(), key=lambda item: str(item[0])):
        try:
            metadata = root.lstat()
        except FileNotFoundError as exc:
            if binding.required:
                raise ValueError("a recovery-critical local run stage is unavailable") from exc
            continue
        except OSError as exc:
            raise ValueError("a saved local run stage cannot be inspected") from exc
        if not stat.S_ISDIR(metadata.st_mode) or root.is_symlink():
            raise ValueError("a saved local run stage is not an ordinary directory")
        inventory.append(
            LocalRecoveryStage(
                root=root,
                owner_refs=tuple(sorted(binding.owner_refs)),
            )
        )
    if not inventory:
        return ()
    try:
        root_metadata = stage_root.lstat()
    except OSError as exc:
        raise ValueError("the local run-stage root is unavailable") from exc
    if not stat.S_ISDIR(root_metadata.st_mode) or stage_root.is_symlink():
        raise ValueError("the local run-stage root is not an ordinary directory")
    return tuple(inventory)


class AgentOutputProblem(ValueError):
    """The agent finished but its patch file is missing or does not validate.

    These are the failures the recovery ladder can act on by talking to the same
    live session again. Agent-authored graph rejection follows the same ladder.
    """


def _looks_like_patch(text: str) -> bool:
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return False
    # Keep this deliberately looser than AgentPatch validation: malformed
    # semantic patches still need to reach the correction ladder and receive the
    # exact schema diagnostic. Historical canonical drafts also carry summary +
    # ops, so retained pre-migration work remains recoverable.
    return isinstance(value, dict) and "ops" in value and "summary" in value


def _existing_patch_digest(
    workspace: Path,
    remote_stage: RemoteRunStage | None,
) -> str | None:
    """Fingerprint a patch already sitting in the stage before a launch.

    A continuation runs in the stage its earlier attempt was given, and
    invariant 9 keeps that attempt's `patch.json` on disk. Without this
    fingerprint a provider that writes nothing at all has its predecessor's
    file collected as this launch's deliverable.
    """
    try:
        text, _ = _collect_patch_text(workspace, remote_stage)
    except (AgentOutputProblem, OSError, StateUnavailable):
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _retry_deliverable_is_unchanged(
    execution: AgentTaskExecution | None,
    *,
    filename: str,
    predecessor_digest: str | None,
    current_text: str | None,
    allow_unchanged: bool = False,
) -> bool:
    """Record whether a reused Retry stage still contains its predecessor's output."""

    if execution is None or execution.continuation != "retry":
        return False
    current_digest = (
        hashlib.sha256(current_text.encode("utf-8")).hexdigest()
        if current_text is not None
        else None
    )
    unchanged = predecessor_digest is not None and current_digest == predecessor_digest
    execution.store.record_agent_task_receipt(
        execution.operation_id,
        "retry_deliverable_comparison",
        {
            "filename": filename,
            "predecessor_sha256": predecessor_digest,
            "retry_sha256": current_digest,
            "unchanged": unchanged,
            "consumed": current_text is not None and (not unchanged or allow_unchanged),
        },
        tier="diagnostic",
    )
    return unchanged and not allow_unchanged


def _collect_patch_text(
    workspace: Path,
    remote_stage: RemoteRunStage | None,
) -> tuple[str, str]:
    """Recover the patch the agent wrote, whatever it chose to call the file.

    Rung 1 of the recovery ladder: a filename mismatch used to discard a whole
    run's work, so the entire scratch folder is searched rather than one path.
    """
    if remote_stage is not None:
        names = remote_stage.list_workspace_files()
        reader = remote_stage.read_text

        def read(name: str) -> str:
            return reader(remote_stage.workspace / name)
    else:
        names = sorted(item.name for item in workspace.iterdir() if item.is_file())

        def read(name: str) -> str:
            return (workspace / name).read_text(encoding="utf-8")

    ordered = sorted(
        (name for name in names if name.casefold().endswith(".json")),
        key=lambda name: (
            name != "patch.json",
            name.casefold() != "patch.json",
            name,
        ),
    )
    if not ordered:
        raise AgentOutputProblem(
            "The agent finished without writing any JSON file to its scratch folder."
        )
    matches: list[tuple[str, str]] = []
    for name in ordered[:_MAX_PATCH_CANDIDATES]:
        try:
            text = read(name)
        except (OSError, ValueError):
            continue
        if name == "patch.json" and _looks_like_patch(text):
            return text, name
        if _looks_like_patch(text):
            matches.append((text, name))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise AgentOutputProblem(
            "The agent left more than one patch-shaped JSON file in its scratch folder: "
            + ", ".join(name for _, name in matches)
            + ". Write exactly one, named patch.json."
        )
    raise AgentOutputProblem(
        "The agent finished without writing a patch object to patch.json. "
        f"The scratch folder holds: {', '.join(ordered) or 'nothing'}."
    )


def _sweep_stale_stages(root: Path, *, now: float) -> None:
    """Age out retained scratch folders. Failed runs keep theirs until then."""
    if not root.is_dir():
        return
    for candidate in root.iterdir():
        if not candidate.is_dir():
            continue
        try:
            age = now - candidate.stat().st_mtime
        except OSError:
            continue
        if age > _STAGE_RETENTION_SECONDS:
            try:
                _remove_local_tree(candidate, root)
            except (OSError, ValueError):
                # Retention is best effort; a live run must not fail because an
                # unrelated expired stage could not be reclaimed.
                continue


def _remove_local_tree(target: Path, boundary: Path) -> None:
    """Remove one exact tree beneath ``boundary``, including read-only copies."""
    if target.parent != boundary:
        raise ValueError("local cleanup target is outside its exact stage boundary")
    if not os.path.lexists(target):
        return
    if target.is_symlink() or not target.is_dir():
        target.unlink()
    else:
        _make_local_tree_writable(target)
        shutil.rmtree(target)
    if os.path.lexists(target):
        raise OSError(f"local cleanup left {target} behind")


def _make_local_tree_writable(target: Path) -> None:
    if target.is_symlink():
        return
    target.chmod(0o700 if target.is_dir() else 0o600)
    if not target.is_dir():
        return
    for child in target.iterdir():
        _make_local_tree_writable(child)


def _swept_stage_root(data_dir: Path) -> Path:
    """The local scratch root, with expired folders reclaimed before it is used."""
    stage_root = data_dir / "run-stage"
    _sweep_stale_stages(stage_root, now=time.time())
    return stage_root


def _stage_task_input(
    local_stage: Path | None,
    remote_stage: RemoteRunStage | None,
    label: str,
    content: str,
) -> str:
    """Create one immutable task input and return its execution-host path."""
    if (local_stage is None) == (remote_stage is None):
        raise ValueError("exactly one task stage must be selected")
    safe_label = _safe_stage_name(label)
    if safe_label != label:
        raise ValueError("task input label contains unsupported characters")
    if remote_stage is not None:
        with tempfile.TemporaryDirectory(prefix="rcp-task-input-") as temporary:
            source = Path(temporary) / safe_label
            source.write_text(content, encoding="utf-8")
            source.chmod(0o400)
            return remote_stage.put_file(source, safe_label)

    assert local_stage is not None
    inputs = local_stage / "inputs"
    inputs.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = inputs / safe_label
    if target.exists():
        raise ValueError(f"immutable task input already exists: {safe_label}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{safe_label}.", dir=inputs)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o400)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return str(target)


def _stage_or_reuse_task_input(
    local_stage: Path | None,
    remote_stage: RemoteRunStage | None,
    label: str,
    content: str,
) -> str:
    """Stage content-addressed immutable input, reusing an identical existing file."""

    if (local_stage is None) == (remote_stage is None):
        raise ValueError("exactly one task stage must be selected")
    safe_label = _safe_stage_name(label)
    if safe_label != label:
        raise ValueError("task input label contains unsupported characters")
    if remote_stage is not None:
        try:
            existing = remote_stage.read_input_text(label)
        except ValueError:
            return _stage_task_input(local_stage, remote_stage, label, content)
        if existing != content:
            raise ValueError(f"immutable remote task input already differs: {label}")
        assert remote_stage.root is not None
        return str(remote_stage.root / "inputs" / label)

    assert local_stage is not None
    target = local_stage / "inputs" / label
    if not target.exists():
        return _stage_task_input(local_stage, remote_stage, label, content)
    if target.is_symlink() or not target.is_file() or target.read_text(encoding="utf-8") != content:
        raise ValueError(f"immutable task input already differs: {label}")
    return str(target)


def _task_token(execution: AgentTaskExecution | None) -> str:
    return _safe_stage_name(execution.operation_id if execution is not None else uuid.uuid4().hex)


def _parent_task_contract_path(
    execution: AgentTaskExecution,
    local_stage: Path | None,
    remote_stage: RemoteRunStage | None,
) -> str:
    record = execution.store.agent_task(execution.operation_id)
    if record is None or record.parent_operation_id is None:
        raise ValueError("The resumed operation has no original task contract.")

    stage_identity = (record.stage_host or "", record.stage_root)
    ancestor_id: str | None = record.parent_operation_id
    while ancestor_id is not None:
        ancestor = execution.store.agent_task(ancestor_id)
        if ancestor is None:
            break
        ancestor_stage = (ancestor.stage_host or "", ancestor.stage_root)
        same_stage = ancestor_stage == stage_identity
        legacy_stage = ancestor.stage_host is None and ancestor.stage_root is None
        if not same_stage and not legacy_stage:
            break

        receipts = execution.store.agent_task_receipts(ancestor_id)
        candidates = [
            receipt.payload.get("contract_path")
            for receipt in receipts
            if receipt.category == "agent_prompt"
        ]
        contract_path = next(
            (value for value in reversed(candidates) if isinstance(value, str) and value), None
        )
        if contract_path is not None:
            if remote_stage is not None:
                assert remote_stage.root is not None
                if PurePosixPath(contract_path).parent != remote_stage.root / "inputs":
                    raise ValueError(
                        "The resumed operation's task contract is outside its saved stage."
                    )
            else:
                assert local_stage is not None
                if Path(contract_path).resolve().parent != (local_stage / "inputs").resolve():
                    raise ValueError(
                        "The resumed operation's task contract is outside its saved stage."
                    )
            return contract_path

        # Pre-stage-provenance tasks remain recoverable through their retained, path-validated
        # prompt receipt above. Without that receipt there is no safe evidence that an older
        # durable contract belongs to this exact execution stage.
        if not same_stage:
            break

        durable_contracts = [
            contract
            for contract in execution.store.agent_task_contracts(ancestor_id)
            if contract.role not in _NON_PROMPT_CONTRACT_ROLES
        ]
        if durable_contracts:
            durable = durable_contracts[-1]
            digest = hashlib.sha256(durable.content.encode("utf-8")).hexdigest()
            if digest != durable.sha256:
                raise ValueError("The resumed operation's durable task contract is corrupt.")
            label = f"task-{_safe_stage_name(ancestor_id)}-recovered-{digest[:16]}.md"
            contract_path = _stage_or_reuse_task_input(
                local_stage,
                remote_stage,
                label,
                durable.content,
            )
            execution.store.record_agent_task_receipt(
                execution.operation_id,
                "original_contract_recovered",
                {
                    "ancestor_operation_id": ancestor_id,
                    "role": durable.role,
                    "sha256": digest,
                    "contract_path": contract_path,
                },
            )
            return contract_path

        ancestor_id = ancestor.parent_operation_id

    raise ValueError("The resumed operation has no recorded original task contract.")


def _stage_json_task_input(
    local_stage: Path | None,
    remote_stage: RemoteRunStage | None,
    label: str,
    value: object,
) -> str:
    return _stage_task_input(
        local_stage,
        remote_stage,
        label,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _stage_task_contract(
    local_stage: Path | None,
    remote_stage: RemoteRunStage | None,
    label: str,
    content: str,
    *,
    execution: AgentTaskExecution | None = None,
    role: str | None = None,
) -> tuple[str, str]:
    if execution is not None:
        execution.store.record_agent_task_contract(
            execution.operation_id,
            role or label,
            content,
            hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )
    contract_path = _stage_task_input(local_stage, remote_stage, label, content)
    return contract_path, PromptFactory.launch_prompt(contract_path)


def _pinned_to_profile(request: _RequestT, profile: AgentSurfaceConfig) -> _RequestT:
    """Pin the resolved launch configuration onto the request the run will use."""
    return request.model_copy(
        update={
            "provider": profile.provider,
            "model": profile.model,
            "reasoning": profile.reasoning,
            "run_on": profile.run_on,
        }
    )


def _record_agent_launch_receipt(
    execution: AgentTaskExecution | None,
    request: RunRequest | CoachRequest,
    *,
    prompt: str,
    contract_path: str,
    remote: bool,
    resumed: bool,
    write_scope: ProjectWriteScope | None = None,
    continuation: str | None = None,
    extra: dict[str, object],
) -> None:
    capability = extra.get("capability")
    scope_payload: dict[str, object] = {}
    if capability in {"work_auto", "orchestrate"}:
        if write_scope is None:
            raise ValueError(f"{capability} launch receipt requires a project write scope")
        if execution is not None:
            execution.bind_write_scope(write_scope)
        scope_payload = {
            "project_id": write_scope.project_id,
            "execution_machine": write_scope.execution_machine,
            "execution_host": write_scope.execution_host,
            "capability": write_scope.capability,
            "canonical_write_roots": write_scope.writable_roots,
            "canonical_repository_roots": write_scope.repository_roots,
            "protected_write_paths": write_scope.protected_write_paths,
            "write_scope_fingerprint": write_scope.fingerprint,
            "provider_enforcement_mode": project_write_enforcement_mode(request.provider),
            "canonical_state_boundary": "provider_enforced",
        }
    elif write_scope is not None:
        raise ValueError(f"{capability!r} launch cannot record a project write scope")
    if execution is None:
        return
    execution.store.record_agent_task_receipt(
        execution.operation_id,
        "agent_launch",
        {
            "provider": request.provider,
            "run_on": request.run_on,
            "remote": remote,
            "resumed": resumed,
            **({"continuation_cause": continuation} if continuation is not None else {}),
            **extra,
            **scope_payload,
        },
    )
    encoded = prompt.encode("utf-8")
    execution.store.record_agent_task_receipt(
        execution.operation_id,
        "agent_prompt",
        {
            "prompt": prompt,
            "contract_path": contract_path,
            "byte_length": len(encoded),
            "line_count": len(prompt.splitlines()),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "resumed": resumed,
            **({"continuation_cause": continuation} if continuation is not None else {}),
            **extra,
            **scope_payload,
        },
        tier="diagnostic",
    )


@dataclass
class _ProviderOutcome:
    """What one pass of a provider stream leaves behind for its caller.

    `answers` collects only the provider's labelled final assistant messages; a
    `message` is a trace and is never promoted into it. What an answer is worth
    is the caller's decision.
    """

    session_id: str | None = None
    completed: bool = False
    failed: bool = False
    paused: bool = False
    answers: list[str] = dataclasses.field(default_factory=list)
    trace_messages: list[str] = dataclasses.field(default_factory=list)
    exit_evidence: dict[str, object] | None = None
    exit_recorded: bool = False


async def _stream_agent_events(
    launcher: AgentLauncher,
    request: RunRequest,
    prompt: str,
    *,
    workspace: Path,
    session_id: str | None,
    read_dirs: list[Path],
    write_dirs: list[Path],
    write_scope: ProjectWriteScope | None,
    execution_host: str,
    execution: AgentTaskExecution | None,
    remote_stage: RemoteRunStage | None,
    capability: AgentCapability,
    outcome: _ProviderOutcome,
    binary: str | None,
    invocation_gate: ProviderInvocationGate | None = None,
    required_session_id: str | None = None,
) -> AsyncIterator[str]:
    """Run one provider pass, recording its outcome and forwarding wire events.

    Terminal and labelled events are withheld from the wire: the caller decides
    what a completed run, an answer, or a trace is worth in its own protocol.
    """
    if remote_stage is not None:
        try:
            await asyncio.to_thread(remote_stage.finalize_inputs)
        except (OSError, StateUnavailable, ValueError) as exc:
            outcome.failed = True
            yield _sse(AgentEvent(event="error", text=str(exc)))
            return
    async with aclosing(
        launcher.stream(
            request.provider,
            prompt,
            cwd=workspace,
            model=request.model,
            reasoning=request.reasoning,
            session_id=session_id,
            read_dirs=read_dirs,
            write_dirs=write_dirs,
            write_scope=write_scope,
            host=execution_host,
            control=execution.control if execution is not None else None,
            remote_pid_file=(
                str(remote_stage.root / "agent.pid")
                if execution is not None and remote_stage is not None and remote_stage.root
                else None
            ),
            invocation_gate=invocation_gate,
            capability=capability,
            binary=binary,
            runtime_id=(execution.runtime_id or None) if execution is not None else None,
        )
    ) as stream:
        async for event in stream:
            if event.event == "provider_exit":
                try:
                    evidence = json.loads(event.text)
                except (json.JSONDecodeError, TypeError, ValueError):
                    evidence = {"unparsed": event.text[:400]}
                outcome.exit_evidence = (
                    evidence if isinstance(evidence, dict) else {"unparsed": event.text[:400]}
                )
                _record_provider_exit(
                    execution,
                    outcome,
                    workspace=workspace,
                    remote_stage=remote_stage,
                )
                continue
            if event.event in {"runtime", "runtime_fallback"}:
                # Background consumes these. It durably checkpoints the runtime
                # before the launcher writes the prompt, and records why an
                # earlier candidate was passed over rather than showing it.
                yield _sse(event)
                continue
            if event.event == "paused":
                outcome.paused = True
            if event.event == "session" and event.session_id:
                if required_session_id is not None and event.session_id != required_session_id:
                    outcome.failed = True
                    if execution is not None:
                        execution.store.record_agent_task_receipt(
                            execution.operation_id,
                            "continuation_context_unavailable",
                            {
                                "reason": "native_session_mismatch",
                                "retry_required": True,
                            },
                        )
                    yield _sse(
                        AgentEvent(
                            event="error",
                            text=(
                                "The provider did not continue the exact saved native session. "
                                "This continuation was stopped before accepting any result."
                            ),
                        )
                    )
                    return
                outcome.session_id = event.session_id
                if execution_host and execution is None:
                    continue
            if event.event == "answer":
                outcome.answers.append(event.text)
                continue
            if event.event == "message":
                if event.text.strip() and len(outcome.trace_messages) < 16:
                    outcome.trace_messages.append(event.text.strip()[:16_000])
                continue
            if event.event == "error":
                outcome.failed = True
            if event.event == "done":
                outcome.completed = True
                continue
            yield _sse(event)


def _record_provider_exit(
    execution: AgentTaskExecution | None,
    outcome: _ProviderOutcome,
    *,
    workspace: Path,
    remote_stage: RemoteRunStage | None,
) -> None:
    if execution is None or outcome.exit_evidence is None or outcome.exit_recorded:
        return
    payload = dict(outcome.exit_evidence)
    try:
        if remote_stage is not None:
            patch_exists = "patch.json" in remote_stage.list_workspace_files()
        else:
            patch_exists = (workspace / "patch.json").is_file()
        payload["patch_json_exists"] = patch_exists
    except (OSError, StateUnavailable, ValueError) as exc:
        payload["patch_json_exists"] = None
        payload["patch_check_error"] = " ".join(str(exc).split())[:400]
    execution.store.record_agent_task_receipt(
        execution.operation_id,
        "provider_exit",
        payload,
        tier="diagnostic",
    )
    outcome.exit_recorded = True


def _record_patch_receipt(
    execution: AgentTaskExecution | None,
    patch: Patch,
    *,
    byte_length: int,
) -> None:
    if execution is None:
        return
    operation_counts: dict[str, int] = {}
    created_node_count = 0
    created_edge_count = 0
    for operation in patch.ops:
        operation_kind = operation.op
        operation_counts[operation_kind] = operation_counts.get(operation_kind, 0) + 1
        if isinstance(operation, CreateNodesOperation):
            created_node_count += len(operation.nodes)
        if isinstance(operation, CreateEdgesOperation):
            created_edge_count += len(operation.edges)
    execution.store.record_agent_task_receipt(
        execution.operation_id,
        "patch_parsed",
        {
            "byte_length": byte_length,
            "kind": patch.kind,
            "author": patch.author,
            "operation_count": len(patch.ops),
            "operation_counts": operation_counts,
            "created_node_count": created_node_count,
            "created_edge_count": created_edge_count,
            "processed_cursor_count": len(patch.processed_cursors),
            "change_summary_count": len(patch.change_summary),
        },
        tier="diagnostic",
    )


def _record_patch_applied_receipt(
    execution: AgentTaskExecution | None,
    state: GraphState,
) -> None:
    if execution is None:
        return
    execution.applied_revision = state.revision
    execution.applied_graph_state = state
    execution.store.record_agent_task_receipt(
        execution.operation_id,
        "patch_applied",
        {
            "revision": state.revision,
            "node_count": len(state.nodes),
            "edge_count": len(state.edges),
            "validation_message_count": len(state.validation_messages),
        },
    )


def _safe_stage_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not name:
        raise ValueError("run stage name is empty")
    return name


def _stage_context_paths(
    context: RunContext | ChatContext,
    service: ProjectService,
    stage: RemoteRunStage,
    execution_machine: str,
) -> dict[str, object]:
    """Repository and materialized-state pointers a remote agent can open."""

    repositories = []
    for repository in context.repositories:
        if repository.machine == execution_machine:
            repositories.append(repository.model_copy(update={"host": ""}))
            continue
        if not repository.host:
            raise StateUnavailable(
                f"Repository {repository.alias!r} is on {repository.machine!r}, which has no "
                f"SSH host reachable from execution machine {execution_machine!r}. Run the "
                "agent on that repository's machine or configure a reachable host."
            )
        repositories.append(repository)
    updates: dict[str, object] = {"repositories": repositories}
    state_repository = service.manifest.repository_map[service.manifest.state.repository]
    if state_repository.machine != execution_machine:
        raise StateUnavailable(
            "Graph-writing agents must run on the canonical state machine; "
            "cross-machine state staging is forbidden."
        )
    canonical = PurePosixPath(state_repository.path) / ".research"
    for field in _STATE_PATH_FIELDS:
        raw_path = getattr(context, field)
        if raw_path:
            updates[field] = str(canonical / Path(raw_path).name)
    updates["facts_dir"] = str(canonical / "facts")
    return updates


def _sse(event: AgentEvent) -> str:
    return f"data: {event.model_dump_json()}\n\n"
