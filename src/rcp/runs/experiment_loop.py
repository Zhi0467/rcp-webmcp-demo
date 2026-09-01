from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from rcp.agents.schema import (
    CreateEdgesOperation as AgentCreateEdgesOperation,
)
from rcp.agents.schema import (
    CreateProposalsOperation as AgentCreateProposalsOperation,
)
from rcp.agents.schema import (
    UpdateNodesOperation as AgentUpdateNodesOperation,
)
from rcp.agents.schema import (
    parse_agent_patch_json,
)
from rcp.background import AgentTaskExecution
from rcp.control import decision_drift
from rcp.core.models import ExperimentDecisionPin, GraphState, Patch
from rcp.core.operations import UpdateNodesOperation as CoreUpdateNodesOperation
from rcp.core.transition_models import GraphTargetRef
from rcp.limits import AGENT_TASK_RECEIPT_MAX_BYTES
from rcp.runs.shared import _stage_json_task_input
from rcp.service import GraphUpdateResult, ProjectService, RunRequest
from rcp.storage import (
    AgentTaskRecord,
    ExperimentEpisodeRecord,
    ExperimentLoopRuntime,
    ExperimentWatcherResourceRecord,
    GraphCondition,
    GraphWatcherRecord,
    StoredWatcherRecord,
    WatcherRecord,
    WatcherStopRequest,
)
from rcp.transport import RemoteRunStage, StateUnavailable
from rcp.watchers import (
    ExperimentWatchSpec,
    WatcherBinding,
    WatcherCheckResult,
    graph_condition_result,
    validate_graph_conditions,
)

if TYPE_CHECKING:
    from rcp.runs.episodes.wrapup import EpisodeWrapupSpec

_EXIT_STATUSES = frozenset({"completed"})
_COMPLETED_NEXT_ACTION_PROBLEM = (
    "Experiment status completed conflicts with a non-empty next_action. Continue useful work "
    "until next_action can truthfully be null, or keep a nonterminal status and arm a watcher "
    "for real detached work or explicitly pause for human authority."
)
_EPISODE_CONTEXT_CANDIDATE_ROLE = "experiment_episode_context_candidate"
_MAX_RECEIPT_ATTEMPTS = 8
_MAX_RECEIPT_TEXT = 400
_MAX_RECEIPT_LIST = 3
_MAX_RECEIPT_IDENTIFIERS = 4
_MAX_RECEIPT_DECISIONS = 4
_MAX_RECEIPT_ID = 100
_MAX_RECEIPT_LIST_TEXT = 120

EpisodeWakeReadiness = Literal["ready", "transient", "incompatible", "unavailable"]

ExperimentLoopPhase = Literal[
    "initial_run",
    "human_reauthorization",
    "watcher_wake",
    "resume",
    "retry",
]


@dataclass(frozen=True)
class EpisodeWakePreflight:
    """Whether a completed group may claim the current episode's native session.

    `transient` and `incompatible` both leave the watchers completed and
    unnotified for a later pass; only `unavailable` is a durable Needs-action
    fact about the episode itself.
    """

    readiness: EpisodeWakeReadiness
    diagnostic: str | None = None
    session_id: str | None = None
    stage_host: str | None = None
    stage_root: str | None = None


ExperimentLoopEnding = Literal["completed", "human_pause"]
ExperimentLoopEndingSignal = Literal[
    "experiment_completed",
    "proposal_created",
    "decision_awaits_human",
    "blocker_linked",
]


@dataclass(frozen=True)
class ExperimentLoopSemanticEnding:
    """One accepted Patch's explicit semantic ending, independent of loop mechanics."""

    ending: ExperimentLoopEnding
    signals: tuple[ExperimentLoopEndingSignal, ...]

    @property
    def partial(self) -> bool:
        return self.ending == "human_pause"


_EXPERIMENT_WATCH_OUTPUT_PREFIX = "experiment-watch-"
_EXPERIMENT_WATCH_OUTPUT_SUFFIX = ".json"


@dataclass(frozen=True)
class StagedExperimentWatcherResource:
    """One live node-and-episode watcher resource exposed to a chat turn."""

    resource: ExperimentWatcherResourceRecord
    watcher_state_path: str
    watch_path: str

    def prompt_value(self) -> dict[str, str]:
        return {
            "control_node_id": self.resource.control_node_id,
            "episode_id": self.resource.episode_id,
            "execution_host": self.resource.execution_host,
            "watcher_state_path": self.watcher_state_path,
            "watch_path": self.watch_path,
        }


def experiment_watcher_output_name(
    control_node_id: str,
    graph_target: GraphTargetRef | None = None,
) -> str:
    """Return the stable physical filename that selects one Experiment resource."""

    target = graph_target or GraphTargetRef()
    digest = hashlib.sha256(f"{target.key}\0{control_node_id}".encode()).hexdigest()
    return f"{_EXPERIMENT_WATCH_OUTPUT_PREFIX}{digest}{_EXPERIMENT_WATCH_OUTPUT_SUFFIX}"


def _experiment_watcher_output_names(
    workspace: Path,
    remote_stage: RemoteRunStage | None,
) -> list[str]:
    if remote_stage is not None:
        return [
            name
            for name in remote_stage.list_workspace_files()
            if name.startswith(_EXPERIMENT_WATCH_OUTPUT_PREFIX)
            and name.endswith(_EXPERIMENT_WATCH_OUTPUT_SUFFIX)
        ]
    try:
        entries = list(os.scandir(workspace))
    except OSError as exc:
        raise StateUnavailable(f"could not inspect chat watcher outputs: {exc}") from exc
    names: list[str] = []
    for entry in entries:
        if not (
            entry.name.startswith(_EXPERIMENT_WATCH_OUTPUT_PREFIX)
            and entry.name.endswith(_EXPERIMENT_WATCH_OUTPUT_SUFFIX)
        ):
            continue
        if not entry.is_file(follow_symlinks=False):
            raise ValueError(
                f"Experiment watcher output is not a direct regular file: {entry.name}"
            )
        names.append(entry.name)
    return sorted(names)


def clear_stale_experiment_watcher_outputs(
    workspace: Path,
    remote_stage: RemoteRunStage | None,
    resources: list[ExperimentWatcherResourceRecord],
) -> None:
    """Remove prior-turn resource outputs from a reusable chat workspace."""

    if remote_stage is not None:
        names = set(_experiment_watcher_output_names(workspace, remote_stage))
    else:
        try:
            with os.scandir(workspace) as entries:
                names = {
                    entry.name
                    for entry in entries
                    if entry.name.startswith(_EXPERIMENT_WATCH_OUTPUT_PREFIX)
                    and entry.name.endswith(_EXPERIMENT_WATCH_OUTPUT_SUFFIX)
                }
        except OSError as exc:
            raise StateUnavailable(f"could not inspect stale chat watcher outputs: {exc}") from exc
    # Exact current paths are removed even when a previous agent replaced one
    # with a symlink; the remote regular-file listing deliberately omits symlinks.
    names.update(
        experiment_watcher_output_name(item.control_node_id, item.graph_target)
        for item in resources
    )
    for name in sorted(names):
        if remote_stage is not None:
            remote_stage.remove_workspace_file(name)
        else:
            path = workspace / name
            if path.is_dir() and not path.is_symlink():
                raise ValueError(f"Experiment watcher output path is an unsafe directory: {name}")
            path.unlink(missing_ok=True)


def read_experiment_watcher_outputs(
    workspace: Path,
    remote_stage: RemoteRunStage | None,
) -> dict[str, str]:
    """Read each physical Experiment watcher file written by this turn."""

    outputs: dict[str, str] = {}
    for name in _experiment_watcher_output_names(workspace, remote_stage):
        if remote_stage is not None:
            outputs[name] = remote_stage.read_text(remote_stage.workspace / name)
        else:
            outputs[name] = (workspace / name).read_text(encoding="utf-8")
    return outputs


def preflight_episode_wake(
    runtime: ExperimentLoopRuntime,
    episode: ExperimentEpisodeRecord | None,
    group: list[StoredWatcherRecord],
) -> EpisodeWakePreflight:
    """Prove the episode session and exact stage before any claim or budget spend.

    Watcher provenance never selects the session — the newest human-authorized
    episode does — so an older group that no longer matches the current binding
    stays pending instead of switching sessions.
    """

    if not group:
        raise ValueError("An Experiment watcher wake requires a completed watcher group.")
    if episode is None or not episode.session_bound:
        return EpisodeWakePreflight(
            readiness="unavailable",
            diagnostic=(
                "This episode has no validated native provider session to continue. "
                "Use Stop loop and press Run to start a fresh episode."
            ),
        )
    if runtime.episode_id != episode.episode_id:
        return EpisodeWakePreflight(
            readiness="unavailable",
            diagnostic=(
                "The completed watcher group no longer belongs to the current Experiment episode. "
                "Use Stop loop and press Run to start a fresh episode."
            ),
        )
    mismatched: list[str] = []
    for record in group:
        continuation = record.continuation
        for label, expected, actual in (
            ("project", episode.project_id, record.project_id),
            ("Experiment", episode.control_node_id, record.node_id),
            ("episode", episode.episode_id, record.episode_id),
            ("check host", episode.execution_host, record.execution_host),
            ("continuation Experiment", episode.control_node_id, continuation.control_node_id),
            ("continuation episode", episode.episode_id, continuation.control_episode_id),
            ("Patch authority", "experiment_loop", continuation.patch_kind),
        ):
            if expected != actual and label not in mismatched:
                mismatched.append(label)
    if mismatched:
        return EpisodeWakePreflight(
            readiness="incompatible",
            diagnostic=(
                "This completed watcher group does not match the current node-attached episode's "
                f"{', '.join(mismatched)}; it stays pending for an explicit human Run."
            ),
        )
    assert episode.stage_root is not None
    if episode.stage_host:
        exists = RemoteRunStage(episode.stage_host).directory_exists(episode.stage_root)
    else:
        stage = Path(episode.stage_root)
        exists = stage.is_dir() and not stage.is_symlink()
    if exists is None:
        return EpisodeWakePreflight(
            readiness="transient",
            diagnostic="The episode's execution machine could not be reached for this pass.",
        )
    if not exists:
        return EpisodeWakePreflight(
            readiness="unavailable",
            diagnostic=(
                "The episode's saved provider workspace is gone from its execution machine. "
                "Use Stop loop and press Run to start a fresh episode."
            ),
        )
    return EpisodeWakePreflight(
        readiness="ready",
        session_id=episode.native_session_id,
        stage_host=episode.stage_host,
        stage_root=episode.stage_root,
    )


def experiment_watcher_delivery_request(
    group: list[StoredWatcherRecord],
    *,
    trigger: Literal["experiment_run", "watcher"],
    episode_id: str,
    invocation: int,
    invocation_ceiling: int,
    control_revision: int,
    decision_bundle: list[ExperimentDecisionPin],
    completion_criteria: list[str],
    session_id: str | None = None,
) -> RunRequest:
    """Build one explicitly attributed Experiment watcher delivery request.

    An automatic wake carries the episode's session id; a human Run that
    reauthorizes pending completion carries none, because it is a fresh episode.
    """

    if trigger == "experiment_run" and session_id:
        raise ValueError("A human Experiment Run always starts a fresh native session.")

    first = group[0]
    continuation = first.continuation
    if continuation.patch_kind != "experiment_loop" or not continuation.control_node_id:
        raise ValueError("An Experiment watcher must retain its origin control binding.")
    try:
        UUID(continuation.control_episode_id or "")
    except ValueError as exc:
        raise ValueError("An Experiment watcher has an invalid origin episode id.") from exc
    if (
        continuation.control_invocation is None
        or continuation.control_invocation_ceiling is None
        or continuation.control_invocation > continuation.control_invocation_ceiling
    ):
        raise ValueError("An Experiment watcher has an invalid origin invocation binding.")
    return RunRequest(
        provider=continuation.provider,
        # Older persisted watcher envelopes used null for the provider default.
        # Make it explicit before profile resolution so a later Settings change
        # cannot reinterpret this frozen continuation.
        model=continuation.model if continuation.model is not None else "",
        reasoning=continuation.reasoning,
        run_on=continuation.run_on,
        run_truth_scope=continuation.run_truth_scope,
        chat_scope="node" if first.origin_task_kind == "node_chat" else "project",
        node_id=first.node_id,
        message="Continue the bounded Experiment loop from its staged watcher state.",
        chat_id=first.chat_id,
        session_id=session_id,
        mode="work",
        trigger=trigger,
        patch_kind="experiment_loop",
        control_node_id=continuation.control_node_id,
        control_revision=control_revision,
        control_episode_id=episode_id,
        control_invocation=invocation,
        control_invocation_ceiling=invocation_ceiling,
        control_decision_bundle=decision_bundle,
        control_completion_criteria=completion_criteria,
        workflow_ids=continuation.workflow_ids,
        skill_ids=continuation.skill_ids,
        invoked_workflow_ids=[],
        invoked_skill_ids=[],
        resolved_skill_packages=continuation.resolved_skill_packages,
        watcher_ids=[item.watcher_id for item in group],
    )


def _completion_problem(operations: list[object], control_node_id: str) -> str | None:
    for operation in operations:
        if not isinstance(operation, (AgentUpdateNodesOperation, CoreUpdateNodesOperation)):
            continue
        for update in operation.nodes:
            if update.id != control_node_id:
                continue
            changes = update.changes
            if changes.get("status") != "completed":
                continue
            next_action = changes.get("next_action")
            if isinstance(next_action, str) and next_action.strip():
                return _COMPLETED_NEXT_ACTION_PROBLEM
    return None


def validate_experiment_completion(patch: Patch, control_node_id: str) -> None:
    """Reject a terminal claim that still names unfinished Experiment work."""

    problem = _completion_problem(list(patch.ops), control_node_id)
    if problem is not None:
        raise ValueError(problem)


def experiment_exit_problem(patch_text: str | None, control_node_id: str) -> str | None:
    """Explain a contradictory attempted exit before watcher correction."""

    if patch_text is None:
        return None
    try:
        patch = parse_agent_patch_json(patch_text)
    except ValueError:
        return None
    return _completion_problem(list(patch.ops), control_node_id)


def experiment_loop_semantic_ending(
    patch_text: str | None,
    control_node_id: str,
) -> ExperimentLoopSemanticEnding | None:
    """Classify one semantic Patch without turning mechanical failure into an ending."""

    if patch_text is None:
        return None
    try:
        patch = parse_agent_patch_json(patch_text)
    except ValueError:
        return None
    operations = list(patch.ops)
    completed = _completion_problem(operations, control_node_id) is None and any(
        update.id == control_node_id and update.changes.get("status") in _EXIT_STATUSES
        for operation in operations
        if isinstance(operation, AgentUpdateNodesOperation)
        for update in operation.nodes
    )
    pause_signals: list[ExperimentLoopEndingSignal] = []
    if any(
        isinstance(operation, AgentCreateProposalsOperation) and operation.proposals
        for operation in operations
    ):
        pause_signals.append("proposal_created")
    for operation in operations:
        if isinstance(operation, AgentUpdateNodesOperation):
            for update in operation.nodes:
                changes = update.changes
                if update.id != control_node_id and changes.get("status") in {"ready", "revisit"}:
                    pause_signals.append("decision_awaits_human")
        if isinstance(operation, AgentCreateEdgesOperation) and any(
            edge.source == control_node_id and edge.relation == "blocked_by"
            for edge in operation.edges
        ):
            pause_signals.append("blocker_linked")
    signals = tuple(dict.fromkeys(pause_signals))
    if completed:
        return ExperimentLoopSemanticEnding(
            ending="completed",
            signals=("experiment_completed", *signals),
        )
    if signals:
        return ExperimentLoopSemanticEnding(ending="human_pause", signals=signals)
    return None


def experiment_loop_ending_signal(
    *,
    semantic_ending: ExperimentLoopSemanticEnding,
    episode_id: str,
    control_node_id: str,
    invocation: int,
    invocation_ceiling: int,
    control_snapshot: dict[str, object],
    patch_text: str,
    graph_update: GraphUpdateResult,
    watcher_ids: list[str],
    stopped_watcher_ids: list[str],
    decision_bundle: list[ExperimentDecisionPin],
) -> dict[str, object]:
    """Build the compact mode-owned facts persisted after a successful handoff."""

    if graph_update.status != "applied":
        raise ValueError("Only an applied Experiment Patch can carry a semantic ending.")
    if not episode_id or not control_node_id or invocation < 1 or invocation_ceiling < invocation:
        raise ValueError("Experiment ending signal is missing its bounded episode identity.")
    attempts, attempt_count, current_summary, next_action = _receipt_control_result(
        control_snapshot,
        patch_text,
        control_node_id,
    )
    candidate_attempts = attempts[-_MAX_RECEIPT_ATTEMPTS:]
    selected_attempts: list[dict[str, object]] = []
    selected_decisions = decision_bundle[:_MAX_RECEIPT_DECISIONS]
    method = {
        "design": _receipt_text(control_snapshot.get("design")),
        "expected_outcomes": _receipt_text_list(control_snapshot.get("expected_outcomes")),
        "interpretation_rules": _receipt_text_list(control_snapshot.get("interpretation_rules")),
        "completion_criteria": _receipt_text_list(control_snapshot.get("completion_criteria")),
    }
    receipt: dict[str, object] = {
        "control": {
            "node_id": control_node_id,
            "title": _receipt_text(control_snapshot.get("title")),
            "objective": _receipt_text(control_snapshot.get("objective")),
            "method": method,
            "current_summary": _receipt_text(current_summary),
            "next_action": _receipt_optional_text(next_action),
        },
        "invocation": {
            "number": invocation,
            "ceiling": invocation_ceiling,
            "decision_bundle": [
                {
                    "decision_id": _receipt_text(item.decision_id, limit=_MAX_RECEIPT_ID),
                    "decision_revision": item.decision_revision,
                    "selected_option": _receipt_text(item.selected_option, limit=240),
                }
                for item in selected_decisions
            ],
            "omitted_decision_count": len(decision_bundle) - len(selected_decisions),
        },
        "attempt_observations": selected_attempts,
        "omitted_attempt_count": attempt_count,
        "watcher_summary": {
            "armed_count": len(watcher_ids),
            "armed_ids": _receipt_identifier_list(watcher_ids),
            "stopped_count": len(stopped_watcher_ids),
            "stopped_ids": _receipt_identifier_list(stopped_watcher_ids),
        },
        "graph_result": {
            "status": graph_update.status,
            "applied_revision": graph_update.applied_revision,
        },
        "semantic_signals": list(semantic_ending.signals),
    }
    signal: dict[str, object] = {
        "episode_id": episode_id,
        "ending": semantic_ending.ending,
        "partial": semantic_ending.partial,
        "receipt": receipt,
    }
    for observation in reversed(candidate_attempts):
        selected_attempts.insert(0, observation)
        receipt["omitted_attempt_count"] = attempt_count - len(selected_attempts)
        if _receipt_payload_size(signal) > AGENT_TASK_RECEIPT_MAX_BYTES:
            selected_attempts.pop(0)
            receipt["omitted_attempt_count"] = attempt_count - len(selected_attempts)
            break
    if _receipt_payload_size(signal) > AGENT_TASK_RECEIPT_MAX_BYTES:
        raise ValueError("The compact Experiment ending receipt exceeds its storage boundary.")
    return signal


def experiment_loop_wrapup_spec(
    continuation_operation_id: str,
    signal: dict[str, object],
) -> EpisodeWrapupSpec:
    """Turn one persisted mode signal into the shared wrap-up admission contract."""

    from rcp.runs.episodes.wrapup import EpisodeWrapupSpec

    episode_id = signal.get("episode_id")
    ending = signal.get("ending")
    partial = signal.get("partial")
    receipt = signal.get("receipt")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("Experiment ending signal has no episode id.")
    if ending not in {"completed", "human_pause"}:
        raise ValueError("Experiment ending signal has no semantic ending.")
    if not isinstance(partial, bool) or partial != (ending == "human_pause"):
        raise ValueError("Experiment ending signal has inconsistent partial state.")
    if not isinstance(receipt, dict):
        raise ValueError("Experiment ending signal has no compact receipt.")
    return EpisodeWrapupSpec(
        episode_id=episode_id,
        ending=ending,
        partial=partial,
        continuation_operation_id=continuation_operation_id,
        receipt=receipt,
    )


_LAUNCH_FAILURE_DIAGNOSTIC = (
    "This Experiment turn failed before it started its agent session, so the episode has "
    "no session to continue. Press Run to start a fresh episode."
)


def experiment_loop_launch_failure_diagnostic(continuation: AgentTaskRecord) -> str:
    """Explain a turn that died before binding a session, naming its real cause.

    The lineage recovery check cannot classify this: it looks for the retained
    episode context candidate, which a turn that never reached prompt assembly
    has not written yet, and so reports a pre-migration lineage defect for an
    episode created seconds ago.
    """

    cause = (continuation.error or continuation.status_message or "").strip()
    if not cause:
        return _LAUNCH_FAILURE_DIAGNOSTIC
    return f"{_LAUNCH_FAILURE_DIAGNOSTIC} It failed with: {cause}"


def experiment_loop_operational_ending_wrapup_spec(
    *,
    continuation: AgentTaskRecord,
    request: RunRequest,
    episode: ExperimentEpisodeRecord,
    ending: Literal["exhausted", "failed"],
    diagnostic: str,
) -> EpisodeWrapupSpec:
    """Adapt an operational ending without rebuilding the resumed session context."""

    from rcp.runs.episodes.wrapup import EpisodeWrapupSpec

    if (
        continuation.episode_id != episode.episode_id
        or continuation.project_id != episode.project_id
        or request.control_episode_id != episode.episode_id
        or request.control_node_id != episode.control_node_id
    ):
        raise ValueError("The Experiment operational ending lost its exact episode lineage.")
    invocation = request.control_invocation
    ceiling = request.control_invocation_ceiling
    if invocation is None or ceiling is None:
        raise ValueError("The Experiment operational ending lost its invocation boundary.")
    receipt: dict[str, object] = {
        "control": {
            "node_id": episode.control_node_id,
            "completion_criteria": _receipt_text_list(
                request.control_completion_criteria,
                limit=_MAX_RECEIPT_TEXT,
            ),
        },
        "invocation": {
            "number": invocation,
            "ceiling": ceiling,
            "decision_bundle": [
                {
                    "decision_id": _receipt_text(item.decision_id, limit=_MAX_RECEIPT_ID),
                    "decision_revision": item.decision_revision,
                    "selected_option": _receipt_text(
                        item.selected_option,
                        limit=_MAX_RECEIPT_TEXT,
                    ),
                }
                for item in request.control_decision_bundle[:_MAX_RECEIPT_DECISIONS]
            ],
        },
        "accepted_handoff": {
            "operation_id": continuation.operation_id,
            "last_graph_result": _receipt_optional_text(
                episode.last_graph_result,
                limit=_MAX_RECEIPT_TEXT,
            ),
            "watcher_ids": _receipt_identifier_list(episode.last_watcher_ids),
            "omitted_watcher_count": max(
                0,
                len(episode.last_watcher_ids) - _MAX_RECEIPT_IDENTIFIERS,
            ),
        },
        "operational_ending": {
            "status": continuation.status,
            "diagnostic": _receipt_text(diagnostic, limit=_MAX_RECEIPT_TEXT),
        },
    }
    return EpisodeWrapupSpec(
        episode_id=episode.episode_id,
        ending=ending,
        partial=True,
        continuation_operation_id=continuation.operation_id,
        receipt=receipt,
        diagnostic=diagnostic,
    )


def _receipt_control_result(
    control_snapshot: dict[str, object],
    patch_text: str,
    control_node_id: str,
) -> tuple[list[dict[str, object]], int, object, object]:
    attempts = control_snapshot.get("attempts")
    current_summary = control_snapshot.get("current_summary")
    next_action = control_snapshot.get("next_action")
    try:
        patch = parse_agent_patch_json(patch_text)
    except ValueError:
        patch = None
    if patch is not None:
        for operation in patch.ops:
            if not isinstance(operation, AgentUpdateNodesOperation):
                continue
            for update in operation.nodes:
                if update.id != control_node_id:
                    continue
                changes = update.changes
                if isinstance(changes.get("attempts"), list):
                    attempts = changes["attempts"]
                if "current_summary" in changes:
                    current_summary = changes["current_summary"]
                if "next_action" in changes:
                    next_action = changes["next_action"]
    if not isinstance(attempts, list):
        attempts = []
    attempt_count = len(attempts)
    observations: list[dict[str, object]] = []
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        observation: dict[str, object] = {
            "id": _receipt_text(attempt.get("id"), limit=_MAX_RECEIPT_ID),
            "sequence": attempt.get("sequence"),
            "purpose": _receipt_text(attempt.get("purpose"), limit=200),
            "attempt_kind": attempt.get("attempt_kind", "external_run"),
            "configuration": _receipt_text(attempt.get("configuration"), limit=300),
            "status": attempt.get("status"),
            "observation": _receipt_optional_text(
                attempt.get("outcome") or attempt.get("failure_reason"),
                limit=300,
            ),
            "job_refs": _receipt_text_list(
                attempt.get("job_refs"),
                limit=_MAX_RECEIPT_ID,
            ),
        }
        debug = attempt.get("debug")
        if isinstance(debug, dict):
            observation["debug"] = {
                key: _receipt_text(debug.get(key), limit=160)
                for key in ("mechanical_fault", "change", "predicted_effect")
            }
        observations.append(observation)
    return observations, attempt_count, current_summary, next_action


def _receipt_text(value: object, *, limit: int = _MAX_RECEIPT_TEXT) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:limit]


def _receipt_optional_text(value: object, *, limit: int = _MAX_RECEIPT_TEXT) -> str | None:
    text = _receipt_text(value, limit=limit)
    return text or None


def _receipt_text_list(
    value: object,
    *,
    limit: int = _MAX_RECEIPT_LIST_TEXT,
) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        text for item in value[:_MAX_RECEIPT_LIST] if (text := _receipt_text(item, limit=limit))
    ]


def _receipt_identifier_list(values: list[str]) -> list[str]:
    return [
        text
        for item in values[:_MAX_RECEIPT_IDENTIFIERS]
        if (text := _receipt_text(item, limit=_MAX_RECEIPT_ID))
    ]


def _receipt_payload_size(payload: dict[str, object]) -> int:
    return len(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    )


def root_experiment_loop_operation_id(execution: AgentTaskExecution) -> str:
    """Resolve the durable root task through explicit parent links, failing on broken lineage."""

    operation_id = execution.operation_id
    seen: set[str] = set()
    while True:
        if operation_id in seen:
            raise ValueError("Experiment-loop task lineage contains a cycle.")
        seen.add(operation_id)
        record = execution.store.agent_task(operation_id)
        if record is None:
            raise ValueError("Experiment-loop task lineage is incomplete.")
        if record.parent_operation_id is None:
            return operation_id
        operation_id = record.parent_operation_id


def prepare_experiment_watcher_records(
    execution: AgentTaskExecution,
    specs: list[ExperimentWatchSpec],
    results: list[WatcherCheckResult],
    binding: WatcherBinding,
    *,
    graph_conditions: list[GraphCondition] | None = None,
    graph_state: GraphState | None = None,
    armed_revision: int | None = None,
) -> list[StoredWatcherRecord]:
    """Prepare one deterministic watcher handoff without writing storage."""

    if len(specs) != len(results):
        raise ValueError("Experiment-loop watcher checks do not match their specifications.")
    conditions = list(graph_conditions or [])
    if conditions:
        if graph_state is None:
            raise ValueError("Experiment graph conditions require current canonical graph state.")
        if armed_revision is None:
            raise ValueError("Experiment graph conditions require their validated base revision.")
        if armed_revision > graph_state.revision:
            raise ValueError("Experiment graph condition baseline is ahead of canonical state.")
        serialized = [item.model_dump_json() for item in conditions]
        if len(serialized) != len(set(serialized)):
            raise ValueError("an Experiment handoff cannot repeat a graph condition")
        validate_graph_conditions(conditions, graph_state)
    created_at = execution.store.now()
    desired: list[StoredWatcherRecord] = []
    for index, (spec, result) in enumerate(zip(specs, results, strict=True)):
        group = getattr(spec, "group", None)
        identity = json.dumps(
            {
                "origin": binding.origin_operation_id,
                "node_id": binding.node_id,
                "episode_id": binding.continuation.control_episode_id,
                "index": index,
                "check_command": spec.check_command,
                "log_path": spec.log_path,
                "cwd": spec.cwd,
                "group": group,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        completed = result.state == "complete"
        desired.append(
            WatcherRecord(
                watcher_id=str(uuid5(NAMESPACE_URL, f"rcp-experiment-watcher:{identity}")),
                project_id=binding.project_id,
                origin_operation_id=binding.origin_operation_id,
                origin_task_kind=binding.origin_task_kind,
                chat_id=binding.chat_id,
                node_id=binding.node_id,
                episode_id=binding.episode_id,
                graph_target=binding.graph_target,
                execution_host=binding.execution_host,
                check_command=spec.check_command,
                log_path=spec.log_path,
                cwd=spec.cwd,
                continuation=binding.continuation,
                status="completed" if completed else "active",
                created_at=created_at,
                last_checked_at=result.checked_at,
                last_exit_code=result.exit_code,
                completed_at=result.checked_at if completed else None,
                group_id=(
                    str(
                        uuid5(
                            NAMESPACE_URL,
                            "rcp-experiment-watcher-group:"
                            f"{binding.origin_operation_id}:{binding.node_id}:"
                            f"{binding.continuation.control_episode_id}:{spec.group}",
                        )
                    )
                    if group is not None
                    else None
                ),
                group_label=group,
            )
        )
    for index, condition in enumerate(conditions):
        identity = json.dumps(
            {
                "origin": binding.origin_operation_id,
                "node_id": binding.node_id,
                "episode_id": binding.continuation.control_episode_id,
                "index": index,
                "condition": condition.model_dump(mode="json"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        assert graph_state is not None
        assert armed_revision is not None
        result = graph_condition_result(
            condition,
            graph_state,
            armed_revision=armed_revision,
        )
        if result == "removed":
            raise ValueError(
                f"Experiment graph condition target {condition.node_id!r} is not canonical."
            )
        completed = result == "completed"
        desired.append(
            GraphWatcherRecord(
                watcher_id=str(uuid5(NAMESPACE_URL, f"rcp-experiment-graph-watcher:{identity}")),
                project_id=binding.project_id,
                origin_operation_id=binding.origin_operation_id,
                origin_task_kind=binding.origin_task_kind,
                chat_id=binding.chat_id,
                node_id=binding.node_id,
                episode_id=binding.episode_id,
                graph_target=binding.graph_target,
                execution_host=binding.execution_host,
                condition=condition,
                armed_revision=armed_revision,
                continuation=binding.continuation,
                status="completed" if completed else "active",
                created_at=created_at,
                last_evaluated_at=created_at,
                completed_at=created_at if completed else None,
            )
        )

    return desired


def persist_experiment_watchers_idempotently(
    execution: AgentTaskExecution,
    specs: list[ExperimentWatchSpec],
    results: list[WatcherCheckResult],
    binding: WatcherBinding,
    stops: list[WatcherStopRequest] | None = None,
    *,
    graph_conditions: list[GraphCondition] | None = None,
    graph_state: GraphState | None = None,
    armed_revision: int | None = None,
    expected_watcher_snapshot_token: str | None = None,
) -> list[StoredWatcherRecord]:
    """Persist one validated handoff once across Retry/crash recovery."""

    desired = prepare_experiment_watcher_records(
        execution,
        specs,
        results,
        binding,
        graph_conditions=graph_conditions,
        graph_state=graph_state,
        armed_revision=armed_revision,
    )
    # The store owns the BEGIN IMMEDIATE boundary shared with Stop loop. It
    # atomically deduplicates this deterministic handoff and, when stop intent
    # won the race, persists/returns every watcher as stopped and notified.
    return execution.store.persist_experiment_watchers_idempotently(
        desired,
        stops=stops,
        binding=binding,
        expected_watcher_snapshot_token=expected_watcher_snapshot_token,
    )


def _stopped_history_episode_id(
    execution: AgentTaskExecution,
    project_id: str,
    control_node_id: str,
    episode_id: str,
) -> str | None:
    """The immediately preceding episode's id, but only when a human stopped it.

    A fresh Run after S72 **Stop loop** stages that episode's watcher records so
    the agent can inspect external work that may still exist. Any other preceding
    episode contributes nothing: a stopped observer is context, never a trigger.
    """

    previous = execution.store.previous_experiment_episode(
        project_id,
        control_node_id,
        episode_id,
    )
    if previous is None or previous.stop_requested_at is None:
        return None
    return previous.episode_id


def _watcher_state(
    execution: AgentTaskExecution,
    control_node_id: str,
    delivered_watcher_ids: list[str],
    episode_id: str,
    phase: ExperimentLoopPhase,
) -> list[dict[str, object]]:
    """Return current operational watcher evidence without duplicating loop control.

    Every shape carries the relevant active, degraded, and completed-unnotified
    observers. A wake and a human reauthorization additionally retain their own
    delivered group after its atomic claim marked it notified, and a fresh Run
    after a human stop additionally retains that stopped episode's records.
    """

    task = execution.store.agent_task(execution.operation_id)
    if task is None:
        raise ValueError("The Experiment-loop operation is no longer available.")
    delivered = set(delivered_watcher_ids)
    all_records = execution.store.watchers(task.project_id)
    delivered_group_ids = {
        record.group_id
        for record in all_records
        if record.watcher_id in delivered and record.group_id is not None
    }
    stopped_history_episode_id = (
        _stopped_history_episode_id(execution, task.project_id, control_node_id, episode_id)
        if phase == "initial_run"
        else None
    )
    records = [
        record
        for record in all_records
        if record.continuation.patch_kind == "experiment_loop"
        and record.graph_target == task.graph_target
        and record.continuation.control_node_id == control_node_id
        and (
            record.watcher_id in delivered
            or record.group_id in delivered_group_ids
            or (record.status in {"active", "degraded"} and not record.notified)
            or (record.status == "completed" and not record.notified)
            or (record.status == "stopped" and record.continuation.control_episode_id == episode_id)
            or (
                stopped_history_episode_id is not None
                and record.status == "stopped"
                and execution.store.experiment_watcher_compatible_with_episode(
                    record.watcher_id,
                    stopped_history_episode_id,
                )
            )
        )
    ]
    state: list[dict[str, object]] = []
    for record in records:
        item: dict[str, object] = {
            "watcher_id": record.watcher_id,
            "origin_operation_id": record.origin_operation_id,
            "execution_host": record.execution_host,
            "status": record.status,
            "created_at": record.created_at,
            "completed_at": record.completed_at,
            "notified": record.notified,
            "notification_operation_id": record.notification_operation_id,
            "stopped_by": record.stopped_by,
            "stop_reason": record.stop_reason,
            "stopped_at": record.stopped_at,
            "stop_operation_id": record.stop_operation_id,
            "episode_id": record.continuation.control_episode_id,
            "invocation": record.continuation.control_invocation,
            "invocation_ceiling": record.continuation.control_invocation_ceiling,
            "control_revision": record.continuation.control_revision,
            "decision_bundle": record.continuation.control_decision_bundle,
        }
        if isinstance(record, GraphWatcherRecord):
            item["condition"] = record.condition.model_dump(mode="json")
            item["armed_revision"] = record.armed_revision
            item["last_evaluated_at"] = record.last_evaluated_at
        else:
            item.update(
                {
                    "check_command": record.check_command,
                    "log_path": record.log_path,
                    "cwd": record.cwd,
                    "last_checked_at": record.last_checked_at,
                    "last_exit_code": record.last_exit_code,
                    "last_error": record.last_error,
                    "next_check_at": record.next_check_at,
                    "consecutive_error_count": record.consecutive_error_count,
                    "group_id": record.group_id,
                    "group_label": record.group_label,
                }
            )
        state.append(item)
    return state


async def stage_chat_experiment_watcher_resources(
    request: RunRequest,
    execution: AgentTaskExecution | None,
    local_stage: Path | None,
    remote_stage: RemoteRunStage | None,
    *,
    workspace: Path,
    token: str,
    clear_stale: bool,
) -> list[StagedExperimentWatcherResource]:
    """Stage live Experiment watcher state within the chat's resolved node scope."""

    if execution is None:
        return []
    task = execution.store.agent_task(execution.operation_id)
    if task is None:
        raise ValueError("The chat operation is no longer available for watcher resource staging.")
    all_resources = execution.store.experiment_watcher_resources(
        task.project_id,
        graph_target=task.graph_target,
    )
    if clear_stale:
        clear_stale_experiment_watcher_outputs(workspace, remote_stage, all_resources)

    def visible_resources(
        resources: list[ExperimentWatcherResourceRecord],
    ) -> list[ExperimentWatcherResourceRecord]:
        return (
            [item for item in resources if item.control_node_id == request.node_id]
            if request.chat_scope == "node"
            else resources
        )

    visible = visible_resources(all_resources)
    for _attempt in range(3):
        watcher_states = await asyncio.gather(
            *(
                asyncio.to_thread(
                    _watcher_state,
                    execution,
                    resource.control_node_id,
                    [],
                    resource.episode_id,
                    "resume",
                )
                for resource in visible
            )
        )
        refreshed = visible_resources(
            execution.store.experiment_watcher_resources(
                task.project_id,
                graph_target=task.graph_target,
            )
        )
        if {
            (item.control_node_id, item.graph_target.key): item.watcher_snapshot_token
            for item in visible
        } == {
            (item.control_node_id, item.graph_target.key): item.watcher_snapshot_token
            for item in refreshed
        }:
            break
        visible = refreshed
    else:
        raise StateUnavailable(
            "Experiment watcher state changed repeatedly while staging this chat turn."
        )
    staged: list[StagedExperimentWatcherResource] = []
    for resource, watcher_state in zip(visible, watcher_states, strict=True):
        digest = hashlib.sha256(
            f"{resource.graph_target.key}\0{resource.control_node_id}".encode()
        ).hexdigest()[:16]
        watcher_state_path = _stage_json_task_input(
            local_stage,
            remote_stage,
            f"task-{token}-experiment-watchers-{digest}.json",
            watcher_state,
        )
        staged.append(
            StagedExperimentWatcherResource(
                resource=resource,
                watcher_state_path=watcher_state_path,
                watch_path=str(
                    workspace
                    / experiment_watcher_output_name(
                        resource.control_node_id,
                        resource.graph_target,
                    )
                ),
            )
        )
    return staged


def _delivered_watcher_groups(
    watcher_state: list[dict[str, object]],
    delivered_watcher_ids: list[str],
) -> list[dict[str, object]]:
    """Retain each delivered group's identity and complete staged membership."""

    delivered = set(delivered_watcher_ids)
    delivered_group_ids = {
        record["group_id"]
        for record in watcher_state
        if isinstance(record.get("group_id"), str) and record.get("watcher_id") in delivered
    }
    return [
        {
            "group_id": group_id,
            "label": next(
                record.get("group_label")
                for record in watcher_state
                if record.get("group_id") == group_id
            ),
            "members": [record for record in watcher_state if record.get("group_id") == group_id],
        }
        for group_id in sorted(delivered_group_ids)
    ]


def experiment_loop_phase(request: RunRequest, continuation: str) -> ExperimentLoopPhase:
    """Name the agent-facing phase for one bounded-loop turn."""

    if continuation == "resume":
        return "resume"
    if continuation in {"retry", "handoff"}:
        return "retry"
    if request.trigger == "experiment_run" and request.watcher_ids:
        return "human_reauthorization"
    if request.trigger == "watcher":
        return "watcher_wake"
    return "initial_run"


def experiment_episode_context_values(
    *,
    ontology_extensions: bool,
    ontology: dict[str, object],
    repositories: list[dict[str, str]],
    skill_pointers: list[dict[str, object]],
) -> dict[str, object]:
    """The episode context that may change between turns of one native session.

    Provider, model, reasoning, machine, truth scope, and authority are pinned for
    the episode, and graph/research/schema/output pointers are refreshed in every
    turn's own message, so neither belongs in the replacement baseline.
    """

    normalized_ontology = json.dumps(
        ontology,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "ontology": {
            "extensions": ontology_extensions,
            "sha256": hashlib.sha256(normalized_ontology.encode("utf-8")).hexdigest(),
        },
        "repositories": repositories,
        "skills": {"pointers": skill_pointers},
    }


def prepare_experiment_episode_context_candidate(
    execution: AgentTaskExecution,
    current_values: dict[str, object],
) -> dict[str, object]:
    """Persist the context one invocation actually sent before it can succeed.

    Resume and in-session Retry keep their original narrow contract, so they must
    commit the originating invocation's candidate rather than whatever happens to
    be current when recovery finishes. Fresh human turns and automatic wakes each
    establish their own immutable candidate.
    """

    if execution.continuation in {"resume", "retry"}:
        root_operation_id = root_experiment_loop_operation_id(execution)
        content = execution.store.agent_task_contract(
            root_operation_id,
            _EPISODE_CONTEXT_CANDIDATE_ROLE,
        )
        if content is None:
            raise ValueError(
                "The continued Experiment-loop turn has no retained episode context candidate."
            )
        try:
            candidate = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "The retained Experiment-loop episode context candidate is invalid."
            ) from exc
        if not isinstance(candidate, dict):
            raise ValueError("The retained Experiment-loop episode context must be an object.")
        return candidate

    content = json.dumps(
        current_values,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    execution.store.record_agent_task_contract(
        execution.operation_id,
        _EPISODE_CONTEXT_CANDIDATE_ROLE,
        content,
        hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )
    return json.loads(content)


def experiment_graph_result_summary(graph_update: GraphUpdateResult) -> str:
    """Say truthfully what RCP did with this turn's Patch, for the next wake."""

    if graph_update.status == "applied":
        return f"applied as revision {graph_update.applied_revision}"
    if graph_update.status == "rejected":
        detail = (
            graph_update.validation_messages[0]
            if graph_update.validation_messages
            else "the graph rejected it"
        )
        return f"rejected: {detail[:400]}"
    return "no graph change"


def _prepare_experiment_episode_binding_intent(
    execution: AgentTaskExecution,
    request: RunRequest,
    *,
    native_session_id: str | None,
    execution_host: str,
    stage_host: str | None,
    stage_root: str | None,
    ending_signal: dict[str, object] | None,
) -> tuple[
    AgentTaskRecord,
    ExperimentEpisodeRecord | None,
    bool,
    dict[str, object] | None,
]:
    """Validate a completed turn and prepare its binding-replacement intent."""

    if ending_signal is not None and ending_signal.get("episode_id") != request.control_episode_id:
        raise ValueError("Experiment ending signal names another episode.")
    task = execution.store.agent_task(execution.operation_id)
    if task is None:
        raise ValueError("The completed Experiment-loop task record is unavailable.")
    if (
        not request.control_episode_id
        or not request.control_node_id
        or request.control_invocation is None
        or not request.provider
        or not request.run_on
        or not request.chat_id
    ):
        raise ValueError("A completed Experiment-loop turn is missing its episode binding.")
    if not native_session_id or not stage_root:
        raise ValueError(
            "A successful Experiment-loop turn did not retain its native session and exact stage."
        )
    episode = execution.store.experiment_episode(request.control_episode_id)
    binding_replacement = bool(
        episode is not None
        and episode.session_bound
        and (
            episode.provider != request.provider
            or episode.native_session_id != native_session_id
            or episode.stage_host != stage_host
            or episode.stage_root != stage_root
        )
    )
    recovery_ancestor_id = task.parent_operation_id
    explicit_handoff_lineage = execution.continuation == "handoff" and bool(recovery_ancestor_id)
    while binding_replacement and recovery_ancestor_id and not explicit_handoff_lineage:
        explicit_handoff_lineage = (
            execution.store.agent_task_continuation_cause(recovery_ancestor_id) == "handoff"
        )
        ancestor = execution.store.agent_task(recovery_ancestor_id)
        recovery_ancestor_id = ancestor.parent_operation_id if ancestor is not None else None
    replacement_authorized = binding_replacement and explicit_handoff_lineage
    if binding_replacement and not replacement_authorized:
        raise ValueError(
            "Only an explicit human Experiment recovery may replace the committed provider session."
        )
    if (
        request.trigger == "watcher"
        and execution.continuation == "watcher_wake"
        and (episode is None or episode.native_session_id != native_session_id)
    ):
        raise ValueError(
            "An automatic Experiment wake cannot replace its committed native session."
        )
    replacement_provenance = (
        {
            "episode_id": request.control_episode_id,
            "invocation": request.control_invocation,
            "parent_operation_id": task.parent_operation_id,
            "previous": {
                "provider": episode.provider,
                "native_session_id": episode.native_session_id,
                "stage_host": episode.stage_host,
                "stage_root": episode.stage_root,
            },
            "replacement": {
                "provider": request.provider,
                "native_session_id": native_session_id,
                "stage_host": stage_host,
                "stage_root": stage_root,
            },
        }
        if replacement_authorized and episode is not None
        else None
    )
    return task, episode, replacement_authorized, replacement_provenance


def commit_experiment_episode_binding(
    execution: AgentTaskExecution,
    request: RunRequest,
    *,
    native_session_id: str | None,
    execution_host: str,
    stage_host: str | None,
    stage_root: str | None,
    graph_result: str,
    watcher_ids: list[str],
    context_baseline: dict[str, object],
    ending_signal: dict[str, object] | None = None,
) -> None:
    """Bind this episode to the session and stage a later automatic wake resumes.

    Only a turn with a mechanically successful joint Patch/watch handoff commits,
    so a provider, task, or handoff failure never moves the binding or baseline.
    A graph rejection is still recorded truthfully because the turn and its
    accepted watcher handoff completed.
    """

    task, _episode, replacement_authorized, replacement_provenance = (
        _prepare_experiment_episode_binding_intent(
            execution,
            request,
            native_session_id=native_session_id,
            execution_host=execution_host,
            stage_host=stage_host,
            stage_root=stage_root,
            ending_signal=ending_signal,
        )
    )
    execution.store.commit_experiment_episode_turn(
        episode_id=request.control_episode_id,
        project_id=task.project_id,
        control_node_id=request.control_node_id,
        provider=request.provider,
        execution_machine=request.run_on,
        execution_host=execution_host,
        native_session_id=native_session_id,
        stage_host=stage_host,
        stage_root=stage_root,
        chat_id=request.chat_id,
        operation_id=execution.operation_id,
        invocation=request.control_invocation,
        graph_result=graph_result,
        watcher_ids=watcher_ids,
        context_baseline=context_baseline,
        ending_signal=ending_signal,
        replace_binding=replacement_authorized,
        replacement_provenance=replacement_provenance,
    )
    execution.store.record_agent_task_receipt(
        execution.operation_id,
        "experiment_episode_binding",
        {
            "episode_id": request.control_episode_id,
            "invocation": request.control_invocation,
            "provider": request.provider,
            "execution_machine": request.run_on,
            "stage_host": stage_host,
            "stage_root": stage_root,
            "graph_result": graph_result,
            "watcher_ids": watcher_ids,
            "binding_replaced": replacement_authorized,
        },
    )


def commit_experiment_episode_handoff(
    execution: AgentTaskExecution,
    request: RunRequest,
    watcher_records: list[StoredWatcherRecord],
    binding: WatcherBinding,
    *,
    native_session_id: str | None,
    execution_host: str,
    stage_host: str | None,
    stage_root: str | None,
    graph_result: str,
    context_baseline: dict[str, object],
    stops: list[WatcherStopRequest] | None = None,
    expected_watcher_snapshot_token: str | None = None,
    ending_signal: dict[str, object] | None = None,
) -> list[StoredWatcherRecord]:
    """Commit prepared watchers and the episode binding as one handoff."""

    task, _episode, replacement_authorized, replacement_provenance = (
        _prepare_experiment_episode_binding_intent(
            execution,
            request,
            native_session_id=native_session_id,
            execution_host=execution_host,
            stage_host=stage_host,
            stage_root=stage_root,
            ending_signal=ending_signal,
        )
    )
    continuation = binding.continuation
    if (
        binding.origin_operation_id != root_experiment_loop_operation_id(execution)
        or binding.project_id != task.project_id
        or binding.chat_id != request.chat_id
        or binding.node_id != request.node_id
        or binding.episode_id != request.control_episode_id
        or binding.execution_host != execution_host
        or continuation.provider != request.provider
        or continuation.run_on != request.run_on
        or continuation.patch_kind != request.patch_kind
        or continuation.control_node_id != request.control_node_id
        or continuation.control_episode_id != request.control_episode_id
        or continuation.control_invocation != request.control_invocation
        or continuation.control_invocation_ceiling != request.control_invocation_ceiling
    ):
        raise ValueError("Experiment handoff binding does not match the current task scope.")
    stored_watchers, _stored_episode = execution.store.commit_experiment_episode_handoff(
        watcher_records,
        binding=binding,
        operation_id=execution.operation_id,
        native_session_id=native_session_id,
        stage_host=stage_host,
        stage_root=stage_root,
        graph_result=graph_result,
        context_baseline=context_baseline,
        stops=stops,
        expected_watcher_snapshot_token=expected_watcher_snapshot_token,
        ending_signal=ending_signal,
        replace_binding=replacement_authorized,
        replacement_provenance=replacement_provenance,
    )
    execution.store.record_agent_task_receipt(
        execution.operation_id,
        "experiment_episode_binding",
        {
            "episode_id": request.control_episode_id,
            "invocation": request.control_invocation,
            "provider": request.provider,
            "execution_machine": request.run_on,
            "stage_host": stage_host,
            "stage_root": stage_root,
            "graph_result": graph_result,
            "watcher_ids": [item.watcher_id for item in stored_watchers],
            "binding_replaced": replacement_authorized,
        },
    )
    return stored_watchers


async def stage_experiment_loop_context(
    service: ProjectService,
    request: RunRequest,
    execution: AgentTaskExecution | None,
    local_stage: Path | None,
    remote_stage: RemoteRunStage | None,
    *,
    token: str,
    continuation: str,
) -> tuple[str, str]:
    """Stage irreducible loop control plus separately readable watcher state."""

    if not request.control_node_id or request.control_revision is None:
        raise ValueError("Experiment-loop work is missing its RCP control binding.")
    if (
        not request.control_episode_id
        or request.control_invocation is None
        or request.control_invocation_ceiling is None
    ):
        raise ValueError("Experiment-loop work is missing its episode invocation binding.")
    if execution is None:
        raise ValueError("Experiment-loop work requires a durable RCP operation.")

    phase = experiment_loop_phase(request, continuation)
    state, watcher_state = await asyncio.gather(
        asyncio.to_thread(service.history.state),
        asyncio.to_thread(
            _watcher_state,
            execution,
            request.control_node_id,
            request.watcher_ids,
            request.control_episode_id,
            phase,
        ),
    )
    watcher_state_path = _stage_json_task_input(
        local_stage,
        remote_stage,
        f"task-{token}-experiment-watchers.json",
        watcher_state,
    )
    delivered_groups = _delivered_watcher_groups(watcher_state, request.watcher_ids)
    drift = decision_drift(state, request.control_decision_bundle)
    if request.control_invocation > request.control_invocation_ceiling:
        raise ValueError("Experiment-loop invocation exceeds its pinned ceiling.")
    control_path = _stage_json_task_input(
        local_stage,
        remote_stage,
        f"task-{token}-experiment-control-{phase}.json",
        {
            "phase": phase,
            "episode_id": request.control_episode_id,
            "invocation": request.control_invocation,
            "invocation_ceiling": request.control_invocation_ceiling,
            "remaining_invocations": (
                request.control_invocation_ceiling - request.control_invocation
            ),
            "decision_bundle": [
                item.model_dump(mode="json") for item in request.control_decision_bundle
            ],
            "decision_drift": [item.model_dump(mode="json") for item in drift],
            "completion_criteria": request.control_completion_criteria,
            "delivered_watcher_ids": request.watcher_ids,
            "delivered_watcher_groups": delivered_groups,
            "watcher_state_path": watcher_state_path,
        },
    )
    return control_path, watcher_state_path
