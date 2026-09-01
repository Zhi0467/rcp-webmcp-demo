from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Literal, TypeVar

from pydantic import BaseModel, ValidationError

from rcp.core.authority import AgentDispatchAuthority, AgentDispatchScope, require_dispatch
from rcp.runs.auto_research import AutoResearchRunRequest
from rcp.runs.branch_merge_request import BranchMergeRunRequest
from rcp.runs.tasks.episode_report import EpisodeReportRunRequest
from rcp.service import CoachRequest, RunRequest
from rcp.skill_registry import SkillSelection
from rcp.storage import AgentTaskKind, AgentTaskRecord, AppStore
from rcp.storage.request_compat import migrate_stored_task_request

AgentTaskRequest = (
    RunRequest
    | CoachRequest
    | AutoResearchRunRequest
    | BranchMergeRunRequest
    | EpisodeReportRunRequest
)
DispatchAuthorityResolver = Callable[
    [AgentTaskKind, AgentTaskRequest],
    AgentDispatchAuthority | None,
]
AgentTaskContinuation = Literal[
    "fresh",
    "resume",
    "retry",
    "handoff",
    "graph_repair",
    "watcher_wake",
    "graph_condition_wake",
    "message_wake",
    "lifecycle_wake",
    "auto_research_continuation",
    "episode_report",
]

_StoredRequest = TypeVar("_StoredRequest", bound=BaseModel)

_AUTO_RESEARCH_GRAPH_ROLES = frozenset({"orchestrator", "worker"})
_CHAT_TASK_KINDS = frozenset({"node_chat", "project_chat"})
_INGEST_TASK_KINDS = frozenset({"seed", "refresh"})


def load_stored_request(
    model: type[_StoredRequest],
    stored: Mapping[str, object],
    *,
    operation_id: str | None = None,
) -> _StoredRequest:
    """Parse one request RCP persisted through the shared compatibility allowlist.

    SQLite task rows are normalized before they become ``AgentTaskRecord``
    instances, so every ordinary stored-task path shares the same migration.
    This helper covers stored mappings assembled outside that row boundary, such
    as a retry candidate merged with human overrides. It removes only explicitly
    retired fields; every unknown, current, or malformed field remains subject
    to the request model's strict validation.
    """

    kind = "auto_research" if model is AutoResearchRunRequest else ""
    migrated = migrate_stored_task_request(
        kind,
        stored,
        operation_id=operation_id,
    )
    return model.model_validate(migrated)


def task_graph_capable(kind: str, request: object) -> bool:
    """Return whether a persisted or live task request may produce a graph patch."""

    if kind in _INGEST_TASK_KINDS:
        return _run_request(request) is not None
    if kind in _CHAT_TASK_KINDS:
        run_request = _run_request(request)
        return run_request is not None and run_request.mode == "work"
    if kind == "auto_research":
        auto_research_request = _auto_research_request(request)
        return (
            auto_research_request is not None
            and auto_research_request.role in _AUTO_RESEARCH_GRAPH_ROLES
        )
    if kind == "branch_merge":
        return _branch_merge_request(request) is not None
    return False


def task_experiment_episode_id(request: object) -> str | None:
    """Return the bounded-experiment episode selected by a live Work request."""

    if isinstance(request, RunRequest) and request.patch_kind == "experiment_loop":
        return request.control_episode_id or ""
    return None


def _run_request(request: object) -> RunRequest | None:
    if isinstance(request, RunRequest):
        return request
    if not isinstance(request, dict):
        return None
    try:
        return RunRequest.model_validate(request)
    except ValidationError:
        return None


def _auto_research_request(request: object) -> AutoResearchRunRequest | None:
    if isinstance(request, AutoResearchRunRequest):
        return request
    if not isinstance(request, dict):
        return None
    role = request.get("role")
    if not isinstance(role, str) or role not in _AUTO_RESEARCH_GRAPH_ROLES:
        return None
    try:
        return AutoResearchRunRequest.model_validate(request)
    except ValidationError:
        return None


def _branch_merge_request(request: object) -> BranchMergeRunRequest | None:
    if isinstance(request, BranchMergeRunRequest):
        return request
    if not isinstance(request, dict):
        return None
    try:
        return BranchMergeRunRequest.model_validate(request)
    except ValidationError:
        return None


def resolved_dispatch_authority(
    store: AppStore,
    resolver: DispatchAuthorityResolver,
    kind: AgentTaskKind,
    request: AgentTaskRequest,
    *,
    project_id: str,
    parent: AgentTaskRecord | None = None,
    operation_id: str | None = None,
    continuation: AgentTaskContinuation = "fresh",
) -> AgentDispatchAuthority | None:
    """Resolve and gate the durable dispatch authority one admission may launch under.

    Admission policy, not engine plumbing: it decides what a task is allowed to
    run as, and every admission owner needs it once those owners leave
    ``background.py``.  It takes the store and the resolver rather than the
    engine because it needs nothing else from it.
    """

    if kind == "branch_merge":
        if not isinstance(request, BranchMergeRunRequest):
            raise TypeError("branch_merge dispatch requires a BranchMergeRunRequest")
        authority = AgentDispatchAuthority(
            profile="orchestrator",
            task_contract="orchestrate",
            scope=AgentDispatchScope(
                run_truth_scope=list(request.run_truth_scope or ()),
                episode_id=request.episode_id,
                patch_kind="work",
            ),
        )
    else:
        authority = resolver(kind, request)
    if kind == "episode_report":
        if not isinstance(request, EpisodeReportRunRequest):
            raise TypeError("episode_report dispatch requires an EpisodeReportRunRequest")
        if authority is not None:
            raise ValueError(
                "Authority refused action 'dispatch': an episode report has no graph "
                "authority binding."
            )
        return None
    if kind == "auto_research":
        if not isinstance(request, AutoResearchRunRequest):
            raise TypeError("auto_research dispatch requires an AutoResearchRunRequest")
        if authority is None:
            raise ValueError(
                "Authority refused action 'dispatch': the Auto-research actor has no "
                "authority binding."
            )
    elif authority is None:
        raise ValueError("Authority refused action 'dispatch': the task has no authority binding.")
    assert authority is not None
    require_dispatch(authority)

    if kind == "auto_research":
        assert isinstance(request, AutoResearchRunRequest)
        if operation_id is None:
            raise ValueError(
                "Authority refused action 'dispatch': Auto-research admission has no operation id."
            )
        actor_operation_id = request.actor_operation_id
        if parent is None:
            if (
                request.role != "orchestrator"
                or actor_operation_id != operation_id
                or request.wake_cause is not None
            ):
                raise ValueError(
                    "Authority refused action 'dispatch': an Auto-research root must be its "
                    "sole orchestrator actor."
                )
            return authority

        stored_parent = store.agent_task(parent.operation_id)
        if stored_parent is None:
            raise ValueError(
                "Authority refused action 'dispatch': the Auto-research parent is missing."
            )
        if (
            stored_parent.project_id != project_id
            or stored_parent.kind != "auto_research"
            or stored_parent.episode_id != request.episode_id
        ):
            raise ValueError(
                "Authority refused action 'dispatch': an Auto-research continuation must "
                "preserve its parent project and episode."
            )
        binding = store.auto_research_actor_binding(parent.operation_id)
        if actor_operation_id == operation_id:
            if (
                request.role != "worker"
                or binding.role != "orchestrator"
                or request.wake_cause is not None
            ):
                raise ValueError(
                    "Authority refused action 'dispatch': only the orchestrator may seat one "
                    "new ordinary worker actor."
                )
            return authority
        if actor_operation_id != binding.actor_operation_id or request.role != binding.role:
            raise ValueError(
                "Authority refused action 'dispatch': an Auto-research continuation cannot "
                "change its canonical actor or role."
            )
        origin = store.agent_task(binding.actor_operation_id)
        if origin is None:
            raise ValueError(
                "Authority refused action 'dispatch': the canonical Auto-research actor is missing."
            )
        if origin.dispatch_authority is None:
            if continuation not in {"resume", "retry"}:
                raise ValueError(
                    "Authority refused action 'dispatch': the canonical Auto-research actor "
                    "has no durable authority binding."
                )
            # A pre-authority Auto-research allocation remains recoverable. Its recovery is
            # still checked against today's closed profile contract before launch.
            return authority
        if authority != origin.dispatch_authority:
            raise ValueError(
                "Authority refused action 'dispatch': an Auto-research continuation cannot "
                "change its canonical actor's authority binding."
            )
        return origin.dispatch_authority

    if parent is not None:
        stored_parent = store.agent_task(parent.operation_id)
        if stored_parent is None:
            raise ValueError(
                "Authority refused action 'dispatch': the continuation parent is missing."
            )
        if stored_parent.project_id != project_id or stored_parent.kind != kind:
            raise ValueError(
                "Authority refused action 'dispatch': a continuation must preserve its "
                "parent's project and task kind."
            )
        # A parent recorded before dispatch authority existed carries none. The
        # continuation still resolved and gated its own authority above; there is
        # simply no earlier binding to hold it to.
        if (
            stored_parent.dispatch_authority is not None
            and authority != stored_parent.dispatch_authority
        ):
            raise ValueError(
                "Authority refused action 'dispatch': a continuation cannot change its "
                "parent's authority binding."
            )
    return authority


def skill_update(
    skills: SkillSelection | None,
    *,
    mode: Literal["python", "json"] = "python",
) -> dict[str, object]:
    """Refresh a continued attempt's recorded packages with what it will stage.

    Every launch re-resolves the selected ids, so a record that kept the earlier
    attempt's versions would misreport an upgraded package.
    """

    if skills is None:
        return {}
    if mode == "json":
        return skills.model_dump(mode="json")
    return {
        "workflow_ids": list(skills.workflow_ids),
        "skill_ids": list(skills.skill_ids),
        "resolved_skill_packages": list(skills.resolved_skill_packages),
    }
