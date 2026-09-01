from __future__ import annotations

from typing import cast

from rcp.config import AgentExecutionProfile, AgentSurface
from rcp.providers import profile_for
from rcp.runs.auto_research import AutoResearchRunRequest
from rcp.runs.task_policy import AgentTaskRequest
from rcp.service import ProjectService, RunRequest
from rcp.storage import AgentTaskKind, AppStore


def _resolved_graph_request(
    service: ProjectService,
    kind: AgentTaskKind,
    request: RunRequest,
) -> RunRequest:
    surface: AgentSurface = kind
    profile = service.resolve_agent_profile(
        surface,
        provider=request.provider,
        model=request.model,
        reasoning=request.reasoning,
        run_on=request.run_on,
    )
    resolved = request.model_copy(
        update={
            "provider": profile.provider,
            # An empty string is the explicit provider-default sentinel. Once a
            # request is resolved it must not collapse back to None, which means
            # "inherit the current surface setting" on a later continuation.
            "model": profile.model,
            "reasoning": profile.reasoning,
            "run_on": profile.run_on,
            "run_truth_scope": list(
                request.run_truth_scope or service.manifest.agent.default_run_truth_scope
            ),
        }
    )
    result = service.resolve_skill_request(resolved)
    assert isinstance(result, RunRequest)
    return result


def _resolved_auto_research_request(
    service: ProjectService,
    request: AutoResearchRunRequest,
) -> AutoResearchRunRequest:
    if (
        request.provider is None
        or request.model is None
        or request.reasoning is None
        or request.run_on is None
        or request.run_truth_scope is None
    ):
        raise ValueError("Auto-research recovery requires its exact pinned execution profile.")
    profile_for(request.provider)
    if request.run_on not in service.manifest.machine_map:
        raise ValueError(f"unknown execution machine: {request.run_on}")
    skill_resolved = service.resolve_skill_request(cast(RunRequest, request))
    if not isinstance(skill_resolved, AutoResearchRunRequest):
        raise TypeError("Auto-research skill resolution changed the task request type.")
    return skill_resolved


def resolved_agent_surface(
    store: AppStore,
    kind: AgentTaskKind,
    request: AgentTaskRequest,
    *,
    parent_operation_id: str | None = None,
) -> AgentExecutionProfile:
    """Which project agent profile one task invocation executes under.

    This lives beside `_resolved_graph_request` because it is the same policy:
    which profile a request belongs to. Chat surfaces answer from `chat_scope`,
    the way every run task owner does, so no caller can pin one field of a
    profile while the owner runs a different one.
    """

    if kind in {"seed", "refresh", "paper_coach"}:
        return cast(AgentExecutionProfile, kind)
    if kind in {"node_chat", "project_chat"}:
        if not isinstance(request, RunRequest):
            raise TypeError("A chat task requires its pinned run request.")
        return "project_chat" if request.chat_scope == "project" else "node_chat"
    if kind == "branch_merge":
        return "orchestrator"
    if kind == "auto_research":
        if not isinstance(request, AutoResearchRunRequest):
            raise TypeError("An Auto-research task requires its pinned actor request.")
        return "orchestrator" if request.role == "orchestrator" else "node_chat"
    if kind == "episode_report":
        parent = store.agent_task(parent_operation_id or "")
        if parent is None:
            raise ValueError("The episode report lost its concluding provider task.")
        if parent.kind == "auto_research":
            return resolved_agent_surface(
                store,
                "auto_research",
                AutoResearchRunRequest.model_validate(parent.request),
            )
        if parent.kind in {"node_chat", "project_chat"}:
            return resolved_agent_surface(
                store,
                parent.kind,
                RunRequest.model_validate(parent.request),
            )
        raise ValueError("The episode report has no current provider profile.")
    raise ValueError(f"Unknown provider task kind: {kind}")


__all__ = [
    "_resolved_auto_research_request",
    "_resolved_graph_request",
    "resolved_agent_surface",
]
