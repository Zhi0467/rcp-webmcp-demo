from __future__ import annotations

from typing import Literal

from rcp.control import ExperimentControlState
from rcp.core.models import Experiment
from rcp.service import ProjectService, RunRequest


def experiment_start_message(message: str | None, node_id: str) -> str:
    """Preserve an explicit goal and use the canonical fallback only for blank input."""

    if message is not None and message.strip():
        return message
    return f"Begin a bounded Experiment-loop episode for {node_id}."


def resolve_experiment_node_work_request(
    service: ProjectService,
    request: RunRequest,
) -> RunRequest:
    """Resolve one Experiment turn from the current human node-Work profile."""

    profile = service.resolve_agent_profile("node_chat")
    resolved = request.model_copy(
        update={
            "provider": profile.provider,
            "model": profile.model,
            "reasoning": profile.reasoning,
            "run_on": profile.run_on,
            "run_truth_scope": list(
                request.run_truth_scope or service.manifest.agent.default_run_truth_scope
            ),
        }
    )
    skill_resolved = service.resolve_skill_request(resolved)
    if not isinstance(skill_resolved, RunRequest):
        raise TypeError("Experiment skill resolution changed the task request type.")
    return skill_resolved


def fresh_experiment_run_request(
    service: ProjectService,
    supplied: RunRequest,
    *,
    node: Experiment,
    state_revision: int,
    control: ExperimentControlState,
    episode_id: str,
    invocation_ceiling: int | None = None,
    trigger: Literal["experiment_run", "orchestrator"] = "experiment_run",
) -> RunRequest:
    """Build the immutable invocation-one contract for a fresh Experiment episode."""

    ceiling = node.invocation_ceiling if invocation_ceiling is None else invocation_ceiling
    if ceiling < 1:
        raise ValueError("Experiment invocation limit must be positive.")
    request = supplied.model_copy(
        update={
            "chat_scope": "node",
            "node_id": node.id,
            "message": experiment_start_message(supplied.message, node.id),
            "session_id": None,
            "mode": "work",
            "trigger": trigger,
            "patch_kind": "experiment_loop",
            "control_node_id": node.id,
            "control_revision": state_revision,
            "control_episode_id": episode_id,
            "control_invocation": 1,
            "control_invocation_ceiling": ceiling,
            "control_decision_bundle": control.governing_decisions,
            "control_completion_criteria": list(node.completion_criteria),
            "watcher_ids": [],
        }
    )
    return resolve_experiment_node_work_request(service, request)
