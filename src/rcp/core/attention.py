"""Backend-owned graph attention membership projections."""

from __future__ import annotations

from rcp.core.models import Blocker, Decision, GraphState, ProjectNode, Standing
from rcp.core.transition_models import GraphAttentionProjection


def decision_awaits_choice(node: ProjectNode) -> bool:
    """Whether one canonical Decision belongs in human attention."""

    return isinstance(node, Decision) and node.status in {"ready", "revisit"}


def project_graph_attention(state: GraphState) -> GraphAttentionProjection:
    """Project the exact canonical memberships used by Inbox and Runs."""

    for node_id, node in state.nodes.items():
        if node_id != node.id:
            raise ValueError(
                f"Graph node mapping key {node_id!r} does not match embedded id {node.id!r}."
            )
    for proposal_id, proposal in state.proposals.items():
        if proposal_id != proposal.id:
            raise ValueError(
                "Graph Proposal mapping key "
                f"{proposal_id!r} does not match embedded id {proposal.id!r}."
            )
    return GraphAttentionProjection(
        pending_proposal_ids=sorted(
            proposal_id
            for proposal_id, proposal in state.proposals.items()
            if proposal.status == "pending"
        ),
        decisions_awaiting_choice_ids=sorted(
            node.id for node in state.nodes.values() if decision_awaits_choice(node)
        ),
        open_blocker_ids=sorted(
            node.id
            for node in state.nodes.values()
            if isinstance(node, Blocker)
            and node.status == "open"
            and node.standing == Standing.ASSERTED
        ),
    )


__all__ = ["decision_awaits_choice", "project_graph_attention"]
