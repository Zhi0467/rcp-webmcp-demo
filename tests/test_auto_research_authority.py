from __future__ import annotations

import json

import pytest

from rcp.agents.schema import (
    agent_output_schema,
    parse_agent_patch_json,
    prepare_agent_patch,
)
from rcp.core.authority import (
    CREATE_AMBIGUITY,
    DECIDE_DECISION,
    ORCHESTRATOR_AGENT_GRAPH_ACTIONS,
    ORDINARY_AGENT_GRAPH_ACTIONS,
    RESOLVE_PROPOSAL,
    SET_COVERAGE,
    SET_ONTOLOGY,
    SET_PROJECT_TRUTH_SCOPE,
    SET_STANDING,
    UPDATE_PROTECTED_EPISTEMIC,
    UPSERT_GLOSSARY,
    AgentDispatchAuthority,
    AgentDispatchScope,
    AgentTaskAuthority,
    operation_actions,
    require_apply,
    require_dispatch,
)
from rcp.core.models import (
    Decision,
    Evidence,
    GraphState,
    Patch,
    ResearchQuestion,
)
from rcp.core.transition_models import GraphTargetRef
from rcp.core.validation.patch import validate_patch

from .helpers import fabricated_authorizer, seated_on_every_project


def _orchestrator_patch(*ops: dict, agent_action: str | None = None) -> Patch:
    return Patch(
        kind="work",
        author="agent",
        producer="agent",
        summary="Advanced Auto-research.",
        ops=list(ops),
        run_truth_scope=["repo-a"],
        authorized_by=fabricated_authorizer("Auto-research owner"),
        profile="orchestrator",
        task_id="episode-turn-one",
        agent_action=agent_action,
    )


def _report(state: GraphState, patch: Patch):
    return validate_patch(state, patch, ["repo-a"])


def _rejection_codes(state: GraphState, patch: Patch) -> set[str]:
    return {message.code for message in _report(state, patch).messages if message.level == "reject"}


def test_orchestrator_is_one_closed_profile_contract_and_episode_scope() -> None:
    scope = AgentDispatchScope(
        run_truth_scope=["repo-a"],
        episode_id="episode-one",
        patch_kind="work",
    )
    authority = AgentDispatchAuthority(
        profile="orchestrator",
        task_contract="orchestrate",
        scope=scope,
    )

    assert require_dispatch(authority) == authority
    assert (
        require_apply(
            AgentTaskAuthority(
                operation_id="episode-turn-one",
                project_id="project-one",
                apply_target=GraphTargetRef(),
                authorized_by=fabricated_authorizer("Auto-research owner"),
                dispatch_authority=authority,
            ),
            _orchestrator_patch(),
            is_project_member=seated_on_every_project,
        )
        == authority
    )
    task = AgentTaskAuthority(
        operation_id="episode-turn-one",
        project_id="project-one",
        apply_target=GraphTargetRef(),
        authorized_by=fabricated_authorizer("Auto-research owner"),
        dispatch_authority=authority,
    )
    with pytest.raises(ValueError, match="Patch profile does not match"):
        require_apply(
            task,
            _orchestrator_patch().model_copy(update={"profile": "ordinary"}),
            is_project_member=seated_on_every_project,
        )

    for profile, contract in (
        ("ordinary", "orchestrate"),
        ("orchestrator", "work_auto"),
    ):
        with pytest.raises(ValueError, match="does not permit task contract"):
            require_dispatch(
                AgentDispatchAuthority(
                    profile=profile,
                    task_contract=contract,
                    scope=scope,
                )
            )

    with pytest.raises(ValueError, match="orchestrate requires an exact episode"):
        require_dispatch(
            AgentDispatchAuthority(
                profile="orchestrator",
                task_contract="orchestrate",
                scope=AgentDispatchScope(run_truth_scope=["repo-a"], patch_kind="work"),
            )
        )


def test_auto_research_worker_has_episode_scope_but_no_chat_or_control_scope() -> None:
    worker = AgentDispatchAuthority(
        profile="ordinary",
        task_contract="work_auto",
        scope=AgentDispatchScope(
            run_truth_scope=["repo-a"],
            episode_id="episode-one",
            patch_kind="work",
        ),
    )

    assert require_dispatch(worker) == worker
    with pytest.raises(ValueError, match="episode worker requires"):
        require_dispatch(
            worker.model_copy(
                update={
                    "scope": AgentDispatchScope(
                        run_truth_scope=["repo-a"],
                        episode_id="episode-one",
                        chat_scope="node",
                        chat_id="worker-chat",
                        node_id="exp/seat",
                        patch_kind="work",
                    )
                }
            )
        )


def test_orchestrator_action_table_adds_only_decision_and_standing() -> None:
    assert (
        ORDINARY_AGENT_GRAPH_ACTIONS
        | {
            DECIDE_DECISION,
            SET_STANDING,
        }
        == ORCHESTRATOR_AGENT_GRAPH_ACTIONS
    )
    assert {
        UPDATE_PROTECTED_EPISTEMIC,
        RESOLVE_PROPOSAL,
        SET_COVERAGE,
        SET_PROJECT_TRUTH_SCOPE,
        SET_ONTOLOGY,
        UPSERT_GLOSSARY,
        CREATE_AMBIGUITY,
    }.isdisjoint(ORCHESTRATOR_AGENT_GRAPH_ACTIONS)


def test_orchestrator_directly_decides_and_judges_action_layer_nodes() -> None:
    state = GraphState(
        project_truth_scope=["repo-a"],
        nodes={
            "dec/budget": Decision(
                id="dec/budget",
                type="decision",
                title="Evaluation budget",
                question="Which budget?",
                options=["small", "large"],
                status="ready",
            ),
            "ev/result": Evidence(
                id="ev/result",
                type="evidence",
                title="Result",
                observation="The larger evaluation was diagnostic.",
                origin="internal_run",
            ),
        },
    )
    patch = _orchestrator_patch(
        {
            "op": "update_nodes",
            "nodes": [
                {
                    "id": "dec/budget",
                    "changes": {"selected_option": "large", "status": "decided"},
                }
            ],
        },
        {"op": "set_standing", "node_id": "ev/result", "standing": "accepted"},
        agent_action="decision_choice",
    )

    assert not _report(state, patch).rejected


def test_orchestrator_decision_outcome_requires_a_real_declared_action() -> None:
    state = GraphState(
        project_truth_scope=["repo-a"],
        nodes={
            "dec/budget": Decision(
                id="dec/budget",
                type="decision",
                title="Evaluation budget",
                question="Which budget?",
                options=["small", "large"],
                status="ready",
            )
        },
    )
    update = {
        "op": "update_nodes",
        "nodes": [
            {
                "id": "dec/budget",
                "changes": {"selected_option": "large", "status": "decided"},
            }
        ],
    }

    assert "missing-decision-action" in _rejection_codes(state, _orchestrator_patch(update))
    assert "unused-agent-action" in _rejection_codes(
        state,
        _orchestrator_patch(
            {"op": "update_nodes", "nodes": [{"id": "dec/budget", "changes": {"title": "B"}}]},
            agent_action="decision_choice",
        ),
    )


def test_orchestrator_creates_beliefs_but_cannot_move_existing_beliefs() -> None:
    empty = GraphState(project_truth_scope=["repo-a"])
    creation = _orchestrator_patch(
        {
            "op": "create_nodes",
            "nodes": [
                {
                    "id": "rq/new-question",
                    "type": "research_question",
                    "title": "New question",
                    "question": "What explains the result?",
                },
                {
                    "id": "hyp/new-explanation",
                    "type": "hypothesis",
                    "title": "New explanation",
                    "statement": "Replanning explains the result.",
                },
            ],
        }
    )
    assert not _report(empty, creation).rejected

    state = GraphState(
        project_truth_scope=["repo-a"],
        nodes={
            "rq/existing": ResearchQuestion(
                id="rq/existing",
                type="research_question",
                title="Existing question",
                question="What already exists?",
            )
        },
    )
    assert "graph-action-refused" in _rejection_codes(
        state,
        _orchestrator_patch(
            {
                "op": "update_nodes",
                "nodes": [{"id": "rq/existing", "changes": {"question": "A changed question?"}}],
            }
        ),
    )
    assert "graph-action-refused" in _rejection_codes(
        state,
        _orchestrator_patch(
            {"op": "set_standing", "node_id": "rq/existing", "standing": "accepted"}
        ),
    )


@pytest.mark.parametrize(
    ("node", "node_id"),
    (
        (
            {
                "id": "rq/new-question",
                "type": "research_question",
                "title": "New question",
                "question": "What explains the result?",
            },
            "rq/new-question",
        ),
        (
            {
                "id": "hyp/new-explanation",
                "type": "hypothesis",
                "title": "New explanation",
                "statement": "Replanning explains the result.",
            },
            "hyp/new-explanation",
        ),
    ),
)
def test_orchestrator_cannot_set_standing_on_belief_created_in_same_patch(
    node: dict[str, str],
    node_id: str,
) -> None:
    state = GraphState(project_truth_scope=["repo-a"])
    patch = _orchestrator_patch(
        {"op": "create_nodes", "nodes": [node]},
        {"op": "set_standing", "node_id": node_id, "standing": "accepted"},
    )

    assert operation_actions(state, patch, patch.ops[1]) == frozenset({UPDATE_PROTECTED_EPISTEMIC})
    assert "graph-action-refused" in _rejection_codes(state, patch)


@pytest.mark.parametrize(
    "node",
    (
        {
            "id": "dec/new-route",
            "type": "decision",
            "title": "New route",
            "question": "Which route?",
            "options": ["matched", "shifted"],
        },
        {
            "id": "exp/new-evaluation",
            "type": "experiment",
            "title": "New evaluation",
            "objective": "Evaluate the route.",
        },
        {
            "id": "blk/new-input",
            "type": "blocker",
            "title": "New input blocker",
            "description": "The input is missing.",
        },
        {
            "id": "ev/new-result",
            "type": "evidence",
            "title": "New result",
            "observation": "The route exposed the failure mode.",
            "origin": "internal_run",
        },
    ),
)
def test_orchestrator_can_set_standing_on_action_node_created_in_same_patch(
    node: dict[str, object],
) -> None:
    state = GraphState(project_truth_scope=["repo-a"])
    node_id = str(node["id"])
    patch = _orchestrator_patch(
        {"op": "create_nodes", "nodes": [node]},
        {"op": "set_standing", "node_id": node_id, "standing": "accepted"},
    )

    assert operation_actions(state, patch, patch.ops[1]) == frozenset({SET_STANDING})
    assert not _report(state, patch).rejected


def test_orchestrator_schema_is_elevated_without_widening_ordinary_schema() -> None:
    ordinary = json.dumps(agent_output_schema(), sort_keys=True)
    orchestrator = json.dumps(agent_output_schema(profile="orchestrator"), sort_keys=True)
    assert '"set_standing"' not in ordinary
    assert '"agent_action"' not in ordinary
    assert '"set_standing"' in orchestrator
    assert '"agent_action"' in orchestrator

    raw = json.dumps(
        {
            "summary": "Chose the governed option.",
            "agent_action": "decision_choice",
            "ops": [
                {
                    "op": "update_nodes",
                    "nodes": [
                        {
                            "id": "dec/budget",
                            "changes": {"selected_option": "large", "status": "decided"},
                        }
                    ],
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="graph operation schema"):
        parse_agent_patch_json(raw)
    draft = parse_agent_patch_json(raw, profile="orchestrator")
    prepared = prepare_agent_patch(
        draft,
        kind="work",
        run_truth_scope=["repo-a"],
        profile="orchestrator",
    )
    assert prepared.agent_action == "decision_choice"
    assert prepared.profile is None
