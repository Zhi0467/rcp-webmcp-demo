from __future__ import annotations

from pathlib import Path

import pytest

from rcp.agents import AgentEvent
from rcp.control import derive_experiment_control_state
from rcp.core.authority import (
    AgentDispatchAuthority,
    AgentDispatchScope,
    AgentTaskAuthority,
)
from rcp.core.materialize import apply_valid_patch, prepare_patch_bookkeeping
from rcp.core.models import (
    Blocker,
    Decision,
    Edge,
    Evidence,
    Experiment,
    GraphState,
    Patch,
    ResearchQuestion,
)
from rcp.core.transition_models import GraphTargetRef
from rcp.core.validation import validate_patch
from rcp.history import HistoryManager, PatchRejected
from rcp.paper import PaperService
from rcp.runs.auto_research import AutoResearchCommandDispatcher, AutoResearchRunRequest
from rcp.runs.tasks.auto_research_stream import stream_auto_research_worker_run
from rcp.service import ProjectService, ProposalDecisionRequest
from rcp.storage import AppStore

from .helpers import fabricated_authorizer, seated_on_every_project
from .test_auto_research_commands import _Effects, _setup_auto_research, _spawn_request
from .test_auto_research_stream import (
    _dispatcher,
    _events,
    _execution,
    _service,
    _WorkerLauncher,
)
from .test_auto_research_stream import (
    _setup_auto_research as _setup_stream_auto_research,
)

_SCOPE = ["repo-a"]


def _orchestrator_patch(
    *ops: dict[str, object],
    revision: int = 2,
    agent_action: str | None = None,
    authorized_by=None,
) -> Patch:
    authorized_by = authorized_by or fabricated_authorizer("Auto-researcher")
    return Patch(
        revision=revision,
        kind="work",
        author="agent",
        producer="agent",
        summary="Advanced the Auto-research episode.",
        ops=list(ops),
        run_truth_scope=_SCOPE,
        repositories_read=_SCOPE,
        source_operation_id="auto-research-root",
        agent_action=agent_action,
        authorized_by=authorized_by,
        profile="orchestrator",
        task_id="auto-research-root",
        episode_id="episode",
    )


def _reject_codes(state: GraphState, patch: Patch) -> set[str]:
    return {
        message.code
        for message in validate_patch(state, patch, _SCOPE).messages
        if message.level == "reject"
    }


def _action_state() -> GraphState:
    retained = (
        Decision(
            id="dec/route",
            type="decision",
            title="Evaluation route",
            question="Which route should govern the evaluation?",
            options=["matched", "shifted"],
            status="ready",
            updated_rev=1,
        ),
        Experiment(
            id="exp/evaluate",
            type="experiment",
            title="Evaluate",
            objective="Evaluate the selected route.",
            status="designing",
        ),
        Blocker(
            id="blk/input",
            type="blocker",
            title="Missing input",
            description="The input must be located.",
            status="open",
        ),
        Evidence(
            id="ev/trace",
            type="evidence",
            title="Trace",
            observation="The initial trace is incomplete.",
            origin="internal_run",
        ),
    )
    expendable = (
        Decision(
            id="dec/discard",
            type="decision",
            title="Discarded decision",
            question="Is this still needed?",
            options=["yes", "no"],
        ),
        Experiment(
            id="exp/discard",
            type="experiment",
            title="Discarded experiment",
            objective="No longer needed.",
        ),
        Blocker(
            id="blk/discard",
            type="blocker",
            title="Discarded blocker",
            description="Already irrelevant.",
        ),
        Evidence(
            id="ev/discard",
            type="evidence",
            title="Discarded evidence",
            observation="A superseded observation.",
            origin="internal_run",
        ),
    )
    nodes = {node.id: node for node in (*retained, *expendable)}
    return GraphState(
        revision=1,
        project_truth_scope=_SCOPE,
        nodes=nodes,
        edges={
            "edge/old-signal": Edge(
                id="edge/old-signal",
                source="ev/trace",
                target="dec/route",
                relation="informs",
            )
        },
    )


def test_s77_orchestrator_directly_controls_action_layer_and_creates_beliefs() -> None:
    state = _action_state()
    patch = _orchestrator_patch(
        {
            "op": "create_nodes",
            "nodes": [
                {
                    "id": "rq/new-route",
                    "type": "research_question",
                    "title": "New route question",
                    "question": "Which route explains the shift?",
                },
                {
                    "id": "hyp/new-route",
                    "type": "hypothesis",
                    "title": "New route hypothesis",
                    "statement": "The shifted route explains the result.",
                },
                {
                    "id": "ev/new-result",
                    "type": "evidence",
                    "title": "New result",
                    "observation": "The shifted route exposed the failure mode.",
                    "origin": "internal_run",
                },
            ],
        },
        {
            "op": "create_edges",
            "edges": [
                {
                    "id": "edge/new-hypothesis",
                    "source": "rq/new-route",
                    "target": "hyp/new-route",
                    "relation": "has_hypothesis",
                },
                {
                    "id": "edge/new-result",
                    "source": "exp/evaluate",
                    "target": "ev/new-result",
                    "relation": "produces",
                },
            ],
        },
        {
            "op": "update_nodes",
            "nodes": [
                {
                    "id": "dec/route",
                    "changes": {
                        "selected_option": "shifted",
                        "status": "decided",
                        "rationale": "The new trace discriminates the routes.",
                    },
                },
                {
                    "id": "exp/evaluate",
                    "changes": {
                        "status": "analyzing",
                        "current_summary": "The diagnostic run completed.",
                    },
                },
                {
                    "id": "blk/input",
                    "changes": {
                        "status": "resolved",
                        "recommended_action": "Use the recovered input.",
                    },
                },
                {
                    "id": "ev/trace",
                    "changes": {
                        "observation": "The complete trace isolates the shifted route.",
                        "interpretation": "The route is diagnostic.",
                    },
                },
            ],
        },
        *(
            {"op": "set_standing", "node_id": node_id, "standing": "accepted"}
            for node_id in ("dec/route", "exp/evaluate", "blk/input", "ev/trace")
        ),
        {"op": "remove_edges", "edge_ids": ["edge/old-signal"]},
        {
            "op": "remove_nodes",
            "node_ids": ["dec/discard", "exp/discard", "blk/discard", "ev/discard"],
        },
        agent_action="decision_choice",
    )

    report = validate_patch(state, patch, _SCOPE)
    assert not report.rejected, [message.message for message in report.messages]

    result = apply_valid_patch(state, patch)
    assert result.nodes["dec/route"].status == "decided"
    assert result.nodes["dec/route"].selected_option == "shifted"
    assert result.nodes["exp/evaluate"].status == "analyzing"
    assert result.nodes["blk/input"].status == "resolved"
    assert result.nodes["ev/trace"].observation.startswith("The complete trace")
    assert all(
        result.nodes[node_id].standing == "accepted"
        for node_id in ("dec/route", "exp/evaluate", "blk/input", "ev/trace")
    )
    assert {"rq/new-route", "hyp/new-route", "ev/new-result"} <= set(result.nodes)
    assert {"edge/new-hypothesis", "edge/new-result"} <= set(result.edges)
    assert "edge/old-signal" not in result.edges
    assert {
        "dec/discard",
        "exp/discard",
        "blk/discard",
        "ev/discard",
    }.isdisjoint(result.nodes)


def _protected_seed_patch() -> Patch:
    return Patch(
        kind="refresh",
        author="agent",
        producer="agent",
        summary="Prepared the existing protected beliefs.",
        run_truth_scope=_SCOPE,
        repositories_read=_SCOPE,
        ops=[
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": "rq/existing",
                        "type": "research_question",
                        "title": "Existing question",
                        "question": "What explains the shift?",
                    },
                    {
                        "id": "hyp/existing",
                        "type": "hypothesis",
                        "title": "Existing hypothesis",
                        "statement": "Replanning explains the shift.",
                    },
                    {
                        "id": "hyp/retire",
                        "type": "hypothesis",
                        "title": "Retirement candidate",
                        "statement": "The obsolete mechanism explains the shift.",
                    },
                    {
                        "id": "ev/cause",
                        "type": "evidence",
                        "title": "Causal result",
                        "observation": "The replanning intervention restored performance.",
                        "origin": "internal_run",
                    },
                ],
            },
            {
                "op": "create_edges",
                "edges": [
                    {
                        "id": "edge/protected",
                        "source": "rq/existing",
                        "target": "hyp/existing",
                        "relation": "has_hypothesis",
                    },
                    {
                        "id": "edge/support",
                        "source": "ev/cause",
                        "target": "hyp/existing",
                        "relation": "supports",
                        "assessment": {
                            "relevance": "direct",
                            "weight": "strong",
                        },
                    },
                ],
            },
        ],
    )


def _protected_proposals(authorizer) -> Patch:
    def proposal(proposal_id: str, title: str, operation: dict[str, object]) -> dict:
        return {
            "id": proposal_id,
            "title": title,
            "card": {"decision_needed": f"Approve or reject {title.lower()}."},
            "ops": [operation],
        }

    return _orchestrator_patch(
        {
            "op": "create_proposals",
            "proposals": [
                proposal(
                    "prop/content",
                    "Clarify the existing question",
                    {
                        "op": "update_nodes",
                        "intent": "content_change",
                        "nodes": [
                            {
                                "id": "rq/existing",
                                "changes": {"question": "What explains the repeated shift?"},
                            }
                        ],
                    },
                ),
                proposal(
                    "prop/question-status",
                    "Answer the existing question",
                    {
                        "op": "update_nodes",
                        "intent": "content_change",
                        "nodes": [{"id": "rq/existing", "changes": {"status": "answered"}}],
                    },
                ),
                proposal(
                    "prop/hypothesis-status",
                    "Support the existing hypothesis",
                    {
                        "op": "update_nodes",
                        "intent": "status_change",
                        "nodes": [
                            {
                                "id": "hyp/existing",
                                "changes": {"status": "supported"},
                                "cause": {"kind": "evidence_edge", "ref_id": "edge/support"},
                            }
                        ],
                    },
                ),
                proposal(
                    "prop/relation",
                    "Detach the existing hypothesis",
                    {
                        "op": "remove_edges",
                        "intent": "protected_relation_change",
                        "edge_ids": ["edge/protected"],
                    },
                ),
                proposal(
                    "prop/removal",
                    "Remove the obsolete hypothesis",
                    {
                        "op": "remove_nodes",
                        "intent": "removal",
                        "node_ids": ["hyp/retire"],
                    },
                ),
            ],
        },
        authorized_by=authorizer,
    )


def test_s77_protected_changes_wait_for_human_judgment(manifest, tmp_path: Path) -> None:
    history = HistoryManager(manifest)
    history.append(_protected_seed_patch())
    original = history.state()

    direct_attempts = (
        {
            "op": "update_nodes",
            "nodes": [{"id": "rq/existing", "changes": {"question": "A direct rewrite?"}}],
        },
        {
            "op": "update_nodes",
            "nodes": [
                {
                    "id": "hyp/existing",
                    "changes": {"status": "supported"},
                    "cause": {"kind": "evidence_edge", "ref_id": "edge/support"},
                }
            ],
        },
        {"op": "set_standing", "node_id": "rq/existing", "standing": "accepted"},
        {"op": "remove_edges", "edge_ids": ["edge/protected"]},
        {"op": "remove_nodes", "node_ids": ["hyp/retire"]},
    )
    for operation in direct_attempts:
        assert "graph-action-refused" in _reject_codes(
            original,
            _orchestrator_patch(operation, revision=original.revision + 1),
        )
    assert history.state() == original

    authorizer = fabricated_authorizer("Auto-researcher")
    authority = AgentTaskAuthority(
        operation_id="auto-research-root",
        project_id="project",
        apply_target=GraphTargetRef(),
        authorized_by=authorizer,
        dispatch_authority=AgentDispatchAuthority(
            profile="orchestrator",
            task_contract="orchestrate",
            scope=AgentDispatchScope(
                run_truth_scope=_SCOPE,
                episode_id="episode",
                patch_kind="work",
            ),
        ),
        episode_id="episode",
    )
    history.project_id = "project"
    history.require_attribution = True
    history.agent_authority_resolver = lambda _project_id, _operation_id: authority
    history.project_membership_check = seated_on_every_project

    raised, result = history.append(_protected_proposals(authorizer))
    assert raised.profile == "orchestrator"
    assert raised.task_id == "auto-research-root"
    assert raised.authorized_by == authorizer
    assert {proposal.status for proposal in result.state.proposals.values()} == {"pending"}
    assert {proposal.created_by for proposal in result.state.proposals.values()} == {"agent"}
    assert {proposal.created_by_operation_id for proposal in result.state.proposals.values()} == {
        "auto-research-root"
    }
    assert result.state.nodes["rq/existing"].standing == "asserted"

    with pytest.raises(PatchRejected) as caught:
        history.append(
            _orchestrator_patch(
                {
                    "op": "resolve_proposals",
                    "resolutions": [{"id": "prop/content", "status": "approved"}],
                },
                revision=result.state.revision + 1,
                authorized_by=authorizer,
            ),
            discard_on_reject=True,
        )
    assert "graph-action-refused" in {
        message.code for message in caught.value.report.messages if message.level == "reject"
    }
    assert history.state().proposals["prop/content"].status == "pending"

    service = ProjectService(
        manifest,
        history,
        PaperService(manifest, AppStore(tmp_path / "paper.sqlite3")),
    )
    judged = service.decide_proposal(
        "prop/content",
        ProposalDecisionRequest(decision="approved"),
        authorized_by=authorizer,
    )
    assert judged.proposals["prop/content"].status == "approved"
    assert judged.proposals["prop/content"].resolved_by == "human"
    assert judged.nodes["rq/existing"].question == "What explains the repeated shift?"
    assert judged.nodes["rq/existing"].standing == "accepted"
    assert {
        judged.proposals[proposal_id].status
        for proposal_id in (
            "prop/question-status",
            "prop/hypothesis-status",
            "prop/relation",
            "prop/removal",
        )
    } == {"pending"}


def test_s77_worker_seating_boundary_does_not_narrow_decision_authority(tmp_path: Path) -> None:
    store, episode, root = _setup_auto_research(tmp_path)
    effects = _Effects(store, episode, root)
    dispatcher = AutoResearchCommandDispatcher(
        store,
        effects.bundle(),
        command_file_reader=lambda _filename, _max_bytes: (
            "Inspect the seated node and return the concrete result."
        ),
    )

    for index, (node_type, node_id) in enumerate(
        (("decision", "dec/route"), ("research_question", "rq/existing")), start=1
    ):
        effects.seat_type = node_type
        response = dispatcher.dispatch(
            root.operation_id,
            _spawn_request(str(index) * 32, key=f"refuse-{node_type}", seat_node_id=node_id),
        )
        assert response.status == "invalid"
        assert "Experiments and Blockers" in (response.message or "")

    for index, (node_type, node_id) in enumerate(
        (("experiment", "exp/evaluate"), ("blocker", "blk/input")), start=3
    ):
        effects.seat_type = node_type
        response = dispatcher.dispatch(
            root.operation_id,
            _spawn_request(str(index) * 32, key=f"seat-{node_type}", seat_node_id=node_id),
        )
        assert response.status == "ok"
        worker = store.agent_task(str(response.result["worker_id"]))
        assert worker is not None
        assert worker.kind == "node_chat"
        assert worker.request["node_id"] == node_id
        assert worker.request["control_node_id"] is None
        assert "scope" not in worker.request

    state = GraphState(
        project_truth_scope=_SCOPE,
        nodes={
            "dec/route": Decision(
                id="dec/route",
                type="decision",
                title="Route",
                question="Which route?",
                options=["matched", "shifted"],
                status="ready",
            )
        },
    )
    decision = _orchestrator_patch(
        {
            "op": "update_nodes",
            "nodes": [
                {
                    "id": "dec/route",
                    "changes": {"selected_option": "shifted", "status": "decided"},
                }
            ],
        },
        revision=1,
        agent_action="decision_choice",
    )
    assert not validate_patch(state, decision, _SCOPE).rejected


@pytest.mark.asyncio
async def test_s77_blocked_child_answer_is_preserved_without_graph_change(
    manifest,
    tmp_path: Path,
) -> None:
    service = _service(manifest, tmp_path)
    before = service.history.state().model_dump(mode="json")
    store, _episode, _root, worker = _setup_stream_auto_research(tmp_path / "store")

    def inspect_contract(contract: str, _workspace: Path) -> None:
        compact = " ".join(contract.split())
        assert "existing ResearchQuestion or Hypothesis" in compact
        assert "human judgment; never apply it directly" in compact
        assert "what failed" in compact

    class DifficultyLauncher(_WorkerLauncher):
        async def stream(self, provider, prompt, **kwargs):
            async for event in super().stream(provider, prompt, **kwargs):
                if event.event == "answer":
                    yield AgentEvent(
                        event="answer",
                        text=(
                            "I could not finish because the evidence contradicts the existing "
                            "Hypothesis; changing that belief requires human judgment."
                        ),
                    )
                else:
                    yield event

    events = await _events(
        stream_auto_research_worker_run(
            service,
            DifficultyLauncher(writer=inspect_contract),
            AutoResearchRunRequest.model_validate(worker.request),
            tmp_path / "data",
            _execution(store, worker),
            command_dispatcher=_dispatcher(store),
        )
    )

    answers = [event.text for event in events if event.event == "answer"]
    assert answers == [
        "I could not finish because the evidence contradicts the existing Hypothesis; "
        "changing that belief requires human judgment."
    ]
    assert events[-1].event == "done"
    assert service.history.state().model_dump(mode="json") == before


def test_s77_decision_unblocks_experiment_while_belief_review_remains_pending() -> None:
    state = GraphState(
        revision=1,
        project_truth_scope=_SCOPE,
        nodes={
            "rq/unrelated": ResearchQuestion(
                id="rq/unrelated",
                type="research_question",
                title="Unrelated question",
                question="What remains unresolved?",
            ),
            "dec/gate": Decision(
                id="dec/gate",
                type="decision",
                title="Experiment gate",
                question="Which configuration should run?",
                options=["small", "large"],
                status="ready",
                updated_rev=1,
            ),
            "exp/gated": Experiment(
                id="exp/gated",
                type="experiment",
                title="Gated experiment",
                objective="Run the selected configuration.",
            ),
            "blk/independent": Blocker(
                id="blk/independent",
                type="blocker",
                title="Independent blocker",
                description="A separate input must be recovered.",
            ),
        },
        edges={
            "edge/governed": Edge(
                id="edge/governed",
                source="exp/gated",
                target="dec/gate",
                relation="governed_by",
            )
        },
    )
    proposal_patch = prepare_patch_bookkeeping(
        state,
        _orchestrator_patch(
            {
                "op": "create_proposals",
                "proposals": [
                    {
                        "id": "prop/unrelated",
                        "title": "Clarify the unrelated question",
                        "card": {"decision_needed": "Approve or reject the clarification."},
                        "ops": [
                            {
                                "op": "update_nodes",
                                "intent": "content_change",
                                "nodes": [
                                    {
                                        "id": "rq/unrelated",
                                        "changes": {"question": "What still remains unresolved?"},
                                    }
                                ],
                            }
                        ],
                    }
                ],
            },
            revision=2,
        ),
    )
    assert not validate_patch(state, proposal_patch, _SCOPE).rejected
    state = apply_valid_patch(state, proposal_patch)
    assert not derive_experiment_control_state(state, "exp/gated").ready

    independent = _orchestrator_patch(
        {
            "op": "update_nodes",
            "nodes": [
                {
                    "id": "dec/gate",
                    "changes": {"selected_option": "large", "status": "decided"},
                },
                {
                    "id": "blk/independent",
                    "changes": {"status": "resolved"},
                },
            ],
        },
        {
            "op": "create_nodes",
            "nodes": [
                {
                    "id": "ev/independent",
                    "type": "evidence",
                    "title": "Independent evidence",
                    "observation": "The separate input was recovered.",
                    "origin": "internal_run",
                }
            ],
        },
        revision=3,
        agent_action="decision_choice",
    )
    report = validate_patch(state, independent, _SCOPE)
    assert not report.rejected, [message.message for message in report.messages]
    state = apply_valid_patch(state, independent)

    control = derive_experiment_control_state(state, "exp/gated")
    assert control.ready
    assert control.governing_decisions[0].selected_option == "large"
    assert state.nodes["blk/independent"].status == "resolved"
    assert "ev/independent" in state.nodes
    assert state.proposals["prop/unrelated"].status == "pending"
