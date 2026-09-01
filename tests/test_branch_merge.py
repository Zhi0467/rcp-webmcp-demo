from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from rcp.agents import AgentEvent
from rcp.agents.write_scope import ProjectWriteScope, WritableRepositoryRoot
from rcp.core.models import (
    AuthorizedHuman,
    Blocker,
    Edge,
    Experiment,
    GraphBranchMetadata,
    GraphState,
    Proposal,
    ResearchQuestion,
)
from rcp.core.transition_models import GraphHeadRef, GraphTargetRef
from rcp.core.transitions import GraphTransitionManager
from rcp.core.validation import ValidationReport
from rcp.history import RevisionConflict
from rcp.runs.branch_merge import (
    BranchMergeCandidateProblem,
    BranchMergeContext,
    BranchMergeEligibility,
    BranchMergeRunOutcome,
    BranchMergeSourceChanged,
    BranchMergeStage,
    branch_merge_can_resolve_without_patch,
    branch_merge_id,
    build_semantic_delta,
    classify_refreshed_context,
    detect_branch_merge_conflicts,
    parse_branch_merge_candidate,
    prepare_branch_merge_with_history,
    require_graph_only_merge_scope,
    semantic_delta_is_subsumed,
    stream_branch_merge_run,
)
from rcp.service import RunRequest


def _head(
    revision: int,
    *,
    branch_id: str | None = None,
    marker: str = "a",
) -> GraphHeadRef:
    return GraphHeadRef(
        target=(
            GraphTargetRef(kind="branch", branch_id=branch_id)
            if branch_id is not None
            else GraphTargetRef()
        ),
        revision=revision,
        transition_id=marker * 64 if revision else None,
    )


def _question(
    *,
    title: str = "Question",
    motivation: str = "Base motivation",
    status: str = "open",
    updated_rev: int = 2,
) -> ResearchQuestion:
    return ResearchQuestion(
        id="rq/merge",
        type="research_question",
        title=title,
        question="What should the project test?",
        motivation=motivation,
        status=status,
        created_rev=2,
        updated_rev=updated_rev,
    )


def _state(
    revision: int,
    *,
    question: ResearchQuestion | None = None,
    extra: ResearchQuestion | None = None,
) -> GraphState:
    nodes = {"rq/merge": question or _question(updated_rev=revision)}
    if extra is not None:
        nodes[extra.id] = extra
    return GraphState(
        revision=revision,
        project_truth_scope=["repo"],
        nodes=nodes,
    )


def _context(
    *,
    main_revision: int = 4,
    main_question: ResearchQuestion | None = None,
    branch_question: ResearchQuestion | None = None,
    branch_id: str | None = None,
    task_id: str = "merge-task",
) -> BranchMergeContext:
    branch_id = branch_id or str(uuid.uuid4())
    authorizer = AuthorizedHuman(
        space_id=str(uuid.uuid4()),
        user_id=str(uuid.uuid4()),
        display_name="Episode owner",
    )
    merge_dispatcher = AuthorizedHuman(
        space_id=str(uuid.uuid4()),
        user_id=str(uuid.uuid4()),
        display_name="Merge dispatcher",
    )
    base_head = _head(2, marker="b")
    branch_head = _head(3, branch_id=branch_id, marker="c")
    metadata = GraphBranchMetadata(
        branch_id=branch_id,
        episode_id=branch_id,
        project_id="project-one",
        base_head=base_head,
        head=branch_head,
        authorized_by=authorizer,
    )
    return BranchMergeContext.create(
        merge_task_id=task_id,
        authorized_by=merge_dispatcher,
        metadata=metadata,
        eligibility=BranchMergeEligibility(
            branch_head=branch_head,
            episode_ending="completed",
        ),
        base_graph=_state(2),
        branch_graph=_state(
            3,
            question=branch_question or _question(motivation="Branch motivation", updated_rev=3),
        ),
        main_head=_head(main_revision, marker="d" if main_revision == 4 else "e"),
        main_graph=_state(
            main_revision,
            question=main_question or _question(updated_rev=main_revision),
        ),
        run_truth_scope=["repo"],
    )


def _candidate_json(*, repositories_read: list[str] | None = None) -> str:
    return json.dumps(
        {
            "summary": "Carry the branch motivation onto main.",
            "ops": [
                {
                    "op": "update_nodes",
                    "nodes": [
                        {
                            "id": "rq/merge",
                            "changes": {"motivation": "Branch motivation"},
                        }
                    ],
                }
            ],
            "repositories_read": repositories_read or [],
            "change_summary": ["Merged the branch motivation."],
        }
    )


def _local_scope(stage: BranchMergeStage, context: BranchMergeContext) -> ProjectWriteScope:
    assert stage.local_stage is not None
    return ProjectWriteScope.create(
        project_id=context.metadata.project_id,
        execution_machine="laptop",
        execution_host="",
        capability="orchestrate",
        stage_root=str(stage.local_stage),
        workspace_root=str(stage.workspace),
        repositories=[],
        protected_write_paths=[str(stage.local_stage.parent / "state" / ".research")],
    )


def test_semantic_delta_is_typed_and_ignores_revision_bookkeeping() -> None:
    branch_id = str(uuid.uuid4())
    base_head = _head(2, marker="b")
    branch_head = _head(3, branch_id=branch_id, marker="c")
    bookkeeping_only = build_semantic_delta(
        _state(2),
        _state(3, question=_question(updated_rev=3)),
        base_head=base_head,
        branch_head=branch_head,
    )
    changed = build_semantic_delta(
        _state(2),
        _state(3, question=_question(motivation="Branch motivation", updated_rev=3)),
        base_head=base_head,
        branch_head=branch_head,
    )

    assert bookkeeping_only.is_empty
    assert len(changed.nodes) == 1
    assert changed.nodes[0].change == "updated"
    assert changed.nodes[0].node_id == "rq/merge"
    assert changed.nodes[0].before is not None
    assert changed.nodes[0].after is not None
    assert changed.nodes[0].after.motivation == "Branch motivation"


def test_three_way_conflicts_are_field_specific_and_never_auto_resolve() -> None:
    base = _state(2)
    branch = _state(3, question=_question(motivation="Branch", updated_rev=3))
    compatible_main = _state(4, question=_question(status="answered", updated_rev=4))
    conflicting_main = _state(4, question=_question(motivation="Main", updated_rev=4))

    assert detect_branch_merge_conflicts(base, branch, compatible_main) == []
    conflicts = detect_branch_merge_conflicts(base, branch, conflicting_main)

    assert len(conflicts) == 1
    assert conflicts[0].code == "both_changed"
    assert conflicts[0].collection == "nodes"
    assert conflicts[0].entity_id == "rq/merge"
    assert conflicts[0].field_path == "motivation"
    assert conflicts[0].branch == "Branch"
    assert conflicts[0].main == "Main"


@pytest.mark.parametrize(
    ("base_fields", "branch_fields", "main_fields"),
    [
        ({}, {"novelty": "branch"}, {"novelty": "main"}),
        ({"novelty": "base"}, {}, {"novelty": "main"}),
    ],
)
def test_dictionary_key_conflicts_can_resolve_to_main_or_a_third_value(
    base_fields: dict[str, str],
    branch_fields: dict[str, str],
    main_fields: dict[str, str],
) -> None:
    template = _context()
    base_question = _question(updated_rev=2).model_copy(update={"extension_fields": base_fields})
    branch_question = _question(updated_rev=3).model_copy(
        update={"extension_fields": branch_fields}
    )
    main_question = _question(updated_rev=4).model_copy(update={"extension_fields": main_fields})
    context = BranchMergeContext.create(
        merge_task_id=template.merge_task_id,
        authorized_by=template.authorized_by,
        metadata=template.metadata,
        eligibility=template.eligibility,
        base_graph=_state(2, question=base_question),
        branch_graph=_state(3, question=branch_question),
        main_head=template.main_head,
        main_graph=_state(4, question=main_question),
        run_truth_scope=template.run_truth_scope,
    )

    assert [item.field_path for item in context.deterministic_conflicts] == [
        "extension_fields.novelty"
    ]
    assert branch_merge_can_resolve_without_patch(context)

    candidate = parse_branch_merge_candidate(
        json.dumps(
            {
                "summary": "Resolve the extension conflict to a third value.",
                "ops": [
                    {
                        "op": "update_nodes",
                        "nodes": [
                            {
                                "id": "rq/merge",
                                "changes": {"extension_fields": {"novelty": "third"}},
                            }
                        ],
                    }
                ],
                "repositories_read": [],
            }
        ),
        context,
    )
    prepared = prepare_branch_merge_with_history(
        _RejectingHistory(context, reject_count=0),
        candidate,
        expected_main_head=context.main_head,
        context=context,
    )

    assert prepared.projection.graph.nodes["rq/merge"].extension_fields == {"novelty": "third"}


def test_effective_delta_accepts_compatible_main_changes_and_ignores_proposal_producer_ids() -> (
    None
):
    context = _context(
        main_question=_question(
            motivation="Branch motivation",
            status="answered",
            updated_rev=4,
        )
    )
    assert semantic_delta_is_subsumed(context.semantic_delta, context.main_graph)

    proposal = Proposal.model_validate(
        {
            "id": "prop/merge",
            "title": "Merge proposal",
            "card": {},
            "ops": [
                {
                    "op": "update_nodes",
                    "intent": "content_change",
                    "nodes": [
                        {
                            "id": "rq/merge",
                            "changes": {"motivation": "Branch motivation"},
                        }
                    ],
                }
            ],
            "created_by_operation_id": "branch-producer",
        }
    )
    main_proposal = proposal.model_copy(update={"created_by_operation_id": "main-producer"})
    base = _state(2)
    branch = _state(3).model_copy(update={"proposals": {proposal.id: proposal}})
    main = _state(4).model_copy(update={"proposals": {main_proposal.id: main_proposal}})
    delta = build_semantic_delta(
        base,
        branch,
        base_head=_head(2, marker="b"),
        branch_head=_head(3, branch_id=str(uuid.uuid4()), marker="c"),
    )

    assert len(delta.proposals) == 1
    assert semantic_delta_is_subsumed(delta, main)
    assert detect_branch_merge_conflicts(base, branch, main) == []


def test_merge_identity_is_branch_lineage_only_and_provenance_is_exact() -> None:
    context = _context(main_revision=4)
    moved = _context(
        main_revision=5,
        branch_id=context.metadata.branch_id,
        task_id=context.merge_task_id,
    )
    # Rebuild with the exact original authorizer so the branch metadata itself is identical.
    moved = BranchMergeContext.create(
        merge_task_id=context.merge_task_id,
        authorized_by=context.authorized_by,
        metadata=context.metadata,
        eligibility=context.eligibility,
        base_graph=context.base_graph,
        branch_graph=context.branch_graph,
        main_head=moved.main_head,
        main_graph=moved.main_graph,
        run_truth_scope=context.run_truth_scope,
    )

    assert branch_merge_id(context.metadata) == branch_merge_id(moved.metadata)
    assert classify_refreshed_context(context, moved) == "main_moved"

    candidate = parse_branch_merge_candidate(_candidate_json(), moved)
    assert candidate.kind == "work"
    assert candidate.profile == "orchestrator"
    assert candidate.authorized_by == context.authorized_by
    assert candidate.authorized_by != context.metadata.authorized_by
    assert candidate.task_id == context.merge_task_id
    assert candidate.episode_id == context.metadata.episode_id
    assert candidate.repositories_read == []
    assert candidate.branch_merge is not None
    assert candidate.branch_merge.merge_id == branch_merge_id(context.metadata)
    assert candidate.branch_merge.branch_base_head == context.metadata.base_head
    assert candidate.branch_merge.branch_head == context.metadata.head
    assert candidate.branch_merge.rebased_main_head == moved.main_head


def test_candidate_rejects_bookkeeping_repository_claims_and_stale_branch_context() -> None:
    context = _context()
    payload = json.loads(_candidate_json())
    payload["branch_merge"] = {"merge_id": "f" * 64}

    with pytest.raises(BranchMergeCandidateProblem, match="schema"):
        parse_branch_merge_candidate(json.dumps(payload), context)
    with pytest.raises(BranchMergeCandidateProblem, match="repositories_read"):
        parse_branch_merge_candidate(_candidate_json(repositories_read=["repo"]), context)

    changed_branch = _context(task_id=context.merge_task_id)
    with pytest.raises(BranchMergeSourceChanged, match="source branch"):
        classify_refreshed_context(context, changed_branch)


def test_graph_only_scope_has_no_repository_roots(tmp_path: Path) -> None:
    root = tmp_path / "stage"
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    stage = BranchMergeStage(local_stage=root, remote_stage=None, workspace=workspace)
    context = _context()
    scope = _local_scope(stage, context)

    require_graph_only_merge_scope(scope, context=context, stage=stage)
    assert scope.repository_roots == []
    assert scope.writable_roots == [str(workspace)]

    repo = tmp_path / "repo"
    repo.mkdir()
    broadened = ProjectWriteScope.create(
        project_id=context.metadata.project_id,
        execution_machine="laptop",
        execution_host="",
        capability="orchestrate",
        stage_root=str(root),
        workspace_root=str(workspace),
        repositories=[WritableRepositoryRoot(alias="repo", machine="laptop", path=str(repo))],
        protected_write_paths=[str(tmp_path / "state" / ".research")],
    )
    with pytest.raises(ValueError, match="no repository write roots"):
        require_graph_only_merge_scope(broadened, context=context, stage=stage)


class _FakeLauncher:
    def __init__(self, patch_text: str) -> None:
        self.patch_text = patch_text
        self.sessions: list[str | None] = []
        self.write_dirs: list[list[Path]] = []

    async def stream(
        self,
        _provider: str,
        _prompt: str,
        *,
        cwd: Path,
        session_id: str | None,
        write_dirs: list[Path],
        **_kwargs: object,
    ) -> AsyncIterator[AgentEvent]:
        self.sessions.append(session_id)
        self.write_dirs.append(write_dirs)
        (cwd / "patch.json").write_text(self.patch_text, encoding="utf-8")
        yield AgentEvent(event="session", session_id=session_id or "native-session")
        yield AgentEvent(event="provider_exit", text='{"return_code":0}')
        yield AgentEvent(event="done")


class _MovingMainHistory:
    def __init__(
        self,
        first: BranchMergeContext,
        second: BranchMergeContext,
    ) -> None:
        self.first = first
        self.second = second
        self.current = first
        self.move_once = True

    def validate_candidate(self, patch, **_kwargs):
        prepared = GraphTransitionManager().prepare_validated(
            self.current.main_graph,
            [patch],
            pre_head=self.current.main_head,
        )
        return prepared.patch, ValidationReport(), self.current.main_graph

    def append(self, patch, **_kwargs):
        if self.move_once:
            self.move_once = False
            self.current = self.second
            raise RevisionConflict("main moved during append")
        prepared = GraphTransitionManager().prepare_validated(
            self.current.main_graph,
            [patch],
            pre_head=self.current.main_head,
        )
        return prepared.patch, SimpleNamespace(state=prepared.projection.graph)


class _RejectingHistory:
    def __init__(self, context: BranchMergeContext, *, reject_count: int) -> None:
        self.context = context
        self.reject_count = reject_count
        self.validation_count = 0
        self.append_count = 0

    def validate_candidate(self, patch, **_kwargs):
        self.validation_count += 1
        if self.validation_count <= self.reject_count:
            report = ValidationReport()
            report.reject(
                "transition-conflict",
                f"Candidate conflict {self.validation_count}.",
                self.context.main_head.revision + 1,
                operation_index=0,
                rule_id="merge.test",
                failed_invariant="test semantic conflict",
            )
            return patch, report, self.context.main_graph
        prepared = GraphTransitionManager().prepare_validated(
            self.context.main_graph,
            [patch],
            pre_head=self.context.main_head,
        )
        return prepared.patch, ValidationReport(), self.context.main_graph

    def append(self, patch, **_kwargs):
        self.append_count += 1
        prepared = GraphTransitionManager().prepare_validated(
            self.context.main_graph,
            [patch],
            pre_head=self.context.main_head,
        )
        return prepared.patch, SimpleNamespace(state=prepared.projection.graph)


class _SequenceLauncher(_FakeLauncher):
    def __init__(self, patch_texts: list[str]) -> None:
        super().__init__(patch_texts[0])
        self.patch_texts = patch_texts

    async def stream(
        self,
        _provider: str,
        _prompt: str,
        *,
        cwd: Path,
        session_id: str | None,
        write_dirs: list[Path],
        **_kwargs: object,
    ) -> AsyncIterator[AgentEvent]:
        index = len(self.sessions)
        self.sessions.append(session_id)
        self.write_dirs.append(write_dirs)
        (cwd / "patch.json").write_text(self.patch_texts[index], encoding="utf-8")
        yield AgentEvent(event="session", session_id=session_id or "native-session")
        yield AgentEvent(event="provider_exit", text='{"return_code":0}')
        yield AgentEvent(event="done")


@pytest.mark.parametrize("touch_then_revert", [False, True])
def test_candidate_conformance_rejects_out_of_delta_writes_even_when_reverted(
    touch_then_revert: bool,
) -> None:
    context = _context()
    payload = json.loads(_candidate_json())
    if touch_then_revert:
        payload["ops"] = [
            {
                "op": "update_nodes",
                "nodes": [{"id": "rq/merge", "changes": {"title": "Temporary title"}}],
            },
            {
                "op": "update_nodes",
                "nodes": [{"id": "rq/merge", "changes": {"title": "Question"}}],
            },
            *payload["ops"],
        ]
    else:
        payload["ops"][0]["nodes"][0]["changes"]["title"] = "Unrelated rewrite"
    candidate = parse_branch_merge_candidate(json.dumps(payload), context)
    history = _RejectingHistory(context, reject_count=0)

    with pytest.raises(BranchMergeCandidateProblem, match="outside the source branch delta"):
        prepare_branch_merge_with_history(
            history,
            candidate,
            expected_main_head=context.main_head,
            context=context,
        )


def test_candidate_conformance_rejects_declared_same_value_write_outside_delta() -> None:
    template = _context()
    unrelated = _question(title="Unrelated", updated_rev=2).model_copy(
        update={"id": "rq/unrelated"}
    )
    context = BranchMergeContext.create(
        merge_task_id=template.merge_task_id,
        authorized_by=template.authorized_by,
        metadata=template.metadata,
        eligibility=template.eligibility,
        base_graph=_state(2, extra=unrelated),
        branch_graph=_state(
            3,
            question=_question(motivation="Branch motivation", updated_rev=3),
            extra=unrelated.model_copy(update={"updated_rev": 3}),
        ),
        main_head=template.main_head,
        main_graph=_state(
            4,
            extra=unrelated.model_copy(update={"updated_rev": 4}),
        ),
        run_truth_scope=template.run_truth_scope,
    )
    payload = json.loads(_candidate_json())
    payload["ops"][0]["nodes"].append({"id": "rq/unrelated", "changes": {"title": "Unrelated"}})
    candidate = parse_branch_merge_candidate(json.dumps(payload), context)

    with pytest.raises(BranchMergeCandidateProblem, match="declares writes outside"):
        prepare_branch_merge_with_history(
            _RejectingHistory(context, reject_count=0),
            candidate,
            expected_main_head=context.main_head,
            context=context,
        )


def test_candidate_conformance_includes_generated_guidance_closure() -> None:
    template = _context()
    blocker = Blocker(
        id="blk/capacity",
        type="blocker",
        title="Capacity",
        description="Capacity is unavailable.",
        status="open",
        created_rev=2,
        updated_rev=2,
    )
    experiment = Experiment(
        id="exp/guidance",
        type="experiment",
        title="Guided experiment",
        objective="Exercise transition closure.",
        current_summary="Waiting for capacity.",
        next_action="Resolve the blocker.",
        created_rev=2,
        updated_rev=2,
    )
    edge = Edge(
        id="edge/blocked",
        source=experiment.id,
        target=blocker.id,
        relation="blocked_by",
        layer="action",
        created_rev=2,
    )
    base = GraphState(
        revision=2,
        project_truth_scope=["repo"],
        nodes={blocker.id: blocker, experiment.id: experiment},
        edges={edge.id: edge},
    )
    branch = base.model_copy(
        update={
            "revision": 3,
            "nodes": {
                blocker.id: blocker.model_copy(update={"status": "resolved", "updated_rev": 3}),
                experiment.id: experiment.model_copy(
                    update={
                        "current_summary_stale": True,
                        "next_action_stale": True,
                        "updated_rev": 3,
                    }
                ),
            },
        }
    )
    context = BranchMergeContext.create(
        merge_task_id=template.merge_task_id,
        authorized_by=template.authorized_by,
        metadata=template.metadata,
        eligibility=template.eligibility,
        base_graph=base,
        branch_graph=branch,
        main_head=template.main_head,
        main_graph=base.model_copy(update={"revision": template.main_head.revision}),
        run_truth_scope=template.run_truth_scope,
    )
    candidate = parse_branch_merge_candidate(
        json.dumps(
            {
                "summary": "Resolve the branch blocker on main.",
                "ops": [
                    {
                        "op": "update_nodes",
                        "nodes": [{"id": blocker.id, "changes": {"status": "resolved"}}],
                    }
                ],
                "repositories_read": [],
            }
        ),
        context,
    )

    prepared = prepare_branch_merge_with_history(
        _RejectingHistory(context, reject_count=0),
        candidate,
        expected_main_head=context.main_head,
        context=context,
    )

    merged_experiment = prepared.projection.graph.nodes[experiment.id]
    assert isinstance(merged_experiment, Experiment)
    assert merged_experiment.current_summary_stale is True
    assert merged_experiment.next_action_stale is True


@pytest.mark.asyncio
async def test_conflict_only_empty_candidate_resolves_to_main_without_append(
    tmp_path: Path,
) -> None:
    context = _context(main_question=_question(motivation="Main motivation", updated_rev=4))
    assert context.deterministic_conflicts
    assert branch_merge_can_resolve_without_patch(context)
    empty = json.dumps(
        {
            "summary": "Keep current main for the conflicting field.",
            "ops": [],
            "repositories_read": [],
        }
    )
    root = tmp_path / "stage"
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    stage = BranchMergeStage(local_stage=root, remote_stage=None, workspace=workspace)
    launcher = _FakeLauncher(empty)
    outcome = BranchMergeRunOutcome()

    frames = [
        frame
        async for frame in stream_branch_merge_run(
            RunRequest(provider="codex", run_on="laptop"),
            launcher,  # type: ignore[arg-type]
            load_context=lambda: context,
            main_history=SimpleNamespace(),
            stage=stage,
            write_scope=_local_scope(stage, context),
            validator_command="python3 validator.py validate patch.json",
            outcome=outcome,
        )
    ]

    assert outcome.status == "noop"
    assert outcome.receipt is not None
    assert outcome.receipt.outcome == "no_change"
    assert outcome.receipt.result_main_head == context.main_head
    assert launcher.sessions == [None]
    assert any("resolved conflicting" in frame for frame in frames)


@pytest.mark.asyncio
async def test_moving_main_discards_candidate_and_rebases_same_session(tmp_path: Path) -> None:
    first = _context(main_revision=4)
    moved_template = _context(
        main_revision=5,
        branch_id=first.metadata.branch_id,
        task_id=first.merge_task_id,
    )
    second = BranchMergeContext.create(
        merge_task_id=first.merge_task_id,
        authorized_by=first.authorized_by,
        metadata=first.metadata,
        eligibility=first.eligibility,
        base_graph=first.base_graph,
        branch_graph=first.branch_graph,
        main_head=moved_template.main_head,
        main_graph=moved_template.main_graph,
        run_truth_scope=first.run_truth_scope,
    )
    history = _MovingMainHistory(first, second)
    root = tmp_path / "stage"
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    stage = BranchMergeStage(local_stage=root, remote_stage=None, workspace=workspace)
    scope = _local_scope(stage, first)
    launcher = _FakeLauncher(_candidate_json())
    outcome = BranchMergeRunOutcome()

    frames = [
        frame
        async for frame in stream_branch_merge_run(
            RunRequest(provider="codex", run_on="laptop"),
            launcher,  # type: ignore[arg-type]
            load_context=lambda: history.current,
            main_history=history,
            stage=stage,
            write_scope=scope,
            validator_command="python3 validator.py validate patch.json",
            outcome=outcome,
        )
    ]

    assert outcome.status == "committed"
    assert outcome.rebase_rounds == 1
    assert outcome.correction_rounds == 0
    assert outcome.rebased_main_head == second.main_head
    assert outcome.result_main_head is not None
    assert outcome.result_main_head.revision == 6
    assert outcome.receipt is not None
    assert outcome.receipt.outcome == "committed"
    assert outcome.receipt.provenance.rebased_main_head == second.main_head
    assert launcher.sessions == [None, "native-session"]
    assert launcher.write_dirs == [[], []]
    assert any('"event":"done"' in frame for frame in frames)


@pytest.mark.asyncio
async def test_semantic_conflict_uses_two_bounded_same_session_corrections(
    tmp_path: Path,
) -> None:
    context = _context()
    history = _RejectingHistory(context, reject_count=3)
    root = tmp_path / "stage"
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    stage = BranchMergeStage(local_stage=root, remote_stage=None, workspace=workspace)
    scope = _local_scope(stage, context)
    payloads: list[str] = []
    for index in range(3):
        payload = json.loads(_candidate_json())
        payload["summary"] = f"Merge candidate {index}."
        payloads.append(json.dumps(payload))
    launcher = _SequenceLauncher(payloads)
    outcome = BranchMergeRunOutcome()

    frames = [
        frame
        async for frame in stream_branch_merge_run(
            RunRequest(provider="codex", run_on="laptop"),
            launcher,  # type: ignore[arg-type]
            load_context=lambda: context,
            main_history=history,
            stage=stage,
            write_scope=scope,
            validator_command="python3 validator.py validate patch.json",
            outcome=outcome,
        )
    ]

    assert outcome.status == "rejected"
    assert outcome.correction_rounds == 2
    assert outcome.diagnostic == "Candidate conflict 3."
    assert launcher.sessions == [None, "native-session", "native-session"]
    assert history.validation_count == 3
    assert history.append_count == 0
    assert any("Candidate conflict 3" in frame for frame in frames)


@pytest.mark.asyncio
async def test_empty_semantic_delta_returns_no_change_receipt_without_provider(
    tmp_path: Path,
) -> None:
    context = _context()
    context = BranchMergeContext.create(
        merge_task_id=context.merge_task_id,
        authorized_by=context.authorized_by,
        metadata=context.metadata,
        eligibility=context.eligibility,
        base_graph=context.base_graph,
        branch_graph=_state(3, question=_question(updated_rev=3)),
        main_head=context.main_head,
        main_graph=context.main_graph,
        run_truth_scope=context.run_truth_scope,
    )
    root = tmp_path / "stage"
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    stage = BranchMergeStage(local_stage=root, remote_stage=None, workspace=workspace)
    scope = _local_scope(stage, context)
    launcher = _FakeLauncher(_candidate_json())
    outcome = BranchMergeRunOutcome()

    frames = [
        frame
        async for frame in stream_branch_merge_run(
            RunRequest(provider="codex", run_on="laptop"),
            launcher,  # type: ignore[arg-type]
            load_context=lambda: context,
            main_history=SimpleNamespace(),
            stage=stage,
            write_scope=scope,
            validator_command="python3 validator.py validate patch.json",
            outcome=outcome,
        )
    ]

    assert outcome.status == "noop"
    assert outcome.result_main_head is None
    assert outcome.receipt is not None
    assert outcome.receipt.outcome == "no_change"
    assert outcome.receipt.result_main_head == context.main_head
    assert launcher.sessions == []
    assert any("no net semantic graph change" in frame for frame in frames)


@pytest.mark.asyncio
async def test_effective_main_no_change_returns_receipt_without_provider(
    tmp_path: Path,
) -> None:
    context = _context(
        main_question=_question(
            motivation="Branch motivation",
            status="answered",
            updated_rev=4,
        )
    )
    root = tmp_path / "stage"
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    stage = BranchMergeStage(local_stage=root, remote_stage=None, workspace=workspace)
    launcher = _FakeLauncher(_candidate_json())
    outcome = BranchMergeRunOutcome()

    frames = [
        frame
        async for frame in stream_branch_merge_run(
            RunRequest(provider="codex", run_on="laptop"),
            launcher,  # type: ignore[arg-type]
            load_context=lambda: context,
            main_history=SimpleNamespace(),
            stage=stage,
            write_scope=_local_scope(stage, context),
            validator_command="python3 validator.py validate patch.json",
            outcome=outcome,
        )
    ]

    assert outcome.status == "noop"
    assert outcome.receipt is not None
    assert outcome.receipt.result_main_head == context.main_head
    assert launcher.sessions == []
    assert any("already contains" in frame for frame in frames)
