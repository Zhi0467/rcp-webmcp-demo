from __future__ import annotations

import json
import uuid
from datetime import timedelta
from pathlib import Path

import pytest

import rcp.history.branches as branch_module
from rcp.core.authority import AgentDispatchAuthority, AgentDispatchScope, AgentTaskAuthority
from rcp.core.models import (
    AuthorizedHuman,
    BranchMergeProvenance,
    BranchMergeReceipt,
    GraphBranchMetadata,
    Patch,
)
from rcp.core.transition_models import GraphHeadRef, GraphTargetRef
from rcp.history import BranchMergeAlreadyResolved, HistoryManager, ReplayHalted
from rcp.runs.branch_merge import branch_merge_id
from rcp.transport import BatchPublishFailed, StateWorkspace
from rcp.transport.state import (
    _validated_branch_commit_path,
    _validated_patch_path,
    _validated_relative_path,
)
from tests.helpers import seated_on_every_project, seed_patch


def _authorizer() -> AuthorizedHuman:
    return AuthorizedHuman(
        space_id=str(uuid.uuid4()),
        user_id=str(uuid.uuid4()),
        display_name="Branch owner",
    )


def _branch_metadata(
    history: HistoryManager,
    *,
    branch_id: str | None = None,
    authorized_by: AuthorizedHuman | None = None,
) -> GraphBranchMetadata:
    episode_id = branch_id or str(uuid.uuid4())
    base = history.head_ref()
    return GraphBranchMetadata(
        branch_id=episode_id,
        episode_id=episode_id,
        project_id="project",
        base_head=base,
        head=GraphHeadRef(
            target=GraphTargetRef(kind="branch", branch_id=episode_id),
            revision=base.revision,
            transition_id=base.transition_id,
        ),
        authorized_by=authorized_by or _authorizer(),
    )


def _branch_patch(node_id: str, title: str = "Branch-only result") -> Patch:
    return Patch(
        kind="work",
        author="agent",
        summary=f"Created {node_id} on the episode branch.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": node_id,
                        "type": "evidence",
                        "title": title,
                        "observation": "The branch recorded an isolated result.",
                        "origin": "internal_run",
                    }
                ],
            }
        ],
    )


def _task_authority(
    operation_id: str,
    metadata: GraphBranchMetadata,
    apply_target: GraphTargetRef,
) -> AgentTaskAuthority:
    return AgentTaskAuthority(
        operation_id=operation_id,
        project_id=metadata.project_id,
        apply_target=apply_target,
        authorized_by=metadata.authorized_by,
        dispatch_authority=AgentDispatchAuthority(
            profile="ordinary",
            task_contract="work_auto",
            scope=AgentDispatchScope(
                run_truth_scope=["repo-a"],
                episode_id=metadata.episode_id,
                patch_kind="work",
            ),
        ),
        episode_id=metadata.episode_id,
    )


def test_branch_starts_at_exact_main_head_and_advances_without_mutating_main(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    metadata = _branch_metadata(history)

    branch = history.create_auto_research_branch(metadata)
    appended, result = branch.append(
        _branch_patch("ev/branch-only"),
        expected_revision=metadata.base_head.revision,
    )

    assert history.state().revision == metadata.base_head.revision
    assert "ev/branch-only" not in history.state().nodes
    assert result.state.revision == metadata.base_head.revision + 1
    assert "ev/branch-only" in result.state.nodes
    assert appended.transition is not None
    assert appended.transition.pre_head == metadata.head
    assert appended.transition.pre_head.target == branch.graph_target
    assert branch.head_ref().transition_id == appended.transition.transition_id
    assert branch.branch_metadata().head == branch.head_ref()

    root = manifest.research_dir / "branches" / metadata.branch_id
    assert (root / "patches" / f"{appended.revision:06d}.json").is_file()
    assert {path.name for path in root.iterdir()} >= {
        "branch.json",
        "patches",
        "merges",
        "graph.json",
        "glossary.json",
        "proposals.json",
        "coverage.json",
        "research.md",
    }
    assert not (root / "manifest.toml").exists()
    assert not (root / "scope-base.json").exists()
    assert not (root / "cursors.json").exists()


def test_apply_authority_is_bound_to_the_exact_main_or_branch_target(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    metadata = _branch_metadata(history)
    branch = history.create_auto_research_branch(metadata)
    exact_branch = _task_authority("branch-exact", metadata, branch.graph_target)
    main_only = _task_authority("main-only", metadata, GraphTargetRef())
    branch_only = _task_authority("branch-only", metadata, branch.graph_target)
    authorities = {
        authority.operation_id: authority for authority in (exact_branch, main_only, branch_only)
    }
    history.project_id = metadata.project_id
    history.require_attribution = True
    history.project_membership_check = seated_on_every_project
    history.agent_authority_resolver = lambda _project_id, operation_id: authorities[operation_id]

    appended, _result = branch.append(
        _branch_patch("ev/exact-branch").model_copy(
            update={"source_operation_id": exact_branch.operation_id}
        )
    )
    assert appended.task_id == exact_branch.operation_id
    assert appended.episode_id == metadata.episode_id
    branch_revision = branch.state().revision

    with pytest.raises(ValueError, match="authorized to Apply to main, not branch"):
        branch.append(
            _branch_patch("ev/main-authority-on-branch").model_copy(
                update={"source_operation_id": main_only.operation_id}
            )
        )
    assert branch.state().revision == branch_revision

    main_revision = history.state().revision
    with pytest.raises(ValueError, match="authorized to Apply to branch:.*not main"):
        history.append(
            _branch_patch("ev/branch-authority-on-main").model_copy(
                update={"source_operation_id": branch_only.operation_id}
            )
        )
    assert history.state().revision == main_revision


def test_branch_replays_immutable_main_prefix_after_main_moves(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    metadata = _branch_metadata(history)
    branch = history.create_auto_research_branch(metadata)

    history.append(_branch_patch("ev/main-later", "Main-only result"))
    branch.append(_branch_patch("ev/branch-later"))

    assert history.state().revision == metadata.base_head.revision + 1
    assert "ev/main-later" in history.state().nodes
    assert "ev/branch-later" not in history.state().nodes
    assert branch.state().revision == metadata.base_head.revision + 1
    assert "ev/branch-later" in branch.state().nodes
    assert "ev/main-later" not in branch.state().nodes
    assert branch.base_state().revision == metadata.base_head.revision
    assert "ev/main-later" not in branch.base_state().nodes
    assert "ev/branch-later" not in branch.base_state().nodes

    reopened = history.branch(
        metadata.branch_id,
        expected_episode_id=metadata.episode_id,
        expected_project_id=metadata.project_id,
    )
    assert reopened.state() == branch.state()
    assert history.create_auto_research_branch(metadata).head_ref() == branch.head_ref()


def test_branch_base_state_fails_closed_when_accepted_main_prefix_is_tampered(manifest) -> None:
    history = HistoryManager(manifest)
    appended, _result = history.append(seed_patch())
    metadata = _branch_metadata(history)
    branch = history.create_auto_research_branch(metadata)
    assert appended.transition is not None
    path = manifest.research_dir / "patches" / f"{appended.revision:06d}.json"
    forged = appended.transition.model_copy(
        update={
            "pre_head": appended.transition.pre_head.model_copy(
                update={
                    "target": GraphTargetRef(
                        kind="branch",
                        branch_id=metadata.branch_id,
                    )
                }
            )
        }
    )
    path.write_text(
        appended.model_copy(update={"transition": forged}).model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ReplayHalted):
        branch.base_state()


def test_branch_open_rejects_identity_mismatch_and_unsafe_paths(manifest, tmp_path) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    metadata = _branch_metadata(history)
    history.create_auto_research_branch(metadata)

    with pytest.raises(ValueError, match="different episode"):
        history.branch(metadata.branch_id, expected_episode_id=str(uuid.uuid4()))
    with pytest.raises(ValueError, match="different project"):
        history.branch(metadata.branch_id, expected_project_id="other")
    with pytest.raises(ValueError, match="canonical episode UUIDv4"):
        history.branch("../patches")

    symlink_id = str(uuid.uuid4())
    outside = tmp_path / "outside"
    outside.mkdir()
    (manifest.research_dir / "branches" / symlink_id).symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="not a regular directory"):
        history.branch(symlink_id)


def test_branch_transition_target_or_chain_tampering_halts_replay(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    metadata = _branch_metadata(history)
    branch = history.create_auto_research_branch(metadata)
    appended, _result = branch.append(_branch_patch("ev/tampered"))
    assert appended.transition is not None
    path = branch.patches_dir / f"{appended.revision:06d}.json"
    forged_trace = appended.transition.model_copy(
        update={
            "pre_head": appended.transition.pre_head.model_copy(update={"target": GraphTargetRef()})
        }
    )
    path.write_text(
        appended.model_copy(update={"transition": forged_trace}).model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    state = branch.current_materialization().state
    assert state.replay_status == "degraded"
    assert state.replay_failure is not None
    assert state.replay_failure.code == "transition-head-mismatch"
    with pytest.raises(ReplayHalted):
        branch.append(_branch_patch("ev/refused"))


class _RecordingRemoteWorkspace(StateWorkspace):
    def __init__(self, root: Path) -> None:
        super().__init__(root, "worker:/state/.research")
        self.remote = True
        self.branch_commits: list[tuple[list[str], str]] = []
        self.patch_commits: list[tuple[list[str], str]] = []
        self.patch_failure: str | None = None
        self.branch_failure: str | None = None
        self.events: list[str] = []

    def publish(self, relative_paths) -> None:
        self.events.append("repair")

    def publish_committed_branch_file(self, relative_paths, commit_path) -> None:
        self.events.append("branch")
        self.branch_commits.append(
            ([Path(item).as_posix() for item in relative_paths], Path(commit_path).as_posix())
        )
        if self.branch_failure is not None:
            raise BatchPublishFailed(
                "simulated remote branch-file failure",
                commit_status=self.branch_failure,
            )

    def publish_committed_patch(self, relative_paths, patch_path) -> None:
        self.events.append("patch")
        self.patch_commits.append(
            ([Path(item).as_posix() for item in relative_paths], Path(patch_path).as_posix())
        )
        if self.patch_failure is not None:
            raise BatchPublishFailed(
                "simulated remote failure",
                commit_status=self.patch_failure,
            )


def test_branch_remote_publication_uses_nested_atomic_commit_points(manifest) -> None:
    workspace = _RecordingRemoteWorkspace(manifest.research_dir)
    history = HistoryManager(manifest, workspace)
    history.append(seed_patch())
    metadata = _branch_metadata(history)

    branch = history.create_auto_research_branch(metadata)
    appended, _result = branch.append(_branch_patch("ev/remote"))

    branch_prefix = f"branches/{metadata.branch_id}"
    assert workspace.branch_commits[0][1] == f"{branch_prefix}/branch.json"
    published, commit = workspace.patch_commits[-1]
    assert commit == f"{branch_prefix}/patches/{appended.revision:06d}.json"
    assert commit in published
    assert f"{branch_prefix}/graph.json" in published
    assert f"{branch_prefix}/branch.json" in published


@pytest.mark.parametrize("commit_status", ["absent", "present"])
def test_branch_remote_patch_failure_reconciles_commit_point(manifest, commit_status: str) -> None:
    workspace = _RecordingRemoteWorkspace(manifest.research_dir)
    history = HistoryManager(manifest, workspace)
    history.append(seed_patch())
    metadata = _branch_metadata(history)
    branch = history.create_auto_research_branch(metadata)
    workspace.patch_failure = commit_status

    if commit_status == "present":
        appended, result = branch.append(_branch_patch("ev/remote-reconciled"))
        assert result.state.revision == appended.revision
        assert "ev/remote-reconciled" in branch.state().nodes
    else:
        with pytest.raises(BatchPublishFailed):
            branch.append(_branch_patch("ev/remote-rolled-back"))
        assert branch.state().revision == metadata.base_head.revision
        assert "ev/remote-rolled-back" not in branch.state().nodes


def test_confirmed_remote_branch_patch_fences_and_repairs_before_the_next_write(manifest) -> None:
    workspace = _RecordingRemoteWorkspace(manifest.research_dir)
    history = HistoryManager(manifest, workspace)
    history.append(seed_patch())
    metadata = _branch_metadata(history)
    branch = history.create_auto_research_branch(metadata)
    workspace.patch_failure = "present"

    branch.append(_branch_patch("ev/confirmed-before-repair"))
    assert history._branch_materialization_repair_required(metadata.branch_id)
    (branch.root / "graph.json").unlink()

    workspace.patch_failure = None
    workspace.events.clear()
    branch.append(_branch_patch("ev/after-repair"))

    assert workspace.events[:2] == ["repair", "patch"]
    assert not history._branch_materialization_repair_required(metadata.branch_id)
    assert {"ev/confirmed-before-repair", "ev/after-repair"} <= set(branch.state().nodes)


def test_unknown_remote_branch_commit_is_repaired_if_a_later_refresh_proves_it(manifest) -> None:
    workspace = _RecordingRemoteWorkspace(manifest.research_dir)
    history = HistoryManager(manifest, workspace)
    history.append(seed_patch())
    metadata = _branch_metadata(history)
    branch = history.create_auto_research_branch(metadata)
    workspace.patch_failure = "unknown"

    with pytest.raises(BatchPublishFailed):
        branch.append(_branch_patch("ev/unknown-then-proved"))
    quarantined = next(branch.patches_dir.glob(".unconfirmed-*.json-*"))
    proved = branch.patches_dir / f"{metadata.base_head.revision + 1:06d}.json"
    quarantined.rename(proved)
    assert history._branch_materialization_repair_required(metadata.branch_id)

    workspace.patch_failure = None
    workspace.events.clear()
    branch.append(_branch_patch("ev/after-unknown-repair"))

    assert workspace.events[:2] == ["repair", "patch"]
    assert not history._branch_materialization_repair_required(metadata.branch_id)
    assert {"ev/unknown-then-proved", "ev/after-unknown-repair"} <= set(branch.state().nodes)


def test_local_branch_output_failure_fences_the_committed_patch_until_repair(
    manifest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    metadata = _branch_metadata(history)
    branch = history.create_auto_research_branch(metadata)
    write_outputs = branch._write_materialized_outputs

    def fail_outputs(_result) -> None:
        raise OSError("simulated derived-output failure")

    monkeypatch.setattr(branch, "_write_materialized_outputs", fail_outputs)
    with pytest.raises(OSError, match="derived-output"):
        branch.append(_branch_patch("ev/local-committed-before-repair"))
    assert history._branch_materialization_repair_required(metadata.branch_id)

    monkeypatch.setattr(branch, "_write_materialized_outputs", write_outputs)
    branch.append(_branch_patch("ev/local-after-repair"))
    assert not history._branch_materialization_repair_required(metadata.branch_id)
    assert {"ev/local-committed-before-repair", "ev/local-after-repair"} <= set(
        branch.state().nodes
    )


def test_branch_read_snapshot_semantically_replays_the_tail_and_fails_closed(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    metadata = _branch_metadata(history)
    branch = history.create_auto_research_branch(metadata)
    appended, _ = branch.append(_branch_patch("ev/read-tail"))
    path = branch.patches_dir / f"{appended.revision:06d}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["ops"] = [
        {
            "op": "update_nodes",
            "nodes": [{"id": "ev/missing", "changes": {"title": "Corrupted"}}],
        }
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ReplayHalted):
        history.branch_read_snapshots(
            [(metadata.branch_id, metadata.episode_id, metadata.project_id)]
        )


def test_branch_read_snapshot_keeps_the_exact_base_before_a_rejected_main_patch(
    manifest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    metadata = _branch_metadata(history)
    branch = history.create_auto_research_branch(metadata)
    rejected, _result = history.append(
        Patch(
            kind="refresh",
            author="agent",
            summary="Persist a rejected main revision after the branch base.",
            run_truth_scope=["repo-a"],
            repositories_read=["repo-a"],
            ops=[
                {
                    "op": "create_edges",
                    "edges": [
                        {
                            "source": "rq/learning-after-shift",
                            "target": "hyp/replanning-restores-plasticity",
                            "relation": "not_a_relation",
                        }
                    ],
                }
            ],
        ),
        raise_on_reject=False,
    )
    assert rejected.admission == "rejected"
    exact_base = branch.base_state()
    observed: list[object] = []
    replay_tail = branch_module._replay_branch_tail

    def capture_base(candidate_branch, base_state, patches):
        observed.append(base_state)
        return replay_tail(candidate_branch, base_state, patches)

    monkeypatch.setattr(branch_module, "_replay_branch_tail", capture_base)

    history.branch_read_snapshots([(metadata.branch_id, metadata.episode_id, metadata.project_id)])

    assert observed == [exact_base]


def test_merge_receipts_are_append_only_exact_and_support_no_change(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    metadata = _branch_metadata(history)
    branch = history.create_auto_research_branch(metadata)
    branch.append(_branch_patch("ev/merge-source"))
    branch_head = branch.head_ref()
    main_head = history.head_ref()
    provenance = BranchMergeProvenance(
        merge_id=branch_merge_id(metadata.model_copy(update={"head": branch_head})),
        branch_id=metadata.branch_id,
        episode_id=metadata.episode_id,
        branch_base_head=metadata.base_head,
        branch_head=branch_head,
        rebased_main_head=main_head,
        merge_task_id="merge-task",
    )
    receipt = BranchMergeReceipt(
        outcome="no_change",
        provenance=provenance,
        result_main_head=main_head,
        authorized_by=metadata.authorized_by,
    )

    assert branch.write_merge_receipt(receipt) == receipt
    assert branch.write_merge_receipt(receipt) == receipt
    assert branch.merge_receipts() == [receipt]

    changed = receipt.model_copy(update={"created_at": receipt.created_at + timedelta(seconds=1)})
    assert branch.write_merge_receipt(changed) == receipt

    inconsistent = receipt.model_copy(
        update={
            "provenance": provenance.model_copy(update={"branch_head": metadata.head}),
        }
    )
    with pytest.raises(ValueError, match="receipt id does not match"):
        branch.write_merge_receipt(inconsistent)

    invalid_provenance = provenance.model_copy(update={"merge_id": "a" * 64})
    invalid_receipt = receipt.model_copy(update={"provenance": invalid_provenance})
    invalid_path = branch.merges_dir / f"{invalid_provenance.merge_id}.json"
    invalid_path.write_text(invalid_receipt.model_dump_json(indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="receipt id does not match"):
        history.branch_read_snapshots(
            [(metadata.branch_id, metadata.episode_id, metadata.project_id)]
        )


def test_independent_no_change_writers_return_the_first_canonical_receipt(manifest) -> None:
    first_history = HistoryManager(manifest)
    first_history.append(seed_patch())
    metadata = _branch_metadata(first_history)
    first_branch = first_history.create_auto_research_branch(metadata)
    first_branch.append(_branch_patch("ev/no-change-race"))
    branch_head = first_branch.head_ref()
    main_head = first_history.head_ref()
    merge_id = branch_merge_id(metadata.model_copy(update={"head": branch_head}))
    winner = BranchMergeReceipt(
        outcome="no_change",
        provenance=BranchMergeProvenance(
            merge_id=merge_id,
            branch_id=metadata.branch_id,
            episode_id=metadata.episode_id,
            branch_base_head=metadata.base_head,
            branch_head=branch_head,
            rebased_main_head=main_head,
            merge_task_id="first-merge-task",
        ),
        result_main_head=main_head,
        authorized_by=metadata.authorized_by,
    )
    assert first_branch.write_merge_receipt(winner) == winner

    second_history = HistoryManager(manifest)
    second_branch = second_history.branch(
        metadata.branch_id,
        expected_episode_id=metadata.episode_id,
        expected_project_id=metadata.project_id,
    )
    contender = winner.model_copy(
        update={
            "provenance": winner.provenance.model_copy(
                update={"merge_task_id": "second-merge-task"}
            ),
            "authorized_by": _authorizer(),
            "created_at": winner.created_at + timedelta(seconds=1),
        }
    )

    assert second_branch.write_merge_receipt(contender) == winner
    assert second_branch.merge_receipts() == [winner]


def test_remote_present_no_change_receipt_is_parsed_and_validated(manifest) -> None:
    workspace = _RecordingRemoteWorkspace(manifest.research_dir)
    history = HistoryManager(manifest, workspace)
    history.append(seed_patch())
    metadata = _branch_metadata(history)
    branch = history.create_auto_research_branch(metadata)
    branch.append(_branch_patch("ev/remote-no-change"))
    branch_head = branch.head_ref()
    main_head = history.head_ref()
    receipt = BranchMergeReceipt(
        outcome="no_change",
        provenance=BranchMergeProvenance(
            merge_id=branch_merge_id(metadata.model_copy(update={"head": branch_head})),
            branch_id=metadata.branch_id,
            episode_id=metadata.episode_id,
            branch_base_head=metadata.base_head,
            branch_head=branch_head,
            rebased_main_head=main_head,
            merge_task_id="remote-receipt-task",
        ),
        result_main_head=main_head,
        authorized_by=metadata.authorized_by,
    )
    workspace.branch_failure = "present"

    assert branch.write_merge_receipt(receipt) == receipt
    assert branch.merge_receipts() == [receipt]


def test_merge_receipt_can_be_reconciled_from_committed_main_patch(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    metadata = _branch_metadata(history)
    branch = history.create_auto_research_branch(metadata)
    branch.append(_branch_patch("ev/merge-reconcile"))
    main_head = history.head_ref()
    provenance = BranchMergeProvenance(
        merge_id=branch_merge_id(metadata.model_copy(update={"head": branch.head_ref()})),
        branch_id=metadata.branch_id,
        episode_id=metadata.episode_id,
        branch_base_head=metadata.base_head,
        branch_head=branch.head_ref(),
        rebased_main_head=main_head,
        merge_task_id="merge-task",
    )
    merge_patch = _branch_patch("ev/merged-main").model_copy(
        update={
            "profile": "orchestrator",
            "task_id": "merge-task",
            "episode_id": metadata.episode_id,
            "authorized_by": metadata.authorized_by,
            "branch_merge": provenance,
        }
    )
    appended, _result = history.append(merge_patch)
    assert appended.transition is not None

    receipt = branch.reconcile_merge_receipt(provenance.merge_id)

    assert receipt is not None
    assert receipt.outcome == "committed"
    assert receipt.provenance == provenance
    assert receipt.result_main_head == GraphHeadRef(
        revision=appended.revision,
        transition_id=appended.transition.transition_id,
    )
    assert branch.reconcile_merge_receipt(provenance.merge_id) == receipt


def test_no_change_writer_reconciles_a_concurrently_committed_main_winner(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    metadata = _branch_metadata(history)
    branch = history.create_auto_research_branch(metadata)
    branch.append(_branch_patch("ev/merge-race"))
    main_head = history.head_ref()
    provenance = BranchMergeProvenance(
        merge_id=branch_merge_id(metadata.model_copy(update={"head": branch.head_ref()})),
        branch_id=metadata.branch_id,
        episode_id=metadata.episode_id,
        branch_base_head=metadata.base_head,
        branch_head=branch.head_ref(),
        rebased_main_head=main_head,
        merge_task_id="committed-winner",
    )
    merge_patch = _branch_patch("ev/merged-race").model_copy(
        update={
            "profile": "orchestrator",
            "task_id": provenance.merge_task_id,
            "episode_id": metadata.episode_id,
            "authorized_by": metadata.authorized_by,
            "branch_merge": provenance,
        }
    )
    appended, _result = history.append(merge_patch)
    assert appended.transition is not None
    contender = BranchMergeReceipt(
        outcome="no_change",
        provenance=provenance.model_copy(update={"merge_task_id": "late-no-change"}),
        result_main_head=main_head,
        authorized_by=_authorizer(),
    )

    winner = branch.write_merge_receipt(contender)

    assert winner.outcome == "committed"
    assert winner.provenance == provenance
    assert winner.result_main_head == GraphHeadRef(
        revision=appended.revision,
        transition_id=appended.transition.transition_id,
    )
    assert branch.merge_receipts() == [winner]


def test_no_change_receipt_wins_before_a_concurrent_main_append(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    metadata = _branch_metadata(history)
    branch = history.create_auto_research_branch(metadata)
    branch.append(_branch_patch("ev/no-change-wins"))
    main_head = history.head_ref()
    provenance = BranchMergeProvenance(
        merge_id=branch_merge_id(metadata.model_copy(update={"head": branch.head_ref()})),
        branch_id=metadata.branch_id,
        episode_id=metadata.episode_id,
        branch_base_head=metadata.base_head,
        branch_head=branch.head_ref(),
        rebased_main_head=main_head,
        merge_task_id="no-change-winner",
    )
    receipt = branch.write_merge_receipt(
        BranchMergeReceipt(
            outcome="no_change",
            provenance=provenance,
            result_main_head=main_head,
            authorized_by=metadata.authorized_by,
        )
    )
    candidate = _branch_patch("ev/late-main-merge").model_copy(
        update={
            "profile": "orchestrator",
            "task_id": provenance.merge_task_id,
            "episode_id": metadata.episode_id,
            "authorized_by": metadata.authorized_by,
            "branch_merge": provenance,
        }
    )

    with pytest.raises(BranchMergeAlreadyResolved) as resolved:
        history.append(candidate, expected_revision=main_head.revision)

    assert resolved.value.receipt == receipt
    assert history.head_ref() == main_head


def test_reconcile_rejects_corrupted_existing_no_change_receipt(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    metadata = _branch_metadata(history)
    branch = history.create_auto_research_branch(metadata)
    branch.append(_branch_patch("ev/corrupt-no-change"))
    branch_head = branch.head_ref()
    main_head = history.head_ref()
    receipt = branch.write_merge_receipt(
        BranchMergeReceipt(
            outcome="no_change",
            provenance=BranchMergeProvenance(
                merge_id=branch_merge_id(metadata.model_copy(update={"head": branch_head})),
                branch_id=metadata.branch_id,
                episode_id=metadata.episode_id,
                branch_base_head=metadata.base_head,
                branch_head=branch_head,
                rebased_main_head=main_head,
                merge_task_id="no-change-corruption",
            ),
            result_main_head=main_head,
            authorized_by=metadata.authorized_by,
        )
    )
    forged_head = main_head.model_copy(update={"transition_id": "f" * 64})
    corrupted = receipt.model_copy(
        update={
            "provenance": receipt.provenance.model_copy(update={"rebased_main_head": forged_head}),
            "result_main_head": forged_head,
        }
    )
    path = branch.merges_dir / f"{receipt.provenance.merge_id}.json"
    path.write_text(corrupted.model_dump_json(indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exact main head"):
        branch.reconcile_merge_receipt(receipt.provenance.merge_id)


def test_reconcile_rejects_corrupted_existing_committed_receipt(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    metadata = _branch_metadata(history)
    branch = history.create_auto_research_branch(metadata)
    branch.append(_branch_patch("ev/corrupt-committed"))
    provenance = BranchMergeProvenance(
        merge_id=branch_merge_id(metadata.model_copy(update={"head": branch.head_ref()})),
        branch_id=metadata.branch_id,
        episode_id=metadata.episode_id,
        branch_base_head=metadata.base_head,
        branch_head=branch.head_ref(),
        rebased_main_head=history.head_ref(),
        merge_task_id="committed-corruption",
    )
    appended, _result = history.append(
        _branch_patch("ev/committed-corruption").model_copy(
            update={
                "profile": "orchestrator",
                "task_id": provenance.merge_task_id,
                "episode_id": metadata.episode_id,
                "authorized_by": metadata.authorized_by,
                "branch_merge": provenance,
            }
        )
    )
    assert appended.transition is not None
    receipt = branch.reconcile_merge_receipt(provenance.merge_id)
    assert receipt is not None
    path = branch.merges_dir / f"{provenance.merge_id}.json"
    corrupted = receipt.model_copy(update={"created_at": receipt.created_at + timedelta(seconds=1)})
    path.write_text(corrupted.model_dump_json(indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="disagrees with main provenance"):
        branch.reconcile_merge_receipt(provenance.merge_id)


@pytest.mark.parametrize(
    "path",
    [
        "branches/not-a-uuid/patches/000001.json",
        "branches/00000000-0000-4000-8000-000000000000/../patches/000001.json",
        "branches/00000000-0000-4000-8000-000000000000/patches/latest.json",
        "branches/00000000-0000-4000-8000-000000000000/aliases/000001.json",
    ],
)
def test_transport_refuses_noncanonical_branch_patch_paths(path: str) -> None:
    with pytest.raises(ValueError):
        _validated_patch_path(path)
    with pytest.raises(ValueError):
        _validated_relative_path(path)


@pytest.mark.parametrize("child", ["patches", "merges"])
def test_branch_child_enumeration_refuses_symlinked_directories(
    manifest,
    tmp_path: Path,
    child: str,
) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    metadata = _branch_metadata(history)
    branch = history.create_auto_research_branch(metadata)
    path = branch.root / child
    path.rmdir()
    outside = tmp_path / f"outside-{child}"
    outside.mkdir()
    path.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="not a regular directory"):
        history.branch(
            metadata.branch_id,
            expected_episode_id=metadata.episode_id,
            expected_project_id=metadata.project_id,
        )


def test_open_branch_refuses_symlinked_metadata(manifest, tmp_path: Path) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    metadata = _branch_metadata(history)
    branch = history.create_auto_research_branch(metadata)
    metadata_path = branch.root / "branch.json"
    outside = tmp_path / "outside-branch.json"
    outside.write_text(metadata_path.read_text(encoding="utf-8"), encoding="utf-8")
    metadata_path.unlink()
    metadata_path.symlink_to(outside)

    with pytest.raises(ValueError, match="not a regular file"):
        history.branch(metadata.branch_id)


def test_transport_accepts_only_exact_branch_commit_shapes() -> None:
    branch_id = "00000000-0000-4000-8000-000000000000"
    assert (
        _validated_patch_path(f"branches/{branch_id}/patches/000042.json").as_posix()
        == f"branches/{branch_id}/patches/000042.json"
    )
    assert (
        _validated_branch_commit_path(f"branches/{branch_id}/branch.json").as_posix()
        == f"branches/{branch_id}/branch.json"
    )
    assert (
        _validated_branch_commit_path(f"branches/{branch_id}/merges/{'a' * 64}.json").as_posix()
        == f"branches/{branch_id}/merges/{'a' * 64}.json"
    )


def test_branch_metadata_json_is_current_after_reopen(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    metadata = _branch_metadata(history)
    branch = history.create_auto_research_branch(metadata)
    branch.append(_branch_patch("ev/current-head"))

    raw = json.loads((branch.root / "branch.json").read_text(encoding="utf-8"))
    assert raw["head"] == branch.head_ref().model_dump(mode="json")
