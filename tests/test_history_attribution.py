from __future__ import annotations

import json
import uuid

import pytest

from rcp.config import load_manifest
from rcp.core.authority import (
    AgentDispatchAuthority,
    AgentDispatchScope,
    AgentTaskAuthority,
    require_apply,
    require_dispatch,
)
from rcp.core.models import AuthorizedHuman, GraphState, Patch
from rcp.core.transition_models import GraphTargetRef
from rcp.core.validation import validate_patch
from rcp.history import HistoryManager, build_revision_summaries
from tests.helpers import refresh_patch, seated_on_every_project, seed_patch

from .helpers import fabricated_authorizer

PROJECT_ID = "project-one"


def _approval(summary: str = "Human approval") -> Patch:
    return Patch(kind="approval", author="human", summary=summary, ops=[])


def _review(node_id: str, standing: str, summary: str) -> Patch:
    return Patch(
        kind="approval",
        author="human",
        summary=summary,
        ops=[{"op": "set_standing", "node_id": node_id, "standing": standing}],
    )


def _agent_patch(operation_id: str | None = "operation-1") -> Patch:
    return seed_patch().model_copy(update={"source_operation_id": operation_id})


def _task_authority(
    operation_id: str,
    authorizer: AuthorizedHuman | None,
    *,
    patch_kind: str = "seed",
    project_id: str = PROJECT_ID,
    profile: str = "ordinary",
    task_contract: str = "scratch_patch",
    episode_id: str | None = None,
) -> AgentTaskAuthority:
    return AgentTaskAuthority(
        operation_id=operation_id,
        project_id=project_id,
        apply_target=GraphTargetRef(),
        authorized_by=authorizer,
        dispatch_authority=AgentDispatchAuthority(
            profile=profile,
            task_contract=task_contract,
            scope=AgentDispatchScope(
                run_truth_scope=["repo-a"],
                episode_id=episode_id,
                patch_kind=patch_kind,
            ),
        ),
        episode_id=episode_id,
    )


def _write_persisted_patch(manifest, document: dict[str, object]) -> None:
    patches_dir = manifest.research_dir / "patches"
    patches_dir.mkdir(exist_ok=True)
    (patches_dir / "000001.json").write_text(
        json.dumps(document),
        encoding="utf-8",
    )


def test_new_patch_lineage_serializes_only_episode_id_and_rejects_live_campaign_id() -> None:
    patch = seed_patch().model_copy(
        update={
            "revision": 1,
            "episode_id": "episode-one",
        }
    )

    document = json.loads(patch.model_dump_json())

    assert document["episode_id"] == "episode-one"
    assert "campaign_id" not in document
    document["campaign_id"] = document.pop("episode_id")
    with pytest.raises(ValueError, match="campaign_id.*episode_id"):
        Patch.model_validate(document)


def test_orchestrator_dispatch_requires_an_exact_episode_id() -> None:
    authority = AgentDispatchAuthority(
        profile="orchestrator",
        task_contract="orchestrate",
        scope=AgentDispatchScope(
            run_truth_scope=["repo-a"],
            patch_kind="work",
        ),
    )

    with pytest.raises(ValueError, match="orchestrate requires an exact episode"):
        require_dispatch(authority)


def test_persisted_legacy_campaign_id_replays_as_episode_id(manifest) -> None:
    history = HistoryManager(manifest)
    history.ensure_layout()
    document = (
        seed_patch()
        .model_copy(
            update={
                "revision": 1,
                "episode_id": "episode-one",
            }
        )
        .model_dump(mode="json")
    )
    document["campaign_id"] = document.pop("episode_id")
    _write_persisted_patch(manifest, document)

    loaded = history.load_patches()
    replayed = history.materialize(write_outputs=False)

    assert loaded[0].episode_id == "episode-one"
    assert replayed.patches[0].episode_id == "episode-one"
    assert replayed.state.revision == 1
    assert replayed.state.replay_status == "complete"


def test_persisted_patch_with_both_lineage_keys_is_rejected(manifest) -> None:
    history = HistoryManager(manifest)
    history.ensure_layout()
    document = (
        seed_patch()
        .model_copy(
            update={
                "revision": 1,
                "episode_id": "episode-new",
            }
        )
        .model_dump(mode="json")
    )
    document["campaign_id"] = "episode-legacy"
    _write_persisted_patch(manifest, document)

    with pytest.raises(ValueError, match="both campaign_id and episode_id"):
        history.load_patches()
    replayed = history.materialize(write_outputs=False)

    assert replayed.state.replay_status == "degraded"
    assert replayed.state.replay_failure is not None
    assert replayed.state.replay_failure.code == "patch-schema-invalid"


def test_opt_in_human_single_and_batch_use_explicit_snapshot(manifest) -> None:
    authorizer = fabricated_authorizer("Alice")
    history = HistoryManager(
        manifest,
        project_id=PROJECT_ID,
        require_attribution=True,
        project_membership_check=seated_on_every_project,
        agent_authority_resolver=lambda _project_id, operation_id: _task_authority(
            operation_id, authorizer
        ),
    )
    history.append(_agent_patch("seed-operation"))

    single, _ = history.append(
        _review("rq/learning-after-shift", "accepted", "Accepted the question").model_copy(
            update={"episode_id": "agent-episode"}
        ),
        authorized_by=authorizer,
    )
    batch, result = history.append_batch(
        [
            _review("rq/learning-after-shift", "contested", "Contested the question"),
            _review(
                "hyp/replanning-restores-plasticity",
                "accepted",
                "Accepted the hypothesis",
            ),
        ],
        authorized_by=authorizer,
    )

    assert single.authorized_by == authorizer
    assert single.profile is None
    assert single.task_id is None
    assert single.episode_id is None
    assert [patch.authorized_by for patch in batch] == [authorizer]
    assert batch[0].transition is not None
    assert len(batch[0].transition.initiating_groups) == 2
    assert result.state.revision == 3
    assert [patch.authorized_by for patch in history.load_patches()[1:]] == [
        authorizer,
        authorizer,
    ]


def test_opt_in_human_from_state_stamps_the_whole_transaction(manifest) -> None:
    authorizer = fabricated_authorizer("Alice")
    history = HistoryManager(
        manifest,
        project_id=PROJECT_ID,
        require_attribution=True,
        project_membership_check=seated_on_every_project,
        agent_authority_resolver=lambda _project_id, operation_id: _task_authority(
            operation_id, authorizer
        ),
    )
    history.append(_agent_patch("seed-operation"))

    prepared, result = history.append_batch_from_state(
        lambda state: [
            _review(
                "rq/learning-after-shift",
                "accepted",
                f"Approval from revision {state.revision}",
            )
        ],
        authorized_by=authorizer,
    )

    assert prepared[0].authorized_by == authorizer
    assert result.state.revision == 2


def test_opt_in_human_rejects_missing_explicit_snapshot_without_revision(manifest) -> None:
    authorizer = fabricated_authorizer("Alice")
    history = HistoryManager(manifest, require_attribution=True)
    preattributed = _approval().model_copy(update={"authorized_by": authorizer})

    with pytest.raises(ValueError, match="explicit authorized_by"):
        history.append(preattributed)
    with pytest.raises(ValueError, match="explicit authorized_by"):
        history.append_batch([_approval(), _approval()])

    assert history.load_patches() == []
    assert history.state().revision == 0


def test_agent_candidate_and_append_use_resolved_direct_task_snapshot(manifest) -> None:
    authorizer = fabricated_authorizer("Alice")
    operation_id = "operation-1"
    history = HistoryManager(
        manifest,
        project_id=PROJECT_ID,
        require_attribution=True,
        project_membership_check=seated_on_every_project,
        agent_authority_resolver=lambda _project_id, requested: (
            _task_authority(requested, authorizer)
            if requested == operation_id
            else _task_authority(requested, None)
        ),
    )
    raw = _agent_patch(operation_id)

    candidate, report, state = history.validate_candidate(raw)

    assert report.rejected is False
    assert state.revision == 0
    assert candidate.authorized_by == authorizer
    assert candidate.profile == "ordinary"
    assert candidate.task_id == operation_id
    assert candidate.episode_id is None
    assert history.load_patches() == []

    appended, result = history.append(raw)
    assert appended.authorized_by == authorizer
    assert appended.profile == "ordinary"
    assert appended.task_id == appended.source_operation_id == operation_id
    assert appended.episode_id is None
    assert result.state.revision == 1


def test_episode_worker_keeps_ordinary_profile_and_episode_id(manifest) -> None:
    authorizer = fabricated_authorizer("Alice")
    operation_id = "worker-operation"
    episode_id = "episode-one"
    authority = _task_authority(
        operation_id,
        authorizer,
        patch_kind="work",
        profile="ordinary",
        task_contract="work_auto",
        episode_id=episode_id,
    )
    history = HistoryManager(
        manifest,
        project_id=PROJECT_ID,
        require_attribution=True,
        project_membership_check=seated_on_every_project,
        agent_authority_resolver=lambda _project_id, _operation_id: authority,
    )
    raw = Patch(
        kind="work",
        author="agent",
        summary="Episode worker result",
        ops=[
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": "rq/worker-result",
                        "type": "research_question",
                        "title": "Worker result",
                        "question": "What did the worker learn?",
                    }
                ],
            }
        ],
        run_truth_scope=["repo-a"],
        source_operation_id=operation_id,
    )

    appended, result = history.append(raw)

    assert appended.profile == "ordinary"
    assert appended.episode_id == episode_id
    assert history.load_patches()[0].episode_id == episode_id
    summary = build_revision_summaries(history.load_patches(), result)[0]
    assert summary.profile == "ordinary"
    assert summary.episode_id == episode_id


def test_episode_orchestrator_patch_keeps_episode_id(manifest) -> None:
    authorizer = fabricated_authorizer("Alice")
    operation_id = "orchestrator-operation"
    episode_id = "episode-one"
    authority = _task_authority(
        operation_id,
        authorizer,
        patch_kind="work",
        profile="orchestrator",
        task_contract="orchestrate",
        episode_id=episode_id,
    )
    history = HistoryManager(
        manifest,
        project_id=PROJECT_ID,
        require_attribution=True,
        project_membership_check=seated_on_every_project,
        agent_authority_resolver=lambda _project_id, _operation_id: authority,
    )

    appended, _ = history.append(
        Patch(
            kind="work",
            author="agent",
            summary="Orchestrator result",
            ops=[
                {
                    "op": "create_nodes",
                    "nodes": [
                        {
                            "id": "rq/orchestrator-result",
                            "type": "research_question",
                            "title": "Orchestrator result",
                            "question": "What did the orchestrator learn?",
                        }
                    ],
                }
            ],
            run_truth_scope=["repo-a"],
            source_operation_id=operation_id,
        )
    )

    assert appended.profile == "orchestrator"
    assert appended.episode_id == episode_id


def test_orchestrator_without_canonical_episode_id_is_refused(manifest) -> None:
    authorizer = fabricated_authorizer("Alice")
    operation_id = "orchestrator-operation"
    authority = AgentTaskAuthority(
        operation_id=operation_id,
        project_id=PROJECT_ID,
        apply_target=GraphTargetRef(),
        authorized_by=authorizer,
        dispatch_authority=AgentDispatchAuthority(
            profile="orchestrator",
            task_contract="orchestrate",
            scope=AgentDispatchScope(
                run_truth_scope=["repo-a"],
                episode_id="episode-one",
                patch_kind="work",
            ),
        ),
        episode_id=None,
    )
    history = HistoryManager(
        manifest,
        project_id=PROJECT_ID,
        require_attribution=True,
        project_membership_check=seated_on_every_project,
        agent_authority_resolver=lambda _project_id, _operation_id: authority,
    )
    raw = Patch(
        kind="work",
        author="agent",
        summary="Orchestrator result",
        ops=[],
        run_truth_scope=["repo-a"],
        source_operation_id=operation_id,
    )

    with pytest.raises(ValueError, match="orchestrator.*episode_id"):
        history.append(raw)

    assert history.load_patches() == []


def test_supplied_episode_id_must_match_canonical_task(manifest) -> None:
    authorizer = fabricated_authorizer("Alice")
    operation_id = "worker-operation"
    authority = _task_authority(
        operation_id,
        authorizer,
        patch_kind="work",
        task_contract="work_auto",
        episode_id="episode-one",
    )
    history = HistoryManager(
        manifest,
        project_id=PROJECT_ID,
        require_attribution=True,
        project_membership_check=seated_on_every_project,
        agent_authority_resolver=lambda _project_id, _operation_id: authority,
    )
    raw = Patch(
        kind="work",
        author="agent",
        summary="Episode worker result",
        ops=[],
        run_truth_scope=["repo-a"],
        source_operation_id=operation_id,
        episode_id="episode-other",
    )

    with pytest.raises(ValueError, match="does not match the canonical"):
        history.append(raw)

    assert history.load_patches() == []


def test_rogue_agent_attribution_cannot_replace_resolved_snapshot(manifest) -> None:
    canonical = fabricated_authorizer("Canonical human")
    rogue = fabricated_authorizer("Rogue human")
    history = HistoryManager(
        manifest,
        project_id=PROJECT_ID,
        require_attribution=True,
        project_membership_check=seated_on_every_project,
        agent_authority_resolver=lambda _project_id, operation_id: _task_authority(
            operation_id, canonical
        ),
    )
    patch = _agent_patch().model_copy(
        update={
            "authorized_by": rogue,
            "profile": "ordinary",
            "task_id": "operation-1",
        }
    )

    with pytest.raises(ValueError, match="does not match the canonical"):
        history.append(patch)

    assert history.load_patches() == []
    assert history.state().revision == 0


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing-source", "source_operation_id"),
        ("missing-resolver", "agent_authority_resolver"),
        ("unknown-task", "unknown agent task"),
        ("legacy-task", "has no authorizer snapshot"),
        ("unnamed-authorizer", "valid authorizer snapshot"),
    ],
)
def test_agent_attribution_failures_do_not_write_or_spend_revision(
    manifest,
    case: str,
    message: str,
) -> None:
    operation_id = None if case == "missing-source" else "operation-1"
    if case == "missing-resolver":
        resolver = None
    elif case == "unknown-task":

        def resolver(_project_id: str, operation_id: str) -> AgentTaskAuthority:
            raise KeyError(operation_id)

    elif case == "legacy-task":

        def resolver(_project_id: str, operation_id: str) -> AgentTaskAuthority:
            return _task_authority(operation_id, None)

    elif case == "unnamed-authorizer":

        def resolver(_project_id: str, operation_id: str) -> AgentTaskAuthority:
            return _task_authority(
                operation_id,
                AuthorizedHuman.model_construct(
                    space_id=str(uuid.uuid4()),
                    user_id=str(uuid.uuid4()),
                    display_name=" ",
                ),
            )

    else:

        def resolver(_project_id: str, operation_id: str) -> AgentTaskAuthority:
            return _task_authority(operation_id, fabricated_authorizer("Alice"))

    history = HistoryManager(
        manifest,
        project_id=PROJECT_ID,
        require_attribution=True,
        project_membership_check=seated_on_every_project,
        agent_authority_resolver=resolver,
    )

    with pytest.raises(ValueError, match=message):
        history.append(_agent_patch(operation_id))

    assert history.load_patches() == []
    assert history.state().revision == 0


def test_identity_claim_stays_system_owned_under_attribution_policy(manifest) -> None:
    space_id = str(uuid.uuid4())
    history = HistoryManager(
        manifest,
        expected_space_id=space_id,
        require_attribution=True,
    )

    identity = history.claim_project_identity("created")
    stored = history.load_patches()[0]

    assert identity.home_space_id == space_id
    assert stored.kind == "identity"
    assert stored.authorized_by is None
    assert stored.profile is None
    assert stored.task_id is None
    assert stored.episode_id is None


def test_default_manager_remains_attribution_opt_out_compatible(manifest) -> None:
    history = HistoryManager(manifest)

    appended, result = history.append(seed_patch())

    assert appended.authorized_by is None
    assert appended.episode_id is None
    assert result.state.revision == 1


def test_attribution_policy_does_not_apply_during_legacy_replay(manifest) -> None:
    HistoryManager(manifest).append(seed_patch())
    guarded = HistoryManager(
        load_manifest(manifest.path),
        require_attribution=True,
    )

    state = guarded.state()

    assert state.revision == 1
    assert guarded.load_patches()[0].authorized_by is None
    assert guarded.load_patches()[0].episode_id is None


def test_episode_id_is_inert_to_validation_and_apply_permission() -> None:
    authorizer = fabricated_authorizer("Alice")
    operation_id = "operation-1"
    authority = _task_authority(operation_id, authorizer)
    base = _agent_patch(operation_id).model_copy(update={"revision": 1})
    verdicts: list[str] = []

    for episode_id in (None, "episode-one", "garbage-id"):
        patch = base.model_copy(update={"episode_id": episode_id})
        report = validate_patch(
            GraphState(),
            patch,
            ["repo-a"],
            repository_aliases=["repo-a"],
            default_run_truth_scope=["repo-a"],
        )
        dispatch = require_apply(authority, patch, is_project_member=seated_on_every_project)
        verdicts.append(
            json.dumps(
                {
                    "validation": [message.model_dump(mode="json") for message in report.messages],
                    "permission": dispatch.model_dump(mode="json"),
                },
                sort_keys=True,
            )
        )

    assert len(set(verdicts)) == 1


def test_authorizer_rename_does_not_change_existing_agent_snapshot(manifest) -> None:
    current = fabricated_authorizer("Before rename")

    def resolve(_project_id: str, operation_id: str) -> AgentTaskAuthority:
        return _task_authority(
            operation_id,
            current,
            patch_kind="refresh" if operation_id == "operation-2" else "seed",
        )

    history = HistoryManager(
        manifest,
        project_id=PROJECT_ID,
        require_attribution=True,
        project_membership_check=seated_on_every_project,
        agent_authority_resolver=resolve,
    )
    history.append(_agent_patch("operation-1"))
    current = current.model_copy(update={"display_name": "After rename"})
    history.append(
        refresh_patch("rq/after-rename").model_copy(update={"source_operation_id": "operation-2"})
    )

    stored = history.load_patches()
    assert stored[0].authorized_by is not None
    assert stored[0].authorized_by.display_name == "Before rename"
    assert stored[1].authorized_by is not None
    assert stored[1].authorized_by.display_name == "After rename"
