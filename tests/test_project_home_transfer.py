from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from rcp.config import load_manifest
from rcp.core.models import AuthorizedHuman, GraphState, Patch, ProjectHomeTransfer
from rcp.core.validation import validate_patch
from rcp.history import HistoryManager, ProjectIdentityConflict

PROJECT_ID = "7c35b754-10e5-4bdd-8114-4ec35671437f"
SOURCE_SPACE_ID = "6f85ff9c-fe58-4e8f-b790-230e8a3b2f2d"
TARGET_SPACE_ID = "b6e2bfa1-d564-4cf5-8347-e80c30e043f9"
OTHER_SPACE_ID = "77aa9b5a-eaf6-4a9e-9aa0-e7d6bf368529"
SHARED_LOCAL_USER_ID = "67ae7caf-7381-4d17-83cc-5b06267b4615"


def _actor(space_id: str, name: str) -> AuthorizedHuman:
    return AuthorizedHuman(
        space_id=space_id,
        user_id=SHARED_LOCAL_USER_ID,
        display_name=name,
    )


def _transfer(**changes: object) -> ProjectHomeTransfer:
    values: dict[str, object] = {
        "project_id": PROJECT_ID,
        "previous_home_space_id": SOURCE_SPACE_ID,
        "new_home_space_id": TARGET_SPACE_ID,
        "source_released_by": _actor(SOURCE_SPACE_ID, "Source owner"),
        "target_admitted_by": _actor(TARGET_SPACE_ID, "Team reviewer"),
    }
    values.update(changes)
    return ProjectHomeTransfer.model_validate(values)


def _transfer_patch(transfer: ProjectHomeTransfer) -> Patch:
    return Patch(
        kind="identity",
        author=None,
        producer="system",
        summary="Project moved to its admitted team space.",
        ops=[],
        project_home_transfer=transfer,
    )


def _ordinary_patch(truth_scope: list[str]) -> Patch:
    return Patch(
        kind="refresh",
        author="agent",
        producer="agent",
        summary="Recorded work after transfer.",
        ops=[],
        run_truth_scope=truth_scope,
    )


def _transition_patch() -> Patch:
    return Patch(
        kind="refresh",
        author="agent",
        producer="agent",
        summary="Record a deterministic historical transition.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": "rq/historical-transition",
                        "type": "research_question",
                        "title": "Historical transition",
                        "question": "Does the old transition id still replay?",
                    }
                ],
            }
        ],
    )


def test_home_transfer_moves_only_the_derived_home_and_preserves_both_actors(manifest) -> None:
    source = HistoryManager(manifest, expected_space_id=SOURCE_SPACE_ID)
    original = source.claim_project_identity("created", project_id=PROJECT_ID)
    transfer = source.transfer_project_home(
        project_id=PROJECT_ID,
        previous_home_space_id=SOURCE_SPACE_ID,
        new_home_space_id=TARGET_SPACE_ID,
        source_released_by=_actor(SOURCE_SPACE_ID, "Source owner"),
        target_admitted_by=_actor(TARGET_SPACE_ID, "Team reviewer"),
    )

    assert transfer == _transfer()
    assert transfer.source_released_by.user_id == transfer.target_admitted_by.user_id
    assert transfer.source_released_by.space_id != transfer.target_admitted_by.space_id
    moved = source.project_identity()
    assert moved == original.model_copy(update={"home_space_id": TARGET_SPACE_ID})
    assert moved.action == "created"

    patches = source.load_patches()
    assert len(patches) == 2
    assert patches[0].project_identity == original
    assert patches[0].project_home_transfer is None
    assert patches[1].project_identity is None
    assert patches[1].project_home_transfer == transfer
    assert patches[1].author is None
    assert patches[1].producer == "system"

    with pytest.raises(ProjectIdentityConflict, match="belongs to space"):
        source.append(_ordinary_patch(manifest.project.truth_scope))

    target = HistoryManager(load_manifest(manifest.path), expected_space_id=TARGET_SPACE_ID)
    target.update_machine_provider_paths({"laptop": {"codex": "/target/codex"}})
    assert load_manifest(manifest.path).machine_map["laptop"].provider_paths["codex"] == (
        "/target/codex"
    )


def test_old_transition_hash_replays_without_the_new_optional_field(manifest) -> None:
    history = HistoryManager(manifest)
    appended, _result = history.append(_transition_patch())
    assert appended.transition is not None
    assert (
        appended.transition.transition_id
        == "ea0a37a86070a3f1614406c928cfd900dd93c200896f3dbf339153f6fe9172d9"
    )

    path = manifest.research_dir / "patches" / "000001.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document.pop("project_home_transfer")
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    legacy_bytes = path.read_bytes()

    replay = HistoryManager(load_manifest(manifest.path)).materialize(write_outputs=False).state

    assert replay.replay_status == "complete"
    assert replay.revision == 1
    assert "rq/historical-transition" in replay.nodes
    assert path.read_bytes() == legacy_bytes


def test_home_transfer_is_exactly_idempotent_but_different_retry_is_refused(manifest) -> None:
    source = HistoryManager(manifest, expected_space_id=SOURCE_SPACE_ID)
    source.claim_project_identity("created", project_id=PROJECT_ID)
    first = source.transfer_project_home(
        project_id=PROJECT_ID,
        previous_home_space_id=SOURCE_SPACE_ID,
        new_home_space_id=TARGET_SPACE_ID,
        source_released_by=_actor(SOURCE_SPACE_ID, "Source owner"),
        target_admitted_by=_actor(TARGET_SPACE_ID, "Team reviewer"),
    )
    repeated = source.transfer_project_home(
        project_id=PROJECT_ID,
        previous_home_space_id=SOURCE_SPACE_ID,
        new_home_space_id=TARGET_SPACE_ID,
        source_released_by=_actor(SOURCE_SPACE_ID, "Source owner"),
        target_admitted_by=_actor(TARGET_SPACE_ID, "Team reviewer"),
    )

    assert repeated == first
    assert len(source.load_patches()) == 2
    with pytest.raises(ProjectIdentityConflict, match="current home"):
        source.transfer_project_home(
            project_id=PROJECT_ID,
            previous_home_space_id=SOURCE_SPACE_ID,
            new_home_space_id=OTHER_SPACE_ID,
            source_released_by=_actor(SOURCE_SPACE_ID, "Source owner"),
            target_admitted_by=_actor(OTHER_SPACE_ID, "Other reviewer"),
        )
    assert len(source.load_patches()) == 2


def test_concurrent_exact_home_transfer_appends_one_revision(manifest) -> None:
    first = HistoryManager(manifest, expected_space_id=SOURCE_SPACE_ID)
    first.claim_project_identity("created", project_id=PROJECT_ID)
    second = HistoryManager(load_manifest(manifest.path), expected_space_id=SOURCE_SPACE_ID)
    ready = threading.Barrier(3)

    def transfer(history: HistoryManager) -> ProjectHomeTransfer:
        ready.wait()
        return history.transfer_project_home(
            project_id=PROJECT_ID,
            previous_home_space_id=SOURCE_SPACE_ID,
            new_home_space_id=TARGET_SPACE_ID,
            source_released_by=_actor(SOURCE_SPACE_ID, "Source owner"),
            target_admitted_by=_actor(TARGET_SPACE_ID, "Team reviewer"),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(transfer, first)
        second_future = executor.submit(transfer, second)
        ready.wait()
        results = [first_future.result(timeout=5), second_future.result(timeout=5)]

    assert results == [_transfer(), _transfer()]
    assert len(first.load_patches()) == 2


def test_replay_reduces_home_transfers_in_canonical_order(manifest) -> None:
    source = HistoryManager(manifest, expected_space_id=SOURCE_SPACE_ID)
    source.claim_project_identity("created", project_id=PROJECT_ID)
    source.transfer_project_home(
        project_id=PROJECT_ID,
        previous_home_space_id=SOURCE_SPACE_ID,
        new_home_space_id=TARGET_SPACE_ID,
        source_released_by=_actor(SOURCE_SPACE_ID, "Source owner"),
        target_admitted_by=_actor(TARGET_SPACE_ID, "Team reviewer"),
    )
    target = HistoryManager(load_manifest(manifest.path), expected_space_id=TARGET_SPACE_ID)
    target.transfer_project_home(
        project_id=PROJECT_ID,
        previous_home_space_id=TARGET_SPACE_ID,
        new_home_space_id=OTHER_SPACE_ID,
        source_released_by=_actor(TARGET_SPACE_ID, "Team source reviewer"),
        target_admitted_by=_actor(OTHER_SPACE_ID, "Other reviewer"),
    )

    derived = HistoryManager(load_manifest(manifest.path)).project_identity()
    assert derived is not None
    assert derived.project_id == PROJECT_ID
    assert derived.home_space_id == OTHER_SPACE_ID
    assert derived.action == "created"


def test_replay_refuses_a_second_nameplate_matching_the_transferred_home(manifest) -> None:
    source = HistoryManager(manifest, expected_space_id=SOURCE_SPACE_ID)
    source.claim_project_identity("created", project_id=PROJECT_ID)
    source.transfer_project_home(
        project_id=PROJECT_ID,
        previous_home_space_id=SOURCE_SPACE_ID,
        new_home_space_id=TARGET_SPACE_ID,
        source_released_by=_actor(SOURCE_SPACE_ID, "Source owner"),
        target_admitted_by=_actor(TARGET_SPACE_ID, "Team reviewer"),
    )
    HistoryManager(load_manifest(manifest.path)).append(
        Patch(
            kind="identity",
            author=None,
            producer="system",
            summary="Conflicting second nameplate.",
            ops=[],
            project_identity={
                "project_id": PROJECT_ID,
                "home_space_id": TARGET_SPACE_ID,
                "action": "created",
            },
        )
    )

    with pytest.raises(ProjectIdentityConflict, match="conflicting project identity"):
        source.project_identity()


def test_only_the_source_home_can_append_the_transfer(manifest) -> None:
    source = HistoryManager(manifest, expected_space_id=SOURCE_SPACE_ID)
    source.claim_project_identity("created", project_id=PROJECT_ID)
    target = HistoryManager(load_manifest(manifest.path), expected_space_id=TARGET_SPACE_ID)

    with pytest.raises(ProjectIdentityConflict, match="current source space"):
        target.transfer_project_home(
            project_id=PROJECT_ID,
            previous_home_space_id=SOURCE_SPACE_ID,
            new_home_space_id=TARGET_SPACE_ID,
            source_released_by=_actor(SOURCE_SPACE_ID, "Source owner"),
            target_admitted_by=_actor(TARGET_SPACE_ID, "Team reviewer"),
        )
    assert len(source.load_patches()) == 1


@pytest.mark.parametrize(
    "transfer",
    [
        _transfer(project_id="b79f76ec-9635-43bb-85ee-f9dc312fdc09"),
        _transfer(
            previous_home_space_id=OTHER_SPACE_ID,
            source_released_by=_actor(OTHER_SPACE_ID, "Other source owner"),
        ),
    ],
    ids=["different-project", "wrong-previous-home"],
)
def test_replay_refuses_a_transfer_that_does_not_continue_canonical_home(
    manifest,
    transfer: ProjectHomeTransfer,
) -> None:
    guarded = HistoryManager(manifest, expected_space_id=SOURCE_SPACE_ID)
    guarded.claim_project_identity("created", project_id=PROJECT_ID)
    HistoryManager(load_manifest(manifest.path)).append(_transfer_patch(transfer))

    with pytest.raises(ProjectIdentityConflict, match="does not continue"):
        guarded.project_identity()
    with pytest.raises(ProjectIdentityConflict, match="does not continue"):
        guarded.append(_ordinary_patch(manifest.project.truth_scope))


def test_replay_refuses_a_home_transfer_before_project_identity(manifest) -> None:
    HistoryManager(manifest).append(_transfer_patch(_transfer()))
    guarded = HistoryManager(load_manifest(manifest.path), expected_space_id=TARGET_SPACE_ID)

    with pytest.raises(ProjectIdentityConflict, match="before establishing"):
        guarded.project_identity()
    with pytest.raises(ProjectIdentityConflict, match="before establishing"):
        guarded.initialize()


def test_home_transfer_shape_binds_each_actor_to_its_own_space() -> None:
    with pytest.raises(ValidationError, match="source-release actor"):
        _transfer(source_released_by=_actor(TARGET_SPACE_ID, "Wrong source"))
    with pytest.raises(ValidationError, match="target-admission actor"):
        _transfer(target_admitted_by=_actor(SOURCE_SPACE_ID, "Wrong target"))
    with pytest.raises(ValidationError, match="must change spaces"):
        _transfer(
            new_home_space_id=SOURCE_SPACE_ID,
            target_admitted_by=_actor(SOURCE_SPACE_ID, "Same-space reviewer"),
        )


@pytest.mark.parametrize("mode", ["admission", "replay"])
def test_agents_cannot_author_or_attach_a_home_transfer(mode: str) -> None:
    transfer = _transfer()
    authored = _transfer_patch(transfer).model_copy(update={"author": "agent", "producer": "agent"})
    ordinary = _ordinary_patch(["repo"]).model_copy(update={"project_home_transfer": transfer})

    authored_report = validate_patch(GraphState(), authored, [], mode=mode)  # type: ignore[arg-type]
    ordinary_report = validate_patch(
        GraphState(project_truth_scope=["repo"]),
        ordinary,
        ["repo"],
        mode=mode,  # type: ignore[arg-type]
    )

    assert {"wrong-author", "wrong-producer"}.issubset(
        {message.code for message in authored_report.messages}
    )
    assert "unexpected-project-home-transfer" in {
        message.code for message in ordinary_report.messages
    }


def test_existing_identity_patch_remains_byte_compatible(manifest) -> None:
    history = HistoryManager(manifest, expected_space_id=SOURCE_SPACE_ID)
    identity = history.claim_project_identity("created", project_id=PROJECT_ID)
    path = manifest.research_dir / "patches" / "000001.json"
    document = json.loads(path.read_bytes())
    document.pop("project_home_transfer")
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    before = path.read_bytes()

    assert "project_home_transfer" not in document
    assert HistoryManager(load_manifest(manifest.path)).project_identity() == identity
    assert path.read_bytes() == before


def test_transfer_revision_summary_is_visible_without_changing_graph_semantics(manifest) -> None:
    history = HistoryManager(manifest, expected_space_id=SOURCE_SPACE_ID)
    history.claim_project_identity("created", project_id=PROJECT_ID)
    before = history.state()
    history.transfer_project_home(
        project_id=PROJECT_ID,
        previous_home_space_id=SOURCE_SPACE_ID,
        new_home_space_id=TARGET_SPACE_ID,
        source_released_by=_actor(SOURCE_SPACE_ID, "Source owner"),
        target_admitted_by=_actor(TARGET_SPACE_ID, "Team reviewer"),
    )
    after = history.state()

    assert after.nodes == before.nodes
    assert after.edges == before.edges
    assert after.coverage == before.coverage
    assert history.revision_summaries(2, 2)[0]["sentences"] == [
        f"Project moved from {SOURCE_SPACE_ID} to {TARGET_SPACE_ID}."
    ]
