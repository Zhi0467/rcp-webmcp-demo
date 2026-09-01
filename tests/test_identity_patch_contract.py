from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from rcp.agents.schema import AgentPatch
from rcp.core.models import AuthorizedHuman, GraphState, Patch, ProjectIdentity
from rcp.core.operations import CoverageUpdate, SetCoverageOperation
from rcp.core.validation import validate_patch

PROJECT_ID = "123e4567-e89b-42d3-a456-426614174000"
SPACE_ID = "123e4567-e89b-42d3-b456-426614174000"
USER_ID = "123e4567-e89b-42d3-8456-426614174000"


def _authorized_human() -> AuthorizedHuman:
    return AuthorizedHuman(
        space_id=SPACE_ID,
        user_id=USER_ID,
        display_name="Ada Lovelace",
    )


def _identity_patch(*, action: str = "created") -> Patch:
    return Patch(
        kind="identity",
        author=None,
        producer="system",
        summary="Recorded the project's canonical identity.",
        ops=[],
        project_identity={
            "project_id": PROJECT_ID,
            "home_space_id": SPACE_ID,
            "action": action,
        },
    )


@pytest.mark.parametrize("action", ["created", "adopted"])
@pytest.mark.parametrize("mode", ["admission", "replay"])
def test_system_identity_revision_is_valid_without_graph_scope(action: str, mode: str) -> None:
    patch = _identity_patch(action=action)

    report = validate_patch(
        GraphState(project_truth_scope=["repo"]),
        patch,
        ["repo"],
        mode=mode,  # type: ignore[arg-type]
    )

    assert not report.rejected
    assert patch.author is None
    assert patch.producer == "system"


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        (
            "ops",
            [SetCoverageOperation(op="set_coverage", coverage=CoverageUpdate())],
            "identity-has-operations",
        ),
        ("run_truth_scope", ["repo"], "identity-has-run-scope"),
        ("repositories_read", ["repo"], "identity-has-run-scope"),
        ("processed_cursors", {"session": "cursor"}, "identity-has-cursors"),
        ("source_operation_id", "operation", "identity-has-operation-id"),
        ("source_effect_id", "effect", "identity-has-operation-id"),
        ("source_effect_sha256", "0" * 64, "identity-has-operation-id"),
        ("human_action", "decision_choice", "identity-has-human-action"),
        ("agent_action", "decision_choice", "identity-has-agent-action"),
        (
            "experiment_control_node_id",
            "exp/controlled",
            "identity-has-experiment-control",
        ),
        ("authorized_by", _authorized_human(), "identity-has-attribution"),
        ("profile", "ordinary", "identity-has-attribution"),
        ("task_id", "task/direct", "identity-has-attribution"),
    ],
)
@pytest.mark.parametrize("mode", ["admission", "replay"])
def test_identity_revision_rejects_non_identity_metadata(
    field: str, value: object, code: str, mode: str
) -> None:
    patch = _identity_patch().model_copy(update={field: value})

    report = validate_patch(
        GraphState(project_truth_scope=["repo"]),
        patch,
        ["repo"],
        mode=mode,  # type: ignore[arg-type]
    )

    assert report.rejected
    assert code in {message.code for message in report.messages}


@pytest.mark.parametrize("mode", ["admission", "replay"])
def test_identity_payload_is_required_and_reserved_for_identity_revisions(mode: str) -> None:
    missing = _identity_patch().model_copy(update={"project_identity": None})
    ordinary = Patch(
        kind="refresh",
        author="agent",
        summary="Tried to attach identity to research work.",
        ops=[],
        run_truth_scope=["repo"],
        project_identity=ProjectIdentity(
            project_id=PROJECT_ID,
            home_space_id=SPACE_ID,
            action="created",
        ),
    )

    missing_report = validate_patch(GraphState(), missing, [], mode=mode)  # type: ignore[arg-type]
    ordinary_report = validate_patch(
        GraphState(project_truth_scope=["repo"]),
        ordinary,
        ["repo"],
        mode=mode,  # type: ignore[arg-type]
    )

    assert "missing-project-identity" in {message.code for message in missing_report.messages}
    assert "unexpected-project-identity" in {message.code for message in ordinary_report.messages}


@pytest.mark.parametrize("mode", ["admission", "replay"])
def test_system_producer_is_reserved_and_identity_has_no_legacy_author(mode: str) -> None:
    system_refresh = Patch(
        kind="refresh",
        author="agent",
        producer="system",
        summary="Tried to use the reserved producer.",
        ops=[],
        run_truth_scope=["repo"],
    )
    authored_identity = _identity_patch().model_copy(update={"author": "human"})
    human_identity = _identity_patch().model_copy(update={"producer": "human"})

    reports = [
        validate_patch(GraphState(), system_refresh, ["repo"], mode=mode),  # type: ignore[arg-type]
        validate_patch(GraphState(), authored_identity, [], mode=mode),  # type: ignore[arg-type]
        validate_patch(GraphState(), human_identity, [], mode=mode),  # type: ignore[arg-type]
    ]

    assert "system-producer-forbidden" in {message.code for message in reports[0].messages}
    assert "wrong-author" in {message.code for message in reports[1].messages}
    assert "wrong-producer" in {message.code for message in reports[2].messages}


def test_legacy_patch_json_infers_producer_without_changing_author_semantics() -> None:
    patch = Patch.model_validate(
        {
            "kind": "refresh",
            "author": "agent",
            "summary": "A historical patch without producer or attribution.",
            "ops": [],
            "run_truth_scope": ["repo"],
        }
    )

    assert patch.author == "agent"
    assert patch.producer == "agent"
    assert patch.authorized_by is None
    assert patch.profile is None
    assert patch.task_id is None
    assert not validate_patch(
        GraphState(project_truth_scope=["repo"]), patch, ["repo"], mode="replay"
    ).rejected


def test_base_attribution_is_strict_additive_with_nullable_episode_lineage() -> None:
    authorized_by = _authorized_human()
    patch = Patch(
        kind="work",
        author="agent",
        summary="Recorded attributable ordinary work.",
        ops=[],
        run_truth_scope=["repo"],
        authorized_by=authorized_by,
        profile="ordinary",
        task_id="task/direct",
    )

    dumped = patch.model_dump(mode="json")
    assert dumped["authorized_by"] == {
        "space_id": SPACE_ID,
        "user_id": USER_ID,
        "display_name": "Ada Lovelace",
    }
    assert dumped["profile"] == "ordinary"
    assert dumped["task_id"] == "task/direct"
    assert dumped["episode_id"] is None

    with pytest.raises(ValidationError):
        AuthorizedHuman.model_validate(
            {
                **authorized_by.model_dump(),
                "current_display_name": "A name must not be re-resolved during replay.",
            }
        )
    for invalid_name in ("line one\nline two", "x" * 121):
        with pytest.raises(ValidationError):
            AuthorizedHuman.model_validate(
                {**authorized_by.model_dump(), "display_name": invalid_name}
            )
    elevated = Patch.model_validate({**dumped, "profile": "orchestrator"})
    assert elevated.profile == "orchestrator"
    assert elevated.episode_id is None


@pytest.mark.parametrize("mode", ["admission", "replay"])
@pytest.mark.parametrize(
    "attribution",
    [
        {"authorized_by": _authorized_human()},
        {"profile": "ordinary"},
        {"task_id": "task/direct"},
        {"authorized_by": _authorized_human(), "profile": "ordinary"},
        {"authorized_by": _authorized_human(), "task_id": "task/direct"},
        {"profile": "ordinary", "task_id": "task/direct"},
        {
            "authorized_by": _authorized_human(),
            "profile": "ordinary",
            "task_id": "",
        },
    ],
)
def test_agent_attribution_is_atomic(attribution: dict[str, object], mode: str) -> None:
    patch = Patch(
        kind="work",
        author="agent",
        summary="Recorded an incomplete attribution envelope.",
        ops=[],
        run_truth_scope=["repo"],
    ).model_copy(update=attribution)

    report = validate_patch(
        GraphState(project_truth_scope=["repo"]),
        patch,
        ["repo"],
        mode=mode,  # type: ignore[arg-type]
    )

    assert "invalid-agent-attribution" in {message.code for message in report.messages}


@pytest.mark.parametrize("mode", ["admission", "replay"])
def test_complete_agent_attribution_is_structurally_valid(mode: str) -> None:
    patch = Patch(
        kind="work",
        author="agent",
        summary="Recorded complete attribution.",
        ops=[],
        run_truth_scope=["repo"],
        authorized_by=_authorized_human(),
        profile="ordinary",
        task_id="task/direct",
    )

    report = validate_patch(
        GraphState(project_truth_scope=["repo"]),
        patch,
        ["repo"],
        mode=mode,  # type: ignore[arg-type]
    )

    assert "invalid-agent-attribution" not in {message.code for message in report.messages}


@pytest.mark.parametrize("mode", ["admission", "replay"])
@pytest.mark.parametrize(
    "attribution",
    [
        {"profile": "ordinary"},
        {"task_id": "task/direct"},
        {"authorized_by": _authorized_human(), "profile": "ordinary"},
        {"authorized_by": _authorized_human(), "task_id": "task/direct"},
        {
            "authorized_by": _authorized_human(),
            "profile": "ordinary",
            "task_id": "task/direct",
        },
    ],
)
def test_human_attribution_forbids_agent_metadata(
    attribution: dict[str, object], mode: str
) -> None:
    patch = Patch(
        kind="approval",
        author="human",
        summary="Recorded invalid human attribution.",
        ops=[],
    ).model_copy(update=attribution)

    report = validate_patch(GraphState(), patch, [], mode=mode)  # type: ignore[arg-type]

    assert "invalid-human-attribution" in {message.code for message in report.messages}


@pytest.mark.parametrize("mode", ["admission", "replay"])
def test_authorized_by_alone_is_valid_human_attribution(mode: str) -> None:
    patch = Patch(
        kind="approval",
        author="human",
        summary="Recorded complete human attribution.",
        ops=[],
        authorized_by=_authorized_human(),
    )

    report = validate_patch(GraphState(), patch, [], mode=mode)  # type: ignore[arg-type]

    assert "invalid-human-attribution" not in {message.code for message in report.messages}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("project_id", str(uuid.uuid1())),
        ("home_space_id", SPACE_ID.upper()),
        ("home_space_id", "not-a-uuid"),
    ],
)
def test_project_identity_requires_canonical_uuid4(field: str, value: str) -> None:
    identity = {
        "project_id": PROJECT_ID,
        "home_space_id": SPACE_ID,
        "action": "created",
    }
    identity[field] = value

    with pytest.raises(ValidationError, match="canonical UUIDv4"):
        ProjectIdentity.model_validate(identity)

    with pytest.raises(ValidationError):
        ProjectIdentity.model_validate({**identity, field: PROJECT_ID, "extra": True})


def test_agent_deliverable_schema_cannot_claim_identity_or_attribution() -> None:
    with pytest.raises(ValidationError):
        AgentPatch.model_validate(
            {
                "summary": "Tried to mint a project identity.",
                "ops": [],
                "project_identity": _identity_patch().project_identity.model_dump(),  # type: ignore[union-attr]
                "producer": "system",
                "authorized_by": {
                    "space_id": SPACE_ID,
                    "user_id": USER_ID,
                    "display_name": "Ada Lovelace",
                },
            }
        )
