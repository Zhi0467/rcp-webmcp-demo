"""Strict persisted contracts for graph-transition provenance.

Operation payloads remain exclusively in ``Patch.ops``.  The trace records
ordered references to those payloads so replay and operational reconciliation
do not need a second action representation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictTransitionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class GraphTargetRef(_StrictTransitionModel):
    kind: Literal["main", "branch"] = "main"
    branch_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def require_matching_branch_id(self) -> GraphTargetRef:
        if self.kind == "main" and self.branch_id is not None:
            raise ValueError("main graph targets cannot carry a branch_id")
        if self.kind == "branch" and self.branch_id is None:
            raise ValueError("branch graph targets require a branch_id")
        return self

    @property
    def key(self) -> str:
        return "main" if self.kind == "main" else f"branch:{self.branch_id}"


class GraphHeadRef(_StrictTransitionModel):
    target: GraphTargetRef = Field(default_factory=GraphTargetRef)
    revision: int = Field(ge=0)
    transition_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class GraphAttentionProjection(_StrictTransitionModel):
    """Exact graph memberships consumed by attention and Runs surfaces."""

    pending_proposal_ids: list[str] = Field(default_factory=list)
    decisions_awaiting_choice_ids: list[str] = Field(default_factory=list)
    open_blocker_ids: list[str] = Field(default_factory=list)


class TransitionCauseRef(_StrictTransitionModel):
    kind: Literal["action", "event"]
    action_index: int | None = Field(default=None, ge=0)
    event_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def require_exact_reference(self) -> TransitionCauseRef:
        if self.kind == "action" and self.action_index is None:
            raise ValueError("action causes require action_index")
        if self.kind == "event" and self.event_id is None:
            raise ValueError("event causes require event_id")
        if self.kind == "action" and self.event_id is not None:
            raise ValueError("action causes cannot carry event_id")
        if self.kind == "event" and self.action_index is not None:
            raise ValueError("event causes cannot carry action_index")
        return self


class TransitionInitiatingGroup(_StrictTransitionModel):
    group_id: str = Field(min_length=1)
    operation_indexes: list[int] = Field(min_length=1)
    summary: str
    change_summary: list[str] = Field(default_factory=list)
    human_action: Literal["decision_choice"] | None = None
    agent_action: Literal["decision_choice"] | None = None


class TransitionGeneratedAction(_StrictTransitionModel):
    operation_index: int = Field(ge=0)
    rule_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    cause: TransitionCauseRef


class TransitionEvent(_StrictTransitionModel):
    event_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    event_type: Literal[
        "node_status_changed",
        "guidance_invalidated",
        "guidance_refreshed",
    ]
    cause: TransitionCauseRef
    node_id: str
    field: str
    before: str | bool | None = None
    after: str | bool | None = None


class TransitionTrace(_StrictTransitionModel):
    schema_generation: Literal[1] = 1
    transition_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    pre_head: GraphHeadRef
    ruleset_tag: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    initiating_groups: list[TransitionInitiatingGroup] = Field(min_length=1)
    generated_actions: list[TransitionGeneratedAction] = Field(default_factory=list)
    lifecycle_events: list[TransitionEvent] = Field(default_factory=list)
    expanded_ops_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class TransitionTrigger(_StrictTransitionModel):
    operation: str
    node_types: list[str] = Field(default_factory=list)
    node_fields: list[str] = Field(default_factory=list)
    relations: list[str] = Field(default_factory=list)


class TransitionTriggerManifest(_StrictTransitionModel):
    ruleset_tag: str
    triggers: list[TransitionTrigger]


class TransitionConflictDetail(_StrictTransitionModel):
    operation_index: int | None = Field(default=None, ge=0)
    rule_id: str | None = None
    cause_chain: list[TransitionCauseRef] = Field(default_factory=list)
    affected_ids: list[str] = Field(default_factory=list)
    invariant: str
    message: str


class GuidanceFieldValidity(_StrictTransitionModel):
    status: Literal["empty", "current", "stale"]
    invalidated_by_event_id: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )


class ExperimentGuidanceValidity(_StrictTransitionModel):
    current_summary: GuidanceFieldValidity
    next_action: GuidanceFieldValidity
