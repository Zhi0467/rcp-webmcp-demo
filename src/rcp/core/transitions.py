"""Deterministic preparation and replay support for canonical graph transitions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rcp.control import ExperimentControlState, experiment_control_dependencies
from rcp.core.attention import project_graph_attention
from rcp.core.materialize import (
    apply_transition_generated_operation,
    apply_valid_operation,
    apply_valid_patch,
)
from rcp.core.models import Evidence, Experiment, GraphState, Patch, ReplayFailure
from rcp.core.operations import (
    GraphOperation,
    NodeUpdate,
    UpdateNodesOperation,
)
from rcp.core.transition_models import (
    ExperimentGuidanceValidity,
    GraphAttentionProjection,
    GraphHeadRef,
    GraphTargetRef,
    GuidanceFieldValidity,
    TransitionCauseRef,
    TransitionConflictDetail,
    TransitionEvent,
    TransitionGeneratedAction,
    TransitionInitiatingGroup,
    TransitionTrace,
    TransitionTrigger,
    TransitionTriggerManifest,
)

TRANSITION_RULESET_TAG = "rcp.lifecycle.v2"
GUIDANCE_RULE_ID = "experiment.guidance-validity.v1"
STATUS_EVENT_RULE_ID = "lifecycle.status-events.v1"
DEFAULT_MAX_RULE_FIRINGS = 128


@dataclass(frozen=True)
class TransitionRule:
    rule_id: str


RULE_REGISTRY: tuple[TransitionRule, ...] = (
    TransitionRule(STATUS_EVENT_RULE_ID),
    TransitionRule(GUIDANCE_RULE_ID),
)


class TransitionConflict(ValueError):
    def __init__(self, details: Iterable[TransitionConflictDetail]) -> None:
        self.details = list(details)
        super().__init__("; ".join(detail.message for detail in self.details))


class ProjectTransitionProjection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    head: GraphHeadRef
    graph: GraphState
    attention: GraphAttentionProjection
    experiment_control: dict[str, ExperimentControlState]
    guidance_validity: dict[str, ExperimentGuidanceValidity]
    ruleset_tag: str
    transition_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    canonical: bool = True
    base_head: GraphHeadRef | None = None


class PreparedTransition(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    patch: Patch
    projection: ProjectTransitionProjection


class CommittedTransition(PreparedTransition):
    pass


class TransitionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pre_head: GraphHeadRef
    patches: list[Patch] = Field(min_length=1)
    ruleset_tag: str = Field(
        default=TRANSITION_RULESET_TAG,
        pattern=r"^[a-z][a-z0-9_.-]*$",
    )


def transition_trigger_manifest() -> TransitionTriggerManifest:
    """Return the conservative browser routing manifest for this ruleset."""

    return TransitionTriggerManifest(
        ruleset_tag=TRANSITION_RULESET_TAG,
        triggers=[
            TransitionTrigger(
                operation="update_nodes",
                node_types=["blocker", "decision", "experiment", "hypothesis"],
                node_fields=[
                    "status",
                    "selected_option",
                    "current_summary",
                    "next_action",
                ],
            ),
            TransitionTrigger(
                operation="create_edges",
                relations=[
                    "blocked_by",
                    "governed_by",
                    "tests",
                    "supports",
                    "weakens",
                    "refutes",
                    "inconclusive",
                    "contradicts",
                ],
            ),
            TransitionTrigger(
                operation="remove_edges",
                relations=[
                    "blocked_by",
                    "governed_by",
                    "tests",
                    "supports",
                    "weakens",
                    "refutes",
                    "inconclusive",
                    "contradicts",
                ],
            ),
            TransitionTrigger(
                operation="create_proposals",
                node_types=["decision"],
            ),
            TransitionTrigger(
                operation="resolve_proposals",
                node_types=["decision"],
            ),
            TransitionTrigger(
                operation="withdraw_proposals",
                node_types=["decision"],
            ),
        ],
    )


class GraphTransitionManager:
    """Prepare already-attributed initiating patches to deterministic closure."""

    def __init__(self, *, max_rule_firings: int = DEFAULT_MAX_RULE_FIRINGS) -> None:
        if max_rule_firings < 1:
            raise ValueError("max_rule_firings must be positive")
        self.max_rule_firings = max_rule_firings

    def prepare_validated(
        self,
        state: GraphState,
        patches: list[Patch],
        *,
        pre_head: GraphHeadRef | None = None,
    ) -> PreparedTransition:
        if not patches:
            raise ValueError("a graph transition requires at least one initiating patch")
        revision = patches[0].revision
        if revision <= state.revision or any(patch.revision != revision for patch in patches):
            raise ValueError("all initiating patches must share the next canonical revision")
        if any(patch.transition is not None for patch in patches):
            raise ValueError("initiating patches cannot contain a committed transition trace")
        self._require_compatible_envelopes(patches)

        target_head = pre_head or GraphHeadRef(revision=state.revision)
        if target_head.revision != state.revision:
            raise ValueError("transition pre-head does not match the supplied graph state")

        source_actions: list[tuple[Patch, GraphOperation]] = []
        groups: list[TransitionInitiatingGroup] = []
        for group_index, patch in enumerate(patches):
            indexes: list[int] = []
            for operation in patch.ops:
                indexes.append(len(source_actions))
                source_actions.append((patch, operation))
            if not indexes:
                continue
            groups.append(
                TransitionInitiatingGroup(
                    group_id=f"group-{group_index + 1}",
                    operation_indexes=indexes,
                    summary=patch.summary,
                    change_summary=list(patch.change_summary),
                    human_action=patch.human_action,
                    agent_action=patch.agent_action,
                )
            )
        if not source_actions:
            raise ValueError("a graph transition requires at least one semantic operation")

        timeline, initiating_state = self._apply_timeline(state, source_actions)
        dependency_causes = self._dependency_change_causes(timeline)
        explicit_guidance = self._explicit_guidance_updates(source_actions)
        generated_operations = self._guidance_actions(
            initiating_state,
            dependency_causes,
            explicit_guidance,
        )
        if len(generated_operations) > self.max_rule_firings:
            raise TransitionConflict(
                [
                    TransitionConflictDetail(
                        rule_id=GUIDANCE_RULE_ID,
                        affected_ids=sorted(dependency_causes),
                        invariant="bounded deterministic rule closure",
                        message=(
                            "Graph transition rule closure exceeded "
                            f"{self.max_rule_firings} generated actions."
                        ),
                    )
                ]
            )

        final_state = initiating_state
        generated_refs: list[TransitionGeneratedAction] = []
        generated_timeline: list[tuple[int, GraphState, GraphState]] = []
        expanded_operations = [operation for _patch, operation in source_actions]
        for operation, cause_index in generated_operations:
            operation_index = len(expanded_operations)
            before_generated = final_state
            try:
                final_state = apply_transition_generated_operation(
                    final_state,
                    patches[0],
                    operation,
                )
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                raise TransitionConflict(
                    [
                        TransitionConflictDetail(
                            operation_index=operation_index,
                            rule_id=GUIDANCE_RULE_ID,
                            cause_chain=[
                                TransitionCauseRef(kind="action", action_index=cause_index)
                            ],
                            invariant="generated guidance-validity action is applicable",
                            message=f"Generated transition action could not be applied: {exc}",
                        )
                    ]
                ) from exc
            expanded_operations.append(operation)
            generated_timeline.append((operation_index, before_generated, final_state))
            generated_refs.append(
                TransitionGeneratedAction(
                    operation_index=operation_index,
                    rule_id=GUIDANCE_RULE_ID,
                    cause=TransitionCauseRef(kind="action", action_index=cause_index),
                )
            )

        ruleset_tag = TRANSITION_RULESET_TAG
        transition_id = _transition_id(
            target_head,
            patches,
            groups,
            source_actions,
            ruleset_tag=ruleset_tag,
            generated_actions=generated_refs,
        )
        lifecycle_events = self._lifecycle_events(
            transition_id,
            timeline,
            generated_timeline,
            generated_refs,
            expanded_operations,
        )
        trace = TransitionTrace(
            transition_id=transition_id,
            pre_head=target_head,
            ruleset_tag=ruleset_tag,
            initiating_groups=groups,
            generated_actions=generated_refs,
            lifecycle_events=lifecycle_events,
            expanded_ops_sha256=_operations_sha256(expanded_operations),
        )
        combined = _combined_patch(patches, expanded_operations, trace)
        # Apply once with the persisted envelope. This proves the exact replay
        # payload, rather than only the per-action staging path, reaches the
        # same candidate.
        persisted_candidate = apply_valid_patch(state, combined)
        if persisted_candidate != final_state:
            raise TransitionConflict(
                [
                    TransitionConflictDetail(
                        invariant="expanded operation replay equivalence",
                        message="Recorded expanded operations do not reproduce the prepared state.",
                    )
                ]
            )
        projection = project_transition_projection(persisted_candidate, trace, canonical=True)
        return PreparedTransition(patch=combined, projection=projection)

    @staticmethod
    def _require_compatible_envelopes(patches: list[Patch]) -> None:
        first = patches[0]
        fields = (
            "schema_generation",
            "kind",
            "author",
            "producer",
            "source_operation_id",
            "source_effect_id",
            "source_effect_sha256",
            "project_identity",
            "project_home_transfer",
            "authorized_by",
            "profile",
            "task_id",
            "episode_id",
            "experiment_control_node_id",
            "experiment_decision_bundle",
        )
        for patch in patches[1:]:
            mismatched = [
                field for field in fields if getattr(patch, field) != getattr(first, field)
            ]
            if mismatched:
                raise ValueError(
                    "initiating patches in one transition must share producer provenance: "
                    + ", ".join(mismatched)
                )

    @staticmethod
    def _apply_timeline(
        initial: GraphState,
        actions: list[tuple[Patch, GraphOperation]],
    ) -> tuple[list[tuple[int, GraphState, GraphState]], GraphState]:
        state = initial
        timeline: list[tuple[int, GraphState, GraphState]] = []
        for index, (patch, operation) in enumerate(actions):
            before = state
            try:
                state = apply_valid_operation(before, patch, operation)
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                raise TransitionConflict(
                    [
                        TransitionConflictDetail(
                            operation_index=index,
                            invariant="initiating action is atomically applicable",
                            message=f"Initiating operation {index} could not be applied: {exc}",
                        )
                    ]
                ) from exc
            timeline.append((index, before, state))
        state = state.model_copy(update={"revision": actions[0][0].revision})
        return timeline, state

    @staticmethod
    def _dependency_change_causes(
        timeline: list[tuple[int, GraphState, GraphState]],
    ) -> dict[str, int]:
        causes: dict[str, int] = {}
        for action_index, before, after in timeline:
            experiment_ids = sorted(
                node_id
                for node_id, node in {**before.nodes, **after.nodes}.items()
                if isinstance(node, Experiment) and node_id in after.nodes
            )
            for experiment_id in experiment_ids:
                # Creating an Experiment establishes its initial guidance
                # against the graph at that action.  Absence before creation is
                # not itself a dependency change.  A later edge or governance
                # mutation still observes the Experiment on both sides and
                # invalidates that guidance normally.
                if not isinstance(before.nodes.get(experiment_id), Experiment):
                    continue
                before_signature = _experiment_dependency_signature(before, experiment_id)
                after_signature = _experiment_dependency_signature(after, experiment_id)
                if before_signature != after_signature:
                    causes.setdefault(experiment_id, action_index)
        return causes

    @staticmethod
    def _explicit_guidance_updates(
        actions: list[tuple[Patch, GraphOperation]],
    ) -> dict[tuple[str, str], int]:
        updates: dict[tuple[str, str], int] = {}
        for action_index, (_patch, operation) in enumerate(actions):
            if not isinstance(operation, UpdateNodesOperation):
                continue
            for update in operation.nodes:
                for field in ("current_summary", "next_action"):
                    if field in update.changes:
                        updates[(update.id, field)] = action_index
        return updates

    @staticmethod
    def _guidance_actions(
        state: GraphState,
        invalidation_causes: dict[str, int],
        explicit_updates: dict[tuple[str, str], int],
    ) -> list[tuple[UpdateNodesOperation, int]]:
        generated: list[tuple[UpdateNodesOperation, int]] = []
        experiment_ids = sorted(
            node_id for node_id, node in state.nodes.items() if isinstance(node, Experiment)
        )
        for experiment_id in experiment_ids:
            experiment = state.nodes[experiment_id]
            assert isinstance(experiment, Experiment)
            changes_by_cause: dict[int, dict[str, Any]] = {}
            invalidation_cause = invalidation_causes.get(experiment_id)
            for field, stale_field in (
                ("current_summary", "current_summary_stale"),
                ("next_action", "next_action_stale"),
            ):
                value = getattr(experiment, field)
                stale = getattr(experiment, stale_field)
                explicit_cause = explicit_updates.get((experiment_id, field))
                if invalidation_cause is not None:
                    desired = bool(value)
                    cause = invalidation_cause
                elif explicit_cause is not None:
                    desired = False
                    cause = explicit_cause
                else:
                    continue
                if stale != desired:
                    changes_by_cause.setdefault(cause, {})[stale_field] = desired
            for cause in sorted(changes_by_cause):
                generated.append(
                    (
                        UpdateNodesOperation(
                            op="update_nodes",
                            nodes=[
                                NodeUpdate(
                                    id=experiment_id,
                                    changes=changes_by_cause[cause],
                                )
                            ],
                        ),
                        cause,
                    )
                )
        return generated

    @staticmethod
    def _lifecycle_events(
        transition_id: str,
        initiating_timeline: list[tuple[int, GraphState, GraphState]],
        generated_timeline: list[tuple[int, GraphState, GraphState]],
        generated: list[TransitionGeneratedAction],
        expanded_operations: list[GraphOperation],
    ) -> list[TransitionEvent]:
        pending: list[tuple[int, str, str, str | bool | None, str | bool | None, str]] = []
        for action_index, before, after in initiating_timeline:
            for node_id in sorted(set(before.nodes).intersection(after.nodes)):
                previous_status = getattr(before.nodes[node_id], "status", None)
                current_status = getattr(after.nodes[node_id], "status", None)
                if previous_status == current_status:
                    continue
                pending.append(
                    (
                        action_index,
                        node_id,
                        "status",
                        previous_status,
                        current_status,
                        "node_status_changed",
                    )
                )

        generated_by_index = {item.operation_index: item for item in generated}
        for operation_index, before, after in generated_timeline:
            assert operation_index in generated_by_index
            operation = expanded_operations[operation_index]
            assert isinstance(operation, UpdateNodesOperation)
            for update in operation.nodes:
                previous_node = before.nodes.get(update.id)
                current_node = after.nodes.get(update.id)
                for field in sorted(update.changes):
                    previous_value = getattr(previous_node, field, None)
                    current_value = getattr(current_node, field, None)
                    pending.append(
                        (
                            operation_index,
                            update.id,
                            field,
                            previous_value,
                            current_value,
                            "guidance_invalidated" if current_value else "guidance_refreshed",
                        )
                    )

        events: list[TransitionEvent] = []
        for ordinal, (cause, node_id, field, old, new, event_type) in enumerate(pending):
            payload = {
                "ordinal": ordinal,
                "event_type": event_type,
                "cause": cause,
                "node_id": node_id,
                "field": field,
                "before": old,
                "after": new,
            }
            events.append(
                TransitionEvent(
                    event_id=_sha256({"transition_id": transition_id, **payload}),
                    event_type=event_type,
                    cause=TransitionCauseRef(kind="action", action_index=cause),
                    node_id=node_id,
                    field=field,
                    before=old,
                    after=new,
                )
            )
        return events


def project_transition_projection(
    state: GraphState,
    trace: TransitionTrace,
    *,
    canonical: bool,
) -> ProjectTransitionProjection:
    invalidation_event_by_field = {
        (event.node_id, event.field): event.event_id
        for event in trace.lifecycle_events
        if event.event_type == "guidance_invalidated"
    }
    return _project_projection(
        state,
        head=GraphHeadRef(
            target=trace.pre_head.target,
            revision=state.revision,
            transition_id=trace.transition_id,
        ),
        ruleset_tag=trace.ruleset_tag,
        transition_id=trace.transition_id,
        canonical=canonical,
        base_head=None if canonical else trace.pre_head,
        invalidation_event_by_field=invalidation_event_by_field,
    )


def current_project_projection(
    state: GraphState,
    *,
    transition_id: str | None = None,
    target: GraphTargetRef | None = None,
) -> ProjectTransitionProjection:
    """Build a coherent current snapshot when no new mutation was committed."""

    return _project_projection(
        state,
        head=GraphHeadRef(
            target=target or GraphTargetRef(),
            revision=state.revision,
            transition_id=transition_id,
        ),
        ruleset_tag=TRANSITION_RULESET_TAG,
        transition_id=transition_id,
        canonical=True,
        invalidation_event_by_field={},
    )


def _project_projection(
    state: GraphState,
    *,
    head: GraphHeadRef,
    ruleset_tag: str,
    transition_id: str | None,
    canonical: bool,
    invalidation_event_by_field: dict[tuple[str, str], str],
    base_head: GraphHeadRef | None = None,
) -> ProjectTransitionProjection:
    from rcp.control import derive_experiment_control_state

    controls: dict[str, ExperimentControlState] = {}
    guidance: dict[str, ExperimentGuidanceValidity] = {}
    for node_id, node in sorted(state.nodes.items()):
        if not isinstance(node, Experiment):
            continue
        controls[node_id] = derive_experiment_control_state(state, node_id)
        guidance[node_id] = ExperimentGuidanceValidity(
            current_summary=_guidance_field_validity(
                node.current_summary,
                node.current_summary_stale,
                invalidation_event_by_field.get((node_id, "current_summary_stale")),
            ),
            next_action=_guidance_field_validity(
                node.next_action,
                node.next_action_stale,
                invalidation_event_by_field.get((node_id, "next_action_stale")),
            ),
        )
    return ProjectTransitionProjection(
        head=head,
        graph=state,
        attention=project_graph_attention(state),
        experiment_control=controls,
        guidance_validity=guidance,
        ruleset_tag=ruleset_tag,
        transition_id=transition_id,
        canonical=canonical,
        base_head=base_head,
    )


def validate_transition_trace(state: GraphState, patch: Patch) -> list[Patch]:
    """Validate structural provenance and return replay source groups.

    This deliberately does not execute rule functions.  Historical replay
    validates the recorded trace and applies its expanded operations exactly.
    """

    trace = patch.transition
    if trace is None:
        raise ValueError("transition patch is missing its trace")
    if trace.pre_head.revision != state.revision:
        raise ValueError(
            f"transition pre-head {trace.pre_head.revision} does not match replay head "
            f"{state.revision}"
        )
    if _operations_sha256(patch.ops) != trace.expanded_ops_sha256:
        raise ValueError("transition expanded operation digest does not match Patch.ops")

    source_indexes = [
        index for group in trace.initiating_groups for index in group.operation_indexes
    ]
    generated_indexes = [item.operation_index for item in trace.generated_actions]
    if len(source_indexes) != len(set(source_indexes)):
        raise ValueError("transition initiating operation indexes overlap")
    if len(generated_indexes) != len(set(generated_indexes)):
        raise ValueError("transition generated operation indexes overlap")
    if set(source_indexes).intersection(generated_indexes):
        raise ValueError("transition initiating and generated operation indexes overlap")
    if sorted([*source_indexes, *generated_indexes]) != list(range(len(patch.ops))):
        raise ValueError("transition trace must reference every expanded operation exactly once")
    if source_indexes != sorted(source_indexes) or generated_indexes != sorted(generated_indexes):
        raise ValueError("transition action references must retain written order")
    first_generated = min(generated_indexes, default=len(patch.ops))
    if source_indexes and max(source_indexes) >= first_generated:
        raise ValueError("transition generated operations must follow all initiating operations")

    source_patches = [_source_patch_for_group(patch, group) for group in trace.initiating_groups]
    source_actions = [
        (source_patch, operation)
        for source_patch in source_patches
        for operation in source_patch.ops
    ]
    if not source_actions:
        raise ValueError("transition trace requires at least one initiating action")
    expected_id = _transition_id(
        trace.pre_head,
        source_patches,
        trace.initiating_groups,
        source_actions,
        ruleset_tag=trace.ruleset_tag,
        generated_actions=trace.generated_actions,
    )
    if expected_id != trace.transition_id:
        raise ValueError("transition id does not match initiating actions and provenance")

    known_events: set[str] = set()
    for item in trace.generated_actions:
        if item.rule_id != GUIDANCE_RULE_ID:
            raise ValueError(f"unsupported generated transition rule {item.rule_id!r}")
        if item.cause.kind != "action" or item.cause.action_index is None:
            raise ValueError("generated actions must cite an earlier action")
        if item.cause.action_index >= item.operation_index:
            raise ValueError("generated action cause must precede the generated action")
        _validate_guidance_operation(patch.ops[item.operation_index])

    try:
        initiating_timeline, replayed = GraphTransitionManager._apply_timeline(
            state,
            source_actions,
        )
        generated_timeline: list[tuple[int, GraphState, GraphState]] = []
        for item in trace.generated_actions:
            before = replayed
            replayed = apply_transition_generated_operation(
                replayed,
                patch,
                patch.ops[item.operation_index],
            )
            generated_timeline.append((item.operation_index, before, replayed))
    except (AttributeError, KeyError, TypeError, ValueError, TransitionConflict) as exc:
        raise ValueError(
            f"transition operations cannot reproduce their lifecycle trace: {exc}"
        ) from exc

    expected_events = GraphTransitionManager._lifecycle_events(
        trace.transition_id,
        initiating_timeline,
        generated_timeline,
        list(trace.generated_actions),
        list(patch.ops),
    )
    if trace.lifecycle_events != expected_events:
        raise ValueError("transition lifecycle events do not match expanded operation effects")

    for ordinal, event in enumerate(trace.lifecycle_events):
        if event.cause.kind == "action":
            if event.cause.action_index is None or event.cause.action_index >= len(patch.ops):
                raise ValueError("transition event cites an unknown action")
            cause_value: int | str = event.cause.action_index
        else:
            if event.cause.event_id not in known_events:
                raise ValueError("transition event cites an event that is not earlier")
            cause_value = event.cause.event_id or ""
        payload = {
            "ordinal": ordinal,
            "event_type": event.event_type,
            "cause": cause_value,
            "node_id": event.node_id,
            "field": event.field,
            "before": event.before,
            "after": event.after,
        }
        if event.event_id != _sha256({"transition_id": trace.transition_id, **payload}):
            raise ValueError("transition event id does not match its canonical payload")
        known_events.add(event.event_id)
    return source_patches


def accepted_transition_head_chain_failure(
    patches: Iterable[Patch],
    *,
    target: GraphTargetRef,
    initial_transition_id: str | None,
) -> ReplayFailure | None:
    """Return the first accepted transition that does not continue the exact head."""

    transition_id = initial_transition_id
    for patch in patches:
        if patch.admission != "accepted" or patch.transition is None:
            continue
        trace = patch.transition
        if trace.pre_head.target != target or trace.pre_head.transition_id != transition_id:
            return ReplayFailure(
                revision=patch.revision,
                created_at=patch.created_at,
                code="transition-head-mismatch",
                message=(
                    f"Transition at revision {patch.revision} does not continue the exact "
                    f"{target.key} graph head."
                ),
            )
        transition_id = trace.transition_id
    return None


def _source_patch_for_group(patch: Patch, group: TransitionInitiatingGroup) -> Patch:
    return patch.model_copy(
        update={
            "summary": group.summary,
            "change_summary": list(group.change_summary),
            "ops": [patch.ops[index] for index in group.operation_indexes],
            "human_action": group.human_action,
            "agent_action": group.agent_action,
            "transition": None,
        }
    )


def _validate_guidance_operation(operation: GraphOperation) -> None:
    if not isinstance(operation, UpdateNodesOperation) or not operation.nodes:
        raise ValueError("guidance-validity rule may generate only non-empty update_nodes")
    allowed = {"current_summary_stale", "next_action_stale"}
    for update in operation.nodes:
        if not update.changes or not set(update.changes) <= allowed:
            raise ValueError("guidance-validity action changes a non-system field")
        if not all(isinstance(value, bool) for value in update.changes.values()):
            raise ValueError("guidance-validity values must be booleans")


def _combined_patch(
    patches: list[Patch],
    operations: list[GraphOperation],
    trace: TransitionTrace,
) -> Patch:
    first = patches[0]
    cursors: dict[str, str] = {}
    for patch in patches:
        for key, value in patch.processed_cursors.items():
            if key in cursors and cursors[key] != value:
                raise ValueError(f"transition has conflicting processed cursor {key!r}")
            cursors[key] = value
    return first.model_copy(
        update={
            "summary": first.summary
            if len(patches) == 1
            else "Human Sync: " + "; ".join(patch.summary for patch in patches),
            "ops": operations,
            "run_truth_scope": _stable_union(patch.run_truth_scope for patch in patches),
            "repositories_read": _stable_union(patch.repositories_read for patch in patches),
            "processed_cursors": cursors,
            "change_summary": [item for patch in patches for item in patch.change_summary],
            "human_action": None,
            "agent_action": None,
            "admission": "accepted",
            "admission_messages": [],
            "transition": trace,
        }
    )


def _transition_id(
    pre_head: GraphHeadRef,
    patches: list[Patch],
    groups: list[TransitionInitiatingGroup],
    source_actions: list[tuple[Patch, GraphOperation]],
    *,
    ruleset_tag: str,
    generated_actions: Sequence[TransitionGeneratedAction] = (),
) -> str:
    envelopes = []
    for patch in patches:
        document = patch.model_dump(mode="json")
        # T1 added this optional canonical identity payload after transition ids
        # were already durable.  It is not valid on a transition-producing
        # patch, so its model default must not change historical hashes.
        if document.get("project_home_transfer") is None:
            document.pop("project_home_transfer", None)
        for field in (
            "revision",
            "created_at",
            "summary",
            "change_summary",
            "ops",
            "admission",
            "admission_messages",
            "human_action",
            "agent_action",
            "transition",
        ):
            document.pop(field, None)
        envelopes.append(document)
    return _sha256(
        {
            "pre_head": pre_head.model_dump(mode="json"),
            "ruleset_tag": ruleset_tag,
            "groups": [group.model_dump(mode="json") for group in groups],
            "producer_envelopes": envelopes,
            "operations": [
                operation.model_dump(mode="json", exclude_unset=True)
                for _patch, operation in source_actions
            ],
            "generated_actions": [action.model_dump(mode="json") for action in generated_actions],
        }
    )


def _operations_sha256(operations: Iterable[GraphOperation]) -> str:
    return _sha256(
        [operation.model_dump(mode="json", exclude_unset=True) for operation in operations]
    )


def _sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stable_union(values: Iterable[Iterable[str]]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for collection in values:
        for value in collection:
            if value not in seen:
                seen.add(value)
                result.append(value)
    return result


def _experiment_dependency_signature(state: GraphState, experiment_id: str) -> tuple[Any, ...]:
    node = state.nodes.get(experiment_id)
    if not isinstance(node, Experiment):
        return ()
    control_dependencies = experiment_control_dependencies(state, experiment_id).model_dump(
        mode="json"
    )
    tests_relations = {
        (edge.id, edge.target)
        for edge in state.edges.values()
        if edge.source == experiment_id and edge.relation == "tests"
    }
    hypothesis_ids = {target for _edge_id, target in tests_relations}
    assessments: list[tuple[Any, ...]] = []
    epistemic_relations = {
        "supports",
        "weakens",
        "refutes",
        "inconclusive",
        "contradicts",
    }
    for edge in state.edges.values():
        if edge.target not in hypothesis_ids or edge.relation not in epistemic_relations:
            continue
        source = state.nodes.get(edge.source)
        if not isinstance(source, Evidence):
            continue
        assessments.append(
            (
                edge.id,
                edge.source,
                edge.target,
                edge.relation,
                edge.assessment.model_dump(mode="json") if edge.assessment else None,
            )
        )
    return (
        control_dependencies,
        tuple(sorted(tests_relations)),
        tuple(sorted(assessments, key=lambda item: item[0])),
    )


def _guidance_field_validity(
    value: str | None,
    stale: bool,
    event_id: str | None,
) -> GuidanceFieldValidity:
    if not value:
        return GuidanceFieldValidity(status="empty")
    if stale:
        return GuidanceFieldValidity(status="stale", invalidated_by_event_id=event_id)
    return GuidanceFieldValidity(status="current")
