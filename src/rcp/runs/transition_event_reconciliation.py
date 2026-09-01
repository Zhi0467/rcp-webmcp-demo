"""Reconcile accepted transition events into one-shot graph watchers.

Canonical graph history and local SQLite cannot share an ACID transaction.  This
module keeps the boundary explicit: callers construct boundaries only from an
accepted replay, then apply them in canonical order.  Existing watcher terminal
state is the durable idempotency receipt; no event is delivered from a preview.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from rcp.core.models import GraphState, Patch
from rcp.core.transition_models import GraphHeadRef, GraphTargetRef, TransitionEvent
from rcp.storage import AppStore, StoredWatcherRecord
from rcp.watchers import (
    graph_watcher_boundary_result,
    ready_graph_watcher_groups,
)


@dataclass(frozen=True)
class AcceptedGraphBoundary:
    """One proven accepted post-state and its stable transition events."""

    target: GraphTargetRef
    revision: int
    transition_id: str | None
    state: GraphState
    lifecycle_events: tuple[TransitionEvent, ...]

    def __post_init__(self) -> None:
        if self.state.replay_status != "complete":
            raise ValueError("an accepted graph boundary requires a complete replay state")
        if self.state.revision != self.revision:
            raise ValueError("accepted graph boundary state does not match its revision")
        if self.lifecycle_events and self.transition_id is None:
            raise ValueError("accepted lifecycle events require a transition id")
        event_ids = [event.event_id for event in self.lifecycle_events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("an accepted transition cannot repeat a lifecycle event id")

    @classmethod
    def from_replay(
        cls,
        pre_state: GraphState,
        patch: Patch,
        post_state: GraphState,
    ) -> AcceptedGraphBoundary:
        """Build a boundary observed while replaying one accepted canonical Patch."""

        if patch.admission != "accepted":
            raise ValueError("a rejected Patch cannot become an accepted graph boundary")
        if pre_state.replay_status != "complete" or post_state.replay_status != "complete":
            raise ValueError("an accepted graph boundary requires complete replay states")
        if patch.revision != pre_state.revision + 1:
            raise ValueError("an accepted graph boundary must advance exactly one revision")
        if post_state.revision != patch.revision:
            raise ValueError("accepted graph boundary state does not match the Patch revision")

        trace = patch.transition
        if trace is None:
            target = GraphTargetRef()
            transition_id = None
            events: tuple[TransitionEvent, ...] = ()
        else:
            if trace.pre_head.revision != pre_state.revision:
                raise ValueError("transition pre-head does not match the replay pre-state")
            target = trace.pre_head.target
            transition_id = trace.transition_id
            events = tuple(trace.lifecycle_events)

        return cls(
            target=target,
            revision=post_state.revision,
            transition_id=transition_id,
            state=post_state,
            lifecycle_events=events,
        )


def reconcile_accepted_graph_boundaries(
    store: AppStore,
    project_id: str,
    boundaries: Iterable[AcceptedGraphBoundary],
    *,
    current_head: GraphHeadRef | None = None,
) -> list[list[StoredWatcherRecord]]:
    """Apply accepted boundaries in target-local order and return ready wakes.

    Each boundary's watcher updates and durable target-local head advance share
    one SQLite transaction. Repeating a reconciliation pass is therefore safe,
    including after a process crash. A watcher armed at the boundary revision
    cannot consume that boundary's events.
    """

    accepted_boundaries = list(boundaries)
    last_revision_by_target: dict[str, int] = {}
    event_ids: set[str] = set()
    for boundary in accepted_boundaries:
        target_key = boundary.target.key
        previous_revision = last_revision_by_target.get(target_key)
        if previous_revision is not None and boundary.revision <= previous_revision:
            raise ValueError(
                f"accepted graph boundaries for {target_key} are not in canonical order"
            )
        last_revision_by_target[target_key] = boundary.revision

        for event in boundary.lifecycle_events:
            if event.event_id in event_ids:
                raise ValueError("a lifecycle event id was reused across accepted boundaries")
            event_ids.add(event.event_id)

    if current_head is not None:
        boundary_target_keys = {boundary.target.key for boundary in accepted_boundaries}
        if boundary_target_keys and boundary_target_keys != {current_head.target.key}:
            raise ValueError("the current graph head does not match the accepted boundaries")
        if accepted_boundaries and accepted_boundaries[-1].revision > current_head.revision:
            raise ValueError("an accepted graph boundary is ahead of the current graph head")
        consumed_head = store.graph_watcher_reconciliation_head(
            project_id,
            current_head.target,
        )
        if consumed_head is not None and consumed_head.revision > current_head.revision:
            raise RuntimeError("the graph is behind its consumed watcher reconciliation head")
        store.initialize_graph_watcher_target_baselines(
            project_id,
            current_head.target,
            armed_revision=current_head.revision,
        )

    for boundary in accepted_boundaries:
        store.consume_graph_watcher_boundary(
            project_id,
            GraphHeadRef(
                target=boundary.target,
                revision=boundary.revision,
                transition_id=boundary.transition_id,
            ),
            evaluate=lambda record, boundary=boundary: graph_watcher_boundary_result(
                record,
                boundary.state,
                boundary.lifecycle_events,
            ),
        )

    return ready_graph_watcher_groups(store, project_id)
