from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rcp.core.materialize import MaterializationResult, apply_valid_patch
from rcp.core.models import AuthorizedHuman, GraphState, Patch, Standing
from rcp.core.operations import (
    CreateAmbiguitiesOperation,
    CreateEdgesOperation,
    CreateNodesOperation,
    CreateProposalsOperation,
    GraphOperation,
    MergeNodesOperation,
    NewEdge,
    RemoveEdgesOperation,
    RemoveNodesOperation,
    ResolveAmbiguitiesOperation,
    ResolveProposalsOperation,
    SetCoverageOperation,
    SetOntologyOperation,
    SetProjectTruthScopeOperation,
    SetStandingOperation,
    SupersedeNodesOperation,
    UpdateNodesOperation,
    UpsertGlossaryOperation,
    WithdrawProposalsOperation,
)
from rcp.limits import REFRESH_DELTA_MAX_BYTES, REFRESH_DELTA_MAX_ENTRIES

_MAX_TITLE_CHARS = 240
_GRAPH_ID_RE = re.compile(
    r"(?<![a-z0-9_/-])[a-z][a-z0-9]*(?:_[a-z0-9]+)*/"
    r"[a-z0-9]+(?:-[a-z0-9]+)*(?![a-z0-9_/-])"
)
_OPERATION_LABELS = {
    "create_nodes": "added research concepts",
    "update_nodes": "updated research concepts",
    "create_edges": "connected research concepts",
    "remove_edges": "removed graph relationships",
    "remove_nodes": "removed research concepts",
    "supersede_nodes": "superseded research concepts",
    "merge_nodes": "merged research concepts",
    "create_ambiguities": "recorded open questions",
    "resolve_ambiguities": "resolved open questions",
    "create_proposals": "recorded proposals",
    "resolve_proposals": "resolved proposals",
    "withdraw_proposals": "withdrew proposals",
    "upsert_glossary": "updated the glossary",
    "set_coverage": "updated source coverage",
    "set_standing": "updated review standing",
    "set_project_truth_scope": "updated the project truth scope",
    "set_ontology": "updated the project ontology",
}
_EVIDENCE_HYPOTHESIS_RELATIONS = frozenset(
    {"supports", "weakens", "refutes", "inconclusive", "contradicts"}
)


class RevisionSummary(BaseModel):
    """Deterministic reader-facing prose for one accepted canonical patch."""

    model_config = ConfigDict(extra="forbid")

    from_revision: int = Field(ge=0)
    to_revision: int = Field(ge=1)
    kind: Literal["seed", "refresh", "chat", "work", "experiment_loop", "approval", "identity"]
    author: Literal["agent", "human"] | None
    producer: Literal["agent", "human", "system"]
    authorized_by: AuthorizedHuman | None = None
    profile: Literal["ordinary", "orchestrator"] | None = None
    task_id: str | None = None
    episode_id: str | None = None
    created_at: str
    sentences: list[str] = Field(min_length=1)


def build_revision_summaries(
    patches: Iterable[Patch],
    materialization: MaterializationResult,
    *,
    from_revision: int = 1,
    to_revision: int | None = None,
) -> list[RevisionSummary]:
    """Render accepted append-only history without exposing graph implementation labels."""

    end = to_revision if to_revision is not None else 10**12
    state = GraphState()
    summaries: list[RevisionSummary] = []

    for patch in sorted(patches, key=lambda item: item.revision):
        report = materialization.reports.get(patch.revision)
        if report is None or report.rejected:
            continue

        previous_state = state
        state = apply_valid_patch(state, patch)
        if not from_revision <= patch.revision <= end:
            continue
        summaries.append(render_revision_summary(previous_state, patch, state))
    return summaries


def render_revision_summary(
    previous_state: GraphState,
    patch: Patch,
    state: GraphState,
) -> RevisionSummary:
    """Render one successfully applied patch without changing either replay state."""

    if patch.kind == "identity" and patch.project_identity is not None:
        home_space_id = patch.project_identity.home_space_id
        if patch.project_identity.action == "created":
            sentences = [f"Project created in {home_space_id}."]
        else:
            sentences = [f"Project identity adopted in {home_space_id}."]
    elif patch.kind == "identity" and patch.project_home_transfer is not None:
        transfer = patch.project_home_transfer
        sentences = [
            f"Project moved from {transfer.previous_home_space_id} to {transfer.new_home_space_id}."
        ]
    else:
        labels = _state_labels(previous_state) | _state_labels(state)
        sentences = [
            _plain_history_text(item, labels) for item in patch.change_summary if item.strip()
        ]
        sentences = [item for item in sentences if item]
        if not sentences:
            sentences = _operation_fallbacks(patch, previous_state, state, labels)
        sentences.extend(_edge_assessment_sentences(patch, previous_state, state, labels))
        sentences.extend(_proposal_consequence_sentences(patch, state, labels, sentences))
        sentences = _unique_sentences(_plain_history_text(item, labels) for item in sentences)
        if not sentences:
            sentences = ["Recorded a research graph revision."]
    return RevisionSummary(
        from_revision=max(0, patch.revision - 1),
        to_revision=patch.revision,
        kind=patch.kind,
        author=patch.author,
        producer=patch.producer,
        authorized_by=patch.authorized_by,
        profile=patch.profile,
        task_id=patch.task_id,
        episode_id=patch.episode_id,
        created_at=patch.created_at.isoformat(),
        sentences=sentences,
    )


class RefreshDeltaEntry(BaseModel):
    """Routing metadata for one post-refresh graph or human-authority event."""

    model_config = ConfigDict(extra="forbid")

    category: Literal[
        "current_contested",
        "standing_transition",
        "human_prose_edit",
        "node_removal",
        "chat_graph_update",
        "proposal_decision",
        "ambiguity_decision",
    ]
    target_id: str
    target_type: str
    title: str = Field(default="", max_length=_MAX_TITLE_CHARS)
    revision: int = Field(ge=0)
    author: Literal["agent", "human"]
    field_names: list[str] = Field(default_factory=list)
    previous_standing: Standing | None = None
    current_standing: Standing | None = None
    decision: Literal["approved", "rejected", "withdrawn", "resolved", "dismissed"] | None = None


class RefreshDelta(BaseModel):
    """A deterministic, bounded index of changes since the last graph ingest."""

    model_config = ConfigDict(extra="forbid")

    after_revision: int = Field(ge=0)
    through_revision: int = Field(ge=0)
    entries: list[RefreshDeltaEntry] = Field(max_length=REFRESH_DELTA_MAX_ENTRIES)
    omitted_count: int = Field(ge=0)
    omitted_from_revision: int | None = Field(default=None, ge=0)
    omitted_through_revision: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def enforce_bounds(self) -> RefreshDelta:
        if self.through_revision < self.after_revision:
            raise ValueError("through_revision cannot precede after_revision")
        if _encoded_size(self) > REFRESH_DELTA_MAX_BYTES:
            raise ValueError(f"refresh_delta exceeds {REFRESH_DELTA_MAX_BYTES} bytes")
        return self


def build_refresh_delta(
    patches: Iterable[Patch],
    materialization: MaterializationResult,
) -> RefreshDelta:
    """Build refresh routing data from already-loaded canonical history.

    Rejected patches never enter the delta. Current contested nodes are
    deliberately included even when their transition predates the most recent
    successful seed/refresh, then newer eligible events fill the remaining
    bounded space.
    """

    ordered = sorted(patches, key=lambda item: item.revision)
    accepted = [
        patch
        for patch in ordered
        if patch.revision in materialization.reports
        and not materialization.reports[patch.revision].rejected
    ]
    baseline = max(
        (patch.revision for patch in accepted if patch.kind in {"seed", "refresh"}),
        default=0,
    )
    standing_transitions = _standing_transition_entries(
        accepted,
        materialization.state,
    )
    mandatory = _current_contested_entries(
        materialization.state,
        standing_transitions,
    )
    recent = [
        *(entry for entry in standing_transitions if entry.revision > baseline),
        *_recent_entries(
            (patch for patch in accepted if patch.revision > baseline),
            materialization.state,
        ),
    ]
    recent.sort(
        key=lambda item: (
            -item.revision,
            item.category,
            item.target_type,
            item.target_id,
            tuple(item.field_names),
        )
    )
    mandatory_keys = {_entry_identity(entry) for entry in mandatory}
    candidates = [
        *mandatory,
        *(entry for entry in recent if _entry_identity(entry) not in mandatory_keys),
    ]

    selected: list[RefreshDeltaEntry] = []
    for entry in candidates:
        if len(selected) >= REFRESH_DELTA_MAX_ENTRIES:
            break
        candidate = [*selected, entry]
        omitted = len(candidates) - len(candidate)
        omitted_range = _omitted_revision_range(candidates[len(candidate) :])
        if (
            _candidate_encoded_size(
                after_revision=baseline,
                through_revision=materialization.state.revision,
                entries=candidate,
                omitted_count=omitted,
                omitted_from_revision=omitted_range[0] if omitted_range else None,
                omitted_through_revision=omitted_range[1] if omitted_range else None,
            )
            > REFRESH_DELTA_MAX_BYTES
        ):
            break
        selected.append(entry)

    omitted_entries = candidates[len(selected) :]
    omitted_range = _omitted_revision_range(omitted_entries)
    return RefreshDelta(
        after_revision=baseline,
        through_revision=materialization.state.revision,
        entries=selected,
        omitted_count=len(omitted_entries),
        omitted_from_revision=omitted_range[0] if omitted_range else None,
        omitted_through_revision=omitted_range[1] if omitted_range else None,
    )


def _current_contested_entries(
    state: GraphState,
    transitions: list[RefreshDeltaEntry],
) -> list[RefreshDeltaEntry]:
    entries = []
    for node in sorted(state.nodes.values(), key=lambda item: item.id):
        if node.standing != Standing.CONTESTED:
            continue
        transition = next(
            (
                item
                for item in reversed(transitions)
                if item.target_id == node.id and item.current_standing == Standing.CONTESTED
            ),
            None,
        )
        entries.append(
            RefreshDeltaEntry(
                category="current_contested",
                target_id=node.id,
                target_type=node.type,
                title=_bounded_title(node.title),
                revision=transition.revision if transition is not None else node.updated_rev,
                author=transition.author if transition is not None else "human",
                field_names=["standing"],
                previous_standing=(
                    transition.previous_standing if transition is not None else None
                ),
                current_standing=node.standing,
            )
        )
    return entries


def _standing_transition_entries(
    patches: list[Patch],
    state: GraphState,
) -> list[RefreshDeltaEntry]:
    standings: dict[str, Standing] = {}
    entries: list[RefreshDeltaEntry] = []
    for patch in patches:
        for operation in patch.ops:
            if isinstance(operation, CreateNodesOperation):
                for node in operation.nodes:
                    standings[node.id] = node.standing
                continue
            if isinstance(operation, SetStandingOperation):
                node_id = operation.node_id
                before = standings.get(node_id, Standing.ASSERTED)
                after = Standing(operation.standing)
                if before != after:
                    entries.append(_standing_entry(state, patch, node_id, before, after))
                standings[node_id] = after
                continue
            if patch.kind == "approval":
                continue
            for node_id in _nodes_reset_by_operation(operation):
                before = standings.get(node_id, Standing.ASSERTED)
                after = Standing.ASSERTED
                if before != after:
                    entries.append(_standing_entry(state, patch, node_id, before, after))
                standings[node_id] = after
    return entries


def _nodes_reset_by_operation(operation: GraphOperation) -> list[str]:
    if isinstance(operation, (UpdateNodesOperation, SupersedeNodesOperation)):
        return [item.id for item in operation.nodes]
    if isinstance(operation, MergeNodesOperation):
        return [item.duplicate for item in operation.merges]
    return []


def _standing_entry(
    state: GraphState,
    patch: Patch,
    node_id: str,
    before: Standing,
    after: Standing,
) -> RefreshDeltaEntry:
    node = state.nodes.get(node_id)
    return RefreshDeltaEntry(
        category="standing_transition",
        target_id=node_id,
        target_type=node.type if node else "node",
        title=_bounded_title(node.title if node else ""),
        revision=patch.revision,
        author=patch.author,
        field_names=["standing"],
        previous_standing=before,
        current_standing=after,
    )


def _recent_entries(
    patches: Iterable[Patch],
    state: GraphState,
) -> list[RefreshDeltaEntry]:
    entries: list[RefreshDeltaEntry] = []
    for patch in patches:
        for operation in patch.ops:
            if isinstance(operation, UpdateNodesOperation) and patch.kind == "approval":
                for update in operation.nodes:
                    # Direct literal prose edits carry the optimistic concurrency
                    # guard. Proposal replay operations do not and are routed by
                    # their proposal-decision entry instead.
                    if "base_updated_rev" not in update.model_fields_set:
                        continue
                    entries.append(
                        _node_entry(
                            state,
                            patch,
                            "human_prose_edit",
                            update.id,
                            _field_names(update.changes),
                        )
                    )
            elif isinstance(operation, ResolveProposalsOperation):
                for resolution in operation.resolutions:
                    decision = resolution.status
                    proposal_id = resolution.id
                    proposal = state.proposals.get(proposal_id)
                    entries.append(
                        RefreshDeltaEntry(
                            category="proposal_decision",
                            target_id=proposal_id,
                            target_type="proposal",
                            title=_bounded_title(proposal.title if proposal else ""),
                            revision=patch.revision,
                            author=patch.author,
                            field_names=sorted(
                                {"status"}
                                | (
                                    {"rejection_reason"}
                                    if "reason" in resolution.model_fields_set
                                    else set()
                                )
                            ),
                            decision=decision,
                        )
                    )
            elif isinstance(operation, WithdrawProposalsOperation):
                for withdrawal in operation.proposals:
                    proposal_id = withdrawal.id
                    proposal = state.proposals.get(proposal_id)
                    entries.append(
                        RefreshDeltaEntry(
                            category="proposal_decision",
                            target_id=proposal_id,
                            target_type="proposal",
                            title=_bounded_title(proposal.title if proposal else ""),
                            revision=patch.revision,
                            author=patch.author,
                            field_names=sorted(
                                {"status"}
                                | (
                                    {"resolution_reason"}
                                    if "reason" in withdrawal.model_fields_set
                                    else set()
                                )
                            ),
                            decision="withdrawn",
                        )
                    )
            elif isinstance(operation, ResolveAmbiguitiesOperation) and patch.author == "human":
                for resolution in operation.resolutions:
                    decision = resolution.status
                    entries.append(
                        RefreshDeltaEntry(
                            category="ambiguity_decision",
                            target_id=resolution.id,
                            target_type="ambiguity",
                            revision=patch.revision,
                            author=patch.author,
                            field_names=["status"],
                            decision=decision,
                        )
                    )
            elif isinstance(operation, RemoveNodesOperation):
                entries.extend(
                    RefreshDeltaEntry(
                        category="node_removal",
                        target_id=node_id,
                        target_type="node",
                        revision=patch.revision,
                        author=patch.author,
                        field_names=["removed"],
                    )
                    for node_id in operation.node_ids
                )
            if patch.kind in {"chat", "work"} and not isinstance(operation, RemoveNodesOperation):
                entries.extend(_chat_entries(patch, operation, state))
    return sorted(
        entries,
        key=lambda item: (
            -item.revision,
            item.category,
            item.target_type,
            item.target_id,
            tuple(item.field_names),
        ),
    )


def _chat_entries(
    patch: Patch,
    operation: GraphOperation,
    state: GraphState,
) -> list[RefreshDeltaEntry]:
    targets: list[tuple[str, str, str, list[str]]] = []
    if isinstance(operation, CreateNodesOperation):
        for node in operation.nodes:
            targets.append(
                (
                    node.id,
                    node.type,
                    node.title,
                    _field_names(node),
                )
            )
    elif isinstance(operation, UpdateNodesOperation):
        for update in operation.nodes:
            node_id = update.id
            node = state.nodes.get(node_id)
            targets.append(
                (
                    node_id,
                    node.type if node else "node",
                    node.title if node else "",
                    _field_names(update.changes),
                )
            )
    elif isinstance(operation, CreateEdgesOperation):
        for edge in operation.edges:
            edge_id = edge.id or f"{edge.source}::{edge.relation}::{edge.target}"
            targets.append((edge_id, "edge", edge.relation, _field_names(edge)))
    elif isinstance(operation, RemoveEdgesOperation):
        targets.extend((edge_id, "edge", "", ["removed"]) for edge_id in operation.edge_ids)
    elif isinstance(operation, SupersedeNodesOperation):
        for item in operation.nodes:
            node_id = item.id
            node = state.nodes.get(node_id)
            targets.append(
                (
                    node_id,
                    node.type if node else "node",
                    node.title if node else "",
                    ["status"],
                )
            )
    elif isinstance(operation, MergeNodesOperation):
        for item in operation.merges:
            node_id = item.duplicate
            node = state.nodes.get(node_id)
            targets.append(
                (
                    node_id,
                    node.type if node else "node",
                    node.title if node else "",
                    ["status"],
                )
            )
    elif isinstance(operation, CreateAmbiguitiesOperation):
        targets.extend(
            (ambiguity.id, "ambiguity", "", _field_names(ambiguity))
            for ambiguity in operation.ambiguities
        )
    elif isinstance(operation, ResolveAmbiguitiesOperation):
        targets.extend(
            (resolution.id, "ambiguity", "", ["status"]) for resolution in operation.resolutions
        )
    elif isinstance(operation, CreateProposalsOperation):
        targets.extend(
            (
                proposal.id,
                "proposal",
                proposal.title,
                _field_names(proposal),
            )
            for proposal in operation.proposals
        )
    elif isinstance(operation, UpsertGlossaryOperation):
        targets.extend(
            (
                term.term,
                "glossary_term",
                term.term,
                _field_names(term),
            )
            for term in operation.terms
        )
    elif isinstance(operation, SetProjectTruthScopeOperation):
        targets.append(("project_truth_scope", "project", "", ["truth_scope"]))

    return [
        RefreshDeltaEntry(
            category="chat_graph_update",
            target_id=target_id,
            target_type=target_type,
            title=_bounded_title(title),
            revision=patch.revision,
            author=patch.author,
            field_names=field_names,
            current_standing=(
                state.nodes[target_id].standing if target_id in state.nodes else None
            ),
        )
        for target_id, target_type, title, field_names in targets
        if target_id
    ]


def _node_entry(
    state: GraphState,
    patch: Patch,
    category: Literal["human_prose_edit"],
    node_id: str,
    field_names: list[str],
) -> RefreshDeltaEntry:
    node = state.nodes.get(node_id)
    return RefreshDeltaEntry(
        category=category,
        target_id=node_id,
        target_type=node.type if node else "node",
        title=_bounded_title(node.title if node else ""),
        revision=patch.revision,
        author=patch.author,
        field_names=field_names,
        current_standing=node.standing if node else None,
    )


def _field_names(value: BaseModel | dict[str, object]) -> list[str]:
    if isinstance(value, BaseModel):
        return sorted(value.model_fields_set)
    return sorted(value)


def _entry_identity(entry: RefreshDeltaEntry) -> tuple[str, int, str]:
    return entry.target_id, entry.revision, ",".join(entry.field_names)


def _bounded_title(value: str) -> str:
    return value[:_MAX_TITLE_CHARS]


def _encoded_size(value: RefreshDelta) -> int:
    return len(
        json.dumps(
            value.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _candidate_encoded_size(
    *,
    after_revision: int,
    through_revision: int,
    entries: list[RefreshDeltaEntry],
    omitted_count: int,
    omitted_from_revision: int | None,
    omitted_through_revision: int | None,
) -> int:
    return len(
        json.dumps(
            {
                "after_revision": after_revision,
                "through_revision": through_revision,
                "entries": [entry.model_dump(mode="json") for entry in entries],
                "omitted_count": omitted_count,
                "omitted_from_revision": omitted_from_revision,
                "omitted_through_revision": omitted_through_revision,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _omitted_revision_range(
    entries: list[RefreshDeltaEntry],
) -> tuple[int, int] | None:
    if not entries:
        return None
    revisions = [entry.revision for entry in entries]
    return min(revisions), max(revisions)


def _state_labels(state: GraphState) -> dict[str, str]:
    return {
        **{node.id: node.title for node in state.nodes.values()},
        **{proposal.id: proposal.title for proposal in state.proposals.values()},
        **{ambiguity.id: ambiguity.question for ambiguity in state.ambiguities.values()},
    }


def _operation_fallbacks(
    patch: Patch,
    previous_state: GraphState,
    state: GraphState,
    labels: dict[str, str],
) -> list[str]:
    sentences: list[str] = []
    for operation in patch.ops:
        if isinstance(operation, CreateNodesOperation):
            for node in operation.nodes:
                title = _object_label(node.id, labels, node.title)
                noun = node.extension_type or node.type
                sentences.append(f"Recorded a {noun.replace('_', ' ')}: {_quoted(title)}.")
        elif isinstance(operation, UpdateNodesOperation):
            sentences.extend(
                f"Updated {_quoted(_object_label(item.id, labels))}." for item in operation.nodes
            )
        elif isinstance(operation, CreateEdgesOperation):
            for edge in operation.edges:
                source = _object_label(edge.source, labels)
                target = _object_label(edge.target, labels)
                sentences.append(f"Connected {_quoted(source)} with {_quoted(target)}.")
        elif isinstance(operation, RemoveEdgesOperation):
            for edge_id in operation.edge_ids:
                edge = previous_state.edges.get(edge_id) or state.edges.get(edge_id)
                if edge is None:
                    sentences.append("Removed a graph relationship.")
                    continue
                sentences.append(
                    f"Removed the relationship between "
                    f"{_quoted(_object_label(edge.source, labels))} and "
                    f"{_quoted(_object_label(edge.target, labels))}."
                )
        elif isinstance(operation, RemoveNodesOperation):
            sentences.extend(
                f"Removed {_quoted(_object_label(node_id, labels))}."
                for node_id in operation.node_ids
            )
        elif isinstance(operation, SupersedeNodesOperation):
            for item in operation.nodes:
                current = _quoted(_object_label(item.id, labels))
                replacement_id = item.superseded_by
                if replacement_id:
                    replacement = _quoted(_object_label(replacement_id, labels))
                    sentences.append(f"Superseded {current} with {replacement}.")
                else:
                    sentences.append(f"Superseded {current}.")
        elif isinstance(operation, MergeNodesOperation):
            for item in operation.merges:
                duplicate = _quoted(_object_label(item.duplicate, labels))
                canonical = _quoted(_object_label(item.canonical, labels))
                sentences.append(f"Merged {duplicate} into {canonical}.")
        elif isinstance(operation, CreateAmbiguitiesOperation):
            for item in operation.ambiguities:
                label = _object_label(item.id, labels, item.question)
                sentences.append(f"Recorded an open question: {_quoted(label)}.")
        elif isinstance(operation, ResolveAmbiguitiesOperation):
            for item in operation.resolutions:
                label = _quoted(_object_label(item.id, labels))
                verb = "Resolved" if item.status == "resolved" else "Dismissed"
                sentences.append(f"{verb} the open question {label}.")
        elif isinstance(operation, CreateProposalsOperation):
            for item in operation.proposals:
                label = _object_label(item.id, labels, item.title)
                sentences.append(f"Recorded a proposal: {_quoted(label)}.")
        elif isinstance(operation, ResolveProposalsOperation):
            for item in operation.resolutions:
                label = _quoted(_object_label(item.id, labels))
                status = item.status.replace("_", " ").title()
                sentences.append(f"{status} proposal {label}.")
        elif isinstance(operation, WithdrawProposalsOperation):
            for item in operation.proposals:
                label = _quoted(_object_label(item.id, labels))
                sentences.append(f"Withdrew proposal {label}.")
        elif isinstance(operation, UpsertGlossaryOperation):
            sentences.extend(
                f"Updated the glossary entry {_quoted(item.term.strip())}."
                for item in operation.terms
                if item.term.strip()
            )
        elif isinstance(operation, SetCoverageOperation):
            sentences.append("Updated source coverage.")
        elif isinstance(operation, SetStandingOperation):
            label = _quoted(_object_label(operation.node_id, labels))
            sentences.append(f"Marked {label} {operation.standing}.")
        elif isinstance(operation, SetProjectTruthScopeOperation):
            sentences.append("Updated the project truth scope.")
        elif isinstance(operation, SetOntologyOperation):
            sentences.append("Updated the project ontology.")
        else:
            sentences.append("Updated the research graph.")
    return sentences


def _edge_assessment_sentences(
    patch: Patch,
    previous_state: GraphState,
    state: GraphState,
    labels: dict[str, str],
) -> list[str]:
    rendered: list[str] = []
    for operation in patch.ops:
        if not isinstance(operation, CreateEdgesOperation):
            continue
        for edge in operation.edges:
            sentence = _edge_assessment_sentence(
                edge,
                previous_state,
                state,
                source=_object_label(edge.source, labels),
                target=_object_label(edge.target, labels),
            )
            if sentence is not None:
                rendered.append(sentence)
    return rendered


def _edge_assessment_sentence(
    edge: NewEdge,
    previous_state: GraphState,
    state: GraphState,
    *,
    source: str,
    target: str,
) -> str | None:
    source_node = state.nodes.get(edge.source) or previous_state.nodes.get(edge.source)
    target_node = state.nodes.get(edge.target) or previous_state.nodes.get(edge.target)
    is_evidence_hypothesis = (
        edge.relation in _EVIDENCE_HYPOTHESIS_RELATIONS
        and source_node is not None
        and source_node.type == "evidence"
        and target_node is not None
        and target_node.type == "hypothesis"
    )
    assessment = edge.assessment
    if assessment is None:
        if not is_evidence_hypothesis:
            return None
        return (
            f"The {edge.relation} relation from {_quoted(source)} to {_quoted(target)} was "
            "recorded as an unassessed legacy relation."
        )

    details = [
        f"{assessment.relevance} relevance",
        f"{assessment.weight} weight",
    ]
    if assessment.scope is not None:
        details.append(f"scope {_quoted(assessment.scope)}")
    if assessment.qualifications:
        qualifications = "; ".join(_quoted(item) for item in assessment.qualifications)
        details.append(f"qualifications {qualifications}")
    return (
        f"The {edge.relation} relation from {_quoted(source)} to {_quoted(target)} was assessed "
        f"with {_reader_list(details)}."
    )


def _reader_list(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _proposal_consequence_sentences(
    patch: Patch,
    state: GraphState,
    labels: dict[str, str],
    existing: list[str],
) -> list[str]:
    proposal_ids: list[str] = []
    for operation in patch.ops:
        if isinstance(operation, CreateProposalsOperation):
            proposal_ids.extend(item.id for item in operation.proposals)
        elif isinstance(operation, ResolveProposalsOperation):
            proposal_ids.extend(
                item.id for item in operation.resolutions if item.status == "approved"
            )

    rendered: list[str] = []
    for proposal_id in dict.fromkeys(proposal_ids):
        proposal = state.proposals.get(proposal_id)
        if proposal is None:
            continue
        title = proposal.title
        consequence = proposal.card.consequences
        plain_consequence = _plain_history_text(consequence, labels)
        if not plain_consequence or any(plain_consequence in item for item in existing):
            continue
        label = _object_label(proposal_id, labels, title)
        rendered.append(
            f"The proposal {_quoted(label)} records this consequence: {_quoted(plain_consequence)}"
        )
    return rendered


def _object_label(identifier: str, labels: dict[str, str], fallback: str = "") -> str:
    return labels.get(identifier) or fallback.strip() or identifier


def _plain_history_text(value: str, labels: dict[str, str]) -> str:
    def replace_identifier(match: re.Match[str]) -> str:
        identifier = match.group(0)
        label = labels.get(identifier)
        return _strip_internal_tokens(label) if label is not None else identifier

    rendered = _GRAPH_ID_RE.sub(replace_identifier, value.strip())
    operation_names = "|".join(re.escape(name) for name in _OPERATION_LABELS)
    rendered = re.sub(
        rf"\s+(?:through|via|using)\s+(?:{operation_names})\b",
        "",
        rendered,
        flags=re.IGNORECASE,
    )
    for operation, label in _OPERATION_LABELS.items():
        rendered = re.sub(rf"\b{re.escape(operation)}\b", label, rendered)
    return " ".join(rendered.split())


def _strip_internal_tokens(value: str) -> str:
    rendered = value
    for operation, label in _OPERATION_LABELS.items():
        rendered = re.sub(rf"\b{re.escape(operation)}\b", label, rendered)
    return " ".join(rendered.split())


def _quoted(value: str) -> str:
    return f"“{_strip_internal_tokens(value)}”"


def _unique_sentences(sentences: Iterable[str]) -> list[str]:
    unique: list[str] = []
    for sentence in sentences:
        if sentence and sentence not in unique:
            unique.append(sentence)
    return unique
