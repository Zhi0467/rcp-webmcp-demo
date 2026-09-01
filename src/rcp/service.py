from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from rcp.agents import (
    AgentLauncher,
    AgentPatch,
    ChatContext,
    ContextAssembler,
    PromptFactory,
    ProviderReadiness,
    RunContext,
    parse_agent_patch_json,
)
from rcp.agents.write_scope import RegisteredRepositoryRoot, registered_repository_roots
from rcp.attachments import ChatAttachmentDescriptor
from rcp.config import (
    AgentExecutionProfile,
    AgentSurface,
    AgentSurfaceConfig,
    MachineConfig,
    Manifest,
)
from rcp.control import derive_experiment_control_state
from rcp.core.attention import project_graph_attention
from rcp.core.authority import (
    AgentDispatchAuthority,
    AgentDispatchScope,
)
from rcp.core.materialize import apply_valid_patch
from rcp.core.models import (
    ACTIVE_EXPERIMENT_ATTEMPT_STATUSES,
    HUMAN_EDITABLE_NODE_FIELDS,
    AuthorizedHuman,
    Decision,
    ExperimentDecisionPin,
    GraphState,
    OntologyState,
    Patch,
    ProjectNode,
    Proposal,
    Standing,
)
from rcp.core.operations import (
    ProposalContentChangeOperation,
    ProposalMergeOperation,
    ProposalProtectedRelationOperation,
    ProposalRemovalOperation,
    ProposalStatusChangeOperation,
    ProposalSupersedeOperation,
    UpdateNodesOperation,
)
from rcp.core.transition_models import GraphTargetRef
from rcp.core.transitions import (
    CommittedTransition,
    PreparedTransition,
    project_transition_projection,
)
from rcp.core.validation.proposals import (
    normalized_decision_proposal_ops,
    proposal_is_stale,
    proposal_updates_node,
)
from rcp.history import HistoryManager
from rcp.limits import (
    BACKUP_INVENTORY_MAX_ENTRIES,
    CHAT_PAGE_DEFAULT_LIMIT,
    CHAT_PAGE_MAX_LIMIT,
    CHAT_PREVIEW_MAX_CHARS,
    CHAT_TITLE_MAX_CHARS,
)
from rcp.paper import PaperService, PaperSnapshot
from rcp.provider_skills import ProviderSkillInventoryManager
from rcp.providers import (
    PROVIDER_IDS,
    ProviderId,
    ProviderSkillReference,
    configured_runtime,
)
from rcp.runs.auto_research import AutoResearchRunRequest
from rcp.skill_registry import (
    SkillDefaults,
    SkillReference,
    SkillSelection,
    official_registry,
)
from rcp.sources import (
    AppChatOrigin,
    ConversationIndex,
    ConversationIndexer,
    ImportedProviderSourceInventory,
    ImportedProviderSourceStore,
    preflight_provider_roots,
    project_cache_roots,
)
from rcp.transport import repository_access as build_repository_access

_SETTINGS_SURFACES: tuple[AgentExecutionProfile, ...] = (
    "seed",
    "refresh",
    "node_chat",
    "project_chat",
    "paper_coach",
    "orchestrator",
)


class _ProjectSnapshotDraft:
    """Opaque graph-only project snapshot data for the display boundary."""

    __slots__ = ("_payload",)

    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __getitem__(self, key: str) -> object:
        return self._payload[key]

    def __setitem__(self, key: str, value: object) -> None:
        self._payload[key] = value

    def _as_dict(self) -> dict[str, object]:
        return dict(self._payload)


ConversationMode = Literal["discuss", "work"]
TaskTrigger = Literal["human", "orchestrator", "experiment_run", "watcher"]
GraphPatchKind = Literal["work", "experiment_loop"]


def _is_decision_choice(changes: dict[str, Any]) -> bool:
    return "selected_option" in changes or changes.get("status") == "decided"


def _proposal_applies_decision_choice(state: GraphState, proposal: Proposal) -> bool:
    for operation in normalized_decision_proposal_ops(state, proposal):
        if not isinstance(operation, UpdateNodesOperation):
            continue
        for update in operation.nodes:
            if isinstance(state.nodes.get(update.id), Decision) and _is_decision_choice(
                update.changes
            ):
                return True
    return False


def _proposal_approval_standing_targets(proposal: Proposal) -> list[str]:
    """Return belief nodes whose standing is accepted with this Proposal.

    ``related_node_ids`` is a dependency set, not a review-target set. New
    protected-change Proposals therefore use their declared intent to name only
    the belief whose content or lifecycle changed. Historical Proposals did not
    carry an intent and retain their existing related-node behavior.
    """

    if len(proposal.ops) != 1:
        return list(proposal.related_node_ids)
    operation = proposal.ops[0]
    if operation.intent.startswith("legacy_"):
        return list(proposal.related_node_ids)
    if isinstance(operation, (ProposalRemovalOperation, ProposalProtectedRelationOperation)):
        return []
    if isinstance(operation, (ProposalContentChangeOperation, ProposalStatusChangeOperation)):
        return [operation.nodes[0].id]
    if isinstance(operation, ProposalSupersedeOperation):
        return [operation.nodes[0].id]
    if isinstance(operation, ProposalMergeOperation):
        return [operation.merges[0].duplicate]
    # Central compatibility decoding assigns synthetic legacy intents to old
    # scope/ontology proposals. They retain their historical dependency-based
    # standing behavior rather than being reinterpreted as a current intent.
    return list(proposal.related_node_ids)


def _proposal_judgment_patch(
    state: GraphState,
    proposal: Proposal,
    *,
    decision: Literal["approved", "rejected"],
    reason: str | None,
    stale_reason: str | None = None,
) -> Patch:
    if proposal_is_stale(state, proposal):
        withdrawal_reason = (
            stale_reason
            or f"The proposal “{proposal.title}” was stale and was withdrawn without applying "
            "changes."
        )
        return Patch(
            kind="approval",
            author="human",
            summary=f"Withdrew stale proposal “{proposal.title}”.",
            ops=[
                {
                    "op": "resolve_proposals",
                    "resolutions": [
                        {
                            "id": proposal.id,
                            "status": "withdrawn",
                            "reason": withdrawal_reason,
                        }
                    ],
                }
            ],
            change_summary=[withdrawal_reason],
        )

    semantic_ops = (
        normalized_decision_proposal_ops(state, proposal) if decision == "approved" else []
    )
    standing_ops = [
        {"op": "set_standing", "node_id": node_id, "standing": "accepted"}
        for node_id in (
            _proposal_approval_standing_targets(proposal) if decision == "approved" else []
        )
    ]
    return Patch(
        kind="approval",
        author="human",
        summary=f"{decision.title()} proposal “{proposal.title}”.",
        ops=[
            *semantic_ops,
            {
                "op": "resolve_proposals",
                "resolutions": [
                    {
                        "id": proposal.id,
                        "status": decision,
                        "reason": reason,
                    }
                ],
            },
            *standing_ops,
        ],
        change_summary=[f"The proposal “{proposal.title}” was {decision}."],
        human_action=(
            "decision_choice"
            if decision == "approved" and _proposal_applies_decision_choice(state, proposal)
            else None
        ),
    )


def _stage_sync_patch(state: GraphState, patch: Patch) -> GraphState:
    """Advance the in-memory Sync draft exactly as append_batch will."""

    return apply_valid_patch(
        state,
        patch.model_copy(update={"revision": state.revision + 1}),
    )


class GraphUpdateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["none", "applied", "rejected"]
    applied_revision: int | None = Field(default=None, ge=0)
    change_summary: list[str] = Field(default_factory=list)
    proposal_ids: list[str] = Field(default_factory=list)
    validation_messages: list[str] = Field(default_factory=list)
    correction_rounds: int = Field(default=0, ge=0)
    repairable: bool = False


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_id: str
    operation_id: str | None = None
    role: Literal["user", "assistant"]
    text: str
    timestamp: str
    native_session_id: str | None = None
    provider: str | None = None
    model: str | None = None
    reasoning: str | None = None
    execution_machine: str | None = None
    applied_revision: int | None = Field(default=None, ge=0)
    mode: ConversationMode | None = None
    graph_update: GraphUpdateResult | None = None
    trigger: TaskTrigger = "human"
    attachments: list[ChatAttachmentDescriptor] = Field(default_factory=list)


class ChatSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chat_id: str
    kind: Literal["node_chat", "project_chat"]
    node_id: str | None
    title: str
    updated_at: str
    message_count: int = Field(ge=1)
    last_message_preview: str


class ChatSummaryPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ChatSummary]
    total: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=CHAT_PAGE_MAX_LIMIT)


class ChatTranscript(ChatSummary):
    messages: list[ChatMessage] = Field(min_length=1)


@dataclass(frozen=True)
class _ChatSummaryCacheEntry:
    fingerprint: tuple[int, int, int, int, int]
    summary: ChatSummary | None


class _StoredChatRecord(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    session_id: str = Field(alias="sessionId")
    native_session_id: str | None = Field(default=None, alias="nativeSessionId")
    node_id: str | None = Field(default=None, alias="nodeId")
    chat_scope: Literal["node", "project"] = Field(alias="chatScope")
    provider: str | None = None
    model: str | None = None
    reasoning: str | None = None
    execution_machine: str | None = Field(default=None, alias="executionMachine")
    timestamp: str
    uuid: str
    operation_id: str | None = Field(default=None, alias="operationId")
    type: Literal["user", "assistant"]
    role: Literal["user", "assistant"]
    text: str
    applied_revision: int | None = Field(default=None, alias="appliedRevision", ge=0)
    mode: ConversationMode | None = None
    graph_update: GraphUpdateResult | None = Field(default=None, alias="graphUpdate")
    trigger: TaskTrigger = "human"
    attachments: list[ChatAttachmentDescriptor] = Field(default_factory=list)


@dataclass(frozen=True)
class CanonicalChatBackupSource:
    """One safe append-only chat and the exact byte boundary backup may read."""

    path: Path
    observed_bytes: int
    device: int
    inode: int


def _canonical_chat_path(
    root: Path,
    chat_id: str,
    *,
    chat_scope: Literal["node", "project"],
    node_id: str | None,
) -> Path:
    target = node_id if chat_scope == "node" else "project"
    if target is None:
        raise ValueError("Node chat requires a node_id")
    safe_target = re.sub(r"[^A-Za-z0-9._-]+", "_", target).strip("._") or "node"
    return root / "chat" / f"{safe_target}-{chat_id}.jsonl"


def canonical_chat_backup_sources(root: Path) -> tuple[CanonicalChatBackupSource, ...]:
    """Inventory every canonical chat without refreshing or taking its append lock."""

    chat_dir = root / "chat"
    try:
        directory = chat_dir.lstat()
    except FileNotFoundError:
        return ()
    except OSError as exc:
        raise ValueError("The canonical chat directory is unavailable.") from exc
    if not stat.S_ISDIR(directory.st_mode):
        raise ValueError("The canonical chat path is not a safe directory.")
    try:
        candidates = sorted(chat_dir.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        raise ValueError("The canonical chat directory cannot be enumerated.") from exc
    if len(candidates) > BACKUP_INVENTORY_MAX_ENTRIES:
        raise ValueError("The canonical chat inventory exceeds its entry bound.")
    sources: list[CanonicalChatBackupSource] = []
    for path in candidates:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise ValueError("A canonical chat file cannot be inspected.") from exc
        if path.parent != chat_dir or path.suffix != ".jsonl" or not stat.S_ISREG(metadata.st_mode):
            raise ValueError("The canonical chat directory contains an unsafe entry.")
        sources.append(
            CanonicalChatBackupSource(
                path=path,
                observed_bytes=metadata.st_size,
                device=metadata.st_dev,
                inode=metadata.st_ino,
            )
        )
    return tuple(sources)


def iter_canonical_chat_backup_prefix(
    source: CanonicalChatBackupSource,
    *,
    project_id: str,
    operation_projects: Mapping[str, str],
) -> Iterator[bytes]:
    """Yield the typed complete prefix whose operations exist in the DB snapshot."""

    descriptor = os.open(source.path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        initial = os.fstat(descriptor)
        if (
            not stat.S_ISREG(initial.st_mode)
            or (initial.st_dev, initial.st_ino) != (source.device, source.inode)
            or initial.st_size < source.observed_bytes
        ):
            raise ValueError("The canonical chat changed before its backup read.")
        first: _StoredChatRecord | None = None
        remaining = source.observed_bytes
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while remaining:
                line = handle.readline(remaining)
                remaining -= len(line)
                if not line or not line.endswith(b"\n"):
                    break
                try:
                    raw = json.loads(line)
                    record = _StoredChatRecord.model_validate(raw)
                    _validate_stored_chat_record(
                        record,
                        first=first,
                        root=source.path.parent.parent,
                        path=source.path,
                    )
                except (TypeError, UnicodeError, ValueError) as exc:
                    raise ValueError("A canonical chat record is malformed.") from exc
                if first is None:
                    first = record
                if record.operation_id is not None:
                    try:
                        operation_id = str(uuid.UUID(record.operation_id))
                    except ValueError as exc:
                        raise ValueError(
                            "A canonical chat operation identity is malformed."
                        ) from exc
                    if operation_id != record.operation_id:
                        raise ValueError("A canonical chat operation identity is not canonical.")
                    owner = operation_projects.get(operation_id)
                    if owner is None:
                        break
                    if owner != project_id:
                        raise ValueError("A canonical chat references another project's task.")
                yield line
        final = os.fstat(descriptor)
        try:
            current = source.path.lstat()
        except OSError as exc:
            raise ValueError("The canonical chat changed during its backup read.") from exc
        if (
            (final.st_dev, final.st_ino) != (source.device, source.inode)
            or (current.st_dev, current.st_ino) != (source.device, source.inode)
            or final.st_size < source.observed_bytes
        ):
            raise ValueError("The canonical chat changed during its backup read.")
    finally:
        os.close(descriptor)


def iter_canonical_chat_transfer(
    source: CanonicalChatBackupSource,
    *,
    operation_id_map: Mapping[str, str],
) -> Iterator[bytes]:
    """Yield one typed chat with source-native and unmapped task bindings removed."""

    descriptor = os.open(source.path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        initial = os.fstat(descriptor)
        if (
            not stat.S_ISREG(initial.st_mode)
            or (initial.st_dev, initial.st_ino) != (source.device, source.inode)
            or initial.st_size < source.observed_bytes
        ):
            raise ValueError("The canonical chat changed before its transfer read.")
        first: _StoredChatRecord | None = None
        remaining = source.observed_bytes
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while remaining:
                line = handle.readline(remaining)
                remaining -= len(line)
                if not line or not line.endswith(b"\n"):
                    raise ValueError("A canonical chat record is incomplete.")
                try:
                    raw = json.loads(line)
                    record = _StoredChatRecord.model_validate(raw)
                    _validate_stored_chat_record(
                        record,
                        first=first,
                        root=source.path.parent.parent,
                        path=source.path,
                    )
                except (TypeError, UnicodeError, ValueError) as exc:
                    raise ValueError("A canonical chat record is malformed.") from exc
                if first is None:
                    first = record
                mapped_operation_id = None
                if record.operation_id is not None:
                    try:
                        source_operation_id = str(uuid.UUID(record.operation_id))
                    except ValueError as exc:
                        raise ValueError(
                            "A canonical chat operation identity is malformed."
                        ) from exc
                    if source_operation_id != record.operation_id:
                        raise ValueError("A canonical chat operation identity is not canonical.")
                    mapped_operation_id = operation_id_map.get(source_operation_id)
                    if mapped_operation_id is not None:
                        try:
                            canonical_target = str(uuid.UUID(mapped_operation_id))
                        except ValueError as exc:
                            raise ValueError(
                                "A mapped canonical chat operation identity is malformed."
                            ) from exc
                        if canonical_target != mapped_operation_id:
                            raise ValueError(
                                "A mapped canonical chat operation identity is not canonical."
                            )
                transferred = record.model_copy(
                    update={
                        "native_session_id": None,
                        "execution_machine": None,
                        "operation_id": mapped_operation_id,
                    }
                )
                yield (
                    json.dumps(
                        transferred.model_dump(mode="json", by_alias=True, exclude_none=True),
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                        allow_nan=False,
                    ).encode("utf-8")
                    + b"\n"
                )
        if first is None:
            raise ValueError("A canonical chat cannot transfer without messages.")
        final = os.fstat(descriptor)
        try:
            current = source.path.lstat()
        except OSError as exc:
            raise ValueError("The canonical chat changed during its transfer read.") from exc
        if (
            (final.st_dev, final.st_ino) != (source.device, source.inode)
            or (current.st_dev, current.st_ino) != (source.device, source.inode)
            or final.st_size < source.observed_bytes
            or current.st_size < source.observed_bytes
        ):
            raise ValueError("The canonical chat changed during its transfer read.")
    finally:
        os.close(descriptor)


def _validate_stored_chat_record(
    record: _StoredChatRecord,
    *,
    first: _StoredChatRecord | None,
    root: Path,
    path: Path,
) -> None:
    chat_id = str(uuid.UUID(record.session_id))
    message_id = str(uuid.UUID(record.uuid))
    timestamp = datetime.fromisoformat(record.timestamp)
    if (
        chat_id != record.session_id
        or message_id != record.uuid
        or timestamp.tzinfo is None
        or record.type != record.role
    ):
        raise ValueError("canonical chat record identity is invalid")
    anchor = first or record
    if (
        record.session_id != anchor.session_id
        or record.chat_scope != anchor.chat_scope
        or record.node_id != anchor.node_id
    ):
        raise ValueError("canonical chat record context changed")
    if (anchor.chat_scope == "node" and not anchor.node_id) or (
        anchor.chat_scope == "project" and anchor.node_id is not None
    ):
        raise ValueError("canonical chat scope is invalid")
    if path != _canonical_chat_path(
        root,
        anchor.session_id,
        chat_scope=anchor.chat_scope,
        node_id=anchor.node_id,
    ):
        raise ValueError("canonical chat path does not match its records")


class ReviewRequest(BaseModel):
    standing: Literal["asserted", "accepted", "contested"]


class NodeEditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_updated_rev: int = Field(ge=0)
    changes: dict[str, Any] = Field(min_length=1)


class NodeEditConflict(ValueError):
    pass


class ProposalDecisionRequest(BaseModel):
    decision: Literal["approved", "rejected"]
    reason: str | None = None


class GraphSyncNodeChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    base_updated_rev: int = Field(ge=0)
    changes: dict[str, Any] = Field(default_factory=dict)
    standing: Literal["asserted", "accepted", "contested"] | None = None
    cancel_attempt_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_change(self) -> GraphSyncNodeChange:
        if not self.changes and self.standing is None and not self.cancel_attempt_ids:
            raise ValueError("a staged node must change wording, standing, or an open attempt")
        if len(self.cancel_attempt_ids) != len(set(self.cancel_attempt_ids)):
            raise ValueError("a staged node cannot cancel the same attempt twice")
        return self


class GraphSyncProposalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    decision: Literal["approved", "rejected"]
    reason: str | None = None


class GraphSyncRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_revision: int = Field(ge=0)
    nodes: list[GraphSyncNodeChange] = Field(default_factory=list)
    proposals: list[GraphSyncProposalDecision] = Field(default_factory=list)
    ontology: OntologyState | None = None
    custom_nodes: list[ProjectNode] = Field(default_factory=list)
    removed_node_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_unique_targets(self) -> GraphSyncRequest:
        for label, values in (
            ("node", [item.node_id for item in self.nodes]),
            ("proposal", [item.proposal_id for item in self.proposals]),
            ("custom node", [item.id for item in self.custom_nodes]),
            ("removed node", self.removed_node_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"a graph sync cannot contain duplicate {label} targets")
        staged_node_ids = {item.node_id for item in self.nodes}
        removed_node_ids = set(self.removed_node_ids)
        conflicting_node_ids = sorted(staged_node_ids & removed_node_ids)
        if conflicting_node_ids:
            raise ValueError(
                "a graph sync cannot both change and remove the same node: "
                f"{', '.join(conflicting_node_ids)}"
            )
        return self


class CreateResultViewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["create"]


class ReviseResultViewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["revise"]
    view_id: str = Field(pattern=r"^[0-9a-f]{24}$")


ResultViewRequest = Annotated[
    CreateResultViewRequest | ReviseResultViewRequest,
    Field(discriminator="action"),
]


class ArtifactSelectionRect(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def stays_inside_viewport(self) -> ArtifactSelectionRect:
        if self.x + self.width > 1 or self.y + self.height > 1:
            raise ValueError("artifact selection rectangle must stay inside its viewport")
        return self


class ArtifactViewport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    width: int = Field(ge=1, le=32768)
    height: int = Field(ge=1, le=32768)


class ArtifactTextSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    kind: Literal["text"]
    text: str = Field(min_length=1, max_length=4096)
    surrounding_text: str = Field(default="", max_length=6144)
    comment: str = Field(default="", max_length=2048)


class ArtifactBoxSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    kind: Literal["box"]
    rect: ArtifactSelectionRect
    viewport: ArtifactViewport
    labels: str = Field(default="", max_length=4096)
    comment: str = Field(default="", max_length=2048)


ArtifactSelection = Annotated[
    ArtifactTextSelection | ArtifactBoxSelection,
    Field(discriminator="kind"),
]


class ArtifactContextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    source: Literal["task", "episode_report"] = "task"
    operation_id: str = Field(min_length=1)
    artifact_id: str = Field(pattern=r"^[0-9a-f]{24}$")
    episode_id: str | None = None
    selections: list[ArtifactSelection] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def source_identity_is_coherent(self) -> ArtifactContextRequest:
        if (self.source == "episode_report") != (self.episode_id is not None):
            raise ValueError("episode report context requires exactly one episode_id")
        return self


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    provider: ProviderId | None = None
    run_truth_scope: list[str] | None = None
    model: str | None = None
    reasoning: str | None = None
    run_on: str | None = None
    chat_scope: Literal["node", "project"] = "node"
    node_id: str | None = None
    message: str | None = None
    chat_id: str | None = None
    session_id: str | None = None
    mode: ConversationMode = "discuss"
    result_view: ResultViewRequest | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    artifact_context: ArtifactContextRequest | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    trigger: TaskTrigger = "human"
    patch_kind: GraphPatchKind = "work"
    control_node_id: str | None = None
    control_revision: int | None = Field(default=None, ge=0)
    control_episode_id: str | None = None
    control_invocation: int | None = Field(default=None, ge=1)
    control_invocation_ceiling: int | None = Field(default=None, ge=1)
    control_decision_bundle: list[ExperimentDecisionPin] = Field(default_factory=list)
    control_completion_criteria: list[str] = Field(default_factory=list)
    watcher_ids: list[str] = Field(default_factory=list)
    workflow_ids: list[str] | None = None
    skill_ids: list[str] | None = None
    invoked_workflow_ids: list[str] = Field(default_factory=list)
    invoked_skill_ids: list[str] = Field(default_factory=list)
    invoked_provider_skill_names: list[str] = Field(default_factory=list)
    resolved_provider_skills: list[ProviderSkillReference] = Field(default_factory=list)
    resolved_skill_packages: list[SkillReference] | None = None
    # The set/client fields are short-lived ingress capabilities accepted only on a
    # human chat request. RCP replaces them with the claimed batch and metadata
    # before the task record is created.
    attachment_set_id: str | None = None
    attachment_client_id: str | None = None
    attachment_batch_id: str | None = None
    attachments: list[ChatAttachmentDescriptor] = Field(default_factory=list)

    @model_validator(mode="after")
    def result_view_requires_node_work(self) -> RunRequest:
        if self.result_view is None:
            return self
        if self.mode != "work" or self.chat_scope != "node" or not self.node_id:
            raise ValueError("a result view requires node-scoped Work with a node_id")
        return self


class CoachRequest(BaseModel):
    message: str
    provider: ProviderId | None = None
    model: str | None = None
    reasoning: str | None = None
    run_on: str | None = None
    session_id: str | None = None
    workflow_ids: list[str] | None = None
    skill_ids: list[str] | None = None
    invoked_workflow_ids: list[str] = Field(default_factory=list)
    invoked_skill_ids: list[str] = Field(default_factory=list)
    invoked_provider_skill_names: list[str] = Field(default_factory=list)
    resolved_provider_skills: list[ProviderSkillReference] = Field(default_factory=list)
    resolved_skill_packages: list[SkillReference] | None = None


def resolve_dispatch_authority(
    kind: str,
    request: object,
) -> AgentDispatchAuthority | None:
    """Resolve one profile binding from the server-owned task shape."""

    if kind == "auto_research":
        if not isinstance(request, AutoResearchRunRequest):
            raise TypeError("auto_research dispatch requires an AutoResearchRunRequest")
        return AgentDispatchAuthority(
            profile="orchestrator" if request.role == "orchestrator" else "ordinary",
            task_contract="orchestrate" if request.role == "orchestrator" else "work_auto",
            scope=AgentDispatchScope(
                run_truth_scope=sorted(set(request.run_truth_scope or ())),
                episode_id=request.episode_id,
                patch_kind="work",
            ),
        )
    if kind == "paper_coach":
        if not isinstance(request, CoachRequest):
            raise TypeError("paper_coach dispatch requires a CoachRequest")
        return AgentDispatchAuthority(
            profile="ordinary",
            task_contract="paper_readonly",
            scope=AgentDispatchScope(),
        )
    if not isinstance(request, RunRequest):
        raise TypeError(f"{kind} dispatch requires a RunRequest")

    run_truth_scope = sorted(set(request.run_truth_scope or ()))
    if kind in {"seed", "refresh"}:
        return AgentDispatchAuthority(
            profile="ordinary",
            task_contract="scratch_patch",
            scope=AgentDispatchScope(
                run_truth_scope=run_truth_scope,
                patch_kind=kind,
            ),
        )
    if kind not in {"node_chat", "project_chat"}:
        raise ValueError(f"Authority refused action 'dispatch': unknown task kind {kind!r}.")

    chat_scope: Literal["node", "project"] = "node" if kind == "node_chat" else "project"
    if request.chat_scope != chat_scope:
        raise ValueError(
            "Authority refused action 'dispatch': task kind and chat scope do not match."
        )
    work = request.mode == "work"
    patch_kind = request.patch_kind if work else None
    experiment = patch_kind == "experiment_loop"
    return AgentDispatchAuthority(
        profile="ordinary",
        task_contract="work_auto" if work else "discuss",
        scope=AgentDispatchScope(
            run_truth_scope=run_truth_scope,
            chat_scope=chat_scope,
            chat_id=request.chat_id,
            node_id=request.node_id if chat_scope == "node" else None,
            patch_kind=patch_kind,
            control_node_id=request.control_node_id if experiment else None,
            control_episode_id=request.control_episode_id if experiment else None,
        ),
    )


class AgentProfileSettings(BaseModel):
    provider: ProviderId
    runtime: str = ""
    model: str = ""
    reasoning: str = "medium"
    run_on: str = Field(min_length=1)


class ProjectSettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_run_truth_scope: list[str] = Field(min_length=1)
    default_auto_research_invocation_ceiling: int | None = Field(
        default=None,
        ge=1,
        description="Operational invocations per newly authorized episode.",
    )
    agent_profiles: dict[AgentExecutionProfile, AgentProfileSettings]
    skill_defaults: SkillDefaults = Field(default_factory=SkillDefaults)
    # Partial by machine and provider. Omission preserves every recorded path;
    # an empty string explicitly clears one provider's record.
    machine_provider_paths: dict[str, dict[ProviderId, str]] | None = None

    @model_validator(mode="after")
    def require_every_surface(self) -> ProjectSettingsRequest:
        expected = set(_SETTINGS_SURFACES)
        actual = set(self.agent_profiles)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"agent profiles must contain every surface; missing={missing}, extra={extra}"
            )
        return self


def _imported_source_store(
    data_dir: Path | None,
    project_id: str | None,
) -> ImportedProviderSourceStore | None:
    if data_dir is None or project_id is None:
        return None
    try:
        parsed = uuid.UUID(project_id)
    except ValueError:
        return None
    if parsed.version != 4 or str(parsed) != project_id:
        return None
    return ImportedProviderSourceStore(data_dir, project_id)


class ProjectService:
    def __init__(
        self,
        manifest: Manifest,
        history: HistoryManager,
        paper: PaperService,
        launcher: AgentLauncher | None = None,
        data_dir: Path | None = None,
        provider_skills: ProviderSkillInventoryManager | None = None,
        project_id: str | None = None,
        repository_inventory: Callable[[], list[RegisteredRepositoryRoot]] | None = None,
        task_continuation_session: Callable[[str, str], str | None] | None = None,
    ) -> None:
        self.history = history
        self.paper = paper
        self.launcher = launcher or AgentLauncher()
        self.provider_skills = provider_skills
        self._data_dir = data_dir
        self._project_id = project_id or history.project_id
        self.imported_sources = _imported_source_store(data_dir, self._project_id)
        self._repository_inventory = repository_inventory
        self._task_continuation_session = task_continuation_session
        state_repository = manifest.repository_map[manifest.state.repository]
        state_machine = manifest.machine_map[state_repository.machine]
        app_chat_origin = AppChatOrigin(
            machine=state_repository.machine,
            host=state_machine.host if history.workspace.remote else "",
            root=(
                str(PurePosixPath(state_repository.path) / ".research" / "chat")
                if history.workspace.remote
                else str(manifest.research_dir / "chat")
            ),
        )
        canonical_project_id = project_id or history.project_id
        cache_root = None
        if data_dir is not None and canonical_project_id is not None:
            cache_root = project_cache_roots(data_dir, canonical_project_id)[0]
        self.indexer = ConversationIndexer(
            manifest,
            cache_root,
            app_chat_origin=app_chat_origin,
        )
        self._index_lock = threading.Lock()
        self._indexes: dict[str, ConversationIndex] = {}
        self._chat_summary_lock = threading.Lock()
        self._chat_summary_cache: dict[Path, _ChatSummaryCacheEntry] = {}

    @property
    def manifest(self) -> Manifest:
        return self.history.manifest

    def repository_ownership_inventory(
        self,
        *,
        project_id: str,
    ) -> list[RegisteredRepositoryRoot]:
        if self._repository_inventory is not None:
            return self._repository_inventory()
        return registered_repository_roots(self.manifest, project_id=project_id)

    def for_graph_target(
        self,
        target: GraphTargetRef,
        *,
        expected_episode_id: str | None = None,
    ) -> ProjectService:
        """Return a service whose graph reads and writes stay on one exact target."""

        if target.kind == "main":
            if expected_episode_id is not None:
                raise ValueError("an episode-target service requires a graph branch")
            return self
        assert target.branch_id is not None
        history = self.history.branch(
            target.branch_id,
            expected_episode_id=expected_episode_id,
            expected_project_id=self._project_id,
        )
        return ProjectService(
            history.manifest,
            history,
            self.paper,
            self.launcher,
            data_dir=self._data_dir,
            provider_skills=self.provider_skills,
            project_id=self._project_id,
            repository_inventory=self._repository_inventory,
            task_continuation_session=self._task_continuation_session,
        )

    def chat_path(
        self,
        chat_id: str,
        *,
        chat_scope: Literal["node", "project"],
        node_id: str | None,
    ) -> Path:
        return _canonical_chat_path(
            self.history.workspace.root,
            chat_id,
            chat_scope=chat_scope,
            node_id=node_id,
        )

    def restore_canonical_chat(
        self,
        source: Path,
        *,
        expected_sha256: str,
        expected_size: int,
        operation_projects: Mapping[str, str],
    ) -> Path:
        """Validate and publish one archived canonical chat without appending to it."""

        try:
            metadata = source.lstat()
        except OSError as exc:
            raise ValueError("The archived canonical chat is unavailable.") from exc
        if not stat.S_ISREG(metadata.st_mode) or expected_size <= 0:
            raise ValueError("The archived canonical chat is not a nonempty regular file.")
        descriptor = CanonicalChatBackupSource(
            path=source,
            observed_bytes=expected_size,
            device=metadata.st_dev,
            inode=metadata.st_ino,
        )
        digest = hashlib.sha256()
        observed = 0
        for line in iter_canonical_chat_backup_prefix(
            descriptor,
            project_id=self._project_id,
            operation_projects=operation_projects,
        ):
            digest.update(line)
            observed += len(line)
        if observed != expected_size or digest.hexdigest() != expected_sha256:
            raise ValueError("The archived canonical chat differs from its manifest proof.")
        relative = Path("chat") / source.name
        self.history.workspace.restore_exact_file(
            relative,
            source,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
        )
        return self.history.workspace.root / relative

    def chat_summaries(
        self,
        *,
        offset: int = 0,
        limit: int = CHAT_PAGE_DEFAULT_LIMIT,
    ) -> ChatSummaryPage:
        if offset < 0:
            raise ValueError("chat offset must be non-negative")
        if limit < 1 or limit > CHAT_PAGE_MAX_LIMIT:
            raise ValueError(f"chat limit must be between 1 and {CHAT_PAGE_MAX_LIMIT}")
        chats = sorted(
            self._canonical_chat_summaries(),
            key=lambda item: (datetime.fromisoformat(item.updated_at), item.chat_id),
            reverse=True,
        )
        return ChatSummaryPage(
            items=chats[offset : offset + limit],
            total=len(chats),
            offset=offset,
            limit=limit,
        )

    def chat_transcript(self, chat_id: str) -> ChatTranscript | None:
        try:
            normalized = str(uuid.UUID(chat_id))
        except ValueError as exc:
            raise ValueError("chat_id must be a UUID") from exc
        if normalized != chat_id:
            raise ValueError("chat_id must be a canonical UUID")
        suffix = f"-{chat_id}.jsonl"
        with self.history.workspace.snapshot_lock:
            transcripts = [
                transcript
                for path, _ in self._canonical_chat_files()
                if path.name.endswith(suffix)
                and (transcript := self._read_chat_transcript(path)) is not None
                and transcript.chat_id == chat_id
            ]
        # The same UUID under two canonical node/project paths is ambiguous.
        return transcripts[0] if len(transcripts) == 1 else None

    def _canonical_chat_summaries(self) -> list[ChatSummary]:
        with self.history.workspace.snapshot_lock:
            files = self._canonical_chat_files()
            live_paths = {path for path, _ in files}
            summaries: dict[str, ChatSummary | None] = {}
            with self._chat_summary_lock:
                for stale in self._chat_summary_cache.keys() - live_paths:
                    del self._chat_summary_cache[stale]
                for path, fingerprint in files:
                    cached = self._chat_summary_cache.get(path)
                    if cached is None or cached.fingerprint != fingerprint:
                        transcript = self._read_chat_transcript(path)
                        summary = (
                            ChatSummary.model_validate(transcript.model_dump(exclude={"messages"}))
                            if transcript is not None
                            else None
                        )
                        cached = _ChatSummaryCacheEntry(fingerprint, summary)
                        self._chat_summary_cache[path] = cached
                    if cached.summary is None:
                        continue
                    chat_id = cached.summary.chat_id
                    if chat_id in summaries:
                        # One conversation has exactly one canonical file. Ambiguous
                        # duplicates are safer hidden than selected by filesystem order.
                        summaries[chat_id] = None
                    else:
                        summaries[chat_id] = cached.summary
            return [summary for summary in summaries.values() if summary is not None]

    def _canonical_chat_files(
        self,
    ) -> list[tuple[Path, tuple[int, int, int, int, int]]]:
        """Reconcile a stale mirror, then return safe canonical file candidates."""

        workspace = self.history.workspace
        if workspace.remote:
            workspace.refresh_if_stale()
        chat_dir = workspace.root / "chat"
        try:
            directory_stat = chat_dir.lstat()
        except OSError:
            return []
        if stat.S_ISLNK(directory_stat.st_mode) or not stat.S_ISDIR(directory_stat.st_mode):
            return []

        files: list[tuple[Path, tuple[int, int, int, int, int]]] = []
        try:
            candidates = sorted(chat_dir.iterdir(), key=lambda path: path.name)
        except OSError:
            return []
        for path in candidates:
            if path.parent != chat_dir or path.suffix != ".jsonl":
                continue
            try:
                file_stat = path.lstat()
            except OSError:
                continue
            if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
                continue
            files.append(
                (
                    path,
                    (
                        file_stat.st_dev,
                        file_stat.st_ino,
                        file_stat.st_size,
                        file_stat.st_mtime_ns,
                        file_stat.st_ctime_ns,
                    ),
                )
            )
        return files

    def _read_chat_transcript(self, path: Path) -> ChatTranscript | None:
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    return None
                with os.fdopen(descriptor, encoding="utf-8") as handle:
                    descriptor = -1
                    lines = handle.read().splitlines()
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
            raw_records = [json.loads(line) for line in lines if line.strip()]
            records = [_StoredChatRecord.model_validate(record) for record in raw_records]
        except (OSError, TypeError, ValueError):
            return None
        if not records:
            return None

        first = records[0]
        try:
            timestamps = [datetime.fromisoformat(record.timestamp) for record in records]
            for record in records:
                _validate_stored_chat_record(
                    record,
                    first=first,
                    root=self.history.workspace.root,
                    path=path,
                )
        except ValueError:
            return None
        chat_id = first.session_id
        if first.chat_scope == "node":
            kind: Literal["node_chat", "project_chat"] = "node_chat"
        else:
            kind = "project_chat"

        messages: list[ChatMessage] = []
        for record in records:
            native_session_id = record.native_session_id
            if self._task_continuation_session is not None:
                native_session_id = (
                    self._task_continuation_session(self._project_id, record.operation_id)
                    if record.operation_id is not None
                    else None
                )
            messages.append(
                ChatMessage(
                    message_id=record.uuid,
                    operation_id=record.operation_id,
                    role=record.role,
                    text=record.text,
                    timestamp=record.timestamp,
                    native_session_id=native_session_id,
                    provider=record.provider,
                    model=record.model,
                    reasoning=record.reasoning,
                    execution_machine=record.execution_machine,
                    applied_revision=record.applied_revision,
                    mode=record.mode,
                    graph_update=record.graph_update,
                    trigger=record.trigger,
                    attachments=record.attachments,
                )
            )
        first_user = next((message.text for message in messages if message.role == "user"), "")
        title = " ".join(first_user.split())[:CHAT_TITLE_MAX_CHARS]
        if not title:
            title = first.node_id or "Project chat"
        preview_source = next(
            (message.text for message in reversed(messages) if message.text.strip()),
            "",
        )
        preview = " ".join(preview_source.split())[:CHAT_PREVIEW_MAX_CHARS]
        updated_at = records[max(range(len(records)), key=timestamps.__getitem__)].timestamp
        return ChatTranscript(
            chat_id=chat_id,
            kind=kind,
            node_id=first.node_id,
            title=title,
            updated_at=updated_at,
            message_count=len(messages),
            last_message_preview=preview,
            messages=messages,
        )

    def project_snapshot(
        self,
        *,
        state: GraphState | None = None,
        paper: PaperSnapshot | None = None,
    ) -> _ProjectSnapshotDraft:
        if state is None:
            state = self.history.state()
        if paper is None:
            paper = self.paper.snapshot()
        primary = self._primary_question(state)
        attention = project_graph_attention(state)
        refresh_profile = self.manifest.agent_profile("refresh")
        profiles = {
            surface: self.manifest.agent_profile(surface).model_dump(mode="json")
            for surface in _SETTINGS_SURFACES
        }
        return _ProjectSnapshotDraft(
            {
                "name": self.manifest.name,
                "revision": state.revision,
                "state_repository": self.manifest.state.repository,
                "canonical_state": self.history.workspace.status().model_dump(mode="json"),
                "run_on": refresh_profile.run_on,
                "project_truth_scope": state.project_truth_scope,
                "default_run_truth_scope": self.manifest.agent.default_run_truth_scope,
                "default_auto_research_invocation_ceiling": (
                    self.manifest.agent.default_auto_research_invocation_ceiling
                ),
                "repositories": [
                    repository.model_dump() for repository in self.manifest.repositories
                ],
                "machines": [machine.model_dump() for machine in self.manifest.machines],
                "primary_question": primary,
                "last_refresh_at": state.last_refresh_at,
                "attention": attention.model_dump(mode="json"),
                "counts": {
                    "pending_proposals": len(attention.pending_proposal_ids),
                    "decisions_awaiting_choice": len(attention.decisions_awaiting_choice_ids),
                    "open_blockers": len(attention.open_blocker_ids),
                    "asserted": sum(
                        node.standing == Standing.ASSERTED for node in state.nodes.values()
                    ),
                    "accepted": sum(
                        node.standing == Standing.ACCEPTED for node in state.nodes.values()
                    ),
                    "contested": sum(
                        node.standing == Standing.CONTESTED for node in state.nodes.values()
                    ),
                },
                "coverage": state.coverage.model_dump(mode="json"),
                "graph": state.model_dump(mode="json"),
                "paper": paper.model_dump(mode="json"),
                "paper_coach": self.manifest.coach.model_dump(mode="json"),
                "agent_profiles": profiles,
                "skill_catalog": official_registry().catalog(),
                "skill_defaults": self.manifest.agent.skill_defaults.model_dump(mode="json"),
                "provider_readiness": {},
                "provider_skill_inventories": self.provider_skill_inventory_snapshot(),
                "providers": {},
                "cache_metrics": self.indexer.cache_metrics().model_dump(mode="json"),
                "validation_messages": [
                    item.model_dump(mode="json") for item in state.validation_messages
                ],
            }
        )

    def readiness_snapshot(self, *, refresh: bool = False) -> dict[str, object]:
        snapshot = self.readiness_for(self.manifest, self.launcher, refresh=refresh)
        self.wait_for_provider_skill_inventories()
        snapshot["provider_skill_inventories"] = self.provider_skill_inventory_snapshot()
        return snapshot

    def wait_for_provider_skill_inventories(self) -> None:
        """Wait only for an already-scheduled startup refresh; never start one."""

        self.wait_for_provider_skill_inventories_for(self.manifest, self.provider_skills)

    @staticmethod
    def wait_for_provider_skill_inventories_for(
        manifest: Manifest,
        provider_skills: ProviderSkillInventoryManager | None,
    ) -> None:
        if provider_skills is None:
            return
        for machine in manifest.machines:
            for provider in PROVIDER_IDS:
                provider_skills.wait(
                    provider,
                    machine.host,
                    machine.provider_paths.get(provider),
                )

    def provider_skill_inventory_snapshot(self) -> dict[str, dict[ProviderId, object]]:
        """Return startup-cached provider skills without probing a provider."""

        return self.provider_skill_inventories_for(self.manifest, self.provider_skills)

    @staticmethod
    def provider_skill_inventories_for(
        manifest: Manifest,
        provider_skills: ProviderSkillInventoryManager | None,
    ) -> dict[str, dict[ProviderId, object]]:
        """Project provider inventories mapped through its machine aliases."""

        if provider_skills is None:
            return {
                machine.alias: {provider: None for provider in PROVIDER_IDS}
                for machine in manifest.machines
            }
        return {
            machine.alias: {
                provider: provider_skills.snapshot(
                    provider,
                    machine.host,
                    machine.provider_paths.get(provider),
                    machine.alias,
                ).model_dump(mode="json")
                for provider in PROVIDER_IDS
            }
            for machine in manifest.machines
        }

    @staticmethod
    def readiness_for(
        manifest: Manifest,
        launcher: AgentLauncher,
        *,
        refresh: bool = False,
    ) -> dict[str, object]:
        """Probe providers without reading or replaying canonical project history."""

        targets = [
            (
                machine.alias,
                machine.host,
                provider,
                machine.provider_paths.get(provider),
            )
            for machine in manifest.machines
            for provider in PROVIDER_IDS
        ]

        def inspect(
            host: str,
            provider: ProviderId,
            binary: str | None,
        ) -> dict[str, object]:
            kwargs: dict[str, object] = {"host": host}
            if binary is not None:
                kwargs["binary"] = binary
            if refresh:
                kwargs["refresh"] = True
            return launcher.readiness(provider, **kwargs).model_dump(mode="json")

        readiness_by_machine: dict[str, dict[ProviderId, dict[str, object]]] = {
            machine.alias: {} for machine in manifest.machines
        }
        with ThreadPoolExecutor(max_workers=min(len(targets), 8)) as executor:
            probes = [
                (
                    alias,
                    provider,
                    executor.submit(inspect, host, provider, binary),
                )
                for alias, host, provider, binary in targets
            ]
            for alias, provider, probe in probes:
                readiness_by_machine[alias][provider] = probe.result()
        coach_machine = manifest.agent_profile("paper_coach").run_on
        return {
            "provider_readiness": readiness_by_machine,
            "providers": readiness_by_machine[coach_machine],
        }

    def clear_rebuildable_caches(self) -> dict[str, object]:
        with self._index_lock:
            self._indexes.clear()
            return self.indexer.clear_rebuildable_caches().model_dump(mode="json")

    def graph_snapshot(self) -> dict[str, object]:
        state = self.history.state()
        return state.model_dump(mode="json")

    def index_snapshot(
        self,
        *,
        refresh: bool = False,
        execution_machine: str | None = None,
        pin_artifact: Callable[[Path], None] | None = None,
    ) -> ConversationIndex:
        with self._index_lock:
            self.indexer.manifest = self.manifest
            key = execution_machine or "cached-local"
            if refresh or key not in self._indexes:
                self._indexes[key] = self.indexer.build(
                    execution_machine=execution_machine,
                    pin_artifact=pin_artifact,
                )
            return self._indexes[key]

    def invalidate_source_index(self) -> None:
        with self._index_lock:
            self._indexes.clear()

    @staticmethod
    def _settings_profile(
        surface: AgentExecutionProfile,
        requested: AgentProfileSettings,
    ) -> AgentSurfaceConfig:
        """One saved profile, reporting a rejection the human can act on.

        A Pydantic envelope names the model and dumps the whole input. The
        settings form shows this text verbatim, so it says which profile failed
        and nothing else.
        """

        try:
            return AgentSurfaceConfig(
                provider=requested.provider,
                runtime=requested.runtime,
                model=requested.model,
                reasoning=requested.reasoning,
                run_on=requested.run_on,
            )
        except ValidationError as exc:
            detail = exc.errors()[0]["msg"].removeprefix("Value error, ")
            raise ValueError(f"{surface.replace('_', ' ')}: {detail}") from exc

    def update_settings(self, request: ProjectSettingsRequest) -> None:
        profiles = {
            surface: self._settings_profile(surface, request.agent_profiles[surface])
            for surface in _SETTINGS_SURFACES
        }
        provider_path_updates = self._validate_provider_path_updates(request.machine_provider_paths)
        prior_paths = {
            (alias, provider): self.manifest.machine_map[alias].provider_paths.get(provider)
            for alias, updates in (provider_path_updates or {}).items()
            for provider in updates
        }
        self.history.update_agent_settings(
            request.default_run_truth_scope,
            profiles,
            provider_path_updates,
            request.skill_defaults,
            request.default_auto_research_invocation_ceiling,
        )
        for (alias, provider), prior_path in prior_paths.items():
            machine = self.manifest.machine_map[alias]
            current_path = machine.provider_paths.get(provider)
            if prior_path == current_path:
                continue
            if prior_path is not None:
                self.launcher.invalidate_readiness(
                    provider,
                    host=machine.host,
                    binary=prior_path,
                )
            if current_path is not None:
                self.launcher.invalidate_readiness(
                    provider,
                    host=machine.host,
                    binary=current_path,
                )
        self.paper.manifest = self.manifest
        self.invalidate_source_index()

    def resolve_provider_path(
        self,
        machine_alias: str,
        provider: ProviderId,
    ) -> ProviderReadiness:
        try:
            machine = self.manifest.machine_map[machine_alias]
        except KeyError:
            raise ValueError(f"unknown execution machine: {machine_alias}") from None
        discovered = self.launcher.readiness(
            provider,
            host=machine.host,
            refresh=True,
        )
        if discovered.path_state == "unreachable":
            raise ValueError(discovered.reason or f"{machine_alias} is unreachable")
        if not discovered.installed or not discovered.binary_path:
            raise ValueError(
                discovered.reason or f"No {provider} executable was found on {machine_alias}."
            )
        self.history.update_machine_provider_paths(
            {machine_alias: {provider: discovered.binary_path}}
        )
        self.paper.manifest = self.manifest
        return self.launcher.readiness(
            provider,
            host=machine.host,
            binary=discovered.binary_path,
            refresh=True,
        )

    def _validate_provider_path_updates(
        self,
        updates: dict[str, dict[ProviderId, str]] | None,
    ) -> dict[str, dict[ProviderId, str]] | None:
        if updates is None:
            return None
        unknown = set(updates) - set(self.manifest.machine_map)
        if unknown:
            raise ValueError(f"provider paths use unknown machines: {sorted(unknown)}")
        validated: dict[str, dict[ProviderId, str]] = {}
        for alias, provider_updates in updates.items():
            machine = self.manifest.machine_map[alias]
            merged = dict(machine.provider_paths)
            merged.update(provider_updates)
            candidate = MachineConfig(
                alias=machine.alias,
                host=machine.host,
                os_account=machine.os_account,
                provider_paths=merged,
            )
            validated[alias] = {
                provider: candidate.provider_paths.get(provider, "")
                for provider in provider_updates
            }
        return validated

    def review_node(
        self,
        node_id: str,
        request: ReviewRequest,
        *,
        authorized_by: AuthorizedHuman | None = None,
    ) -> GraphState:
        state = self.history.state()
        self.history.require_writable(state)
        if node_id not in state.nodes:
            raise KeyError(node_id)
        node = state.nodes[node_id]
        patch = Patch(
            kind="approval",
            author="human",
            summary=f"Marked “{node.title}” {request.standing}.",
            ops=[{"op": "set_standing", "node_id": node_id, "standing": request.standing}],
            change_summary=[f"“{node.title}” is now {request.standing}."],
        )
        _, result = self.history.append(patch, authorized_by=authorized_by)
        return result.state

    def sync_graph(
        self,
        request: GraphSyncRequest,
        *,
        active_control_node_ids: set[str],
        authorized_by: AuthorizedHuman | None = None,
    ) -> GraphState:
        """Commit one project-wide human draft in one canonical transaction."""

        transition = self.sync_graph_transition(
            request,
            active_control_node_ids=active_control_node_ids,
            authorized_by=authorized_by,
        )
        return (
            transition.projection.graph
            if transition is not None
            else self.history.current_materialization().state
        )

    def preview_sync_graph(
        self,
        request: GraphSyncRequest,
        *,
        active_control_node_ids: set[str],
        authorized_by: AuthorizedHuman | None = None,
    ) -> PreparedTransition:
        """Prepare the same complete Sync candidate without writing canonical history."""

        return self.history.preview_batch_from_state(
            lambda state: self._build_sync_patches(
                request,
                state,
                active_control_node_ids=active_control_node_ids,
            ),
            expected_revision=request.base_revision,
            authorized_by=authorized_by,
        )

    def sync_graph_transition(
        self,
        request: GraphSyncRequest,
        *,
        active_control_node_ids: set[str],
        authorized_by: AuthorizedHuman | None = None,
    ) -> CommittedTransition | None:
        """Commit one project-wide human draft as exactly one graph transition."""

        has_staged_work = (
            any(
                (
                    request.nodes,
                    request.proposals,
                    request.custom_nodes,
                    request.removed_node_ids,
                )
            )
            or request.ontology is not None
        )
        if not has_staged_work:
            return None
        try:
            prepared, result = self.history.append_batch_from_state(
                lambda state: self._build_sync_patches(
                    request,
                    state,
                    active_control_node_ids=active_control_node_ids,
                ),
                expected_revision=request.base_revision,
                authorized_by=authorized_by,
            )
        except ValueError as exc:
            if "graph changed after this draft began" in str(exc):
                raise NodeEditConflict(str(exc)) from exc
            raise
        if not prepared:
            return None
        patch = prepared[0]
        if patch.transition is None:
            raise RuntimeError("graph Sync committed without transition provenance")
        return CommittedTransition(
            patch=patch,
            projection=project_transition_projection(
                result.state,
                patch.transition,
                canonical=True,
            ),
        )

    def _build_sync_patches(
        self,
        request: GraphSyncRequest,
        state: GraphState,
        *,
        active_control_node_ids: set[str],
    ) -> list[Patch]:
        """Build one Sync from the same fresh state that history will append against."""

        ontology_changed = request.ontology is not None and request.ontology != state.ontology
        if (
            not any(
                (
                    request.nodes,
                    request.proposals,
                    request.custom_nodes,
                    request.removed_node_ids,
                )
            )
            and not ontology_changed
        ):
            return []

        patches: list[Patch] = []

        removed_node_ids = set(request.removed_node_ids)
        direct_choice_node_ids = {
            staged.node_id
            for staged in request.nodes
            if isinstance(state.nodes.get(staged.node_id), Decision)
            and _is_decision_choice(staged.changes)
        }
        superseded_proposal_ids = {
            proposal.id
            for proposal in state.proposals.values()
            if proposal.status == "pending"
            and any(
                proposal_updates_node(proposal, decision_id)
                for decision_id in direct_choice_node_ids
            )
        }
        directly_changed_node_ids = {item.node_id for item in request.nodes}
        for judgment in request.proposals:
            if judgment.decision != "approved":
                continue
            proposal = state.proposals.get(judgment.proposal_id)
            if proposal is None or proposal.status != "pending":
                continue
            if proposal.id in superseded_proposal_ids:
                continue
            overlapping_node_ids = sorted(
                directly_changed_node_ids.intersection(proposal.related_node_ids)
            )
            if overlapping_node_ids:
                raise NodeEditConflict(
                    "A graph Sync cannot approve a Proposal and directly change its dependent "
                    "node in the same draft: "
                    f"{', '.join(overlapping_node_ids)}."
                )

        effective_ontology = request.ontology if ontology_changed else state.ontology
        if ontology_changed:
            current_types = {item.name for item in state.ontology.types}
            newly_defined_types = {
                item.name for item in effective_ontology.types if item.name not in current_types
            }
            used_new_types = sorted(
                {
                    node.extension_type
                    for node in request.custom_nodes
                    if node.extension_type in newly_defined_types
                }
            )
            if used_new_types:
                raise ValueError(
                    "This draft both defines and uses a new ontology type "
                    f"({', '.join(used_new_types)}). Sync the ontology first, then create "
                    "nodes of that type in a new draft."
                )
            patches.append(
                Patch(
                    kind="approval",
                    author="human",
                    summary="Updated the project ontology.",
                    ops=[
                        {
                            "op": "set_ontology",
                            "ontology": effective_ontology.model_dump(mode="json"),
                        }
                    ],
                    change_summary=["Updated the project ontology."],
                )
            )

        active_types = {item.name: item for item in effective_ontology.types if not item.deprecated}
        for node in request.custom_nodes:
            extension_type = node.extension_type
            if extension_type is None:
                raise ValueError(
                    "Human-created graph nodes must use an active custom ontology type; "
                    "base-node authoring is not available."
                )
            definition = active_types.get(extension_type)
            if definition is None:
                raise ValueError(
                    f"Custom node {node.id} uses inactive or unknown ontology type "
                    f"{extension_type!r}."
                )
            if node.type != definition.base_type:
                raise ValueError(
                    f"Custom node {node.id} must use base type {definition.base_type!r} "
                    f"for ontology type {extension_type!r}."
                )
            prepared = node.model_copy(
                update={
                    "standing": Standing.ASSERTED,
                    "created_rev": 0,
                    "updated_rev": 0,
                    "source_refs": [],
                }
            )
            patches.append(
                Patch(
                    kind="approval",
                    author="human",
                    summary=f"Created “{node.title}”.",
                    ops=[{"op": "create_nodes", "nodes": [prepared.model_dump(mode="json")]}],
                    change_summary=[
                        f"Created “{node.title}” as a {extension_type.replace('_', ' ')}."
                    ],
                )
            )

        if request.removed_node_ids:
            for node_id in request.removed_node_ids:
                node = state.nodes.get(node_id)
                if node is None:
                    raise KeyError(node_id)
                if node.standing == Standing.ACCEPTED:
                    raise NodeEditConflict(
                        f"Accepted node {node_id} cannot be removed; withdraw its acceptance "
                        "and Sync before removing it."
                    )
                if node.type == "experiment":
                    control = derive_experiment_control_state(
                        state,
                        node_id,
                        active_control_node_ids=active_control_node_ids,
                    )
                    if control.active:
                        raise NodeEditConflict(
                            f"Experiment {node_id} cannot be removed while its bounded "
                            "experiment loop is active."
                        )
            node_ids = list(request.removed_node_ids)
            titles = ", ".join(f"“{state.nodes[node_id].title}”" for node_id in node_ids)
            patches.append(
                Patch(
                    kind="approval",
                    author="human",
                    summary=f"Removed {titles}.",
                    ops=[{"op": "remove_nodes", "node_ids": node_ids}],
                    change_summary=[f"Removed {titles}."],
                )
            )

        decision_state = state
        for patch in patches:
            decision_state = _stage_sync_patch(decision_state, patch)
        for staged in request.proposals:
            proposal = decision_state.proposals.get(staged.proposal_id)
            if proposal is None:
                raise KeyError(staged.proposal_id)
            if proposal.id in superseded_proposal_ids:
                continue
            if proposal.status != "pending":
                raise NodeEditConflict(f"Proposal {proposal.id} is no longer pending.")
            stale_from_removal = bool(removed_node_ids.intersection(proposal.related_node_ids))
            stale_reason = (
                f"The proposal “{proposal.title}” became stale because a related research "
                "concept was removed in this Sync."
                if stale_from_removal
                else None
            )
            patch = _proposal_judgment_patch(
                decision_state,
                proposal,
                decision=staged.decision,
                reason=staged.reason,
                stale_reason=stale_reason,
            )
            patches.append(patch)
            decision_state = _stage_sync_patch(decision_state, patch)

        for staged in request.nodes:
            node = state.nodes.get(staged.node_id)
            if node is None:
                raise KeyError(staged.node_id)
            if staged.base_updated_rev != node.updated_rev:
                raise NodeEditConflict(
                    f"{node.id} changed after this draft began; reload before syncing it."
                )
            ops: list[dict[str, Any]] = []
            change_summary: list[str] = []
            changed_fields: set[str] = set()
            display_title = str(staged.changes.get("title", node.title))
            is_direct_choice = isinstance(node, Decision) and _is_decision_choice(staged.changes)
            if staged.changes:
                allowed = set(HUMAN_EDITABLE_NODE_FIELDS[node.type])
                if (
                    isinstance(node, Decision)
                    and "status" in staged.changes
                    and not is_direct_choice
                    and staged.changes["status"] not in {"open", "ready", "revisit"}
                ):
                    raise ValueError(
                        f"Direct edits to {node.id} may queue it as open, ready, or revisit; "
                        "only the Decision choice control may decide it."
                    )
                if is_direct_choice:
                    if node.status == "superseded":
                        raise ValueError(
                            f"Decision {node.id} is superseded and cannot be decided again."
                        )
                    # Choosing the option a Decision already carries stages only
                    # the status move, so resolve the effective choice against
                    # the node the way the options list already is.
                    selected_option = staged.changes.get("selected_option", node.selected_option)
                    effective_options = staged.changes.get("options", node.options)
                    if staged.changes.get("status") != "decided":
                        raise ValueError(
                            f"Direct choice on {node.id} must set status exactly to decided."
                        )
                    if (
                        not isinstance(selected_option, str)
                        or not selected_option.strip()
                        or not isinstance(effective_options, list)
                        or selected_option not in effective_options
                    ):
                        raise ValueError(
                            f"Direct choice on {node.id} must select one non-empty option from "
                            "its current options."
                        )
                    allowed.update({"selected_option", "status"})
                if "extension_fields" in staged.changes:
                    self._validate_human_extension_fields(state, node, staged.changes)
                    allowed.add("extension_fields")
                disallowed = sorted(set(staged.changes) - allowed)
                if disallowed:
                    raise ValueError(
                        f"Direct edits to {node.id} cannot change: {', '.join(disallowed)}."
                    )
                current = node.model_dump(mode="python")
                changed_fields = {
                    field for field, value in staged.changes.items() if current[field] != value
                }
                if changed_fields or is_direct_choice:
                    candidate = {**current, **staged.changes}
                    try:
                        type(node).model_validate(candidate)
                    except ValueError as exc:
                        raise ValueError(f"Invalid edit for {node.id}: {exc}") from exc
                    ops.append(
                        {
                            "op": "update_nodes",
                            "nodes": [
                                {
                                    "id": node.id,
                                    "base_updated_rev": staged.base_updated_rev,
                                    "changes": staged.changes,
                                }
                            ],
                        }
                    )
                    ordinary_changed_fields = changed_fields - {"selected_option", "status"}
                    if ordinary_changed_fields:
                        change_summary.append(f"Updated wording for “{display_title}”.")
                    if is_direct_choice:
                        change_summary.append(
                            f"Selected “{selected_option}” for “{display_title}”."
                        )
                    elif "status" in changed_fields:
                        change_summary.append(
                            f"Updated lifecycle for “{display_title}”: status is now "
                            f"{staged.changes['status']}."
                        )
            if is_direct_choice:
                withdrawals = [
                    proposal
                    for proposal in state.proposals.values()
                    if proposal.status == "pending" and proposal_updates_node(proposal, node.id)
                ]
                if withdrawals:
                    ops.append(
                        {
                            "op": "resolve_proposals",
                            "resolutions": [
                                {
                                    "id": proposal.id,
                                    "status": "withdrawn",
                                    "reason": (
                                        f"The human decided {node.id} directly, making this "
                                        "Proposal stale."
                                    ),
                                }
                                for proposal in withdrawals
                            ],
                        }
                    )
                    change_summary.extend(
                        f"Withdrew Proposal “{proposal.title}” because the human decided "
                        f"“{display_title}” directly."
                        for proposal in withdrawals
                    )
            if staged.cancel_attempt_ids:
                ops.append(
                    {
                        "op": "update_nodes",
                        "nodes": [
                            {
                                "id": node.id,
                                "base_updated_rev": staged.base_updated_rev,
                                "changes": {
                                    "attempts": self._cancelled_attempts(
                                        node, staged.cancel_attempt_ids
                                    )
                                },
                            }
                        ],
                    }
                )
                change_summary.append(f"Released open experiment attempts for “{display_title}”.")
            resulting_standing = staged.standing
            if (
                changed_fields
                and not is_direct_choice
                and resulting_standing is None
                and node.standing != Standing.ASSERTED
            ):
                resulting_standing = Standing.ASSERTED
            if resulting_standing is not None and resulting_standing != node.standing:
                ops.append(
                    {
                        "op": "set_standing",
                        "node_id": node.id,
                        "standing": resulting_standing,
                    }
                )
                change_summary.append(f"“{display_title}” is now {resulting_standing}.")
            if ops:
                patches.append(
                    Patch(
                        kind="approval",
                        author="human",
                        summary=f"Synced staged changes for “{display_title}”.",
                        ops=ops,
                        change_summary=change_summary,
                        human_action="decision_choice" if is_direct_choice else None,
                    )
                )

        return patches

    @staticmethod
    def _cancelled_attempts(node: ProjectNode, attempt_ids: list[str]) -> list[dict[str, Any]]:
        """Close the named open attempts, leaving every other attempt untouched.

        The human releases an attempt whose watcher can no longer answer. Only an
        open attempt can be released — this never rewrites a finished record.
        """

        from rcp.core.models import Experiment, utc_now

        if not isinstance(node, Experiment):
            raise ValueError(f"{node.id} has no attempts to release.")
        open_ids = {
            attempt.id
            for attempt in node.attempts
            if attempt.status in ACTIVE_EXPERIMENT_ATTEMPT_STATUSES
        }
        unknown = sorted(set(attempt_ids) - open_ids)
        if unknown:
            raise ValueError(f"{node.id} has no open attempt named: {', '.join(unknown)}.")
        finished_at = utc_now()
        return [
            (
                attempt.model_copy(
                    update={
                        "status": "cancelled",
                        "finished_at": finished_at,
                        "failure_reason": "Released by the human.",
                    }
                )
                if attempt.id in set(attempt_ids)
                else attempt
            ).model_dump(mode="json")
            for attempt in node.attempts
        ]

    def edit_node(
        self,
        node_id: str,
        request: NodeEditRequest,
        *,
        authorized_by: AuthorizedHuman | None = None,
    ) -> GraphState:
        state = self.history.state()
        self.history.require_writable(state)
        node = state.nodes.get(node_id)
        if node is None:
            raise KeyError(node_id)
        if request.base_updated_rev != node.updated_rev:
            raise NodeEditConflict(
                f"{node_id} changed after this editor opened; reload it before saving."
            )
        disallowed = sorted(
            set(request.changes)
            - (
                set(HUMAN_EDITABLE_NODE_FIELDS[node.type])
                | ({"extension_fields"} if "extension_fields" in request.changes else set())
            )
        )
        if "extension_fields" in request.changes:
            self._validate_human_extension_fields(state, node, request.changes)
        if disallowed:
            raise ValueError(f"Direct edits to {node_id} cannot change: {', '.join(disallowed)}.")
        current = node.model_dump(mode="python")
        if all(current[field] == value for field, value in request.changes.items()):
            raise ValueError("The submitted node wording is unchanged.")
        candidate = {**current, **request.changes}
        try:
            type(node).model_validate(candidate)
        except ValueError as exc:
            raise ValueError(f"Invalid wording for {node_id}: {exc}") from exc
        patch = Patch(
            kind="approval",
            author="human",
            summary=f"Edited wording for “{request.changes.get('title', node.title)}”.",
            ops=[
                {
                    "op": "update_nodes",
                    "nodes": [
                        {
                            "id": node_id,
                            "base_updated_rev": request.base_updated_rev,
                            "changes": request.changes,
                        }
                    ],
                }
            ],
            change_summary=[
                f"Updated human-authored wording for “{request.changes.get('title', node.title)}”."
            ],
        )
        _, result = self.history.append(patch, authorized_by=authorized_by)
        return result.state

    @staticmethod
    def _validate_human_extension_fields(
        state: GraphState,
        node: ProjectNode,
        changes: dict[str, Any],
    ) -> None:
        values = changes.get("extension_fields")
        if not isinstance(values, dict):
            raise ValueError("Extension fields must be submitted as one complete object.")
        owner_types = {node.type}
        if node.extension_type is not None:
            owner_types.add(node.extension_type)
        definitions = {
            field.name: field for field in state.ontology.fields if field.owner_type in owner_types
        }
        missing = object()
        protected = sorted(
            name
            for name in set(node.extension_fields) | set(values)
            if (definition := definitions.get(name)) is None or definition.deprecated
            if node.extension_fields.get(name, missing) != values.get(name, missing)
        )
        if protected:
            raise ValueError(
                f"Extension fields for {node.id} are not active on its ontology type and "
                f"cannot be changed: {', '.join(protected)}."
            )

    def decide_proposal(
        self,
        proposal_id: str,
        request: ProposalDecisionRequest,
        *,
        authorized_by: AuthorizedHuman | None = None,
    ) -> GraphState:
        def build(state: GraphState) -> list[Patch]:
            proposal = state.proposals.get(proposal_id)
            if proposal is None:
                raise KeyError(proposal_id)
            if proposal.status != "pending":
                raise ValueError("proposal is not pending")
            return [
                _proposal_judgment_patch(
                    state,
                    proposal,
                    decision=request.decision,
                    reason=request.reason,
                )
            ]

        _, result = self.history.append_batch_from_state(
            build,
            authorized_by=authorized_by,
        )
        return result.state

    def resolve_agent_profile(
        self,
        surface: AgentExecutionProfile,
        *,
        provider: ProviderId | None = None,
        model: str | None = None,
        reasoning: str | None = None,
        run_on: str | None = None,
    ) -> AgentSurfaceConfig:
        base = self.manifest.agent_profile(surface)
        updates: dict[str, object] = {}
        if provider is not None:
            updates["provider"] = provider
            if provider != base.provider and model is None:
                updates["model"] = ""
            if provider != base.provider:
                updates["runtime"] = configured_runtime(provider, None)
        if model is not None:
            updates["model"] = model
        if reasoning is not None:
            updates["reasoning"] = reasoning
        if run_on is not None:
            if run_on not in self.manifest.machine_map:
                raise ValueError(f"unknown execution machine: {run_on}")
            state_machine = self.manifest.repository_map[self.manifest.state.repository].machine
            if surface != "paper_coach" and run_on != state_machine:
                raise ValueError(
                    f"{surface.replace('_', ' ')} must run on canonical state machine "
                    f"{state_machine!r}"
                )
            updates["run_on"] = run_on
        return base.model_copy(update=updates)

    def assemble_run(
        self,
        request: RunRequest,
        surface: AgentSurface = "refresh",
        *,
        imported_source_inventory: ImportedProviderSourceInventory | None = None,
    ) -> RunContext:
        materialization = self.history.current_materialization()
        state = self.history.require_writable(materialization.state)
        selected = request.run_truth_scope or self.manifest.agent.default_run_truth_scope
        selected_set = set(selected)
        project_scope = set(state.project_truth_scope or self.manifest.project.truth_scope)
        if not selected_set or not selected_set.issubset(project_scope):
            raise ValueError("run truth scope must be a non-empty subset of project truth scope")
        profile = self.resolve_agent_profile(
            surface,
            provider=request.provider,
            model=request.model,
            reasoning=request.reasoning,
            run_on=request.run_on,
        )
        execution_machine = self.manifest.machine_map[profile.run_on]
        repository_access = {
            alias: build_repository_access(
                self.manifest.repository_map[alias],
                self.manifest.machine_map[self.manifest.repository_map[alias].machine],
            )
            for alias in selected
            if alias in self.manifest.repository_map
        }
        assembler = ContextAssembler(self.manifest)
        source_roots = assembler.source_roots(execution_machine.alias)
        source_errors = preflight_provider_roots(source_roots, execution_machine)
        imported = imported_source_inventory or self.imported_source_inventory(
            surface,
            execution_machine,
        )
        imported_source_roots = (
            imported.roots(self.imported_sources.root)
            if self.imported_sources is not None and imported.files
            else {}
        )
        context = assembler.assemble(
            state,
            request.run_truth_scope,
            repository_access=repository_access,
            refresh_delta=(
                self.history.refresh_delta(materialization) if surface == "refresh" else None
            ),
            source_roots=source_roots,
            imported_source_roots=imported_source_roots,
            imported_source_fingerprint=(
                imported.fingerprint if imported is not None and imported.files else None
            ),
            source_errors=source_errors,
        )
        return context

    def imported_source_inventory(
        self,
        surface: AgentSurface,
        execution_machine: MachineConfig,
    ) -> ImportedProviderSourceInventory | None:
        if self.imported_sources is None or surface not in {"seed", "refresh"}:
            return None
        return self.imported_sources.inventory()

    def validate_imported_source_context(
        self,
        context: RunContext,
        inventory: ImportedProviderSourceInventory | None,
        *,
        expected_roots: dict[str, list[str]] | None = None,
    ) -> None:
        if expected_roots is None:
            expected_roots = (
                inventory.roots(self.imported_sources.root)
                if inventory is not None and inventory.files and self.imported_sources is not None
                else {}
            )
        expected_fingerprint = (
            inventory.fingerprint if inventory is not None and inventory.files else None
        )
        if (
            context.imported_source_roots != expected_roots
            or context.imported_source_fingerprint != expected_fingerprint
        ):
            raise ValueError("imported provider history changed after this context was prepared")

    def graph_task_contract(
        self,
        kind: str,
        *,
        project_name: str,
        ontology_path: str,
        ontology_extensions: bool,
        graph_path: str | None,
        research_path: str | None,
        provider_log_roots: dict[str, list[str]],
        ingestion_watermark: datetime | str | None,
        repositories: list[dict[str, str]],
        patch_path: str,
        output_schema_path: str,
        validator_command: str,
        human_request_path: str | None = None,
        retry_diagnostics_path: str | None = None,
        source_errors: list[str] | None = None,
        skill_pointers: list[dict[str, object]] | None = None,
    ) -> str:
        return PromptFactory.graph_task_contract(
            kind,
            project_name=project_name,
            ontology_path=ontology_path,
            ontology_extensions=ontology_extensions,
            graph_path=graph_path,
            research_path=research_path,
            provider_log_roots=provider_log_roots,
            ingestion_watermark=ingestion_watermark,
            repositories=repositories,
            patch_path=patch_path,
            output_schema_path=output_schema_path,
            human_request_path=human_request_path,
            retry_diagnostics_path=retry_diagnostics_path,
            source_errors=source_errors or [],
            validator_command=validator_command,
            skill_pointers=skill_pointers,
        )

    def assemble_chat(self, request: RunRequest) -> ChatContext:
        """Chat context: the graph and live pointers, never the ingest corpus."""

        state = self.history.state()
        selected = request.run_truth_scope or self.manifest.agent.default_run_truth_scope
        repository_access = {
            alias: build_repository_access(
                self.manifest.repository_map[alias],
                self.manifest.machine_map[self.manifest.repository_map[alias].machine],
            )
            for alias in selected
            if alias in self.manifest.repository_map
        }
        return ContextAssembler(self.manifest).chat_context(
            state,
            node_id=request.node_id if request.chat_scope == "node" else None,
            run_truth_scope=request.run_truth_scope,
            repository_access=repository_access,
        )

    def resolve_skill_selection(
        self,
        request: RunRequest | CoachRequest,
    ) -> SkillSelection:
        """Resolve the official packages enabled by the project Settings defaults.

        A request's recorded `resolved_skill_packages` is deliberately ignored:
        it is the receipt of an earlier attempt, and the registry is what says
        which version this launch gets.
        """

        defaults = self.manifest.agent.skill_defaults
        return official_registry().resolve(defaults=defaults)

    def resolve_skill_request(
        self,
        request: RunRequest | CoachRequest,
    ) -> RunRequest | CoachRequest:
        selection = self.resolve_skill_selection(request)
        available = {(item.kind, item.id) for item in selection.resolved_skill_packages}
        registry = official_registry()

        def validate_invocations(
            values: list[str], kind: Literal["workflow", "skill"]
        ) -> list[str]:
            normalized: list[str] = []
            seen: set[str] = set()
            for value in values:
                registry.package(kind, value)
                if value in seen:
                    continue
                if (kind, value) not in available:
                    raise ValueError(
                        f"invoked {kind} {value!r} is not enabled in project skill defaults"
                    )
                seen.add(value)
                normalized.append(value)
            return normalized

        invoked_workflow_ids = validate_invocations(request.invoked_workflow_ids, "workflow")
        invoked_skill_ids = validate_invocations(request.invoked_skill_ids, "skill")
        resolved_provider_skills: list[ProviderSkillReference] = []
        if request.invoked_provider_skill_names:
            if self.provider_skills is None:
                raise ValueError("provider-native skill inventory is unavailable")
            if request.provider is None or request.run_on is None:
                raise ValueError(
                    "provider-native skills require a resolved provider and execution machine"
                )
            try:
                machine = self.manifest.machine_map[request.run_on]
            except KeyError:
                raise ValueError(f"unknown execution machine: {request.run_on}") from None
            resolved_provider_skills = self.provider_skills.resolve(
                request.provider,
                machine.host,
                machine.provider_paths.get(request.provider),
                machine.alias,
                request.invoked_provider_skill_names,
            )
        return request.model_copy(
            update={
                "workflow_ids": selection.workflow_ids,
                "skill_ids": selection.skill_ids,
                "invoked_workflow_ids": invoked_workflow_ids,
                "invoked_skill_ids": invoked_skill_ids,
                "resolved_provider_skills": resolved_provider_skills,
                "resolved_skill_packages": selection.resolved_skill_packages,
            }
        )

    def coach_context(
        self, request: CoachRequest, draft_path: Path | None = None
    ) -> tuple[dict[str, object], list[Path]]:
        # Resolved for its validation of the requested provider/model/run_on.
        self.resolve_agent_profile(
            "paper_coach",
            provider=request.provider,
            model=request.model,
            reasoning=request.reasoning,
            run_on=request.run_on,
        )
        repository_access = {
            alias: build_repository_access(
                self.manifest.repository_map[alias],
                self.manifest.machine_map[self.manifest.repository_map[alias].machine],
            )
            for alias in self.manifest.project.truth_scope
        }
        pointers = ContextAssembler(self.manifest).paper_pointers(draft_path, repository_access)
        read_dirs = [
            Path(access.path)
            for access in repository_access.values()
            if not access.host and Path(access.path).exists()
        ]
        return pointers, read_dirs

    def pointer_hashes(self) -> tuple[str, int, str]:
        snapshot = self.paper.snapshot()
        intro_hash = (
            snapshot.canonical_hash or hashlib.sha256(snapshot.content.encode("utf-8")).hexdigest()
        )
        state = self.history.state()
        research_path = self.manifest.research_dir / "research.md"
        research = research_path.read_bytes() if research_path.exists() else b""
        return intro_hash, state.revision, hashlib.sha256(research).hexdigest()

    @staticmethod
    def parse_patch_output(chunks: list[str]) -> tuple[AgentPatch, str | None]:
        last_problem: ValueError | None = None
        for chunk in reversed(chunks):
            candidate = chunk.strip()
            try:
                return parse_agent_patch_json(candidate), None
            except ValueError as exc:
                last_problem = exc
                start = candidate.find("{")
                end = candidate.rfind("}")
                if start >= 0 and end > start:
                    try:
                        return parse_agent_patch_json(candidate[start : end + 1]), None
                    except ValueError as exc:
                        last_problem = exc
                        continue
        if last_problem is not None:
            raise last_problem
        raise ValueError("agent completed without a valid semantic Patch object")

    @staticmethod
    def _primary_question(state: GraphState):
        questions = [node for node in state.nodes.values() if node.type == "research_question"]
        questions.sort(
            key=lambda node: (
                {Standing.ACCEPTED: 0, Standing.ASSERTED: 1, Standing.CONTESTED: 2}[node.standing],
                node.id,
            )
        )
        return questions[0].model_dump(mode="json") if questions else None
