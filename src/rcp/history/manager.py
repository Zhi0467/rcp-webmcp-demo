from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import stat
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from rcp.config import (
    AgentExecutionProfile,
    AgentSurfaceConfig,
    Manifest,
    load_manifest,
    validate_project_scope_update,
    write_agent_settings,
    write_machine_provider_paths,
    write_project_scope,
)
from rcp.core.authority import AgentTaskAuthority, ProjectMembershipCheck, require_apply
from rcp.core.materialize import (
    AcceptedPatchObserver,
    MaterializationResult,
    apply_valid_patch,
    finalize_patch_bookkeeping,
    materialize_patches,
    prepare_patch_bookkeeping,
)
from rcp.core.models import (
    AuthorizedHuman,
    BranchMergeReceipt,
    GraphBranchMetadata,
    GraphState,
    Patch,
    ProjectHomeTransfer,
    ProjectIdentity,
    ReplayFailure,
)
from rcp.core.operations import SetProjectTruthScopeOperation, adapt_persisted_patch_document
from rcp.core.research_md import render_research_md
from rcp.core.transition_models import GraphHeadRef, GraphTargetRef, TransitionTrace
from rcp.core.transitions import (
    GraphTransitionManager,
    PreparedTransition,
    TransitionConflict,
    accepted_transition_head_chain_failure,
    project_transition_projection,
)
from rcp.core.validation import ValidationReport, validate_patch
from rcp.history.delta import (
    RefreshDelta,
    RevisionSummary,
    build_refresh_delta,
    render_revision_summary,
)
from rcp.limits import PROJECT_TRANSFER_INVENTORY_MAX_ENTRIES
from rcp.providers import ProviderId
from rcp.skill_registry import SkillDefaults
from rcp.transport import (
    BatchPublishFailed,
    LocalStateWorkspace,
    StateUnavailable,
    StateWorkspace,
)

if TYPE_CHECKING:
    from rcp.history.branches import BranchHistoryManager, BranchReadSnapshot


@dataclass(frozen=True)
class CanonicalFactSource:
    """One safe opaque fact and its observed transfer boundary."""

    path: Path
    relative_path: PurePosixPath
    observed_size: int
    device: int
    inode: int
    modified_ns: int
    changed_ns: int
    root_device: int
    root_inode: int


def canonical_fact_sources(root: Path) -> tuple[CanonicalFactSource, ...]:
    """Inventory only bounded regular files below the canonical facts owner."""

    facts_root = root / "facts"
    try:
        root_descriptor = os.open(
            facts_root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except FileNotFoundError:
        return ()
    except OSError as exc:
        raise ValueError("The facts directory is unavailable.") from exc
    try:
        root_metadata = os.fstat(root_descriptor)
        if not stat.S_ISDIR(root_metadata.st_mode):
            raise ValueError("The facts path is not a safe directory.")
        files: list[CanonicalFactSource] = []
        observed_entries = [0]
        _inventory_canonical_facts(
            root_descriptor,
            facts_root=facts_root,
            relative_root=PurePosixPath(),
            root_metadata=root_metadata,
            files=files,
            observed_entries=observed_entries,
        )
        return tuple(sorted(files, key=lambda source: source.relative_path.as_posix()))
    finally:
        os.close(root_descriptor)


def _inventory_canonical_facts(
    directory_descriptor: int,
    *,
    facts_root: Path,
    relative_root: PurePosixPath,
    root_metadata: os.stat_result,
    files: list[CanonicalFactSource],
    observed_entries: list[int],
) -> None:
    try:
        names = sorted(os.listdir(directory_descriptor))
    except OSError as exc:
        raise ValueError("The facts directory cannot be enumerated.") from exc
    for name in names:
        observed_entries[0] += 1
        if observed_entries[0] > PROJECT_TRANSFER_INVENTORY_MAX_ENTRIES:
            raise ValueError("The facts inventory exceeds its entry bound.")
        relative = relative_root / name
        if {".git", "credentials"}.intersection(relative.parts):
            raise ValueError("The facts tree contains a forbidden path.")
        try:
            item = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        except OSError as exc:
            raise ValueError("A facts entry cannot be inspected.") from exc
        if stat.S_ISDIR(item.st_mode):
            try:
                child_descriptor = os.open(
                    name,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_descriptor,
                )
            except OSError as exc:
                raise ValueError("A facts directory changed during inspection.") from exc
            try:
                opened = os.fstat(child_descriptor)
                if not stat.S_ISDIR(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
                    item.st_dev,
                    item.st_ino,
                ):
                    raise ValueError("A facts directory changed during inspection.")
                _inventory_canonical_facts(
                    child_descriptor,
                    facts_root=facts_root,
                    relative_root=relative,
                    root_metadata=root_metadata,
                    files=files,
                    observed_entries=observed_entries,
                )
            finally:
                os.close(child_descriptor)
        elif stat.S_ISREG(item.st_mode):
            files.append(
                CanonicalFactSource(
                    path=facts_root.joinpath(*relative.parts),
                    relative_path=relative,
                    observed_size=item.st_size,
                    device=item.st_dev,
                    inode=item.st_ino,
                    modified_ns=item.st_mtime_ns,
                    changed_ns=item.st_ctime_ns,
                    root_device=root_metadata.st_dev,
                    root_inode=root_metadata.st_ino,
                )
            )
        else:
            raise ValueError("The facts tree contains an unsafe entry.")


def iter_canonical_fact_bytes(
    root: Path,
    source: CanonicalFactSource,
    *,
    chunk_size: int,
) -> Iterator[bytes]:
    """Read one inventoried fact through a no-follow descriptor chain."""

    relative = source.relative_path
    if (
        chunk_size < 1
        or relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or source.path != (root / "facts").joinpath(*relative.parts)
    ):
        raise ValueError("The canonical fact source is invalid.")
    descriptors: list[int] = []
    try:
        directory_descriptor = os.open(
            root / "facts",
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        descriptors.append(directory_descriptor)
        root_metadata = os.fstat(directory_descriptor)
        if not stat.S_ISDIR(root_metadata.st_mode) or (
            root_metadata.st_dev,
            root_metadata.st_ino,
        ) != (source.root_device, source.root_inode):
            raise ValueError("The facts directory changed before transfer.")
        for part in relative.parts[:-1]:
            child_descriptor = os.open(
                part,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_descriptor,
            )
            descriptors.append(child_descriptor)
            directory_descriptor = child_descriptor
            if not stat.S_ISDIR(os.fstat(directory_descriptor).st_mode):
                raise ValueError("A facts directory changed before transfer.")
        file_descriptor = os.open(
            relative.parts[-1],
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_descriptor,
        )
        descriptors.append(file_descriptor)
        initial = os.fstat(file_descriptor)
        expected = (
            source.device,
            source.inode,
            source.observed_size,
            source.modified_ns,
            source.changed_ns,
        )
        if (
            not stat.S_ISREG(initial.st_mode)
            or (
                initial.st_dev,
                initial.st_ino,
                initial.st_size,
                initial.st_mtime_ns,
                initial.st_ctime_ns,
            )
            != expected
        ):
            raise ValueError("A fact changed before its transfer read.")
        while True:
            chunk = os.read(file_descriptor, chunk_size)
            if not chunk:
                break
            yield chunk
        final = os.fstat(file_descriptor)
        if (
            final.st_dev,
            final.st_ino,
            final.st_size,
            final.st_mtime_ns,
            final.st_ctime_ns,
        ) != expected:
            raise ValueError("A fact changed during its transfer read.")
    except OSError as exc:
        raise ValueError("The canonical fact became unavailable during transfer.") from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


class RevisionConflict(ValueError):
    """A patch was written against a graph revision that is no longer current."""


class BranchMergeAlreadyCommitted(RuntimeError):
    """The same stable branch head already has an accepted main merge."""

    def __init__(self, patch: Patch, materialization: MaterializationResult) -> None:
        self.patch = patch
        self.materialization = materialization
        assert patch.branch_merge is not None
        super().__init__(
            "graph branch merge "
            f"{patch.branch_merge.merge_id} already committed at revision {patch.revision}"
        )


class BranchMergeAlreadyResolved(RuntimeError):
    """The source branch head already has a canonical no-change receipt."""

    def __init__(self, receipt: BranchMergeReceipt) -> None:
        self.receipt = receipt
        super().__init__(
            "graph branch merge "
            f"{receipt.provenance.merge_id} already resolved without a main Patch"
        )


class ReplayHalted(RuntimeError):
    """Canonical history is structurally invalid and therefore read-only."""

    def __init__(self, state: GraphState) -> None:
        failure = state.replay_failure
        if failure is None:
            message = (
                f"Canonical replay is degraded at coherent revision {state.revision}; "
                "history must be repaired before making canonical changes."
            )
            self.failed_revision = None
            self.code = "replay-halted"
        else:
            message = (
                f"Canonical replay halted at revision {failure.revision} "
                f"({failure.code}): {failure.message} The graph is read-only at "
                f"coherent revision {state.revision} until history is repaired."
            )
            self.failed_revision = failure.revision
            self.code = failure.code
        self.coherent_revision = state.revision
        super().__init__(message)


class PatchRejected(ValueError):
    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        super().__init__(
            "; ".join(item.message for item in report.messages if item.level == "reject")
        )


class ProjectIdentityConflict(ValueError):
    """Canonical project identity is missing, conflicting, or owned elsewhere."""


def _is_exact_prepared_identity_prefix(
    materialization: MaterializationResult,
    identity: ProjectIdentity,
) -> bool:
    if len(materialization.patches) != 1:
        return False
    patch = materialization.patches[0]
    return bool(
        patch.revision == 1
        and patch.kind == "identity"
        and patch.admission == "accepted"
        and patch.author is None
        and patch.producer == "system"
        and patch.ops == []
        and patch.project_identity == identity
    )


def _patch_failure_created_at(path: Path) -> datetime:
    """Retain useful failure chronology even when the Patch schema cannot load."""

    with suppress(OSError, TypeError, ValueError):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and isinstance(raw.get("created_at"), str):
            created_at = datetime.fromisoformat(raw["created_at"])
            return created_at if created_at.tzinfo is not None else created_at.replace(tzinfo=UTC)
    with suppress(OSError):
        return datetime.fromtimestamp(path.stat().st_mtime, UTC)
    return datetime.fromtimestamp(0, UTC)


def _first_replay_revision(paths: list[Path]) -> int:
    return int(paths[0].stem) if paths else 0


def _first_replay_created_at(
    paths: list[Path],
    patches: list[Patch],
    *,
    fallback: Path | None = None,
) -> datetime:
    if patches:
        return patches[0].created_at
    if paths:
        return _patch_failure_created_at(paths[0])
    return (
        _patch_failure_created_at(fallback)
        if fallback is not None
        else datetime.fromtimestamp(0, UTC)
    )


class HistoryManager:
    def __init__(
        self,
        manifest: Manifest,
        workspace: StateWorkspace | None = None,
        *,
        expected_space_id: str | None = None,
        project_id: str | None = None,
        require_attribution: bool = False,
        agent_authority_resolver: Callable[[str, str], AgentTaskAuthority] | None = None,
        project_membership_check: ProjectMembershipCheck | None = None,
    ) -> None:
        if expected_space_id is not None:
            parsed = uuid.UUID(expected_space_id)
            if str(parsed) != expected_space_id or parsed.version != 4:
                raise ValueError("expected_space_id must be a canonical UUIDv4")
        if project_id is not None and not project_id.strip():
            raise ValueError("project_id must be non-empty when supplied")
        self.manifest = manifest
        self.workspace = workspace or LocalStateWorkspace(
            manifest.research_dir, str(manifest.research_dir)
        )
        self.root = self.workspace.root
        self.patches_dir = self.root / "patches"
        self._process_lock = self.workspace.snapshot_lock
        self._accepted_revision: int | None = None
        self.expected_space_id = expected_space_id
        self.project_id = project_id
        self.require_attribution = require_attribution
        self.agent_authority_resolver = agent_authority_resolver
        self.project_membership_check = project_membership_check
        self._branch_materialization_repairs: set[str] = set()

    def initialize(self) -> MaterializationResult:
        with self._process_lock:
            self._reload_manifest()
            coherent = self._coherent_materialization()
            if coherent is not None:
                self._require_writable_home_locked(coherent)
                self._remember_accepted_revision(coherent)
                return coherent

        publishing = False
        try:
            with self.workspace.transaction(), self._append_lock():
                self._reload_manifest()
                coherent = self._coherent_materialization()
                if coherent is not None:
                    self._require_writable_home_locked(coherent)
                    self._remember_accepted_revision(coherent)
                    return coherent
                current = self.materialize(write_outputs=False)
                self.require_writable(current.state)
                self._require_writable_home_locked(current)
                self.ensure_layout()
                result = self.materialize()
                self._synchronize_manifest_from_history(result)
                publishing = True
                self.workspace.publish(self._materialized_paths(include_manifest=True))
                self.workspace.complete_materialization_repair()
                self._remember_accepted_revision(result)
                return result
        except StateUnavailable:
            if publishing and self.workspace.remote:
                self.workspace.require_materialization_repair()
            if not (self.root / "manifest.toml").is_file():
                raise
            with self._append_lock():
                self._reload_manifest()
                current = self.materialize(write_outputs=False)
                self.require_writable(current.state)
                self._require_writable_home_locked(current)
                self.ensure_layout()
                result = self.materialize()
                self._synchronize_manifest_from_history(result)
                self._remember_accepted_revision(result)
                return result

    def restore_canonical_history(
        self,
        sources: Mapping[str, tuple[Path, str, int]],
        *,
        expected_main_head: GraphHeadRef,
        expected_branch_heads: tuple[GraphHeadRef, ...],
    ) -> MaterializationResult:
        """Publish an archived append-only history, replay it, and prove every head."""

        if expected_main_head.target.kind != "main" or "manifest.toml" not in sources:
            raise ValueError("restored history requires one exact main head and manifest")
        main_revisions: list[int] = []
        branch_metadata: set[str] = set()
        branch_revisions: dict[str, list[int]] = {}
        branch_merges: set[str] = set()
        ordered: list[tuple[tuple[object, ...], str]] = []
        for value in sources:
            relative = Path(value)
            parts = relative.parts
            if relative == Path("manifest.toml"):
                key: tuple[object, ...] = (0, value)
            elif relative == Path("scope-base.json"):
                key = (1, value)
            elif (
                len(parts) == 2
                and parts[0] == "patches"
                and re.fullmatch(r"[0-9]{6}\.json", parts[1]) is not None
            ) or (
                len(parts) == 3
                and parts[0] == "patches"
                and parts[1].startswith("batch-")
                and re.fullmatch(r"[0-9]{6}\.json", parts[2]) is not None
            ):
                revision = int(relative.stem)
                main_revisions.append(revision)
                key = (2, revision, value)
            elif len(parts) == 3 and parts[0] == "branches" and parts[2] == "branch.json":
                branch_id = parts[1]
                parsed_branch = uuid.UUID(branch_id)
                if str(parsed_branch) != branch_id or parsed_branch.version != 4:
                    raise ValueError("restored branch identity is not canonical")
                branch_metadata.add(branch_id)
                key = (3, branch_id, 0, value)
            elif (
                len(parts) == 4
                and parts[0] == "branches"
                and parts[2] == "patches"
                and re.fullmatch(r"[0-9]{6}\.json", parts[3]) is not None
            ):
                branch_id = parts[1]
                revision = int(relative.stem)
                branch_revisions.setdefault(branch_id, []).append(revision)
                key = (3, branch_id, 1, revision, value)
            elif (
                len(parts) == 4
                and parts[0] == "branches"
                and parts[2] == "merges"
                and re.fullmatch(r"[0-9a-f]{64}\.json", parts[3]) is not None
            ):
                branch_merges.add(parts[1])
                key = (3, parts[1], 2, value)
            else:
                raise ValueError(f"restored canonical history path is unsupported: {value}")
            ordered.append((key, value))
        if sorted(main_revisions) != list(range(1, expected_main_head.revision + 1)):
            raise ValueError("restored main Patch inventory does not match its captured head")
        expected_branches = {
            head.target.branch_id: head
            for head in expected_branch_heads
            if head.target.kind == "branch" and head.target.branch_id is not None
        }
        if len(expected_branches) != len(expected_branch_heads) or set(expected_branches) != (
            branch_metadata
        ):
            raise ValueError("restored branch metadata does not match its captured heads")
        if not set(branch_revisions).union(branch_merges).issubset(branch_metadata):
            raise ValueError("restored branch history requires its captured metadata")
        for branch_id, head in expected_branches.items():
            revisions = branch_revisions.get(branch_id, [])
            if revisions and max(revisions) != head.revision:
                raise ValueError("restored branch Patch inventory does not match its captured head")

        for _key, value in sorted(ordered):
            source, sha256, size = sources[value]
            self.workspace.restore_exact_file(
                value,
                source,
                expected_sha256=sha256,
                expected_size=size,
            )
        self._reload_manifest()
        result = self.initialize()
        if self.head_ref(result) != expected_main_head:
            raise ValueError("restored main history does not replay to its captured head")
        for branch_id, expected in sorted(expected_branches.items()):
            branch = self.branch(
                branch_id,
                expected_project_id=self.project_id,
            )
            branch_result = branch.initialize()
            if branch.head_ref(branch_result) != expected:
                raise ValueError("restored branch history does not replay to its captured head")
            branch.validated_merge_receipts()
        return result

    def ensure_layout(self) -> None:
        for path in (
            self.patches_dir,
            self.root / "chat",
            self.root / "paper",
            self.root / "facts",
        ):
            path.mkdir(parents=True, exist_ok=True)
        defaults = {
            "graph.json": GraphState(
                project_truth_scope=self.manifest.project.truth_scope
            ).model_dump(mode="json"),
            "glossary.json": {},
            "proposals.json": {},
            "coverage.json": GraphState().coverage.model_dump(mode="json"),
            "cursors.json": {},
            "scope-base.json": {
                "truth_scope": self.manifest.project.truth_scope,
                "repository_aliases": sorted(self.manifest.repository_map),
            },
        }
        for name, value in defaults.items():
            path = self.root / name
            if not path.exists():
                self._atomic_json(path, value)
        research_md = self.root / "research.md"
        if not research_md.exists():
            self._atomic_text(research_md, "")

    def load_patches(self) -> list[Patch]:
        with self._process_lock:
            return [
                self._decode_persisted_patch(path.read_text(encoding="utf-8"))
                for path in self._patch_paths()
            ]

    @staticmethod
    def _decode_persisted_patch(payload: str | bytes) -> Patch:
        """Decode append-only history while keeping legacy lineage out of live parsing."""

        document = json.loads(payload)
        if isinstance(document, dict):
            document = adapt_persisted_patch_document(document)
        return Patch.model_validate(document)

    def project_identity(
        self,
        materialization: MaterializationResult | None = None,
    ) -> ProjectIdentity | None:
        """Return the unique accepted nameplate without adding it to graph state."""

        if materialization is not None:
            return self._project_identity_from_replay(materialization)
        with self._process_lock:
            with suppress(StateUnavailable):
                self.workspace.refresh_if_stale()
            self._reload_manifest()
            result = self.materialize(write_outputs=False)
            return self._project_identity_from_replay(result)

    def project_home_space_id_at_revision(
        self,
        revision: int,
        materialization: MaterializationResult | None = None,
    ) -> str | None:
        """Return the canonical project home at one retained main revision."""

        result = materialization or self.current_materialization()
        if revision < 0 or revision > result.state.revision:
            raise ValueError("project home revision is outside retained main history")
        identity = self._project_identity_from_replay(
            result,
            through_revision=revision,
        )
        return identity.home_space_id if identity is not None else None

    def claim_project_identity(
        self,
        action: str,
        *,
        project_id: str | None = None,
    ) -> ProjectIdentity:
        """Idempotently claim an untagged project for this manager's space.

        Ordinary setup leaves ``project_id`` unset and mints one identity here.
        A prepared team-project request has already reserved its identity, so its
        finalizer supplies that exact id and this append owner refuses any other
        retained nameplate.
        """

        if action not in {"created", "adopted"}:
            raise ValueError("project identity action must be 'created' or 'adopted'")
        if self.expected_space_id is None:
            raise ValueError("claiming project identity requires expected_space_id")
        reserved = (
            ProjectIdentity(
                project_id=project_id,
                home_space_id=self.expected_space_id,
                action=action,
            )
            if project_id is not None
            else None
        )

        with self.workspace.transaction(), self._append_lock():
            self._reload_manifest()
            current = self.materialize(write_outputs=False)
            self.require_writable(current.state)
            existing = self._project_identity_from_replay(current)
            if existing is not None:
                self._require_expected_home(existing)
                if reserved is not None and (
                    existing != reserved
                    or not _is_exact_prepared_identity_prefix(current, reserved)
                ):
                    raise ProjectIdentityConflict(
                        "The retained project identity does not match the prepared project."
                    )
                return existing
            if reserved is not None and current.patches:
                raise ProjectIdentityConflict(
                    "The prepared project acquired Patch history before its identity claim."
                )

            identity = reserved or ProjectIdentity(
                project_id=str(uuid.uuid4()),
                home_space_id=self.expected_space_id,
                action=action,
            )
            patch = Patch(
                kind="identity",
                author=None,
                producer="system",
                summary=(
                    "Project created." if action == "created" else "Project identity adopted."
                ),
                ops=[],
                project_identity=identity,
            )
            appended, _result = self._append_locked(
                patch,
                discard_on_reject=True,
                allow_identity_claim=True,
            )
            assert appended.project_identity is not None
            return appended.project_identity

    def transfer_project_home(
        self,
        *,
        project_id: str,
        previous_home_space_id: str,
        new_home_space_id: str,
        source_released_by: AuthorizedHuman,
        target_admitted_by: AuthorizedHuman,
    ) -> ProjectHomeTransfer:
        """Append one exact home change after both spaces recorded human authority."""

        transfer = ProjectHomeTransfer(
            project_id=project_id,
            previous_home_space_id=previous_home_space_id,
            new_home_space_id=new_home_space_id,
            source_released_by=source_released_by,
            target_admitted_by=target_admitted_by,
        )
        if self.expected_space_id != transfer.previous_home_space_id:
            raise ProjectIdentityConflict(
                "Only the project's current source space may append its home transfer."
            )

        with self.workspace.transaction(), self._append_lock():
            self._reload_manifest()
            current = self.materialize(write_outputs=False)
            self.require_writable(current.state)
            identity = self._project_identity_from_replay(current)
            if identity is None:
                raise ProjectIdentityConflict(
                    "Canonical project identity must exist before its home can transfer."
                )
            if identity.project_id != transfer.project_id:
                raise ProjectIdentityConflict(
                    "The home transfer names a different canonical project identity."
                )
            if identity.home_space_id == transfer.new_home_space_id:
                existing = next(
                    (
                        patch.project_home_transfer
                        for patch in reversed(current.patches)
                        if patch.admission == "accepted" and patch.project_home_transfer == transfer
                    ),
                    None,
                )
                if existing is not None:
                    return existing
            if identity.home_space_id != transfer.previous_home_space_id:
                raise ProjectIdentityConflict(
                    "The home transfer does not continue from the project's current home."
                )
            self._require_expected_home(identity)
            patch = Patch(
                kind="identity",
                author=None,
                producer="system",
                summary="Project moved to its admitted team space.",
                ops=[],
                project_home_transfer=transfer,
            )
            appended, _result = self._append_locked(
                patch,
                discard_on_reject=True,
            )
            assert appended.project_home_transfer is not None
            return appended.project_home_transfer

    def current_accepted_revision(self) -> int:
        """Return the cached accepted revision without reading canonical patch bodies."""

        with self._process_lock:
            if self._accepted_revision is None:
                # Project services call ``initialize`` before exposure. This fallback
                # keeps a directly constructed HistoryManager correct without making
                # the steady-state API probe replay or read patch files.
                self._remember_accepted_revision(self.materialize(write_outputs=False))
            assert self._accepted_revision is not None
            return self._accepted_revision

    def state(self) -> GraphState:
        return self.current_materialization().state

    @property
    def graph_target(self) -> GraphTargetRef:
        return GraphTargetRef()

    def head_ref(self, materialization: MaterializationResult | None = None) -> GraphHeadRef:
        """Return the exact current main head, including transition-chain identity."""

        result = materialization or self.current_materialization()
        if result.state.replay_status != "complete":
            raise ReplayHalted(result.state)
        failure = accepted_transition_head_chain_failure(
            result.patches,
            target=self.graph_target,
            initial_transition_id=None,
        )
        if failure is not None:
            raise ValueError(failure.message)
        transition_id: str | None = None
        for patch in result.patches:
            if patch.admission != "accepted" or patch.transition is None:
                continue
            transition_id = patch.transition.transition_id
        return GraphHeadRef(
            target=self.graph_target,
            revision=result.state.revision,
            transition_id=transition_id,
        )

    def create_auto_research_branch(
        self,
        metadata: GraphBranchMetadata,
    ) -> BranchHistoryManager:
        """Create one episode branch at the exact current accepted main head."""

        from rcp.history.branches import create_auto_research_branch

        return create_auto_research_branch(self, metadata)

    def branch(
        self,
        branch_id: str,
        *,
        expected_episode_id: str | None = None,
        expected_project_id: str | None = None,
    ) -> BranchHistoryManager:
        """Open and verify one canonical episode branch."""

        from rcp.history.branches import open_branch

        return open_branch(
            self,
            branch_id,
            expected_episode_id=expected_episode_id,
            expected_project_id=expected_project_id,
        )

    def branch_metadata(
        self,
        branch_id: str,
        *,
        expected_episode_id: str | None = None,
        expected_project_id: str | None = None,
    ) -> GraphBranchMetadata:
        return self.branch(
            branch_id,
            expected_episode_id=expected_episode_id,
            expected_project_id=expected_project_id,
        ).branch_metadata()

    def branch_read_snapshots(
        self,
        identities: list[tuple[str, str, str]],
    ) -> dict[str, BranchReadSnapshot | None]:
        """Read several episode branches from one refreshed, read-only snapshot."""

        from rcp.history.branches import read_branch_snapshots

        return read_branch_snapshots(self, identities)

    def _require_branch_materialization_repair(self, branch_id: str) -> None:
        self._branch_materialization_repairs.add(branch_id)

    def _branch_materialization_repair_required(self, branch_id: str) -> bool:
        return branch_id in self._branch_materialization_repairs

    def _complete_branch_materialization_repair(self, branch_id: str) -> None:
        self._branch_materialization_repairs.discard(branch_id)

    def merge_receipts(self, branch_id: str) -> list[BranchMergeReceipt]:
        return self.branch(branch_id).merge_receipts()

    def write_merge_receipt(self, receipt: BranchMergeReceipt) -> BranchMergeReceipt:
        return self.branch(receipt.provenance.branch_id).write_merge_receipt(receipt)

    def reconcile_branch_merge_receipt(
        self,
        branch_id: str,
        merge_id: str,
    ) -> BranchMergeReceipt | None:
        return self.branch(branch_id).reconcile_merge_receipt(merge_id)

    def current_materialization(self) -> MaterializationResult:
        """Replay current canonical history once and return state plus reports."""

        with self._process_lock:
            self._repair_materializations_if_needed()
            with suppress(StateUnavailable):
                self.workspace.refresh_if_stale()
            self._reload_manifest()
            result = self.materialize(write_outputs=False)
            self._remember_accepted_revision(result)
            return result

    def require_writable(self, state: GraphState | None = None) -> GraphState:
        """Return the coherent state, or refuse a canonical mutation after replay halts."""

        current = state or self.current_materialization().state
        if current.replay_status == "degraded":
            raise ReplayHalted(current)
        return current

    def append(
        self,
        patch: Patch,
        *,
        raise_on_reject: bool = True,
        discard_on_reject: bool = False,
        expected_revision: int | None = None,
        authorized_by: AuthorizedHuman | None = None,
    ) -> tuple[Patch, MaterializationResult]:
        """Append a patch to the log and rematerialize.

        A rejected patch is still written to the append-only log. Callers that
        want to inspect the report themselves pass ``raise_on_reject=False``.
        Agent workflows that must correct an invalid deliverable before it
        enters canonical history pass ``discard_on_reject=True``; validation
        still happens under the append lock, but the rejected candidate is not
        written and does not consume a revision.

        ``expected_revision`` refuses a patch written against state that has since
        moved. The comparison happens under the append lock, which is the same
        lock every other writer takes, so nothing can land between the check and
        the write — a freshness check made outside this lock cannot say that.
        """
        with self.workspace.transaction(), self._append_lock():
            return self._append_locked(
                patch,
                raise_on_reject=raise_on_reject,
                discard_on_reject=discard_on_reject,
                expected_revision=expected_revision,
                authorized_by=authorized_by,
            )

    def _append_locked(
        self,
        patch: Patch,
        *,
        raise_on_reject: bool = True,
        discard_on_reject: bool = False,
        expected_revision: int | None = None,
        allow_identity_claim: bool = False,
        authorized_by: AuthorizedHuman | None = None,
    ) -> tuple[Patch, MaterializationResult]:
        """Use the single-patch commit path while the canonical append lock is held."""

        self._reload_manifest()
        current = self.materialize(write_outputs=False)
        self.require_writable(current.state)
        self._require_writable_home_locked(
            current,
            identity_claim=patch.project_identity if allow_identity_claim else None,
        )
        self.ensure_layout()
        if self.workspace.materialization_repair_required:
            self._repair_materializations_locked()
            current = self.materialize(write_outputs=False)
        if patch.branch_merge is not None:
            matching_merges = [
                existing
                for existing in current.patches
                if existing.admission == "accepted"
                and existing.branch_merge is not None
                and existing.branch_merge.merge_id == patch.branch_merge.merge_id
            ]
            if len(matching_merges) > 1:
                raise ValueError("main history contains duplicate branch merge provenance")
            if matching_merges:
                raise BranchMergeAlreadyCommitted(matching_merges[0], current)
            from rcp.history.branches import existing_receipt_for_main_append

            receipt = existing_receipt_for_main_append(self, current, patch.branch_merge)
            if receipt is not None:
                raise BranchMergeAlreadyResolved(receipt)
        if expected_revision is not None and current.state.revision != expected_revision:
            raise RevisionConflict(
                f"the graph moved from revision {expected_revision} to "
                f"{current.state.revision} while this patch was being written"
            )
        revision = self._next_revision()
        patch, report, _preflight_state = self._validate_candidate_locked(
            current,
            patch,
            revision,
            authorized_by=authorized_by,
        )
        if discard_on_reject and report.rejected:
            raise PatchRejected(report)
        patch = patch.model_copy(
            update={
                "admission": "rejected" if report.rejected else "accepted",
                "admission_messages": list(report.messages),
            }
        )
        target = self.patches_dir / f"{revision:06d}.json"
        manifest_path = self.root / "manifest.toml"
        manifest_before = manifest_path.read_text(encoding="utf-8")
        self._atomic_text(target, patch.model_dump_json(indent=2) + "\n")
        result = self.materialize(write_outputs=True)
        scope_changed = False
        if not result.reports[revision].rejected:
            scope_changed = self._synchronize_manifest_scope(result, patch)
        paths = [
            target.relative_to(self.root),
            *self._materialized_paths(include_manifest=True),
        ]
        try:
            self.workspace.publish_committed_patch(
                paths,
                target.relative_to(self.root),
            )
        except Exception as exc:
            if not self.workspace.remote:
                raise
            if not self._reconcile_remote_publish_failure(
                exc,
                target,
                scope_changed=scope_changed,
                manifest_before=manifest_before,
            ):
                raise
        self._remember_accepted_revision(result)
        if raise_on_reject and result.reports[revision].rejected:
            raise PatchRejected(result.reports[revision])
        return patch, result

    def validate_candidate(
        self,
        patch: Patch,
        *,
        authorized_by: AuthorizedHuman | None = None,
    ) -> tuple[Patch, ValidationReport, GraphState]:
        """Validate without writing, against canonical state held under the append lock."""

        with self.workspace.transaction(), self._append_lock():
            self._reload_manifest()
            current = self.materialize(write_outputs=False)
            self.require_writable(current.state)
            self._require_writable_home_locked(current)
            self.ensure_layout()
            if self.workspace.materialization_repair_required:
                self._repair_materializations_locked()
                current = self.materialize(write_outputs=False)
            prepared, report, _candidate = self._validate_candidate_locked(
                current,
                patch,
                self._next_revision(),
                authorized_by=authorized_by,
            )
            return prepared, report, current.state

    def preview_batch_from_state(
        self,
        build_patches: Callable[[GraphState], list[Patch]],
        *,
        expected_revision: int,
        authorized_by: AuthorizedHuman | None = None,
    ) -> PreparedTransition:
        """Prepare a complete non-canonical transition without appending history."""

        with self.workspace.transaction(), self._append_lock():
            self._reload_manifest()
            current = self.materialize(write_outputs=False)
            self.require_writable(current.state)
            self._require_writable_home_locked(current)
            if current.state.revision != expected_revision:
                raise RevisionConflict(
                    f"the graph moved from revision {expected_revision} to "
                    f"{current.state.revision} while this draft was being previewed"
                )
            patches = build_patches(current.state)
            if not patches:
                raise ValueError("the staged draft has no semantic graph change to preview")
            patch, report, candidate = self._prepare_transition_locked(
                current,
                patches,
                self._next_revision(),
                authorized_by=authorized_by,
            )
            if report.rejected or candidate is None:
                raise PatchRejected(report)
            assert patch.transition is not None
            return PreparedTransition(
                patch=patch,
                projection=project_transition_projection(
                    candidate,
                    patch.transition,
                    canonical=False,
                ),
            )

    def transition_events_after(self, revision: int) -> list[dict[str, object]]:
        """Read canonical transition events after a consumer watermark."""

        result = self.current_materialization()
        events: list[dict[str, object]] = []
        for patch in result.patches:
            if patch.revision <= revision or patch.admission != "accepted":
                continue
            if patch.transition is None:
                continue
            head = {
                "target": patch.transition.pre_head.target.model_dump(mode="json"),
                "revision": patch.revision,
                "transition_id": patch.transition.transition_id,
            }
            for event in patch.transition.lifecycle_events:
                events.append(
                    {
                        "head": head,
                        "event": event.model_dump(mode="json"),
                    }
                )
        return events

    def transition_trace_at_revision(self, revision: int) -> TransitionTrace | None:
        """Read one accepted transition trace without replaying rule code."""

        for path in self._patch_paths():
            if int(path.stem) != revision:
                continue
            patch = self._decode_persisted_patch(path.read_text(encoding="utf-8"))
            return patch.transition if patch.admission == "accepted" else None
        return None

    def _validate_candidate_locked(
        self,
        current: MaterializationResult,
        patch: Patch,
        revision: int,
        *,
        authorized_by: AuthorizedHuman | None = None,
    ) -> tuple[Patch, ValidationReport, GraphState | None]:
        if patch.kind == "identity" and not patch.ops:
            identity_patch = self._stamp_attribution_for_admission(
                patch,
                authorized_by=authorized_by,
                apply_target=self.graph_target,
            ).model_copy(update={"revision": revision})
            report = validate_patch(
                current.state,
                identity_patch,
                current.state.project_truth_scope,
                repository_aliases=self.manifest.repository_map,
                machine_aliases=self.manifest.machine_map,
                default_run_truth_scope=self.manifest.agent.default_run_truth_scope,
                state_repository=self.manifest.state.repository,
            )
            candidate = (
                apply_valid_patch(current.state, identity_patch) if not report.rejected else None
            )
            return identity_patch, report, candidate
        prepared, report, state = self._prepare_transition_locked(
            current,
            [patch],
            revision,
            authorized_by=authorized_by,
        )
        return prepared, report, state

    def _prepare_transition_locked(
        self,
        current: MaterializationResult,
        raw_patches: list[Patch],
        revision: int,
        *,
        authorized_by: AuthorizedHuman | None,
        pre_head: GraphHeadRef | None = None,
        apply_target: GraphTargetRef | None = None,
    ) -> tuple[Patch, ValidationReport, GraphState | None]:
        """Validate initiating groups, then prepare one expanded transition Patch."""

        target = apply_target or self.graph_target
        if pre_head is not None and pre_head.target != target:
            raise ValueError("transition pre-head does not match its Apply graph target")
        aggregate = ValidationReport()
        staged = current.state
        prepared_sources: list[Patch] = []
        for raw_patch in raw_patches:
            patch = self._stamp_attribution_for_admission(
                raw_patch,
                authorized_by=authorized_by,
                apply_target=target,
            ).model_copy(update={"revision": revision})
            validation_state = staged.model_copy(update={"revision": current.state.revision})
            patch = prepare_patch_bookkeeping(validation_state, patch)
            report = validate_patch(
                validation_state,
                patch,
                validation_state.project_truth_scope,
                repository_aliases=self.manifest.repository_map,
                machine_aliases=self.manifest.machine_map,
                default_run_truth_scope=self.manifest.agent.default_run_truth_scope,
                state_repository=self.manifest.state.repository,
            )
            aggregate.messages.extend(report.messages)
            if report.rejected:
                return patch, aggregate, None
            try:
                candidate = apply_valid_patch(validation_state, patch)
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                aggregate.reject(
                    "malformed-operation",
                    f"Patch operations could not be applied atomically: {exc}.",
                    revision,
                )
                return patch, aggregate, None
            if candidate.project_truth_scope != validation_state.project_truth_scope:
                descriptor = next(
                    (
                        op.repository.model_dump(mode="python")
                        for op in patch.ops
                        if isinstance(op, SetProjectTruthScopeOperation)
                        and op.repository is not None
                    ),
                    None,
                )
                try:
                    validate_project_scope_update(
                        self.manifest,
                        candidate.project_truth_scope,
                        descriptor,
                    )
                except ValueError as exc:
                    aggregate.reject("invalid-project-scope", str(exc), revision)
                    return patch, aggregate, None
            patch = finalize_patch_bookkeeping(patch, candidate)
            prepared_sources.append(patch)
            staged = candidate.model_copy(update={"revision": current.state.revision})

        if pre_head is None:
            previous_transition_id = next(
                (
                    item.transition.transition_id
                    for item in reversed(current.patches)
                    if item.admission == "accepted" and item.transition is not None
                ),
                None,
            )
            pre_head = GraphHeadRef(
                revision=current.state.revision,
                transition_id=previous_transition_id,
            )
        try:
            prepared = GraphTransitionManager().prepare_validated(
                current.state,
                prepared_sources,
                pre_head=pre_head,
            )
        except TransitionConflict as exc:
            for detail in exc.details:
                aggregate.reject(
                    "transition-conflict",
                    detail.message,
                    revision,
                    related_node_ids=detail.affected_ids,
                    operation_index=detail.operation_index,
                    rule_id=detail.rule_id,
                    cause_chain=[item.model_dump(mode="json") for item in detail.cause_chain],
                    failed_invariant=detail.invariant,
                )
            return prepared_sources[0], aggregate, None
        transition_patch = prepared.patch.model_copy(
            update={"admission_messages": list(aggregate.messages)}
        )
        return transition_patch, aggregate, prepared.projection.graph

    def _stamp_attribution_for_admission(
        self,
        patch: Patch,
        *,
        authorized_by: AuthorizedHuman | None,
        apply_target: GraphTargetRef,
    ) -> Patch:
        if not self.require_attribution:
            return patch

        if patch.kind == "identity":
            if authorized_by is not None or any(
                value is not None
                for value in (
                    patch.authorized_by,
                    patch.profile,
                    patch.task_id,
                    patch.episode_id,
                )
            ):
                raise ValueError("identity patches are system-owned and cannot carry attribution")
            return patch

        if patch.kind == "approval":
            if authorized_by is None:
                raise ValueError(
                    "human approval patches require an explicit authorized_by snapshot"
                )
            authorizer = self._canonical_authorizer(authorized_by)
            return patch.model_copy(
                update={
                    "authorized_by": authorizer,
                    "profile": None,
                    "task_id": None,
                    "episode_id": None,
                }
            )

        if authorized_by is not None:
            raise ValueError("explicit authorized_by is only valid for human approval patches")
        operation_id = patch.source_operation_id
        if not operation_id or not operation_id.strip():
            raise ValueError("agent patches require a non-empty direct source_operation_id")
        if (
            self.project_id is None
            or self.agent_authority_resolver is None
            or self.project_membership_check is None
        ):
            raise ValueError(
                "agent attribution requires a canonical project-scoped agent_authority_resolver "
                "and project_membership_check"
            )
        try:
            resolved = self.agent_authority_resolver(self.project_id, operation_id)
        except KeyError as exc:
            raise ValueError(f"unknown agent task {operation_id!r}") from exc
        if resolved.operation_id != operation_id or resolved.project_id != self.project_id:
            raise ValueError("agent authority resolver returned another task or project")
        if resolved.apply_target != apply_target:
            raise ValueError(
                f"agent task {operation_id!r} is authorized to Apply to "
                f"{resolved.apply_target.key}, not {apply_target.key}"
            )
        dispatch = require_apply(resolved, patch, is_project_member=self.project_membership_check)
        if dispatch.profile == "orchestrator" and resolved.episode_id is None:
            raise ValueError("orchestrator agent tasks require a canonical episode_id")
        assert resolved.authorized_by is not None
        authorizer = self._canonical_authorizer(resolved.authorized_by)
        canonical = (authorizer, dispatch.profile, operation_id, resolved.episode_id)
        supplied = (patch.authorized_by, patch.profile, patch.task_id, patch.episode_id)
        if any(value is not None for value in supplied) and supplied != canonical:
            raise ValueError(
                "agent patch attribution does not match the canonical task attribution"
            )
        return patch.model_copy(
            update={
                "authorized_by": authorizer,
                "profile": dispatch.profile,
                "task_id": operation_id,
                "episode_id": resolved.episode_id,
            }
        )

    @staticmethod
    def _canonical_authorizer(authorizer: AuthorizedHuman) -> AuthorizedHuman:
        try:
            snapshot = AuthorizedHuman.model_validate(authorizer.model_dump(mode="python"))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("authorized_by must be a valid authorizer snapshot") from exc
        if not snapshot.display_name.strip():
            raise ValueError("authorized_by must name the authorizing human")
        return snapshot

    def _require_no_change_merge_authority(self, receipt: BranchMergeReceipt) -> None:
        """Apply the live task and membership gate before admitting a no-change receipt."""

        if not self.require_attribution:
            return
        operation_id = receipt.provenance.merge_task_id
        if (
            self.project_id is None
            or self.agent_authority_resolver is None
            or self.project_membership_check is None
        ):
            raise ValueError(
                "branch merge receipt attribution requires a canonical project-scoped "
                "agent_authority_resolver and project_membership_check"
            )
        try:
            resolved = self.agent_authority_resolver(self.project_id, operation_id)
        except KeyError as exc:
            raise ValueError(f"unknown agent task {operation_id!r}") from exc
        if resolved.operation_id != operation_id or resolved.project_id != self.project_id:
            raise ValueError("agent authority resolver returned another task or project")
        if resolved.apply_target != GraphTargetRef():
            raise ValueError("a branch merge receipt requires main-target Apply authority")
        dispatch = resolved.dispatch_authority
        if dispatch is None:
            raise ValueError("branch merge receipt task has no dispatch authority binding")
        probe = Patch(
            kind="work",
            author="agent",
            producer="agent",
            summary="Validate no-change branch merge receipt authority.",
            source_operation_id=operation_id,
            run_truth_scope=list(dispatch.scope.run_truth_scope),
            repositories_read=[],
            profile="orchestrator",
            task_id=operation_id,
            episode_id=receipt.provenance.episode_id,
            authorized_by=receipt.authorized_by,
            ops=[],
        )
        bound = require_apply(
            resolved,
            probe,
            is_project_member=self.project_membership_check,
        )
        if bound.profile != "orchestrator":
            raise ValueError("a branch merge receipt requires orchestrator authority")
        if resolved.episode_id != receipt.provenance.episode_id:
            raise ValueError("branch merge receipt episode does not match its task authority")
        if resolved.authorized_by is None:
            raise ValueError("branch merge receipt task has no authorizer snapshot")
        if self._canonical_authorizer(resolved.authorized_by) != receipt.authorized_by:
            raise ValueError("branch merge receipt authorizer does not match its task authority")

    def update_agent_settings(
        self,
        default_run_truth_scope: list[str],
        profiles: dict[AgentExecutionProfile, AgentSurfaceConfig],
        provider_path_updates: dict[str, dict[ProviderId, str]] | None = None,
        skill_defaults: SkillDefaults | None = None,
        default_auto_research_invocation_ceiling: int | None = None,
    ) -> Manifest:
        with self.workspace.transaction(), self._append_lock():
            self._reload_manifest()
            current = self.materialize(write_outputs=False)
            self.require_writable(current.state)
            self._require_writable_home_locked(current)
            self._repair_materializations_locked()
            self.manifest = write_agent_settings(
                self.manifest,
                default_run_truth_scope,
                profiles,
                provider_path_updates,
                skill_defaults,
                default_auto_research_invocation_ceiling,
            )
            self.workspace.publish([Path("manifest.toml")])
        return self.manifest

    def update_machine_provider_paths(
        self,
        provider_path_updates: dict[str, dict[ProviderId, str]],
    ) -> Manifest:
        with self.workspace.transaction(), self._append_lock():
            self._reload_manifest()
            current = self.materialize(write_outputs=False)
            self.require_writable(current.state)
            self._require_writable_home_locked(current)
            self._repair_materializations_locked()
            self.manifest = write_machine_provider_paths(
                self.manifest,
                provider_path_updates,
            )
            self.workspace.publish([Path("manifest.toml")])
        return self.manifest

    def append_batch(
        self,
        patches: list[Patch],
        *,
        expected_revision: int | None = None,
        authorized_by: AuthorizedHuman | None = None,
    ) -> tuple[list[Patch], MaterializationResult]:
        """Append a validated human transaction and publish materializations once."""

        if not patches:
            return [], self.materialize(write_outputs=False)
        return self.append_batch_from_state(
            lambda _state: patches,
            expected_revision=expected_revision,
            authorized_by=authorized_by,
        )

    def append_batch_from_state(
        self,
        build_patches: Callable[[GraphState], list[Patch]],
        *,
        expected_revision: int | None = None,
        authorized_by: AuthorizedHuman | None = None,
    ) -> tuple[list[Patch], MaterializationResult]:
        """Build and append a human transaction from the fresh, append-locked state."""

        with self.workspace.transaction(), self._append_lock():
            self._reload_manifest()
            current = self.materialize(write_outputs=False)
            self.require_writable(current.state)
            self._require_writable_home_locked(current)
            self.ensure_layout()
            if self.workspace.materialization_repair_required:
                self._repair_materializations_locked()
                current = self.materialize(write_outputs=False)
            if expected_revision is not None and current.state.revision != expected_revision:
                raise ValueError(
                    "The graph changed after this draft began; reload before syncing it."
                )
            patches = build_patches(current.state)
            if not patches:
                return [], current
            revision = self._next_revision()
            prepared, report, _candidate = self._prepare_transition_locked(
                current,
                patches,
                revision,
                authorized_by=authorized_by,
            )
            if report.rejected:
                raise PatchRejected(report)
            prepared = prepared.model_copy(
                update={
                    "admission": "accepted",
                    "admission_messages": list(report.messages),
                }
            )
            target = self.patches_dir / f"{revision:06d}.json"
            manifest_path = self.root / "manifest.toml"
            manifest_before = manifest_path.read_text(encoding="utf-8")
            self._atomic_text(target, prepared.model_dump_json(indent=2) + "\n")
            result = self.materialize(write_outputs=True)
            scope_changed = self._synchronize_manifest_scope(result, prepared)
            paths = [
                target.relative_to(self.root),
                *self._materialized_paths(include_manifest=True),
            ]
            try:
                self.workspace.publish_committed_patch(
                    paths,
                    target.relative_to(self.root),
                )
            except Exception as exc:
                if not self.workspace.remote:
                    raise
                if not self._reconcile_remote_publish_failure(
                    exc,
                    target,
                    scope_changed=scope_changed,
                    manifest_before=manifest_before,
                ):
                    raise
            self._remember_accepted_revision(result)
            return [prepared], result

    def materialize(
        self,
        *,
        write_outputs: bool = True,
        pending_patch_paths: list[Path] | None = None,
        accepted_patch_observer: AcceptedPatchObserver | None = None,
    ) -> MaterializationResult:
        with self._process_lock:
            if accepted_patch_observer is None:
                result = self._replay(pending_patch_paths)
            else:
                result = self._replay(
                    pending_patch_paths,
                    accepted_patch_observer=accepted_patch_observer,
                )
            if write_outputs and result.state.replay_status == "complete":
                self.ensure_layout()
                self._write_materialized_outputs(result)
            return result

    def accepted_boundary_states(self) -> tuple[MaterializationResult, list[GraphState]]:
        """Replay canonical history and retain each accepted state in revision order."""

        result, boundaries = self.accepted_patch_boundaries()
        return result, [state for _previous, _patch, state in boundaries]

    def accepted_patch_boundaries(
        self,
    ) -> tuple[MaterializationResult, list[tuple[GraphState, Patch, GraphState]]]:
        """Replay canonical history with each accepted Patch and its exact pre/post state."""

        boundaries: list[tuple[GraphState, Patch, GraphState]] = []

        def collect(previous: GraphState, patch: Patch, state: GraphState) -> None:
            boundaries.append((previous, patch, state))

        with self._process_lock:
            # Graph-condition transitions require a proven current canonical
            # snapshot. Unlike display-oriented reads, stale remote state must
            # fail closed and be retried rather than evaluated as current.
            if not self.workspace.refresh_if_stale():
                raise StateUnavailable("canonical state refresh did not confirm a current snapshot")
            self._reload_manifest()
            result = self.materialize(
                write_outputs=False,
                accepted_patch_observer=collect,
            )
            self._remember_accepted_revision(result)
            return result, boundaries

    def _coherent_materialization(self) -> MaterializationResult | None:
        """Return replayed state only when every cached derived output already matches."""

        if self.workspace.materialization_repair_required:
            return None
        required = self._materialized_paths(include_manifest=True)
        if not all((self.root / path).is_file() for path in required):
            return None
        result = self._replay()
        if result.state.project_truth_scope != self.manifest.project.truth_scope:
            return None
        expected_json = {
            "graph.json": result.state.model_dump(mode="json"),
            "glossary.json": {
                key: value.model_dump(mode="json") for key, value in result.state.glossary.items()
            },
            "proposals.json": {
                key: value.model_dump(mode="json") for key, value in result.state.proposals.items()
            },
            "coverage.json": result.state.coverage.model_dump(mode="json"),
            "cursors.json": result.processed_cursors,
        }
        try:
            for name, expected in expected_json.items():
                if json.loads((self.root / name).read_text(encoding="utf-8")) != expected:
                    return None
            if (self.root / "research.md").read_text(encoding="utf-8") != render_research_md(
                result.state
            ):
                return None
        except (OSError, ValueError):
            return None
        return result

    def _replay(
        self,
        pending_patch_paths: list[Path] | None = None,
        *,
        accepted_patch_observer: AcceptedPatchObserver | None = None,
    ) -> MaterializationResult:
        pending = pending_patch_paths or []
        patch_paths = sorted(
            [*self._patch_paths(), *pending],
            key=lambda path: int(path.stem),
        )
        patches: list[Patch] = []
        structural_failure: ReplayFailure | None = None
        for path in patch_paths:
            try:
                patches.append(self._decode_persisted_patch(path.read_text(encoding="utf-8")))
            except OSError as exc:
                structural_failure = ReplayFailure(
                    revision=int(path.stem),
                    created_at=_patch_failure_created_at(path),
                    code="patch-read-failed",
                    message=str(exc),
                )
                break
            except ValueError as exc:
                structural_failure = ReplayFailure(
                    revision=int(path.stem),
                    created_at=_patch_failure_created_at(path),
                    code="patch-schema-invalid",
                    message=str(exc),
                )
                break
        scope_base_path = self.root / "scope-base.json"
        scope_failure: ReplayFailure | None = None
        if scope_base_path.is_file():
            try:
                initial_truth_scope = json.loads(scope_base_path.read_text(encoding="utf-8"))[
                    "truth_scope"
                ]
                if not isinstance(initial_truth_scope, list) or not all(
                    isinstance(alias, str) for alias in initial_truth_scope
                ):
                    raise ValueError("truth_scope must be a list of repository aliases")
            except (OSError, KeyError, TypeError, ValueError) as exc:
                initial_truth_scope = self.manifest.project.truth_scope
                scope_failure = ReplayFailure(
                    revision=_first_replay_revision(patch_paths),
                    created_at=_first_replay_created_at(
                        patch_paths,
                        patches,
                        fallback=scope_base_path,
                    ),
                    code="scope-provenance-invalid",
                    message=f"Canonical scope provenance {scope_base_path} is invalid: {exc}",
                )
        elif patch_paths:
            initial_truth_scope = self.manifest.project.truth_scope
            scope_failure = ReplayFailure(
                revision=_first_replay_revision(patch_paths),
                created_at=_first_replay_created_at(patch_paths, patches),
                code="scope-provenance-missing",
                message=(
                    f"Canonical scope provenance {scope_base_path} is absent while Patch history "
                    "exists; replay is refused rather than substituting the current manifest scope."
                ),
            )
        else:
            initial_truth_scope = self.manifest.project.truth_scope
        chain_failure = accepted_transition_head_chain_failure(
            patches,
            target=self.graph_target,
            initial_transition_id=None,
        )
        chain_prefix = patches
        if chain_failure is not None:
            chain_prefix = []
            for patch in patches:
                if patch.revision == chain_failure.revision:
                    break
                chain_prefix.append(patch)
        replayable_patches = [] if scope_failure is not None else chain_prefix
        result = materialize_patches(
            replayable_patches,
            initial_truth_scope=list(initial_truth_scope),
            repository_aliases=sorted(self.manifest.repository_map),
            machine_aliases=sorted(self.manifest.machine_map),
            default_run_truth_scope=list(self.manifest.agent.default_run_truth_scope),
            state_repository=self.manifest.state.repository,
            accepted_patch_observer=accepted_patch_observer,
        )
        structural_or_chain_failure = min(
            (item for item in (structural_failure, chain_failure) if item is not None),
            key=lambda item: item.revision,
            default=None,
        )
        failure = scope_failure or structural_or_chain_failure
        if failure is not None and result.state.replay_status == "complete":
            result.state = result.state.model_copy(
                update={
                    "replay_status": "degraded",
                    "replay_failure": failure,
                }
            )
        return result

    def _write_materialized_outputs(self, result: MaterializationResult) -> None:
        self._atomic_json(self.root / "graph.json", result.state.model_dump(mode="json"))
        self._atomic_json(
            self.root / "glossary.json",
            {key: value.model_dump(mode="json") for key, value in result.state.glossary.items()},
        )
        self._atomic_json(
            self.root / "proposals.json",
            {key: value.model_dump(mode="json") for key, value in result.state.proposals.items()},
        )
        self._atomic_json(
            self.root / "coverage.json", result.state.coverage.model_dump(mode="json")
        )
        self._atomic_json(self.root / "cursors.json", result.processed_cursors)
        self._atomic_text(self.root / "research.md", render_research_md(result.state))

    def refresh_delta(
        self,
        materialization: MaterializationResult | None = None,
    ) -> RefreshDelta:
        """Return the bounded delta without coupling context assembly to history I/O."""

        with self._process_lock:
            result = materialization or self.materialize(write_outputs=False)
            return build_refresh_delta(result.patches, result)

    def slice(self, from_revision: int, to_revision: int | None = None) -> list[dict[str, object]]:
        with self._process_lock:
            with suppress(StateUnavailable):
                self.workspace.refresh_if_stale()
            end = to_revision if to_revision is not None else 10**12
            materialization = self.materialize(write_outputs=False)
            return [
                {
                    "revision": patch.revision,
                    "kind": patch.kind,
                    "created_at": patch.created_at.isoformat(),
                    "summary": patch.summary,
                    "change_summary": patch.change_summary,
                }
                for patch in materialization.patches
                if from_revision <= patch.revision <= end
            ]

    def revision_summaries(
        self,
        from_revision: int = 1,
        to_revision: int | None = None,
    ) -> list[dict[str, object]]:
        """Return a reader-facing projection without changing the raw history contract."""

        with self._process_lock:
            with suppress(StateUnavailable):
                self.workspace.refresh_if_stale()
            end = to_revision if to_revision is not None else 10**12
            summaries: list[RevisionSummary] = []

            def collect(previous_state: GraphState, patch: Patch, state: GraphState) -> None:
                if from_revision <= patch.revision <= end:
                    summaries.append(render_revision_summary(previous_state, patch, state))

            self.materialize(
                write_outputs=False,
                accepted_patch_observer=collect,
            )
            return [item.model_dump(mode="json") for item in summaries]

    def _next_revision(self) -> int:
        paths = self._patch_paths()
        return int(paths[-1].stem) + 1 if paths else 1

    def _project_identity_from_replay(
        self,
        result: MaterializationResult,
        *,
        through_revision: int | None = None,
    ) -> ProjectIdentity | None:
        initial_identity: ProjectIdentity | None = None
        current_identity: ProjectIdentity | None = None
        for patch in result.patches:
            if through_revision is not None and patch.revision > through_revision:
                break
            report = result.reports.get(patch.revision)
            if patch.kind != "identity" or report is None or report.rejected:
                continue
            if patch.project_identity is not None:
                candidate = patch.project_identity
                if initial_identity is None:
                    initial_identity = candidate
                    current_identity = candidate
                    continue
                if candidate != initial_identity:
                    raise ProjectIdentityConflict(
                        "Canonical history contains conflicting project identity revisions "
                        f"({initial_identity.project_id} in "
                        f"{initial_identity.home_space_id} and "
                        f"{candidate.project_id} in {candidate.home_space_id}); it is read-only "
                        "until the history is repaired."
                    )
                continue
            transfer = patch.project_home_transfer
            if transfer is None:
                continue
            if current_identity is None:
                raise ProjectIdentityConflict(
                    "Canonical history changes project home before establishing its identity; "
                    "it is read-only until the history is repaired."
                )
            if (
                transfer.project_id != current_identity.project_id
                or transfer.previous_home_space_id != current_identity.home_space_id
            ):
                raise ProjectIdentityConflict(
                    "Canonical project home-transfer history does not continue from its current "
                    "identity and home; it is read-only until the history is repaired."
                )
            current_identity = current_identity.model_copy(
                update={"home_space_id": transfer.new_home_space_id}
            )
        return current_identity

    def _require_expected_home(self, identity: ProjectIdentity) -> None:
        if self.expected_space_id is not None and identity.home_space_id != self.expected_space_id:
            raise ProjectIdentityConflict(
                f"Project {identity.project_id} belongs to space {identity.home_space_id}; "
                f"this space is {self.expected_space_id}. Canonical writes are refused."
            )

    def _require_writable_home_locked(
        self,
        result: MaterializationResult,
        *,
        identity_claim: ProjectIdentity | None = None,
    ) -> None:
        if self.expected_space_id is None:
            return
        identity = self._project_identity_from_replay(result)
        if identity is not None:
            self._require_expected_home(identity)
            if identity_claim is not None:
                raise ProjectIdentityConflict(
                    f"Project {identity.project_id} already has a canonical identity."
                )
            return
        if identity_claim is not None:
            self._require_expected_home(identity_claim)
            return
        raise ProjectIdentityConflict(
            "Canonical project identity must be claimed before this space can write."
        )

    def _remember_accepted_revision(self, result: MaterializationResult) -> None:
        self._accepted_revision = max(
            (revision for revision, report in result.reports.items() if not report.rejected),
            default=0,
        )

    def _patch_paths(self) -> list[Path]:
        flat = list(self.patches_dir.glob("[0-9][0-9][0-9][0-9][0-9][0-9].json"))
        batched = [
            path
            for directory in self.patches_dir.glob("batch-*")
            if directory.is_dir()
            for path in directory.glob("[0-9][0-9][0-9][0-9][0-9][0-9].json")
        ]
        return sorted([*flat, *batched], key=lambda path: int(path.stem))

    def _synchronize_manifest_scope(self, result: MaterializationResult, patch: Patch) -> bool:
        if result.state.project_truth_scope == self.manifest.project.truth_scope:
            return False
        descriptor = next(
            (
                op.repository.model_dump(mode="python")
                for op in patch.ops
                if isinstance(op, SetProjectTruthScopeOperation) and op.repository is not None
            ),
            None,
        )
        self.manifest = write_project_scope(
            self.manifest,
            result.state.project_truth_scope,
            repository_descriptor=descriptor,
        )
        return True

    def _synchronize_manifest_from_history(self, result: MaterializationResult) -> bool:
        """Repair a manifest that lagged a committed truth-scope patch."""

        if result.state.project_truth_scope == self.manifest.project.truth_scope:
            return False
        descriptor = None
        for patch in reversed(result.patches):
            operation = next(
                (op for op in reversed(patch.ops) if isinstance(op, SetProjectTruthScopeOperation)),
                None,
            )
            if operation is not None:
                descriptor = (
                    operation.repository.model_dump(mode="python")
                    if operation.repository is not None
                    else None
                )
                break
        self.manifest = write_project_scope(
            self.manifest,
            result.state.project_truth_scope,
            repository_descriptor=descriptor,
        )
        return True

    def _repair_materializations_if_needed(self) -> None:
        if not self.workspace.materialization_repair_required:
            return
        with self.workspace.transaction(), self._append_lock():
            self._reload_manifest()
            current = self.materialize(write_outputs=False)
            self.require_writable(current.state)
            self._require_writable_home_locked(current)
            self.ensure_layout()
            self._repair_materializations_locked()

    def _repair_materializations_locked(self) -> None:
        if not self.workspace.materialization_repair_required:
            return
        result = self.materialize(write_outputs=True)
        self._synchronize_manifest_from_history(result)
        self.workspace.publish(self._materialized_paths(include_manifest=True))
        self.workspace.complete_materialization_repair()

    def _reconcile_remote_publish_failure(
        self,
        exc: Exception,
        committed: Path,
        *,
        scope_changed: bool,
        manifest_before: str,
    ) -> bool:
        """Reconcile the local mirror with an observed remote commit point.

        ``True`` means the history commit is confirmed and the caller should
        report success. ``False`` means the local copy has been removed from
        replay and the original publish error must be raised.
        """

        commit_status = exc.commit_status if isinstance(exc, BatchPublishFailed) else "unknown"
        if commit_status == "present":
            self.workspace.require_materialization_repair()
            return True

        if committed.exists():
            if commit_status == "unknown":
                self._quarantine_local_commit(committed)
                # A later refresh may prove that the remote commit landed. If it
                # did, derived outputs still need the same repair as a confirmed
                # post-commit failure.
                self.workspace.require_materialization_repair()
            else:
                try:
                    if committed.is_dir():
                        shutil.rmtree(committed)
                    else:
                        committed.unlink()
                except OSError:
                    # Replay safety matters more than deleting this non-canonical
                    # mirror copy immediately.
                    self._quarantine_local_commit(committed)
        self._fsync_directory(self.patches_dir)
        if scope_changed:
            self._atomic_text(self.root / "manifest.toml", manifest_before)
            self._reload_manifest()
        self.materialize(write_outputs=True)
        return False

    def _quarantine_local_commit(self, committed: Path) -> None:
        quarantine = self.patches_dir / (f".unconfirmed-{committed.name}-{uuid.uuid4().hex}")
        os.replace(committed, quarantine)

    def _reload_manifest(self) -> None:
        path = self.root / "manifest.toml"
        if path.is_file():
            self.manifest = load_manifest(path)

    @staticmethod
    def _materialized_paths(*, include_manifest: bool = False) -> list[Path]:
        paths = [
            Path("graph.json"),
            Path("glossary.json"),
            Path("proposals.json"),
            Path("coverage.json"),
            Path("cursors.json"),
            Path("scope-base.json"),
            Path("research.md"),
        ]
        if include_manifest:
            paths.append(Path("manifest.toml"))
        return paths

    @contextmanager
    def _append_lock(self) -> Iterator[None]:
        lock_path = self.root / ".append.lock"
        with self._process_lock, lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _atomic_json(path: Path, value: object) -> None:
        HistoryManager._atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")

    @staticmethod
    def _atomic_text(path: Path, content: str) -> None:
        temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temp.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        HistoryManager._fsync_directory(path.parent)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
