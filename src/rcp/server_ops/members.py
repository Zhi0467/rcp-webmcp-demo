"""Confirmed team-member removal through the running service owner."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from typing import BinaryIO, Protocol

from rcp.background import BackgroundAgentTasks
from rcp.runs.membership_fence import fence_episodes_for_removed_member
from rcp.server_ops.cli import CallerIdentity, PreparedServerCommand, ServerEventEmitter
from rcp.server_ops.control import (
    ServerControlClient,
    ServerControlError,
    ServerControlMemberAdvanceResult,
    ServerControlMemberPlanResult,
    ServerControlMemberSnapshot,
)
from rcp.server_ops.layout import DEFAULT_SERVER_LAYOUT, ServerLayout
from rcp.server_ops.models import (
    CommandAction,
    MachineTarget,
    NonsecretField,
    ServerCommandRequest,
    ServerPlanEvent,
    ServerStep,
    redact_server_text,
)
from rcp.server_runtime import ServerMetadata, data_dir_identity
from rcp.storage import AppStore, MemberRemovalPreviewRecord


class MemberRemovalRefused(ServerControlError):
    def __init__(self, message: str) -> None:
        super().__init__("operation_refused", message)


class MemberRemovalControl(Protocol):
    def member_removal_plan(self, member_id: str) -> ServerControlMemberPlanResult: ...

    def advance_member_removal(
        self,
        member_id: str,
        *,
        boundary_sha256: str,
    ) -> ServerControlMemberAdvanceResult: ...


ControlFactory = Callable[[ServerLayout], MemberRemovalControl]


def prepare_member_remove_command(
    request: ServerCommandRequest,
    identity: CallerIdentity,
    *,
    control_factory: ControlFactory | None = None,
    layout: ServerLayout = DEFAULT_SERVER_LAYOUT,
) -> PreparedServerCommand:
    if request.command != "server member remove" or request.member_id is None:
        raise ValueError("prepare_member_remove_command requires one member removal")
    client = (control_factory or _installed_control)(layout)
    planned = client.member_removal_plan(request.member_id)
    pending = planned.step
    plan = ServerPlanEvent(
        command=request.command,
        timestamp=datetime.now(UTC),
        steps=(pending,),
    )

    def execute(emitter: ServerEventEmitter, _input_stream: BinaryIO) -> None:
        snapshot = planned.snapshot
        blockers = _removal_blockers(snapshot)
        if snapshot.removal_started_at is None and blockers:
            emitter.emit_step(
                pending.model_copy(
                    update={
                        "state": "failed",
                        "message": "Member removal was refused. " + " ".join(blockers),
                    }
                )
            )
            return
        if snapshot.removal_started_at is None and request.member_confirmed_boundary is None:
            resume = _resume_argv(snapshot.member_id, snapshot.boundary_sha256)
            emitter.emit_step(
                pending.model_copy(
                    update={
                        "performed_by": "human",
                        "state": "operator_action_needed",
                        "message": (
                            "Review the exact inventory above, then run the displayed command "
                            "to confirm this member-removal boundary."
                        ),
                        "actions": (CommandAction(argv=resume),),
                        "resume_argv": resume,
                    }
                )
            )
            return
        confirmed = request.member_confirmed_boundary or snapshot.boundary_sha256
        emitter.emit_step(
            pending.model_copy(
                update={
                    "state": "running",
                    "message": "RCP is fencing access and stopping this member's live work.",
                }
            )
        )
        try:
            result = client.advance_member_removal(
                snapshot.member_id,
                boundary_sha256=confirmed,
            )
        except ServerControlError as exc:
            emitter.emit_step(
                pending.model_copy(
                    update={
                        "state": "failed",
                        "message": str(exc),
                    }
                )
            )
            return
        emitter.emit_step(result.step)

    return PreparedServerCommand(plan=plan, execute=execute)


def _installed_control(layout: ServerLayout) -> MemberRemovalControl:
    return ServerControlClient.from_data_dir(
        layout.data_dir,
        expected_server_uid=os.geteuid(),
    )


class MemberRemovalCoordinator:
    """Fence access once, then repeatedly press the existing graceful stop owners."""

    def __init__(
        self,
        store: AppStore,
        background: BackgroundAgentTasks,
        metadata: ServerMetadata,
    ) -> None:
        if store.space_kind != "team":
            raise ValueError("member removal requires a team space")
        if (
            metadata.owner_kind != "cli"
            or metadata.control_socket is None
            or metadata.data_dir_id != data_dir_identity(store.path.parent)
        ):
            raise ValueError("member removal requires this installed team service process")
        self.store = store
        self.background = background
        self.metadata = metadata

    def plan(self, member_id: str) -> ServerControlMemberPlanResult:
        snapshot = self._snapshot(self.store.member_removal_preview(member_id))
        return ServerControlMemberPlanResult(
            **self._identity_fields(),
            snapshot=snapshot,
            step=self._pending_step(snapshot),
        )

    def advance(
        self,
        member_id: str,
        *,
        boundary_sha256: str,
    ) -> ServerControlMemberAdvanceResult:
        before = self.store.member_removal_preview(member_id)
        if before.member.removed_at is None and before.member.removal_started_at is None:
            if before.boundary_sha256 != boundary_sha256:
                raise MemberRemovalRefused(
                    "The member-removal inventory changed after confirmation; rerun the command."
                )
            blockers = _removal_blockers(self._snapshot(before))
            if blockers:
                raise MemberRemovalRefused(" ".join(blockers))
            try:
                before = self.store.begin_member_removal(
                    member_id,
                    expected_boundary_sha256=boundary_sha256,
                )
            except ValueError as exc:
                raise MemberRemovalRefused(str(exc)) from exc
        errors: list[str] = []
        for operation_id in before.active_task_ids:
            try:
                self.background.request_member_removal_pause(operation_id)
            except (KeyError, RuntimeError, ValueError) as exc:
                errors.append(f"task {operation_id}: {type(exc).__name__}")
        try:
            fence_episodes_for_removed_member(self.store, self.background, member_id)
        except (KeyError, RuntimeError, ValueError) as exc:
            errors.append(f"episodes: {type(exc).__name__}")

        after = self.store.member_removal_preview(member_id)
        if not after.active_task_ids and not after.active_episode_ids:
            self.store.complete_member_removal(member_id)
            after = self.store.member_removal_preview(member_id)
        snapshot = self._snapshot(after)
        step = self._result_step(snapshot, errors=errors)
        return ServerControlMemberAdvanceResult(
            **self._identity_fields(),
            confirmed_boundary_sha256=boundary_sha256,
            snapshot=snapshot,
            step=step,
        )

    def reconcile_pending(self) -> tuple[ServerControlMemberAdvanceResult, ...]:
        results: list[ServerControlMemberAdvanceResult] = []
        for member in self.store.members_pending_removal():
            preview = self.store.member_removal_preview(member.user_id)
            results.append(
                self.advance(
                    member.user_id,
                    boundary_sha256=preview.boundary_sha256,
                )
            )
        return tuple(results)

    def pending_snapshots(self) -> tuple[ServerControlMemberSnapshot, ...]:
        return tuple(
            self._snapshot(self.store.member_removal_preview(member.user_id))
            for member in self.store.members_pending_removal()
        )

    def _snapshot(self, preview: MemberRemovalPreviewRecord) -> ServerControlMemberSnapshot:
        labels: list[str] = []
        for project_id in preview.orphaned_project_ids:
            project = self.store.project(project_id)
            name = project.name if project is not None else "Unknown project"
            labels.append(redact_server_text(f"{name} ({project_id})"))
        return ServerControlMemberSnapshot(
            member_id=preview.member.user_id,
            member_display_name=(
                redact_server_text(preview.member.display_name)
                if preview.member.display_name is not None
                else None
            ),
            removal_started_at=preview.member.removal_started_at,
            removed_at=preview.member.removed_at,
            last_authenticating_member=preview.last_authenticating_member,
            project_ids=preview.project_ids,
            orphaned_project_ids=preview.orphaned_project_ids,
            orphaned_project_labels=tuple(labels),
            active_task_ids=preview.active_task_ids,
            active_episode_ids=preview.active_episode_ids,
            active_token_ids=preview.active_token_ids,
            browser_session_count=preview.browser_session_count,
            space_invitation_ids=preview.space_invitation_ids,
            project_invitation_ids=preview.project_invitation_ids,
            boundary_sha256=preview.boundary_sha256,
        )

    def _pending_step(self, snapshot: ServerControlMemberSnapshot) -> ServerStep:
        return ServerStep(
            number=1,
            title="Remove team member",
            purpose=(
                "Confirm one exact consequence set, fence access atomically, and drain the "
                "member's already-authorized work without killing an in-flight provider turn."
            ),
            performed_by="system",
            target=MachineTarget(host=self.metadata.host, os_account="rcp"),
            phase="member_removal",
            state="pending",
            expected_success="The member is tombstoned and no authorized work remains live.",
            message="RCP will verify and reconcile this exact member-removal inventory.",
            fields=_snapshot_fields(snapshot),
        )

    def _result_step(
        self,
        snapshot: ServerControlMemberSnapshot,
        *,
        errors: list[str],
    ) -> ServerStep:
        pending = self._pending_step(snapshot)
        if snapshot.removed_at is not None:
            return pending.model_copy(
                update={
                    "state": "succeeded",
                    "message": "Member removal completed; historical attribution remains.",
                }
            )
        resume = ("rcp", "server", "member", "remove", snapshot.member_id)
        diagnostic = f" Stop attempts reported {len(errors)} bounded error(s)." if errors else ""
        return pending.model_copy(
            update={
                "performed_by": "human",
                "state": "operator_action_needed",
                "message": (
                    "Access is fenced. The exact tasks or episodes shown above are still "
                    f"settling.{diagnostic} Rerun the command to reconcile and read back."
                ),
                "actions": (CommandAction(argv=resume),),
                "resume_argv": resume,
            }
        )

    def _identity_fields(self) -> dict[str, object]:
        return {
            "instance_id": self.metadata.instance_id,
            "pid": self.metadata.pid,
            "data_dir_id": self.metadata.data_dir_id,
            "space_id": self.store.space_id,
        }


def _removal_blockers(snapshot: ServerControlMemberSnapshot) -> tuple[str, ...]:
    blockers: list[str] = []
    if snapshot.last_authenticating_member:
        blockers.append(
            "This is the last enrolled member who can authenticate; enroll another member first."
        )
    if snapshot.orphaned_project_ids:
        blockers.append(
            "This member is the only authenticating member of the named project(s); add another "
            "enrolled project member first."
        )
    return tuple(blockers)


def _snapshot_fields(snapshot: ServerControlMemberSnapshot) -> tuple[NonsecretField, ...]:
    state = (
        "removed"
        if snapshot.removed_at is not None
        else "access_fenced"
        if snapshot.removal_started_at is not None
        else "active"
    )
    fields = [
        NonsecretField(name="member_id", value=snapshot.member_id),
        NonsecretField(name="member_name", value=snapshot.member_display_name or "unnamed"),
        NonsecretField(name="removal_state", value=state),
        NonsecretField(name="project_ids", value=_inventory(snapshot.project_ids)),
        NonsecretField(name="active_task_ids", value=_inventory(snapshot.active_task_ids)),
        NonsecretField(name="active_episode_ids", value=_inventory(snapshot.active_episode_ids)),
        NonsecretField(name="permanent_access_ids", value=_inventory(snapshot.active_token_ids)),
        NonsecretField(name="browser_login_count", value=snapshot.browser_session_count),
        NonsecretField(
            name="space_invitation_ids",
            value=_inventory(snapshot.space_invitation_ids),
        ),
        NonsecretField(
            name="project_invitation_ids",
            value=_inventory(snapshot.project_invitation_ids),
        ),
        NonsecretField(name="boundary_sha256", value=snapshot.boundary_sha256),
    ]
    fields.extend(
        NonsecretField(name=f"orphaned_project_{index}", value=label)
        for index, label in enumerate(snapshot.orphaned_project_labels, start=1)
    )
    return tuple(fields)


def _inventory(values: tuple[str, ...]) -> str:
    return json.dumps(values, ensure_ascii=False, separators=(",", ":")) if values else "none"


def _resume_argv(member_id: str, boundary_sha256: str) -> tuple[str, ...]:
    return (
        "rcp",
        "server",
        "member",
        "remove",
        member_id,
        "--confirm-boundary",
        boundary_sha256,
    )


__all__ = ["MemberRemovalCoordinator", "prepare_member_remove_command"]
