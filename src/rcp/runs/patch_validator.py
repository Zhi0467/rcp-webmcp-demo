from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from rcp.agents.command_mailbox import (
    CommandTurnIdentity,
    StagedCommandMailbox,
    prepare_command_mailbox,
    serve_command_mailbox,
    stage_command_mailbox,
)
from rcp.agents.command_protocol import (
    CommandRequest,
    CommandResponse,
    ValidateCommandRequest,
    staged_command_client_source,
)
from rcp.background import AgentTaskExecution
from rcp.limits import PATCH_SELF_CHECK_MAX_COUNT, PATCH_SELF_CHECK_POLL_SECONDS
from rcp.transport import RemoteRunStage, RunStageMailbox, StateUnavailable

# Compatibility export for callers that assert which tested source was staged.
# The executable itself lives in one stdlib-only module; there is no handwritten copy.
VALIDATOR_CLIENT_SOURCE = staged_command_client_source()


class PatchValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["valid", "invalid", "unavailable"]
    messages: list[str] = Field(default_factory=list)
    live_revision: int | None = Field(default=None, ge=0)
    candidate_revision: int | None = Field(default=None, ge=0)


@dataclass
class PatchValidationBudget:
    count: int = 0


def stage_patch_validation_mailbox(
    *,
    local_stage: Path | None,
    remote_stage: RemoteRunStage | None,
    task_id: str,
    turn_id: str,
    timeout_seconds: float,
) -> StagedCommandMailbox:
    """Stage the one command client with a validate-only, non-episode credential."""

    return stage_command_mailbox(
        local_stage=local_stage,
        remote_stage=remote_stage,
        episode_id=None,
        task_id=task_id,
        turn_id=turn_id,
        timeout_seconds=timeout_seconds,
    )


def prepare_patch_validation_mailbox(
    *,
    mailbox_id: str,
    workspace: Path,
    remote_stage: RemoteRunStage | None,
) -> None:
    """Compatibility hook for the Work call site while its serialized edit is pending."""

    if len(mailbox_id) != 32 or any(
        character not in "0123456789abcdef" for character in mailbox_id
    ):
        raise ValueError("validator mailbox id is malformed")
    mailbox = RunStageMailbox.for_stage(
        local_stage=workspace if remote_stage is None else None,
        remote_stage=remote_stage,
    )
    prepare_command_mailbox(mailbox=mailbox)


async def serve_patch_validation_mailbox(
    *,
    staged: StagedCommandMailbox,
    execution: AgentTaskExecution | None,
    validate: Callable[[str], PatchValidationResult],
    stop: asyncio.Event,
    budget: PatchValidationBudget,
) -> None:
    """Serve bounded live Patch checks over the unified staged command mailbox."""

    async def handle(
        request: CommandRequest,
        _identity: CommandTurnIdentity,
    ) -> CommandResponse:
        if not isinstance(request, ValidateCommandRequest):
            return CommandResponse(
                request_id=request.request_id,
                status="invalid",
                message="This validator credential authorizes Patch validation only.",
            )
        budget.count += 1
        count = budget.count
        if count > PATCH_SELF_CHECK_MAX_COUNT:
            result = PatchValidationResult(
                status="unavailable",
                messages=["This task has reached its bounded RCP validator self-check limit."],
            )
        else:
            result = await asyncio.to_thread(validate, request.arguments.patch)
        _record_self_check(execution, count, result)
        return _command_response(request.request_id, result)

    try:
        await serve_command_mailbox(
            staged=staged,
            handler=handle,
            stop=stop,
            poll_seconds=PATCH_SELF_CHECK_POLL_SECONDS,
        )
    except (OSError, StateUnavailable, ValueError) as exc:
        _record_mailbox_unavailable(execution, str(exc))


def cleanup_patch_validation_mailbox(
    *,
    staged: StagedCommandMailbox,
    execution: AgentTaskExecution | None,
) -> None:
    """Remove unified command receipts while keeping cleanup failure diagnostic-only."""

    try:
        staged.cleanup()
    except (OSError, StateUnavailable, ValueError) as exc:
        _record_mailbox_unavailable(execution, f"mailbox cleanup failed: {exc}")


def _command_response(request_id: str, result: PatchValidationResult) -> CommandResponse:
    status = {
        "valid": "ok",
        "invalid": "invalid",
        "unavailable": "unavailable",
    }[result.status]
    diagnostic = " ".join(message.strip() for message in result.messages if message.strip())
    if status != "ok" and not diagnostic:
        diagnostic = f"Patch validation was {result.status}."
    return CommandResponse(
        request_id=request_id,
        status=status,
        message=diagnostic[:2_000] or None,
        result=result.model_dump(mode="json"),
    )


def _record_self_check(
    execution: AgentTaskExecution | None,
    count: int,
    result: PatchValidationResult,
) -> None:
    if execution is None:
        return
    execution.store.record_agent_task_event(
        execution.operation_id,
        (
            f"Patch self-check {count}/{PATCH_SELF_CHECK_MAX_COUNT}: "
            f"{result.status}"
            + (
                f" against live graph revision {result.live_revision}"
                if result.live_revision is not None
                else ""
            )
            + "."
        ),
        level="info" if result.status == "valid" else "warning",
    )
    execution.store.record_agent_task_receipt(
        execution.operation_id,
        "patch_self_check",
        {
            "count": count,
            "limit": PATCH_SELF_CHECK_MAX_COUNT,
            **result.model_dump(mode="json"),
        },
        tier="diagnostic",
    )


def _record_mailbox_unavailable(execution: AgentTaskExecution | None, detail: str) -> None:
    if execution is None:
        return
    execution.store.record_agent_task_event(
        execution.operation_id,
        f"Patch validator became unavailable: {' '.join(detail.split())[:400]}",
        level="warning",
    )
