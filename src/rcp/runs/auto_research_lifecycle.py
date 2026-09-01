from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rcp.limits import (
    AUTO_RESEARCH_LIFECYCLE_MAX_BYTES,
    AUTO_RESEARCH_LIFECYCLE_MAX_NOTICES,
)
from rcp.storage import AutoResearchLifecycleNoticeRecord
from rcp.transport.workspace_mailbox import RunStageMailbox

AUTO_RESEARCH_LIFECYCLE_PROTOCOL_VERSION = 1
AUTO_RESEARCH_LIFECYCLE_HANDOFF_FILE = "lifecycle.json"


def _wire_size(delivery: BaseModel) -> int:
    return len((delivery.model_dump_json() + "\n").encode("utf-8"))


class AutoResearchLifecycleFact(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    notice_id: str = Field(min_length=1)
    source_kind: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_event: str = Field(min_length=1)
    source_attempt: int = Field(ge=1)
    payload: dict[str, object]
    created_at: str = Field(min_length=1)
    delivered_at: str = Field(min_length=1)


class AutoResearchLifecycleDelivery(BaseModel):
    """Strict RCP-authored lifecycle input, separate from hearsay mail."""

    model_config = ConfigDict(extra="forbid", strict=True)

    version: Literal[1] = AUTO_RESEARCH_LIFECYCLE_PROTOCOL_VERSION
    kind: Literal["auto_research_lifecycle_delivery"] = "auto_research_lifecycle_delivery"
    episode_id: str = Field(min_length=1)
    recipient_task_id: str = Field(min_length=1)
    delivery_operation_id: str = Field(min_length=1)
    authority: Literal["rcp_lifecycle"] = "rcp_lifecycle"
    graph_authority: Literal["none"] = "none"
    notices: list[AutoResearchLifecycleFact] = Field(min_length=1)

    @model_validator(mode="after")
    def delivery_is_bounded_and_unique(self) -> AutoResearchLifecycleDelivery:
        if len(self.notices) > AUTO_RESEARCH_LIFECYCLE_MAX_NOTICES:
            raise ValueError(
                "Auto-research lifecycle delivery exceeds "
                f"{AUTO_RESEARCH_LIFECYCLE_MAX_NOTICES} notices"
            )
        notice_ids = [notice.notice_id for notice in self.notices]
        if len(notice_ids) != len(set(notice_ids)):
            raise ValueError("Auto-research lifecycle delivery repeats a notice")
        if _wire_size(self) > AUTO_RESEARCH_LIFECYCLE_MAX_BYTES:
            raise ValueError(
                "Auto-research lifecycle delivery exceeds "
                f"{AUTO_RESEARCH_LIFECYCLE_MAX_BYTES} bytes"
            )
        return self


def auto_research_lifecycle_delivery(
    *,
    episode_id: str,
    recipient_task_id: str,
    delivery_operation_id: str,
    notices: Sequence[AutoResearchLifecycleNoticeRecord],
) -> AutoResearchLifecycleDelivery:
    facts: list[AutoResearchLifecycleFact] = []
    for notice in notices:
        if (
            notice.episode_id != episode_id
            or notice.delivery_operation_id != delivery_operation_id
            or notice.delivered_at is None
        ):
            raise ValueError("Lifecycle notice does not match its claimed delivery")
        facts.append(
            AutoResearchLifecycleFact(
                notice_id=notice.notice_id,
                source_kind=notice.source_kind,
                source_id=notice.source_id,
                source_event=notice.source_event,
                source_attempt=notice.source_attempt,
                payload=notice.payload,
                created_at=notice.created_at,
                delivered_at=notice.delivered_at,
            )
        )
    return AutoResearchLifecycleDelivery(
        episode_id=episode_id,
        recipient_task_id=recipient_task_id,
        delivery_operation_id=delivery_operation_id,
        notices=facts,
    )


def parse_auto_research_lifecycle_delivery(
    value: str | bytes,
) -> AutoResearchLifecycleDelivery:
    encoded = value.encode("utf-8") if isinstance(value, str) else value
    if len(encoded) > AUTO_RESEARCH_LIFECYCLE_MAX_BYTES:
        raise ValueError(
            f"Auto-research lifecycle delivery exceeds {AUTO_RESEARCH_LIFECYCLE_MAX_BYTES} bytes"
        )
    return AutoResearchLifecycleDelivery.model_validate_json(encoded)


def stage_auto_research_lifecycle_delivery(
    mailbox: RunStageMailbox,
    delivery: AutoResearchLifecycleDelivery,
) -> None:
    validated = AutoResearchLifecycleDelivery.model_validate(delivery.model_dump(mode="python"))
    mailbox.write_text(
        AUTO_RESEARCH_LIFECYCLE_HANDOFF_FILE,
        validated.model_dump_json() + "\n",
    )
