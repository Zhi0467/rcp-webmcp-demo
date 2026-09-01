from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rcp.core.models import AuthorizedHuman
from rcp.limits import (
    AUTO_RESEARCH_MAIL_MAX_BYTES,
    AUTO_RESEARCH_MAIL_MAX_MESSAGES,
)
from rcp.storage import AutoResearchMessageRecord, AutoResearchMessageRole
from rcp.transport.workspace_mailbox import RunStageMailbox

AUTO_RESEARCH_MAIL_PROTOCOL_VERSION = 1
AUTO_RESEARCH_MAIL_HANDOFF_FILE = "messages.json"


def _auto_research_mail_wire_size(delivery: BaseModel) -> int:
    return len((delivery.model_dump_json() + "\n").encode("utf-8"))


class AutoResearchMailMessage(BaseModel):
    """One durable message copied into a claimed delivery without reinterpretation."""

    model_config = ConfigDict(extra="forbid", strict=True)

    message_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    sender_role: AutoResearchMessageRole
    sender_task_id: str | None = Field(default=None, min_length=1)
    authorized_by: AuthorizedHuman | None = None
    recipient_task_id: str = Field(min_length=1)
    control_node_id: str | None = Field(default=None, min_length=1)
    body: str = Field(min_length=1, max_length=16_000)
    created_at: str = Field(min_length=1)
    delivered_at: str = Field(min_length=1)
    delivery_operation_id: str = Field(min_length=1)

    @field_validator("body")
    @classmethod
    def body_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("auto_research mail body must not be blank")
        return value

    @model_validator(mode="after")
    def sender_metadata_is_explicit(self) -> AutoResearchMailMessage:
        if self.sender_role == "human":
            if self.sender_task_id is not None:
                raise ValueError("human auto_research mail cannot claim an agent task sender")
        else:
            if self.sender_task_id is None:
                raise ValueError("agent auto_research mail must name its durable sender task")
            if self.authorized_by is not None:
                raise ValueError("agent auto_research mail cannot claim a human sender snapshot")
        return self


class AutoResearchMailDelivery(BaseModel):
    """One coalesced, already-claimed inbound mail handoff for a provider turn."""

    model_config = ConfigDict(extra="forbid", strict=True)

    version: Literal[1] = AUTO_RESEARCH_MAIL_PROTOCOL_VERSION
    kind: Literal["auto_research_mail_delivery"] = "auto_research_mail_delivery"
    episode_id: str = Field(min_length=1)
    recipient_task_id: str = Field(min_length=1)
    delivery_operation_id: str = Field(min_length=1)
    graph_authority: Literal["none"] = "none"
    epistemic_status: Literal["hearsay"] = "hearsay"
    messages: list[AutoResearchMailMessage] = Field(min_length=1)

    @model_validator(mode="after")
    def messages_match_the_claimed_delivery(self) -> AutoResearchMailDelivery:
        if len(self.messages) > AUTO_RESEARCH_MAIL_MAX_MESSAGES:
            raise ValueError(
                f"auto_research mail delivery exceeds {AUTO_RESEARCH_MAIL_MAX_MESSAGES} messages"
            )
        message_ids: set[str] = set()
        for message in self.messages:
            if message.message_id in message_ids:
                raise ValueError("auto_research mail delivery contains a duplicate message")
            message_ids.add(message.message_id)
            if message.episode_id != self.episode_id:
                raise ValueError("auto_research mail delivery crosses auto_researchs")
            if message.recipient_task_id != self.recipient_task_id:
                raise ValueError("auto_research mail delivery crosses recipients")
            if message.delivery_operation_id != self.delivery_operation_id:
                raise ValueError("auto_research mail delivery crosses claimed wake operations")
        if _auto_research_mail_wire_size(self) > AUTO_RESEARCH_MAIL_MAX_BYTES:
            raise ValueError(
                f"auto_research mail delivery exceeds {AUTO_RESEARCH_MAIL_MAX_BYTES} bytes"
            )
        return self

    @property
    def message_ids(self) -> list[str]:
        return [message.message_id for message in self.messages]


def auto_research_mail_claim_prefix(
    *,
    episode_id: str,
    recipient_task_id: str,
    delivery_operation_id: str,
    delivered_at: str,
    messages: Sequence[AutoResearchMessageRecord],
) -> list[AutoResearchMessageRecord]:
    """Select the largest deterministic pending prefix inside the handoff domain."""

    selected: list[AutoResearchMessageRecord] = []
    projected: list[AutoResearchMailMessage] = []
    message_ids: set[str] = set()
    for message in messages:
        if len(selected) >= AUTO_RESEARCH_MAIL_MAX_MESSAGES:
            break
        if message.message_id in message_ids:
            raise ValueError("auto_research mail claim contains a duplicate message")
        if message.episode_id != episode_id:
            raise ValueError("auto_research mail claim crosses auto_researchs")
        if message.recipient_task_id != recipient_task_id:
            raise ValueError("auto_research mail claim crosses recipients")
        if message.delivered_at is not None or message.delivery_operation_id is not None:
            raise ValueError("auto_research mail claim contains an already delivered message")
        copied = AutoResearchMailMessage.model_validate(
            {
                **message.model_dump(mode="python"),
                "delivered_at": delivered_at,
                "delivery_operation_id": delivery_operation_id,
            }
        )
        candidate = AutoResearchMailDelivery.model_construct(
            episode_id=episode_id,
            recipient_task_id=recipient_task_id,
            delivery_operation_id=delivery_operation_id,
            messages=[*projected, copied],
        )
        if _auto_research_mail_wire_size(candidate) > AUTO_RESEARCH_MAIL_MAX_BYTES:
            break
        selected.append(message)
        projected.append(copied)
        message_ids.add(message.message_id)
    return selected


def auto_research_mail_delivery(
    *,
    episode_id: str,
    recipient_task_id: str,
    delivery_operation_id: str,
    messages: Sequence[AutoResearchMessageRecord],
) -> AutoResearchMailDelivery:
    """Build an inbound handoff only from records claimed by this exact wake."""

    copied = [AutoResearchMailMessage.model_validate(message.model_dump()) for message in messages]
    return AutoResearchMailDelivery(
        episode_id=episode_id,
        recipient_task_id=recipient_task_id,
        delivery_operation_id=delivery_operation_id,
        messages=copied,
    )


def parse_auto_research_mail_delivery(value: str | bytes) -> AutoResearchMailDelivery:
    """Strictly parse a complete versioned ``messages.json`` handoff."""

    encoded = value.encode("utf-8") if isinstance(value, str) else value
    if len(encoded) > AUTO_RESEARCH_MAIL_MAX_BYTES:
        raise ValueError(
            f"auto_research mail delivery exceeds {AUTO_RESEARCH_MAIL_MAX_BYTES} bytes"
        )
    return AutoResearchMailDelivery.model_validate_json(encoded)


def stage_auto_research_mail_delivery(
    mailbox: RunStageMailbox,
    delivery: AutoResearchMailDelivery,
) -> None:
    """Atomically stage inbound mail after the owning turn clears stale handoffs."""

    validated = AutoResearchMailDelivery.model_validate(delivery.model_dump(mode="python"))
    mailbox.write_text(AUTO_RESEARCH_MAIL_HANDOFF_FILE, validated.model_dump_json() + "\n")
