from __future__ import annotations

from rcp.core.models import ValidationMessage


class ValidationReport:
    def __init__(self) -> None:
        self.messages: list[ValidationMessage] = []

    @property
    def rejected(self) -> bool:
        return any(item.level == "reject" for item in self.messages)

    @property
    def flags(self) -> list[ValidationMessage]:
        return [item for item in self.messages if item.level == "flag"]

    def reject(
        self,
        code: str,
        message: str,
        revision: int | None = None,
        *,
        related_node_ids: list[str] | None = None,
        related_edge_ids: list[str] | None = None,
        operation_index: int | None = None,
        rule_id: str | None = None,
        cause_chain: list[dict[str, object]] | None = None,
        failed_invariant: str | None = None,
    ) -> None:
        self.messages.append(
            ValidationMessage(
                level="reject",
                code=code,
                message=message,
                patch_revision=revision,
                related_node_ids=related_node_ids or [],
                related_edge_ids=related_edge_ids or [],
                operation_index=operation_index,
                rule_id=rule_id,
                cause_chain=cause_chain or [],
                failed_invariant=failed_invariant,
            )
        )

    def flag(
        self,
        code: str,
        message: str,
        revision: int | None = None,
        *,
        related_node_ids: list[str] | None = None,
        related_edge_ids: list[str] | None = None,
    ) -> None:
        self.messages.append(
            ValidationMessage(
                level="flag",
                code=code,
                message=message,
                patch_revision=revision,
                related_node_ids=related_node_ids or [],
                related_edge_ids=related_edge_ids or [],
            )
        )
