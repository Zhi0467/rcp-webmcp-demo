from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Final

logger = logging.getLogger(__name__)

# Persisted request compatibility is deliberately closed. A field appears here
# only after RCP shipped it, later retired it, and documented why dropping that
# exact field preserves the request's meaning. Unknown fields are left in place
# so the request model rejects them rather than silently guessing a migration.
_RETIRED_REQUEST_FIELDS: Final[dict[str, frozenset[str]]] = {
    "auto_research": frozenset({"ending"}),
}


def migrate_stored_task_request(
    kind: str,
    stored: Mapping[str, object],
    *,
    operation_id: str | None = None,
    warn: bool = True,
) -> dict[str, object]:
    """Apply the explicit compatibility allowlist to one persisted task request."""

    migrated = dict(stored)
    retired = _RETIRED_REQUEST_FIELDS.get(kind, frozenset())
    removed = sorted(set(migrated) & retired)
    if not removed:
        return migrated
    for field in removed:
        migrated.pop(field)
    if warn:
        logger.warning(
            "Dropped retired field(s) %s from stored %s task %s.",
            ", ".join(removed),
            kind,
            operation_id or "<unknown>",
        )
    return migrated
