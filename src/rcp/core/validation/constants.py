from __future__ import annotations

import re

from pydantic import TypeAdapter

from rcp.core.models import ProjectNode

NODE_PREFIXES = {
    "research_question": "rq",
    "hypothesis": "hyp",
    "decision": "dec",
    "experiment": "exp",
    "evidence": "ev",
    "blocker": "blk",
}
SLUG_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*/[a-z0-9]+(?:-[a-z0-9]+)*$")
IDENTIFIER_RE = re.compile(r"\b(?:[a-z]+_[a-z0-9_]+|[a-z]+-[a-z0-9-]*\d[a-z0-9-]*)\b")
NODE_ADAPTER = TypeAdapter(ProjectNode)
IMMUTABLE_NODE_UPDATE_FIELDS = frozenset(
    {
        "id",
        "type",
        "extension_type",
        "standing",
        "created_rev",
        "updated_rev",
        "legacy_strength",
        "current_summary_stale",
        "next_action_stale",
    }
)

# Loading a pre-generation-2 Patch migrates retired vocabulary in memory, which
# adds these system fields to operations that never carried them on disk. Replay
# must accept them where a live write never could -- see
# `adapt_persisted_patch_document`.
LEGACY_COMPATIBILITY_UPDATE_FIELDS = frozenset(
    {"legacy_strength", "current_summary_stale", "next_action_stale"}
)
