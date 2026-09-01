"""Conversation record parsing shared by local indexing and remote execution.

This module is executed in two places: imported normally in this process, and
**shipped as source text** to a remote host by `indexer.py`, where it is
prepended to a small driver and run with `python3 -c`. That remote host has no
virtualenv and no `rcp` package, so this module may import **only** the standard
library and must not import anything else from `rcp`.

`tests/test_sources.py` enforces both halves of that contract: the import
restriction, and that the shipped program produces byte-identical records to the
local path for the same input.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
from typing import Any

TEXT_CHUNK_TYPES = frozenset({"text", "input_text", "output_text"})
KNOWN_ROLES = frozenset({"user", "assistant", "system", "tool"})


def extract_text(content: Any) -> str:
    """Flatten a provider content field into plain text."""

    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    chunks: list[str] = []
    for item in content:
        if isinstance(item, str):
            chunks.append(item)
        elif isinstance(item, dict) and item.get("type") in TEXT_CHUNK_TYPES:
            value = item.get("text")
            if isinstance(value, str):
                chunks.append(value)
    return "\n".join(chunks)


def fallback_record_id(raw: dict[str, Any], line_number: int) -> str:
    """Derive a stable id for a record whose provider gave it none."""

    digest = hashlib.sha256(
        json.dumps(raw, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    return f"line-{line_number}-{digest}"


def normalize_record(raw: dict[str, Any], provider: str, line_number: int) -> dict[str, Any]:
    """Normalize one provider record.

    Returns a plain dict rather than a model so the remote copy needs no
    pydantic. `timestamp` stays as the provider wrote it; callers that want a
    datetime parse it themselves.
    """

    if provider == "codex":
        payload = raw.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}
        record_id = payload.get("id") or raw.get("id")
        raw_type = f"{raw.get('type', '')}:{payload.get('type', '')}".rstrip(":")
        role = payload.get("role", "unknown")
        text = extract_text(payload.get("content"))
        if not text and payload.get("type") in {"user_message", "agent_message"}:
            text = str(payload.get("message", ""))
            role = "user" if payload.get("type") == "user_message" else "assistant"
        if not text and payload.get("type") == "custom_tool_call":
            tool_input = payload.get("input")
            if isinstance(tool_input, str):
                text = tool_input
                role = "assistant"
        timestamp = payload.get("timestamp") or raw.get("timestamp")
    elif provider == "claude":
        record_id = raw.get("uuid")
        raw_type = str(raw.get("type", ""))
        role = raw_type if raw_type in {"user", "assistant", "system"} else "unknown"
        message = raw.get("message", {})
        text = extract_text(message.get("content") if isinstance(message, dict) else message)
        timestamp = raw.get("timestamp")
    else:
        record_id = raw.get("uuid") or raw.get("id")
        raw_type = str(raw.get("type", ""))
        role = raw.get("role", "unknown")
        text = str(raw.get("text", raw.get("content", "")))
        timestamp = raw.get("timestamp")

    if role not in KNOWN_ROLES:
        role = "unknown"
    return {
        "uuid": str(record_id or fallback_record_id(raw, line_number)),
        "timestamp": timestamp,
        "role": role,
        "text": text,
        "raw_type": raw_type,
    }


def normalize_path(value: str) -> str:
    if not value:
        return ""
    return posixpath.normpath(value.replace("\\", "/"))


def path_matches_roots(cwd: str, roots: list[str]) -> bool:
    """True when `cwd` is one of `roots` or sits inside one of them."""

    normalized = normalize_path(cwd)
    for root in roots:
        normalized_root = normalize_path(root)
        if normalized == normalized_root or normalized.startswith(
            normalized_root.rstrip("/") + "/"
        ):
            return True
    return False
