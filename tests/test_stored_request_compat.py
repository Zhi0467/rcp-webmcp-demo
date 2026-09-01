from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from rcp.runs.auto_research import AutoResearchRunRequest
from rcp.storage.rows import RowMappingMixin


def _task_row(request: dict[str, object]) -> dict[str, object]:
    return {
        "operation_id": "op-1",
        "project_id": "project",
        "kind": "auto_research",
        "status": "queued",
        "request_json": json.dumps(request),
        "created_at": "2026-08-20T00:00:00+00:00",
        "updated_at": "2026-08-20T00:00:00+00:00",
        "status_message": "Waiting.",
        "authorized_space_id": None,
        "authorized_user_id": None,
        "authorized_display_name": None,
    }


def test_task_row_decoder_migrates_before_any_run_path_reads_the_request() -> None:
    record = RowMappingMixin()._agent_task_record(
        _task_row(
            {
                "episode_id": "episode-1",
                "role": "orchestrator",
                "ending": None,
            }
        )
    )

    assert record.request == {"episode_id": "episode-1", "role": "orchestrator"}
    assert AutoResearchRunRequest.model_validate(record.request).episode_id == "episode-1"


def test_task_row_decoder_preserves_unallowlisted_fields_for_strict_rejection() -> None:
    record = RowMappingMixin()._agent_task_record(
        _task_row(
            {
                "episode_id": "episode-1",
                "role": "orchestrator",
                "future_field": True,
            }
        )
    )

    assert record.request["future_field"] is True
    with pytest.raises(ValidationError):
        AutoResearchRunRequest.model_validate(record.request)


def test_task_row_decoder_preserves_persisted_provider_runtime() -> None:
    row = _task_row(
        {
            "episode_id": "episode-1",
            "role": "orchestrator",
            "provider": "codex",
        }
    )
    row["runtime_id"] = "codex.app-server-stdio.v1"

    record = RowMappingMixin()._agent_task_record(row)

    assert record.operation_id == "op-1"
    assert record.runtime_id == "codex.app-server-stdio.v1"
    assert record.request == {
        "episode_id": "episode-1",
        "role": "orchestrator",
        "provider": "codex",
    }
