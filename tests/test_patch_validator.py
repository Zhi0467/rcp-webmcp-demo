from __future__ import annotations

import asyncio
import json
import subprocess
import uuid
from pathlib import Path

import pytest

from rcp.history import HistoryManager
from rcp.limits import PATCH_SELF_CHECK_MAX_COUNT
from rcp.paper import PaperService
from rcp.runs.patch_validator import (
    PatchValidationBudget,
    PatchValidationResult,
    prepare_patch_validation_mailbox,
    serve_patch_validation_mailbox,
    stage_patch_validation_mailbox,
)
from rcp.runs.tasks.graph import _validate_graph_patch_live
from rcp.runs.tasks.work import _apply_work_patch, _validate_work_patch_live
from rcp.service import ProjectService
from rcp.storage import AppStore
from tests.helpers import agent_patch_json, refresh_patch, seed_patch


class _RecordingStore:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, str]] = []
        self.receipts: list[tuple[str, str, dict[str, object], str]] = []

    def record_agent_task_event(self, operation_id: str, message: str, *, level: str) -> None:
        self.events.append((operation_id, message, level))

    def record_agent_task_receipt(
        self,
        operation_id: str,
        category: str,
        payload: dict[str, object],
        *,
        tier: str,
    ) -> None:
        self.receipts.append((operation_id, category, payload, tier))


class _Execution:
    operation_id = "work-operation"

    def __init__(self) -> None:
        self.store = _RecordingStore()


async def _run_client(
    staged,
    patch_path: Path,
    *,
    timeout: float = 2,
) -> subprocess.CompletedProcess[str]:
    return await asyncio.to_thread(
        subprocess.run,
        staged.client_argv("validate", str(patch_path), timeout_seconds=timeout),
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.asyncio
async def test_validator_client_distinguishes_valid_invalid_and_unavailable(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    patch_path = workspace / "patch.json"
    patch_path.write_text("{}", encoding="utf-8")

    for status, expected_code in (("valid", 0), ("invalid", 1)):
        staged = stage_patch_validation_mailbox(
            local_stage=workspace,
            remote_stage=None,
            task_id="validator-task",
            turn_id=f"validator-{status}",
            timeout_seconds=2,
        )
        stop = asyncio.Event()
        server = asyncio.create_task(
            serve_patch_validation_mailbox(
                staged=staged,
                execution=None,
                validate=lambda _text, status=status: PatchValidationResult(
                    status=status,
                    messages=[status],
                    live_revision=4,
                    candidate_revision=5,
                ),
                stop=stop,
                budget=PatchValidationBudget(),
            )
        )
        result = await _run_client(staged, patch_path)
        stop.set()
        await server
        staged.cleanup()
        assert result.returncode == expected_code
        assert json.loads(result.stdout)["status"] == status

    staged = stage_patch_validation_mailbox(
        local_stage=workspace,
        remote_stage=None,
        task_id="validator-task",
        turn_id="validator-unavailable",
        timeout_seconds=0.2,
    )
    unavailable = await _run_client(staged, patch_path, timeout=0.2)
    staged.cleanup()
    assert unavailable.returncode == 2
    assert "did not answer" in unavailable.stdout


@pytest.mark.asyncio
async def test_patch_self_checks_are_bounded_and_each_one_is_a_task_event(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    patch_path = workspace / "patch.json"
    patch_path.write_text("{}", encoding="utf-8")
    staged = stage_patch_validation_mailbox(
        local_stage=workspace,
        remote_stage=None,
        task_id="validator-task",
        turn_id="bounded-validator",
        timeout_seconds=2,
    )
    execution = _Execution()
    calls = 0

    def validate(_text: str) -> PatchValidationResult:
        nonlocal calls
        calls += 1
        return PatchValidationResult(status="valid", live_revision=1, candidate_revision=2)

    stop = asyncio.Event()
    budget = PatchValidationBudget()
    server = asyncio.create_task(
        serve_patch_validation_mailbox(
            staged=staged,
            execution=execution,  # type: ignore[arg-type]
            validate=validate,
            stop=stop,
            budget=budget,
        )
    )
    results = [await _run_client(staged, patch_path) for _ in range(PATCH_SELF_CHECK_MAX_COUNT + 1)]
    stop.set()
    await server
    staged.cleanup()

    assert [result.returncode for result in results[:-1]] == [0] * PATCH_SELF_CHECK_MAX_COUNT
    assert results[-1].returncode == 2
    assert calls == PATCH_SELF_CHECK_MAX_COUNT
    assert budget.count == PATCH_SELF_CHECK_MAX_COUNT + 1
    assert len(execution.store.events) == PATCH_SELF_CHECK_MAX_COUNT + 1
    assert "self-check limit" in results[-1].stdout


def test_stable_validator_mailbox_is_cleaned_before_each_provider_pass(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    mailbox_id = uuid.uuid4().hex
    stale_prefix = f"rcp-command-{mailbox_id}-{uuid.uuid4().hex}"
    stale_request = workspace / f"{stale_prefix}.request.json"
    stale_response = workspace / f"{stale_prefix}.response.json"
    unrelated = workspace / "keep.txt"
    stale_request.write_text("{}", encoding="utf-8")
    stale_response.write_text("{}", encoding="utf-8")
    unrelated.write_text("keep", encoding="utf-8")

    prepare_patch_validation_mailbox(
        mailbox_id=mailbox_id,
        workspace=workspace,
        remote_stage=None,
    )

    assert not stale_request.exists()
    assert not stale_response.exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_stable_validator_mailbox_preparation_fails_when_workspace_is_unavailable(
    tmp_path: Path,
) -> None:
    with pytest.raises(OSError, match="run workspace .* is unavailable"):
        prepare_patch_validation_mailbox(
            mailbox_id=uuid.uuid4().hex,
            workspace=tmp_path / "missing",
            remote_stage=None,
        )


def test_live_self_check_and_apply_share_current_state_validation(manifest, tmp_path: Path) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    service = ProjectService(
        manifest,
        history,
        PaperService(manifest, AppStore(tmp_path / "app.sqlite3")),
        data_dir=tmp_path,
    )
    semantic_patch = json.dumps(
        {
            "summary": "Create a Work result node.",
            "repositories_read": ["repo-a"],
            "change_summary": ["Created the Work result node."],
            "ops": [
                {
                    "op": "create_nodes",
                    "nodes": [
                        {
                            "id": "rq/live-validator-result",
                            "type": "research_question",
                            "title": "Live validator result",
                            "question": "Does the live validator see canonical movement?",
                        }
                    ],
                }
            ],
        }
    )

    checked = _validate_work_patch_live(
        service,
        semantic_patch,
        run_truth_scope=["repo-a"],
    )
    assert checked.status == "valid"
    assert checked.live_revision == 1
    assert history.state().revision == 1

    history.append(refresh_patch("rq/live-validator-result"))
    rechecked = _validate_work_patch_live(
        service,
        semantic_patch,
        run_truth_scope=["repo-a"],
    )
    assert rechecked.status == "invalid"
    assert rechecked.live_revision == 2
    assert any("already exists" in message for message in rechecked.messages)

    applied, failure = _apply_work_patch(
        service,
        None,
        semantic_patch,
        run_truth_scope=["repo-a"],
    )
    assert applied is None
    assert failure is not None
    assert "already exists" in failure.message
    assert history.state().revision == 2


def test_graph_live_self_check_validates_current_state_without_appending(
    manifest, tmp_path: Path
) -> None:
    history = HistoryManager(manifest)
    service = ProjectService(
        manifest,
        history,
        PaperService(manifest, AppStore(tmp_path / "app.sqlite3")),
        data_dir=tmp_path,
    )
    semantic_patch = agent_patch_json(seed_patch())

    checked = _validate_graph_patch_live(
        service,
        semantic_patch,
        kind="seed",
        run_truth_scope=["repo-a"],
    )
    assert checked.status == "valid"
    assert checked.live_revision == 0
    assert checked.candidate_revision == 1
    assert history.state().revision == 0

    history.append(seed_patch())
    rechecked = _validate_graph_patch_live(
        service,
        semantic_patch,
        kind="seed",
        run_truth_scope=["repo-a"],
    )
    assert rechecked.status == "invalid"
    assert rechecked.live_revision == 1
    assert rechecked.candidate_revision == 2
    assert any("already exists" in message for message in rechecked.messages)
    assert history.state().revision == 1
