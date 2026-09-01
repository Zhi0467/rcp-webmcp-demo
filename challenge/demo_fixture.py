"""Seed the challenge session with one ordinary historical task artifact."""

from __future__ import annotations

from pathlib import Path

from rcp.artifacts import descriptor_for
from rcp.service import RunRequest, resolve_dispatch_authority
from rcp.storage import AgentTaskRecord, AppStore

DEMO_CHAT_ID = "20000000-0000-4000-8000-000000000001"
DEMO_ARTIFACT_OPERATION_ID = "20000000-0000-4000-8000-000000000003"
DEMO_ARTIFACT_NAME = "plasticity-reliability-overview.html"
DEMO_DISPLAY_NAME = "Demo Researcher"


def seed_demo_records(store: AppStore, project_id: str, stage_root: Path) -> AgentTaskRecord:
    """Create the fixed pre-Experiment task once inside one isolated demo DB."""

    stage_root = stage_root.resolve()
    owner = store.local_owner
    if owner is None:
        raise ValueError("The RCP Demo requires a personal-space owner.")
    if owner.display_name != DEMO_DISPLAY_NAME:
        store.rename_space_user(owner.user_id, DEMO_DISPLAY_NAME)

    source = Path(__file__).with_name("fixture") / DEMO_ARTIFACT_NAME
    data = source.read_bytes()
    descriptor = descriptor_for(
        DEMO_ARTIFACT_OPERATION_ID,
        DEMO_ARTIFACT_NAME,
        size_bytes=len(data),
    )
    request = RunRequest(
        provider="rcp-demo",
        run_truth_scope=["crlp-demo-state"],
        model="rcp-demo-1",
        reasoning="",
        run_on="laptop",
        chat_scope="node",
        node_id="hyp/search-restores-future-learning",
        message="Summarize what is reliable enough to decide the next research action.",
        chat_id=DEMO_CHAT_ID,
        mode="discuss",
    )
    request_payload = request.model_dump(mode="json")
    result = {
        "messages": [
            "The matched-path checks are reliable enough to run the held-out replicate. "
            "They do not yet establish why future learning differs."
        ],
        "artifacts": [descriptor.model_dump(mode="json")],
    }
    dispatch_authority = resolve_dispatch_authority("node_chat", request)
    existing = store.agent_task(DEMO_ARTIFACT_OPERATION_ID)
    if existing is not None:
        if (
            existing.project_id != project_id
            or existing.kind != "node_chat"
            or existing.status not in {"running", "succeeded"}
            or existing.request != request_payload
            or existing.runtime_id != "rcp-demo.jsonl.v1"
            or existing.stage_root != str(stage_root)
            or existing.dispatch_authority != dispatch_authority
            or (existing.status == "succeeded" and existing.result != result)
        ):
            raise ValueError("The RCP Demo seed identity conflicts with existing task state.")
        _write_demo_artifact(stage_root, data)
        if existing.status == "succeeded":
            return existing
        store.complete_agent_task(
            existing.operation_id,
            applied_revision=None,
            result=result,
        )
        recovered = store.agent_task(existing.operation_id)
        assert recovered is not None
        return recovered

    _write_demo_artifact(stage_root, data)
    now = store.now()
    record = AgentTaskRecord(
        operation_id=DEMO_ARTIFACT_OPERATION_ID,
        project_id=project_id,
        kind="node_chat",
        status="running",
        request=request_payload,
        created_at=now,
        updated_at=now,
        started_at=now,
        status_message="RCP Demo fixture result is ready.",
        runtime_id="rcp-demo.jsonl.v1",
        stage_root=str(stage_root),
        dispatch_authority=dispatch_authority,
    )
    store.create_agent_task(record)
    store.record_agent_task_receipt(
        record.operation_id,
        "operation_created",
        {
            "kind": record.kind,
            "attempt": record.attempt,
            "has_parent": False,
            "resumed": False,
        },
        tier="diagnostic",
    )
    store.complete_agent_task(
        record.operation_id,
        applied_revision=None,
        result=result,
    )
    stored = store.agent_task(record.operation_id)
    assert stored is not None
    return stored


def _write_demo_artifact(stage_root: Path, data: bytes) -> None:
    artifact_directory = stage_root / "turns" / DEMO_ARTIFACT_OPERATION_ID / "artifacts"
    if artifact_directory.is_symlink():
        raise ValueError("The RCP Demo artifact directory cannot be a symlink.")
    artifact_directory.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_directory / DEMO_ARTIFACT_NAME
    if artifact_path.is_symlink():
        raise ValueError("The RCP Demo artifact cannot be a symlink.")
    artifact_path.write_bytes(data)
