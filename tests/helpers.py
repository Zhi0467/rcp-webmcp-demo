from __future__ import annotations

import json
import time
import uuid
from typing import Any

from pydantic import TypeAdapter

from rcp.api import create_app
from rcp.core.models import AuthorizedHuman, Patch
from rcp.core.operations import GraphOperation, ProposalOperation
from rcp.history import HistoryManager
from rcp.storage import ACTIVE_AGENT_TASK_STATUSES, AgentTaskRecord, AppStore

# Background tasks run on their own thread, so this bounds a genuine hang rather
# than the expected duration. The poll returns the moment the task is terminal,
# so a generous bound costs nothing on success, while a tight one invents
# failures whenever the full suite is competing for the CPU.
TASK_SETTLE_TIMEOUT = 60.0
_TASK_POLL_INTERVAL = 0.01

_GRAPH_OPERATION_ADAPTER = TypeAdapter(GraphOperation)
_PROPOSAL_OPERATION_ADAPTER = TypeAdapter(ProposalOperation)

_RCP_OWNED_ITEM_FIELDS = {
    "create_nodes": ("nodes", {"standing", "created_rev", "updated_rev"}),
    "create_edges": ("edges", {"layer", "created_rev"}),
    "create_ambiguities": ("ambiguities", {"raised_rev"}),
    "create_proposals": (
        "proposals",
        {
            "related_node_ids",
            "related_edge_ids",
            "related_config_keys",
            "base_rev",
            "status",
            "raised_rev",
            "resolved_rev",
            "rejection_reason",
        },
    ),
    "upsert_glossary": ("terms", {"updated_rev"}),
}


def create_named_app(*args: Any, **kwargs: Any):
    """Create an app whose personal test owner has accepted the write precondition."""

    app = create_app(*args, **kwargs)
    if app.state.space_kind == "personal":
        store = app.state.background_tasks.store
        owner = store.local_owner
        if owner is not None and owner.display_name is None:
            store.rename_space_user(owner.user_id, "Test researcher")
    return app


def _store_of(app_or_store: Any) -> AppStore:
    """Accept either an app or the store itself, so callers keep whichever they hold."""

    if isinstance(app_or_store, AppStore):
        return app_or_store
    return app_or_store.state.background_tasks.store


def authorized_human(
    app_or_store: Any, *, display_name: str = "Test researcher"
) -> AuthorizedHuman:
    """The local owner as a patch author, naming them if the app has not already."""

    store = _store_of(app_or_store)
    owner = store.local_owner
    assert owner is not None, "app has no local owner"
    if owner.display_name is None:
        owner = store.rename_space_user(owner.user_id, display_name)
    return AuthorizedHuman(
        space_id=store.space_id,
        user_id=owner.user_id,
        display_name=owner.display_name,
    )


def seated_on_every_project(_project_id: str, _user_id: str) -> bool:
    """A membership check for histories built without a project catalog.

    `HistoryManager` requires one whenever it can resolve agent authority, so a
    test that fabricates its own resolver supplies this. Membership itself is
    exercised in `test_project_membership.py` against a real store.
    """

    return True


def fabricated_authorizer(display_name: str = "Campaign owner") -> AuthorizedHuman:
    """A synthetic authorizer for stores that were never opened as an app."""

    return AuthorizedHuman(
        space_id=str(uuid.uuid4()),
        user_id=str(uuid.uuid4()),
        display_name=display_name,
    )


def wait_for_task(
    app_or_store: Any,
    operation_id: str,
    *,
    expect: str | None = None,
    timeout: float = TASK_SETTLE_TIMEOUT,
) -> AgentTaskRecord:
    """Poll the store until the task leaves every active status."""

    store = _store_of(app_or_store)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = store.agent_task(operation_id)
        assert record is not None, f"task {operation_id} was never recorded"
        if record.status not in ACTIVE_AGENT_TASK_STATUSES:
            if expect is not None:
                assert record.status == expect, (
                    f"task {operation_id} settled as {record.status!r}, expected {expect!r}: "
                    f"{record.error or record.status_message}"
                )
            return record
        time.sleep(_TASK_POLL_INTERVAL)
    raise AssertionError(f"task {operation_id} did not settle within {timeout}s")


def wait_for_task_response(
    client: Any,
    project_id: str,
    operation_id: str,
    *,
    expect: str | None = None,
    timeout: float = TASK_SETTLE_TIMEOUT,
) -> dict[str, Any]:
    """Poll the task route until the task leaves every active status."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/projects/{project_id}/tasks/{operation_id}")
        assert response.status_code == 200, response.text
        task = response.json()
        if task["status"] not in ACTIVE_AGENT_TASK_STATUSES:
            if expect is not None:
                assert task["status"] == expect, (
                    f"task {operation_id} settled as {task['status']!r}, expected {expect!r}: "
                    f"{task.get('error') or task.get('status_message')}"
                )
            return task
        time.sleep(_TASK_POLL_INTERVAL)
    raise AssertionError(f"task {operation_id} did not settle within {timeout}s")


def append_fixture_patch(service: Any, patch: Patch, **kwargs: Any):
    """Prepare canonical graph state without impersonating a production agent task."""

    fixture_history = HistoryManager(service.manifest, service.history.workspace)
    appended, result = fixture_history.append(patch, **kwargs)
    # Mirror the cache update that the production manager would have performed
    # if this test-only legacy fixture had gone through its guarded admission.
    service.history._remember_accepted_revision(result)
    return appended, result


def agent_patch_json(patch: Patch) -> str:
    """Render canonical test data as the semantic JSON an agent may write."""

    operations = [operation.model_dump(mode="json", exclude_unset=True) for operation in patch.ops]
    for operation in operations:
        owned = _RCP_OWNED_ITEM_FIELDS.get(operation.get("op"))
        if owned is None:
            continue
        field, excluded = owned
        operation[field] = [
            {key: value for key, value in item.items() if key not in excluded}
            for item in operation.get(field, [])
        ]
    return json.dumps(
        {
            "summary": patch.summary,
            "ops": operations,
            "repositories_read": list(patch.repositories_read),
            "change_summary": list(patch.change_summary),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def refresh_patch(node_id: str = "rq/transfer-after-shift") -> Patch:
    """A minimal refresh patch that applies cleanly on top of ``seed_patch``."""
    return Patch(
        kind="refresh",
        author="agent",
        summary="Recorded a second research question.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": node_id,
                        "type": "research_question",
                        "title": "Transfer after task shift",
                        "question": "Does replanning transfer to an unseen task family?",
                        "motivation": "The seed corpus left transfer unexamined.",
                        "scope": "Matched compute across task families.",
                        "status": "open",
                    }
                ],
            }
        ],
        change_summary=[f"Added {node_id}."],
    )


def shape_invalid_patch() -> Patch:
    """A core-valid Patch using an operation absent from the agent schema."""
    return Patch(
        kind="refresh",
        author="agent",
        summary="Used an operation that is not in the agent schema.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "set_ontology",
                "ontology": {"types": [], "fields": [], "relations": []},
            }
        ],
    )


def graph_operation(document: dict[str, Any]) -> GraphOperation:
    """Parse one exact GraphOperation for direct contract-level test calls."""

    return _GRAPH_OPERATION_ADAPTER.validate_python(document)


def proposal_operation(document: dict[str, Any]) -> ProposalOperation:
    """Parse one exact ProposalOperation for direct contract-level test calls."""

    return _PROPOSAL_OPERATION_ADAPTER.validate_python(document)


def gated_patch() -> Patch:
    """Well formed, but asks for a transition the graph gates behind a Proposal."""
    return Patch(
        kind="refresh",
        author="agent",
        summary="Tried to bypass a gated transition.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "update_nodes",
                "nodes": [
                    {
                        "id": "hyp/replanning-restores-plasticity",
                        "changes": {"status": "supported"},
                    }
                ],
            }
        ],
    )


def seed_patch() -> Patch:
    return Patch(
        kind="seed",
        author="agent",
        summary="Seeded the project question and initial hypothesis.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": "rq/learning-after-shift",
                        "type": "research_question",
                        "title": "Learning after task shift",
                        "question": "Can the learner retain its ability to adapt after the task changes?",
                        "motivation": "Persistent agents encounter repeated changes.",
                        "scope": "Matched compute and update histories.",
                        "status": "open",
                    },
                    {
                        "id": "hyp/replanning-restores-plasticity",
                        "type": "hypothesis",
                        "title": "Replanning restores plasticity",
                        "statement": "Search-time replanning restores future learning ability.",
                        "rationale": "It may reduce dependence on stale value features.",
                        "predictions": ["The unseen-task learning curve recovers."],
                        "status": "proposed",
                    },
                ],
            },
            {
                "op": "create_edges",
                "edges": [
                    {
                        "source": "rq/learning-after-shift",
                        "target": "hyp/replanning-restores-plasticity",
                        "relation": "has_hypothesis",
                    }
                ],
            },
        ],
    )
