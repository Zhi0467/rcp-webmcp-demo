from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from rcp.service import RunRequest
from rcp.storage import (
    AppStore,
    ProjectRecord,
    ResultViewConflict,
    ResultViewRecord,
)

_VIEW_ID = "0123456789abcdef01234567"
_CREATED = datetime(2026, 8, 12, 1, 2, 3, tzinfo=UTC)
_HTML = b"<!doctype html><html><body>original</body></html>"
_REVISED_HTML = b"<!doctype html><html><body>revised</body></html>"


def _view(
    *,
    view_id: str = _VIEW_ID,
    project_id: str = "project-one",
    experiment_id: str = "experiment-one",
    chat_id: str = "chat-one",
    expires_at: datetime | None = None,
) -> ResultViewRecord:
    created_at = _CREATED.isoformat()
    return ResultViewRecord(
        view_id=view_id,
        project_id=project_id,
        experiment_id=experiment_id,
        chat_id=chat_id,
        origin_operation_id="operation-create",
        latest_operation_id="operation-create",
        provider="codex",
        model="",
        reasoning="high",
        run_on="local",
        native_session_id="native-session",
        stage_host="",
        stage_root="/tmp/rcp-run.chat-one",
        source_name="throughput-pilot.html",
        content_sha256=hashlib.sha256(_HTML).hexdigest(),
        size_bytes=len(_HTML),
        created_at=created_at,
        updated_at=created_at,
        expires_at=(expires_at or _CREATED + timedelta(days=7)).isoformat(),
    )


def _project(project_id: str) -> ProjectRecord:
    return ProjectRecord(
        project_id=project_id,
        locator=f"/tmp/{project_id}/research.yaml",
        name=project_id,
        state_location=f"/tmp/{project_id}/.research",
        state_remote=False,
        added_at=_CREATED.isoformat(),
    )


def test_result_view_request_is_a_strict_create_or_revise_union() -> None:
    create = RunRequest.model_validate(
        {
            "mode": "work",
            "chat_scope": "node",
            "node_id": "experiment-one",
            "result_view": {"action": "create"},
        }
    )
    revise = RunRequest.model_validate(
        {
            "mode": "work",
            "chat_scope": "node",
            "node_id": "experiment-one",
            "result_view": {"action": "revise", "view_id": _VIEW_ID},
        }
    )

    assert create.result_view is not None and create.result_view.action == "create"
    assert revise.result_view is not None and revise.result_view.action == "revise"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RunRequest.model_validate(
            {
                "mode": "work",
                "node_id": "experiment-one",
                "result_view": {"action": "create", "view_id": _VIEW_ID},
            }
        )
    with pytest.raises(ValidationError, match="Field required"):
        RunRequest.model_validate(
            {
                "mode": "work",
                "node_id": "experiment-one",
                "result_view": {"action": "revise"},
            }
        )
    with pytest.raises(ValidationError):
        RunRequest.model_validate(
            {
                "mode": "work",
                "node_id": "experiment-one",
                "result_view": {"action": "revise", "view_id": "not-an-opaque-id"},
            }
        )


@pytest.mark.parametrize(
    "values",
    [
        {"mode": "discuss", "chat_scope": "node", "node_id": "experiment-one"},
        {"mode": "work", "chat_scope": "project", "node_id": "experiment-one"},
        {"mode": "work", "chat_scope": "node", "node_id": None},
    ],
)
def test_result_view_request_requires_node_scoped_work(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="node-scoped Work"):
        RunRequest.model_validate({**values, "result_view": {"action": "create"}})


def test_fresh_result_view_schema_stores_html_privately(tmp_path) -> None:
    path = tmp_path / "rcp.sqlite3"
    AppStore(path)

    with sqlite3.connect(path) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(result_views)")]
        indexes = {
            row[1] for row in connection.execute("PRAGMA index_list(result_views)").fetchall()
        }
    assert columns == [
        "view_id",
        "project_id",
        "experiment_id",
        "chat_id",
        "origin_operation_id",
        "latest_operation_id",
        "provider",
        "model",
        "reasoning",
        "run_on",
        "native_session_id",
        "stage_host",
        "stage_root",
        "source_name",
        "content_sha256",
        "size_bytes",
        "html",
        "created_at",
        "updated_at",
        "expires_at",
        "kept_filename",
        "kept_at",
    ]
    assert "result_views_project_experiment" in indexes
    assert "result_views_project_chat" in indexes
    assert not {"bytes", "content", "patch", "proposal", "revision"} & set(columns)


def test_actual_legacy_result_view_schema_migrates_before_indexes_are_created(tmp_path) -> None:
    path = tmp_path / "rcp.sqlite3"
    AppStore(path)
    legacy = _view()
    with sqlite3.connect(path) as connection:
        connection.execute("DROP INDEX result_views_project_experiment")
        connection.execute("DROP INDEX result_views_project_chat")
        connection.execute("DROP INDEX result_views_expiry")
        connection.execute("ALTER TABLE result_views DROP COLUMN html")
        connection.execute(
            """
            INSERT INTO result_views (
                view_id, project_id, experiment_id, chat_id,
                origin_operation_id, latest_operation_id,
                provider, model, reasoning, run_on,
                native_session_id, stage_host, stage_root, source_name,
                content_sha256, size_bytes, created_at, updated_at, expires_at,
                kept_filename, kept_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            tuple(legacy.model_dump(mode="python").values()),
        )

    migrated = AppStore(path)

    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(result_views)")}
        indexes = {row[1] for row in connection.execute("PRAGMA index_list(result_views)")}
        stored = connection.execute(
            "SELECT html FROM result_views WHERE view_id = ?", (legacy.view_id,)
        ).fetchone()
    assert "html" in columns
    assert indexes >= {
        "result_views_project_experiment",
        "result_views_project_chat",
        "result_views_expiry",
    }
    assert stored == ("",)
    assert migrated.result_view_for_diagnostics(legacy.view_id) == legacy
    with pytest.raises(ValueError, match="size does not match"):
        migrated.result_view_bytes(
            legacy.view_id,
            expected_content_sha256=legacy.content_sha256,
        )


def test_result_view_insert_fetch_and_filtered_listing(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    first = store.create_result_view(_view(), html=_HTML)
    second = store.create_result_view(
        _view(
            view_id="1123456789abcdef01234567",
            experiment_id="experiment-two",
            chat_id="chat-two",
        ),
        html=_HTML,
    )

    assert store.result_view(first.view_id, as_of=_CREATED) == first
    assert {item.view_id for item in store.list_result_views("project-one", as_of=_CREATED)} == {
        first.view_id,
        second.view_id,
    }
    assert store.list_result_views(
        "project-one", experiment_id="experiment-one", as_of=_CREATED
    ) == [first]
    assert store.list_result_views("project-one", chat_id="chat-two", as_of=_CREATED) == [second]


def test_expired_temporary_view_is_hidden_but_available_for_diagnostics(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    record = store.create_result_view(_view(expires_at=_CREATED + timedelta(hours=1)), html=_HTML)
    after_expiry = _CREATED + timedelta(hours=2)

    assert store.result_view_for_diagnostics(record.view_id) == record
    assert store.result_view(record.view_id, as_of=after_expiry) is None
    assert store.list_result_views(record.project_id, as_of=after_expiry) == []
    assert store.result_view_for_diagnostics(record.view_id) == record
    assert store.result_view(record.view_id, include_expired=True, as_of=after_expiry) == record


def test_kept_view_survives_expiry_and_keep_is_idempotent(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    record = store.create_result_view(_view(expires_at=_CREATED + timedelta(hours=1)), html=_HTML)
    kept = store.mark_result_view_kept(
        record.view_id,
        expected_content_sha256=record.content_sha256,
        kept_filename="throughput-project-26-08-12.html",
        kept_at=(_CREATED + timedelta(minutes=1)).isoformat(),
    )
    retried = store.mark_result_view_kept(
        record.view_id,
        expected_content_sha256=record.content_sha256,
        kept_filename="must-not-replace-the-first-name.html",
        kept_at=(_CREATED + timedelta(minutes=2)).isoformat(),
    )

    assert retried == kept
    assert kept.kept_filename == "throughput-project-26-08-12.html"
    assert store.list_result_views(record.project_id, as_of=_CREATED + timedelta(days=30)) == [kept]


def test_active_chat_extends_only_unkept_view_retention(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    unkept = store.create_result_view(_view(expires_at=_CREATED + timedelta(hours=1)), html=_HTML)
    kept_source = store.create_result_view(
        _view(
            view_id="2123456789abcdef01234567",
            expires_at=_CREATED + timedelta(hours=1),
        ),
        html=_HTML,
    )
    kept = store.mark_result_view_kept(
        kept_source.view_id,
        expected_content_sha256=kept_source.content_sha256,
        kept_filename="kept-project-26-08-12.html",
        kept_at=(_CREATED + timedelta(minutes=1)).isoformat(),
    )
    extended = _CREATED + timedelta(days=8)

    assert (
        store.refresh_result_view_expiry(
            unkept.project_id,
            unkept.chat_id,
            expires_at=extended.isoformat(),
            as_of=_CREATED,
        )
        == 1
    )
    assert (
        store.refresh_result_view_expiry(
            unkept.project_id,
            unkept.chat_id,
            expires_at=(_CREATED + timedelta(minutes=30)).isoformat(),
            as_of=_CREATED,
        )
        == 0
    )
    assert store.result_view_for_diagnostics(unkept.view_id).expires_at == extended.isoformat()
    assert store.result_view_for_diagnostics(kept.view_id).expires_at == kept.expires_at


def test_active_chat_cannot_revive_an_already_expired_unkept_view(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    created_at = datetime(2000, 1, 1, tzinfo=UTC)
    expired_at = created_at + timedelta(days=1)
    record = store.create_result_view(
        ResultViewRecord.model_validate(
            {
                **_view().model_dump(mode="python"),
                "created_at": created_at.isoformat(),
                "updated_at": created_at.isoformat(),
                "expires_at": expired_at.isoformat(),
            }
        ),
        html=_HTML,
    )

    refreshed = store.refresh_result_view_expiry(
        record.project_id,
        record.chat_id,
        expires_at=(_CREATED + timedelta(days=8)).isoformat(),
        as_of=_CREATED,
    )

    assert refreshed == 0
    unchanged = store.result_view_for_diagnostics(record.view_id)
    assert unchanged is not None
    assert unchanged.expires_at == expired_at.isoformat()
    assert store.result_view(record.view_id, as_of=_CREATED) is None


def test_revision_uses_digest_cas_and_preserves_view_identity(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    record = store.create_result_view(_view(), html=_HTML)
    updated_at = (_CREATED + timedelta(minutes=5)).isoformat()
    revised = store.revise_result_view(
        record.view_id,
        expected_content_sha256=record.content_sha256,
        latest_operation_id="operation-revise",
        content_sha256=hashlib.sha256(_REVISED_HTML).hexdigest(),
        size_bytes=len(_REVISED_HTML),
        html=_REVISED_HTML,
        updated_at=updated_at,
        expires_at=(_CREATED + timedelta(days=8)).isoformat(),
    )

    assert revised.view_id == record.view_id
    assert revised.origin_operation_id == record.origin_operation_id
    assert revised.latest_operation_id == "operation-revise"
    assert revised.content_sha256 == hashlib.sha256(_REVISED_HTML).hexdigest()
    with pytest.raises(ResultViewConflict, match="changed before"):
        store.revise_result_view(
            record.view_id,
            expected_content_sha256=record.content_sha256,
            latest_operation_id="operation-stale",
            content_sha256=hashlib.sha256(b"<html>stale</html>").hexdigest(),
            size_bytes=len(b"<html>stale</html>"),
            html=b"<html>stale</html>",
            updated_at=(_CREATED + timedelta(minutes=6)).isoformat(),
            expires_at=(_CREATED + timedelta(days=8)).isoformat(),
        )
    assert store.result_view_for_diagnostics(record.view_id) == revised


def test_result_view_bytes_are_bounded_digest_validated_and_updated_atomically(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    record = store.create_result_view(_view(), html=_HTML)

    assert (
        store.result_view_bytes(
            record.view_id,
            expected_content_sha256=record.content_sha256,
        )
        == _HTML
    )
    with pytest.raises(ResultViewConflict, match="changed before"):
        store.result_view_bytes(record.view_id, expected_content_sha256="f" * 64)

    with pytest.raises(ValueError, match="digest does not match"):
        store.revise_result_view(
            record.view_id,
            expected_content_sha256=record.content_sha256,
            latest_operation_id="operation-invalid",
            content_sha256=hashlib.sha256(_REVISED_HTML).hexdigest(),
            size_bytes=len(_REVISED_HTML),
            html=_REVISED_HTML.replace(b"revised", b"changed"),
            updated_at=(_CREATED + timedelta(minutes=1)).isoformat(),
            expires_at=(_CREATED + timedelta(days=8)).isoformat(),
        )

    assert store.result_view_for_diagnostics(record.view_id) == record
    assert (
        store.result_view_bytes(
            record.view_id,
            expected_content_sha256=record.content_sha256,
        )
        == _HTML
    )


def test_invalid_create_does_not_leave_metadata_without_bytes(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    record = _view()

    with pytest.raises(ValueError, match="size does not match"):
        store.create_result_view(record, html=b"<html>short</html>")

    assert store.result_view_for_diagnostics(record.view_id) is None


def test_operational_prune_deletes_expired_unkept_html_with_its_record(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    expires_at = _CREATED + timedelta(hours=1)
    expired = store.create_result_view(_view(expires_at=expires_at), html=_HTML)
    kept_source = store.create_result_view(
        _view(view_id="3123456789abcdef01234567", expires_at=expires_at),
        html=_HTML,
    )
    kept = store.mark_result_view_kept(
        kept_source.view_id,
        expected_content_sha256=kept_source.content_sha256,
        kept_filename="kept-project-26-08-12.html",
        kept_at=(_CREATED + timedelta(minutes=1)).isoformat(),
    )

    result = store.prune_operational_storage(now=_CREATED + timedelta(hours=2))

    assert result["result_views"] == 1
    assert store.result_view_for_diagnostics(expired.view_id) is None
    assert store.result_view_for_diagnostics(kept.view_id) == kept
    assert (
        store.result_view_bytes(
            kept.view_id,
            expected_content_sha256=kept.content_sha256,
        )
        == _HTML
    )


def test_named_active_result_view_revision_query_preserves_route_policy(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    record = store.create_result_view(_view(), html=_HTML)
    request = json.dumps(
        {
            "chat_id": record.chat_id,
            "result_view": {"action": "revise", "view_id": record.view_id},
        }
    )
    with store.connection() as connection:
        connection.execute(
            """
            INSERT INTO graph_runs (
                operation_id, project_id, kind, status, request_json,
                created_at, updated_at, status_message
            ) VALUES (?, ?, 'node_chat', 'queued', ?, ?, ?, 'queued')
            """,
            (
                "revision-operation",
                record.project_id,
                request,
                record.created_at,
                record.updated_at,
            ),
        )

    assert store.has_active_result_view_revision(record) is True
    with store.connection() as connection:
        connection.execute(
            "UPDATE graph_runs SET status = 'succeeded' WHERE operation_id = ?",
            ("revision-operation",),
        )
    assert store.has_active_result_view_revision(record) is False


def test_revision_after_keep_conflicts_without_changing_kept_metadata(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    record = store.create_result_view(_view(), html=_HTML)
    kept = store.mark_result_view_kept(
        record.view_id,
        expected_content_sha256=record.content_sha256,
        kept_filename="throughput-project-26-08-12.html",
        kept_at=(_CREATED + timedelta(minutes=1)).isoformat(),
    )

    with pytest.raises(ResultViewConflict, match="kept result view"):
        store.revise_result_view(
            record.view_id,
            expected_content_sha256=record.content_sha256,
            latest_operation_id="operation-revise-after-keep",
            content_sha256=hashlib.sha256(_REVISED_HTML).hexdigest(),
            size_bytes=len(_REVISED_HTML),
            html=_REVISED_HTML,
            updated_at=(_CREATED + timedelta(minutes=2)).isoformat(),
            expires_at=(_CREATED + timedelta(days=8)).isoformat(),
        )

    assert store.result_view_for_diagnostics(record.view_id) == kept
    descriptor = store.result_view_descriptor(kept, as_of=_CREATED)
    assert descriptor.state == "kept"
    assert descriptor.kept_filename == kept.kept_filename
    assert descriptor.can_revise is False


def test_keep_after_revision_conflicts_without_exposing_stale_keep_metadata(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    record = store.create_result_view(_view(), html=_HTML)
    revised = store.revise_result_view(
        record.view_id,
        expected_content_sha256=record.content_sha256,
        latest_operation_id="operation-revise-before-keep",
        content_sha256=hashlib.sha256(_REVISED_HTML).hexdigest(),
        size_bytes=len(_REVISED_HTML),
        html=_REVISED_HTML,
        updated_at=(_CREATED + timedelta(minutes=1)).isoformat(),
        expires_at=(_CREATED + timedelta(days=8)).isoformat(),
    )

    with pytest.raises(ResultViewConflict, match="changed before Keep"):
        store.mark_result_view_kept(
            record.view_id,
            expected_content_sha256=record.content_sha256,
            kept_filename="stale-copy.html",
            kept_at=(_CREATED + timedelta(minutes=2)).isoformat(),
        )

    assert store.result_view_for_diagnostics(record.view_id) == revised
    descriptor = store.result_view_descriptor(revised, as_of=_CREATED)
    assert descriptor.state == "temporary"
    assert descriptor.kept_filename is None
    assert descriptor.kept_at is None
    assert descriptor.can_revise is True


def test_project_identity_migration_and_deletion_include_result_views(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    legacy_id = "legacy-project"
    canonical_id = str(uuid.uuid4())
    store.upsert_project(_project(legacy_id))
    record = store.create_result_view(_view(project_id=legacy_id), html=_HTML)

    store.migrate_project_identity(legacy_id, canonical_id, store.space_id)

    migrated = store.result_view_for_diagnostics(record.view_id)
    assert migrated is not None and migrated.project_id == canonical_id
    counts = store.delete_project_records(canonical_id)
    assert counts["result_views"] == 1
    assert store.result_view_for_diagnostics(record.view_id) is None


def test_public_descriptor_exposes_no_private_binding_fields(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    record = store.create_result_view(_view(), html=_HTML)

    descriptor = store.result_view_descriptor(record, as_of=_CREATED)

    assert descriptor.model_dump() == {
        "view_id": record.view_id,
        "chat_id": record.chat_id,
        "experiment_id": record.experiment_id,
        "name": record.source_name,
        "media_type": "text/html",
        "state": "temporary",
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "expires_at": record.expires_at,
        "kept_filename": None,
        "kept_at": None,
        "can_revise": True,
    }
    private_fields = {
        "project_id",
        "origin_operation_id",
        "latest_operation_id",
        "provider",
        "model",
        "reasoning",
        "run_on",
        "native_session_id",
        "stage_host",
        "stage_root",
        "source_name",
        "content_sha256",
        "size_bytes",
    }
    assert not private_fields & descriptor.model_dump().keys()


@pytest.mark.parametrize("source_name", ["nested/view.html", "view.htm", "view.png"])
def test_result_view_record_requires_one_plain_html_source(source_name: str) -> None:
    with pytest.raises(ValidationError):
        ResultViewRecord.model_validate({**_view().model_dump(), "source_name": source_name})
