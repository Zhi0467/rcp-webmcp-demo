from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from rcp.core.models import AuthorizedHuman
from rcp.core.transition_models import GraphHeadRef, GraphTargetRef
from rcp.project_transfer import PROJECT_TRANSFER_ARCHIVE_CODEC
from rcp.server_ops.backup_models import (
    BACKUP_APP_DATA_CAPTURED,
    BACKUP_APP_DATA_DATABASE,
    BACKUP_APP_DATA_DEFERRED,
    BACKUP_APP_DATA_EXCLUSIONS,
    BACKUP_RESEARCH_CANONICAL_ROOTS,
    BACKUP_RESEARCH_DELEGATED_ROOTS,
    BACKUP_RESEARCH_EXCLUSIONS,
)
from rcp.storage import AppStore
from rcp.transfer import (
    TRANSFER_APP_DATA_CONTROL_ROOTS,
    TRANSFER_APP_DATA_EXCLUDED_ROOTS,
    TRANSFER_APP_DATA_PROJECT_SOURCE_ROOTS,
    TRANSFER_APP_DATA_TYPED_ROOTS,
    TRANSFER_ARCHIVE_CODEC,
    TRANSFER_GLOBAL_TABLES,
    TRANSFER_RESEARCH_CANONICAL_ROOTS,
    TRANSFER_RESEARCH_DELEGATED_ROOTS,
    TRANSFER_RESEARCH_EXCLUDED_ROOTS,
    TRANSFER_RESEARCH_PROVENANCE_ROOTS,
    TransferArchiveActor,
    TransferArchiveAttribution,
    TransferArchiveDiagnostic,
    TransferArchiveEntry,
    TransferArchiveEnvelope,
    TransferArchiveManifest,
    TransferGraphHead,
    TransferGraphTarget,
    inspect_project_linked_tables,
    inspect_transfer_app_data_roots,
    inspect_transfer_research_roots,
    inspect_transfer_table_inventory,
)

PROJECT_ID = "11111111-1111-4111-8111-111111111111"
SOURCE_SPACE_ID = "22222222-2222-4222-8222-222222222222"
TARGET_SPACE_ID = "33333333-3333-4333-8333-333333333333"
SOURCE_REQUEST_ID = "44444444-4444-4444-8444-444444444444"
TARGET_REQUEST_ID = "55555555-5555-4555-8555-555555555555"
ARCHIVE_ACTOR_ID = "66666666-6666-4666-8666-666666666666"
SOURCE_USER_ID = "77777777-7777-4777-8777-777777777777"
BRANCH_ID = "88888888-8888-4888-8888-888888888888"
TARGET_ARCHIVE_ACTOR_ID = "99999999-9999-4999-8999-999999999999"
TARGET_USER_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
UNRELATED_SPACE_ID = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
CREATED_AT = datetime(2026, 8, 31, 18, 0, tzinfo=UTC)

PROJECT_LINKED_TABLES = {
    "agent_usage",
    "auto_research_apply_results",
    "auto_research_child_admissions",
    "auto_research_child_experiments",
    "auto_research_child_work",
    "auto_research_child_work_attempts",
    "auto_research_command_files",
    "auto_research_episodes",
    "auto_research_experiment_invocations",
    "auto_research_finish_receipts",
    "auto_research_inbox_receipts",
    "auto_research_invocations",
    "auto_research_lifecycle_notices",
    "auto_research_messages",
    "auto_research_recoveries",
    "chat_session_contexts",
    "episode_invocations",
    "episode_report_attempts",
    "episode_reports",
    "episode_wrapups",
    "episodes",
    "experiment_episode_state",
    "graph_run_contracts",
    "graph_run_events",
    "graph_run_outputs",
    "graph_run_receipts",
    "graph_runs",
    "graph_watcher_reconciliation",
    "paper_drafts",
    "project_aliases",
    "project_invitations",
    "project_members",
    "project_provisioning_requests",
    "project_provisioning_step_receipts",
    "project_transfer_activations",
    "project_transfer_import_configurations",
    "project_transfer_imports",
    "project_transfer_proofs",
    "project_transfer_requests",
    "project_transfer_restore_reentries",
    "project_transfer_uploads",
    "projects",
    "result_views",
    "watchers",
    "writing_sessions",
}


def _entry(path: str, group: str, *, size: int = 10, digest: str | None = None):
    return TransferArchiveEntry(
        archive_path=path,
        group=group,
        sha256=digest or hashlib.sha256(path.encode()).hexdigest(),
        size_bytes=size,
    )


def _manifest() -> TransferArchiveManifest:
    source_manifest_sha256 = hashlib.sha256(b"source manifest").hexdigest()
    source_release_proof_sha256 = hashlib.sha256(b"x" * 32).hexdigest()
    provider_history_sha256 = hashlib.sha256(b"native").hexdigest()
    entries = tuple(
        sorted(
            (
                _entry("artifacts/kept.txt", "kept_artifact"),
                _entry(f"canonical/branches/{BRANCH_ID}/branch.json", "canonical_history"),
                _entry(
                    f"canonical/branches/{BRANCH_ID}/patches/000001.json",
                    "canonical_history",
                ),
                _entry("canonical/patches/000001.json", "canonical_history"),
                _entry("canonical/patches/000002.json", "canonical_history"),
                _entry("canonical/scope-base.json", "canonical_history"),
                _entry(
                    "control/source-release-proof.bin",
                    "source_release_proof",
                    size=32,
                    digest=source_release_proof_sha256,
                ),
                _entry(f"chats/{uuid.uuid4()}.jsonl", "rcp_chat"),
                _entry("facts/inputs/data.bin", "fact"),
                _entry("paper/introduction.md", "paper_introduction"),
                _entry(
                    f"provider-history/codex/{provider_history_sha256}",
                    "provider_history",
                    digest=provider_history_sha256,
                ),
                _entry(
                    "provenance/manifest.toml",
                    "source_manifest_provenance",
                    digest=source_manifest_sha256,
                ),
                _entry("records/tasks.jsonl", "operational_records"),
                _entry("result-views/kept.html", "legacy_kept_result_view"),
            ),
            key=lambda item: item.archive_path,
        )
    )
    return TransferArchiveManifest(
        project_id=PROJECT_ID,
        source_space_id=SOURCE_SPACE_ID,
        target_space_id=TARGET_SPACE_ID,
        source_request_id=SOURCE_REQUEST_ID,
        target_request_id=TARGET_REQUEST_ID,
        source_rcp_version="0.1.0.dev0+main",
        source_schema_generation=1,
        source_configuration_sha256="a" * 64,
        source_manifest_sha256=source_manifest_sha256,
        source_release_proof_sha256=source_release_proof_sha256,
        target_activation_proof_sha256="f" * 64,
        main_head=TransferGraphHead.capture(GraphHeadRef(revision=2, transition_id="b" * 64)),
        branch_heads=(
            TransferGraphHead.capture(
                GraphHeadRef(
                    target=GraphTargetRef(kind="branch", branch_id=BRANCH_ID),
                    revision=1,
                    transition_id="c" * 64,
                )
            ),
        ),
        attributions=(
            TransferArchiveAttribution(
                archive_actor_id=ARCHIVE_ACTOR_ID,
                source_actor=TransferArchiveActor.capture(
                    AuthorizedHuman(
                        space_id=SOURCE_SPACE_ID,
                        user_id=SOURCE_USER_ID,
                        display_name="Z",
                    )
                ),
            ),
            TransferArchiveAttribution(
                archive_actor_id=TARGET_ARCHIVE_ACTOR_ID,
                source_actor=TransferArchiveActor.capture(
                    AuthorizedHuman(
                        space_id=TARGET_SPACE_ID,
                        user_id=TARGET_USER_ID,
                        display_name="Alice",
                    )
                ),
            ),
        ),
        diagnostics=(
            TransferArchiveDiagnostic(
                code="provider_history_unmatched",
                message="One unmatched provider conversation was omitted.",
            ),
        ),
        entries=entries,
        payload_size_bytes=sum(item.size_bytes for item in entries),
        created_at=CREATED_AT,
    )


def _replace_manifest(manifest: TransferArchiveManifest, **changes: object):
    return TransferArchiveManifest.model_validate_json(
        json.dumps({**manifest.model_dump(mode="json"), **changes})
    )


def test_transfer_root_policy_is_closed_against_current_concrete_owners() -> None:
    app_groups = (
        TRANSFER_APP_DATA_TYPED_ROOTS,
        TRANSFER_APP_DATA_PROJECT_SOURCE_ROOTS,
        TRANSFER_APP_DATA_CONTROL_ROOTS,
        TRANSFER_APP_DATA_EXCLUDED_ROOTS,
    )
    assert set().union(*app_groups) == (
        {BACKUP_APP_DATA_DATABASE}
        | BACKUP_APP_DATA_CAPTURED
        | BACKUP_APP_DATA_DEFERRED
        | BACKUP_APP_DATA_EXCLUSIONS
    )
    assert sum(len(group) for group in app_groups) == len(set().union(*app_groups))

    research_groups = (
        TRANSFER_RESEARCH_PROVENANCE_ROOTS,
        TRANSFER_RESEARCH_CANONICAL_ROOTS,
        TRANSFER_RESEARCH_DELEGATED_ROOTS,
        TRANSFER_RESEARCH_EXCLUDED_ROOTS,
    )
    assert set().union(*research_groups) == (
        BACKUP_RESEARCH_CANONICAL_ROOTS
        | BACKUP_RESEARCH_DELEGATED_ROOTS
        | BACKUP_RESEARCH_EXCLUSIONS
    )
    assert sum(len(group) for group in research_groups) == len(set().union(*research_groups))


def test_read_only_root_inventory_reports_unknown_durable_roots(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "rcp.sqlite3").write_bytes(b"database")
    (data_dir / "project-sources").mkdir()
    (data_dir / "transfer-exports").mkdir()
    (data_dir / "run-stage").mkdir()
    (data_dir / "future-project-history").mkdir()

    app_data = inspect_transfer_app_data_roots(data_dir)
    assert app_data.typed_entries == ("rcp.sqlite3",)
    assert app_data.project_source_entries == ("project-sources",)
    assert app_data.control_entries == ("transfer-exports",)
    assert app_data.excluded_entries == ("run-stage",)
    assert app_data.unclassified_entries == ("future-project-history",)
    assert app_data.complete is False

    research_dir = tmp_path / ".research"
    research_dir.mkdir()
    for name in ("manifest.toml", "patches", "chat", "graph.json", "future-history"):
        path = research_dir / name
        path.write_text("x", encoding="utf-8") if "." in name else path.mkdir()
    research = inspect_transfer_research_roots(research_dir)
    assert research.provenance_entries == ("manifest.toml",)
    assert research.canonical_entries == ("patches",)
    assert research.delegated_entries == ("chat",)
    assert research.excluded_entries == ("graph.json",)
    assert research.unclassified_entries == ("future-history",)


def test_project_linked_sqlite_inventory_is_exact_and_follows_child_tables(tmp_path: Path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    with store.connection() as connection:
        inventory = inspect_transfer_table_inventory(connection)
        assert set(inventory.project_linked_tables) == PROJECT_LINKED_TABLES
        assert set(inventory.global_tables) == TRANSFER_GLOBAL_TABLES
        assert inventory.unclassified_tables == ()
        assert inventory.complete is True
        assert set(inspect_project_linked_tables(connection)) == PROJECT_LINKED_TABLES
        connection.execute(
            """
            CREATE TABLE future_project_history (
                event_id TEXT PRIMARY KEY,
                operation_id TEXT NOT NULL,
                FOREIGN KEY(operation_id) REFERENCES graph_runs(operation_id)
            )
            """
        )
        assert "future_project_history" in inspect_project_linked_tables(connection)
        connection.execute("CREATE TABLE future_global_state (state_id TEXT PRIMARY KEY)")
        changed = inspect_transfer_table_inventory(connection)
        assert changed.unclassified_tables == ("future_global_state",)
        assert changed.complete is False


def test_manifest_binds_every_identity_head_entry_and_exact_payload_size() -> None:
    manifest = _manifest()
    encoded = manifest.canonical_bytes()

    assert json.loads(encoded)["archive_codec"] == TRANSFER_ARCHIVE_CODEC
    assert manifest.sha256() == hashlib.sha256(encoded).hexdigest()
    assert encoded == manifest.canonical_bytes()
    assert b"native_session" not in encoded
    envelope = TransferArchiveEnvelope.bind(
        manifest,
        archive_sha256="d" * 64,
        archive_size_bytes=manifest.payload_size_bytes + len(encoded) + 4096,
    )
    assert envelope.manifest_sha256 == manifest.sha256()
    assert envelope.manifest_size_bytes == len(encoded)
    assert envelope.payload_size_bytes == manifest.payload_size_bytes
    envelope.verify_manifest(manifest)
    with pytest.raises(ValueError, match="does not match"):
        envelope.verify_manifest(_replace_manifest(manifest, source_rcp_version="later"))


def test_manifest_nested_identity_and_head_snapshots_are_immutable() -> None:
    manifest = _manifest()
    encoded = manifest.canonical_bytes()

    with pytest.raises(ValidationError, match="frozen"):
        manifest.main_head.revision = 7
    with pytest.raises(ValidationError, match="frozen"):
        manifest.branch_heads[0].target.branch_id = SOURCE_REQUEST_ID
    with pytest.raises(ValidationError, match="frozen"):
        manifest.attributions[0].source_actor.display_name = "Mallory"

    assert manifest.canonical_bytes() == encoded


def test_attribution_accepts_both_transfer_spaces_but_rejects_an_unrelated_space() -> None:
    manifest = _manifest()
    unrelated = TransferArchiveAttribution(
        archive_actor_id=ARCHIVE_ACTOR_ID,
        source_actor=TransferArchiveActor.capture(
            AuthorizedHuman(
                space_id=UNRELATED_SPACE_ID,
                user_id=SOURCE_USER_ID,
                display_name="Mallory",
            )
        ),
    )
    with pytest.raises(ValidationError, match="unrelated space"):
        _replace_manifest(manifest, attributions=[unrelated.model_dump(mode="json")])


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"payload_size_bytes": 1}, "payload byte total"),
        ({"source_manifest_sha256": "e" * 64}, "source-manifest provenance"),
        ({"source_release_proof_sha256": "f" * 64}, "source-release proof"),
        ({"main_head": TransferGraphHead(revision=3).model_dump(mode="json")}, "main head"),
        ({"branch_heads": []}, "branch heads"),
    ],
)
def test_manifest_rejects_boundary_mismatches(changes: dict[str, object], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        _replace_manifest(_manifest(), **changes)


def test_manifest_accepts_a_branch_tail_that_starts_after_its_main_base() -> None:
    manifest = _manifest()
    entries = tuple(
        entry
        for entry in manifest.entries
        if entry.archive_path != f"canonical/branches/{BRANCH_ID}/patches/000001.json"
    ) + (
        _entry(
            f"canonical/branches/{BRANCH_ID}/patches/000002.json",
            "canonical_history",
        ),
    )
    entries = tuple(sorted(entries, key=lambda entry: entry.archive_path))

    moved = _replace_manifest(
        manifest,
        branch_heads=[
            TransferGraphHead(
                target=TransferGraphTarget(kind="branch", branch_id=BRANCH_ID),
                revision=2,
            ).model_dump(mode="json")
        ],
        entries=[entry.model_dump(mode="json") for entry in entries],
        payload_size_bytes=sum(entry.size_bytes for entry in entries),
    )

    assert moved.branch_heads[0].revision == 2


def test_entry_groups_cannot_smuggle_materializations_credentials_or_target_proof() -> None:
    for path, group in (
        ("canonical/graph.json", "canonical_history"),
        ("credentials/token", "fact"),
        ("control/target-activation-proof.bin", "source_release_proof"),
        ("provider-history/codex/not-content-addressed", "provider_history"),
    ):
        with pytest.raises(ValidationError, match="path|credentials"):
            _entry(path, group)

    with pytest.raises(ValidationError, match="credential-shaped"):
        TransferArchiveDiagnostic(
            code="provider_unreadable",
            message="token=github_pat_abcdefghijklmnopqrstuvwxyz",
        )

    with pytest.raises(ValidationError, match="path"):
        _entry(
            f"provider-history/codex/{'a' * 64}",
            "provider_history",
            digest="b" * 64,
        )


def test_source_negotiation_and_archive_manifest_share_one_codec_constant() -> None:
    assert PROJECT_TRANSFER_ARCHIVE_CODEC == TRANSFER_ARCHIVE_CODEC
