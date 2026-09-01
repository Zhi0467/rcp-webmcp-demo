from __future__ import annotations

import hashlib
import stat
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rcp.core.models import AuthorizedHuman
from rcp.transfer import (
    TransferArchiveActor,
    TransferArchiveAttribution,
    TransferArchiveEntry,
    TransferArchiveManifest,
    TransferGraphHead,
)
from rcp.transfer.source import (
    discard_transfer_archive_stage,
    read_transfer_archive,
    seal_transfer_archive,
    stage_transfer_archive,
)

PROJECT_ID = "11111111-1111-4111-8111-111111111111"
SOURCE_SPACE_ID = "22222222-2222-4222-8222-222222222222"
TARGET_SPACE_ID = "33333333-3333-4333-8333-333333333333"
SOURCE_REQUEST_ID = "44444444-4444-4444-8444-444444444444"
TARGET_REQUEST_ID = "55555555-5555-4555-8555-555555555555"
SOURCE_USER_ID = "66666666-6666-4666-8666-666666666666"


def _entry(path: str, group: str, payload: bytes) -> TransferArchiveEntry:
    return TransferArchiveEntry(
        archive_path=path,
        group=group,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def _manifest_and_payloads() -> tuple[TransferArchiveManifest, dict[str, bytes]]:
    payloads = {
        "canonical/patches/000001.json": b'{"revision":1}\n',
        "canonical/scope-base.json": b'{"scope":[]}\n',
        "control/source-release-proof.bin": b"p" * 32,
        "provenance/manifest.toml": b"[project]\nid = 'source'\n",
        "records/tasks.jsonl": b'{"operation_id":"one"}\n',
    }
    entries = (
        _entry(
            "canonical/patches/000001.json",
            "canonical_history",
            payloads["canonical/patches/000001.json"],
        ),
        _entry(
            "canonical/scope-base.json", "canonical_history", payloads["canonical/scope-base.json"]
        ),
        _entry(
            "control/source-release-proof.bin",
            "source_release_proof",
            payloads["control/source-release-proof.bin"],
        ),
        _entry(
            "provenance/manifest.toml",
            "source_manifest_provenance",
            payloads["provenance/manifest.toml"],
        ),
        _entry("records/tasks.jsonl", "operational_records", payloads["records/tasks.jsonl"]),
    )
    entries = tuple(sorted(entries, key=lambda entry: entry.archive_path))
    manifest = TransferArchiveManifest(
        project_id=PROJECT_ID,
        source_space_id=SOURCE_SPACE_ID,
        target_space_id=TARGET_SPACE_ID,
        source_request_id=SOURCE_REQUEST_ID,
        target_request_id=TARGET_REQUEST_ID,
        source_rcp_version="0.1.0.dev0+main",
        source_schema_generation=1,
        source_configuration_sha256="a" * 64,
        source_manifest_sha256=hashlib.sha256(payloads["provenance/manifest.toml"]).hexdigest(),
        source_release_proof_sha256=hashlib.sha256(
            payloads["control/source-release-proof.bin"]
        ).hexdigest(),
        target_activation_proof_sha256="b" * 64,
        main_head=TransferGraphHead(revision=1, transition_id="c" * 64),
        branch_heads=(),
        attributions=(
            TransferArchiveAttribution(
                archive_actor_id=SOURCE_USER_ID,
                source_actor=TransferArchiveActor.capture(
                    AuthorizedHuman(
                        space_id=SOURCE_SPACE_ID,
                        user_id=SOURCE_USER_ID,
                        display_name="Z",
                    )
                ),
            ),
        ),
        diagnostics=(),
        entries=entries,
        payload_size_bytes=sum(len(payload) for payload in payloads.values()),
        created_at=datetime(2026, 8, 31, 18, 0, tzinfo=UTC),
    )
    return manifest, payloads


def _capture(root: Path, payloads: dict[str, bytes]) -> None:
    root.mkdir(mode=0o700, parents=True)
    for relative, payload in payloads.items():
        destination = root / relative
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        current = destination.parent
        while current != root:
            current.chmod(0o700)
            current = current.parent
        destination.write_bytes(payload)
        destination.chmod(0o400)


def _sealed_fixture(tmp_path: Path):
    manifest, payloads = _manifest_and_payloads()
    capture = tmp_path / "capture"
    _capture(capture, payloads)
    destination_parent = tmp_path / "exports"
    destination_parent.mkdir(mode=0o700)
    sealed = seal_transfer_archive(
        manifest=manifest,
        capture_root=capture,
        destination=destination_parent / "request.rcp-transfer",
    )
    return manifest, payloads, sealed


def test_source_archive_is_deterministic_and_contains_only_declared_files(tmp_path: Path) -> None:
    manifest, payloads, sealed = _sealed_fixture(tmp_path / "first")
    second_manifest, second_payloads = _manifest_and_payloads()
    second_capture = tmp_path / "second" / "capture"
    _capture(second_capture, second_payloads)
    second_parent = tmp_path / "second" / "exports"
    second_parent.mkdir(mode=0o700)
    second = seal_transfer_archive(
        manifest=second_manifest,
        capture_root=second_capture,
        destination=second_parent / "request.rcp-transfer",
    )

    assert sealed.archive_path.read_bytes() == second.archive_path.read_bytes()
    assert stat.S_IMODE(sealed.archive_path.stat().st_mode) == 0o600
    with tarfile.open(sealed.archive_path, mode="r:") as archive:
        assert [member.name for member in archive] == [
            "manifest.json",
            *sorted(payloads),
        ]
    readback = read_transfer_archive(sealed.archive_path, expected_envelope=sealed.envelope)
    assert readback.manifest == manifest
    assert readback.envelope == sealed.envelope


def test_existing_destination_is_never_replaced_and_failed_seal_cleans_partial(
    tmp_path: Path,
) -> None:
    manifest, payloads, sealed = _sealed_fixture(tmp_path / "good")
    original = sealed.archive_path.read_bytes()
    capture = tmp_path / "retry" / "capture"
    _capture(capture, payloads)
    with pytest.raises(FileExistsError, match="already exists"):
        seal_transfer_archive(
            manifest=manifest,
            capture_root=capture,
            destination=sealed.archive_path,
        )
    assert sealed.archive_path.read_bytes() == original

    bad_capture = tmp_path / "bad" / "capture"
    _capture(bad_capture, payloads)
    (bad_capture / "records/tasks.jsonl").chmod(0o600)
    (bad_capture / "records/tasks.jsonl").write_bytes(b"changed\n")
    (bad_capture / "records/tasks.jsonl").chmod(0o400)
    bad_parent = tmp_path / "bad" / "exports"
    bad_parent.mkdir(mode=0o700)
    with pytest.raises(ValueError, match="differs from its manifest"):
        seal_transfer_archive(
            manifest=manifest,
            capture_root=bad_capture,
            destination=bad_parent / "request.rcp-transfer",
        )
    assert not list(bad_parent.glob(".*.partial"))
    assert not (bad_parent / "request.rcp-transfer").exists()


def test_stage_streams_verified_archive_into_private_tree(tmp_path: Path) -> None:
    manifest, payloads, sealed = _sealed_fixture(tmp_path)
    staged = tmp_path / "staged"
    readback = stage_transfer_archive(
        sealed.archive_path,
        staged,
        expected_envelope=sealed.envelope,
    )
    assert readback.manifest == manifest
    assert stat.S_IMODE(staged.stat().st_mode) == 0o700
    assert not (staged / "manifest.json").exists()
    assert {
        path.relative_to(staged).as_posix() for path in staged.rglob("*") if path.is_file()
    } == set(payloads)
    for relative, payload in payloads.items():
        path = staged / relative
        assert path.read_bytes() == payload
        assert stat.S_IMODE(path.stat().st_mode) == 0o400

    discard_transfer_archive_stage(staged)
    assert not staged.exists()
    discard_transfer_archive_stage(staged)


@pytest.mark.parametrize("mutation", ["missing", "symlink", "wrong_mode", "corrupt"])
def test_readback_refuses_missing_symlink_wrong_mode_or_corrupt_archive(
    tmp_path: Path,
    mutation: str,
) -> None:
    _manifest, _payloads, sealed = _sealed_fixture(tmp_path / "fixture")
    candidate = tmp_path / f"{mutation}.rcp-transfer"
    if mutation == "missing":
        candidate = tmp_path / "does-not-exist.rcp-transfer"
    elif mutation == "symlink":
        candidate.symlink_to(sealed.archive_path)
    else:
        candidate.write_bytes(sealed.archive_path.read_bytes())
        candidate.chmod(0o644 if mutation == "wrong_mode" else 0o600)
        if mutation == "corrupt":
            data = bytearray(candidate.read_bytes())
            data[data.index(ord("p"))] ^= 1
            candidate.write_bytes(data)
            candidate.chmod(0o600)
    with pytest.raises(ValueError, match="missing|symlink|mode|bytes|corrupt|malformed"):
        read_transfer_archive(candidate, expected_envelope=sealed.envelope)


def test_stage_failure_removes_only_new_staging_root(tmp_path: Path) -> None:
    _manifest, _payloads, sealed = _sealed_fixture(tmp_path / "fixture")
    data = bytearray(sealed.archive_path.read_bytes())
    data[data.index(ord("p"))] ^= 1
    tampered = tmp_path / "tampered.rcp-transfer"
    tampered.write_bytes(data)
    tampered.chmod(0o600)
    staged = tmp_path / "staged"
    with pytest.raises(ValueError):
        stage_transfer_archive(tampered, staged)
    assert not staged.exists()
