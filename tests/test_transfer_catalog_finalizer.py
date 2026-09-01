from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from rcp.agents import AgentLauncher
from rcp.setup import ProjectSetupManager
from rcp.transfer.configuration import TransferTargetConfiguration
from rcp.transfer.importer import import_project_transfer

from .test_transfer_import import _archive_fixture


def _published_fixture(manifest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fixture = _archive_fixture(manifest, tmp_path, monkeypatch)
    import_project_transfer(
        fixture["catalog"],
        archive=fixture["archive"],
        envelope=fixture["envelope"],
        archive_root=fixture["archive_root"],
        target_configuration=fixture["configuration"],
    )
    request = fixture["target"].project_provisioning_request(fixture["archive"].target_request_id)
    assert request is not None
    setup = ProjectSetupManager(
        fixture["catalog"].data_dir,
        fixture["catalog"],
        AgentLauncher(),
    )
    return fixture, request, setup


def test_incoming_transfer_finalizer_accepts_retained_history_without_publishing(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, request, setup = _published_fixture(manifest, tmp_path, monkeypatch)
    target_state_root = fixture["target_state_root"]
    patch_payloads_before = {
        path.relative_to(target_state_root).as_posix(): path.read_bytes()
        for path in sorted((target_state_root / "patches").rglob("*.json"))
    }
    assert len(patch_payloads_before) > 1

    record = setup.prepare_incoming_transfer_project(
        request,
        target_configuration=fixture["configuration"],
    )

    archive = fixture["archive"]
    assert record.project_id == archive.project_id
    assert record.home_space_id == fixture["target"].space_id
    assert record.locator == str(target_state_root / "manifest.toml")
    assert record.state_location == str(target_state_root)
    assert fixture["target"].project(archive.project_id) is None
    patch_payloads_after = {
        path.relative_to(target_state_root).as_posix(): path.read_bytes()
        for path in sorted((target_state_root / "patches").rglob("*.json"))
    }
    assert patch_payloads_after == patch_payloads_before


@pytest.mark.parametrize("wrong_identity", ("project", "home"))
def test_incoming_transfer_catalog_prevalidation_rejects_wrong_identity(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wrong_identity: str,
) -> None:
    fixture, request, _setup = _published_fixture(manifest, tmp_path, monkeypatch)
    target_state_root = fixture["target_state_root"]
    project_id = request.proposed_project_id
    home_space_id = request.target_space_id
    if wrong_identity == "project":
        project_id = str(uuid.uuid4())
    else:
        home_space_id = str(uuid.uuid4())

    with pytest.raises(ValueError, match="another RCP space|differs from its reviewed target"):
        fixture["catalog"].prepare_incoming_transfer_registration(
            str(target_state_root / "manifest.toml"),
            project_id=project_id,
            home_space_id=home_space_id,
            expected_manifest_content=fixture["configuration"].manifest_content,
        )

    assert fixture["target"].project(fixture["archive"].project_id) is None


def test_incoming_transfer_finalizer_rejects_wrong_review(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, request, setup = _published_fixture(manifest, tmp_path, monkeypatch)
    configuration = TransferTargetConfiguration(
        manifest_content=fixture["configuration"].manifest_content,
        receipt=fixture["configuration"].receipt.model_copy(
            update={"final_review_sha256": "0" * 64}
        ),
    )

    with pytest.raises(ValueError, match="does not bind"):
        setup.prepare_incoming_transfer_project(
            request,
            target_configuration=configuration,
        )

    assert fixture["target"].project(fixture["archive"].project_id) is None
