from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rcp.config import AGENT_EXECUTION_PROFILES, load_manifest
from rcp.core.models import AuthorizedHuman, GraphBranchMetadata, Patch
from rcp.core.transition_models import GraphHeadRef, GraphTargetRef
from rcp.history import HistoryManager
from rcp.providers import configured_runtime_id
from rcp.server_ops.github import parse_github_repository_ref
from rcp.storage import (
    ProjectProvisioningGitCheckRecord,
    ProjectProvisioningMachineRecord,
    ProjectProvisioningProviderCheckRecord,
    ProjectProvisioningRepositoryRecord,
    ProjectProvisioningRequestRecord,
    ProjectTransferLinkReceipt,
    ProjectTransferRepositoryBinding,
    ProjectTransferRepositorySource,
    ProjectTransferSourceConfiguration,
)
from rcp.storage.provisioning import (
    project_provisioning_review_digest,
    project_transfer_source_configuration_sha256,
)
from rcp.transfer import (
    TRANSFER_ARCHIVE_CODEC,
    TransferArchiveActor,
    TransferArchiveAttribution,
    TransferArchiveEntry,
    TransferArchiveManifest,
    TransferGraphHead,
)
from rcp.transfer.configuration import build_transfer_target_configuration
from rcp.transport import StateUnavailable

PROJECT_ID = "11111111-1111-4111-8111-111111111111"
SOURCE_SPACE_ID = "22222222-2222-4222-8222-222222222222"
TARGET_SPACE_ID = "33333333-3333-4333-8333-333333333333"
SOURCE_REQUEST_ID = "44444444-4444-4444-8444-444444444444"
TARGET_REQUEST_ID = "55555555-5555-4555-8555-555555555555"
SOURCE_USER_ID = "66666666-6666-4666-8666-666666666666"
TARGET_USER_ID = "77777777-7777-4777-8777-777777777777"
ARCHIVE_ACTOR_ID = "88888888-8888-4888-8888-888888888888"
BRANCH_ID = "99999999-9999-4999-8999-999999999999"


def _actor(space_id: str, user_id: str, display_name: str) -> AuthorizedHuman:
    return AuthorizedHuman(
        space_id=space_id,
        user_id=user_id,
        display_name=display_name,
    )


def _capture_entry(root: Path, source: Path, archive_path: str, group: str):
    destination = root / archive_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    payload = destination.read_bytes()
    return TransferArchiveEntry(
        archive_path=archive_path,
        group=group,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def _configuration_fixture(manifest, tmp_path: Path):
    source_actor = _actor(SOURCE_SPACE_ID, SOURCE_USER_ID, "Z")
    target_actor = _actor(TARGET_SPACE_ID, TARGET_USER_ID, "Alice")
    history = HistoryManager(
        manifest,
        expected_space_id=SOURCE_SPACE_ID,
        project_id=PROJECT_ID,
    )
    history.claim_project_identity("created", project_id=PROJECT_ID)
    branch_base = history.head_ref(history.materialize(write_outputs=False))
    branch = history.create_auto_research_branch(
        GraphBranchMetadata(
            branch_id=BRANCH_ID,
            episode_id=BRANCH_ID,
            project_id=PROJECT_ID,
            base_head=branch_base,
            head=GraphHeadRef(
                target=GraphTargetRef(kind="branch", branch_id=BRANCH_ID),
                revision=branch_base.revision,
                transition_id=branch_base.transition_id,
            ),
            authorized_by=source_actor,
        )
    )
    branch.append(
        Patch(
            kind="work",
            author="agent",
            summary="Recorded branch-only transfer evidence.",
            run_truth_scope=["repo-a"],
            repositories_read=["repo-a"],
            ops=[
                {
                    "op": "create_nodes",
                    "nodes": [
                        {
                            "id": "ev/transfer-branch",
                            "type": "evidence",
                            "title": "Transfer branch evidence",
                            "observation": "The retained branch replays on the target.",
                            "origin": "internal_run",
                        }
                    ],
                }
            ],
        ),
        expected_revision=branch_base.revision,
    )
    history.transfer_project_home(
        project_id=PROJECT_ID,
        previous_home_space_id=SOURCE_SPACE_ID,
        new_home_space_id=TARGET_SPACE_ID,
        source_released_by=source_actor,
        target_admitted_by=target_actor,
    )
    replay = history.materialize(write_outputs=False)

    repositories = tuple(
        ProjectTransferRepositorySource(
            alias=alias,
            repository=parse_github_repository_ref(f"git@github.com:Example/{alias}.git"),
            machine_alias="laptop",
        )
        for alias in ("repo-a", "repo-b")
    )
    source_manifest_bytes = manifest.path.read_bytes()
    source_configuration = ProjectTransferSourceConfiguration(
        source_rcp_version="0.1.0.dev0+main",
        source_schema_generation=1,
        supported_archive_codecs=(TRANSFER_ARCHIVE_CODEC,),
        machine_aliases=("laptop",),
        repositories=repositories,
        state_repository="repo-a",
        project_truth_scope=("repo-a", "repo-b"),
        default_run_truth_scope=("repo-a",),
        source_manifest_sha256=hashlib.sha256(source_manifest_bytes).hexdigest(),
    )
    source_configuration_sha256 = project_transfer_source_configuration_sha256(source_configuration)

    target_root = "/srv/rcp/projects"
    machine = ProjectProvisioningMachineRecord(
        alias="laptop",
        location="ssh",
        host="server.example",
        os_account="rcp",
        central_root=target_root,
        resolved_central_root=target_root,
    )
    checked_at = "2026-08-31T12:00:00+00:00"
    target_repositories = [
        ProjectProvisioningRepositoryRecord(
            alias=item.alias,
            repository=item.repository,
            machine_alias="laptop",
            intended_path=(f"{target_root}/{PROJECT_ID}/repositories/{item.alias}"),
            resolved_path=(f"{target_root}/{PROJECT_ID}/repositories/{item.alias}"),
            checkout_disposition="request_created",
            git_check=ProjectProvisioningGitCheckRecord(
                status="ready",
                commit="a" * 40,
                write_verified=True,
                deploy_key_label=(f"rcp:{TARGET_SPACE_ID}:{PROJECT_ID}:{item.alias}"),
                public_key_fingerprint="SHA256:" + ("A" * 43),
                checked_at=checked_at,
            ),
        )
        for item in repositories
    ]
    provider_checks = [
        ProjectProvisioningProviderCheckRecord(
            profile=profile,
            provider="codex",
            runtime_id="codex:exec",
            model="gpt-test",
            reasoning="medium",
            machine_alias="laptop",
            status="ready",
            binary_path="/usr/local/bin/codex",
            version="codex-cli 1.2.3",
            resolved_runtime_id=configured_runtime_id("codex", "exec"),
            execution_account="rcp",
            checked_at=checked_at,
        )
        for profile in AGENT_EXECUTION_PROFILES
    ]
    provisioning = ProjectProvisioningRequestRecord(
        request_id=TARGET_REQUEST_ID,
        kind="incoming_transfer",
        status="ready_for_review",
        target_space_id=TARGET_SPACE_ID,
        authorized_by=target_actor,
        proposed_project_id=PROJECT_ID,
        name="Transferred project",
        state_repository="repo-a",
        project_truth_scope=["repo-a", "repo-b"],
        default_run_truth_scope=["repo-a"],
        machines=[machine],
        repositories=target_repositories,
        provider_checks=provider_checks,
        final_review_digest="0" * 64,
        revision=2,
        created_at=checked_at,
        updated_at=checked_at,
        setup_started_at=checked_at,
        ready_at=checked_at,
    )
    provisioning = provisioning.model_copy(
        update={"final_review_digest": project_provisioning_review_digest(provisioning)}
    )

    source_proof = b"p" * 32
    source_proof_sha256 = hashlib.sha256(source_proof).hexdigest()
    target_proof_sha256 = "b" * 64
    link = ProjectTransferLinkReceipt(
        source_request_id=SOURCE_REQUEST_ID,
        target_request_id=TARGET_REQUEST_ID,
        project_id=PROJECT_ID,
        source_space_id=SOURCE_SPACE_ID,
        target_space_id=TARGET_SPACE_ID,
        source_configuration_sha256=source_configuration_sha256,
        target_repositories=tuple(
            ProjectTransferRepositoryBinding(
                alias=item.alias,
                repository=item.repository,
            )
            for item in repositories
        ),
        accepted_schema_generation=1,
        accepted_archive_codec=TRANSFER_ARCHIVE_CODEC,
        source_release_proof_sha256=source_proof_sha256,
        target_activation_proof_sha256=target_proof_sha256,
        created_at=checked_at,
    )

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    entries = [
        _capture_entry(
            archive_root,
            manifest.path,
            "provenance/manifest.toml",
            "source_manifest_provenance",
        )
    ]
    entries.append(
        _capture_entry(
            archive_root,
            manifest.research_dir / "scope-base.json",
            "canonical/scope-base.json",
            "canonical_history",
        )
    )
    for path in sorted((manifest.research_dir / "patches").rglob("*.json")):
        relative = path.relative_to(manifest.research_dir).as_posix()
        entries.append(
            _capture_entry(
                archive_root,
                path,
                f"canonical/{relative}",
                "canonical_history",
            )
        )
    branch_root = manifest.research_dir / "branches" / BRANCH_ID
    for path in sorted(branch_root.rglob("*")):
        if not path.is_file() or path.name in {
            "coverage.json",
            "glossary.json",
            "graph.json",
            "proposals.json",
            "research.md",
        }:
            continue
        relative = path.relative_to(manifest.research_dir).as_posix()
        entries.append(
            _capture_entry(
                archive_root,
                path,
                f"canonical/{relative}",
                "canonical_history",
            )
        )
    proof_path = archive_root / "control/source-release-proof.bin"
    proof_path.parent.mkdir(parents=True)
    proof_path.write_bytes(source_proof)
    entries.append(
        TransferArchiveEntry(
            archive_path="control/source-release-proof.bin",
            group="source_release_proof",
            sha256=source_proof_sha256,
            size_bytes=len(source_proof),
        )
    )
    ordered_entries = tuple(sorted(entries, key=lambda item: item.archive_path))
    archive = TransferArchiveManifest(
        project_id=PROJECT_ID,
        source_space_id=SOURCE_SPACE_ID,
        target_space_id=TARGET_SPACE_ID,
        source_request_id=SOURCE_REQUEST_ID,
        target_request_id=TARGET_REQUEST_ID,
        source_rcp_version=source_configuration.source_rcp_version,
        source_schema_generation=source_configuration.source_schema_generation,
        source_configuration_sha256=source_configuration_sha256,
        source_manifest_sha256=source_configuration.source_manifest_sha256,
        source_release_proof_sha256=source_proof_sha256,
        target_activation_proof_sha256=target_proof_sha256,
        main_head=TransferGraphHead.capture(history.head_ref(replay)),
        branch_heads=(TransferGraphHead.capture(branch.head_ref()),),
        attributions=(
            TransferArchiveAttribution(
                archive_actor_id=ARCHIVE_ACTOR_ID,
                source_actor=TransferArchiveActor.capture(source_actor),
            ),
        ),
        diagnostics=(),
        entries=ordered_entries,
        payload_size_bytes=sum(item.size_bytes for item in ordered_entries),
        created_at=datetime(2026, 8, 31, 12, 0, tzinfo=UTC),
    )
    return provisioning, source_configuration, link, archive, archive_root


def test_target_configuration_uses_only_reviewed_target_execution_and_replays(
    manifest,
    tmp_path: Path,
) -> None:
    provisioning, source, link, archive, archive_root = _configuration_fixture(manifest, tmp_path)

    configured = build_transfer_target_configuration(
        provisioning,
        source,
        link,
        archive,
        archive_root,
    )

    target = load_manifest(_write_manifest(tmp_path / "target.toml", configured.manifest_content))
    assert target.repository_map["repo-a"].path == (
        f"/srv/rcp/projects/{PROJECT_ID}/repositories/repo-a"
    )
    assert target.machine_map["laptop"].host == "server.example"
    assert target.machine_map["laptop"].provider_paths == {"codex": "/usr/local/bin/codex"}
    assert str(manifest.repository_map["repo-a"].path) not in configured.manifest_content
    assert configured.receipt.main_head == archive.main_head
    assert configured.receipt.final_review_sha256 == provisioning.final_review_digest
    assert configured.receipt.archive_manifest_sha256 == archive.sha256()
    assert configured.receipt.retained_history.state == "empty"


def test_target_configuration_accepts_only_an_exact_retained_prefix(
    manifest,
    tmp_path: Path,
) -> None:
    provisioning, source, link, archive, archive_root = _configuration_fixture(manifest, tmp_path)
    retained = tmp_path / "checkout" / ".research"
    (retained / "patches").mkdir(parents=True)
    shutil.copyfile(archive_root / "provenance/manifest.toml", retained / "manifest.toml")
    shutil.copyfile(archive_root / "canonical/scope-base.json", retained / "scope-base.json")
    shutil.copyfile(
        archive_root / "canonical/patches/000001.json",
        retained / "patches/000001.json",
    )
    retained_branch = retained / "branches" / BRANCH_ID
    (retained_branch / "patches").mkdir(parents=True)
    final_metadata = GraphBranchMetadata.model_validate_json(
        (archive_root / f"canonical/branches/{BRANCH_ID}/branch.json").read_text(encoding="utf-8")
    )
    retained_metadata = final_metadata.model_copy(
        update={
            "head": GraphHeadRef(
                target=GraphTargetRef(kind="branch", branch_id=BRANCH_ID),
                revision=final_metadata.base_head.revision,
                transition_id=final_metadata.base_head.transition_id,
            )
        }
    )
    (retained_branch / "branch.json").write_text(
        retained_metadata.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    lagging = build_transfer_target_configuration(
        provisioning,
        source,
        link,
        archive,
        archive_root,
        retained_research_root=retained,
    )

    assert lagging.receipt.retained_history.branches[0].revision == 1

    shutil.copyfile(
        archive_root / f"canonical/branches/{BRANCH_ID}/branch.json",
        retained_branch / "branch.json",
    )
    shutil.copyfile(
        archive_root / f"canonical/branches/{BRANCH_ID}/patches/000002.json",
        retained_branch / "patches/000002.json",
    )

    configured = build_transfer_target_configuration(
        provisioning,
        source,
        link,
        archive,
        archive_root,
        retained_research_root=retained,
    )

    assert configured.receipt.retained_history.state == "matching"
    assert configured.receipt.retained_history.main_revision == 1
    assert configured.receipt.retained_history.branches[0].branch_id == BRANCH_ID
    assert configured.receipt.retained_history.branches[0].revision == 2

    (retained / "patches/000001.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from the archive"):
        build_transfer_target_configuration(
            provisioning,
            source,
            link,
            archive,
            archive_root,
            retained_research_root=retained,
        )

    shutil.copyfile(
        archive_root / "canonical/patches/000001.json",
        retained / "patches/000001.json",
    )
    shutil.copyfile(
        archive_root / "canonical/patches/000002.json",
        retained / "patches/000003.json",
    )
    with pytest.raises(ValueError, match="outside the archive"):
        build_transfer_target_configuration(
            provisioning,
            source,
            link,
            archive,
            archive_root,
            retained_research_root=retained,
        )


def test_target_configuration_refuses_scope_or_archive_binding_drift(
    manifest,
    tmp_path: Path,
) -> None:
    provisioning, source, link, archive, archive_root = _configuration_fixture(manifest, tmp_path)
    changed = provisioning.model_copy(
        update={
            "project_truth_scope": ["repo-a"],
            "default_run_truth_scope": ["repo-a"],
        }
    )
    changed = changed.model_copy(
        update={"final_review_digest": project_provisioning_review_digest(changed)}
    )

    with pytest.raises(ValueError, match="scope provenance"):
        build_transfer_target_configuration(
            changed,
            source,
            link,
            archive,
            archive_root,
        )

    rebound = archive.model_copy(update={"target_request_id": str(uuid.uuid4())})
    with pytest.raises(ValueError, match="reviewed source/target link"):
        build_transfer_target_configuration(
            provisioning,
            source,
            link,
            rebound,
            archive_root,
        )


def test_target_configuration_preserves_historical_machine_aliases(
    manifest,
    tmp_path: Path,
) -> None:
    provisioning, source, link, archive, archive_root = _configuration_fixture(manifest, tmp_path)
    renamed = provisioning.model_copy(
        update={
            "machines": [provisioning.machines[0].model_copy(update={"alias": "server"})],
            "repositories": [
                repository.model_copy(update={"machine_alias": "server"})
                for repository in provisioning.repositories
            ],
            "provider_checks": [
                check.model_copy(update={"machine_alias": "server"})
                for check in provisioning.provider_checks
            ],
        }
    )
    renamed = renamed.model_copy(
        update={"final_review_digest": project_provisioning_review_digest(renamed)}
    )

    with pytest.raises(ValueError, match="renames a historical repository machine alias"):
        build_transfer_target_configuration(
            renamed,
            source,
            link,
            archive,
            archive_root,
        )


def test_target_configuration_rechecks_the_source_codec_offer(
    manifest,
    tmp_path: Path,
) -> None:
    provisioning, source, link, archive, archive_root = _configuration_fixture(manifest, tmp_path)
    changed_source = source.model_copy(update={"supported_archive_codecs": ("other",)})
    changed_digest = project_transfer_source_configuration_sha256(changed_source)
    changed_link = link.model_copy(update={"source_configuration_sha256": changed_digest})
    changed_archive = archive.model_copy(update={"source_configuration_sha256": changed_digest})

    with pytest.raises(ValueError, match="different accepted format"):
        build_transfer_target_configuration(
            provisioning,
            changed_source,
            changed_link,
            changed_archive,
            archive_root,
        )


def test_target_configuration_rejects_special_retained_and_archive_files(
    manifest,
    tmp_path: Path,
) -> None:
    provisioning, source, link, archive, archive_root = _configuration_fixture(manifest, tmp_path)
    retained = tmp_path / "checkout" / ".research"
    (retained / "patches").mkdir(parents=True)
    shutil.copyfile(archive_root / "provenance/manifest.toml", retained / "manifest.toml")
    shutil.copyfile(archive_root / "canonical/scope-base.json", retained / "scope-base.json")
    shutil.copyfile(
        archive_root / "canonical/patches/000001.json",
        retained / "patches/000001.json",
    )
    os.mkfifo(retained / "chat")

    with pytest.raises(ValueError, match="not a regular directory"):
        build_transfer_target_configuration(
            provisioning,
            source,
            link,
            archive,
            archive_root,
            retained_research_root=retained,
        )

    canonical = archive_root / "canonical/patches/000001.json"
    canonical.unlink()
    os.mkfifo(canonical)
    with pytest.raises(StateUnavailable, match="changed before publication"):
        build_transfer_target_configuration(
            provisioning,
            source,
            link,
            archive,
            archive_root,
        )


def test_prior_home_can_read_its_branch_but_cannot_authorize_new_team_work(
    manifest,
    tmp_path: Path,
) -> None:
    _configuration_fixture(manifest, tmp_path)
    target = HistoryManager(
        load_manifest(manifest.path),
        expected_space_id=TARGET_SPACE_ID,
        project_id=PROJECT_ID,
    )

    assert target.branch(BRANCH_ID, expected_project_id=PROJECT_ID).head_ref().revision == 2

    metadata_path = manifest.research_dir / "branches" / BRANCH_ID / "branch.json"
    original = GraphBranchMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
    forged = original.model_copy(
        update={"authorized_by": _actor(TARGET_SPACE_ID, TARGET_USER_ID, "Alice")}
    )
    metadata_path.write_text(forged.model_dump_json(indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="authorizer belongs to a different space"):
        target.branch(BRANCH_ID, expected_project_id=PROJECT_ID)
    metadata_path.write_text(original.model_dump_json(indent=2) + "\n", encoding="utf-8")

    base = target.head_ref(target.materialize(write_outputs=False))
    new_branch_id = str(uuid.uuid4())
    with pytest.raises(ValueError, match="authorizer belongs to a different space"):
        target.create_auto_research_branch(
            GraphBranchMetadata(
                branch_id=new_branch_id,
                episode_id=new_branch_id,
                project_id=PROJECT_ID,
                base_head=base,
                head=GraphHeadRef(
                    target=GraphTargetRef(kind="branch", branch_id=new_branch_id),
                    revision=base.revision,
                    transition_id=base.transition_id,
                ),
                authorized_by=_actor(SOURCE_SPACE_ID, SOURCE_USER_ID, "Z"),
            )
        )

    accepted_branch_id = str(uuid.uuid4())
    accepted = target.create_auto_research_branch(
        GraphBranchMetadata(
            branch_id=accepted_branch_id,
            episode_id=accepted_branch_id,
            project_id=PROJECT_ID,
            base_head=base,
            head=GraphHeadRef(
                target=GraphTargetRef(kind="branch", branch_id=accepted_branch_id),
                revision=base.revision,
                transition_id=base.transition_id,
            ),
            authorized_by=_actor(TARGET_SPACE_ID, TARGET_USER_ID, "Alice"),
        )
    )
    assert accepted.head_ref().revision == base.revision


def _write_manifest(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path
