"""Durable, human-authorized team-project preparation state."""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import uuid
from pathlib import PurePosixPath

from pydantic import TypeAdapter

from rcp.config import DEFAULT_AUTO_RESEARCH_INVOCATION_CEILING
from rcp.core.models import AuthorizedHuman
from rcp.core.transition_models import GraphHeadRef
from rcp.server_ops.layout import DEFAULT_SERVER_LAYOUT
from rcp.server_ops.models import CommandAction, MachineTarget, MessageText, ServerStep
from rcp.storage.models import (
    ProjectMemberRecord,
    ProjectProvisioningCancellationDisposition,
    ProjectProvisioningGitCheckRecord,
    ProjectProvisioningKind,
    ProjectProvisioningMachineIntent,
    ProjectProvisioningMachineRecord,
    ProjectProvisioningProviderCheckRecord,
    ProjectProvisioningProviderIntent,
    ProjectProvisioningRepositoryIntent,
    ProjectProvisioningRepositoryRecord,
    ProjectProvisioningRequestRecord,
    ProjectProvisioningStatus,
    ProjectProvisioningStepReceiptRecord,
    ProjectRecord,
    ProjectTransferActivationReceipt,
    ProjectTransferCleanupAcknowledgment,
    ProjectTransferImportRecord,
    ProjectTransferLinkReceipt,
    ProjectTransferPhase,
    ProjectTransferRegisteredProject,
    ProjectTransferRepositoryBinding,
    ProjectTransferRequestRecord,
    ProjectTransferResolvedPath,
    ProjectTransferRestoreReentryReceipt,
    ProjectTransferSide,
    ProjectTransferSourceConfiguration,
    ProjectTransferSourceReleaseReceipt,
    ProjectTransferTargetAdmissionReceipt,
    ProjectTransferUploadCompleteReceipt,
    ProjectTransferUploadRecord,
    _canonical_uuid4,
)

_PROVISIONING_TRANSITIONS: dict[ProjectProvisioningStatus, frozenset[ProjectProvisioningStatus]] = {
    "waiting_for_server_setup": frozenset(
        {"setup_in_progress", "operator_action_needed", "cancelled"}
    ),
    "setup_in_progress": frozenset(
        {"setup_in_progress", "operator_action_needed", "ready_for_review", "cancelled"}
    ),
    "operator_action_needed": frozenset(
        {"setup_in_progress", "operator_action_needed", "cancelled"}
    ),
    "ready_for_review": frozenset({"setup_in_progress", "completed", "cancelled"}),
    "completed": frozenset(),
    "cancelled": frozenset(),
}
_MESSAGE_TEXT_ADAPTER = TypeAdapter(MessageText)
_PROJECT_CONFIG_FIELDS = frozenset(
    {
        "name",
        "state_repository",
        "project_truth_scope",
        "default_run_truth_scope",
        "default_auto_research_invocation_ceiling",
    }
)
_PROJECT_TRANSFER_PHASES: frozenset[ProjectTransferPhase] = frozenset(
    {
        "awaiting_link",
        "linked",
        "target_admitted",
        "source_released",
        "source_fenced",
        "archive_bound",
        "target_activated",
        "cleanup_acknowledged",
        "completed",
        "operator_action_needed",
    }
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def project_transfer_source_configuration_sha256(
    configuration: ProjectTransferSourceConfiguration,
) -> str:
    """Bind exactly the credential-free source provenance exchanged at link time."""

    normalized = ProjectTransferSourceConfiguration.model_validate_json(
        configuration.model_dump_json()
    )
    return hashlib.sha256(
        _canonical_json(normalized.model_dump(mode="json")).encode("utf-8")
    ).hexdigest()


def project_transfer_receipt_sha256(
    receipt: (
        ProjectTransferLinkReceipt
        | ProjectTransferTargetAdmissionReceipt
        | ProjectTransferSourceReleaseReceipt
        | ProjectTransferActivationReceipt
        | ProjectTransferCleanupAcknowledgment
    ),
) -> str:
    """Return the public digest that binds one cross-space receipt."""

    return hashlib.sha256(
        _canonical_json(receipt.model_dump(mode="json")).encode("utf-8")
    ).hexdigest()


def project_provisioning_review_digest(record: ProjectProvisioningRequestRecord) -> str:
    """Bind the exact identity, placement, Git, and provider review payload."""

    payload = {
        "request_id": record.request_id,
        "kind": record.kind,
        "target_space_id": record.target_space_id,
        "authorized_by": record.authorized_by.model_dump(mode="json"),
        "proposed_project_id": record.proposed_project_id,
        "name": record.name,
        "state_repository": record.state_repository,
        "project_truth_scope": record.project_truth_scope,
        "default_run_truth_scope": record.default_run_truth_scope,
        "default_auto_research_invocation_ceiling": (
            record.default_auto_research_invocation_ceiling
        ),
        "machines": [machine.model_dump(mode="json") for machine in record.machines],
        "repositories": [repository.model_dump(mode="json") for repository in record.repositories],
        "provider_checks": [check.model_dump(mode="json") for check in record.provider_checks],
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _legacy_project_provisioning_review_digest(
    record: ProjectProvisioningRequestRecord,
) -> str:
    """Reproduce the pre-P4 digest for an already reviewable persisted request."""

    payload = {
        "request_id": record.request_id,
        "kind": record.kind,
        "target_space_id": record.target_space_id,
        "authorized_by": record.authorized_by.model_dump(mode="json"),
        "proposed_project_id": record.proposed_project_id,
        "machines": [machine.model_dump(mode="json") for machine in record.machines],
        "repositories": [
            repository.model_dump(mode="json", exclude={"checkout_disposition"})
            for repository in record.repositories
        ],
        "provider_checks": [check.model_dump(mode="json") for check in record.provider_checks],
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _verify_project_provisioning_review_digest(
    record: ProjectProvisioningRequestRecord,
) -> ProjectProvisioningRequestRecord:
    if record.status in {"ready_for_review", "completed"}:
        current_digest = project_provisioning_review_digest(record)
        legacy_digest = (
            _legacy_project_provisioning_review_digest(record)
            if not record.configuration_complete
            and all(repository.checkout_disposition is None for repository in record.repositories)
            else None
        )
        if record.final_review_digest not in {current_digest, legacy_digest}:
            raise ValueError("project provisioning final-review digest does not match its payload")
    return record


class ProjectProvisioningStoreMixin:
    """One transactional state machine for every team-project preparation."""

    def require_project_accepts_new_work(self, project_id: str) -> None:
        """Refuse new source-side work after the human release fences the project."""

        canonical_project_id = self._transfer_uuid(project_id, "project identity")
        with self.connection() as connection:
            self._require_project_accepts_new_work(connection, canonical_project_id)

    @staticmethod
    def _require_project_accepts_new_work(
        connection: sqlite3.Connection,
        project_id: str,
    ) -> None:
        row = connection.execute(
            """
            SELECT request_id FROM project_transfer_requests
            WHERE side = 'source' AND project_id = ?
              AND phase IN (
                'source_released', 'source_fenced', 'archive_bound',
                'cleanup_acknowledged', 'completed'
              )
            LIMIT 1
            """,
            (project_id,),
        ).fetchone()
        if row is not None:
            raise ValueError(
                "This project is moving to its admitted team space and cannot accept new work."
            )

    def retire_source_project_transfer(self, request_id: str) -> ProjectTransferRequestRecord:
        """Hide one departed source project after its target proof was verified."""

        canonical_request_id = self._transfer_uuid(request_id, "transfer request identity")
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._project_transfer_request_from_connection(
                connection,
                canonical_request_id,
            )
            if (
                current.side != "source"
                or current.phase not in {"cleanup_acknowledged", "completed"}
                or current.proof_state not in {"acknowledged", "consumed"}
            ):
                raise ValueError("source transfer is not ready to retire its catalog entry")
            project = connection.execute(
                "SELECT * FROM projects WHERE project_id = ?",
                (current.project_id,),
            ).fetchone()
            if project is None:
                raise RuntimeError("source transfer lost its retained catalog receipt")
            bound_request_id = project["retired_transfer_request_id"]
            if bound_request_id is not None:
                if bound_request_id != canonical_request_id or project["retired_at"] is None:
                    raise ValueError("source project was retired by another operation")
                return current
            if project["retired_at"] is not None:
                raise ValueError("source project has an unbound retirement marker")
            if project["home_space_id"] != current.source_space_id:
                raise ValueError("source catalog entry no longer belongs to this transfer")
            connection.execute(
                """
                UPDATE projects
                SET retired_at = ?, retired_transfer_request_id = ?
                WHERE project_id = ? AND retired_at IS NULL
                """,
                (self.now(), canonical_request_id, current.project_id),
            )
        return current

    def create_source_project_transfer_request(
        self,
        *,
        project_id: str,
        target_space_id: str,
        initiated_by: AuthorizedHuman,
        source_configuration: ProjectTransferSourceConfiguration,
        request_id: str | None = None,
    ) -> ProjectTransferRequestRecord:
        """Create one personal-source request and its protected release proof."""

        canonical_project_id = self._transfer_uuid(project_id, "transfer project identity")
        canonical_target_space_id = self._transfer_uuid(
            target_space_id,
            "transfer target space identity",
        )
        canonical_request_id = (
            str(uuid.uuid4())
            if request_id is None
            else self._transfer_uuid(request_id, "transfer request identity")
        )
        actor = AuthorizedHuman.model_validate(initiated_by.model_dump(mode="json"))
        configuration = ProjectTransferSourceConfiguration.model_validate_json(
            source_configuration.model_dump_json()
        )
        configuration_sha256 = project_transfer_source_configuration_sha256(configuration)
        secret = secrets.token_bytes(32)
        commitment = hashlib.sha256(secret).hexdigest()
        now = self.now()
        record = ProjectTransferRequestRecord(
            request_id=canonical_request_id,
            side="source",
            phase="awaiting_link",
            project_id=canonical_project_id,
            source_space_id=self.space_id,
            target_space_id=canonical_target_space_id,
            initiated_by=actor,
            source_configuration=configuration,
            source_configuration_sha256=configuration_sha256,
            source_release_proof_sha256=commitment,
            revision=0,
            created_at=now,
            updated_at=now,
        )
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._require_transfer_space(
                connection,
                expected_kind="personal",
                expected_space_id=record.source_space_id,
            )
            self._require_transfer_actor(connection, actor)
            project = connection.execute(
                "SELECT project_id, home_space_id FROM projects WHERE project_id = ?",
                (canonical_project_id,),
            ).fetchone()
            if project is None or project["home_space_id"] != record.source_space_id:
                raise ValueError("source transfer project is stale or belongs to another space")
            existing = connection.execute(
                "SELECT * FROM project_transfer_requests WHERE request_id = ?",
                (canonical_request_id,),
            ).fetchone()
            if existing is not None:
                current = self._project_transfer_request_from_connection(
                    connection,
                    canonical_request_id,
                )
                fields = {
                    "request_id",
                    "side",
                    "project_id",
                    "source_space_id",
                    "target_space_id",
                    "initiated_by",
                    "source_configuration",
                    "source_configuration_sha256",
                }
                expected_payload = record.model_dump(mode="json", include=fields)
                current_payload = current.model_dump(mode="json", include=fields)
                if current_payload != expected_payload:
                    raise ValueError(
                        "transfer request identity already names another source intent"
                    )
                return current
            try:
                self._insert_project_transfer_request(
                    connection,
                    record,
                    proof_kind="source_release",
                    secret=secret,
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("the source project already has another transfer request") from exc
        return record

    def create_target_project_transfer_request(
        self,
        *,
        provisioning_request_id: str,
        source_request_id: str,
        source_project_id: str,
        source_space_id: str,
        initiated_by: AuthorizedHuman,
        source_configuration: ProjectTransferSourceConfiguration,
        source_configuration_sha256: str,
        source_release_proof_sha256: str,
        accepted_schema_generation: int,
        accepted_archive_codec: str,
    ) -> ProjectTransferRequestRecord:
        """Link one incoming team provisioning request without creating project authority."""

        canonical_request_id = self._transfer_uuid(
            provisioning_request_id,
            "target transfer request identity",
        )
        canonical_source_request_id = self._transfer_uuid(
            source_request_id,
            "source transfer request identity",
        )
        canonical_project_id = self._transfer_uuid(
            source_project_id,
            "source transfer project identity",
        )
        canonical_source_space_id = self._transfer_uuid(
            source_space_id,
            "transfer source space identity",
        )
        actor = AuthorizedHuman.model_validate(initiated_by.model_dump(mode="json"))
        configuration = ProjectTransferSourceConfiguration.model_validate_json(
            source_configuration.model_dump_json()
        )
        actual_configuration_sha256 = project_transfer_source_configuration_sha256(configuration)
        if actual_configuration_sha256 != source_configuration_sha256:
            raise ValueError("source transfer configuration digest does not match its payload")
        if accepted_schema_generation != configuration.source_schema_generation:
            raise ValueError("target does not accept the source transfer schema")
        if accepted_archive_codec not in configuration.supported_archive_codecs:
            raise ValueError("target selected an archive codec the source did not offer")
        self._transfer_digest(source_release_proof_sha256, "source release proof commitment")
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._require_transfer_space(
                connection,
                expected_kind="team",
                expected_space_id=self.space_id,
            )
            self._require_transfer_actor(connection, actor)
            provisioning_row = connection.execute(
                "SELECT * FROM project_provisioning_requests WHERE request_id = ?",
                (canonical_request_id,),
            ).fetchone()
            if provisioning_row is None:
                raise KeyError(canonical_request_id)
            provisioning = self._project_provisioning_record(provisioning_row)
            if (
                provisioning.kind != "incoming_transfer"
                or provisioning.target_space_id != self.space_id
                or provisioning.authorized_by != actor
                or provisioning.proposed_project_id != canonical_project_id
            ):
                raise ValueError("target transfer does not match its incoming provisioning request")
            target_repositories = tuple(
                ProjectTransferRepositoryBinding(
                    alias=repository.alias,
                    repository=repository.repository,
                )
                for repository in sorted(provisioning.repositories, key=lambda item: item.alias)
            )
            source_repositories = {
                repository.alias: repository.repository.identity
                for repository in configuration.repositories
            }
            if {item.alias: item.repository.identity for item in target_repositories} != (
                source_repositories
            ):
                raise ValueError(
                    "target transfer repositories do not match the source configuration"
                )
            existing = connection.execute(
                "SELECT * FROM project_transfer_requests WHERE request_id = ?",
                (canonical_request_id,),
            ).fetchone()
            if existing is not None:
                current = self._project_transfer_request_from_connection(
                    connection,
                    canonical_request_id,
                )
                fields = {
                    "request_id",
                    "side",
                    "linked_request_id",
                    "project_id",
                    "source_space_id",
                    "target_space_id",
                    "initiated_by",
                    "source_configuration",
                    "source_configuration_sha256",
                    "accepted_schema_generation",
                    "accepted_archive_codec",
                    "source_release_proof_sha256",
                }
                expected_payload = {
                    "request_id": canonical_request_id,
                    "side": "target",
                    "linked_request_id": canonical_source_request_id,
                    "project_id": canonical_project_id,
                    "source_space_id": canonical_source_space_id,
                    "target_space_id": self.space_id,
                    "initiated_by": actor.model_dump(mode="json"),
                    "source_configuration": configuration.model_dump(mode="json"),
                    "source_configuration_sha256": actual_configuration_sha256,
                    "accepted_schema_generation": accepted_schema_generation,
                    "accepted_archive_codec": accepted_archive_codec,
                    "source_release_proof_sha256": source_release_proof_sha256,
                }
                if current.model_dump(mode="json", include=fields) != expected_payload:
                    raise ValueError("target transfer request already names another source intent")
                assert current.link_receipt is not None
                if current.link_receipt.target_repositories != target_repositories:
                    raise ValueError("target transfer request already names other repositories")
                return current
            secret = secrets.token_bytes(32)
            target_commitment = hashlib.sha256(secret).hexdigest()
            now = self.now()
            link_receipt = ProjectTransferLinkReceipt(
                source_request_id=canonical_source_request_id,
                target_request_id=canonical_request_id,
                project_id=canonical_project_id,
                source_space_id=canonical_source_space_id,
                target_space_id=self.space_id,
                source_configuration_sha256=actual_configuration_sha256,
                target_repositories=target_repositories,
                accepted_schema_generation=accepted_schema_generation,
                accepted_archive_codec=accepted_archive_codec,
                source_release_proof_sha256=source_release_proof_sha256,
                target_activation_proof_sha256=target_commitment,
                created_at=now,
            )
            record = ProjectTransferRequestRecord(
                request_id=canonical_request_id,
                side="target",
                phase="linked",
                linked_request_id=canonical_source_request_id,
                project_id=canonical_project_id,
                source_space_id=canonical_source_space_id,
                target_space_id=self.space_id,
                initiated_by=actor,
                source_configuration=configuration,
                source_configuration_sha256=actual_configuration_sha256,
                accepted_schema_generation=accepted_schema_generation,
                accepted_archive_codec=accepted_archive_codec,
                source_release_proof_sha256=source_release_proof_sha256,
                target_activation_proof_sha256=target_commitment,
                link_receipt=link_receipt,
                revision=0,
                created_at=now,
                updated_at=now,
            )
            self._insert_project_transfer_request(
                connection,
                record,
                proof_kind="target_activation",
                secret=secret,
            )
        return record

    def link_source_project_transfer_request(
        self,
        request_id: str,
        *,
        receipt: ProjectTransferLinkReceipt,
    ) -> ProjectTransferRequestRecord:
        """Commit the target identity, commitment, and negotiated codec on the source."""

        canonical_request_id = self._transfer_uuid(request_id, "transfer request identity")
        normalized = ProjectTransferLinkReceipt.model_validate_json(receipt.model_dump_json())
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._project_transfer_request_from_connection(
                connection,
                canonical_request_id,
            )
            if current.side != "source":
                raise ValueError("only a source transfer request can accept a target link")
            self._require_link_receipt_matches(current, normalized)
            values = {
                "phase": "linked",
                "linked_request_id": normalized.target_request_id,
                "accepted_schema_generation": normalized.accepted_schema_generation,
                "accepted_archive_codec": normalized.accepted_archive_codec,
                "target_activation_proof_sha256": normalized.target_activation_proof_sha256,
                "link_receipt": normalized.model_dump(mode="json"),
            }
            if current.phase != "awaiting_link":
                if current.link_receipt != normalized:
                    raise ValueError("source transfer request is already linked differently")
                return current
            updated = self._updated_project_transfer_record(current, **values)
            self._replace_project_transfer_record(connection, current, updated)
        return updated

    def project_transfer_request(self, request_id: str) -> ProjectTransferRequestRecord | None:
        canonical_request_id = self._transfer_uuid(request_id, "transfer request identity")
        with self.connection() as connection:
            exists = connection.execute(
                "SELECT 1 FROM project_transfer_requests WHERE request_id = ?",
                (canonical_request_id,),
            ).fetchone()
            if exists is None:
                return None
            return self._project_transfer_request_from_connection(
                connection,
                canonical_request_id,
            )

    def project_transfer_requests(
        self,
        *,
        side: ProjectTransferSide | None = None,
        phase: ProjectTransferPhase | None = None,
    ) -> list[ProjectTransferRequestRecord]:
        if side is not None and side not in {"source", "target"}:
            raise ValueError("transfer request side is invalid")
        if phase is not None and phase not in _PROJECT_TRANSFER_PHASES:
            raise ValueError("transfer request phase is invalid")
        values: list[str] = []
        clauses: list[str] = []
        if side is not None:
            clauses.append("side = ?")
            values.append(side)
        if phase is not None:
            clauses.append("phase = ?")
            values.append(phase)
        where = "" if not clauses else "WHERE " + " AND ".join(clauses)
        with self.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM project_transfer_requests
                {where}
                ORDER BY created_at DESC, request_id
                """,
                values,
            ).fetchall()
            return [
                self._project_transfer_request_from_connection(connection, row["request_id"])
                for row in rows
            ]

    def record_target_project_transfer_admission(
        self,
        request_id: str,
        *,
        admitted_by: AuthorizedHuman,
    ) -> ProjectTransferRequestRecord:
        """Bind one target review without creating the canonical project."""

        canonical_request_id = self._transfer_uuid(request_id, "transfer request identity")
        actor = AuthorizedHuman.model_validate(admitted_by.model_dump(mode="json"))
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._project_transfer_request_from_connection(
                connection,
                canonical_request_id,
            )
            if current.side != "target":
                raise ValueError("target admission belongs only to the target transfer request")
            self._require_transfer_actor(connection, actor)
            if current.target_admission_receipt is not None:
                if current.target_admission_receipt.admitted_by != actor:
                    raise ValueError("target transfer admission already names another actor")
                return current
            provisioning_row = connection.execute(
                "SELECT * FROM project_provisioning_requests WHERE request_id = ?",
                (canonical_request_id,),
            ).fetchone()
            if provisioning_row is None:
                raise KeyError(canonical_request_id)
            provisioning = self._project_provisioning_record(provisioning_row)
            if (
                provisioning.kind != "incoming_transfer"
                or provisioning.status != "ready_for_review"
                or provisioning.final_review_digest is None
                or current.linked_request_id is None
                or current.accepted_schema_generation is None
                or current.accepted_archive_codec is None
                or current.target_activation_proof_sha256 is None
            ):
                raise ValueError("incoming transfer is not ready for target admission")
            resolved_paths = tuple(
                ProjectTransferResolvedPath(
                    repository_alias=repository.alias,
                    machine_alias=repository.machine_alias,
                    path=repository.resolved_path,
                )
                for repository in sorted(
                    provisioning.repositories,
                    key=lambda item: item.alias,
                )
                if repository.resolved_path is not None
            )
            if len(resolved_paths) != len(provisioning.repositories):
                raise ValueError("target admission requires every resolved central path")
            if {item.repository_alias for item in resolved_paths} != {
                item.alias for item in current.source_configuration.repositories
            }:
                raise ValueError("target preparation repository aliases changed after linking")
            receipt = ProjectTransferTargetAdmissionReceipt(
                source_request_id=current.linked_request_id,
                target_request_id=current.request_id,
                project_id=current.project_id,
                source_space_id=current.source_space_id,
                target_space_id=current.target_space_id,
                admitted_by=actor,
                source_configuration_sha256=current.source_configuration_sha256,
                target_preparation_revision=provisioning.revision,
                target_preparation_sha256=provisioning.final_review_digest,
                resolved_paths=resolved_paths,
                accepted_schema_generation=current.accepted_schema_generation,
                accepted_archive_codec=current.accepted_archive_codec,
                source_release_proof_sha256=current.source_release_proof_sha256,
                target_activation_proof_sha256=current.target_activation_proof_sha256,
                created_at=(
                    current.target_admission_receipt.created_at
                    if current.target_admission_receipt is not None
                    else self.now()
                ),
            )
            if current.phase != "linked":
                raise ValueError("target transfer request is not awaiting admission")
            updated = self._updated_project_transfer_record(
                current,
                phase="target_admitted",
                target_admission_receipt=receipt.model_dump(mode="json"),
            )
            self._replace_project_transfer_record(connection, current, updated)
        return updated

    def accept_target_project_transfer_admission(
        self,
        request_id: str,
        *,
        receipt: ProjectTransferTargetAdmissionReceipt,
    ) -> ProjectTransferRequestRecord:
        """Persist the exact target-space receipt on the linked personal source."""

        canonical_request_id = self._transfer_uuid(request_id, "transfer request identity")
        normalized = ProjectTransferTargetAdmissionReceipt.model_validate_json(
            receipt.model_dump_json()
        )
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._project_transfer_request_from_connection(
                connection,
                canonical_request_id,
            )
            self._require_target_admission_matches(current, normalized)
            if current.target_admission_receipt is not None:
                if current.target_admission_receipt != normalized:
                    raise ValueError("source request already stores another target admission")
                return current
            if current.side != "source" or current.phase != "linked":
                raise ValueError("source transfer request is not awaiting target admission")
            updated = self._updated_project_transfer_record(
                current,
                phase="target_admitted",
                target_admission_receipt=normalized.model_dump(mode="json"),
            )
            self._replace_project_transfer_record(connection, current, updated)
        return updated

    def record_source_project_transfer_release(
        self,
        request_id: str,
        *,
        released_by: AuthorizedHuman,
        revalidated_configuration: ProjectTransferSourceConfiguration,
        source_head: GraphHeadRef,
    ) -> ProjectTransferRequestRecord:
        """Record source authority only after the bound configuration is revalidated."""

        canonical_request_id = self._transfer_uuid(request_id, "transfer request identity")
        actor = AuthorizedHuman.model_validate(released_by.model_dump(mode="json"))
        configuration = ProjectTransferSourceConfiguration.model_validate_json(
            revalidated_configuration.model_dump_json()
        )
        head = GraphHeadRef.model_validate_json(source_head.model_dump_json())
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._project_transfer_request_from_connection(
                connection,
                canonical_request_id,
            )
            self._require_transfer_actor(connection, actor)
            if current.side != "source" or current.target_admission_receipt is None:
                raise ValueError("source release requires the linked target admission")
            configuration_sha256 = project_transfer_source_configuration_sha256(configuration)
            if current.source_release_receipt is not None:
                if (
                    current.source_release_receipt.released_by != actor
                    or current.source_release_receipt.source_configuration_sha256
                    != configuration_sha256
                    or current.source_release_receipt.source_head != head
                ):
                    raise ValueError("source transfer release already binds another boundary")
                return current
            project = connection.execute(
                "SELECT project_id, home_space_id FROM projects WHERE project_id = ?",
                (current.project_id,),
            ).fetchone()
            if project is None or project["home_space_id"] != current.source_space_id:
                raise ValueError("source transfer project is stale or already left this space")
            if configuration_sha256 != current.source_configuration_sha256:
                raise ValueError("source configuration changed after transfer preparation")
            admission = current.target_admission_receipt
            assert current.linked_request_id is not None
            assert current.accepted_schema_generation is not None
            assert current.accepted_archive_codec is not None
            assert current.target_activation_proof_sha256 is not None
            receipt = ProjectTransferSourceReleaseReceipt(
                source_request_id=current.request_id,
                target_request_id=current.linked_request_id,
                project_id=current.project_id,
                source_space_id=current.source_space_id,
                target_space_id=current.target_space_id,
                released_by=actor,
                source_configuration_sha256=current.source_configuration_sha256,
                target_admission_sha256=project_transfer_receipt_sha256(admission),
                target_preparation_revision=admission.target_preparation_revision,
                target_preparation_sha256=admission.target_preparation_sha256,
                source_head=head,
                accepted_schema_generation=current.accepted_schema_generation,
                accepted_archive_codec=current.accepted_archive_codec,
                source_release_proof_sha256=current.source_release_proof_sha256,
                target_activation_proof_sha256=current.target_activation_proof_sha256,
                created_at=(
                    current.source_release_receipt.created_at
                    if current.source_release_receipt is not None
                    else self.now()
                ),
            )
            if current.phase != "target_admitted":
                raise ValueError("source transfer request is not awaiting release")
            updated = self._updated_project_transfer_record(
                current,
                phase="source_released",
                source_release_receipt=receipt.model_dump(mode="json"),
            )
            self._replace_project_transfer_record(connection, current, updated)
        return updated

    def accept_source_project_transfer_release(
        self,
        request_id: str,
        *,
        receipt: ProjectTransferSourceReleaseReceipt,
    ) -> ProjectTransferRequestRecord:
        """Persist the exact source-space receipt on the admitted team target."""

        canonical_request_id = self._transfer_uuid(request_id, "transfer request identity")
        normalized = ProjectTransferSourceReleaseReceipt.model_validate_json(
            receipt.model_dump_json()
        )
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._project_transfer_request_from_connection(
                connection,
                canonical_request_id,
            )
            self._require_source_release_matches(current, normalized)
            if current.source_release_receipt is not None:
                if current.source_release_receipt != normalized:
                    raise ValueError("target request already stores another source release")
                return current
            if current.side != "target" or current.phase != "target_admitted":
                raise ValueError("target transfer request is not awaiting source release")
            updated = self._updated_project_transfer_record(
                current,
                phase="source_released",
                source_release_receipt=normalized.model_dump(mode="json"),
            )
            self._replace_project_transfer_record(connection, current, updated)
        return updated

    def mark_source_project_transfer_fenced(
        self,
        request_id: str,
        *,
        source_head: GraphHeadRef,
    ) -> ProjectTransferRequestRecord:
        """Advance only the source request whose exact released head was fenced."""

        canonical_request_id = self._transfer_uuid(request_id, "transfer request identity")
        head = GraphHeadRef.model_validate_json(source_head.model_dump_json())
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._project_transfer_request_from_connection(
                connection,
                canonical_request_id,
            )
            if (
                current.side != "source"
                or current.source_release_receipt is None
                or head.target.kind != "main"
                or head.revision <= current.source_release_receipt.source_head.revision
            ):
                raise ValueError("source fence does not advance the released canonical head")
            if current.phase == "source_fenced":
                if current.source_fence_head != head:
                    raise ValueError("source transfer already binds another fence head")
                return current
            if current.phase != "source_released":
                raise ValueError("source transfer request is not awaiting its fence")
            updated = self._updated_project_transfer_record(
                current,
                phase="source_fenced",
                source_fence_head=head.model_dump(mode="json"),
            )
            self._replace_project_transfer_record(connection, current, updated)
        return updated

    def bind_project_transfer_archive(
        self,
        request_id: str,
        *,
        archive_sha256: str,
        archive_size_bytes: int,
        source_fence_head: GraphHeadRef | None = None,
    ) -> ProjectTransferRequestRecord:
        """Bind one exact archive on either linked request, idempotently."""

        canonical_request_id = self._transfer_uuid(request_id, "transfer request identity")
        digest = self._transfer_digest(archive_sha256, "transfer archive digest")
        if archive_size_bytes < 1:
            raise ValueError("transfer archive size must be positive")
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._project_transfer_request_from_connection(
                connection,
                canonical_request_id,
            )
            requested_fence_head = (
                current.source_fence_head
                if source_fence_head is None
                else GraphHeadRef.model_validate_json(source_fence_head.model_dump_json())
            )
            expected_phase = "source_fenced" if current.side == "source" else "source_released"
            if current.archive_sha256 is not None:
                if (
                    current.archive_sha256 != digest
                    or current.archive_size_bytes != archive_size_bytes
                    or current.source_fence_head != requested_fence_head
                ):
                    raise ValueError("transfer request already binds another archive")
                return current
            if current.phase != expected_phase:
                raise ValueError("transfer request is not ready to bind its archive")
            if current.side == "source" and current.proof_state == "unexposed":
                raise ValueError("sealed source archive must include the exposed release proof")
            if current.side == "source":
                proof = connection.execute(
                    "SELECT * FROM project_transfer_proofs WHERE request_id = ?",
                    (canonical_request_id,),
                ).fetchone()
                if (
                    current.proof_state != "exposed"
                    or proof is None
                    or proof["state"] != "exposed"
                    or proof["secret"] is None
                ):
                    raise ValueError("sealed source archive requires its exposed release proof")
                secret = bytes(proof["secret"])
                if (
                    len(secret) != 32
                    or hashlib.sha256(secret).hexdigest() != current.source_release_proof_sha256
                ):
                    raise RuntimeError("stored source release proof does not match its commitment")
            if (
                requested_fence_head is None
                or current.source_release_receipt is None
                or requested_fence_head.target.kind != "main"
                or requested_fence_head.revision
                <= current.source_release_receipt.source_head.revision
            ):
                raise ValueError("transfer archive does not bind the fenced source head")
            updated = self._updated_project_transfer_record(
                current,
                phase="archive_bound",
                source_fence_head=requested_fence_head.model_dump(mode="json"),
                archive_sha256=digest,
                archive_size_bytes=archive_size_bytes,
            )
            self._replace_project_transfer_record(connection, current, updated)
        return updated

    def begin_target_project_transfer_upload(
        self,
        request_id: str,
    ) -> ProjectTransferUploadRecord:
        """Begin or replay the one target-side lease for a bound archive."""

        canonical_request_id = self._transfer_uuid(request_id, "transfer request identity")
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._project_transfer_request_from_connection(
                connection,
                canonical_request_id,
            )
            row = connection.execute(
                "SELECT * FROM project_transfer_uploads WHERE request_id = ?",
                (canonical_request_id,),
            ).fetchone()
            if row is not None:
                existing = self._project_transfer_upload_record(row)
                self._require_target_upload_matches(current, existing)
                if existing.status in {"complete", "consumed"} and current.phase in {
                    "archive_bound",
                    "target_activated",
                    "cleanup_acknowledged",
                    "completed",
                }:
                    return existing
            self._require_target_upload_request(current)
            assert current.archive_sha256 is not None
            assert current.archive_size_bytes is not None
            if row is not None:
                if existing.status == "active":
                    return existing
                if existing.status != "invalidated":
                    raise RuntimeError("stored project transfer upload state is invalid")

            now = self.now()
            lease_boundary = self._new_transfer_upload_boundary(connection, canonical_request_id)
            created_at = now if row is None else existing.created_at
            replacement = ProjectTransferUploadRecord(
                request_id=current.request_id,
                project_id=current.project_id,
                archive_sha256=current.archive_sha256,
                archive_size_bytes=current.archive_size_bytes,
                lease_boundary_sha256=lease_boundary,
                status="active",
                created_at=created_at,
                updated_at=now,
            )
            if row is None:
                connection.execute(
                    """
                    INSERT INTO project_transfer_uploads (
                        request_id, project_id, archive_sha256, archive_size_bytes,
                        lease_boundary_sha256, status, receipt_json, created_at,
                        updated_at, invalidated_at
                    ) VALUES (?, ?, ?, ?, ?, 'active', NULL, ?, ?, NULL)
                    """,
                    (
                        replacement.request_id,
                        replacement.project_id,
                        replacement.archive_sha256,
                        replacement.archive_size_bytes,
                        replacement.lease_boundary_sha256,
                        replacement.created_at,
                        replacement.updated_at,
                    ),
                )
            else:
                changed = connection.execute(
                    """
                    UPDATE project_transfer_uploads
                    SET archive_sha256 = ?, archive_size_bytes = ?,
                        lease_boundary_sha256 = ?, status = 'active',
                        receipt_json = NULL, updated_at = ?, invalidated_at = NULL
                    WHERE request_id = ? AND status = 'invalidated'
                    """,
                    (
                        replacement.archive_sha256,
                        replacement.archive_size_bytes,
                        replacement.lease_boundary_sha256,
                        replacement.updated_at,
                        replacement.request_id,
                    ),
                ).rowcount
                if changed != 1:
                    raise RuntimeError("project transfer upload restart lost its transaction guard")
            stored = connection.execute(
                "SELECT * FROM project_transfer_uploads WHERE request_id = ?",
                (canonical_request_id,),
            ).fetchone()
            if stored is None:
                raise RuntimeError("project transfer upload disappeared during begin")
            return self._project_transfer_upload_record(stored)

    def target_project_transfer_upload(
        self,
        request_id: str,
    ) -> ProjectTransferUploadRecord | None:
        """Read one target upload lease or retained terminal receipt."""

        canonical_request_id = self._transfer_uuid(request_id, "transfer request identity")
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM project_transfer_uploads WHERE request_id = ?",
                (canonical_request_id,),
            ).fetchone()
            if row is None:
                return None
            upload = self._project_transfer_upload_record(row)
            current = self._project_transfer_request_from_connection(
                connection,
                canonical_request_id,
            )
            self._require_target_upload_matches(current, upload)
            return upload

    def target_project_transfer_uploads(self) -> list[ProjectTransferUploadRecord]:
        """List durable target upload boundaries in request order."""

        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM project_transfer_uploads ORDER BY request_id"
            ).fetchall()
            uploads = [self._project_transfer_upload_record(row) for row in rows]
            for upload in uploads:
                current = self._project_transfer_request_from_connection(
                    connection,
                    upload.request_id,
                )
                self._require_target_upload_matches(current, upload)
            return uploads

    def complete_target_project_transfer_upload(
        self,
        request_id: str,
        *,
        lease_boundary_sha256: str,
    ) -> ProjectTransferUploadRecord:
        """Commit one verified upload and retain its typed completion receipt."""

        canonical_request_id = self._transfer_uuid(request_id, "transfer request identity")
        boundary = self._transfer_digest(lease_boundary_sha256, "transfer upload lease boundary")
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._project_transfer_request_from_connection(
                connection,
                canonical_request_id,
            )
            row = connection.execute(
                "SELECT * FROM project_transfer_uploads WHERE request_id = ?",
                (canonical_request_id,),
            ).fetchone()
            if row is None:
                raise KeyError(canonical_request_id)
            existing = self._project_transfer_upload_record(row)
            self._require_target_upload_matches(current, existing)
            if existing.status in {"complete", "consumed"} and current.phase in {
                "archive_bound",
                "target_activated",
                "cleanup_acknowledged",
                "completed",
            }:
                if existing.lease_boundary_sha256 != boundary:
                    raise ValueError("project transfer upload already has another lease boundary")
                return existing
            self._require_target_upload_request(current)
            if existing.status != "active":
                raise ValueError("project transfer upload lease is no longer active")
            if existing.lease_boundary_sha256 != boundary:
                raise ValueError("project transfer upload lease boundary does not match")
            now = self.now()
            completed = ProjectTransferUploadCompleteReceipt(
                request_id=current.request_id,
                project_id=current.project_id,
                archive_sha256=existing.archive_sha256,
                archive_size_bytes=existing.archive_size_bytes,
                lease_boundary_sha256=boundary,
                completed_at=now,
            )
            values = existing.model_dump(mode="json")
            values.update(
                {
                    "status": "complete",
                    "receipt": completed.model_dump(mode="json"),
                    "updated_at": now,
                }
            )
            updated = ProjectTransferUploadRecord.model_validate(values)
            changed = connection.execute(
                """
                UPDATE project_transfer_uploads
                SET status = 'complete', receipt_json = ?, updated_at = ?
                WHERE request_id = ? AND status = 'active' AND lease_boundary_sha256 = ?
                """,
                (
                    _canonical_json(completed.model_dump(mode="json")),
                    updated.updated_at,
                    canonical_request_id,
                    boundary,
                ),
            ).rowcount
            if changed != 1:
                raise RuntimeError("project transfer upload completion lost its transaction guard")
            return updated

    def activate_target_project_transfer(
        self,
        request_id: str,
        *,
        project: ProjectRecord,
        expected_provisioning_revision: int,
        expected_final_review_digest: str,
    ) -> ProjectTransferActivationReceipt:
        """Atomically register, seat, complete provisioning, and activate one import."""

        canonical_request_id = self._transfer_uuid(request_id, "transfer request identity")
        prepared_project = ProjectRecord.model_validate_json(project.model_dump_json())
        if type(expected_provisioning_revision) is not int or expected_provisioning_revision < 0:
            raise ValueError("target activation provisioning revision is invalid")
        expected_review = self._transfer_digest(
            expected_final_review_digest,
            "target activation final-review digest",
        )
        try:
            with self.connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                current = self._project_transfer_request_from_connection(
                    connection,
                    canonical_request_id,
                )
                existing_row = connection.execute(
                    "SELECT * FROM project_transfer_activations WHERE target_request_id = ?",
                    (canonical_request_id,),
                ).fetchone()
                if existing_row is not None:
                    existing = self._project_transfer_activation_record(existing_row)
                    self._validate_target_activation_receipt(
                        connection,
                        current,
                        existing,
                    )
                    if (
                        existing.provisioning_revision != expected_provisioning_revision + 1
                        or existing.provisioning_final_review_sha256 != expected_review
                        or self._activation_project_intent(existing.registered_project)
                        != self._activation_project_intent(prepared_project)
                    ):
                        raise ValueError(
                            "target activation retry does not match its committed boundary"
                        )
                    return existing

                if current.side != "target" or current.phase != "archive_bound":
                    raise ValueError("target transfer request is not ready to activate")
                if (
                    current.linked_request_id is None
                    or current.source_fence_head is None
                    or current.archive_sha256 is None
                    or current.target_activation_proof_sha256 is None
                    or current.target_admission_receipt is None
                    or current.source_release_receipt is None
                ):
                    raise RuntimeError("archive-bound target transfer lost activation authority")
                admitted_by = current.target_admission_receipt.admitted_by
                self._require_transfer_actor(connection, admitted_by)
                if (
                    prepared_project.project_id != current.project_id
                    or prepared_project.home_space_id != current.target_space_id
                    or prepared_project.retired_at is not None
                    or prepared_project.retired_transfer_request_id is not None
                ):
                    raise ValueError("target activation project does not match its admitted home")

                provisioning_row = connection.execute(
                    "SELECT * FROM project_provisioning_requests WHERE request_id = ?",
                    (canonical_request_id,),
                ).fetchone()
                if provisioning_row is None:
                    raise RuntimeError("target transfer lost its provisioning request")
                provisioning = self._project_provisioning_record(provisioning_row)
                restore_reentry = self._latest_project_transfer_restore_reentry(
                    connection,
                    canonical_request_id,
                )
                reviewed_revision = (
                    current.target_admission_receipt.target_preparation_revision
                    if restore_reentry is None
                    else restore_reentry.provisioning_revision
                )
                reviewed_digest = (
                    current.target_admission_receipt.target_preparation_sha256
                    if restore_reentry is None
                    else restore_reentry.provisioning_final_review_sha256
                )
                if (
                    provisioning.kind != "incoming_transfer"
                    or provisioning.status != "ready_for_review"
                    or provisioning.target_space_id != current.target_space_id
                    or provisioning.proposed_project_id != current.project_id
                    or provisioning.revision != expected_provisioning_revision
                    or provisioning.final_review_digest != expected_review
                    or reviewed_revision != expected_provisioning_revision
                    or reviewed_digest != expected_review
                    or (restore_reentry is not None and restore_reentry.confirmed_by != admitted_by)
                ):
                    raise ValueError("reviewed transfer provisioning changed before activation")

                upload_row = connection.execute(
                    "SELECT * FROM project_transfer_uploads WHERE request_id = ?",
                    (canonical_request_id,),
                ).fetchone()
                if upload_row is None:
                    raise ValueError("target activation requires its completed upload")
                upload = self._project_transfer_upload_record(upload_row)
                self._require_target_upload_matches(current, upload)
                if upload.status != "complete" or upload.receipt is None:
                    raise ValueError("target activation requires its completed upload")

                import_row = connection.execute(
                    "SELECT * FROM project_transfer_imports WHERE request_id = ?",
                    (canonical_request_id,),
                ).fetchone()
                if import_row is None:
                    raise ValueError("target activation requires its completed import")
                imported = self._project_transfer_import_record(import_row)
                if (
                    imported.status != "complete"
                    or imported.project_id != current.project_id
                    or imported.publication_sha256 is None
                    or imported.completed_at is None
                ):
                    raise ValueError("target activation requires its completed import")

                if (
                    connection.execute(
                        "SELECT 1 FROM projects WHERE project_id = ? OR locator = ?",
                        (current.project_id, prepared_project.locator),
                    ).fetchone()
                    is not None
                ):
                    raise ValueError("target activation project is already registered")
                if (
                    connection.execute(
                        "SELECT 1 FROM project_members WHERE project_id = ?",
                        (current.project_id,),
                    ).fetchone()
                    is not None
                ):
                    raise ValueError("target activation project already has a member seat")

                now = self.now()
                registered_project = ProjectTransferRegisteredProject(
                    project_id=prepared_project.project_id,
                    home_space_id=current.target_space_id,
                    locator=prepared_project.locator,
                    name=prepared_project.name,
                    state_location=prepared_project.state_location,
                    state_remote=prepared_project.state_remote,
                    revision=prepared_project.revision,
                    registered_at=prepared_project.added_at,
                )
                first_member = ProjectMemberRecord(
                    project_id=current.project_id,
                    user_id=admitted_by.user_id,
                    seated_at=now,
                    seated_by=admitted_by.user_id,
                )
                completed_provisioning = self._completed_transfer_provisioning(
                    provisioning,
                    completed_at=now,
                )
                activation = ProjectTransferActivationReceipt(
                    target_request_id=current.request_id,
                    source_request_id=current.linked_request_id,
                    project_id=current.project_id,
                    source_space_id=current.source_space_id,
                    target_space_id=current.target_space_id,
                    archive_sha256=current.archive_sha256,
                    source_fence_head=current.source_fence_head,
                    source_release_proof_sha256=current.source_release_proof_sha256,
                    target_activation_proof_sha256=current.target_activation_proof_sha256,
                    upload_lease_boundary_sha256=upload.lease_boundary_sha256,
                    upload_archive_sha256=upload.archive_sha256,
                    upload_archive_size_bytes=upload.archive_size_bytes,
                    upload_completed_at=upload.receipt.completed_at,
                    archive_manifest_sha256=imported.archive_manifest_sha256,
                    target_manifest_sha256=imported.target_manifest_sha256,
                    operational_payload_sha256=imported.operational_payload_sha256,
                    publication_sha256=imported.publication_sha256,
                    import_completed_at=imported.completed_at,
                    provisioning_request_id=provisioning.request_id,
                    provisioning_revision=completed_provisioning.revision,
                    provisioning_final_review_sha256=expected_review,
                    provisioning_completed_at=now,
                    admitted_by=admitted_by,
                    registered_project=registered_project,
                    first_member=first_member,
                    activated_at=now,
                )
                self._insert_target_activation_project(connection, prepared_project)
                connection.execute(
                    """
                    INSERT INTO project_members (project_id, user_id, seated_at, seated_by)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        first_member.project_id,
                        first_member.user_id,
                        first_member.seated_at,
                        first_member.seated_by,
                    ),
                )
                self._commit_transfer_provisioning_completion(
                    connection,
                    provisioning,
                    completed_provisioning,
                    registered_project,
                    admitted_by,
                )
                consumed = connection.execute(
                    """
                    UPDATE project_transfer_uploads
                    SET status = 'consumed', updated_at = ?
                    WHERE request_id = ? AND status = 'complete'
                    """,
                    (now, canonical_request_id),
                ).rowcount
                if consumed != 1:
                    raise RuntimeError("target activation lost its completed upload boundary")
                connection.execute(
                    """
                    INSERT INTO project_transfer_activations (
                        target_request_id, project_id, receipt_json, activated_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        activation.target_request_id,
                        activation.project_id,
                        _canonical_json(activation.model_dump(mode="json")),
                        activation.activated_at,
                    ),
                )
                updated = self._updated_project_transfer_record(
                    current,
                    phase="target_activated",
                )
                self._replace_project_transfer_record(connection, current, updated)
                stored = connection.execute(
                    "SELECT * FROM project_transfer_activations WHERE target_request_id = ?",
                    (canonical_request_id,),
                ).fetchone()
                if stored is None:
                    raise RuntimeError("target activation receipt disappeared during commit")
                return self._project_transfer_activation_record(stored)
        except sqlite3.IntegrityError as exc:
            raise ValueError("target activation conflicts with existing target state") from exc

    def target_project_transfer_activation(
        self,
        request_id: str,
    ) -> ProjectTransferActivationReceipt | None:
        """Read and validate one committed target activation receipt."""

        canonical_request_id = self._transfer_uuid(request_id, "transfer request identity")
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM project_transfer_activations WHERE target_request_id = ?",
                (canonical_request_id,),
            ).fetchone()
            if row is None:
                return None
            receipt = self._project_transfer_activation_record(row)
            current = self._project_transfer_request_from_connection(
                connection,
                canonical_request_id,
            )
            self._validate_target_activation_receipt(connection, current, receipt)
            return receipt

    def mark_target_project_transfer_activated(
        self,
        request_id: str,
    ) -> ProjectTransferRequestRecord:
        """Read an already-compound-activated request; never create a partial boundary."""

        activation = self.target_project_transfer_activation(request_id)
        if activation is None:
            raise ValueError("target activation requires its compound activation receipt")
        current = self.project_transfer_request(activation.target_request_id)
        if current is None:  # pragma: no cover - receipt validation already proves this
            raise RuntimeError("target activation receipt lost its transfer request")
        return current

    def reenter_restored_target_project_transfer(
        self,
        request_id: str,
        *,
        expected_restored_revision: int,
        expected_resume_phase: ProjectTransferPhase,
        expected_final_review_digest: str,
        confirmed_by: AuthorizedHuman,
    ) -> tuple[ProjectTransferRequestRecord, ProjectTransferUploadRecord]:
        """Revalidate one restored archive boundary and issue only its fresh relay lease."""

        canonical_request_id = self._transfer_uuid(request_id, "transfer request identity")
        if type(expected_restored_revision) is not int or expected_restored_revision < 0:
            raise ValueError("restored transfer revision is invalid")
        if expected_resume_phase != "archive_bound":
            raise ValueError("restored target import can resume only its archive-bound phase")
        expected_review = self._transfer_digest(
            expected_final_review_digest,
            "restored transfer final-review digest",
        )
        actor = AuthorizedHuman.model_validate_json(confirmed_by.model_dump_json())
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._project_transfer_request_from_connection(
                connection,
                canonical_request_id,
            )
            admission = current.target_admission_receipt
            if (
                current.side != "target"
                or admission is None
                or admission.admitted_by.space_id != actor.space_id
                or admission.admitted_by.user_id != actor.user_id
            ):
                raise ValueError("restored target transfer requires its exact confirmer")
            self._require_transfer_actor(connection, actor)
            confirmer_snapshot = admission.admitted_by

            provisioning_row = connection.execute(
                "SELECT * FROM project_provisioning_requests WHERE request_id = ?",
                (canonical_request_id,),
            ).fetchone()
            if provisioning_row is None:
                raise RuntimeError("restored target transfer lost its provisioning request")
            provisioning = self._project_provisioning_record(provisioning_row)
            if (
                provisioning.kind != "incoming_transfer"
                or provisioning.status != "ready_for_review"
                or provisioning.target_space_id != current.target_space_id
                or provisioning.proposed_project_id != current.project_id
                or provisioning.final_review_digest != expected_review
            ):
                raise ValueError("restored transfer provisioning is no longer ready for import")

            upload_row = connection.execute(
                "SELECT * FROM project_transfer_uploads WHERE request_id = ?",
                (canonical_request_id,),
            ).fetchone()
            if upload_row is None:
                raise ValueError("restored target transfer has no invalidated upload to replace")
            upload = self._project_transfer_upload_record(upload_row)
            self._require_target_upload_matches(current, upload)

            prior_row = connection.execute(
                """
                SELECT * FROM project_transfer_restore_reentries
                WHERE target_request_id = ? AND restored_revision = ?
                """,
                (canonical_request_id, expected_restored_revision),
            ).fetchone()
            if prior_row is not None:
                prior = self._project_transfer_restore_reentry_record(prior_row)
                if (
                    current.phase != "archive_bound"
                    or current.revision != expected_restored_revision + 1
                    or prior.resume_phase != expected_resume_phase
                    or prior.provisioning_revision != provisioning.revision
                    or prior.provisioning_final_review_sha256 != expected_review
                    or prior.confirmed_by != confirmer_snapshot
                    or upload.status not in {"active", "complete"}
                    or upload.lease_boundary_sha256 != prior.replacement_lease_boundary_sha256
                ):
                    raise RuntimeError(
                        "stored restored-transfer re-entry does not match its current boundary"
                    )
                return current, upload
            if (
                current.phase != "operator_action_needed"
                or current.revision != expected_restored_revision
                or current.restore_resume_phase != expected_resume_phase
            ):
                raise ValueError("restored target transfer changed before re-entry")
            if upload.status != "invalidated":
                raise ValueError("restored target transfer upload is not invalidated")
            if current.archive_sha256 is None or current.archive_size_bytes is None:
                raise RuntimeError("restored archive-bound transfer lost its archive metadata")

            now = self.now()
            replacement = ProjectTransferUploadRecord(
                request_id=current.request_id,
                project_id=current.project_id,
                archive_sha256=current.archive_sha256,
                archive_size_bytes=current.archive_size_bytes,
                lease_boundary_sha256=self._new_transfer_upload_boundary(
                    connection,
                    canonical_request_id,
                ),
                status="active",
                created_at=upload.created_at,
                updated_at=now,
            )
            reentry = ProjectTransferRestoreReentryReceipt(
                target_request_id=current.request_id,
                source_request_id=current.linked_request_id,
                project_id=current.project_id,
                source_space_id=current.source_space_id,
                target_space_id=current.target_space_id,
                restored_revision=current.revision,
                resume_phase="archive_bound",
                provisioning_revision=provisioning.revision,
                provisioning_final_review_sha256=expected_review,
                confirmed_by=confirmer_snapshot,
                archive_sha256=current.archive_sha256,
                archive_size_bytes=current.archive_size_bytes,
                replacement_lease_boundary_sha256=replacement.lease_boundary_sha256,
                created_at=now,
            )
            changed = connection.execute(
                """
                UPDATE project_transfer_uploads
                SET archive_sha256 = ?, archive_size_bytes = ?,
                    lease_boundary_sha256 = ?, status = 'active', receipt_json = NULL,
                    updated_at = ?, invalidated_at = NULL
                WHERE request_id = ? AND status = 'invalidated'
                  AND lease_boundary_sha256 = ?
                """,
                (
                    replacement.archive_sha256,
                    replacement.archive_size_bytes,
                    replacement.lease_boundary_sha256,
                    replacement.updated_at,
                    canonical_request_id,
                    upload.lease_boundary_sha256,
                ),
            ).rowcount
            if changed != 1:
                raise RuntimeError("restored target upload replacement lost its transaction guard")
            updated = self._updated_project_transfer_record(
                current,
                phase="archive_bound",
                restore_resume_phase=None,
                restore_diagnostic=None,
            )
            self._replace_project_transfer_record(connection, current, updated)
            connection.execute(
                """
                INSERT INTO project_transfer_restore_reentries (
                    target_request_id, restored_revision, receipt_json, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    reentry.target_request_id,
                    reentry.restored_revision,
                    _canonical_json(reentry.model_dump(mode="json")),
                    reentry.created_at,
                ),
            )
            return updated, replacement

    def expose_project_transfer_proof(self, request_id: str) -> bytes:
        """Return the local raw proof only after its committed legal boundary."""

        canonical_request_id = self._transfer_uuid(request_id, "transfer request identity")
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._project_transfer_request_from_connection(
                connection,
                canonical_request_id,
            )
            if current.proof_state == "consumed":
                raise ValueError("transfer proof was already consumed")
            legal = (
                current.side == "source" and current.phase in {"source_fenced", "archive_bound"}
            ) or (
                current.side == "target"
                and current.phase
                in {
                    "target_activated",
                    "cleanup_acknowledged",
                    "completed",
                }
            )
            if not legal:
                raise ValueError("transfer proof is not exposed at this protocol boundary")
            proof = connection.execute(
                "SELECT * FROM project_transfer_proofs WHERE request_id = ?",
                (canonical_request_id,),
            ).fetchone()
            if proof is None:
                raise RuntimeError("transfer request lost its protected proof")
            if proof["state"] == "consumed" or proof["secret"] is None:
                raise ValueError("transfer proof was already consumed")
            secret = bytes(proof["secret"])
            if (
                len(secret) != 32
                or hashlib.sha256(secret).hexdigest() != proof["commitment_sha256"]
            ):
                raise RuntimeError("stored transfer proof does not match its commitment")
            if proof["state"] == "unexposed":
                now = self.now()
                changed = connection.execute(
                    """
                    UPDATE project_transfer_proofs
                    SET state = 'exposed', exposed_at = ?
                    WHERE request_id = ? AND state = 'unexposed'
                    """,
                    (now, canonical_request_id),
                ).rowcount
                if changed != 1:
                    raise RuntimeError("transfer proof exposure lost its transaction guard")
                updated = self._updated_project_transfer_record(
                    current,
                    proof_state="exposed",
                )
                self._replace_project_transfer_record(connection, current, updated)
            elif proof["state"] not in {"exposed", "acknowledged"}:
                raise RuntimeError("stored transfer proof state is invalid")
        return secret

    def verify_target_project_transfer_activation(
        self,
        request_id: str,
        *,
        proof: bytes,
    ) -> ProjectTransferCleanupAcknowledgment:
        """Verify target activation on the source and publish one bound acknowledgment."""

        canonical_request_id = self._transfer_uuid(request_id, "transfer request identity")
        if not isinstance(proof, bytes) or len(proof) != 32:
            raise ValueError("target activation proof must be exactly 32 bytes")
        commitment = hashlib.sha256(proof).hexdigest()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._project_transfer_request_from_connection(
                connection,
                canonical_request_id,
            )
            if (
                current.side != "source"
                or current.phase not in {"archive_bound", "cleanup_acknowledged", "completed"}
                or current.target_activation_proof_sha256 != commitment
                or current.linked_request_id is None
                or current.source_fence_head is None
                or current.archive_sha256 is None
            ):
                raise ValueError("target activation proof does not match this source transfer")
            acknowledgment = ProjectTransferCleanupAcknowledgment(
                source_request_id=current.request_id,
                target_request_id=current.linked_request_id,
                project_id=current.project_id,
                source_space_id=current.source_space_id,
                target_space_id=current.target_space_id,
                source_release_proof_sha256=current.source_release_proof_sha256,
                target_activation_proof_sha256=commitment,
                archive_sha256=current.archive_sha256,
                source_fence_head=current.source_fence_head,
            )
            acknowledgment_sha256 = project_transfer_receipt_sha256(acknowledgment)
            protected = connection.execute(
                "SELECT * FROM project_transfer_proofs WHERE request_id = ?",
                (canonical_request_id,),
            ).fetchone()
            if protected is None:
                raise RuntimeError("transfer request lost its protected proof")
            if protected["state"] in {"acknowledged", "consumed"}:
                if protected["acknowledgement_sha256"] != acknowledgment_sha256:
                    raise ValueError("source transfer already verified another target boundary")
                return acknowledgment
            if protected["state"] != "exposed" or current.phase != "archive_bound":
                raise ValueError("source transfer proof is not ready for target verification")
            now = self.now()
            changed = connection.execute(
                """
                UPDATE project_transfer_proofs
                SET state = 'acknowledged', acknowledgement_sha256 = ?, acknowledged_at = ?
                WHERE request_id = ? AND state = 'exposed'
                """,
                (acknowledgment_sha256, now, canonical_request_id),
            ).rowcount
            if changed != 1:
                raise RuntimeError("source transfer proof verification lost its transaction guard")
            updated = self._updated_project_transfer_record(
                current,
                proof_state="acknowledged",
                proof_acknowledgement_sha256=acknowledgment_sha256,
            )
            self._replace_project_transfer_record(connection, current, updated)
        return acknowledgment

    def accept_project_transfer_cleanup_acknowledgment(
        self,
        request_id: str,
        *,
        acknowledgment: ProjectTransferCleanupAcknowledgment,
        accepted_by: AuthorizedHuman,
    ) -> ProjectTransferRequestRecord:
        """Consume the target proof only for the source backend's exact acknowledgment."""

        canonical_request_id = self._transfer_uuid(request_id, "transfer request identity")
        normalized = ProjectTransferCleanupAcknowledgment.model_validate_json(
            acknowledgment.model_dump_json()
        )
        actor = AuthorizedHuman.model_validate(accepted_by.model_dump(mode="json"))
        acknowledgment_sha256 = project_transfer_receipt_sha256(normalized)
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._project_transfer_request_from_connection(
                connection,
                canonical_request_id,
            )
            self._require_transfer_actor(connection, actor)
            self._require_cleanup_acknowledgment_matches(current, normalized)
            admission = current.target_admission_receipt
            if admission is None or admission.admitted_by != actor:
                raise ValueError("only the target confirmer may accept transfer cleanup")
            protected = connection.execute(
                "SELECT * FROM project_transfer_proofs WHERE request_id = ?",
                (canonical_request_id,),
            ).fetchone()
            if protected is None:
                raise RuntimeError("transfer request lost its protected proof")
            if current.phase == "completed":
                if (
                    current.proof_state != "consumed"
                    or protected["state"] != "consumed"
                    or protected["acknowledgement_sha256"] != acknowledgment_sha256
                ):
                    raise ValueError("completed transfer has another cleanup acknowledgment")
                return current
            if current.phase != "target_activated" or current.proof_state != "exposed":
                raise ValueError("target transfer is not ready for cleanup")
            now = self.now()
            changed = connection.execute(
                """
                UPDATE project_transfer_proofs
                SET state = 'consumed', acknowledgement_sha256 = ?,
                    acknowledged_at = ?, consumed_at = ?, secret = NULL
                WHERE request_id = ? AND state = 'exposed'
                """,
                (acknowledgment_sha256, now, now, canonical_request_id),
            ).rowcount
            if changed != 1:
                raise RuntimeError("target transfer cleanup lost its transaction guard")
            updated = self._updated_project_transfer_record(
                current,
                phase="completed",
                proof_state="consumed",
                proof_acknowledgement_sha256=acknowledgment_sha256,
            )
            self._replace_project_transfer_record(connection, current, updated)
        return updated

    def acknowledge_project_transfer_proof(
        self,
        request_id: str,
        *,
        acknowledgement_sha256: str,
    ) -> ProjectTransferRequestRecord:
        """Retain a public verification receipt before erasing the local raw proof."""

        canonical_request_id = self._transfer_uuid(request_id, "transfer request identity")
        acknowledgement = self._transfer_digest(
            acknowledgement_sha256,
            "transfer proof acknowledgment",
        )
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._project_transfer_request_from_connection(
                connection,
                canonical_request_id,
            )
            proof = connection.execute(
                "SELECT * FROM project_transfer_proofs WHERE request_id = ?",
                (canonical_request_id,),
            ).fetchone()
            if proof is None:
                raise RuntimeError("transfer request lost its protected proof")
            if proof["state"] in {"acknowledged", "consumed"}:
                if proof["acknowledgement_sha256"] != acknowledgement:
                    raise ValueError("transfer proof already has another acknowledgment")
                return current
            if proof["state"] != "exposed":
                raise ValueError("transfer proof cannot be acknowledged before exposure")
            expected_phase = "archive_bound" if current.side == "source" else "target_activated"
            if current.phase != expected_phase:
                raise ValueError("transfer proof cannot be acknowledged before its boundary")
            now = self.now()
            connection.execute(
                """
                UPDATE project_transfer_proofs
                SET state = 'acknowledged', acknowledgement_sha256 = ?, acknowledged_at = ?
                WHERE request_id = ? AND state = 'exposed'
                """,
                (acknowledgement, now, canonical_request_id),
            )
            updated = self._updated_project_transfer_record(
                current,
                proof_state="acknowledged",
                proof_acknowledgement_sha256=acknowledgement,
            )
            self._replace_project_transfer_record(connection, current, updated)
        return updated

    def consume_project_transfer_proof(
        self,
        request_id: str,
        *,
        acknowledgement_sha256: str,
    ) -> ProjectTransferRequestRecord:
        """Erase one acknowledged raw proof while retaining commitments and receipts."""

        canonical_request_id = self._transfer_uuid(request_id, "transfer request identity")
        acknowledgement = self._transfer_digest(
            acknowledgement_sha256,
            "transfer proof acknowledgment",
        )
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._project_transfer_request_from_connection(
                connection,
                canonical_request_id,
            )
            proof = connection.execute(
                "SELECT * FROM project_transfer_proofs WHERE request_id = ?",
                (canonical_request_id,),
            ).fetchone()
            if proof is None:
                raise RuntimeError("transfer request lost its protected proof")
            if proof["acknowledgement_sha256"] != acknowledgement:
                raise ValueError("transfer proof acknowledgment does not match")
            if proof["state"] == "consumed":
                return current
            if proof["state"] != "acknowledged":
                raise ValueError("transfer proof must be acknowledged before consumption")
            if current.phase != "cleanup_acknowledged":
                raise ValueError("transfer proof cannot be consumed before cleanup acknowledgment")
            now = self.now()
            changed = connection.execute(
                """
                UPDATE project_transfer_proofs
                SET state = 'consumed', secret = NULL, consumed_at = ?
                WHERE request_id = ? AND state = 'acknowledged'
                """,
                (now, canonical_request_id),
            ).rowcount
            if changed != 1:
                raise RuntimeError("transfer proof consumption lost its transaction guard")
            updated = self._updated_project_transfer_record(current, proof_state="consumed")
            self._replace_project_transfer_record(connection, current, updated)
        return updated

    def complete_project_transfer_request(
        self,
        request_id: str,
    ) -> ProjectTransferRequestRecord:
        """Close one side only after its raw proof has been consumed."""

        canonical_request_id = self._transfer_uuid(request_id, "transfer request identity")
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._project_transfer_request_from_connection(
                connection,
                canonical_request_id,
            )
            if current.phase == "completed":
                return current
            if current.phase != "cleanup_acknowledged" or current.proof_state != "consumed":
                raise ValueError("transfer request cannot complete before proof consumption")
            updated = self._updated_project_transfer_record(current, phase="completed")
            self._replace_project_transfer_record(connection, current, updated)
        return updated

    def acknowledge_project_transfer_cleanup(
        self,
        request_id: str,
    ) -> ProjectTransferRequestRecord:
        """Record the public cleanup boundary after the opposite proof was verified."""

        canonical_request_id = self._transfer_uuid(request_id, "transfer request identity")
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._project_transfer_request_from_connection(
                connection,
                canonical_request_id,
            )
            if current.phase in {"cleanup_acknowledged", "completed"}:
                return current
            expected_phase = "archive_bound" if current.side == "source" else "target_activated"
            if current.phase != expected_phase or current.proof_state != "acknowledged":
                raise ValueError("transfer cleanup cannot be acknowledged at this boundary")
            updated = self._updated_project_transfer_record(
                current,
                phase="cleanup_acknowledged",
            )
            self._replace_project_transfer_record(connection, current, updated)
        return updated

    @staticmethod
    def _require_target_upload_request(
        current: ProjectTransferRequestRecord,
    ) -> None:
        if current.side != "target":
            raise ValueError("only a target transfer request may own an upload")
        if current.phase != "archive_bound":
            if current.phase == "operator_action_needed":
                raise ValueError("target transfer upload is unavailable until restore re-entry")
            raise ValueError("target transfer must be archive-bound for upload")
        if current.archive_sha256 is None or current.archive_size_bytes is None:
            raise RuntimeError("archive-bound target transfer lost its archive metadata")

    @staticmethod
    def _require_target_upload_matches(
        current: ProjectTransferRequestRecord,
        upload: ProjectTransferUploadRecord,
    ) -> None:
        if (
            current.side != "target"
            or upload.request_id != current.request_id
            or upload.project_id != current.project_id
            or upload.archive_sha256 != current.archive_sha256
            or upload.archive_size_bytes != current.archive_size_bytes
        ):
            raise RuntimeError("stored project transfer upload does not match its target request")

    @staticmethod
    def _new_transfer_upload_boundary(
        connection: sqlite3.Connection,
        request_id: str,
    ) -> str:
        while True:
            boundary = secrets.token_hex(32)
            if (
                connection.execute(
                    """
                    SELECT 1 FROM project_transfer_uploads
                    WHERE request_id = ? AND lease_boundary_sha256 = ?
                    """,
                    (request_id, boundary),
                ).fetchone()
                is None
            ):
                return boundary

    @staticmethod
    def _project_transfer_upload_record(
        row: sqlite3.Row,
    ) -> ProjectTransferUploadRecord:
        try:
            receipt = (
                None
                if row["receipt_json"] is None
                else ProjectTransferUploadCompleteReceipt.model_validate_json(row["receipt_json"])
            )
            record = ProjectTransferUploadRecord(
                request_id=row["request_id"],
                project_id=row["project_id"],
                archive_sha256=row["archive_sha256"],
                archive_size_bytes=row["archive_size_bytes"],
                lease_boundary_sha256=row["lease_boundary_sha256"],
                status=row["status"],
                receipt=receipt,
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                invalidated_at=row["invalidated_at"],
            )
            return record
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("stored project transfer upload is invalid") from exc

    @staticmethod
    def _project_transfer_activation_record(
        row: sqlite3.Row,
    ) -> ProjectTransferActivationReceipt:
        try:
            receipt = ProjectTransferActivationReceipt.model_validate_json(row["receipt_json"])
            if (
                row["target_request_id"] != receipt.target_request_id
                or row["project_id"] != receipt.project_id
                or row["activated_at"] != receipt.activated_at
            ):
                raise ValueError("activation receipt columns do not match its payload")
            return receipt
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("stored project transfer activation receipt is invalid") from exc

    @staticmethod
    def _project_transfer_restore_reentry_record(
        row: sqlite3.Row,
    ) -> ProjectTransferRestoreReentryReceipt:
        try:
            receipt = ProjectTransferRestoreReentryReceipt.model_validate_json(row["receipt_json"])
            if (
                row["target_request_id"] != receipt.target_request_id
                or row["restored_revision"] != receipt.restored_revision
                or row["created_at"] != receipt.created_at
            ):
                raise ValueError("restored-transfer receipt columns do not match its payload")
            return receipt
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("stored project transfer restore re-entry is invalid") from exc

    @classmethod
    def _latest_project_transfer_restore_reentry(
        cls,
        connection: sqlite3.Connection,
        request_id: str,
    ) -> ProjectTransferRestoreReentryReceipt | None:
        row = connection.execute(
            """
            SELECT * FROM project_transfer_restore_reentries
            WHERE target_request_id = ?
            ORDER BY restored_revision DESC
            LIMIT 1
            """,
            (request_id,),
        ).fetchone()
        return None if row is None else cls._project_transfer_restore_reentry_record(row)

    @staticmethod
    def _project_transfer_import_record(row: sqlite3.Row) -> ProjectTransferImportRecord:
        from rcp.storage.transfer import ProjectTransferStoreMixin

        return ProjectTransferStoreMixin._project_transfer_import_record(row)

    @staticmethod
    def _activation_project_intent(
        project: ProjectRecord | ProjectTransferRegisteredProject,
    ) -> tuple[object, ...]:
        return (
            project.project_id,
            project.home_space_id,
            project.locator,
            project.name,
            project.state_location,
            project.state_remote,
            project.revision,
            project.added_at if isinstance(project, ProjectRecord) else project.registered_at,
        )

    @staticmethod
    def _completed_transfer_provisioning(
        current: ProjectProvisioningRequestRecord,
        *,
        completed_at: str,
    ) -> ProjectProvisioningRequestRecord:
        values = current.model_dump(mode="json")
        values.update(
            {
                "status": "completed",
                "revision": current.revision + 1,
                "updated_at": completed_at,
                "completed_at": completed_at,
            }
        )
        return _verify_project_provisioning_review_digest(
            ProjectProvisioningRequestRecord.model_validate_json(_canonical_json(values))
        )

    @staticmethod
    def _insert_target_activation_project(
        connection: sqlite3.Connection,
        project: ProjectRecord,
    ) -> None:
        connection.execute(
            """
            INSERT INTO projects (
                project_id, home_space_id, locator, name, state_location, state_remote,
                added_at, last_opened_at, revision, primary_question, attention_count,
                last_refresh_at, reachable, error, retired_at, retired_transfer_request_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
            """,
            (
                project.project_id,
                project.home_space_id,
                project.locator,
                project.name,
                project.state_location,
                int(project.state_remote),
                project.added_at,
                project.last_opened_at,
                project.revision,
                project.primary_question,
                project.attention_count,
                project.last_refresh_at,
                None if project.reachable is None else int(project.reachable),
                project.error,
            ),
        )

    @staticmethod
    def _commit_transfer_provisioning_completion(
        connection: sqlite3.Connection,
        current: ProjectProvisioningRequestRecord,
        completed: ProjectProvisioningRequestRecord,
        registered_project: ProjectTransferRegisteredProject,
        admitted_by: AuthorizedHuman,
    ) -> None:
        transition_payload = {
            "request_id": current.request_id,
            "phase": "transfer_activation",
            "expected_revision": current.revision,
            "final_review_digest": current.final_review_digest,
            "registered_project": registered_project.model_dump(mode="json"),
            "admitted_by": admitted_by.model_dump(mode="json"),
        }
        transition_sha256 = hashlib.sha256(
            _canonical_json(transition_payload).encode("utf-8")
        ).hexdigest()
        changed = connection.execute(
            """
            UPDATE project_provisioning_requests
            SET status = 'completed', revision = ?, updated_at = ?, completed_at = ?
            WHERE request_id = ? AND revision = ? AND status = 'ready_for_review'
              AND final_review_digest = ?
            """,
            (
                completed.revision,
                completed.updated_at,
                completed.completed_at,
                current.request_id,
                current.revision,
                current.final_review_digest,
            ),
        ).rowcount
        if changed != 1:
            raise RuntimeError("target activation lost its provisioning transaction guard")
        receipt = ProjectProvisioningStepReceiptRecord(
            request_id=current.request_id,
            receipt_id="transfer-activation",
            phase="transfer_activation",
            from_status="ready_for_review",
            to_status="completed",
            transition_sha256=transition_sha256,
            resulting_revision=completed.revision,
            created_at=completed.completed_at,
        )
        connection.execute(
            """
            INSERT INTO project_provisioning_step_receipts (
                request_id, receipt_id, phase, from_status, to_status,
                transition_sha256, resulting_revision, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.request_id,
                receipt.receipt_id,
                receipt.phase,
                receipt.from_status,
                receipt.to_status,
                receipt.transition_sha256,
                receipt.resulting_revision,
                receipt.created_at,
            ),
        )

    @classmethod
    def _validate_target_activation_receipt(
        cls,
        connection: sqlite3.Connection,
        current: ProjectTransferRequestRecord,
        receipt: ProjectTransferActivationReceipt,
    ) -> None:
        admission = current.target_admission_receipt
        effective_phase = (
            current.restore_resume_phase
            if current.phase == "operator_action_needed"
            else current.phase
        )
        if (
            current.side != "target"
            or effective_phase not in {"target_activated", "cleanup_acknowledged", "completed"}
            or current.linked_request_id != receipt.source_request_id
            or current.request_id != receipt.target_request_id
            or current.project_id != receipt.project_id
            or current.source_space_id != receipt.source_space_id
            or current.target_space_id != receipt.target_space_id
            or current.archive_sha256 != receipt.archive_sha256
            or current.source_fence_head != receipt.source_fence_head
            or current.source_release_proof_sha256 != receipt.source_release_proof_sha256
            or current.target_activation_proof_sha256 != receipt.target_activation_proof_sha256
            or admission is None
            or admission.admitted_by != receipt.admitted_by
        ):
            raise RuntimeError("stored target activation does not match its transfer request")

        upload_row = connection.execute(
            "SELECT * FROM project_transfer_uploads WHERE request_id = ?",
            (current.request_id,),
        ).fetchone()
        if upload_row is None:
            raise RuntimeError("stored target activation lost its upload receipt")
        upload = cls._project_transfer_upload_record(upload_row)
        if (
            upload.status != "consumed"
            or upload.receipt is None
            or upload.lease_boundary_sha256 != receipt.upload_lease_boundary_sha256
            or upload.archive_sha256 != receipt.upload_archive_sha256
            or upload.archive_size_bytes != receipt.upload_archive_size_bytes
            or upload.receipt.completed_at != receipt.upload_completed_at
        ):
            raise RuntimeError("stored target activation does not match its upload receipt")

        import_row = connection.execute(
            "SELECT * FROM project_transfer_imports WHERE request_id = ?",
            (current.request_id,),
        ).fetchone()
        if import_row is None:
            raise RuntimeError("stored target activation lost its import receipt")
        imported = cls._project_transfer_import_record(import_row)
        if (
            imported.status != "complete"
            or imported.archive_manifest_sha256 != receipt.archive_manifest_sha256
            or imported.target_manifest_sha256 != receipt.target_manifest_sha256
            or imported.operational_payload_sha256 != receipt.operational_payload_sha256
            or imported.publication_sha256 != receipt.publication_sha256
            or imported.completed_at != receipt.import_completed_at
        ):
            raise RuntimeError("stored target activation does not match its import receipt")

        provisioning_row = connection.execute(
            "SELECT * FROM project_provisioning_requests WHERE request_id = ?",
            (current.request_id,),
        ).fetchone()
        if provisioning_row is None:
            raise RuntimeError("stored target activation lost its provisioning receipt")
        provisioning = cls._project_provisioning_record(provisioning_row)
        if (
            provisioning.status != "completed"
            or provisioning.revision != receipt.provisioning_revision
            or provisioning.final_review_digest != receipt.provisioning_final_review_sha256
            or provisioning.completed_at != receipt.provisioning_completed_at
        ):
            raise RuntimeError("stored target activation does not match completed provisioning")

        project_row = connection.execute(
            "SELECT project_id, home_space_id FROM projects WHERE project_id = ?",
            (current.project_id,),
        ).fetchone()
        if (
            project_row is None
            or project_row["project_id"] != receipt.registered_project.project_id
            or project_row["home_space_id"] != receipt.registered_project.home_space_id
        ):
            raise RuntimeError("stored target activation lost its registered target project")

    @classmethod
    def _invalidate_project_transfer_upload(
        cls,
        connection: sqlite3.Connection,
        current: ProjectTransferRequestRecord,
        *,
        now: str,
    ) -> None:
        row = connection.execute(
            "SELECT * FROM project_transfer_uploads WHERE request_id = ?",
            (current.request_id,),
        ).fetchone()
        if row is None:
            return
        upload = cls._project_transfer_upload_record(row)
        cls._require_target_upload_matches(current, upload)
        if upload.status in {"invalidated", "consumed"}:
            return
        changed = connection.execute(
            """
            UPDATE project_transfer_uploads
            SET status = 'invalidated', receipt_json = NULL,
                invalidated_at = ?, updated_at = ?
            WHERE request_id = ? AND status IN ('active', 'complete')
            """,
            (now, now, current.request_id),
        ).rowcount
        if changed != 1:
            raise RuntimeError("restore upload invalidation lost its transaction guard")

    @staticmethod
    def _transfer_uuid(value: str, label: str) -> str:
        try:
            return _canonical_uuid4(value, label=label)
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc

    @staticmethod
    def _transfer_digest(value: str, label: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{label} must be lowercase SHA-256")
        return value

    @staticmethod
    def _require_transfer_space(
        connection: sqlite3.Connection,
        *,
        expected_kind: str,
        expected_space_id: str,
    ) -> None:
        identity = connection.execute(
            "SELECT space_id, space_kind FROM space_identity WHERE singleton = 1"
        ).fetchone()
        if (
            identity is None
            or identity["space_id"] != expected_space_id
            or identity["space_kind"] != expected_kind
        ):
            raise ValueError("project transfer requires this exact source or target space")

    @staticmethod
    def _require_transfer_actor(
        connection: sqlite3.Connection,
        actor: AuthorizedHuman,
    ) -> None:
        identity = connection.execute(
            "SELECT space_id FROM space_identity WHERE singleton = 1"
        ).fetchone()
        member = connection.execute(
            """
            SELECT user_id, removal_started_at, removed_at
            FROM space_users WHERE user_id = ?
            """,
            (actor.user_id,),
        ).fetchone()
        if (
            identity is None
            or actor.space_id != identity["space_id"]
            or member is None
            or member["removal_started_at"] is not None
            or member["removed_at"] is not None
        ):
            raise ValueError("project transfer actor is not current in this exact space")

    @staticmethod
    def _insert_project_transfer_request(
        connection: sqlite3.Connection,
        record: ProjectTransferRequestRecord,
        *,
        proof_kind: str,
        secret: bytes,
    ) -> None:
        commitment = hashlib.sha256(secret).hexdigest()
        expected_commitment = (
            record.source_release_proof_sha256
            if record.side == "source"
            else record.target_activation_proof_sha256
        )
        if len(secret) != 32 or commitment != expected_commitment:
            raise ValueError("transfer proof does not match its request commitment")
        connection.execute(
            """
            INSERT INTO project_transfer_requests (
                request_id, side, phase, project_id, source_space_id,
                target_space_id, linked_request_id, record_json, revision,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.request_id,
                record.side,
                record.phase,
                record.project_id,
                record.source_space_id,
                record.target_space_id,
                record.linked_request_id,
                _canonical_json(record.model_dump(mode="json")),
                record.revision,
                record.created_at,
                record.updated_at,
            ),
        )
        connection.execute(
            """
            INSERT INTO project_transfer_proofs (
                request_id, proof_kind, state, commitment_sha256, secret
            ) VALUES (?, ?, 'unexposed', ?, ?)
            """,
            (record.request_id, proof_kind, commitment, secret),
        )

    def _project_transfer_request_from_connection(
        self,
        connection: sqlite3.Connection,
        request_id: str,
    ) -> ProjectTransferRequestRecord:
        row = connection.execute(
            "SELECT * FROM project_transfer_requests WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if row is None:
            raise KeyError(request_id)
        record = self._project_transfer_record(row)
        proof = connection.execute(
            "SELECT * FROM project_transfer_proofs WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if proof is None:
            raise RuntimeError("stored transfer request lost its protected proof")
        expected_kind = "source_release" if record.side == "source" else "target_activation"
        expected_commitment = (
            record.source_release_proof_sha256
            if record.side == "source"
            else record.target_activation_proof_sha256
        )
        if (
            proof["proof_kind"] != expected_kind
            or proof["commitment_sha256"] != expected_commitment
            or proof["state"] != record.proof_state
            or proof["acknowledgement_sha256"] != record.proof_acknowledgement_sha256
        ):
            raise RuntimeError("stored transfer proof does not match its public request state")
        return record

    @classmethod
    def _project_transfer_record(
        cls,
        row: sqlite3.Row,
    ) -> ProjectTransferRequestRecord:
        try:
            record = ProjectTransferRequestRecord.model_validate_json(row["record_json"])
            if (
                record.request_id != row["request_id"]
                or record.side != row["side"]
                or record.phase != row["phase"]
                or record.project_id != row["project_id"]
                or record.source_space_id != row["source_space_id"]
                or record.target_space_id != row["target_space_id"]
                or record.linked_request_id != row["linked_request_id"]
                or record.revision != row["revision"]
                or record.created_at != row["created_at"]
                or record.updated_at != row["updated_at"]
                or project_transfer_source_configuration_sha256(record.source_configuration)
                != record.source_configuration_sha256
            ):
                raise ValueError("stored transfer request columns do not match its payload")
            if record.target_admission_receipt is not None:
                cls._require_target_admission_matches(
                    record,
                    record.target_admission_receipt,
                )
            if record.link_receipt is not None:
                cls._require_link_receipt_matches(record, record.link_receipt)
            if record.source_release_receipt is not None:
                cls._require_source_release_matches(
                    record,
                    record.source_release_receipt,
                )
            return record
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise RuntimeError("stored project transfer request is invalid") from exc

    def _updated_project_transfer_record(
        self,
        current: ProjectTransferRequestRecord,
        **changes: object,
    ) -> ProjectTransferRequestRecord:
        return ProjectTransferRequestRecord.model_validate_json(
            _canonical_json(
                {
                    **current.model_dump(mode="json"),
                    **changes,
                    "revision": current.revision + 1,
                    "updated_at": self.now(),
                }
            )
        )

    @staticmethod
    def _replace_project_transfer_record(
        connection: sqlite3.Connection,
        current: ProjectTransferRequestRecord,
        updated: ProjectTransferRequestRecord,
    ) -> None:
        changed = connection.execute(
            """
            UPDATE project_transfer_requests
            SET phase = ?, linked_request_id = ?, record_json = ?, revision = ?, updated_at = ?
            WHERE request_id = ? AND revision = ?
            """,
            (
                updated.phase,
                updated.linked_request_id,
                _canonical_json(updated.model_dump(mode="json")),
                updated.revision,
                updated.updated_at,
                current.request_id,
                current.revision,
            ),
        ).rowcount
        if changed != 1:
            raise RuntimeError("project transfer update lost its transaction guard")

    @staticmethod
    def _require_link_receipt_matches(
        current: ProjectTransferRequestRecord,
        receipt: ProjectTransferLinkReceipt,
    ) -> None:
        expected_source_request_id = (
            current.request_id if current.side == "source" else current.linked_request_id
        )
        expected_target_request_id = (
            current.linked_request_id if current.side == "source" else current.request_id
        )
        source_repositories = {
            repository.alias: repository.repository.identity
            for repository in current.source_configuration.repositories
        }
        target_repositories = {
            repository.alias: repository.repository.identity
            for repository in receipt.target_repositories
        }
        if (
            expected_source_request_id is None
            or receipt.source_request_id != expected_source_request_id
            or (
                expected_target_request_id is not None
                and receipt.target_request_id != expected_target_request_id
            )
            or receipt.project_id != current.project_id
            or receipt.source_space_id != current.source_space_id
            or receipt.target_space_id != current.target_space_id
            or receipt.source_configuration_sha256 != current.source_configuration_sha256
            or receipt.accepted_schema_generation
            != current.source_configuration.source_schema_generation
            or receipt.accepted_archive_codec
            not in current.source_configuration.supported_archive_codecs
            or receipt.source_release_proof_sha256 != current.source_release_proof_sha256
            or source_repositories != target_repositories
        ):
            raise ValueError("transfer link receipt does not match the source request")
        if current.link_receipt is not None and (
            receipt != current.link_receipt
            or receipt.target_activation_proof_sha256 != current.target_activation_proof_sha256
            or receipt.accepted_schema_generation != current.accepted_schema_generation
            or receipt.accepted_archive_codec != current.accepted_archive_codec
        ):
            raise ValueError("transfer link receipt does not match the target request")

    @staticmethod
    def _require_target_admission_matches(
        current: ProjectTransferRequestRecord,
        receipt: ProjectTransferTargetAdmissionReceipt,
    ) -> None:
        source_request_id = (
            current.request_id if current.side == "source" else current.linked_request_id
        )
        target_request_id = (
            current.linked_request_id if current.side == "source" else current.request_id
        )
        if (
            source_request_id is None
            or target_request_id is None
            or receipt.source_request_id != source_request_id
            or receipt.target_request_id != target_request_id
            or receipt.project_id != current.project_id
            or receipt.source_space_id != current.source_space_id
            or receipt.target_space_id != current.target_space_id
            or receipt.source_configuration_sha256 != current.source_configuration_sha256
            or receipt.accepted_schema_generation != current.accepted_schema_generation
            or receipt.accepted_archive_codec != current.accepted_archive_codec
            or receipt.source_release_proof_sha256 != current.source_release_proof_sha256
            or receipt.target_activation_proof_sha256 != current.target_activation_proof_sha256
        ):
            raise ValueError("target admission receipt does not match the linked source request")

    @staticmethod
    def _require_source_release_matches(
        current: ProjectTransferRequestRecord,
        receipt: ProjectTransferSourceReleaseReceipt,
    ) -> None:
        admission = current.target_admission_receipt
        source_request_id = (
            current.request_id if current.side == "source" else current.linked_request_id
        )
        target_request_id = (
            current.linked_request_id if current.side == "source" else current.request_id
        )
        if (
            admission is None
            or source_request_id is None
            or target_request_id is None
            or receipt.source_request_id != source_request_id
            or receipt.target_request_id != target_request_id
            or receipt.project_id != current.project_id
            or receipt.source_space_id != current.source_space_id
            or receipt.target_space_id != current.target_space_id
            or receipt.source_configuration_sha256 != current.source_configuration_sha256
            or receipt.target_admission_sha256 != project_transfer_receipt_sha256(admission)
            or receipt.target_preparation_revision != admission.target_preparation_revision
            or receipt.target_preparation_sha256 != admission.target_preparation_sha256
            or receipt.accepted_schema_generation != current.accepted_schema_generation
            or receipt.accepted_archive_codec != current.accepted_archive_codec
            or receipt.source_release_proof_sha256 != current.source_release_proof_sha256
            or receipt.target_activation_proof_sha256 != current.target_activation_proof_sha256
        ):
            raise ValueError("source release receipt does not match the admitted target request")

    @staticmethod
    def _require_cleanup_acknowledgment_matches(
        current: ProjectTransferRequestRecord,
        acknowledgment: ProjectTransferCleanupAcknowledgment,
    ) -> None:
        if (
            current.side != "target"
            or current.linked_request_id is None
            or current.target_activation_proof_sha256 is None
            or current.source_fence_head is None
            or current.archive_sha256 is None
            or acknowledgment.source_request_id != current.linked_request_id
            or acknowledgment.target_request_id != current.request_id
            or acknowledgment.project_id != current.project_id
            or acknowledgment.source_space_id != current.source_space_id
            or acknowledgment.target_space_id != current.target_space_id
            or acknowledgment.source_release_proof_sha256 != current.source_release_proof_sha256
            or acknowledgment.target_activation_proof_sha256
            != current.target_activation_proof_sha256
            or acknowledgment.archive_sha256 != current.archive_sha256
            or acknowledgment.source_fence_head != current.source_fence_head
        ):
            raise ValueError("transfer cleanup acknowledgment does not match the target request")

    def detach_project_provisioning_for_restore(
        self,
        connection: sqlite3.Connection,
        *,
        diagnostic: str,
        now: str,
    ) -> None:
        """Invalidate captured machine progress while preserving completed receipts."""

        if not connection.in_transaction:
            raise RuntimeError("restore provisioning detachment requires one active transaction")
        normalized = _MESSAGE_TEXT_ADAPTER.validate_python(diagnostic)
        rows = connection.execute(
            """
            SELECT * FROM project_provisioning_requests
            WHERE status NOT IN ('completed', 'cancelled')
            ORDER BY request_id
            """
        ).fetchall()
        for row in rows:
            current = self._project_provisioning_record(row)
            if (
                current.status == "operator_action_needed"
                and current.retryable_diagnostic == normalized
                and current.operator_action is not None
                and current.operator_action.phase == "restore_reentry"
            ):
                continue
            machines = [
                machine.model_copy(update={"resolved_central_root": None})
                for machine in current.machines
            ]
            repositories = [
                repository.model_copy(
                    update={
                        "resolved_path": None,
                        "checkout_disposition": None,
                        "git_check": ProjectProvisioningGitCheckRecord(),
                    }
                )
                for repository in current.repositories
            ]
            provider_checks = [
                ProjectProvisioningProviderCheckRecord(
                    **check.model_dump(
                        include={
                            "profile",
                            "provider",
                            "runtime_id",
                            "model",
                            "reasoning",
                            "machine_alias",
                        }
                    )
                )
                for check in current.provider_checks
            ]
            resume = (
                str(DEFAULT_SERVER_LAYOUT.cli_wrapper),
                "server",
                "project",
                "provision",
                current.request_id,
            )
            action = ServerStep(
                number=1,
                title="Resume restored project setup",
                purpose=(
                    "Recheck replacement-machine paths, repository keys, checkouts, and provider "
                    "readiness before this unfinished request can continue."
                ),
                performed_by="human",
                target=MachineTarget(
                    host=(
                        machines[0].host if machines[0].location == "ssh" else "replacement-server"
                    ),
                    os_account=machines[0].os_account,
                ),
                phase="restore_reentry",
                state="operator_action_needed",
                expected_success=(
                    "The replacement machine publishes fresh setup receipts for this request."
                ),
                message=(
                    "The archived request lost every old machine claim during replacement restore."
                ),
                actions=(CommandAction(argv=resume),),
                resume_argv=resume,
            )
            connection.execute(
                """
                UPDATE project_provisioning_requests
                SET status = 'operator_action_needed', machines_json = ?,
                    repositories_json = ?, provider_checks_json = ?,
                    retryable_diagnostic = ?, operator_action_json = ?,
                    final_review_digest = NULL, revision = revision + 1,
                    updated_at = ?, setup_started_at = COALESCE(setup_started_at, ?),
                    ready_at = NULL
                WHERE request_id = ?
                """,
                (
                    _canonical_json([item.model_dump(mode="json") for item in machines]),
                    _canonical_json([item.model_dump(mode="json") for item in repositories]),
                    _canonical_json([item.model_dump(mode="json") for item in provider_checks]),
                    normalized,
                    _canonical_json(action.model_dump(mode="json")),
                    now,
                    now,
                    current.request_id,
                ),
            )

    def detach_project_transfers_for_restore(
        self,
        connection: sqlite3.Connection,
        *,
        diagnostic: str,
        now: str,
    ) -> None:
        """Freeze unfinished target transfers at their last committed boundary."""

        if not connection.in_transaction:
            raise RuntimeError("restore transfer detachment requires one active transaction")
        normalized = _MESSAGE_TEXT_ADAPTER.validate_python(diagnostic)
        rows = connection.execute(
            """
            SELECT * FROM project_transfer_requests
            WHERE side = 'target' AND phase != 'completed'
            ORDER BY request_id
            """
        ).fetchall()
        for row in rows:
            current = self._project_transfer_request_from_connection(
                connection,
                row["request_id"],
            )
            if current.phase == "operator_action_needed":
                if current.restore_diagnostic == normalized:
                    self._invalidate_project_transfer_upload(
                        connection,
                        current,
                        now=now,
                    )
                    continue
                resume_phase = current.restore_resume_phase
            else:
                resume_phase = current.phase
            if resume_phase is None:
                raise RuntimeError("restored target transfer lost its committed phase")
            self._invalidate_project_transfer_upload(
                connection,
                current,
                now=now,
            )
            updated = ProjectTransferRequestRecord.model_validate_json(
                _canonical_json(
                    {
                        **current.model_dump(mode="json"),
                        "phase": "operator_action_needed",
                        "restore_resume_phase": resume_phase,
                        "restore_diagnostic": normalized,
                        "revision": current.revision + 1,
                        "updated_at": now,
                    }
                )
            )
            self._replace_project_transfer_record(connection, current, updated)

    def create_project_provisioning_request(
        self,
        *,
        kind: ProjectProvisioningKind,
        authorized_by: AuthorizedHuman,
        machines: list[ProjectProvisioningMachineIntent],
        repositories: list[ProjectProvisioningRepositoryIntent],
        provider_checks: list[ProjectProvisioningProviderIntent],
        source_project_id: str | None = None,
        name: str | None = None,
        state_repository: str | None = None,
        project_truth_scope: list[str] | None = None,
        default_run_truth_scope: list[str] | None = None,
        default_auto_research_invocation_ceiling: int = (DEFAULT_AUTO_RESEARCH_INVOCATION_CEILING),
        request_id: str | None = None,
    ) -> ProjectProvisioningRequestRecord:
        """Reserve one project namespace without creating project authority."""

        if kind not in {"create_team_project", "incoming_transfer"}:
            raise ValueError("project provisioning kind is invalid")
        authorizer = AuthorizedHuman.model_validate(authorized_by.model_dump(mode="json"))
        machine_intents = [
            ProjectProvisioningMachineIntent.model_validate(machine.model_dump(mode="json"))
            for machine in machines
        ]
        repository_intents = [
            ProjectProvisioningRepositoryIntent.model_validate(repository.model_dump(mode="json"))
            for repository in repositories
        ]
        provider_intents = [
            ProjectProvisioningProviderIntent.model_validate(check.model_dump(mode="json"))
            for check in provider_checks
        ]
        if kind == "create_team_project":
            if source_project_id is not None:
                raise ValueError("new team-project provisioning cannot name a source project id")
            proposed_project_id = str(uuid.uuid4())
        else:
            if source_project_id is None:
                raise ValueError("incoming transfer provisioning requires the source project id")
            try:
                proposed_project_id = _canonical_uuid4(
                    source_project_id,
                    label="incoming transfer project identity",
                )
            except RuntimeError as exc:
                raise ValueError(str(exc)) from exc
        target_space_id = self.space_id
        if request_id is None:
            canonical_request_id = str(uuid.uuid4())
        else:
            try:
                canonical_request_id = _canonical_uuid4(
                    request_id,
                    label="provisioning request identity",
                )
            except RuntimeError as exc:
                raise ValueError(str(exc)) from exc
        machine_records = [
            ProjectProvisioningMachineRecord(
                **machine.model_dump(mode="json"),
                resolved_central_root=None,
            )
            for machine in machine_intents
        ]
        machine_map = {machine.alias: machine for machine in machine_records}
        repository_records: list[ProjectProvisioningRepositoryRecord] = []
        for repository in repository_intents:
            machine = machine_map.get(repository.machine_alias)
            if machine is None:
                raise ValueError("provisioning repository names an unknown machine")
            intended_path = (
                None
                if machine.central_root is None
                else str(
                    PurePosixPath(machine.central_root)
                    / proposed_project_id
                    / "repositories"
                    / repository.alias
                )
            )
            repository_records.append(
                ProjectProvisioningRepositoryRecord(
                    **repository.model_dump(mode="json"),
                    intended_path=intended_path,
                )
            )
        provider_records = [
            ProjectProvisioningProviderCheckRecord(**check.model_dump(mode="json"))
            for check in provider_intents
        ]
        now = self.now()
        record = ProjectProvisioningRequestRecord(
            request_id=canonical_request_id,
            kind=kind,
            status="waiting_for_server_setup",
            target_space_id=target_space_id,
            authorized_by=authorizer,
            proposed_project_id=proposed_project_id,
            name=name,
            state_repository=state_repository,
            project_truth_scope=list(project_truth_scope or []),
            default_run_truth_scope=list(default_run_truth_scope or []),
            default_auto_research_invocation_ceiling=(default_auto_research_invocation_ceiling),
            machines=machine_records,
            repositories=repository_records,
            provider_checks=provider_records,
            revision=0,
            created_at=now,
            updated_at=now,
        )
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            identity = connection.execute(
                "SELECT space_id, space_kind FROM space_identity WHERE singleton = 1"
            ).fetchone()
            if (
                identity is None
                or identity["space_id"] != target_space_id
                or identity["space_kind"] != "team"
            ):
                raise ValueError("project provisioning requires this exact team space")
            if authorizer.space_id != target_space_id:
                raise ValueError("project provisioning authorizer belongs to another space")
            if (
                connection.execute(
                    "SELECT 1 FROM space_users WHERE user_id = ?",
                    (authorizer.user_id,),
                ).fetchone()
                is None
            ):
                raise ValueError("project provisioning authorizer is not a current space member")
            existing = connection.execute(
                "SELECT * FROM project_provisioning_requests WHERE request_id = ?",
                (canonical_request_id,),
            ).fetchone()
            if existing is not None:
                current = self._project_provisioning_record(existing)
                if self._project_provisioning_creation_payload(current) != (
                    self._project_provisioning_creation_payload(record)
                ):
                    raise ValueError(
                        "provisioning request identity already names another project intent"
                    )
                return current
            if (
                connection.execute(
                    "SELECT 1 FROM projects WHERE project_id = ?",
                    (proposed_project_id,),
                ).fetchone()
                is not None
            ):
                raise ValueError("the proposed project identity already exists in this space")
            try:
                self._insert_project_provisioning_request(connection, record)
            except sqlite3.IntegrityError as exc:
                raise ValueError("the proposed project identity is already reserved") from exc
        return record

    @staticmethod
    def _project_provisioning_creation_payload(
        record: ProjectProvisioningRequestRecord,
    ) -> dict[str, object]:
        return {
            "request_id": record.request_id,
            "kind": record.kind,
            "target_space_id": record.target_space_id,
            "authorized_by": record.authorized_by.model_dump(mode="json"),
            "proposed_project_id": record.proposed_project_id,
            "name": record.name,
            "state_repository": record.state_repository,
            "project_truth_scope": record.project_truth_scope,
            "default_run_truth_scope": record.default_run_truth_scope,
            "default_auto_research_invocation_ceiling": (
                record.default_auto_research_invocation_ceiling
            ),
            "machines": [
                machine.model_dump(mode="json", exclude={"resolved_central_root"})
                for machine in record.machines
            ],
            "repositories": [
                repository.model_dump(
                    mode="json",
                    exclude={
                        "resolved_path",
                        "checkout_disposition",
                        "git_check",
                    },
                )
                for repository in record.repositories
            ],
            "provider_checks": [
                check.model_dump(
                    mode="json",
                    include={
                        "profile",
                        "provider",
                        "runtime_id",
                        "model",
                        "reasoning",
                        "machine_alias",
                    },
                )
                for check in record.provider_checks
            ],
        }

    def project_provisioning_request(
        self,
        request_id: str,
    ) -> ProjectProvisioningRequestRecord | None:
        try:
            canonical_request_id = _canonical_uuid4(
                request_id,
                label="provisioning request identity",
            )
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM project_provisioning_requests WHERE request_id = ?",
                (canonical_request_id,),
            ).fetchone()
        return self._project_provisioning_record(row) if row is not None else None

    def project_provisioning_requests(
        self,
        *,
        status: ProjectProvisioningStatus | None = None,
    ) -> list[ProjectProvisioningRequestRecord]:
        if status is not None and status not in _PROVISIONING_TRANSITIONS:
            raise ValueError("project provisioning status is invalid")
        with self.connection() as connection:
            if status is None:
                rows = connection.execute(
                    """
                    SELECT * FROM project_provisioning_requests
                    ORDER BY created_at DESC, request_id
                    """
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM project_provisioning_requests
                    WHERE status = ?
                    ORDER BY created_at DESC, request_id
                    """,
                    (status,),
                ).fetchall()
        return [self._project_provisioning_record(row) for row in rows]

    def completed_project_provisioning_requests(
        self,
        project_id: str,
    ) -> list[ProjectProvisioningRequestRecord]:
        """Return completed reconstruction proofs for one exact project identity."""

        try:
            canonical_project_id = _canonical_uuid4(
                project_id,
                label="project identity",
            )
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM project_provisioning_requests
                WHERE proposed_project_id = ? AND status = 'completed'
                ORDER BY created_at, request_id
                """,
                (canonical_project_id,),
            ).fetchall()
        return [self._project_provisioning_record(row) for row in rows]

    def transition_project_provisioning_request(
        self,
        request_id: str,
        *,
        receipt_id: str,
        phase: str,
        expected_revision: int,
        expected_status: ProjectProvisioningStatus,
        to_status: ProjectProvisioningStatus,
        machines: list[ProjectProvisioningMachineRecord],
        repositories: list[ProjectProvisioningRepositoryRecord],
        provider_checks: list[ProjectProvisioningProviderCheckRecord],
        retryable_diagnostic: str | None = None,
        operator_action: ServerStep | None = None,
        cancellation_disposition: ProjectProvisioningCancellationDisposition | None = None,
    ) -> ProjectProvisioningRequestRecord:
        """Commit one named machine step or return its already-committed result."""

        try:
            canonical_request_id = _canonical_uuid4(
                request_id,
                label="provisioning request identity",
            )
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc
        if expected_revision < 0:
            raise ValueError("project provisioning expected revision cannot be negative")
        if (
            expected_status not in _PROVISIONING_TRANSITIONS
            or to_status not in _PROVISIONING_TRANSITIONS
        ):
            raise ValueError("project provisioning transition status is invalid")
        machine_records = [
            ProjectProvisioningMachineRecord.model_validate(machine.model_dump(mode="json"))
            for machine in machines
        ]
        repository_records = [
            ProjectProvisioningRepositoryRecord.model_validate(repository.model_dump(mode="json"))
            for repository in repositories
        ]
        provider_records = [
            ProjectProvisioningProviderCheckRecord.model_validate(check.model_dump(mode="json"))
            for check in provider_checks
        ]
        normalized_action = (
            None
            if operator_action is None
            else ServerStep.model_validate_json(operator_action.model_dump_json())
        )
        normalized_diagnostic = (
            None
            if retryable_diagnostic is None
            else _MESSAGE_TEXT_ADAPTER.validate_python(retryable_diagnostic)
        )
        transition_payload = {
            "request_id": canonical_request_id,
            "receipt_id": receipt_id,
            "phase": phase,
            "expected_revision": expected_revision,
            "expected_status": expected_status,
            "to_status": to_status,
            "machines": [machine.model_dump(mode="json") for machine in machine_records],
            "repositories": [
                repository.model_dump(mode="json") for repository in repository_records
            ],
            "provider_checks": [check.model_dump(mode="json") for check in provider_records],
            "retryable_diagnostic": normalized_diagnostic,
            "operator_action": (
                None if normalized_action is None else normalized_action.model_dump(mode="json")
            ),
            "cancellation_disposition": cancellation_disposition,
        }
        transition_sha256 = hashlib.sha256(
            _canonical_json(transition_payload).encode("utf-8")
        ).hexdigest()
        now = self.now()
        receipt_shape = ProjectProvisioningStepReceiptRecord(
            request_id=canonical_request_id,
            receipt_id=receipt_id,
            phase=phase,
            from_status=expected_status,
            to_status=to_status,
            transition_sha256=transition_sha256,
            resulting_revision=expected_revision + 1,
            created_at=now,
        )
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            prior_receipt_row = connection.execute(
                """
                SELECT * FROM project_provisioning_step_receipts
                WHERE request_id = ? AND receipt_id = ?
                """,
                (canonical_request_id, receipt_id),
            ).fetchone()
            if prior_receipt_row is not None:
                prior_receipt = self._project_provisioning_receipt(prior_receipt_row)
                if prior_receipt.transition_sha256 != transition_sha256:
                    raise ValueError("project provisioning receipt already names another step")
                current_row = connection.execute(
                    "SELECT * FROM project_provisioning_requests WHERE request_id = ?",
                    (canonical_request_id,),
                ).fetchone()
                if current_row is None:
                    raise RuntimeError("project provisioning receipt lost its request")
                return self._project_provisioning_record(current_row)
            current_row = connection.execute(
                "SELECT * FROM project_provisioning_requests WHERE request_id = ?",
                (canonical_request_id,),
            ).fetchone()
            if current_row is None:
                raise KeyError(canonical_request_id)
            current = self._project_provisioning_record(current_row)
            if current.revision != expected_revision or current.status != expected_status:
                raise ValueError("project provisioning request changed; reload it before retrying")
            if to_status not in _PROVISIONING_TRANSITIONS[current.status]:
                raise ValueError(
                    f"project provisioning cannot move from {current.status} to {to_status}"
                )
            setup_started_at = current.setup_started_at
            if (
                to_status
                in {
                    "setup_in_progress",
                    "operator_action_needed",
                    "ready_for_review",
                    "completed",
                }
                and setup_started_at is None
            ):
                setup_started_at = now
            ready_at = now if to_status == "ready_for_review" else None
            if to_status == "completed":
                ready_at = current.ready_at
                if (
                    machine_records != current.machines
                    or repository_records != current.repositories
                    or provider_records != current.provider_checks
                ):
                    raise ValueError("project completion cannot change the reviewed machine state")
            values = current.model_dump(mode="json")
            values.update(
                {
                    "status": to_status,
                    "machines": [machine.model_dump(mode="json") for machine in machine_records],
                    "repositories": [
                        repository.model_dump(mode="json") for repository in repository_records
                    ],
                    "provider_checks": [
                        check.model_dump(mode="json") for check in provider_records
                    ],
                    "retryable_diagnostic": normalized_diagnostic,
                    "operator_action": (
                        None
                        if normalized_action is None
                        else normalized_action.model_dump(mode="json")
                    ),
                    "final_review_digest": None,
                    "cancellation_disposition": cancellation_disposition,
                    "revision": expected_revision + 1,
                    "updated_at": now,
                    "setup_started_at": setup_started_at,
                    "ready_at": ready_at,
                    "completed_at": now if to_status == "completed" else None,
                    "cancelled_at": now if to_status == "cancelled" else None,
                }
            )
            if to_status in {"ready_for_review", "completed"}:
                values["final_review_digest"] = "0" * 64
                draft = ProjectProvisioningRequestRecord.model_validate_json(
                    _canonical_json(values)
                )
                values["final_review_digest"] = project_provisioning_review_digest(draft)
                if (
                    to_status == "completed"
                    and values["final_review_digest"] != current.final_review_digest
                ):
                    raise ValueError("project provisioning final review changed before completion")
            updated = _verify_project_provisioning_review_digest(
                ProjectProvisioningRequestRecord.model_validate_json(_canonical_json(values))
            )
            changed = connection.execute(
                """
                UPDATE project_provisioning_requests
                SET status = ?, machines_json = ?, repositories_json = ?,
                    provider_checks_json = ?, retryable_diagnostic = ?,
                    operator_action_json = ?, final_review_digest = ?,
                    cancellation_disposition = ?, revision = ?, updated_at = ?,
                    setup_started_at = ?, ready_at = ?, completed_at = ?, cancelled_at = ?
                WHERE request_id = ? AND revision = ? AND status = ?
                """,
                (
                    updated.status,
                    _canonical_json([item.model_dump(mode="json") for item in updated.machines]),
                    _canonical_json(
                        [item.model_dump(mode="json") for item in updated.repositories]
                    ),
                    _canonical_json(
                        [item.model_dump(mode="json") for item in updated.provider_checks]
                    ),
                    updated.retryable_diagnostic,
                    (
                        None
                        if updated.operator_action is None
                        else _canonical_json(updated.operator_action.model_dump(mode="json"))
                    ),
                    updated.final_review_digest,
                    updated.cancellation_disposition,
                    updated.revision,
                    updated.updated_at,
                    updated.setup_started_at,
                    updated.ready_at,
                    updated.completed_at,
                    updated.cancelled_at,
                    canonical_request_id,
                    expected_revision,
                    expected_status,
                ),
            ).rowcount
            if changed != 1:
                raise RuntimeError("project provisioning transition lost its transaction guard")
            connection.execute(
                """
                INSERT INTO project_provisioning_step_receipts (
                    request_id, receipt_id, phase, from_status, to_status,
                    transition_sha256, resulting_revision, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt_shape.request_id,
                    receipt_shape.receipt_id,
                    receipt_shape.phase,
                    receipt_shape.from_status,
                    receipt_shape.to_status,
                    receipt_shape.transition_sha256,
                    receipt_shape.resulting_revision,
                    receipt_shape.created_at,
                ),
            )
        return updated

    def project_provisioning_step_receipts(
        self,
        request_id: str,
    ) -> list[ProjectProvisioningStepReceiptRecord]:
        try:
            canonical_request_id = _canonical_uuid4(
                request_id,
                label="provisioning request identity",
            )
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM project_provisioning_step_receipts
                WHERE request_id = ?
                ORDER BY resulting_revision, receipt_id
                """,
                (canonical_request_id,),
            ).fetchall()
        return [self._project_provisioning_receipt(row) for row in rows]

    @staticmethod
    def _insert_project_provisioning_request(
        connection: sqlite3.Connection,
        record: ProjectProvisioningRequestRecord,
    ) -> None:
        connection.execute(
            """
            INSERT INTO project_provisioning_requests (
                request_id, kind, status, target_space_id, authorized_by_json,
                proposed_project_id, project_config_json, machines_json, repositories_json,
                provider_checks_json, retryable_diagnostic, operator_action_json,
                final_review_digest, cancellation_disposition, revision,
                created_at, updated_at, setup_started_at, ready_at,
                completed_at, cancelled_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.request_id,
                record.kind,
                record.status,
                record.target_space_id,
                _canonical_json(record.authorized_by.model_dump(mode="json")),
                record.proposed_project_id,
                (
                    _canonical_json(
                        {
                            "name": record.name,
                            "state_repository": record.state_repository,
                            "project_truth_scope": record.project_truth_scope,
                            "default_run_truth_scope": record.default_run_truth_scope,
                            "default_auto_research_invocation_ceiling": (
                                record.default_auto_research_invocation_ceiling
                            ),
                        }
                    )
                    if record.configuration_complete
                    else None
                ),
                _canonical_json([item.model_dump(mode="json") for item in record.machines]),
                _canonical_json([item.model_dump(mode="json") for item in record.repositories]),
                _canonical_json([item.model_dump(mode="json") for item in record.provider_checks]),
                record.retryable_diagnostic,
                None,
                record.final_review_digest,
                record.cancellation_disposition,
                record.revision,
                record.created_at,
                record.updated_at,
                record.setup_started_at,
                record.ready_at,
                record.completed_at,
                record.cancelled_at,
            ),
        )

    @staticmethod
    def _project_provisioning_record(row: sqlite3.Row) -> ProjectProvisioningRequestRecord:
        try:
            project_config_json = row["project_config_json"]
            project_config = {} if project_config_json is None else json.loads(project_config_json)
            if not isinstance(project_config, dict) or (
                project_config_json is not None and set(project_config) != _PROJECT_CONFIG_FIELDS
            ):
                raise ValueError("stored project provisioning configuration is invalid")
            return _verify_project_provisioning_review_digest(
                ProjectProvisioningRequestRecord.model_validate_json(
                    _canonical_json(
                        {
                            "request_id": row["request_id"],
                            "kind": row["kind"],
                            "status": row["status"],
                            "target_space_id": row["target_space_id"],
                            "authorized_by": json.loads(row["authorized_by_json"]),
                            "proposed_project_id": row["proposed_project_id"],
                            **project_config,
                            "machines": json.loads(row["machines_json"]),
                            "repositories": json.loads(row["repositories_json"]),
                            "provider_checks": json.loads(row["provider_checks_json"]),
                            "retryable_diagnostic": row["retryable_diagnostic"],
                            "operator_action": (
                                None
                                if row["operator_action_json"] is None
                                else json.loads(row["operator_action_json"])
                            ),
                            "final_review_digest": row["final_review_digest"],
                            "cancellation_disposition": row["cancellation_disposition"],
                            "revision": row["revision"],
                            "created_at": row["created_at"],
                            "updated_at": row["updated_at"],
                            "setup_started_at": row["setup_started_at"],
                            "ready_at": row["ready_at"],
                            "completed_at": row["completed_at"],
                            "cancelled_at": row["cancelled_at"],
                        }
                    )
                )
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise RuntimeError("stored project provisioning request is invalid") from exc

    @staticmethod
    def _project_provisioning_receipt(row: sqlite3.Row) -> ProjectProvisioningStepReceiptRecord:
        try:
            return ProjectProvisioningStepReceiptRecord.model_validate(dict(row))
        except ValueError as exc:
            raise RuntimeError("stored project provisioning receipt is invalid") from exc


__all__ = [
    "ProjectProvisioningStoreMixin",
    "project_provisioning_review_digest",
    "project_transfer_receipt_sha256",
    "project_transfer_source_configuration_sha256",
]
