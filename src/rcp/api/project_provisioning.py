"""Member-authorized product requests for team-project preparation."""

from __future__ import annotations

import json
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rcp.api.dependencies import (
    get_catalog,
    get_experiment_operation_lock,
    get_identity_access,
    get_setup,
    get_store,
)
from rcp.api.identity import IdentityAccess
from rcp.config import (
    AGENT_EXECUTION_PROFILES,
    DEFAULT_AUTO_RESEARCH_INVOCATION_CEILING,
    GRAPH_AGENT_EXECUTION_PROFILES,
    AgentExecutionProfile,
)
from rcp.core.models import AuthorizedHuman
from rcp.core.transition_models import GraphHeadRef
from rcp.history import ProjectIdentityConflict
from rcp.keyed_locks import KeyedLocks
from rcp.project_transfer import capture_project_transfer_source
from rcp.projects import ProjectCatalog
from rcp.providers import ProviderId
from rcp.server_ops.github import GitHubRepositoryRef, parse_github_repository_ref
from rcp.server_ops.layout import DEFAULT_SERVER_LAYOUT
from rcp.server_ops.models import ServerStep
from rcp.setup import ProjectSetupManager
from rcp.storage import (
    AppStore,
    ProjectProvisioningCancellationDisposition,
    ProjectProvisioningCheckStatus,
    ProjectProvisioningMachineIntent,
    ProjectProvisioningMachineRecord,
    ProjectProvisioningProviderCheckRecord,
    ProjectProvisioningProviderIntent,
    ProjectProvisioningRepositoryIntent,
    ProjectProvisioningRepositoryRecord,
    ProjectProvisioningRequestRecord,
    ProjectProvisioningStatus,
    ProjectTransferCleanupAcknowledgment,
    ProjectTransferLinkReceipt,
    ProjectTransferPhase,
    ProjectTransferRequestRecord,
    ProjectTransferSourceConfiguration,
    ProjectTransferSourceReleaseReceipt,
    ProjectTransferTargetAdmissionReceipt,
    SpaceKind,
    TeamAuthenticationError,
)
from rcp.storage.provisioning import project_transfer_source_configuration_sha256
from rcp.transfer.source import (
    advance_source_project_transfer,
    complete_source_project_transfer,
    read_transfer_archive,
    source_transfer_export_path,
    stream_source_transfer_archive,
)
from rcp.transport import StateUnavailable

router = APIRouter()

IdentityDependency = Annotated[IdentityAccess, Depends(get_identity_access)]
StoreDependency = Annotated[AppStore, Depends(get_store)]
SetupDependency = Annotated[ProjectSetupManager, Depends(get_setup)]
CatalogDependency = Annotated[ProjectCatalog, Depends(get_catalog)]
OperationLockDependency = Annotated[KeyedLocks, Depends(get_experiment_operation_lock)]

ProjectCreationIntent = Literal[
    "use_existing_checkout_personally",
    "create_shared_team_project",
    "move_personal_project_to_team",
]

_STATUS_LABELS: dict[ProjectProvisioningStatus, str] = {
    "waiting_for_server_setup": "Waiting for server setup",
    "setup_in_progress": "Setup in progress",
    "operator_action_needed": "Operator action needed",
    "ready_for_review": "Ready for review",
    "completed": "Completed",
    "cancelled": "Cancelled",
}
_CHECK_LABELS: dict[ProjectProvisioningCheckStatus, str] = {
    "pending": "Waiting for setup",
    "checking": "Checking",
    "operator_action_needed": "Operator action needed",
    "ready": "Ready",
}
_TRANSFER_PHASE_LABELS: dict[ProjectTransferPhase, str] = {
    "awaiting_link": "Awaiting target link",
    "linked": "Target setup in progress",
    "target_admitted": "Target admitted; source release pending",
    "source_released": "Source released; archive relay pending",
    "source_fenced": "Source fenced; archive binding pending",
    "archive_bound": "Archive bound; target activation pending",
    "target_activated": "Target activated; cleanup pending",
    "cleanup_acknowledged": "Cleanup acknowledged",
    "completed": "Completed",
    "operator_action_needed": "Operator action needed",
}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ProjectCreationIntentControl(_StrictModel):
    intent: ProjectCreationIntent
    eligible: bool
    preselected: bool
    primary_action_label: str
    required_fields: tuple[str, ...]
    pinned_source_project_id: str | None = None
    unavailable_reason: str | None = None


class ProjectCreationControl(_StrictModel):
    intents: tuple[ProjectCreationIntentControl, ...]
    requires_authenticated_member: bool


class ProjectProvisioningMachineRequest(_StrictModel):
    alias: str
    location: Literal["local", "ssh"]
    host: str = ""
    os_account: str
    central_root: str | None = None

    def intent(self) -> ProjectProvisioningMachineIntent:
        central_root = self.central_root
        if self.location == "local" and central_root is None:
            central_root = str(DEFAULT_SERVER_LAYOUT.projects_root)
        return ProjectProvisioningMachineIntent(
            alias=self.alias,
            location=self.location,
            host=self.host,
            os_account=self.os_account,
            central_root=central_root,
        )


class ProjectProvisioningRepositoryRequest(_StrictModel):
    alias: str
    source: str
    machine_alias: str

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        parse_github_repository_ref(value)
        return value

    def intent(self) -> ProjectProvisioningRepositoryIntent:
        return ProjectProvisioningRepositoryIntent(
            alias=self.alias,
            repository=parse_github_repository_ref(self.source),
            machine_alias=self.machine_alias,
        )


class ProjectProvisioningCreateRequest(_StrictModel):
    name: str = Field(min_length=1, max_length=120)
    state_repository: str
    project_truth_scope: list[str] = Field(min_length=1, max_length=64)
    default_run_truth_scope: list[str] = Field(min_length=1, max_length=64)
    default_auto_research_invocation_ceiling: int = Field(
        default=DEFAULT_AUTO_RESEARCH_INVOCATION_CEILING,
        ge=1,
    )
    machines: list[ProjectProvisioningMachineRequest] = Field(min_length=1, max_length=32)
    repositories: list[ProjectProvisioningRepositoryRequest] = Field(
        min_length=1,
        max_length=64,
    )
    provider_checks: list[ProjectProvisioningProviderIntent] = Field(
        min_length=1,
        max_length=32,
    )

    @model_validator(mode="after")
    def require_every_execution_profile(self) -> ProjectProvisioningCreateRequest:
        profiles = [check.profile for check in self.provider_checks]
        if len(profiles) != len(set(profiles)):
            raise ValueError("project provisioning provider profiles must be unique")
        missing = set(AGENT_EXECUTION_PROFILES) - set(profiles)
        if missing:
            raise ValueError(
                "project provisioning must configure every agent execution profile; "
                f"missing={sorted(missing)}"
            )
        repository_machines = {
            repository.alias: repository.machine_alias for repository in self.repositories
        }
        canonical_machine = repository_machines.get(self.state_repository)
        if canonical_machine is not None and any(
            check.profile in GRAPH_AGENT_EXECUTION_PROFILES
            and check.machine_alias != canonical_machine
            for check in self.provider_checks
        ):
            raise ValueError(
                "graph-writing provider profiles must run on the canonical state machine"
            )
        return self


class ProjectProvisioningCompleteRequest(_StrictModel):
    final_review_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class ProjectTransferRepositorySourceRequest(_StrictModel):
    alias: str
    repository: GitHubRepositoryRef
    machine_alias: str


class ProjectTransferSourceConfigurationRequest(_StrictModel):
    source_rcp_version: str
    source_schema_generation: int = Field(ge=1)
    supported_archive_codecs: list[str] = Field(min_length=1, max_length=16)
    machine_aliases: list[str] = Field(min_length=1, max_length=32)
    repositories: list[ProjectTransferRepositorySourceRequest] = Field(
        min_length=1,
        max_length=64,
    )
    state_repository: str
    project_truth_scope: list[str] = Field(min_length=1, max_length=64)
    default_run_truth_scope: list[str] = Field(min_length=1, max_length=64)
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def storage_model(self) -> ProjectTransferSourceConfiguration:
        return ProjectTransferSourceConfiguration.model_validate_json(self.model_dump_json())


class ProjectTransferSourceCreateRequest(_StrictModel):
    request_id: str
    project_id: str
    target_space_id: str
    expected_source_configuration_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )


class ProjectTransferIncomingProvisioningCreateRequest(ProjectProvisioningCreateRequest):
    request_id: str
    source_project_id: str


class ProjectTransferTargetCreateRequest(_StrictModel):
    provisioning_request_id: str
    source_request_id: str
    source_project_id: str
    source_space_id: str
    source_configuration: ProjectTransferSourceConfigurationRequest
    source_configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_release_proof_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    accepted_schema_generation: int = Field(ge=1)
    accepted_archive_codec: str


class ProjectTransferLinkRequest(_StrictModel):
    receipt: dict[str, object]

    def storage_model(self) -> ProjectTransferLinkReceipt:
        return ProjectTransferLinkReceipt.model_validate_json(json.dumps(self.receipt))


class ProjectTransferTargetAdmissionRequest(_StrictModel):
    receipt: dict[str, object]

    def storage_model(self) -> ProjectTransferTargetAdmissionReceipt:
        return ProjectTransferTargetAdmissionReceipt.model_validate_json(json.dumps(self.receipt))


class ProjectTransferSourceReleaseRequest(_StrictModel):
    expected_source_configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_source_head: GraphHeadRef


class ProjectTransferRestoreReentryRequest(_StrictModel):
    expected_restored_revision: int = Field(ge=0)
    expected_resume_phase: Literal["archive_bound"]
    expected_final_review_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class ProjectTransferSourceReceiptRequest(_StrictModel):
    receipt: dict[str, object]

    def storage_model(self) -> ProjectTransferSourceReleaseReceipt:
        return ProjectTransferSourceReleaseReceipt.model_validate_json(json.dumps(self.receipt))


class ProjectTransferArchiveRequest(_StrictModel):
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    archive_size_bytes: int = Field(ge=1)
    source_fence_head: GraphHeadRef | None = None


class ProjectTransferCleanupAcknowledgmentRequest(_StrictModel):
    acknowledgment: dict[str, object]

    def storage_model(self) -> ProjectTransferCleanupAcknowledgment:
        return ProjectTransferCleanupAcknowledgment.model_validate_json(
            json.dumps(self.acknowledgment)
        )


class ProjectProvisioningMachineProjection(_StrictModel):
    alias: str
    location: Literal["local", "ssh"]
    host: str
    os_account: str
    intended_central_root: str | None
    resolved_central_root: str | None
    ready: bool
    status_label: str


class ProjectProvisioningRepositoryProjection(_StrictModel):
    alias: str
    repository: GitHubRepositoryRef
    https_clone_url: str
    ssh_clone_url: str
    settings_url: str
    machine_alias: str
    intended_path: str | None
    resolved_path: str | None
    checkout_disposition: Literal["request_created", "reused_existing"] | None
    status: ProjectProvisioningCheckStatus
    status_label: str
    ready: bool
    commit: str | None
    write_verified: bool
    deploy_key_label: str | None
    public_key_fingerprint: str | None
    checked_at: str | None
    diagnostic: str | None


class ProjectProvisioningProviderProjection(_StrictModel):
    profile: AgentExecutionProfile
    provider: ProviderId
    runtime_id: str
    model: str
    reasoning: str
    machine_alias: str
    status: ProjectProvisioningCheckStatus
    status_label: str
    ready: bool
    binary_path: str | None
    version: str | None
    resolved_runtime_id: str | None
    execution_account: str | None
    checked_at: str | None
    diagnostic: str | None


class ProjectProvisioningReadinessProjection(_StrictModel):
    machines_ready: int
    machines_total: int
    repositories_ready: int
    repositories_total: int
    providers_ready: int
    providers_total: int
    all_ready: bool


class ProjectProvisioningFinalReview(_StrictModel):
    digest: str
    proposed_project_id: str
    authorized_by: AuthorizedHuman
    ready_at: str


class ProjectProvisioningResponse(_StrictModel):
    request_id: str
    kind: Literal["create_team_project", "incoming_transfer"]
    status: ProjectProvisioningStatus
    status_label: str
    next_action: str | None
    can_run_setup: bool
    can_review: bool
    can_cancel: bool
    target_space_id: str
    proposed_project_id: str
    name: str | None
    state_repository: str | None
    project_truth_scope: list[str]
    default_run_truth_scope: list[str]
    default_auto_research_invocation_ceiling: int
    authorized_by: AuthorizedHuman
    machines: list[ProjectProvisioningMachineProjection]
    repositories: list[ProjectProvisioningRepositoryProjection]
    provider_checks: list[ProjectProvisioningProviderProjection]
    readiness: ProjectProvisioningReadinessProjection
    diagnostic: str | None
    operator_action: ServerStep | None
    operator_argv: tuple[str, ...]
    final_review: ProjectProvisioningFinalReview | None
    cancellation_disposition: ProjectProvisioningCancellationDisposition | None
    revision: int
    created_at: str
    updated_at: str
    setup_started_at: str | None
    completed_at: str | None
    cancelled_at: str | None


class ProjectTransferResponse(ProjectTransferRequestRecord):
    """Safe transfer read model with backend-owned lifecycle decisions.

    The durable record contains only commitments and public receipts; it has no
    raw proof or upload lease secret.  Keep mutations on the existing record
    responses while reads use this projection so the Web client does not have
    to interpret transfer phases itself.
    """

    phase_label: str
    next_action: str | None
    can_link: bool
    can_run_setup: bool
    can_review: bool
    can_admit: bool
    can_accept_admission: bool
    can_release: bool
    can_accept_release: bool
    can_relay: bool
    can_restore_reentry: bool
    can_complete: bool
    finished: bool


class ProjectTransferSourceBoundaryResponse(_StrictModel):
    """The exact read-only source boundary a release request must echo."""

    source_configuration: ProjectTransferSourceConfiguration
    source_configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_head: GraphHeadRef


def project_creation_control(space_kind: SpaceKind) -> ProjectCreationControl:
    personal = space_kind == "personal"
    return ProjectCreationControl(
        requires_authenticated_member=not personal,
        intents=(
            ProjectCreationIntentControl(
                intent="use_existing_checkout_personally",
                eligible=personal,
                preselected=personal,
                primary_action_label="Use existing checkout",
                required_fields=(
                    "name",
                    "repositories",
                    "state_repository",
                    "execution",
                    "confirmed",
                ),
                unavailable_reason=(
                    None if personal else "Existing-checkout setup belongs to a personal space."
                ),
            ),
            ProjectCreationIntentControl(
                intent="create_shared_team_project",
                eligible=not personal,
                preselected=not personal,
                primary_action_label="Create shared team project",
                required_fields=("machines", "repositories", "provider_checks"),
                unavailable_reason=(
                    "Connect to a team space to create a shared project." if personal else None
                ),
            ),
            ProjectCreationIntentControl(
                intent="move_personal_project_to_team",
                eligible=personal,
                preselected=False,
                primary_action_label="Move to team space",
                required_fields=("source_project", "team_connection"),
                unavailable_reason=(
                    None if personal else "Move-to-team setup begins in a personal space."
                ),
            ),
        ),
    )


@router.post(
    "/api/project-transfers/incoming-provisioning-requests",
    response_model=ProjectProvisioningResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_incoming_transfer_provisioning_request(
    body: ProjectTransferIncomingProvisioningCreateRequest,
    request: Request,
    *,
    identity_access: IdentityDependency,
    store: StoreDependency,
) -> ProjectProvisioningResponse:
    identity_access.require_team_space()
    actor = identity_access.require_patch_capable_identity(request)
    try:
        record = store.create_project_provisioning_request(
            kind="incoming_transfer",
            authorized_by=actor,
            machines=[machine.intent() for machine in body.machines],
            repositories=[repository.intent() for repository in body.repositories],
            provider_checks=body.provider_checks,
            source_project_id=body.source_project_id,
            name=body.name,
            state_repository=body.state_repository,
            project_truth_scope=body.project_truth_scope,
            default_run_truth_scope=body.default_run_truth_scope,
            default_auto_research_invocation_ceiling=(
                body.default_auto_research_invocation_ceiling
            ),
            request_id=body.request_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _project_provisioning_response(record, viewer_user_id=actor.user_id)


@router.get(
    "/api/project-transfers/incoming-provisioning-requests",
    response_model=list[ProjectProvisioningResponse],
)
def incoming_transfer_provisioning_requests(
    request: Request,
    *,
    identity_access: IdentityDependency,
    store: StoreDependency,
) -> list[ProjectProvisioningResponse]:
    identity_access.require_team_space()
    viewer = identity_access.acting_user(request)
    return [
        _project_provisioning_response(record, viewer_user_id=viewer.user_id)
        for record in store.project_provisioning_requests()
        if record.kind == "incoming_transfer"
    ]


@router.get(
    "/api/project-transfers/incoming-provisioning-requests/{request_id}",
    response_model=ProjectProvisioningResponse,
)
def incoming_transfer_provisioning_request(
    request_id: str,
    request: Request,
    *,
    identity_access: IdentityDependency,
    store: StoreDependency,
) -> ProjectProvisioningResponse:
    identity_access.require_team_space()
    viewer = identity_access.acting_user(request)
    record = _request_or_404(store, request_id)
    if record.kind != "incoming_transfer":
        raise HTTPException(status_code=404, detail="Provisioning request not found")
    return _project_provisioning_response(record, viewer_user_id=viewer.user_id)


@router.post(
    "/api/project-transfers/source-requests",
    response_model=ProjectTransferRequestRecord,
    status_code=status.HTTP_201_CREATED,
)
def create_source_project_transfer_request(
    body: ProjectTransferSourceCreateRequest,
    request: Request,
    *,
    identity_access: IdentityDependency,
    store: StoreDependency,
    catalog: CatalogDependency,
) -> ProjectTransferRequestRecord:
    _require_transfer_space(store, "personal")
    actor = identity_access.require_patch_capable_identity(request)
    try:
        existing = store.project_transfer_request(body.request_id)
        if existing is None:
            service = catalog.open(body.project_id)
            configuration, _head = capture_project_transfer_source(service)
        else:
            configuration = existing.source_configuration
        actual_digest = project_transfer_source_configuration_sha256(configuration)
        if body.expected_source_configuration_sha256 not in {None, actual_digest}:
            raise ValueError("source configuration changed before transfer creation")
        return store.create_source_project_transfer_request(
            project_id=body.project_id,
            target_space_id=body.target_space_id,
            initiated_by=actor,
            source_configuration=configuration,
            request_id=body.request_id,
        )
    except (KeyError, OSError, StateUnavailable, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/api/project-transfers/target-requests",
    response_model=ProjectTransferRequestRecord,
    status_code=status.HTTP_201_CREATED,
)
def create_target_project_transfer_request(
    body: ProjectTransferTargetCreateRequest,
    request: Request,
    *,
    identity_access: IdentityDependency,
    store: StoreDependency,
) -> ProjectTransferRequestRecord:
    identity_access.require_team_space()
    actor = identity_access.require_patch_capable_identity(request)
    try:
        return store.create_target_project_transfer_request(
            provisioning_request_id=body.provisioning_request_id,
            source_request_id=body.source_request_id,
            source_project_id=body.source_project_id,
            source_space_id=body.source_space_id,
            initiated_by=actor,
            source_configuration=body.source_configuration.storage_model(),
            source_configuration_sha256=body.source_configuration_sha256,
            source_release_proof_sha256=body.source_release_proof_sha256,
            accepted_schema_generation=body.accepted_schema_generation,
            accepted_archive_codec=body.accepted_archive_codec,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get(
    "/api/project-transfers/requests",
    response_model=list[ProjectTransferResponse],
)
def project_transfer_requests(
    request: Request,
    *,
    identity_access: IdentityDependency,
    store: StoreDependency,
) -> list[ProjectTransferResponse]:
    viewer = identity_access.acting_user(request)
    return [
        _project_transfer_response(record, store=store, viewer_user_id=viewer.user_id)
        for record in store.project_transfer_requests()
    ]


@router.get(
    "/api/project-transfers/requests/{request_id}",
    response_model=ProjectTransferResponse,
)
def project_transfer_request(
    request_id: str,
    request: Request,
    *,
    identity_access: IdentityDependency,
    store: StoreDependency,
) -> ProjectTransferResponse:
    viewer = identity_access.acting_user(request)
    return _project_transfer_response(
        _transfer_request_or_404(store, request_id),
        store=store,
        viewer_user_id=viewer.user_id,
    )


@router.post(
    "/api/project-transfers/source-requests/{request_id}/link",
    response_model=ProjectTransferRequestRecord,
)
def link_source_project_transfer_request(
    request_id: str,
    body: ProjectTransferLinkRequest,
    request: Request,
    *,
    identity_access: IdentityDependency,
    store: StoreDependency,
) -> ProjectTransferRequestRecord:
    _require_transfer_space(store, "personal")
    identity_access.require_patch_capable_identity(request)
    try:
        return store.link_source_project_transfer_request(
            request_id,
            receipt=body.storage_model(),
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/api/project-transfers/target-requests/{request_id}/admit",
    response_model=ProjectTransferRequestRecord,
)
def admit_target_project_transfer_request(
    request_id: str,
    request: Request,
    *,
    identity_access: IdentityDependency,
    store: StoreDependency,
) -> ProjectTransferRequestRecord:
    identity_access.require_team_space()
    actor = identity_access.require_patch_capable_identity(request)
    try:
        return store.record_target_project_transfer_admission(
            request_id,
            admitted_by=actor,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/api/project-transfers/source-requests/{request_id}/target-admission",
    response_model=ProjectTransferRequestRecord,
)
def accept_target_project_transfer_admission(
    request_id: str,
    body: ProjectTransferTargetAdmissionRequest,
    request: Request,
    *,
    identity_access: IdentityDependency,
    store: StoreDependency,
) -> ProjectTransferRequestRecord:
    _require_transfer_space(store, "personal")
    identity_access.require_patch_capable_identity(request)
    try:
        return store.accept_target_project_transfer_admission(
            request_id,
            receipt=body.storage_model(),
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get(
    "/api/project-transfers/source-requests/{request_id}/release-boundary",
    response_model=ProjectTransferSourceBoundaryResponse,
)
def read_source_project_transfer_release_boundary(
    request_id: str,
    request: Request,
    *,
    identity_access: IdentityDependency,
    store: StoreDependency,
    catalog: CatalogDependency,
    operation_lock: OperationLockDependency,
) -> ProjectTransferSourceBoundaryResponse:
    """Capture the source boundary that the release mutation must echo."""

    _require_transfer_space(store, "personal")
    actor = identity_access.require_patch_capable_identity(request)
    try:
        transfer = _transfer_request_or_404(store, request_id)
        if transfer.side != "source":
            raise ValueError("source release boundary belongs only to a source transfer")
        if not store.is_project_member(transfer.project_id, actor.user_id):
            raise HTTPException(status_code=404, detail="Project not found")
        with operation_lock(transfer.project_id):
            current = _transfer_request_or_404(store, request_id)
            if current.side != "source":
                raise ValueError("source release boundary belongs only to a source transfer")
            if not store.is_project_member(current.project_id, actor.user_id):
                raise HTTPException(status_code=404, detail="Project not found")
            if current.phase != "target_admitted":
                raise ValueError("source transfer is not awaiting its release boundary")
            service = catalog.open(current.project_id)
            configuration, source_head = capture_project_transfer_source(service)
    except HTTPException:
        raise
    except (KeyError, OSError, StateUnavailable, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return ProjectTransferSourceBoundaryResponse(
        source_configuration=configuration,
        source_configuration_sha256=project_transfer_source_configuration_sha256(configuration),
        source_head=source_head,
    )


@router.post(
    "/api/project-transfers/source-requests/{request_id}/release",
    response_model=ProjectTransferRequestRecord,
)
def release_source_project_transfer_request(
    request_id: str,
    body: ProjectTransferSourceReleaseRequest,
    request: Request,
    *,
    identity_access: IdentityDependency,
    store: StoreDependency,
    catalog: CatalogDependency,
    operation_lock: OperationLockDependency,
) -> ProjectTransferRequestRecord:
    _require_transfer_space(store, "personal")
    actor = identity_access.require_patch_capable_identity(request)
    try:
        transfer = _transfer_request_or_404(store, request_id)
        with operation_lock(transfer.project_id):
            transfer = _transfer_request_or_404(store, request_id)
            existing_receipt = transfer.source_release_receipt
            if existing_receipt is None:
                if not store.is_project_member(transfer.project_id, actor.user_id):
                    raise ValueError("source release requires current project membership")
                service = catalog.open(transfer.project_id)
                configuration, source_head = capture_project_transfer_source(service)
                if (
                    project_transfer_source_configuration_sha256(configuration)
                    != body.expected_source_configuration_sha256
                    or source_head != body.expected_source_head
                ):
                    raise ValueError(
                        "source configuration or canonical head changed before release"
                    )
                store.record_source_project_transfer_release(
                    request_id,
                    released_by=actor,
                    revalidated_configuration=configuration,
                    source_head=source_head,
                )
            else:
                if (
                    existing_receipt.source_configuration_sha256
                    != body.expected_source_configuration_sha256
                    or existing_receipt.source_head != body.expected_source_head
                ):
                    raise ValueError("source release already binds another canonical boundary")
            return advance_source_project_transfer(store, catalog, request_id)
    except (KeyError, OSError, StateUnavailable, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/api/project-transfers/target-requests/{request_id}/source-release",
    response_model=ProjectTransferRequestRecord,
)
def accept_source_project_transfer_release(
    request_id: str,
    body: ProjectTransferSourceReceiptRequest,
    request: Request,
    *,
    identity_access: IdentityDependency,
    store: StoreDependency,
) -> ProjectTransferRequestRecord:
    identity_access.require_team_space()
    identity_access.acting_user(request)
    try:
        return store.accept_source_project_transfer_release(
            request_id,
            receipt=body.storage_model(),
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/api/project-transfers/target-requests/{request_id}/restore-reentry",
    response_model=ProjectTransferRequestRecord,
)
def reenter_restored_target_project_transfer(
    request_id: str,
    body: ProjectTransferRestoreReentryRequest,
    request: Request,
    *,
    identity_access: IdentityDependency,
    store: StoreDependency,
) -> ProjectTransferRequestRecord:
    """Resume one restored archive-bound target transfer with a new lease."""

    identity_access.require_team_space()
    confirmed_by = identity_access.require_patch_capable_identity(request)
    try:
        transfer, _upload = store.reenter_restored_target_project_transfer(
            request_id,
            expected_restored_revision=body.expected_restored_revision,
            expected_resume_phase=body.expected_resume_phase,
            expected_final_review_digest=body.expected_final_review_digest,
            confirmed_by=confirmed_by,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return transfer


@router.post(
    "/api/project-transfers/requests/{request_id}/archive",
    response_model=ProjectTransferRequestRecord,
)
def bind_project_transfer_archive(
    request_id: str,
    body: ProjectTransferArchiveRequest,
    request: Request,
    *,
    identity_access: IdentityDependency,
    store: StoreDependency,
) -> ProjectTransferRequestRecord:
    actor = identity_access.acting_user(request)
    try:
        transfer = _transfer_request_or_404(store, request_id)
        if transfer.side != "target":
            raise ValueError("source archives are bound only by the source release workflow")
        admission = transfer.target_admission_receipt
        if admission is None or admission.admitted_by.user_id != actor.user_id:
            raise HTTPException(
                status_code=403,
                detail="Only the member who admitted this transfer may bind its archive.",
            )
        return store.bind_project_transfer_archive(
            request_id,
            archive_sha256=body.archive_sha256,
            archive_size_bytes=body.archive_size_bytes,
            source_fence_head=body.source_fence_head,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get(
    "/api/native/project-transfers/source-requests/{request_id}/archive",
    response_class=StreamingResponse,
    include_in_schema=False,
)
def download_source_project_transfer_archive(
    request_id: str,
    request: Request,
    *,
    identity_access: IdentityDependency,
    store: StoreDependency,
    catalog: CatalogDependency,
) -> StreamingResponse:
    """Stream only the exact sealed source archive bound to this request."""

    _require_transfer_space(store, "personal")
    identity_access.require_patch_capable_identity(request)
    expected_instance_id = request.app.state.instance_metadata.instance_id
    if request.headers.get("X-RCP-Instance-ID") != expected_instance_id:
        raise HTTPException(
            status_code=409,
            detail="The native transfer relay must pin the current personal backend instance.",
        )
    try:
        transfer = _transfer_request_or_404(store, request_id)
        if (
            transfer.side != "source"
            or transfer.phase != "archive_bound"
            or transfer.archive_sha256 is None
            or transfer.archive_size_bytes is None
        ):
            raise ValueError("source transfer has no sealed archive to download")
        archive_path = source_transfer_export_path(catalog.data_dir, request_id)
        readback = read_transfer_archive(archive_path)
        if (
            readback.envelope.archive_sha256 != transfer.archive_sha256
            or readback.envelope.archive_size_bytes != transfer.archive_size_bytes
        ):
            raise ValueError("sealed source archive differs from its durable receipt")
    except (KeyError, OSError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return StreamingResponse(
        stream_source_transfer_archive(
            archive_path,
            expected_sha256=transfer.archive_sha256,
            expected_size_bytes=transfer.archive_size_bytes,
        ),
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "no-store",
            "Content-Length": str(transfer.archive_size_bytes),
            "Content-Disposition": f'attachment; filename="{request_id}.rcp-transfer"',
            "X-RCP-Archive-SHA256": transfer.archive_sha256,
        },
    )


@router.post(
    "/api/native/project-transfers/target-requests/{request_id}/cleanup-acknowledgment",
    response_model=ProjectTransferRequestRecord,
    include_in_schema=False,
)
def acknowledge_project_transfer_cleanup(
    request_id: str,
    body: ProjectTransferCleanupAcknowledgmentRequest,
    request: Request,
    *,
    identity_access: IdentityDependency,
    store: StoreDependency,
) -> ProjectTransferRequestRecord:
    identity_access.require_team_space()
    token = _native_bearer_token(request)
    try:
        member = store.authenticate_team_member_token(token)
        transfer = _transfer_request_or_404(store, request_id)
        admission = transfer.target_admission_receipt
        if (
            transfer.side != "target"
            or admission is None
            or admission.admitted_by.user_id != member.user_id
        ):
            raise HTTPException(
                status_code=403,
                detail="Only the member who admitted this transfer may acknowledge cleanup.",
            )
        return store.accept_project_transfer_cleanup_acknowledgment(
            request_id,
            acknowledgment=body.storage_model(),
            accepted_by=AuthorizedHuman(
                space_id=store.space_id,
                user_id=member.user_id,
                display_name=member.display_name,
            ),
        )
    except TeamAuthenticationError as exc:
        raise HTTPException(
            status_code=401,
            detail={
                "code": "team_token_invalid",
                "message": "The member token is invalid or revoked.",
            },
        ) from exc
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/api/native/project-transfers/source-requests/{request_id}/target-activation-proof",
    response_model=ProjectTransferCleanupAcknowledgment,
    include_in_schema=False,
)
async def verify_target_activation_proof(
    request_id: str,
    request: Request,
    *,
    identity_access: IdentityDependency,
    store: StoreDependency,
    catalog: CatalogDependency,
    operation_lock: OperationLockDependency,
) -> ProjectTransferCleanupAcknowledgment:
    _require_transfer_space(store, "personal")
    identity_access.require_patch_capable_identity(request)
    if request.headers.get("content-type", "").split(";", 1)[0] != "application/octet-stream":
        raise HTTPException(status_code=415, detail="Target activation proof must be binary.")
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > 32:
            raise HTTPException(status_code=413, detail="Target activation proof is too large.")
        chunks.append(chunk)
    proof = b"".join(chunks)
    try:
        transfer = _transfer_request_or_404(store, request_id)
        with operation_lock(transfer.project_id):
            return complete_source_project_transfer(
                store,
                catalog,
                request_id,
                target_activation_proof=proof,
            )
    except (KeyError, OSError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get(
    "/api/native/project-transfers/target-requests/{request_id}/activation-proof",
    response_class=Response,
    include_in_schema=False,
)
def retrieve_target_activation_proof(
    request_id: str,
    request: Request,
    *,
    identity_access: IdentityDependency,
    store: StoreDependency,
) -> Response:
    identity_access.require_team_space()
    token = _native_bearer_token(request)
    try:
        member = store.authenticate_team_member_token(token)
        transfer = _transfer_request_or_404(store, request_id)
        admission = transfer.target_admission_receipt
        if (
            transfer.side != "target"
            or admission is None
            or admission.admitted_by.user_id != member.user_id
        ):
            raise HTTPException(
                status_code=403,
                detail="Only the member who admitted this transfer may retrieve its proof.",
            )
        proof = store.expose_project_transfer_proof(request_id)
    except TeamAuthenticationError as exc:
        raise HTTPException(
            status_code=401,
            detail={
                "code": "team_token_invalid",
                "message": "The member token is invalid or revoked.",
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return Response(
        content=proof,
        media_type="application/octet-stream",
        headers={"Cache-Control": "no-store"},
    )


@router.post(
    "/api/project-provisioning/requests",
    response_model=ProjectProvisioningResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_project_provisioning_request(
    body: ProjectProvisioningCreateRequest,
    request: Request,
    *,
    identity_access: IdentityDependency,
    store: StoreDependency,
) -> ProjectProvisioningResponse:
    identity_access.require_team_space()
    authorized_by = identity_access.require_patch_capable_identity(request)
    try:
        record = store.create_project_provisioning_request(
            kind="create_team_project",
            authorized_by=authorized_by,
            machines=[machine.intent() for machine in body.machines],
            repositories=[repository.intent() for repository in body.repositories],
            provider_checks=body.provider_checks,
            name=body.name,
            state_repository=body.state_repository,
            project_truth_scope=body.project_truth_scope,
            default_run_truth_scope=body.default_run_truth_scope,
            default_auto_research_invocation_ceiling=(
                body.default_auto_research_invocation_ceiling
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _project_provisioning_response(record, viewer_user_id=authorized_by.user_id)


@router.get(
    "/api/project-provisioning/requests",
    response_model=list[ProjectProvisioningResponse],
)
def project_provisioning_requests(
    request: Request,
    *,
    identity_access: IdentityDependency,
    store: StoreDependency,
) -> list[ProjectProvisioningResponse]:
    identity_access.require_team_space()
    viewer = identity_access.acting_user(request)
    return [
        _project_provisioning_response(record, viewer_user_id=viewer.user_id)
        for record in store.project_provisioning_requests()
        if record.kind == "create_team_project"
    ]


@router.get(
    "/api/project-provisioning/requests/{request_id}",
    response_model=ProjectProvisioningResponse,
)
def project_provisioning_request(
    request_id: str,
    request: Request,
    *,
    identity_access: IdentityDependency,
    store: StoreDependency,
) -> ProjectProvisioningResponse:
    identity_access.require_team_space()
    viewer = identity_access.acting_user(request)
    record = _request_or_404(store, request_id)
    if record.kind != "create_team_project":
        raise HTTPException(status_code=404, detail="Provisioning request not found")
    return _project_provisioning_response(record, viewer_user_id=viewer.user_id)


@router.post(
    "/api/project-provisioning/requests/{request_id}/cancel",
    response_model=ProjectProvisioningResponse,
)
def cancel_project_provisioning_request(
    request_id: str,
    request: Request,
    *,
    identity_access: IdentityDependency,
    store: StoreDependency,
) -> ProjectProvisioningResponse:
    identity_access.require_team_space()
    viewer = identity_access.acting_user(request)
    record = _request_or_404(store, request_id)
    if record.kind != "create_team_project":
        raise HTTPException(status_code=404, detail="Provisioning request not found")
    if record.authorized_by.user_id != viewer.user_id:
        raise HTTPException(
            status_code=403,
            detail="Only the member who authorized this preparation may cancel it.",
        )
    if record.status == "cancelled":
        return _project_provisioning_response(record, viewer_user_id=viewer.user_id)
    if record.status != "waiting_for_server_setup":
        raise HTTPException(
            status_code=409,
            detail=(
                "Server preparation has started. Its machine-owned setup flow must first "
                "record the exact cleanup or reuse disposition."
            ),
        )
    try:
        cancelled = store.transition_project_provisioning_request(
            request_id,
            receipt_id=f"member-cancel-{record.revision}",
            phase="member_cancel",
            expected_revision=record.revision,
            expected_status=record.status,
            to_status="cancelled",
            machines=record.machines,
            repositories=record.repositories,
            provider_checks=record.provider_checks,
            cancellation_disposition="nothing_to_remove",
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=409,
            detail="The provisioning request changed; reload it before cancelling.",
        ) from exc
    return _project_provisioning_response(cancelled, viewer_user_id=viewer.user_id)


@router.post(
    "/api/project-provisioning/requests/{request_id}/complete",
    response_model=ProjectProvisioningResponse,
)
def complete_project_provisioning_request(
    request_id: str,
    body: ProjectProvisioningCompleteRequest,
    request: Request,
    *,
    identity_access: IdentityDependency,
    setup: SetupDependency,
    store: StoreDependency,
) -> ProjectProvisioningResponse:
    """Create exactly the project reviewed by one current named team member."""

    identity_access.require_team_space()
    reviewer = identity_access.require_patch_capable_identity(request)
    record = _request_or_404(store, request_id)
    if record.kind != "create_team_project":
        raise HTTPException(status_code=404, detail="Provisioning request not found")
    if body.final_review_digest != record.final_review_digest:
        raise HTTPException(
            status_code=409,
            detail="The provisioning review changed; reload it before creating the project.",
        )
    if record.status == "completed":
        _require_completed_project(store, record)
        return _project_provisioning_response(record, viewer_user_id=reviewer.user_id)
    if record.status != "ready_for_review":
        raise HTTPException(
            status_code=409,
            detail="Only a request that is ready for final review can create a project.",
        )
    authorizer = store.space_user(record.authorized_by.user_id)
    if authorizer is None or authorizer.identity_kind != "team_member":
        raise HTTPException(
            status_code=409,
            detail=(
                "The member who authorized preparation is no longer enrolled. Create and "
                "review a new provisioning request."
            ),
        )

    try:
        card = setup.create_prepared_team_project(record, seat_member=reviewer.user_id)
    except (ProjectIdentityConflict, KeyError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (FileNotFoundError, OSError, StateUnavailable) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if card.get("id") != record.proposed_project_id:
        raise RuntimeError("prepared project registration returned another project identity")

    current = _request_or_404(store, request_id)
    if current.status == "completed":
        _require_completed_project(store, current)
        return _project_provisioning_response(current, viewer_user_id=reviewer.user_id)
    if (
        current.status != "ready_for_review"
        or current.revision != record.revision
        or current.final_review_digest != body.final_review_digest
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "The provisioning request changed while the reviewed project was being "
                "registered. Reload the same request to reconcile it."
            ),
        )
    try:
        completed = store.transition_project_provisioning_request(
            request_id,
            receipt_id=f"member-finalize:{reviewer.user_id}",
            phase="member_finalize",
            expected_revision=current.revision,
            expected_status="ready_for_review",
            to_status="completed",
            machines=current.machines,
            repositories=current.repositories,
            provider_checks=current.provider_checks,
        )
    except (KeyError, ValueError) as exc:
        reconciled = _request_or_404(store, request_id)
        if (
            reconciled.status == "completed"
            and reconciled.final_review_digest == body.final_review_digest
        ):
            _require_completed_project(store, reconciled)
            return _project_provisioning_response(
                reconciled,
                viewer_user_id=reviewer.user_id,
            )
        raise HTTPException(
            status_code=409,
            detail="The provisioning request changed; reload it before creating the project.",
        ) from exc
    _require_completed_project(store, completed)
    return _project_provisioning_response(completed, viewer_user_id=reviewer.user_id)


def _require_completed_project(
    store: AppStore,
    request: ProjectProvisioningRequestRecord,
) -> None:
    project = store.project(request.proposed_project_id)
    if project is None or project.home_space_id != request.target_space_id:
        raise HTTPException(
            status_code=503,
            detail="The completed provisioning request lost its exact registered project.",
        )


def _request_or_404(store: AppStore, request_id: str) -> ProjectProvisioningRequestRecord:
    try:
        record = store.project_provisioning_request(request_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Provisioning request not found") from exc
    if record is None:
        raise HTTPException(status_code=404, detail="Provisioning request not found")
    return record


def _transfer_request_or_404(
    store: AppStore,
    request_id: str,
) -> ProjectTransferRequestRecord:
    try:
        record = store.project_transfer_request(request_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Transfer request not found") from exc
    if record is None:
        raise HTTPException(status_code=404, detail="Transfer request not found")
    return record


def _require_transfer_space(store: AppStore, expected: SpaceKind) -> None:
    if store.space_kind != expected:
        raise HTTPException(status_code=404, detail="Transfer request not found")


def _native_bearer_token(request: Request) -> str:
    authorization = request.headers.get("authorization", "")
    scheme, separator, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not separator or not token or " " in token:
        raise HTTPException(
            status_code=401,
            detail={
                "code": "team_token_required",
                "message": "This native route requires a permanent team member token.",
            },
        )
    return token


def _project_transfer_response(
    record: ProjectTransferRequestRecord,
    *,
    store: AppStore,
    viewer_user_id: str,
) -> ProjectTransferResponse:
    """Project a transfer record into the decisions needed by its UI owner."""

    provisioning = (
        store.project_provisioning_request(record.request_id) if record.side == "target" else None
    )
    effective_phase = (
        record.restore_resume_phase if record.phase == "operator_action_needed" else record.phase
    )
    target_ready_for_review = provisioning is not None and provisioning.status == "ready_for_review"
    can_link = record.side == "source" and record.phase == "awaiting_link"
    can_run_setup = (
        record.side == "target"
        and effective_phase == "linked"
        and provisioning is not None
        and provisioning.status
        in {"waiting_for_server_setup", "setup_in_progress", "operator_action_needed"}
    )
    can_admit = (
        record.side == "target"
        and record.phase == "linked"
        and target_ready_for_review
        and record.target_admission_receipt is None
    )
    can_accept_admission = (
        record.side == "source"
        and record.phase == "linked"
        and record.target_admission_receipt is None
    )
    can_release = (
        record.side == "source"
        and record.phase == "target_admitted"
        and record.source_release_receipt is None
    )
    can_accept_release = (
        record.side == "target"
        and record.phase == "target_admitted"
        and record.source_release_receipt is None
    )
    can_relay = (record.side == "source" and record.phase == "archive_bound") or (
        record.side == "target" and record.phase == "source_released"
    )
    can_restore_reentry = (
        record.side == "target"
        and record.phase == "operator_action_needed"
        and record.restore_resume_phase == "archive_bound"
        and target_ready_for_review
        and record.target_admission_receipt is not None
        and record.target_admission_receipt.admitted_by.space_id == store.space_id
        and record.target_admission_receipt.admitted_by.user_id == viewer_user_id
    )
    # Target cleanup is completed by the native proof relay. There is no
    # browser mutation to advertise as a generic "complete" action.
    can_complete = False

    if record.phase == "operator_action_needed":
        next_action = record.restore_diagnostic or "Re-enter the restored transfer boundary."
    elif record.side == "source":
        next_action = {
            "awaiting_link": "Link the target transfer request.",
            "linked": "Wait for target preparation and admission.",
            "target_admitted": "Review and release the source project.",
            "source_released": "Wait for source fencing and archive sealing.",
            "source_fenced": "Bind and relay the sealed source archive.",
            "archive_bound": "Relay the sealed source archive to the target.",
            "target_activated": "Wait for target cleanup confirmation.",
            "cleanup_acknowledged": "Finish source transfer cleanup.",
            "completed": None,
        }[record.phase]
    else:
        if record.phase == "linked":
            if target_ready_for_review:
                next_action = "Review and admit the prepared target project."
            elif provisioning is not None and provisioning.status == "cancelled":
                next_action = "The target preparation was cancelled."
            else:
                next_action = "Run target server setup."
        else:
            next_action = {
                "awaiting_link": None,
                "target_admitted": "Wait for the source release.",
                "source_released": "Relay and bind the sealed source archive.",
                "source_fenced": None,
                "archive_bound": "Activate the imported target project.",
                "target_activated": "Wait for native transfer cleanup confirmation.",
                "cleanup_acknowledged": "Finish target transfer cleanup.",
                "completed": None,
            }[record.phase]

    # Keep tuple-valued protocol fields as tuples while validating this strict
    # response model; FastAPI performs the JSON conversion at the boundary.
    payload = record.model_dump()
    payload.update(
        {
            "phase_label": _TRANSFER_PHASE_LABELS[record.phase],
            "next_action": next_action,
            "can_link": can_link,
            "can_run_setup": can_run_setup,
            "can_review": can_admit,
            "can_admit": can_admit,
            "can_accept_admission": can_accept_admission,
            "can_release": can_release,
            "can_accept_release": can_accept_release,
            "can_relay": can_relay,
            "can_restore_reentry": can_restore_reentry,
            "can_complete": can_complete,
            "finished": record.phase == "completed",
        }
    )
    return ProjectTransferResponse.model_validate(payload)


def _project_provisioning_response(
    record: ProjectProvisioningRequestRecord,
    *,
    viewer_user_id: str,
) -> ProjectProvisioningResponse:
    machines = [_machine_projection(machine) for machine in record.machines]
    repositories = [_repository_projection(repository) for repository in record.repositories]
    providers = [_provider_projection(check) for check in record.provider_checks]
    readiness = ProjectProvisioningReadinessProjection(
        machines_ready=sum(machine.ready for machine in machines),
        machines_total=len(machines),
        repositories_ready=sum(repository.ready for repository in repositories),
        repositories_total=len(repositories),
        providers_ready=sum(provider.ready for provider in providers),
        providers_total=len(providers),
        all_ready=all(machine.ready for machine in machines)
        and all(repository.ready for repository in repositories)
        and all(provider.ready for provider in providers),
    )
    return ProjectProvisioningResponse(
        request_id=record.request_id,
        kind=record.kind,
        status=record.status,
        status_label=_STATUS_LABELS[record.status],
        next_action=_next_action(record),
        can_run_setup=record.status
        in {"waiting_for_server_setup", "setup_in_progress", "operator_action_needed"},
        can_review=record.status == "ready_for_review",
        can_cancel=(
            record.kind == "create_team_project"
            and record.status == "waiting_for_server_setup"
            and record.authorized_by.user_id == viewer_user_id
        ),
        target_space_id=record.target_space_id,
        proposed_project_id=record.proposed_project_id,
        name=record.name,
        state_repository=record.state_repository,
        project_truth_scope=record.project_truth_scope,
        default_run_truth_scope=record.default_run_truth_scope,
        default_auto_research_invocation_ceiling=(record.default_auto_research_invocation_ceiling),
        authorized_by=record.authorized_by,
        machines=machines,
        repositories=repositories,
        provider_checks=providers,
        readiness=readiness,
        diagnostic=record.retryable_diagnostic,
        operator_action=record.operator_action,
        operator_argv=(
            str(DEFAULT_SERVER_LAYOUT.cli_wrapper),
            "server",
            "project",
            "provision",
            record.request_id,
        ),
        final_review=(
            ProjectProvisioningFinalReview(
                digest=record.final_review_digest,
                proposed_project_id=record.proposed_project_id,
                authorized_by=record.authorized_by,
                ready_at=record.ready_at,
            )
            if record.final_review_digest is not None and record.ready_at is not None
            else None
        ),
        cancellation_disposition=record.cancellation_disposition,
        revision=record.revision,
        created_at=record.created_at,
        updated_at=record.updated_at,
        setup_started_at=record.setup_started_at,
        completed_at=record.completed_at,
        cancelled_at=record.cancelled_at,
    )


def _next_action(record: ProjectProvisioningRequestRecord) -> str | None:
    if record.status == "waiting_for_server_setup":
        return "Run server setup."
    if record.status == "setup_in_progress":
        return "Wait for server setup, or resume the same command after an interruption."
    if record.status == "operator_action_needed":
        assert record.operator_action is not None
        return record.operator_action.message
    if record.status == "ready_for_review":
        return "Review the prepared project."
    return None


def _machine_projection(
    machine: ProjectProvisioningMachineRecord,
) -> ProjectProvisioningMachineProjection:
    ready = machine.resolved_central_root is not None
    return ProjectProvisioningMachineProjection(
        alias=machine.alias,
        location=machine.location,
        host=machine.host,
        os_account=machine.os_account,
        intended_central_root=machine.central_root,
        resolved_central_root=machine.resolved_central_root,
        ready=ready,
        status_label="Ready" if ready else "Waiting for setup",
    )


def _repository_projection(
    repository: ProjectProvisioningRepositoryRecord,
) -> ProjectProvisioningRepositoryProjection:
    check = repository.git_check
    return ProjectProvisioningRepositoryProjection(
        alias=repository.alias,
        repository=repository.repository,
        https_clone_url=repository.repository.https_clone_url,
        ssh_clone_url=repository.repository.ssh_clone_url,
        settings_url=repository.repository.settings_url,
        machine_alias=repository.machine_alias,
        intended_path=repository.intended_path,
        resolved_path=repository.resolved_path,
        checkout_disposition=repository.checkout_disposition,
        status=check.status,
        status_label=_CHECK_LABELS[check.status],
        ready=check.status == "ready",
        commit=check.commit,
        write_verified=check.write_verified,
        deploy_key_label=check.deploy_key_label,
        public_key_fingerprint=check.public_key_fingerprint,
        checked_at=check.checked_at,
        diagnostic=check.diagnostic,
    )


def _provider_projection(
    check: ProjectProvisioningProviderCheckRecord,
) -> ProjectProvisioningProviderProjection:
    return ProjectProvisioningProviderProjection(
        profile=check.profile,
        provider=check.provider,
        runtime_id=check.runtime_id,
        model=check.model,
        reasoning=check.reasoning,
        machine_alias=check.machine_alias,
        status=check.status,
        status_label=_CHECK_LABELS[check.status],
        ready=check.status == "ready",
        binary_path=check.binary_path,
        version=check.version,
        resolved_runtime_id=check.resolved_runtime_id,
        execution_account=check.execution_account,
        checked_at=check.checked_at,
        diagnostic=check.diagnostic,
    )


__all__ = [
    "ProjectCreationControl",
    "ProjectProvisioningCompleteRequest",
    "ProjectProvisioningCreateRequest",
    "ProjectProvisioningResponse",
    "ProjectTransferResponse",
    "ProjectTransferRestoreReentryRequest",
    "ProjectTransferSourceBoundaryResponse",
    "project_creation_control",
    "router",
]
