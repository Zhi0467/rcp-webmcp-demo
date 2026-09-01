"""Resumable machine preparation for one authorized team-project request."""

from __future__ import annotations

import hashlib
import json
import os
import socket
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import BinaryIO, Literal, Protocol

from rcp.server_ops.cli import CallerIdentity, PreparedServerCommand, ServerEventEmitter
from rcp.server_ops.control import (
    ServerControlClient,
    ServerControlError,
    ServerControlProjectPlanResult,
    ServerControlProjectStepResult,
    ServerControlProjectTarget,
)
from rcp.server_ops.git_credentials import (
    DeployKeyMaterial,
    GitCredentialManager,
    GitCredentialRefused,
    GitWriteProbe,
    cleanup_ref_operator_step,
    deploy_key_operator_step,
    empty_repository_operator_step,
)
from rcp.server_ops.layout import DEFAULT_SERVER_LAYOUT, ServerLayout
from rcp.server_ops.models import (
    ExternalAction,
    ExternalServiceTarget,
    MachineTarget,
    NonsecretField,
    ServerCommandRequest,
    ServerPlanEvent,
    ServerStep,
)
from rcp.server_ops.project_checkout import (
    ProjectCheckoutManager,
    ProjectCheckoutRefused,
    retained_research_operator_step,
)
from rcp.server_ops.provider_readiness import ProviderReadinessCoordinator
from rcp.server_runtime import ServerMetadata, data_dir_identity
from rcp.storage import (
    AppStore,
    ProjectProvisioningGitCheckRecord,
    ProjectProvisioningMachineRecord,
    ProjectProvisioningProviderCheckRecord,
    ProjectProvisioningRepositoryRecord,
    ProjectProvisioningRequestRecord,
)

_ProvisionTargetKind = Literal[
    "start",
    "repository_key",
    "repository_write",
    "repository_checkout",
    "provider",
    "final_review",
]


class ProjectProvisionRefused(ServerControlError):
    """The selected durable preparation boundary is no longer safe to run."""

    def __init__(self, message: str) -> None:
        super().__init__("operation_refused", message)


class ProjectProvisionControl(Protocol):
    def project_provision_plan(self, *, request_id: str) -> ServerControlProjectPlanResult: ...

    def advance_project_provision(
        self,
        *,
        request_id: str,
        boundary_sha256: str,
        target_id: str,
    ) -> ServerControlProjectStepResult: ...


@dataclass(frozen=True)
class _ProvisionTarget:
    target_id: str
    kind: _ProvisionTargetKind
    step: ServerStep
    repository_index: int | None = None
    provider_index: int | None = None
    provider_target_id: str | None = None


@dataclass(frozen=True)
class _ProvisionBoundary:
    request: ProjectProvisioningRequestRecord
    boundary_sha256: str
    targets: tuple[_ProvisionTarget, ...]


def prepare_project_provision_command(
    request: ServerCommandRequest,
    identity: CallerIdentity,
    *,
    control: ProjectProvisionControl | None = None,
    layout: ServerLayout = DEFAULT_SERVER_LAYOUT,
) -> PreparedServerCommand:
    """Publish the full request plan before advancing its first durable step."""

    if request.command != "server project provision" or request.request_id is None:
        raise ValueError("prepare_project_provision_command requires one provisioning request")
    client = control or ServerControlClient.from_data_dir(
        layout.data_dir,
        expected_server_uid=os.geteuid(),
    )
    resolved = client.project_provision_plan(request_id=request.request_id)
    plan = ServerPlanEvent(
        command=request.command,
        timestamp=datetime.now(UTC),
        steps=tuple(target.step for target in resolved.targets),
    )

    def execute(emitter: ServerEventEmitter, _input_stream: BinaryIO) -> None:
        boundary = resolved.boundary_sha256
        last_result: ServerControlProjectStepResult | None = None
        for target_index, target in enumerate(resolved.targets):
            pending = target.step
            emitter.emit_step(
                pending.model_copy(
                    update={
                        "state": "running",
                        "message": "RCP is advancing this exact durable preparation step.",
                    }
                )
            )
            try:
                advanced = client.advance_project_provision(
                    request_id=request.request_id,
                    boundary_sha256=boundary,
                    target_id=target.target_id,
                )
            except ServerControlError as exc:
                emitter.emit_step(
                    pending.model_copy(update={"state": "failed", "message": str(exc)})
                )
                return
            if advanced.target_id != target.target_id:
                raise ProjectProvisionRefused(
                    "The control result names another project-provisioning target."
                )
            terminal = advanced.step
            if (
                terminal.state == "succeeded"
                and target_index == len(resolved.targets) - 1
                and advanced.request_status != "ready_for_review"
            ):
                terminal = terminal.model_copy(
                    update={
                        "state": "failed",
                        "message": (
                            "The machine steps finished without a durable ready-for-review "
                            "readback. Inspect the server log and rerun this exact request."
                        ),
                    }
                )
            emitter.emit_step(terminal)
            if terminal.state != "succeeded":
                return
            boundary = advanced.next_boundary_sha256
            last_result = advanced
        if last_result is None or last_result.request_status != "ready_for_review":
            raise ProjectProvisionRefused(
                "The command completed its plan without a durable ready-for-review readback."
            )

    return PreparedServerCommand(plan=plan, execute=execute)


class ProjectProvisionCoordinator:
    """Compose Git identity, write proof, checkout, and provider owners durably."""

    def __init__(
        self,
        store: AppStore,
        metadata: ServerMetadata,
        provider_coordinator: ProviderReadinessCoordinator,
        *,
        credential_manager: GitCredentialManager | None = None,
        checkout_manager: ProjectCheckoutManager | None = None,
        layout: ServerLayout = DEFAULT_SERVER_LAYOUT,
        local_host: str | None = None,
    ) -> None:
        if store.space_kind != "team":
            raise ValueError("project provisioning requires an installed team space")
        if (
            metadata.owner_kind != "cli"
            or metadata.control_socket is None
            or metadata.data_dir_id != data_dir_identity(store.path.parent)
        ):
            raise ValueError(
                "project provisioning requires this exact CLI-owned team service process"
            )
        self.store = store
        self.metadata = metadata
        self.provider_coordinator = provider_coordinator
        self.layout = layout
        self.credential_manager = credential_manager or GitCredentialManager(layout)
        self.checkout_manager = checkout_manager or ProjectCheckoutManager(
            layout,
            credential_manager=self.credential_manager,
        )
        self.local_host = local_host or socket.gethostname()

    def plan(self, request_id: str) -> ServerControlProjectPlanResult:
        boundary = self._resolve(request_id)
        return ServerControlProjectPlanResult(
            **self._identity_fields(),
            request_id=request_id,
            request_status=boundary.request.status,
            revision=boundary.request.revision,
            boundary_sha256=boundary.boundary_sha256,
            targets=tuple(
                ServerControlProjectTarget(target_id=target.target_id, step=target.step)
                for target in boundary.targets
            ),
        )

    def project_provision_plan(self, *, request_id: str) -> ServerControlProjectPlanResult:
        return self.plan(request_id)

    def advance(
        self,
        request_id: str,
        *,
        boundary_sha256: str,
        target_id: str,
    ) -> ServerControlProjectStepResult:
        boundary = self._resolve(request_id)
        if boundary.boundary_sha256 != boundary_sha256:
            raise ProjectProvisionRefused(
                "The provisioning request changed after the plan was shown; rerun the command."
            )
        target = next(
            (candidate for candidate in boundary.targets if candidate.target_id == target_id),
            None,
        )
        if target is None:
            raise ProjectProvisionRefused(
                "The provisioning target changed after the plan was shown; rerun the command."
            )
        step = self._advance_target(boundary.request, target)
        current = self.store.project_provisioning_request(request_id)
        if current is None or current.target_space_id != self.store.space_id:
            raise ProjectProvisionRefused("The provisioning request disappeared after its step.")
        if current.status in {"completed", "cancelled"}:
            raise ProjectProvisionRefused(
                f"The provisioning request became {current.status} during preparation."
            )
        next_boundary = self._resolve(request_id)
        return ServerControlProjectStepResult(
            **self._identity_fields(),
            request_id=request_id,
            request_status=current.status,
            revision=current.revision,
            target_id=target_id,
            boundary_sha256=boundary_sha256,
            next_boundary_sha256=next_boundary.boundary_sha256,
            step=step,
        )

    def advance_project_provision(
        self,
        *,
        request_id: str,
        boundary_sha256: str,
        target_id: str,
    ) -> ServerControlProjectStepResult:
        return self.advance(
            request_id,
            boundary_sha256=boundary_sha256,
            target_id=target_id,
        )

    def _resolve(self, request_id: str) -> _ProvisionBoundary:
        request = self.store.project_provisioning_request(request_id)
        if request is None or request.target_space_id != self.store.space_id:
            raise ProjectProvisionRefused("The selected provisioning request does not exist.")
        if request.status in {"completed", "cancelled"}:
            raise ProjectProvisionRefused(
                f"The selected provisioning request is already {request.status}."
            )
        if request.cancellation_disposition is not None:
            raise ProjectProvisionRefused(
                "The provisioning request is in explicit cancellation handling; preparation "
                "cannot clear or reinterpret that disposition."
            )
        if self.store.project(request.proposed_project_id) is not None:
            raise ProjectProvisionRefused(
                "The proposed project already exists; server preparation cannot alter it."
            )
        targets = self._targets(request)
        payload = {
            "request_id": request.request_id,
            "revision": request.revision,
            "status": request.status,
            "targets": [target.target_id for target in targets],
        }
        return _ProvisionBoundary(
            request=request,
            boundary_sha256=_digest(payload),
            targets=targets,
        )

    def _targets(
        self,
        request: ProjectProvisioningRequestRecord,
    ) -> tuple[_ProvisionTarget, ...]:
        if not request.configuration_complete:
            return (self._configuration_target(request),)
        machine_map = {machine.alias: machine for machine in request.machines}
        targets: list[_ProvisionTarget] = []

        def add(
            kind: _ProvisionTargetKind,
            step: ServerStep,
            *,
            repository_index: int | None = None,
            provider_index: int | None = None,
            provider_target_id: str | None = None,
            identity: object,
        ) -> None:
            targets.append(
                _ProvisionTarget(
                    target_id=_digest(
                        {
                            "request_id": request.request_id,
                            "kind": kind,
                            "identity": identity,
                        }
                    ),
                    kind=kind,
                    step=step.model_copy(update={"number": len(targets) + 1}),
                    repository_index=repository_index,
                    provider_index=provider_index,
                    provider_target_id=provider_target_id,
                )
            )

        add(
            "start",
            self._machine_step(
                1,
                title="Enter server preparation",
                purpose="Claim the authorized request's next durable preparation revision.",
                phase="provisioning_start",
                expected="The request is durably marked as server setup in progress.",
                message="RCP will enter or resume server preparation.",
            ),
            identity="start",
        )
        for index, repository in enumerate(request.repositories):
            machine = machine_map[repository.machine_alias]
            machine_target = self._machine_target(machine)
            repository_target = ExternalServiceTarget(
                service="github.com",
                resource=repository.repository.identity,
                destination_url=repository.repository.settings_url,
                required_authority_role="repository administrator",
            )
            add(
                "repository_key",
                ServerStep(
                    number=1,
                    title=f"Prepare deploy key for {repository.alias}",
                    purpose="Prepare one repository-scoped key on the exact checkout account.",
                    performed_by="system",
                    target=machine_target,
                    phase="repository_key",
                    state="pending",
                    expected_success="The exact key label and public fingerprint are durable.",
                    message=f"RCP will prepare the deploy key for {repository.alias}.",
                ),
                repository_index=index,
                identity={"repository": repository.alias, "phase": "key"},
            )
            add(
                "repository_write",
                ServerStep(
                    number=1,
                    title=f"Verify Git write access for {repository.alias}",
                    purpose="Prove repository-scoped read and write access without changing code.",
                    performed_by="system",
                    target=repository_target,
                    phase="repository_write",
                    state="pending",
                    expected_success=(
                        "A request-scoped ref is written, read back exactly, and removed."
                    ),
                    message=f"RCP will verify Git write access for {repository.alias}.",
                ),
                repository_index=index,
                identity={"repository": repository.alias, "phase": "write"},
            )
            add(
                "repository_checkout",
                ServerStep(
                    number=1,
                    title=f"Prepare central checkout for {repository.alias}",
                    purpose="Clone or verify the exact central checkout without rewriting work.",
                    performed_by="system",
                    target=machine_target,
                    phase="repository_checkout",
                    state="pending",
                    expected_success=(
                        "The exact checkout path, origin, commit, and ownership are durable."
                    ),
                    message=f"RCP will prepare the central checkout for {repository.alias}.",
                ),
                repository_index=index,
                identity={"repository": repository.alias, "phase": "checkout"},
            )
        provider_plan = self.provider_coordinator.plan("request", request.request_id)
        for provider_index, (provider_target, profile) in enumerate(
            zip(provider_plan.targets, request.provider_checks, strict=True)
        ):
            machine = machine_map[profile.machine_alias]
            add(
                "provider",
                ServerStep(
                    number=1,
                    title=f"Verify provider profile {profile.profile}",
                    purpose=(
                        "Check the saved provider on its exact account without managing its auth."
                    ),
                    performed_by="system",
                    target=self._machine_target(machine),
                    phase="provider_readiness",
                    state="pending",
                    expected_success=(
                        "Executable, authentication, runtime, model, and account all match."
                    ),
                    message=f"RCP will verify provider profile {profile.profile}.",
                ),
                provider_index=provider_index,
                provider_target_id=provider_target.target_id,
                identity={"provider_target": provider_target.target_id},
            )
        add(
            "final_review",
            self._machine_step(
                1,
                title="Publish final preparation review",
                purpose="Bind the exact machine, Git, checkout, and provider results for review.",
                phase="provisioning_review",
                expected="The request is durably ready for a human's final creation decision.",
                message="RCP will publish the final preparation digest.",
            ),
            identity="final-review",
        )
        return tuple(targets)

    def _configuration_target(
        self,
        request: ProjectProvisioningRequestRecord,
    ) -> _ProvisionTarget:
        step = self._machine_step(
            1,
            title="Replace the incomplete legacy request",
            purpose="Refuse machine work until the complete reviewed project configuration exists.",
            phase="provisioning_configuration",
            expected="A new request names its project, state repository, and truth scopes.",
            message="RCP will verify that this request carries complete project configuration.",
            performed_by="human",
        )
        return _ProvisionTarget(
            target_id=_digest({"request_id": request.request_id, "kind": "configuration-required"}),
            kind="start",
            step=step,
        )

    def _advance_target(
        self,
        request: ProjectProvisioningRequestRecord,
        target: _ProvisionTarget,
    ) -> ServerStep:
        if not request.configuration_complete:
            return self._pause(
                request,
                target.step,
                message=(
                    "This legacy request does not contain the reviewed project configuration. "
                    "Cancel it and create a new team-project request in the current setup flow."
                ),
                actions=(
                    ExternalAction(
                        instruction=(
                            "Cancel this request in RCP, then create a new team project with its "
                            "name, state repository, truth scopes, machines, repositories, and "
                            "provider profiles."
                        )
                    ),
                ),
                phase="provisioning_configuration",
            )
        if target.kind == "start":
            return self._start(request, target.step)
        if target.kind == "final_review":
            return self._final_review(request, target.step)
        if target.kind == "provider":
            assert target.provider_target_id is not None and target.provider_index is not None
            check = request.provider_checks[target.provider_index]
            if check.status == "ready" and all(
                value is not None
                for value in (
                    check.execution_account,
                    check.binary_path,
                    check.version,
                    check.resolved_runtime_id,
                )
            ):
                return target.step.model_copy(
                    update={
                        "state": "succeeded",
                        "message": f"Provider profile {check.profile} is durably ready.",
                        "fields": self._provider_fields(check),
                    }
                )
            return self.provider_coordinator.check_for_project_provision(
                request.request_id,
                target_id=target.provider_target_id,
                pending=target.step,
            )
        assert target.repository_index is not None
        if target.kind == "repository_key":
            return self._prepare_key(request, target.repository_index, target.step)
        if target.kind == "repository_write":
            return self._verify_write(request, target.repository_index, target.step)
        return self._prepare_checkout(request, target.repository_index, target.step)

    def _start(self, request: ProjectProvisioningRequestRecord, pending: ServerStep) -> ServerStep:
        if request.status in {"setup_in_progress", "ready_for_review"}:
            return pending.model_copy(
                update={
                    "state": "succeeded",
                    "message": f"The request is durably {request.status.replace('_', ' ')}.",
                    "fields": (NonsecretField(name="revision", value=request.revision),),
                }
            )
        updated = self._transition(
            request,
            phase="provisioning_start",
            to_status="setup_in_progress",
        )
        return pending.model_copy(
            update={
                "state": "succeeded",
                "message": "Server preparation is durably in progress.",
                "fields": (NonsecretField(name="revision", value=updated.revision),),
            }
        )

    def _prepare_key(
        self,
        request: ProjectProvisioningRequestRecord,
        repository_index: int,
        pending: ServerStep,
    ) -> ServerStep:
        repository = request.repositories[repository_index]
        machine = self._machine(request, repository.machine_alias)
        try:
            material = self.credential_manager.prepare_key(
                machine,
                repository.repository,
                space_id=request.target_space_id,
                project_id=request.proposed_project_id,
                repository_alias=repository.alias,
            )
        except GitCredentialRefused as exc:
            return self._repository_failure(
                request,
                repository_index,
                pending,
                message=str(exc),
                instruction=(
                    f"Repair the exact {machine.os_account} account path or SSH route, then "
                    "resume this request. RCP did not use another account."
                ),
                phase="repository_key",
            )
        check = repository.git_check
        if (
            check.deploy_key_label == material.label
            and check.public_key_fingerprint == material.public_key_fingerprint
        ):
            return pending.model_copy(
                update={
                    "state": "succeeded",
                    "message": f"The deploy key for {repository.alias} is durably identified.",
                    "fields": self._key_fields(material),
                }
            )
        updated_check = self._pending_git_check(material)
        repositories = list(request.repositories)
        repositories[repository_index] = repository.model_copy(update={"git_check": updated_check})
        self._transition(
            request,
            phase="repository_key",
            to_status="setup_in_progress",
            repositories=repositories,
        )
        return pending.model_copy(
            update={
                "state": "succeeded",
                "message": f"The deploy key for {repository.alias} is durably identified.",
                "fields": self._key_fields(material),
            }
        )

    def _verify_write(
        self,
        request: ProjectProvisioningRequestRecord,
        repository_index: int,
        pending: ServerStep,
    ) -> ServerStep:
        repository = request.repositories[repository_index]
        machine = self._machine(request, repository.machine_alias)
        try:
            material = self.credential_manager.prepare_key(
                machine,
                repository.repository,
                space_id=request.target_space_id,
                project_id=request.proposed_project_id,
                repository_alias=repository.alias,
            )
        except GitCredentialRefused as exc:
            return self._repository_failure(
                request,
                repository_index,
                pending,
                message=str(exc),
                instruction="Repair the exact deploy-key path or SSH route, then resume.",
                phase="repository_write",
            )
        check = repository.git_check
        if (
            check.deploy_key_label != material.label
            or check.public_key_fingerprint != material.public_key_fingerprint
        ):
            source = deploy_key_operator_step(
                self.credential_manager,
                machine,
                material,
                number=pending.number,
                request_id=request.request_id,
                resume_argv=self._resume_argv(request.request_id),
            )
            return self._persist_git_pause(
                request,
                repository_index,
                pending,
                source,
                material,
            )
        if check.status == "ready":
            assert check.commit is not None
            return pending.model_copy(
                update={
                    "state": "succeeded",
                    "message": f"Git write access for {repository.alias} is durably proven.",
                    "fields": self._git_fields(repository, check),
                }
            )
        try:
            probe = self.credential_manager.probe_write(
                machine,
                material,
                request_id=request.request_id,
            )
        except GitCredentialRefused as exc:
            return self._repository_failure(
                request,
                repository_index,
                pending,
                message=str(exc),
                instruction="Inspect the exact request-owned Git probe path, then resume.",
                phase="repository_write",
                material=material,
            )
        if probe.ready:
            assert probe.commit is not None
            ready = ProjectProvisioningGitCheckRecord(
                status="ready",
                commit=probe.commit,
                write_verified=True,
                deploy_key_label=material.label,
                public_key_fingerprint=material.public_key_fingerprint,
                checked_at=self.store.now(),
            )
            repositories = list(request.repositories)
            repositories[repository_index] = repository.model_copy(update={"git_check": ready})
            self._transition(
                request,
                phase="repository_write",
                to_status="setup_in_progress",
                repositories=repositories,
            )
            return pending.model_copy(
                update={
                    "state": "succeeded",
                    "message": probe.diagnostic,
                    "fields": self._git_fields(repository, ready),
                }
            )
        source = self._probe_operator_step(
            request,
            machine,
            material,
            probe,
            pending,
        )
        return self._persist_git_pause(
            request,
            repository_index,
            pending,
            source,
            material,
        )

    def _prepare_checkout(
        self,
        request: ProjectProvisioningRequestRecord,
        repository_index: int,
        pending: ServerStep,
    ) -> ServerStep:
        repository = request.repositories[repository_index]
        check = repository.git_check
        if (
            check.status == "ready"
            and repository.resolved_path is not None
            and repository.checkout_disposition is not None
        ):
            return pending.model_copy(
                update={
                    "state": "succeeded",
                    "message": f"The central checkout for {repository.alias} is durably ready.",
                    "fields": self._checkout_fields(repository),
                }
            )
        if check.status != "ready" or check.commit is None:
            raise ProjectProvisionRefused(
                "A repository checkout cannot run before its exact Git write proof."
            )
        machine = self._machine(request, repository.machine_alias)
        try:
            material = self.credential_manager.prepare_key(
                machine,
                repository.repository,
                space_id=request.target_space_id,
                project_id=request.proposed_project_id,
                repository_alias=repository.alias,
            )
        except GitCredentialRefused as exc:
            return self._repository_failure(
                request,
                repository_index,
                pending,
                message=str(exc),
                instruction="Repair the exact deploy-key path or SSH route, then resume.",
                phase="repository_checkout",
            )
        if (
            check.deploy_key_label != material.label
            or check.public_key_fingerprint != material.public_key_fingerprint
        ):
            return self._repository_failure(
                request,
                repository_index,
                pending,
                message=(
                    "The deploy key changed after its Git write proof. RCP invalidated that "
                    "proof before touching the checkout."
                ),
                instruction=(
                    "Resume this exact request so RCP can prove Git write access for the changed "
                    "key before retrying the checkout."
                ),
                phase="repository_checkout",
                material=material,
            )
        try:
            result = self.checkout_manager.prepare(
                machine,
                material,
                request_kind=request.kind,
                project_id=request.proposed_project_id,
                repository_alias=repository.alias,
                state_repository=request.state_repository == repository.alias,
                expected_commit=check.commit,
            )
        except ProjectCheckoutRefused as exc:
            source = self._checkout_operator_step(request, machine, exc, pending)
            return self._pause(
                request,
                self._copy_operator_contract(pending, source),
                message=source.message,
                actions=source.actions,
                fields=source.fields,
                phase="repository_checkout",
            )
        machines = list(request.machines)
        machine_index = next(
            index for index, candidate in enumerate(machines) if candidate.alias == machine.alias
        )
        machines[machine_index] = machine.model_copy(
            update={"resolved_central_root": result.central_root}
        )
        repositories = list(request.repositories)
        repositories[repository_index] = repository.model_copy(
            update={
                "resolved_path": result.repository_path,
                "checkout_disposition": result.checkout_disposition,
            }
        )
        self._transition(
            request,
            phase="repository_checkout",
            to_status="setup_in_progress",
            machines=machines,
            repositories=repositories,
        )
        updated_repository = repositories[repository_index]
        return pending.model_copy(
            update={
                "state": "succeeded",
                "message": f"The central checkout for {repository.alias} is ready.",
                "fields": self._checkout_fields(updated_repository),
            }
        )

    def _final_review(
        self,
        request: ProjectProvisioningRequestRecord,
        pending: ServerStep,
    ) -> ServerStep:
        if request.status == "ready_for_review":
            assert request.final_review_digest is not None
            return pending.model_copy(
                update={
                    "state": "succeeded",
                    "message": "The final preparation review is durably published.",
                    "fields": (
                        NonsecretField(
                            name="final_review_digest",
                            value=request.final_review_digest,
                        ),
                        NonsecretField(name="revision", value=request.revision),
                    ),
                }
            )
        repository_machines = {repository.machine_alias for repository in request.repositories}
        if (
            any(
                machine.alias in repository_machines and machine.resolved_central_root is None
                for machine in request.machines
            )
            or any(
                repository.git_check.status != "ready"
                or repository.resolved_path is None
                or repository.checkout_disposition is None
                for repository in request.repositories
            )
            or any(check.status != "ready" for check in request.provider_checks)
        ):
            return pending.model_copy(
                update={
                    "state": "failed",
                    "message": (
                        "The durable request is not fully prepared after its named steps. "
                        "Rerun the command after inspecting the server log."
                    ),
                }
            )
        updated = self._transition(
            request,
            phase="provisioning_review",
            to_status="ready_for_review",
        )
        assert updated.final_review_digest is not None
        return pending.model_copy(
            update={
                "state": "succeeded",
                "message": "The final preparation review is durably published.",
                "fields": (
                    NonsecretField(
                        name="final_review_digest",
                        value=updated.final_review_digest,
                    ),
                    NonsecretField(name="revision", value=updated.revision),
                ),
            }
        )

    def _probe_operator_step(
        self,
        request: ProjectProvisioningRequestRecord,
        machine: ProjectProvisioningMachineRecord,
        material: DeployKeyMaterial,
        probe: GitWriteProbe,
        pending: ServerStep,
    ) -> ServerStep:
        resume = self._resume_argv(request.request_id)
        if probe.status in {"github_host_trust_needed", "github_grant_needed"}:
            return deploy_key_operator_step(
                self.credential_manager,
                machine,
                material,
                number=pending.number,
                request_id=request.request_id,
                resume_argv=resume,
            )
        if probe.status == "empty_repository":
            return empty_repository_operator_step(
                material,
                number=pending.number,
                request_id=request.request_id,
                resume_argv=resume,
            )
        if probe.status == "cleanup_failed":
            return cleanup_ref_operator_step(
                material,
                probe,
                number=pending.number,
                request_id=request.request_id,
                resume_argv=resume,
            )
        instruction = (
            f"Inspect only {probe.temporary_ref!r} on {material.repository.identity}; remove it "
            "only after proving it belongs to this request, then resume."
            if probe.temporary_ref is not None
            else (
                f"Repair GitHub access or transport for exact account {machine.os_account}, "
                "then resume this request."
            )
        )
        return pending.model_copy(
            update={
                "state": "operator_action_needed",
                "performed_by": "human",
                "message": probe.diagnostic,
                "actions": (ExternalAction(instruction=instruction),),
                "fields": tuple(
                    field
                    for field in (
                        NonsecretField(name="repository", value=material.repository.identity),
                        (
                            NonsecretField(name="temporary_ref", value=probe.temporary_ref)
                            if probe.temporary_ref is not None
                            else None
                        ),
                    )
                    if field is not None
                ),
                "resume_argv": resume,
            }
        )

    def _checkout_operator_step(
        self,
        request: ProjectProvisioningRequestRecord,
        machine: ProjectProvisioningMachineRecord,
        refusal: ProjectCheckoutRefused,
        pending: ServerStep,
    ) -> ServerStep:
        resume = self._resume_argv(request.request_id)
        if refusal.kind == "retained_research":
            return retained_research_operator_step(
                machine,
                refusal,
                number=pending.number,
                request_id=request.request_id,
                resume_argv=resume,
                local_host=self.local_host,
            )
        fields = tuple(
            field
            for field in (
                (
                    NonsecretField(name="repository_path", value=refusal.repository_path)
                    if refusal.repository_path is not None
                    else None
                ),
                (
                    NonsecretField(
                        name="checkout_disposition",
                        value=refusal.checkout_disposition,
                    )
                    if refusal.checkout_disposition is not None
                    else None
                ),
            )
            if field is not None
        )
        return pending.model_copy(
            update={
                "state": "operator_action_needed",
                "performed_by": "human",
                "message": str(refusal),
                "actions": (
                    ExternalAction(
                        instruction=(
                            f"Inspect the exact checkout as {machine.os_account}; repair the "
                            "reported path, Git configuration, or access without resetting or "
                            "cleaning retained work, then resume."
                        )
                    ),
                ),
                "fields": fields,
                "resume_argv": resume,
            }
        )

    def _persist_git_pause(
        self,
        request: ProjectProvisioningRequestRecord,
        repository_index: int,
        pending: ServerStep,
        source: ServerStep,
        material: DeployKeyMaterial,
    ) -> ServerStep:
        terminal = self._copy_operator_contract(pending, source)
        repository = request.repositories[repository_index]
        paused = ProjectProvisioningGitCheckRecord(
            status="operator_action_needed",
            deploy_key_label=material.label,
            public_key_fingerprint=material.public_key_fingerprint,
            checked_at=self.store.now(),
            diagnostic=terminal.message,
        )
        repositories = list(request.repositories)
        repositories[repository_index] = repository.model_copy(update={"git_check": paused})
        self._transition(
            request,
            phase=pending.phase,
            to_status="operator_action_needed",
            repositories=repositories,
            retryable_diagnostic=terminal.message,
            operator_action=terminal,
        )
        return terminal

    def _repository_failure(
        self,
        request: ProjectProvisioningRequestRecord,
        repository_index: int,
        pending: ServerStep,
        *,
        message: str,
        instruction: str,
        phase: str,
        material: DeployKeyMaterial | None = None,
    ) -> ServerStep:
        terminal = pending.model_copy(
            update={
                "state": "operator_action_needed",
                "performed_by": "human",
                "message": message,
                "actions": (ExternalAction(instruction=instruction),),
                "fields": (() if material is None else self._key_fields(material)),
                "resume_argv": self._resume_argv(request.request_id),
            }
        )
        repository = request.repositories[repository_index]
        check = repository.git_check
        paused = ProjectProvisioningGitCheckRecord(
            status="operator_action_needed",
            deploy_key_label=(material.label if material is not None else check.deploy_key_label),
            public_key_fingerprint=(
                material.public_key_fingerprint
                if material is not None
                else check.public_key_fingerprint
            ),
            checked_at=self.store.now(),
            diagnostic=message,
        )
        repositories = list(request.repositories)
        repositories[repository_index] = repository.model_copy(update={"git_check": paused})
        self._transition(
            request,
            phase=phase,
            to_status="operator_action_needed",
            repositories=repositories,
            retryable_diagnostic=message,
            operator_action=terminal,
        )
        return terminal

    def _pause(
        self,
        request: ProjectProvisioningRequestRecord,
        pending: ServerStep,
        *,
        message: str,
        actions: tuple[ExternalAction, ...] = (),
        fields: tuple[NonsecretField, ...] = (),
        phase: str,
    ) -> ServerStep:
        terminal = pending.model_copy(
            update={
                "state": "operator_action_needed",
                "performed_by": "human",
                "message": message,
                "actions": actions,
                "fields": fields,
                "resume_argv": self._resume_argv(request.request_id),
            }
        )
        self._transition(
            request,
            phase=phase,
            to_status="operator_action_needed",
            retryable_diagnostic=message,
            operator_action=terminal,
        )
        return terminal

    def _transition(
        self,
        request: ProjectProvisioningRequestRecord,
        *,
        phase: str,
        to_status: Literal[
            "setup_in_progress",
            "operator_action_needed",
            "ready_for_review",
        ],
        machines: list[ProjectProvisioningMachineRecord] | None = None,
        repositories: list[ProjectProvisioningRepositoryRecord] | None = None,
        retryable_diagnostic: str | None = None,
        operator_action: ServerStep | None = None,
    ) -> ProjectProvisioningRequestRecord:
        return self.store.transition_project_provisioning_request(
            request.request_id,
            receipt_id=f"{phase}-r{request.revision}",
            phase=phase,
            expected_revision=request.revision,
            expected_status=request.status,
            to_status=to_status,
            machines=machines or request.machines,
            repositories=repositories or request.repositories,
            provider_checks=request.provider_checks,
            retryable_diagnostic=retryable_diagnostic,
            operator_action=operator_action,
            cancellation_disposition=None,
        )

    def _machine(
        self,
        request: ProjectProvisioningRequestRecord,
        alias: str,
    ) -> ProjectProvisioningMachineRecord:
        return next(machine for machine in request.machines if machine.alias == alias)

    def _machine_target(self, machine: ProjectProvisioningMachineRecord) -> MachineTarget:
        return MachineTarget(
            host=machine.host or self.local_host,
            os_account=machine.os_account,
        )

    def _machine_step(
        self,
        number: int,
        *,
        title: str,
        purpose: str,
        phase: str,
        expected: str,
        message: str,
        performed_by: Literal["system", "human"] = "system",
    ) -> ServerStep:
        return ServerStep(
            number=number,
            title=title,
            purpose=purpose,
            performed_by=performed_by,
            target=MachineTarget(host=self.local_host, os_account=self.layout.service_account),
            phase=phase,
            state="pending",
            expected_success=expected,
            message=message,
        )

    def _resume_argv(self, request_id: str) -> tuple[str, ...]:
        return (
            "sudo",
            "-n",
            "-u",
            self.layout.service_account,
            "-H",
            str(self.layout.cli_wrapper),
            "server",
            "project",
            "provision",
            request_id,
        )

    def _identity_fields(self) -> dict[str, object]:
        return {
            "instance_id": self.metadata.instance_id,
            "pid": self.metadata.pid,
            "data_dir_id": self.metadata.data_dir_id,
            "space_id": self.store.space_id,
        }

    def _pending_git_check(self, material: DeployKeyMaterial) -> ProjectProvisioningGitCheckRecord:
        return ProjectProvisioningGitCheckRecord(
            status="pending",
            deploy_key_label=material.label,
            public_key_fingerprint=material.public_key_fingerprint,
            checked_at=self.store.now(),
        )

    @staticmethod
    def _copy_operator_contract(pending: ServerStep, source: ServerStep) -> ServerStep:
        if (
            source.state != "operator_action_needed"
            or source.performed_by != "human"
            or source.target != pending.target
        ):
            raise ValueError("project operator action must retain the planned typed target")
        return pending.model_copy(
            update={
                "state": "operator_action_needed",
                "performed_by": source.performed_by,
                "message": source.message,
                "actions": source.actions,
                "fields": source.fields,
                "resume_argv": source.resume_argv,
            }
        )

    @staticmethod
    def _key_fields(material: DeployKeyMaterial) -> tuple[NonsecretField, ...]:
        return (
            NonsecretField(name="repository", value=material.repository.identity),
            NonsecretField(name="deploy_key_label", value=material.label),
            NonsecretField(
                name="public_key_fingerprint",
                value=material.public_key_fingerprint,
            ),
        )

    @staticmethod
    def _git_fields(
        repository: ProjectProvisioningRepositoryRecord,
        check: ProjectProvisioningGitCheckRecord,
    ) -> tuple[NonsecretField, ...]:
        assert check.commit is not None and check.public_key_fingerprint is not None
        return (
            NonsecretField(name="repository", value=repository.repository.identity),
            NonsecretField(name="git_commit", value=check.commit),
            NonsecretField(
                name="public_key_fingerprint",
                value=check.public_key_fingerprint,
            ),
        )

    @staticmethod
    def _checkout_fields(
        repository: ProjectProvisioningRepositoryRecord,
    ) -> tuple[NonsecretField, ...]:
        assert repository.resolved_path is not None
        assert repository.checkout_disposition is not None
        return (
            NonsecretField(name="repository_path", value=repository.resolved_path),
            NonsecretField(
                name="checkout_disposition",
                value=repository.checkout_disposition,
            ),
        )

    @staticmethod
    def _provider_fields(
        check: ProjectProvisioningProviderCheckRecord,
    ) -> tuple[NonsecretField, ...]:
        assert check.execution_account is not None
        assert check.binary_path is not None
        assert check.version is not None
        assert check.resolved_runtime_id is not None
        return (
            NonsecretField(name="provider_profile", value=check.profile),
            NonsecretField(name="provider", value=check.provider),
            NonsecretField(name="execution_account", value=check.execution_account),
            NonsecretField(name="binary_path", value=check.binary_path),
            NonsecretField(name="provider_version", value=check.version),
            NonsecretField(name="runtime_id", value=check.resolved_runtime_id),
        )


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "ProjectProvisionCoordinator",
    "ProjectProvisionRefused",
    "prepare_project_provision_command",
]
