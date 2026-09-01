"""Exact-account provider checks for installed team-server operations."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import socket
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import BinaryIO, Literal, Protocol

from rcp.agents.launcher import AgentLauncher, ProviderReadiness
from rcp.config import AGENT_EXECUTION_PROFILES, AgentExecutionProfile, load_manifest
from rcp.providers import AgentCapability, ProviderId, profile_for
from rcp.server_ops.cli import CallerIdentity, PreparedServerCommand, ServerEventEmitter
from rcp.server_ops.control import (
    ServerControlClient,
    ServerControlError,
    ServerControlProviderCheckResult,
    ServerControlProviderPlanResult,
    ServerControlProviderTarget,
)
from rcp.server_ops.layout import DEFAULT_SERVER_LAYOUT, ServerLayout
from rcp.server_ops.models import (
    CommandAction,
    ExternalAction,
    MachineTarget,
    NonsecretField,
    ServerCommandRequest,
    ServerPlanEvent,
    ServerStep,
    redact_server_text,
)
from rcp.server_runtime import ServerMetadata, data_dir_identity
from rcp.storage import (
    AppStore,
    ProjectProvisioningMachineRecord,
    ProjectProvisioningProviderCheckRecord,
    ProjectProvisioningRequestRecord,
)

ProviderSelectorKind = Literal["request", "project"]
_PROFILE_CAPABILITY: dict[AgentExecutionProfile, AgentCapability] = {
    "seed": "scratch_patch",
    "refresh": "scratch_patch",
    "node_chat": "discuss",
    "project_chat": "discuss",
    "paper_coach": "paper_readonly",
    "orchestrator": "orchestrate",
}


class ProviderReadinessRefused(ServerControlError):
    """The selected durable provider boundary cannot be checked safely."""

    def __init__(self, message: str) -> None:
        super().__init__("operation_refused", message)


class ProviderReadinessControl(Protocol):
    def provider_readiness_plan(
        self,
        *,
        selector_kind: Literal["request", "project"],
        selector_id: str,
    ) -> ServerControlProviderPlanResult: ...

    def check_provider_readiness(
        self,
        *,
        selector_kind: Literal["request", "project"],
        selector_id: str,
        boundary_sha256: str,
        target_id: str,
    ) -> ServerControlProviderCheckResult: ...


ControlFactory = Callable[[ServerLayout], ProviderReadinessControl]


def prepare_provider_check_command(
    request: ServerCommandRequest,
    identity: CallerIdentity,
    *,
    control_factory: ControlFactory | None = None,
    layout: ServerLayout = DEFAULT_SERVER_LAYOUT,
) -> PreparedServerCommand:
    """Prepare one complete CLI plan before any provider subprocess is touched."""

    if request.command != "server provider check":
        raise ValueError("prepare_provider_check_command requires one provider check")
    selector_kind: ProviderSelectorKind
    selector_id: str
    if request.request_id is not None:
        selector_kind, selector_id = "request", request.request_id
    else:
        assert request.project_id is not None
        selector_kind, selector_id = "project", request.project_id
    client = (control_factory or _installed_control)(layout)
    resolved = client.provider_readiness_plan(
        selector_kind=selector_kind,
        selector_id=selector_id,
    )
    plan = ServerPlanEvent(
        command=request.command,
        timestamp=datetime.now(UTC),
        steps=tuple(target.step for target in resolved.targets),
    )

    def execute(emitter: ServerEventEmitter, _input_stream: BinaryIO) -> None:
        boundary = resolved.boundary_sha256
        for target in resolved.targets:
            pending = target.step
            emitter.emit_step(
                pending.model_copy(
                    update={
                        "state": "running",
                        "message": (
                            "RCP is checking the saved provider configuration on this exact "
                            "machine account."
                        ),
                    }
                )
            )
            try:
                checked = client.check_provider_readiness(
                    selector_kind=selector_kind,
                    selector_id=selector_id,
                    boundary_sha256=boundary,
                    target_id=target.target_id,
                )
            except ServerControlError as exc:
                emitter.emit_step(
                    pending.model_copy(
                        update={
                            "state": "failed",
                            "message": str(exc),
                        }
                    )
                )
                return
            if checked.target_id != target.target_id:
                raise ProviderReadinessRefused("The control result names another provider target.")
            emitter.emit_step(checked.step)
            if checked.step.state != "succeeded":
                return
            boundary = checked.next_boundary_sha256

    return PreparedServerCommand(plan=plan, execute=execute)


def _installed_control(layout: ServerLayout) -> ProviderReadinessControl:
    return ServerControlClient.from_data_dir(
        layout.data_dir,
        expected_server_uid=os.geteuid(),
    )


@dataclass(frozen=True)
class _ProviderTarget:
    target_id: str
    number: int
    profile: AgentExecutionProfile
    provider: ProviderId
    runtime_id: str
    model: str
    reasoning: str
    machine_alias: str
    location: Literal["local", "ssh"]
    host: str
    os_account: str
    binary_path: str | None
    provider_index: int | None


@dataclass(frozen=True)
class _ProviderBoundary:
    selector_kind: ProviderSelectorKind
    selector_id: str
    boundary_sha256: str
    targets: tuple[_ProviderTarget, ...]
    request: ProjectProvisioningRequestRecord | None


@dataclass(frozen=True)
class _ReadinessProblem:
    kind: Literal["install", "login", "configuration", "transport"]
    message: str


class ProviderReadinessCoordinator:
    """Resolve, probe, and durably publish only saved provider targets."""

    def __init__(
        self,
        store: AppStore,
        launcher: AgentLauncher,
        metadata: ServerMetadata,
        *,
        layout: ServerLayout = DEFAULT_SERVER_LAYOUT,
        local_host: str | None = None,
    ) -> None:
        if store.space_kind != "team":
            raise ValueError("provider readiness requires an installed team space")
        if (
            metadata.owner_kind != "cli"
            or metadata.control_socket is None
            or metadata.data_dir_id != data_dir_identity(store.path.parent)
        ):
            raise ValueError(
                "provider readiness requires this exact CLI-owned team service process"
            )
        self.store = store
        self.launcher = launcher
        self.metadata = metadata
        self.layout = layout
        self.local_host = local_host or socket.gethostname()

    def plan(
        self,
        selector_kind: ProviderSelectorKind,
        selector_id: str,
    ) -> ServerControlProviderPlanResult:
        boundary = self._resolve(selector_kind, selector_id)
        return ServerControlProviderPlanResult(
            **self._identity_fields(),
            selector_kind=selector_kind,
            selector_id=selector_id,
            boundary_sha256=boundary.boundary_sha256,
            targets=tuple(
                ServerControlProviderTarget(
                    target_id=target.target_id,
                    step=self._pending_step(target),
                )
                for target in boundary.targets
            ),
        )

    def check(
        self,
        selector_kind: ProviderSelectorKind,
        selector_id: str,
        *,
        boundary_sha256: str,
        target_id: str,
    ) -> ServerControlProviderCheckResult:
        boundary = self._resolve(selector_kind, selector_id)
        if boundary.boundary_sha256 != boundary_sha256:
            raise ProviderReadinessRefused(
                "The provider configuration changed after the plan was shown; rerun the command."
            )
        target = next(
            (candidate for candidate in boundary.targets if candidate.target_id == target_id),
            None,
        )
        if target is None:
            raise ProviderReadinessRefused(
                "The provider target changed after the plan was shown; rerun the command."
            )
        step, ready_proof = self._check_target(target, selector_kind, selector_id)
        if boundary.request is not None:
            self._persist_request_result(boundary.request, target, step, ready_proof)
        next_boundary = self._resolve(selector_kind, selector_id)
        return ServerControlProviderCheckResult(
            **self._identity_fields(),
            selector_kind=selector_kind,
            selector_id=selector_id,
            target_id=target_id,
            boundary_sha256=boundary_sha256,
            next_boundary_sha256=next_boundary.boundary_sha256,
            step=step,
        )

    def check_for_project_provision(
        self,
        request_id: str,
        *,
        target_id: str,
        pending: ServerStep,
    ) -> ServerStep:
        """Run one request target with the unified project-provision contract."""

        boundary = self._resolve_request(request_id)
        target = next(
            (candidate for candidate in boundary.targets if candidate.target_id == target_id),
            None,
        )
        if target is None:
            raise ProviderReadinessRefused(
                "The provider target changed after the project plan was shown; rerun the command."
            )
        if pending.state != "pending" or pending.phase != "provider_readiness":
            raise ValueError("project provisioning supplied an invalid provider step")
        step, ready_proof = self._check_target(
            target,
            "request",
            request_id,
            pending=pending,
            resume_argv=self._project_resume_argv(request_id),
        )
        assert boundary.request is not None
        self._persist_request_result(boundary.request, target, step, ready_proof)
        return step

    def _resolve(
        self,
        selector_kind: ProviderSelectorKind,
        selector_id: str,
    ) -> _ProviderBoundary:
        if selector_kind == "request":
            return self._resolve_request(selector_id)
        return self._resolve_project(selector_id)

    def _resolve_request(self, request_id: str) -> _ProviderBoundary:
        request = self.store.project_provisioning_request(request_id)
        if request is None or request.target_space_id != self.store.space_id:
            raise ProviderReadinessRefused("The selected provisioning request does not exist.")
        if request.status in {"completed", "cancelled"}:
            raise ProviderReadinessRefused(
                f"The selected provisioning request is already {request.status}."
            )
        machines = {machine.alias: machine for machine in request.machines}
        targets = tuple(
            self._request_target(number, index, check, machines[check.machine_alias])
            for number, (index, check) in enumerate(enumerate(request.provider_checks), start=1)
        )
        return _ProviderBoundary(
            selector_kind="request",
            selector_id=request_id,
            boundary_sha256=_boundary_digest(
                "request",
                request_id,
                targets,
                revision=request.revision,
                status=request.status,
            ),
            targets=targets,
            request=request,
        )

    def _request_target(
        self,
        number: int,
        index: int,
        check: ProjectProvisioningProviderCheckRecord,
        machine: ProjectProvisioningMachineRecord,
    ) -> _ProviderTarget:
        return _target(
            number=number,
            profile=check.profile,
            provider=check.provider,
            runtime_id=check.runtime_id,
            model=check.model,
            reasoning=check.reasoning,
            machine_alias=check.machine_alias,
            location=machine.location,
            host=machine.host,
            os_account=machine.os_account,
            binary_path=check.binary_path,
            provider_index=index,
        )

    def _resolve_project(self, project_id: str) -> _ProviderBoundary:
        record = self.store.project(project_id)
        if (
            record is None
            or record.home_space_id != self.store.space_id
            or self.store.space_kind != "team"
        ):
            raise ProviderReadinessRefused("The selected team project does not exist.")
        try:
            manifest = load_manifest(record.locator)
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise ProviderReadinessRefused(
                "The selected project manifest is unavailable or invalid."
            ) from exc
        targets: list[_ProviderTarget] = []
        for number, profile_name in enumerate(AGENT_EXECUTION_PROFILES, start=1):
            configured = manifest.agent_profile(profile_name)
            machine = manifest.machine_map[configured.run_on]
            os_account = machine.os_account or (
                self.layout.service_account if not machine.host else ""
            )
            if not os_account:
                raise ProviderReadinessRefused(
                    f"Project machine {machine.alias!r} has no recorded remote execution account."
                )
            targets.append(
                _target(
                    number=number,
                    profile=profile_name,
                    provider=configured.provider,
                    runtime_id=configured.runtime,
                    model=configured.model,
                    reasoning=configured.reasoning,
                    machine_alias=machine.alias,
                    location="ssh" if machine.host else "local",
                    host=machine.host,
                    os_account=os_account,
                    binary_path=machine.provider_paths.get(configured.provider),
                    provider_index=None,
                )
            )
        resolved = tuple(targets)
        return _ProviderBoundary(
            selector_kind="project",
            selector_id=project_id,
            boundary_sha256=_boundary_digest("project", project_id, resolved),
            targets=resolved,
            request=None,
        )

    def _pending_step(self, target: _ProviderTarget) -> ServerStep:
        label = profile_for(target.provider).label
        return ServerStep(
            number=target.number,
            title=f"Verify {label} for {target.profile}",
            purpose=(
                "Check the saved provider profile on its exact operating-system account and "
                "ask a human to act only when native readiness is missing."
            ),
            performed_by="human",
            target=self._machine_target(target),
            phase="provider_readiness",
            state="pending",
            expected_success=(
                "The executable, version, native authentication, runtime, model, and execution "
                "account all match the saved profile."
            ),
            message=f"RCP will verify the saved {target.profile} provider profile.",
        )

    def _check_target(
        self,
        target: _ProviderTarget,
        selector_kind: ProviderSelectorKind,
        selector_id: str,
        *,
        pending: ServerStep | None = None,
        resume_argv: tuple[str, ...] | None = None,
    ) -> tuple[ServerStep, dict[str, str] | None]:
        pending = pending or self._pending_step(target)
        account = self.launcher.execution_account(host=target.host)
        if not account.reachable or account.os_account is None:
            return (
                self._operator_step(
                    pending,
                    target,
                    selector_kind,
                    selector_id,
                    account.reason or "The configured execution account is unavailable.",
                    actions=self._transport_actions(target),
                    resume_argv=resume_argv,
                ),
                None,
            )
        if account.os_account != target.os_account:
            return (
                self._operator_step(
                    pending,
                    target,
                    selector_kind,
                    selector_id,
                    (
                        f"The saved target requires OS account {target.os_account}, but the "
                        f"configured route reached {account.os_account}. RCP did not probe the "
                        "provider under the wrong account."
                    ),
                    actions=self._transport_actions(target),
                    resume_argv=resume_argv,
                ),
                None,
            )
        readiness = self.launcher.readiness(
            target.provider,
            host=target.host,
            binary=target.binary_path,
            refresh=True,
        )
        problem = self._readiness_problem(target, readiness)
        if problem is not None:
            actions = self._provider_actions(target, readiness, problem.kind)
            return (
                self._operator_step(
                    pending,
                    target,
                    selector_kind,
                    selector_id,
                    problem.message,
                    actions=actions,
                    resume_argv=resume_argv,
                ),
                None,
            )
        assert readiness.binary_path is not None
        assert readiness.version is not None
        resolved_runtime = _resolved_runtime_id(target.provider, target.runtime_id)
        proof = {
            "binary_path": readiness.binary_path,
            "version": readiness.version,
            "resolved_runtime_id": resolved_runtime,
            "execution_account": account.os_account,
        }
        return (
            pending.model_copy(
                update={
                    "state": "succeeded",
                    "message": (
                        f"{profile_for(target.provider).label} is ready for {target.profile} "
                        f"as {account.os_account}."
                    ),
                    "fields": self._proof_fields(target, proof),
                }
            ),
            proof,
        )

    def _readiness_problem(
        self,
        target: _ProviderTarget,
        readiness: ProviderReadiness,
    ) -> _ReadinessProblem | None:
        label = profile_for(target.provider).label
        if readiness.path_state == "unreachable":
            return _ReadinessProblem(
                "transport",
                readiness.reason or f"The configured route to {label} became unavailable.",
            )
        if not readiness.installed or readiness.binary_path is None:
            if target.binary_path is not None:
                return _ReadinessProblem(
                    "configuration",
                    readiness.reason or f"The saved {label} executable is unavailable.",
                )
            return _ReadinessProblem(
                "install",
                readiness.reason or f"{label} is not installed on the selected account.",
            )
        if readiness.path_state not in {"resolved", "unconfigured"}:
            return _ReadinessProblem(
                "install",
                readiness.reason or f"The saved {label} executable is unavailable.",
            )
        if not readiness.version or not _safe_version(readiness.version):
            return _ReadinessProblem(
                "install",
                f"{label} did not report one bounded nonsecret version.",
            )
        try:
            profile_for(target.provider).validate_readiness_version(
                readiness.version,
                capability=_PROFILE_CAPABILITY[target.profile],
            )
        except ValueError as exc:
            return _ReadinessProblem("configuration", str(exc))
        if not readiness.authenticated:
            return _ReadinessProblem(
                "login",
                readiness.reason or f"{label} is not authenticated on the selected account.",
            )
        try:
            _resolved_runtime_id(target.provider, target.runtime_id)
        except ValueError:
            return _ReadinessProblem(
                "configuration",
                f"The saved runtime {target.runtime_id!r} is not supported by this {label} build.",
            )
        if target.model:
            if not readiness.models:
                return _ReadinessProblem(
                    "configuration",
                    f"{label} did not return a model catalog, so RCP could not verify the "
                    f"saved model {target.model!r}.",
                )
            selected = next((model for model in readiness.models if model.id == target.model), None)
            if selected is None and readiness.models:
                return _ReadinessProblem(
                    "configuration",
                    f"The saved model {target.model!r} is not offered by this {label} build.",
                )
            if (
                selected is not None
                and selected.reasoning
                and target.reasoning not in selected.reasoning
            ):
                return _ReadinessProblem(
                    "configuration",
                    f"The saved reasoning effort {target.reasoning!r} is not offered by model "
                    f"{target.model!r}.",
                )
        elif readiness.models and not any(
            not model.reasoning or target.reasoning in model.reasoning for model in readiness.models
        ):
            return _ReadinessProblem(
                "configuration",
                f"The saved reasoning effort {target.reasoning!r} is not offered for the "
                f"{label} provider default.",
            )
        return None

    def _provider_actions(
        self,
        target: _ProviderTarget,
        readiness: ProviderReadiness,
        problem_kind: Literal["install", "login", "configuration", "transport"],
    ) -> tuple[CommandAction | ExternalAction, ...]:
        label = profile_for(target.provider).label
        if problem_kind == "transport":
            return self._transport_actions(target)
        if problem_kind == "login" and readiness.binary_path:
            login = tuple(profile_for(target.provider).login_command(readiness.binary_path))
            command = (
                (
                    "sudo",
                    "-u",
                    self.layout.service_account,
                    "-H",
                    "ssh",
                    "-t",
                    target.host,
                    shlex.join(login),
                )
                if target.host
                else (
                    "sudo",
                    "-u",
                    self.layout.service_account,
                    "-H",
                    *login,
                )
            )
            return (
                CommandAction(argv=command),
                ExternalAction(
                    instruction=(
                        f"Complete {label}'s native login directly as OS account "
                        f"{target.os_account}; do not paste provider credentials into RCP."
                    )
                ),
            )
        if problem_kind == "configuration":
            return (
                ExternalAction(
                    instruction=(
                        f"Correct the saved {target.profile} provider profile through the "
                        "project setup or settings flow; the server CLI will not substitute a "
                        "runtime, model, or reasoning value."
                    )
                ),
            )
        return (
            ExternalAction(
                instruction=(
                    f"Install {label} for OS account {target.os_account} through the provider's "
                    "ordinary installation workflow, then rerun the exact check command."
                )
            ),
        )

    def _transport_actions(
        self,
        target: _ProviderTarget,
    ) -> tuple[CommandAction | ExternalAction, ...]:
        if target.host:
            return (
                ExternalAction(
                    instruction=(
                        f"Repair the rcp service account's existing OpenSSH route to "
                        f"{target.host} so it reaches only OS account {target.os_account}."
                    )
                ),
                CommandAction(
                    argv=(
                        "sudo",
                        "-u",
                        self.layout.service_account,
                        "-H",
                        "ssh",
                        target.host,
                        "id -un",
                    )
                ),
            )
        return (
            ExternalAction(
                instruction=(
                    f"Run the provider check from the installed service as OS account "
                    f"{target.os_account}."
                )
            ),
        )

    def _operator_step(
        self,
        pending: ServerStep,
        target: _ProviderTarget,
        selector_kind: ProviderSelectorKind,
        selector_id: str,
        message: str,
        *,
        actions: tuple[CommandAction | ExternalAction, ...],
        resume_argv: tuple[str, ...] | None = None,
    ) -> ServerStep:
        return pending.model_copy(
            update={
                "state": "operator_action_needed",
                "performed_by": "human",
                "message": message,
                "actions": actions,
                "fields": (
                    NonsecretField(name="provider_profile", value=target.profile),
                    NonsecretField(name="provider", value=target.provider),
                    NonsecretField(name="machine", value=target.machine_alias),
                    NonsecretField(name="execution_account", value=target.os_account),
                ),
                "resume_argv": resume_argv or self._resume_argv(selector_kind, selector_id),
            }
        )

    def _persist_request_result(
        self,
        request: ProjectProvisioningRequestRecord,
        target: _ProviderTarget,
        step: ServerStep,
        ready_proof: dict[str, str] | None,
    ) -> None:
        assert target.provider_index is not None
        current = request.provider_checks[target.provider_index]
        values = current.model_dump(mode="json")
        values.update(
            {
                "status": "ready" if ready_proof is not None else "operator_action_needed",
                "binary_path": None,
                "version": None,
                "resolved_runtime_id": None,
                "execution_account": None,
                "checked_at": self.store.now(),
                "diagnostic": None if ready_proof is not None else step.message,
            }
        )
        if ready_proof is not None:
            values.update(ready_proof)
        updated_check = ProjectProvisioningProviderCheckRecord.model_validate(values)
        checks = list(request.provider_checks)
        checks[target.provider_index] = updated_check

        unrelated_action = request.operator_action is not None and not _action_matches_target(
            request.operator_action,
            target,
        )
        if unrelated_action:
            to_status = "operator_action_needed"
            operator_action = request.operator_action
            diagnostic = request.retryable_diagnostic
        elif ready_proof is None:
            to_status = "operator_action_needed"
            operator_action = step
            diagnostic = step.message
        else:
            to_status = "setup_in_progress"
            operator_action = None
            diagnostic = None
        self.store.transition_project_provisioning_request(
            request.request_id,
            receipt_id=f"provider-{target.target_id[:24]}-r{request.revision}",
            phase="provider_readiness",
            expected_revision=request.revision,
            expected_status=request.status,
            to_status=to_status,
            machines=request.machines,
            repositories=request.repositories,
            provider_checks=checks,
            retryable_diagnostic=diagnostic,
            operator_action=operator_action,
            cancellation_disposition=request.cancellation_disposition,
        )

    def _machine_target(self, target: _ProviderTarget) -> MachineTarget:
        return MachineTarget(
            host=target.host or self.local_host,
            os_account=target.os_account,
        )

    def _proof_fields(
        self,
        target: _ProviderTarget,
        proof: dict[str, str],
    ) -> tuple[NonsecretField, ...]:
        return (
            NonsecretField(name="provider_profile", value=target.profile),
            NonsecretField(name="provider", value=target.provider),
            NonsecretField(name="machine", value=target.machine_alias),
            NonsecretField(name="execution_account", value=proof["execution_account"]),
            NonsecretField(name="binary_path", value=proof["binary_path"]),
            NonsecretField(name="provider_version", value=proof["version"]),
            NonsecretField(name="runtime_id", value=proof["resolved_runtime_id"]),
            NonsecretField(name="model", value=target.model or "provider default"),
            NonsecretField(name="reasoning", value=target.reasoning),
        )

    def _resume_argv(
        self,
        selector_kind: ProviderSelectorKind,
        selector_id: str,
    ) -> tuple[str, ...]:
        return (
            "sudo",
            "-n",
            "-u",
            self.layout.service_account,
            "-H",
            str(self.layout.cli_wrapper),
            "server",
            "provider",
            "check",
            f"--{selector_kind}",
            selector_id,
        )

    def _project_resume_argv(self, request_id: str) -> tuple[str, ...]:
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


def _target(
    *,
    number: int,
    profile: AgentExecutionProfile,
    provider: ProviderId,
    runtime_id: str,
    model: str,
    reasoning: str,
    machine_alias: str,
    location: Literal["local", "ssh"],
    host: str,
    os_account: str,
    binary_path: str | None,
    provider_index: int | None,
) -> _ProviderTarget:
    identity = {
        "profile": profile,
        "provider": provider,
        "runtime_id": runtime_id,
        "model": model,
        "reasoning": reasoning,
        "machine_alias": machine_alias,
        "location": location,
        "host": host,
        "os_account": os_account,
        "binary_path": binary_path,
    }
    return _ProviderTarget(
        target_id=hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest(),
        number=number,
        profile=profile,
        provider=provider,
        runtime_id=runtime_id,
        model=model,
        reasoning=reasoning,
        machine_alias=machine_alias,
        location=location,
        host=host,
        os_account=os_account,
        binary_path=binary_path,
        provider_index=provider_index,
    )


def _boundary_digest(
    selector_kind: ProviderSelectorKind,
    selector_id: str,
    targets: tuple[_ProviderTarget, ...],
    *,
    revision: int | None = None,
    status: str | None = None,
) -> str:
    payload = {
        "selector_kind": selector_kind,
        "selector_id": selector_id,
        "revision": revision,
        "status": status,
        "targets": [
            {
                "target_id": target.target_id,
                "number": target.number,
                "provider_index": target.provider_index,
            }
            for target in targets
        ],
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _resolved_runtime_id(provider: ProviderId, value: str) -> str:
    prefix = f"{provider}:"
    configured = value[len(prefix) :] if value.startswith(prefix) else value
    return profile_for(provider).configured_runtime_id(configured)


def _safe_version(value: str) -> bool:
    return bool(
        value
        and value == value.strip()
        and len(value) <= 240
        and not any(ord(character) < 32 or ord(character) == 127 for character in value)
        and redact_server_text(value) == value
    )


def _action_matches_target(step: ServerStep, target: _ProviderTarget) -> bool:
    if step.phase != "provider_readiness":
        return False
    fields = {field.name: field.value for field in step.fields}
    return fields == {
        "provider_profile": target.profile,
        "provider": target.provider,
        "machine": target.machine_alias,
        "execution_account": target.os_account,
    }


__all__ = ["ProviderReadinessCoordinator", "ProviderReadinessRefused"]
