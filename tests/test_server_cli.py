from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from io import BytesIO, StringIO

import pytest
from pydantic import TypeAdapter, ValidationError

from rcp.__main__ import build_parser, main
from rcp.server_ops.cli import (
    SERVER_CLI_EXIT_FAILED,
    SERVER_CLI_EXIT_OPERATOR_ACTION,
    SERVER_CLI_EXIT_WRONG_IDENTITY,
    SERVER_CLI_TERMINAL_RESERVE_BYTES,
    CallerIdentity,
    PreparedServerCommand,
    ServerEventEmitter,
    render_server_execution,
    request_from_namespace,
    run_server_command,
)
from rcp.server_ops.models import (
    SERVER_CLI_MAX_EXECUTION_BYTES,
    SERVER_CLI_MAX_STEPS,
    CommandAction,
    ExternalAction,
    ExternalServiceTarget,
    MachineTarget,
    NonsecretField,
    ServerCommandEvent,
    ServerCommandExecution,
    ServerCommandRequest,
    ServerPlanEvent,
    ServerStep,
    ServerStepEvent,
    server_event_stream_size,
)

REQUEST_ID = "123e4567-e89b-42d3-a456-426614174000"
PROJECT_ID = "123e4567-e89b-42d3-b456-426614174001"
MEMBER_ID = "123e4567-e89b-42d3-8456-426614174002"
UPDATE_COMMIT = "a" * 40
NOW = datetime(2026, 8, 28, 12, 0, tzinfo=UTC)
AGE_RECIPIENT = "age1qypqxpq9qcrsszg2pvxq6rs0zqg3yyc5z5tpwxqergd3c8g7rusqmwn7f2"
BACKUP_CONFIGURE_ARGV = (
    "server",
    "backup",
    "configure",
    "--destination",
    "/srv/rcp-backups",
    "--recipient",
    AGE_RECIPIENT,
    "--confirm",
)


def _parse(*argv: str):
    return build_parser().parse_args(argv)


def _machine_step(
    command: str,
    *,
    state: str,
    identity: CallerIdentity | None = None,
) -> ServerStep:
    identity = identity or CallerIdentity(uid=501, username="rcp", host="lab.example")
    return ServerStep(
        number=1,
        title="Read server health",
        purpose="Read the concrete server operation without changing state.",
        performed_by="system",
        target=MachineTarget(host=identity.host, os_account=identity.username),
        phase="health_readback",
        state=state,
        expected_success=f"{command} publishes one verified health result.",
        message=f"{command} is {state}.",
    )


def _successful_command(
    request: ServerCommandRequest,
    identity: CallerIdentity,
) -> PreparedServerCommand:
    pending = _machine_step(request.command, state="pending", identity=identity)
    running = _machine_step(request.command, state="running", identity=identity)
    succeeded = _machine_step(request.command, state="succeeded", identity=identity)

    def execute(emitter, _input_stream) -> None:
        emitter.emit_step(running, timestamp=NOW)
        emitter.emit_step(succeeded, timestamp=NOW)

    return PreparedServerCommand(
        plan=ServerPlanEvent(command=request.command, timestamp=NOW, steps=(pending,)),
        execute=execute,
    )


def _operator_execution() -> ServerCommandExecution:
    target = ExternalServiceTarget(
        service="github.com",
        resource="openai/rcp",
        destination_url="https://github.com/openai/rcp/settings/keys",
        required_authority_role="repository administrator",
    )
    common = {
        "number": 1,
        "title": "Grant repository write access",
        "purpose": "Install the central checkout's public deploy key.",
        "performed_by": "human",
        "target": target,
        "phase": "git_write_grant",
        "expected_success": "The request-scoped Git write probe succeeds.",
    }
    pending = ServerStep(
        **common,
        state="pending",
        message="RCP will wait for the repository write grant.",
    )
    paused = ServerStep(
        **common,
        state="operator_action_needed",
        message="GitHub has not granted write access to this deploy key.",
        actions=(
            ExternalAction(
                instruction="Add the displayed public key and enable Allow write access."
            ),
        ),
        fields=(NonsecretField(name="public_key", value="ssh-ed25519 AAAAC3 public@example"),),
        resume_argv=("rcp", "server", "project", "provision", REQUEST_ID),
    )
    return ServerCommandExecution(
        events=(
            ServerPlanEvent(
                command="server project provision",
                timestamp=NOW,
                steps=(pending,),
            ),
            ServerStepEvent(
                command="server project provision",
                timestamp=NOW,
                step=paused,
            ),
        ),
        exit_code=SERVER_CLI_EXIT_OPERATOR_ACTION,
    )


@pytest.mark.parametrize(
    ("argv", "command", "fields"),
    [
        (
            ("server", "install", "--team-name", "Upgrade Fixture Lab"),
            "server install",
            {"team_name": "Upgrade Fixture Lab"},
        ),
        (("server", "doctor"), "server doctor", {}),
        (
            ("server", "provider", "check", "--request", REQUEST_ID),
            "server provider check",
            {"request_id": REQUEST_ID},
        ),
        (
            ("server", "provider", "check", "--project", PROJECT_ID),
            "server provider check",
            {"project_id": PROJECT_ID},
        ),
        (
            ("server", "project", "provision", REQUEST_ID),
            "server project provision",
            {"request_id": REQUEST_ID},
        ),
        (
            ("server", "project", "transfer-import", REQUEST_ID),
            "server project transfer-import",
            {"request_id": REQUEST_ID},
        ),
        (
            BACKUP_CONFIGURE_ARGV,
            "server backup configure",
            {
                "backup_destination": "/srv/rcp-backups",
                "backup_schedule": "02:00",
                "backup_retention": 30,
                "backup_age_recipient": AGE_RECIPIENT,
                "backup_confirmed": True,
            },
        ),
        (("server", "backup", "run"), "server backup run", {}),
        (
            ("server", "restore", "/backups/lab.age", "--identity-file", "/safe/age.key"),
            "server restore",
            {"archive_path": "/backups/lab.age", "recovery_identity_file": "/safe/age.key"},
        ),
        (
            (
                "server",
                "restore",
                "/backups/lab.age",
                "--identity-file",
                "/safe/age.key",
                "--confirm-data-dir",
                "/home/rcp/rcp-server/data",
            ),
            "server restore",
            {
                "archive_path": "/backups/lab.age",
                "recovery_identity_file": "/safe/age.key",
                "restore_confirmed_data_dir": "/home/rcp/rcp-server/data",
            },
        ),
        (
            (
                "server",
                "restore",
                "/backups/lab.age",
                "--identity-file",
                "/safe/age.key",
                "--confirm-data-dir",
                "/home/rcp/rcp-server/data",
                "--old-authority-disposition",
                "old-machine-destroyed",
                "--confirm-old-authority",
                "b" * 64,
                "--remove-stale-member",
                MEMBER_ID,
            ),
            "server restore",
            {
                "archive_path": "/backups/lab.age",
                "recovery_identity_file": "/safe/age.key",
                "restore_confirmed_data_dir": "/home/rcp/rcp-server/data",
                "restore_old_authority_disposition": "old-machine-destroyed",
                "restore_confirmed_old_authority": "b" * 64,
                "restore_stale_member_id": MEMBER_ID,
            },
        ),
        (
            ("server", "member", "remove", MEMBER_ID),
            "server member remove",
            {"member_id": MEMBER_ID},
        ),
        (
            (
                "server",
                "member",
                "remove",
                MEMBER_ID,
                "--confirm-boundary",
                "d" * 64,
            ),
            "server member remove",
            {"member_id": MEMBER_ID, "member_confirmed_boundary": "d" * 64},
        ),
        (("server", "update"), "server update", {}),
        (
            ("server", "update", "--confirm-target", UPDATE_COMMIT),
            "server update",
            {"update_confirmed_commit": UPDATE_COMMIT},
        ),
    ],
)
def test_server_command_tree_builds_one_strict_request(argv, command, fields) -> None:
    request = request_from_namespace(_parse(*argv))

    assert request.command == command
    for name, value in fields.items():
        assert getattr(request, name) == value


def test_machine_readable_is_a_renderer_choice_before_or_after_the_leaf() -> None:
    before = _parse("server", "--machine-readable", "project", "provision", REQUEST_ID)
    after = _parse("server", "project", "provision", REQUEST_ID, "--machine-readable")

    assert before.machine_readable is True
    assert after.machine_readable is True
    assert request_from_namespace(before) == request_from_namespace(after)


@pytest.mark.parametrize(
    "argv",
    [
        ("server", "install"),
        ("server", "install", "--team-name", ""),
        ("server", "provider", "check"),
        (
            "server",
            "provider",
            "check",
            "--request",
            REQUEST_ID,
            "--project",
            PROJECT_ID,
        ),
        ("server", "project", "provision", REQUEST_ID.upper()),
        ("server", "member", "remove", MEMBER_ID, "--confirm-boundary", "not-a-digest"),
        ("server", "project", "transfer-import", REQUEST_ID, "/tmp/archive"),
        ("server", "project", "provision", REQUEST_ID, "--host", "other.example"),
        ("server", "provider", "check", "--request", REQUEST_ID, "--account", "alice"),
        ("server", "member", "remove", "not-a-uuid"),
        ("server", "restore", "relative.age", "--identity-file", "/safe/age.key"),
        ("server", "restore", "/backups/lab.age", "--identity-file", "relative.key"),
        (
            "server",
            "restore",
            "/backups/lab.age",
            "--identity-file",
            "/safe/age.key",
            "--confirm-data-dir",
            "relative",
        ),
        ("server", "restore", "/backups/lab.age", "--identity", "AGE-SECRET-KEY-1ABC"),
        ("server", "update", "--confirm-target", "abc123"),
        ("server", "update", "--confirm-target", "A" * 40),
    ],
)
def test_server_parser_rejects_ambiguous_ids_overrides_and_raw_restore_identity(argv) -> None:
    with pytest.raises(SystemExit) as raised:
        _parse(*argv)

    assert raised.value.code == 2


def test_restore_ignores_raw_identity_environment_and_keeps_it_out_of_request(
    monkeypatch,
) -> None:
    monkeypatch.setenv("RCP_RECOVERY_IDENTITY", "AGE-SECRET-KEY-1SUPERSECRET")

    request = request_from_namespace(
        _parse(
            "server",
            "restore",
            "/backups/lab.age",
            "--identity-file",
            "/safe/age.key",
        )
    )

    serialized = request.model_dump_json()
    assert "SUPERSECRET" not in serialized
    assert request.recovery_identity_file == "/safe/age.key"


def test_request_model_rejects_fields_that_do_not_belong_to_the_command() -> None:
    with pytest.raises(ValidationError, match="does not accept request_id"):
        ServerCommandRequest(command="server doctor", request_id=REQUEST_ID)
    with pytest.raises(ValidationError, match="exactly one"):
        ServerCommandRequest(command="server provider check")


def test_event_contract_is_bounded_strict_and_plan_stable() -> None:
    pending = _machine_step("server doctor", state="pending")
    with pytest.raises(ValidationError):
        ServerStep(**{**pending.model_dump(), "title": "x" * 121})
    with pytest.raises(ValidationError):
        ServerPlanEvent(
            command="server doctor",
            timestamp=NOW,
            steps=tuple(pending for _ in range(SERVER_CLI_MAX_STEPS + 1)),
        )
    changed_target = _machine_step("server doctor", state="succeeded").model_copy(
        update={"target": MachineTarget(host="other.example", os_account="rcp")}
    )
    with pytest.raises(ValidationError, match="cannot change planned target"):
        ServerCommandExecution(
            events=(
                ServerPlanEvent(command="server doctor", timestamp=NOW, steps=(pending,)),
                ServerStepEvent(command="server doctor", timestamp=NOW, step=changed_target),
            ),
            exit_code=0,
        )


def test_execution_rejects_out_of_order_success_and_oversized_total_output() -> None:
    first = _machine_step("server doctor", state="pending")
    second = ServerStep(
        **{
            **first.model_dump(),
            "number": 2,
            "title": "Read second health source",
        }
    )
    premature = ServerStep(
        **{
            **second.model_dump(),
            "state": "succeeded",
            "message": "The second source passed before the first ran.",
        }
    )
    with pytest.raises(ValidationError, match="earlier step succeeds"):
        ServerCommandExecution(
            events=(
                ServerPlanEvent(command="server doctor", timestamp=NOW, steps=(first, second)),
                ServerStepEvent(command="server doctor", timestamp=NOW, step=premature),
            ),
            exit_code=1,
        )

    fields = tuple(NonsecretField(name=f"field_{index}", value="x" * 2048) for index in range(32))
    large_steps = tuple(
        ServerStep(
            **{
                **first.model_dump(),
                "number": number,
                "title": f"Large step {number}",
                "fields": fields,
            }
        )
        for number in range(1, SERVER_CLI_MAX_STEPS + 1)
    )
    failed = ServerStep(
        **{
            **large_steps[0].model_dump(),
            "state": "failed",
            "message": "The bounded output check failed.",
        }
    )
    with pytest.raises(ValidationError, match=str(SERVER_CLI_MAX_EXECUTION_BYTES)):
        ServerCommandExecution(
            events=(
                ServerPlanEvent(command="server doctor", timestamp=NOW, steps=large_steps),
                ServerStepEvent(command="server doctor", timestamp=NOW, step=failed),
            ),
            exit_code=1,
        )


def test_live_emitter_reserves_enough_space_for_a_maximal_safe_failure() -> None:
    destination_prefix = "https://example.com/"
    target = ExternalServiceTarget(
        service="🧪" * 120,
        resource="🧪" * 512,
        destination_url=destination_prefix + ("a" * (2048 - len(destination_prefix))),
        required_authority_role="🧪" * 256,
    )
    pending = ServerStep(
        number=1,
        title="🧪" * 120,
        purpose="🧪" * 600,
        performed_by="system",
        target=target,
        phase="p" * 64,
        state="pending",
        expected_success="🧪" * 1000,
        message="🧪" * 4000,
        fields=tuple(
            NonsecretField(name=f"field_{index}", value="🧪" * 2048) for index in range(32)
        ),
    )
    emitter = ServerEventEmitter(
        ServerPlanEvent(command="server doctor", timestamp=NOW, steps=(pending,)),
        machine_readable=True,
        stream=StringIO(),
    )

    emitter.fail_unexpected()

    failure = emitter.events[-1]
    assert isinstance(failure, ServerStepEvent)
    assert failure.step.fields == ()
    assert server_event_stream_size((failure,)) < SERVER_CLI_TERMINAL_RESERVE_BYTES


def test_event_text_is_single_line_and_terminal_safe() -> None:
    pending = _machine_step("server doctor", state="pending")
    with pytest.raises(ValidationError, match="control characters"):
        ServerStep(**{**pending.model_dump(), "message": "unsafe\x1b[31mtext"})
    with pytest.raises(ValidationError, match="control characters"):
        MachineTarget(host="lab\nother", os_account="rcp")


def test_operator_action_requires_human_responsibility_actions_and_resume() -> None:
    target = ExternalServiceTarget(
        service="github.com",
        resource="openai/rcp",
        destination_url="https://github.com/openai/rcp/settings/keys",
        required_authority_role="repository administrator",
    )
    common = {
        "number": 1,
        "title": "Grant write access",
        "purpose": "Install a public deploy key.",
        "target": target,
        "phase": "git_write_grant",
        "state": "operator_action_needed",
        "expected_success": "The write probe succeeds.",
        "message": "The grant is missing.",
    }
    with pytest.raises(ValidationError, match="performed by a human"):
        ServerStep(
            **common,
            performed_by="system",
            actions=(ExternalAction(instruction="Enable write access."),),
            resume_argv=("rcp", "server", "project", "provision", REQUEST_ID),
        )
    with pytest.raises(ValidationError, match="require actions and resume"):
        ServerStep(**common, performed_by="human")


def test_system_step_may_transfer_responsibility_only_for_a_human_action_pause() -> None:
    pending = _machine_step("server project provision", state="pending")
    paused = pending.model_copy(
        update={
            "performed_by": "human",
            "state": "operator_action_needed",
            "message": "The exact account path needs operator repair.",
            "actions": (ExternalAction(instruction="Repair the named path, then resume."),),
            "resume_argv": ("rcp", "server", "project", "provision", REQUEST_ID),
        }
    )
    execution = ServerCommandExecution(
        events=(
            ServerPlanEvent(
                command="server project provision",
                timestamp=NOW,
                steps=(pending,),
            ),
            ServerStepEvent(
                command="server project provision",
                timestamp=NOW,
                step=paused,
            ),
        ),
        exit_code=SERVER_CLI_EXIT_OPERATOR_ACTION,
    )

    assert execution.events[-1].step.performed_by == "human"
    with pytest.raises(ValidationError, match="transfer responsibility"):
        ServerCommandExecution(
            events=(
                execution.events[0],
                ServerStepEvent(
                    command="server project provision",
                    timestamp=NOW,
                    step=pending.model_copy(update={"performed_by": "human", "state": "failed"}),
                ),
            ),
            exit_code=1,
        )


def test_external_target_names_a_role_without_accepting_an_invented_user() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExternalServiceTarget.model_validate(
            {
                "service": "github.com",
                "resource": "openai/rcp",
                "destination_url": "https://github.com/openai/rcp/settings/keys",
                "required_authority_role": "repository administrator",
                "user_account": "alice",
            }
        )
    with pytest.raises(ValidationError, match="credential-free HTTPS"):
        ExternalServiceTarget(
            service="github.com",
            resource="openai/rcp",
            destination_url="https://alice:secret@github.com/openai/rcp/settings/keys",
            required_authority_role="repository administrator",
        )
    with pytest.raises(ValidationError, match="credential-free HTTPS"):
        ExternalServiceTarget(
            service="github.com",
            resource="openai/rcp",
            destination_url="https://github.com/openai/rcp/settings/keys?token=secret",
            required_authority_role="repository administrator",
        )


def test_cli_events_redact_secret_shaped_text_and_reject_it_in_argv_or_fields() -> None:
    step = _machine_step("server doctor", state="failed").model_copy(
        update={
            "message": (
                "token=rcp_member_supersecret Authorization: Bearer abcdefghijk "
                "github_pat_abcdefghijklmnop"
            )
        }
    )
    event = ServerStepEvent(command="server doctor", timestamp=NOW, step=step)
    serialized = event.model_dump_json()

    assert "supersecret" not in serialized
    assert "abcdefghijk" not in serialized
    assert "abcdefghijklmnop" not in serialized
    assert "REDACTED" in serialized
    with pytest.raises(ValidationError, match="credential-shaped"):
        CommandAction(argv=("provider", "--token", "ghp_abcdefghijklmnop"))
    with pytest.raises(ValidationError, match="credential-shaped fields"):
        NonsecretField(name="access_token", value="anything")
    with pytest.raises(ValidationError, match="credential-shaped fields"):
        NonsecretField(name="recovery_identity", value="anything")
    with pytest.raises(ValidationError, match="credential-shaped fields"):
        NonsecretField(name="api_key", value="anything")
    with pytest.raises(ValidationError, match="credential-shaped"):
        CommandAction(argv=("age", "--identity", "AGE-SECRET-KEY-1SUPERSECRET"))
    with pytest.raises(ValidationError, match="raw credential flags"):
        CommandAction(argv=("provider", "--token", "plain-value"))
    paused = _operator_execution().events[-1].step
    with pytest.raises(ValidationError, match="raw credential flags"):
        ServerStep(
            **{
                **paused.model_dump(),
                "resume_argv": ("rcp", "server", "restore", "--identity", "plain-value"),
            }
        )
    redacted_action = ExternalAction(
        instruction="Use AGE-SECRET-KEY-1SUPERSECRET or sk-ant-abcdefghijklmnop."
    )
    assert "SUPERSECRET" not in redacted_action.instruction
    assert "abcdefghijklmnop" not in redacted_action.instruction


def test_interactive_and_machine_renderers_use_the_same_external_action() -> None:
    execution = _operator_execution()
    interactive = StringIO()
    machine = StringIO()

    render_server_execution(execution, machine_readable=False, stream=interactive)
    render_server_execution(execution, machine_readable=True, stream=machine)

    interactive_text = interactive.getvalue()
    machine_lines = machine.getvalue().splitlines()
    assert "repository administrator" in interactive_text
    assert "Performed by: human operator" in interactive_text
    assert "State: pending" in interactive_text
    assert "https://github.com/openai/rcp/settings/keys" in interactive_text
    assert f"rcp server project provision {REQUEST_ID}" in interactive_text
    assert len(machine_lines) == len(execution.events)
    adapter = TypeAdapter(ServerCommandEvent)
    parsed = [adapter.validate_json(line) for line in machine_lines]
    assert tuple(parsed) == execution.events
    assert "user_account" not in machine.getvalue()


def test_renderer_selection_never_changes_the_command_handler_call() -> None:
    calls: list[tuple[ServerCommandRequest, CallerIdentity]] = []
    identity = CallerIdentity(uid=501, username="rcp", host="lab.example")

    def handler(request, caller):
        calls.append((request, caller))
        return _successful_command(request, caller)

    interactive = StringIO()
    machine = StringIO()
    first = run_server_command(
        _parse("server", "doctor"),
        handler=handler,
        identity=identity,
        stream=interactive,
    )
    second = run_server_command(
        _parse("server", "doctor", "--machine-readable"),
        handler=handler,
        identity=identity,
        stream=machine,
    )

    assert first == second == 0
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert interactive.getvalue().startswith("Plan for `rcp server doctor`")
    assert json.loads(machine.getvalue().splitlines()[0])["event"] == "plan"


def test_plan_is_visible_before_the_executor_can_perform_work() -> None:
    output = StringIO()
    side_effects = []
    identity = CallerIdentity(uid=501, username="rcp", host="lab.example")

    def handler(request, caller):
        prepared = _successful_command(request, caller)

        def execute(emitter, input_stream) -> None:
            assert output.getvalue().startswith("Plan for `rcp server doctor`")
            side_effects.append("machine work began")
            prepared.execute(emitter, input_stream)

        return PreparedServerCommand(plan=prepared.plan, execute=execute)

    exit_code = run_server_command(
        _parse("server", "doctor"),
        handler=handler,
        identity=identity,
        stream=output,
    )

    assert exit_code == 0
    assert side_effects == ["machine work began"]


def test_executor_exception_becomes_a_secret_safe_terminal_event() -> None:
    output = StringIO()
    identity = CallerIdentity(uid=501, username="rcp", host="lab.example")

    def handler(request, caller):
        prepared = _successful_command(request, caller)

        def execute(_emitter, _input_stream) -> None:
            raise RuntimeError("AGE-SECRET-KEY-1SUPERSECRET token=ghp_abcdefghijklmnop")

        return PreparedServerCommand(plan=prepared.plan, execute=execute)

    exit_code = run_server_command(
        _parse("server", "doctor", "--machine-readable"),
        handler=handler,
        identity=identity,
        stream=output,
    )
    events = [json.loads(line) for line in output.getvalue().splitlines()]

    assert exit_code == SERVER_CLI_EXIT_FAILED
    assert events[0]["event"] == "plan"
    assert events[-1]["step"]["state"] == "failed"
    assert "SUPERSECRET" not in output.getvalue()
    assert "abcdefghijklmnop" not in output.getvalue()


def test_preparer_exception_becomes_a_secret_safe_terminal_event() -> None:
    output = StringIO()

    def handler(_request, _caller):
        raise RuntimeError("sk-proj-abcdefghijklmnop")

    exit_code = run_server_command(
        _parse("server", "doctor", "--machine-readable"),
        handler=handler,
        identity=CallerIdentity(uid=501, username="rcp", host="lab.example"),
        stream=output,
    )

    assert exit_code == SERVER_CLI_EXIT_FAILED
    assert "abcdefghijklmnop" not in output.getvalue()
    assert json.loads(output.getvalue().splitlines()[-1])["step"]["state"] == "failed"


@pytest.mark.parametrize(
    ("argv", "required_account"),
    [
        (("server", "install", "--team-name", "Upgrade Fixture Lab"), "root"),
        (BACKUP_CONFIGURE_ARGV, "root"),
        (
            ("server", "restore", "/backups/lab.age", "--identity-file", "/safe/age.key"),
            "root",
        ),
        (("server", "update"), "root"),
        (("server", "doctor"), "rcp"),
        (("server", "provider", "check", "--request", REQUEST_ID), "rcp"),
        (("server", "project", "provision", REQUEST_ID), "rcp"),
        (("server", "project", "transfer-import", REQUEST_ID), "rcp"),
        (("server", "backup", "run"), "rcp"),
        (("server", "member", "remove", MEMBER_ID), "rcp"),
    ],
)
def test_every_server_command_enforces_its_exact_entry_identity(argv, required_account) -> None:
    identity = CallerIdentity(
        uid=0 if required_account == "root" else 501,
        username=required_account,
        host="lab",
    )
    calls = []

    def handler(request, caller):
        calls.append((request.command, caller))
        return _successful_command(request, caller)

    exit_code = run_server_command(
        _parse(*argv),
        handler=handler,
        identity=identity,
        stream=StringIO(),
    )

    assert exit_code == 0
    assert calls == [(request_from_namespace(_parse(*argv)).command, identity)]

    wrong_identities = (
        CallerIdentity(uid=501, username="rcp", host="lab"),
        CallerIdentity(uid=502, username="alice", host="lab"),
    )
    if required_account == "rcp":
        wrong_identities = (
            CallerIdentity(uid=0, username="root", host="lab"),
            CallerIdentity(uid=502, username="alice", host="lab"),
        )
    for wrong_identity in wrong_identities:
        output = StringIO()

        def refused_handler(*_args):
            pytest.fail("wrong entry identity reached the concrete handler")

        exit_code = run_server_command(
            _parse(*argv, "--machine-readable"),
            handler=refused_handler,
            identity=wrong_identity,
            stream=output,
        )
        events = [json.loads(line) for line in output.getvalue().splitlines()]

        assert exit_code == SERVER_CLI_EXIT_WRONG_IDENTITY
        assert events[-1]["step"]["state"] == "failed"
        assert events[-1]["step"]["phase"] == "entry_identity"


def test_transfer_import_passes_stdin_to_the_handler_without_an_archive_argument() -> None:
    archive = BytesIO(b"versioned transfer bytes")
    observed = []
    identity = CallerIdentity(uid=501, username="rcp", host="lab")

    def handler(request, caller):
        prepared = _successful_command(request, caller)

        def execute(emitter, input_stream) -> None:
            observed.append((request, caller, input_stream.read()))
            prepared.execute(emitter, input_stream)

        return PreparedServerCommand(plan=prepared.plan, execute=execute)

    exit_code = run_server_command(
        _parse("server", "project", "transfer-import", REQUEST_ID),
        handler=handler,
        identity=identity,
        input_stream=archive,
        stream=StringIO(),
    )

    assert exit_code == 0
    assert observed == [
        (
            ServerCommandRequest(
                command="server project transfer-import",
                request_id=REQUEST_ID,
            ),
            identity,
            b"versioned transfer bytes",
        )
    ]


def test_transfer_import_dispatches_to_its_concrete_owner(monkeypatch) -> None:
    output = StringIO()
    calls = []

    def prepare(request, identity):
        calls.append((request, identity))
        return _successful_command(request, identity)

    monkeypatch.setattr("rcp.transfer.target.prepare_transfer_import_command", prepare)

    exit_code = run_server_command(
        _parse("server", "project", "transfer-import", REQUEST_ID, "--machine-readable"),
        identity=CallerIdentity(uid=501, username="rcp", host="lab"),
        stream=output,
    )

    assert exit_code == 0
    assert calls[0][0].request_id == REQUEST_ID
    assert calls[0][1].username == "rcp"


def test_top_level_main_routes_server_commands_before_personal_data_resolution(
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setattr(sys, "argv", ["rcp", "server", "doctor"])
    monkeypatch.setattr(
        "rcp.__main__.default_data_dir",
        lambda: pytest.fail("server CLI resolved the personal app data directory"),
    )
    monkeypatch.setattr("rcp.__main__.run_server_command", lambda args: calls.append(args) or 0)

    main()

    assert len(calls) == 1
    assert calls[0].server_operation == "server doctor"
