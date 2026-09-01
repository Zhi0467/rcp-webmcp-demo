"""Stdlib-only client staged into an agent run workspace.

This file is deliberately self-contained. RCP ships its source verbatim to a
local or SSH execution stage, where no RCP installation is assumed.
"""

import argparse
import json
import math
import os
import re
import socket
import stat
import sys
import tempfile
import time
import uuid

VERSION = 1
OK = 0
INVALID = 1
UNAVAILABLE = 2
COMMAND_MAILBOX_MAX_REQUEST_BYTES = 16 * 1024 * 1024
# Read the trusted broker's single response through EOF. Guarded Finish hydrates
# its complete durable blocker snapshot outside the compact event ledger, and no
# finite response bound is proved by admission. The socket timeout still bounds a
# peer that stalls before closing the newline-framed response.
PROMPT_FILE_MAX_BYTES = 16 * 1024
_MAILBOX_ID = re.compile(r"^[a-f0-9]{32}$")
_TOKEN = re.compile(r"^[a-f0-9]{64}$")
_SAFE_FILE = re.compile(r"^[A-Za-z0-9._-]+$")
_MUTATING = frozenset(
    (
        "apply",
        "spawn",
        "pause",
        "resume",
        "stop",
        "message",
        "watch_graph",
        "episode",
        "inbox",
        "finish",
    )
)


class ClientInputError(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ClientInputError(message)


def _encoded_request(value):
    content = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
    if len(content) > COMMAND_MAILBOX_MAX_REQUEST_BYTES:
        raise ClientInputError(
            "serialized command request exceeds the "
            f"{COMMAND_MAILBOX_MAX_REQUEST_BYTES}-byte command request limit"
        )
    return content


def _atomic_request(path, content):
    directory = os.path.dirname(path)
    descriptor, temporary = tempfile.mkstemp(prefix=".rcp-command-", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _regular_workspace_file(workspace, path, label):
    absolute = os.path.abspath(path)
    if os.path.dirname(absolute) != workspace:
        raise ClientInputError(f"{label} must be a direct file in this run workspace")
    name = os.path.basename(absolute)
    if not _SAFE_FILE.match(name):
        raise ClientInputError(f"{label} has an unsafe file name")
    if os.path.islink(absolute) or not os.path.isfile(absolute):
        raise ClientInputError(f"{label} is unavailable or not a regular file")
    return absolute


def _workspace_text_filename(workspace, path, label, max_bytes, require_nonblank=False):
    absolute = _regular_workspace_file(workspace, path, label)
    try:
        descriptor = os.open(
            absolute,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode):
                raise ClientInputError(f"{label} is unavailable or not a regular file")
            if file_stat.st_size > max_bytes:
                raise ClientInputError(f"{label} exceeds the {max_bytes}-byte limit")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                content = stream.read(max_bytes + 1)
        finally:
            os.close(descriptor)
    except ClientInputError:
        raise
    except OSError as exc:
        raise ClientInputError(f"{label} could not be read: {exc}") from exc
    if len(content) > max_bytes:
        raise ClientInputError(f"{label} exceeds the {max_bytes}-byte limit")
    try:
        text = content.decode("utf-8")
    except UnicodeError as exc:
        raise ClientInputError(f"{label} must be UTF-8 text") from exc
    if require_nonblank and not text.strip():
        raise ClientInputError(f"{label} must not be blank")
    return os.path.basename(absolute)


def _read_json(path, label):
    try:
        with open(path, encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ClientInputError(f"{label} could not be read: {exc}") from exc
    if not isinstance(value, dict):
        raise ClientInputError(f"{label} must contain one JSON object")
    return value


def _credential(workspace, path):
    path = _regular_workspace_file(workspace, path, "credential")
    value = _read_json(path, "credential")
    if value.get("version") != VERSION:
        raise ClientInputError("credential protocol version is unsupported")
    mailbox_id = value.get("mailbox_id")
    token = value.get("token")
    if not isinstance(mailbox_id, str) or not _MAILBOX_ID.match(mailbox_id):
        raise ClientInputError("credential mailbox id is malformed")
    if not isinstance(token, str) or not _TOKEN.match(token):
        raise ClientInputError("credential token is malformed")
    return mailbox_id, token


def _parser():
    parser = _Parser(prog="rcp-agent-client")
    authority = parser.add_mutually_exclusive_group(required=True)
    authority.add_argument("--credential")
    authority.add_argument("--broker")
    parser.add_argument("--mailbox-id")
    parser.add_argument("--timeout", required=True, type=float)
    parser.add_argument("--workspace", required=True)
    subparsers = parser.add_subparsers(dest="verb", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("patch_path")

    apply = subparsers.add_parser("apply")
    apply.add_argument("--key", required=True)
    apply.add_argument("patch_path")

    status = subparsers.add_parser("status")
    status_target = status.add_mutually_exclusive_group()
    status_target.add_argument("--worker-id")
    status_target.add_argument("--episode-id")

    spawn = subparsers.add_parser("spawn")
    spawn.add_argument("--key", required=True)
    spawn.add_argument("--seat-node", required=True)
    spawn.add_argument("--instruction-file", required=True)

    for verb in ("pause", "resume", "stop"):
        control = subparsers.add_parser(verb)
        control.add_argument("--key", required=True)
        control.add_argument("worker_id")

    message = subparsers.add_parser("message")
    message.add_argument("--key", required=True)
    message.add_argument("--recipient")
    message.add_argument("body")

    watch = subparsers.add_parser("watch-graph")
    watch.add_argument("--key", required=True)
    watch.add_argument("--condition-json", required=True)
    watch.add_argument("--reason", required=True)

    episode = subparsers.add_parser("episode")
    episode.add_argument("--key", required=True)
    episode_action = episode.add_mutually_exclusive_group(required=True)
    episode_action.add_argument("--kick-off-experiment", action="store_true")
    episode_action.add_argument("--stop", metavar="EPISODE_ID")
    episode_action.add_argument("--resume", metavar="EPISODE_ID")
    episode.add_argument("--node")
    episode.add_argument("--goal-file")
    episode.add_argument("--invocation-limit", type=int)

    inbox = subparsers.add_parser("inbox")
    inbox.add_argument("--key", required=True)
    inbox_action = inbox.add_mutually_exclusive_group(required=True)
    inbox_action.add_argument("--harvest", action="store_true")
    inbox_action.add_argument("--clear", action="store_true")

    finish = subparsers.add_parser("finish")
    finish.add_argument("--key", required=True)
    return parser


def _nonblank(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ClientInputError(f"{label} must not be blank")
    return value.strip()


def _request_arguments(namespace, workspace):
    verb = namespace.verb.replace("-", "_")
    if verb == "validate":
        patch_path = _regular_workspace_file(workspace, namespace.patch_path, "patch.json")
        if os.path.basename(patch_path) != "patch.json":
            raise ClientInputError("validation accepts only this run workspace's patch.json")
        try:
            with open(patch_path, "rb") as stream:
                if os.fstat(stream.fileno()).st_size > COMMAND_MAILBOX_MAX_REQUEST_BYTES:
                    raise ClientInputError(
                        "patch.json exceeds the "
                        f"{COMMAND_MAILBOX_MAX_REQUEST_BYTES}-byte command request limit"
                    )
                content = stream.read(COMMAND_MAILBOX_MAX_REQUEST_BYTES + 1)
            if len(content) > COMMAND_MAILBOX_MAX_REQUEST_BYTES:
                raise ClientInputError(
                    "patch.json exceeds the "
                    f"{COMMAND_MAILBOX_MAX_REQUEST_BYTES}-byte command request limit"
                )
            return verb, None, {"patch": content.decode("utf-8")}
        except (OSError, UnicodeError) as exc:
            raise ClientInputError(f"patch.json could not be read: {exc}") from exc
    if verb == "status":
        worker_id = namespace.worker_id
        if worker_id is not None:
            worker_id = _nonblank(worker_id, "worker id")
            if len(worker_id) > 200:
                raise ClientInputError("worker id must be at most 200 characters")
        episode_id = namespace.episode_id
        if episode_id is not None:
            episode_id = _nonblank(episode_id, "episode id")
            if len(episode_id) > 200:
                raise ClientInputError("episode id must be at most 200 characters")
        return verb, None, {"worker_id": worker_id, "episode_id": episode_id}
    key = _nonblank(namespace.key, "idempotency key")
    if verb == "apply":
        patch_file = _workspace_text_filename(
            workspace,
            namespace.patch_path,
            "patch.json",
            COMMAND_MAILBOX_MAX_REQUEST_BYTES,
        )
        if patch_file != "patch.json":
            raise ClientInputError("Apply accepts only this run workspace's patch.json")
        arguments = {"patch_file": patch_file}
    elif verb == "spawn":
        instruction_file = _workspace_text_filename(
            workspace,
            namespace.instruction_file,
            "instruction file",
            PROMPT_FILE_MAX_BYTES,
            require_nonblank=True,
        )
        arguments = {
            "seat_node_id": _nonblank(namespace.seat_node, "seat node"),
            "instruction_file": instruction_file,
        }
    elif verb in ("pause", "resume", "stop"):
        arguments = {"worker_id": _nonblank(namespace.worker_id, "worker id")}
    elif verb == "message":
        recipient = namespace.recipient
        if recipient is not None:
            recipient = _nonblank(recipient, "recipient")
        arguments = {
            "recipient_task_id": recipient,
            "body": _nonblank(namespace.body, "message body"),
        }
    elif verb == "watch_graph":
        try:
            condition = json.loads(namespace.condition_json)
        except ValueError as exc:
            raise ClientInputError(f"graph condition is not valid JSON: {exc}") from exc
        if not isinstance(condition, dict):
            raise ClientInputError("graph condition must be one JSON object")
        arguments = {
            "condition": condition,
            "reason": _nonblank(namespace.reason, "watch reason"),
        }
    elif verb == "episode":
        if namespace.kick_off_experiment:
            if namespace.stop is not None or namespace.resume is not None:
                raise ClientInputError("episode action is ambiguous")
            node_id = namespace.node
            if node_id is None:
                raise ClientInputError("Experiment kickoff requires --node")
            invocation_limit = namespace.invocation_limit
            if invocation_limit is not None and invocation_limit <= 0:
                raise ClientInputError("invocation limit must be a positive integer")
            goal_file = namespace.goal_file
            if goal_file is not None:
                goal_file = _workspace_text_filename(
                    workspace,
                    goal_file,
                    "goal file",
                    PROMPT_FILE_MAX_BYTES,
                    require_nonblank=True,
                )
            arguments = {
                "action": "kick_off_experiment",
                "node_id": _nonblank(node_id, "Experiment node"),
                "goal_file": goal_file,
                "invocation_limit": invocation_limit,
            }
        else:
            if namespace.node is not None:
                raise ClientInputError("--node is available only for Experiment kickoff")
            if namespace.goal_file is not None:
                raise ClientInputError("--goal-file is available only for Experiment kickoff")
            if namespace.invocation_limit is not None:
                raise ClientInputError(
                    "--invocation-limit is available only for Experiment kickoff"
                )
            if namespace.stop is not None:
                action = "stop"
                episode_id = namespace.stop
            elif namespace.resume is not None:
                action = "resume"
                episode_id = namespace.resume
            else:
                raise ClientInputError("episode action is required")
            arguments = {
                "action": action,
                "episode_id": _nonblank(episode_id, "episode id"),
            }
    elif verb == "inbox":
        if namespace.harvest:
            action = "harvest"
        elif namespace.clear:
            action = "clear"
        else:
            raise ClientInputError("inbox action is required")
        arguments = {"action": action}
    elif verb == "finish":
        arguments = {}
    else:
        raise ClientInputError("unsupported command verb")
    if verb not in _MUTATING:
        raise ClientInputError("unsupported mutating command verb")
    return verb, key, arguments


def _print_json(value):
    print(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def _client_failure(verb, status, message):
    if verb == "validate":
        _print_json({"status": status, "messages": [message]})
    else:
        _print_json({"status": status, "message": message, "result": {}})
    return INVALID if status == "invalid" else UNAVAILABLE


def _response_exit_code(response):
    status = response.get("status")
    if status == "ok":
        return OK
    if status == "invalid":
        return INVALID
    if status == "unavailable":
        return UNAVAILABLE
    return None


def _handle_response(response, verb, request_id):
    if not isinstance(response, dict) or response.get("request_id") != request_id:
        return _client_failure(
            verb,
            "unavailable",
            "RCP command returned a malformed or mismatched response.",
        )
    exit_code = _response_exit_code(response)
    if exit_code is None:
        return _client_failure(
            verb,
            "unavailable",
            "RCP command returned an unsupported status.",
        )
    _print_json(_display_response(response, verb))
    return exit_code


def _run(namespace):
    if not math.isfinite(namespace.timeout) or namespace.timeout <= 0:
        raise ClientInputError("timeout must be a positive finite number")
    workspace = os.path.abspath(namespace.workspace)
    if os.path.islink(workspace) or not os.path.isdir(workspace):
        raise ClientInputError("run workspace is unavailable")
    if namespace.broker is not None:
        broker = os.path.abspath(namespace.broker)
        if not broker.startswith("/tmp/rcp-command-") or not broker.endswith(".sock"):
            raise ClientInputError("broker path is outside the bounded temporary namespace")
        mailbox_id = namespace.mailbox_id
        if not isinstance(mailbox_id, str) or not _MAILBOX_ID.fullmatch(mailbox_id):
            raise ClientInputError("broker mailbox id is malformed")
        token = None
    else:
        if namespace.mailbox_id is not None:
            raise ClientInputError("mailbox id is supplied by the credential")
        mailbox_id, token = _credential(workspace, namespace.credential)
    verb, key, arguments = _request_arguments(namespace, workspace)
    request_id = uuid.uuid4().hex
    prefix = f"rcp-command-{mailbox_id}-{request_id}"
    request = {
        "version": VERSION,
        "mailbox_id": mailbox_id,
        "request_id": request_id,
        "credential": token or ("0" * 64),
        "verb": verb,
        "idempotency_key": key,
        "arguments": arguments,
    }
    request_content = _encoded_request(request)
    if namespace.broker is not None:
        return _run_brokered(namespace, broker, request_content, request_id)

    request_path = os.path.join(workspace, prefix + ".request.json")
    response_path = os.path.join(workspace, prefix + ".response.json")
    try:
        _atomic_request(request_path, request_content)
    except OSError as exc:
        return _client_failure(
            namespace.verb,
            "unavailable",
            f"RCP command request could not be written: {exc}",
        )

    deadline = time.monotonic() + namespace.timeout
    while time.monotonic() < deadline:
        try:
            with open(response_path, encoding="utf-8") as stream:
                response = json.load(stream)
        except FileNotFoundError:
            time.sleep(0.1)
            continue
        except (OSError, UnicodeError, ValueError) as exc:
            return _client_failure(
                namespace.verb,
                "unavailable",
                f"RCP command response could not be read: {exc}",
            )
        return _handle_response(response, namespace.verb, request_id)
    return _client_failure(
        namespace.verb,
        "unavailable",
        "RCP command did not answer before the timeout.",
    )


def _run_brokered(namespace, broker, request_content, request_id):
    connection = None
    content = bytearray()
    try:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(namespace.timeout)
        connection.connect(broker)
        connection.sendall(request_content)
        while True:
            chunk = connection.recv(65536)
            if not chunk:
                break
            content.extend(chunk)
    except (OSError, TimeoutError) as exc:
        return _client_failure(
            namespace.verb,
            "unavailable",
            f"RCP command broker is unavailable: {exc}",
        )
    finally:
        if connection is not None:
            connection.close()
    if not content.endswith(b"\n") or b"\n" in content[:-1]:
        return _client_failure(
            namespace.verb,
            "unavailable",
            "RCP command broker did not return one complete newline-delimited response.",
        )
    try:
        response = json.loads(content[:-1])
    except (UnicodeError, ValueError) as exc:
        return _client_failure(
            namespace.verb,
            "unavailable",
            f"RCP command broker returned invalid JSON: {exc}",
        )
    return _handle_response(response, namespace.verb, request_id)


def _display_response(response, verb):
    """Keep the established validator stdout shape over the generic envelope."""

    if verb == "validate":
        result = response.get("result")
        if isinstance(result, dict) and result.get("status") in (
            "valid",
            "invalid",
            "unavailable",
        ):
            messages = result.get("messages")
            if not isinstance(messages, list) or not all(
                isinstance(message, str) for message in messages
            ):
                messages = []
            return {"status": result["status"], "messages": messages}
        status = response.get("status")
        validation_status = "valid" if status == "ok" else status
        message = response.get("message")
        messages = [message] if isinstance(message, str) and message else []
        return {"status": validation_status, "messages": messages}
    message = response.get("message")
    if message is not None and not isinstance(message, str):
        message = "RCP returned a response with an invalid message."
    result = response.get("result")
    if not isinstance(result, dict):
        result = {}
    return {
        "status": response["status"],
        "message": message,
        "result": result,
    }


def _requested_verb(argv):
    for argument in argv:
        if argument in (
            "validate",
            "apply",
            "status",
            "spawn",
            "pause",
            "resume",
            "stop",
            "message",
            "watch-graph",
            "episode",
            "inbox",
            "finish",
        ):
            return argument
    return None


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    verb = _requested_verb(arguments)
    try:
        namespace = _parser().parse_args(arguments)
        return _run(namespace)
    except ClientInputError as exc:
        return _client_failure(verb, "invalid", f"RCP command is invalid: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
