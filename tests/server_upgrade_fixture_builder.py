"""Create one position-independent, secret-free prior-version server fixture."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import inspect
import json
import shutil
import sqlite3
import uuid
from contextlib import chdir
from pathlib import Path

from rcp.config import load_manifest
from rcp.core.models import AuthorizedHuman
from rcp.history import HistoryManager
from rcp.service import RunRequest
from rcp.storage import AgentTaskRecord, AppStore, ProjectRecord

FIXTURE_SCHEMA_VERSION = 1


def build_fixture(output: Path, *, boundary: str, commit: str) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"fixture output must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    data_dir = output / "data"
    data_dir.mkdir()
    manifest_path = _write_project(output)

    store, bootstrap = AppStore.initialize_team_space(
        data_dir / "rcp.sqlite3",
        "Upgrade Fixture Lab",
    )
    member, member_token = store.enroll_team_member(bootstrap, "Upgrade Fixture Member")
    session, authenticated_member = store.create_team_session(member_token)
    if authenticated_member.user_id != member.user_id:
        raise RuntimeError("fixture session belongs to the wrong member")
    session_hash = hashlib.sha256(session.encode()).hexdigest()
    with store.connection() as connection:
        credential = connection.execute(
            """
            SELECT token_id, token_hash, revoked_at
            FROM team_member_tokens WHERE user_id = ?
            """,
            (member.user_id,),
        ).fetchone()
        persisted_session = connection.execute(
            "SELECT session_hash FROM team_sessions WHERE user_id = ?",
            (member.user_id,),
        ).fetchone()
    if credential is None or credential["revoked_at"] is not None:
        raise RuntimeError("fixture member has no active credential")
    if persisted_session is None or persisted_session["session_hash"] != session_hash:
        raise RuntimeError("fixture member session was not persisted")
    manifest = load_manifest(manifest_path)
    HistoryManager(manifest).initialize()
    history = HistoryManager(manifest, expected_space_id=store.space_id)
    identity = history.claim_project_identity("created")
    materialized = history.initialize().state
    relative_locator = "project/.research/manifest.toml"
    relative_state = "project/.research"
    project = store.upsert_project(
        ProjectRecord(
            project_id=identity.project_id,
            home_space_id=store.space_id,
            locator=relative_locator,
            name=manifest.name,
            state_location=relative_state,
            state_remote=False,
            added_at=store.now(),
            revision=materialized.revision,
            reachable=True,
        )
    )
    seat_member = getattr(store, "seat_project_member", None)
    if callable(seat_member):
        seat_member(project.project_id, member.user_id)
    authorized_by = AuthorizedHuman(
        space_id=store.space_id,
        user_id=member.user_id,
        display_name=member.display_name,
    )
    store, experiment_episode_id, experiment_operation_id = _create_experiment_fixture(
        store,
        project_id=project.project_id,
        authorized_by=authorized_by,
    )
    store, provisioning_request_id, provisioning_project_id = _provisioning_fixture(
        store,
        authorized_by=authorized_by,
    )
    provisioning_configuration_complete = _provisioning_configuration_complete(
        store,
        provisioning_request_id,
    )

    operation_id = str(uuid.uuid4())
    now = store.now()
    request = RunRequest(
        provider="codex",
        run_truth_scope=["repo"],
        model="",
        reasoning="medium",
        run_on="laptop",
    )
    store.create_agent_task(
        AgentTaskRecord(
            operation_id=operation_id,
            project_id=project.project_id,
            kind="refresh",
            status="running",
            request=request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            started_at=now,
            status_message="Fixture task was running when the prior server stopped.",
            phase="running",
            last_activity_at=now,
        )
    )

    space_id = store.space_id
    user_id = member.user_id
    project_id = project.project_id
    expected_repair = _expected_experiment_repair(store, experiment_episode_id)
    _settle_fixture_files(store.path, manifest.research_dir)

    _assert_secret_free(output, bootstrap, member_token, session)
    _compress_database(store.path)
    metadata = {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "boundary": boundary,
        "created_with_commit": commit,
        "space_id": space_id,
        "user_id": user_id,
        "member_token_id": str(credential["token_id"]),
        "member_token_hash": str(credential["token_hash"]),
        "member_session_hash": session_hash,
        "project_id": project_id,
        "active_operation_id": operation_id,
        "experiment_episode_id": experiment_episode_id,
        "experiment_operation_id": experiment_operation_id,
        "provisioning_request_id": provisioning_request_id,
        "provisioning_project_id": provisioning_project_id,
        "provisioning_configuration_complete": provisioning_configuration_complete,
        "expected_repair": expected_repair,
        "expected_revision": materialized.revision,
        "files": _file_hashes(output),
    }
    (output / "fixture.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def upgrade_fixture(
    source: Path,
    output: Path,
    *,
    boundary: str,
    commit: str,
    complete_experiment_task: bool,
) -> None:
    if output.exists():
        raise ValueError(f"fixture output must be absent: {output}")
    metadata = json.loads((source / "fixture.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != FIXTURE_SCHEMA_VERSION:
        raise ValueError("source fixture schema is unsupported")
    shutil.copytree(source, output)
    (output / "fixture.json").unlink()
    database = _materialize_database(output)
    with chdir(output):
        store = AppStore(database)
        if store.space_id != metadata.get("space_id"):
            raise RuntimeError("fixture upgrade changed its space identity")
        project_id = metadata.get("project_id")
        if not isinstance(project_id, str):
            raise ValueError("source fixture has no project identity")
        project = store.project(project_id)
        if project is None:
            raise RuntimeError("fixture upgrade lost its project")
        manifest = load_manifest(project.locator)
        materialized = HistoryManager(
            manifest,
            expected_space_id=store.space_id,
            project_id=project_id,
            require_attribution=True,
        ).initialize()
        if materialized.state.replay_status != "complete":
            raise RuntimeError("fixture upgrade could not replay canonical history")
        experiment_operation_id = metadata.get("experiment_operation_id")
        if complete_experiment_task:
            if not isinstance(experiment_operation_id, str):
                raise ValueError("source fixture has no Experiment task to complete")
            before = store.agent_task(experiment_operation_id)
            if before is None or before.status != "running":
                raise RuntimeError("fixture Experiment task is not running before completion")
            store.complete_agent_task(
                experiment_operation_id,
                applied_revision=materialized.state.revision,
                result={"graph_update": {"status": "applied"}},
            )
            after = store.agent_task(experiment_operation_id)
            if after is None or after.status != "succeeded":
                raise RuntimeError("fixture Experiment task did not complete")
        experiment_episode_id = metadata.get("experiment_episode_id")
        expected_repair = _expected_experiment_repair(
            store,
            experiment_episode_id if isinstance(experiment_episode_id, str) else None,
        )
        user_id = metadata.get("user_id")
        if not isinstance(user_id, str):
            raise ValueError("source fixture has no member identity")
        member = store.space_user(user_id)
        if member is None:
            raise RuntimeError("fixture upgrade lost its member")
        prior_provisioning_request_id = metadata.get("provisioning_request_id")
        store, provisioning_request_id, provisioning_project_id = _provisioning_fixture(
            store,
            authorized_by=AuthorizedHuman(
                space_id=store.space_id,
                user_id=member.user_id,
                display_name=member.display_name,
            ),
            existing_request_id=metadata.get("provisioning_request_id"),
            existing_project_id=metadata.get("provisioning_project_id"),
        )
        legacy_provisioning_request_id = metadata.get("legacy_provisioning_request_id")
        if (
            isinstance(prior_provisioning_request_id, str)
            and prior_provisioning_request_id != provisioning_request_id
        ):
            legacy_provisioning_request_id = prior_provisioning_request_id
        provisioning_configuration_complete = _provisioning_configuration_complete(
            store,
            provisioning_request_id,
        )
    _settle_fixture_files(store.path, output / "project" / ".research")
    _compress_database(store.path)
    metadata.update(
        {
            "boundary": boundary,
            "created_with_commit": commit,
            "expected_revision": materialized.state.revision,
            "expected_repair": expected_repair,
            "provisioning_request_id": provisioning_request_id,
            "provisioning_project_id": provisioning_project_id,
            "legacy_provisioning_request_id": legacy_provisioning_request_id,
            "provisioning_configuration_complete": provisioning_configuration_complete,
            "files": _file_hashes(output),
        }
    )
    (output / "fixture.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _create_experiment_fixture(
    store: AppStore,
    *,
    project_id: str,
    authorized_by: AuthorizedHuman,
) -> tuple[AppStore, str | None, str | None]:
    create = getattr(store, "create_experiment_episode_with_invocation", None)
    episode_id = str(uuid.uuid4())
    operation_id = str(uuid.uuid4())
    now = store.now()
    request = {
        "provider": "codex",
        "model": "gpt-5",
        "reasoning": "medium",
        "run_on": "laptop",
        "run_truth_scope": ["repo"],
        "chat_id": "upgrade-fixture-experiment",
        "node_id": "exp/upgrade-fixture",
        "message": "Continue the bounded fixture experiment.",
        "mode": "work",
        "trigger": "experiment_run",
        "patch_kind": "experiment_loop",
        "control_node_id": "exp/upgrade-fixture",
        "control_revision": 1,
        "control_episode_id": episode_id,
        "control_invocation": 1,
        "control_invocation_ceiling": 2,
        "control_decision_bundle": [],
        "control_completion_criteria": ["The fixture evaluation has finished."],
        "watcher_ids": [],
        "session_id": None,
        "workflow_ids": [],
        "skill_ids": [],
        "invoked_workflow_ids": [],
        "invoked_skill_ids": [],
        "resolved_skill_packages": [],
    }
    if callable(create):
        create(
            AgentTaskRecord(
                operation_id=operation_id,
                project_id=project_id,
                episode_id=episode_id,
                kind="node_chat",
                status="queued",
                request=request,
                created_at=now,
                updated_at=now,
                status_message="Fixture Experiment invocation was queued.",
                phase="queued",
                last_activity_at=now,
                authorized_by=authorized_by,
            )
        )
    else:
        commit = getattr(store, "commit_experiment_episode_turn", None)
        if not callable(commit):
            return store, None, None
        store.create_agent_task(
            AgentTaskRecord(
                operation_id=operation_id,
                project_id=project_id,
                kind="node_chat",
                status="running",
                request=request,
                created_at=now,
                updated_at=now,
                started_at=now,
                status_message="Fixture Experiment invocation was running.",
                phase="running",
                last_activity_at=now,
                authorized_by=authorized_by,
            )
        )
        commit(
            episode_id=episode_id,
            project_id=project_id,
            control_node_id="exp/upgrade-fixture",
            provider="codex",
            execution_machine="laptop",
            execution_host="",
            native_session_id="fixture-native-session",
            stage_host=None,
            stage_root="project/.research/fixture-stage",
            chat_id="upgrade-fixture-experiment",
            operation_id=operation_id,
            invocation=1,
            graph_result="applied",
            watcher_ids=[],
            context_baseline={},
        )
    # Persist the exact state this boundary produced after a normal server
    # restart before the next historical boundary upgrades it.
    reopened = AppStore(store.path)
    if reopened.experiment_episode(episode_id) is None:
        raise RuntimeError("fixture Experiment episode did not survive restart")
    return reopened, episode_id, operation_id


def _provisioning_fixture(
    store: AppStore,
    *,
    authorized_by: AuthorizedHuman,
    existing_request_id: object = None,
    existing_project_id: object = None,
) -> tuple[AppStore, str | None, str | None]:
    create = getattr(store, "create_project_provisioning_request", None)
    read = getattr(store, "project_provisioning_request", None)
    receipts = getattr(store, "project_provisioning_step_receipts", None)
    if not callable(create) or not callable(read) or not callable(receipts):
        if existing_request_id is not None or existing_project_id is not None:
            raise RuntimeError("fixture provisioning state is unsupported by this boundary")
        return store, None, None

    from rcp.server_ops.github import parse_github_repository_ref
    from rcp.server_ops.layout import DEFAULT_SERVER_LAYOUT
    from rcp.storage import (
        ProjectProvisioningMachineIntent,
        ProjectProvisioningProviderIntent,
        ProjectProvisioningRepositoryIntent,
    )

    configuration = (
        {
            "name": "Upgrade Fixture Prepared Project",
            "state_repository": "repo",
            "project_truth_scope": ["repo"],
            "default_run_truth_scope": ["repo"],
        }
        if "name" in inspect.signature(create).parameters
        else {}
    )

    def create_started_request():
        created = create(
            kind="create_team_project",
            authorized_by=authorized_by,
            machines=[
                ProjectProvisioningMachineIntent(
                    alias="server",
                    location="local",
                    os_account="rcp",
                    central_root=str(DEFAULT_SERVER_LAYOUT.projects_root),
                )
            ],
            repositories=[
                ProjectProvisioningRepositoryIntent(
                    alias="repo",
                    repository=parse_github_repository_ref("https://github.com/openai/rcp.git"),
                    machine_alias="server",
                )
            ],
            provider_checks=[
                ProjectProvisioningProviderIntent(
                    profile="seed",
                    provider="codex",
                    runtime_id="codex:exec",
                    model="gpt-5",
                    reasoning="medium",
                    machine_alias="server",
                )
            ],
            **configuration,
        )
        return store.transition_project_provisioning_request(
            created.request_id,
            receipt_id="fixture-setup-started",
            phase="setup_start",
            expected_revision=0,
            expected_status="waiting_for_server_setup",
            to_status="setup_in_progress",
            machines=created.machines,
            repositories=created.repositories,
            provider_checks=created.provider_checks,
        )

    if existing_request_id is None and existing_project_id is None:
        request = create_started_request()
    else:
        if not isinstance(existing_request_id, str) or not isinstance(existing_project_id, str):
            raise ValueError("fixture provisioning metadata is incomplete")
        request = read(existing_request_id)
        if request is None or request.proposed_project_id != existing_project_id:
            raise RuntimeError("fixture upgrade lost its provisioning request")
        if configuration and not getattr(request, "configuration_complete", False):
            request = create_started_request()

    reopened = AppStore(store.path)
    reopened_read = reopened.project_provisioning_request
    reopened_receipts = reopened.project_provisioning_step_receipts
    persisted = reopened_read(request.request_id)
    if persisted is None or persisted.status != "setup_in_progress" or persisted.revision != 1:
        raise RuntimeError("fixture provisioning request did not survive restart")
    persisted_receipts = reopened_receipts(request.request_id)
    if [receipt.receipt_id for receipt in persisted_receipts] != ["fixture-setup-started"]:
        raise RuntimeError("fixture provisioning receipt did not survive restart")
    if reopened.project(persisted.proposed_project_id) is not None:
        raise RuntimeError("fixture preparation created project authority")
    return reopened, persisted.request_id, persisted.proposed_project_id


def _provisioning_configuration_complete(
    store: AppStore,
    request_id: str | None,
) -> bool | None:
    if request_id is None:
        return None
    request = store.project_provisioning_request(request_id)
    if request is None:
        raise RuntimeError("fixture provisioning request disappeared")
    return bool(getattr(request, "configuration_complete", False))


def _expected_experiment_repair(store: AppStore, episode_id: str | None) -> str | None:
    if episode_id is None:
        return None
    with store.connection() as connection:
        tables = {
            str(row["name"])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        if not {"episodes", "episode_wrapups"}.issubset(tables):
            return None
        episode = connection.execute(
            "SELECT status, ending, wrapup_state FROM episodes WHERE episode_id = ?",
            (episode_id,),
        ).fetchone()
        wrapup = connection.execute(
            "SELECT state FROM episode_wrapups WHERE episode_id = ?",
            (episode_id,),
        ).fetchone()
    if (
        episode is not None
        and episode["status"] in {"queued", "running", "stopping"}
        and episode["ending"] is None
        and episode["wrapup_state"] == "not_started"
        and wrapup is not None
        and wrapup["state"] == "legacy_unavailable"
    ):
        return "impossible_legacy_experiment_wrapup"
    return None


def _settle_fixture_files(database: Path, research: Path) -> None:
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    for suffix in ("-wal", "-shm"):
        database.with_name(f"{database.name}{suffix}").unlink(missing_ok=True)
    for name in (".append.lock", ".state.lock"):
        (research / name).unlink(missing_ok=True)


def _compress_database(database: Path) -> None:
    compressed = database.with_name(f"{database.name}.gz")
    compressed.write_bytes(gzip.compress(database.read_bytes(), compresslevel=9, mtime=0))
    database.unlink()


def _materialize_database(root: Path) -> Path:
    database = root / "data" / "rcp.sqlite3"
    compressed = database.with_name(f"{database.name}.gz")
    if not compressed.is_file() or database.exists():
        raise ValueError("source fixture database compression is invalid")
    database.write_bytes(gzip.decompress(compressed.read_bytes()))
    compressed.unlink()
    return database


def _write_project(root: Path) -> Path:
    project = root / "project"
    research = project / ".research"
    research.mkdir(parents=True)
    (project / "provider-sources" / "claude").mkdir(parents=True)
    (project / "provider-sources" / "codex").mkdir(parents=True)
    manifest = research / "manifest.toml"
    manifest.write_text(
        """name = "Upgrade Fixture Project"

[[machines]]
alias = "laptop"
host = ""

[[repositories]]
alias = "repo"
machine = "laptop"
path = "."

[project]
truth_scope = ["repo"]

[state]
repository = "repo"

[agent]
default_run_truth_scope = ["repo"]

[sources]
claude_roots = ["provider-sources/claude"]
codex_roots = ["provider-sources/codex"]

[execution]
run_on = "laptop"

[paper.coach]
default_provider = "codex"
default_model = ""
default_reasoning = "medium"
""",
        encoding="utf-8",
    )
    return manifest


def _assert_secret_free(root: Path, *secrets: str) -> None:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        payload = path.read_bytes()
        for secret in secrets:
            if secret.encode() in payload:
                raise RuntimeError(f"raw fixture credential leaked into {path.relative_to(root)}")


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "fixture.json"
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--boundary", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--source-fixture", type=Path)
    parser.add_argument("--complete-experiment-task", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.complete_experiment_task and args.source_fixture is None:
        parser.error("--complete-experiment-task requires --source-fixture")
    if args.source_fixture is None:
        build_fixture(output, boundary=args.boundary, commit=args.commit)
    else:
        upgrade_fixture(
            args.source_fixture.resolve(),
            output,
            boundary=args.boundary,
            commit=args.commit,
            complete_experiment_task=args.complete_experiment_task,
        )


if __name__ == "__main__":
    main()
