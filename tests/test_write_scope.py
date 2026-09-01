from __future__ import annotations

import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from threading import Barrier
from types import SimpleNamespace
from typing import Any

import pytest

from rcp.agents import AgentLauncher
from rcp.agents.context import ChatContext, RepositoryPointer
from rcp.agents.write_scope import (
    RegisteredRepositoryRoot,
    registered_repository_roots,
    resolve_project_write_scope,
)
from rcp.config import Manifest
from rcp.projects import ProjectCatalog
from rcp.runs.shared import _stage_context_paths
from rcp.storage import AgentTaskRecord, AppStore, ProjectRecord
from rcp.transport import RemoteRunStage, StateUnavailable


def _local_pointers(manifest: Manifest, aliases: list[str]) -> list[RepositoryPointer]:
    return [
        RepositoryPointer(
            alias=alias,
            machine=manifest.repository_map[alias].machine,
            host=manifest.machine_map[manifest.repository_map[alias].machine].host,
            path=manifest.repository_map[alias].path,
        )
        for alias in aliases
    ]


def _resolve_local(
    manifest: Manifest,
    tmp_path: Path,
    *,
    aliases: list[str] | None = None,
    pointers: list[RepositoryPointer] | None = None,
    project_id: str = "project",
    stage_root: Path | None = None,
    workspace_root: Path | None = None,
    app_data_dir: Path | None = None,
    repository_inventory: list[RegisteredRepositoryRoot] | None = None,
):
    selected = aliases or ["repo-a"]
    stage = stage_root or tmp_path / "stage"
    workspace = workspace_root or stage / "workspace"
    stage.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)
    return resolve_project_write_scope(
        manifest=manifest,
        project_id=project_id,
        execution_machine="laptop",
        capability="work_auto",
        stage_root=str(stage),
        workspace_root=str(workspace),
        admitted_aliases=selected,
        repository_pointers=(_local_pointers(manifest, selected) if pointers is None else pointers),
        remote_stage=None,
        app_data_dir=app_data_dir or tmp_path / "app-data",
        repository_inventory=(
            registered_repository_roots(manifest, project_id=project_id or "test-invalid-project")
            if repository_inventory is None
            else repository_inventory
        ),
    )


def test_local_scope_contains_only_exact_admitted_roots_and_protects_research(
    manifest: Manifest, tmp_path: Path
) -> None:
    scope = _resolve_local(manifest, tmp_path, aliases=["repo-a", "repo-b"])

    repo_a = str(Path(manifest.repository_map["repo-a"].path).resolve())
    repo_b = str(Path(manifest.repository_map["repo-b"].path).resolve())
    assert scope.repository_roots == [repo_a, repo_b]
    assert scope.writable_roots == [
        str((tmp_path / "stage" / "workspace").resolve()),
        repo_a,
        repo_b,
    ]
    assert scope.protected_write_paths == sorted(
        [str(Path(repo_a) / ".research"), str(Path(repo_b) / ".research")]
    )


def test_state_research_is_protected_when_state_repository_is_not_admitted(
    manifest: Manifest, tmp_path: Path
) -> None:
    scope = _resolve_local(manifest, tmp_path, aliases=["repo-b"])

    assert scope.repository_roots == [str(Path(manifest.repository_map["repo-b"].path).resolve())]
    assert str(Path(manifest.repository_map["repo-a"].path).resolve() / ".research") in (
        scope.protected_write_paths
    )
    assert str(Path(manifest.repository_map["repo-b"].path).resolve() / ".research") in (
        scope.protected_write_paths
    )


def test_scope_fingerprint_is_deterministic_and_binds_material_scope_fields(
    manifest: Manifest, tmp_path: Path
) -> None:
    first = _resolve_local(manifest, tmp_path, aliases=["repo-a", "repo-b"])
    reordered_pointers = list(reversed(_local_pointers(manifest, ["repo-a", "repo-b"])))
    second = _resolve_local(
        manifest,
        tmp_path,
        aliases=["repo-a", "repo-b"],
        pointers=reordered_pointers,
    )

    assert first.fingerprint == second.fingerprint

    other_project = _resolve_local(
        manifest,
        tmp_path,
        aliases=["repo-a", "repo-b"],
        project_id="other-project",
    )
    assert other_project.fingerprint != first.fingerprint


def _register_catalog_project(
    store: AppStore,
    manifest: Manifest,
    *,
    project_id: str,
    locator: str | None = None,
) -> None:
    state_repository = manifest.repository_map[manifest.state.repository]
    state_machine = manifest.machine_map[state_repository.machine]
    store.upsert_project(
        ProjectRecord(
            project_id=project_id,
            locator=locator or str(manifest.path),
            name=manifest.name,
            state_location=str(manifest.research_dir),
            state_remote=bool(state_machine.host),
            added_at=store.now(),
        )
    )


def test_catalog_repository_inventory_loads_registered_project_roots(
    manifest: Manifest,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    store = AppStore(data_dir / "rcp.sqlite3")
    _register_catalog_project(store, manifest, project_id="registered-project")
    catalog = ProjectCatalog(data_dir, store, AgentLauncher())

    inventory = catalog.repository_ownership_inventory()

    assert {(item.project_id, item.alias) for item in inventory} == {
        ("registered-project", "repo-a"),
        ("registered-project", "repo-b"),
    }


def test_catalog_repository_inventory_fails_closed_for_unavailable_registration(
    manifest: Manifest,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    store = AppStore(data_dir / "rcp.sqlite3")
    _register_catalog_project(
        store,
        manifest,
        project_id="unavailable-project",
        locator=str(tmp_path / "missing-project"),
    )
    catalog = ProjectCatalog(data_dir, store, AgentLauncher())

    with pytest.raises(ValueError, match="Cannot establish the repository ownership inventory"):
        catalog.repository_ownership_inventory()


@pytest.mark.parametrize(
    ("pointers", "message"),
    [
        ([], "missing repository pointer"),
        (
            [
                RepositoryPointer(alias="repo-b", machine="laptop", path="/unused"),
            ],
            "missing repository pointer",
        ),
    ],
)
def test_scope_rejects_missing_or_wrong_pointer_alias(
    manifest: Manifest,
    tmp_path: Path,
    pointers: list[RepositoryPointer],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _resolve_local(manifest, tmp_path, pointers=pointers)


def test_scope_rejects_duplicate_pointer_alias(manifest: Manifest, tmp_path: Path) -> None:
    pointer = _local_pointers(manifest, ["repo-a"])[0]

    with pytest.raises(ValueError, match="duplicate repository pointer"):
        _resolve_local(manifest, tmp_path, pointers=[pointer, pointer])


def test_scope_rejects_pointer_symlink_escape(manifest: Manifest, tmp_path: Path) -> None:
    escaped = Path(manifest.repository_map["repo-a"].path) / "escaped"
    escaped.symlink_to(Path(manifest.repository_map["repo-b"].path), target_is_directory=True)
    pointer = RepositoryPointer(
        alias="repo-a",
        machine="laptop",
        path=str(escaped),
    )

    with pytest.raises(ValueError, match="no longer matches its registered project root"):
        _resolve_local(manifest, tmp_path, pointers=[pointer])


@pytest.mark.parametrize(
    ("repository_root", "message"),
    [
        (Path("/"), "(?:filesystem root|repository root is not writable)"),
        (Path.home(), "execution account home"),
        (Path(tempfile.gettempdir()), "broad temporary directory"),
    ],
)
def test_scope_rejects_broad_local_repository_roots(
    manifest: Manifest,
    tmp_path: Path,
    repository_root: Path,
    message: str,
) -> None:
    changed = manifest.model_copy(deep=True)
    changed.repository_map["repo-a"].path = str(repository_root)
    pointer = RepositoryPointer(alias="repo-a", machine="laptop", path=str(repository_root))

    with pytest.raises(ValueError, match=message):
        _resolve_local(changed, tmp_path, pointers=[pointer])


def test_scope_rejects_repository_overlapping_app_data(manifest: Manifest, tmp_path: Path) -> None:
    repository_root = Path(manifest.repository_map["repo-a"].path)

    with pytest.raises(ValueError, match="application data directory"):
        _resolve_local(manifest, tmp_path, app_data_dir=repository_root / "cache")


@pytest.mark.parametrize("relation", ["equal", "ancestor", "descendant"])
def test_scope_rejects_overlap_with_same_project_unadmitted_repository(
    manifest: Manifest,
    tmp_path: Path,
    relation: str,
) -> None:
    changed = manifest.model_copy(deep=True)
    admitted = Path(changed.repository_map["repo-a"].path)
    if relation == "equal":
        unadmitted = admitted
    elif relation == "ancestor":
        unadmitted = admitted.parent
    else:
        unadmitted = admitted / "nested-unadmitted"
        unadmitted.mkdir()
    changed.repository_map["repo-b"].path = str(unadmitted)

    with pytest.raises(ValueError, match="overlaps an unadmitted repository"):
        _resolve_local(changed, tmp_path, aliases=["repo-a"])


@pytest.mark.parametrize("relation", ["equal", "ancestor", "descendant"])
def test_scope_rejects_overlap_with_repository_owned_by_another_project(
    manifest: Manifest,
    tmp_path: Path,
    relation: str,
) -> None:
    admitted = Path(manifest.repository_map["repo-a"].path)
    if relation == "equal":
        foreign = admitted
    elif relation == "ancestor":
        foreign = admitted.parent
    else:
        foreign = admitted / "foreign-child"
        foreign.mkdir()
    inventory = registered_repository_roots(manifest, project_id="project")
    inventory.append(
        RegisteredRepositoryRoot(
            project_id="other-project",
            alias="foreign",
            machine="local",
            execution_host="",
            path=str(foreign),
        )
    )

    with pytest.raises(ValueError, match="overlaps another project"):
        _resolve_local(
            manifest,
            tmp_path,
            aliases=["repo-a"],
            repository_inventory=inventory,
        )


def test_scope_rejects_canonical_symlink_overlap_hidden_by_distinct_lexical_roots(
    manifest: Manifest,
    tmp_path: Path,
) -> None:
    admitted = Path(manifest.repository_map["repo-a"].path)
    foreign_target = admitted / "foreign-target"
    foreign_target.mkdir()
    foreign_alias = tmp_path / "foreign-alias"
    foreign_alias.symlink_to(foreign_target, target_is_directory=True)
    inventory = registered_repository_roots(manifest, project_id="project")
    inventory.append(
        RegisteredRepositoryRoot(
            project_id="other-project",
            alias="foreign",
            machine="local",
            execution_host="",
            path=str(foreign_alias),
        )
    )

    with pytest.raises(ValueError, match="overlaps another project"):
        _resolve_local(
            manifest,
            tmp_path,
            aliases=["repo-a"],
            repository_inventory=inventory,
        )


def test_scope_rejects_workspace_outside_exact_stage(manifest: Manifest, tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    workspace = tmp_path / "other-workspace"

    with pytest.raises(ValueError, match="workspace must be inside its exact task stage"):
        _resolve_local(
            manifest,
            tmp_path,
            stage_root=stage,
            workspace_root=workspace,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"project_id": ""}, "durable project id"),
        ({"aliases": ["not-in-project"], "pointers": []}, "outside this project"),
    ],
)
def test_scope_rejects_project_mismatch(
    manifest: Manifest,
    tmp_path: Path,
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _resolve_local(manifest, tmp_path, **kwargs)


def test_scope_rejects_repository_pointer_machine_mismatch(
    manifest: Manifest, tmp_path: Path
) -> None:
    pointer = RepositoryPointer(
        alias="repo-a",
        machine="somewhere-else",
        host="",
        path=manifest.repository_map["repo-a"].path,
    )

    with pytest.raises(ValueError, match="does not match its project execution machine"):
        _resolve_local(manifest, tmp_path, pointers=[pointer])


def test_scope_accepts_the_transport_host_the_staged_context_actually_carries(
    manifest: Manifest, tmp_path: Path
) -> None:
    """A pointer's host describes agent transport, not machine topology.

    `_stage_context_paths` blanks it for a repository on the execution machine,
    so comparing it against the machine's SSH host refused every remote run.
    """

    pointer = RepositoryPointer(
        alias="repo-a",
        machine="laptop",
        host="unexpected-host",
        path=manifest.repository_map["repo-a"].path,
    )

    scope = _resolve_local(manifest, tmp_path, pointers=[pointer])

    # The host never reaches the resulting sandbox: the roots come from the
    # manifest and are canonicalized on the execution machine itself.
    assert scope.execution_host == ""
    assert scope.repository_roots == [str(Path(manifest.repository_map["repo-a"].path).resolve())]


class _RemoteScopeStage:
    def __init__(
        self,
        *,
        host: str = "worker.example",
        overrides: dict[str, str] | None = None,
        failure: Exception | None = None,
    ) -> None:
        self.host = host
        self.root = PurePosixPath("/tmp/rcp-run.scope")
        self.overrides = overrides or {}
        self.failure = failure
        self.calls: list[tuple[list[str], bool]] = []

    @property
    def workspace(self) -> PurePosixPath:
        return self.root / "workspace"

    def canonical_directories(
        self, paths: list[str], *, require_writable: bool
    ) -> tuple[dict[str, str], str]:
        self.calls.append((paths, require_writable))
        if self.failure is not None:
            raise self.failure
        return ({path: self.overrides.get(path, path) for path in paths}, "/home/worker")


def _remote_manifest(manifest: Manifest) -> Manifest:
    remote = manifest.model_copy(deep=True)
    remote.machine_map["laptop"].host = "worker.example"
    remote.repository_map["repo-a"].path = "/declared/repo-a"
    remote.repository_map["repo-b"].path = "/declared/repo-b"
    return remote


def _resolve_remote(
    manifest: Manifest,
    stage: _RemoteScopeStage,
    *,
    pointer_path: str = "/pointer/repo-a",
    repository_inventory: list[RegisteredRepositoryRoot] | None = None,
):
    return resolve_project_write_scope(
        manifest=manifest,
        project_id="project",
        execution_machine="laptop",
        capability="orchestrate",
        stage_root=str(stage.root),
        workspace_root=str(stage.workspace),
        admitted_aliases=["repo-a"],
        repository_pointers=[
            RepositoryPointer(
                alias="repo-a",
                machine="laptop",
                host="worker.example",
                path=pointer_path,
            )
        ],
        remote_stage=stage,  # type: ignore[arg-type]
        app_data_dir=None,
        repository_inventory=(
            registered_repository_roots(manifest, project_id="project")
            if repository_inventory is None
            else repository_inventory
        ),
    )


def test_remote_scope_accepts_the_pointer_the_real_staging_step_produces(
    manifest: Manifest,
) -> None:
    """Drive the real `_stage_context_paths` instead of hand-building the pointer.

    Hand-writing the pointer the way the check wanted is how a refusal that fired
    on every remote Work turn shipped green: staging blanks the host of a
    repository on the execution machine, so the check could only ever pass
    locally, where the machine host is blank too.
    """

    remote = _remote_manifest(manifest)
    stage = _RemoteScopeStage(
        host="worker.example",
        overrides={
            "/declared/repo-a": "/srv/repo-a",
            "/declared/repo-a/.research": "/srv/repo-a/.research",
        },
    )
    context = ChatContext.model_construct(
        repositories=[
            RepositoryPointer(
                alias="repo-a",
                machine="laptop",
                host=remote.machine_map["laptop"].host,
                path=remote.repository_map["repo-a"].path,
            )
        ],
        run_truth_scope=["repo-a"],
        graph_path="/x/graph.json",
        research_md_path="/x/research.md",
        introduction_path=None,
        glossary_path="/x/glossary.json",
        coverage_path="/x/coverage.json",
    )
    service = SimpleNamespace(manifest=remote)

    context = context.model_copy(
        update=_stage_context_paths(context, service, stage, "laptop")  # type: ignore[arg-type]
    )
    assert context.repositories[0].host == ""

    scope = resolve_project_write_scope(
        manifest=remote,
        project_id="project",
        execution_machine="laptop",
        capability="work_auto",
        stage_root=str(stage.root),
        workspace_root=str(stage.workspace),
        admitted_aliases=["repo-a"],
        repository_pointers=context.repositories,
        remote_stage=stage,  # type: ignore[arg-type]
        app_data_dir=None,
        repository_inventory=registered_repository_roots(remote, project_id="project"),
    )

    assert scope.execution_host == "worker.example"
    assert scope.repository_roots == ["/srv/repo-a"]


def test_remote_scope_uses_execution_host_canonical_roots(manifest: Manifest) -> None:
    remote = _remote_manifest(manifest)
    stage = _RemoteScopeStage(
        overrides={
            "/declared/repo-a": "/srv/repo-a",
            "/pointer/repo-a": "/srv/repo-a",
            "/declared/repo-a/.research": "/srv/repo-a/.research",
        }
    )

    scope = _resolve_remote(remote, stage)

    assert scope.execution_host == "worker.example"
    assert scope.repository_roots == ["/srv/repo-a"]
    assert scope.protected_write_paths == [
        "/declared/repo-a/.research",
        "/srv/repo-a/.research",
    ]
    assert [writable for _paths, writable in stage.calls] == [True, False, False]


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (StateUnavailable("offline"), StateUnavailable),
        (ValueError("not writable"), ValueError),
    ],
)
def test_remote_scope_fails_closed_when_canonical_resolution_fails(
    manifest: Manifest,
    failure: Exception,
    expected: type[Exception],
) -> None:
    remote = _remote_manifest(manifest)
    stage = _RemoteScopeStage(failure=failure)

    with pytest.raises(expected, match=str(failure)):
        _resolve_remote(remote, stage)


def test_remote_scope_rejects_stage_or_repository_canonical_mismatch(
    manifest: Manifest,
) -> None:
    remote = _remote_manifest(manifest)
    wrong_stage = _RemoteScopeStage(
        overrides={str(PurePosixPath("/tmp/rcp-run.scope")): "/tmp/rcp-run.other"}
    )
    with pytest.raises(ValueError, match="does not match its exact RCP task stage"):
        _resolve_remote(remote, wrong_stage)

    escaped_repository = _RemoteScopeStage(
        overrides={
            "/declared/repo-a": "/srv/repo-a",
            "/pointer/repo-a": "/srv/escaped",
        }
    )
    with pytest.raises(ValueError, match="no longer matches its registered project root"):
        _resolve_remote(remote, escaped_repository)


def test_remote_scope_rejects_task_stage_host_mismatch(manifest: Manifest) -> None:
    remote = _remote_manifest(manifest)
    stage = _RemoteScopeStage(host="other.example")

    with pytest.raises(ValueError, match="different execution host"):
        _resolve_remote(remote, stage)


def test_remote_scope_compares_only_the_exact_execution_host_identity(
    manifest: Manifest,
) -> None:
    remote = _remote_manifest(manifest)
    current = registered_repository_roots(remote, project_id="project")
    foreign = RegisteredRepositoryRoot(
        project_id="other-project",
        alias="foreign",
        machine="worker",
        execution_host="other.example",
        path="/srv/repo-a",
    )
    stage = _RemoteScopeStage(
        overrides={
            "/declared/repo-a": "/srv/repo-a",
            "/pointer/repo-a": "/srv/repo-a",
        }
    )

    scope = _resolve_remote(remote, stage, repository_inventory=[*current, foreign])

    assert scope.repository_roots == ["/srv/repo-a"]


def test_remote_scope_rejects_overlap_on_same_execution_host_identity(
    manifest: Manifest,
) -> None:
    remote = _remote_manifest(manifest)
    current = registered_repository_roots(remote, project_id="project")
    foreign = RegisteredRepositoryRoot(
        project_id="other-project",
        alias="foreign",
        machine="worker",
        execution_host="worker.example",
        path="/declared/foreign",
    )
    stage = _RemoteScopeStage(
        overrides={
            "/declared/repo-a": "/srv/repo-a",
            "/pointer/repo-a": "/srv/repo-a",
            "/declared/foreign": "/srv/repo-a/nested",
        }
    )

    with pytest.raises(ValueError, match="overlaps another project"):
        _resolve_remote(remote, stage, repository_inventory=[*current, foreign])


def test_remote_stage_canonical_directory_probe_validates_response(monkeypatch) -> None:
    stage = RemoteRunStage("worker.example")
    payload = json.dumps(
        {"home": "/home/worker", "paths": {"/declared": "/canonical"}},
        sort_keys=True,
    )
    monkeypatch.setattr(
        stage,
        "_ssh",
        lambda _arguments: subprocess.CompletedProcess([], 0, stdout=payload, stderr=""),
    )

    assert stage.canonical_directories(["/declared"], require_writable=True) == (
        {"/declared": "/canonical"},
        "/home/worker",
    )

    invalid_payload = json.dumps({"home": "/home/worker", "paths": {"/different": "/canonical"}})
    monkeypatch.setattr(
        stage,
        "_ssh",
        lambda _arguments: subprocess.CompletedProcess([], 0, stdout=invalid_payload, stderr=""),
    )
    with pytest.raises(StateUnavailable, match="invalid paths"):
        stage.canonical_directories(["/declared"], require_writable=False)


@pytest.mark.parametrize(
    ("returncode", "message", "expected"),
    [
        (255, "host unavailable", StateUnavailable),
        (42, "repository is not writable", ValueError),
    ],
)
def test_remote_stage_canonical_directory_probe_distinguishes_transport_and_policy_failures(
    monkeypatch,
    returncode: int,
    message: str,
    expected: type[Exception],
) -> None:
    stage = RemoteRunStage("worker.example")
    monkeypatch.setattr(
        stage,
        "_ssh",
        lambda _arguments: subprocess.CompletedProcess([], returncode, stdout="", stderr=message),
    )

    with pytest.raises(expected, match=message):
        stage.canonical_directories(["/declared"], require_writable=True)


def _create_scoped_task(
    store: AppStore,
    operation_id: str,
    *,
    project_id: str = "project",
    stage_root: str = "/tmp/rcp-run.scope",
    kind: str = "refresh",
) -> None:
    now = store.now()
    store.create_agent_task(
        AgentTaskRecord(
            operation_id=operation_id,
            project_id=project_id,
            kind=kind,  # type: ignore[arg-type]
            status="queued",
            request={},
            created_at=now,
            updated_at=now,
            status_message="queued",
            stage_root=stage_root,
        )
    )


def test_durable_task_scope_binding_is_idempotent_and_rejects_identity_mismatch(
    tmp_path: Path,
) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _create_scoped_task(store, "operation")
    fingerprint = "a" * 64

    for _attempt in range(2):
        store.bind_agent_task_write_scope(
            "operation",
            project_id="project",
            stage_host="",
            stage_root="/tmp/rcp-run.scope",
            fingerprint=fingerprint,
        )
    assert (
        AppStore(tmp_path / "rcp.sqlite3").agent_task("operation").write_scope_fingerprint
        == fingerprint
    )

    with pytest.raises(ValueError, match="different project"):
        store.bind_agent_task_write_scope(
            "operation",
            project_id="other-project",
            stage_host="",
            stage_root="/tmp/rcp-run.scope",
            fingerprint=fingerprint,
        )
    with pytest.raises(ValueError, match="saved execution stage"):
        store.bind_agent_task_write_scope(
            "operation",
            project_id="project",
            stage_host="other-host",
            stage_root="/tmp/rcp-run.scope",
            fingerprint=fingerprint,
        )
    with pytest.raises(ValueError, match="changed after it was durably bound"):
        store.bind_agent_task_write_scope(
            "operation",
            project_id="project",
            stage_host="",
            stage_root="/tmp/rcp-run.scope",
            fingerprint="b" * 64,
        )


def test_durable_task_scope_binding_is_compare_and_set(tmp_path: Path) -> None:
    database = tmp_path / "rcp.sqlite3"
    store = AppStore(database)
    _create_scoped_task(store, "operation")
    first_store = AppStore(database)
    second_store = AppStore(database)
    barrier = Barrier(2)

    def bind(candidate_store: AppStore, fingerprint: str) -> str | None:
        barrier.wait()
        try:
            candidate_store.bind_agent_task_write_scope(
                "operation",
                project_id="project",
                stage_host="",
                stage_root="/tmp/rcp-run.scope",
                fingerprint=fingerprint,
            )
        except ValueError:
            return None
        return fingerprint

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda item: bind(*item),
                [(first_store, "a" * 64), (second_store, "b" * 64)],
            )
        )

    winner = store.agent_task("operation").write_scope_fingerprint
    assert results.count(None) == 1
    assert winner in {"a" * 64, "b" * 64}
    assert winner in results


def test_continuation_scope_binding_rejects_mismatch_on_same_stage(tmp_path: Path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _create_scoped_task(store, "first")
    _create_scoped_task(store, "continuation")
    store.bind_agent_task_write_scope(
        "first",
        project_id="project",
        stage_host="",
        stage_root="/tmp/rcp-run.scope",
        fingerprint="a" * 64,
    )

    with pytest.raises(ValueError, match="continuation conflicts"):
        store.bind_agent_task_write_scope(
            "continuation",
            project_id="project",
            stage_host="",
            stage_root="/tmp/rcp-run.scope",
            fingerprint="b" * 64,
        )
