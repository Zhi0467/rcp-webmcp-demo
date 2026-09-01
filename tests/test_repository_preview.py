from __future__ import annotations

import shlex
import subprocess
from pathlib import Path, PurePosixPath

import pytest
from fastapi.testclient import TestClient

import rcp.repository_preview as preview_module
from rcp.repository_preview import (
    RepositorySource,
    load_repository_source,
    load_repository_source_for_path,
    repository_source_document,
)
from rcp.transport import StateUnavailable
from rcp.transport.ssh import SSH_OPTIONS

from .helpers import create_named_app as create_app


def test_local_repository_source_is_bounded_utf8_and_does_not_follow_symlinks(
    manifest,
) -> None:
    root = Path(manifest.repository_map["repo-a"].path)
    nested = root / "src"
    nested.mkdir()
    (nested / "safe.py").write_text("print('safe')\n", encoding="utf-8")

    source = load_repository_source(manifest, "repo-a", "src/safe.py", max_bytes=64)

    assert source.text == "print('safe')\n"
    with pytest.raises(ValueError, match="size limit|bounded"):
        load_repository_source(manifest, "repo-a", "src/safe.py", max_bytes=4)
    (nested / "binary.dat").write_bytes(b"\xff")
    with pytest.raises(ValueError, match="UTF-8"):
        load_repository_source(manifest, "repo-a", "src/binary.dat")
    for name, content in (("nul.txt", b"safe\x00unsafe"), ("escape.txt", b"safe\x1bunsafe")):
        (nested / name).write_bytes(content)
        with pytest.raises(ValueError, match="control"):
            load_repository_source(manifest, "repo-a", f"src/{name}")
    (nested / "ordinary-controls.txt").write_bytes(b"tab\tok\r\n")
    assert (
        load_repository_source(manifest, "repo-a", "src/ordinary-controls.txt").text
        == "tab\tok\r\n"
    )
    with pytest.raises(ValueError, match="bounded"):
        load_repository_source(manifest, "repo-a", "src")
    (nested / "linked.py").symlink_to(nested / "safe.py")
    with pytest.raises(ValueError, match="safely"):
        load_repository_source(manifest, "repo-a", "src/linked.py")
    (root / "linked-src").symlink_to(nested, target_is_directory=True)
    with pytest.raises(ValueError, match="regular file|safely"):
        load_repository_source(manifest, "repo-a", "linked-src/safe.py")


@pytest.mark.parametrize(
    "path",
    ["", ".", "..", "/etc/passwd", "src/../secret", "src/./safe.py", "src//safe.py"],
)
def test_repository_source_rejects_unsafe_paths(manifest, path: str) -> None:
    with pytest.raises(ValueError, match="relative|unsafe"):
        load_repository_source(manifest, "repo-a", path)


def test_absolute_repository_path_resolves_one_segment_boundary_match(manifest) -> None:
    root = Path(manifest.repository_map["repo-a"].path)
    target = root / "src" / "safe.py"
    target.parent.mkdir()
    target.write_text("safe", encoding="utf-8")

    source = load_repository_source_for_path(manifest, target.as_posix())

    assert source.repository_alias == "repo-a"
    assert source.relative_path == "src/safe.py"
    assert source.text == "safe"
    with pytest.raises(ValueError, match="outside every"):
        load_repository_source_for_path(manifest, "/outside/configured/repositories.py")
    with pytest.raises(ValueError, match="outside every"):
        load_repository_source_for_path(manifest, f"{root}-sibling/file.py")


@pytest.mark.parametrize("nested", [False, True], ids=["equal-roots", "nested-roots"])
def test_absolute_repository_path_refuses_ambiguous_roots_before_reading(
    manifest,
    monkeypatch,
    nested: bool,
) -> None:
    root = PurePosixPath(manifest.repository_map["repo-a"].path)
    manifest.repository_map["repo-b"].path = str(root / "nested" if nested else root)
    target = root / "nested" / "answer.py"

    def unexpected_reader(*_args, **_kwargs):
        raise AssertionError("ambiguous paths must not reach a repository reader")

    monkeypatch.setattr(preview_module, "_read_local_file", unexpected_reader)
    monkeypatch.setattr(preview_module, "_read_remote_file", unexpected_reader)

    with pytest.raises(ValueError, match="repo-a, repo-b"):
        load_repository_source_for_path(manifest, target.as_posix())


def test_repository_source_document_escapes_content_and_highlights_requested_line() -> None:
    source = RepositorySource(
        repository_alias='repo"><script>alert(1)</script>',
        relative_path="src/<unsafe>.py",
        text='first\n<script>alert("source")</script>',
    )

    document = repository_source_document(source, line=2).decode("utf-8")

    assert "<script>" not in document
    assert "&lt;script&gt;alert(&quot;source&quot;)&lt;/script&gt;" in document
    assert 'id="L1" class="line"' in document
    assert 'id="L2" class="line selected"' in document
    with pytest.raises(ValueError, match="outside"):
        repository_source_document(source, line=3)


def test_remote_repository_source_uses_multiplexed_ssh_reader(
    manifest,
    monkeypatch,
) -> None:
    manifest.machine_map["laptop"].host = "research@example.test"
    captured: dict[str, object] = {}

    def run(arguments, **kwargs):
        captured["arguments"] = arguments
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(arguments, 0, b"remote text\n", b"")

    monkeypatch.setattr(preview_module.subprocess, "run", run)

    source = load_repository_source_for_path(
        manifest,
        f"{manifest.repository_map['repo-a'].path}/nested/file.py",
        max_bytes=123,
    )

    assert source.text == "remote text\n"
    arguments = captured["arguments"]
    assert isinstance(arguments, list)
    assert arguments[: 1 + len(SSH_OPTIONS)] == ["ssh", *SSH_OPTIONS]
    assert arguments[-2] == "research@example.test"
    remote_arguments = shlex.split(arguments[-1])
    assert remote_arguments[:2] == ["python3", "-c"]
    assert remote_arguments[-3:] == [
        manifest.repository_map["repo-a"].path,
        "nested/file.py",
        "123",
    ]
    assert captured["kwargs"] == {
        "capture_output": True,
        "timeout": preview_module.REPOSITORY_PREVIEW_TIMEOUT_SECONDS,
        "check": False,
    }


@pytest.mark.parametrize(
    ("returncode", "error"),
    [
        (44, FileNotFoundError),
        (45, ValueError),
        (255, StateUnavailable),
    ],
)
def test_remote_repository_source_maps_reader_failures(
    manifest,
    monkeypatch,
    returncode: int,
    error: type[Exception],
) -> None:
    manifest.machine_map["laptop"].host = "research@example.test"

    def run(arguments, **_kwargs):
        return subprocess.CompletedProcess(arguments, returncode, b"", b"unavailable")

    monkeypatch.setattr(preview_module.subprocess, "run", run)

    with pytest.raises(error):
        load_repository_source(manifest, "repo-a", "nested/file.py")


def test_repository_preview_route_returns_escaped_get_and_empty_head(manifest, tmp_path) -> None:
    source_path = Path(manifest.repository_map["repo-b"].path) / "answer.py"
    source_path.write_text("first\n<script>alert(1)</script>\n", encoding="utf-8")
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    client = TestClient(app)
    url = f"/api/projects/{project_id}/repositories/files/preview"

    response = client.get(url, params={"path": str(source_path), "line": 2})

    assert response.status_code == 200
    assert "<script>" not in response.text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in response.text
    assert 'id="L2" class="line selected"' in response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["content-security-policy"].startswith("sandbox;")

    head = client.head(url, params={"path": str(source_path), "line": 2})
    assert head.status_code == 200
    assert head.content == b""
    assert head.headers["cache-control"] == "no-store"


def test_repository_preview_route_maps_missing_and_invalid_requests(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    client = TestClient(app)
    url = f"/api/projects/{project_id}/repositories/files/preview"
    repo_a = Path(manifest.repository_map["repo-a"].path)

    assert client.get(url, params={"path": "/outside/repositories/answer.py"}).status_code == 422
    assert client.get(url, params={"path": str(repo_a / "missing.py")}).status_code == 404
    assert client.get(url, params={"path": f"{repo_a}/../secret"}).status_code == 422
    assert (
        client.get(
            url,
            params={"path": str(repo_a / "missing.py"), "line": 0},
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/api/projects/not-registered/repositories/files/preview",
            params={"path": str(repo_a / "answer.py")},
        ).status_code
        == 404
    )


def test_repository_preview_route_names_ambiguous_aliases_before_reading(
    manifest,
    tmp_path,
    monkeypatch,
) -> None:
    root = Path(manifest.repository_map["repo-a"].path)
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    manifest_text = manifest.path.read_text(encoding="utf-8")
    repo_b_root = manifest.repository_map["repo-b"].path
    manifest.path.write_text(
        manifest_text.replace(
            f'path = "{repo_b_root}"',
            f'path = "{root}"',
        ),
        encoding="utf-8",
    )

    def unexpected_reader(*_args, **_kwargs):
        raise AssertionError("ambiguous paths must not reach a repository reader")

    monkeypatch.setattr(preview_module, "_read_local_file", unexpected_reader)
    monkeypatch.setattr(preview_module, "_read_remote_file", unexpected_reader)

    response = TestClient(app).get(
        f"/api/projects/{project_id}/repositories/files/preview",
        params={"path": str(root / "answer.py")},
    )

    assert response.status_code == 422
    assert response.json()["detail"].endswith("repo-a, repo-b")


def test_repository_preview_route_reloads_the_registered_manifest(manifest, tmp_path) -> None:
    original_root = Path(manifest.repository_map["repo-b"].path)
    replacement_root = tmp_path / "replacement-repo"
    replacement_root.mkdir()
    (original_root / "answer.py").write_text("stale", encoding="utf-8")
    (replacement_root / "answer.py").write_text("live", encoding="utf-8")
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    manifest_text = manifest.path.read_text(encoding="utf-8")
    manifest.path.write_text(
        manifest_text.replace(
            f'path = "{original_root}"',
            f'path = "{replacement_root}"',
        ),
        encoding="utf-8",
    )

    response = TestClient(app).get(
        f"/api/projects/{project_id}/repositories/files/preview",
        params={"path": str(replacement_root / "answer.py")},
    )

    assert response.status_code == 200
    assert ">live</span>" in response.text
    assert ">stale</span>" not in response.text
