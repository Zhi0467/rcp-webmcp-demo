"""Build prior RCP source exactly and verify immutable upgrade fixture integrity."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPOSITORY_ROOT / "tests" / "fixtures" / "server_upgrade"
EXACT_BASE_ENV = "RCP_RUN_EXACT_BASE_UPGRADE"
# Ordered by the real upgrade chain. Each entry starts a distinct stored-shape
# or migration-interpretation era; transaction-only refactors are not eras.
EXPECTED_BOUNDARIES: dict[str, tuple[str, str]] = {
    "team-server-v1-78be62b": (
        "78be62b775fd62d7888c2e22d87569c103bffc83",
        "c6fc54845354bb000a9ae9dc26ac40446ba14f96f4f00b8ad8412338ec65da42",
    ),
    "episode-vocabulary-v2-885fa3a": (
        "885fa3abe2f514def4a5612601b58da1dc7292ee",
        "1bc33ba48978e18cad7d609c8afa6d91a035419fa432e4eab12acbc638f65527",
    ),
    "orchestrated-children-v3-bb2f5aa": (
        "bb2f5aa0684239c84b4a4a25b54502099bdc0eb0",
        "96a1205235716360258c7bb0ed367411cd07092b25868c21a421aa81517bb475",
    ),
    "graph-targets-v4-f6085b0": (
        "f6085b0d08f4779f9a38707342f6d2567b5ab53c",
        "c8d606e23d9a6c8744c6887c6542903818533915971c5e9640d3f9124efa9509",
    ),
    "provider-runtime-v5-65b2a08": (
        "65b2a080454ae3699b2da9dbb4a1c8e826b7f43b",
        "2b8245f182bbe44eab59095a5a0bbb2ee461435b5f3d5b9413bdb61e0edb603b",
    ),
    "pre-experiment-repair-v6-af52e03": (
        "af52e03f0466e49afd9dc00878a47ecc0caa57f7",
        "2da06d721ef30316f8a4bd8866c016515a583b263c2e9b7fd1e508ea4fd9a578",
    ),
    "source-server-install-v7-638c19e": (
        "638c19e17252e0e441a698e628b49449df088c81",
        "3f2c9a6cac26424882a7ec64f35d0c0410ea64d86597a3e7359c2ba5951c8a69",
    ),
    "project-provisioning-v8-227f964": (
        "227f9645e850d20cb19a49be7e944ded64309e43",
        "59c77fd91519935483a93ab6bb6e1c5c4b5dff7f3e21496443ce12a8fb2f029d",
    ),
    "central-checkout-v9-a499be3": (
        "a499be3f80618ffe495d7a1a565a33683998502b",
        "cb5afc65b24bd7693c3396816cf03114ce9d430aa06500a2dee6083ca47d53d8",
    ),
    "update-cutover-v10-db3173b": (
        "db3173b4ba31c89cd5370463bb180c007e013368",
        "c9cce4a77d79f30d79cf603215469a379735fa6ed13d1c49d7e9f2238d128ea7",
    ),
    "pre-member-removal-v11-27c9682": (
        "27c9682ed3679ff0063a96995b14ae184dbaff12",
        "5efeb8cde52d346dcbaa8af20f8cc35bd6b2ddabe0572c1c1c4c1c388bb30cb2",
    ),
}


def immutable_fixture_directories() -> list[Path]:
    return [FIXTURE_ROOT / name for name in EXPECTED_BOUNDARIES]


def verify_fixture_registry() -> None:
    actual = {path.name for path in FIXTURE_ROOT.iterdir() if path.is_dir()}
    expected = set(EXPECTED_BOUNDARIES)
    if actual != expected:
        raise ValueError("server-upgrade fixture boundary registry changed")
    for name, (source_commit, root_digest) in EXPECTED_BOUNDARIES.items():
        root = FIXTURE_ROOT / name
        metadata = verify_fixture_integrity(root)
        if metadata.get("boundary") != name:
            raise ValueError(f"server-upgrade fixture boundary was relabeled: {name}")
        if metadata.get("created_with_commit") != source_commit:
            raise ValueError(f"server-upgrade fixture provenance changed: {name}")
        if fixture_bundle_digest(root) != root_digest:
            raise ValueError(f"server-upgrade fixture root digest changed: {name}")


def fixture_bundle_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(relative)
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def verify_fixture_integrity(root: Path) -> dict[str, object]:
    metadata = json.loads((root / "fixture.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ValueError("unsupported server-upgrade fixture schema")
    expected = metadata.get("files")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("server-upgrade fixture has no immutable file inventory")
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "fixture.json"
    }
    if actual_paths != set(expected):
        raise ValueError("server-upgrade fixture file inventory changed")
    for relative, digest in expected.items():
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ValueError("server-upgrade fixture hash inventory is invalid")
        actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"server-upgrade fixture changed: {relative}")
    return metadata


def exact_candidate_base() -> tuple[str, str]:
    dirty = bool(_capture(["git", "status", "--porcelain"], cwd=REPOSITORY_ROOT).strip())
    base_ref = "HEAD" if dirty else "HEAD^1"
    return base_ref, _capture(["git", "rev-parse", base_ref], cwd=REPOSITORY_ROOT).strip()


def build_exact_base_fixture(work_root: Path) -> tuple[Path, str]:
    base_ref, base_commit = exact_candidate_base()
    work_root.mkdir(parents=True)
    archive = work_root / "base.tar"
    checkout = work_root / "base"
    checkout.mkdir()
    _run(
        ["git", "archive", "--format=tar", f"--output={archive}", base_ref],
        cwd=REPOSITORY_ROOT,
    )
    _extract_git_archive(archive, checkout)
    _run(["npm", "ci"], cwd=checkout / "web")
    _run(["npm", "run", "build"], cwd=checkout / "web")
    _run(["uv", "sync", "--project", str(checkout), "--frozen"], cwd=REPOSITORY_ROOT)

    fixture = work_root / "fixture"
    builder = REPOSITORY_ROOT / "tests" / "server_upgrade_fixture_builder.py"
    _run(
        [
            "uv",
            "run",
            "--project",
            str(checkout),
            "--frozen",
            "python",
            str(builder),
            str(fixture),
            "--boundary",
            "exact-candidate-base",
            "--commit",
            base_commit,
        ],
        cwd=work_root,
    )
    return fixture, base_commit


def exact_base_gate_enabled() -> bool:
    return os.environ.get(EXACT_BASE_ENV) == "1"


def _extract_git_archive(archive: Path, destination: Path) -> None:
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("git archive contains an unsafe path")
        bundle.extractall(destination, filter="data")


def _capture(argv: list[str], *, cwd: Path) -> str:
    return subprocess.run(
        argv,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout


def _run(argv: list[str], *, cwd: Path) -> None:
    subprocess.run(argv, cwd=cwd, check=True)


__all__ = [
    "EXACT_BASE_ENV",
    "EXPECTED_BOUNDARIES",
    "build_exact_base_fixture",
    "exact_base_gate_enabled",
    "exact_candidate_base",
    "fixture_bundle_digest",
    "immutable_fixture_directories",
    "verify_fixture_integrity",
    "verify_fixture_registry",
]
