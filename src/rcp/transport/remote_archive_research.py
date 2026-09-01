"""Archive the canonical ``.research`` directory on the execution machine.

RCP ships this module's *own source* to the execution machine and runs it with
``python -c``; nothing in RCP imports it. See ``remote_lock_holder`` for why
these live as real modules rather than string literals.

Protocol. ``argv`` is ``(root, timestamp, expected_fingerprint)``, where the
fingerprint is ``-`` to skip the retained-history check. On success the new
archive path is printed and the exit status is 0. Exit 2 is a rejected argument,
3 is a retained-history fingerprint mismatch, and 1 is any other failure.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import sys
import uuid
from pathlib import Path

BRANCH_PATCH_PATTERN = re.compile(r"[0-9]{6}[.]json")
BRANCH_MERGE_PATTERN = re.compile(r"[a-f0-9]{64}[.]json")
BRANCH_DERIVED_NAMES = frozenset(
    {"graph.json", "research.md", "glossary.json", "proposals.json", "coverage.json"}
)


def require_regular_file(path: Path) -> None:
    try:
        mode = os.lstat(path).st_mode
    except OSError as exc:
        raise ValueError(str(exc)) from exc
    if not stat.S_ISREG(mode):
        raise ValueError(f"retained history input is not a regular file: {path}")


def require_regular_directory(path: Path, label: str) -> None:
    try:
        mode = os.lstat(path).st_mode
    except OSError as exc:
        raise ValueError(str(exc)) from exc
    if not stat.S_ISDIR(mode):
        raise ValueError(f"{label} is not a regular directory: {path}")


def retained_history_paths(root: Path) -> list[Path]:
    require_regular_directory(root, "retained research path")
    manifest = root / "manifest.toml"
    require_regular_file(manifest)
    paths = [manifest]
    scope_base = root / "scope-base.json"
    if os.path.lexists(scope_base):
        require_regular_file(scope_base)
        paths.append(scope_base)
    patches = root / "patches"
    if os.path.lexists(patches):
        if not stat.S_ISDIR(os.lstat(patches).st_mode):
            raise ValueError(f"retained patch path is not a regular directory: {patches}")
        for child in patches.iterdir():
            if re.fullmatch(r"[0-9]{6}[.]json", child.name):
                require_regular_file(child)
                paths.append(child)
            elif child.name.startswith("batch-"):
                if not stat.S_ISDIR(os.lstat(child).st_mode):
                    raise ValueError(f"retained patch batch is not a regular directory: {child}")
                for patch in child.iterdir():
                    if re.fullmatch(r"[0-9]{6}[.]json", patch.name):
                        require_regular_file(patch)
                        paths.append(patch)

    branches = root / "branches"
    if os.path.lexists(branches):
        require_regular_directory(branches, "retained graph branches path")
        for branch in sorted(branches.iterdir(), key=lambda path: path.name):
            try:
                branch_id = uuid.UUID(branch.name)
            except ValueError as exc:
                raise ValueError(
                    f"retained graph branch has a malformed canonical name: {branch}"
                ) from exc
            if str(branch_id) != branch.name or branch_id.version != 4:
                raise ValueError(f"retained graph branch has a malformed canonical name: {branch}")
            require_regular_directory(branch, "retained graph branch path")
            entries = {child.name: child for child in branch.iterdir()}
            allowed = {"branch.json", "patches", "merges", *BRANCH_DERIVED_NAMES}
            malformed = sorted(entries.keys() - allowed)
            if malformed:
                raise ValueError(
                    "retained graph branch contains a malformed canonical entry: "
                    f"{entries[malformed[0]]}"
                )

            metadata = branch / "branch.json"
            require_regular_file(metadata)
            paths.append(metadata)
            for name in BRANCH_DERIVED_NAMES:
                derived = entries.get(name)
                if derived is not None:
                    require_regular_file(derived)

            for directory_name, pattern, label in (
                ("patches", BRANCH_PATCH_PATTERN, "retained graph branch patch path"),
                ("merges", BRANCH_MERGE_PATTERN, "retained graph branch merge path"),
            ):
                directory = branch / directory_name
                require_regular_directory(directory, label)
                for child in directory.iterdir():
                    if pattern.fullmatch(child.name) is None:
                        raise ValueError(f"{label} has a malformed canonical name: {child}")
                    require_regular_file(child)
                    paths.append(child)
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def retained_history_fingerprint(root: Path) -> str:
    digest = hashlib.sha256(b"rcp-retained-history-v2\0")
    for path in retained_history_paths(root):
        relative = path.relative_to(root).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def main() -> None:
    root = Path(sys.argv[1])
    timestamp = sys.argv[2]
    expected_fingerprint = sys.argv[3]
    if (
        not root.is_absolute()
        or str(root) == "/"
        or root.name != ".research"
        or not re.fullmatch(r"[0-9]{8}T[0-9]{12}Z", timestamp)
        or (expected_fingerprint != "-" and not re.fullmatch(r"[0-9a-f]{64}", expected_fingerprint))
    ):
        print("invalid canonical research directory or archive timestamp", file=sys.stderr)
        raise SystemExit(2)
    try:
        mode = os.lstat(root).st_mode
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None
    if not stat.S_ISDIR(mode):
        print("canonical research path is not a regular directory", file=sys.stderr)
        raise SystemExit(1)

    if expected_fingerprint != "-":
        try:
            actual_fingerprint = retained_history_fingerprint(root)
        except (OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(1) from None
        if actual_fingerprint != expected_fingerprint:
            print("retained research changed since preflight", file=sys.stderr)
            raise SystemExit(3)

    base_name = f"{root.name}.archive-{timestamp}"
    for index in range(1, 10000):
        name = base_name if index == 1 else f"{base_name}-{index}"
        archive = root.with_name(name)
        if os.path.lexists(archive):
            continue
        try:
            os.rename(root, archive)
        except FileExistsError:
            continue
        except OSError as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(1) from None
        print(archive)
        raise SystemExit(0)
    print("too many canonical research archive name collisions", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
