"""Bounded direct-root inventory for a remote lock-free backup export."""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

_MAX_DIRECT_ENTRIES = 100_000
_MAX_NAME_BYTES = 4096


class BackupRootInspectionError(RuntimeError):
    """The remote canonical root cannot be classified safely."""


def inspect_direct_root(root_path: str) -> list[dict[str, str]]:
    root = Path(root_path)
    if not root.is_absolute() or root == Path("/") or ".." in root.parts:
        raise BackupRootInspectionError("The canonical root path is invalid.")
    try:
        root_metadata = root.lstat()
        entries = sorted(root.iterdir(), key=lambda path: path.name)
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise BackupRootInspectionError("The canonical root is unavailable.") from exc
    if not stat.S_ISDIR(root_metadata.st_mode) or len(entries) > _MAX_DIRECT_ENTRIES:
        raise BackupRootInspectionError("The canonical root is unsafe or oversized.")
    result: list[dict[str, str]] = []
    for entry in entries:
        name = entry.name
        if (
            not name
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
            or "\x00" in name
            or len(name.encode("utf-8")) > _MAX_NAME_BYTES
        ):
            raise BackupRootInspectionError("A canonical root entry name is unsafe.")
        try:
            metadata = entry.lstat()
        except OSError as exc:
            raise BackupRootInspectionError("A canonical root entry is unavailable.") from exc
        if stat.S_ISDIR(metadata.st_mode):
            kind = "directory"
        elif stat.S_ISREG(metadata.st_mode):
            kind = "file"
        else:
            kind = "other"
        result.append({"name": name, "kind": kind})
    return result


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        return 2
    try:
        result = inspect_direct_root(argv[1])
    except BackupRootInspectionError:
        return 3
    sys.stdout.write(json.dumps(result, separators=(",", ":"), sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through SSH source execution
    raise SystemExit(main(sys.argv))
