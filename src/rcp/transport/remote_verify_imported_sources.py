"""Verify one immutable imported provider-source directory on an execution host."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys

INVALID = 46


def verify(
    root: str,
    label: str,
    project_id: str,
    expected_json: str,
    expected_fingerprint: str,
) -> dict[str, int | str]:
    expected = json.loads(expected_json)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptors: list[int] = []

    def open_directory(name: str, *, dir_fd: int | None = None) -> int:
        descriptor = os.open(name, directory_flags, dir_fd=dir_fd)
        descriptors.append(descriptor)
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("staged source is not a directory")
        if info.st_mode & 0o222:
            raise ValueError("staged source directory is writable")
        return descriptor

    try:
        root_descriptor = os.open(root, directory_flags)
        descriptors.append(root_descriptor)
        inputs_descriptor = os.open("inputs", directory_flags, dir_fd=root_descriptor)
        descriptors.append(inputs_descriptor)
        target_descriptor = open_directory(label, dir_fd=inputs_descriptor)
        by_provider: dict[str, list[dict[str, object]]] = {}
        for item in expected:
            by_provider.setdefault(item["provider"], []).append(item)
        if sorted(os.listdir(target_descriptor)) != sorted(by_provider):
            raise ValueError("staged provider sources differ from their inventory")
        observed = []
        for provider in sorted(by_provider):
            provider_descriptor = open_directory(provider, dir_fd=target_descriptor)
            items = by_provider[provider]
            if sorted(os.listdir(provider_descriptor)) != [item["sha256"] for item in items]:
                raise ValueError("staged provider sources differ from their inventory")
            for item in items:
                descriptor = os.open(
                    item["sha256"],
                    file_flags,
                    dir_fd=provider_descriptor,
                )
                try:
                    info = os.fstat(descriptor)
                    if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o222:
                        raise ValueError("staged provider source is not an immutable regular file")
                    digest = hashlib.sha256()
                    size = 0
                    while True:
                        chunk = os.read(descriptor, 1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
                        size += len(chunk)
                finally:
                    os.close(descriptor)
                if digest.hexdigest() != item["sha256"] or size != item["size_bytes"]:
                    raise ValueError("staged provider source content differs from its inventory")
                observed.append(
                    {
                        "provider": provider,
                        "sha256": item["sha256"],
                        "size_bytes": size,
                    }
                )
        payload = {"project_id": project_id, "files": observed}
        fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if fingerprint != expected_fingerprint:
            raise ValueError("staged provider source fingerprint differs from its inventory")
        return {
            "fingerprint": fingerprint,
            "file_count": len(observed),
            "payload_size_bytes": sum(item["size_bytes"] for item in observed),
        }
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def main(argv: list[str]) -> int:
    if len(argv) != 6:
        return 2
    try:
        root, label, project_id, expected_fingerprint, raw_limit = argv[1:]
        limit = int(raw_limit)
        encoded_inventory = sys.stdin.buffer.read(limit + 1)
        if len(encoded_inventory) > limit:
            raise ValueError("imported provider source inventory exceeds its byte bound")
        payload = verify(
            root,
            label,
            project_id,
            encoded_inventory.decode("utf-8"),
            expected_fingerprint,
        )
    except BaseException as exc:
        print(str(exc), file=sys.stderr)
        return INVALID
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through SSH source execution
    raise SystemExit(main(sys.argv))
