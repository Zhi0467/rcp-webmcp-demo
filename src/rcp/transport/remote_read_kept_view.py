"""Read one kept result view out of a repository on the execution machine.

RCP ships this module's *own source* to the execution machine and runs it with
``python -c``. Unlike its siblings this module is also imported in-process, for
the exit-status constants below — the caller and the script must agree on them,
and importing is the only way to guarantee they cannot drift apart.

Protocol. ``argv`` is ``(repository, name, max_bytes)`` for legacy views or
``(repository, directory, name, max_bytes)`` for a generic kept artifact.
"""

from __future__ import annotations

import os
import re
import stat
import sys
from pathlib import Path

MISSING = 44
TOO_LARGE = 45
UNSAFE = 46


def main() -> None:
    repository = Path(sys.argv[1])
    if len(sys.argv) == 4:
        directory, name, max_bytes = "views", sys.argv[2], int(sys.argv[3])
        name_pattern = r"[a-z0-9](?:[a-z0-9-]{0,238})[.]html"
    else:
        directory, name, max_bytes = sys.argv[2], sys.argv[3], int(sys.argv[4])
        name_pattern = r"[a-z0-9](?:[a-z0-9-]{0,220})[.](?:html?|png|jpe?g|gif|webp|svg)"
    if (
        not repository.is_absolute()
        or str(repository) == "/"
        or directory not in {"views", "artifacts"}
        or not re.fullmatch(name_pattern, name)
        or max_bytes < 1
        or max_bytes > 16 * 1024 * 1024
        or not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "O_NOFOLLOW")
    ):
        raise SystemExit(UNSAFE)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        repository_fd = os.open(repository, directory_flags)
    except FileNotFoundError:
        raise SystemExit(MISSING) from None
    except OSError:
        raise SystemExit(UNSAFE) from None
    try:
        try:
            views_fd = os.open(directory, directory_flags, dir_fd=repository_fd)
        except FileNotFoundError:
            raise SystemExit(MISSING) from None
        except OSError:
            raise SystemExit(UNSAFE) from None
        try:
            try:
                file_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=views_fd)
            except FileNotFoundError:
                raise SystemExit(MISSING) from None
            except OSError:
                raise SystemExit(UNSAFE) from None
            try:
                info = os.fstat(file_fd)
                if not stat.S_ISREG(info.st_mode):
                    raise SystemExit(UNSAFE)
                if info.st_size > max_bytes:
                    raise SystemExit(TOO_LARGE)
                remaining = max_bytes + 1
                chunks = []
                while remaining > 0:
                    chunk = os.read(file_fd, min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                data = b"".join(chunks)
                if len(data) > max_bytes:
                    raise SystemExit(TOO_LARGE)
                sys.stdout.buffer.write(data)
            finally:
                os.close(file_fd)
        finally:
            os.close(views_fd)
    finally:
        os.close(repository_fd)


if __name__ == "__main__":
    main()
