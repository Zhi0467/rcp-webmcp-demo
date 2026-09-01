"""Execute one shipped command with private file-creation modes.

The server sends this module's source through ``python3 -c`` on the selected
local or SSH checkout account. The argv remains separate from the source text.
"""

from __future__ import annotations

import os
import sys

PRIVATE_UMASK = 0o077


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        return 2
    os.umask(PRIVATE_UMASK)
    try:
        os.execvp(arguments[0], arguments)
    except OSError:
        return 126


if __name__ == "__main__":
    raise SystemExit(main())
