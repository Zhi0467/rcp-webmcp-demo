from __future__ import annotations

import shlex

SSH_OPTIONS = [
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "ControlMaster=auto",
    "-o",
    "ControlPersist=60",
    "-o",
    "ControlPath=/tmp/rcp-ssh-%C",
]


def ssh_arguments(host: str, command: str) -> list[str]:
    return ["ssh", *SSH_OPTIONS, host, command]


def rsync_ssh_arguments() -> list[str]:
    return ["-e", shlex.join(["ssh", *SSH_OPTIONS])]
