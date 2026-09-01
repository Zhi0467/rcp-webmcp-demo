from __future__ import annotations

import asyncio
import json
import re
import shlex
import subprocess
from pathlib import Path

import pytest

import rcp.runs.tasks.graph as graph_run
from rcp.agents import AgentEvent, AgentProcessControl
from rcp.background import AgentTaskExecution
from rcp.core.models import AuthorizedHuman
from rcp.runs.patch_validator import VALIDATOR_CLIENT_SOURCE
from rcp.runs.tasks.graph import stream_graph_run
from rcp.service import RunRequest, resolve_dispatch_authority
from rcp.storage import AgentTaskRecord
from tests.helpers import agent_patch_json, seed_patch
from tests.helpers import create_named_app as create_app


@pytest.mark.asyncio
async def test_seed_attempt_stages_and_serves_live_validator_before_final_append(
    manifest, tmp_path: Path, monkeypatch
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    store = app.state.background_tasks.store
    owner = store.local_owner
    assert owner is not None and owner.display_name is not None
    operation_id = "seed-live-validator"
    request = RunRequest(run_truth_scope=["repo-a"])
    dispatch_authority = resolve_dispatch_authority("seed", request)
    assert dispatch_authority is not None
    now = store.now()
    store.create_agent_task(
        AgentTaskRecord(
            operation_id=operation_id,
            project_id=app.state.default_project_id,
            kind="seed",
            status="running",
            request=request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="running",
            authorized_by=AuthorizedHuman(
                space_id=app.state.space_id,
                user_id=owner.user_id,
                display_name=owner.display_name,
            ),
            dispatch_authority=dispatch_authority,
        )
    )
    execution = AgentTaskExecution(
        operation_id=operation_id,
        store=store,
        control=AgentProcessControl(),
    )
    monkeypatch.setattr(graph_run, "PATCH_SELF_CHECK_TIMEOUT_SECONDS", 2)

    class ValidatingLauncher:
        validator_result: subprocess.CompletedProcess[str] | None = None

        async def stream(self, _provider, prompt, **kwargs):
            workspace = Path(kwargs["cwd"])
            contract_path = Path(prompt.splitlines()[1])
            contract = contract_path.read_text(encoding="utf-8")
            command_match = re.search(
                r"After writing `patch\.json`, run this exact command: `([^`]+)`",
                contract,
            )
            assert command_match is not None
            command = shlex.split(command_match.group(1))
            validator_client = Path(command[1])
            assert validator_client.read_text(encoding="utf-8") == VALIDATOR_CLIENT_SOURCE

            (workspace / "patch.json").write_text(agent_patch_json(seed_patch()), encoding="utf-8")
            self.validator_result = await asyncio.to_thread(
                subprocess.run,
                command,
                capture_output=True,
                text=True,
                check=False,
            )
            assert self.validator_result.returncode == 0
            assert json.loads(self.validator_result.stdout)["status"] == "valid"
            assert service.history.state().revision == 1
            yield AgentEvent(event="session", session_id="seed-validator-session")
            yield AgentEvent(event="done")

    launcher = ValidatingLauncher()
    frames = [
        frame
        async for frame in stream_graph_run(
            service,
            launcher,
            "seed",
            request,
            tmp_path / "data",
            execution=execution,
        )
    ]

    assert launcher.validator_result is not None
    assert service.history.state().revision == 2
    assert any("applied_revision" in frame for frame in frames)
    assert '"event":"done"' in frames[-1]
