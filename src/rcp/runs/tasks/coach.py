from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from contextlib import aclosing
from datetime import UTC, datetime
from pathlib import Path

from rcp.agents import AgentEvent, AgentLauncher, PromptFactory
from rcp.agents.prompts import invoked_package_pointers
from rcp.background import AgentTaskExecution
from rcp.paper import PaperService, WritingSession
from rcp.providers import configured_runtime_id
from rcp.runs.shared import (
    _parent_task_contract_path,
    _pinned_to_profile,
    _ProviderOutcome,
    _record_agent_launch_receipt,
    _record_provider_exit,
    _sse,
    _stage_json_task_input,
    _stage_task_contract,
    _stage_task_input,
    _swept_stage_root,
    _task_token,
)
from rcp.service import CoachRequest, ProjectService
from rcp.skills.staging import stage_skill_selection
from rcp.transport import StateUnavailable


def _resolved_coach_request(
    service: ProjectService,
    request: CoachRequest,
) -> CoachRequest:
    existing = None
    if request.session_id:
        existing = next(
            (
                session
                for session in service.paper.sessions()
                if session.native_session_id == request.session_id
            ),
            None,
        )
        if existing is None:
            raise ValueError("That native session was not created by this Paper workspace.")
        if request.provider is not None and request.provider != existing.provider:
            raise ValueError("A resumed session cannot change provider.")
        if request.model is not None and (request.model or "provider-default") != existing.model:
            raise ValueError("A resumed session cannot change model.")
        if request.reasoning is not None and request.reasoning != existing.reasoning:
            raise ValueError("A resumed session cannot change reasoning.")
        if request.run_on is not None and request.run_on != existing.execution_machine:
            raise ValueError("A resumed session cannot change execution machine.")
    profile = service.resolve_agent_profile(
        "paper_coach",
        provider=request.provider or (existing.provider if existing else None),
        model=(
            request.model
            if request.model is not None
            else ("" if existing.model == "provider-default" else existing.model)
            if existing
            else None
        ),
        reasoning=request.reasoning or (existing.reasoning if existing else None),
        run_on=request.run_on or (existing.execution_machine if existing else None),
    )
    resolved = request.model_copy(
        update={
            "provider": profile.provider,
            "model": profile.model or None,
            "reasoning": profile.reasoning,
            "run_on": profile.run_on,
        }
    )
    result = service.resolve_skill_request(resolved)
    assert isinstance(result, CoachRequest)
    return result


async def stream_coach(
    service: ProjectService,
    launcher: AgentLauncher,
    paper: PaperService,
    request: CoachRequest,
    data_dir: Path,
    execution: AgentTaskExecution | None = None,
) -> AsyncIterator[str]:
    continuation = execution.continuation if execution is not None else "fresh"
    reusing_checkpoint = bool(execution is not None and execution.reuses_native_checkpoint)
    resuming = continuation == "resume"
    retrying = continuation == "retry"
    retry_attempt = continuation in {"retry", "handoff"}
    existing = None
    if request.session_id:
        existing = next(
            (item for item in paper.sessions() if item.native_session_id == request.session_id),
            None,
        )
    try:
        if not (reusing_checkpoint and existing is None):
            request = _resolved_coach_request(service, request)
        profile = service.resolve_agent_profile(
            "paper_coach",
            provider=request.provider,
            model=request.model,
            reasoning=request.reasoning,
            run_on=request.run_on,
        )
    except ValueError as exc:
        yield _sse(AgentEvent(event="error", text=str(exc)))
        return
    request = _pinned_to_profile(request, profile)
    execution_machine = service.manifest.machine_map[profile.run_on]
    execution_host = execution_machine.host
    provider_binary = execution_machine.provider_paths.get(profile.provider)
    if execution_host:
        yield _sse(
            AgentEvent(
                event="error",
                text=(
                    "Remote writing-coach sessions need persistent read-only staging for native "
                    "resume. Choose a local machine for this invocation."
                ),
            )
        )
        return
    stage_root = _swept_stage_root(data_dir)
    if reusing_checkpoint:
        if not execution.stage_root:
            yield _sse(
                AgentEvent(
                    event="error",
                    text="The interrupted paper-coach task has no staging checkpoint; retry it.",
                )
            )
            return
        local_stage = Path(execution.stage_root).resolve()
        if local_stage.parent != stage_root.resolve() or not local_stage.is_dir():
            yield _sse(
                AgentEvent(
                    event="error",
                    text="The interrupted paper-coach staging checkpoint is unavailable; retry it.",
                )
            )
            return
    else:
        token = _task_token(execution)
        local_stage = stage_root / f"paper-{token}"
        local_stage.mkdir(parents=True, exist_ok=False)
        if execution is not None:
            execution.checkpoint_stage("", str(local_stage))
    snapshot = paper.snapshot()
    draft_override = None
    if not resuming and snapshot.sync_state in {"unsynced", "conflict"}:
        draft_override = Path(
            _stage_task_input(
                local_stage,
                None,
                f"task-{_task_token(execution)}-paper-draft.md",
                snapshot.content,
            )
        )
    try:
        skill_pointers = stage_skill_selection(
            service.resolve_skill_selection(request),
            local_stage=local_stage,
            remote_stage=None,
            label=f"rcp-skills-{_task_token(execution)}",
        )
        if reusing_checkpoint and not request.session_id:
            raise ValueError(
                "The continued paper-coach task has no native agent session; retry it from a "
                "clean attempt instead."
            )
        if resuming:
            assert execution is not None
            original_contract_path = _parent_task_contract_path(execution, local_stage, None)
            contract = PromptFactory.continuation_task_contract(
                original_contract_path=original_contract_path,
                mode="resume",
                invoked_skill_pointers=invoked_package_pointers(
                    skill_pointers,
                    workflow_ids=request.invoked_workflow_ids,
                    skill_ids=request.invoked_skill_ids,
                ),
                invoked_provider_skills=request.resolved_provider_skills,
            )
            contract_path, prompt = _stage_task_contract(
                local_stage,
                None,
                f"task-{_task_token(execution)}-resume.md",
                contract,
                execution=execution,
                role="paper_coach_resume",
            )
            read_dirs = [service.manifest.research_dir, local_stage / "inputs"]
        else:
            pointers, read_dirs = service.coach_context(request, draft_override)
            token = _task_token(execution)
            retry_diagnostics_path = (
                _stage_json_task_input(
                    local_stage,
                    None,
                    f"task-{token}-retry-diagnostics.json",
                    {"prior_attempt_diagnostics": list(execution.retry_feedback)},
                )
                if execution is not None and (execution.retry_feedback or retry_attempt)
                else None
            )
            raw_repositories = pointers["truth_repositories"]
            assert isinstance(raw_repositories, list)
            # A retry that still holds its native session already has the contract in the
            # conversation; it gets a follow-up naming what changed, not a rebuilt contract.
            resumed_retry = retrying and reusing_checkpoint
            current_contract_path = None
            current_prompt = None
            if not resumed_retry:
                human_request_path = _stage_task_input(
                    local_stage,
                    None,
                    f"task-{token}-human-request.txt",
                    request.message,
                )
                contract = PromptFactory.paper_coach_task_contract(
                    introduction_path=str(pointers["introduction"]),
                    graph_path=str(pointers["graph"]),
                    research_path=str(pointers["research_md"]),
                    repositories=[
                        {
                            "alias": str(item["alias"]),
                            "host": str(item["host"]),
                            "path": str(item["path"]),
                        }
                        for item in raw_repositories
                        if isinstance(item, dict)
                    ],
                    human_request_path=human_request_path,
                    retry_diagnostics_path=retry_diagnostics_path,
                    skill_pointers=skill_pointers,
                    invoked_skill_pointers=invoked_package_pointers(
                        skill_pointers,
                        workflow_ids=request.invoked_workflow_ids,
                        skill_ids=request.invoked_skill_ids,
                    ),
                    invoked_provider_skills=request.resolved_provider_skills,
                )
                current_contract_path, current_prompt = _stage_task_contract(
                    local_stage,
                    None,
                    f"task-{token}-{'base' if retry_attempt else 'initial'}.md",
                    contract,
                    execution=execution,
                    role="paper_coach_retry_base" if retry_attempt else "paper_coach",
                )
            if retrying:
                assert execution is not None
                assert retry_diagnostics_path is not None
                original_contract_path = _parent_task_contract_path(execution, local_stage, None)
                retry_contract = PromptFactory.continuation_task_contract(
                    original_contract_path=original_contract_path,
                    current_contract_path=current_contract_path,
                    diagnostics_path=retry_diagnostics_path,
                    mode="retry",
                    skill_pointers=skill_pointers if resumed_retry else None,
                    invoked_skill_pointers=invoked_package_pointers(
                        skill_pointers,
                        workflow_ids=request.invoked_workflow_ids,
                        skill_ids=request.invoked_skill_ids,
                    ),
                    invoked_provider_skills=request.resolved_provider_skills,
                )
                contract_path, prompt = _stage_task_contract(
                    local_stage,
                    None,
                    f"task-{token}-retry.md",
                    retry_contract,
                    execution=execution,
                    role="paper_coach_retry",
                )
            else:
                contract_path, prompt = current_contract_path, current_prompt
            read_dirs.extend([service.manifest.research_dir, local_stage / "inputs"])
    except (StateUnavailable, ValueError) as exc:
        yield _sse(AgentEvent(event="error", text=str(exc)))
        return
    model = (
        (None if existing.model == "provider-default" else existing.model)
        if existing
        else request.model
    )
    reasoning = existing.reasoning if existing else request.reasoning
    native_session_id = request.session_id
    actual_runtime_id = (
        execution.runtime_id
        if execution is not None
        else configured_runtime_id(profile.provider, profile.runtime)
    )
    completed = False
    provider_outcome = _ProviderOutcome(session_id=native_session_id)
    _record_agent_launch_receipt(
        execution,
        request,
        prompt=prompt,
        contract_path=contract_path,
        remote=False,
        resumed=reusing_checkpoint,
        continuation=continuation,
        extra={
            "surface": "paper_coach",
            "capability": "paper_readonly",
            "network_access": True,
            "launch_kind": "retry" if retry_attempt else "resume" if resuming else "initial",
        },
    )
    async with aclosing(
        launcher.stream(
            request.provider,
            prompt,
            cwd=service.manifest.research_dir,
            model=model,
            reasoning=reasoning,
            session_id=request.session_id,
            read_dirs=read_dirs,
            capability="paper_readonly",
            control=execution.control if execution is not None else None,
            binary=provider_binary,
            runtime_id=(execution.runtime_id or None) if execution is not None else None,
        )
    ) as stream:
        async for event in stream:
            if event.event == "provider_exit":
                try:
                    evidence = json.loads(event.text)
                except (json.JSONDecodeError, TypeError, ValueError):
                    evidence = {"unparsed": event.text[:400]}
                provider_outcome.exit_evidence = (
                    evidence if isinstance(evidence, dict) else {"unparsed": event.text[:400]}
                )
                _record_provider_exit(
                    execution,
                    provider_outcome,
                    workspace=local_stage,
                    remote_stage=None,
                )
                continue
            if event.event == "runtime":
                actual_runtime_id = event.text
            if event.event == "session" and event.session_id:
                native_session_id = event.session_id
            if event.event == "done":
                completed = True
            yield _sse(event)
    if completed and native_session_id:
        intro_hash, graph_revision, research_hash = service.pointer_hashes()
        now = datetime.now(UTC)
        paper.record_session(
            WritingSession(
                provider=request.provider,
                runtime_id=actual_runtime_id,
                native_session_id=native_session_id,
                execution_machine=profile.run_on,
                project_id=paper.project_id,
                title=existing.title if existing else request.message[:72],
                model=model or "provider-default",
                reasoning=reasoning,
                created_at=existing.created_at if existing else now,
                last_resumed_at=now,
                introduction_hash_examined=intro_hash,
                graph_revision_examined=graph_revision,
                research_md_hash_examined=research_hash,
            )
        )


def _paper_snapshot_path(data_dir: Path, project_id: str) -> Path:
    exports = data_dir / "paper-snapshots"
    exports.mkdir(parents=True, exist_ok=True)
    safe_project_id = re.sub(r"[^A-Za-z0-9._-]+", "_", project_id).strip("._")
    return exports / f"{(safe_project_id or 'project')[:80]}-introduction.md"
