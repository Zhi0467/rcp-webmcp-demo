from __future__ import annotations

import uuid
from collections.abc import Callable
from contextlib import suppress
from typing import TYPE_CHECKING

from rcp.agents.command_protocol import (
    ApplyArguments,
    CommandRequest,
    EpisodeCommandRequest,
    EpisodeControlArguments,
    ExperimentKickoffArguments,
    FinishCommandRequest,
    InboxArguments,
    InboxClearArguments,
    InboxCommandRequest,
    InboxHarvestArguments,
    MessageArguments,
    MessageCommandRequest,
    PauseCommandRequest,
    ResumeCommandRequest,
    SpawnArguments,
    StatusArguments,
    StopCommandRequest,
    WatchGraphArguments,
    WatchGraphCommandRequest,
)
from rcp.core.models import GraphState
from rcp.runs.auto_research import (
    AutoResearchCommandContext,
    AutoResearchCommandEffectResult,
    AutoResearchCommandEffects,
    AutoResearchCommandInvalid,
    AutoResearchCommandUnavailable,
    AutoResearchValidateCommand,
)
from rcp.runs.auto_research_admission import (
    ensure_auto_research_child_work_spawned,
    pause_auto_research_child_work,
    resume_auto_research_child_work,
    start_auto_research_child_work,
    stop_auto_research_child_work,
)
from rcp.runs.auto_research_delivery import (
    AutoResearchWatcherReadyHook,
    arm_auto_research_graph_condition,
    deliver_pending_auto_research_mail,
    reconcile_auto_research_graph_condition,
    record_auto_research_message,
)
from rcp.runs.auto_research_experiments import (
    AutoResearchExperimentCoordinator,
    AutoResearchExperimentLimitInvalid,
)
from rcp.service import GraphUpdateResult, RunRequest
from rcp.storage import (
    AgentTaskRecord,
    AppStore,
    AutoResearchApplyResultRecord,
    AutoResearchChildWorkRecord,
    AutoResearchExperimentAllowanceReached,
    AutoResearchFinishReceiptRecord,
    AutoResearchInboxReceiptRecord,
    AutoResearchMessageRecord,
    EpisodeNotRunning,
    GraphWatcherRecord,
)
from rcp.storage.auto_research_children import (
    AutoResearchInboxClearTooLarge,
    AutoResearchInboxHarvestTooLarge,
    AutoResearchInboxNoticeUnacknowledgeable,
    auto_research_inbox_projection,
)

if TYPE_CHECKING:
    from rcp.background import BackgroundAgentTasks


AutoResearchWorkerRequestFactory = Callable[
    [AutoResearchCommandContext, SpawnArguments, str, str],
    RunRequest,
]
AutoResearchGraphState = Callable[[], GraphState]
AutoResearchApplyPatch = Callable[
    [AutoResearchCommandContext, str, str],
    tuple[GraphUpdateResult | None, str | None, bool],
]
AutoResearchGraphApplied = Callable[[], None]


def auto_research_command_effects(
    *,
    store: AppStore,
    background: BackgroundAgentTasks,
    validate: AutoResearchValidateCommand,
    worker_request_factory: AutoResearchWorkerRequestFactory,
    graph_state: AutoResearchGraphState,
    execution_host: str,
    apply_patch: AutoResearchApplyPatch | None = None,
    on_graph_applied: AutoResearchGraphApplied | None = None,
    on_watcher_ready: AutoResearchWatcherReadyHook | None = None,
    experiment_coordinator: AutoResearchExperimentCoordinator | None = None,
) -> AutoResearchCommandEffects:
    """Bind staged auto_research commands to the existing durable runtime seams.

    Semantic Patch authority and worker profile selection remain injected. This
    module only composes already-authoritative graph, task, watcher, and mail
    primitives behind the staged command dispatcher.
    """

    if background.store is not store:
        raise ValueError("Auto-research command effects require one shared task store.")

    def apply(
        context: AutoResearchCommandContext,
        arguments: ApplyArguments,
        planned_apply_id: str,
    ) -> AutoResearchCommandEffectResult:
        if apply_patch is None:
            raise AutoResearchCommandUnavailable("In-turn Apply is unavailable.")
        snapshot = context.command_file
        if snapshot is None or snapshot.filename != arguments.patch_file:
            raise AutoResearchCommandUnavailable(
                "Apply lost its immutable Patch snapshot before validation."
            )
        graph_update, failure, correctable = apply_patch(
            context,
            snapshot.text,
            planned_apply_id,
        )
        if graph_update is None:
            message = failure or "The Patch could not be applied."
            live_revision, graph_path, research_path = _refreshed_paths(context, graph_state)
            rejected_update = (
                GraphUpdateResult(
                    status="rejected",
                    validation_messages=[message[:1_600]],
                    repairable=True,
                )
                if correctable
                else None
            )
            outcome = AutoResearchCommandEffectResult(
                status="invalid" if correctable else "unavailable",
                message=message,
                result={
                    "disposition": "invalid" if correctable else "unavailable",
                    "applied_revision": None,
                    "live_revision": live_revision,
                    "patch_sha256": snapshot.sha256,
                    "validation_messages": [message[:1_600]],
                    "graph_path": graph_path,
                    "research_path": research_path,
                    **(
                        {"graph_update": rejected_update.model_dump(mode="json")}
                        if rejected_update is not None
                        else {}
                    ),
                },
            )
            if not correctable:
                return outcome
            return _save_apply_result(store, context, planned_apply_id, snapshot.sha256, outcome)
        if graph_update.status not in {"none", "applied"}:
            raise AutoResearchCommandUnavailable(
                "In-turn Apply returned an unsupported graph disposition."
            )
        if context.consume_command_file is None:
            raise AutoResearchCommandUnavailable(
                "Apply completed but the exact Patch handoff could not be consumed."
            )
        consumed = context.consume_command_file(snapshot.filename, snapshot.sha256)
        if graph_update.status == "applied" and on_graph_applied is not None:
            on_graph_applied()
        live_revision, graph_path, research_path = _refreshed_paths(context, graph_state)
        disposition = "applied" if graph_update.status == "applied" else "valid_empty"
        if graph_update.status == "applied":
            message = f"Patch applied at revision {graph_update.applied_revision}."
        elif consumed:
            message = "The valid empty Patch was consumed without spending a revision."
        else:
            message = "The valid empty Patch spent no revision; a newer patch.json remains."
        outcome = AutoResearchCommandEffectResult(
            message=message,
            result={
                "disposition": disposition,
                "applied_revision": graph_update.applied_revision,
                "live_revision": live_revision,
                "patch_sha256": snapshot.sha256,
                "validation_messages": list(graph_update.validation_messages),
                "graph_path": graph_path,
                "research_path": research_path,
                "patch_consumed": consumed,
                "graph_update": graph_update.model_dump(mode="json"),
            },
        )
        return _save_apply_result(store, context, planned_apply_id, snapshot.sha256, outcome)

    def status(
        context: AutoResearchCommandContext,
        arguments: StatusArguments,
    ) -> AutoResearchCommandEffectResult:
        episode = store.episode(context.episode.episode_id)
        if episode is None or episode.mode != "auto_research":
            raise AutoResearchCommandUnavailable(
                "The Auto-research episode status is no longer available."
            )
        meter = store.episode_budget_meter(episode.episode_id)
        result: dict[str, object] = {
            "episode": {
                "episode_id": episode.episode_id,
                "status": episode.status,
                "ending": episode.ending,
                "stop_requested": episode.stop_requested_at is not None,
                "operational_invocations_remaining": meter.invocations_remaining,
            },
            "budget": meter.model_dump(mode="json"),
            "experiment_allowance": store.auto_research_experiment_allowance(
                episode.episode_id
            ).model_dump(mode="json"),
            "children": _child_registry_status(store, episode.episode_id),
            "lifecycle": _lifecycle_registry_status(store, episode.episode_id),
        }
        if arguments.worker_id is not None:
            route, leaf = _worker_leaf(store, context, arguments.worker_id)
            result["worker"] = _worker_status(route, leaf)
        if arguments.episode_id is not None:
            route = store.auto_research_child_experiment(arguments.episode_id)
            child = store.episode(arguments.episode_id)
            if (
                route is None
                or route.auto_research_episode_id != context.episode.episode_id
                or child is None
            ):
                raise AutoResearchCommandInvalid(
                    "Experiment status target is outside this Auto-research episode."
                )
            result["child_episode"] = {
                "episode_id": child.episode_id,
                "control_node_id": child.control_node_id,
                "status": child.status,
                "ending": child.ending,
                "route_state": route.state,
                "stop_requested": child.stop_requested_at is not None,
            }
        return AutoResearchCommandEffectResult(
            message="Auto-research status is current.",
            result=result,
        )

    def spawn(
        context: AutoResearchCommandContext,
        arguments: SpawnArguments,
        planned_worker_id: str,
    ) -> AutoResearchCommandEffectResult:
        snapshot = context.command_file
        if (
            snapshot is None
            or snapshot.kind != "instruction"
            or snapshot.filename != arguments.instruction_file
        ):
            raise AutoResearchCommandUnavailable(
                "Spawn lost its immutable instruction snapshot before admission."
            )
        try:
            request = worker_request_factory(
                context,
                arguments,
                snapshot.text,
                planned_worker_id,
            )
            _validate_worker_request(
                context,
                arguments,
                planned_worker_id,
                snapshot.text,
                request,
            )
            worker = start_auto_research_child_work(
                background,
                context.episode.episode_id,
                request,
                admitted_by_operation_id=context.task.operation_id,
                worker_id=planned_worker_id,
                instruction=snapshot.text,
                instruction_sha256=snapshot.sha256,
                admission_id=planned_worker_id,
            )
        except ValueError:
            if store.auto_research_child_work(planned_worker_id) is None:
                with suppress(KeyError, ValueError):
                    store.cancel_auto_research_child_admission(planned_worker_id)
            raise
        return AutoResearchCommandEffectResult(
            message="Ordinary Work was seated and queued.",
            result={
                "worker_id": planned_worker_id,
                "current_operation_id": worker.operation_id,
                "status": worker.status,
                "disposition": "created",
            },
        )

    def pause(
        context: AutoResearchCommandContext,
        worker_id: str,
    ) -> AutoResearchCommandEffectResult:
        route, _ = _worker_leaf(store, context, worker_id)
        try:
            paused = pause_auto_research_child_work(
                background,
                context.episode.episode_id,
                worker_id,
            )
        except EpisodeNotRunning as exc:
            raise AutoResearchCommandUnavailable(str(exc)) from exc
        return AutoResearchCommandEffectResult(
            message="Pause was requested for the current worker task attempt.",
            result=_worker_control_result(route, paused),
        )

    def resume(
        context: AutoResearchCommandContext,
        worker_id: str,
        planned_operation_id: str,
    ) -> AutoResearchCommandEffectResult:
        try:
            resumed = resume_auto_research_child_work(
                background,
                context.episode.episode_id,
                worker_id,
                operation_id=planned_operation_id,
            )
        except EpisodeNotRunning as exc:
            raise AutoResearchCommandUnavailable(str(exc)) from exc
        if resumed.disposition == "resume_unavailable":
            return AutoResearchCommandEffectResult(
                status="invalid",
                message=(
                    f"Worker Resume is unavailable because {resumed.reason}; "
                    "use Spawn with a fresh instruction file."
                ),
                result={
                    "disposition": resumed.disposition,
                    "worker_id": worker_id,
                    "current_operation_id": resumed.current_operation_id,
                    "reason": resumed.reason,
                    "replacement_command": resumed.replacement_command,
                },
            )
        assert resumed.task is not None
        return AutoResearchCommandEffectResult(
            message=(
                "The current worker task attempt is resuming from its exact saved "
                "session and allocation."
            ),
            result={
                "disposition": "resumed",
                "worker_id": worker_id,
                "current_operation_id": resumed.current_operation_id,
                "status": resumed.task.status,
            },
        )

    def stop(
        context: AutoResearchCommandContext,
        worker_id: str,
    ) -> AutoResearchCommandEffectResult:
        route, _ = _worker_leaf(store, context, worker_id)
        stopped = stop_auto_research_child_work(
            background,
            context.episode.episode_id,
            worker_id,
        )
        return AutoResearchCommandEffectResult(
            message="Stop was recorded and the current worker attempt is stopping gracefully.",
            result={
                **_worker_control_result(route, stopped),
                "disposition": "stopped",
            },
        )

    def message(
        context: AutoResearchCommandContext,
        arguments: MessageArguments,
        planned_message_id: str,
    ) -> AutoResearchCommandEffectResult:
        recipient_task_id = arguments.recipient_task_id or context.episode.root_operation_id
        if recipient_task_id is None:
            raise AutoResearchCommandUnavailable(
                "The Auto-research episode has no orchestrator mail recipient."
            )
        try:
            saved = record_auto_research_message(
                store,
                message_id=planned_message_id,
                episode_id=context.episode.episode_id,
                sender_role=context.request.role,
                sender_task_id=context.task.operation_id,
                authorized_by=None,
                recipient_task_id=recipient_task_id,
                control_node_id=context.request.control_node_id,
                body=arguments.body,
            )
        except EpisodeNotRunning as exc:
            raise AutoResearchCommandUnavailable(str(exc)) from exc
        started_operation_id = deliver_pending_auto_research_mail(
            background,
            episode_id=context.episode.episode_id,
            recipient_task_id=recipient_task_id,
        )
        canonical = store.auto_research_message(saved.message_id)
        if canonical is None:
            raise AutoResearchCommandUnavailable(
                "The persisted Auto-research message disappeared before delivery was recorded."
            )
        delivery_operation_id = canonical.delivery_operation_id
        return AutoResearchCommandEffectResult(
            message=(
                "The message was accepted and a delivery turn started."
                if delivery_operation_id is not None
                else (
                    "The message was queued behind an older pending delivery."
                    if started_operation_id is not None
                    else "The message was queued for the recipient's next delivery turn."
                )
            ),
            result=_message_command_result(
                canonical,
                delivery_operation_id=delivery_operation_id,
                disposition="created",
            ),
        )

    def watch_graph(
        context: AutoResearchCommandContext,
        arguments: WatchGraphArguments,
        planned_watcher_id: str,
    ) -> AutoResearchCommandEffectResult:
        watcher = arm_auto_research_graph_condition(
            store,
            context,
            arguments,
            watcher_id=planned_watcher_id,
            state=graph_state(),
            execution_host=execution_host,
            on_ready=on_watcher_ready,
        )
        return AutoResearchCommandEffectResult(
            message="The graph condition was armed.",
            result=_watcher_command_result(watcher, disposition="created"),
        )

    def episode(
        context: AutoResearchCommandContext,
        arguments: ExperimentKickoffArguments | EpisodeControlArguments,
        planned_episode_effect_id: str,
    ) -> AutoResearchCommandEffectResult:
        if experiment_coordinator is None:
            raise AutoResearchCommandUnavailable("Experiment episode control is unavailable.")
        if isinstance(arguments, ExperimentKickoffArguments):
            snapshot = context.command_file
            if arguments.goal_file is not None:
                if (
                    snapshot is None
                    or snapshot.kind != "goal"
                    or snapshot.filename != arguments.goal_file
                ):
                    raise AutoResearchCommandUnavailable(
                        "Experiment kickoff lost its immutable goal snapshot."
                    )
                goal = snapshot.text
                goal_sha256 = snapshot.sha256
            else:
                if snapshot is not None:
                    raise AutoResearchCommandUnavailable(
                        "Experiment kickoff received an unexpected command-file snapshot."
                    )
                goal = None
                goal_sha256 = None
            try:
                action = experiment_coordinator.kick_off(
                    auto_research_episode_id=context.episode.episode_id,
                    parent_operation_id=context.task.operation_id,
                    child_episode_id=planned_episode_effect_id,
                    node_id=arguments.node_id,
                    goal=goal,
                    goal_sha256=goal_sha256,
                    invocation_limit=arguments.invocation_limit,
                    admission_id=planned_episode_effect_id,
                )
            except AutoResearchExperimentLimitInvalid as exc:
                return AutoResearchCommandEffectResult(
                    status="invalid",
                    message=str(exc),
                    result={
                        "disposition": "limit_too_high",
                        "experiment_allowance": exc.allowance.model_dump(mode="json"),
                    },
                )
            except AutoResearchExperimentAllowanceReached as exc:
                return AutoResearchCommandEffectResult(
                    status="invalid",
                    message=(
                        "The Auto-research child Experiment allowance is exhausted; "
                        "no Experiment was started."
                    ),
                    result={
                        "disposition": "allowance_exhausted",
                        "experiment_allowance": exc.allowance.model_dump(mode="json"),
                    },
                )
        elif arguments.action == "stop":
            action = experiment_coordinator.stop(
                context.episode.episode_id,
                arguments.episode_id,
            )
        else:
            action = experiment_coordinator.resume(
                context.episode.episode_id,
                arguments.episode_id,
                operation_id=planned_episode_effect_id,
            )
        result = {
            "disposition": action.disposition,
            "episode_id": action.episode_id,
            "status": action.status,
            "experiment_allowance": action.allowance.model_dump(mode="json"),
        }
        if action.operation_id is not None:
            result["operation_id"] = action.operation_id
        if action.reason is not None:
            result["reason"] = action.reason
        if action.replacement_command is not None:
            result["replacement_command"] = action.replacement_command
        status = "invalid" if action.disposition == "resume_unavailable" else "ok"
        message = {
            "created": "Experiment episode was created and queued.",
            "replacement_pending": (
                "Experiment replacement was reserved while the prior episode stops gracefully."
            ),
            "stopping": "Experiment episode is stopping gracefully.",
            "stopped": "Experiment episode is stopped.",
            "cancelled": "Pending Experiment replacement was cancelled.",
            "existing": "The existing Experiment kickoff result was recovered.",
            "resumed": "Experiment episode resumed its exact saved allocation.",
            "resume_unavailable": (
                action.reason
                or "Experiment Resume is unavailable; kick off a fresh replacement episode."
            ),
        }[action.disposition]
        return AutoResearchCommandEffectResult(
            status=status,
            message=message,
            result=result,
        )

    def finish(
        context: AutoResearchCommandContext,
        planned_finish_effect_id: str,
    ) -> AutoResearchCommandEffectResult:
        actor_operation_id = _finish_actor_operation_id(store, context)
        try:
            receipt = store.guard_auto_research_finish(
                context.episode.episode_id,
                effect_id=planned_finish_effect_id,
                actor_operation_id=actor_operation_id,
            )
        except EpisodeNotRunning as exc:
            raise AutoResearchCommandUnavailable(str(exc)) from exc
        return _finish_command_result(receipt)

    def inbox(
        context: AutoResearchCommandContext,
        arguments: InboxArguments,
        planned_inbox_effect_id: str,
    ) -> AutoResearchCommandEffectResult:
        if not isinstance(arguments, (InboxHarvestArguments, InboxClearArguments)):
            raise AssertionError(f"unhandled inbox action: {arguments}")
        try:
            receipt = store.process_auto_research_lifecycle_inbox(
                context.episode.episode_id,
                effect_id=planned_inbox_effect_id,
                mode=arguments.action,
                acknowledged_by=(context.request.actor_operation_id or context.task.operation_id),
            )
        except AutoResearchInboxClearTooLarge as exc:
            return AutoResearchCommandEffectResult(
                status="invalid",
                message=str(exc),
                result={
                    "action": "clear",
                    "disposition": "response_too_large",
                },
            )
        except AutoResearchInboxHarvestTooLarge as exc:
            return AutoResearchCommandEffectResult(
                status="invalid",
                message=str(exc),
                result={
                    "action": "harvest",
                    "disposition": "notice_too_large",
                    "replacement_command": "inbox --key <new-key> --clear",
                },
            )
        except AutoResearchInboxNoticeUnacknowledgeable as exc:
            return AutoResearchCommandEffectResult(
                status="unavailable",
                message=str(exc),
                result={
                    "action": "harvest",
                    "disposition": "response_too_large",
                },
            )
        return _inbox_command_result(receipt)

    def reconcile_unknown(
        context: AutoResearchCommandContext,
        request: CommandRequest,
        planned_effect_id: str | None,
    ) -> AutoResearchCommandEffectResult | None:
        if isinstance(request, ResumeCommandRequest) and planned_effect_id is not None:
            if store.agent_task(planned_effect_id) is None:
                return None
            return resume(
                context,
                request.arguments.worker_id,
                planned_effect_id,
            )
        if isinstance(request, PauseCommandRequest):
            _, current = _worker_leaf(store, context, request.arguments.worker_id)
            if current.status in {"pausing", "paused"}:
                return pause(context, request.arguments.worker_id)
            return None
        if isinstance(request, StopCommandRequest):
            route, current = _worker_leaf(store, context, request.arguments.worker_id)
            if route.stop_requested_at is not None and current.status not in {
                "queued",
                "running",
            }:
                return stop(context, request.arguments.worker_id)
            return None
        if (
            isinstance(request, EpisodeCommandRequest)
            and isinstance(request.arguments, EpisodeControlArguments)
            and request.arguments.action == "resume"
            and planned_effect_id is not None
        ):
            if store.agent_task(planned_effect_id) is None:
                return None
            return episode(context, request.arguments, planned_effect_id)
        if (
            isinstance(request, EpisodeCommandRequest)
            and isinstance(request.arguments, EpisodeControlArguments)
            and request.arguments.action == "stop"
            and planned_effect_id is not None
            and experiment_coordinator is not None
        ):
            route = store.auto_research_child_experiment(request.arguments.episode_id)
            child = store.episode(request.arguments.episode_id)
            if route is None or (
                route.state not in {"cancelled", "terminal"}
                and (child is None or child.stop_requested_at is None)
            ):
                return None
            return episode(context, request.arguments, planned_effect_id)
        if request.verb == "apply" and planned_effect_id is not None:
            saved = store.auto_research_apply_result(planned_effect_id)
            if saved is None:
                return None
            snapshot = context.command_file
            if (
                saved.episode_id != context.episode.episode_id
                or saved.operation_id != context.task.operation_id
                or snapshot is None
                or snapshot.kind != "apply"
                or saved.patch_sha256 != snapshot.sha256
            ):
                raise AutoResearchCommandUnavailable(
                    "The interrupted Apply result does not match its immutable command."
                )
            disposition = (
                saved.result.get("result", {}).get("disposition")
                if isinstance(saved.result.get("result"), dict)
                else None
            )
            if disposition in {"applied", "valid_empty"}:
                if context.consume_command_file is None:
                    raise AutoResearchCommandUnavailable(
                        "The interrupted Apply cannot consume its exact Patch handoff."
                    )
                context.consume_command_file(snapshot.filename, snapshot.sha256)
            return AutoResearchCommandEffectResult.model_validate(saved.result)
        if isinstance(request, FinishCommandRequest) and planned_effect_id is not None:
            receipt = store.auto_research_finish_receipt(planned_effect_id)
            if receipt is None:
                return None
            if (
                receipt.episode_id != context.episode.episode_id
                or receipt.actor_operation_id != _finish_actor_operation_id(store, context)
            ):
                raise AutoResearchCommandUnavailable(
                    "The interrupted Finish result does not match its original command."
                )
            return _finish_command_result(receipt)
        if isinstance(request, InboxCommandRequest) and planned_effect_id is not None:
            receipt = store.auto_research_inbox_receipt(planned_effect_id)
            if receipt is None:
                return None
            if (
                receipt.episode_id != context.episode.episode_id
                or receipt.mode != request.arguments.action
                or receipt.acknowledged_by
                != (context.request.actor_operation_id or context.task.operation_id)
            ):
                raise AutoResearchCommandUnavailable(
                    "The interrupted inbox result does not match its original command."
                )
            return _inbox_command_result(receipt)
        if not _is_canonical_uuid(planned_effect_id):
            return None
        assert planned_effect_id is not None
        if isinstance(request, MessageCommandRequest):
            saved = store.auto_research_message(planned_effect_id)
            if saved is None or not _auto_research_message_matches(
                store,
                context,
                request.arguments,
                saved,
            ):
                return None
            return AutoResearchCommandEffectResult(
                message="The existing message was recovered after an interrupted delivery.",
                result=_message_command_result(
                    saved,
                    delivery_operation_id=saved.delivery_operation_id,
                    disposition="existing",
                ),
            )
        if isinstance(request, WatchGraphCommandRequest):
            watcher = reconcile_auto_research_graph_condition(
                store,
                context,
                request.arguments,
                watcher_id=planned_effect_id,
                execution_host=execution_host,
            )
            if watcher is None:
                return None
            return AutoResearchCommandEffectResult(
                message=("The existing graph condition was recovered after interrupted arming."),
                result=_watcher_command_result(watcher, disposition="existing"),
            )
        return None

    def seat_node_type(_project_id: str, node_id: str) -> str | None:
        node = graph_state().nodes.get(node_id)
        return node.type if node is not None else None

    def worker_lookup(
        context: AutoResearchCommandContext,
        worker_id: str,
    ) -> AgentTaskRecord:
        _, current = _worker_leaf(store, context, worker_id)
        return current

    def verify_spawn(
        context: AutoResearchCommandContext,
        arguments: SpawnArguments,
        worker_id: str,
    ) -> AgentTaskRecord:
        route, _ = _worker_leaf(store, context, worker_id)
        root = store.agent_task(route.root_operation_id)
        snapshot = context.command_file
        if (
            root is None
            or route.worker_id != worker_id
            or route.root_operation_id != worker_id
            or route.admitted_by_operation_id != context.task.operation_id
            or route.control_node_id != arguments.seat_node_id
            or snapshot is None
            or snapshot.kind != "instruction"
            or snapshot.filename != arguments.instruction_file
            or route.instruction != snapshot.text
            or route.instruction_sha256 != snapshot.sha256
            or root.kind != "node_chat"
            or root.project_id != context.episode.project_id
            or root.episode_id != context.episode.episode_id
            or root.parent_operation_id is not None
        ):
            raise AutoResearchCommandUnavailable(
                "Auto-research Spawn created an ordinary Work task with incorrect routing."
            )
        request = RunRequest.model_validate(root.request)
        _validate_worker_request(
            context,
            arguments,
            worker_id,
            snapshot.text,
            request,
        )
        return ensure_auto_research_child_work_spawned(
            background,
            context.episode.episode_id,
            worker_id,
            operation_id=root.operation_id,
            continuation="fresh",
        )

    return AutoResearchCommandEffects(
        validate=validate,
        apply=apply,
        status=status,
        spawn=spawn,
        pause=pause,
        resume=resume,
        stop=stop,
        message=message,
        watch_graph=watch_graph,
        episode=episode,
        inbox=inbox,
        finish=finish,
        seat_node_type=seat_node_type,
        reconcile_unknown=reconcile_unknown,
        worker_lookup=worker_lookup,
        verify_spawn=verify_spawn,
    )


def _refreshed_paths(
    context: AutoResearchCommandContext,
    graph_state: AutoResearchGraphState,
) -> tuple[int, str | None, str | None]:
    if context.refresh_command_state is None:
        return graph_state().revision, None, None
    return context.refresh_command_state()


def _save_apply_result(
    store: AppStore,
    context: AutoResearchCommandContext,
    apply_id: str,
    patch_sha256: str,
    outcome: AutoResearchCommandEffectResult,
) -> AutoResearchCommandEffectResult:
    stored = store.save_auto_research_apply_result(
        AutoResearchApplyResultRecord(
            apply_id=apply_id,
            episode_id=context.episode.episode_id,
            operation_id=context.task.operation_id,
            patch_sha256=patch_sha256,
            result=outcome.model_dump(mode="json"),
            created_at=store.now(),
        )
    )
    return AutoResearchCommandEffectResult.model_validate(stored.result)


def _inbox_command_result(
    receipt: AutoResearchInboxReceiptRecord,
) -> AutoResearchCommandEffectResult:
    result, message = auto_research_inbox_projection(
        receipt.mode,
        notice_ids=receipt.notice_ids,
        notices=receipt.notices,
    )
    return AutoResearchCommandEffectResult(message=message, result=result)


def _finish_command_result(
    receipt: AutoResearchFinishReceiptRecord,
) -> AutoResearchCommandEffectResult:
    compact_result: dict[str, object] = {
        "finish_receipt_id": receipt.effect_id,
        "disposition": receipt.disposition,
        "blocker_count": receipt.blocker_count,
        "digest": receipt.result_sha256,
    }
    if receipt.disposition == "blocked":
        count = receipt.blocker_count
        return AutoResearchCommandEffectResult(
            status="invalid",
            message=(
                f"Auto-research has {count} unsettled obligation"
                f"{'s' if count != 1 else ''}; settle them, then call finish with a new key."
            ),
            result=compact_result,
        )
    return AutoResearchCommandEffectResult(
        message=(
            "Auto-research accepted Finish; episode wrap-up will begin after every "
            "already-admitted turn settles."
        ),
        result=compact_result,
    )


def _finish_actor_operation_id(
    store: AppStore,
    context: AutoResearchCommandContext,
) -> str:
    binding = store.auto_research_actor_binding(context.task.operation_id)
    if (
        binding.episode_id != context.episode.episode_id
        or binding.role != "orchestrator"
        or context.request.role != "orchestrator"
    ):
        raise AutoResearchCommandUnavailable(
            "Guarded Finish requires the canonical Auto-research orchestrator actor."
        )
    return binding.actor_operation_id


def _child_registry_status(store: AppStore, episode_id: str) -> dict[str, object]:
    work_routes = store.auto_research_child_works(episode_id)
    experiment_routes = store.auto_research_child_experiments(episode_id)
    work: list[dict[str, object]] = []
    for route in work_routes[-16:]:
        current = store.agent_task(route.current_operation_id)
        work.append(
            {
                "worker_id": route.worker_id,
                "control_node_id": route.control_node_id,
                "current_operation_id": route.current_operation_id,
                "status": current.status if current is not None else "missing",
                "attempt": current.attempt if current is not None else None,
                "stop_requested": route.stop_requested_at is not None,
            }
        )
    experiments: list[dict[str, object]] = []
    for route in experiment_routes[-16:]:
        child = store.episode(route.child_episode_id)
        experiments.append(
            {
                "episode_id": route.child_episode_id,
                "control_node_id": route.control_node_id,
                "route_state": route.state,
                "status": child.status if child is not None else route.state,
                "ending": child.ending if child is not None else None,
                "replaces_episode_id": route.replaces_episode_id,
            }
        )
    admissions = store.pending_auto_research_child_admissions(episode_id)
    return {
        "work_count": len(work_routes),
        "work": work,
        "omitted_work_count": max(0, len(work_routes) - len(work)),
        "experiment_count": len(experiment_routes),
        "experiments": experiments,
        "omitted_experiment_count": max(0, len(experiment_routes) - len(experiments)),
        "pending_admission_count": len(admissions),
        "pending_admission_ids": [item.admission_id for item in admissions[:16]],
    }


def _lifecycle_registry_status(store: AppStore, episode_id: str) -> dict[str, object]:
    notices = store.auto_research_lifecycle_notices(episode_id)
    counts = {"pending": 0, "delivered": 0, "acknowledged": 0}
    for notice in notices:
        counts[notice.state] += 1
    pending = [notice.notice_id for notice in notices if notice.state == "pending"]
    return {
        "counts": counts,
        "pending_notice_ids": pending[:16],
        "omitted_pending_count": max(0, len(pending) - 16),
    }


def _is_canonical_uuid(value: str | None) -> bool:
    if value is None:
        return False
    try:
        return str(uuid.UUID(value)) == value
    except ValueError:
        return False


def _auto_research_message_matches(
    store: AppStore,
    context: AutoResearchCommandContext,
    arguments: MessageArguments,
    saved: AutoResearchMessageRecord,
) -> bool:
    if context.request.role not in {"orchestrator", "worker"}:
        return False
    recipient_task_id = arguments.recipient_task_id or context.episode.root_operation_id
    if recipient_task_id is None:
        return False
    try:
        sender = store.auto_research_actor_binding(context.task.operation_id)
        recipient = store.auto_research_actor_binding(recipient_task_id)
    except KeyError:
        return False
    expected_actor_id = context.request.actor_operation_id or context.task.operation_id
    if (
        sender.episode_id != context.episode.episode_id
        or sender.actor_operation_id != expected_actor_id
        or sender.role != context.request.role
        or store.auto_research_invocation_role(context.task.operation_id) != sender.role
        or recipient.episode_id != context.episode.episode_id
        or recipient.actor_operation_id != recipient_task_id
    ):
        return False
    if sender.role == "worker":
        if (
            recipient.role != "orchestrator"
            or recipient_task_id != context.episode.root_operation_id
        ):
            return False
    elif recipient.role != "worker":
        return False
    return (
        saved.episode_id == context.episode.episode_id
        and saved.sender_role == sender.role
        and saved.sender_task_id == context.task.operation_id
        and saved.authorized_by is None
        and saved.recipient_task_id == recipient_task_id
        and saved.control_node_id == context.request.control_node_id
        and saved.body == arguments.body
    )


def _message_command_result(
    saved: AutoResearchMessageRecord,
    *,
    delivery_operation_id: str | None,
    disposition: str,
) -> dict[str, object]:
    return {
        "message_id": saved.message_id,
        "recipient_task_id": saved.recipient_task_id,
        "delivery_operation_id": delivery_operation_id,
        "delivery": "started" if delivery_operation_id is not None else "pending",
        "graph_authority": "none",
        "epistemic_status": "hearsay",
        "disposition": disposition,
    }


def _watcher_command_result(
    watcher: GraphWatcherRecord,
    *,
    disposition: str,
) -> dict[str, object]:
    completed_immediately = (
        watcher.status == "completed"
        and watcher.completed_at is not None
        and watcher.completed_at == watcher.created_at
        and watcher.last_evaluated_at == watcher.created_at
    )
    return {
        "watcher_id": watcher.watcher_id,
        "status": watcher.status,
        "completed_immediately": completed_immediately,
        "disposition": disposition,
    }


def _validate_worker_request(
    context: AutoResearchCommandContext,
    arguments: SpawnArguments,
    planned_worker_id: str,
    instruction: str,
    request: RunRequest,
) -> None:
    if (
        request.mode != "work"
        or request.trigger != "orchestrator"
        or request.patch_kind != "work"
        or request.chat_scope != "node"
        or request.node_id != arguments.seat_node_id
        or request.message != instruction
        or request.chat_id != planned_worker_id
    ):
        raise AutoResearchCommandInvalid(
            "The resolved worker request changed its ordinary Work mode, seat, or instruction."
        )
    if request.provider is None or request.run_on is None:
        raise AutoResearchCommandUnavailable(
            "The Auto-research worker profile did not resolve a provider and execution machine."
        )
    if request.session_id is not None or request.watcher_ids or request.result_view is not None:
        raise AutoResearchCommandInvalid(
            "A newly seated Auto-research worker must start with a fresh session and no wake state."
        )


def _worker_leaf(
    store: AppStore,
    context: AutoResearchCommandContext,
    worker_id: str,
) -> tuple[AutoResearchChildWorkRecord, AgentTaskRecord]:
    route = store.auto_research_child_work(worker_id)
    if route is None or route.episode_id != context.episode.episode_id:
        raise AutoResearchCommandInvalid(
            "The worker control target is outside this Auto-research episode."
        )
    leaf = store.agent_task(route.current_operation_id)
    if leaf is None:
        raise AutoResearchCommandUnavailable(
            "The Auto-research worker's current task attempt is no longer available."
        )
    return route, leaf


def _worker_status(
    route: AutoResearchChildWorkRecord,
    leaf: AgentTaskRecord,
) -> dict[str, object]:
    return {
        "worker_id": route.worker_id,
        "current_operation_id": leaf.operation_id,
        "control_node_id": route.control_node_id,
        "status": leaf.status,
        "status_message": leaf.status_message[:2_000],
        "stop_requested": route.stop_requested_at is not None,
        "can_pause": leaf.can_pause,
        "can_resume": leaf.can_resume,
    }


def _worker_control_result(
    route: AutoResearchChildWorkRecord,
    task: AgentTaskRecord,
) -> dict[str, object]:
    return {
        "worker_id": route.worker_id,
        "current_operation_id": task.operation_id,
        "status": task.status,
    }
