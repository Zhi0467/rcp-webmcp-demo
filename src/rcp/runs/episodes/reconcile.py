from __future__ import annotations

import logging
from contextlib import suppress
from typing import TYPE_CHECKING

from rcp.runs.auto_research import (
    AutoResearchEndingSignal,
    AutoResearchRunRequest,
    auto_research_exhaustion_signal,
    auto_research_wrapup_spec,
    settle_auto_research_stop,
)
from rcp.runs.auto_research_delivery import (
    reconcile_pending_auto_research_lifecycle,
    reconcile_pending_auto_research_mail,
)
from rcp.runs.auto_research_recovery import (
    reconcile_auto_research_task_settlement,
    reconcile_due_auto_research_recoveries,
)
from rcp.runs.episodes.report import start_episode_report
from rcp.runs.episodes.wrapup import begin_episode_report_wrapup
from rcp.runs.experiment_loop import (
    experiment_loop_launch_failure_diagnostic,
    experiment_loop_operational_ending_wrapup_spec,
    experiment_loop_wrapup_spec,
)
from rcp.service import RunRequest
from rcp.storage import ACTIVE_AGENT_TASK_STATUSES, AgentTaskRecord, AppStore, EpisodeRecord

if TYPE_CHECKING:
    from rcp.background import AgentTaskExecution, BackgroundAgentTasks


class EpisodeReconciler:
    """Settle episode tasks, endings, recovery, and hidden report admission."""

    def __init__(
        self,
        store: AppStore,
        background: BackgroundAgentTasks,
        *,
        logger: logging.Logger,
    ) -> None:
        self.store = store
        self.background = background
        self.logger = logger

    def _has_unsettled_visible_episode_task(self, episode_id: str) -> bool:
        """Whether already-admitted visible work still owns an unfinished turn."""

        return any(
            task.visible and task.status in {*ACTIVE_AGENT_TASK_STATUSES, "paused"}
            for task in self.store.episode_tasks(episode_id, include_hidden=True)
        )

    def reconcile_auto_research_wrapup(
        self,
        signal: AutoResearchEndingSignal,
        *,
        source: str,
        operation_id: str | None = None,
    ) -> bool:
        """Admit the shared hidden report only after Auto-research is quiescent."""

        if self._has_unsettled_visible_episode_task(
            signal.episode_id
        ) or not self.store.auto_research_is_quiescent(signal.episode_id):
            return False
        try:
            admission = begin_episode_report_wrapup(
                self.store,
                auto_research_wrapup_spec(self.store, signal),
            )
            if admission.launchable:
                start_episode_report(self.background, signal.episode_id)
            return True
        except Exception as exc:
            self.logger.warning(
                "Could not reconcile episode report for %s after %s: %s",
                signal.episode_id,
                source,
                exc,
            )
            if operation_id is not None:
                with suppress(Exception):
                    self.store.record_agent_task_receipt(
                        operation_id,
                        "episode_report_reconciliation_failed",
                        {
                            "episode_id": signal.episode_id,
                            "source": source,
                            "exception_type": type(exc).__name__,
                            "detail": str(exc),
                        },
                        tier="diagnostic",
                    )
            return False

    def reconcile_auto_research_episode(
        self,
        episode_id: str,
        *,
        source: str,
        operation_id: str | None = None,
        signal: AutoResearchEndingSignal | None = None,
    ) -> None:
        episode = self.store.episode(episode_id)
        if episode is None or episode.mode != "auto_research":
            return
        if episode.stop_requested_at is not None:
            settle_auto_research_stop(self.store, episode_id)
            return
        if episode.wrapup_state in {"ready", "failed", "skipped", "legacy_unavailable"}:
            return
        if episode.wrapup_state in {"pending", "running"}:
            if self._has_unsettled_visible_episode_task(episode_id):
                return
            try:
                start_episode_report(self.background, episode_id)
            except Exception as exc:
                self.logger.warning(
                    "Could not restart episode report for %s after %s: %s",
                    episode_id,
                    source,
                    exc,
                )
                if operation_id is not None:
                    with suppress(Exception):
                        self.store.record_agent_task_receipt(
                            operation_id,
                            "episode_report_reconciliation_failed",
                            {
                                "episode_id": episode_id,
                                "source": source,
                                "exception_type": type(exc).__name__,
                                "detail": str(exc),
                            },
                            tier="diagnostic",
                        )
            return
        if signal is None and episode.ending is None:
            if (
                self.store.auto_research_is_quiescent(episode_id)
                and episode.invocations_used >= episode.invocation_ceiling
            ):
                signal = auto_research_exhaustion_signal(
                    self.store,
                    episode_id,
                    diagnostic="The authorized operational invocation ceiling was exhausted.",
                )
            else:
                return
        if signal is None:
            assert episode.ending is not None and episode.ending != "stopped"
            signal = AutoResearchEndingSignal(
                episode_id=episode_id,
                ending=episode.ending,
                partial=episode.ending != "completed",
                diagnostic=episode.ending_diagnostic,
            )
        self.reconcile_auto_research_wrapup(
            signal,
            source=source,
            operation_id=operation_id,
        )

    def settle_auto_research_task(
        self,
        auto_request: AutoResearchRunRequest,
        execution: AgentTaskExecution,
    ) -> None:
        """Own the Auto-research half of task settlement, Stop fence included.

        The engine used to load the episode, settle a requested Stop, and call a
        second Auto-research-only callback.  Settlement crosses one boundary now,
        so that sequence lives with the owner instead.  Its failure is recorded
        under its own diagnostic and never replaces the task verdict.
        """

        try:
            episode = self.store.episode(auto_request.episode_id)
            if episode is None:
                return
            if episode.stop_requested_at is not None:
                settled = settle_auto_research_stop(self.store, episode.episode_id)
                if settled is not None:
                    episode = settled
            self.reconcile_auto_research_task(episode, auto_request, execution)
        except Exception as exc:
            with suppress(Exception):
                self.store.record_agent_task_receipt(
                    execution.operation_id,
                    "auto_research_task_settled_callback_failed",
                    {"exception_type": type(exc).__name__},
                    tier="diagnostic",
                )

    def reconcile_auto_research_task(
        self,
        episode: EpisodeRecord,
        auto_request: AutoResearchRunRequest,
        execution: AgentTaskExecution,
    ) -> None:
        signal = reconcile_auto_research_task_settlement(
            self.background,
            episode,
            auto_request,
            execution,
        )
        current = self.store.episode(episode.episode_id)
        if current is None:
            return
        if current.stop_requested_at is None and current.ending is None:
            try:
                reconcile_pending_auto_research_lifecycle(
                    self.background,
                    episode_id=current.episode_id,
                )
                reconcile_pending_auto_research_mail(
                    self.background,
                    episode_id=current.episode_id,
                )
            except Exception as exc:
                self.logger.warning(
                    "Could not deliver pending Auto-research lifecycle or mail after task %s: %s",
                    execution.operation_id,
                    exc,
                )
                with suppress(Exception):
                    self.store.record_agent_task_receipt(
                        execution.operation_id,
                        "auto_research_delivery_retry_failed",
                        {"exception_type": type(exc).__name__, "detail": str(exc)},
                        tier="diagnostic",
                    )
        self.reconcile_auto_research_episode(
            current.episode_id,
            source=f"task {execution.operation_id} settlement",
            operation_id=execution.operation_id,
            signal=signal,
        )

    def reconcile_auto_research_recovery_pass(self) -> None:
        reconcile_due_auto_research_recoveries(self.background)

    def latest_experiment_leaf(self, episode_id: str) -> AgentTaskRecord | None:
        tasks = self.store.episode_tasks(episode_id)
        parents = {
            task.parent_operation_id for task in tasks if task.parent_operation_id is not None
        }
        leaves = [task for task in tasks if task.operation_id not in parents]
        return leaves[-1] if leaves else None

    def reconcile_experiment_episode(
        self,
        episode_id: str,
        *,
        source: str,
        operation_id: str | None = None,
    ) -> None:
        """Fence one Experiment ending, retire its observers, and start shared wrap-up."""

        episode = self.store.episode(episode_id)
        if episode is None or episode.mode != "experiment_loop":
            return
        if episode.stop_requested_at is not None:
            if episode.control_node_id is not None:
                self.store.settle_experiment_loop_stop(
                    episode.project_id,
                    episode.control_node_id,
                    episode_id=episode.episode_id,
                    graph_target=episode.graph_target,
                )
            return
        if episode.wrapup_state in {"ready", "failed", "skipped", "legacy_unavailable"}:
            return
        state = self.store.experiment_episode(episode_id)
        if state is None:
            return
        ending_signal = self.store.experiment_episode_ending_signal(episode_id)
        if ending_signal is not None:
            continuation_operation_id, signal = ending_signal
            spec = experiment_loop_wrapup_spec(continuation_operation_id, signal)
        else:
            continuation = (
                self.store.agent_task(operation_id)
                if operation_id is not None
                else self.latest_experiment_leaf(episode_id)
            )
            if continuation is None:
                return
            try:
                request = RunRequest.model_validate(continuation.request)
            except ValueError:
                return
            ending = episode.ending
            diagnostic = episode.ending_diagnostic
            if ending is None:
                if (
                    continuation.status == "succeeded"
                    and episode.invocations_used >= episode.invocation_ceiling
                ):
                    ending = "exhausted"
                    diagnostic = "The authorized operational invocation ceiling was exhausted."
                elif continuation.status == "failed":
                    if not continuation.native_session_id:
                        # No session was ever bound, so there is no lineage to
                        # classify and nothing to resume.
                        ending = "failed"
                        diagnostic = experiment_loop_launch_failure_diagnostic(continuation)
                    else:
                        recovery_problem = self.store.experiment_episode_recovery_context_problem(
                            continuation.operation_id
                        )
                        if recovery_problem is None:
                            return
                        ending = "failed"
                        diagnostic = recovery_problem
                else:
                    return
            if ending not in {"exhausted", "failed"} or not diagnostic:
                return
            spec = experiment_loop_operational_ending_wrapup_spec(
                continuation=continuation,
                request=request,
                episode=state,
                ending=ending,
                diagnostic=diagnostic,
            )
        try:
            self.store.fence_episode_ending(
                spec.episode_id,
                spec.ending,
                diagnostic=spec.diagnostic,
            )
            if not self.store.settle_experiment_episode_wrapup(spec.episode_id):
                return
            admission = begin_episode_report_wrapup(self.store, spec)
            if admission.launchable:
                start_episode_report(self.background, spec.episode_id)
        except Exception as exc:
            self.logger.warning(
                "Could not reconcile Experiment episode %s after %s: %s",
                episode_id,
                source,
                exc,
            )
            if operation_id is not None:
                with suppress(Exception):
                    self.store.record_agent_task_receipt(
                        operation_id,
                        "episode_report_reconciliation_failed",
                        {
                            "episode_id": episode_id,
                            "source": source,
                            "exception_type": type(exc).__name__,
                            "detail": str(exc),
                        },
                        tier="diagnostic",
                    )
