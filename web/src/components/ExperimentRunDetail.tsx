import { ExternalLink, FlaskConical } from "lucide-react";
import { useState, type ReactNode } from "react";
import {
  type ExperimentRun,
  type ExperimentWatcherGroup,
  type ExperimentWatcherItem,
  experimentRecommendation,
  graphConditionLabel,
  isExternalWatcherRecord,
  watcherLastObservedAt,
} from "../runProjection";
import { currentExperimentGuidance, experimentGuidanceDetail } from "../experimentGuidance";
import type { ExperimentLoopHealth, WatcherRecord } from "../types";
import { EpisodeReportLink } from "./EpisodeReportLink";

const healthLabels: Record<ExperimentLoopHealth, string> = {
  starting: "Starting",
  agent_active: "Agent active",
  waiting_on_watchers: "Waiting on watchers",
  degraded: "Watcher degraded",
  stopping: "Stopping gracefully",
  wrapping_up: "Wrapping up visualization and report",
  failed: "Failed",
  human_stopped: "Human-stopped",
  paused_at_limit: "Paused at invocation limit",
  needs_action: "Needs action",
  completed: "Completed",
};

const healthTones: Record<ExperimentLoopHealth, string> = {
  starting: "running",
  agent_active: "running",
  waiting_on_watchers: "waiting",
  degraded: "degraded",
  stopping: "stopping",
  wrapping_up: "running",
  failed: "degraded",
  human_stopped: "stopped",
  paused_at_limit: "paused",
  needs_action: "actionable",
  completed: "completed",
};

export function experimentHealthLabel(health: ExperimentLoopHealth): string {
  return healthLabels[health];
}

export function experimentHealthTone(health: ExperimentLoopHealth): string {
  return healthTones[health];
}

interface Props {
  run: ExperimentRun;
  runBusy: boolean;
  runDisabled: boolean;
  stopBusy: boolean;
  recoveryBusy: boolean;
  watcherCheckBusyId: string | null;
  providerLabel?: string;
  conversation?: ReactNode;
  allowStart?: boolean;
  startDisabled?: boolean;
  onRun: () => void;
  onStopLoop: () => void;
  onRecover: (action: "resume" | "retry") => void;
  onSwitchProvider: () => void;
  onCheckWatcher: (watcherId: string) => void;
  episodeReportHref: (episodeId: string) => string;
}

export function ExperimentRunDetail({
  run,
  runBusy,
  runDisabled,
  stopBusy,
  recoveryBusy,
  watcherCheckBusyId,
  providerLabel,
  conversation,
  allowStart = true,
  startDisabled = false,
  onRun,
  onStopLoop,
  onRecover,
  onSwitchProvider,
  onCheckWatcher,
  episodeReportHref,
}: Props) {
  const [reportOpenError, setReportOpenError] = useState<string | null>(null);
  const { node, control, taskGroup, currentTask, health } = run;
  const operational = control.operational;
  const session = operational.session;
  const episode = control.episode;
  const stopUnsettled = control.stop_pending;
  const currentOperationId =
    currentTask?.operation_id ??
    operational?.current_operation_id ??
    taskGroup?.latest.operation_id;
  const attempts = node.attempts ?? [];
  const completionCriteria = node.completion_criteria ?? [];
  const stoppedWatcherItems = run.watcherItems.filter(watcherItemIsStopped);
  const currentWatcherItems = run.watcherItems.filter((item) => !watcherItemIsStopped(item));
  const stoppedWatcherCount = stoppedWatcherItems.reduce(watcherItemCount, 0);
  const currentWatcherCount = currentWatcherItems.reduce(watcherItemCount, 0);
  const lastActivity = formatMoment(
    currentTask?.last_activity_at ?? operational?.current_last_activity_at,
  );
  const recoveryAction = currentTask ? control.task_control : null;
  const recoveryProvider =
    providerLabel ||
    capitalize(String(currentTask?.request.provider || session?.provider || "agent"));
  const canSwitchProvider = Boolean(currentTask && control.can_switch_provider);
  const showStop = Boolean(control.episode_id && (stopBusy || control.can_stop));
  const baseRecommendation = experimentRecommendation(run);
  const recommendation =
    !allowStart && baseRecommendation.step === "start_episode"
      ? { step: "review" as const, label: "Review the owning Auto-research episode" }
      : startDisabled && baseRecommendation.step === "start_episode"
        ? { step: "review" as const, label: "Sync staged changes before starting" }
        : baseRecommendation;
  const summaryGuidance = experimentGuidanceDetail(node, "current_summary");
  const nextActionGuidance = experimentGuidanceDetail(node, "next_action");
  const currentSummary = currentExperimentGuidance(node, "current_summary");
  const currentNextAction = currentExperimentGuidance(node, "next_action");
  const watcherActionsDisabled =
    runDisabled || runBusy || stopBusy || recoveryBusy || watcherCheckBusyId !== null;

  return (
    <div className={`experiment-run-detail ${healthTones[health]}`}>
      <div className="experiment-run-topline">
        <div
          className={`experiment-run-health ${healthTones[health]}`}
          role="status"
          aria-live="polite"
          aria-atomic="true"
        >
          <strong>{healthLabels[health]}</strong>
        </div>
        <div className="experiment-run-actions" aria-label="Experiment loop actions">
          {recoveryAction && (
            <button
              type="button"
              className="button primary compact experiment-recovery-button"
              disabled={runDisabled || recoveryBusy || stopBusy}
              aria-busy={recoveryBusy}
              onClick={() => onRecover(recoveryAction)}
            >
              {recoveryBusy
                ? recoveryAction === "resume"
                  ? `Resuming ${recoveryProvider}…`
                  : `Retrying ${recoveryProvider}…`
                : recoveryAction === "resume"
                  ? `Resume ${recoveryProvider}`
                  : `Retry ${recoveryProvider}`}
            </button>
          )}
          {canSwitchProvider && (
            <button
              type="button"
              className="button compact"
              disabled={runDisabled || recoveryBusy || stopBusy}
              onClick={onSwitchProvider}
            >
              Switch provider…
            </button>
          )}
          {showStop && (
            <button
              type="button"
              className="button compact experiment-stop-loop"
              disabled={runDisabled || stopBusy || recoveryBusy || stopUnsettled}
              onClick={onStopLoop}
            >
              {stopBusy || stopUnsettled ? "Stopping" : "Stop loop"}
            </button>
          )}
          {control.can_open_report && control.report_episode_id && episode && (
            <EpisodeReportLink
              className="button primary compact"
              href={episodeReportHref(control.report_episode_id)}
              projectId={episode.project_id}
              episodeId={control.report_episode_id}
              onOpenError={setReportOpenError}
            >
              <ExternalLink size={12} aria-hidden="true" /> Open report
            </EpisodeReportLink>
          )}
          {allowStart && !control.node_closed && (
            <button
              type="button"
              className="button primary compact experiment-run-button"
              disabled={
                runDisabled || startDisabled || runBusy || stopUnsettled || !control.can_start
              }
              onClick={onRun}
              aria-describedby={control.reasons.length ? `${node.id}-run-requirements` : undefined}
            >
              <FlaskConical size={13} aria-hidden="true" />{" "}
              {runBusy ? "Starting" : control.episode_id ? "Start new episode" : "Start episode"}
            </button>
          )}
        </div>
      </div>

      <div className={`experiment-run-recommendation ${recommendation.step}`}>
        <span className="eyebrow">Recommended next step</span>
        <strong>{recommendation.label}</strong>
      </div>

      {episode?.ending_diagnostic && (
        <div className="campaign-run-error" role="alert">
          {episode.ending_diagnostic}
        </div>
      )}

      {/* The report is a deliverable of an ended episode, so its failure is reported
          after the reason the episode ended and never in place of it. */}
      {episode?.wrapup_state === "failed" && (
        <div className="campaign-run-note">
          Report generation error: {episode.wrapup_error || "The report could not be generated."}
        </div>
      )}

      {reportOpenError && (
        <div className="campaign-run-error" role="alert">
          {reportOpenError}
        </div>
      )}

      <p className="experiment-run-meta">
        {control && (
          <span>
            <span className="eyebrow">Invocation</span>
            {operational?.current_invocation ??
              taskInvocation(currentTask) ??
              control.invocations_used}{" "}
            / {control.invocation_ceiling}
          </span>
        )}
        {control?.episode_id && (
          <span>
            <span className="eyebrow">Next episode limit</span>
            {node.invocation_ceiling}
          </span>
        )}
        {lastActivity !== "—" && (
          <span>
            <span className="eyebrow">Active</span>
            {lastActivity}
          </span>
        )}
      </p>

      {control && !control.node_closed && control.reasons.length > 0 && (
        <ul
          id={`${node.id}-run-requirements`}
          className="experiment-gate-reasons"
          aria-label="Run requirements"
        >
          {control.reasons.map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
        </ul>
      )}

      <section className="experiment-run-block">
        <div className="experiment-run-block-heading">
          <h4>Research summary</h4>
        </div>
        <p className="experiment-run-prose">
          {String(currentSummary || node.objective || "No current summary recorded")}
        </p>
        {currentNextAction && (
          <p className="experiment-run-prose experiment-run-next-action">
            <span className="eyebrow">Next action</span>
            {currentNextAction}
          </p>
        )}
        {summaryGuidance.status === "stale" && (
          <p className="experiment-run-prose">
            <span className="eyebrow">{summaryGuidance.label}</span>
            {summaryGuidance.text}
          </p>
        )}
        {nextActionGuidance.status === "stale" && (
          <p className="experiment-run-prose experiment-run-next-action">
            <span className="eyebrow">{nextActionGuidance.label}</span>
            {nextActionGuidance.text}
          </p>
        )}
      </section>

      {(control?.decision_drift ?? []).length > 0 && (
        <ul className="experiment-run-drift" aria-label="Decision drift">
          {(control?.decision_drift ?? []).map((drift) => (
            <li key={drift.decision_id}>
              {`${drift.decision_id} moved to ${drift.current_option ?? drift.current_status ?? "an unavailable state"} after this episode was pinned to ${drift.pinned_option}.`}
            </li>
          ))}
        </ul>
      )}

      <Fold title="Watchers" count={currentWatcherCount} defaultOpen>
        {currentWatcherItems.length === 0 ? (
          <p className="experiment-run-empty">
            {stoppedWatcherCount > 0
              ? "No current watchers."
              : "No detached work has been handed off."}
          </p>
        ) : (
          <ul className="experiment-run-watchers" aria-label="Experiment watchers">
            <WatcherItems
              items={currentWatcherItems}
              watcherCheckBusyId={watcherCheckBusyId}
              actionsDisabled={watcherActionsDisabled}
              onCheckWatcher={onCheckWatcher}
            />
          </ul>
        )}
        {stoppedWatcherCount > 0 && (
          <Fold title="Stopped watchers" count={stoppedWatcherCount} nested>
            <ul className="experiment-run-watchers" aria-label="Stopped experiment watchers">
              <WatcherItems
                items={stoppedWatcherItems}
                watcherCheckBusyId={watcherCheckBusyId}
                actionsDisabled={watcherActionsDisabled}
                onCheckWatcher={onCheckWatcher}
              />
            </ul>
          </Fold>
        )}
      </Fold>

      {attempts.length > 0 && (
        <Fold title="Semantic attempts" count={attempts.length}>
          <ol className="experiment-run-attempts" aria-label="Semantic attempts">
            {attempts.map((attempt) => (
              <li key={attempt.id}>
                <span className="experiment-run-attempt-seq">
                  {String(attempt.sequence).padStart(2, "0")}
                </span>
                <span className="experiment-run-attempt-copy">
                  <strong>{attempt.purpose}</strong>
                  <span>{attempt.outcome || attempt.failure_reason || "No outcome recorded"}</span>
                  {attempt.job_refs.length > 0 && (
                    <span className="mono experiment-run-breakable">
                      {attempt.job_refs.join(", ")}
                    </span>
                  )}
                </span>
                <span className={`status-pill ${attempt.status}`}>{attempt.status}</span>
              </li>
            ))}
          </ol>
        </Fold>
      )}

      {(session || control?.episode_id || currentOperationId) && (
        <Fold title="Execution">
          <Facts
            entries={[
              {
                label: "Agent",
                value: joinFacts([
                  session?.provider,
                  session?.model || "provider default",
                  session?.reasoning,
                ]),
              },
              { label: "Machine", value: joinFacts([session?.run_on, session?.execution_host]) },
              {
                label: "Truth scope",
                value: session?.run_truth_scope?.join(", "),
                breakable: true,
              },
              {
                label: "Native continuity",
                value: session
                  ? joinFacts([
                      session.native_session_bound ? "Bound" : "Not bound",
                      session.diagnostic,
                    ])
                  : null,
              },
              {
                label: "Last task error",
                value: currentTask?.error,
                breakable: true,
              },
              { label: "Episode", value: control?.episode_id, mono: true, breakable: true },
              { label: "Current task", value: currentOperationId, mono: true, breakable: true },
            ]}
          />
        </Fold>
      )}

      {completionCriteria.length > 0 && (
        <Fold title="Completion criteria" count={completionCriteria.length}>
          <ul className="experiment-run-list">
            {completionCriteria.map((criterion) => (
              <li key={criterion}>{criterion}</li>
            ))}
          </ul>
        </Fold>
      )}

      {(control?.governing_decisions ?? []).length > 0 && (
        <Fold title="Governing decisions" count={(control?.governing_decisions ?? []).length}>
          <ul className="experiment-run-list">
            {(control?.governing_decisions ?? []).map((pin) => (
              <li key={pin.decision_id}>
                <span className="mono">{pin.decision_id}</span> · r{pin.decision_revision} ·{" "}
                {pin.selected_option}
              </li>
            ))}
          </ul>
        </Fold>
      )}

      {conversation && (
        <section className="experiment-run-conversation" aria-label="Run conversation">
          <div className="experiment-run-block-heading">
            <h4>Conversation</h4>
          </div>
          {conversation}
        </section>
      )}
    </div>
  );
}

function Fold({
  title,
  count,
  defaultOpen,
  nested,
  children,
}: {
  title: string;
  count?: number;
  defaultOpen?: boolean;
  nested?: boolean;
  children: ReactNode;
}) {
  return (
    <details className={nested ? "experiment-fold nested" : "experiment-fold"} open={defaultOpen}>
      <summary>
        <span className="experiment-fold-title">{title}</span>
        {count !== undefined && <span className="experiment-fold-count">{count}</span>}
      </summary>
      <div className="experiment-fold-body">{children}</div>
    </details>
  );
}

interface Fact {
  label: string;
  value: ReactNode;
  mono?: boolean;
  breakable?: boolean;
  className?: string;
}

function Facts({ entries, className }: { entries: Fact[]; className?: string }) {
  const present = entries.filter(
    (entry) =>
      entry.value !== null &&
      entry.value !== undefined &&
      entry.value !== "" &&
      entry.value !== "—",
  );
  if (present.length === 0) return null;
  return (
    <dl className={className ? `experiment-run-facts ${className}` : "experiment-run-facts"}>
      {present.map((entry) => (
        <div key={entry.label}>
          <dt>{entry.label}</dt>
          <dd
            className={
              [
                entry.mono ? "mono" : "",
                entry.breakable ? "experiment-run-breakable" : "",
                entry.className ?? "",
              ]
                .filter(Boolean)
                .join(" ") || undefined
            }
          >
            {entry.value}
          </dd>
        </div>
      ))}
    </dl>
  );
}

function joinFacts(parts: (string | null | undefined)[]): string | null {
  const kept = parts.filter((part): part is string => Boolean(part));
  return kept.length > 0 ? kept.join(" · ") : null;
}

function WatcherItems({
  items,
  watcherCheckBusyId,
  actionsDisabled,
  onCheckWatcher,
}: {
  items: ExperimentWatcherItem[];
  watcherCheckBusyId: string | null;
  actionsDisabled: boolean;
  onCheckWatcher: (watcherId: string) => void;
}) {
  return items.map((item) =>
    item.kind === "group" ? (
      <WatcherGroupDetail
        group={item.group}
        watcherCheckBusyId={watcherCheckBusyId}
        actionsDisabled={actionsDisabled}
        onCheckWatcher={onCheckWatcher}
        key={item.group.groupId}
      />
    ) : (
      <WatcherDetail
        watcher={item.watcher}
        watcherCheckBusyId={watcherCheckBusyId}
        actionsDisabled={actionsDisabled}
        onCheckWatcher={onCheckWatcher}
        key={item.watcher.watcher_id}
      />
    ),
  );
}

function watcherItemCount(total: number, item: ExperimentWatcherItem): number {
  return total + (item.kind === "group" ? item.group.watchers.length : 1);
}

function watcherItemIsStopped(item: ExperimentWatcherItem): boolean {
  return item.kind === "group"
    ? item.group.watchers.every((watcher) => watcher.status === "stopped")
    : item.watcher.status === "stopped";
}

function WatcherGroupDetail({
  group,
  watcherCheckBusyId,
  actionsDisabled,
  onCheckWatcher,
}: {
  group: ExperimentWatcherGroup;
  watcherCheckBusyId: string | null;
  actionsDisabled: boolean;
  onCheckWatcher: (watcherId: string) => void;
}) {
  return (
    <li className="experiment-run-watcher-group">
      <details>
        <summary>
          <span>
            <span className="eyebrow">Watcher group</span>
            <strong>{group.label}</strong>
          </span>
          <span className="experiment-run-watcher-group-counts">{watcherGroupSummary(group)}</span>
        </summary>
        <p className="experiment-run-watcher-group-id">
          <span className="eyebrow">Group ID</span>
          <code>{group.groupId}</code>
        </p>
        <ul className="experiment-run-watcher-group-members" aria-label={`${group.label} watchers`}>
          {group.watchers.map((watcher) => (
            <WatcherDetail
              watcher={watcher}
              watcherCheckBusyId={watcherCheckBusyId}
              actionsDisabled={actionsDisabled}
              onCheckWatcher={onCheckWatcher}
              key={watcher.watcher_id}
            />
          ))}
        </ul>
      </details>
    </li>
  );
}

function WatcherDetail({
  watcher,
  watcherCheckBusyId,
  actionsDisabled,
  onCheckWatcher,
}: {
  watcher: WatcherRecord;
  watcherCheckBusyId: string | null;
  actionsDisabled: boolean;
  onCheckWatcher: (watcherId: string) => void;
}) {
  const external = isExternalWatcherRecord(watcher);
  const canCheckNow = external && watcher.status === "degraded" && !watcher.notified;
  const checkBusy = watcherCheckBusyId === watcher.watcher_id;
  return (
    <li className={`experiment-run-watcher ${watcher.status}`}>
      <details>
        <summary className="experiment-run-watcher-heading">
          <span className={`status-pill ${watcher.status}`}>{watcher.status}</span>
          <strong className="mono experiment-run-breakable">
            {external ? watcher.watcher_id : graphConditionLabel(watcher.condition)}
          </strong>
          <span>{watcherDeliveryLabel(watcher)}</span>
        </summary>
        <Facts
          className="experiment-run-watcher-facts"
          entries={[
            {
              label: "Origin invocation",
              value: watcher.origin_operation_id,
              mono: true,
              breakable: true,
            },
            {
              label: "Watcher ID",
              value: external ? null : watcher.watcher_id,
              mono: true,
              breakable: true,
            },
            { label: "Provenance", value: watcherProvenance(watcher) },
            {
              label: external ? "Last check" : "Last evaluation",
              value: formatMoment(watcherLastObservedAt(watcher)),
            },
            {
              label: "Next check",
              value: external
                ? watcher.next_check_at
                  ? formatMoment(watcher.next_check_at)
                  : "Not scheduled"
                : null,
            },
            {
              label: "Consecutive failures",
              value: external ? watcher.consecutive_error_count : null,
            },
            { label: "Exit code", value: external ? watcher.last_exit_code : null },
            { label: "Completed", value: formatMoment(watcher.completed_at) },
            { label: "Machine", value: watcher.execution_host || "Local" },
            {
              label: "Delivery task",
              value: watcher.notification_operation_id,
              mono: true,
              breakable: true,
            },
            {
              label: "Stopped by",
              value: watcher.status === "stopped" ? watcherStopDisposition(watcher) : null,
            },
            {
              label: "Current error",
              value: external ? watcher.last_error : null,
              className: "experiment-run-watcher-current-error",
            },
          ]}
        />
        {canCheckNow && (
          <div className="experiment-run-watcher-actions">
            <button
              type="button"
              className="button compact"
              disabled={actionsDisabled}
              aria-busy={checkBusy}
              onClick={() => onCheckWatcher(watcher.watcher_id)}
            >
              {checkBusy ? "Checking…" : "Check now"}
            </button>
          </div>
        )}
        {external ? (
          <>
            <div className="experiment-run-watcher-command">
              <span className="eyebrow">Check command</span>
              <code>{watcher.check_command}</code>
            </div>
            <div className="experiment-run-watcher-paths">
              <span>
                <span className="eyebrow">Log</span>
                <code>{watcher.log_path}</code>
              </span>
              <span>
                <span className="eyebrow">Working directory</span>
                <code>{watcher.cwd}</code>
              </span>
            </div>
          </>
        ) : (
          <div className="experiment-run-watcher-command">
            <span className="eyebrow">Graph condition</span>
            <code>{graphConditionLabel(watcher.condition)}</code>
          </div>
        )}
        {watcher.stop_reason && (
          <p className="experiment-run-watcher-stop-reason">
            <strong>{watcher.stopped_by === "agent" ? "Agent reason" : "Stop reason"}</strong>
            {watcher.stop_reason}
          </p>
        )}
      </details>
    </li>
  );
}

function watcherDeliveryLabel(watcher: WatcherRecord): string {
  if (watcher.notification_operation_id) return "Delivery claimed";
  if (watcher.status === "stopped") return "Stopped · not delivered";
  if (watcher.status === "completed" && !watcher.notified) return "Pending delivery";
  if (watcher.notified) return "Acknowledged · not delivered";
  return "Not delivered";
}

function watcherGroupSummary(group: ExperimentWatcherGroup): string {
  const { finished, degraded, running, stopped } = group.counts;
  const summary = [`${finished} finished`, `${degraded} degraded`, `${running} running`];
  if (stopped > 0) summary.push(`${stopped} stopped`);
  return summary.join(" · ");
}

function watcherProvenance(watcher: WatcherRecord): string {
  const episode = watcher.continuation.control_episode_id ?? "—";
  const invocation = watcher.continuation.control_invocation ?? "—";
  const ceiling = watcher.continuation.control_invocation_ceiling;
  return `episode ${episode} · invocation ${invocation}${ceiling ? ` / ${ceiling}` : ""}`;
}

function watcherStopDisposition(watcher: WatcherRecord): string {
  const actor = watcher.stopped_by ? `${capitalize(watcher.stopped_by)} stopped` : "Stopped";
  const stoppedAt = formatMoment(watcher.stopped_at);
  return stoppedAt === "—" ? actor : `${actor} · ${stoppedAt}`;
}

function capitalize(value: string): string {
  return `${value[0].toUpperCase()}${value.slice(1)}`;
}

function taskInvocation(task: ExperimentRun["currentTask"]): number | null {
  const value = task?.request.control_invocation;
  return typeof value === "number" && Number.isInteger(value) ? value : null;
}

function formatMoment(value: string | null | undefined): string {
  if (!value || !Number.isFinite(Date.parse(value))) return "—";
  return new Date(value).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}
