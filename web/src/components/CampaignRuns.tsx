import {
  ChevronDown,
  CirclePause,
  ExternalLink,
  LoaderCircle,
  MessageCircle,
  Network,
  Play,
  RotateCcw,
  Send,
  Square,
  Telescope,
} from "lucide-react";
import { useEffect, useId, useMemo, useState } from "react";
import { taskStatusLabel } from "../agentTasks";
import {
  episodeEndingLabel,
  episodeProjection,
  episodeReportPreviewUrl,
  episodeTaskRoleLabel,
  episodeTaskRows,
  formatTokenCount,
} from "../campaigns";
import { MarkdownAnswer } from "../chatMarkdown";
import type { AgentTask, Episode, EpisodeMessage } from "../types";
import { EpisodeReportLink } from "./EpisodeReportLink";

export function AutoResearchEpisodeCard({
  episode,
  tasks,
  messages,
  initiallyExpanded,
  busyAction,
  taskActionId,
  onInspectTask,
  onLoadMessages,
  onStop,
  onMerge,
  onReauthorize,
  onSendMessage,
  onOperateTask,
}: {
  episode: Episode;
  tasks: AgentTask[];
  messages: EpisodeMessage[];
  initiallyExpanded: boolean;
  busyAction: string | null;
  taskActionId: string | null;
  onInspectTask: (operationId: string) => void;
  onLoadMessages: (episodeId: string) => Promise<void>;
  onStop: (episodeId: string) => Promise<void>;
  onMerge: (episodeId: string) => Promise<void>;
  onReauthorize: (episodeId: string, invocationCeiling: number) => Promise<void>;
  onSendMessage: (episodeId: string, body: string) => Promise<void>;
  onOperateTask: (task: AgentTask, action: "pause" | "resume" | "retry") => Promise<void>;
}) {
  const detailId = useId();
  const [expanded, setExpanded] = useState(initiallyExpanded);
  const [additionalInvocations, setAdditionalInvocations] = useState("");
  const [message, setMessage] = useState("");
  const [localError, setLocalError] = useState<string | null>(null);
  const taskRows = useMemo(
    () => episodeTaskRows(episode, episode.tasks.length > 0 ? episode.tasks : tasks),
    [episode, tasks],
  );
  const projection = useMemo(
    () =>
      episodeProjection(
        episode,
        taskRows.map(({ task }) => task),
      ),
    [episode, taskRows],
  );
  const { recommendation, taskControl } = projection;
  const parsedAdditionalInvocations = Number(additionalInvocations);
  const reauthorizationIsValid =
    additionalInvocations.trim().length > 0 &&
    Number.isSafeInteger(parsedAdditionalInvocations) &&
    parsedAdditionalInvocations >= 1;
  const stopBusy = busyAction === `stop:${episode.episode_id}`;
  const mergeBusy = busyAction === `merge:${episode.episode_id}`;
  const reauthorizeBusy = busyAction === `reauthorize:${episode.episode_id}`;
  const messageBusy = busyAction === `message:${episode.episode_id}`;
  const controlTaskBusy = taskControl !== null && taskActionId === taskControl.task.operation_id;
  const anotherActionBusy = busyAction !== null || taskActionId !== null;
  const episodeTimestamp = formatTimestamp(episode.created_at);
  const showStop =
    episode.can_stop &&
    projection.health !== "stopping" &&
    projection.health !== "completed" &&
    projection.health !== "stopped" &&
    projection.health !== "failed";

  useEffect(() => {
    if (!expanded) return;
    void onLoadMessages(episode.episode_id).catch((error) => {
      setLocalError(error instanceof Error ? error.message : String(error));
    });
  }, [episode.episode_id, expanded, onLoadMessages]);

  useEffect(() => {
    if (episode.can_reauthorize) setExpanded(true);
  }, [episode.can_reauthorize]);

  const submitReauthorization = async () => {
    if (!reauthorizationIsValid || anotherActionBusy) return;
    setLocalError(null);
    try {
      await onReauthorize(episode.episode_id, parsedAdditionalInvocations);
      setAdditionalInvocations("");
    } catch (error) {
      setLocalError(error instanceof Error ? error.message : String(error));
    }
  };

  const submitMessage = async () => {
    const body = message.trim();
    if (!body || anotherActionBusy) return;
    setLocalError(null);
    try {
      await onSendMessage(episode.episode_id, body);
      setMessage("");
    } catch (error) {
      setLocalError(error instanceof Error ? error.message : String(error));
    }
  };

  const operateControlTask = async () => {
    if (anotherActionBusy || !taskControl) return;
    setLocalError(null);
    try {
      await onOperateTask(taskControl.task, taskControl.kind);
    } catch (error) {
      setLocalError(error instanceof Error ? error.message : String(error));
    }
  };

  const mergeToMain = async () => {
    if (!episode.graph_branch?.merge_eligible || anotherActionBusy) return;
    setLocalError(null);
    try {
      await onMerge(episode.episode_id);
    } catch (error) {
      setLocalError(error instanceof Error ? error.message : String(error));
    }
  };

  return (
    <article className={`campaign-run ${projection.health}`}>
      <span className="campaign-state-rail" aria-hidden="true" />
      <div className="campaign-run-heading">
        <button
          className="campaign-run-toggle"
          type="button"
          aria-label={`${expanded ? "Collapse" : "Expand"} auto-research episode started ${episodeTimestamp}`}
          aria-expanded={expanded}
          aria-controls={detailId}
          onClick={() => setExpanded((current) => !current)}
        />
        <span className="campaign-run-identity">
          <strong className="campaign-run-title">
            <Telescope size={14} aria-hidden="true" />
            <span>Auto-research</span>
          </strong>
          <span className="campaign-run-meta">
            <span className={`status-pill ${projection.health}`}>{projection.healthLabel}</span>
            <time dateTime={episode.created_at}>{episodeTimestamp}</time>
          </span>
        </span>
        <EpisodeBudgetMeter episode={episode} />
        <span className="campaign-run-time">
          <ChevronDown size={15} aria-hidden="true" />
        </span>
      </div>
      {expanded && (
        <div className="campaign-run-detail" id={detailId}>
          <div className="campaign-run-actions">
            <div
              className={`campaign-run-health ${projection.health}`}
              role="status"
              aria-live="polite"
              aria-atomic="true"
            >
              <strong>{projection.healthLabel}</strong>
            </div>
            {episode.report && episode.wrapup_state === "ready" && (
              <div className="campaign-report-actions">
                <EpisodeReportLink
                  className={`button compact ${recommendation.kind === "open_report" ? "primary" : "secondary"}`}
                  href={episodeReportPreviewUrl(episode.project_id, episode.episode_id)}
                  aria-label={`Open ${episodeEndingLabel(episode.report.ending)} report from ${formatTimestamp(episode.report.created_at, true)}`}
                  projectId={episode.project_id}
                  episodeId={episode.episode_id}
                  onOpenError={setLocalError}
                >
                  <ExternalLink size={12} /> Open report
                </EpisodeReportLink>
              </div>
            )}
            {taskControl && (
              <button
                className={`button compact ${
                  taskControl.kind === "resume" ? "primary" : "secondary"
                }`}
                type="button"
                disabled={anotherActionBusy}
                onClick={() => void operateControlTask()}
              >
                {controlTaskBusy ? (
                  <LoaderCircle className="spin" size={12} />
                ) : taskControl.kind === "pause" ? (
                  <CirclePause size={12} />
                ) : taskControl.kind === "resume" ? (
                  <Play size={12} />
                ) : (
                  <RotateCcw size={12} />
                )}
                {controlTaskBusy
                  ? `${episodeActionLabel(taskControl.kind)}…`
                  : episodeActionLabel(taskControl.kind)}
              </button>
            )}
            {episode.can_reauthorize && projection.health === "needs_action" && (
              <form
                className="campaign-reauthorize"
                onSubmit={(event) => {
                  event.preventDefault();
                  void submitReauthorization();
                }}
              >
                <input
                  type="number"
                  min={1}
                  step={1}
                  inputMode="numeric"
                  aria-label="New episode invocation ceiling"
                  value={additionalInvocations}
                  disabled={anotherActionBusy}
                  onChange={(event) => setAdditionalInvocations(event.target.value)}
                />
                <button
                  className="button primary compact"
                  type="submit"
                  disabled={!reauthorizationIsValid || anotherActionBusy}
                >
                  {reauthorizeBusy && <LoaderCircle className="spin" size={12} />}
                  Reauthorize
                </button>
              </form>
            )}
            {showStop && (
              <button
                className="button secondary compact campaign-stop"
                type="button"
                disabled={anotherActionBusy}
                onClick={() => {
                  setLocalError(null);
                  void onStop(episode.episode_id).catch((error) => {
                    setLocalError(error instanceof Error ? error.message : String(error));
                  });
                }}
              >
                {stopBusy ? <LoaderCircle className="spin" size={12} /> : <Square size={11} />}
                {stopBusy ? "Stopping…" : "Stop"}
              </button>
            )}
          </div>

          <div className={`campaign-run-recommendation ${recommendation.kind}`}>
            <span className="eyebrow">Recommended next step</span>
            <strong>{recommendation.label}</strong>
          </div>

          {episode.graph_branch && (
            <section
              className={`campaign-graph-branch ${episode.graph_branch.merge_state}`}
              aria-label="Episode graph branch"
            >
              <header>
                <span className="campaign-branch-identity">
                  <span className="eyebrow">Graph branch</span>
                  <strong title={episode.graph_branch.branch_id}>
                    {compactIdentity(episode.graph_branch.branch_id)}
                  </strong>
                </span>
                <span className={`status-pill branch-${episode.graph_branch.merge_state}`}>
                  {branchMergeStateLabel(episode.graph_branch.merge_state)}
                </span>
              </header>
              <div className="campaign-branch-heads">
                <GraphHeadFact label="Base on main" head={episode.graph_branch.base_head} />
                <span className="campaign-branch-head-path" aria-hidden="true" />
                <GraphHeadFact label="Branch head" head={episode.graph_branch.head} />
                {episode.graph_branch.latest_successful_merge && (
                  <GraphHeadFact
                    label={
                      episode.graph_branch.latest_successful_merge.outcome === "committed"
                        ? "Merged on main"
                        : "Main unchanged"
                    }
                    head={episode.graph_branch.latest_successful_merge.result_main_head}
                  />
                )}
              </div>
              {(episode.graph_branch.merge_state === "needs_action" ||
                episode.graph_branch.merge_state === "failed") &&
                episode.graph_branch.merge_diagnostic && (
                  <div
                    className={`campaign-branch-diagnostic ${episode.graph_branch.merge_state}`}
                    role={episode.graph_branch.merge_state === "failed" ? "alert" : "status"}
                  >
                    {episode.graph_branch.merge_diagnostic}
                  </div>
                )}
              {episode.graph_branch.merge_eligible &&
                episode.graph_branch.merge_state !== "running" && (
                  <button
                    className="button primary compact campaign-branch-merge"
                    type="button"
                    disabled={anotherActionBusy}
                    onClick={() => void mergeToMain()}
                  >
                    {mergeBusy ? (
                      <LoaderCircle className="spin" size={12} />
                    ) : (
                      <Network size={12} />
                    )}
                    {mergeBusy ? "Starting merge…" : "Merge to main"}
                  </button>
                )}
            </section>
          )}

          {episode.starting_instruction && (
            <div className="campaign-starting-instruction">
              <span className="field-label">Starting instruction</span>
              <MarkdownAnswer text={episode.starting_instruction} />
            </div>
          )}

          <section className="campaign-turns" aria-label="Episode turns">
            <header>
              <h3>Turns</h3>
              <span>{taskRows.length}</span>
            </header>
            {taskRows.length > 0 ? (
              <ul>
                {taskRows.map(({ task, role, depth }) => {
                  const target = taskTarget(task);
                  const roleLabel =
                    task.kind === "branch_merge" ? "Branch merge" : episodeTaskRoleLabel(role);
                  return (
                    <li className={`campaign-task depth-${depth}`} key={task.operation_id}>
                      <button type="button" onClick={() => onInspectTask(task.operation_id)}>
                        <span className={`campaign-task-role ${role}`}>{roleLabel}</span>
                        <span className="campaign-task-copy">
                          <strong>{target || roleLabel}</strong>
                          <span>{task.status_message}</span>
                        </span>
                        <span className={`status-pill ${task.status}`}>
                          {taskStatusLabel(task)}
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            ) : (
              <p className="campaign-empty">No turns recorded.</p>
            )}
          </section>

          <section className="campaign-mail" aria-label="Episode mail">
            <header>
              <h3>
                <MessageCircle size={13} /> Mail
              </h3>
              <span>{messages.length}</span>
            </header>
            <div className="campaign-mail-thread" role="log" aria-live="polite">
              {messages.length > 0 ? (
                [...messages]
                  .sort((left, right) => Date.parse(left.created_at) - Date.parse(right.created_at))
                  .map((item) => (
                    <article
                      className={`campaign-message ${item.sender_role}`}
                      key={item.message_id}
                    >
                      <header>
                        <strong>{messageSenderLabel(item)}</strong>
                        <span>
                          {item.delivered_at ? "Delivered" : "Pending"} ·{" "}
                          <time dateTime={item.created_at}>{formatTimestamp(item.created_at)}</time>
                        </span>
                      </header>
                      <div className="chat-markdown">
                        <MarkdownAnswer text={item.body} />
                      </div>
                    </article>
                  ))
              ) : (
                <p className="campaign-empty">No messages yet.</p>
              )}
            </div>
            {episode.can_message && (
              <form
                className="campaign-message-composer"
                onSubmit={(event) => {
                  event.preventDefault();
                  void submitMessage();
                }}
              >
                <textarea
                  rows={2}
                  aria-label="Message orchestrator"
                  value={message}
                  disabled={anotherActionBusy}
                  onChange={(event) => setMessage(event.target.value)}
                />
                <button
                  className="button primary compact"
                  type="submit"
                  disabled={!message.trim() || anotherActionBusy}
                >
                  {messageBusy ? <LoaderCircle className="spin" size={12} /> : <Send size={12} />}
                  Send
                </button>
              </form>
            )}
          </section>
          {/* A stopped episode reaches here with no report, no report error, and no
              ending diagnostic, because the projection withholds all three. There is
              nothing left for this view to suppress. */}
          {(localError ||
            (episode.wrapup_state === "failed" ? episode.wrapup_error : null) ||
            episode.ending_diagnostic) && (
            <div className="campaign-run-error" role="alert">
              {localError ||
                (episode.wrapup_state === "failed"
                  ? `Report generation error: ${episode.wrapup_error || "The report could not be generated."}`
                  : episode.ending_diagnostic)}
            </div>
          )}
        </div>
      )}
    </article>
  );
}

function GraphHeadFact({ label, head }: { label: string; head: Episode["graph_base_head"] }) {
  if (!head) return null;
  return (
    <span className="campaign-branch-head">
      <span>{label}</span>
      <strong>r{head.revision}</strong>
      <code title={head.transition_id ?? undefined}>
        {head.transition_id ? compactIdentity(head.transition_id) : "origin"}
      </code>
    </span>
  );
}

function branchMergeStateLabel(state: NonNullable<Episode["graph_branch"]>["merge_state"]): string {
  switch (state) {
    case "unmerged":
      return "Unmerged";
    case "running":
      return "Merge running";
    case "merged":
      return "Merged";
    case "needs_action":
      return "Merge needs action";
    case "failed":
      return "Merge failed";
  }
}

function compactIdentity(value: string): string {
  return value.length <= 18 ? value : `${value.slice(0, 8)}\u2026${value.slice(-6)}`;
}

export function EpisodeBudgetMeter({ episode }: { episode: Episode }) {
  const budget = episode.budget;
  const usedPercent = Math.min(100, (budget.invocations_used / budget.invocation_ceiling) * 100);
  const label = `${budget.invocations_used} of ${budget.invocation_ceiling} operational invocations used; ${budget.observed_input_tokens} observed input tokens and ${budget.observed_generated_tokens} observed generated tokens`;
  return (
    <span
      className="campaign-budget-meter"
      role="meter"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={budget.invocation_ceiling}
      aria-valuenow={budget.invocations_used}
    >
      <span className="campaign-budget-copy">
        <strong>
          {budget.invocations_used} / {budget.invocation_ceiling} invocations
        </strong>
        <span>
          {formatTokenCount(budget.observed_input_tokens)} input ·{" "}
          {formatTokenCount(budget.observed_generated_tokens)} generated
        </span>
      </span>
      <span className="campaign-budget-track" aria-hidden="true">
        <span className="campaign-budget-spent" style={{ width: `${usedPercent}%` }} />
      </span>
    </span>
  );
}

function taskTarget(task: AgentTask): string | null {
  const value = task.request.control_node_id ?? task.request.node_id;
  return typeof value === "string" && value.trim() ? value : null;
}

function messageSenderLabel(message: EpisodeMessage): string {
  if (message.sender_role === "human") {
    return message.authorized_by?.display_name ?? "Unattributed";
  }
  if (message.sender_role === "orchestrator") return "Orchestrator";
  const worker = message.control_node_id || message.sender_task_id;
  return worker ? `Worker · ${worker}` : "Worker";
}

function formatTimestamp(value: string, includeSeconds = false): string {
  const parsed = Date.parse(value);
  if (Number.isNaN(parsed)) return value;
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    ...(includeSeconds ? { second: "2-digit" } : {}),
  }).format(parsed);
}

function episodeActionLabel(action: "pause" | "resume" | "retry"): string {
  if (action === "pause") return "Pause";
  if (action === "resume") return "Resume";
  return "Retry";
}
