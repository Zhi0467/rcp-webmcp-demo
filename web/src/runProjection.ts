import type {
  AgentTask,
  ExternalWatcherRecord,
  ExperimentControlState,
  ExperimentLoopHealth,
  ExperimentRecommendedStep,
  GraphCondition,
  GraphWatcherRecord,
  GraphNode,
  WatcherRecord,
} from "./types";

export interface AgentTaskGroup {
  rootId: string;
  root: AgentTask;
  latest: AgentTask;
  attempts: AgentTask[];
}

export interface RunTaskProjection {
  actionable: AgentTaskGroup[];
  running: AgentTaskGroup[];
  completed: AgentTaskGroup[];
}

export type RunSectionKey = "running" | "actionable" | "completed";

export interface ExperimentRun {
  node: GraphNode;
  control: ExperimentControlState;
  taskGroup: AgentTaskGroup | null;
  currentTask: AgentTask | null;
  watchers: WatcherRecord[];
  watcherItems: ExperimentWatcherItem[];
  currentWatchers: WatcherRecord[];
  health: ExperimentLoopHealth;
}

export interface ExperimentRecommendation {
  step: ExperimentRecommendedStep;
  label: string;
}

export interface ExperimentWatcherCounts {
  finished: number;
  degraded: number;
  running: number;
  stopped: number;
}

export interface ExperimentWatcherGroup {
  groupId: string;
  label: string;
  watchers: ExternalWatcherRecord[];
  counts: ExperimentWatcherCounts;
}

export type ExperimentWatcherItem =
  { kind: "group"; group: ExperimentWatcherGroup } | { kind: "watcher"; watcher: WatcherRecord };

export function watcherIsActive(watcher: WatcherRecord): boolean {
  return watcher.status === "active" || watcher.status === "degraded";
}

export function isExternalWatcherRecord(watcher: WatcherRecord): watcher is ExternalWatcherRecord {
  return "check_command" in watcher;
}

export function isGraphWatcherRecord(watcher: WatcherRecord): watcher is GraphWatcherRecord {
  return "condition" in watcher;
}

export function graphConditionLabel(condition: GraphCondition): string {
  return "status_in" in condition
    ? `${condition.node_id} reaches ${condition.status_in.join(" or ")}`
    : `Proposal on ${condition.node_id} is resolved`;
}

export function watcherLastObservedAt(watcher: WatcherRecord): string | null {
  return isGraphWatcherRecord(watcher) ? watcher.last_evaluated_at : watcher.last_checked_at;
}

/**
 * Chats show the resources they can observe: an Experiment node's loop watchers and the exact
 * conversation's own self-wake watchers. Experiment-loop provenance does not make a chat its
 * owner, while generic self-wake watchers never leak into another conversation.
 */
export function visibleChatWatchers(
  watchers: WatcherRecord[],
  chatId: string,
  node: GraphNode | null | undefined,
): WatcherRecord[] {
  const experimentNodeId = node?.type === "experiment" ? node.id : null;
  const visible = new Map<string, WatcherRecord>();
  for (const watcher of watchers) {
    if (!watcherIsActive(watcher)) continue;
    const nodeLoopWatcher =
      experimentNodeId !== null &&
      watcher.continuation.patch_kind === "experiment_loop" &&
      watcher.continuation.control_node_id === experimentNodeId;
    const chatSelfWakeWatcher =
      watcher.chat_id === chatId && watcher.continuation.patch_kind === "work";
    if (nodeLoopWatcher || chatSelfWakeWatcher) visible.set(watcher.watcher_id, watcher);
  }
  return [...visible.values()];
}

export type RunEntry =
  | { kind: "task"; id: string; observedAt: string | null; group: AgentTaskGroup }
  | { kind: "experiment"; id: string; observedAt: string | null; experiment: ExperimentRun }
  | { kind: "blocker"; id: string; observedAt: string | null; node: GraphNode };

export interface RunProjection {
  running: RunEntry[];
  actionable: RunEntry[];
  completed: RunEntry[];
}

export interface RunProjectionInput {
  nodes: GraphNode[];
  tasks: AgentTask[];
  watchers?: WatcherRecord[];
  experimentControl: Record<string, ExperimentControlState>;
  actionableBlockerIds: ReadonlySet<string>;
  dismissedTaskIds?: ReadonlySet<string>;
}

export function buildRunTaskProjection(
  tasks: AgentTask[],
  dismissedTaskIds: ReadonlySet<string> = new Set(),
): RunTaskProjection {
  const groups = groupAgentTasks(tasks).filter(
    (group) => !isTaskNotificationSuperseded(group.latest, tasks),
  );
  return {
    actionable: groups.filter(
      (group) => group.latest.awaiting_human && !dismissedTaskIds.has(group.latest.operation_id),
    ),
    running: groups.filter((group) => group.latest.active),
    completed: groups.filter((group) => group.latest.settled),
  };
}

export function isExperimentLoopTask(task: AgentTask): boolean {
  return (
    task.request?.patch_kind === "experiment_loop" && Boolean(task.request?.control_node_id ?? null)
  );
}

/** Map the server's recommendation to presentation copy without re-deciding it. */
export function experimentRecommendation(run: ExperimentRun): ExperimentRecommendation {
  const step = run.control.recommendation;
  const labels: Record<ExperimentRecommendedStep, string> = {
    wait:
      run.health === "stopping"
        ? "Wait for the current turn to finish"
        : run.health === "wrapping_up"
          ? "Wrapping up visualization and report"
          : run.health === "waiting_on_watchers"
            ? "Wait for watcher completion"
            : "Wait for the agent",
    resume: run.control.can_switch_provider
      ? "Resume this episode, or switch provider"
      : "Resume this episode",
    retry: "Retry this episode, or switch provider",
    keep_loop: "Keep loop running; check now if needed",
    start_episode: run.control.episode_id ? "Start a new episode" : "Start an episode",
    stop_and_restart: "Stop loop, then start a new episode",
    resolve_requirements: "Resolve the run requirements",
    open_report: "Open report",
    review: "Review the loop state",
    none: run.control.node_closed
      ? `Experiment is ${run.node.status}`
      : run.control.episode?.wrapup_state === "legacy_unavailable"
        ? "Episode report unavailable"
        : run.health === "failed"
          ? "Episode ended"
          : "No action needed",
  };
  return { step, label: labels[step] };
}

export function buildExperimentRun(
  node: GraphNode,
  control: ExperimentControlState,
  tasks: AgentTask[],
  allWatchers: WatcherRecord[],
): ExperimentRun {
  const watchers = allWatchers
    .filter(
      (watcher) =>
        watcher.continuation.patch_kind === "experiment_loop" &&
        watcher.continuation.control_node_id === node.id,
    )
    .sort(
      (left, right) =>
        right.created_at.localeCompare(left.created_at) ||
        left.watcher_id.localeCompare(right.watcher_id),
    );
  const currentWatchers = watchers.filter(
    (watcher) =>
      Boolean(control.episode_id) && watcher.continuation.control_episode_id === control.episode_id,
  );
  const { taskGroup, currentTask } = currentExperimentTaskGroup(node.id, control, tasks);
  return {
    node,
    control,
    taskGroup,
    currentTask,
    watchers,
    watcherItems: experimentWatcherDisplayItems(watchers),
    currentWatchers,
    health: control.health,
  };
}

/** Keep an Experiment's immutable watcher group visible as one operational unit. */
export function experimentWatcherDisplayItems(watchers: WatcherRecord[]): ExperimentWatcherItem[] {
  const groups = new Map<string, ExperimentWatcherGroup>();
  const items: ExperimentWatcherItem[] = [];
  for (const watcher of watchers) {
    if (!isExternalWatcherRecord(watcher) || !watcher.group_id) {
      items.push({ kind: "watcher", watcher });
      continue;
    }
    let group = groups.get(watcher.group_id);
    if (!group) {
      group = {
        groupId: watcher.group_id,
        label: watcher.group_label ?? watcher.group_id,
        watchers: [],
        counts: { finished: 0, degraded: 0, running: 0, stopped: 0 },
      };
      groups.set(watcher.group_id, group);
      items.push({ kind: "group", group });
    }
    group.watchers.push(watcher);
    group.counts[watcherGroupCountKey(watcher)] += 1;
  }
  return items;
}

function watcherGroupCountKey(watcher: ExternalWatcherRecord): keyof ExperimentWatcherCounts {
  switch (watcher.status) {
    case "completed":
      return "finished";
    case "active":
      return "running";
    case "degraded":
      return "degraded";
    case "stopped":
      return "stopped";
  }
}

export function buildRunProjection(input: RunProjectionInput): RunProjection {
  const watchers = input.watchers ?? [];
  const experimentControl = input.experimentControl;
  const ingestion = buildRunTaskProjection(
    input.tasks.filter((task) => task.kind === "seed" || task.kind === "refresh"),
    input.dismissedTaskIds ?? new Set<string>(),
  );
  const sections: Record<RunSectionKey, RunEntry[]> = {
    running: ingestion.running.map(taskEntry),
    actionable: ingestion.actionable.map(taskEntry),
    completed: ingestion.completed.map(taskEntry),
  };
  input.nodes
    .filter((node) => node.type === "experiment")
    .forEach((node) => {
      const control = experimentControl[node.id];
      if (!control) {
        throw new Error(`Experiment ${node.id} is missing its backend control projection.`);
      }
      if (!control.health || !control.recommendation || !control.run_section) {
        throw new Error(`Experiment ${node.id} has an incomplete backend control projection.`);
      }
      const run = buildExperimentRun(node, control, input.tasks, watchers);
      sections[control.run_section].push(experimentEntry(run));
    });
  input.nodes
    .filter((node) => input.actionableBlockerIds.has(node.id))
    .forEach((node) => {
      if (node.type !== "blocker") {
        throw new Error(`Attention member ${node.id} is not a Blocker.`);
      }
      sections.actionable.push({
        kind: "blocker",
        id: node.id,
        observedAt: newestTimestamp(node.source_refs.map((source) => source.timestamp)),
        node,
      });
    });
  return {
    running: sortRunEntries(sections.running),
    actionable: sortRunEntries(sections.actionable),
    completed: sortRunEntries(sections.completed),
  };
}

export function groupAgentTasks(tasks: AgentTask[]): AgentTaskGroup[] {
  const byId = new Map(tasks.map((task) => [task.operation_id, task]));
  const grouped = new Map<string, AgentTask[]>();
  tasks.forEach((task) => {
    const rootId = logicalRootId(task, byId);
    const attempts = grouped.get(rootId) ?? [];
    attempts.push(task);
    grouped.set(rootId, attempts);
  });
  return [...grouped.entries()]
    .map(([rootId, attempts]) => {
      attempts.sort(compareTaskAscending);
      return {
        rootId,
        root: byId.get(rootId) ?? attempts[0],
        latest: attempts.at(-1) ?? attempts[0],
        attempts,
      };
    })
    .sort((left, right) => compareTaskAscending(right.latest, left.latest));
}

function currentExperimentTaskGroup(
  nodeId: string,
  control: ExperimentControlState,
  tasks: AgentTask[],
): { taskGroup: AgentTaskGroup | null; currentTask: AgentTask | null } {
  const nodeTasks = tasks.filter(
    (task) => isExperimentLoopTask(task) && task.request.control_node_id === nodeId,
  );
  const currentOperationId = control.operational.current_operation_id;
  const currentTask = currentOperationId
    ? (nodeTasks.find((task) => task.operation_id === currentOperationId) ?? null)
    : null;
  const episodeTasks = control.episode_id
    ? nodeTasks.filter((task) => taskEpisodeId(task) === control.episode_id)
    : nodeTasks;
  const groups = groupAgentTasks(episodeTasks);
  const taskGroup =
    (currentOperationId
      ? groups.find((group) =>
          group.attempts.some((task) => task.operation_id === currentOperationId),
        )
      : null) ??
    groups[0] ??
    null;
  return { taskGroup, currentTask };
}

function taskEpisodeId(task: AgentTask): string | null {
  const value = task.request.control_episode_id;
  return typeof value === "string" && value ? value : null;
}

function taskEntry(group: AgentTaskGroup): RunEntry {
  return { kind: "task", id: group.rootId, observedAt: group.latest.updated_at, group };
}

function experimentEntry(experiment: ExperimentRun): RunEntry {
  const observedAt = newestTimestamp([
    experiment.taskGroup?.latest.updated_at,
    experiment.control.operational.current_last_activity_at,
    ...experiment.watchers.flatMap((watcher) => [
      watcher.completed_at,
      watcherLastObservedAt(watcher),
      watcher.created_at,
    ]),
  ]);
  return { kind: "experiment", id: experiment.node.id, observedAt, experiment };
}

/** An Experiment-loop watcher is released by Stop loop, never one watcher at a time. */
export function watcherIsIndividuallyStoppable(watcher: WatcherRecord): boolean {
  return watcher.continuation?.patch_kind !== "experiment_loop";
}

function sortRunEntries(entries: RunEntry[]): RunEntry[] {
  return [...entries].sort((left, right) => {
    const leftAt = left.observedAt ? Date.parse(left.observedAt) : Number.NaN;
    const rightAt = right.observedAt ? Date.parse(right.observedAt) : Number.NaN;
    const leftKnown = Number.isFinite(leftAt);
    const rightKnown = Number.isFinite(rightAt);
    if (leftKnown && rightKnown && leftAt !== rightAt) return rightAt - leftAt;
    if (leftKnown !== rightKnown) return leftKnown ? -1 : 1;
    return left.id.localeCompare(right.id);
  });
}

function newestTimestamp(values: (string | null | undefined)[]): string | null {
  return (
    values
      .filter(
        (value): value is string => typeof value === "string" && Number.isFinite(Date.parse(value)),
      )
      .sort((left, right) => Date.parse(right) - Date.parse(left))[0] ?? null
  );
}

function logicalRootId(task: AgentTask, byId: Map<string, AgentTask>): string {
  const seen = new Set([task.operation_id]);
  let current = task;
  while (
    current.parent_operation_id &&
    byId.has(current.parent_operation_id) &&
    !seen.has(current.parent_operation_id)
  ) {
    seen.add(current.parent_operation_id);
    current = byId.get(current.parent_operation_id) as AgentTask;
  }
  return current.operation_id;
}

function compareTaskAscending(left: AgentTask, right: AgentTask): number {
  return (
    left.created_at.localeCompare(right.created_at) ||
    left.operation_id.localeCompare(right.operation_id)
  );
}

function isTaskNotificationSuperseded(task: AgentTask, tasks: AgentTask[]): boolean {
  if ((task.kind !== "seed" && task.kind !== "refresh") || !task.awaiting_human || task.paused)
    return false;
  return tasks.some(
    (candidate) =>
      (candidate.kind === "seed" || candidate.kind === "refresh") &&
      candidate.settled &&
      compareTaskAscending(candidate, task) > 0,
  );
}
