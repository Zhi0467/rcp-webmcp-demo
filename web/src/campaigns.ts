import type {
  AgentTask,
  Episode,
  EpisodeEnding,
  EpisodeHealth,
  EpisodeRecommendationKind,
  EpisodeTaskControlKind,
} from "./types";

export type { EpisodeHealth, EpisodeRecommendationKind };

const EPISODE_HEALTH_LABELS: Record<EpisodeHealth, string> = {
  starting: "Starting",
  active: "Active",
  recovering: "Recovering",
  needs_action: "Needs action",
  stopping: "Stopping gracefully",
  wrapping_up: "Wrapping up visualization and report",
  completed: "Completed",
  stopped: "Stopped",
  failed: "Failed",
};

export type EpisodeTaskRole = "orchestrator" | "worker" | "wake";

export interface EpisodeTaskRow {
  task: AgentTask;
  role: EpisodeTaskRole;
  depth: number;
}

export interface EpisodeRecommendation {
  kind: EpisodeRecommendationKind;
  label: string;
  task: AgentTask | null;
}

export interface EpisodeTaskControl {
  kind: EpisodeTaskControlKind;
  task: AgentTask;
}

export interface EpisodeProjection {
  health: EpisodeHealth;
  healthLabel: string;
  recommendation: EpisodeRecommendation;
  taskControl: EpisodeTaskControl | null;
}

export function isLiveEpisode(episode: Episode): boolean {
  // The projection says whether a parent is still live. Restating the storage
  // status list here is how two surfaces came to disagree about one Experiment.
  return episode.live;
}

export function mergeEpisode(episodes: Episode[], nextEpisode: Episode): Episode[] {
  return [
    nextEpisode,
    ...episodes.filter((episode) => episode.episode_id !== nextEpisode.episode_id),
  ].sort(compareEpisodesNewestFirst);
}

export function runsEpisodeCards(
  episodes: Episode[],
  currentExperimentEpisodeIds: ReadonlySet<string>,
): Episode[] {
  return [...episodes]
    .sort(compareEpisodesNewestFirst)
    .filter(
      (episode) =>
        episode.mode === "auto_research" || currentExperimentEpisodeIds.has(episode.episode_id),
    );
}

export function currentEpisodeControlTask(
  episode: Episode,
  tasks: AgentTask[] = episode.tasks,
): AgentTask | null {
  if (!episode.current_control_task_id) return null;
  return tasks.find((task) => task.operation_id === episode.current_control_task_id) ?? null;
}

const EPISODE_RECOMMENDATION_LABELS: Record<EpisodeRecommendationKind, string> = {
  continue: "Let auto-research continue",
  wait: "Wait for the current step",
  resume: "Resume the current turn",
  retry: "Retry the current turn",
  reauthorize: "Start a new authorized episode",
  open_report: "Open report",
  review: "Review the episode state",
  none: "No further action is needed",
};

/** The wording for the wait and review the projection distinguishes by state. */
const EPISODE_RECOMMENDATION_LABELS_BY_HEALTH: Partial<
  Record<EpisodeHealth, Partial<Record<EpisodeRecommendationKind, string>>>
> = {
  wrapping_up: { wait: "Wrapping up visualization and report" },
  recovering: { wait: "Wait for automatic turn recovery" },
  stopping: { wait: "Wait for the current turn to finish" },
  starting: { wait: "Wait for auto-research to start" },
  active: { wait: "Wait for the current turn to pause" },
  stopped: { none: "No further action is available" },
  failed: { review: "Review the episode failure" },
  needs_action: { review: "Review the blocked turn" },
};

export function episodeProjection(
  episode: Episode,
  tasks: AgentTask[] = episode.tasks,
): EpisodeProjection {
  // Lifecycle state, next step, and available control are decided by the server.
  // This function resolves them to copy and to the task object a control acts on.
  const task = currentEpisodeControlTask(episode, tasks);
  const defaultLabel =
    EPISODE_RECOMMENDATION_LABELS_BY_HEALTH[episode.health]?.[episode.recommendation] ??
    EPISODE_RECOMMENDATION_LABELS[episode.recommendation];
  const label =
    episode.mode === "experiment_loop"
      ? experimentEpisodeRecommendationLabel(episode, defaultLabel)
      : defaultLabel;
  return {
    health: episode.health,
    healthLabel: EPISODE_HEALTH_LABELS[episode.health],
    recommendation: { kind: episode.recommendation, label, task },
    taskControl: episode.task_control && task ? { kind: episode.task_control, task } : null,
  };
}

function experimentEpisodeRecommendationLabel(episode: Episode, fallback: string): string {
  if (episode.recommendation === "continue") return "Let the experiment loop continue";
  if (episode.health === "starting" && episode.recommendation === "wait") {
    return "Wait for the experiment loop to start";
  }
  if (episode.health === "active" && episode.recommendation === "wait") {
    return "Wait for the current experiment turn to pause";
  }
  return fallback;
}

export function episodeEndingLabel(ending: EpisodeEnding): string {
  switch (ending) {
    case "completed":
      return "Completed";
    case "exhausted":
      return "Exhausted";
    case "stopped":
      return "Stopped";
    case "failed":
      return "Failed";
    case "human_pause":
      return "Human-authority pause";
  }
}

export function episodeReportPreviewUrl(projectId: string, episodeId: string): string {
  return `/api/projects/${encodeURIComponent(projectId)}/episodes/${encodeURIComponent(episodeId)}/report/viewer`;
}

export function episodeTaskRows(episode: Episode, tasks: AgentTask[]): EpisodeTaskRow[] {
  const episodeTasks = tasks
    .filter((task) => task.episode_id === episode.episode_id)
    .sort(compareTaskTime);
  const byId = new Map(episodeTasks.map((task) => [task.operation_id, task]));
  return episodeTasks.map((task) => ({
    task,
    role: episodeTaskRole(episode, task),
    depth: episodeTaskDepth(task, byId),
  }));
}

export function episodeTaskRole(episode: Episode, task: AgentTask): EpisodeTaskRole {
  const declared = task.request.role ?? task.request.invocation_role;
  if (
    task.request.wake_cause ||
    task.request.trigger === "watcher" ||
    task.request.continuation_cause === "graph_condition" ||
    task.request.continuation_cause === "message"
  ) {
    return "wake";
  }
  if (declared === "worker") return "worker";
  if (task.operation_id === episode.root_operation_id) return "orchestrator";
  if (declared === "orchestrator" || declared === "wake") return declared;
  if (task.kind !== "auto_research" || task.request.control_node_id || task.request.node_id) {
    return "worker";
  }
  return "orchestrator";
}

export function episodeTaskRoleLabel(role: EpisodeTaskRole): string {
  switch (role) {
    case "orchestrator":
      return "Orchestrator";
    case "worker":
      return "Worker";
    case "wake":
      return "Wake";
  }
}

export function formatTokenCount(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}

function episodeTaskDepth(task: AgentTask, byId: ReadonlyMap<string, AgentTask>): number {
  const actorOperationId = episodeActorOperationId(task);
  const actorOrigin = actorOperationId === null ? null : byId.get(actorOperationId);
  if (actorOrigin) return episodeActorDepth(actorOrigin, byId);
  return episodeAncestryDepth(task, byId);
}

function episodeActorDepth(task: AgentTask, byId: ReadonlyMap<string, AgentTask>): number {
  let depth = 0;
  let current = task;
  let parentId = current.parent_operation_id;
  const seen = new Set<string>([current.operation_id]);
  while (parentId && byId.has(parentId) && !seen.has(parentId)) {
    seen.add(parentId);
    const parent = byId.get(parentId);
    if (!parent) break;
    if (
      (episodeActorOperationId(current) ?? current.operation_id) !==
      (episodeActorOperationId(parent) ?? parent.operation_id)
    ) {
      depth += 1;
    }
    current = parent;
    parentId = current.parent_operation_id;
  }
  return Math.min(depth, 4);
}

function episodeActorOperationId(task: AgentTask): string | null {
  const value = task.request.actor_operation_id;
  return typeof value === "string" && value.length > 0 ? value : null;
}

function episodeAncestryDepth(task: AgentTask, byId: ReadonlyMap<string, AgentTask>): number {
  let depth = 0;
  let parentId = task.parent_operation_id;
  const seen = new Set<string>();
  while (parentId && byId.has(parentId) && !seen.has(parentId)) {
    seen.add(parentId);
    depth += 1;
    parentId = byId.get(parentId)?.parent_operation_id ?? null;
  }
  return Math.min(depth, 4);
}

function compareTaskTime(left: AgentTask, right: AgentTask): number {
  return (
    comparableTime(left.created_at) - comparableTime(right.created_at) ||
    left.operation_id.localeCompare(right.operation_id)
  );
}

function comparableTime(value: string): number {
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? 0 : parsed;
}

function compareEpisodesNewestFirst(left: Episode, right: Episode): number {
  return (
    comparableTime(right.created_at) - comparableTime(left.created_at) ||
    right.episode_id.localeCompare(left.episode_id)
  );
}
