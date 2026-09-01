import type {
  AgentArtifactDescriptor,
  AgentTask,
  AgentTaskKind,
  ChatMessage,
  ChatAttachmentDescriptor,
  ConversationMode,
  GraphUpdateResult,
  TaskTrigger,
} from "./types";

export interface TaskTranscriptLine {
  role: "human" | "agent" | "error" | "meta";
  text: string;
  taskId: string;
  timestamp: string;
  artifacts?: AgentArtifactDescriptor[];
  attachments?: ChatAttachmentDescriptor[];
  mode?: ConversationMode | null;
  trigger?: TaskTrigger;
  graphUpdate?: GraphUpdateResult | null;
}

export function isActiveTask(task: AgentTask): boolean {
  return task.active;
}

export function taskNotificationStorageKey(projectId: string | null): string {
  return `rcp:dismissed-task-notifications:${projectId ?? "none"}`;
}

export function parseDismissedTaskIds(value: string | null): Set<string> {
  if (!value) return new Set();
  try {
    const parsed: unknown = JSON.parse(value);
    return new Set(
      Array.isArray(parsed)
        ? parsed.filter((item): item is string => typeof item === "string" && item.length > 0)
        : [],
    );
  } catch {
    return new Set();
  }
}

export function serializeDismissedTaskIds(taskIds: ReadonlySet<string>): string {
  return JSON.stringify([...taskIds].sort());
}

export function isTaskNotificationSuperseded(task: AgentTask, tasks: AgentTask[]): boolean {
  // A paused turn is the one awaiting-human state the human can still resume, and
  // its notification is the way back to it, so a later success never retires it.
  if ((task.kind !== "seed" && task.kind !== "refresh") || !task.awaiting_human || task.paused)
    return false;
  return tasks.some(
    (candidate) =>
      (candidate.kind === "seed" || candidate.kind === "refresh") &&
      candidate.settled &&
      compareTaskTime(candidate, task) > 0,
  );
}

/** Style follows the answers the projection published, never a status of its own. */
export function agentTaskTone(task: AgentTask): "running" | "failed" | "paused" | "succeeded" {
  if (task.failed) return "failed";
  if (task.awaiting_human) return "paused";
  if (task.active) return "running";
  return "succeeded";
}

export function taskStatusLabel(task: AgentTask): string {
  return task.status_label;
}

export function projectActivityTask(
  tasks: AgentTask[],
  observedTaskId: string | null,
): AgentTask | null {
  const active = tasks.find(isActiveTask);
  if (active) return active;
  const continuedTaskIds = new Set(
    tasks.flatMap((task) => (task.parent_operation_id ? [task.parent_operation_id] : [])),
  );
  const paused = tasks.find((task) => task.paused && !continuedTaskIds.has(task.operation_id));
  if (paused) return isTaskNotificationSuperseded(paused, tasks) ? null : paused;
  if (!observedTaskId) return null;
  const observed = tasks.find((task) => task.operation_id === observedTaskId);
  if (!observed || isTaskNotificationSuperseded(observed, tasks)) return null;
  const byId = new Map(tasks.map((task) => [task.operation_id, task]));
  const descendants = tasks
    .filter((task) => task.operation_id !== observed.operation_id)
    .filter((task) => hasAncestor(task, observed.operation_id, byId))
    .sort(compareTaskTime);
  const latestDescendant = descendants.at(-1);
  if (latestDescendant) {
    if (latestDescendant.settled) return null;
    return latestDescendant;
  }
  return observed;
}

export function taskKindLabel(kind: AgentTaskKind): string {
  switch (kind) {
    case "seed":
      return "Seed project graph";
    case "refresh":
      return "Refresh project graph";
    case "node_chat":
      return "Node conversation";
    case "project_chat":
      return "Project conversation";
    case "paper_coach":
      return "Writing coach";
    case "auto_research":
      return "Auto-research";
    case "branch_merge":
      return "Branch merge";
  }
}

export function relatedChatTasks(
  tasks: AgentTask[],
  kind: "node_chat" | "project_chat",
  nodeId?: string | null,
  requestedChatId?: string | null,
): AgentTask[] {
  const candidates = tasks
    .filter(
      (task) => task.kind === kind && (kind === "project_chat" || task.request.node_id === nodeId),
    )
    .sort(compareTaskTime);
  if (requestedChatId) return candidates.filter((task) => task.request.chat_id === requestedChatId);
  const latest = candidates.at(-1);
  if (!latest) return [];
  const chatId = textValue(latest.request.chat_id);
  return chatId ? candidates.filter((task) => task.request.chat_id === chatId) : [latest];
}

export function relatedCoachTasks(tasks: AgentTask[], sessionId: string | null): AgentTask[] {
  const candidates = tasks.filter((task) => task.kind === "paper_coach").sort(compareTaskTime);
  if (sessionId) {
    return candidates.filter(
      (task) => task.native_session_id === sessionId || task.request.session_id === sessionId,
    );
  }
  return [];
}

export function resumablePausedChatTask(tasks: AgentTask[]): AgentTask | null {
  const continuedTaskIds = new Set(
    tasks.flatMap((task) => (task.parent_operation_id ? [task.parent_operation_id] : [])),
  );
  return (
    [...tasks]
      .reverse()
      .find((task) => task.paused && task.can_resume && !continuedTaskIds.has(task.operation_id)) ??
    null
  );
}

export function chatTasksMissingFromHistory(
  tasks: AgentTask[],
  messages: ChatMessage[],
): AgentTask[] {
  const persistedOperationIds = new Set(
    messages.flatMap((message) => (message.operation_id ? [message.operation_id] : [])),
  );
  // Operation id is the turn identity. Matching by prompt text loses one of
  // two legitimate turns when the human sends the same message twice.
  return tasks.filter((task) => !persistedOperationIds.has(task.operation_id));
}

export function chatMessageTranscriptLine(message: ChatMessage): TaskTranscriptLine {
  return {
    role: message.role === "user" && message.trigger !== "watcher" ? "human" : "agent",
    text: message.text,
    taskId: message.operation_id ?? message.message_id,
    timestamp: message.timestamp,
    mode: message.mode,
    trigger: message.trigger,
    graphUpdate: message.graph_update,
    attachments: message.attachments,
  };
}

export function reconcileChatHistoryArtifacts(
  messages: ChatMessage[],
  tasks: AgentTask[],
): TaskTranscriptLine[] {
  const lines = messages.map(chatMessageTranscriptLine);
  const answerLineByOperationId = new Map<string, number>();
  messages.forEach((message, index) => {
    if (
      message.role === "assistant" &&
      message.operation_id &&
      !answerLineByOperationId.has(message.operation_id)
    ) {
      answerLineByOperationId.set(message.operation_id, index);
    }
  });
  tasks.forEach((task) => {
    const artifacts = taskArtifacts(task);
    const lineIndex = answerLineByOperationId.get(task.operation_id);
    if (!artifacts.length || lineIndex === undefined) return;
    lines[lineIndex] = { ...lines[lineIndex], artifacts };
  });
  return lines;
}

export function reconstructTaskTranscript(tasks: AgentTask[]): TaskTranscriptLine[] {
  return [...tasks].sort(compareTaskTime).flatMap((task) => {
    const lines: TaskTranscriptLine[] = [];
    const message = textValue(task.request.message);
    const mode = conversationMode(task.request.mode);
    const trigger = taskTrigger(task.request.trigger);
    const graphUpdate = task.result?.graph_update ?? null;
    if (message && trigger === "human") {
      const attachments = taskAttachments(task.request.attachments);
      lines.push({
        role: "human",
        text: message,
        taskId: task.operation_id,
        timestamp: task.created_at,
        mode,
        trigger,
        ...(attachments.length ? { attachments } : {}),
      });
    }
    const messages = Array.isArray(task.result?.messages)
      ? task.result.messages.filter(
          (item): item is string => typeof item === "string" && item.trim().length > 0,
        )
      : [];
    const artifacts = taskArtifacts(task);
    messages.forEach((text, index) =>
      lines.push({
        role: "agent",
        text,
        taskId: task.operation_id,
        timestamp: task.created_at,
        mode,
        trigger,
        ...(index === messages.length - 1 && artifacts.length ? { artifacts } : {}),
        ...(index === messages.length - 1 && graphUpdate ? { graphUpdate } : {}),
      }),
    );
    if (!messages.length && (artifacts.length || (graphUpdate && graphUpdate.status !== "none"))) {
      lines.push({
        role: "agent",
        text: "",
        taskId: task.operation_id,
        timestamp: task.created_at,
        mode,
        trigger,
        ...(artifacts.length ? { artifacts } : {}),
        ...(graphUpdate ? { graphUpdate } : {}),
      });
    }
    const graphOnlyRejection = task.settled && graphUpdate?.status === "rejected";
    if (task.error && !graphOnlyRejection) {
      lines.push({
        role: "error",
        text: task.error,
        taskId: task.operation_id,
        timestamp: task.created_at,
        trigger,
      });
    } else if (task.awaiting_human) {
      lines.push({
        role: task.failed ? "error" : "meta",
        text: task.status_message,
        taskId: task.operation_id,
        timestamp: task.created_at,
        trigger,
      });
    }
    return lines;
  });
}

function taskAttachments(value: unknown): ChatAttachmentDescriptor[] {
  if (!Array.isArray(value)) return [];
  return value.filter(
    (item): item is ChatAttachmentDescriptor =>
      typeof item === "object" &&
      item !== null &&
      typeof item.attachment_id === "string" &&
      typeof item.name === "string" &&
      typeof item.media_type === "string" &&
      typeof item.size === "number" &&
      typeof item.expires_at === "string",
  );
}

export function orderTranscriptLines(lines: TaskTranscriptLine[]): TaskTranscriptLine[] {
  return [...lines].sort(
    (left, right) => comparableTime(left.timestamp) - comparableTime(right.timestamp),
  );
}

export function artifactUrl(
  projectId: string,
  taskId: string,
  artifactId: string,
  action: "content" | "preview" | "viewer" | "download",
): string {
  return `/api/projects/${encodeURIComponent(projectId)}/tasks/${encodeURIComponent(taskId)}/artifacts/${encodeURIComponent(artifactId)}/${action}`;
}

export function latestNativeSessionId(tasks: AgentTask[]): string | null {
  return (
    [...tasks]
      .sort(compareTaskTime)
      .reverse()
      .find((task) => task.native_session_id)?.native_session_id ?? null
  );
}

function compareTaskTime(left: AgentTask, right: AgentTask): number {
  return (
    Date.parse(left.created_at) - Date.parse(right.created_at) ||
    left.operation_id.localeCompare(right.operation_id)
  );
}

function comparableTime(value: string): number {
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? 0 : parsed;
}

function hasAncestor(
  task: AgentTask,
  ancestorId: string,
  byId: ReadonlyMap<string, AgentTask>,
): boolean {
  const seen = new Set<string>();
  let parentId = task.parent_operation_id;
  while (parentId && !seen.has(parentId)) {
    if (parentId === ancestorId) return true;
    seen.add(parentId);
    parentId = byId.get(parentId)?.parent_operation_id ?? null;
  }
  return false;
}

function textValue(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value : null;
}

function conversationMode(value: unknown): ConversationMode | null {
  return value === "discuss" || value === "work" ? value : null;
}

function taskTrigger(value: unknown): TaskTrigger {
  return value === "experiment_run" || value === "watcher" ? value : "human";
}

function taskArtifacts(task: AgentTask): AgentArtifactDescriptor[] {
  if (!Array.isArray(task.result?.artifacts)) return [];
  return task.result.artifacts.filter(
    (item): item is AgentArtifactDescriptor =>
      typeof item === "object" &&
      item !== null &&
      typeof item.artifact_id === "string" &&
      typeof item.name === "string" &&
      typeof item.media_type === "string" &&
      typeof item.available === "boolean" &&
      (item.unavailable_reason === null || typeof item.unavailable_reason === "string") &&
      typeof item.can_open === "boolean" &&
      typeof item.can_download === "boolean" &&
      typeof item.can_keep === "boolean" &&
      typeof item.can_revise === "boolean",
  );
}
