import type {
  AgentRunConfig,
  AgentTask,
  AgentTaskStatus,
  ChatMessage,
  ChatSummary,
  ConversationMode,
} from "./types";

export type ChatKind = "node_chat" | "project_chat";

export interface ChatConversation {
  chatId: string;
  kind: ChatKind;
  nodeId: string | null;
  title: string;
  tasks: AgentTask[];
  updatedAt: string;
}

export interface DraftConversation {
  chatId: string;
  kind: ChatKind;
  nodeId: string | null;
  title: string;
}

export function chatDraftStorageKey(projectId: string, chatId: string): string {
  return `rcp:chat-draft:${projectId}:${chatId}`;
}

export function chatModeStorageKey(projectId: string, chatId: string): string {
  return `rcp:chat-mode:${projectId}:${chatId}`;
}

export function parseConversationMode(value: unknown): ConversationMode | null {
  return value === "discuss" || value === "work" ? value : null;
}

export function toggleConversationMode(mode: ConversationMode): ConversationMode {
  return mode === "discuss" ? "work" : "discuss";
}

export function isConversationModeShortcut(key: string, shiftKey: boolean): boolean {
  return key === "Tab" && shiftKey;
}

export function latestPersistedConversationMode(
  messages: ChatMessage[],
  tasks: AgentTask[],
): ConversationMode {
  const candidates: Array<{ mode: ConversationMode; timestamp: number; order: number }> = [];
  let order = 0;
  messages.forEach((message) => {
    const mode = parseConversationMode(message.mode);
    if (mode) candidates.push({ mode, timestamp: comparableTime(message.timestamp), order });
    order += 1;
  });
  tasks.forEach((task) => {
    const mode = parseConversationMode(task.request.mode);
    if (mode) candidates.push({ mode, timestamp: comparableTime(task.created_at), order });
    order += 1;
  });
  candidates.sort((left, right) => left.timestamp - right.timestamp || left.order - right.order);
  return candidates.at(-1)?.mode ?? "discuss";
}

/** The transcript records "use the provider default" as a literal sentinel; a
 *  request records it as an empty string. Continuing a conversation reads the
 *  transcript, so the sentinel has to be translated back or it is sent to the
 *  provider as though it were a real model name. */
function persistedModel(value: string | null | undefined): string {
  return !value || value === "provider-default" ? "" : value;
}

export function latestPersistedChatConfig(
  messages: ChatMessage[],
  tasks: AgentTask[],
  fallback: AgentRunConfig,
): AgentRunConfig {
  const candidates: Array<{ config: AgentRunConfig; timestamp: number; order: number }> = [];
  let order = 0;
  messages.forEach((message) => {
    if (
      typeof message.provider === "string" &&
      message.provider.trim() &&
      typeof message.execution_machine === "string" &&
      message.execution_machine.trim()
    ) {
      candidates.push({
        config: {
          ...fallback,
          provider: message.provider,
          model: persistedModel(message.model),
          reasoning: message.reasoning ?? fallback.reasoning,
          run_on: message.execution_machine,
        },
        timestamp: comparableTime(message.timestamp),
        order,
      });
    }
    order += 1;
  });
  tasks.forEach((task) => {
    const request = task.request;
    if (
      typeof request.provider === "string" &&
      request.provider.trim() &&
      typeof request.run_on === "string" &&
      request.run_on.trim()
    ) {
      candidates.push({
        config: {
          ...fallback,
          provider: request.provider,
          model: typeof request.model === "string" ? persistedModel(request.model) : "",
          reasoning: typeof request.reasoning === "string" ? request.reasoning : fallback.reasoning,
          run_on: request.run_on,
        },
        timestamp: comparableTime(task.created_at),
        order,
      });
    }
    order += 1;
  });
  candidates.sort((left, right) => left.timestamp - right.timestamp || left.order - right.order);
  return candidates.at(-1)?.config ?? fallback;
}

export function chatIdForTask(task: AgentTask): string | null {
  if (task.kind !== "node_chat" && task.kind !== "project_chat") return null;
  const value = task.request.chat_id;
  return typeof value === "string" && value.trim() ? value : null;
}

export function groupChatConversations(
  summaries: ChatSummary[],
  tasks: AgentTask[],
  nodeTitles: Record<string, string>,
  projectTitle: string,
  drafts: DraftConversation[] = [],
): ChatConversation[] {
  const grouped = new Map<string, ChatConversation>();
  for (const summary of summaries) {
    const title =
      summary.kind === "node_chat" && summary.node_id
        ? (nodeTitles[summary.node_id] ?? summary.node_id)
        : summary.title;
    grouped.set(summary.chat_id, {
      chatId: summary.chat_id,
      kind: summary.kind,
      nodeId: summary.node_id,
      title,
      tasks: [],
      updatedAt: summary.updated_at,
    });
  }
  for (const draft of drafts) {
    if (!grouped.has(draft.chatId))
      grouped.set(draft.chatId, { ...draft, tasks: [], updatedAt: "" });
  }
  for (const task of tasks) {
    const chatId = chatIdForTask(task);
    if (!chatId) continue;
    const kind = task.kind as ChatKind;
    const nodeId =
      kind === "node_chat" && typeof task.request.node_id === "string"
        ? task.request.node_id
        : null;
    const existing = grouped.get(chatId);
    const title = nodeId ? (nodeTitles[nodeId] ?? nodeId) : projectTitle;
    if (existing) {
      existing.tasks.push(task);
      if (Date.parse(task.updated_at) > Date.parse(existing.updatedAt || "1970-01-01")) {
        existing.updatedAt = task.updated_at;
      }
    } else {
      grouped.set(chatId, {
        chatId,
        kind,
        nodeId,
        title,
        tasks: [task],
        updatedAt: task.updated_at,
      });
    }
  }
  grouped.forEach((conversation) =>
    conversation.tasks.sort(
      (left, right) =>
        Date.parse(left.created_at) - Date.parse(right.created_at) ||
        left.operation_id.localeCompare(right.operation_id),
    ),
  );
  return [...grouped.values()].sort(
    (left, right) =>
      Date.parse(right.updatedAt || "9999-01-01") - Date.parse(left.updatedAt || "9999-01-01") ||
      left.title.localeCompare(right.title),
  );
}

export function latestConversation(
  conversations: ChatConversation[],
  kind: ChatKind,
  nodeId: string | null = null,
): ChatConversation | null {
  return (
    conversations.find(
      (conversation) =>
        conversation.kind === kind && (kind === "project_chat" || conversation.nodeId === nodeId),
    ) ?? null
  );
}

export function chatIndicator(
  tasks: AgentTask[],
  unreadTaskIds: Set<string>,
): "active" | "unread" | null {
  if (tasks.some((task) => chatIdForTask(task) && chatTaskNeedsAttention(task))) return "active";
  if (tasks.some((task) => unreadTaskIds.has(task.operation_id) && chatIdForTask(task)))
    return "unread";
  return null;
}

export function chatTaskNeedsAttention(task: AgentTask): boolean {
  return task.active || task.paused;
}

export function chatEntryConversationId(
  conversations: ChatConversation[],
  activityTask: AgentTask | null,
  unreadTaskIds: Set<string>,
  previousChatId: string | null,
): string | null {
  if (previousChatId && conversations.some((item) => item.chatId === previousChatId))
    return previousChatId;
  const activeChatId =
    activityTask && chatTaskNeedsAttention(activityTask) ? chatIdForTask(activityTask) : null;
  if (activeChatId && conversations.some((item) => item.chatId === activeChatId))
    return activeChatId;
  const unread = conversations.find((conversation) =>
    conversation.tasks.some((task) => unreadTaskIds.has(task.operation_id)),
  );
  if (unread) return unread.chatId;
  return conversations[0]?.chatId ?? null;
}

export function newlyUnreadChatTaskIds(
  tasks: AgentTask[],
  previousStatuses: ReadonlyMap<string, AgentTaskStatus>,
  visibleChatId: string | null,
): string[] {
  return tasks.flatMap((task) => {
    const chatId = chatIdForTask(task);
    const previous = previousStatuses.get(task.operation_id);
    const becameTerminal =
      previous !== undefined && previous !== task.status && !chatTaskNeedsAttention(task);
    return chatId && chatId !== visibleChatId && becameTerminal ? [task.operation_id] : [];
  });
}

export function conversationHasUnread(
  conversation: ChatConversation,
  unreadTaskIds: ReadonlySet<string>,
): boolean {
  return conversation.tasks.some((task) => unreadTaskIds.has(task.operation_id));
}

function comparableTime(value: string): number {
  const timestamp = Date.parse(value);
  return Number.isFinite(timestamp) ? timestamp : 0;
}
