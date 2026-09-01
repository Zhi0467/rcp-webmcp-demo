import {
  artifactUrl,
  latestNativeSessionId,
  relatedChatTasks,
  resumablePausedChatTask,
  taskArtifacts,
} from "./agentTasks";
import { episodeReportPreviewUrl } from "./campaigns";
import {
  latestPersistedChatConfig,
  type ChatKind,
  type ConversationTurnSubmission,
} from "./chatWorkspace";
import { filterSkillCatalogToDefaults } from "./skillPicker";
import type {
  AgentRunConfig,
  AgentTask,
  ChatMessage,
  ChatTranscript,
  Episode,
  GraphNode,
  ProjectCard,
  ProjectSnapshot,
  WatcherRecord,
} from "./types";

export type WebMcpJsonSchema = {
  type: "object";
  properties?: Record<string, unknown>;
  required?: string[];
  additionalProperties: false;
};

export type WebMcpToolResult = {
  content: Array<{ type: "text"; text: string }>;
};

export type WebMcpToolDefinition = {
  name: string;
  description: string;
  inputSchema: WebMcpJsonSchema;
  annotations?: {
    readOnlyHint?: boolean;
    untrustedContentHint?: boolean;
  };
  execute: (input: Record<string, unknown>) => WebMcpToolResult | Promise<WebMcpToolResult>;
};

export type WebMcpModelContext = {
  registerTool: (
    definition: WebMcpToolDefinition,
    options?: { signal?: AbortSignal },
  ) => void | Promise<void>;
};

type DocumentWithModelContext = {
  modelContext?: unknown;
};

export type WebMcpRegistration = {
  controller: AbortController;
  dispose: () => void;
};

export type WebMcpToolRegistry = {
  update: (definitions: WebMcpToolDefinition[]) => void;
  dispose: () => void;
};

export const WEBMCP_RESULT_MAX_CHARS = 1_500;
const WEBMCP_NODE_RESULT_MAX_CHARS = 12_000;
const WEBMCP_ARTIFACT_RESULT_MAX_CHARS = 8_000;
const WEBMCP_CONVERSATION_RESULT_MAX_CHARS = 12_000;
const WEBMCP_EXPERIMENT_RESULT_MAX_CHARS = 4_000;
const WEBMCP_PROJECT_INDEX_RESULT_MAX_CHARS = 6_000;
const PROJECT_LIST_LIMIT = 8;
const OVERVIEW_LIST_LIMIT = 2;
const NODE_RELATION_LIMIT = 32;
const ARTIFACT_LIST_LIMIT = 8;
const CONVERSATION_MESSAGE_LIMIT = 6;
const CONVERSATION_MESSAGE_TEXT_LIMIT = 320;
const CONVERSATION_ARTIFACT_LIMIT = 4;
const RCP_SKILL_LIST_LIMIT = 4;
const PROVIDER_SKILL_LIST_LIMIT = 6;

export function modelContextFromDocument(value: unknown): WebMcpModelContext | null {
  if (!value || typeof value !== "object") return null;
  const candidate = (value as DocumentWithModelContext).modelContext;
  if (!candidate || typeof candidate !== "object") return null;
  const registerTool = (candidate as { registerTool?: unknown }).registerTool;
  if (typeof registerTool !== "function") return null;
  return candidate as WebMcpModelContext;
}

export function currentWebMcpContext(): WebMcpModelContext | null {
  return typeof document === "undefined" ? null : modelContextFromDocument(document);
}

function observeWebMcpRegistration(result: void | Promise<void>, signal: AbortSignal): void {
  if (!result) return;
  void result.catch((error: unknown) => {
    if (
      signal.aborted &&
      typeof error === "object" &&
      error !== null &&
      "name" in error &&
      error.name === "AbortError"
    ) {
      return;
    }
    console.error("WebMCP tool registration failed.", error);
  });
}

export function registerWebMcpTools(
  definitions: WebMcpToolDefinition[],
  context: WebMcpModelContext | null = currentWebMcpContext(),
): WebMcpRegistration | null {
  if (!context || definitions.length === 0) return null;
  const controller = new AbortController();
  try {
    definitions.forEach((definition) => {
      observeWebMcpRegistration(
        context.registerTool(definition, { signal: controller.signal }),
        controller.signal,
      );
    });
  } catch (error) {
    controller.abort();
    throw error;
  }
  return {
    controller,
    dispose: () => controller.abort(),
  };
}

export function createWebMcpToolRegistry(
  definitions: WebMcpToolDefinition[],
  context: WebMcpModelContext | null = currentWebMcpContext(),
): WebMcpToolRegistry | null {
  if (!context || definitions.length === 0) return null;
  const current = new Map<string, WebMcpToolDefinition>();
  const registrations = new Map<
    string,
    {
      registration: WebMcpRegistration;
      activeCalls: number;
      retired: boolean;
      retireTimer: ReturnType<typeof setTimeout> | null;
    }
  >();
  let disposed = false;

  const finishCall = (name: string): void => {
    const entry = registrations.get(name);
    if (!entry) return;
    entry.activeCalls -= 1;
    if (!entry.retired || entry.activeCalls !== 0 || entry.retireTimer !== null) return;
    entry.retireTimer = setTimeout(() => {
      entry.retireTimer = null;
      if (!entry.retired || entry.activeCalls !== 0) return;
      entry.registration.dispose();
      registrations.delete(name);
    }, 0);
  };

  const executeCurrent = (
    name: string,
    input: Record<string, unknown>,
  ): WebMcpToolResult | Promise<WebMcpToolResult> => {
    const latest = current.get(name);
    const entry = registrations.get(name);
    if (!latest || !entry || entry.retired) {
      throw new Error(`WebMCP tool ${name} is not currently available.`);
    }
    entry.activeCalls += 1;
    try {
      const result = latest.execute(input);
      if (result && typeof (result as Promise<WebMcpToolResult>).then === "function") {
        return Promise.resolve(result).finally(() => finishCall(name));
      }
      finishCall(name);
      return result;
    } catch (error) {
      finishCall(name);
      throw error;
    }
  };

  const update = (nextDefinitions: WebMcpToolDefinition[]): void => {
    if (disposed) throw new Error("Cannot update a disposed WebMCP tool registry.");
    const nextNames = new Set(nextDefinitions.map((definition) => definition.name));
    for (const name of current.keys()) {
      if (nextNames.has(name)) continue;
      current.delete(name);
      const entry = registrations.get(name);
      if (!entry) continue;
      entry.retired = true;
      if (entry.activeCalls === 0) {
        entry.registration.dispose();
        registrations.delete(name);
      }
    }
    for (const definition of nextDefinitions) {
      current.set(definition.name, definition);
      const existing = registrations.get(definition.name);
      if (existing) {
        existing.retired = false;
        if (existing.retireTimer !== null) {
          clearTimeout(existing.retireTimer);
          existing.retireTimer = null;
        }
        continue;
      }
      const proxy: WebMcpToolDefinition = {
        ...definition,
        execute: (input) => executeCurrent(definition.name, input),
      };
      const registration = registerWebMcpTools([proxy], context);
      if (registration) {
        registrations.set(definition.name, {
          registration,
          activeCalls: 0,
          retired: false,
          retireTimer: null,
        });
      }
    }
  };

  update(definitions);
  return {
    update,
    dispose: () => {
      if (disposed) return;
      disposed = true;
      registrations.forEach((entry) => {
        if (entry.retireTimer !== null) clearTimeout(entry.retireTimer);
        entry.registration.dispose();
      });
      registrations.clear();
      current.clear();
    },
  };
}

export function webMcpTextResult(
  value: unknown,
  maxChars: number = WEBMCP_RESULT_MAX_CHARS,
): WebMcpToolResult {
  const text = JSON.stringify(value);
  if (text === undefined) throw new Error("WebMCP tool result is not JSON serializable.");
  if (text.length > maxChars) {
    throw new Error(`WebMCP tool result exceeds ${maxChars} characters.`);
  }
  return { content: [{ type: "text", text }] };
}

function compactText(value: string, maxChars: number): string {
  return value.length <= maxChars ? value : `${value.slice(0, maxChars - 1)}…`;
}

function compactNode(node: GraphNode): Record<string, unknown> {
  return {
    id: node.id,
    type: node.type,
    title: compactText(node.title, 96),
    standing: node.standing,
    ...(node.status ? { status: node.status } : {}),
  };
}

function compactProjectCard(project: ProjectCard): Record<string, unknown> {
  return {
    id: project.id,
    name: compactText(project.name, 96),
    primary_question: project.primary_question ? compactText(project.primary_question, 160) : null,
    remote: project.remote,
    reachable: project.reachable ?? null,
    revision: project.revision ?? null,
    attention_count: project.attention_count,
  };
}

export function listProjectsForWebMcp(
  projects: ProjectCard[],
  input: Record<string, unknown>,
): Record<string, unknown> {
  const query = optionalStringInput(input, "query")?.trim().toLowerCase() ?? null;
  const matches = query
    ? projects.filter((project) =>
        [project.id, project.name, project.primary_question ?? ""].some((value) =>
          value.toLowerCase().includes(query),
        ),
      )
    : projects;
  const visible = matches.slice(0, PROJECT_LIST_LIMIT);
  return {
    total: projects.length,
    matched: matches.length,
    returned: visible.length,
    truncated: matches.length > visible.length,
    projects: visible.map(compactProjectCard),
  };
}

export async function openProjectFromIndex(
  projects: ProjectCard[],
  input: Record<string, unknown>,
  openProject: (projectId: string) => boolean | void | Promise<boolean | void>,
): Promise<Record<string, unknown>> {
  const projectId = requiredStringInput(input, "project_id");
  const project = projects.find((candidate) => candidate.id === projectId);
  if (!project)
    throw new Error(`Project ${projectId} is not present in the current project index.`);
  if ((await openProject(project.id)) === false) {
    throw new Error(
      `Project ${projectId} requires the existing desktop access review before it can open.`,
    );
  }
  return {
    navigation_requested: true,
    target_view: "project",
    project: compactProjectCard(project),
  };
}

export function projectIndexToolDefinitions(
  currentProjects: () => ProjectCard[],
  openProject: (projectId: string) => boolean | void | Promise<boolean | void>,
): WebMcpToolDefinition[] {
  return [
    {
      name: "rcp_list_projects",
      description: "List the RCP projects available from the current project index.",
      inputSchema: {
        type: "object",
        properties: {
          query: {
            type: "string",
            minLength: 1,
            description: "Optional words from the project name or primary research question.",
          },
        },
        additionalProperties: false,
      },
      annotations: { readOnlyHint: true, untrustedContentHint: true },
      execute: (input) =>
        webMcpTextResult(
          listProjectsForWebMcp(currentProjects(), input),
          WEBMCP_PROJECT_INDEX_RESULT_MAX_CHARS,
        ),
    },
    {
      name: "rcp_open_project",
      description: "Open one exact listed RCP project in the current browser page.",
      inputSchema: {
        type: "object",
        properties: {
          project_id: {
            type: "string",
            minLength: 1,
            description: "Exact project id returned by rcp_list_projects.",
          },
        },
        required: ["project_id"],
        additionalProperties: false,
      },
      annotations: { readOnlyHint: true, untrustedContentHint: true },
      execute: async (input) =>
        webMcpTextResult(await openProjectFromIndex(currentProjects(), input, openProject)),
    },
  ];
}

function recentNodes(project: ProjectSnapshot, type: GraphNode["type"]): GraphNode[] {
  return Object.values(project.graph.nodes)
    .filter((node) => node.type === type)
    .sort((left, right) => right.updated_rev - left.updated_rev || left.id.localeCompare(right.id))
    .slice(0, OVERVIEW_LIST_LIMIT);
}

export function projectOverview(project: ProjectSnapshot): Record<string, unknown> {
  const attentionIds = [
    ...project.attention.open_blocker_ids,
    ...project.attention.decisions_awaiting_choice_ids,
    ...project.attention.pending_proposal_ids,
  ];
  return {
    project: {
      id: project.id,
      name: compactText(project.name, 96),
      revision: project.revision,
      freshness: project.snapshot_freshness,
    },
    primary_question: project.primary_question ? compactNode(project.primary_question) : null,
    counts: project.counts,
    recent: {
      hypotheses: recentNodes(project, "hypothesis").map(compactNode),
      experiments: recentNodes(project, "experiment").map(compactNode),
      evidence: recentNodes(project, "evidence").map(compactNode),
      blockers: recentNodes(project, "blocker").map(compactNode),
    },
    suggested_node_ids: [...new Set(attentionIds)].slice(0, 6),
  };
}

function requiredStringInput(input: Record<string, unknown>, name: string): string {
  const value = input[name];
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`${name} must be a non-blank string.`);
  }
  return value;
}

function optionalStringInput(input: Record<string, unknown>, name: string): string | null {
  const value = input[name];
  if (value === undefined) return null;
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`${name} must be a non-blank string when supplied.`);
  }
  return value;
}

function stringListInput(input: Record<string, unknown>, name: string, maximum = 32): string[] {
  const value = input[name];
  if (value === undefined) return [];
  if (!Array.isArray(value) || value.length > maximum) {
    throw new Error(`${name} must be an array of at most ${maximum} strings.`);
  }
  const items = value.map((item) => {
    if (typeof item !== "string" || !item.trim()) {
      throw new Error(`${name} must contain only non-blank strings.`);
    }
    return item;
  });
  if (new Set(items).size !== items.length) throw new Error(`${name} must not contain duplicates.`);
  return items;
}

export function inspectProjectNode(
  project: ProjectSnapshot,
  input: Record<string, unknown>,
): Record<string, unknown> {
  const nodeId = requiredStringInput(input, "node_id");
  const node = project.graph.nodes[nodeId];
  if (!node) throw new Error(`Node ${nodeId} is not present in the current project graph.`);
  const allRelations = Object.values(project.graph.edges)
    .filter((edge) => edge.source === nodeId || edge.target === nodeId)
    .sort((left, right) => left.id.localeCompare(right.id))
    .map((edge) => ({
      edge_id: edge.id,
      source_id: edge.source,
      target_id: edge.target,
      relation: edge.relation,
      layer: edge.layer,
    }));
  const relations = allRelations.slice(0, NODE_RELATION_LIMIT);
  const allRelatedNodeIds = allRelations.map((edge) =>
    edge.source_id === nodeId ? edge.target_id : edge.source_id,
  );
  const relatedNodeIds = [...new Set(allRelatedNodeIds)];
  return {
    project_id: project.id,
    graph_revision: project.graph.revision,
    node,
    relation_count: allRelations.length,
    relations_truncated: allRelations.length > relations.length,
    relations,
    related_node_ids: relatedNodeIds.slice(0, NODE_RELATION_LIMIT),
    related_node_ids_truncated: relatedNodeIds.length > NODE_RELATION_LIMIT,
    ...(node.type === "experiment"
      ? { experiment_control: project.experiment_control[node.id] ?? null }
      : {}),
  };
}

export function projectReadToolDefinitions(project: ProjectSnapshot): WebMcpToolDefinition[] {
  return [
    {
      name: "rcp_get_project_overview",
      description: "Read a compact map of the open RCP research project.",
      inputSchema: { type: "object", additionalProperties: false },
      annotations: { readOnlyHint: true, untrustedContentHint: true },
      execute: () => webMcpTextResult(projectOverview(project)),
    },
    {
      name: "rcp_inspect_node",
      description: "Read one exact saved RCP graph node and its direct relations.",
      inputSchema: {
        type: "object",
        properties: {
          node_id: {
            type: "string",
            minLength: 1,
            description: "Exact current graph node id returned by an RCP read tool.",
          },
        },
        required: ["node_id"],
        additionalProperties: false,
      },
      annotations: { readOnlyHint: true, untrustedContentHint: true },
      execute: (input) =>
        webMcpTextResult(inspectProjectNode(project, input), WEBMCP_NODE_RESULT_MAX_CHARS),
    },
  ];
}

type ArtifactFilter = {
  nodeId: string | null;
  chatId: string | null;
  taskId: string | null;
  episodeId: string | null;
};

type ProjectArtifactRecord = {
  viewer_id: string;
  kind: "task_artifact" | "episode_report";
  name: string;
  media_type: string;
  available: boolean;
  can_open: boolean;
  task_id: string | null;
  chat_id: string | null;
  node_id: string | null;
  episode_id: string | null;
  kept_filename: string | null;
  unavailable_reason: string | null;
  viewer_url: string;
  content_url: string;
  sort_time: string;
};

function artifactFilter(input: Record<string, unknown>): ArtifactFilter {
  const filter = {
    nodeId: optionalStringInput(input, "node_id"),
    chatId: optionalStringInput(input, "chat_id"),
    taskId: optionalStringInput(input, "task_id"),
    episodeId: optionalStringInput(input, "episode_id"),
  };
  if (Object.values(filter).filter(Boolean).length > 1) {
    throw new Error("Supply at most one artifact filter.");
  }
  return filter;
}

function taskNodeId(task: AgentTask): string | null {
  const value = task.request.control_node_id ?? task.request.node_id;
  return typeof value === "string" && value ? value : null;
}

function taskChatId(task: AgentTask): string | null {
  const value = task.request.chat_id;
  return typeof value === "string" && value ? value : null;
}

function taskMatchesArtifactFilter(task: AgentTask, filter: ArtifactFilter): boolean {
  if (filter.nodeId) return taskNodeId(task) === filter.nodeId;
  if (filter.chatId) return taskChatId(task) === filter.chatId;
  if (filter.taskId) return task.operation_id === filter.taskId;
  if (filter.episodeId) return task.episode_id === filter.episodeId;
  return true;
}

function episodeMatchesArtifactFilter(
  episode: Episode,
  tasks: AgentTask[],
  filter: ArtifactFilter,
): boolean {
  if (filter.nodeId) return episode.control_node_id === filter.nodeId;
  if (filter.chatId) {
    return tasks.some(
      (task) => task.episode_id === episode.episode_id && taskChatId(task) === filter.chatId,
    );
  }
  if (filter.taskId) return false;
  if (filter.episodeId) return episode.episode_id === filter.episodeId;
  return true;
}

function assertArtifactFilterTarget(
  project: ProjectSnapshot,
  tasks: AgentTask[],
  episodes: Episode[],
  filter: ArtifactFilter,
): void {
  if (filter.nodeId && !project.graph.nodes[filter.nodeId]) {
    throw new Error(`Node ${filter.nodeId} is not present in the current project graph.`);
  }
  if (filter.taskId && !tasks.some((task) => task.operation_id === filter.taskId)) {
    throw new Error(`Task ${filter.taskId} is not present in the current project.`);
  }
  if (filter.chatId && !tasks.some((task) => taskChatId(task) === filter.chatId)) {
    throw new Error(`Conversation ${filter.chatId} is not present in the current project.`);
  }
  if (filter.episodeId && !episodes.some((episode) => episode.episode_id === filter.episodeId)) {
    throw new Error(`Episode ${filter.episodeId} is not present in the current project.`);
  }
}

function projectArtifactRecords(
  project: ProjectSnapshot,
  tasks: AgentTask[],
  episodes: Episode[],
  input: Record<string, unknown>,
): ProjectArtifactRecord[] {
  const filter = artifactFilter(input);
  assertArtifactFilterTarget(project, tasks, episodes, filter);
  const artifacts = tasks
    .filter((task) => taskMatchesArtifactFilter(task, filter))
    .flatMap((task) =>
      taskArtifacts(task).map((artifact) => ({
        viewer_id: `task:${task.operation_id}:${artifact.artifact_id}`,
        kind: "task_artifact" as const,
        name: compactText(artifact.name, 120),
        media_type: artifact.media_type,
        available: artifact.available,
        can_open: artifact.can_open,
        task_id: task.operation_id,
        chat_id: taskChatId(task),
        node_id: taskNodeId(task),
        episode_id: task.episode_id ?? null,
        kept_filename: artifact.kept_filename ?? null,
        unavailable_reason: artifact.unavailable_reason
          ? compactText(artifact.unavailable_reason, 160)
          : null,
        viewer_url: artifactUrl(project.id, task.operation_id, artifact.artifact_id, "viewer"),
        content_url: artifactUrl(project.id, task.operation_id, artifact.artifact_id, "content"),
        sort_time: task.updated_at,
      })),
    );
  const reports = episodes
    .filter(
      (episode) =>
        episode.report &&
        episode.wrapup_state === "ready" &&
        episodeMatchesArtifactFilter(episode, tasks, filter),
    )
    .map((episode) => ({
      viewer_id: `report:${episode.episode_id}`,
      kind: "episode_report" as const,
      name: `${episode.ending ?? "Experiment"} episode report`,
      media_type: "text/html",
      available: true,
      can_open: true,
      task_id: null,
      chat_id: null,
      node_id: episode.control_node_id,
      episode_id: episode.episode_id,
      kept_filename: null,
      unavailable_reason: null,
      viewer_url: episodeReportPreviewUrl(project.id, episode.episode_id),
      content_url: `/api/projects/${encodeURIComponent(project.id)}/episodes/${encodeURIComponent(episode.episode_id)}/report/content`,
      sort_time: episode.report?.created_at ?? episode.updated_at,
    }));
  return [...artifacts, ...reports].sort(
    (left, right) =>
      Date.parse(right.sort_time) - Date.parse(left.sort_time) ||
      left.viewer_id.localeCompare(right.viewer_id),
  );
}

export function listProjectArtifacts(
  project: ProjectSnapshot,
  tasks: AgentTask[],
  episodes: Episode[],
  input: Record<string, unknown>,
): Record<string, unknown> {
  const records = projectArtifactRecords(project, tasks, episodes, input);
  return {
    project_id: project.id,
    total: records.length,
    truncated: records.length > ARTIFACT_LIST_LIMIT,
    artifacts: records
      .slice(0, ARTIFACT_LIST_LIMIT)
      .map(({ sort_time: _, viewer_url: __, content_url: ___, ...record }) => record),
  };
}

export async function openProjectArtifact(
  project: ProjectSnapshot,
  tasks: AgentTask[],
  episodes: Episode[],
  input: Record<string, unknown>,
  openViewer: (viewerUrl: string, contentUrl: string) => boolean | Promise<boolean>,
): Promise<Record<string, unknown>> {
  const viewerId = requiredStringInput(input, "viewer_id");
  const record = projectArtifactRecords(project, tasks, episodes, {}).find(
    (candidate) => candidate.viewer_id === viewerId,
  );
  if (!record)
    throw new Error(`Artifact viewer ${viewerId} is not present in the current project.`);
  if (!record.available || !record.can_open) {
    throw new Error(record.unavailable_reason ?? `Artifact viewer ${viewerId} is unavailable.`);
  }
  if (!(await openViewer(record.viewer_url, record.content_url))) {
    throw new Error("The RCP artifact viewer could not be shown.");
  }
  return {
    project_id: project.id,
    viewer_id: record.viewer_id,
    kind: record.kind,
    opened: true,
  };
}

export function projectArtifactToolDefinitions(
  project: ProjectSnapshot,
  tasks: AgentTask[],
  episodes: Episode[],
  openViewer: (viewerUrl: string, contentUrl: string) => boolean | Promise<boolean>,
): WebMcpToolDefinition[] {
  return [
    {
      name: "rcp_list_artifacts",
      description: "List RCP task artifacts and immutable episode reports in the open project.",
      inputSchema: {
        type: "object",
        properties: {
          node_id: {
            type: "string",
            minLength: 1,
            description: "Optional exact graph node id whose artifacts should be listed.",
          },
          chat_id: {
            type: "string",
            minLength: 1,
            description: "Optional exact conversation id whose artifacts should be listed.",
          },
          task_id: {
            type: "string",
            minLength: 1,
            description: "Optional exact task id whose artifacts should be listed.",
          },
          episode_id: {
            type: "string",
            minLength: 1,
            description: "Optional exact episode id whose artifacts and report should be listed.",
          },
        },
        additionalProperties: false,
      },
      annotations: { readOnlyHint: true, untrustedContentHint: true },
      execute: (input) =>
        webMcpTextResult(
          listProjectArtifacts(project, tasks, episodes, input),
          WEBMCP_ARTIFACT_RESULT_MAX_CHARS,
        ),
    },
    {
      name: "rcp_open_artifact",
      description: "Open one listed artifact or report in RCP's existing visual viewer.",
      inputSchema: {
        type: "object",
        properties: {
          viewer_id: {
            type: "string",
            minLength: 1,
            description: "Exact viewer id returned by rcp_list_artifacts.",
          },
        },
        required: ["viewer_id"],
        additionalProperties: false,
      },
      annotations: { readOnlyHint: true, untrustedContentHint: true },
      execute: async (input) =>
        webMcpTextResult(await openProjectArtifact(project, tasks, episodes, input, openViewer)),
    },
  ];
}

function compactConversationMessage(message: ChatMessage): Record<string, unknown> {
  return {
    message_id: message.message_id,
    task_id: message.operation_id ?? null,
    role: message.role,
    mode: message.mode,
    trigger: message.trigger,
    timestamp: message.timestamp,
    text: compactText(message.text, CONVERSATION_MESSAGE_TEXT_LIMIT),
    text_truncated: message.text.length > CONVERSATION_MESSAGE_TEXT_LIMIT,
  };
}

function latestConversationTask(tasks: AgentTask[]): AgentTask | null {
  return (
    [...tasks]
      .sort(
        (left, right) =>
          Date.parse(left.created_at) - Date.parse(right.created_at) ||
          left.operation_id.localeCompare(right.operation_id),
      )
      .at(-1) ?? null
  );
}

function latestAssistantAnswer(
  transcript: ChatTranscript,
  latestTask: AgentTask | null,
): string | null {
  const matching = latestTask
    ? [...transcript.messages]
        .reverse()
        .find(
          (message) =>
            message.role === "assistant" && message.operation_id === latestTask.operation_id,
        )
    : null;
  const fallback = [...transcript.messages]
    .reverse()
    .find((message) => message.role === "assistant");
  const taskFallback = Array.isArray(latestTask?.result?.messages)
    ? latestTask.result.messages
        .filter(
          (message): message is string => typeof message === "string" && Boolean(message.trim()),
        )
        .at(-1)
    : null;
  const answer = matching?.text ?? taskFallback ?? fallback?.text;
  return answer ? compactText(answer, 1_000) : null;
}

function compactGraphUpdate(task: AgentTask | null): Record<string, unknown> | null {
  const update = task?.result?.graph_update;
  if (!update) return null;
  return {
    status: update.status,
    applied_revision: update.applied_revision,
    change_summary: update.change_summary.slice(0, 3).map((item) => compactText(item, 140)),
    proposal_ids: update.proposal_ids.slice(0, 8),
    validation_messages: update.validation_messages
      .slice(0, 3)
      .map((item) => compactText(item, 140)),
    correction_rounds: update.correction_rounds,
    repairable: update.repairable,
  };
}

function conversationRefusal(
  latestTask: AgentTask | null,
  relatedTasks: AgentTask[],
  taskStartPending: boolean,
  runTruthScope: string[],
  providerReady: boolean,
): string | null {
  if (latestTask?.graph_target.kind === "branch") {
    return "This Auto-research branch conversation is read-only.";
  }
  const active = relatedTasks.find((task) => task.active);
  if (active) return active.status_message || "A turn in this conversation is already active.";
  const paused = resumablePausedChatTask(relatedTasks);
  if (paused) return paused.status_message || "Resume or retry the paused turn before sending.";
  if (taskStartPending) return "Another task start is already being submitted.";
  if (!runTruthScope.length) return "This conversation has no repository scope.";
  if (!providerReady) return "The configured provider is not ready on its execution machine.";
  return null;
}

async function resolveProjectConversationContext(
  project: ProjectSnapshot,
  tasks: AgentTask[],
  chatId: string,
  loadTranscript: (chatId: string) => Promise<ChatTranscript>,
  taskStartPending: boolean,
) {
  const transcript = await loadTranscript(chatId);
  if (transcript.chat_id !== chatId) {
    throw new Error(`Conversation ${chatId} returned a mismatched transcript.`);
  }
  const surface = transcript.kind;
  const node = transcript.node_id ? project.graph.nodes[transcript.node_id] : null;
  if (surface === "node_chat" && !node) {
    throw new Error(`Conversation ${chatId} points to a node outside the current project graph.`);
  }
  const relatedTasks = relatedChatTasks(tasks, surface, transcript.node_id, chatId);
  const latestTask = latestConversationTask(relatedTasks);
  const profile = project.agent_profiles[surface];
  const fallbackConfig: AgentRunConfig = {
    provider: profile.provider,
    model: profile.model,
    reasoning: profile.reasoning,
    run_on: profile.run_on,
  };
  const config = latestPersistedChatConfig(transcript.messages, relatedTasks, fallbackConfig);
  const readiness = project.provider_readiness[config.run_on]?.[config.provider];
  const providerReady =
    readiness === undefined || Boolean(readiness.installed && readiness.authenticated);
  const runTruthScope =
    latestTask?.request.run_truth_scope ?? project.default_run_truth_scope ?? [];
  return {
    transcript,
    surface,
    node,
    relatedTasks,
    latestTask,
    profile,
    config,
    readiness,
    providerReady,
    runTruthScope,
    sessionId:
      latestNativeSessionId(relatedTasks) ??
      [...transcript.messages].reverse().find((message) => message.native_session_id)
        ?.native_session_id ??
      null,
    refusal: conversationRefusal(
      latestTask,
      relatedTasks,
      taskStartPending,
      runTruthScope,
      providerReady,
    ),
  };
}

export async function inspectProjectConversation(
  project: ProjectSnapshot,
  tasks: AgentTask[],
  input: Record<string, unknown>,
  loadTranscript: (chatId: string) => Promise<ChatTranscript>,
  taskStartPending = false,
): Promise<Record<string, unknown>> {
  const chatId = requiredStringInput(input, "chat_id");
  const context = await resolveProjectConversationContext(
    project,
    tasks,
    chatId,
    loadTranscript,
    taskStartPending,
  );
  const {
    transcript,
    surface,
    node,
    latestTask,
    profile,
    config,
    readiness,
    providerReady,
    runTruthScope,
    sessionId,
    refusal,
  } = context;
  const enabledCatalog = filterSkillCatalogToDefaults(
    project.skill_catalog ?? [],
    project.skill_defaults ?? { workflow_ids: [], skill_ids: [] },
  );
  const providerInventory =
    project.provider_skill_inventories?.[config.run_on]?.[config.provider] ?? null;
  const enabledProviderSkills = (providerInventory?.skills ?? []).filter((item) => item.enabled);
  const enabledWorkflows = enabledCatalog.filter((item) => item.kind === "workflow");
  const enabledSkills = enabledCatalog.filter((item) => item.kind === "skill");
  const recentMessages = transcript.messages.slice(-CONVERSATION_MESSAGE_LIMIT);
  return {
    project_id: project.id,
    chat_id: chatId,
    kind: surface,
    node_id: transcript.node_id,
    node_title: node ? compactText(node.title, 160) : null,
    title: compactText(transcript.title, 160),
    updated_at: transcript.updated_at,
    message_count: transcript.message_count,
    transcript_truncated: transcript.messages.length > recentMessages.length,
    recent_messages: recentMessages.map(compactConversationMessage),
    latest_task: latestTask
      ? {
          task_id: latestTask.operation_id,
          status: latestTask.status_label,
          status_message: compactText(latestTask.status_message, 180),
          active: latestTask.active,
          awaiting_human: latestTask.awaiting_human,
          paused: latestTask.paused,
          failed: latestTask.failed,
          settled: latestTask.settled,
          runtime_used: latestTask.runtime_id,
          final_answer: latestAssistantAnswer(transcript, latestTask),
          graph_update: compactGraphUpdate(latestTask),
          artifacts: taskArtifacts(latestTask)
            .slice(0, CONVERSATION_ARTIFACT_LIMIT)
            .map((artifact) => ({
              viewer_id: `task:${latestTask.operation_id}:${artifact.artifact_id}`,
              name: compactText(artifact.name, 96),
              media_type: artifact.media_type,
              available: artifact.available,
              can_open: artifact.can_open,
              kept_filename: artifact.kept_filename ?? null,
            })),
        }
      : null,
    send_options: {
      can_send: refusal === null,
      refusal_reason: refusal,
      modes: ["discuss", "work"],
      run_truth_scope: runTruthScope.slice(0, 8),
      run_truth_scope_truncated: runTruthScope.length > 8,
      stable_session_id: sessionId,
      provider: config.provider,
      provider_label: readiness?.label ?? config.provider,
      configured_runtime: profile.runtime,
      last_runtime_used: latestTask?.runtime_id ?? null,
      model: config.model || "provider-default",
      reasoning: config.reasoning,
      run_on: config.run_on,
      provider_ready: providerReady,
      workflows_total: enabledWorkflows.length,
      workflows_truncated: enabledWorkflows.length > RCP_SKILL_LIST_LIMIT,
      workflows: enabledWorkflows.slice(0, RCP_SKILL_LIST_LIMIT).map((item) => ({
        id: item.id,
        label: compactText(item.label, 80),
        description: compactText(item.description, 120),
      })),
      skills_total: enabledSkills.length,
      skills_truncated: enabledSkills.length > RCP_SKILL_LIST_LIMIT,
      skills: enabledSkills.slice(0, RCP_SKILL_LIST_LIMIT).map((item) => ({
        id: item.id,
        label: compactText(item.label, 80),
        description: compactText(item.description, 120),
      })),
      provider_skills: {
        status: providerInventory?.status ?? "unavailable",
        stale: providerInventory?.stale ?? false,
        diagnostic: providerInventory?.diagnostic
          ? compactText(providerInventory.diagnostic, 180)
          : null,
        total: enabledProviderSkills.length,
        truncated: enabledProviderSkills.length > PROVIDER_SKILL_LIST_LIMIT,
        items: enabledProviderSkills.slice(0, PROVIDER_SKILL_LIST_LIMIT).map((item) => ({
          name: compactText(item.name, 80),
          label: compactText(item.label, 80),
        })),
      },
    },
  };
}

export function projectConversationToolDefinitions(
  project: ProjectSnapshot,
  tasks: AgentTask[],
  loadTranscript: (chatId: string) => Promise<ChatTranscript>,
  taskStartPending = false,
): WebMcpToolDefinition[] {
  return [
    {
      name: "rcp_inspect_conversation",
      description: "Read one bounded RCP conversation and its current Send options.",
      inputSchema: {
        type: "object",
        properties: {
          chat_id: {
            type: "string",
            minLength: 1,
            description: "Exact current RCP conversation id.",
          },
        },
        required: ["chat_id"],
        additionalProperties: false,
      },
      annotations: { readOnlyHint: true, untrustedContentHint: true },
      execute: async (input) =>
        webMcpTextResult(
          await inspectProjectConversation(project, tasks, input, loadTranscript, taskStartPending),
          WEBMCP_CONVERSATION_RESULT_MAX_CHARS,
        ),
    },
  ];
}

type CreateWebMcpConversation = (kind: ChatKind, node: GraphNode | null) => string;
type StartWebMcpConversationTurn = (submission: ConversationTurnSubmission) => Promise<AgentTask>;

function exactEnabledSkillIds(
  project: ProjectSnapshot,
  kind: "workflow" | "skill",
  requested: string[],
): string[] {
  const enabled = new Set(
    filterSkillCatalogToDefaults(
      project.skill_catalog ?? [],
      project.skill_defaults ?? { workflow_ids: [], skill_ids: [] },
    )
      .filter((item) => item.kind === kind)
      .map((item) => item.id),
  );
  const unknown = requested.filter((id) => !enabled.has(id));
  if (unknown.length) {
    throw new Error(`Unknown or disabled RCP ${kind} ids: ${unknown.join(", ")}.`);
  }
  return requested;
}

function exactEnabledProviderSkillNames(
  project: ProjectSnapshot,
  config: AgentRunConfig,
  requested: string[],
): string[] {
  const inventory = project.provider_skill_inventories?.[config.run_on]?.[config.provider];
  const enabled = new Set(
    (inventory?.skills ?? []).filter((item) => item.enabled).map((item) => item.name),
  );
  const unknown = requested.filter((name) => !enabled.has(name));
  if (unknown.length) {
    throw new Error(`Unknown or disabled provider skill names: ${unknown.join(", ")}.`);
  }
  return requested;
}

export async function sendProjectConversationMessage(
  project: ProjectSnapshot,
  tasks: AgentTask[],
  input: Record<string, unknown>,
  loadTranscript: (chatId: string) => Promise<ChatTranscript>,
  taskStartPending: boolean,
  createConversation: CreateWebMcpConversation,
  startTurn: StartWebMcpConversationTurn,
): Promise<Record<string, unknown>> {
  const message = requiredStringInput(input, "message").trim();
  if (message.length > 2_000) throw new Error("message must contain at most 2000 characters.");
  const mode = requiredStringInput(input, "mode");
  if (mode !== "discuss" && mode !== "work") {
    throw new Error("mode must be discuss or work.");
  }
  const requestedChatId = optionalStringInput(input, "chat_id");
  const requestedNodeId = optionalStringInput(input, "node_id");
  if (requestedChatId && requestedNodeId) {
    throw new Error("chat_id and node_id cannot be supplied together.");
  }
  const workflowIds = exactEnabledSkillIds(
    project,
    "workflow",
    stringListInput(input, "workflow_ids"),
  );
  const skillIds = exactEnabledSkillIds(project, "skill", stringListInput(input, "skill_ids"));
  const existing = requestedChatId
    ? await resolveProjectConversationContext(
        project,
        tasks,
        requestedChatId,
        loadTranscript,
        taskStartPending,
      )
    : null;
  const node = existing
    ? existing.node
    : requestedNodeId
      ? project.graph.nodes[requestedNodeId]
      : null;
  if (!existing && requestedNodeId && !node) {
    throw new Error(`Node ${requestedNodeId} is not present in the current project graph.`);
  }
  const surface: ChatKind = existing?.surface ?? (node ? "node_chat" : "project_chat");
  const profile = existing?.profile ?? project.agent_profiles[surface];
  const config: AgentRunConfig = existing?.config ?? {
    provider: profile.provider,
    model: profile.model,
    reasoning: profile.reasoning,
    run_on: profile.run_on,
  };
  const runTruthScope = existing?.runTruthScope ?? project.default_run_truth_scope ?? [];
  const readiness = project.provider_readiness[config.run_on]?.[config.provider];
  const providerReady =
    readiness === undefined || Boolean(readiness.installed && readiness.authenticated);
  const refusal =
    existing?.refusal ??
    conversationRefusal(null, [], taskStartPending, runTruthScope, providerReady);
  if (refusal) throw new Error(refusal);
  const providerSkillNames = exactEnabledProviderSkillNames(
    project,
    config,
    stringListInput(input, "provider_skill_names"),
  );
  const chatId = existing?.transcript.chat_id ?? createConversation(surface, node);
  const task = await startTurn({
    kind: surface,
    config,
    runTruthScope,
    nodeId: node?.id ?? null,
    message,
    chatId,
    sessionId: existing?.sessionId ?? null,
    mode,
    skills: { workflow_ids: workflowIds, skill_ids: skillIds },
    providerSkillNames,
  });
  return {
    project_id: project.id,
    chat_id: chatId,
    task_id: task.operation_id,
    kind: surface,
    node_id: node?.id ?? null,
    mode,
    accepted: true,
    status: task.status_label,
    active: task.active,
    queued: task.queued,
  };
}

export function projectConversationSendToolDefinitions(
  project: ProjectSnapshot,
  tasks: AgentTask[],
  loadTranscript: (chatId: string) => Promise<ChatTranscript>,
  taskStartPending: boolean,
  createConversation: CreateWebMcpConversation,
  startTurn: StartWebMcpConversationTurn,
): WebMcpToolDefinition[] {
  return [
    {
      name: "rcp_send_conversation_message",
      description:
        "Start one asynchronous RCP Discuss or Work turn in a new or existing conversation.",
      inputSchema: {
        type: "object",
        properties: {
          message: {
            type: "string",
            minLength: 1,
            maxLength: 2_000,
            description: "Natural-language request for the provider turn.",
          },
          mode: {
            type: "string",
            enum: ["discuss", "work"],
            description:
              "Discuss cannot change project truth; Work uses RCP's bounded Work authority.",
          },
          chat_id: {
            type: "string",
            minLength: 1,
            description:
              "Existing conversation to resume; omit for a fresh project or node conversation.",
          },
          node_id: {
            type: "string",
            minLength: 1,
            description:
              "Current node for a fresh node conversation; omit when chat_id is supplied.",
          },
          workflow_ids: {
            type: "array",
            items: { type: "string", minLength: 1 },
            maxItems: 32,
            description: "Exact enabled workflow ids returned by conversation inspection.",
          },
          skill_ids: {
            type: "array",
            items: { type: "string", minLength: 1 },
            maxItems: 32,
            description: "Exact enabled RCP skill ids returned by conversation inspection.",
          },
          provider_skill_names: {
            type: "array",
            items: { type: "string", minLength: 1 },
            maxItems: 32,
            description: "Exact provider-native skill names returned by conversation inspection.",
          },
        },
        required: ["message", "mode"],
        additionalProperties: false,
      },
      annotations: { readOnlyHint: false },
      execute: async (toolInput) =>
        webMcpTextResult(
          await sendProjectConversationMessage(
            project,
            tasks,
            toolInput,
            loadTranscript,
            taskStartPending,
            createConversation,
            startTurn,
          ),
        ),
    },
  ];
}

function exactExperiment(project: ProjectSnapshot, input: Record<string, unknown>): GraphNode {
  const experimentId = requiredStringInput(input, "experiment_id");
  const node = project.graph.nodes[experimentId];
  if (!node || node.type !== "experiment") {
    throw new Error(`Experiment ${experimentId} is not present in the current project graph.`);
  }
  return node;
}

function experimentTasks(tasks: AgentTask[], experimentId: string): AgentTask[] {
  return tasks
    .filter(
      (task) =>
        task.request.control_node_id === experimentId ||
        (task.kind === "node_chat" &&
          task.request.node_id === experimentId &&
          task.request.patch_kind === "experiment_loop"),
    )
    .sort(
      (left, right) =>
        Date.parse(right.created_at) - Date.parse(left.created_at) ||
        right.operation_id.localeCompare(left.operation_id),
    );
}

function experimentWatchers(watchers: WatcherRecord[], experimentId: string): WatcherRecord[] {
  return watchers
    .filter((watcher) => watcher.continuation.control_node_id === experimentId)
    .sort(
      (left, right) =>
        Date.parse(right.created_at) - Date.parse(left.created_at) ||
        right.watcher_id.localeCompare(left.watcher_id),
    );
}

function compactExperimentNode(node: GraphNode): Record<string, unknown> {
  const attempts = node.attempts ?? [];
  const latestAttempt = attempts.at(-1) ?? null;
  return {
    ...compactNode(node),
    current_summary: node.current_summary ? compactText(node.current_summary, 280) : null,
    next_action: node.next_action ? compactText(node.next_action, 220) : null,
    completion_criteria: (node.completion_criteria ?? [])
      .slice(0, 4)
      .map((criterion) => compactText(criterion, 140)),
    attempt_count: attempts.length,
    latest_attempt: latestAttempt
      ? {
          id: latestAttempt.id,
          sequence: latestAttempt.sequence,
          purpose: compactText(latestAttempt.purpose, 140),
          status: latestAttempt.status,
          outcome: latestAttempt.outcome ? compactText(latestAttempt.outcome, 240) : null,
          failure_reason: latestAttempt.failure_reason
            ? compactText(latestAttempt.failure_reason, 180)
            : null,
          job_refs: latestAttempt.job_refs.slice(0, 4),
        }
      : null,
  };
}

function compactExperimentControl(
  control: ProjectSnapshot["experiment_control"][string],
): Record<string, unknown> {
  const operational = control.operational;
  const episode = control.episode;
  return {
    ready: control.ready,
    reasons: (control.reasons ?? []).slice(0, 4).map((reason) => compactText(reason, 180)),
    graph_reasons: (control.graph_reasons ?? [])
      .slice(0, 4)
      .map((reason) => compactText(reason, 180)),
    invocations: {
      used: control.invocations_used,
      ceiling: control.invocation_ceiling,
      remaining: control.invocations_remaining,
    },
    episode_id: control.episode_id,
    paused: control.paused,
    active: control.active,
    health: control.health,
    recommendation: control.recommendation,
    run_section: control.run_section,
    live: control.live,
    can_start: control.can_start,
    can_stop: control.can_stop,
    stop_pending: control.stop_pending,
    task_control: control.task_control,
    can_switch_provider: control.can_switch_provider,
    can_open_report: control.can_open_report,
    report_episode_id: control.report_episode_id,
    node_closed: control.node_closed,
    governing_decision_ids: (control.governing_decisions ?? [])
      .slice(0, 8)
      .map((decision) => decision.decision_id),
    decision_drift_count: (control.decision_drift ?? []).length,
    operational: operational
      ? {
          task_active: operational.task_active,
          detached_work_active: operational.detached_work_active,
          watcher_degraded: operational.watcher_degraded,
          watcher_completion_pending: operational.watcher_completion_pending,
          episode_exited: operational.episode_exited,
          episode_live: operational.episode_live,
          stop_requested: operational.stop_requested,
          stop_settled: operational.stop_settled,
          chat_id: operational.chat_id,
          current_task_id: operational.current_operation_id,
          current_queued: operational.current_queued,
          current_active: operational.current_active,
          current_awaiting_human: operational.current_awaiting_human,
          current_phase: operational.current_phase,
          current_status_message: operational.current_status_message
            ? compactText(operational.current_status_message, 180)
            : null,
          current_invocation: operational.current_invocation,
          session: {
            provider: operational.session.provider,
            model: operational.session.model,
            reasoning: operational.session.reasoning,
            run_on: operational.session.run_on,
            execution_host: operational.session.execution_host,
            run_truth_scope: operational.session.run_truth_scope?.slice(0, 8) ?? null,
            native_session_bound: operational.session.native_session_bound,
            diagnostic: operational.session.diagnostic
              ? compactText(operational.session.diagnostic, 180)
              : null,
          },
        }
      : null,
    episode: episode
      ? {
          episode_id: episode.episode_id,
          graph_target: episode.graph_target,
          recovery: episode.recovery,
          budget: episode.budget,
          ending: episode.ending,
          ending_diagnostic: episode.ending_diagnostic
            ? compactText(episode.ending_diagnostic, 180)
            : null,
          wrapup_state: episode.wrapup_state,
          wrapup_error: episode.wrapup_error ? compactText(episode.wrapup_error, 180) : null,
          updated_at: episode.updated_at,
          report: episode.report,
        }
      : null,
  };
}

export function inspectProjectExperiment(
  project: ProjectSnapshot,
  tasks: AgentTask[],
  watchers: WatcherRecord[],
  input: Record<string, unknown>,
  taskStartPending = false,
  mutationsDisabled = false,
  startRequiresSync = false,
): Record<string, unknown> {
  const node = exactExperiment(project, input);
  const control = project.experiment_control[node.id];
  if (!control) throw new Error(`Experiment ${node.id} has no current control projection.`);
  const relatedTasks = experimentTasks(tasks, node.id);
  const relatedWatchers = experimentWatchers(watchers, node.id);
  const pageStartRefusal = mutationsDisabled
    ? "Graph mutations are currently disabled."
    : startRequiresSync
      ? "Sync staged graph changes before starting an episode."
      : taskStartPending
        ? "Another task start is already being submitted."
        : null;
  return {
    project_id: project.id,
    graph_revision: project.graph.revision,
    experiment: compactExperimentNode(node),
    control: compactExperimentControl(control),
    page_start_refusal: pageStartRefusal,
    start_available: control.can_start && pageStartRefusal === null,
    tasks: relatedTasks.slice(0, 3).map((task) => ({
      task_id: task.operation_id,
      episode_id: task.episode_id ?? task.request.control_episode_id ?? null,
      invocation: task.request.control_invocation ?? null,
      status: task.status_label,
      status_message: compactText(task.status_message, 160),
      active: task.active,
      awaiting_human: task.awaiting_human,
      settled: task.settled,
      runtime_used: task.runtime_id,
    })),
    watchers: relatedWatchers.slice(0, 4).map((watcher) => ({
      watcher_id: watcher.watcher_id,
      episode_id: watcher.episode_id,
      status: watcher.status,
      kind: "check_command" in watcher ? "external" : "graph",
      next_check_at: "next_check_at" in watcher ? watcher.next_check_at : null,
      stop_reason: watcher.stop_reason ? compactText(watcher.stop_reason, 160) : null,
      last_error:
        "last_error" in watcher && watcher.last_error ? compactText(watcher.last_error, 180) : null,
    })),
    report_viewer_id: control.can_open_report ? `report:${control.report_episode_id}` : null,
  };
}

type StartWebMcpExperiment = (node: GraphNode) => Promise<AgentTask>;

export async function startProjectExperiment(
  project: ProjectSnapshot,
  input: Record<string, unknown>,
  startExperiment: StartWebMcpExperiment,
): Promise<Record<string, unknown>> {
  const node = exactExperiment(project, input);
  const task = await startExperiment(node);
  return {
    project_id: project.id,
    experiment_id: node.id,
    task_id: task.operation_id,
    episode_id: task.episode_id ?? task.request.control_episode_id ?? null,
    accepted: true,
    status: task.status_label,
    active: task.active,
    queued: task.queued,
  };
}

export function projectExperimentToolDefinitions(
  project: ProjectSnapshot,
  tasks: AgentTask[],
  watchers: WatcherRecord[],
  taskStartPending: boolean,
  startReturning: boolean,
  mutationsDisabled: boolean,
  startRequiresSync: boolean,
  startExperiment: StartWebMcpExperiment,
): WebMcpToolDefinition[] {
  const inputSchema: WebMcpJsonSchema = {
    type: "object",
    properties: {
      experiment_id: {
        type: "string",
        minLength: 1,
        description: "Exact current Experiment node id returned by an RCP read tool.",
      },
    },
    required: ["experiment_id"],
    additionalProperties: false,
  };
  const inspectTool: WebMcpToolDefinition = {
    name: "rcp_inspect_experiment",
    description: "Read one RCP Experiment's current control, work, watchers, and report state.",
    inputSchema,
    annotations: { readOnlyHint: true, untrustedContentHint: true },
    execute: (toolInput) =>
      webMcpTextResult(
        inspectProjectExperiment(
          project,
          tasks,
          watchers,
          toolInput,
          taskStartPending,
          mutationsDisabled,
          startRequiresSync,
        ),
        WEBMCP_EXPERIMENT_RESULT_MAX_CHARS,
      ),
  };
  const startIsDiscoverable =
    startReturning ||
    (!taskStartPending &&
      !mutationsDisabled &&
      !startRequiresSync &&
      Object.values(project.experiment_control).some((control) => control.can_start));
  if (!startIsDiscoverable) return [inspectTool];
  return [
    inspectTool,
    {
      name: "rcp_start_experiment",
      description: "Start the next bounded episode for one exact RCP Experiment.",
      inputSchema,
      annotations: { readOnlyHint: false },
      execute: async (toolInput) =>
        webMcpTextResult(await startProjectExperiment(project, toolInput, startExperiment)),
    },
  ];
}

type StopWebMcpExperiment = (experimentId: string, episodeId: string) => Promise<void>;

export async function stopProjectExperimentEpisode(
  project: ProjectSnapshot,
  input: Record<string, unknown>,
  stopExperiment: StopWebMcpExperiment,
): Promise<Record<string, unknown>> {
  const node = exactExperiment(project, input);
  const episodeId = requiredStringInput(input, "episode_id");
  const control = project.experiment_control[node.id];
  if (!control) throw new Error(`Experiment ${node.id} has no current control projection.`);
  if (control.episode_id !== episodeId) {
    throw new Error(`Episode ${episodeId} is not the live episode for Experiment ${node.id}.`);
  }
  if (!control.can_stop) {
    throw new Error(control.reasons.join(" ") || `Experiment ${node.id} cannot be stopped now.`);
  }
  await stopExperiment(node.id, episodeId);
  return {
    project_id: project.id,
    experiment_id: node.id,
    episode_id: episodeId,
    stop_requested: true,
    graceful: true,
  };
}

export function projectExperimentStopToolDefinitions(
  project: ProjectSnapshot,
  stopExperiment: StopWebMcpExperiment,
  stopPending = false,
): WebMcpToolDefinition[] {
  if (
    !stopPending &&
    !Object.values(project.experiment_control).some((control) => control.can_stop)
  ) {
    return [];
  }
  return [
    {
      name: "rcp_stop_episode",
      description: "Request RCP's graceful Stop fence for one exact live Experiment episode.",
      inputSchema: {
        type: "object",
        properties: {
          experiment_id: {
            type: "string",
            minLength: 1,
            description: "Exact current Experiment node id returned by rcp_inspect_experiment.",
          },
          episode_id: {
            type: "string",
            minLength: 1,
            description: "Exact live episode id returned by rcp_inspect_experiment.",
          },
        },
        required: ["experiment_id", "episode_id"],
        additionalProperties: false,
      },
      annotations: { readOnlyHint: false },
      execute: async (toolInput) =>
        webMcpTextResult(await stopProjectExperimentEpisode(project, toolInput, stopExperiment)),
    },
  ];
}
