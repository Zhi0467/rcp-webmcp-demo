import {
  AlertTriangle,
  Download,
  ExternalLink,
  File,
  History,
  Inbox,
  LoaderCircle,
  MessageCircle,
  MessageCirclePlus,
  Mic,
  MicOff,
  Play,
  Plus,
  RadioTower,
  RotateCcw,
  Send,
  X,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { removeChatAttachment, uploadChatAttachment } from "../api";
import {
  artifactUrl,
  chatTasksMissingFromHistory,
  isActiveTask,
  latestNativeSessionId,
  orderTranscriptLines,
  reconcileChatHistoryArtifacts,
  reconstructTaskTranscript,
  relatedChatTasks,
  resumablePausedChatTask,
  taskKindLabel,
} from "../agentTasks";
import {
  chatDraftStorageKey,
  chatModeStorageKey,
  isConversationModeShortcut,
  latestPersistedChatConfig,
  latestPersistedConversationMode,
  parseConversationMode,
  toggleConversationMode,
} from "../chatWorkspace";
import { MarkdownAnswer } from "../chatMarkdown";
import { replaceTextSpan } from "../chatInput";
import type { GlossaryIndex } from "../glossary";
import { skillInvocationFields } from "../skillPicker";
import {
  graphConditionLabel,
  isExternalWatcherRecord,
  visibleChatWatchers,
  watcherIsIndividuallyStoppable,
  watcherLastObservedAt,
} from "../runProjection";
import {
  downloadDesktopArtifact,
  type DictationResultEvent,
  type DictationStateEvent,
  isDesktopRuntime,
  listenDesktopEvent,
  openDesktopArtifactPreview,
  openDesktopRepositoryFilePreview,
  startDesktopDictation,
  stopDesktopDictation,
} from "../desktopRuntime";
import { repositoryFilePreviewUrl, resolveRepositoryFileHref } from "../repositoryFileLinks";
import type {
  AgentArtifactDescriptor,
  ArtifactContextRequest,
  ArtifactSelection,
  AgentTask,
  ChatMessage,
  ChatAttachmentDescriptor,
  ConversationMode,
  GraphNode,
  GraphUpdateResult,
  ProjectSnapshot,
  StartAgentTask,
  WatcherRecord,
} from "../types";
import {
  CHAT_SCROLL_BOTTOM_TOLERANCE_PX,
  CHAT_USER_MESSAGE_COLLAPSE_THRESHOLD,
} from "../uiConstants";
import { profileRunConfig } from "./AgentConfigControls";
import { SkillPicker, useSkillPicker } from "./SkillPicker";
import { RepositoryScope } from "./RepositoryScope";

interface Props {
  project: ProjectSnapshot;
  node?: GraphNode | null;
  nodes?: Readonly<Record<string, GraphNode>>;
  glossaryIndex?: GlossaryIndex;
  conversationTitle?: string;
  runScope: string[];
  tasks: AgentTask[];
  watchers?: WatcherRecord[];
  historyMessages?: ChatMessage[];
  chatId: string;
  presentation?: "floating" | "workspace";
  fixedConversation?: boolean;
  readOnly?: boolean;
  reviewPending?: boolean;
  graphChangesDisabled?: boolean;
  onStartTask: StartAgentTask;
  onInspectTask: (taskId: string) => void;
  onOpenInbox: () => void;
  onRepairGraphUpdate: (taskId: string) => Promise<void>;
  onOpenNode?: (nodeId: string) => void;
  onStopWatcher?: (watcherId: string) => void;
  onNewSession: () => void;
  onClose: () => void;
  onResumeTask: (task: AgentTask) => void;
  onRetryTask: (task: AgentTask) => void;
}

interface PendingChatTurn {
  clientId: string;
  text: string;
  timestamp: string;
  mode: ConversationMode;
  attachments: ChatAttachmentDescriptor[];
}

type AttachmentStatus = "preparing" | "ready" | "error";

interface ComposerAttachment {
  localId: string;
  file: File;
  status: AttachmentStatus;
  descriptor?: ChatAttachmentDescriptor;
  error?: string;
}

interface DictationSpan {
  sessionId: string;
  start: number;
  end: number;
}

const ARTIFACT_ID_PATTERN = /^[0-9a-f]{24}$/;
const INLINE_ARTIFACT_MAX_BYTES = 2 * 1024 * 1024;

interface ArtifactContextPayload {
  type: "rcp-artifact-context";
  version: 1;
  project_id: string;
  chat_id: string;
  operation_id: string;
  source?: "task" | "episode_report";
  episode_id?: string | null;
  artifact_id: string;
  artifact_name: string;
  media_type: string;
  selections: ArtifactSelection[];
}

function artifactContextStorageKey(projectId: string, chatId: string): string {
  return `rcp:artifact-context:${encodeURIComponent(projectId)}:${encodeURIComponent(chatId)}`;
}

export function parseArtifactContextPayload(value: unknown): ArtifactContextPayload | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const candidate = value as Record<string, unknown>;
  const allowedKeys = new Set([
    "type",
    "version",
    "project_id",
    "chat_id",
    "operation_id",
    "source",
    "episode_id",
    "artifact_id",
    "artifact_name",
    "media_type",
    "selections",
  ]);
  if (
    Object.keys(candidate).some((key) => !allowedKeys.has(key)) ||
    candidate.type !== "rcp-artifact-context" ||
    candidate.version !== 1 ||
    !isBoundedArtifactText(candidate.project_id, 1, 512) ||
    !isBoundedArtifactText(candidate.chat_id, 1, 512) ||
    !isBoundedArtifactText(candidate.operation_id, 1, 512) ||
    (candidate.source !== undefined &&
      candidate.source !== "task" &&
      candidate.source !== "episode_report") ||
    (candidate.source === "episode_report" && typeof candidate.episode_id !== "string") ||
    (candidate.source !== "episode_report" &&
      candidate.episode_id !== undefined &&
      candidate.episode_id !== null) ||
    typeof candidate.artifact_id !== "string" ||
    !ARTIFACT_ID_PATTERN.test(candidate.artifact_id) ||
    !isBoundedArtifactText(candidate.artifact_name, 1, 255) ||
    !isBoundedArtifactText(candidate.media_type, 1, 64) ||
    !Array.isArray(candidate.selections) ||
    candidate.selections.length < 1 ||
    candidate.selections.length > 12
  )
    return null;
  const selections: ArtifactSelection[] = [];
  for (const raw of candidate.selections) {
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
    const selection = raw as Record<string, unknown>;
    if (
      selection.kind === "text" &&
      isBoundedArtifactText(selection.text, 1, 4096) &&
      isBoundedArtifactText(selection.surrounding_text, 0, 6144) &&
      isBoundedArtifactText(selection.comment, 0, 2048)
    ) {
      selections.push({
        kind: "text",
        text: selection.text,
        surrounding_text: selection.surrounding_text,
        comment: selection.comment,
      });
      continue;
    }
    if (
      selection.kind === "box" &&
      isArtifactSelectionRect(selection.rect) &&
      isArtifactViewport(selection.viewport) &&
      isBoundedArtifactText(selection.labels, 0, 4096) &&
      isBoundedArtifactText(selection.comment, 0, 2048)
    ) {
      selections.push({
        kind: "box",
        rect: selection.rect,
        viewport: selection.viewport,
        labels: selection.labels,
        comment: selection.comment,
      });
      continue;
    }
    return null;
  }
  return { ...(candidate as unknown as ArtifactContextPayload), selections };
}

function isBoundedArtifactText(value: unknown, minimum: number, maximum: number): value is string {
  return typeof value === "string" && value.length >= minimum && value.length <= maximum;
}

function isArtifactSelectionRect(
  value: unknown,
): value is { x: number; y: number; width: number; height: number } {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const rect = value as Record<string, unknown>;
  if (
    Object.keys(rect).some((key) => !["x", "y", "width", "height"].includes(key)) ||
    ![rect.x, rect.y, rect.width, rect.height].every(
      (part) => typeof part === "number" && Number.isFinite(part),
    )
  )
    return false;
  const x = rect.x as number;
  const y = rect.y as number;
  const width = rect.width as number;
  const height = rect.height as number;
  return x >= 0 && y >= 0 && width > 0 && height > 0 && x + width <= 1 && y + height <= 1;
}

function isArtifactViewport(value: unknown): value is { width: number; height: number } {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const viewport = value as Record<string, unknown>;
  return (
    Object.keys(viewport).every((key) => key === "width" || key === "height") &&
    [viewport.width, viewport.height].every(
      (part) => typeof part === "number" && Number.isInteger(part) && part >= 1 && part <= 32768,
    )
  );
}

export function artifactContextDraft(payload: ArtifactContextPayload): string {
  return payload.selections
    .map((selection, index) => {
      const selected =
        selection.kind === "text"
          ? `Selected text: ${selection.text}`
          : `Boxed region: ${selection.labels || `${Math.round(selection.rect.x * 100)}%, ${Math.round(selection.rect.y * 100)}%`}`;
      const comment = selection.comment.trim() ? `\n${selection.comment.trim()}` : "";
      return `${selected}${comment}\n:rcp-artifact-selection{index="${index + 1}"}`;
    })
    .join("\n\n");
}

export function NodeChat({
  project,
  node,
  nodes = {},
  glossaryIndex,
  conversationTitle,
  runScope,
  tasks,
  watchers = [],
  historyMessages = [],
  chatId,
  presentation = "floating",
  fixedConversation = false,
  readOnly = false,
  reviewPending = false,
  graphChangesDisabled = false,
  onStartTask,
  onInspectTask,
  onOpenInbox,
  onRepairGraphUpdate,
  onOpenNode,
  onStopWatcher,
  onNewSession,
  onClose,
  onResumeTask,
  onRetryTask,
}: Props) {
  const surface = node ? "node_chat" : "project_chat";
  const skillCatalog = project.skill_catalog ?? [];
  const skillDefaults = project.skill_defaults ?? { workflow_ids: [], skill_ids: [] };
  const relatedTasks = useMemo(
    () => relatedChatTasks(tasks, surface, node?.id, chatId),
    [chatId, node?.id, surface, tasks],
  );
  const [pendingTurn, setPendingTurn] = useState<PendingChatTurn | null>(null);
  const transcript = useMemo(
    () =>
      orderTranscriptLines([
        ...reconcileChatHistoryArtifacts(historyMessages, relatedTasks),
        ...reconstructTaskTranscript(chatTasksMissingFromHistory(relatedTasks, historyMessages)),
        ...(pendingTurn
          ? [
              {
                role: "human" as const,
                text: pendingTurn.text,
                taskId: pendingTurn.clientId,
                timestamp: pendingTurn.timestamp,
                mode: pendingTurn.mode,
                attachments: pendingTurn.attachments,
                trigger: "human" as const,
              },
            ]
          : []),
      ]),
    [historyMessages, pendingTurn, relatedTasks],
  );
  const config = useMemo(
    () =>
      latestPersistedChatConfig(
        historyMessages,
        relatedTasks,
        profileRunConfig(project.agent_profiles[surface]),
      ),
    [historyMessages, project.agent_profiles, relatedTasks, surface],
  );
  const [scope, setScope] = useState(runScope);
  const draftKey = chatDraftStorageKey(project.id, chatId);
  const modeKey = chatModeStorageKey(project.id, chatId);
  const artifactContextKey = artifactContextStorageKey(project.id, chatId);
  const derivedMode = useMemo(
    () => latestPersistedConversationMode(historyMessages, relatedTasks),
    [historyMessages, relatedTasks],
  );
  const [message, setMessage] = useState(() => readStorage(draftKey) ?? "");
  const [artifactContext, setArtifactContext] = useState<ArtifactContextRequest | null>(null);
  const lastArtifactContextRef = useRef<string | null>(null);
  const [modeState, setModeState] = useState<{ value: ConversationMode; pinned: boolean }>(() => {
    const storedMode = parseConversationMode(readStorage(modeKey));
    return { value: storedMode ?? derivedMode, pinned: Boolean(storedMode) };
  });
  const modeRef = useRef(modeState.value);
  const [submitting, setSubmitting] = useState(false);
  const [attachments, setAttachments] = useState<ComposerAttachment[]>([]);
  const [attachmentSetId, setAttachmentSetId] = useState<string | null>(null);
  const [draggingFiles, setDraggingFiles] = useState(false);
  const [dictationState, setDictationState] = useState<
    "idle" | "starting" | "recording" | "stopping" | "error"
  >("idle");
  const [dictationError, setDictationError] = useState<string | null>(null);
  const [expiryClock, setExpiryClock] = useState(() => Date.now());
  const [expandedHumanMessageIds, setExpandedHumanMessageIds] = useState<Set<string>>(
    () => new Set(),
  );
  const chatLinesRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const attachmentInputRef = useRef<HTMLInputElement | null>(null);
  const attachmentSetIdRef = useRef<string | null>(null);
  const attachmentUploadBusyRef = useRef(false);
  const cancelledAttachmentIdsRef = useRef<Set<string>>(new Set());
  const dictationSpanRef = useRef<DictationSpan | null>(null);
  const dictationTimerRef = useRef<number | null>(null);
  const shouldStickToBottomRef = useRef(true);
  const lastChatIdRef = useRef(chatId);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [repairingTaskId, setRepairingTaskId] = useState<string | null>(null);
  const [repairErrors, setRepairErrors] = useState<Map<string, string>>(() => new Map());
  const [unavailableArtifacts, setUnavailableArtifacts] = useState<Set<string>>(() => new Set());
  const [artifactShellErrors, setArtifactShellErrors] = useState<Map<string, string>>(
    () => new Map(),
  );
  const [repositoryFileErrors, setRepositoryFileErrors] = useState<Map<string, string>>(
    () => new Map(),
  );
  const [watchersOpen, setWatchersOpen] = useState(false);
  const readiness = project.provider_readiness[config.run_on]?.[config.provider];
  const skills = useSkillPicker({
    catalog: skillCatalog,
    defaults: skillDefaults,
    provider: config.provider,
    providerLabel: readiness?.label || config.provider,
    machine: config.run_on,
    inventory: project.provider_skill_inventories?.[config.run_on]?.[config.provider],
    message,
    onComplete: (next) => {
      setMessage(next);
      setSubmitError(null);
      window.requestAnimationFrame(() => {
        textareaRef.current?.focus();
        textareaRef.current?.setSelectionRange(next.length, next.length);
      });
    },
  });
  const desktop = useMemo(() => isDesktopRuntime(), []);
  const relatedActive = relatedTasks.some(isActiveTask);

  useEffect(() => {
    const accept = (raw: unknown) => {
      const payload = parseArtifactContextPayload(raw);
      if (!payload || payload.project_id !== project.id || payload.chat_id !== chatId) return;
      const source = relatedTasks.find((task) => task.operation_id === payload.operation_id);
      const sourceKind = payload.source ?? "task";
      const sourceArtifact = source?.result?.artifacts?.find(
        (artifact) => artifact.artifact_id === payload.artifact_id,
      );
      if (!source || (sourceKind === "task" && (!sourceArtifact || !sourceArtifact.can_revise)))
        return;
      const signature = JSON.stringify(payload);
      if (lastArtifactContextRef.current === signature) return;
      lastArtifactContextRef.current = signature;
      setArtifactContext({
        source: sourceKind,
        operation_id: payload.operation_id,
        artifact_id: payload.artifact_id,
        ...(sourceKind === "episode_report" && payload.episode_id
          ? { episode_id: payload.episode_id }
          : {}),
        selections: payload.selections,
      });
      const addition = artifactContextDraft(payload);
      setMessage((current) => {
        const next = current.trimEnd() ? `${current.trimEnd()}\n\n${addition}` : addition;
        writeStorage(draftKey, next);
        return next;
      });
      removeStorage(artifactContextKey);
      window.requestAnimationFrame(() => textareaRef.current?.focus());
    };
    const stored = readStorage(artifactContextKey);
    if (stored) {
      try {
        accept(JSON.parse(stored));
      } catch {
        removeStorage(artifactContextKey);
      }
    }
    const storage = (event: StorageEvent) => {
      if (event.key !== artifactContextKey || !event.newValue) return;
      try {
        accept(JSON.parse(event.newValue));
      } catch {
        removeStorage(artifactContextKey);
      }
    };
    window.addEventListener("storage", storage);
    let channel: BroadcastChannel | null = null;
    try {
      channel = new BroadcastChannel("rcp-artifact-context");
      channel.addEventListener("message", (event) => accept(event.data));
    } catch {
      // Storage events remain the cross-window path where BroadcastChannel is unavailable.
    }
    return () => {
      window.removeEventListener("storage", storage);
      channel?.close();
    };
  }, [artifactContextKey, chatId, draftKey, project.id, relatedTasks]);
  const liveWatchers = useMemo(
    () => visibleChatWatchers(watchers, chatId, node),
    [chatId, node, watchers],
  );
  const continuedTaskIds = useMemo(
    () =>
      new Set(
        relatedTasks.flatMap((task) =>
          task.parent_operation_id ? [task.parent_operation_id] : [],
        ),
      ),
    [relatedTasks],
  );
  const pausedAttempt = resumablePausedChatTask(relatedTasks);
  const providerReady =
    readiness === undefined || Boolean(readiness.installed && readiness.authenticated);
  const sessionId =
    latestNativeSessionId(relatedTasks) ??
    [...historyMessages].reverse().find((message) => message.native_session_id)
      ?.native_session_id ??
    null;
  const mode = modeState.value;
  modeRef.current = mode;
  const chatTitle = node?.title || conversationTitle || project.name;
  const apiBase = `/api/projects/${encodeURIComponent(project.id)}`;
  const attachmentClientId = useMemo(() => chatAttachmentClientId(), []);
  const readyAttachments = attachments.flatMap((item) =>
    item.status === "ready" && item.descriptor ? [item.descriptor] : [],
  );
  const attachmentsPreparing = attachments.some((item) => item.status === "preparing");
  const attachmentsUnready = attachments.some((item) => item.status !== "ready");
  const dictating = dictationState !== "idle" && dictationState !== "error";
  useEffect(() => {
    setModeState((current) =>
      current.pinned || current.value === derivedMode
        ? current
        : { ...current, value: derivedMode },
    );
  }, [derivedMode]);

  useEffect(() => {
    skills.reset();
    // Settings supplies fresh conversation defaults; an open turn keeps its own.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chatId, project.id]);

  useEffect(() => {
    if (message) writeStorage(draftKey, message);
    else removeStorage(draftKey);
  }, [draftKey, message]);

  useEffect(() => {
    if (lastChatIdRef.current !== chatId) {
      lastChatIdRef.current = chatId;
      shouldStickToBottomRef.current = true;
      setRepositoryFileErrors(new Map());
    }
    const element = chatLinesRef.current;
    if (!element || !shouldStickToBottomRef.current) return;
    element.scrollTop = element.scrollHeight;
  }, [chatId, transcript]);

  useEffect(() => {
    attachmentSetIdRef.current = attachmentSetId;
  }, [attachmentSetId]);

  useEffect(() => {
    if (!transcript.some((line) => line.attachments?.length)) return;
    const timer = window.setInterval(() => setExpiryClock(Date.now()), 60_000);
    return () => window.clearInterval(timer);
  }, [transcript]);

  useEffect(() => {
    if (!desktop) return;
    let disposed = false;
    const unlisten: Array<() => void> = [];
    void listenDesktopEvent<DictationResultEvent>("rcp://dictation-result", (payload) => {
      const span = dictationSpanRef.current;
      if (!span || span.sessionId !== payload.session_id) return;
      setMessage((current) => {
        const next = replaceTextSpan(current, span, payload.text);
        span.end = next.end;
        skills.readMessage(next.value);
        return next.value;
      });
      window.requestAnimationFrame(() => {
        const active = dictationSpanRef.current;
        if (!active || active.sessionId !== payload.session_id) return;
        textareaRef.current?.setSelectionRange(active.end, active.end);
      });
    }).then((dispose) => (disposed ? dispose() : unlisten.push(dispose)));
    void listenDesktopEvent<DictationStateEvent>("rcp://dictation-state", (payload) => {
      if (dictationSpanRef.current?.sessionId !== payload.session_id) return;
      if (payload.state === "recording") setDictationState("recording");
      if (payload.state === "error") {
        clearDictationTimer(dictationTimerRef);
        dictationSpanRef.current = null;
        setDictationState("error");
        setDictationError(payload.error || "Dictation stopped unexpectedly.");
      }
      if (payload.state === "stopped") {
        clearDictationTimer(dictationTimerRef);
        dictationSpanRef.current = null;
        setDictationState("idle");
      }
    }).then((dispose) => (disposed ? dispose() : unlisten.push(dispose)));
    return () => {
      disposed = true;
      unlisten.forEach((dispose) => dispose());
      clearDictationTimer(dictationTimerRef);
      const sessionId = dictationSpanRef.current?.sessionId;
      dictationSpanRef.current = null;
      if (sessionId) void stopDesktopDictation(sessionId);
    };
    // The event bridge belongs to the native shell lifetime, not each draft render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [desktop]);

  const selectMode = useCallback(
    (next: ConversationMode) => {
      modeRef.current = next;
      writeStorage(modeKey, next);
      setModeState({ value: next, pinned: true });
    },
    [modeKey],
  );

  const toggleMode = useCallback(() => {
    selectMode(toggleConversationMode(modeRef.current));
  }, [selectMode]);

  useEffect(() => {
    if (presentation !== "workspace" || readOnly) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.repeat) return;
      if (!isConversationModeShortcut(event.key, event.shiftKey)) return;
      event.preventDefault();
      toggleMode();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [presentation, readOnly, toggleMode]);

  const updateMessage = (next: string) => {
    if (readOnly) return;
    if (next && !modeState.pinned) {
      writeStorage(modeKey, mode);
      setModeState({ value: mode, pinned: true });
    }
    setMessage(next);
    skills.readMessage(next);
    setSubmitError(null);
  };

  const stopDictation = (invalidate = false) => {
    const sessionId = dictationSpanRef.current?.sessionId;
    if (!sessionId) return;
    clearDictationTimer(dictationTimerRef);
    setDictationState("stopping");
    if (invalidate) {
      dictationSpanRef.current = null;
      setDictationState("idle");
    }
    void stopDesktopDictation(sessionId).catch((error) => {
      dictationSpanRef.current = null;
      setDictationState("error");
      setDictationError(error instanceof Error ? error.message : String(error));
    });
  };

  const toggleDictation = async () => {
    if (readOnly) return;
    if (dictating) {
      stopDictation();
      return;
    }
    const textarea = textareaRef.current;
    const start = textarea?.selectionStart ?? message.length;
    const sessionId = crypto.randomUUID();
    dictationSpanRef.current = { sessionId, start, end: start };
    writeStorage(modeKey, mode);
    setModeState({ value: mode, pinned: true });
    setSubmitError(null);
    setDictationError(null);
    setDictationState("starting");
    try {
      await startDesktopDictation(sessionId);
      if (dictationSpanRef.current?.sessionId !== sessionId) return;
      setDictationState("recording");
      dictationTimerRef.current = window.setTimeout(() => stopDictation(), 55_000);
    } catch (error) {
      dictationSpanRef.current = null;
      setDictationState("error");
      setDictationError(error instanceof Error ? error.message : String(error));
    }
  };

  const addFiles = async (incoming: File[]) => {
    if (readOnly) return;
    setDraggingFiles(false);
    if (attachmentUploadBusyRef.current) {
      setSubmitError("Wait for the current files to finish preparing before adding more.");
      return;
    }
    attachmentUploadBusyRef.current = true;
    setSubmitError(null);
    const available = Math.max(0, MAX_CHAT_ATTACHMENTS - attachments.length);
    if (incoming.length > available) {
      setSubmitError(`A turn can include at most ${MAX_CHAT_ATTACHMENTS} files.`);
    }
    const candidates = incoming.slice(0, available).map<ComposerAttachment>((file) => ({
      localId: crypto.randomUUID(),
      file,
      status: "preparing",
    }));
    let total = attachments.reduce(
      (sum, item) => (item.status === "error" ? sum : sum + item.file.size),
      0,
    );
    for (const item of candidates) {
      const validation = validateChatAttachment(item.file, total);
      if (validation) {
        item.status = "error";
        item.error = validation;
      } else {
        total += item.file.size;
      }
    }
    setAttachments((current) => [...current, ...candidates]);

    const uploadCandidates = candidates.filter((candidate) => candidate.status === "preparing");
    for (const [index, item] of uploadCandidates.entries()) {
      try {
        const result = await uploadChatAttachment(
          apiBase,
          chatId,
          item.file,
          attachmentClientId,
          attachmentSetIdRef.current,
        );
        attachmentSetIdRef.current = result.attachment_set_id;
        setAttachmentSetId(result.attachment_set_id);
        if (cancelledAttachmentIdsRef.current.delete(item.localId)) {
          await removeChatAttachment(
            apiBase,
            chatId,
            result.attachment_set_id,
            result.attachment.attachment_id,
            attachmentClientId,
          );
          continue;
        }
        setAttachments((current) =>
          current.map((candidate) =>
            candidate.localId === item.localId
              ? { ...candidate, status: "ready", descriptor: result.attachment }
              : candidate,
          ),
        );
      } catch (error) {
        const cancelled = cancelledAttachmentIdsRef.current.delete(item.localId);
        const detail = error instanceof Error ? error.message : String(error);
        if (!cancelled) {
          setAttachments((current) =>
            current.map((candidate) =>
              candidate.localId === item.localId
                ? {
                    ...candidate,
                    status: "error",
                    error: detail,
                  }
                : candidate,
            ),
          );
        }
        if (!attachmentSetIdRef.current) {
          const remaining = new Set(
            uploadCandidates.slice(index + 1).map((candidate) => candidate.localId),
          );
          setAttachments((current) =>
            current.map((candidate) =>
              remaining.has(candidate.localId)
                ? { ...candidate, status: "error", error: "Attachment set could not be created" }
                : candidate,
            ),
          );
          break;
        }
      }
    }
    attachmentUploadBusyRef.current = false;
  };

  const removeAttachment = (item: ComposerAttachment) => {
    if (item.status === "preparing") cancelledAttachmentIdsRef.current.add(item.localId);
    setAttachments((current) => current.filter((candidate) => candidate.localId !== item.localId));
    const setId = attachmentSetIdRef.current;
    if (setId && item.descriptor) {
      void removeChatAttachment(
        apiBase,
        chatId,
        setId,
        item.descriptor.attachment_id,
        attachmentClientId,
      ).catch((error) => setSubmitError(error instanceof Error ? error.message : String(error)));
    }
  };

  const toggleHumanMessage = (messageId: string) => {
    setExpandedHumanMessageIds((current) => {
      const next = new Set(current);
      if (next.has(messageId)) next.delete(messageId);
      else next.add(messageId);
      return next;
    });
  };

  const handleChatScroll = (event: React.UIEvent<HTMLDivElement>) => {
    const element = event.currentTarget;
    shouldStickToBottomRef.current =
      element.scrollHeight - element.scrollTop - element.clientHeight <=
      CHAT_SCROLL_BOTTOM_TOLERANCE_PX;
  };

  const send = async () => {
    if (readOnly) return;
    const text = message.trim();
    if (
      !text ||
      attachmentsUnready ||
      relatedActive ||
      pausedAttempt ||
      submitting ||
      repairingTaskId ||
      reviewPending
    )
      return;
    if (dictating) stopDictation(true);
    shouldStickToBottomRef.current = true;
    const clientId = `pending-${crypto.randomUUID()}`;
    setPendingTurn({
      clientId,
      text,
      timestamp: new Date().toISOString(),
      mode,
      attachments: readyAttachments,
    });
    setMessage("");
    setSubmitError(null);
    setSubmitting(true);
    try {
      await onStartTask(surface, {
        ...config,
        model: config.model || null,
        run_truth_scope: scope,
        node_id: node?.id ?? null,
        message: text,
        chat_id: chatId,
        session_id: sessionId,
        mode,
        ...(artifactContext ? { artifact_context: artifactContext } : {}),
        ...(readyAttachments.length && attachmentSetId
          ? {
              attachment_set_id: attachmentSetId,
              attachment_client_id: attachmentClientId,
            }
          : {}),
        ...skillInvocationFields(skills.selection, skills.providerSkillNames),
      });
      setPendingTurn((current) => (current?.clientId === clientId ? null : current));
      skills.reset();
      setAttachments([]);
      setArtifactContext(null);
      lastArtifactContextRef.current = null;
      setAttachmentSetId(null);
      attachmentSetIdRef.current = null;
      selectMode(mode);
    } catch (error) {
      setPendingTurn((current) => (current?.clientId === clientId ? null : current));
      setMessage((current) => (current ? current : text));
      setSubmitError(error instanceof Error ? error.message : String(error));
    } finally {
      setSubmitting(false);
    }
  };

  const repairGraphUpdate = async (taskId: string) => {
    if (readOnly || repairingTaskId) return;
    setRepairingTaskId(taskId);
    setRepairErrors((current) => withoutMapKey(current, taskId));
    try {
      await onRepairGraphUpdate(taskId);
    } catch (error) {
      setRepairErrors((current) =>
        withMapValue(current, taskId, error instanceof Error ? error.message : String(error)),
      );
    } finally {
      setRepairingTaskId(null);
    }
  };

  const markArtifactUnavailable = (taskId: string, artifactId: string) => {
    setUnavailableArtifacts((current) => {
      const next = new Set(current);
      next.add(`${taskId}:${artifactId}`);
      return next;
    });
  };

  const openArtifact = async (taskId: string, artifact: AgentArtifactDescriptor) => {
    if (!artifact.can_open) return;
    if (desktop) {
      const key = `${taskId}:${artifact.artifact_id}`;
      setArtifactShellErrors((current) => withoutMapKey(current, key));
      try {
        await openDesktopArtifactPreview({
          projectId: project.id,
          taskId,
          artifactId: artifact.artifact_id,
        });
      } catch (error) {
        setArtifactShellErrors((current) =>
          withMapValue(
            current,
            key,
            `Open failed: ${error instanceof Error ? error.message : String(error)}`,
          ),
        );
      }
      return;
    }
    const target = window.open("about:blank", "_blank");
    if (!target) {
      markArtifactUnavailable(taskId, artifact.artifact_id);
      return;
    }
    target.opener = null;
    try {
      target.location.replace(artifactUrl(project.id, taskId, artifact.artifact_id, "viewer"));
    } catch {
      target.close();
      markArtifactUnavailable(taskId, artifact.artifact_id);
    }
  };

  const downloadArtifact = async (taskId: string, artifact: AgentArtifactDescriptor) => {
    if (!artifact.can_download) return;
    const key = `${taskId}:${artifact.artifact_id}`;
    setArtifactShellErrors((current) => withoutMapKey(current, key));
    try {
      await downloadDesktopArtifact({
        projectId: project.id,
        taskId,
        artifactId: artifact.artifact_id,
        suggestedName: artifact.name,
      });
    } catch (error) {
      setArtifactShellErrors((current) =>
        withMapValue(
          current,
          key,
          `Download failed: ${error instanceof Error ? error.message : String(error)}`,
        ),
      );
    }
  };

  const openRepositoryFile = async (messageId: string, href: string) => {
    const resolution = resolveRepositoryFileHref(href, project.repositories);
    if (resolution.kind === "error") {
      setRepositoryFileErrors((current) => withMapValue(current, messageId, resolution.message));
      return;
    }

    setRepositoryFileErrors((current) => withoutMapKey(current, messageId));
    const target = resolution.target;
    if (desktop) {
      try {
        await openDesktopRepositoryFilePreview({ projectId: project.id, ...target });
      } catch (error) {
        setRepositoryFileErrors((current) =>
          withMapValue(
            current,
            messageId,
            `Open failed: ${error instanceof Error ? error.message : String(error)}`,
          ),
        );
      }
      return;
    }

    const preview = window.open("about:blank", "_blank");
    if (!preview) {
      setRepositoryFileErrors((current) =>
        withMapValue(current, messageId, "Repository file preview could not be opened."),
      );
      return;
    }
    preview.opener = null;
    const url = repositoryFilePreviewUrl(project.id, target);
    if (!(await resourceIsAvailable(url))) {
      preview.close();
      setRepositoryFileErrors((current) =>
        withMapValue(current, messageId, "Repository file preview is unavailable."),
      );
      return;
    }
    try {
      preview.location.replace(url);
    } catch {
      preview.close();
      setRepositoryFileErrors((current) =>
        withMapValue(current, messageId, "Repository file preview could not be opened."),
      );
    }
  };

  const watcherToggle = liveWatchers.length > 0 && (
    <button
      className={`chat-watcher-count${watchersOpen ? " is-open" : ""}`}
      type="button"
      aria-expanded={watchersOpen}
      aria-label={`${liveWatchers.length} active watcher${liveWatchers.length === 1 ? "" : "s"}`}
      onClick={() => setWatchersOpen((open) => !open)}
    >
      <RadioTower size={12} /> {liveWatchers.length}
    </button>
  );

  return (
    <div
      className={`chat-dock ${presentation}`}
      data-mode={mode}
      role={presentation === "floating" ? "dialog" : "region"}
      aria-modal="false"
      aria-label={node || conversationTitle ? `Chat about ${chatTitle}` : "Project chat"}
      aria-keyshortcuts={presentation === "workspace" && !readOnly ? "Shift+Tab" : undefined}
    >
      {presentation === "floating" && (
        <header data-drag-handle="true">
          <MessageCircle size={17} />
          <strong>{chatTitle}</strong>
          {watcherToggle}
          <button
            className="icon-button"
            onClick={onClose}
            aria-label="Minimize chat; background work will continue"
          >
            <X size={17} />
          </button>
        </header>
      )}
      <div className="chat-context-controls">
        <div
          className="agent-provider-label"
          aria-busy={readiness === undefined}
          aria-label={`Chat provider: ${readiness?.label || config.provider}`}
        >
          {readiness?.label || config.provider}
          {readiness === undefined && (
            <LoaderCircle className="spin" size={12} aria-label="Checking provider" />
          )}
        </div>
        {!fixedConversation && !readOnly && (
          <button className="chat-new-session" type="button" onClick={onNewSession}>
            <MessageCirclePlus size={13} /> New session
          </button>
        )}
        {presentation === "workspace" && watcherToggle}
        {!readOnly && (
          <div className="chat-scope-control">
            <RepositoryScope
              repositories={project.repositories}
              projectScope={project.project_truth_scope}
              stateRepository={project.state_repository}
              selected={scope}
              onChange={relatedActive || reviewPending ? () => undefined : setScope}
            />
          </div>
        )}
      </div>
      {liveWatchers.length > 0 && watchersOpen && (
        <section className="chat-watchers" aria-label="Active watchers">
          {liveWatchers.map((watcher) => {
            const external = isExternalWatcherRecord(watcher);
            const observedAt = watcherLastObservedAt(watcher);
            return (
              <div className={`chat-watcher-row ${watcher.status}`} key={watcher.watcher_id}>
                <strong>
                  {external ? fileName(watcher.log_path) : graphConditionLabel(watcher.condition)}
                </strong>
                <time dateTime={observedAt ?? undefined}>
                  {observedAt
                    ? `${external ? "Checked" : "Evaluated"} ${new Date(observedAt).toLocaleString()}`
                    : external
                      ? "Not checked yet"
                      : "Not evaluated yet"}
                </time>
                {external && watcher.last_error && <span role="alert">{watcher.last_error}</span>}
                {!readOnly && onStopWatcher && watcherIsIndividuallyStoppable(watcher) && (
                  <button
                    className="button compact"
                    type="button"
                    onClick={() => onStopWatcher(watcher.watcher_id)}
                  >
                    Stop watching
                  </button>
                )}
              </div>
            );
          })}
        </section>
      )}
      <div
        className="node-chat-lines"
        aria-live="polite"
        onScroll={handleChatScroll}
        ref={chatLinesRef}
      >
        {transcript.map((line, index) => {
          const messageId = `${line.taskId}:${index}`;
          const task = relatedTasks.find((candidate) => candidate.operation_id === line.taskId);
          const activeLineTask = task && isActiveTask(task) ? task : null;
          const pausedLineTask =
            task?.paused && task.can_resume && !continuedTaskIds.has(task.operation_id)
              ? task
              : null;
          const pendingLine = pendingTurn?.clientId === line.taskId;
          const collapsible =
            line.role === "human" && line.text.length > CHAT_USER_MESSAGE_COLLAPSE_THRESHOLD;
          const expanded = expandedHumanMessageIds.has(messageId);
          return (
            <div className={`node-chat-line ${line.role}`} key={`${line.taskId}-${index}`}>
              {line.role === "human" && line.mode && (
                <span className={`chat-turn-mode ${line.mode}`}>{modeLabel(line.mode)}</span>
              )}
              {line.trigger === "watcher" && (
                <span className="chat-turn-trigger watcher">Watcher</span>
              )}
              {line.role === "agent" ? (
                line.text && (
                  <div className="chat-markdown">
                    <MarkdownAnswer
                      text={line.text}
                      nodes={nodes}
                      glossaryIndex={glossaryIndex}
                      onOpenNode={onOpenNode}
                      onOpenRepositoryFileLink={(href) => void openRepositoryFile(messageId, href)}
                    />
                    {repositoryFileErrors.get(messageId) && (
                      <strong className="chat-repository-file-error" role="alert">
                        {repositoryFileErrors.get(messageId)}
                      </strong>
                    )}
                  </div>
                )
              ) : line.role === "human" ? (
                <>
                  <div
                    className={`chat-human-message${collapsible && !expanded ? " collapsed" : ""}`}
                  >
                    <span className="node-chat-text">{line.text}</span>
                  </div>
                  {collapsible && (
                    <button
                      type="button"
                      className="chat-message-toggle"
                      aria-expanded={expanded}
                      onClick={() => toggleHumanMessage(messageId)}
                    >
                      {expanded ? "See less" : "See more"}
                    </button>
                  )}
                  {line.attachments?.map((attachment) => {
                    const expired = Date.parse(attachment.expires_at) <= expiryClock;
                    return (
                      <div
                        className={`chat-input-attachment${expired ? " expired" : ""}`}
                        key={attachment.attachment_id}
                      >
                        <File size={13} />
                        <span>
                          <strong>{attachment.name}</strong>
                          <small>
                            {attachment.media_type} · {formatBytes(attachment.size)}
                          </small>
                        </span>
                        {expired && <em>Expired</em>}
                      </div>
                    );
                  })}
                  {pausedLineTask ? (
                    <InlinePausedTask
                      task={pausedLineTask}
                      disabled={readOnly}
                      onResume={() => onResumeTask(pausedLineTask)}
                      onRetry={() => onRetryTask(pausedLineTask)}
                    />
                  ) : activeLineTask ? (
                    <InlineTaskProgress task={activeLineTask} />
                  ) : pendingLine ? (
                    <InlineTaskProgress task={null} />
                  ) : null}
                </>
              ) : pausedLineTask ? null : (
                <span className="node-chat-text">{line.text}</span>
              )}
              {line.artifacts?.map((artifact) => {
                const runtimeUnavailable = unavailableArtifacts.has(
                  `${line.taskId}:${artifact.artifact_id}`,
                );
                const unavailable = !artifact.available || runtimeUnavailable;
                const unavailableReason =
                  (!artifact.available && artifact.unavailable_reason) ||
                  (runtimeUnavailable ? "Preview unavailable" : null);
                const shellError = artifactShellErrors.get(
                  `${line.taskId}:${artifact.artifact_id}`,
                );
                return (
                  <div
                    className={`chat-artifact${unavailable ? " unavailable" : ""}`}
                    key={artifact.artifact_id}
                  >
                    {artifact.media_type !== "text/html" &&
                      artifact.size_bytes != null &&
                      artifact.size_bytes <= INLINE_ARTIFACT_MAX_BYTES &&
                      artifact.can_open &&
                      !unavailable && (
                        <button
                          className="chat-artifact-inline"
                          type="button"
                          aria-label={`Open ${artifact.name}`}
                          onClick={() => void openArtifact(line.taskId, artifact)}
                        >
                          <img
                            src={artifactUrl(
                              project.id,
                              line.taskId,
                              artifact.artifact_id,
                              "content",
                            )}
                            alt={artifact.name}
                            onError={() =>
                              markArtifactUnavailable(line.taskId, artifact.artifact_id)
                            }
                          />
                        </button>
                      )}
                    <File size={14} />
                    <span>
                      {artifact.name}
                      {artifact.kept_filename && <em>Kept</em>}
                    </span>
                    {unavailable ? (
                      <strong>{unavailableReason ?? "Preview unavailable"}</strong>
                    ) : (
                      <div className="chat-artifact-actions">
                        {artifact.can_open && (
                          <button
                            type="button"
                            onClick={() => void openArtifact(line.taskId, artifact)}
                          >
                            <ExternalLink size={12} /> Open
                          </button>
                        )}
                        {artifact.can_download &&
                          (desktop ? (
                            <button
                              type="button"
                              onClick={() => void downloadArtifact(line.taskId, artifact)}
                            >
                              <Download size={12} /> Download
                            </button>
                          ) : (
                            <a
                              href={artifactUrl(
                                project.id,
                                line.taskId,
                                artifact.artifact_id,
                                "download",
                              )}
                              download={artifact.name}
                            >
                              <Download size={12} /> Download
                            </a>
                          ))}
                      </div>
                    )}
                    {shellError && (
                      <strong className="chat-artifact-shell-error" role="alert">
                        {shellError}
                      </strong>
                    )}
                  </div>
                );
              })}
              {line.role === "agent" && line.graphUpdate && (
                <GraphUpdateReceipt
                  update={line.graphUpdate}
                  taskId={line.taskId}
                  repairBusy={repairingTaskId === line.taskId}
                  repairDisabled={
                    readOnly || graphChangesDisabled || relatedActive || submitting || reviewPending
                  }
                  repairContinued={continuedTaskIds.has(line.taskId)}
                  repairError={repairErrors.get(line.taskId) ?? null}
                  onInspectTask={onInspectTask}
                  onOpenInbox={onOpenInbox}
                  onRepair={() => void repairGraphUpdate(line.taskId)}
                />
              )}
            </div>
          );
        })}
        {submitError && <div className="node-chat-line error">{submitError}</div>}
      </div>
      {!readOnly && (
        <div
          className={`chat-composer${draggingFiles ? " is-dragging-files" : ""}`}
          data-mode={mode}
          onDragEnter={(event) => {
            if (event.dataTransfer.types.includes("Files")) setDraggingFiles(true);
          }}
          onDragOver={(event) => {
            if (!event.dataTransfer.types.includes("Files")) return;
            event.preventDefault();
            event.dataTransfer.dropEffect = "copy";
          }}
          onDragLeave={(event) => {
            if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
              setDraggingFiles(false);
            }
          }}
          onDrop={(event) => {
            if (!event.dataTransfer.files.length) return;
            event.preventDefault();
            void addFiles(Array.from(event.dataTransfer.files));
          }}
        >
          <SkillPicker {...skills.props} />
          {artifactContext && (
            <div className="artifact-context-chip">
              <span>Artifact selections · {artifactContext.selections.length}</span>
              <button
                type="button"
                aria-label="Remove artifact selections"
                onClick={() => {
                  setArtifactContext(null);
                  lastArtifactContextRef.current = null;
                }}
              >
                <X size={12} />
              </button>
            </div>
          )}
          {attachments.length > 0 && (
            <div className="chat-attachment-chips" aria-label="Files for this turn">
              {attachments.map((item) => (
                <div className={`chat-attachment-chip ${item.status}`} key={item.localId}>
                  {item.status === "preparing" ? (
                    <LoaderCircle className="spin" size={12} />
                  ) : (
                    <File size={12} />
                  )}
                  <span>
                    <strong>{item.file.name}</strong>
                    <small>
                      {item.status === "preparing"
                        ? "Preparing"
                        : item.status === "ready"
                          ? `Ready · ${formatBytes(item.file.size)}`
                          : item.error || "Could not prepare file"}
                    </small>
                  </span>
                  <button
                    type="button"
                    aria-label={`Remove ${item.file.name}`}
                    onClick={() => removeAttachment(item)}
                  >
                    <X size={12} />
                  </button>
                </div>
              ))}
            </div>
          )}
          <input
            ref={attachmentInputRef}
            className="visually-hidden"
            type="file"
            multiple
            accept={CHAT_ATTACHMENT_ACCEPT}
            onChange={(event) => {
              const files = Array.from(event.currentTarget.files ?? []);
              event.currentTarget.value = "";
              if (files.length) void addFiles(files);
            }}
          />
          <button
            className="icon-button chat-add-file"
            type="button"
            aria-label="Add files"
            disabled={
              attachments.length >= MAX_CHAT_ATTACHMENTS || attachmentsPreparing || submitting
            }
            onClick={() => attachmentInputRef.current?.click()}
          >
            <Plus size={16} />
          </button>
          <textarea
            ref={textareaRef}
            aria-label="Message"
            aria-keyshortcuts="Shift+Tab"
            value={message}
            onChange={(event) => {
              if (dictating) stopDictation(true);
              updateMessage(event.target.value);
            }}
            onPaste={(event) => {
              const files = Array.from(event.clipboardData.files);
              if (!files.length) return;
              event.preventDefault();
              void addFiles(files);
            }}
            onKeyDown={(event) => {
              if (skills.handleKeyDown(event)) return;
              if (isConversationModeShortcut(event.key, event.shiftKey)) {
                if (presentation !== "workspace") {
                  event.preventDefault();
                  toggleMode();
                }
                return;
              }
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void send();
              }
            }}
          />
          <div className="chat-send">
            <div className="chat-mode-toggle" role="group" aria-label="Conversation mode">
              {(["discuss", "work"] as const).map((option) => (
                <button
                  type="button"
                  className={option}
                  aria-pressed={mode === option}
                  onClick={() => selectMode(option)}
                  key={option}
                >
                  {modeLabel(option)}
                </button>
              ))}
            </div>
            <div className="chat-send-actions">
              {desktop && (
                <button
                  className={`icon-button chat-dictation-button${dictating ? " recording" : ""}`}
                  type="button"
                  aria-label={dictating ? "Stop dictation" : "Start dictation"}
                  aria-pressed={dictating}
                  title={dictationError || (dictating ? "Stop dictation" : "Dictate")}
                  disabled={submitting}
                  onClick={() => void toggleDictation()}
                >
                  {dictating ? <MicOff size={15} /> : <Mic size={15} />}
                </button>
              )}
              <button
                className="icon-button primary chat-send-button"
                disabled={
                  !message.trim() ||
                  attachmentsUnready ||
                  relatedActive ||
                  Boolean(pausedAttempt) ||
                  submitting ||
                  Boolean(repairingTaskId) ||
                  reviewPending ||
                  scope.length === 0 ||
                  !providerReady
                }
                onClick={() => void send()}
                aria-label={`Start ${modeLabel(mode)} turn`}
              >
                <Send size={15} />
              </button>
            </div>
            {dictationError && (
              <span className="chat-dictation-error" role="alert">
                {dictationError}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function InlineTaskProgress({ task }: { task: AgentTask | null }) {
  const label = task
    ? task.status_message || `${taskKindLabel(task.kind)} is running`
    : "Starting task";
  return (
    <div className="chat-task-inline running" role="status" aria-label="Agent task running">
      <LoaderCircle className="spin" size={12} />
      <span>{label}</span>
      <div className="chat-task-progress" role="progressbar" aria-label="Task in progress">
        <span style={{ width: `${Math.round((task?.progress ?? 0) * 100)}%` }} />
      </div>
    </div>
  );
}

function InlinePausedTask({
  task,
  disabled,
  onResume,
  onRetry,
}: {
  task: AgentTask;
  disabled: boolean;
  onResume: () => void;
  onRetry: () => void;
}) {
  return (
    <div className="chat-task-inline paused" role="status" aria-label="Agent task paused">
      <span>{task.status_message}</span>
      <button
        type="button"
        className="button compact primary"
        disabled={disabled}
        onClick={onResume}
      >
        <Play size={11} /> Resume
      </button>
      <button
        type="button"
        className="button compact secondary"
        disabled={disabled}
        onClick={onRetry}
      >
        <RotateCcw size={11} /> Retry
      </button>
    </div>
  );
}

function GraphUpdateReceipt({
  update,
  taskId,
  repairBusy,
  repairDisabled,
  repairContinued,
  repairError,
  onInspectTask,
  onOpenInbox,
  onRepair,
}: {
  update: GraphUpdateResult;
  taskId: string;
  repairBusy: boolean;
  repairDisabled: boolean;
  repairContinued: boolean;
  repairError: string | null;
  onInspectTask: (taskId: string) => void;
  onOpenInbox: () => void;
  onRepair: () => void;
}) {
  if (update.status === "none") return null;
  const proposalCount = update.proposal_ids.length;
  return (
    <div className={`chat-graph-receipt ${update.status}`}>
      <div className="chat-graph-receipt-actions">
        {update.status === "applied" && (
          <button type="button" onClick={() => onInspectTask(taskId)}>
            <History size={12} />
            {update.applied_revision === null
              ? "Graph updated"
              : `Graph updated · r${update.applied_revision}`}
          </button>
        )}
        {update.status === "rejected" && (
          <strong>
            <AlertTriangle size={12} /> Graph update rejected
          </strong>
        )}
        {update.status === "applied" && proposalCount > 0 && (
          <button type="button" onClick={onOpenInbox}>
            <Inbox size={12} />
            {proposalCount} proposal{proposalCount === 1 ? "" : "s"} sent to Inbox
          </button>
        )}
        {update.status === "rejected" && update.repairable && !repairContinued && (
          <button type="button" disabled={repairBusy || repairDisabled} onClick={onRepair}>
            <RotateCcw className={repairBusy ? "spin" : undefined} size={12} />
            Repair graph update
          </button>
        )}
      </div>
      {update.change_summary.length > 0 && (
        <ul className="chat-graph-change-summary">
          {update.change_summary.map((item, index) => (
            <li key={`${index}:${item}`}>{item}</li>
          ))}
        </ul>
      )}
      {update.status === "rejected" && update.validation_messages.length > 0 && (
        <ul className="chat-graph-validation">
          {update.validation_messages.map((item, index) => (
            <li key={`${index}:${item}`}>{item}</li>
          ))}
        </ul>
      )}
      {repairError && (
        <strong className="chat-graph-repair-error" role="alert">
          {repairError}
        </strong>
      )}
    </div>
  );
}

function modeLabel(mode: ConversationMode): "Discuss" | "Work" {
  return mode === "discuss" ? "Discuss" : "Work";
}

const MAX_CHAT_ATTACHMENTS = 8;
const MAX_CHAT_ATTACHMENT_BYTES = 16 * 1024 * 1024;
const MAX_CHAT_ATTACHMENT_TOTAL_BYTES = 32 * 1024 * 1024;
const CHAT_ATTACHMENT_CLIENT_KEY = "rcp:chat-attachment-client";
const CHAT_ATTACHMENT_EXTENSIONS = new Set([
  "c",
  "cc",
  "cpp",
  "cs",
  "css",
  "csv",
  "fish",
  "go",
  "h",
  "hpp",
  "htm",
  "html",
  "java",
  "js",
  "json",
  "jsx",
  "kt",
  "kts",
  "lua",
  "markdown",
  "md",
  "mjs",
  "mm",
  "php",
  "py",
  "r",
  "rb",
  "rs",
  "scala",
  "sh",
  "sql",
  "svg",
  "swift",
  "toml",
  "ts",
  "tsv",
  "tsx",
  "txt",
  "xml",
  "yaml",
  "yml",
  "zsh",
]);
const CHAT_ATTACHMENT_BINARY_EXTENSIONS = new Set(["jpeg", "jpg", "pdf", "png", "webp"]);
const CHAT_ATTACHMENT_ACCEPT = [
  ".txt",
  ".md",
  ".csv",
  ".tsv",
  ".json",
  ".html",
  ".htm",
  ".svg",
  ".pdf",
  ".png",
  ".jpg",
  ".jpeg",
  ".webp",
  ...[...CHAT_ATTACHMENT_EXTENSIONS].map((extension) => `.${extension}`),
].join(",");

function validateChatAttachment(file: File, currentTotal: number): string | null {
  const extension = file.name.split(".").at(-1)?.toLowerCase() ?? "";
  if (
    !CHAT_ATTACHMENT_BINARY_EXTENSIONS.has(extension) &&
    !CHAT_ATTACHMENT_EXTENSIONS.has(extension)
  ) {
    return "Unsupported file type";
  }
  if (file.size > MAX_CHAT_ATTACHMENT_BYTES) return "File exceeds 16 MiB";
  if (currentTotal + file.size > MAX_CHAT_ATTACHMENT_TOTAL_BYTES) {
    return "Turn exceeds 32 MiB total";
  }
  return null;
}

function chatAttachmentClientId(): string {
  try {
    const current = sessionStorage.getItem(CHAT_ATTACHMENT_CLIENT_KEY);
    if (current) return current;
    const created = crypto.randomUUID();
    sessionStorage.setItem(CHAT_ATTACHMENT_CLIENT_KEY, created);
    return created;
  } catch {
    return crypto.randomUUID();
  }
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(bytes < 10 * 1024 ? 1 : 0)} KiB`;
  return `${(bytes / (1024 * 1024)).toFixed(bytes < 10 * 1024 * 1024 ? 1 : 0)} MiB`;
}

function clearDictationTimer(ref: React.MutableRefObject<number | null>): void {
  if (ref.current !== null) window.clearTimeout(ref.current);
  ref.current = null;
}

function fileName(path: string): string {
  return path.split("/").filter(Boolean).at(-1) ?? path;
}

function readStorage(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function writeStorage(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch {}
}

function removeStorage(key: string): void {
  try {
    localStorage.removeItem(key);
  } catch {}
}

function withMapValue(map: Map<string, string>, key: string, value: string): Map<string, string> {
  const next = new Map(map);
  next.set(key, value);
  return next;
}

function withoutMapKey(map: Map<string, string>, key: string): Map<string, string> {
  if (!map.has(key)) return map;
  const next = new Map(map);
  next.delete(key);
  return next;
}

async function resourceIsAvailable(url: string): Promise<boolean> {
  try {
    const response = await fetch(url, { method: "HEAD" });
    return response.ok;
  } catch {
    return false;
  }
}
