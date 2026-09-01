import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError } from "../api";
import {
  loadChatSummaryPage,
  loadChatTranscript,
  mergeChatSummaryPage,
  nextChatSummaryOffset,
  reconcileChatSelectionAfterRefresh,
} from "../chatApi";
import {
  chatIdForTask,
  latestConversation,
  newlyUnreadChatTaskIds,
  type ChatConversation,
  type ChatKind,
  type DraftConversation,
} from "../chatWorkspace";
import type { AgentTask, AppView, ChatSummary, ChatTranscript, GraphNode } from "../types";

export interface FloatingChat {
  chatId: string;
  nodeId: string;
}

export interface ChatStateSnapshot {
  floatingChat: FloatingChat | null;
  draftConversations: DraftConversation[];
  selectedChatId: string | null;
  unreadChatTaskIds: Set<string>;
  chatSummaries: ChatSummary[];
  chatSummaryTotal: number;
  chatSummaryNextOffset: number;
  chatTranscripts: Map<string, ChatTranscript>;
  selectedCanonicalChat: ChatSummary | null;
  chatTaskStatuses: Map<string, AgentTask["status"]>;
}

interface UseChatStateOptions {
  projectId: string | null;
  apiBase: string;
  selectedExperimentChatId: string | null;
  isActiveProject: (projectId: string) => boolean;
  visibleTranscriptIds: (selectedChatId: string | null, floatingChatId: string | null) => string[];
  reportError: (message: string) => void;
}

export function cloneChatStateSnapshot(snapshot: ChatStateSnapshot): ChatStateSnapshot {
  return {
    ...snapshot,
    floatingChat: snapshot.floatingChat ? { ...snapshot.floatingChat } : null,
    draftConversations: [...snapshot.draftConversations],
    unreadChatTaskIds: new Set(snapshot.unreadChatTaskIds),
    chatSummaries: [...snapshot.chatSummaries],
    chatTranscripts: new Map(snapshot.chatTranscripts),
    chatTaskStatuses: new Map(snapshot.chatTaskStatuses),
  };
}

export function visibleChatTranscriptIds(
  view: AppView,
  selectedChatId: string | null,
  floatingChatId: string | null,
  experimentChatId: string | null,
): string[] {
  return [
    ...new Set([
      ...(view === "chats" && selectedChatId ? [selectedChatId] : []),
      ...(view === "execution" && experimentChatId ? [experimentChatId] : []),
      ...(floatingChatId ? [floatingChatId] : []),
    ]),
  ];
}

export function visibleUnreadChatId(
  view: AppView,
  selectedChatId: string | null,
  experimentChatId: string | null,
): string | null {
  if (view === "chats") return selectedChatId;
  if (view === "execution") return experimentChatId;
  return null;
}

export function shouldLoadVisibleChatTranscript(
  chatId: string,
  summaries: readonly Pick<ChatSummary, "chat_id">[],
  selectedExperimentChatId: string | null,
): boolean {
  return (
    chatId === selectedExperimentChatId || summaries.some((summary) => summary.chat_id === chatId)
  );
}

export function useChatState({
  projectId,
  apiBase,
  selectedExperimentChatId,
  isActiveProject,
  visibleTranscriptIds,
  reportError,
}: UseChatStateOptions) {
  const [floatingChat, setFloatingChatState] = useState<FloatingChat | null>(null);
  const [draftConversations, setDraftConversations] = useState<DraftConversation[]>([]);
  const [selectedChatId, setSelectedChatId] = useState<string | null>(null);
  const [unreadChatTaskIds, setUnreadChatTaskIds] = useState<Set<string>>(() => new Set());
  const [chatSummaries, setChatSummaries] = useState<ChatSummary[]>([]);
  const [chatSummaryTotal, setChatSummaryTotal] = useState(0);
  const [chatSummaryNextOffset, setChatSummaryNextOffset] = useState(0);
  const [chatSummariesLoading, setChatSummariesLoading] = useState(false);
  const [chatTranscripts, setChatTranscripts] = useState<Map<string, ChatTranscript>>(
    () => new Map(),
  );
  const [selectedCanonicalChat, setSelectedCanonicalChat] = useState<ChatSummary | null>(null);
  const chatTaskStatuses = useRef<Map<string, AgentTask["status"]>>(new Map());
  const chatSummariesRef = useRef<ChatSummary[]>([]);
  const selectedChatIdRef = useRef<string | null>(null);
  const selectedCanonicalChatRef = useRef<ChatSummary | null>(null);
  const chatSummaryRefreshGeneration = useRef(0);

  const visibleChatSummaries = useMemo(
    () =>
      selectedCanonicalChat &&
      !chatSummaries.some((summary) => summary.chat_id === selectedCanonicalChat.chat_id)
        ? [...chatSummaries, selectedCanonicalChat]
        : chatSummaries,
    [chatSummaries, selectedCanonicalChat],
  );
  const visibleChatIds = useMemo(
    () => visibleTranscriptIds(selectedChatId, floatingChat?.chatId ?? null),
    [floatingChat?.chatId, selectedChatId, visibleTranscriptIds],
  );
  const visibleChatVersions = visibleChatIds
    .map(
      (chatId) =>
        `${chatId}:${visibleChatSummaries.find((summary) => summary.chat_id === chatId)?.updated_at ?? ""}`,
    )
    .join("|");

  useEffect(() => {
    if (!apiBase || visibleChatIds.length === 0) return;
    let cancelled = false;
    visibleChatIds.forEach((chatId) => {
      if (
        !shouldLoadVisibleChatTranscript(chatId, visibleChatSummaries, selectedExperimentChatId)
      ) {
        return;
      }
      void loadChatTranscript(apiBase, chatId, api)
        .then((transcript) => {
          if (cancelled) return;
          setChatTranscripts((current) => new Map(current).set(chatId, transcript));
        })
        .catch((error) => {
          if (!cancelled) {
            reportError(
              `Conversation could not be loaded: ${error instanceof Error ? error.message : String(error)}`,
            );
          }
        });
    });
    return () => {
      cancelled = true;
    };
  }, [apiBase, selectedExperimentChatId, visibleChatVersions]);

  const selectChat = useCallback((chatId: string | null) => {
    selectedChatIdRef.current = chatId;
    setSelectedChatId(chatId);
    if (selectedCanonicalChatRef.current?.chat_id !== chatId) {
      selectedCanonicalChatRef.current = null;
      setSelectedCanonicalChat(null);
    }
  }, []);

  const setFloatingChat = useCallback((next: FloatingChat | null) => {
    setFloatingChatState(next);
  }, []);

  const reconcileFloatingChat = useCallback(
    (nodes: Record<string, GraphNode>, retainMissing: boolean) => {
      setFloatingChatState((current) =>
        current && (nodes[current.nodeId] || retainMissing) ? current : null,
      );
    },
    [],
  );

  const startConversation = useCallback(
    (kind: ChatKind, node: GraphNode | null, projectTitle: string): string => {
      const chatId = window.crypto.randomUUID();
      const draft: DraftConversation = {
        chatId,
        kind,
        nodeId: node?.id ?? null,
        title: node?.title ?? projectTitle,
      };
      setDraftConversations((current) => [draft, ...current]);
      return chatId;
    },
    [],
  );

  const ensureConversation = useCallback(
    (
      conversations: ChatConversation[],
      kind: ChatKind,
      node: GraphNode | null,
      projectTitle: string,
    ): string => {
      const existing = latestConversation(conversations, kind, node?.id ?? null);
      return existing?.chatId ?? startConversation(kind, node, projectTitle);
    },
    [startConversation],
  );

  const refreshChatSummaries = useCallback(
    async (requestedProjectId: string, base: string) => {
      const generation = ++chatSummaryRefreshGeneration.current;
      setChatSummariesLoading(true);
      try {
        const page = await loadChatSummaryPage(base, 0, api);
        if (
          !isActiveProject(requestedProjectId) ||
          generation !== chatSummaryRefreshGeneration.current
        )
          return;
        const selectedId = selectedChatIdRef.current;
        const previousSummary = selectedId
          ? (chatSummariesRef.current.find((summary) => summary.chat_id === selectedId) ??
            (selectedCanonicalChatRef.current?.chat_id === selectedId
              ? selectedCanonicalChatRef.current
              : null))
          : null;
        let validation: ChatTranscript | null | undefined;
        if (
          selectedId &&
          previousSummary &&
          !page.items.some((summary) => summary.chat_id === selectedId)
        ) {
          try {
            validation = await loadChatTranscript(base, selectedId, api);
          } catch (error) {
            if (error instanceof ApiError && error.status === 404) validation = null;
            else throw error;
          }
        }
        if (
          !isActiveProject(requestedProjectId) ||
          generation !== chatSummaryRefreshGeneration.current
        )
          return;
        const nextSummaries = mergeChatSummaryPage([], page.items, "refresh");
        chatSummariesRef.current = nextSummaries;
        setChatSummaries(nextSummaries);
        setChatSummaryTotal(page.total);
        setChatSummaryNextOffset(nextChatSummaryOffset(page));
        if (selectedChatIdRef.current === selectedId) {
          const reconciliation = reconcileChatSelectionAfterRefresh(
            selectedId,
            previousSummary,
            nextSummaries,
            validation,
          );
          selectedChatIdRef.current = reconciliation.selectedChatId;
          setSelectedChatId(reconciliation.selectedChatId);
          selectedCanonicalChatRef.current = reconciliation.retainedSummary;
          setSelectedCanonicalChat(reconciliation.retainedSummary);
          if (validation) {
            setChatTranscripts((current) => new Map(current).set(validation.chat_id, validation));
          } else if (reconciliation.deleteTranscript && selectedId) {
            setDraftConversations((current) =>
              current.filter((draft) => draft.chatId !== selectedId),
            );
            setChatTranscripts((current) => {
              if (!current.has(selectedId)) return current;
              const next = new Map(current);
              next.delete(selectedId);
              return next;
            });
          }
        }
      } finally {
        if (
          isActiveProject(requestedProjectId) &&
          generation === chatSummaryRefreshGeneration.current
        ) {
          setChatSummariesLoading(false);
        }
      }
    },
    [isActiveProject],
  );

  const loadMoreChatSummaries = useCallback(async () => {
    if (!projectId || !apiBase || chatSummariesLoading || chatSummaryNextOffset >= chatSummaryTotal)
      return;
    const requestedProjectId = projectId;
    const generation = chatSummaryRefreshGeneration.current;
    const offset = chatSummaryNextOffset;
    setChatSummariesLoading(true);
    try {
      const page = await loadChatSummaryPage(apiBase, offset, api);
      if (
        !isActiveProject(requestedProjectId) ||
        generation !== chatSummaryRefreshGeneration.current
      )
        return;
      const nextSummaries = mergeChatSummaryPage(chatSummariesRef.current, page.items, "append");
      chatSummariesRef.current = nextSummaries;
      setChatSummaries(nextSummaries);
      setChatSummaryTotal(page.total);
      setChatSummaryNextOffset(nextChatSummaryOffset(page));
    } catch (error) {
      if (
        isActiveProject(requestedProjectId) &&
        generation === chatSummaryRefreshGeneration.current
      ) {
        reportError(
          `Chats could not be loaded: ${error instanceof Error ? error.message : String(error)}`,
        );
      }
    } finally {
      if (
        isActiveProject(requestedProjectId) &&
        generation === chatSummaryRefreshGeneration.current
      ) {
        setChatSummariesLoading(false);
      }
    }
  }, [apiBase, chatSummariesLoading, chatSummaryNextOffset, chatSummaryTotal, projectId]);

  const recordTaskUpdates = useCallback((tasks: AgentTask[], visibleChatId: string | null) => {
    const previousStatuses = chatTaskStatuses.current;
    const nextStatuses = new Map(previousStatuses);
    const newlyTerminal = newlyUnreadChatTaskIds(tasks, previousStatuses, visibleChatId);
    const completedChatTasks = newlyUnreadChatTaskIds(tasks, previousStatuses, null);
    for (const task of tasks) {
      if (!chatIdForTask(task)) continue;
      nextStatuses.set(task.operation_id, task.status);
    }
    chatTaskStatuses.current = nextStatuses;
    if (newlyTerminal.length) {
      setUnreadChatTaskIds((current) => new Set([...current, ...newlyTerminal]));
    }
    return completedChatTasks.length > 0;
  }, []);

  const recordWatcherResults = useCallback((tasks: AgentTask[]) => {
    const unseen = tasks.filter(
      (task) =>
        task.request.trigger === "watcher" &&
        !chatTaskStatuses.current.has(task.operation_id) &&
        task.finished,
    );
    if (unseen.length > 0) {
      setUnreadChatTaskIds(
        (current) =>
          new Set([
            ...current,
            ...unseen.flatMap((task) => {
              const chatId = chatIdForTask(task);
              return chatId && chatId !== selectedChatIdRef.current ? [task.operation_id] : [];
            }),
          ]),
      );
    }
    return unseen.length > 0;
  }, []);

  const markVisibleChatRead = useCallback((tasks: AgentTask[], visibleChatId: string | null) => {
    if (!visibleChatId) return;
    setUnreadChatTaskIds((current) => {
      const next = new Set(current);
      tasks.forEach((task) => {
        if (task.request.chat_id === visibleChatId) next.delete(task.operation_id);
      });
      return next.size === current.size ? current : next;
    });
  }, []);

  const resetProjectChats = useCallback(() => {
    setFloatingChatState(null);
    setDraftConversations([]);
    selectChat(null);
    setUnreadChatTaskIds(new Set());
    chatSummaryRefreshGeneration.current += 1;
    chatSummariesRef.current = [];
    setChatSummaries([]);
    setChatSummaryTotal(0);
    setChatSummaryNextOffset(0);
    setChatSummariesLoading(false);
    setChatTranscripts(new Map());
    chatTaskStatuses.current = new Map();
  }, [selectChat]);

  const restoreProjectChats = useCallback(
    (snapshot: ChatStateSnapshot, nodes: Record<string, GraphNode>) => {
      setFloatingChatState(
        snapshot.floatingChat && nodes[snapshot.floatingChat.nodeId]
          ? { ...snapshot.floatingChat }
          : null,
      );
      setDraftConversations([...snapshot.draftConversations]);
      selectChat(snapshot.selectedChatId);
      setUnreadChatTaskIds(new Set(snapshot.unreadChatTaskIds));
      chatSummaryRefreshGeneration.current += 1;
      chatSummariesRef.current = [...snapshot.chatSummaries];
      setChatSummaries([...snapshot.chatSummaries]);
      setChatSummaryTotal(snapshot.chatSummaryTotal);
      setChatSummaryNextOffset(snapshot.chatSummaryNextOffset);
      setChatSummariesLoading(false);
      setChatTranscripts(new Map(snapshot.chatTranscripts));
      selectedCanonicalChatRef.current = snapshot.selectedCanonicalChat;
      setSelectedCanonicalChat(snapshot.selectedCanonicalChat);
      chatTaskStatuses.current = new Map(snapshot.chatTaskStatuses);
    },
    [selectChat],
  );

  const snapshot: ChatStateSnapshot = {
    floatingChat,
    draftConversations,
    selectedChatId,
    unreadChatTaskIds,
    chatSummaries,
    chatSummaryTotal,
    chatSummaryNextOffset,
    chatTranscripts,
    selectedCanonicalChat,
    chatTaskStatuses: chatTaskStatuses.current,
  };

  return {
    snapshot,
    chatSummariesLoading,
    visibleChatSummaries,
    selectChat,
    setFloatingChat,
    reconcileFloatingChat,
    startConversation,
    ensureConversation,
    refreshChatSummaries,
    loadMoreChatSummaries,
    recordTaskUpdates,
    recordWatcherResults,
    markVisibleChatRead,
    resetProjectChats,
    restoreProjectChats,
  };
}
