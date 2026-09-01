import { ChevronLeft, ChevronRight, Circle, LoaderCircle, MessageCircle } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { isActiveTask } from "../agentTasks";
import { conversationHasUnread, type ChatConversation } from "../chatWorkspace";
import type { GlossaryIndex } from "../glossary";
import {
  CHAT_LIST_DEFAULT_WIDTH,
  CHAT_LIST_COLLAPSED_WIDTH,
  CHAT_LIST_DIVIDER_WIDTH,
  CHAT_LIST_MIN_WIDTH_COMPACT,
  chatListWidthBounds,
  clampChatListWidth,
  isChatListToggleShortcut,
  type ChatListWidthBounds,
} from "../chatLayout";
import type {
  AgentTask,
  ChatTranscript,
  GraphNode,
  ProjectSnapshot,
  StartAgentTask,
  WatcherRecord,
} from "../types";
import { NodeChat } from "../components/NodeChat";

interface Props {
  project: ProjectSnapshot;
  conversations: ChatConversation[];
  selectedChatId: string | null;
  nodes: Record<string, GraphNode>;
  glossaryIndex: GlossaryIndex;
  runScope: string[];
  tasks: AgentTask[];
  watchers: WatcherRecord[];
  graphChangesDisabled: boolean;
  unreadTaskIds: ReadonlySet<string>;
  chatTranscripts: ReadonlyMap<string, ChatTranscript>;
  hasMore: boolean;
  loadingMore: boolean;
  onSelect: (chatId: string) => void;
  onLoadMore: () => void;
  onStartTask: StartAgentTask;
  onResumeTask: (task: AgentTask) => void;
  onRetryTask: (task: AgentTask) => void;
  onInspectTask: (taskId: string) => void;
  onOpenInbox: () => void;
  onOpenNode?: (nodeId: string) => void;
  onRepairGraphUpdate: (taskId: string) => Promise<void>;
  onStopWatcher?: (watcherId: string) => void;
  onNewSession: (conversation: ChatConversation) => void;
}

function chatListWidthStorageKey(projectId: string): string {
  return `rcp:chat-list-width:${projectId}`;
}

function chatListCollapsedStorageKey(projectId: string): string {
  return `rcp:chat-list-collapsed:${projectId}`;
}

function readChatListWidth(projectId: string): number {
  if (typeof window === "undefined") return CHAT_LIST_DEFAULT_WIDTH;
  try {
    const value = Number(window.localStorage.getItem(chatListWidthStorageKey(projectId)));
    return Number.isFinite(value) && value >= CHAT_LIST_MIN_WIDTH_COMPACT
      ? value
      : CHAT_LIST_DEFAULT_WIDTH;
  } catch {
    return CHAT_LIST_DEFAULT_WIDTH;
  }
}

function readChatListCollapsed(projectId: string): boolean {
  if (typeof window === "undefined") return false;
  try {
    return window.localStorage.getItem(chatListCollapsedStorageKey(projectId)) === "true";
  } catch {
    return false;
  }
}

export function ChatsWorkspace({
  project,
  conversations,
  selectedChatId,
  nodes,
  glossaryIndex,
  runScope,
  tasks,
  watchers,
  graphChangesDisabled,
  unreadTaskIds,
  chatTranscripts,
  hasMore,
  loadingMore,
  onSelect,
  onLoadMore,
  onStartTask,
  onResumeTask,
  onRetryTask,
  onInspectTask,
  onOpenInbox,
  onOpenNode,
  onRepairGraphUpdate,
  onStopWatcher,
  onNewSession,
}: Props) {
  const [listWidth, setListWidth] = useState(() => readChatListWidth(project.id));
  const [listCollapsed, setListCollapsed] = useState(() => readChatListCollapsed(project.id));
  const [widthBounds, setWidthBounds] = useState<ChatListWidthBounds>(() =>
    chatListWidthBounds(typeof window === "undefined" ? 1200 : window.innerWidth),
  );
  const workspace = useRef<HTMLElement>(null);
  const selected =
    conversations.find((conversation) => conversation.chatId === selectedChatId) ??
    conversations[0] ??
    null;

  useEffect(() => {
    setListWidth(readChatListWidth(project.id));
    setListCollapsed(readChatListCollapsed(project.id));
  }, [project.id]);

  useEffect(() => {
    if (!project.id) return;
    try {
      window.localStorage.setItem(chatListWidthStorageKey(project.id), String(listWidth));
    } catch {
      // Layout state is a convenience; storage failures must not affect chat.
    }
  }, [listWidth, project.id]);

  useEffect(() => {
    if (!project.id) return;
    try {
      window.localStorage.setItem(chatListCollapsedStorageKey(project.id), String(listCollapsed));
    } catch {
      // Layout state is a convenience; storage failures must not affect chat.
    }
  }, [listCollapsed, project.id]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!isChatListToggleShortcut(event)) return;
      event.preventDefault();
      setListCollapsed((current) => !current);
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  useEffect(() => {
    const element = workspace.current;
    if (!element) return;

    const updateBounds = (width: number) => {
      const nextBounds = chatListWidthBounds(width);
      setWidthBounds(nextBounds);
      setListWidth((current) => clampChatListWidth(current, nextBounds));
    };
    updateBounds(element.getBoundingClientRect().width);
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) updateBounds(entry.contentRect.width);
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, [project.id]);

  const resizeFromPointer = (clientX: number) => {
    const bounds = workspace.current?.getBoundingClientRect();
    if (!bounds) return;
    setListWidth(clampChatListWidth(clientX - bounds.left, widthBounds));
  };

  return (
    <section
      className="chats-workspace"
      ref={workspace}
      style={{
        gridTemplateColumns: `${listCollapsed ? CHAT_LIST_COLLAPSED_WIDTH : listWidth}px ${CHAT_LIST_DIVIDER_WIDTH}px minmax(0, 1fr)`,
      }}
    >
      <aside
        className="conversation-list"
        aria-label="Project conversations"
        hidden={listCollapsed}
        id="conversation-list-panel"
      >
        <header>
          <MessageCircle size={16} />
          <strong>Chats</strong>
        </header>
        <div role="listbox" aria-label="Conversations">
          {conversations.map((conversation) => {
            const latest = conversation.tasks.at(-1);
            const active = conversation.tasks.some(isActiveTask);
            const unread = conversationHasUnread(conversation, unreadTaskIds);
            const selectedConversation = conversation.chatId === selected?.chatId;
            return (
              <button
                type="button"
                role="option"
                aria-selected={selectedConversation}
                aria-current={selectedConversation ? "page" : undefined}
                aria-label={`${conversation.title}, ${conversation.kind === "project_chat" ? "project" : "node"} conversation${unread ? ", unread result" : ""}`}
                className={`${selectedConversation ? "active" : ""}${unread ? " unread" : ""}`}
                onClick={() => onSelect(conversation.chatId)}
                key={conversation.chatId}
              >
                <span>{conversation.title}</span>
                <small>{conversation.kind === "project_chat" ? "Project" : "Node"}</small>
                {active && <Circle className="conversation-active" size={8} fill="currentColor" />}
                {!active && unread && (
                  <Circle
                    className="conversation-unread"
                    size={8}
                    fill="currentColor"
                    aria-hidden="true"
                  />
                )}
                {!active && !unread && latest && (
                  <time>{new Date(latest.updated_at).toLocaleDateString()}</time>
                )}
              </button>
            );
          })}
        </div>
        {hasMore && (
          <footer className="conversation-list-more">
            <button
              className="button primary compact"
              type="button"
              disabled={loadingMore}
              onClick={onLoadMore}
            >
              {loadingMore && <LoaderCircle className="spin" size={12} />}
              Load more
            </button>
          </footer>
        )}
      </aside>
      <div className={`conversation-divider${listCollapsed ? " is-collapsed" : ""}`}>
        <div
          aria-controls="conversation-list-panel conversation-surface-panel"
          aria-hidden={listCollapsed}
          aria-label="Resize conversation list"
          aria-orientation="vertical"
          aria-valuemax={widthBounds.maximum}
          aria-valuemin={widthBounds.minimum}
          aria-valuenow={Math.round(listWidth)}
          className="conversation-resize-handle"
          onKeyDown={
            listCollapsed
              ? undefined
              : (event) => {
                  if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
                    event.preventDefault();
                    setListWidth((current) =>
                      clampChatListWidth(
                        current + (event.key === "ArrowLeft" ? -16 : 16),
                        widthBounds,
                      ),
                    );
                  }
                  if (event.key === "Home" || event.key === "End") {
                    event.preventDefault();
                    setListWidth(event.key === "Home" ? widthBounds.minimum : widthBounds.maximum);
                  }
                }
          }
          onPointerCancel={
            listCollapsed
              ? undefined
              : (event) => {
                  if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                    event.currentTarget.releasePointerCapture(event.pointerId);
                  }
                }
          }
          onPointerDown={
            listCollapsed
              ? undefined
              : (event) => {
                  event.currentTarget.setPointerCapture(event.pointerId);
                  resizeFromPointer(event.clientX);
                }
          }
          onPointerMove={
            listCollapsed
              ? undefined
              : (event) => {
                  if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                    resizeFromPointer(event.clientX);
                  }
                }
          }
          onPointerUp={
            listCollapsed
              ? undefined
              : (event) => {
                  if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                    event.currentTarget.releasePointerCapture(event.pointerId);
                  }
                }
          }
          role="separator"
          tabIndex={listCollapsed ? -1 : 0}
        />
        <button
          aria-controls="conversation-list-panel"
          aria-expanded={!listCollapsed}
          aria-keyshortcuts="Meta+B"
          aria-label={listCollapsed ? "Expand conversation list" : "Collapse conversation list"}
          className="conversation-divider-toggle"
          onClick={() => setListCollapsed((current) => !current)}
          type="button"
        >
          {listCollapsed ? <ChevronRight size={13} /> : <ChevronLeft size={13} />}
        </button>
      </div>
      <div className="conversation-surface" id="conversation-surface-panel">
        {selected ? (
          <NodeChat
            key={selected.chatId}
            project={project}
            node={selected.nodeId ? (nodes[selected.nodeId] ?? null) : null}
            nodes={nodes}
            glossaryIndex={glossaryIndex}
            conversationTitle={selected.kind === "node_chat" ? selected.title : undefined}
            runScope={runScope}
            tasks={tasks}
            watchers={watchers}
            historyMessages={chatTranscripts.get(selected.chatId)?.messages}
            chatId={selected.chatId}
            presentation="workspace"
            graphChangesDisabled={graphChangesDisabled}
            onStartTask={onStartTask}
            onResumeTask={onResumeTask}
            onRetryTask={onRetryTask}
            onInspectTask={onInspectTask}
            onOpenInbox={onOpenInbox}
            onRepairGraphUpdate={onRepairGraphUpdate}
            onOpenNode={onOpenNode}
            onStopWatcher={onStopWatcher}
            onNewSession={() => onNewSession(selected)}
            onClose={() => undefined}
          />
        ) : null}
      </div>
    </section>
  );
}
