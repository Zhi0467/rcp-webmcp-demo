import assert from "node:assert/strict";
import { withTaskAnswers } from "./taskAnswers.mjs";
import test from "node:test";

import {
  chatDraftStorageKey,
  chatIdForTask,
  chatIndicator,
  chatEntryConversationId,
  chatModeStorageKey,
  conversationHasUnread,
  groupChatConversations,
  isConversationModeShortcut,
  latestConversation,
  latestPersistedChatConfig,
  latestPersistedConversationMode,
  newlyUnreadChatTaskIds,
  parseConversationMode,
  toggleConversationMode,
} from "../src/chatWorkspace.ts";

function task(overrides) {
  return withTaskAnswers({
    operation_id: overrides.operation_id,
    project_id: "project",
    kind: "node_chat",
    status: "succeeded",
    request: {},
    created_at: "2026-07-28T00:00:00Z",
    updated_at: "2026-07-28T00:00:00Z",
    status_message: "Done",
    attempt: 1,
    estimate_seconds: 1,
    estimate_samples: 0,
    phase: "done",
    elapsed_seconds: 1,
    progress: 1,
    can_pause: false,
    can_resume: false,
    can_retry: false,
    ...overrides,
  });
}

test("conversations group by chat id rather than latest node", () => {
  const tasks = [
    task({ operation_id: "a1", request: { chat_id: "chat-a", node_id: "node/a" } }),
    task({
      operation_id: "b1",
      created_at: "2026-07-28T00:01:00Z",
      updated_at: "2026-07-28T00:01:00Z",
      request: { chat_id: "chat-b", node_id: "node/a" },
    }),
    task({
      operation_id: "p1",
      kind: "project_chat",
      created_at: "2026-07-28T00:02:00Z",
      updated_at: "2026-07-28T00:02:00Z",
      request: { chat_id: "chat-p" },
    }),
  ];
  const summaries = tasks.map((item) => ({
    chat_id: item.request.chat_id,
    kind: item.kind,
    node_id: item.request.node_id ?? null,
    title: item.kind === "project_chat" ? "Project question" : "Node question",
    updated_at: item.updated_at,
    message_count: 2,
    last_message_preview: "Answer",
  }));
  const conversations = groupChatConversations(summaries, tasks, { "node/a": "Node A" }, "Project");
  assert.deepEqual(
    conversations.map((item) => item.chatId),
    ["chat-p", "chat-b", "chat-a"],
  );
  assert.equal(conversations.find((item) => item.chatId === "chat-a")?.title, "Node A");
  assert.equal(latestConversation(conversations, "node_chat", "node/a")?.chatId, "chat-b");
  assert.equal(chatIdForTask(tasks[2]), "chat-p");
});

test("draft conversations survive without tasks and indicators distinguish active and unread", () => {
  const draft = { chatId: "empty", kind: "project_chat", nodeId: null, title: "Project" };
  assert.equal(groupChatConversations([], [], {}, "Project", [draft])[0].chatId, "empty");
  const running = task({
    operation_id: "running",
    status: "running",
    request: { chat_id: "live", node_id: "node/a" },
  });
  const done = task({ operation_id: "done", request: { chat_id: "done", node_id: "node/a" } });
  assert.equal(chatIndicator([done, running], new Set(["done"])), "active");
  assert.equal(chatIndicator([done], new Set(["done"])), "unread");
  assert.equal(chatIndicator([done], new Set()), null);
});

test("a terminal task stays reachable when durable transcript persistence failed", () => {
  const completed = task({
    operation_id: "unpersisted",
    request: { chat_id: "task-only", node_id: "node/a", message: "Keep this answer" },
    result: { messages: ["Still available from the task receipt"] },
  });
  const conversations = groupChatConversations([], [completed], { "node/a": "Node A" }, "Project");
  assert.equal(conversations.length, 1);
  assert.equal(conversations[0].chatId, "task-only");
  assert.deepEqual(conversations[0].tasks, [completed]);
});

test("entry preserves the previous chat before routing active or unread work", () => {
  const active = task({
    operation_id: "active",
    status: "running",
    request: { chat_id: "active-chat", node_id: "node/a" },
  });
  const terminal = task({
    operation_id: "terminal",
    request: { chat_id: "read-chat", node_id: "node/a" },
  });
  const summaries = [
    {
      chat_id: "active-chat",
      kind: "node_chat",
      node_id: "node/a",
      title: "Active",
      updated_at: "2026-07-28T00:03:00Z",
      message_count: 1,
      last_message_preview: "",
    },
    {
      chat_id: "unread-chat",
      kind: "project_chat",
      node_id: null,
      title: "Unread",
      updated_at: "2026-07-28T00:02:00Z",
      message_count: 2,
      last_message_preview: "",
    },
    {
      chat_id: "read-chat",
      kind: "node_chat",
      node_id: "node/a",
      title: "Read",
      updated_at: "2026-07-28T00:01:00Z",
      message_count: 2,
      last_message_preview: "",
    },
  ];
  const unreadTask = task({
    operation_id: "unread",
    kind: "project_chat",
    request: { chat_id: "unread-chat" },
  });
  const conversations = groupChatConversations(
    summaries,
    [active, unreadTask, terminal],
    {},
    "Project",
  );
  assert.equal(
    chatEntryConversationId(conversations, active, new Set(["unread"]), "read-chat"),
    "read-chat",
  );
  assert.equal(
    chatEntryConversationId(conversations, terminal, new Set(["unread"]), "read-chat"),
    "read-chat",
  );
  assert.equal(
    chatEntryConversationId(conversations, active, new Set(["unread"]), "missing-chat"),
    "active-chat",
  );
  assert.equal(
    chatEntryConversationId(conversations, terminal, new Set(["unread"]), "missing-chat"),
    "unread-chat",
  );
});

test("a completion is unread unless its exact conversation is selected and visible", () => {
  const completed = task({
    operation_id: "done",
    request: { chat_id: "chat-b", node_id: "node/b" },
  });
  const previous = new Map([["done", "running"]]);
  assert.deepEqual(newlyUnreadChatTaskIds([completed], previous, "chat-a"), ["done"]);
  assert.deepEqual(newlyUnreadChatTaskIds([completed], previous, "chat-b"), []);
  const conversations = groupChatConversations(
    [
      {
        chat_id: "chat-b",
        kind: "node_chat",
        node_id: "node/b",
        title: "B",
        updated_at: completed.updated_at,
        message_count: 2,
        last_message_preview: "done",
      },
    ],
    [completed],
    {},
    "Project",
  );
  assert.equal(conversationHasUnread(conversations[0], new Set(["done"])), true);
});

test("conversation mode controls have stable storage keys and Shift+Tab semantics", () => {
  assert.equal(chatDraftStorageKey("project", "chat"), "rcp:chat-draft:project:chat");
  assert.equal(chatModeStorageKey("project", "chat"), "rcp:chat-mode:project:chat");
  assert.equal(toggleConversationMode("discuss"), "work");
  assert.equal(toggleConversationMode("work"), "discuss");
  assert.equal(isConversationModeShortcut("Tab", true), true);
  assert.equal(isConversationModeShortcut("Tab", false), false);
  assert.equal(isConversationModeShortcut("Enter", true), false);
  assert.equal(parseConversationMode("work"), "work");
  assert.equal(parseConversationMode("legacy"), null);
});

test("chat provider configuration follows the persisted conversation over the project fallback", () => {
  const fallback = { provider: "codex", model: "", reasoning: "medium", run_on: "local" };
  const claude = { provider: "claude", model: "opus", reasoning: "high", run_on: "local" };
  assert.deepEqual(latestPersistedChatConfig([], [], fallback), fallback);
  assert.deepEqual(
    latestPersistedChatConfig(
      [
        {
          provider: "claude",
          model: "opus",
          reasoning: "high",
          execution_machine: "local",
          timestamp: "2026-07-28T00:02:00Z",
        },
      ],
      [],
      fallback,
    ),
    claude,
  );
});

test("continuing a chat translates the stored provider-default sentinel back to a real default", () => {
  // The transcript writes "provider-default" where a request writes "". Sending
  // the sentinel onward makes the provider reject it as an unknown model name.
  const fallback = { provider: "codex", model: "", reasoning: "medium", run_on: "local" };
  assert.deepEqual(
    latestPersistedChatConfig(
      [
        {
          provider: "codex",
          model: "provider-default",
          reasoning: "medium",
          execution_machine: "local",
          timestamp: "2026-08-06T00:01:00Z",
        },
      ],
      [],
      fallback,
    ),
    fallback,
  );
  assert.deepEqual(
    latestPersistedChatConfig(
      [],
      [
        {
          operation_id: "t1",
          created_at: "2026-08-06T00:02:00Z",
          request: {
            chat_id: "chat",
            provider: "codex",
            model: "provider-default",
            reasoning: "medium",
            run_on: "local",
          },
        },
      ],
      fallback,
    ),
    fallback,
  );
});

test("the next turn derives from the latest explicit mode without relabelling legacy history", () => {
  const messages = [
    { mode: null, timestamp: "2026-07-28T00:00:00Z" },
    { mode: "work", timestamp: "2026-07-28T00:01:00Z" },
  ];
  const laterDiscuss = task({
    operation_id: "later",
    created_at: "2026-07-28T00:02:00Z",
    request: { chat_id: "chat", mode: "discuss" },
  });
  assert.equal(latestPersistedConversationMode(messages, [laterDiscuss]), "discuss");
  assert.equal(
    latestPersistedConversationMode([{ mode: null, timestamp: "invalid" }], []),
    "discuss",
  );
});
