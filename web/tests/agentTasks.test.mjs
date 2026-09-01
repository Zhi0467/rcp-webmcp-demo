import assert from "node:assert/strict";
import { withTaskAnswers } from "./taskAnswers.mjs";
import test from "node:test";

import {
  artifactUrl,
  chatMessageTranscriptLine,
  chatTasksMissingFromHistory,
  latestNativeSessionId,
  orderTranscriptLines,
  parseDismissedTaskIds,
  projectActivityTask,
  reconcileChatHistoryArtifacts,
  reconstructTaskTranscript,
  relatedChatTasks,
  resumablePausedChatTask,
  serializeDismissedTaskIds,
  taskKindLabel,
  taskNotificationStorageKey,
} from "../src/agentTasks.ts";

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

function artifact(overrides = {}) {
  return {
    artifact_id: "a".repeat(24),
    name: "chart.html",
    media_type: "text/html",
    available: true,
    unavailable_reason: null,
    can_open: true,
    can_download: true,
    can_keep: true,
    can_revise: true,
    ...overrides,
  };
}

test("branch merge tasks keep a human-readable activity label", () => {
  assert.equal(taskKindLabel("branch_merge"), "Branch merge");
});

test("node chat reconstruction follows the latest chat id for that node", () => {
  const tasks = [
    task({
      operation_id: "old",
      request: { node_id: "node/a", chat_id: "old-chat", message: "Old question" },
    }),
    task({
      operation_id: "first",
      created_at: "2026-07-28T00:01:00Z",
      request: { node_id: "node/a", chat_id: "new-chat", message: "What does this mean?" },
      result: { messages: ["First answer"] },
      native_session_id: "native-1",
    }),
    task({
      operation_id: "other",
      created_at: "2026-07-28T00:02:00Z",
      request: { node_id: "node/b", chat_id: "other-chat", message: "Different node" },
    }),
    task({
      operation_id: "followup",
      created_at: "2026-07-28T00:03:00Z",
      request: { node_id: "node/a", chat_id: "new-chat", message: "Clarify it" },
      result: { messages: ["Clearer answer"] },
      native_session_id: "native-1",
    }),
  ];

  const related = relatedChatTasks(tasks, "node_chat", "node/a");
  assert.deepEqual(
    related.map((item) => item.operation_id),
    ["first", "followup"],
  );
  assert.deepEqual(
    reconstructTaskTranscript(related).map(({ role, text }) => ({ role, text })),
    [
      { role: "human", text: "What does this mean?" },
      { role: "agent", text: "First answer" },
      { role: "human", text: "Clarify it" },
      { role: "agent", text: "Clearer answer" },
    ],
  );
  assert.equal(latestNativeSessionId(related), "native-1");
});

test("temporary input attachment metadata follows the human turn only", () => {
  const attachment = {
    attachment_id: "attachment-a",
    name: "evidence.csv",
    media_type: "text/csv",
    size: 42,
    expires_at: "2026-08-15T00:00:00Z",
  };
  const lines = reconstructTaskTranscript([
    task({
      operation_id: "with-attachment",
      request: { message: "Read this", attachments: [attachment] },
      result: { messages: ["Read"] },
    }),
  ]);
  assert.deepEqual(lines[0].attachments, [attachment]);
  assert.equal(lines[1].attachments, undefined);
});

test("node chat reconstruction can select an older explicit chat id", () => {
  const tasks = [
    task({
      operation_id: "old",
      request: { node_id: "node/a", chat_id: "old-chat", message: "Old question" },
    }),
    task({
      operation_id: "new",
      created_at: "2026-07-28T00:01:00Z",
      request: { node_id: "node/a", chat_id: "new-chat", message: "New question" },
    }),
  ];
  assert.deepEqual(
    relatedChatTasks(tasks, "node_chat", "node/a", "old-chat").map((item) => item.operation_id),
    ["old"],
  );
});

test("paused resumable attempts block a new chat turn", () => {
  const paused = task({ operation_id: "paused", status: "paused", can_resume: true });
  assert.equal(resumablePausedChatTask([paused])?.operation_id, "paused");
  assert.equal(resumablePausedChatTask([{ ...paused, can_resume: false }]), null);
  const resumed = task({
    operation_id: "resumed",
    parent_operation_id: "paused",
    status: "succeeded",
  });
  assert.equal(resumablePausedChatTask([paused, resumed]), null);
});

test("task identity, not repeated prompt text, decides reconstruction", () => {
  const older = task({
    operation_id: "older",
    created_at: "2026-07-28T00:00:00Z",
    request: { message: "Same prompt" },
  });
  const active = task({
    operation_id: "active",
    status: "running",
    created_at: "2026-07-28T01:00:00Z",
    request: { message: "Same prompt" },
  });
  const messages = [
    {
      message_id: "message",
      operation_id: "older",
      role: "user",
      text: "Same prompt",
      timestamp: "2026-07-28T00:00:01Z",
      native_session_id: null,
      provider: null,
      model: null,
      reasoning: null,
      execution_machine: null,
      applied_revision: null,
    },
  ];
  assert.deepEqual(
    chatTasksMissingFromHistory([older, active], messages).map((item) => item.operation_id),
    ["active"],
  );
});

test("two identical task prompts both remain visible when neither has durable history", () => {
  const first = task({
    operation_id: "first",
    created_at: "2026-07-28T00:00:00Z",
    request: { message: "Same prompt" },
  });
  const second = task({
    operation_id: "second",
    created_at: "2026-07-28T00:01:00Z",
    request: { message: "Same prompt" },
  });
  assert.deepEqual(
    chatTasksMissingFromHistory([first, second], []).map((item) => item.operation_id),
    ["first", "second"],
  );
});

test("an older failed repeated prompt remains beside a newer durable turn", () => {
  const failed = task({
    operation_id: "failed",
    created_at: "2026-07-28T00:00:00Z",
    status: "failed",
    request: { message: "Same prompt" },
    error: "Provider exited",
  });
  const durable = {
    message_id: "new-user",
    operation_id: "succeeded",
    role: "user",
    text: "Same prompt",
    timestamp: "2026-07-28T00:01:00Z",
    mode: null,
    graph_update: null,
  };
  const missing = chatTasksMissingFromHistory([failed], [durable]);
  assert.deepEqual(
    missing.map((item) => item.operation_id),
    ["failed"],
  );
  assert.deepEqual(
    orderTranscriptLines([
      ...reconstructTaskTranscript(missing),
      chatMessageTranscriptLine(durable),
    ]).map(({ role, text }) => ({ role, text })),
    [
      { role: "human", text: "Same prompt" },
      { role: "error", text: "Provider exited" },
      { role: "human", text: "Same prompt" },
    ],
  );
});

test("task-only failures are merged into chronological chat history", () => {
  const durable = {
    message_id: "answer",
    operation_id: "later",
    role: "assistant",
    text: "Later answer",
    timestamp: "2026-07-28T00:02:00Z",
    mode: null,
    graph_update: null,
  };
  const failed = task({
    operation_id: "earlier",
    created_at: "2026-07-28T00:01:00Z",
    status: "failed",
    request: { message: "Earlier prompt" },
    error: "Provider exited",
  });
  assert.deepEqual(
    orderTranscriptLines([
      ...reconstructTaskTranscript([failed]),
      chatMessageTranscriptLine(durable),
    ]).map(({ role, text }) => ({ role, text })),
    [
      { role: "human", text: "Earlier prompt" },
      { role: "error", text: "Provider exited" },
      { role: "agent", text: "Later answer" },
    ],
  );
});

test("an assistant-only repair receipt suppresses reconstruction of its child task", () => {
  const repair = task({
    operation_id: "repair",
    parent_operation_id: "original",
    request: { message: "Original work", mode: "work" },
  });
  const messages = [
    {
      message_id: "receipt",
      operation_id: "repair",
      role: "assistant",
      text: "",
      timestamp: "2026-07-28T00:01:00Z",
      mode: "work",
      graph_update: null,
    },
  ];
  assert.deepEqual(chatTasksMissingFromHistory([repair], messages), []);
});

test("failed tasks preserve the human prompt and surfaced error", () => {
  const failed = task({
    operation_id: "failed",
    status: "failed",
    request: { node_id: "node/a", chat_id: "chat", message: "Rewrite this" },
    result: null,
    error: "Provider exited",
  });

  assert.deepEqual(
    reconstructTaskTranscript([failed]).map(({ role, text }) => ({ role, text })),
    [
      { role: "human", text: "Rewrite this" },
      { role: "error", text: "Provider exited" },
    ],
  );
});

test("failed chat tasks render a preserved answer before the error", () => {
  const failed = task({
    operation_id: "rejected-change",
    status: "failed",
    request: {
      node_id: "node/a",
      chat_id: "chat",
      message: "Explain this and update the graph",
    },
    result: { messages: ["Here is the explanation that completed before the edit failed."] },
    error: "The graph moved while this patch was being written",
  });

  assert.deepEqual(
    reconstructTaskTranscript([failed]).map(({ role, text }) => ({ role, text })),
    [
      { role: "human", text: "Explain this and update the graph" },
      { role: "agent", text: "Here is the explanation that completed before the edit failed." },
      { role: "error", text: "The graph moved while this patch was being written" },
    ],
  );
});

test("artifacts stay attached to the answer when a later task error is present", () => {
  const artifacts = [artifact()];
  const failed = task({
    operation_id: "artifact-change-rejected",
    status: "failed",
    request: { node_id: "node/a", chat_id: "chat", message: "Show it and update the graph" },
    result: { messages: ["The result is **ready**."], artifacts },
    error: "Graph change rejected",
  });

  const transcript = reconstructTaskTranscript([failed]);
  assert.deepEqual(
    transcript.map(({ role, text }) => ({ role, text })),
    [
      { role: "human", text: "Show it and update the graph" },
      { role: "agent", text: "The result is **ready**." },
      { role: "error", text: "Graph change rejected" },
    ],
  );
  assert.deepEqual(transcript[1].artifacts, artifacts);
});

test("persisted chat reconciliation keeps task artifacts on the assistant answer", () => {
  const artifacts = [artifact({ artifact_id: "b".repeat(24), name: "preview.html" })];
  const completed = task({
    operation_id: "artifact-turn",
    request: { chat_id: "chat", message: "Build a preview" },
    result: { messages: ["Preview ready."], artifacts },
  });
  const messages = [
    {
      message_id: "human-message",
      operation_id: "artifact-turn",
      role: "user",
      text: "Build a preview",
      timestamp: "2026-07-28T00:00:01Z",
      mode: "discuss",
      graph_update: null,
      attachments: [],
      trigger: "human",
    },
    {
      message_id: "agent-message",
      operation_id: "artifact-turn",
      role: "assistant",
      text: "Preview ready.",
      timestamp: "2026-07-28T00:00:02Z",
      mode: "discuss",
      graph_update: null,
      attachments: [],
      trigger: "human",
    },
  ];

  assert.deepEqual(chatTasksMissingFromHistory([completed], messages), []);
  assert.deepEqual(reconcileChatHistoryArtifacts(messages, [completed])[1].artifacts, artifacts);
});

test("historical artifact decisions survive transcript reconciliation without UI inference", () => {
  const unavailable = artifact({
    artifact_id: "c".repeat(24),
    name: "expired.html",
    available: false,
    unavailable_reason: "Artifact bytes were not retained with this task history.",
    can_open: false,
    can_download: false,
    can_keep: false,
    can_revise: false,
  });
  const completed = task({
    operation_id: "historical-artifact",
    history_only: true,
    native_session_id: null,
    result: { messages: ["Historical result."], artifacts: [unavailable] },
  });

  assert.deepEqual(reconstructTaskTranscript([completed])[0].artifacts, [unavailable]);
  assert.equal(latestNativeSessionId([completed]), null);
});

test("artifact metadata without backend decisions is not rendered", () => {
  const incomplete = {
    artifact_id: "d".repeat(24),
    name: "legacy.html",
    media_type: "text/html",
  };
  const completed = task({
    operation_id: "incomplete-artifact",
    result: { messages: ["Legacy result."], artifacts: [incomplete] },
  });

  assert.equal(reconstructTaskTranscript([completed])[0].artifacts, undefined);
});

test("conversation reconstruction preserves immutable mode and graph receipt metadata", () => {
  const graphUpdate = {
    status: "applied",
    applied_revision: 12,
    change_summary: ["Recorded attempt exp/demo/attempt-2"],
    proposal_ids: ["proposal/decision"],
    validation_messages: [],
    correction_rounds: 1,
    repairable: false,
  };
  const completed = task({
    operation_id: "work-turn",
    request: { chat_id: "chat", message: "Run it", mode: "work" },
    result: { messages: ["The experiment completed."], graph_update: graphUpdate },
  });

  const transcript = reconstructTaskTranscript([completed]);
  assert.equal(transcript[0].mode, "work");
  assert.equal(transcript[1].mode, "work");
  assert.deepEqual(transcript[1].graphUpdate, graphUpdate);
});

test("legacy turns remain unlabelled and graph-only rejection is not an operational error", () => {
  const completed = task({
    operation_id: "rejected-reflection",
    request: { chat_id: "chat", message: "Run it" },
    status: "succeeded",
    error: "Graph update rejected",
    result: {
      messages: ["The experiment completed."],
      graph_update: {
        status: "rejected",
        applied_revision: null,
        change_summary: [],
        proposal_ids: [],
        validation_messages: ["The graph moved."],
        correction_rounds: 2,
        repairable: true,
      },
    },
  });

  const transcript = reconstructTaskTranscript([completed]);
  assert.equal(transcript[0].mode, null);
  assert.deepEqual(
    transcript.map(({ role, text }) => ({ role, text })),
    [
      { role: "human", text: "Run it" },
      { role: "agent", text: "The experiment completed." },
    ],
  );
  assert.equal(transcript[1].graphUpdate.status, "rejected");
});

test("durable chat lines retain their task identity with a legacy message fallback", () => {
  const record = {
    message_id: "message-id",
    operation_id: "task-id",
    role: "assistant",
    text: "Done",
    mode: "work",
    graph_update: null,
  };
  assert.equal(chatMessageTranscriptLine(record).taskId, "task-id");
  assert.equal(chatMessageTranscriptLine({ ...record, operation_id: null }).taskId, "message-id");
});

test("a watcher wake reconstructs only an attributed agent line", () => {
  const wake = task({
    operation_id: "watcher-wake",
    request: {
      chat_id: "chat",
      trigger: "watcher",
      message: "Inspect watcher/one and watcher/two",
      mode: "work",
    },
    result: { messages: ["Both detached jobs are finished."] },
  });
  const transcript = reconstructTaskTranscript([wake]);
  assert.deepEqual(
    transcript.map(({ role, text, trigger }) => ({ role, text, trigger })),
    [
      {
        role: "agent",
        text: "Both detached jobs are finished.",
        trigger: "watcher",
      },
    ],
  );
});

test("durable watcher messages never occupy the human side of a conversation", () => {
  const line = chatMessageTranscriptLine({
    message_id: "watcher-message",
    operation_id: "watcher-task",
    role: "assistant",
    text: "The watched work is gone.",
    mode: "work",
    graph_update: null,
    trigger: "watcher",
  });
  assert.equal(line.role, "agent");
  assert.equal(line.trigger, "watcher");
});

test("artifact URLs contain only RCP identifiers and the explicit action", () => {
  assert.equal(
    artifactUrl("project/id", "task id", "artifact#id", "content"),
    "/api/projects/project%2Fid/tasks/task%20id/artifacts/artifact%23id/content",
  );
  assert.equal(
    artifactUrl("project/id", "task id", "artifact#id", "preview"),
    "/api/projects/project%2Fid/tasks/task%20id/artifacts/artifact%23id/preview",
  );
  assert.equal(
    artifactUrl("project/id", "task id", "artifact#id", "download"),
    "/api/projects/project%2Fid/tasks/task%20id/artifacts/artifact%23id/download",
  );
});

test("project entry does not promote terminal task history into the activity strip", () => {
  const failed = task({ operation_id: "failed", status: "failed", error: "Provider exited" });
  const succeeded = task({ operation_id: "succeeded" });

  assert.equal(projectActivityTask([failed, succeeded], null), null);
});

test("project entry does not promote a paused attempt that already has a child", () => {
  const oldestPaused = task({ operation_id: "oldest-paused", status: "paused" });
  const pausedAncestor = task({
    operation_id: "paused-ancestor",
    status: "paused",
    parent_operation_id: "oldest-paused",
  });
  const failedChild = task({
    operation_id: "failed-child",
    status: "failed",
    parent_operation_id: "paused-ancestor",
    error: "Provider exited",
  });
  const laterRefresh = task({ operation_id: "later-refresh", kind: "refresh" });

  assert.equal(
    projectActivityTask([laterRefresh, failedChild, pausedAncestor, oldestPaused], null),
    null,
  );
});

test("project activity follows ongoing work and keeps its observed terminal result", () => {
  const running = task({ operation_id: "running", status: "running" });
  const paused = task({ operation_id: "paused", status: "paused" });
  const failed = task({ operation_id: "running", status: "failed", error: "Provider exited" });

  assert.equal(projectActivityTask([running], null)?.operation_id, "running");
  assert.equal(projectActivityTask([paused], null)?.operation_id, "paused");
  assert.equal(projectActivityTask([failed], "running")?.status, "failed");
});

test("a later graph success clears an older failed graph notification", () => {
  const failedSeed = task({
    operation_id: "failed-seed",
    kind: "seed",
    status: "failed",
    created_at: "2026-07-28T00:00:00Z",
  });
  const completedRefresh = task({
    operation_id: "completed-refresh",
    kind: "refresh",
    status: "succeeded",
    created_at: "2026-07-28T01:00:00Z",
  });
  assert.equal(projectActivityTask([completedRefresh, failedSeed], "failed-seed"), null);
});

test("a later graph success does not clear a paused graph task", () => {
  const pausedSeed = task({
    operation_id: "paused-seed",
    kind: "seed",
    status: "paused",
    created_at: "2026-07-28T00:00:00Z",
  });
  const completedRefresh = task({
    operation_id: "completed-refresh",
    kind: "refresh",
    status: "succeeded",
    created_at: "2026-07-28T01:00:00Z",
  });
  assert.equal(
    projectActivityTask([completedRefresh, pausedSeed], null)?.operation_id,
    "paused-seed",
  );
});

test("a completed retry clears its failed parent, while a failed retry stays actionable", () => {
  const failedSeed = task({
    operation_id: "failed-seed",
    kind: "seed",
    status: "failed",
    created_at: "2026-07-28T00:00:00Z",
  });
  const completedRetry = task({
    operation_id: "completed-retry",
    kind: "seed",
    status: "succeeded",
    parent_operation_id: "failed-seed",
    created_at: "2026-07-28T01:00:00Z",
  });
  assert.equal(projectActivityTask([completedRetry, failedSeed], "failed-seed"), null);

  const failedRetry = withTaskAnswers({
    ...completedRetry,
    operation_id: "failed-retry",
    status: "failed",
  });
  assert.equal(
    projectActivityTask([failedRetry, failedSeed], "failed-seed")?.operation_id,
    "failed-retry",
  );
});

test("dismissed task notification ids round trip by project", () => {
  const ids = new Set(["task-b", "task-a"]);
  assert.equal(taskNotificationStorageKey("project"), "rcp:dismissed-task-notifications:project");
  assert.equal(serializeDismissedTaskIds(ids), '["task-a","task-b"]');
  assert.deepEqual(parseDismissedTaskIds('["task-b","task-a",3]'), ids);
  assert.deepEqual(parseDismissedTaskIds("not json"), new Set());
});
