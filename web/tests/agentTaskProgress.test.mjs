import assert from "node:assert/strict";
import { withTaskAnswers } from "./taskAnswers.mjs";
import { after, test } from "node:test";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";

const server = await createServer({
  root: new URL("..", import.meta.url).pathname,
  configFile: false,
  logLevel: "silent",
  server: { middlewareMode: true, hmr: false },
  optimizeDeps: { noDiscovery: true },
});
const { AgentTaskInspector } = await server.ssrLoadModule("/src/components/AgentTaskInspector.tsx");

after(() => server.close());

function task(status) {
  const now = "2026-08-03T12:00:00Z";
  return withTaskAnswers({
    operation_id: `task-${status}`,
    project_id: "project",
    kind: "seed",
    status,
    request: { provider: "codex", run_on: "local" },
    created_at: now,
    updated_at: now,
    status_message: status === "failed" ? "Provider failed" : "Task status",
    attempt: 1,
    estimate_seconds: 300,
    estimate_samples: 1,
    phase: "agent",
    elapsed_seconds: 10,
    progress: 0.03,
    can_pause: status === "running",
    can_resume: status === "paused",
    can_retry: status === "failed" || status === "interrupted",
    events: [],
  });
}

function renderInspector(status) {
  const selectedTask = task(status);
  return renderToStaticMarkup(
    React.createElement(AgentTaskInspector, {
      tasks: [selectedTask],
      task: selectedTask,
      loading: false,
      actionBusy: false,
      onSelect() {},
      onPause() {},
      onResume() {},
      onRetry() {},
      onDismiss() {},
      onClose() {},
    }),
  );
}

test("active tasks show status without a progress bar", () => {
  const inspector = renderInspector("running");

  assert.match(inspector, /Running in the background/);
  assert.doesNotMatch(inspector, /progressbar|Estimated (?:agent )?progress/);
  assert.doesNotMatch(inspector, />3%<\/(?:span|strong)>/);
  assert.doesNotMatch(inspector, /about 5m left/);
});

test("terminal tasks show no live progress in the inspector", () => {
  for (const status of ["failed", "succeeded", "interrupted", "paused"]) {
    const html = renderInspector(status);
    assert.doesNotMatch(html, /Estimated (?:agent )?progress/);
    assert.doesNotMatch(html, />3%<\/(?:span|strong)>/);
    assert.doesNotMatch(html, /about 5m left/);
  }
});

test("Auto-research tasks are inspection-only while ordinary task controls remain available", () => {
  const ordinary = renderInspector("running");
  const episodeTask = { ...task("running"), kind: "auto_research", episode_id: "episode" };
  const episodeInspector = renderToStaticMarkup(
    React.createElement(AgentTaskInspector, {
      tasks: [episodeTask],
      task: episodeTask,
      loading: false,
      actionBusy: false,
      onSelect() {},
      onPause() {},
      onResume() {},
      onRetry() {},
      onDismiss() {},
      onClose() {},
    }),
  );

  assert.match(ordinary, /run-inspector-actions/);
  assert.doesNotMatch(episodeInspector, /run-inspector-actions/);
});
