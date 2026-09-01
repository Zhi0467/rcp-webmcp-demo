import assert from "node:assert/strict";
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
const {
  AddTeamSpaceDialog,
  TeamConnectionGroup,
  createTeamReconciliationTracker,
  replaceTeamConnectionView,
} = await server.ssrLoadModule("/src/components/TeamSpaceGroups.tsx");

after(() => server.close());

const connection = {
  connection_id: "11111111-1111-4111-8111-111111111111",
  display_name: "Causal Systems Lab",
  ssh_target: "rcp@lab-server",
  remote_loopback_port: 8421,
  expected_space_id: "22222222-2222-4222-8222-222222222222",
  local_origin: "https://rcp-11111111111141118111111111111111.localhost:18421",
  minimum_shell_version: "0.3.2",
  last_known_cards: [
    {
      id: "33333333-3333-4333-8333-333333333333",
      name: "Plasticity study",
      primary_question: "What remains stable?",
      attention_count: 2,
    },
  ],
  operator_route: null,
};

test("an available team group exposes its verified cached project", () => {
  const html = renderToStaticMarkup(
    React.createElement(TeamConnectionGroup, {
      view: { connection, state: "available", error: null },
      onReconnect() {},
      onOpenProject() {},
    }),
  );

  assert.match(html, /Causal Systems Lab/);
  assert.match(html, /Plasticity study/);
  assert.match(html, /2 waiting/);
  assert.doesNotMatch(html, /disabled=""/);
  assert.doesNotMatch(html, /Reconnect/);
});

test("an unavailable team group keeps cached cards visible but inert", () => {
  const html = renderToStaticMarkup(
    React.createElement(TeamConnectionGroup, {
      view: { connection, state: "unavailable", error: "server unavailable" },
      onReconnect() {},
      onOpenProject() {},
    }),
  );

  assert.match(html, /Plasticity study/);
  assert.match(html, /<button[^>]*>.*Reconnect/s);
  assert.match(html, /class="team-project-card"[^>]*disabled=""/);
  assert.match(html, /role="alert"[^>]*>server unavailable/);
});

test("connection updates preserve saved registry order", () => {
  const second = {
    ...connection,
    connection_id: "44444444-4444-4444-8444-444444444444",
    display_name: "Second Lab",
  };
  const checking = [connection, second].map((item) => ({
    connection: item,
    state: "checking",
    error: null,
  }));
  const updated = replaceTeamConnectionView(checking, {
    connection: second,
    state: "available",
    error: null,
  });

  assert.deepEqual(
    updated.map((view) => view.connection.connection_id),
    [connection.connection_id, second.connection_id],
  );
  assert.equal(updated[1].state, "available");
});

test("superseded and stopped reconciliation attempts cannot publish", () => {
  const tracker = createTeamReconciliationTracker();
  const first = tracker.begin(connection.connection_id);
  const second = tracker.begin(connection.connection_id);

  assert.equal(tracker.isCurrent(connection.connection_id, first), false);
  assert.equal(tracker.isCurrent(connection.connection_id, second), true);
  tracker.stop();
  assert.equal(tracker.isCurrent(connection.connection_id, second), false);
});

test("Add team space keeps the one credential in a password field and out of URLs", () => {
  const html = renderToStaticMarkup(
    React.createElement(AddTeamSpaceDialog, {
      onClose() {},
      onEstablished() {},
    }),
  );

  assert.match(html, /<form[^>]*autoComplete="off"/);
  assert.match(html, /SSH target/);
  assert.match(html, /Bootstrap or invitation code/);
  assert.match(html, /<input[^>]*type="password"/);
  assert.doesNotMatch(html, /action=|localStorage|sessionStorage|[?&](token|code)=/i);
});
