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
const { AcceptanceAgentIndicator } = await server.ssrLoadModule("/src/App.tsx");

after(() => server.close());

test("acceptance mode is visibly distinct from real provider operation", () => {
  const html = renderToStaticMarkup(
    React.createElement(AcceptanceAgentIndicator, { agentMode: "acceptance" }),
  );

  assert.match(html, /role="status"/);
  assert.match(html, /Fake acceptance agent active/);
  assert.match(html, /Acceptance mode · no real provider calls/);
});

test("provider mode does not show the acceptance-agent indicator", () => {
  const html = renderToStaticMarkup(
    React.createElement(AcceptanceAgentIndicator, { agentMode: "provider" }),
  );

  assert.equal(html, "");
});
