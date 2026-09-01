import assert from "node:assert/strict";
import { after, test } from "node:test";
import { readFile } from "node:fs/promises";
import { createServer } from "vite";

import { loadServerStatus } from "../src/api.ts";

const server = await createServer({
  root: new URL("..", import.meta.url).pathname,
  configFile: false,
  logLevel: "silent",
  server: { middlewareMode: true, hmr: false },
  optimizeDeps: { noDiscovery: true },
});
const { formatServerBytes, formatServerProjectCounts, formatServerTimestamp, shortCommit } =
  await server.ssrLoadModule("/src/components/ServerSettings.tsx");

after(() => server.close());

test("server status uses one authenticated read-only API call", async () => {
  const originalFetch = globalThis.fetch;
  const requests = [];
  const response = { overall: { label: "Server is healthy", tone: "good" } };
  globalThis.fetch = async (path, init) => {
    requests.push({ path, init });
    return new Response(JSON.stringify(response), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };
  try {
    assert.deepEqual(await loadServerStatus(), response);
    assert.equal(requests.length, 1);
    assert.equal(requests[0].path, "/api/server-status");
    assert.equal(requests[0].init.method, undefined);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("server status formats exact backend facts without deriving lifecycle state", () => {
  assert.equal(shortCommit("a".repeat(40)), "a".repeat(10));
  assert.equal(shortCommit(null), "Not available");
  assert.equal(formatServerBytes(4096), "4.0 KB");
  assert.equal(formatServerBytes(null), "Not recorded");
  assert.equal(formatServerProjectCounts(3, 1), "3 protected · 1 uncaptured");
  assert.equal(formatServerProjectCounts(0, 0), "0 protected · 0 uncaptured");
  assert.equal(formatServerProjectCounts(null, null), "Not recorded");
  assert.notEqual(formatServerTimestamp("2026-08-30T12:00:00Z"), "Not recorded");
});

test("server settings exposes command names as text and no machine mutation handler", async () => {
  const source = await readFile(
    new URL("../src/components/ServerSettings.tsx", import.meta.url),
    "utf8",
  );

  assert.match(source, /status\.operator_commands\.map/);
  assert.match(source, /<code>\{item\.command\}<\/code>/);
  assert.match(source, /catch \(failure\) \{\s*setStatus\(null\);\s*setError\(/);
  assert.doesNotMatch(source, /method:\s*["'](?:POST|PUT|PATCH|DELETE)/);
  assert.doesNotMatch(source, /runDesktopServerCommand|invokeServerCommand/);
});
