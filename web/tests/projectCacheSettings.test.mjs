import assert from "node:assert/strict";
import { after, test } from "node:test";
import { createServer } from "vite";

const server = await createServer({
  root: new URL("..", import.meta.url).pathname,
  configFile: false,
  logLevel: "silent",
  server: { middlewareMode: true, hmr: false },
  optimizeDeps: { noDiscovery: true },
});
const { publishCacheMetrics, showClearAllCachesWarning } = await server.ssrLoadModule(
  "/src/views/ProjectSettings.tsx",
);

after(() => server.close());

test("global clear replaces nonzero visible and parent cache metrics", () => {
  const nonzero = {
    remote_sources: { bytes: 128, count: 2 },
    session_slices: { bytes: 64, count: 1 },
  };
  const cleared = {
    remote_sources: { bytes: 0, count: 0 },
    session_slices: { bytes: 0, count: 0 },
  };
  let visible = nonzero;
  let parent = nonzero;

  publishCacheMetrics(
    cleared,
    (metrics) => {
      visible = metrics;
    },
    (metrics) => {
      parent = metrics;
    },
  );

  assert.deepEqual(visible, cleared);
  assert.deepEqual(parent, cleared);
});

test("opening the global warning clears unrelated settings status first", () => {
  const events = [];

  showClearAllCachesWarning(
    () => events.push("status-cleared"),
    () => events.push("warning-opened"),
  );

  assert.deepEqual(events, ["status-cleared", "warning-opened"]);
});
