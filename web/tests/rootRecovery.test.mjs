import assert from "node:assert/strict";
import { after, test } from "node:test";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";
import viteConfig from "../vite.config.ts";

const server = await createServer({
  root: new URL("..", import.meta.url).pathname,
  configFile: false,
  logLevel: "silent",
  server: { middlewareMode: true, hmr: false },
  optimizeDeps: { noDiscovery: true },
});
const {
  PRELOAD_RECOVERY_KEY,
  PRELOAD_RECOVERY_WINDOW_MS,
  recoverFromPreloadError,
  RootErrorBoundary,
} = await server.ssrLoadModule("/src/rootRecovery.tsx");

after(() => server.close());

test("ordinary source builds preserve assets for already-open windows", () => {
  assert.equal(viteConfig.build.emptyOutDir, false);
});

function storageWith(value = null) {
  const values = new Map(value === null ? [] : [[PRELOAD_RECOVERY_KEY, String(value)]]);
  return {
    getItem(key) {
      return values.get(key) ?? null;
    },
    setItem(key, next) {
      values.set(key, next);
    },
  };
}

test("a missing lazy chunk reloads the document once", () => {
  const event = new Event("vite:preloadError", { cancelable: true });
  let reloads = 0;

  assert.equal(
    recoverFromPreloadError(event, storageWith(), () => (reloads += 1), 10_000),
    true,
  );
  assert.equal(event.defaultPrevented, true);
  assert.equal(reloads, 1);
});

test("a repeated missing chunk reaches the visible error boundary instead of reloading forever", () => {
  const event = new Event("vite:preloadError", { cancelable: true });
  let reloads = 0;

  assert.equal(
    recoverFromPreloadError(
      event,
      storageWith(10_000),
      () => (reloads += 1),
      10_000 + PRELOAD_RECOVERY_WINDOW_MS - 1,
    ),
    false,
  );
  assert.equal(event.defaultPrevented, false);
  assert.equal(reloads, 0);

  const boundary = new RootErrorBoundary({ children: React.createElement("span", {}, "ready") });
  boundary.state = { error: new Error("chunk missing") };
  const html = renderToStaticMarkup(boundary.render());
  assert.match(html, /role="alert"/);
  assert.match(html, /RCP needs to reload/);
  assert.match(html, />Reload RCP</);
});
