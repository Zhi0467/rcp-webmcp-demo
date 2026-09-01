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
const { EpisodeReportLink } = await server.ssrLoadModule("/src/components/EpisodeReportLink.tsx");

after(() => server.close());

const props = {
  projectId: "project one",
  episodeId: "episode/one",
  href: "/api/projects/project%20one/episodes/episode%2Fone/report/viewer",
  children: "Open report",
  onOpenError() {},
};

test("the report link preserves browser target behavior", () => {
  const html = renderToStaticMarkup(React.createElement(EpisodeReportLink, props));

  assert.match(
    html,
    /href="\/api\/projects\/project%20one\/episodes\/episode%2Fone\/report\/viewer"/,
  );
  assert.match(html, /target="_blank"/);
  assert.match(html, /rel="noopener noreferrer"/);
});

test("the report link claims desktop clicks and invokes the native preview command", async () => {
  const originalWindow = globalThis.window;
  let prevented = false;
  const invocations = [];
  const desktopWindow = new EventTarget();
  desktopWindow.__TAURI_INTERNALS__ = {
    invoke: async (command, args) => {
      invocations.push({ command, args });
      return { opened: true };
    },
  };
  globalThis.window = desktopWindow;
  try {
    const link = EpisodeReportLink(props);
    await link.props.onClick({ preventDefault: () => (prevented = true) });

    assert.equal(prevented, true);
    assert.deepEqual(invocations, [
      {
        command: "open_episode_report_preview",
        args: { projectId: "project one", episodeId: "episode/one" },
      },
    ]);
  } finally {
    if (originalWindow === undefined) delete globalThis.window;
    else globalThis.window = originalWindow;
  }
});

test("the report link surfaces a native preview failure", async () => {
  const originalWindow = globalThis.window;
  const errors = [];
  let prevented = false;
  const desktopWindow = new EventTarget();
  desktopWindow.__TAURI_INTERNALS__ = {
    invoke: async () => {
      throw new Error("native preview failed");
    },
  };
  globalThis.window = desktopWindow;
  try {
    const link = EpisodeReportLink({ ...props, onOpenError: (message) => errors.push(message) });
    await link.props.onClick({ preventDefault: () => (prevented = true) });

    assert.equal(prevented, true);
    assert.deepEqual(errors, ["native preview failed"]);
  } finally {
    if (originalWindow === undefined) delete globalThis.window;
    else globalThis.window = originalWindow;
  }
});
