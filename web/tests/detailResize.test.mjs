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
const { DraggableWindow } = await server.ssrLoadModule("/src/components/DraggableWindow.tsx");
const { DetailDrawer } = await server.ssrLoadModule("/src/components/DetailDrawer.tsx");

after(() => server.close());

test("node detail and chat get four pointer-only resize corners", () => {
  const previousWindow = globalThis.window;
  globalThis.window = {
    innerWidth: 1000,
    innerHeight: 800,
    localStorage: {
      getItem(key) {
        return key === "detail-size" ? '{"width":780,"height":650}' : null;
      },
      setItem() {},
    },
  };
  try {
    const node = {
      id: "hyp/detail",
      type: "hypothesis",
      title: "Resizable detail",
      statement: "The detail can be resized.",
      standing: "asserted",
      created_rev: 1,
      updated_rev: 1,
      source_refs: [],
      extension_fields: {},
    };
    const detail = renderToStaticMarkup(
      React.createElement(DetailDrawer, {
        node,
        edges: [],
        allNodes: { [node.id]: node },
        glossary: {},
        beliefTransitions: [],
        validationMessages: [],
        ontology: { types: [], fields: [], relations: [] },
        sizeStorageKey: "detail-size",
        onClose() {},
        onDock() {},
        onBeginEdit() {},
        onStanding() {},
        onStage() {},
        onOpenChat() {},
        onExploreRelations() {},
        onSelectNode() {},
      }),
    );
    const chat = renderToStaticMarkup(
      React.createElement(
        DraggableWindow,
        { className: "node-chat-window", kind: "chat", resizable: true },
        React.createElement("aside", null, "Chat"),
      ),
    );

    assert.match(detail, /width:780px;height:650px/);
    for (const markup of [detail, chat]) {
      assert.equal(markup.match(/data-resize-corner=/g)?.length, 4);
      for (const corner of ["top-left", "top-right", "bottom-left", "bottom-right"]) {
        assert.match(
          markup,
          new RegExp(
            `<div class="floating-window-resize-corner ${corner}" data-resize-corner="${corner}"`,
          ),
        );
      }
      assert.doesNotMatch(
        markup,
        /floating-window-resize-handle|aria-keyshortcuts|Resize node detail window/,
      );
    }
  } finally {
    if (previousWindow === undefined) delete globalThis.window;
    else globalThis.window = previousWindow;
  }
});
