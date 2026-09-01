import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { after, test } from "node:test";
import { createServer } from "vite";

const server = await createServer({
  root: new URL("..", import.meta.url).pathname,
  configFile: false,
  logLevel: "silent",
  server: { middlewareMode: true, hmr: false },
  optimizeDeps: { noDiscovery: true },
});
const { artifactContextDraft, parseArtifactContextPayload } = await server.ssrLoadModule(
  "/src/components/NodeChat.tsx",
);
const nodeChatSource = await readFile(
  new URL("../src/components/NodeChat.tsx", import.meta.url),
  "utf8",
);
const appSource = await readFile(new URL("../src/App.tsx", import.meta.url), "utf8");

after(() => server.close());

const payload = {
  type: "rcp-artifact-context",
  version: 1,
  project_id: "project",
  chat_id: "chat",
  operation_id: "operation",
  artifact_id: "0123456789abcdef01234567",
  artifact_name: "curves.html",
  media_type: "text/html",
  source: "task",
  episode_id: null,
  selections: [
    {
      kind: "text",
      text: "the final spike",
      surrounding_text: "loss rises around the final spike",
      comment: "Why does this happen?",
    },
    {
      kind: "box",
      rect: { x: 0.5, y: 0.2, width: 0.25, height: 0.3 },
      viewport: { width: 1200, height: 800 },
      labels: "seed three",
      comment: "Compare this with seed one.",
    },
  ],
};

test("artifact selections decode as bounded context for exactly one originating chat", () => {
  assert.deepEqual(parseArtifactContextPayload(payload), payload);
  assert.equal(parseArtifactContextPayload({ ...payload, artifact_id: "bad" }), null);
  assert.equal(parseArtifactContextPayload({ ...payload, selections: [] }), null);
  assert.equal(
    parseArtifactContextPayload({
      ...payload,
      selections: [{ ...payload.selections[0], comment: "x".repeat(2049) }],
    }),
    null,
  );
  assert.equal(
    parseArtifactContextPayload({
      ...payload,
      selections: [
        {
          ...payload.selections[1],
          rect: { x: 0.9, y: 0.2, width: 0.25, height: 0.3 },
        },
      ],
    }),
    null,
  );
  assert.equal(
    parseArtifactContextPayload({ ...payload, source: "episode_report", episode_id: null }),
    null,
  );
});

test("artifact selection comments assemble into a visible annotation-style draft", () => {
  const draft = artifactContextDraft(payload);
  assert.match(draft, /Selected text: the final spike/);
  assert.match(draft, /Why does this happen\?/);
  assert.match(draft, /Boxed region: seed three/);
  assert.match(draft, /Compare this with seed one\./);
  assert.match(draft, /:rcp-artifact-selection\{index="1"\}/);
  assert.match(draft, /:rcp-artifact-selection\{index="2"\}/);
});

test("the unified artifact handoff does not switch mode or dispatch automatically", () => {
  const handoff = nodeChatSource.slice(
    nodeChatSource.indexOf("const accept = (raw: unknown)"),
    nodeChatSource.indexOf("const stored = readStorage(artifactContextKey)"),
  );
  assert.match(handoff, /setArtifactContext/);
  assert.match(handoff, /setMessage/);
  assert.doesNotMatch(handoff, /selectMode\("work"\)/);
  assert.doesNotMatch(handoff, /onStartTask|send\(/);
  assert.match(nodeChatSource, /artifact_context: artifactContext/);
});

test("Experiment run conversations no longer receive special result-view props", () => {
  const selectedConversation = appSource.slice(
    appSource.indexOf("const selectedExperimentConversation"),
    appSource.indexOf("return (", appSource.indexOf("const selectedExperimentConversation")),
  );
  assert.doesNotMatch(selectedConversation, /resultViews|onKeepResultView/);
  assert.match(nodeChatSource, /artifact\.artifact_id, "viewer"/);
  assert.match(nodeChatSource, /artifact\.media_type !== "text\/html"/);
});

test("artifact cards consume backend decisions and do not preflight disabled routes", () => {
  const artifactActions = nodeChatSource.slice(
    nodeChatSource.indexOf("const openArtifact = async"),
    nodeChatSource.indexOf("const openRepositoryFile = async"),
  );
  assert.match(artifactActions, /if \(!artifact\.can_open\) return/);
  assert.match(artifactActions, /if \(!artifact\.can_download\) return/);
  assert.doesNotMatch(artifactActions, /method: "HEAD"|resourceIsAvailable/);
  assert.match(nodeChatSource, /!artifact\.available && artifact\.unavailable_reason/);
  assert.match(nodeChatSource, /artifact\.can_open &&/);
  assert.match(nodeChatSource, /artifact\.can_download &&/);
  assert.match(nodeChatSource, /!sourceArtifact\.can_revise/);
});
