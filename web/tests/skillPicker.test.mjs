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
  addProviderSkillSelection,
  addSkillSelection,
  buildSkillPickerEntries,
  completeSkillTrigger,
  filterSkillCatalog,
  filterSkillCatalogToDefaults,
  filterSkillPickerEntries,
  hasSkillSelection,
  isSkillPickerChooseKey,
  moveSkillHighlight,
  readSkillTrigger,
  removeSkillSelection,
  selectedSkillRefs,
  skillInvocationFields,
} = await server.ssrLoadModule("/src/skillPicker.ts");
const { SkillPicker } = await server.ssrLoadModule("/src/components/SkillPicker.tsx");

after(() => server.close());

const catalog = [
  {
    id: "research-graph-audit",
    kind: "workflow",
    label: "Research graph audit",
    version: "1.0.0",
    description: "Review graph structure.",
    dependencies: [],
  },
  {
    id: "graph-audit",
    kind: "skill",
    label: "Graph audit",
    version: "1.0.0",
    description: "Inspect the project graph.",
    dependencies: [],
  },
];
const EMPTY_DEFAULTS = { workflow_ids: [], skill_ids: [] };

const codexInventory = {
  provider: "codex",
  machine: "local",
  host: "",
  provider_version: "0.146.1",
  inventory_hash: "codex-hash",
  command: ["codex", "app-server"],
  protocol: "jsonrpc",
  status: "fresh",
  stale: false,
  skills: [
    {
      name: "frontend-design:frontend-design",
      label: "Frontend design",
      description: "Shape a distinctive interface.",
      scope: "user",
      path: "/skills/frontend-design/SKILL.md",
      enabled: true,
    },
    {
      name: "disabled-skill",
      label: "Disabled skill",
      description: "Do not offer this.",
      enabled: false,
    },
  ],
};

test("a trigger opens only at the end of a word boundary", () => {
  assert.equal(readSkillTrigger("/"), "");
  assert.equal(readSkillTrigger("$gra"), null);
  assert.equal(readSkillTrigger("check this /graph"), "graph");
  assert.equal(
    readSkillTrigger("use /frontend-design:frontend-design"),
    "frontend-design:frontend-design",
  );
  assert.equal(readSkillTrigger("look at src/rcp/runs"), null);
  assert.equal(readSkillTrigger("/graph then more words"), null);
  assert.equal(readSkillTrigger(""), null);
});

test("selection completes official and provider triggers without changing preceding text", () => {
  const entries = buildSkillPickerEntries(
    catalog,
    { workflow_ids: ["research-graph-audit"], skill_ids: [] },
    {
      provider: "codex",
      providerLabel: "Codex",
      machine: "local",
      inventory: codexInventory,
    },
  );

  assert.equal(completeSkillTrigger("/res", entries[0]), "/research-graph-audit ");
  assert.equal(
    completeSkillTrigger("Please use /front", entries[1]),
    "Please use /frontend-design:frontend-design ",
  );
  assert.equal(completeSkillTrigger("No active trigger", entries[0]), "No active trigger");
});

test("Enter and unmodified Tab choose an entry while Shift+Tab remains available", () => {
  assert.equal(isSkillPickerChooseKey("Enter", false, false, false, false), true);
  assert.equal(isSkillPickerChooseKey("Tab", false, false, false, false), true);
  assert.equal(isSkillPickerChooseKey("Tab", true, false, false, false), false);
  assert.equal(isSkillPickerChooseKey("Tab", false, true, false, false), false);
});

test("the menu orders RCP official groups before the selected provider and machine", () => {
  const entries = buildSkillPickerEntries(
    catalog,
    { workflow_ids: ["research-graph-audit"], skill_ids: ["graph-audit"] },
    {
      provider: "codex",
      providerLabel: "Codex",
      machine: "local",
      inventory: codexInventory,
    },
  );

  assert.deepEqual(
    entries.map((entry) => [entry.source, entry.group, entry.label]),
    [
      ["rcp", "RCP Official Workflows", "Research graph audit"],
      ["rcp", "RCP Official Skills", "Graph audit"],
      ["provider", "Codex Skills · local", "Frontend design"],
    ],
  );
  assert.deepEqual(
    filterSkillPickerEntries(entries, "frontend-design:frontend-design").map(
      (entry) => entry.label,
    ),
    ["Frontend design"],
  );
});

test("switching provider or machine replaces only the provider-native group", () => {
  const defaults = { workflow_ids: ["research-graph-audit"], skill_ids: ["graph-audit"] };
  const codex = buildSkillPickerEntries(catalog, defaults, {
    provider: "codex",
    providerLabel: "Codex",
    machine: "local",
    inventory: codexInventory,
  });
  const claude = buildSkillPickerEntries(catalog, defaults, {
    provider: "claude",
    providerLabel: "Claude",
    machine: "gpu",
    inventory: {
      ...codexInventory,
      provider: "claude",
      machine: "gpu",
      host: "gpu.example",
      skills: [
        {
          name: "review-pr",
          label: "Review PR",
          description: "Review a change.",
          enabled: true,
        },
      ],
    },
  });

  assert.deepEqual(
    codex.filter((entry) => entry.source === "rcp"),
    claude.filter((entry) => entry.source === "rcp"),
  );
  assert.deepEqual(
    claude.filter((entry) => entry.source === "provider").map((entry) => entry.group),
    ["Claude Skills · gpu"],
  );
  assert.equal(
    claude.some((entry) => "name" in entry && entry.name.includes("frontend")),
    false,
  );
});

test("stale native skills remain selectable and produce separate request metadata", () => {
  const message = "Please apply /frontend-design:frontend-design";
  const [native] = buildSkillPickerEntries(catalog, EMPTY_DEFAULTS, {
    provider: "codex",
    providerLabel: "Codex",
    machine: "local",
    inventory: {
      ...codexInventory,
      status: "stale",
      stale: true,
      diagnostic: "SSH host is unavailable",
    },
  });
  const names = addProviderSkillSelection([], native);

  assert.equal(message, "Please apply /frontend-design:frontend-design");
  assert.deepEqual(skillInvocationFields(EMPTY_DEFAULTS, names), {
    invoked_workflow_ids: [],
    invoked_skill_ids: [],
    invoked_provider_skill_names: ["frontend-design:frontend-design"],
  });

  const html = renderToStaticMarkup(
    React.createElement(SkillPicker, {
      catalog,
      selection: EMPTY_DEFAULTS,
      entries: [native],
      open: true,
      loading: false,
      highlight: 0,
      onHighlight() {},
      onChoose() {},
    }),
  );
  assert.match(html, /Codex Skills · local/);
  assert.match(html, /stale · Shape a distinctive interface/);
  assert.match(html, /Last refresh failed: SSH host is unavailable/);
});

test("refreshing inventory leaves official entries usable and shows loading state", () => {
  const entries = buildSkillPickerEntries(
    catalog,
    { workflow_ids: ["research-graph-audit"], skill_ids: [] },
    {
      provider: "codex",
      providerLabel: "Codex",
      machine: "local",
      inventory: { ...codexInventory, status: "refreshing", skills: [] },
    },
  );
  const html = renderToStaticMarkup(
    React.createElement(SkillPicker, {
      catalog,
      selection: EMPTY_DEFAULTS,
      entries,
      open: true,
      loading: true,
      highlight: 0,
      onHighlight() {},
      onChoose() {},
    }),
  );

  assert.match(html, /RCP Official Workflows/);
  assert.match(html, /role="option"/);
  assert.match(html, /Checking provider skills…/);
});

test("the dropdown filters on id, label, and kind", () => {
  assert.deepEqual(
    filterSkillCatalog(catalog, "").map((item) => item.id),
    ["research-graph-audit", "graph-audit"],
  );
  assert.deepEqual(
    filterSkillCatalog(catalog, "workflow").map((item) => item.id),
    ["research-graph-audit"],
  );
  // Label matching is case-insensitive.
  assert.deepEqual(
    filterSkillCatalog(catalog, "Graph Audit").map((item) => item.id),
    ["research-graph-audit", "graph-audit"],
  );
  assert.deepEqual(
    filterSkillCatalog(catalog, "nothing").map((item) => item.id),
    [],
  );
  assert.deepEqual(
    filterSkillCatalog(catalog, "graph-audit").map((item) => item.id),
    ["research-graph-audit", "graph-audit"],
  );
});

test("slash commands expose only packages enabled in Settings", () => {
  assert.deepEqual(
    filterSkillCatalogToDefaults(catalog, {
      workflow_ids: ["research-graph-audit"],
      skill_ids: [],
    }).map((item) => item.id),
    ["research-graph-audit"],
  );
});

test("the picker never renders persistent selection chips", () => {
  const html = renderToStaticMarkup(
    React.createElement(SkillPicker, {
      catalog,
      selection: { workflow_ids: ["research-graph-audit"], skill_ids: [] },
      entries: [],
      open: false,
      loading: false,
      highlight: 0,
      onHighlight() {},
      onChoose() {},
    }),
  );

  assert.doesNotMatch(html, /chat-skill-chip/);
});

test("arrow keys wrap around both ends of the dropdown", () => {
  assert.equal(moveSkillHighlight(0, 2, 1), 1);
  assert.equal(moveSkillHighlight(1, 2, 1), 0);
  assert.equal(moveSkillHighlight(0, 2, -1), 1);
  assert.equal(moveSkillHighlight(0, 0, 1), 0);
});

test("selection is structured by kind and never duplicates an entry", () => {
  let selection = { workflow_ids: [], skill_ids: [] };
  assert.equal(hasSkillSelection(selection), false);

  selection = addSkillSelection(selection, catalog[0]);
  selection = addSkillSelection(selection, catalog[1]);
  selection = addSkillSelection(selection, catalog[1]);

  assert.deepEqual(selection, {
    workflow_ids: ["research-graph-audit"],
    skill_ids: ["graph-audit"],
  });
  assert.deepEqual(selectedSkillRefs(selection), [
    ["workflow", "research-graph-audit"],
    ["skill", "graph-audit"],
  ]);

  selection = removeSkillSelection(selection, "workflow", "research-graph-audit");
  assert.deepEqual(selection, { workflow_ids: [], skill_ids: ["graph-audit"] });
});
