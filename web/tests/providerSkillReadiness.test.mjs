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
const { shouldPollProviderSkillReadiness } = await server.ssrLoadModule("/src/App.tsx");

after(() => server.close());

const inventory = (status) => ({ status });

test("provider skill readiness follow-ups continue only for refreshing inventories", () => {
  assert.equal(
    shouldPollProviderSkillReadiness(
      {
        local: { codex: inventory("fresh") },
        gpu: { claude: inventory("refreshing") },
      },
      0,
    ),
    true,
  );
  assert.equal(
    shouldPollProviderSkillReadiness({ local: { codex: inventory("stale") } }, 0),
    false,
  );
  assert.equal(shouldPollProviderSkillReadiness(undefined, 0), false);
});

test("provider skill readiness follow-ups stop at the startup bound", () => {
  const inventories = { local: { codex: inventory("refreshing") } };

  assert.equal(shouldPollProviderSkillReadiness(inventories, 19), true);
  assert.equal(shouldPollProviderSkillReadiness(inventories, 20), false);
});
