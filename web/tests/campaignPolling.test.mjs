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
const { LIVE_EPISODE_POLL_INTERVAL_MS, startLiveEpisodePolling } =
  await server.ssrLoadModule("/src/App.tsx");
const { episodePollingTarget } = await server.ssrLoadModule("/src/hooks/useEpisodeDialogs.ts");

after(() => server.close());

test("episode polling follows active work and an ended episode's running branch merge", () => {
  const live = { episode_id: "live", status: "running", live: true, graph_branch: null };
  const merging = {
    episode_id: "merging",
    status: "completed",
    live: false,
    graph_branch: { merge_state: "running" },
  };
  const merged = {
    ...merging,
    graph_branch: { merge_state: "merged" },
  };

  assert.equal(episodePollingTarget([merging])?.episode_id, "merging");
  assert.equal(episodePollingTarget([merging, live])?.episode_id, "live");
  assert.equal(episodePollingTarget([merged]), null);
});

test("live episode polling is single-flight and keeps failures visible until recovery", async () => {
  const timers = new Map();
  const cleared = [];
  let nextTimerId = 1;
  const clock = {
    setTimeout(callback, delay) {
      const timerId = nextTimerId++;
      timers.set(timerId, { callback, delay });
      return timerId;
    },
    clearTimeout(timerId) {
      cleared.push(timerId);
      timers.delete(timerId);
    },
  };
  const nextTimer = () => {
    assert.equal(timers.size, 1);
    const [timerId, timer] = timers.entries().next().value;
    timers.delete(timerId);
    return timer;
  };
  const settle = () => new Promise((resolve) => setImmediate(resolve));

  let firstRefreshDone;
  let refreshCount = 0;
  let visibleError = null;
  const stop = startLiveEpisodePolling(
    clock,
    () => {
      refreshCount += 1;
      if (refreshCount === 1) {
        return new Promise((resolve) => {
          firstRefreshDone = resolve;
        });
      }
      if (refreshCount === 2) return Promise.reject(new Error("backend unavailable"));
      return Promise.resolve();
    },
    (error) => {
      visibleError = error.message;
    },
    () => {
      visibleError = null;
    },
  );

  const firstTimer = nextTimer();
  assert.equal(firstTimer.delay, LIVE_EPISODE_POLL_INTERVAL_MS);
  firstTimer.callback();
  await settle();
  assert.equal(refreshCount, 1);
  assert.equal(timers.size, 0, "an unresolved refresh must not schedule an overlapping poll");

  firstRefreshDone();
  await settle();
  assert.equal(visibleError, null);

  nextTimer().callback();
  await settle();
  assert.equal(refreshCount, 2);
  assert.equal(visibleError, "backend unavailable");

  await settle();
  assert.equal(visibleError, "backend unavailable", "the error remains visible between polls");
  nextTimer().callback();
  await settle();
  assert.equal(refreshCount, 3);
  assert.equal(visibleError, null, "a successful episode refresh clears the failure");

  const scheduledTimerId = timers.keys().next().value;
  stop();
  assert.deepEqual(cleared, [scheduledTimerId]);
  assert.equal(timers.size, 0);
});
