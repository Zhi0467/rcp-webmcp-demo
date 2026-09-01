import assert from "node:assert/strict";
import test from "node:test";

import {
  modelChange,
  modelOptions,
  modelsFor,
  providerChange,
  providerOptions,
  reasoningFor,
  reasoningOptions,
} from "../src/providers.ts";

/** Shaped like a real `codex debug models` probe: efforts differ per model. */
const CODEX = {
  provider: "codex",
  label: "Codex",
  installed: true,
  authenticated: true,
  models: [
    {
      id: "gpt-5.6-sol",
      label: "GPT-5.6-Sol",
      reasoning: ["low", "high", "ultra"],
      default_reasoning: "low",
    },
    { id: "gpt-5.5", label: "GPT-5.5", reasoning: ["low", "high"], default_reasoning: "medium" },
  ],
};
const CLAUDE = {
  provider: "claude",
  label: "Claude",
  installed: true,
  authenticated: true,
  models: [{ id: "opus", label: "Opus", reasoning: ["low", "max"], default_reasoning: "medium" }],
};

test("every provider the backend probed is offered, under its own label", () => {
  const options = providerOptions([CODEX, CLAUDE], "codex");
  assert.deepEqual(options, [
    { id: "codex", label: "Codex" },
    { id: "claude", label: "Claude" },
  ]);
});

test("reasoning narrows to the selected model rather than the provider", () => {
  assert.deepEqual(reasoningFor(CODEX.models, "gpt-5.6-sol"), ["low", "high", "ultra"]);
  assert.deepEqual(reasoningFor(CODEX.models, "gpt-5.5"), ["low", "high"]);
});

test("provider default offers every effort any of its models accepts", () => {
  assert.deepEqual(reasoningFor(CODEX.models, ""), ["low", "high", "ultra"]);
});

test("moving to a model that rejects the current effort falls back to its default", () => {
  // `ultra` is real on sol and absent on 5.5; carrying it over would be
  // rejected at the API, which is the bug this whole path exists to prevent.
  assert.deepEqual(modelChange(CODEX.models, "gpt-5.5", "ultra"), {
    model: "gpt-5.5",
    reasoning: "medium",
  });
});

test("an effort the new model still accepts is left alone", () => {
  assert.deepEqual(modelChange(CODEX.models, "gpt-5.5", "high"), { model: "gpt-5.5" });
});

test("a saved value the provider no longer offers stays selectable", () => {
  // An unreachable CLI reports no models at all. Silently dropping the saved
  // model would rewrite the manifest choice the human made.
  assert.deepEqual(modelOptions([], "gpt-5.4"), [
    { id: "", label: "Provider default" },
    { id: "gpt-5.4", label: "gpt-5.4" },
  ]);
  assert.deepEqual(reasoningOptions(CODEX.models, "gpt-5.5", "minimal").at(-1), {
    id: "minimal",
    label: "minimal",
  });
});

test("provider default is offered ahead of the catalog", () => {
  assert.deepEqual(modelOptions(CLAUDE.models, ""), [
    { id: "", label: "Provider default" },
    { id: "opus", label: "Opus" },
  ]);
});

test("an unknown provider contributes no models instead of throwing", () => {
  assert.deepEqual(modelsFor([CODEX, CLAUDE], "gemini"), []);
  assert.deepEqual(modelsFor([CODEX, CLAUDE], "claude"), CLAUDE.models);
});

test("switching provider drops the other provider's model", () => {
  // Offering `gpt-5.5` under Claude would be a value Claude rejects; the model
  // choice belongs to the provider, so it resets to the provider default.
  // `max` is shared, so only the model resets.
  assert.deepEqual(providerChange(CLAUDE.models, "claude", "max"), {
    provider: "claude",
    model: "",
  });
});

test("an effort the new provider does not share is replaced, not carried over", () => {
  assert.deepEqual(providerChange(CLAUDE.models, "claude", "ultra"), {
    provider: "claude",
    model: "",
    reasoning: "low",
  });
});
