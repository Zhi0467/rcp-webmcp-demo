import assert from "node:assert/strict";
import test from "node:test";

import {
  activeExperimentGuidanceText,
  currentExperimentGuidance,
  experimentGuidanceDetail,
} from "../src/experimentGuidance.ts";

test("active Experiment guidance never returns text marked stale", () => {
  const node = {
    current_summary: "This summary predates the resolved gate.",
    current_summary_stale: true,
    next_action: "Wait for input that is no longer required.",
    next_action_stale: true,
  };

  assert.equal(currentExperimentGuidance(node, "current_summary"), null);
  assert.equal(currentExperimentGuidance(node, "next_action"), null);
  assert.equal(activeExperimentGuidanceText(node), null);
});

test("active list guidance prefers a current next action and falls back to a current summary", () => {
  assert.equal(
    activeExperimentGuidanceText({
      current_summary: "Current summary",
      next_action: "Current next action",
    }),
    "Current next action",
  );
  assert.equal(
    activeExperimentGuidanceText({
      current_summary: "Current summary",
      next_action: "Stale next action",
      next_action_stale: true,
    }),
    "Current summary",
  );
});

test("detail guidance retains stale authored text with an explicit stale label", () => {
  assert.deepEqual(
    experimentGuidanceDetail(
      { current_summary: "Historical summary", current_summary_stale: true },
      "current_summary",
    ),
    {
      field: "current_summary",
      status: "stale",
      text: "Historical summary",
      label: "Previous research summary (stale)",
    },
  );
  assert.deepEqual(experimentGuidanceDetail({ next_action: "  " }, "next_action"), {
    field: "next_action",
    status: "empty",
    text: null,
    label: "Next action",
  });
});
