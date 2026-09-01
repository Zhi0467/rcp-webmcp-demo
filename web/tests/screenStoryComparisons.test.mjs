import assert from "node:assert/strict";
import test from "node:test";

import {
  SCREEN_STORY_COMPARISONS,
  estimatedScreenplayTokens,
  pickScreenStoryComparison,
  projectUsageTokens,
  screenStoryComparisonCopy,
} from "../src/screenStoryComparisons.ts";

test("comparison ledger contains only complete series and film IP entries", () => {
  assert.deepEqual(
    SCREEN_STORY_COMPARISONS.map(({ id }) => id),
    [
      "before-trilogy",
      "back-to-the-future-trilogy",
      "lord-of-the-rings-trilogy",
      "stranger-things",
      "breaking-bad",
      "better-call-saul",
      "big-bang-theory",
      "the-simpsons",
    ],
  );
  assert.equal(new Set(SCREEN_STORY_COMPARISONS.map(({ id }) => id)).size, 8);
  for (const comparison of SCREEN_STORY_COMPARISONS) {
    assert.match(comparison.kind, /^(film_ip|series)$/);
    assert.doesNotMatch(comparison.label, /season|episode|\bS\d/i);
    assert.ok(comparison.estimatedScriptWords > 0);
    assert.ok(comparison.sources.length > 0);
    assert.ok(comparison.basis.length > 0);
  }
});

test("estimated screenplay tokens use one English token conversion", () => {
  const rings = SCREEN_STORY_COMPARISONS.find(({ id }) => id === "lord-of-the-rings-trilogy");
  assert.ok(rings);
  assert.equal(estimatedScreenplayTokens(rings), 131_621);
});

test("project usage adds processed input and generated output without adding cache again", () => {
  assert.equal(projectUsageTokens(900_000, 100_000), 1_000_000);
  assert.equal(projectUsageTokens(-1, 100), 100);
});

test("comparison selection is bounded and injectable", () => {
  assert.equal(pickScreenStoryComparison(() => 0).id, "before-trilogy");
  assert.equal(pickScreenStoryComparison(() => 0.999999).id, "the-simpsons");
  assert.equal(pickScreenStoryComparison(() => 1).id, "the-simpsons");
});

test("comparison copy uses percent, decimal, and whole-number ratios", () => {
  const story = {
    id: "test",
    label: "Test Story",
    kind: "series",
    estimatedScriptWords: 750,
    confidence: "measured",
    basis: "test",
    sources: ["https://example.com"],
  };
  assert.equal(screenStoryComparisonCopy(0, story), null);
  assert.equal(
    screenStoryComparisonCopy(630, story),
    "This project has used about 63% as many tokens as the scripts for Test Story.",
  );
  assert.equal(
    screenStoryComparisonCopy(7_400, story),
    "This project has used about 7.4× as many tokens as the scripts for Test Story.",
  );
  assert.equal(
    screenStoryComparisonCopy(10_000, story),
    "This project has used about 10.0× as many tokens as the scripts for Test Story.",
  );
  assert.equal(
    screenStoryComparisonCopy(12_600, story),
    "This project has used about 13× as many tokens as the scripts for Test Story.",
  );
});
