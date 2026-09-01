import assert from "node:assert/strict";
import test from "node:test";

import { replaceTextSpan } from "../src/chatInput.ts";

test("dictation inserts at the captured cursor and revises only its active span", () => {
  const first = replaceTextSpan("before  after", { start: 7, end: 7 }, "partial");
  assert.deepEqual(first, { value: "before partial after", end: 14 });

  const revised = replaceTextSpan(first.value, { start: 7, end: first.end }, "final words");
  assert.deepEqual(revised, { value: "before final words after", end: 18 });
});
