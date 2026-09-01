import assert from "node:assert/strict";
import test from "node:test";

import { changeTextScale, normalizeTextScale, textScaleShortcut } from "../src/textScale.ts";

test("text scale is stepped and bounded", () => {
  assert.equal(normalizeTextScale(null), 100);
  assert.equal(normalizeTextScale(""), 100);
  assert.equal(normalizeTextScale("bad"), 100);
  assert.equal(normalizeTextScale(73), 80);
  assert.equal(normalizeTextScale(149), 140);
  assert.equal(changeTextScale(140, "increase"), 140);
  assert.equal(changeTextScale(80, "decrease"), 80);
  assert.equal(changeTextScale(130, "reset"), 100);
});

test("desktop shortcuts map command plus, minus, and zero exactly once", () => {
  const key = (value, extra = {}) => ({
    key: value,
    metaKey: true,
    altKey: false,
    ctrlKey: false,
    shiftKey: false,
    ...extra,
  });
  assert.equal(textScaleShortcut(key("+")), "increase");
  assert.equal(textScaleShortcut(key("=")), "increase");
  assert.equal(textScaleShortcut(key("-")), "decrease");
  assert.equal(textScaleShortcut(key("0")), "reset");
  assert.equal(textScaleShortcut(key("+", { metaKey: false, ctrlKey: true })), null);
});
