import assert from "node:assert/strict";
import test from "node:test";

import {
  CHAT_LIST_MAX_WIDTH,
  CHAT_LIST_MIN_WIDTH,
  CHAT_LIST_MIN_WIDTH_COMPACT,
  chatListWidthBounds,
  clampChatListWidth,
  isChatListToggleShortcut,
} from "../src/chatLayout.ts";

test("chat list width keeps a full chat surface at desktop sizes", () => {
  assert.deepEqual(chatListWidthBounds(1200), {
    minimum: CHAT_LIST_MIN_WIDTH,
    maximum: CHAT_LIST_MAX_WIDTH,
  });
  assert.deepEqual(chatListWidthBounds(780), {
    minimum: CHAT_LIST_MIN_WIDTH,
    maximum: 420,
  });
});

test("chat list width uses a compact minimum on narrow screens", () => {
  assert.deepEqual(chatListWidthBounds(390), {
    minimum: CHAT_LIST_MIN_WIDTH_COMPACT,
    maximum: CHAT_LIST_MIN_WIDTH_COMPACT,
  });
});

test("chat list width clamps pointer and keyboard adjustments to its bounds", () => {
  const bounds = { minimum: 190, maximum: 420 };
  assert.equal(clampChatListWidth(80, bounds), 190);
  assert.equal(clampChatListWidth(300, bounds), 300);
  assert.equal(clampChatListWidth(600, bounds), 420);
});

test("command B toggles the chat list exactly once", () => {
  const key = (value, extra = {}) => ({
    key: value,
    metaKey: true,
    altKey: false,
    ctrlKey: false,
    shiftKey: false,
    repeat: false,
    ...extra,
  });
  assert.equal(isChatListToggleShortcut(key("b")), true);
  assert.equal(isChatListToggleShortcut(key("B")), true);
  assert.equal(isChatListToggleShortcut(key("b", { repeat: true })), false);
  assert.equal(isChatListToggleShortcut(key("b", { metaKey: false, ctrlKey: true })), false);
  assert.equal(isChatListToggleShortcut(key("b", { shiftKey: true })), false);
});
