export const CHAT_LIST_DEFAULT_WIDTH = 260;
export const CHAT_LIST_COLLAPSED_WIDTH = 0;
export const CHAT_LIST_DIVIDER_WIDTH = 24;
export const CHAT_LIST_MIN_WIDTH = 190;
export const CHAT_LIST_MIN_WIDTH_COMPACT = 110;
export const CHAT_LIST_MAX_WIDTH = 420;
export const CHAT_SURFACE_MIN_WIDTH = 300;

export interface ChatListWidthBounds {
  minimum: number;
  maximum: number;
}

export function chatListWidthBounds(containerWidth: number): ChatListWidthBounds {
  const minimum = containerWidth < 720 ? CHAT_LIST_MIN_WIDTH_COMPACT : CHAT_LIST_MIN_WIDTH;
  const maximum = Math.max(
    minimum,
    Math.min(CHAT_LIST_MAX_WIDTH, containerWidth - CHAT_SURFACE_MIN_WIDTH),
  );
  return { minimum, maximum };
}

export function clampChatListWidth(value: number, bounds: ChatListWidthBounds): number {
  return Math.min(bounds.maximum, Math.max(bounds.minimum, value));
}

export function isChatListToggleShortcut(
  event: Pick<KeyboardEvent, "key" | "metaKey" | "altKey" | "ctrlKey" | "shiftKey" | "repeat">,
): boolean {
  return (
    event.metaKey &&
    !event.altKey &&
    !event.ctrlKey &&
    !event.shiftKey &&
    !event.repeat &&
    event.key.toLowerCase() === "b"
  );
}
