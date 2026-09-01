import type { ChatSummary, ChatSummaryPage, ChatTranscript } from "./types";

export type ChatPageRequest = (path: string) => Promise<ChatSummaryPage>;

export const CHAT_SUMMARY_PAGE_SIZE = 200;

export async function loadChatSummaryPage(
  apiBase: string,
  offset: number,
  request: ChatPageRequest,
): Promise<ChatSummaryPage> {
  return request(`${apiBase}/chats?offset=${offset}&limit=${CHAT_SUMMARY_PAGE_SIZE}`);
}

export function mergeChatSummaryPage(
  current: ChatSummary[],
  page: ChatSummary[],
  placement: "append" | "refresh",
): ChatSummary[] {
  const ordered = placement === "refresh" ? page : [...current, ...page];
  const seen = new Set<string>();
  return ordered.filter((summary) => {
    if (seen.has(summary.chat_id)) return false;
    seen.add(summary.chat_id);
    return true;
  });
}

export function nextChatSummaryOffset(page: ChatSummaryPage): number {
  return page.offset + page.items.length;
}

export interface ChatSelectionReconciliation {
  selectedChatId: string | null;
  retainedSummary: ChatSummary | null;
  deleteTranscript: boolean;
}

export function reconcileChatSelectionAfterRefresh(
  selectedChatId: string | null,
  previousSummary: ChatSummary | null,
  refreshedSummaries: ChatSummary[],
  validation: ChatTranscript | null | undefined,
): ChatSelectionReconciliation {
  if (!selectedChatId) {
    return { selectedChatId: null, retainedSummary: null, deleteTranscript: false };
  }
  if (refreshedSummaries.some((summary) => summary.chat_id === selectedChatId)) {
    return { selectedChatId, retainedSummary: null, deleteTranscript: false };
  }
  if (!previousSummary) {
    return { selectedChatId, retainedSummary: null, deleteTranscript: false };
  }
  if (validation) {
    return { selectedChatId, retainedSummary: validation, deleteTranscript: false };
  }
  if (validation === null) {
    return { selectedChatId: null, retainedSummary: null, deleteTranscript: true };
  }
  return { selectedChatId, retainedSummary: previousSummary, deleteTranscript: false };
}

export async function loadChatTranscript(
  apiBase: string,
  chatId: string,
  request: (path: string) => Promise<ChatTranscript>,
): Promise<ChatTranscript> {
  return request(`${apiBase}/chats/${encodeURIComponent(chatId)}`);
}
