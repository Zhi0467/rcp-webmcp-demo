import type { GlossaryTerm } from "./types";

export interface GlossaryIndexEntry {
  term: string;
  normalizedTerm: string;
  plainDefinition: string;
}

export interface GlossaryIndex {
  entriesByInitial: ReadonlyMap<string, readonly GlossaryIndexEntry[]>;
}

export type GlossaryTextSegment =
  | { kind: "text"; text: string }
  | {
      kind: "definition";
      text: string;
      term: string;
      plainDefinition: string;
    };

const TERM_CONTINUATION = /[\p{L}\p{N}_/-]/u;

export function buildGlossaryIndex(
  glossary: Readonly<Record<string, GlossaryTerm>>,
): GlossaryIndex {
  const entries = Object.values(glossary)
    .filter((entry) => entry.term.trim())
    .map((entry) => ({
      term: entry.term,
      normalizedTerm: normalizeTerm(entry.term),
      plainDefinition: entry.plain_definition,
    }))
    .sort(compareEntries);
  const seen = new Set<string>();
  const entriesByInitial = new Map<string, GlossaryIndexEntry[]>();

  for (const entry of entries) {
    if (seen.has(entry.normalizedTerm)) continue;
    seen.add(entry.normalizedTerm);
    const initial = entry.normalizedTerm[0];
    if (!initial) continue;
    const bucket = entriesByInitial.get(initial) ?? [];
    bucket.push(entry);
    entriesByInitial.set(initial, bucket);
  }

  return { entriesByInitial };
}

export function segmentGlossaryText(text: string, index: GlossaryIndex): GlossaryTextSegment[] {
  if (!text || index.entriesByInitial.size === 0) {
    return text ? [{ kind: "text", text }] : [];
  }

  const segments: GlossaryTextSegment[] = [];
  let plainTextStart = 0;
  let cursor = 0;
  while (cursor < text.length) {
    const initial = normalizeTerm(text[cursor] ?? "")[0];
    const candidates = initial ? index.entriesByInitial.get(initial) : undefined;
    const match = candidates?.find((entry) => matchesAt(text, cursor, entry));
    if (!match) {
      cursor += 1;
      continue;
    }

    if (cursor > plainTextStart) {
      segments.push({ kind: "text", text: text.slice(plainTextStart, cursor) });
    }
    const end = cursor + match.term.length;
    segments.push({
      kind: "definition",
      text: text.slice(cursor, end),
      term: match.term,
      plainDefinition: match.plainDefinition,
    });
    cursor = end;
    plainTextStart = end;
  }

  if (plainTextStart < text.length) {
    segments.push({ kind: "text", text: text.slice(plainTextStart) });
  }
  return segments;
}

function matchesAt(text: string, start: number, entry: GlossaryIndexEntry): boolean {
  const end = start + entry.term.length;
  if (normalizeTerm(text.slice(start, end)) !== entry.normalizedTerm) return false;
  return !isTermContinuation(text[start - 1]) && !isTermContinuation(text[end]);
}

function isTermContinuation(value: string | undefined): boolean {
  return Boolean(value && TERM_CONTINUATION.test(value));
}

function normalizeTerm(value: string): string {
  return value.toLowerCase();
}

function compareEntries(left: GlossaryIndexEntry, right: GlossaryIndexEntry): number {
  const lengthDifference = right.term.length - left.term.length;
  if (lengthDifference) return lengthDifference;
  if (left.normalizedTerm < right.normalizedTerm) return -1;
  if (left.normalizedTerm > right.normalizedTerm) return 1;
  if (left.term < right.term) return -1;
  if (left.term > right.term) return 1;
  return 0;
}
