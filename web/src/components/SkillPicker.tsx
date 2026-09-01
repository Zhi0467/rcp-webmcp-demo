import { useEffect, useState } from "react";
import {
  EMPTY_SKILL_SELECTION,
  addProviderSkillSelection,
  addSkillSelection,
  buildSkillPickerEntries,
  completeSkillTrigger,
  filterSkillPickerEntries,
  isSkillPickerChooseKey,
  moveSkillHighlight,
  readSkillTrigger,
} from "../skillPicker";
import type { ProviderSkillPickerEntry, SkillPickerEntry } from "../skillPicker";
import type {
  ProviderId,
  ProviderSkillInventory,
  SkillCatalogEntry,
  SkillDefaults,
} from "../types";

interface Options {
  catalog: SkillCatalogEntry[];
  defaults: SkillDefaults;
  provider: ProviderId;
  providerLabel: string;
  machine: string;
  inventory?: ProviderSkillInventory | null;
  message: string;
  onComplete: (message: string) => void;
}

/**
 * The `/`-dropdown controller shared by chat and paper coaching.
 *
 * `handleKeyDown` belongs to the composer's own textarea and must run before
 * its send shortcut, so an open dropdown claims the arrows, Enter, Tab, and
 * Escape instead of sending the turn.
 */
export function useSkillPicker({
  catalog,
  defaults,
  provider,
  providerLabel,
  machine,
  inventory,
  message,
  onComplete,
}: Options) {
  const [selection, setSelection] = useState<SkillDefaults>(EMPTY_SKILL_SELECTION);
  const [providerSkillNames, setProviderSkillNames] = useState<string[]>([]);
  const [query, setQuery] = useState<string | null>(null);
  const [highlight, setHighlight] = useState(0);
  const availableEntries = buildSkillPickerEntries(catalog, defaults, {
    provider,
    providerLabel,
    machine,
    inventory,
  });
  const entries = query === null ? [] : filterSkillPickerEntries(availableEntries, query);
  const loading = Boolean(query !== null && inventory?.status === "refreshing");
  const open = query !== null && (entries.length > 0 || loading);
  const defaultsKey = `${defaults.workflow_ids.join(",")}|${defaults.skill_ids.join(",")}`;
  const targetKey = `${provider}:${machine}`;

  useEffect(() => {
    setHighlight(0);
  }, [query]);

  useEffect(() => {
    setSelection(EMPTY_SKILL_SELECTION);
    setProviderSkillNames([]);
    setQuery(null);
  }, [defaultsKey]);

  useEffect(() => {
    setProviderSkillNames([]);
    setQuery(null);
  }, [targetKey]);

  const close = () => setQuery(null);

  const reset = () => {
    setSelection(EMPTY_SKILL_SELECTION);
    setProviderSkillNames([]);
    close();
  };

  /** Track the trigger word as the composer text changes. */
  const readMessage = (next: string) => setQuery(readSkillTrigger(next));

  const choose = (entry: SkillPickerEntry) => {
    onComplete(completeSkillTrigger(message, entry));
    if (entry.source === "rcp") {
      setSelection((current) => addSkillSelection(current, entry));
    } else {
      setProviderSkillNames((current) => addProviderSkillSelection(current, entry));
    }
    close();
  };

  const handleKeyDown = (event: React.KeyboardEvent): boolean => {
    if (!open) return false;
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      setHighlight((current) =>
        moveSkillHighlight(current, entries.length, event.key === "ArrowDown" ? 1 : -1),
      );
      return true;
    }
    if (
      isSkillPickerChooseKey(event.key, event.shiftKey, event.altKey, event.ctrlKey, event.metaKey)
    ) {
      event.preventDefault();
      const entry = entries[Math.min(highlight, entries.length - 1)];
      if (entry) choose(entry);
      return true;
    }
    if (event.key === "Escape") {
      event.preventDefault();
      close();
      return true;
    }
    return false;
  };

  return {
    selection,
    providerSkillNames,
    setSelection,
    reset,
    readMessage,
    handleKeyDown,
    props: {
      catalog,
      selection,
      entries,
      open,
      loading,
      highlight: Math.min(highlight, Math.max(entries.length - 1, 0)),
      onHighlight: setHighlight,
      onChoose: choose,
    },
  };
}

export type SkillPickerProps = ReturnType<typeof useSkillPicker>["props"];

function skillEntryKey(entry: SkillPickerEntry): string {
  return entry.source === "rcp"
    ? `rcp:${entry.kind}:${entry.id}`
    : `provider:${entry.provider}:${entry.machine}:${entry.name}`;
}

function entryDetail(entry: SkillPickerEntry): string {
  if (entry.source === "rcp") return entry.description;
  return entry.stale ? `stale · ${entry.description}` : entry.description;
}

function staleInventory(entries: SkillPickerEntry[]): ProviderSkillPickerEntry | undefined {
  return entries.find(
    (entry): entry is ProviderSkillPickerEntry => entry.source === "provider" && entry.stale,
  );
}

export function SkillPicker({
  entries,
  open,
  loading,
  highlight,
  onHighlight,
  onChoose,
}: SkillPickerProps) {
  const stale = staleInventory(entries);
  let previousGroup: string | null = null;
  let optionIndex = 0;
  return (
    <>
      {open && (
        <div className="chat-skill-menu" role="listbox" aria-label="Select a skill or workflow">
          {entries.map((item) => {
            const showGroup = item.group !== previousGroup;
            previousGroup = item.group;
            const index = optionIndex++;
            return (
              <div className="chat-skill-menu-item" role="presentation" key={skillEntryKey(item)}>
                {showGroup && <div className="chat-skill-group-label">{item.group}</div>}
                <button
                  type="button"
                  role="option"
                  aria-selected={index === highlight}
                  className={index === highlight ? "highlighted" : undefined}
                  onMouseDown={(event) => event.preventDefault()}
                  onMouseEnter={() => onHighlight(index)}
                  onClick={() => onChoose(item)}
                >
                  <span>
                    <strong>{item.label}</strong>
                    <small>{entryDetail(item)}</small>
                  </span>
                </button>
              </div>
            );
          })}
          {loading && (
            <div className="chat-skill-menu-status" role="status">
              Checking provider skills…
            </div>
          )}
          {stale?.diagnostic && (
            <div className="chat-skill-menu-status stale" role="status">
              Last refresh failed: {stale.diagnostic}
            </div>
          )}
        </div>
      )}
    </>
  );
}
