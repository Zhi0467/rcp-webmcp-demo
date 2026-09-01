import type {
  ProviderId,
  ProviderSkill,
  ProviderSkillInventory,
  SkillCatalogEntry,
  SkillDefaults,
  SkillKind,
} from "./types";

/** The slash trigger word being typed at the end of the composer. */
const TRIGGER = /(?:^|\s)\/([A-Za-z0-9_:-]*)$/;

export const EMPTY_SKILL_SELECTION: SkillDefaults = { workflow_ids: [], skill_ids: [] };

export interface OfficialSkillPickerEntry extends SkillCatalogEntry {
  source: "rcp";
  group: "RCP Official Workflows" | "RCP Official Skills";
}

export interface ProviderSkillPickerEntry extends ProviderSkill {
  source: "provider";
  provider: ProviderId;
  machine: string;
  group: string;
  stale: boolean;
  diagnostic?: string | null;
}

export type SkillPickerEntry = OfficialSkillPickerEntry | ProviderSkillPickerEntry;

export interface ProviderSkillTarget {
  provider: ProviderId;
  providerLabel: string;
  machine: string;
  inventory?: ProviderSkillInventory | null;
}

/** The generic slash token shown in the composer for a selected entry. */
export function skillSlashToken(entry: SkillPickerEntry): string {
  return `/${entry.source === "rcp" ? entry.id : entry.name}`;
}

/** Replace the active trailing trigger with the selected entry's exact token. */
export function completeSkillTrigger(message: string, entry: SkillPickerEntry): string {
  const match = message.match(TRIGGER);
  if (!match || match.index === undefined) return message;
  const boundaryLength = match[0].length - match[1].length - 1;
  const triggerStart = match.index + boundaryLength;
  return `${message.slice(0, triggerStart)}${skillSlashToken(entry)} `;
}

export function isSkillPickerChooseKey(
  key: string,
  shiftKey: boolean,
  altKey: boolean,
  ctrlKey: boolean,
  metaKey: boolean,
): boolean {
  return (
    (key === "Enter" && !shiftKey) ||
    (key === "Tab" && !shiftKey && !altKey && !ctrlKey && !metaKey)
  );
}

function skillSelectionKey(kind: SkillKind): keyof SkillDefaults {
  return kind === "workflow" ? "workflow_ids" : "skill_ids";
}

/** The trigger query the composer text ends with, or null when none is open. */
export function readSkillTrigger(message: string): string | null {
  const match = message.match(TRIGGER);
  return match ? match[1] : null;
}

export function filterSkillCatalog(
  catalog: SkillCatalogEntry[],
  query: string,
): SkillCatalogEntry[] {
  const needle = query.trim().toLowerCase();
  if (!needle) return catalog;
  return catalog.filter(
    (item) =>
      item.id.includes(needle) ||
      item.label.toLowerCase().includes(needle) ||
      item.kind.includes(needle),
  );
}

/** Build the ordered slash menu without mixing provider-native and RCP identities. */
export function buildSkillPickerEntries(
  catalog: SkillCatalogEntry[],
  defaults: SkillDefaults,
  target: ProviderSkillTarget,
): SkillPickerEntry[] {
  const enabledOfficial = filterSkillCatalogToDefaults(catalog, defaults);
  const workflows: OfficialSkillPickerEntry[] = enabledOfficial
    .filter((item) => item.kind === "workflow")
    .map((item) => ({ ...item, source: "rcp", group: "RCP Official Workflows" }));
  const skills: OfficialSkillPickerEntry[] = enabledOfficial
    .filter((item) => item.kind === "skill")
    .map((item) => ({ ...item, source: "rcp", group: "RCP Official Skills" }));
  const native: ProviderSkillPickerEntry[] = (target.inventory?.skills ?? [])
    .filter((item) => item.enabled)
    .map((item) => ({
      ...item,
      source: "provider",
      provider: target.provider,
      machine: target.machine,
      group: `${target.providerLabel} Skills · ${target.machine}`,
      stale: target.inventory?.status === "stale",
      diagnostic: target.inventory?.diagnostic,
    }));
  return [...workflows, ...skills, ...native];
}

export function filterSkillPickerEntries(
  entries: SkillPickerEntry[],
  query: string,
): SkillPickerEntry[] {
  const needle = query.trim().toLowerCase();
  if (!needle) return entries;
  return entries.filter((item) => {
    const identity =
      item.source === "rcp" ? `${item.id} ${item.kind}` : `${item.name} ${item.provider}`;
    return [identity, item.label, item.description, item.group].some((value) =>
      value.toLowerCase().includes(needle),
    );
  });
}

export function addProviderSkillSelection(
  selection: string[],
  entry: ProviderSkillPickerEntry,
): string[] {
  return selection.includes(entry.name) ? selection : [...selection, entry.name];
}

export function skillInvocationFields(selection: SkillDefaults, providerSkillNames: string[]) {
  return {
    invoked_workflow_ids: selection.workflow_ids,
    invoked_skill_ids: selection.skill_ids,
    invoked_provider_skill_names: providerSkillNames,
  };
}

/** Slash commands may invoke only packages explicitly enabled in Settings. */
export function filterSkillCatalogToDefaults(
  catalog: SkillCatalogEntry[],
  defaults: SkillDefaults,
): SkillCatalogEntry[] {
  const allowed = new Set([
    ...defaults.workflow_ids.map((id) => `workflow:${id}`),
    ...defaults.skill_ids.map((id) => `skill:${id}`),
  ]);
  return catalog.filter((item) => allowed.has(`${item.kind}:${item.id}`));
}

/** Wrap the highlight so arrowing past either end returns to the other. */
export function moveSkillHighlight(index: number, count: number, delta: number): number {
  if (count <= 0) return 0;
  return (((index + delta) % count) + count) % count;
}

export function addSkillSelection(
  selection: SkillDefaults,
  entry: SkillCatalogEntry,
): SkillDefaults {
  const key = skillSelectionKey(entry.kind);
  if (selection[key].includes(entry.id)) return selection;
  return { ...selection, [key]: [...selection[key], entry.id] };
}

export function removeSkillSelection(
  selection: SkillDefaults,
  kind: SkillKind,
  id: string,
): SkillDefaults {
  const key = skillSelectionKey(kind);
  return { ...selection, [key]: selection[key].filter((item) => item !== id) };
}

export function selectedSkillRefs(selection: SkillDefaults): [SkillKind, string][] {
  return [
    ...selection.workflow_ids.map((id) => ["workflow", id] as [SkillKind, string]),
    ...selection.skill_ids.map((id) => ["skill", id] as [SkillKind, string]),
  ];
}

export function hasSkillSelection(selection: SkillDefaults): boolean {
  return selection.workflow_ids.length > 0 || selection.skill_ids.length > 0;
}
