import type { AppView } from "./types";

export interface ProjectTab {
  id: string;
  name: string;
}

export interface ProjectTabCloseResult {
  tabs: ProjectTab[];
  activeProjectId: string | null;
}

export interface ProjectViewState {
  view: AppView;
  panelScroll: Array<[AppView, number]>;
  researchSubview: AppView;
  dagViewport: { zoom: number; scrollLeft: number; scrollTop: number } | null;
}

export interface ProjectViewportRef<T> {
  current: T | null;
}

export function initialProjectHash(
  hash: string,
  navigationType: PerformanceNavigationTiming["type"] | null | undefined,
): string {
  return navigationType === "reload" ? "" : hash;
}

export function projectViewportRef<T>(
  refs: Map<string, ProjectViewportRef<T>>,
  projectId: string,
): ProjectViewportRef<T> {
  const existing = refs.get(projectId);
  if (existing) return existing;
  const created = { current: null };
  refs.set(projectId, created);
  return created;
}

export function openProjectTab(tabs: ProjectTab[], project: ProjectTab): ProjectTab[] {
  const existing = tabs.findIndex((tab) => tab.id === project.id);
  if (existing === -1) return [...tabs, project];
  if (tabs[existing].name === project.name) return tabs;
  return tabs.map((tab, index) => (index === existing ? project : tab));
}

export function closeProjectTab(
  tabs: ProjectTab[],
  activeProjectId: string | null,
  projectId: string,
): ProjectTabCloseResult {
  const closingIndex = tabs.findIndex((tab) => tab.id === projectId);
  if (closingIndex === -1) return { tabs, activeProjectId };
  const nextTabs = tabs.filter((tab) => tab.id !== projectId);
  if (activeProjectId !== projectId) return { tabs: nextTabs, activeProjectId };
  return {
    tabs: nextTabs,
    activeProjectId: tabs[closingIndex + 1]?.id ?? tabs[closingIndex - 1]?.id ?? null,
  };
}

export function adjacentProjectTabId(
  tabs: ProjectTab[],
  activeProjectId: string | null,
  direction: -1 | 1,
): string | null {
  if (tabs.length === 0) return null;
  const activeIndex = tabs.findIndex((tab) => tab.id === activeProjectId);
  if (activeIndex === -1) return direction === 1 ? tabs[0].id : tabs[tabs.length - 1].id;
  return tabs[(activeIndex + direction + tabs.length) % tabs.length].id;
}

export function projectTabShortcut(
  event: Pick<KeyboardEvent, "key" | "metaKey" | "altKey" | "ctrlKey" | "shiftKey">,
  editable: boolean,
): "index" | "previous" | "next" | null {
  if (!event.metaKey || event.ctrlKey || event.shiftKey) return null;
  if (!event.altKey && event.key.toLowerCase() === "t") return "index";
  if (editable || !event.altKey) return null;
  if (event.key === "ArrowLeft") return "previous";
  if (event.key === "ArrowRight") return "next";
  return null;
}

export function isEditableShortcutTarget(target: EventTarget | null): boolean {
  if (!(target instanceof Element)) return false;
  return Boolean(
    target.closest("input, textarea, select, [contenteditable]:not([contenteditable='false'])"),
  );
}
