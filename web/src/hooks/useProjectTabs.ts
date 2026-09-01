import { useCallback, useEffect, useRef, useState } from "react";
import { api, loadExperimentEpisodes } from "../api";
import { experimentBoardHref } from "../experimentBoard";
import {
  adjacentProjectTabId,
  closeProjectTab,
  openProjectTab,
  type ProjectTab,
} from "../projectTabs";
import type { ExperimentLoopIndexEntry, ProjectCard, ProjectSnapshot } from "../types";

const PROJECT_HEADER_COLLAPSED_KEY = "rcp:project-header-collapsed";
export const EXPERIMENT_BOARD_POLL_DELAY_MS = 5_000;
export const PROJECT_TAB_CACHE_LIMIT = 8;
export const OPEN_PROJECT_HEARTBEAT_INTERVAL_MS = 3_000;
export const ACTIVE_PROJECT_CACHE_OBSERVE_INTERVAL_MS = 1_000;

interface ProjectCachePollingClock {
  setInterval(callback: () => void, delay: number): number;
  clearInterval(intervalId: number): void;
}

interface ProjectCachePollingVisibility {
  isHidden(): boolean;
  listen(callback: () => void): () => void;
}

interface UseProjectTabsOptions {
  initialProjectId: string | null;
  initialSetupOpen: boolean;
  projectIndexReady: boolean;
  reportError: (message: string) => void;
}

export function cacheProjectTabState<T>(
  cache: Map<string, T>,
  projectId: string,
  state: T,
  limit = PROJECT_TAB_CACHE_LIMIT,
): void {
  cache.delete(projectId);
  cache.set(projectId, state);
  while (cache.size > limit) {
    const oldest = cache.keys().next().value;
    if (oldest === undefined) break;
    cache.delete(oldest);
  }
}

export function projectTabStateForOpen<T>(
  cache: Map<string, T>,
  projectId: string,
): { state: T; loading: false } | null {
  const state = cache.get(projectId);
  if (!state) return null;
  cacheProjectTabState(cache, projectId, state);
  return { state, loading: false };
}

export function projectIdsForCacheHeartbeat(tabs: ProjectTab[]): string[] {
  return [...new Set(tabs.map((tab) => tab.id))];
}

export function inactiveProjectTabState<T>(
  cache: Map<string, T>,
  tabs: ProjectTab[],
  activeProjectId: string | null,
  requestedProjectId: string,
): T | null {
  if (activeProjectId === requestedProjectId || !tabs.some((tab) => tab.id === requestedProjectId))
    return null;
  return cache.get(requestedProjectId) ?? null;
}

export function startProjectCachePolling(
  clock: ProjectCachePollingClock,
  visibility: ProjectCachePollingVisibility,
  sweepOpenProjects: () => void,
  observeActiveProject: () => void,
): () => void {
  const runWhenVisible = (callback: () => void) => () => {
    if (!visibility.isHidden()) callback();
  };
  const sweepInterval = clock.setInterval(
    runWhenVisible(sweepOpenProjects),
    OPEN_PROJECT_HEARTBEAT_INTERVAL_MS,
  );
  const activeInterval = clock.setInterval(
    runWhenVisible(observeActiveProject),
    ACTIVE_PROJECT_CACHE_OBSERVE_INTERVAL_MS,
  );
  const stopListening = visibility.listen(runWhenVisible(sweepOpenProjects));
  return () => {
    clock.clearInterval(sweepInterval);
    clock.clearInterval(activeInterval);
    stopListening();
  };
}

export function singleFlightProjectCacheHeartbeat(
  inFlight: Map<string, Promise<void>>,
  projectId: string,
  heartbeat: () => Promise<void>,
): Promise<void> {
  const pending = inFlight.get(projectId);
  if (pending) return pending;
  const request = heartbeat().finally(() => {
    if (inFlight.get(projectId) === request) inFlight.delete(projectId);
  });
  inFlight.set(projectId, request);
  return request;
}

export function useProjectTabs<T extends { project: ProjectSnapshot }>({
  initialProjectId,
  initialSetupOpen,
  projectIndexReady,
  reportError,
}: UseProjectTabsOptions) {
  const [projectId, setProjectId] = useState<string | null>(initialProjectId);
  const [setupOpen, setSetupOpen] = useState(initialSetupOpen);
  const [projects, setProjects] = useState<ProjectCard[]>([]);
  const [openProjectTabs, setOpenProjectTabs] = useState<ProjectTab[]>([]);
  const [experimentLoops, setExperimentLoops] = useState<ExperimentLoopIndexEntry[]>([]);
  const [project, setProject] = useState<ProjectSnapshot | null>(null);
  const [projectHeaderCollapsed, setProjectHeaderCollapsed] = useState(() =>
    readProjectHeaderCollapsed(initialProjectId),
  );
  const activeProjectId = useRef(projectId);
  const openProjectTabsRef = useRef(openProjectTabs);
  const projectTabStatesRef = useRef(new Map<string, T>());
  const projectCacheHeartbeatInFlight = useRef(new Map<string, Promise<void>>());
  activeProjectId.current = projectId;
  openProjectTabsRef.current = openProjectTabs;

  const isActiveProject = useCallback(
    (requestedProjectId: string) => activeProjectId.current === requestedProjectId,
    [],
  );
  const getActiveProjectId = useCallback(() => activeProjectId.current, []);
  const replaceProject = useCallback((nextProject: ProjectSnapshot | null) => {
    setProject(nextProject);
  }, []);
  const updateProject = useCallback(
    (update: (current: ProjectSnapshot | null) => ProjectSnapshot | null) => {
      setProject(update);
    },
    [],
  );
  const replaceProjects = useCallback((nextProjects: ProjectCard[]) => {
    setProjects(nextProjects);
  }, []);
  const refreshExperimentLoops = useCallback(async () => {
    const nextEntries = await loadExperimentEpisodes();
    setExperimentLoops(nextEntries);
    return nextEntries;
  }, []);
  const loadProjectIndex = useCallback(async () => {
    setProjects(await api<ProjectCard[]>("/api/projects"));
  }, []);
  const applyHashRoute = useCallback((nextProjectId: string | null, nextSetupOpen: boolean) => {
    setSetupOpen(nextSetupOpen);
    setProjectId(nextProjectId);
  }, []);
  const clearProjectRoute = useCallback(() => {
    setSetupOpen(false);
    setProjectId(null);
  }, []);
  const openSetup = useCallback(() => {
    setSetupOpen(true);
    setProjectId(null);
    window.location.hash = "/projects/new";
  }, []);
  const returnToProjects = useCallback(() => {
    setSetupOpen(false);
    setProjectId(null);
    window.location.hash = "";
  }, []);

  const setTabs = useCallback((nextTabs: ProjectTab[]) => {
    openProjectTabsRef.current = nextTabs;
    setOpenProjectTabs(nextTabs);
  }, []);
  const tabForProject = useCallback(
    (id: string): ProjectTab => ({
      id,
      name:
        projects.find((item) => item.id === id)?.name ??
        experimentLoops.find((item) => item.project_id === id)?.project_name ??
        (project?.id === id ? project.name : id),
    }),
    [experimentLoops, project, projects],
  );
  const commitProjectOpen = useCallback(
    (id: string, experimentRoute: string | null = null) => {
      setTabs(openProjectTab(openProjectTabsRef.current, tabForProject(id)));
      setSetupOpen(false);
      window.location.hash = experimentRoute
        ? experimentBoardHref(id, experimentRoute).slice(1)
        : `/projects/${encodeURIComponent(id)}`;
    },
    [setTabs, tabForProject],
  );
  const activateProjectTab = useCallback(
    (id: string) => {
      if (id === activeProjectId.current) return;
      commitProjectOpen(id);
    },
    [commitProjectOpen],
  );
  const closeDockedProject = useCallback(
    (id: string): boolean => {
      const result = closeProjectTab(openProjectTabsRef.current, activeProjectId.current, id);
      if (result.tabs === openProjectTabsRef.current) return false;
      projectTabStatesRef.current.delete(id);
      setTabs(result.tabs);
      if (id !== activeProjectId.current) return true;
      if (result.activeProjectId) {
        setSetupOpen(false);
        window.location.hash = `/projects/${encodeURIComponent(result.activeProjectId)}`;
      } else {
        setSetupOpen(false);
        setProjectId(null);
        window.location.hash = "";
      }
      return true;
    },
    [setTabs],
  );
  const removeProject = useCallback(
    (id: string) => {
      setProjects((current) => current.filter((item) => item.id !== id));
      setExperimentLoops((current) => current.filter((item) => item.project_id !== id));
      projectTabStatesRef.current.delete(id);
      setTabs(closeProjectTab(openProjectTabsRef.current, activeProjectId.current, id).tabs);
    },
    [setTabs],
  );

  const resetProjectHeader = useCallback((nextProjectId: string | null) => {
    setProjectHeaderCollapsed(readProjectHeaderCollapsed(nextProjectId));
  }, []);
  const restoreProjectHeader = useCallback((collapsed: boolean) => {
    setProjectHeaderCollapsed(collapsed);
  }, []);
  const toggleProjectHeader = useCallback(() => {
    setProjectHeaderCollapsed((collapsed) => !collapsed);
  }, []);

  const cacheProjectState = useCallback((id: string, state: T) => {
    cacheProjectTabState(projectTabStatesRef.current, id, state);
  }, []);
  const cachedProjectStateForOpen = useCallback(
    (id: string) => projectTabStateForOpen(projectTabStatesRef.current, id),
    [],
  );
  const inactiveCachedProjectState = useCallback(
    (id: string) =>
      inactiveProjectTabState(
        projectTabStatesRef.current,
        openProjectTabsRef.current,
        activeProjectId.current,
        id,
      ),
    [],
  );
  const isProjectTabOpen = useCallback(
    (id: string) => openProjectTabsRef.current.some((tab) => tab.id === id),
    [],
  );
  const projectIdsForHeartbeat = useCallback(
    () => projectIdsForCacheHeartbeat(openProjectTabsRef.current),
    [],
  );
  const adjacentProjectId = useCallback(
    (offset: -1 | 1) =>
      adjacentProjectTabId(openProjectTabsRef.current, activeProjectId.current, offset),
    [],
  );
  const runProjectHeartbeat = useCallback(
    (id: string, heartbeat: () => Promise<void>) =>
      singleFlightProjectCacheHeartbeat(projectCacheHeartbeatInFlight.current, id, heartbeat),
    [],
  );

  useEffect(() => {
    if (!project || project.id !== projectId) return;
    const nextTabs = openProjectTab(openProjectTabsRef.current, {
      id: project.id,
      name: project.name,
    });
    if (nextTabs === openProjectTabsRef.current) return;
    setTabs(nextTabs);
  }, [project, projectId, setTabs]);

  useEffect(() => {
    if (!projectId) return;
    try {
      localStorage.setItem(
        projectHeaderCollapsedStorageKey(projectId),
        String(projectHeaderCollapsed),
      );
    } catch {
      // Layout state is a convenience; storage failures must not affect the project.
    }
  }, [projectHeaderCollapsed, projectId]);

  useEffect(() => {
    if (!projectIndexReady || projectId || setupOpen) return;
    let stopped = false;
    let timer = 0;
    const schedule = () => {
      timer = window.setTimeout(() => void poll(), EXPERIMENT_BOARD_POLL_DELAY_MS);
    };
    const poll = async () => {
      if (stopped) return;
      if (pageIsHidden()) {
        schedule();
        return;
      }
      try {
        const nextEntries = await loadExperimentEpisodes();
        if (!stopped) setExperimentLoops(nextEntries);
      } catch (error) {
        if (!stopped) {
          reportError(
            `Experiment board could not be refreshed: ${error instanceof Error ? error.message : String(error)}`,
          );
        }
      }
      if (!stopped) schedule();
    };
    void poll();
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
  }, [projectIndexReady, projectId, reportError, setupOpen]);

  return {
    projectId,
    setupOpen,
    projects,
    openProjectTabs,
    experimentLoops,
    project,
    projectHeaderCollapsed,
    isActiveProject,
    getActiveProjectId,
    replaceProject,
    updateProject,
    replaceProjects,
    loadProjectIndex,
    refreshExperimentLoops,
    applyHashRoute,
    clearProjectRoute,
    openSetup,
    returnToProjects,
    commitProjectOpen,
    activateProjectTab,
    closeDockedProject,
    removeProject,
    resetProjectHeader,
    restoreProjectHeader,
    toggleProjectHeader,
    cacheProjectState,
    cachedProjectStateForOpen,
    inactiveCachedProjectState,
    isProjectTabOpen,
    projectIdsForHeartbeat,
    adjacentProjectId,
    runProjectHeartbeat,
  };
}

function projectHeaderCollapsedStorageKey(projectId: string): string {
  return `${PROJECT_HEADER_COLLAPSED_KEY}:${projectId}`;
}

function readProjectHeaderCollapsed(projectId: string | null): boolean {
  if (!projectId) return false;
  try {
    return localStorage.getItem(projectHeaderCollapsedStorageKey(projectId)) === "true";
  } catch {
    return false;
  }
}

function pageIsHidden(): boolean {
  if (typeof document === "undefined") return false;
  return document.visibilityState === "hidden";
}
