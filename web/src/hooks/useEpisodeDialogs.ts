import { useCallback, useEffect, useState } from "react";
import { loadEpisodeMessages, loadEpisodes } from "../api";
import { isLiveEpisode, mergeEpisode } from "../campaigns";
import type { Episode, EpisodeMessage } from "../types";

export const LIVE_EPISODE_POLL_INTERVAL_MS = 1_500;

interface LiveEpisodePollingClock {
  setTimeout(callback: () => void, delay: number): number;
  clearTimeout(timeoutId: number): void;
}

export function startLiveEpisodePolling(
  clock: LiveEpisodePollingClock,
  refresh: () => Promise<void>,
  onError: (error: unknown) => void,
  onSuccess: () => void,
): () => void {
  let stopped = false;
  let timeoutId = 0;
  const schedule = () => {
    timeoutId = clock.setTimeout(() => void poll(), LIVE_EPISODE_POLL_INTERVAL_MS);
  };
  const poll = async () => {
    try {
      await refresh();
      if (!stopped) onSuccess();
    } catch (error) {
      if (!stopped) onError(error);
    } finally {
      if (!stopped) schedule();
    }
  };
  schedule();
  return () => {
    stopped = true;
    clock.clearTimeout(timeoutId);
  };
}

export function episodePollingTarget(episodes: Episode[]): Episode | null {
  return (
    episodes.find(isLiveEpisode) ??
    episodes.find((episode) => episode.graph_branch?.merge_state === "running") ??
    null
  );
}

interface EpisodeState {
  projectId: string | null;
  episodes: Episode[];
  messages: Record<string, EpisodeMessage[]>;
}

interface UseEpisodeDialogsOptions {
  projectId: string | null;
  apiBase: string;
  isActiveProject: (projectId: string) => boolean;
}

export function useEpisodeDialogs({
  projectId,
  apiBase,
  isActiveProject,
}: UseEpisodeDialogsOptions) {
  const [runDialogOpen, setRunDialogOpen] = useState(false);
  const [autoResearchDialogOpen, setAutoResearchDialogOpen] = useState(false);
  const [autoResearchStartError, setAutoResearchStartError] = useState<string | null>(null);
  const [episodeAction, setEpisodeAction] = useState<string | null>(null);
  const [episodeRefreshError, setEpisodeRefreshError] = useState<string | null>(null);
  const [episodeState, setEpisodeState] = useState<EpisodeState>({
    projectId: null,
    episodes: [],
    messages: {},
  });

  const episodes = episodeState.projectId === projectId ? episodeState.episodes : [];
  const episodeMessages = episodeState.projectId === projectId ? episodeState.messages : {};
  const liveAutoResearchEpisode =
    episodes.find((episode) => episode.mode === "auto_research" && isLiveEpisode(episode)) ?? null;
  const pollingEpisode = episodePollingTarget(episodes);
  const pollingAutoResearchEpisode = episodePollingTarget(
    episodes.filter((episode) => episode.mode === "auto_research"),
  );

  const refreshEpisodes = useCallback(async () => {
    if (!projectId || !apiBase) return;
    const requestedProjectId = projectId;
    const nextEpisodes = await loadEpisodes(apiBase);
    if (!isActiveProject(requestedProjectId)) return;
    setEpisodeState((current) => ({
      projectId: requestedProjectId,
      episodes: nextEpisodes,
      messages: current.projectId === requestedProjectId ? current.messages : {},
    }));
  }, [apiBase, projectId]);

  const refreshEpisodeMessages = useCallback(
    async (episodeId: string) => {
      if (!projectId || !apiBase) return;
      const requestedProjectId = projectId;
      const nextMessages = await loadEpisodeMessages(apiBase, episodeId);
      if (!isActiveProject(requestedProjectId)) return;
      setEpisodeState((current) =>
        current.projectId === requestedProjectId
          ? {
              ...current,
              messages: { ...current.messages, [episodeId]: nextMessages },
            }
          : current,
      );
    },
    [apiBase, projectId],
  );

  useEffect(() => {
    if (!projectId || !apiBase) {
      setEpisodeRefreshError(null);
      setEpisodeState({ projectId: null, episodes: [], messages: {} });
      return;
    }
    const requestedProjectId = projectId;
    setEpisodeRefreshError(null);
    setEpisodeState((current) =>
      current.projectId === requestedProjectId
        ? current
        : { projectId: requestedProjectId, episodes: [], messages: {} },
    );
    void refreshEpisodes()
      .then(() => {
        if (isActiveProject(requestedProjectId)) setEpisodeRefreshError(null);
      })
      .catch((error) => {
        if (!isActiveProject(requestedProjectId)) return;
        setEpisodeRefreshError(
          `Episodes could not be loaded: ${error instanceof Error ? error.message : String(error)}`,
        );
      });
  }, [apiBase, projectId, refreshEpisodes]);

  useEffect(() => {
    const episodeId = pollingEpisode?.episode_id;
    if (!episodeId) return;
    return startLiveEpisodePolling(
      {
        setTimeout: (callback, delay) => window.setTimeout(callback, delay),
        clearTimeout: (timeoutId) => window.clearTimeout(timeoutId),
      },
      async () => {
        await Promise.all([
          refreshEpisodes(),
          pollingAutoResearchEpisode
            ? refreshEpisodeMessages(pollingAutoResearchEpisode.episode_id)
            : Promise.resolve(),
        ]);
      },
      (error) => {
        setEpisodeRefreshError(
          `Episodes could not refresh: ${error instanceof Error ? error.message : String(error)}`,
        );
      },
      () => setEpisodeRefreshError(null),
    );
  }, [
    pollingAutoResearchEpisode?.episode_id,
    pollingEpisode?.episode_id,
    refreshEpisodeMessages,
    refreshEpisodes,
  ]);

  const replaceEpisode = useCallback(
    (nextEpisode: Episode) => {
      setEpisodeState((current) => {
        if (!isActiveProject(nextEpisode.project_id)) return current;
        const currentEpisodes =
          current.projectId === nextEpisode.project_id ? current.episodes : [];
        return {
          projectId: nextEpisode.project_id,
          messages: current.projectId === nextEpisode.project_id ? current.messages : {},
          episodes: mergeEpisode(currentEpisodes, nextEpisode),
        };
      });
    },
    [isActiveProject],
  );

  const recordEpisodeMessage = useCallback(
    (requestedProjectId: string, episodeId: string, saved: EpisodeMessage) => {
      setEpisodeState((current) =>
        current.projectId === requestedProjectId
          ? {
              ...current,
              messages: {
                ...current.messages,
                [episodeId]: [
                  ...(current.messages[episodeId] ?? []).filter(
                    (item) => item.message_id !== saved.message_id,
                  ),
                  saved,
                ],
              },
            }
          : current,
      );
    },
    [],
  );

  const openRunDialog = useCallback(() => setRunDialogOpen(true), []);
  const closeRunDialog = useCallback(() => setRunDialogOpen(false), []);
  const openAutoResearchDialog = useCallback(() => {
    setAutoResearchStartError(null);
    setAutoResearchDialogOpen(true);
  }, []);
  const closeAutoResearchDialog = useCallback(() => setAutoResearchDialogOpen(false), []);
  const reportAutoResearchStartError = useCallback(
    (message: string | null) => setAutoResearchStartError(message),
    [],
  );
  const beginEpisodeAction = useCallback(
    (action: string) => {
      if (episodeAction) return null;
      setEpisodeAction(action);
      return () => setEpisodeAction(null);
    },
    [episodeAction],
  );

  return {
    runDialogOpen,
    autoResearchDialogOpen,
    autoResearchStartError,
    episodeAction,
    episodeRefreshError,
    episodes,
    episodeMessages,
    liveAutoResearchEpisode,
    openRunDialog,
    closeRunDialog,
    openAutoResearchDialog,
    closeAutoResearchDialog,
    reportAutoResearchStartError,
    beginEpisodeAction,
    replaceEpisode,
    recordEpisodeMessage,
    refreshEpisodes,
    refreshEpisodeMessages,
  };
}
