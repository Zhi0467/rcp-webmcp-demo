import { useCallback, useState } from "react";
import {
  acceptCurrentBackendIdentity,
  applyDesktopUpdate,
  checkDesktopUpdate,
  DESKTOP_FOLDER_ACCESS_ACK_KEY,
  desktopFolderAccessAcknowledgementValue,
  desktopReconnectBackend,
  needsDesktopFolderAccessAcknowledgement,
  type DesktopUpdate,
} from "../desktopRuntime";

interface PendingDesktopProject {
  projectId: string;
  experimentId: string | null;
}

interface UpdateIdentityCheck {
  ok: boolean;
  activeAgentTasks: number;
}

export function useDesktopShell(desktop: boolean) {
  const [reconnecting, setReconnecting] = useState(false);
  const [desktopUpdate, setDesktopUpdate] = useState<DesktopUpdate | null>(null);
  const [updateExpanded, setUpdateExpanded] = useState(false);
  const [updateApplying, setUpdateApplying] = useState(false);
  const [updateError, setUpdateError] = useState<string | null>(null);
  const [pendingDesktopProject, setPendingDesktopProject] = useState<PendingDesktopProject | null>(
    null,
  );
  const [desktopAccessError, setDesktopAccessError] = useState<string | null>(null);

  const refreshDesktopUpdate = useCallback(async () => {
    if (!desktop) return;
    try {
      const result = await checkDesktopUpdate();
      setDesktopUpdate(result?.available ? result : null);
      setUpdateError(result?.enabled === false && result.reason ? result.reason : null);
    } catch (error) {
      setUpdateError(error instanceof Error ? error.message : String(error));
    }
  }, [desktop]);

  const recordDesktopUpdateReady = useCallback(
    (version: string | undefined, activeAgentTasks: number) => {
      setDesktopUpdate((current) => ({
        enabled: true,
        available: true,
        version: version ?? current?.version,
        current_version: current?.current_version,
        active_agent_tasks: current?.active_agent_tasks ?? activeAgentTasks,
      }));
    },
    [],
  );

  const requestDesktopProjectOpen = (projectId: string, experimentId: string | null) => {
    if (!desktop) return false;
    let storedAcknowledgement: string | null = null;
    try {
      storedAcknowledgement = localStorage.getItem(DESKTOP_FOLDER_ACCESS_ACK_KEY);
    } catch {}
    if (!needsDesktopFolderAccessAcknowledgement(true, storedAcknowledgement)) return false;
    setDesktopAccessError(null);
    setPendingDesktopProject({ projectId, experimentId });
    return true;
  };

  const continueDesktopProjectOpen = (
    openProject: (projectId: string, experimentId: string | null) => void,
  ) => {
    if (!pendingDesktopProject) return;
    try {
      localStorage.setItem(
        DESKTOP_FOLDER_ACCESS_ACK_KEY,
        desktopFolderAccessAcknowledgementValue(),
      );
      const projectToOpen = pendingDesktopProject;
      setPendingDesktopProject(null);
      setDesktopAccessError(null);
      openProject(projectToOpen.projectId, projectToOpen.experimentId);
    } catch (error) {
      setDesktopAccessError(
        `RCP could not record this choice: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  };

  const dismissDesktopProjectOpen = () => {
    setPendingDesktopProject(null);
    setDesktopAccessError(null);
  };

  const reconnectBackend = async (reportIssue: (message: string) => void) => {
    if (reconnecting) return;
    setReconnecting(true);
    try {
      if (desktop) {
        const status = await desktopReconnectBackend();
        if (window.location.origin !== new URL(status.base_url).origin) {
          window.location.replace(`${status.base_url}/${window.location.hash}`);
          return;
        }
      }
      await acceptCurrentBackendIdentity();
    } catch (error) {
      reportIssue(error instanceof Error ? error.message : String(error));
    } finally {
      setReconnecting(false);
    }
  };

  const applyUpdate = async (
    activeTask: boolean,
    verifyIdentity: () => Promise<UpdateIdentityCheck>,
  ) => {
    if (!desktopUpdate || updateApplying) return;
    setUpdateApplying(true);
    setUpdateError(null);
    try {
      const identity = await verifyIdentity();
      if (!identity.ok) return;
      const hasActiveWork = activeTask || identity.activeAgentTasks > 0;
      if (hasActiveWork && !updateExpanded) {
        setUpdateExpanded(true);
        return;
      }
      await applyDesktopUpdate(hasActiveWork);
    } catch (error) {
      setUpdateError(error instanceof Error ? error.message : String(error));
    } finally {
      setUpdateApplying(false);
    }
  };

  const dismissUpdate = () => {
    setDesktopUpdate(null);
    setUpdateError(null);
    setUpdateExpanded(false);
  };

  return {
    reconnecting,
    desktopUpdate,
    updateExpanded,
    updateApplying,
    updateError,
    pendingDesktopProject,
    desktopAccessError,
    refreshDesktopUpdate,
    recordDesktopUpdateReady,
    requestDesktopProjectOpen,
    continueDesktopProjectOpen,
    dismissDesktopProjectOpen,
    reconnectBackend,
    applyUpdate,
    expandUpdate: () => setUpdateExpanded(true),
    dismissUpdate,
  };
}
