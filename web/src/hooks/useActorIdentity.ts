import { useCallback, useEffect, useRef, useState } from "react";
import {
  api,
  ApiError,
  exchangeTeamSession,
  pinApiInstance,
  registerIdentityNameRequiredHandler,
  registerMutationFailureHandler,
} from "../api";
import {
  BACKEND_IDENTITY_EVENT,
  establishBackendIdentity,
  reverifyBackendIdentity,
  verifyIdentityAfterMutationFailure,
  type BackendIdentityEventDetail,
} from "../desktopRuntime";
import type { Health, IdentityResponse } from "../types";

export function useActorIdentity() {
  const [identityReady, setIdentityReady] = useState(false);
  const [identityIssue, setIdentityIssue] = useState<string | null>(null);
  const [verifiedHealth, setVerifiedHealth] = useState<Health | null>(null);
  const [actorIdentity, setActorIdentity] = useState<IdentityResponse | null>(null);
  const [actorIdentityError, setActorIdentityError] = useState<string | null>(null);
  const [actorIdentityChecked, setActorIdentityChecked] = useState(false);
  const [teamSessionRequired, setTeamSessionRequired] = useState(false);
  const [actorNamePromptOpen, setActorNamePromptOpen] = useState(false);
  const [actorNameDraft, setActorNameDraft] = useState("");
  const [actorNameSaving, setActorNameSaving] = useState(false);
  const [actorNameError, setActorNameError] = useState<string | null>(null);

  const actorIdentityRef = useRef<IdentityResponse | null>(null);
  const actorNamePromptResolver = useRef<((saved: boolean) => void) | null>(null);
  const verifiedHealthRef = useRef<Health | null>(null);
  actorIdentityRef.current = actorIdentity;

  const requestActorName = useCallback((): Promise<boolean> => {
    if (actorNamePromptResolver.current) return Promise.resolve(false);
    setActorNameDraft(actorIdentityRef.current?.user.display_name ?? "");
    setActorNameError(null);
    setActorNamePromptOpen(true);
    return new Promise((resolve) => {
      actorNamePromptResolver.current = resolve;
    });
  }, []);

  const settleActorNamePrompt = useCallback((saved: boolean) => {
    const resolve = actorNamePromptResolver.current;
    actorNamePromptResolver.current = null;
    setActorNamePromptOpen(false);
    setActorNameSaving(false);
    setActorNameError(null);
    resolve?.(saved);
  }, []);

  const saveActorName = useCallback(async () => {
    const displayName = actorNameDraft.trim();
    if (!displayName || actorNameSaving) return;
    setActorNameSaving(true);
    setActorNameError(null);
    try {
      const saved = await api<IdentityResponse>("/api/identity", {
        method: "PATCH",
        body: JSON.stringify({ display_name: displayName }),
      });
      setActorIdentity(saved);
      setActorIdentityError(null);
      settleActorNamePrompt(true);
    } catch (error) {
      setActorNameError(error instanceof Error ? error.message : String(error));
      setActorNameSaving(false);
    }
  }, [actorNameDraft, actorNameSaving, settleActorNamePrompt]);

  useEffect(() => {
    const onIdentity = (event: Event) => {
      const detail = (event as CustomEvent<BackendIdentityEventDetail>).detail;
      setIdentityReady(true);
      setIdentityIssue(detail.ok ? null : detail.message || "RCP could not verify its backend.");
      if (detail.health) {
        verifiedHealthRef.current = detail.health;
        setVerifiedHealth(detail.health);
        if (detail.ok) pinApiInstance(detail.health.instance_id);
      }
    };
    window.addEventListener(BACKEND_IDENTITY_EVENT, onIdentity);
    registerMutationFailureHandler(verifyIdentityAfterMutationFailure);
    void establishBackendIdentity();
    return () => {
      registerMutationFailureHandler(null);
      window.removeEventListener(BACKEND_IDENTITY_EVENT, onIdentity);
    };
  }, []);

  useEffect(() => {
    registerIdentityNameRequiredHandler(requestActorName);
    return () => {
      registerIdentityNameRequiredHandler(null);
      const resolve = actorNamePromptResolver.current;
      actorNamePromptResolver.current = null;
      resolve?.(false);
    };
  }, [requestActorName]);

  useEffect(() => {
    if (!identityReady || identityIssue || !verifiedHealth) return;
    let stopped = false;
    setActorIdentityChecked(false);
    setTeamSessionRequired(false);
    setActorIdentityError(null);
    void api<IdentityResponse>("/api/identity")
      .then((identity) => {
        if (stopped) return;
        setActorIdentity(identity);
        setActorIdentityChecked(true);
      })
      .catch((error) => {
        if (stopped) return;
        setActorIdentity(null);
        if (
          verifiedHealth.space_kind === "team" &&
          error instanceof ApiError &&
          error.status === 401
        ) {
          setTeamSessionRequired(true);
        } else {
          setActorIdentityError(error instanceof Error ? error.message : String(error));
        }
        setActorIdentityChecked(true);
      });
    return () => {
      stopped = true;
    };
  }, [identityIssue, identityReady, verifiedHealth?.space_id, verifiedHealth?.space_kind]);

  const authenticateTeamSession = useCallback(async (token: string) => {
    const identity = await exchangeTeamSession(token);
    setActorIdentity(identity);
    setActorIdentityError(null);
    setActorIdentityChecked(true);
    setTeamSessionRequired(false);
  }, []);

  const reportIdentityIssue = useCallback((message: string) => {
    setIdentityIssue(message);
  }, []);

  const reverifyIdentity = useCallback((reason: string) => reverifyBackendIdentity(reason), []);

  const currentActiveAgentTasks = useCallback(
    () => verifiedHealthRef.current?.active_agent_tasks ?? 0,
    [],
  );

  const updateActorNameDraft = useCallback((value: string) => {
    setActorNameDraft(value);
    setActorNameError(null);
  }, []);

  return {
    identityReady,
    identityIssue,
    verifiedHealth,
    actorIdentity,
    actorIdentityError,
    actorIdentityChecked,
    teamSessionRequired,
    actorNamePromptOpen,
    actorNameDraft,
    actorNameSaving,
    actorNameError,
    requestActorName,
    settleActorNamePrompt,
    saveActorName,
    authenticateTeamSession,
    reportIdentityIssue,
    reverifyIdentity,
    currentActiveAgentTasks,
    updateActorNameDraft,
  };
}
