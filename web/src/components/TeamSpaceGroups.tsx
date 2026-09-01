import { LoaderCircle, Plus, RefreshCw, Server, WifiOff, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  addExistingDesktopTeamConnection,
  enrollDesktopTeamConnection,
  establishDesktopTeamSession,
  isDesktopRuntime,
  listDesktopTeamConnections,
  navigateDesktopToTeam,
  type EstablishedTeamSession,
  type TeamConnectionMetadata,
} from "../desktopRuntime";

type ConnectionState = "checking" | "available" | "unavailable";

export interface TeamConnectionView {
  connection: TeamConnectionMetadata;
  state: ConnectionState;
  error: string | null;
}

export function replaceTeamConnectionView(
  current: TeamConnectionView[],
  next: TeamConnectionView,
): TeamConnectionView[] {
  const index = current.findIndex(
    (item) => item.connection.connection_id === next.connection.connection_id,
  );
  if (index < 0) return [...current, next];
  const updated = [...current];
  updated[index] = next;
  return updated;
}

export function createTeamReconciliationTracker() {
  let active = true;
  const attempts = new Map<string, number>();
  return {
    begin(connectionId: string): number {
      const attempt = (attempts.get(connectionId) ?? 0) + 1;
      attempts.set(connectionId, attempt);
      return attempt;
    },
    isCurrent(connectionId: string, attempt: number): boolean {
      return active && attempts.get(connectionId) === attempt;
    },
    stop(): void {
      active = false;
      attempts.clear();
    },
  };
}

export function TeamSpaceGroups({
  addOpen,
  onOpenAdd,
  onCloseAdd,
}: {
  addOpen: boolean;
  onOpenAdd: () => void;
  onCloseAdd: () => void;
}) {
  const desktop = isDesktopRuntime();
  const [connections, setConnections] = useState<TeamConnectionView[]>([]);
  const [loading, setLoading] = useState(desktop);
  const [listError, setListError] = useState<string | null>(null);
  const reconciliation = useRef(createTeamReconciliationTracker());

  const updateConnection = useCallback((next: TeamConnectionView) => {
    setConnections((current) => replaceTeamConnectionView(current, next));
  }, []);

  const reconcile = useCallback(
    async (connection: TeamConnectionMetadata) => {
      const tracker = reconciliation.current;
      const attempt = tracker.begin(connection.connection_id);
      updateConnection({ connection, state: "checking", error: null });
      try {
        const session = await establishDesktopTeamSession(connection.connection_id);
        if (!tracker.isCurrent(connection.connection_id, attempt)) return null;
        updateConnection({ connection: session.connection, state: "available", error: null });
        return session;
      } catch (error) {
        if (!tracker.isCurrent(connection.connection_id, attempt)) return null;
        updateConnection({
          connection,
          state: "unavailable",
          error: error instanceof Error ? error.message : String(error),
        });
        throw error;
      }
    },
    [updateConnection],
  );

  useEffect(() => {
    if (!desktop) return;
    const tracker = createTeamReconciliationTracker();
    reconciliation.current = tracker;
    let stopped = false;
    setLoading(true);
    setListError(null);
    void listDesktopTeamConnections()
      .then(async (saved) => {
        if (stopped) return;
        setConnections(saved.map((connection) => ({ connection, state: "checking", error: null })));
        await Promise.allSettled(saved.map((connection) => reconcile(connection)));
      })
      .catch((error) => {
        if (!stopped) setListError(error instanceof Error ? error.message : String(error));
      })
      .finally(() => {
        if (!stopped) setLoading(false);
      });
    return () => {
      stopped = true;
      tracker.stop();
    };
  }, [desktop, reconcile]);

  const openProject = async (view: TeamConnectionView, projectId: string) => {
    const session = await reconcile(view.connection);
    if (!session) return;
    await navigateDesktopToTeam(session.connection.connection_id, projectId);
  };

  if (!desktop) return null;

  return (
    <section className="team-space-groups" aria-label="Team spaces" aria-busy={loading}>
      <header className="team-space-groups-header">
        <h2>Team spaces</h2>
        <button type="button" onClick={onOpenAdd}>
          <Plus size={14} aria-hidden="true" /> Add team space
        </button>
      </header>

      {listError && (
        <p className="team-space-list-error" role="alert">
          {listError}
        </p>
      )}

      {connections.map((view) => (
        <TeamConnectionGroup
          key={view.connection.connection_id}
          view={view}
          onReconnect={() => void reconcile(view.connection).catch(() => undefined)}
          onOpenProject={(projectId) => void openProject(view, projectId).catch(() => undefined)}
        />
      ))}

      {!loading && connections.length === 0 && !listError && (
        <button className="team-space-empty" type="button" onClick={onOpenAdd}>
          <Plus size={16} aria-hidden="true" /> Add your lab server
        </button>
      )}

      {addOpen && (
        <AddTeamSpaceDialog
          onClose={onCloseAdd}
          onEstablished={(session) => {
            updateConnection({ connection: session.connection, state: "available", error: null });
            onCloseAdd();
          }}
        />
      )}
    </section>
  );
}

export function TeamConnectionGroup({
  view,
  onReconnect,
  onOpenProject,
}: {
  view: TeamConnectionView;
  onReconnect: () => void;
  onOpenProject: (projectId: string) => void;
}) {
  const { connection, state, error } = view;
  return (
    <section
      className={`team-space-group ${state}`}
      aria-label={`${connection.display_name} team space`}
    >
      <header>
        <span className="team-space-state" aria-label={`Connection ${state}`}>
          {state === "checking" ? (
            <LoaderCircle className="spin" size={14} aria-hidden="true" />
          ) : state === "available" ? (
            <Server size={14} aria-hidden="true" />
          ) : (
            <WifiOff size={14} aria-hidden="true" />
          )}
        </span>
        <h3>{connection.display_name}</h3>
        {state === "unavailable" && (
          <button type="button" onClick={onReconnect}>
            <RefreshCw size={13} aria-hidden="true" /> Reconnect
          </button>
        )}
      </header>
      {state === "unavailable" && error && (
        <p className="team-space-connection-error" role="alert">
          {error}
        </p>
      )}
      <div className="team-project-shelf">
        {connection.last_known_cards.map((project) => (
          <button
            className="team-project-card"
            type="button"
            key={project.id}
            disabled={state !== "available"}
            onClick={() => onOpenProject(project.id)}
          >
            <strong>{project.name}</strong>
            <span>
              {project.attention_count > 0 ? `${project.attention_count} waiting` : "Open"}
            </span>
          </button>
        ))}
        {state === "available" && connection.last_known_cards.length === 0 && (
          <span className="team-space-no-projects">No projects yet</span>
        )}
      </div>
    </section>
  );
}

export function AddTeamSpaceDialog({
  onClose,
  onEstablished,
}: {
  onClose: () => void;
  onEstablished: (session: EstablishedTeamSession) => void;
}) {
  const [mode, setMode] = useState<"enroll" | "existing">("enroll");
  const [sshTarget, setSshTarget] = useState("");
  const [port, setPort] = useState("8421");
  const [displayName, setDisplayName] = useState("");
  const [secret, setSecret] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const dialogRef = useRef<HTMLElement>(null);
  const firstInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const returnFocus =
      document.activeElement instanceof HTMLElement ? document.activeElement : null;
    firstInputRef.current?.focus();
    return () => returnFocus?.focus();
  }, []);

  const close = () => {
    if (submitting) return;
    setSecret("");
    onClose();
  };

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (submitting) return;
    setSubmitting(true);
    setError(null);
    const remoteLoopbackPort = Number(port);
    try {
      const session =
        mode === "enroll"
          ? await enrollDesktopTeamConnection({
              ssh_target: sshTarget,
              remote_loopback_port: remoteLoopbackPort,
              enrollment_code: secret,
              member_display_name: displayName,
            })
          : await addExistingDesktopTeamConnection({
              ssh_target: sshTarget,
              remote_loopback_port: remoteLoopbackPort,
              member_token: secret,
            });
      onEstablished(session);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setSecret("");
      setSubmitting(false);
    }
  };

  return (
    <div
      className="modal-backdrop"
      onKeyDown={(event) => {
        if (event.key === "Escape") {
          event.preventDefault();
          close();
          return;
        }
        if (event.key !== "Tab") return;
        const focusable = Array.from(
          dialogRef.current?.querySelectorAll<HTMLElement>(
            'button:not([disabled]), input:not([disabled]), [tabindex]:not([tabindex="-1"])',
          ) ?? [],
        );
        if (focusable.length === 0) return;
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first.focus();
        }
      }}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) close();
      }}
    >
      <section
        ref={dialogRef}
        className="add-team-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="add-team-title"
      >
        <header>
          <h2 id="add-team-title">Add team space</h2>
          <button type="button" aria-label="Close Add team space" onClick={close}>
            <X size={16} aria-hidden="true" />
          </button>
        </header>
        <div className="add-team-mode" role="group" aria-label="Team membership">
          <button
            type="button"
            aria-pressed={mode === "enroll"}
            onClick={() => {
              setMode("enroll");
              setSecret("");
            }}
          >
            New member
          </button>
          <button
            type="button"
            aria-pressed={mode === "existing"}
            onClick={() => {
              setMode("existing");
              setSecret("");
            }}
          >
            Existing member
          </button>
        </div>
        <form autoComplete="off" onSubmit={(event) => void submit(event)}>
          <label>
            SSH target
            <input
              ref={firstInputRef}
              required
              spellCheck={false}
              placeholder="rcp@lab-server"
              value={sshTarget}
              onChange={(event) => setSshTarget(event.target.value)}
            />
          </label>
          <label>
            RCP server port
            <input
              required
              type="number"
              min="1"
              max="65535"
              value={port}
              onChange={(event) => setPort(event.target.value)}
            />
          </label>
          {mode === "enroll" && (
            <label>
              Your display name
              <input
                required
                value={displayName}
                onChange={(event) => setDisplayName(event.target.value)}
              />
            </label>
          )}
          <label>
            {mode === "enroll" ? "Bootstrap or invitation code" : "Permanent member token"}
            <input
              required
              type="password"
              name="team-credential"
              value={secret}
              onChange={(event) => setSecret(event.target.value)}
            />
          </label>
          {error && (
            <p className="add-team-error" role="alert">
              {error}
            </p>
          )}
          <footer>
            <button
              type="button"
              className="button secondary"
              disabled={submitting}
              onClick={close}
            >
              Cancel
            </button>
            <button type="submit" className="button" disabled={submitting}>
              {submitting ? "Connecting…" : "Add team space"}
            </button>
          </footer>
        </form>
      </section>
    </div>
  );
}
