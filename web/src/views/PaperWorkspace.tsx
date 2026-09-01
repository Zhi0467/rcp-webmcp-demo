import {
  AlertTriangle,
  Check,
  ChevronRight,
  FilePenLine,
  History,
  LoaderCircle,
  MessageSquarePlus,
  Send,
  WifiOff,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { isActiveTask, reconstructTaskTranscript, relatedCoachTasks } from "../agentTasks";
import { api } from "../api";
import { MarkdownAnswer } from "../chatMarkdown";
import { profileRunConfig } from "../components/AgentConfigControls";
import { SkillPicker, useSkillPicker } from "../components/SkillPicker";
import { EMPTY_SKILL_SELECTION, skillInvocationFields } from "../skillPicker";
import type {
  AgentRunConfig,
  AgentTask,
  PaperSnapshot,
  ProjectSnapshot,
  StartAgentTask,
  WritingSession,
} from "../types";

interface Props {
  apiBase: string;
  project: ProjectSnapshot;
  initialPaper: PaperSnapshot;
  tasks: AgentTask[];
  onStartTask: StartAgentTask;
  onPaperChange: (paper: PaperSnapshot) => void;
}

const MIN_EDITOR_SHARE = 35;
const MAX_EDITOR_SHARE = 70;
const MIN_EDITOR_WIDTH = 360;
const MIN_COACH_WIDTH = 320;
const DIVIDER_WIDTH = 9;
const PAPER_VIEW_STORAGE_PREFIX = "rcp:paper-view";
const PAPER_REFRESH_INTERVAL_MS = 5_000;

type PaperView = "write" | "preview" | "incoming";

interface EditorShareBounds {
  minimum: number;
  maximum: number;
}

function editorShareBoundsForWidth(width: number): EditorShareBounds {
  if (!width) return { minimum: MIN_EDITOR_SHARE, maximum: MAX_EDITOR_SHARE };
  const minimum = Math.min(
    MAX_EDITOR_SHARE,
    Math.max(MIN_EDITOR_SHARE, (MIN_EDITOR_WIDTH / width) * 100),
  );
  const coachConstrainedMaximum = Math.min(
    MAX_EDITOR_SHARE,
    ((width - DIVIDER_WIDTH - MIN_COACH_WIDTH) / width) * 100,
  );
  return { minimum, maximum: Math.max(minimum, coachConstrainedMaximum) };
}

function clampEditorShare(value: number, bounds: EditorShareBounds): number {
  return Math.min(bounds.maximum, Math.max(bounds.minimum, value));
}

function storedPaperView(projectId: string, allowIncoming: boolean): PaperView {
  try {
    const stored = localStorage.getItem(`${PAPER_VIEW_STORAGE_PREFIX}:${projectId}`);
    if (stored === "preview" || (allowIncoming && stored === "incoming")) return stored;
    return "write";
  } catch {
    return "write";
  }
}

export function swapPaperBuffers(editor: string, incoming: string): [string, string] {
  return [incoming, editor];
}

export function loadPaperSnapshot(
  load: (path: string) => Promise<PaperSnapshot>,
  apiBase: string,
): Promise<PaperSnapshot> {
  return load(`${apiBase}/paper`);
}

export function PaperWorkspace({
  apiBase,
  project,
  initialPaper,
  tasks,
  onStartTask,
  onPaperChange,
}: Props) {
  const [paper, setPaper] = useState(initialPaper);
  const [content, setContent] = useState(initialPaper.content);
  const [incomingContent, setIncomingContent] = useState(initialPaper.incoming_content ?? "");
  // Keep the version paired with the visible Incoming buffer. Canonical may
  // advance again while that buffer temporarily holds the displaced draft.
  const [incomingCanonicalHash, setIncomingCanonicalHash] = useState(
    initialPaper.sync_state === "behind" ? (initialPaper.canonical_hash ?? null) : null,
  );
  const [saveBaseHash, setSaveBaseHash] = useState(initialPaper.base_hash ?? null);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [sessions, setSessions] = useState<WritingSession[]>([]);
  const [activeSession, setActiveSession] = useState<WritingSession | null>(null);
  const [freshSession, setFreshSession] = useState(false);
  const [pendingCoachTaskId, setPendingCoachTaskId] = useState<string | null>(null);
  const [coachSubmitting, setCoachSubmitting] = useState(false);
  const [config, setConfig] = useState<AgentRunConfig>(() =>
    profileRunConfig(project.agent_profiles.paper_coach),
  );
  const [message, setMessage] = useState("");
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [paperView, setPaperView] = useState<PaperView>(() =>
    storedPaperView(project.id, initialPaper.sync_state === "behind"),
  );
  const [editorShare, setEditorShare] = useState(62);
  const [editorShareBounds, setEditorShareBounds] = useState<EditorShareBounds>({
    minimum: MIN_EDITOR_SHARE,
    maximum: MAX_EDITOR_SHARE,
  });
  const workspace = useRef<HTMLElement>(null);
  const textarea = useRef<HTMLTextAreaElement>(null);
  const coachTextarea = useRef<HTMLTextAreaElement>(null);
  const latestContent = useRef(content);
  const buffersSwapped = useRef(false);
  const paperRequestGeneration = useRef(0);
  const handledCoachTask = useRef<string | null>(
    tasks.find((task) => task.kind === "paper_coach" && task.settled)?.operation_id ?? null,
  );

  useEffect(() => {
    latestContent.current = content;
  }, [content]);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;
    const poll = async () => {
      const generation = ++paperRequestGeneration.current;
      try {
        const next = await loadPaperSnapshot(api, apiBase);
        if (cancelled || generation !== paperRequestGeneration.current) return;
        setPaper(next);
        if (!buffersSwapped.current) {
          setIncomingContent(next.incoming_content ?? "");
          setIncomingCanonicalHash(
            next.sync_state === "behind" ? (next.canonical_hash ?? null) : null,
          );
        }
        if (next.sync_state !== "behind") setSaveBaseHash(next.base_hash ?? null);
        onPaperChange(next);
      } catch {
        // A background freshness check must not turn a usable local editor into an error state.
      } finally {
        if (!cancelled) timer = window.setTimeout(() => void poll(), PAPER_REFRESH_INTERVAL_MS);
      }
    };
    timer = window.setTimeout(() => void poll(), PAPER_REFRESH_INTERVAL_MS);
    return () => {
      cancelled = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [apiBase, onPaperChange]);

  const attachSession = (session: WritingSession) => {
    setActiveSession(session);
    setFreshSession(false);
    setPendingCoachTaskId(null);
    setConfig({
      provider: session.provider,
      model: session.model === "provider-default" ? "" : session.model,
      reasoning: session.reasoning ?? "medium",
      run_on: session.execution_machine,
    });
    setSubmitError(null);
  };

  const loadSessions = async (preferredSessionId?: string | null) => {
    const items = await api<WritingSession[]>(`${apiBase}/paper/sessions`);
    setSessions(items);
    if (preferredSessionId) {
      const preferredSession = items.find((item) => item.native_session_id === preferredSessionId);
      if (preferredSession) attachSession(preferredSession);
      else {
        setActiveSession(null);
        setFreshSession(false);
      }
    }
    return items;
  };

  useEffect(() => {
    const latestSessionId = tasks.find(
      (task) => task.kind === "paper_coach" && task.native_session_id,
    )?.native_session_id;
    void loadSessions(freshSession ? null : latestSessionId).catch((error) => {
      setSubmitError(error instanceof Error ? error.message : String(error));
    });
  }, [apiBase]);

  const latestCoachTask = tasks.find((task) => task.kind === "paper_coach") ?? null;
  const pendingCoachTask = pendingCoachTaskId
    ? (tasks.find((task) => task.operation_id === pendingCoachTaskId) ?? null)
    : latestCoachTask && isActiveTask(latestCoachTask)
      ? latestCoachTask
      : null;
  const activeCoachTask = tasks.find(
    (task) =>
      task.kind === "paper_coach" &&
      isActiveTask(task) &&
      (activeSession
        ? task.native_session_id === activeSession.native_session_id
        : task.operation_id === pendingCoachTask?.operation_id),
  );
  const skillCatalog = project.skill_catalog ?? [];
  const skillDefaults = project.skill_defaults ?? EMPTY_SKILL_SELECTION;
  const readiness = project.provider_readiness[config.run_on]?.[config.provider];
  const skills = useSkillPicker({
    catalog: skillCatalog,
    defaults: skillDefaults,
    provider: config.provider,
    providerLabel: readiness?.label || config.provider,
    machine: config.run_on,
    inventory: project.provider_skill_inventories?.[config.run_on]?.[config.provider],
    message,
    onComplete: (next) => {
      setMessage(next);
      setSubmitError(null);
      window.requestAnimationFrame(() => {
        coachTextarea.current?.focus();
        coachTextarea.current?.setSelectionRange(next.length, next.length);
      });
    },
  });

  useEffect(() => {
    skills.reset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project.id]);
  useEffect(() => {
    if (!latestCoachTask?.native_session_id || !latestCoachTask.settled) return;
    if (handledCoachTask.current === latestCoachTask.operation_id) return;
    handledCoachTask.current = latestCoachTask.operation_id;
    void loadSessions(latestCoachTask.native_session_id)
      .catch((error) => setSubmitError(error instanceof Error ? error.message : String(error)))
      .finally(() => setPendingCoachTaskId(null));
  }, [latestCoachTask?.native_session_id, latestCoachTask?.operation_id, latestCoachTask?.status]);

  useEffect(() => {
    if (!dirty || saving || saveError || paper.sync_state === "not_created") return;
    const timer = window.setTimeout(async () => {
      setSaving(true);
      const savedContent = latestContent.current;
      const generation = ++paperRequestGeneration.current;
      try {
        const next = await api<PaperSnapshot>(`${apiBase}/paper`, {
          method: "PUT",
          body: JSON.stringify({ content: savedContent, base_hash: saveBaseHash }),
        });
        if (generation !== paperRequestGeneration.current) return;
        setPaper(next);
        setIncomingContent(next.incoming_content ?? "");
        setIncomingCanonicalHash(
          next.sync_state === "behind" ? (next.canonical_hash ?? null) : null,
        );
        if (next.sync_state !== "behind") setSaveBaseHash(next.base_hash ?? null);
        buffersSwapped.current = false;
        onPaperChange(next);
        setDirty(next.content !== latestContent.current);
        setSaveError(null);
      } catch (error) {
        setSaveError(error instanceof Error ? error.message : String(error));
      } finally {
        setSaving(false);
      }
    }, 650);
    return () => window.clearTimeout(timer);
  }, [apiBase, dirty, onPaperChange, paper.sync_state, saveBaseHash, saveError, saving]);

  const create = async () => {
    if (creating) return;
    setCreating(true);
    setSaveError(null);
    try {
      const next = await api<PaperSnapshot>(`${apiBase}/paper/create`, { method: "POST" });
      setPaper(next);
      setContent(next.content);
      setIncomingContent(next.incoming_content ?? "");
      setIncomingCanonicalHash(next.sync_state === "behind" ? (next.canonical_hash ?? null) : null);
      setSaveBaseHash(next.base_hash ?? null);
      buffersSwapped.current = false;
      latestContent.current = next.content;
      setDirty(next.sync_state === "unsynced");
      onPaperChange(next);
      requestAnimationFrame(() => textarea.current?.focus());
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : String(error));
    } finally {
      setCreating(false);
    }
  };

  const updateMessage = (next: string) => {
    setMessage(next);
    skills.readMessage(next);
    setSubmitError(null);
  };

  const send = async () => {
    const text = message.trim();
    if (!text || activeCoachTask || coachSubmitting) return;
    setSubmitError(null);
    setCoachSubmitting(true);
    try {
      const task = await onStartTask("paper_coach", {
        message: text,
        ...config,
        model: config.model || null,
        session_id: activeSession?.native_session_id ?? null,
        ...skillInvocationFields(skills.selection, skills.providerSkillNames),
      });
      setPendingCoachTaskId(task.operation_id);
      setMessage("");
      skills.reset();
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : String(error));
    } finally {
      setCoachSubmitting(false);
    }
  };

  const sessionTasks = activeSession
    ? relatedCoachTasks(tasks, activeSession.native_session_id)
    : pendingCoachTask
      ? [pendingCoachTask]
      : [];
  const transcript = reconstructTaskTranscript(sessionTasks);
  const reviewStale = activeSession
    ? activeSession.graph_revision_examined < project.revision
    : false;
  const syncLabel = paper.sync_state.replace("_", " ");
  const wordCount = useMemo(
    () => (content.trim() ? content.trim().split(/\s+/).length : 0),
    [content],
  );
  const coachActive = Boolean(activeCoachTask);
  const constrainEditorShare = (value: number) => {
    const width = workspace.current?.getBoundingClientRect().width ?? 0;
    return clampEditorShare(value, editorShareBoundsForWidth(width));
  };
  const resizeFromPointer = (clientX: number) => {
    const bounds = workspace.current?.getBoundingClientRect();
    if (!bounds) return;
    setEditorShare(constrainEditorShare(((clientX - bounds.left) / bounds.width) * 100));
  };
  const selectPaperView = (next: PaperView) => {
    setPaperView(next);
    try {
      localStorage.setItem(`${PAPER_VIEW_STORAGE_PREFIX}:${project.id}`, next);
    } catch {}
  };
  const applyIncoming = () => {
    const [nextContent, nextIncoming] = swapPaperBuffers(content, incomingContent);
    setContent(nextContent);
    setIncomingContent(nextIncoming);
    latestContent.current = nextContent;
    setSaveBaseHash(incomingCanonicalHash);
    buffersSwapped.current = !buffersSwapped.current;
    setDirty(false);
    setSaveError(null);
  };

  useEffect(() => {
    if (paperView === "incoming") setSaveBaseHash(incomingCanonicalHash);
  }, [incomingCanonicalHash, paperView]);

  useEffect(() => {
    if (paper.sync_state === "behind" || paperView !== "incoming") return;
    setPaperView("write");
    try {
      localStorage.setItem(`${PAPER_VIEW_STORAGE_PREFIX}:${project.id}`, "write");
    } catch {}
  }, [paper.sync_state, paperView, project.id]);

  const paperCreated = paper.sync_state !== "not_created";
  useEffect(() => {
    const element = workspace.current;
    if (!element) return;

    const updateBounds = (width: number) => {
      const bounds = editorShareBoundsForWidth(width);
      setEditorShareBounds(bounds);
      setEditorShare((current) => clampEditorShare(current, bounds));
    };
    updateBounds(element.getBoundingClientRect().width);

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) updateBounds(entry.contentRect.width);
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, [paperCreated]);

  if (paper.sync_state === "not_created") {
    return (
      <section className="paper-empty">
        <div className="paper-sheet-preview">
          <span># Introduction</span>
          <i />
          <i />
          <i />
        </div>
        <div className="paper-empty-action">
          {saveError && (
            <div className="paper-inline-error" role="alert">
              {saveError}
            </div>
          )}
          <button className="button primary" disabled={creating} onClick={() => void create()}>
            <FilePenLine size={15} /> {creating ? "Opening editor…" : "Create introduction"}
          </button>
        </div>
      </section>
    );
  }

  return (
    <section
      className="paper-workspace"
      ref={workspace}
      style={{
        gridTemplateColumns: `minmax(${MIN_EDITOR_WIDTH}px, ${editorShare}%) ${DIVIDER_WIDTH}px minmax(${MIN_COACH_WIDTH}px, 1fr)`,
      }}
    >
      <div className="paper-editor-column">
        <header className="paper-toolbar">
          <div className="paper-view-controls">
            <div className="paper-view-toggle" role="group" aria-label="Paper view">
              {(["write", "preview"] as const).map((view) => (
                <button
                  aria-pressed={paperView === view}
                  key={view}
                  onClick={() => selectPaperView(view)}
                  type="button"
                >
                  {view === "write" ? "Write" : "Preview"}
                </button>
              ))}
              {paper.sync_state === "behind" && (
                <button
                  aria-pressed={paperView === "incoming"}
                  onClick={() => selectPaperView("incoming")}
                  type="button"
                >
                  Incoming
                </button>
              )}
            </div>
            {paper.sync_state === "behind" && (
              <button
                aria-label="Swap editor and incoming introduction"
                className="button compact secondary paper-apply-incoming"
                disabled={saving}
                onClick={applyIncoming}
                type="button"
              >
                Apply
              </button>
            )}
          </div>
          <div className="paper-status">
            <span>{wordCount} words</span>
            <span className={`sync-state ${saveError ? "save-error" : paper.sync_state}`}>
              {!saveError && paper.sync_state === "synced" && <Check size={13} />}
              {!saveError && paper.sync_state === "unsynced" && <WifiOff size={13} />}
              {(saveError || paper.sync_state === "behind") && <AlertTriangle size={13} />}
              {saveError
                ? "save failed"
                : saving
                  ? "saving"
                  : dirty
                    ? "unsaved changes"
                    : syncLabel}
            </span>
          </div>
        </header>
        {saveError && (
          <div className="paper-save-error" role="alert">
            <AlertTriangle size={16} />
            <span>
              <strong>Your text is still in this editor.</strong> {saveError}
            </span>
            <button className="button compact secondary" onClick={() => setSaveError(null)}>
              Retry save
            </button>
          </div>
        )}
        {paperView === "write" ? (
          <textarea
            ref={textarea}
            className="markdown-editor"
            aria-label="Paper introduction Markdown"
            spellCheck
            value={content}
            onChange={(event) => {
              setContent(event.target.value);
              setDirty(true);
              setSaveError(null);
            }}
          />
        ) : (
          <article
            aria-label="Paper introduction preview"
            className="markdown-editor paper-markdown-preview chat-markdown"
          >
            <MarkdownAnswer text={paperView === "incoming" ? incomingContent : content} />
          </article>
        )}
      </div>

      <div
        aria-label="Resize paper editor and writing coach"
        aria-orientation="vertical"
        aria-valuemax={editorShareBounds.maximum}
        aria-valuemin={editorShareBounds.minimum}
        aria-valuenow={Math.round(editorShare)}
        className="paper-divider"
        onKeyDown={(event) => {
          if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
            event.preventDefault();
            setEditorShare((current) =>
              constrainEditorShare(current + (event.key === "ArrowLeft" ? -2 : 2)),
            );
          }
          if (event.key === "Home" || event.key === "End") {
            event.preventDefault();
            setEditorShare(
              constrainEditorShare(event.key === "Home" ? MIN_EDITOR_SHARE : MAX_EDITOR_SHARE),
            );
          }
        }}
        onPointerDown={(event) => {
          event.currentTarget.setPointerCapture(event.pointerId);
          resizeFromPointer(event.clientX);
        }}
        onPointerMove={(event) => {
          if (event.currentTarget.hasPointerCapture(event.pointerId))
            resizeFromPointer(event.clientX);
        }}
        onPointerUp={(event) => {
          if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            event.currentTarget.releasePointerCapture(event.pointerId);
          }
        }}
        role="separator"
        tabIndex={0}
      />

      <aside className="coach-column">
        <header className="coach-header">
          <h2>Writing coach</h2>
          <button
            className="button ghost compact"
            onClick={() => {
              setActiveSession(null);
              setFreshSession(true);
              setPendingCoachTaskId(null);
              setConfig(profileRunConfig(project.agent_profiles.paper_coach));
              skills.reset();
              setSubmitError(null);
            }}
          >
            <MessageSquarePlus size={14} /> New chat
          </button>
        </header>
        <details className="session-history">
          <summary>
            <History size={14} />
            <strong>Chat history</strong>
            <small>{sessions.length}</small>
          </summary>
          <div className="session-strip">
            {sessions.length === 0 ? (
              <span className="muted">No prior writing sessions</span>
            ) : (
              sessions.map((session) => (
                <button
                  className={
                    activeSession?.native_session_id === session.native_session_id ? "active" : ""
                  }
                  key={session.native_session_id}
                  onClick={() => {
                    attachSession(session);
                  }}
                >
                  <span>{session.title || "Untitled coach session"}</span>
                  <span className="session-meta">
                    {session.provider}
                    {session.runtime_label ? ` · ${session.runtime_label}` : ""} · rev{" "}
                    {session.graph_revision_examined}
                  </span>
                  <ChevronRight size={13} />
                </button>
              ))
            )}
          </div>
        </details>

        <div
          className="agent-provider-label"
          aria-busy={readiness === undefined}
          aria-label={`Writing coach provider: ${readiness?.label || config.provider}`}
        >
          {readiness?.label || config.provider}
          {readiness === undefined && (
            <LoaderCircle className="spin" size={12} aria-label="Checking provider" />
          )}
        </div>
        {reviewStale && (
          <div className="stale-review">
            <AlertTriangle size={14} /> Project understanding changed since this session reviewed
            the draft.
          </div>
        )}
        {readiness && !readiness.authenticated && (
          <div className="provider-unavailable">
            <WifiOff size={14} /> {readiness?.reason || `${config.provider} is unavailable.`}
          </div>
        )}

        <div className="coach-transcript" aria-live="polite">
          {transcript.map((line, index) => (
            <div
              className={`chat-line ${line.role === "agent" ? "coach" : line.role}`}
              key={`${line.taskId}-${index}`}
            >
              {line.role === "human" ? `You: ${line.text}` : line.text}
            </div>
          ))}
          {submitError && <div className="chat-line error">{submitError}</div>}
          {coachActive && (
            <div className="thinking">
              <i />
              <i />
              <i /> {activeCoachTask?.status_message || "Writing coach is running"}
            </div>
          )}
        </div>
        <div className="coach-composer">
          <SkillPicker {...skills.props} />
          <textarea
            ref={coachTextarea}
            aria-label="Message"
            value={message}
            onChange={(event) => updateMessage(event.target.value)}
            onKeyDown={(event) => {
              if (skills.handleKeyDown(event)) return;
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void send();
              }
            }}
          />
          <button
            className="icon-button primary"
            aria-label={
              activeSession ? "Resume session in background" : "Start coach task in background"
            }
            disabled={
              !message.trim() ||
              Boolean(activeCoachTask) ||
              coachSubmitting ||
              (readiness !== undefined && !readiness.authenticated)
            }
            onClick={() => void send()}
          >
            <Send size={16} />
          </button>
        </div>
      </aside>
    </section>
  );
}
