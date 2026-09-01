import { Mail, MoreHorizontal, Server, Trash2, WifiOff } from "lucide-react";
import { useEffect, useState } from "react";
import { ExperimentBoard } from "../components/ExperimentBoard";
import { LandingIdentityMenu } from "../components/LandingIdentityMenu";
import { ProjectDock } from "../components/ProjectDock";
import { TeamSpaceGroups } from "../components/TeamSpaceGroups";
import { isDesktopRuntime } from "../desktopRuntime";
import type { ProjectTab } from "../projectTabs";
import type {
  ExperimentLoopIndexEntry,
  IdentityResponse,
  ProjectCard,
  ProjectCreationControl,
  ProjectInvitation,
} from "../types";
import { projectCreationPrimaryLabel } from "../projectSetup";

interface Props {
  projects: ProjectCard[];
  invitations: ProjectInvitation[];
  onAnswerInvitation: (invitationId: string, response: "accept" | "decline") => Promise<void>;
  experimentLoops: ExperimentLoopIndexEntry[];
  onOpen: (projectId: string) => void;
  onOpenExperiment: (projectId: string, experimentRoute: string) => void;
  onCreate: () => void;
  projectCreation: ProjectCreationControl;
  onDelete: (projectId: string) => Promise<void> | void;
  openProjectTabs: ProjectTab[];
  onActivateProjectTab: (projectId: string) => void;
  onCloseProjectTab: (projectId: string) => void;
  identity: IdentityResponse | null;
  identityError: string | null;
  onRequestIdentityName: () => Promise<boolean> | void;
}

const COVER_STYLES = ["plain", "dye", "mosaic", "wood", "marble", "diffusion"] as const;
type CoverStyle = (typeof COVER_STYLES)[number];

const COVER_TONES = [
  "oxblood",
  "teal",
  "mustard",
  "coral",
  "indigo",
  "plum",
  "moss",
  "slate",
] as const;

const COVER_LABELS: Record<CoverStyle, string> = {
  plain: "Plain",
  dye: "Tie-dye",
  mosaic: "Mosaic",
  wood: "Wood",
  marble: "Marble",
  diffusion: "Diffusion",
};

interface ProjectActionsMenuProps {
  project: ProjectCard;
  cover: CoverStyle;
  onChooseCover: (cover: CoverStyle) => void;
  onDelete: () => void;
}

export function ProjectActionsMenu({
  project,
  cover,
  onChooseCover,
  onDelete,
}: ProjectActionsMenuProps) {
  return (
    <div className="project-cover-menu" role="menu" aria-label={`Actions for ${project.name}`}>
      <span className="project-cover-menu-label" role="presentation">
        Cover
      </span>
      <div className="project-cover-options" role="presentation">
        {COVER_STYLES.map((style) => (
          <button
            className="project-cover-option"
            type="button"
            role="menuitemradio"
            key={style}
            aria-checked={cover === style}
            onClick={() => onChooseCover(style)}
          >
            <span className="project-cover-swatch">
              <span
                className={`project-cover-swatch-zoom project-material-${style}`}
                aria-hidden="true"
              />
            </span>
            <span className="project-cover-option-label">{COVER_LABELS[style]}</span>
          </button>
        ))}
      </div>
      {project.can_delete && (
        <button className="project-delete-action" type="button" role="menuitem" onClick={onDelete}>
          <Trash2 size={13} aria-hidden="true" />
          Delete project
        </button>
      )}
    </div>
  );
}

export function ProjectLanding({
  projects,
  invitations,
  onAnswerInvitation,
  experimentLoops,
  onOpen,
  onOpenExperiment,
  onCreate,
  projectCreation,
  onDelete,
  openProjectTabs,
  onActivateProjectTab,
  onCloseProjectTab,
  identity,
  identityError,
  onRequestIdentityName,
}: Props) {
  const desktop = isDesktopRuntime();
  const [covers, setCovers] = useState<Record<string, CoverStyle>>(() => readCoverPreferences());
  const [openMenuProject, setOpenMenuProject] = useState<string | null>(null);
  const [deleteProjectId, setDeleteProjectId] = useState<string | null>(null);
  const [deleteBusy, setDeleteBusy] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [addTeamOpen, setAddTeamOpen] = useState(false);

  useEffect(() => {
    try {
      localStorage.setItem("rcp:project-covers", JSON.stringify(covers));
    } catch {
      // Cover choices are a convenience; storage failures must not affect the project list.
    }
  }, [covers]);

  useEffect(() => {
    const closeOnPointerDown = (event: PointerEvent) => {
      if (!(event.target instanceof Element) || event.target.closest(".project-cover-shell"))
        return;
      setOpenMenuProject(null);
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      setOpenMenuProject(null);
      if (!deleteBusy) {
        setDeleteProjectId(null);
        setDeleteError(null);
      }
    };
    document.addEventListener("pointerdown", closeOnPointerDown);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOnPointerDown);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [deleteBusy]);

  const deleteProject =
    projects.find((project) => project.id === deleteProjectId && project.can_delete) ?? null;

  const closeDeleteConfirmation = () => {
    if (deleteBusy) return;
    setDeleteProjectId(null);
    setDeleteError(null);
  };

  const confirmDelete = async () => {
    if (!deleteProject || deleteBusy) return;
    setDeleteBusy(true);
    setDeleteError(null);
    try {
      await onDelete(deleteProject.id);
      setCovers((current) => {
        const next = { ...current };
        delete next[deleteProject.id];
        return next;
      });
      setDeleteProjectId(null);
    } catch (error) {
      setDeleteError(error instanceof Error ? error.message : String(error));
    } finally {
      setDeleteBusy(false);
    }
  };

  return (
    <div className="landing-shell">
      <header className="landing-header">
        <a className="rcp-mark" href="#" aria-label="RCP project index">
          <span className="rcp-wordmark" aria-hidden="true">
            RCP
          </span>
        </a>
        <ProjectDock
          tabs={openProjectTabs}
          activeProjectId={null}
          onActivate={onActivateProjectTab}
          onClose={onCloseProjectTab}
        />
        <LandingIdentityMenu
          identity={identity}
          identityError={identityError}
          onRequestName={onRequestIdentityName}
          onAddTeamSpace={desktop ? () => setAddTeamOpen(true) : undefined}
        />
      </header>

      <main className="landing-main">
        {identity?.space_kind === "personal" && (
          <h1 className="space-group-title">Personal space</h1>
        )}
        <section className="project-shelf" aria-label="RCP projects">
          {invitations.map((invitation) => (
            <ProjectInvitationCard
              key={invitation.invitation_id}
              invitation={invitation}
              onAnswer={onAnswerInvitation}
            />
          ))}
          {projects.map((project, index) => {
            const unavailable = project.reachable === false;
            const cover = covers[project.id] || "wood";
            const tone = COVER_TONES[index % COVER_TONES.length];
            return (
              <div className={`project-cover-shell project-cover-tone-${tone}`} key={project.id}>
                <button
                  className={`project-cover project-material-${cover}`}
                  onClick={() => onOpen(project.id)}
                >
                  <span className="project-cover-spine" aria-hidden="true" />
                  <span
                    className={unavailable ? "project-cover-state offline" : "project-cover-state"}
                  >
                    {unavailable ? <WifiOff size={12} /> : <Server size={12} />}
                    {unavailable ? "Cached" : project.remote ? "Remote" : "Local"}
                  </span>
                  <strong>{project.name}</strong>
                  <span className="project-cover-meta">
                    {project.revision == null ? "Not opened" : `Revision ${project.revision}`}
                    {project.attention_count > 0 && <> · {project.attention_count} waiting</>}
                    {project.last_opened_at && <> · {formatReturn(project.last_opened_at)}</>}
                  </span>
                  <span className="project-cover-open" aria-hidden="true">
                    Open →
                  </span>
                </button>
                <button
                  className="project-cover-trigger"
                  type="button"
                  aria-haspopup="menu"
                  aria-expanded={openMenuProject === project.id}
                  aria-label={`Project actions for ${project.name}`}
                  title="Project actions"
                  onClick={() =>
                    setOpenMenuProject((current) => (current === project.id ? null : project.id))
                  }
                >
                  <MoreHorizontal size={14} aria-hidden="true" />
                </button>
                {openMenuProject === project.id && (
                  <ProjectActionsMenu
                    project={project}
                    cover={cover}
                    onChooseCover={(style) => {
                      setCovers((current) => ({ ...current, [project.id]: style }));
                      setOpenMenuProject(null);
                    }}
                    onDelete={() => {
                      setOpenMenuProject(null);
                      setDeleteError(null);
                      setDeleteProjectId(project.id);
                    }}
                  />
                )}
              </div>
            );
          })}

          <button className="project-cover new-project-cover" onClick={onCreate}>
            <span className="new-project-plus" aria-hidden="true">
              +
            </span>
            <strong>{projectCreationPrimaryLabel(projectCreation)}</strong>
            <span className="project-cover-open" aria-hidden="true">
              Create →
            </span>
          </button>
        </section>

        {identity?.space_kind === "personal" && (
          <TeamSpaceGroups
            addOpen={addTeamOpen}
            onOpenAdd={() => setAddTeamOpen(true)}
            onCloseAdd={() => setAddTeamOpen(false)}
          />
        )}

        <ExperimentBoard entries={experimentLoops} onOpen={onOpenExperiment} />
      </main>

      {deleteProject && (
        <div
          className="modal-backdrop"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) closeDeleteConfirmation();
          }}
        >
          <section
            className="project-delete-dialog"
            role="alertdialog"
            aria-modal="true"
            aria-labelledby="project-delete-title"
            aria-describedby="project-delete-warning"
          >
            <header>
              <Trash2 size={18} aria-hidden="true" />
              <h2 id="project-delete-title">Delete {deleteProject.name}?</h2>
            </header>
            <p id="project-delete-warning">
              RCP records, task history, and staged run data for {deleteProject.name} will be
              permanently erased. Repositories and their <code>.research</code> directories remain
              untouched. Paused, interrupted, failed, and completed history will become unreachable.
            </p>
            {deleteError && (
              <div className="project-delete-error" role="alert">
                {deleteError}
              </div>
            )}
            <footer>
              <button
                className="button secondary"
                type="button"
                autoFocus
                disabled={deleteBusy}
                onClick={closeDeleteConfirmation}
              >
                Cancel
              </button>
              <button
                className="button danger"
                type="button"
                disabled={deleteBusy}
                onClick={confirmDelete}
              >
                {deleteBusy ? "Deleting…" : "Delete project"}
              </button>
            </footer>
          </section>
        </div>
      )}
    </div>
  );
}

function ProjectInvitationCard({
  invitation,
  onAnswer,
}: {
  invitation: ProjectInvitation;
  onAnswer: (invitationId: string, response: "accept" | "decline") => Promise<void>;
}) {
  const [busy, setBusy] = useState<"accept" | "decline" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const answer = async (response: "accept" | "decline") => {
    setBusy(response);
    setError(null);
    try {
      await onAnswer(invitation.invitation_id, response);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
      setBusy(null);
    }
  };
  const inviter = invitation.invited_by_name || invitation.invited_by;
  return (
    <div className="project-cover-shell project-invitation-shell">
      <article className="project-cover project-invitation">
        <span className="project-cover-spine" aria-hidden="true" />
        <span className="project-cover-state">
          <Mail size={12} />
          Invitation
        </span>
        <strong>{invitation.project_name}</strong>
        <span className="project-cover-meta">
          {invitation.space_name ? `${invitation.space_name} · ` : ""}
          {inviter}
        </span>
        {error ? <p className="project-invitation-error">{error}</p> : null}
        <span className="project-invitation-actions">
          <button type="button" disabled={busy !== null} onClick={() => answer("accept")}>
            {busy === "accept" ? "Accepting…" : "Accept"}
          </button>
          <button type="button" disabled={busy !== null} onClick={() => answer("decline")}>
            {busy === "decline" ? "Declining…" : "Decline"}
          </button>
        </span>
      </article>
    </div>
  );
}

function formatReturn(timestamp?: string | null): string {
  if (!timestamp) return "";
  return new Date(timestamp).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

function readCoverPreferences(): Record<string, CoverStyle> {
  try {
    const parsed: unknown = JSON.parse(localStorage.getItem("rcp:project-covers") || "{}");
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return {};
    const covers: Record<string, CoverStyle> = {};
    for (const [projectId, style] of Object.entries(parsed)) {
      if (isCoverStyle(style)) covers[projectId] = style;
    }
    return covers;
  } catch {
    return {};
  }
}

function isCoverStyle(value: unknown): value is CoverStyle {
  return typeof value === "string" && (COVER_STYLES as readonly string[]).includes(value);
}
