import { Check, ChevronDown, Copy, Link2, Pencil, UserPlus, UserRound } from "lucide-react";
import { useEffect, useId, useRef, useState } from "react";
import { createTeamInvitation, loadTeamInvitations } from "../api";
import type { IdentityResponse, TeamInvitation, TeamInvitationIssue } from "../types";

interface Props {
  identity: IdentityResponse | null;
  identityError: string | null;
  onRequestName: () => Promise<boolean> | void;
  onAddTeamSpace?: () => void;
}

interface IdentityProvenanceSlipProps {
  identity: IdentityResponse;
  identityError: string | null;
  teamNoticeId: string;
  copyStatus: "idle" | "copied" | "failed";
  onCopy: () => void;
  onEdit: () => void;
  teamPanelActive?: boolean;
  onAddTeamSpace?: () => void;
}

export async function copyIdentityId(
  userId: string,
  clipboard: Pick<Clipboard, "writeText"> | undefined = typeof navigator === "undefined"
    ? undefined
    : navigator.clipboard,
): Promise<void> {
  if (!clipboard) throw new Error("Clipboard access is unavailable.");
  await clipboard.writeText(userId);
}

export function IdentityProvenanceSlip({
  identity,
  identityError,
  teamNoticeId,
  copyStatus,
  onCopy,
  onEdit,
  teamPanelActive = true,
  onAddTeamSpace,
}: IdentityProvenanceSlipProps) {
  const displayName = identity.user.display_name ?? "";
  const spaceLabel = identity.space_kind === "personal" ? "Personal space" : "Team space";

  return (
    <>
      <div className="landing-identity-slip" data-identity-record="provenance-slip">
        <header>
          <span>Identity record</span>
          <span>{identity.user.identity_kind === "local_owner" ? "Local" : "Member"}</span>
        </header>
        <div className="landing-identity-slip-person">
          <span className="landing-identity-avatar large" aria-hidden="true">
            {identityInitial(displayName)}
          </span>
          <span>
            <strong>{displayName}</strong>
            <small>{spaceLabel}</small>
          </span>
          <button
            className="landing-identity-edit"
            type="button"
            data-identity-action="edit"
            onClick={onEdit}
          >
            <Pencil size={12} aria-hidden="true" />
            Edit
          </button>
        </div>
        <dl>
          <div>
            <dt>User ID</dt>
            <dd>
              <code tabIndex={0} aria-label={`User ID ${identity.user.user_id}`}>
                {identity.user.user_id}
              </code>
              <button
                className="landing-identity-copy"
                type="button"
                data-identity-action="copy-id"
                aria-label="Copy user ID"
                onClick={onCopy}
              >
                {copyStatus === "copied" ? (
                  <Check size={12} aria-hidden="true" />
                ) : (
                  <Copy size={12} aria-hidden="true" />
                )}
                {copyStatus === "copied" ? "Copied" : "Copy"}
              </button>
            </dd>
          </div>
          <div>
            <dt>Scope</dt>
            <dd>{spaceLabel}</dd>
          </div>
        </dl>
        {copyStatus === "failed" && (
          <p className="landing-identity-copy-error" role="alert">
            User ID could not be copied. Select it above to copy manually.
          </p>
        )}
        {identityError && (
          <p className="landing-identity-panel-error" role="alert">
            {identityError}
          </p>
        )}
      </div>

      {identity.space_kind === "team" ? (
        <TeamInvitationPanel identity={identity} active={teamPanelActive} />
      ) : (
        <PersonalTeamSeam noticeId={teamNoticeId} onAddTeamSpace={onAddTeamSpace} />
      )}
    </>
  );
}

export function PersonalTeamSeam({
  noticeId,
  onAddTeamSpace,
}: {
  noticeId: string;
  onAddTeamSpace?: () => void;
}) {
  return (
    <section
      className="landing-team-seam"
      aria-labelledby={`${noticeId}-title`}
      data-team-space-seam="available"
    >
      <header>
        <span id={`${noticeId}-title`}>Team spaces</span>
        <span>Desktop</span>
      </header>
      <div className="landing-team-seam-actions">
        {onAddTeamSpace && (
          <button type="button" onClick={onAddTeamSpace}>
            <Link2 size={13} aria-hidden="true" />
            Add team space
          </button>
        )}
      </div>
    </section>
  );
}

export function TeamInvitationPanel({
  identity,
  active = true,
}: {
  identity: IdentityResponse;
  active?: boolean;
}) {
  const [invitations, setInvitations] = useState<TeamInvitation[]>([]);
  const [issued, setIssued] = useState<TeamInvitationIssue | null>(null);
  const [loading, setLoading] = useState(false);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copyStatus, setCopyStatus] = useState<"idle" | "copied" | "failed">("idle");
  const titleId = useId();

  useEffect(() => {
    if (!active) {
      // A raw invitation code is shown once, at the moment it is created.
      // Closing the panel discards it, so reopening shows only metadata.
      setIssued(null);
      setCopyStatus("idle");
      return;
    }
    let stopped = false;
    setLoading(true);
    setError(null);
    void loadTeamInvitations()
      .then((next) => {
        if (!stopped) setInvitations(next);
      })
      .catch(() => {
        if (!stopped) setError("Invitations could not be loaded. Try opening this panel again.");
      })
      .finally(() => {
        if (!stopped) setLoading(false);
      });
    return () => {
      stopped = true;
    };
  }, [active, identity.user.user_id]);

  const createInvitation = async () => {
    if (creating) return;
    setCreating(true);
    setError(null);
    setCopyStatus("idle");
    try {
      const next = await createTeamInvitation();
      setIssued(next);
      setInvitations((current) => [
        next.invitation,
        ...current.filter(
          (invitation) => invitation.invitation_id !== next.invitation.invitation_id,
        ),
      ]);
    } catch {
      setError("An invitation could not be created. Try again.");
    } finally {
      setCreating(false);
    }
  };

  const copyInvitation = async () => {
    if (!issued) return;
    try {
      await copyIdentityId(invitationCopyBlock(issued));
      setCopyStatus("copied");
    } catch {
      setCopyStatus("failed");
    }
  };

  return (
    <section className="landing-team-invitations" aria-labelledby={titleId}>
      <header>
        <span id={titleId}>Team invitations</span>
        <span>{identity.space_name || "Team space"}</span>
      </header>
      <button
        className="landing-team-invite-action"
        type="button"
        disabled={creating}
        onClick={() => void createInvitation()}
      >
        <UserPlus size={13} aria-hidden="true" />
        {creating ? "Creating" : "Invite member"}
      </button>

      {issued && (
        <div className="landing-team-invitation-code" aria-live="polite">
          <div>
            <span>Invitation for</span>
            <strong>{issued.space_name}</strong>
          </div>
          <code tabIndex={0} aria-label={`Invitation code ${issued.code}`}>
            {issued.code}
          </code>
          <div className="landing-team-invitation-expiry">
            Expires {formatInvitationTime(issued.invitation.expires_at)}
          </div>
          <button type="button" onClick={() => void copyInvitation()}>
            {copyStatus === "copied" ? (
              <Check size={12} aria-hidden="true" />
            ) : (
              <Copy size={12} aria-hidden="true" />
            )}
            {copyStatus === "copied" ? "Copied" : "Copy invitation"}
          </button>
          {copyStatus === "failed" && (
            <p role="alert">Invitation could not be copied. Select the code to copy it manually.</p>
          )}
        </div>
      )}

      {error && (
        <p className="landing-team-invitation-error" role="alert">
          {error}
        </p>
      )}

      <TeamInvitationLedger invitations={invitations} loading={loading} />
    </section>
  );
}

export function TeamInvitationLedger({
  invitations,
  loading = false,
}: {
  invitations: TeamInvitation[];
  loading?: boolean;
}) {
  return (
    <div className="landing-team-invitation-ledger" aria-busy={loading}>
      <span>Created by you</span>
      {!loading && invitations.length === 0 && <p>No invitations created yet.</p>}
      {invitations.length > 0 && (
        <ul>
          {invitations.map((invitation) => (
            <li key={invitation.invitation_id}>
              <span>{invitationStatus(invitation)}</span>
              <time dateTime={invitation.expires_at}>
                Expires {formatInvitationTime(invitation.expires_at)}
              </time>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function invitationCopyBlock(issue: TeamInvitationIssue): string {
  return `${issue.space_name}\n${issue.code}\nExpires ${formatInvitationTime(issue.invitation.expires_at)}`;
}

function invitationStatus(invitation: TeamInvitation): string {
  if (invitation.consumed_at) return "Used";
  if (invitation.locked_at) return "Locked";
  if (new Date(invitation.expires_at).getTime() <= Date.now()) return "Expired";
  return "Available";
}

function formatInvitationTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

export function LandingIdentityMenu({
  identity,
  identityError,
  onRequestName,
  onAddTeamSpace,
}: Props) {
  const [open, setOpen] = useState(false);
  const [copyStatus, setCopyStatus] = useState<"idle" | "copied" | "failed">("idle");
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const panelId = useId();
  const teamNoticeId = useId();
  const displayName = identity?.user.display_name?.trim() ?? "";
  const namedIdentity = identity && displayName ? identity : null;
  const spaceLabel = identity?.space_kind === "team" ? "Team space" : "Personal space";

  useEffect(() => {
    if (namedIdentity) return;
    setOpen(false);
    setCopyStatus("idle");
  }, [namedIdentity]);

  useEffect(() => {
    if (!open) return;
    const closeOnPointerDown = (event: PointerEvent) => {
      if (event.target instanceof Node && rootRef.current?.contains(event.target)) return;
      setOpen(false);
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      setOpen(false);
      triggerRef.current?.focus();
    };
    document.addEventListener("pointerdown", closeOnPointerDown);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOnPointerDown);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [open]);

  const requestName = () => {
    setOpen(false);
    void onRequestName();
  };

  const copyUserId = async () => {
    if (!namedIdentity) return;
    try {
      await copyIdentityId(namedIdentity.user.user_id);
      setCopyStatus("copied");
    } catch {
      setCopyStatus("failed");
    }
  };

  return (
    <div className={`landing-identity-menu${identityError ? " has-error" : ""}`} ref={rootRef}>
      <button
        className="landing-identity-trigger"
        type="button"
        ref={triggerRef}
        aria-haspopup={namedIdentity ? "dialog" : undefined}
        aria-expanded={namedIdentity ? open : undefined}
        aria-controls={namedIdentity ? panelId : undefined}
        onClick={() => {
          if (!namedIdentity) {
            requestName();
            return;
          }
          setCopyStatus("idle");
          setOpen((current) => !current);
        }}
      >
        <span className="landing-identity-avatar" aria-hidden="true">
          {namedIdentity ? identityInitial(displayName) : <UserRound size={14} />}
        </span>
        <span className="landing-identity-trigger-copy">
          <strong>{namedIdentity ? displayName : "Sign in"}</strong>
          {namedIdentity && <small>{spaceLabel}</small>}
        </span>
        {namedIdentity && <ChevronDown size={13} aria-hidden="true" />}
      </button>

      {identityError && (
        <span className="landing-identity-trigger-error" role="alert">
          {identityError}
        </span>
      )}

      {namedIdentity && (
        <section
          className="landing-identity-panel"
          id={panelId}
          role="dialog"
          aria-modal="false"
          aria-label="Your identity and spaces"
          hidden={!open}
        >
          <IdentityProvenanceSlip
            identity={namedIdentity}
            identityError={identityError}
            teamNoticeId={teamNoticeId}
            copyStatus={copyStatus}
            onCopy={() => void copyUserId()}
            onEdit={requestName}
            teamPanelActive={open}
            onAddTeamSpace={onAddTeamSpace}
          />
        </section>
      )}
    </div>
  );
}

function identityInitial(displayName: string): string {
  return Array.from(displayName.trim())[0]?.toLocaleUpperCase() ?? "?";
}
