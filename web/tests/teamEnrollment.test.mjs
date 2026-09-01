import assert from "node:assert/strict";
import { after, test } from "node:test";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";

const server = await createServer({
  root: new URL("..", import.meta.url).pathname,
  configFile: false,
  logLevel: "silent",
  server: { middlewareMode: true, hmr: false },
  optimizeDeps: { noDiscovery: true },
});
const { ApiError } = await server.ssrLoadModule("/src/api.ts");
const { TeamLoginBoundary, teamLoginFailureMessage } = await server.ssrLoadModule(
  "/src/components/TeamLoginBoundary.tsx",
);
const { IdentityProvenanceSlip, TeamInvitationLedger, invitationCopyBlock } =
  await server.ssrLoadModule("/src/components/LandingIdentityMenu.tsx");

after(() => server.close());

const teamIdentity = {
  space_id: "123e4567-e89b-42d3-a456-426614174000",
  space_kind: "team",
  space_name: "Causal Systems Lab",
  user: {
    user_id: "123e4567-e89b-42d3-a456-426614174001",
    display_name: "Ada Researcher",
    identity_kind: "team_member",
    created_at: "2026-08-12T00:00:00Z",
    updated_at: "2026-08-12T00:00:00Z",
  },
};

// Relative to now on purpose. A literal expiry silently turns this into a
// time bomb: the ledger renders "Expired" once the date passes and the test
// starts failing on a calendar day rather than on a code change.
const DAY_MS = 24 * 60 * 60 * 1000;
const invitation = {
  invitation_id: "invite-a",
  created_by: teamIdentity.user.user_id,
  created_at: new Date(Date.now() - 7 * DAY_MS).toISOString(),
  expires_at: new Date(Date.now() + 7 * DAY_MS).toISOString(),
  consumed_at: null,
  consumed_by: null,
  failed_attempts: 0,
  locked_at: null,
};

test("team login uses a focused secret field without a URL or storage seam", () => {
  const html = renderToStaticMarkup(
    React.createElement(TeamLoginBoundary, {
      spaceName: teamIdentity.space_name,
      async onAuthenticate() {},
    }),
  );

  assert.match(html, /Sign in to Causal Systems Lab/);
  assert.match(html, /<input[^>]*type="password"/);
  assert.match(html, /data-team-login="credential-slip"/);
  assert.match(html, /<form[^>]*autoComplete="off"/);
  assert.doesNotMatch(html, /action=|localStorage|sessionStorage|[?&](token|code)=/i);
});

test("team login errors never reflect the submitted token", () => {
  const rawToken = "rcp_super-secret-member-token";
  const rejected = teamLoginFailureMessage(new ApiError(rawToken, 401));
  const unavailable = teamLoginFailureMessage(new Error(rawToken));

  assert.match(rejected, /not accepted/);
  assert.doesNotMatch(rejected, new RegExp(rawToken));
  assert.doesNotMatch(unavailable, new RegExp(rawToken));
});

test("an authenticated team member gets the active invitation seam", () => {
  const html = renderToStaticMarkup(
    React.createElement(IdentityProvenanceSlip, {
      identity: teamIdentity,
      identityError: null,
      teamNoticeId: "team-status",
      copyStatus: "idle",
      onCopy() {},
      onEdit() {},
      teamPanelActive: false,
    }),
  );

  assert.match(html, /Team invitations/);
  assert.match(html, /Causal Systems Lab/);
  assert.match(html, /<button[^>]*>.*Invite member/s);
  assert.doesNotMatch(html, /data-team-space-seam="unimplemented"/);
  assert.doesNotMatch(html, /Join team space|Accept invitation|type="password"/);
});

test("invitation metadata is visible without retaining raw codes in the ledger", () => {
  const ledger = renderToStaticMarkup(
    React.createElement(TeamInvitationLedger, { invitations: [invitation] }),
  );
  const rawCode = "rcp_invitation-secret";
  const copyBlock = invitationCopyBlock({
    invitation,
    code: rawCode,
    space_name: teamIdentity.space_name,
  });

  assert.match(ledger, /Created by you/);
  assert.match(ledger, /Available/);
  assert.match(ledger, /Expires/);
  assert.doesNotMatch(ledger, new RegExp(rawCode));
  assert.match(copyBlock, /Causal Systems Lab/);
  assert.match(copyBlock, new RegExp(rawCode));
  assert.match(copyBlock, /Expires/);
});
