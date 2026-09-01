import assert from "node:assert/strict";
import { after, beforeEach, test } from "node:test";
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
const { ProjectActionsMenu, ProjectLanding } = await server.ssrLoadModule(
  "/src/views/ProjectLanding.tsx",
);
const { IdentityProvenanceSlip, copyIdentityId } = await server.ssrLoadModule(
  "/src/components/LandingIdentityMenu.tsx",
);

after(() => server.close());

beforeEach(() => {
  globalThis.localStorage = {
    getItem: () => null,
    setItem() {},
    removeItem() {},
  };
});

const userId = "123e4567-e89b-42d3-a456-426614174001";
const identity = {
  space_id: "123e4567-e89b-42d3-a456-426614174000",
  space_kind: "personal",
  user: {
    user_id: userId,
    display_name: "Ada Researcher",
    identity_kind: "local_owner",
    created_at: "2026-08-12T00:00:00Z",
    updated_at: "2026-08-12T00:00:00Z",
  },
};

function landingProps(identityValue = identity) {
  return {
    projects: [],
    invitations: [],
    async onAnswerInvitation() {},
    experimentLoops: [],
    onOpen() {},
    onOpenExperiment() {},
    onCreate() {},
    projectCreation: {
      requires_authenticated_member: false,
      intents: [
        {
          intent: "use_existing_checkout_personally",
          eligible: true,
          preselected: true,
          primary_action_label: "Use existing checkout",
          required_fields: ["repositories"],
          pinned_source_project_id: null,
          unavailable_reason: null,
        },
      ],
    },
    onDelete() {},
    openProjectTabs: [],
    onActivateProjectTab() {},
    onCloseProjectTab() {},
    identity: identityValue,
    identityError: null,
    onRequestIdentityName() {},
  };
}

test("the project index uses the backend-selected creation label", () => {
  const html = renderToStaticMarkup(React.createElement(ProjectLanding, landingProps()));

  assert.match(html, /Use existing checkout/);
  assert.doesNotMatch(html, />New project</);
});

test("the project index starts with covers and exposes the named identity with its full ID", () => {
  const html = renderToStaticMarkup(React.createElement(ProjectLanding, landingProps()));

  assert.doesNotMatch(html, /Choose a project/i);
  assert.match(html, /Ada Researcher/);
  assert.match(html, /Personal space/);
  assert.match(html, new RegExp(userId));
  assert.match(html, /data-identity-record="provenance-slip"/);
});

test("the project menu renders Delete only from the backend-owned card decision", () => {
  const personal = {
    id: "project-1",
    home_space_id: identity.space_id,
    name: "Personal paper",
    locator: "/tmp/personal/.research/manifest.toml",
    state_location: "/tmp/personal",
    remote: false,
    attention_count: 0,
    can_delete: true,
    delete_unavailable_reason: null,
  };
  const team = {
    ...personal,
    id: "project-2",
    name: "Team paper",
    can_delete: false,
    delete_unavailable_reason:
      "Team projects cannot be deleted here. A server operator must deprovision the managed checkout and Git deploy keys.",
  };
  const props = {
    cover: "wood",
    onChooseCover() {},
    onDelete() {},
  };

  const personalMenu = renderToStaticMarkup(
    React.createElement(ProjectActionsMenu, { ...props, project: personal }),
  );
  const teamMenu = renderToStaticMarkup(
    React.createElement(ProjectActionsMenu, { ...props, project: team }),
  );

  assert.match(personalMenu, />Delete project</);
  assert.doesNotMatch(teamMenu, /Delete project/);
  assert.match(teamMenu, />Cover</);
  assert.doesNotMatch(teamMenu, /server operator|deploy key/i);
});

test("an unnamed personal identity presents the landing sign-in action", () => {
  const unnamed = { ...identity, user: { ...identity.user, display_name: null } };
  const html = renderToStaticMarkup(React.createElement(ProjectLanding, landingProps(unnamed)));

  assert.match(html, />Sign in</);
  assert.doesNotMatch(html, /data-identity-record="provenance-slip"/);
});

test("the identity slip delegates Edit and Copy without changing identity state", async () => {
  let edits = 0;
  let copyClicks = 0;
  let clipboardValue = null;
  const tree = IdentityProvenanceSlip({
    identity,
    identityError: null,
    teamNoticeId: "team-status",
    copyStatus: "idle",
    onCopy() {
      copyClicks += 1;
    },
    onEdit() {
      edits += 1;
    },
  });
  const edit = findElement(tree, (element) => element.props["data-identity-action"] === "edit");
  const copy = findElement(tree, (element) => element.props["data-identity-action"] === "copy-id");

  assert.ok(edit);
  assert.ok(copy);
  edit.props.onClick();
  copy.props.onClick();
  await copyIdentityId(userId, {
    async writeText(value) {
      clipboardValue = value;
    },
  });

  assert.equal(edits, 1);
  assert.equal(copyClicks, 1);
  assert.equal(clipboardValue, userId);
  assert.equal(identity.user.user_id, userId);
});

test("the personal identity panel opens the desktop Add team space flow", () => {
  const html = renderToStaticMarkup(
    React.createElement(IdentityProvenanceSlip, {
      identity,
      identityError: null,
      teamNoticeId: "team-status",
      copyStatus: "idle",
      onCopy() {},
      onEdit() {},
      onAddTeamSpace() {},
    }),
  );

  assert.match(html, /data-team-space-seam="available"/);
  assert.match(html, /<button[^>]*>.*Add team space/s);
  assert.doesNotMatch(html, /not implemented|coming later/i);
  assert.doesNotMatch(html, /<(form|input|textarea|select)\b/i);
  assert.doesNotMatch(html, /password|access token|private key/i);
});

test("the ordinary browser does not advertise the desktop Add team space action", () => {
  const html = renderToStaticMarkup(React.createElement(ProjectLanding, landingProps()));

  assert.doesNotMatch(html, /Add team space/);
  assert.doesNotMatch(html, /Add your lab server/);
});

function findElement(node, predicate) {
  if (Array.isArray(node)) {
    for (const child of node) {
      const match = findElement(child, predicate);
      if (match) return match;
    }
    return null;
  }
  if (!node || typeof node !== "object") return null;
  if (node.props && predicate(node)) return node;
  const children = node.props?.children;
  for (const child of Array.isArray(children) ? children : [children]) {
    const match = findElement(child, predicate);
    if (match) return match;
  }
  return null;
}

test("a pending project invitation is shelved beside the projects you have", () => {
  const html = renderToStaticMarkup(
    React.createElement(ProjectLanding, {
      ...landingProps(),
      invitations: [
        {
          invitation_id: "invitation-1",
          project_id: "project-1",
          project_name: "Plasticity study",
          space_name: "Lab space",
          invited_by: "1e6a2f6c-2b6f-4d4a-9d0e-2f0f5a8b2c3d",
          invited_by_name: "Ada Researcher",
          created_at: "2026-08-15T00:00:00Z",
        },
      ],
    }),
  );

  assert.match(html, /Plasticity study/);
  assert.match(html, /Lab space/);
  assert.match(html, /Ada Researcher/);
  assert.match(html, />Accept</);
  assert.match(html, />Decline</);
});

test("an invitation card carries no explanatory line under it", () => {
  const html = renderToStaticMarkup(
    React.createElement(ProjectLanding, {
      ...landingProps(),
      invitations: [
        {
          invitation_id: "invitation-1",
          project_id: "project-1",
          project_name: "Plasticity study",
          space_name: null,
          invited_by: "someone",
          invited_by_name: null,
          created_at: "2026-08-15T00:00:00Z",
        },
      ],
    }),
  );

  assert.doesNotMatch(html, /you have been invited/i);
  assert.doesNotMatch(html, /accept to join/i);
});
