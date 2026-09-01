import assert from "node:assert/strict";
import { after, test } from "node:test";
import { readFile } from "node:fs/promises";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";

import {
  assertSupportedProjectCreationIntent,
  buildTeamProvisioningRequest,
  formatCommandArgv,
  invalidProjectProvisioningHash,
  parseProjectSetupRoute,
  projectCreationPrimaryLabel,
  projectMoveSetupHash,
  projectProvisioningHash,
  projectProvisioningRequestId,
  repositoryPickerPresentation,
  selectedProjectCreationIntent,
  stateRepositoryAfterRemoval,
} from "../src/projectSetup.ts";

const server = await createServer({
  root: new URL("..", import.meta.url).pathname,
  configFile: false,
  logLevel: "silent",
  server: { middlewareMode: true, hmr: false },
  optimizeDeps: { noDiscovery: true },
});
const { ProjectSetup, RepositoryEditor } = await server.ssrLoadModule(
  "/src/views/ProjectSetup.tsx",
);
const { ProjectSettings } = await server.ssrLoadModule("/src/views/ProjectSettings.tsx");
const {
  TransferProjectSetup,
  transferActiveWorkSummary,
  transferFinished,
  transferRelayFailure,
  transferTargetIsReady,
} = await server.ssrLoadModule("/src/views/TransferProjectSetup.tsx");
const {
  ProvisioningStatus,
  gitWriteFact,
  projectProvisioningCreateModeAvailable,
  serverOperatorProbeMatchesDraft,
  TeamProjectSetup,
} = await server.ssrLoadModule("/src/views/TeamProjectSetup.tsx");

after(() => server.close());

const repositories = [
  { id: 1, alias: "research" },
  { id: 2, alias: "analysis" },
  { id: 3, alias: "paper" },
];

test("removing the canonical repository selects the first remaining alias", () => {
  assert.equal(stateRepositoryAfterRemoval(repositories, 1, "research"), "analysis");
});

test("repository removal preserves another selection and handles an empty remainder", () => {
  assert.equal(stateRepositoryAfterRemoval(repositories, 2, "research"), "research");
  assert.equal(stateRepositoryAfterRemoval([repositories[0]], 1, "research"), "");
});

test("only desktop local repositories offer the native folder picker", () => {
  assert.deepEqual(repositoryPickerPresentation("local", true), {
    showPicker: true,
    hint: null,
  });
  assert.deepEqual(repositoryPickerPresentation("local", false), {
    showPicker: false,
    hint: "Paste an absolute path. Finder selection is available in the desktop app.",
  });
  assert.deepEqual(repositoryPickerPresentation("ssh", true), {
    showPicker: false,
    hint: null,
  });
  assert.deepEqual(repositoryPickerPresentation("ssh", false), {
    showPicker: false,
    hint: null,
  });
});

test("the repository path label targets only its input while the picker stays a sibling", () => {
  const originalWindow = globalThis.window;
  globalThis.window = { __TAURI_INTERNALS__: {} };
  try {
    const html = renderToStaticMarkup(
      React.createElement(RepositoryEditor, {
        repository: {
          id: 7,
          alias: "research",
          location: "local",
          path: "/Users/example/research",
          host: "",
          default_read: true,
        },
        canonical: true,
        only: true,
        onCanonical() {},
        onChange() {},
      }),
    );
    const labelStart = html.indexOf('<label for="repository-path-7">');
    const labelEnd = html.indexOf("</label>", labelStart);
    const inputStart = html.indexOf('id="repository-path-7"', labelEnd);
    const pickerStart = html.indexOf("Choose folder…", inputStart);

    assert.ok(labelStart >= 0);
    assert.ok(labelEnd > labelStart);
    assert.ok(inputStart > labelEnd);
    assert.ok(pickerStart > inputStart);
  } finally {
    if (originalWindow === undefined) delete globalThis.window;
    else globalThis.window = originalWindow;
  }
});

const personalCreation = {
  requires_authenticated_member: false,
  intents: [
    {
      intent: "use_existing_checkout_personally",
      eligible: true,
      preselected: true,
      primary_action_label: "Use existing checkout",
      required_fields: ["name", "repositories", "state_repository", "execution", "confirmed"],
      pinned_source_project_id: null,
      unavailable_reason: null,
    },
    {
      intent: "create_shared_team_project",
      eligible: false,
      preselected: false,
      primary_action_label: "Create shared team project",
      required_fields: ["machines", "repositories", "provider_checks"],
      pinned_source_project_id: null,
      unavailable_reason: "Connect to a team space.",
    },
    {
      intent: "move_personal_project_to_team",
      eligible: false,
      preselected: false,
      primary_action_label: "Move to team space",
      required_fields: [],
      pinned_source_project_id: null,
      unavailable_reason: "Not available yet.",
    },
  ],
};

const teamCreation = {
  requires_authenticated_member: true,
  intents: personalCreation.intents.map((intent) => ({
    ...intent,
    eligible: intent.intent === "create_shared_team_project",
    preselected: intent.intent === "create_shared_team_project",
    unavailable_reason:
      intent.intent === "create_shared_team_project" ? null : intent.unavailable_reason,
  })),
};

test("the backend selects one setup intent and its primary label", () => {
  assert.equal(selectedProjectCreationIntent(personalCreation), "use_existing_checkout_personally");
  assert.equal(projectCreationPrimaryLabel(teamCreation), "Create shared team project");
  assert.throws(
    () => selectedProjectCreationIntent({ requires_authenticated_member: false, intents: [] }),
    /did not select/,
  );
  assert.doesNotThrow(() =>
    assertSupportedProjectCreationIntent(teamCreation, "create_shared_team_project"),
  );
  assert.throws(
    () =>
      assertSupportedProjectCreationIntent(
        {
          ...teamCreation,
          intents: teamCreation.intents.map((intent) =>
            intent.intent === "create_shared_team_project"
              ? { ...intent, required_fields: ["invented_field"] }
              : intent,
          ),
        },
        "create_shared_team_project",
      ),
    /field contract/,
  );
  assert.throws(
    () =>
      assertSupportedProjectCreationIntent(
        {
          ...teamCreation,
          intents: teamCreation.intents.map((intent) =>
            intent.intent === "create_shared_team_project"
              ? { ...intent, required_fields: ["machines", "machines", "repositories"] }
              : intent,
          ),
        },
        "create_shared_team_project",
      ),
    /field contract/,
  );
});

test("personal and team setup use one visible wizard route with plainly named intents", () => {
  const personal = renderToStaticMarkup(
    React.createElement(ProjectSetup, {
      projectCreation: personalCreation,
      onCancel() {},
      onCreated() {},
    }),
  );
  const team = renderToStaticMarkup(
    React.createElement(ProjectSetup, {
      projectCreation: teamCreation,
      onCancel() {},
      onCreated() {},
    }),
  );

  for (const html of [personal, team]) {
    assert.match(html, /Use an existing checkout personally/);
    assert.match(html, /Create a shared team project/);
    assert.match(html, /Move an existing personal project to a team/);
    assert.equal((html.match(/class="setup-shell/g) ?? []).length, 1);
    assert.match(html, /role="group" aria-label="Project setup kind"/);
    assert.match(html, /aria-pressed="true"/);
    assert.match(html, /aria-current="step"/);
  }
  assert.match(personal, /Absolute repository path/);
  assert.match(team, /GitHub repository/);
  assert.doesNotMatch(team, /Absolute repository path/);
});

test("a provisioning request deep link accepts only one canonical UUID4", () => {
  const requestId = "11111111-1111-4111-8111-111111111111";
  assert.equal(projectProvisioningHash(requestId), `#/projects/new?request=${requestId}`);
  assert.equal(projectProvisioningRequestId(projectProvisioningHash(requestId)), requestId);
  assert.equal(projectProvisioningRequestId("#/projects/new?request=../other"), null);
  assert.equal(invalidProjectProvisioningHash("#/projects/new?request=../other"), true);
  assert.equal(invalidProjectProvisioningHash("#/projects/new"), false);
  assert.equal(projectProvisioningRequestId("#/projects/other"), null);
});

test("move setup links pin the source and round-trip the linked request identities", () => {
  const sourceProjectId = "11111111-1111-4111-8111-111111111111";
  const sourceRequestId = "33333333-3333-4333-8333-333333333333";
  const targetRequestId = "44444444-4444-4444-8444-444444444444";
  const hash = projectMoveSetupHash({
    sourceProjectId,
    sourceRequestId,
    targetRequestId,
  });

  assert.equal(
    hash,
    `#/projects/new?intent=move_personal_project_to_team&source_project_id=${sourceProjectId}&source_request_id=${sourceRequestId}&target_request_id=${targetRequestId}`,
  );
  assert.deepEqual(parseProjectSetupRoute(hash), {
    kind: "move",
    intent: "move_personal_project_to_team",
    sourceProjectId,
    sourceRequestId,
    targetRequestId,
  });
  assert.deepEqual(parseProjectSetupRoute(projectMoveSetupHash({ sourceProjectId })), {
    kind: "move",
    intent: "move_personal_project_to_team",
    sourceProjectId,
    sourceRequestId: null,
    targetRequestId: null,
  });
  assert.throws(
    () => projectMoveSetupHash({ sourceProjectId: "../other" }),
    /Source project identity must be a canonical UUID4/,
  );
});

test("move setup links fail closed for missing, forged, or duplicate identities", () => {
  assert.deepEqual(parseProjectSetupRoute("#/projects/new?intent=move_personal_project_to_team"), {
    kind: "invalid",
    reason: "invalid_move_route",
  });
  assert.deepEqual(
    parseProjectSetupRoute(
      "#/projects/new?intent=move_personal_project_to_team&source_project_id=../other",
    ),
    { kind: "invalid", reason: "invalid_move_route" },
  );
  assert.deepEqual(
    parseProjectSetupRoute(
      "#/projects/new?intent=move_personal_project_to_team&source_project_id=11111111-1111-4111-8111-111111111111&source_project_id=22222222-2222-4222-8222-222222222222",
    ),
    { kind: "invalid", reason: "invalid_move_route" },
  );
  assert.deepEqual(
    parseProjectSetupRoute(
      "#/projects/new?intent=move_personal_project_to_team&source_project_id=11111111-1111-4111-8111-111111111111&transfer_request_id=22222222-2222-4222-8222-222222222222",
    ),
    { kind: "invalid", reason: "invalid_move_route" },
  );
  assert.deepEqual(
    parseProjectSetupRoute(
      "#/projects/new?intent=move_personal_project_to_team&source_project_id=11111111-1111-4111-8111-111111111111&source_request_id=33333333-3333-4333-8333-333333333333",
    ),
    { kind: "invalid", reason: "invalid_move_route" },
  );
  assert.equal(
    invalidProjectProvisioningHash(
      "#/projects/new?intent=move_personal_project_to_team&source_project_id=11111111-1111-4111-8111-111111111111",
    ),
    false,
  );
  assert.throws(
    () =>
      projectMoveSetupHash({
        sourceProjectId: "11111111-1111-4111-8111-111111111111",
        sourceRequestId: "33333333-3333-4333-8333-333333333333",
      }),
    /created as one pair/,
  );
});

const moveCreation = {
  ...personalCreation,
  intents: personalCreation.intents.map((intent) =>
    intent.intent === "move_personal_project_to_team"
      ? {
          ...intent,
          eligible: true,
          preselected: false,
          required_fields: ["source_project", "team_connection"],
          unavailable_reason: null,
        }
      : { ...intent, eligible: false, preselected: false },
  ),
};

test("the move route is consumed by the one wizard and locks its intent", () => {
  const originalWindow = globalThis.window;
  globalThis.window = { __TAURI_INTERNALS__: {} };
  try {
    const html = renderToStaticMarkup(
      React.createElement(ProjectSetup, {
        projectCreation: moveCreation,
        setupRoute: parseProjectSetupRoute(
          "#/projects/new?intent=move_personal_project_to_team&source_project_id=11111111-1111-4111-8111-111111111111",
        ),
        onCancel() {},
        onCreated() {},
      }),
    );
    assert.match(html, /Move an existing personal project to a team/);
    assert.match(html, /Source project pinned/);
    assert.match(html, /11111111-1111-4111-8111-111111111111/);
    assert.match(html, /aria-pressed="true"/);
    assert.match(html, /disabled=""/);
  } finally {
    if (originalWindow === undefined) delete globalThis.window;
    else globalThis.window = originalWindow;
  }
});

test("invalid setup routes fail visibly without mounting a setup form", () => {
  const html = renderToStaticMarkup(
    React.createElement(ProjectSetup, {
      projectCreation: personalCreation,
      setupRoute: { kind: "invalid", reason: "invalid_move_route" },
      onCancel() {},
      onCreated() {},
    }),
  );
  assert.match(html, /role="alert"/);
  assert.match(html, /setup link is invalid/);
  assert.doesNotMatch(html, /Project name|Absolute repository path/);
});

test("move setup is visibly unavailable outside the desktop runtime", () => {
  const originalWindow = globalThis.window;
  delete globalThis.window;
  try {
    const html = renderToStaticMarkup(
      React.createElement(TransferProjectSetup, {
        route: parseProjectSetupRoute(
          "#/projects/new?intent=move_personal_project_to_team&source_project_id=11111111-1111-4111-8111-111111111111",
        ),
        intentChooser: React.createElement("div", null, "locked move intent"),
        onCancel() {},
      }),
    );
    assert.match(html, /unavailable in a browser/);
    assert.match(html, /source-built desktop app/);
  } finally {
    if (originalWindow !== undefined) globalThis.window = originalWindow;
  }
});

test("move active-work counts use backend active and live booleans", () => {
  assert.deepEqual(
    transferActiveWorkSummary(
      [
        { active: true, status: "succeeded" },
        { active: false, status: "running" },
      ],
      [
        { live: true, status: "completed" },
        { live: false, status: "active" },
      ],
    ),
    { activeTaskCount: 1, liveEpisodeCount: 1, totalCount: 2 },
  );
});

test("move target readiness requires a saved origin and operator route", () => {
  const ready = {
    connection_id: "team-1",
    expected_space_id: "space-team-1",
    local_origin: "https://rcp-team-1.localhost:9001",
    operator_route: { ssh_target: "rcp@example.org", mode: "direct_rcp" },
  };
  assert.equal(transferTargetIsReady(ready), true);
  assert.equal(transferTargetIsReady({ ...ready, operator_route: null }), false);
  assert.equal(transferTargetIsReady(null), false);
});

test("move completion renders the native cross-space decision", () => {
  assert.equal(
    transferFinished({
      source: { finished: false },
      target: { finished: false },
      finished: true,
    }),
    true,
  );
  assert.equal(
    transferFinished({
      source: { finished: true },
      target: { finished: true },
      finished: false,
    }),
    false,
  );
  assert.equal(transferFinished(null), false);
});

test("move relay failures stay loud and point to the explicit manual path", () => {
  const failed = transferRelayFailure({
    exit_code: 17,
    proof_verified: false,
    cleanup_acknowledged: false,
  });
  assert.match(failed, /Automatic relay failed/);
  assert.match(failed, /code 17/);
  assert.match(failed, /Manual relay/);
  assert.equal(
    transferRelayFailure({
      exit_code: 0,
      proof_verified: true,
      cleanup_acknowledged: true,
    }),
    null,
  );
  assert.equal(transferRelayFailure(null), null);
});

function settingsProject() {
  const permissions = {
    read_graph: true,
    read_research_md: true,
    read_introduction: false,
    read_repositories: "run_scope",
    read_conversations: "none",
    write_graph_patch: false,
    write_project_files: false,
    write_paper: false,
  };
  const profile = {
    provider: "codex",
    model: "",
    reasoning: "medium",
    run_on: "local",
    permissions,
  };
  const metric = {
    bytes: 0,
    count: 0,
    limits: { max_bytes: 1, max_count: 1, ttl_seconds: 1 },
    reclaimable_bytes: 0,
    reclaimable_count: 0,
  };
  return {
    id: "11111111-1111-4111-8111-111111111111",
    name: "Personal project",
    state_repository: "research",
    run_on: "local",
    default_run_truth_scope: ["research"],
    default_auto_research_invocation_ceiling: 10,
    repositories: [{ alias: "research", machine: "local", path: "/repo" }],
    machines: [{ alias: "local", host: "", provider_paths: { codex: "codex" } }],
    agent_profiles: {
      seed: profile,
      refresh: profile,
      node_chat: profile,
      project_chat: profile,
      paper_coach: profile,
      orchestrator: profile,
    },
    providers: {},
    provider_readiness: {},
    cache_metrics: { remote_sources: metric, session_slices: metric },
  };
}

test("Project Settings opens the move route only for a personal project", () => {
  const project = settingsProject();
  const renderSettings = (spaceKind, onMove) =>
    renderToStaticMarkup(
      React.createElement(ProjectSettings, {
        apiBase: "/api/projects/project",
        project,
        identity: null,
        onLeftProject() {},
        usage: null,
        onRefreshUsage: async () => {},
        cacheClearDisabled: false,
        onSaved() {},
        onCacheMetricsChange() {},
        onRefreshReadiness: async () => {},
        showDisplaySettings: false,
        spaceKind,
        onMovePersonalProjectToTeam: onMove,
        textScale: 100,
        onTextScaleChange() {},
      }),
    );

  const personal = renderSettings("personal", () => {});
  const team = renderSettings("team", () => {});
  assert.match(personal, /Project home/);
  assert.match(personal, /Move to team space/);
  assert.doesNotMatch(team, /Move to team space/);
});

test("copyable server argv preserves exact token boundaries", () => {
  assert.equal(
    formatCommandArgv(["/usr/local/bin/rcp", "server", "path with spaces", "a'b"]),
    "/usr/local/bin/rcp server 'path with spaces' 'a'\\''b'",
  );
});

test("an operator probe is valid only for the exact displayed route", () => {
  const probe = {
    connection_id: "team-1",
    available: true,
    route: { ssh_target: "operator@server", mode: "sudo_rcp" },
    diagnostic: null,
  };
  assert.equal(serverOperatorProbeMatchesDraft(probe, " operator@server ", "sudo_rcp"), true);
  assert.equal(serverOperatorProbeMatchesDraft(probe, "rcp@server", "sudo_rcp"), false);
  assert.equal(serverOperatorProbeMatchesDraft(probe, "operator@server", "direct_rcp"), false);
  assert.equal(gitWriteFact(true), "Git write verified");
  assert.equal(gitWriteFact(false), "Git write not verified");
});

test("a deep-linked request blocks the blank create form before its durable read", () => {
  const originalWindow = globalThis.window;
  globalThis.window = {
    location: {
      hash: "#/projects/new?request=11111111-1111-4111-8111-111111111111",
      origin: "http://127.0.0.1:8421",
    },
  };
  try {
    const html = renderToStaticMarkup(
      React.createElement(TeamProjectSetup, {
        intentChooser: React.createElement("div", null, "intent chooser"),
        onCancel() {},
        onCreated() {},
      }),
    );
    assert.match(html, /Loading the existing setup request/);
    assert.doesNotMatch(
      html,
      /Name the project and its first GitHub repository|Create setup request/,
    );
    assert.doesNotMatch(html, /Team boundary|Canonical state|Central checkout owner/);
  } finally {
    if (originalWindow === undefined) delete globalThis.window;
    else globalThis.window = originalWindow;
  }
  assert.equal(projectProvisioningCreateModeAvailable(true, true, false), false);
  assert.equal(projectProvisioningCreateModeAvailable(false, true, false), false);
  assert.equal(projectProvisioningCreateModeAvailable(false, false, false), true);
});

test("an invalid deep link fails visibly without exposing the create form", () => {
  const originalWindow = globalThis.window;
  globalThis.window = {
    location: {
      hash: "#/projects/new?request=../other",
      origin: "http://127.0.0.1:8421",
    },
  };
  try {
    const html = renderToStaticMarkup(
      React.createElement(TeamProjectSetup, {
        intentChooser: React.createElement("div", null, "intent chooser"),
        onCancel() {},
        onCreated() {},
      }),
    );
    assert.match(html, /invalid provisioning request identity/);
    assert.doesNotMatch(
      html,
      /Name the project and its first GitHub repository|Create setup request/,
    );
  } finally {
    if (originalWindow === undefined) delete globalThis.window;
    else globalThis.window = originalWindow;
  }
});

test("the team request derives truth scopes while preserving machine and provider intent", () => {
  const request = buildTeamProvisioningRequest({
    name: " Shared project ",
    stateRepository: "paper",
    defaultAutoResearchInvocationCeiling: 10,
    machines: [{ alias: "server", location: "local", os_account: "rcp" }],
    repositories: [
      {
        alias: "paper",
        source: "https://github.com/OpenAI/RCP.git",
        machine_alias: "server",
        default_read: true,
      },
      {
        alias: "private-notes",
        source: "git@github.com:OpenAI/private-notes.git",
        machine_alias: "server",
        default_read: false,
      },
    ],
    providerChecks: [
      {
        profile: "seed",
        provider: "codex",
        runtime_id: "codex:exec",
        model: "gpt-5.6-luna",
        reasoning: "medium",
        machine_alias: "server",
      },
    ],
  });

  assert.equal(request.name, "Shared project");
  assert.deepEqual(request.project_truth_scope, ["paper", "private-notes"]);
  assert.deepEqual(request.default_run_truth_scope, ["paper"]);
  assert.equal(request.repositories[0].source, "https://github.com/OpenAI/RCP.git");
  assert.equal("default_read" in request.repositories[0], false);
  assert.equal(request.provider_checks[0].machine_alias, "server");
});

test("the provisioning view renders backend answers and hides native actions in a browser", async () => {
  const request = {
    request_id: "11111111-1111-4111-8111-111111111111",
    kind: "create_team_project",
    status: "operator_action_needed",
    status_label: "Operator action needed",
    next_action: "Grant repository write access, then resume setup.",
    can_run_setup: true,
    can_review: false,
    can_cancel: false,
    target_space_id: "space-1",
    proposed_project_id: "project-1",
    name: "Shared project",
    state_repository: "paper",
    project_truth_scope: ["paper"],
    default_run_truth_scope: ["paper"],
    default_auto_research_invocation_ceiling: 10,
    authorized_by: { space_id: "space-1", user_id: "alice", display_name: "Alice" },
    machines: [
      {
        alias: "server",
        location: "local",
        host: "",
        os_account: "rcp",
        intended_central_root: "/var/lib/rcp/projects/project-1",
        resolved_central_root: "/var/lib/rcp/projects/project-1",
        ready: true,
        status_label: "Ready",
      },
    ],
    repositories: [
      {
        alias: "paper",
        repository: { identity: "openai/rcp" },
        https_clone_url: "https://github.com/openai/rcp.git",
        ssh_clone_url: "git@github.com:openai/rcp.git",
        settings_url: "https://github.com/openai/rcp/settings/keys",
        machine_alias: "server",
        intended_path: "/var/lib/rcp/projects/project-1/paper",
        resolved_path: "/var/lib/rcp/projects/project-1/paper",
        checkout_disposition: "request_created",
        status: "operator_action_needed",
        status_label: "Operator action needed",
        ready: false,
        commit: "a".repeat(40),
        write_verified: false,
        deploy_key_label: "rcp:space-1:project-1:paper",
        public_key_fingerprint: "SHA256:key",
        checked_at: "2026-08-30T00:00:00Z",
        diagnostic: null,
      },
    ],
    provider_checks: [
      {
        profile: "seed",
        provider: "codex",
        runtime_id: "codex:exec",
        model: "gpt-5.6-luna",
        reasoning: "medium",
        machine_alias: "server",
        status: "ready",
        status_label: "Provider ready",
        ready: true,
        binary_path: "/usr/local/bin/codex",
        version: "1.0",
        resolved_runtime_id: "codex:exec",
        execution_account: "rcp",
        checked_at: "2026-08-30T00:00:00Z",
        diagnostic: null,
      },
    ],
    readiness: {
      machines_ready: 1,
      machines_total: 1,
      repositories_ready: 0,
      repositories_total: 1,
      providers_ready: 1,
      providers_total: 1,
      all_ready: false,
    },
    diagnostic: "Exact backend diagnostic",
    operator_action: {
      number: 4,
      title: "Grant repository write access",
      message: "Add this public deploy key with Allow write access.",
      performed_by: "human",
      target: {
        kind: "external_service",
        service: "github.com",
        resource: "openai/rcp",
        destination_url: "https://github.com/openai/rcp/settings/keys",
        required_authority_role: "repository administrator",
      },
      purpose: "Authorize the central checkout.",
      phase: "deploy_key",
      state: "operator_action_needed",
      expected_success: "The write check succeeds.",
      actions: [{ kind: "external", instruction: "Enable Allow write access." }],
      fields: [{ name: "public_key", value: "ssh-ed25519 public" }],
      resume_argv: ["rcp", "server", "project", "provision", "request with space"],
    },
    operator_argv: ["rcp", "server", "project", "provision", "request with space"],
    final_review: null,
    cancellation_disposition: null,
    revision: 3,
    created_at: "2026-08-30T00:00:00Z",
    updated_at: "2026-08-30T00:00:00Z",
    setup_started_at: "2026-08-30T00:00:00Z",
    completed_at: null,
    cancelled_at: null,
  };
  const noop = () => {};
  const operatorHtml = renderToStaticMarkup(
    React.createElement(ProvisioningStatus, {
      request,
      events: [],
      desktop: false,
      connection: null,
      operatorTarget: "",
      operatorMode: "sudo_rcp",
      probe: null,
      busy: null,
      onOperatorTarget: noop,
      onOperatorMode: noop,
      onSaveAndProbe: noop,
      onCopy: noop,
      onRefresh: noop,
      onRun: noop,
      onTerminal: noop,
      onCancel: noop,
      onComplete: noop,
    }),
  );

  assert.match(operatorHtml, /Operator action needed/);
  assert.match(operatorHtml, /Grant repository write access, then resume setup/);
  assert.match(operatorHtml, /Exact backend diagnostic/);
  assert.match(operatorHtml, /openai\/rcp/);
  assert.match(operatorHtml, /\/var\/lib\/rcp\/projects\/project-1\/paper/);
  assert.match(operatorHtml, /Provider ready/);
  assert.match(operatorHtml, /repository administrator/);
  assert.match(operatorHtml, /Authorize the central checkout/);
  assert.doesNotMatch(operatorHtml, /deploy_key|operator_action_needed/);
  assert.match(operatorHtml, /Git write not verified/);
  assert.match(operatorHtml, /Allow write access/);
  assert.doesNotMatch(operatorHtml, /Final review|Confirm and create project/);
  assert.match(operatorHtml, /Copy server command/);
  assert.doesNotMatch(operatorHtml, /Run setup now|Open in Terminal/);

  const readyRequest = {
    ...request,
    status: "ready_for_review",
    status_label: "Ready for review",
    next_action: "Review the prepared project.",
    can_run_setup: false,
    can_review: true,
    diagnostic: null,
    operator_action: null,
    repositories: request.repositories.map((repository) => ({
      ...repository,
      status: "ready",
      status_label: "Git write ready",
      ready: true,
      write_verified: true,
    })),
    readiness: {
      ...request.readiness,
      repositories_ready: 1,
      all_ready: true,
    },
    final_review: {
      digest: "b".repeat(64),
      proposed_project_id: "project-1",
      authorized_by: { space_id: "space-1", user_id: "alice", display_name: "Alice" },
      ready_at: "2026-08-30T00:00:00Z",
    },
  };
  const readyHtml = renderToStaticMarkup(
    React.createElement(ProvisioningStatus, {
      request: readyRequest,
      events: [],
      desktop: false,
      connection: null,
      operatorTarget: "",
      operatorMode: "sudo_rcp",
      probe: null,
      busy: null,
      onOperatorTarget: noop,
      onOperatorMode: noop,
      onSaveAndProbe: noop,
      onCopy: noop,
      onRefresh: noop,
      onRun: noop,
      onTerminal: noop,
      onCancel: noop,
      onComplete: noop,
    }),
  );
  assert.match(readyHtml, /Ready for review/);
  assert.match(readyHtml, /Review the prepared project/);
  assert.match(readyHtml, /Git write verified/);
  assert.match(readyHtml, /Final review/);
  assert.match(readyHtml, /Confirm and create project/);
  assert.doesNotMatch(readyHtml, /Human action required|Run setup now|Open in Terminal/);
  const finalReviewHtml = readyHtml.slice(readyHtml.indexOf('class="provisioning-final-review"'));
  assert.match(finalReviewHtml, /https:\/\/github\.com\/openai\/rcp\.git/);
  assert.match(finalReviewHtml, /\/var\/lib\/rcp\/projects\/project-1\/paper/);
  assert.match(finalReviewHtml, /Git write verified/);
  assert.match(finalReviewHtml, /Provider ready/);
  assert.match(finalReviewHtml, />Alice</);
  assert.ok(
    finalReviewHtml.indexOf("https://github.com/openai/rcp.git") <
      finalReviewHtml.indexOf("Confirm and create project"),
  );

  const source = await readFile(
    new URL("../src/views/TeamProjectSetup.tsx", import.meta.url),
    "utf8",
  );
  assert.doesNotMatch(source, /request\.status\b/);
  assert.match(source, /role="log"[\s\S]*aria-live="polite"[\s\S]*aria-relevant="additions"/);
});
