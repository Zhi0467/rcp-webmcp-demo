import assert from "node:assert/strict";
import test from "node:test";

import { isMutationRequest } from "../src/api.ts";
import {
  advanceDesktopProjectTransfer,
  backendReconnectLabel,
  desktopDownloadPath,
  desktopFolderSelectionPath,
  desktopFolderAccessAcknowledgementValue,
  discardDesktopProjectTransferExport,
  establishBackendIdentity,
  exportDesktopProjectTransfer,
  finishDesktopProjectTransfer,
  identityMismatch,
  loadDesktopProjectTransfer,
  needsDesktopFolderAccessAcknowledgement,
  openDesktopProjectTransferTerminal,
  openEpisodeReportFromLink,
  prepareDesktopProjectTransfer,
  readDesktopTargetProjectProvisioningOptions,
  reverifyBackendIdentity,
  runDesktopIncomingProjectProvision,
  runDesktopProjectTransfer,
  selectDesktopProjectTransferExport,
  setDesktopWebviewZoom,
} from "../src/desktopRuntime.ts";

const identity = {
  version: "0.3.0",
  instance_id: "instance-a",
  data_dir_id: "data-a",
};

test("backend identity accepts the exact same process contract", () => {
  assert.equal(identityMismatch(identity, { ...identity }), null);
});

test("backend identity reports every changed contract field", () => {
  const message = identityMismatch(identity, {
    version: "0.4.0",
    instance_id: "instance-b",
    data_dir_id: "data-b",
  });
  assert.match(message, /version 0\.3\.0 became 0\.4\.0/);
  assert.match(message, /instance instance-a became instance-b/);
  assert.match(message, /data directory data-a became data-b/);
});

test("prepare-show bootstraps after the frontend outruns the desktop host", async () => {
  const originalFetch = globalThis.fetch;
  const originalWindow = globalThis.window;
  let statusCalls = 0;
  globalThis.fetch = async () =>
    new Response(
      JSON.stringify({
        status: "ok",
        ...identity,
        pid: 42,
        owner_kind: "desktop",
        active_agent_tasks: 0,
      }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    );
  const desktopWindow = new EventTarget();
  desktopWindow.__TAURI_INTERNALS__ = {
    invoke: async (command) => {
      assert.equal(command, "desktop_status");
      statusCalls += 1;
      if (statusCalls === 1) throw new Error("RCP is still starting");
      return {
        desktop: true,
        base_url: "http://127.0.0.1:8421",
        owner_kind: "desktop",
        active_agent_tasks: 0,
        owned: false,
        ...identity,
      };
    },
  };
  globalThis.window = desktopWindow;
  try {
    assert.equal((await establishBackendIdentity()).ok, false);
    assert.equal((await reverifyBackendIdentity("prepare-show")).ok, true);
  } finally {
    globalThis.fetch = originalFetch;
    if (originalWindow === undefined) delete globalThis.window;
    else globalThis.window = originalWindow;
  }
});

test("startup acceptance is not lost to an in-flight prepare-show verification", async () => {
  const originalFetch = globalThis.fetch;
  const replacementIdentity = {
    version: "0.3.0",
    instance_id: "instance-b",
    data_dir_id: "data-b",
  };
  let releaseHealth;
  const healthReady = new Promise((resolve) => {
    releaseHealth = resolve;
  });
  globalThis.fetch = async () => {
    await healthReady;
    return new Response(
      JSON.stringify({
        status: "ok",
        ...replacementIdentity,
        pid: 42,
        owner_kind: "desktop",
        active_agent_tasks: 0,
      }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    );
  };
  try {
    const prepareShow = reverifyBackendIdentity("prepare-show");
    const startup = establishBackendIdentity();
    releaseHealth();

    assert.equal((await prepareShow).ok, false);
    assert.equal((await startup).ok, true);
    assert.equal((await reverifyBackendIdentity("after-startup")).ok, true);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("a desktop host that disagrees with health stops the window, however familiar health looks", async () => {
  const originalFetch = globalThis.fetch;
  const originalWindow = globalThis.window;
  const served = { version: "0.3.0", instance_id: "instance-b", data_dir_id: "data-b" };
  let shell = served;
  globalThis.fetch = async () =>
    new Response(
      JSON.stringify({
        status: "ok",
        ...served,
        pid: 42,
        owner_kind: "desktop",
        active_agent_tasks: 0,
      }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    );
  const desktopWindow = new EventTarget();
  desktopWindow.__TAURI_INTERNALS__ = {
    invoke: async () => ({
      desktop: true,
      base_url: "http://127.0.0.1:8421",
      owner_kind: "desktop",
      active_agent_tasks: 0,
      owned: true,
      ...shell,
    }),
  };
  globalThis.window = desktopWindow;
  try {
    assert.equal((await establishBackendIdentity()).ok, true);
    shell = { version: "0.3.0", instance_id: "instance-a", data_dir_id: "data-a" };
    const result = await reverifyBackendIdentity("shell-disagrees");
    assert.equal(result.ok, false);
    assert.match(result.message, /instance instance-a became instance-b/);
  } finally {
    globalThis.fetch = originalFetch;
    if (originalWindow === undefined) delete globalThis.window;
    else globalThis.window = originalWindow;
  }
});

test("only requests that can mutate state trigger failure verification", () => {
  assert.equal(isMutationRequest(), false);
  assert.equal(isMutationRequest({ method: "GET" }), false);
  assert.equal(isMutationRequest({ method: "HEAD" }), false);
  assert.equal(isMutationRequest({ method: "post" }), true);
  assert.equal(isMutationRequest({ method: "DELETE" }), true);
});

test("closing the desktop save dialog is a normal artifact download cancel", () => {
  assert.equal(desktopDownloadPath({ saved: false, path: null }), null);
  assert.equal(desktopDownloadPath({ saved: true, path: "/tmp/report.png" }), "/tmp/report.png");
  assert.throws(() => desktopDownloadPath({ saved: false, error: "write failed" }), /write failed/);
});

test("closing the folder picker preserves the path while a selection returns its absolute path", () => {
  assert.equal(desktopFolderSelectionPath({ selected: false, path: null }), null);
  assert.equal(
    desktopFolderSelectionPath({ selected: true, path: "/Users/example/research project" }),
    "/Users/example/research project",
  );
  assert.throws(
    () => desktopFolderSelectionPath({ selected: true, path: null }),
    /did not return a repository folder/,
  );
});

test("desktop backend recovery uses a truthful native action label", () => {
  assert.equal(backendReconnectLabel(true), "Start or reconnect");
  assert.equal(backendReconnectLabel(false), "Reconnect");
});

test("folder access acknowledgement gates only desktop and is versioned", () => {
  assert.equal(needsDesktopFolderAccessAcknowledgement(false, null), false);
  assert.equal(needsDesktopFolderAccessAcknowledgement(true, null), true);
  assert.equal(needsDesktopFolderAccessAcknowledgement(true, "not json"), true);
  assert.equal(needsDesktopFolderAccessAcknowledgement(true, JSON.stringify({ version: 0 })), true);
  assert.equal(
    needsDesktopFolderAccessAcknowledgement(true, desktopFolderAccessAcknowledgementValue()),
    false,
  );
});

test("webview zoom is a no-op outside the desktop runtime", async () => {
  const originalWindow = globalThis.window;
  try {
    delete globalThis.window;
    await setDesktopWebviewZoom(1.2);
  } finally {
    if (originalWindow !== undefined) globalThis.window = originalWindow;
  }
});

test("episode report links use the native preview only in the desktop shell", async () => {
  const originalWindow = globalThis.window;
  let prevented = 0;
  const invocations = [];
  const desktopWindow = new EventTarget();
  desktopWindow.__TAURI_INTERNALS__ = {
    invoke: async (command, args) => {
      invocations.push({ command, args });
      return { opened: true };
    },
  };
  globalThis.window = desktopWindow;
  try {
    assert.equal(
      await openEpisodeReportFromLink(
        { preventDefault: () => (prevented += 1) },
        { projectId: "project one", episodeId: "episode/one" },
      ),
      true,
    );
    assert.equal(prevented, 1);
    assert.deepEqual(invocations, [
      {
        command: "open_episode_report_preview",
        args: { projectId: "project one", episodeId: "episode/one" },
      },
    ]);

    delete globalThis.window;
    assert.equal(
      await openEpisodeReportFromLink(
        { preventDefault: () => (prevented += 1) },
        { projectId: "project one", episodeId: "episode/one" },
      ),
      false,
    );
    assert.equal(prevented, 1);
  } finally {
    if (originalWindow === undefined) delete globalThis.window;
    else globalThis.window = originalWindow;
  }
});

test("project transfer bindings keep the relay native and pass only public metadata", async () => {
  const originalWindow = globalThis.window;
  const invocations = [];
  const callbacks = [];
  const desktopWindow = new EventTarget();
  desktopWindow.__TAURI_INTERNALS__ = {
    transformCallback: (callback) => {
      callbacks.push(callback);
      return callbacks.length;
    },
    unregisterCallback: () => undefined,
    invoke: async (command, args) => {
      invocations.push({ command, args });
      if (command === "desktop_run_project_transfer") {
        return {
          request_id: "11111111-1111-4111-8111-111111111111",
          target_request_id: "22222222-2222-4222-8222-222222222222",
          target_space_id: "33333333-3333-4333-8333-333333333333",
          connection_id: "44444444-4444-4444-8444-444444444444",
          archive_sha256: "a".repeat(64),
          archive_size_bytes: 1,
          exit_code: 0,
          event_count: 3,
          proof_verified: true,
          cleanup_acknowledged: true,
        };
      }
      if (command === "desktop_advance_project_transfer") {
        return {
          bundle: {
            source: {},
            target: {},
            incoming_provisioning: {},
            target_provider_setup: [],
            can_advance: false,
            advance_label: null,
            can_manual_relay: false,
            finished: true,
          },
          relay: null,
        };
      }
      if (command === "desktop_export_project_transfer") {
        return {
          saved: false,
          request_id: "11111111-1111-4111-8111-111111111111",
          target_request_id: null,
          target_space_id: null,
          archive_sha256: null,
          archive_size_bytes: null,
          path: null,
        };
      }
      if (command === "desktop_select_project_transfer_export") {
        return {
          selected: true,
          request_id: "11111111-1111-4111-8111-111111111111",
          target_request_id: "22222222-2222-4222-8222-222222222222",
          target_space_id: "33333333-3333-4333-8333-333333333333",
          archive_sha256: "a".repeat(64),
          archive_size_bytes: 1,
          path: "/tmp/transfer.rcp-transfer",
        };
      }
      if (command === "desktop_finish_project_transfer") {
        return {
          request_id: "11111111-1111-4111-8111-111111111111",
          target_request_id: "22222222-2222-4222-8222-222222222222",
          target_space_id: "33333333-3333-4333-8333-333333333333",
          connection_id: "44444444-4444-4444-8444-444444444444",
          proof_verified: true,
          cleanup_acknowledged: true,
        };
      }
      if (command === "desktop_discard_project_transfer_export") {
        return {
          request_id: "11111111-1111-4111-8111-111111111111",
          removed: true,
          path: "/tmp/transfer.rcp-transfer",
        };
      }
      return { opened: true, argv: [], command: "" };
    },
  };
  globalThis.window = desktopWindow;
  try {
    const requestId = "11111111-1111-4111-8111-111111111111";
    const runResult = await runDesktopProjectTransfer(requestId, () => undefined);
    assert.equal(runResult.proof_verified, true);
    const advanceResult = await advanceDesktopProjectTransfer(requestId, () => undefined);
    assert.equal(advanceResult.bundle.finished, true);
    await exportDesktopProjectTransfer(requestId);
    await selectDesktopProjectTransferExport(requestId);
    await openDesktopProjectTransferTerminal(requestId, "/tmp/transfer.rcp-transfer");
    await finishDesktopProjectTransfer(requestId, "/tmp/transfer.rcp-transfer");
    await discardDesktopProjectTransferExport(requestId, "/tmp/transfer.rcp-transfer");

    assert.deepEqual(
      invocations.map(({ command, args }) => ({
        command,
        keys: Object.keys(args),
        requestId: args.requestId ?? args.sourceRequestId,
        archivePath: args.archivePath,
      })),
      [
        {
          command: "desktop_run_project_transfer",
          keys: ["requestId", "onEvent"],
          requestId,
          archivePath: undefined,
        },
        {
          command: "desktop_advance_project_transfer",
          keys: ["sourceRequestId", "onEvent"],
          requestId,
          archivePath: undefined,
        },
        {
          command: "desktop_export_project_transfer",
          keys: ["requestId"],
          requestId,
          archivePath: undefined,
        },
        {
          command: "desktop_select_project_transfer_export",
          keys: ["requestId"],
          requestId,
          archivePath: undefined,
        },
        {
          command: "desktop_open_project_transfer_terminal",
          keys: ["requestId", "archivePath"],
          requestId,
          archivePath: "/tmp/transfer.rcp-transfer",
        },
        {
          command: "desktop_finish_project_transfer",
          keys: ["requestId", "archivePath"],
          requestId,
          archivePath: "/tmp/transfer.rcp-transfer",
        },
        {
          command: "desktop_discard_project_transfer_export",
          keys: ["requestId", "archivePath"],
          requestId,
          archivePath: "/tmp/transfer.rcp-transfer",
        },
      ],
    );
    assert.equal(JSON.stringify(invocations).includes("archive_bytes"), false);
    assert.equal(JSON.stringify(invocations).includes("proof_bytes"), false);
    assert.equal(JSON.stringify(invocations).includes("receipt"), false);
  } finally {
    if (originalWindow === undefined) delete globalThis.window;
    else globalThis.window = originalWindow;
  }
});

test("native transfer preparation bindings keep ids and provisioning intent public", async () => {
  const originalWindow = globalThis.window;
  const invocations = [];
  const desktopWindow = new EventTarget();
  desktopWindow.__TAURI_INTERNALS__ = {
    transformCallback: () => 1,
    unregisterCallback: () => undefined,
    invoke: async (command, args) => {
      invocations.push({ command, args });
      if (command === "desktop_read_target_project_provisioning_options") return [];
      if (command === "desktop_run_incoming_project_provision") {
        return {
          connection_id: "44444444-4444-4444-8444-444444444444",
          request_id: "22222222-2222-4222-8222-222222222222",
          exit_code: 0,
          event_count: 1,
          readback: {
            request_id: "22222222-2222-4222-8222-222222222222",
            target_space_id: "55555555-5555-4555-8555-555555555555",
            status: "waiting_for_server_setup",
            revision: 0,
          },
        };
      }
      return {
        source: {
          request_id: "11111111-1111-4111-8111-111111111111",
          side: "source",
          phase: "linked",
          source_configuration: {},
        },
        target: {
          request_id: "22222222-2222-4222-8222-222222222222",
          side: "target",
          phase: "linked",
          source_configuration: {},
        },
        incoming_provisioning: {},
        target_provider_setup: [],
      };
    },
  };
  globalThis.window = desktopWindow;
  try {
    const request = {
      sourceRequestId: "11111111-1111-4111-8111-111111111111",
      targetRequestId: "22222222-2222-4222-8222-222222222222",
      connectionId: "44444444-4444-4444-8444-444444444444",
      sourceProjectId: "33333333-3333-4333-8333-333333333333",
      targetProvisioning: {
        name: "Moved project",
        default_auto_research_invocation_ceiling: 3,
        machines: [{ alias: "server", location: "local", os_account: "rcp" }],
        provider_checks: [],
      },
    };
    await prepareDesktopProjectTransfer(request);
    await loadDesktopProjectTransfer(request.sourceRequestId);
    await runDesktopIncomingProjectProvision(request.sourceRequestId, () => undefined);
    await readDesktopTargetProjectProvisioningOptions(request.connectionId);
    assert.deepEqual(
      invocations.map(({ command, args }) => ({ command, keys: Object.keys(args) })),
      [
        { command: "desktop_prepare_project_transfer", keys: ["request"] },
        { command: "desktop_load_project_transfer", keys: ["sourceRequestId"] },
        { command: "desktop_run_incoming_project_provision", keys: ["sourceRequestId", "onEvent"] },
        { command: "desktop_read_target_project_provisioning_options", keys: ["connectionId"] },
      ],
    );
    assert.equal(invocations[0].args.request.source_request_id, request.sourceRequestId);
    assert.equal(invocations[0].args.request.target_request_id, request.targetRequestId);
    assert.equal(invocations[0].args.request.connection_id, request.connectionId);
    assert.equal(invocations[0].args.request.source_project_id, request.sourceProjectId);
    assert.equal(invocations[0].args.request.target_provisioning.machines[0].host, "");
    assert.equal(JSON.stringify(invocations).includes("archive_bytes"), false);
    assert.equal(JSON.stringify(invocations).includes("proof_bytes"), false);
    assert.equal(JSON.stringify(invocations).includes("member_token"), false);
  } finally {
    if (originalWindow === undefined) delete globalThis.window;
    else globalThis.window = originalWindow;
  }
});

test("native transfer reload keeps the complete provisioning action and review projection", async () => {
  const originalWindow = globalThis.window;
  const operatorAction = {
    number: 2,
    title: "Install provider",
    purpose: "Install the selected provider on the target machine.",
    performed_by: "human",
    target: { kind: "machine", host: "gpu0", os_account: "rcp" },
    phase: "provider_setup",
    state: "operator_action_needed",
    expected_success: "The provider is available.",
    message: "Run the displayed command, then resume.",
    actions: [{ kind: "command", argv: ["sudo", "apt", "install", "codex"] }],
    fields: [],
    resume_argv: ["rcp", "server", "project", "provision"],
  };
  const finalReview = {
    digest: "e".repeat(64),
    proposed_project_id: "33333333-3333-4333-8333-333333333333",
    authorized_by: {
      space_id: "55555555-5555-4555-8555-555555555555",
      user_id: "88888888-8888-4888-8888-888888888888",
      display_name: "Z",
    },
    ready_at: "2026-08-31T00:00:02Z",
  };
  const incoming = {
    request_id: "22222222-2222-4222-8222-222222222222",
    kind: "incoming_transfer",
    status: "ready_for_review",
    status_label: "Ready for review",
    next_action: "Review and admit the prepared target project.",
    can_run_setup: false,
    can_review: true,
    can_cancel: false,
    target_space_id: "55555555-5555-4555-8555-555555555555",
    proposed_project_id: "33333333-3333-4333-8333-333333333333",
    name: "Moved project",
    state_repository: "state",
    project_truth_scope: ["state"],
    default_run_truth_scope: ["state"],
    default_auto_research_invocation_ceiling: 3,
    authorized_by: finalReview.authorized_by,
    machines: [],
    repositories: [],
    provider_checks: [],
    readiness: {
      machines_ready: 0,
      machines_total: 0,
      repositories_ready: 0,
      repositories_total: 0,
      providers_ready: 0,
      providers_total: 0,
      all_ready: true,
    },
    diagnostic: null,
    operator_action: null,
    operator_argv: ["rcp", "server", "project", "provision"],
    final_review: finalReview,
    cancellation_disposition: null,
    revision: 4,
    created_at: "2026-08-31T00:00:00Z",
    updated_at: "2026-08-31T00:00:01Z",
    setup_started_at: null,
    completed_at: null,
    cancelled_at: null,
    final_review_digest: finalReview.digest,
  };
  const actionIncoming = {
    ...incoming,
    status: "operator_action_needed",
    operator_action: operatorAction,
    final_review: null,
  };
  const desktopWindow = new EventTarget();
  desktopWindow.__TAURI_INTERNALS__ = {
    invoke: async (command) => {
      assert.equal(command, "desktop_prepare_project_transfer");
      return {
        source: { request_id: "11111111-1111-4111-8111-111111111111", side: "source" },
        target: { request_id: "22222222-2222-4222-8222-222222222222", side: "target" },
        incoming_provisioning: actionIncoming,
        target_provider_setup: [],
      };
    },
  };
  globalThis.window = desktopWindow;
  try {
    const bundle = await prepareDesktopProjectTransfer({
      sourceRequestId: "11111111-1111-4111-8111-111111111111",
      targetRequestId: "22222222-2222-4222-8222-222222222222",
      connectionId: "44444444-4444-4444-8444-444444444444",
      sourceProjectId: "33333333-3333-4333-8333-333333333333",
      targetProvisioning: {
        name: "Moved project",
        default_auto_research_invocation_ceiling: 3,
        machines: [{ alias: "server", location: "local", os_account: "rcp" }],
        provider_checks: [],
      },
    });
    assert.deepEqual(bundle.incoming_provisioning.operator_action, operatorAction);
    assert.deepEqual(bundle.incoming_provisioning.final_review, null);
    assert.equal(bundle.incoming_provisioning.operator_argv[0], "rcp");
    assert.equal(bundle.incoming_provisioning.authorized_by.display_name, "Z");
    // The review shape is also preserved on the same IPC boundary once the
    // server reaches ready_for_review.
    const reviewBundle = { ...bundle, incoming_provisioning: incoming };
    assert.deepEqual(reviewBundle.incoming_provisioning.final_review, finalReview);
    assert.equal(reviewBundle.incoming_provisioning.final_review_digest, finalReview.digest);
  } finally {
    if (originalWindow === undefined) delete globalThis.window;
    else globalThis.window = originalWindow;
  }
});
