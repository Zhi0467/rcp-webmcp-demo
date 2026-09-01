import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  cancelProjectProvisioningRequest,
  completeProjectProvisioningRequest,
  createTeamProjectProvisioning,
  loadProjectProvisioningRequest,
  loadProjectProvisioningRequests,
} from "../src/api.ts";

const requestBody = {
  name: "Shared paper project",
  state_repository: "paper",
  project_truth_scope: ["paper"],
  default_run_truth_scope: ["paper"],
  default_auto_research_invocation_ceiling: 10,
  machines: [
    {
      alias: "server",
      location: "local",
      os_account: "rcp",
    },
  ],
  repositories: [
    {
      alias: "paper",
      source: "https://github.com/OpenAI/RCP.git",
      machine_alias: "server",
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
    },
  ],
};

const projectedResponse = {
  request_id: "request / one",
  kind: "create_team_project",
  status: "waiting_for_server_setup",
  status_label: "Waiting for server setup",
  next_action: "Run server setup.",
  can_run_setup: true,
  can_review: false,
  can_cancel: true,
  target_space_id: "space",
  proposed_project_id: "project",
  name: "Shared paper project",
  state_repository: "paper",
  project_truth_scope: ["paper"],
  default_run_truth_scope: ["paper"],
  default_auto_research_invocation_ceiling: 10,
  authorized_by: { space_id: "space", user_id: "alice", display_name: "Alice" },
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
    all_ready: false,
  },
  diagnostic: null,
  operator_action: null,
  operator_argv: ["/usr/local/bin/rcp", "server", "project", "provision", "request / one"],
  final_review: null,
  cancellation_disposition: null,
  revision: 0,
  created_at: "2026-08-29T00:00:00Z",
  updated_at: "2026-08-29T00:00:00Z",
  setup_started_at: null,
  completed_at: null,
  cancelled_at: null,
};

test("project provisioning calls preserve the backend projection and exact request identity", async () => {
  const originalFetch = globalThis.fetch;
  const requests = [];
  globalThis.fetch = async (path, init) => {
    requests.push({ path, init });
    const body =
      path === "/api/project-provisioning/requests" && init?.method === undefined
        ? [projectedResponse]
        : projectedResponse;
    return new Response(JSON.stringify(body), {
      status: path === "/api/project-provisioning/requests" && init?.method === "POST" ? 201 : 200,
      headers: { "Content-Type": "application/json" },
    });
  };
  try {
    assert.deepEqual(await createTeamProjectProvisioning(requestBody), projectedResponse);
    assert.deepEqual(await loadProjectProvisioningRequests(), [projectedResponse]);
    assert.deepEqual(await loadProjectProvisioningRequest("request / one"), projectedResponse);
    assert.deepEqual(await cancelProjectProvisioningRequest("request / one"), projectedResponse);
    assert.deepEqual(
      await completeProjectProvisioningRequest("request / one", "a".repeat(64)),
      projectedResponse,
    );

    assert.equal(requests[0].path, "/api/project-provisioning/requests");
    assert.equal(requests[0].init.method, "POST");
    assert.deepEqual(JSON.parse(requests[0].init.body), requestBody);
    assert.equal(requests[1].path, "/api/project-provisioning/requests");
    assert.equal(requests[1].init.method, undefined);
    assert.equal(requests[2].path, "/api/project-provisioning/requests/request%20%2F%20one");
    assert.equal(requests[2].init.method, undefined);
    assert.equal(requests[3].path, "/api/project-provisioning/requests/request%20%2F%20one/cancel");
    assert.equal(requests[3].init.method, "POST");
    assert.deepEqual(JSON.parse(requests[3].init.body), {});
    assert.equal(
      requests[4].path,
      "/api/project-provisioning/requests/request%20%2F%20one/complete",
    );
    assert.equal(requests[4].init.method, "POST");
    assert.deepEqual(JSON.parse(requests[4].init.body), { final_review_digest: "a".repeat(64) });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("complete provisioning and transfer lifecycles remain opaque to browser code", async () => {
  const types = await readFile(new URL("../src/types.ts", import.meta.url), "utf8");

  assert.match(types, /declare const OPAQUE_PROJECT_PROVISIONING_STATUS: unique symbol/);
  assert.match(types, /declare const OPAQUE_PROJECT_PROVISIONING_CHECK_STATUS: unique symbol/);
  assert.match(types, /declare const OPAQUE_PROJECT_TRANSFER_PHASE: unique symbol/);
  assert.match(types, /declare const OPAQUE_PROJECT_TRANSFER_PROOF_STATE: unique symbol/);
  assert.doesNotMatch(types, /type ProjectProvisioningStatus\s*=\s*"/);
  assert.doesNotMatch(types, /type ProjectProvisioningCheckStatus\s*=\s*"/);
  assert.doesNotMatch(types, /type ProjectTransferPhase\s*=\s*"/);
  assert.doesNotMatch(types, /type ProjectTransferProofState\s*=\s*"/);
});
