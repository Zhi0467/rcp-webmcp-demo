import assert from "node:assert/strict";
import test from "node:test";

import {
  ApiError,
  api,
  clearAllProjectCaches,
  clearProjectCaches,
  createTeamInvitation,
  exchangeTeamSession,
  loadTeamInvitations,
  loadProjectReadiness,
  pinApiInstance,
  registerIdentityNameRequiredHandler,
  registerMutationFailureHandler,
  removeChatAttachment,
  uploadChatAttachment,
} from "../src/api.ts";

const metrics = {
  remote_sources: {
    bytes: 128,
    count: 2,
    limits: { max_bytes: 1024, max_count: 8, ttl_seconds: 86400 },
    oldest_accessed_at: "2026-07-28T00:00:00Z",
    reclaimable_bytes: 64,
    reclaimable_count: 1,
  },
  session_slices: {
    bytes: 0,
    count: 0,
    limits: { max_bytes: 2048, max_count: 16, ttl_seconds: 172800 },
    oldest_accessed_at: null,
    reclaimable_bytes: 0,
    reclaimable_count: 0,
  },
};

test("clearProjectCaches issues one DELETE and returns replacement metrics", async () => {
  const originalFetch = globalThis.fetch;
  let request;
  globalThis.fetch = async (path, init) => {
    request = { path, init };
    return new Response(JSON.stringify(metrics), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };
  try {
    assert.deepEqual(await clearProjectCaches("/api/projects/demo"), metrics);
    assert.equal(request.path, "/api/projects/demo/caches");
    assert.equal(request.init.method, "DELETE");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("clearAllProjectCaches returns authoritative zero metrics for the open project", async () => {
  const originalFetch = globalThis.fetch;
  let request;
  const clearedMetrics = {
    remote_sources: { ...metrics.remote_sources, bytes: 0, count: 0 },
    session_slices: { ...metrics.session_slices, bytes: 0, count: 0 },
  };
  globalThis.fetch = async (path, init) => {
    request = { path, init };
    return new Response(JSON.stringify(clearedMetrics), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };
  try {
    assert.deepEqual(await clearAllProjectCaches("project/with spaces"), clearedMetrics);
    assert.equal(request.path, "/api/caches?project_id=project%2Fwith%20spaces");
    assert.equal(request.init.method, "DELETE");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("automatic readiness reads cached state while explicit refresh bypasses it", async () => {
  const originalFetch = globalThis.fetch;
  const paths = [];
  globalThis.fetch = async (path) => {
    paths.push(path);
    return new Response(JSON.stringify({ provider_readiness: {}, providers: {} }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };
  try {
    await loadProjectReadiness("/api/projects/demo");
    await loadProjectReadiness("/api/projects/demo", true);
    assert.deepEqual(paths, [
      "/api/projects/demo/readiness",
      "/api/projects/demo/readiness?refresh=true",
    ]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("API errors retain status for stale and active-task handling", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () =>
    new Response(JSON.stringify({ detail: "Agent task is active" }), {
      status: 409,
      headers: { "Content-Type": "application/json" },
    });
  try {
    await assert.rejects(
      clearProjectCaches("/api/projects/demo"),
      (error) =>
        error instanceof ApiError &&
        error.status === 409 &&
        error.message === "Agent task is active",
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("a failed mutation runs the registered identity verifier once", async () => {
  const originalFetch = globalThis.fetch;
  let checkedPath = null;
  globalThis.fetch = async () =>
    new Response(JSON.stringify({ detail: "Conflict" }), {
      status: 409,
      headers: { "Content-Type": "application/json" },
    });
  registerMutationFailureHandler(async (path) => {
    checkedPath = path;
  });
  try {
    await assert.rejects(
      api("/api/projects/demo/sync", { method: "POST", body: "{}" }),
      (error) => error instanceof ApiError && error.status === 409,
    );
    assert.equal(checkedPath, "/api/projects/demo/sync");
  } finally {
    registerMutationFailureHandler(null);
    globalThis.fetch = originalFetch;
  }
});

test("a named identity retries the exact mutation once before reconnect handling", async () => {
  const originalFetch = globalThis.fetch;
  const requests = [];
  let prompted = 0;
  let reconnects = 0;
  const init = { method: "POST", headers: { "X-Caller": "kept" }, body: '{"value":1}' };
  globalThis.fetch = async (path, requestInit) => {
    requests.push({ path, init: requestInit });
    return new Response(
      requests.length === 1
        ? JSON.stringify({
            detail: { code: "identity_name_required", message: "Choose a name." },
          })
        : JSON.stringify({ saved: true }),
      {
        status: requests.length === 1 ? 428 : 200,
        headers: { "Content-Type": "application/json" },
      },
    );
  };
  registerIdentityNameRequiredHandler(async () => {
    prompted += 1;
    return true;
  });
  registerMutationFailureHandler(async () => {
    reconnects += 1;
  });
  try {
    assert.deepEqual(await api("/api/projects/demo/sync", init), { saved: true });
    assert.equal(prompted, 1);
    assert.equal(reconnects, 0);
    assert.equal(requests.length, 2);
    assert.equal(requests[0].path, requests[1].path);
    assert.equal(requests[0].init.method, requests[1].init.method);
    assert.equal(requests[0].init.body, requests[1].init.body);
    assert.equal(new Headers(requests[1].init.headers).get("X-Caller"), "kept");
  } finally {
    registerIdentityNameRequiredHandler(null);
    registerMutationFailureHandler(null);
    globalThis.fetch = originalFetch;
  }
});

test("cancelling the identity prompt rejects the original 428 without reconnect handling", async () => {
  const originalFetch = globalThis.fetch;
  let requests = 0;
  let reconnects = 0;
  globalThis.fetch = async () => {
    requests += 1;
    return new Response(
      JSON.stringify({
        detail: { code: "identity_name_required", message: "Choose a name." },
      }),
      { status: 428, headers: { "Content-Type": "application/json" } },
    );
  };
  registerIdentityNameRequiredHandler(async () => false);
  registerMutationFailureHandler(async () => {
    reconnects += 1;
  });
  try {
    await assert.rejects(
      api("/api/projects/demo/sync", { method: "POST", body: "{}" }),
      (error) =>
        error instanceof ApiError &&
        error.status === 428 &&
        error.message.includes("identity_name_required"),
    );
    assert.equal(requests, 1);
    assert.equal(reconnects, 0);
  } finally {
    registerIdentityNameRequiredHandler(null);
    registerMutationFailureHandler(null);
    globalThis.fetch = originalFetch;
  }
});

test("a repeated identity 428 is not prompted or retried again", async () => {
  const originalFetch = globalThis.fetch;
  let requests = 0;
  let prompts = 0;
  let reconnects = 0;
  globalThis.fetch = async () => {
    requests += 1;
    return new Response(
      JSON.stringify({
        detail: { code: "identity_name_required", message: "Choose a name." },
      }),
      { status: 428, headers: { "Content-Type": "application/json" } },
    );
  };
  registerIdentityNameRequiredHandler(async () => {
    prompts += 1;
    return true;
  });
  registerMutationFailureHandler(async () => {
    reconnects += 1;
  });
  try {
    await assert.rejects(
      api("/api/projects/demo/sync", { method: "POST", body: "{}" }),
      (error) => error instanceof ApiError && error.status === 428,
    );
    assert.equal(requests, 2);
    assert.equal(prompts, 1);
    assert.equal(reconnects, 1);
  } finally {
    registerIdentityNameRequiredHandler(null);
    registerMutationFailureHandler(null);
    globalThis.fetch = originalFetch;
  }
});

test("mutations carry the pinned backend identity and preserve caller headers", async () => {
  const originalFetch = globalThis.fetch;
  let request;
  globalThis.fetch = async (path, init) => {
    request = { path, init };
    return new Response("{}", { status: 200, headers: { "Content-Type": "application/json" } });
  };
  pinApiInstance("instance-a");
  try {
    await api("/api/projects/demo/sync", {
      method: "POST",
      headers: new Headers({ Authorization: "Bearer test", "X-Caller": "kept" }),
      body: "{}",
    });
    const headers = new Headers(request.init.headers);
    assert.equal(headers.get("X-RCP-Instance-ID"), "instance-a");
    assert.equal(headers.get("Authorization"), "Bearer test");
    assert.equal(headers.get("X-Caller"), "kept");
  } finally {
    pinApiInstance(null);
    globalThis.fetch = originalFetch;
  }
});

test("reads never carry the pinned backend identity", async () => {
  const originalFetch = globalThis.fetch;
  let request;
  globalThis.fetch = async (path, init) => {
    request = { path, init };
    return new Response("{}", { status: 200, headers: { "Content-Type": "application/json" } });
  };
  pinApiInstance("instance-a");
  try {
    await api("/api/projects/demo", { headers: { "X-Caller": "kept" } });
    const headers = new Headers(request.init.headers);
    assert.equal(headers.get("X-RCP-Instance-ID"), null);
    assert.equal(headers.get("X-Caller"), "kept");
  } finally {
    pinApiInstance(null);
    globalThis.fetch = originalFetch;
  }
});

test("chat attachment ingress preserves multipart content and client scope", async () => {
  const originalFetch = globalThis.fetch;
  let request;
  const descriptor = {
    attachment_id: "attachment-a",
    name: "notes.md",
    media_type: "text/markdown",
    size: 5,
    expires_at: "2026-08-15T00:00:00Z",
  };
  globalThis.fetch = async (path, init) => {
    request = { path, init };
    return new Response(JSON.stringify({ attachment_set_id: "set-a", attachment: descriptor }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };
  try {
    const file = new File(["hello"], "notes.md", { type: "text/markdown" });
    const result = await uploadChatAttachment(
      "/api/projects/demo",
      "chat-a",
      file,
      "client-a",
      "set-a",
    );
    assert.equal(result.attachment.name, "notes.md");
    assert.equal(request.path, "/api/projects/demo/chats/chat-a/attachments");
    assert.equal(request.init.method, "POST");
    assert.ok(request.init.body instanceof FormData);
    assert.equal(request.init.body.get("client_id"), "client-a");
    assert.equal(request.init.body.get("attachment_set_id"), "set-a");
    assert.equal(new Headers(request.init.headers).get("Content-Type"), null);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("removing an unsent attachment preserves the claimed set scope", async () => {
  const originalFetch = globalThis.fetch;
  let request;
  globalThis.fetch = async (path, init) => {
    request = { path, init };
    return new Response(JSON.stringify({ removed: true }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };
  try {
    await removeChatAttachment("/api/projects/demo", "chat-a", "set-a", "attachment-a", "client-a");
    assert.match(request.path, /attachment-a\?/);
    assert.match(request.path, /attachment_set_id=set-a/);
    assert.match(request.path, /client_id=client-a/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("team session exchange sends the raw token once in the JSON body only", async () => {
  const originalFetch = globalThis.fetch;
  const rawToken = "rcp_super-secret-member-token";
  let request;
  const identity = {
    space_id: "space-a",
    space_kind: "team",
    space_name: "Causal Systems Lab",
    user: { user_id: "user-a", display_name: "Ada", identity_kind: "team_member" },
  };
  globalThis.fetch = async (path, init) => {
    request = { path, init };
    return new Response(JSON.stringify(identity), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };
  try {
    assert.deepEqual(await exchangeTeamSession(rawToken), identity);
    assert.equal(request.path, "/api/team/session/exchange");
    assert.equal(request.init.method, "POST");
    assert.deepEqual(JSON.parse(request.init.body), { token: rawToken });
    assert.doesNotMatch(request.path, new RegExp(rawToken));
    assert.equal(new Headers(request.init.headers).get("Authorization"), null);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("team invitation helpers use the member-scoped collection without code URLs", async () => {
  const originalFetch = globalThis.fetch;
  const requests = [];
  const invitation = {
    invitation_id: "invite-a",
    created_by: "user-a",
    created_at: "2026-08-12T00:00:00Z",
    expires_at: "2026-08-19T00:00:00Z",
    consumed_at: null,
    consumed_by: null,
    failed_attempts: 0,
    locked_at: null,
  };
  globalThis.fetch = async (path, init) => {
    requests.push({ path, init });
    const body =
      init?.method === "POST"
        ? { invitation, code: "rcp_invite-secret", space_name: "Causal Systems Lab" }
        : [invitation];
    return new Response(JSON.stringify(body), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };
  try {
    assert.deepEqual(await loadTeamInvitations(), [invitation]);
    assert.equal((await createTeamInvitation()).code, "rcp_invite-secret");
    assert.deepEqual(
      requests.map(({ path, init }) => [path, init?.method ?? "GET"]),
      [
        ["/api/team/invitations", "GET"],
        ["/api/team/invitations", "POST"],
      ],
    );
    assert.deepEqual(JSON.parse(requests[1].init.body), {});
    assert.ok(requests.every(({ path }) => !path.includes("rcp_invite-secret")));
  } finally {
    globalThis.fetch = originalFetch;
  }
});
