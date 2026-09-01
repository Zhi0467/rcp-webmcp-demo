import assert from "node:assert/strict";
import { withTaskAnswers } from "./taskAnswers.mjs";
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
const { ProjectOverview } = await server.ssrLoadModule("/src/views/ProjectOverview.tsx");
const { setupExistingResearchSelection, setupFinalConfirmation } = await server.ssrLoadModule(
  "/src/views/ProjectSetup.tsx",
);
const { ProjectHistoryDrawer } = await server.ssrLoadModule(
  "/src/components/ProjectHistoryDrawer.tsx",
);
const { revisionSummariesUrl } = await server.ssrLoadModule("/src/App.tsx");

after(() => server.close());

const graph = {
  revision: 5,
  nodes: {
    "rq/main": {
      id: "rq/main",
      type: "research_question",
      title: "Future plasticity",
      question: "What predicts future plasticity?",
      updated_rev: 4,
    },
    "hyp/latest": {
      id: "hyp/latest",
      type: "hypothesis",
      title: "Latest graph concept",
      updated_rev: 5,
    },
    "dec/ready": {
      id: "dec/ready",
      type: "decision",
      title: "Choose the queued direction",
      status: "ready",
      updated_rev: 3,
    },
    "dec/open": {
      id: "dec/open",
      type: "decision",
      title: "Keep framing this choice",
      status: "open",
      updated_rev: 2,
    },
  },
  edges: {},
  proposals: {},
  ambiguities: {
    legacy: { id: "legacy", question: "Historical ambiguity", status: "open" },
  },
};

const project = {
  primary_question: graph.nodes["rq/main"],
  counts: { accepted: 2 },
  canonical_state: { remote: false },
  last_refresh_at: "2026-08-03T08:00:00Z",
};

const latestSummary = {
  from_revision: 4,
  to_revision: 5,
  kind: "refresh",
  author: "agent",
  producer: "agent",
  authorized_by: null,
  profile: null,
  task_id: null,
  created_at: "2026-08-03T08:00:00Z",
  sentences: [
    "Updated the third-stream ordering.",
    "Recorded two additional probe checkpoints.",
    "This complete detail belongs in History only.",
  ],
};

test("revision summary URLs scope Overview to one revision and leave drawer history complete", () => {
  const base = "/api/projects/project-one";
  assert.equal(
    revisionSummariesUrl(base, 5),
    `${base}/history/summaries?from_revision=5&to_revision=5`,
  );
  assert.equal(revisionSummariesUrl(base), `${base}/history/summaries`);
});

test("Overview uses the latest revision prose and preserves every other question row", () => {
  const html = renderToStaticMarkup(
    React.createElement(ProjectOverview, {
      project,
      graph,
      pendingProposals: [],
      decisionsAwaitingChoice: [graph.nodes["dec/ready"]],
      latestRevisionSummary: latestSummary,
      onNavigate() {},
    }),
  );

  assert.match(
    html,
    /Updated the third-stream ordering\. Recorded two additional probe checkpoints\./,
  );
  assert.doesNotMatch(html, /This complete detail belongs in History only/);
  assert.match(html, /Revision 4 to revision 5/);
  assert.match(html, /What are we asking\?/);
  assert.match(html, /Where are we\?/);
  assert.match(html, /What is blocked\?/);
  assert.match(html, /What needs you\?/);
  assert.match(html, /Choose the queued direction/);
  assert.match(html, /0 proposals · 1 decisions awaiting choice/);
  assert.doesNotMatch(html, /Historical ambiguity|ambiguities/);
  assert.match(html, /What happens next\?/);
  assert.equal(html.match(/class="overview-number"/g)?.length, 6);
});

test("Overview keeps its previous latest-node fallback when no summary is supplied", () => {
  const html = renderToStaticMarkup(
    React.createElement(ProjectOverview, {
      project,
      graph,
      pendingProposals: [],
      decisionsAwaitingChoice: [],
      onNavigate() {},
    }),
  );

  assert.match(html, /Latest graph concept/);
  assert.doesNotMatch(html, /Updated the third-stream ordering/);
  assert.match(html, /Last refresh/);
});

test("Project history separates revision prose from the complete clickable Agent task list", () => {
  const inspected = [];
  const tasks = [
    task("seed-task", "seed", "succeeded", 1),
    task("coach-task", "paper_coach", "failed", 2),
  ];
  const summaries = [
    { ...latestSummary, from_revision: 3, to_revision: 4, sentences: ["Earlier change."] },
    latestSummary,
  ];
  const props = {
    projectId: "project",
    summaries,
    tasks,
    loading: false,
    error: null,
    onInspectTask(taskId) {
      inspected.push(taskId);
    },
    episodeReportHref: (episodeId) => `/preview/${episodeId}`,
    onClose() {},
  };
  const html = renderToStaticMarkup(React.createElement(ProjectHistoryDrawer, props));

  assert.match(html, /aria-label="Agent tasks"/);
  assert.match(html, /aria-label="Project revision summaries"/);
  assert.match(html, /Project revisions/);
  assert.match(html, /Revision 4 to revision 5/);
  assert.match(html, /Updated the third-stream ordering\./);
  assert.ok(html.indexOf("Revision 4 to revision 5") < html.indexOf("Revision 3 to revision 4"));
  assert.match(html, /Seed project graph · attempt 1/);
  assert.match(html, /Writing coach · attempt 2/);
  assert.equal(html.match(/data-task-id=/g)?.length, tasks.length);

  const tree = ProjectHistoryDrawer(props);
  const coachButton = findElement(
    tree,
    (element) => element.props["data-task-id"] === "coach-task",
  );
  assert.ok(coachButton);
  coachButton.props.onClick();
  assert.deepEqual(inspected, ["coach-task"]);
});

test("Project history reports loading without hiding the already-loaded Agent tasks", () => {
  const html = renderToStaticMarkup(
    React.createElement(ProjectHistoryDrawer, {
      projectId: "project",
      summaries: [],
      tasks: [task("active-task", "refresh", "running", 1)],
      loading: true,
      error: null,
      onInspectTask() {},
      episodeReportHref: (episodeId) => `/preview/${episodeId}`,
      onClose() {},
    }),
  );

  assert.match(html, /role="status"/);
  assert.match(html, /Loading project revisions…/);
  assert.doesNotMatch(html, /No project revisions yet/);
  assert.match(html, /data-task-id="active-task"/);
});

test("Project history labels system, attributed, and legacy revisions without inferring identity", () => {
  const authorized = {
    space_id: "123e4567-e89b-42d3-a456-426614174000",
    user_id: "123e4567-e89b-42d3-a456-426614174001",
    display_name: "Ada Researcher",
  };
  const html = renderToStaticMarkup(
    React.createElement(ProjectHistoryDrawer, {
      projectId: "project",
      summaries: [
        {
          ...latestSummary,
          from_revision: 0,
          to_revision: 1,
          kind: "identity",
          author: null,
          producer: "system",
          sentences: ["Project identity adopted."],
        },
        {
          ...latestSummary,
          from_revision: 1,
          to_revision: 2,
          producer: "human",
          author: "human",
          authorized_by: authorized,
          sentences: ["Recorded a human change."],
        },
        {
          ...latestSummary,
          from_revision: 2,
          to_revision: 3,
          authorized_by: authorized,
          profile: "ordinary",
          task_id: "task-ordinary",
          sentences: ["Recorded an Agent change."],
        },
        {
          ...latestSummary,
          from_revision: 3,
          to_revision: 4,
          producer: "human",
          author: "human",
          authorized_by: null,
          sentences: ["Recorded a legacy change."],
        },
      ],
      tasks: [],
      loading: false,
      error: null,
      onInspectTask() {},
      episodeReportHref: (episodeId) => `/preview/${episodeId}`,
      onClose() {},
    }),
  );

  assert.match(html, /Identity · RCP/);
  assert.match(html, /Refresh · Ada Researcher/);
  assert.match(html, /Ordinary Agent task · task-ordinary/);
  assert.match(html, /Refresh · Human · Unattributed/);
});

test("Project history groups episode envelopes while preserving revision attribution", () => {
  const authorized = {
    space_id: "123e4567-e89b-42d3-a456-426614174000",
    user_id: "123e4567-e89b-42d3-a456-426614174001",
    display_name: "Ada Researcher",
  };
  const report = {
    report_id: "report-final",
    ending: "exhausted",
    created_at: "2026-08-03T09:00:00Z",
  };
  const summaries = [
    {
      ...latestSummary,
      from_revision: 5,
      to_revision: 6,
      authorized_by: authorized,
      profile: "orchestrator",
      task_id: "task-orchestrator",
      episode_id: "episode-live",
      episode: {
        mode: "auto_research",
        state_label: "Exhausted",
        ending: "exhausted",
        wrapup_state: "ready",
        report,
      },
      created_at: "2026-08-03T08:06:00Z",
      sentences: ["Coordinated the episode."],
    },
    {
      ...latestSummary,
      from_revision: 4,
      to_revision: 5,
      authorized_by: authorized,
      profile: "ordinary",
      task_id: "task-missing-new",
      episode_id: "episode-missing",
      episode: null,
      created_at: "2026-08-03T08:05:00Z",
      sentences: ["Recorded missing episode work."],
    },
    {
      ...latestSummary,
      from_revision: 3,
      to_revision: 4,
      kind: "approval",
      producer: "human",
      author: "human",
      authorized_by: authorized,
      profile: null,
      task_id: null,
      episode_id: null,
      episode: null,
      created_at: "2026-08-03T08:04:00Z",
      sentences: ["Approved the episode proposal."],
    },
    {
      ...latestSummary,
      from_revision: 2,
      to_revision: 3,
      authorized_by: authorized,
      profile: "ordinary",
      task_id: "task-worker",
      episode_id: "episode-live",
      episode: {
        mode: "auto_research",
        state_label: "Exhausted",
        ending: "exhausted",
        wrapup_state: "ready",
        report,
      },
      created_at: "2026-08-03T08:03:00Z",
      sentences: ["Recorded worker research."],
    },
    {
      ...latestSummary,
      from_revision: 1,
      to_revision: 2,
      authorized_by: authorized,
      profile: "ordinary",
      task_id: "task-missing-old",
      episode_id: "episode-missing",
      episode: null,
      created_at: "2026-08-03T08:02:00Z",
      sentences: ["Recorded earlier missing episode work."],
    },
    {
      ...latestSummary,
      from_revision: 0,
      to_revision: 1,
      authorized_by: authorized,
      profile: "ordinary",
      task_id: "task-independent",
      episode_id: null,
      episode: null,
      created_at: "2026-08-03T08:01:00Z",
      sentences: ["Recorded independent work."],
    },
  ];

  const html = renderToStaticMarkup(
    React.createElement(ProjectHistoryDrawer, {
      projectId: "project",
      summaries,
      tasks: [],
      loading: false,
      error: null,
      onInspectTask() {},
      episodeReportHref: (episodeId) => `/preview/${episodeId}`,
      onClose() {},
    }),
  );

  assert.equal(html.match(/class="history-campaign-group"/g)?.length, 2);
  assert.match(html, /Auto-research episode · Exhausted/);
  assert.match(html, /Authorized by Ada Researcher/);
  assert.match(html, /2 revisions/);
  assert.match(html, /Episode no longer recorded/);
  // An episode id may appear in the report link's href, but never as text a
  // human reads: a group identifies itself by state, not by a bare id.
  assert.doesNotMatch(html.replace(/<[^>]*>/g, " "), /episode-live|episode-missing/);
  assert.match(html, /Orchestrator Agent task · task-orchestrator/);
  assert.match(html, /Ordinary Agent task · task-worker/);
  assert.match(html, /Approval · Ada Researcher/);
  assert.equal(html.match(/Open report/g)?.length, 1);
  assert.ok(
    html.indexOf("Auto-research episode · Exhausted") < html.indexOf("Episode no longer recorded"),
  );
  assert.ok(
    html.indexOf("Episode no longer recorded") < html.indexOf("Approved the episode proposal"),
  );
  assert.ok(
    html.indexOf("</section></li>", html.indexOf("Episode no longer recorded")) <
      html.indexOf("Approved the episode proposal"),
  );
});

test("Project history report control links to the decorated report's preview", () => {
  const report = {
    report_id: "report-final",
    ending: "completed",
    created_at: "2026-08-03T09:00:00Z",
  };
  const requested = [];

  const html = renderToStaticMarkup(
    React.createElement(ProjectHistoryDrawer, {
      projectId: "project",
      summaries: [
        {
          ...latestSummary,
          episode_id: "episode-live",
          episode: {
            mode: "experiment_loop",
            status: "completed",
            ending: "completed",
            wrapup_state: "ready",
            report,
          },
        },
      ],
      tasks: [],
      loading: false,
      error: null,
      onInspectTask() {},
      episodeReportHref(episodeId) {
        requested.push(episodeId);
        return `/preview/${episodeId}`;
      },
      onClose() {},
    }),
  );

  assert.deepEqual(requested, ["episode-live"]);
  assert.match(html, /href="\/preview\/episode-live"/);
  assert.match(html, /Open report/);
});

test("an identity revision does not make an unseeded project look refreshed", () => {
  const identityOnlyGraph = { ...graph, revision: 1, nodes: {} };
  const html = renderToStaticMarkup(
    React.createElement(ProjectOverview, {
      project: {
        ...project,
        primary_question: null,
        last_refresh_at: null,
      },
      graph: identityOnlyGraph,
      pendingProposals: [],
      decisionsAwaitingChoice: [],
      latestRevisionSummary: null,
      onNavigate() {},
    }),
  );

  assert.match(html, /The project has not been seeded yet\./);
  assert.match(html, /Seed the graph from the selected truth repositories\./);
  assert.match(html, /Project revision 1/);
  assert.doesNotMatch(html, /Graph revision/);
});

test("connecting an existing project preserves its retained canonical state", () => {
  assert.match(
    setupFinalConfirmation({
      action: "connect",
      remote_write: false,
      canonical_location: "/research/project",
    }),
    /retained canonical state without replacing its manifest/,
  );
});

test("archive confirmation carries the exact retained-history token from preflight", () => {
  const preview = {
    existing_research: { archive_token: "a".repeat(64) },
  };

  assert.deepEqual(setupExistingResearchSelection(preview, "archive_and_create"), {
    existing_research_action: "archive_and_create",
    existing_research_token: "a".repeat(64),
  });
  assert.deepEqual(setupExistingResearchSelection(preview, "open_existing"), {
    existing_research_action: "open_existing",
    existing_research_token: null,
  });
});

function task(operationId, kind, status, attempt) {
  return withTaskAnswers({
    operation_id: operationId,
    project_id: "project",
    kind,
    status,
    request: {},
    created_at: "2026-08-03T08:00:00Z",
    updated_at: "2026-08-03T08:00:00Z",
    status_message: status,
    attempt,
    estimate_seconds: 0,
    estimate_samples: 0,
    phase: status,
    elapsed_seconds: 0,
    progress: 1,
    can_pause: false,
    can_resume: false,
    can_retry: false,
  });
}

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
