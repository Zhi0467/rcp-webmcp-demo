# Research Control Panel — WebMCP Challenge Demo

Research Control Panel (RCP) turns provider conversations, bounded autonomous
episodes, evidence, and human decisions into one durable research workspace.
This challenge build adds WebMCP Site Tools so a researcher can operate that
workspace from the familiar ChatGPT or Codex conversation beside the live page.

**Live demo:** <https://rcp-webmcp-demo.onrender.com>

This repository is a source-only public snapshot prepared for the
[WebMCP Challenge](https://webmcp.devpost.com/). The `pre-webmcp` tag captures
the audited product baseline; the default branch adds the challenge work.

## Why WebMCP helps

Without WebMCP, a browser agent must repeatedly inspect rendered UI, rediscover
identities, and reproduce low-level navigation. RCP now gives the browser agent
compact research-level actions such as “explain this project,” “inspect this
Experiment,” and “start the next episode.” RCP still owns admission,
configuration, provider routing, persistence, lifecycle, and human authority.

WebMCP is an alternate entrance to existing RCP operations, not a second API or
state machine. The ordinary app remains fully usable when Site Tools are absent.

## Demo journey

The hosted site contains a synthetic continual-learning project and a
deterministic challenge provider named **RCP Demo**. A researcher can ask the
browser agent to:

1. list and open the project;
2. explain its research question, hypotheses, evidence, and blockers;
3. inspect one exact graph claim;
4. open an existing visual reliability artifact in RCP's sandboxed viewer;
5. send a Discuss turn and read the durable task result;
6. inspect and start a ready held-out Experiment;
7. observe the real episode move from ready to live to completed; and
8. open the newly persisted visual result and explain its limited conclusion.

The deterministic result completes only the scoped synthetic replicate. It
does not resolve the broader research question or silently change hypothesis
standing.

## Site Tools

The landing page exposes two tools:

| Tool | Purpose |
|---|---|
| `rcp_list_projects` | Return the bounded project index. |
| `rcp_open_project` | Navigate to one exact listed project. |

An open project can expose nine more tools:

| Tool | Purpose |
|---|---|
| `rcp_get_project_overview` | Return a compact research map and current attention state. |
| `rcp_inspect_node` | Read one exact typed graph node and direct relations. |
| `rcp_list_artifacts` | List current artifacts and immutable episode reports. |
| `rcp_open_artifact` | Open one exact item in RCP's existing visual viewer. |
| `rcp_inspect_conversation` | Read a bounded transcript, latest task result, routing, and legal Send options. |
| `rcp_send_conversation_message` | Start a normal Discuss or Work turn through RCP's conversation owner. |
| `rcp_inspect_experiment` | Read backend-decided Experiment, task, episode, watcher, and report state. |
| `rcp_start_experiment` | Start the next bounded episode through the normal Run path. |
| `rcp_stop_episode` | Request RCP's graceful continuation fence for the exact live episode. |

Mutation tools appear only when the current backend projection has a valid
target. Start disappears while an episode is live, Stop appears for that exact
episode, and both disappear after completion. Every call revalidates its target
at execution time.

Protected graph operations, Proposal approval, Decision choice, credentials,
project administration, raw Patch input, and server administration are not
exposed through WebMCP.

## Architecture

```text
ChatGPT Work or Codex
        |
        | discovers top-level document.modelContext tools
        v
RCP React page ── shared UI owners ── RCP backend projections and routes
                                             |
                                             v
                          task / provider / episode / artifact storage
```

The hosted demo adds a challenge-only gateway in front of ordinary RCP:

- an opaque persistent browser cookie maps to one isolated fixture copy;
- only the cookie hash is stored;
- one loopback-only RCP child owns each active copy;
- refresh, browser reopen, and child restart preserve that browser's progress;
- a different browser identity receives a different copy;
- management and filesystem-facing routes are blocked at the public boundary;
- storage, process, request, and concurrency limits fail closed; and
- **Start over demo** is an explicit human-only action, never a WebMCP tool.

The gateway and RCP Demo provider exist to make public judging safe and
reproducible. They are challenge deployment scaffolding, not claimed as durable
RCP multi-user architecture.

## Run locally

Requirements:

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Node.js 24 and npm

Build in the required order:

```bash
npm --prefix web ci
npm --prefix web run build
uv sync --frozen
```

Start the isolated challenge gateway on plain local HTTP:

```bash
RCP_DEMO_ROOT=/tmp/rcp-webmcp-demo \
  uv run python -m challenge.gateway \
  --host 127.0.0.1 --port 8420 --insecure-cookie
```

Open `http://127.0.0.1:8420`. A normal browser shows the complete RCP UI. Site
Tools require a supported WebMCP browser host; current OpenAI guidance supports
ChatGPT Work and Codex in the ChatGPT desktop built-in browser using GPT-5.6
Sol or Terra.

## Verification

The hosted HTTPS build and complete private challenge branch have passed:

- 3,208 backend tests with nine expected skips;
- 508 Web tests;
- the production TypeScript build;
- Ruff and all repository pre-commit hooks;
- an actual Codex in-app-browser journey through project discovery, exact node
  and conversation reads, a durable Discuss turn, Experiment Start and
  completion, and both visual artifact and episode-report opening;
- production `Secure`, `HttpOnly`, `SameSite=Strict` cookie checks, distinct
  browser session copies, refresh and tab-reopen continuity, and persistence
  through a Render service restart;
- clean hosted browser and Render application logs; and
- local production-gateway checks for cookie resume, restart persistence,
  two-browser isolation, route refusal, and bounded concurrent children.

This source-only public snapshot intentionally omits the private documentation
tree, agent instructions, and the two test files that enforce that internal
documentation layout. Its exact runnable backend suite contains the remaining
3,200 product tests with nine expected skips.

Checked-in journey cases live in `challenge/evals/webmcp_journeys.json`.

Run the principal checks with:

```bash
npm --prefix web test
npm --prefix web run build
uv run pytest
uv run ruff check src tests challenge
```

## Deploy

`render.yaml` defines one Render Web Service with a persistent disk, HTTPS
gateway health check, frozen production dependency install, and explicit
resource limits. Create a Blueprint from this public repository's sole `main`
branch. This repository is only the audited deployment snapshot; product
development continues in the private RCP repository.

## Challenge scope and provenance

RCP's research graph, authority queue, conversations, provider execution,
Experiment loop, watchers, artifacts, and ordinary UI predate the challenge.
The challenge work adds:

- imperative top-level WebMCP registration;
- eleven bounded schemas and results;
- state-dependent tool discovery and invocation-time revalidation;
- shared navigation, Send, Start, Stop, and artifact-viewer ownership;
- the deterministic synthetic evaluation fixture;
- browser-isolated public demo hosting; and
- target-browser journey evaluations.

Compare the default branch with the `pre-webmcp` tag to inspect that boundary.
Private development history, internal operational documents, machine-specific
fixtures, and credentials are intentionally absent from this public snapshot.

## License

MIT. See [LICENSE](LICENSE).
