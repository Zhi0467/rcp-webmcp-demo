"""Challenge-only deterministic provider executable.

This is a normal provider CLI from RCP's point of view. It deliberately does
not interpret the researcher's message or selected skills; it only reads the
RCP-authored contract pointer needed to identify the operation and its staged
output locations.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

VERSION = "rcp-demo 1.0"

_EXPERIMENT_HEADING = "# RCP Experiment-loop task contract"
_REPORT_HEADING = "# RCP episode report contract"
_CONTRACT_POINTER_PREFIX = "Open and follow the immutable RCP task contract at:\n"
_CONTRACT_POINTER_SUFFIX = "\nThat contract is the sole RCP task and authority source"
_DEMO_EXPERIMENT_ID = "exp/two-update-matched-trajectory"
_DEMO_HYPOTHESIS_ID = "hyp/search-restores-future-learning"
_DEMO_EVIDENCE_ID = "ev/held-out-plasticity-replicate"


@dataclass(frozen=True)
class DemoContract:
    operation: str
    path: Path | None = None
    patch_path: Path | None = None
    watch_path: Path | None = None
    artifact_path: Path | None = None
    report_path: Path | None = None
    graph_path: Path | None = None
    experiment_id: str | None = None


def resolve_contract(prompt: str, *, capability: str) -> DemoContract:
    """Resolve only RCP-authored operation metadata and exact output paths."""

    if capability == "discuss":
        return DemoContract(operation="discuss")
    contract_path = _contract_path(prompt)
    if contract_path is None:
        return DemoContract(operation=capability)
    contract = contract_path.read_text(encoding="utf-8")
    if contract.startswith(_EXPERIMENT_HEADING):
        return DemoContract(
            operation="experiment",
            path=contract_path,
            patch_path=_required_contract_path(contract, "- Optional semantic graph Patch: `"),
            watch_path=_required_contract_path(
                contract,
                "- Required watcher handoff that continues this Experiment's bounded loop: `",
            ),
            artifact_path=_required_contract_path(
                contract,
                "- Optional preview artifact directory: `",
            ),
            graph_path=_required_contract_path(
                contract,
                "- Current graph, including the Experiment's attempts: `",
            ),
            experiment_id=_required_contract_value(contract, "- Focused Experiment id: `"),
        )
    if contract.startswith(_REPORT_HEADING):
        return DemoContract(
            operation="report",
            path=contract_path,
            report_path=_required_contract_path(
                contract,
                "- self-contained sandbox-safe HTML report: `",
            ),
        )
    return DemoContract(operation=capability, path=contract_path)


def _contract_path(prompt: str) -> Path | None:
    if not prompt.startswith(_CONTRACT_POINTER_PREFIX):
        return None
    remainder = prompt[len(_CONTRACT_POINTER_PREFIX) :]
    raw_path, separator, _rest = remainder.partition(_CONTRACT_POINTER_SUFFIX)
    if not separator or not raw_path or "\n" in raw_path:
        raise ValueError("RCP Demo received a malformed task-contract pointer.")
    path = Path(raw_path)
    if not path.is_absolute():
        raise ValueError("RCP Demo requires an absolute task-contract path.")
    return path


def _required_contract_path(contract: str, prefix: str) -> Path:
    path = Path(_required_contract_value(contract, prefix))
    if not path.is_absolute():
        raise ValueError("RCP Demo output paths must be absolute.")
    return path


def _required_contract_value(contract: str, prefix: str) -> str:
    matches = [
        line[len(prefix) : -1]
        for line in contract.splitlines()
        if line.startswith(prefix) and line.endswith("`")
    ]
    if len(matches) != 1:
        raise ValueError(f"RCP Demo contract must contain one {prefix.strip()!r} value.")
    return matches[0]


def _require_stage_path(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(Path.cwd().resolve())
    except ValueError as exc:
        raise ValueError("RCP Demo refuses an output path outside the exact task stage.") from exc
    return resolved


def _write_experiment_result(contract: DemoContract) -> None:
    if (
        contract.experiment_id != _DEMO_EXPERIMENT_ID
        or contract.graph_path is None
        or contract.patch_path is None
        or contract.watch_path is None
        or contract.artifact_path is None
    ):
        raise ValueError("RCP Demo received an unknown or incomplete Experiment contract.")
    graph = json.loads(contract.graph_path.read_text(encoding="utf-8"))
    nodes = graph.get("nodes") if isinstance(graph, dict) else None
    experiment = nodes.get(_DEMO_EXPERIMENT_ID) if isinstance(nodes, dict) else None
    if not isinstance(experiment, dict):
        raise ValueError("RCP Demo fixture is missing its expected Experiment.")
    attempts = copy.deepcopy(experiment.get("attempts"))
    if not isinstance(attempts, list):
        raise ValueError("RCP Demo fixture Experiment has no attempt ledger.")
    held_out = [
        item for item in attempts if isinstance(item, dict) and item.get("id") == "attempt/04"
    ]
    if len(held_out) != 1 or held_out[0].get("status") != "planned":
        raise ValueError("RCP Demo held-out replicate is not ready for its fixed completion.")
    held_out[0].update(
        {
            "status": "completed",
            "outcome": (
                "All 30 rows passed the match gate. Maximum first-shift gaps were 0.01 "
                "for return and 0.002 for policy KL. Mean second-shift slopes were "
                "0.178333 for search-assisted and 0.025333 for value-only, a difference "
                "of 0.153."
            ),
            "failure_reason": None,
            "finished_at": "2026-09-01T19:00:00Z",
        }
    )

    patch = {
        "summary": "Recorded the completed held-out plasticity replicate.",
        "repositories_read": ["crlp-demo-state"],
        "change_summary": [
            "The held-out replicate completed and produced scoped internal-run Evidence.",
            "The broader plasticity question and Hypothesis standing remain unchanged.",
        ],
        "ops": [
            {
                "op": "update_nodes",
                "nodes": [
                    {
                        "id": _DEMO_EXPERIMENT_ID,
                        "changes": {
                            "attempts": attempts,
                            "status": "completed",
                            "current_summary": (
                                "The retained 30-row synthetic replicate passed the first-shift "
                                "match gate. The search-assisted mean second-shift slope was "
                                "0.178333 versus 0.025333 for value-only."
                            ),
                            "next_action": None,
                        },
                    }
                ],
            },
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": _DEMO_EVIDENCE_ID,
                        "type": "evidence",
                        "title": "Held-out second-shift learning curves",
                        "observation": (
                            "All 30 retained rows passed the match gate. The search-assisted "
                            "second-shift slopes were 0.179, 0.172, and 0.184; the value-only "
                            "slopes were 0.026, 0.025, and 0.025."
                        ),
                        "interpretation": (
                            "The replicate supports a scoped difference in future learning under "
                            "this matched setup, but does not identify the mechanism or settle the "
                            "broader plasticity question."
                        ),
                        "role": "result",
                        "validity": "qualified",
                        "origin": "internal_run",
                        "artifact_refs": [
                            "study/held_out_trajectory.csv",
                            "study/analyze_held_out.py",
                        ],
                    }
                ],
            },
            {
                "op": "create_edges",
                "edges": [
                    {
                        "id": "edge/demo-experiment-produces-held-out-evidence",
                        "source": _DEMO_EXPERIMENT_ID,
                        "target": _DEMO_EVIDENCE_ID,
                        "relation": "produces",
                        "explanation": "The completed replicate produced this result.",
                    },
                    {
                        "id": "edge/demo-held-out-evidence-inconclusive-hypothesis",
                        "source": _DEMO_EVIDENCE_ID,
                        "target": _DEMO_HYPOTHESIS_ID,
                        "relation": "inconclusive",
                        "explanation": (
                            "The scoped result is consistent with preserved future learning but "
                            "does not isolate search as the causal mechanism."
                        ),
                        "assessment": {
                            "relevance": "direct",
                            "weight": "moderate",
                            "scope": (
                                "Three fixed synthetic seeds per arm across five held-out updates."
                            ),
                            "qualifications": [
                                "The fixture is synthetic.",
                                "The replicate does not identify the causal mechanism.",
                            ],
                        },
                    },
                ],
            },
        ],
    }
    patch_path = _require_stage_path(contract.patch_path)
    watch_path = _require_stage_path(contract.watch_path)
    artifact_path = _require_stage_path(contract.artifact_path)
    artifact_path.mkdir(parents=True, exist_ok=True)
    patch_path.write_text(json.dumps(patch, indent=2) + "\n", encoding="utf-8")
    watch_path.write_text('{"external":[],"graph":[]}\n', encoding="utf-8")
    (artifact_path / "held-out-plasticity-replicate.html").write_text(
        _RESULT_ARTIFACT,
        encoding="utf-8",
    )


def _write_episode_report(contract: DemoContract) -> None:
    if contract.report_path is None:
        raise ValueError("RCP Demo received an incomplete episode report contract.")
    report_path = _require_stage_path(contract.report_path)
    report_path.write_text(_EPISODE_REPORT, encoding="utf-8")


def _emit(value: dict[str, object]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":")), flush=True)


def _run_turn(args: argparse.Namespace) -> int:
    prompt = sys.stdin.read()
    contract = resolve_contract(prompt, capability=args.capability)
    if contract.operation == "experiment":
        _write_experiment_result(contract)
    elif contract.operation == "report":
        _write_episode_report(contract)
    session_id = args.session or str(uuid5(NAMESPACE_URL, f"rcp-demo:{Path.cwd()}"))
    _emit({"type": "session.started", "session_id": session_id})
    _emit(
        {
            "type": "message",
            "message": f"RCP Demo is running the fixed {contract.operation} response.",
        }
    )
    answer = {
        "discuss": (
            "The retained 30-row table passes the first-shift matching gate. It is ready for the "
            "bounded second-shift slope comparison, while the broader plasticity claim remains "
            "open."
        ),
        "experiment": "RCP Demo inspected the fixed held-out replicate and recorded its result.",
        "report": "RCP Demo wrote the fixed synthetic episode report.",
    }.get(contract.operation, "RCP Demo completed the fixed challenge response.")
    _emit({"type": "result", "result": answer})
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rcp-demo")
    parser.add_argument("--version", action="store_true")
    subparsers = parser.add_subparsers(dest="command")
    auth = subparsers.add_parser("auth")
    auth.add_argument("action", choices=("status", "login"))
    skills = subparsers.add_parser("skills")
    skills.add_argument("action", choices=("list",))
    turn = subparsers.add_parser("turn")
    turn.add_argument("--capability", required=True)
    turn.add_argument("--session")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.version:
        print(VERSION)
        return 0
    if args.command == "auth":
        print("RCP Demo is ready")
        return 0
    if args.command == "skills":
        print(json.dumps({"skills": []}, separators=(",", ":")))
        return 0
    if args.command == "turn":
        try:
            return _run_turn(args)
        except (OSError, UnicodeError, ValueError) as exc:
            _emit({"type": "error", "error": str(exc)})
            return 1
    _parser().error("a command is required")
    return 2


_EPISODE_REPORT = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Held-out plasticity episode report</title>
<style>
  :root { color-scheme: light; font-family: ui-sans-serif, system-ui, sans-serif; }
  body { margin: 0; background: #f5f0e7; color: #29251f; }
  main { max-width: 820px; margin: 0 auto; padding: 44px 30px 60px; }
  h1, h2 { font-family: Georgia, serif; }
  h1 { max-width: 680px; margin: 8px 0 12px; font-size: 42px; line-height: 1.06; }
  h2 { margin: 0 0 12px; font-size: 24px; }
  p, li { line-height: 1.6; }
  .eyebrow { color: #8f3e32; font-size: 12px; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; }
  .lede { max-width: 700px; color: #645c50; font-size: 18px; }
  .outcome { margin: 28px 0; padding: 20px 22px; border-left: 5px solid #9b3e32; background: #fffdf8; }
  .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }
  .card { padding: 22px; border: 1px solid #cbbda8; background: #fffdf8; }
  .label { color: #776d60; font-size: 12px; font-weight: 800; letter-spacing: .1em; text-transform: uppercase; }
  .metric { margin: 8px 0 0; font: 700 26px/1.2 Georgia, serif; }
  .scope { margin-top: 24px; padding-top: 20px; border-top: 1px solid #cbbda8; color: #5e564b; }
  @media (max-width: 620px) { .grid { grid-template-columns: 1fr; } h1 { font-size: 34px; } }
</style>
</head>
<body>
<main>
  <div class="eyebrow">Synthetic challenge episode · completed</div>
  <h1>Held-out learning separated after the second shift</h1>
  <p class="lede">All 30 retained rows passed the first-shift match gate. Across three fixed
  synthetic seeds, the search-assisted arm had a mean second-shift slope of 0.178333 versus
  0.025333 for value-only.</p>

  <section class="outcome" aria-labelledby="outcome-title">
    <h2 id="outcome-title">Outcome</h2>
    <p>The mean slope difference was 0.153. The run produced qualified internal Evidence for
    this scoped difference in future learning; it did not identify the causal mechanism, settle
    the broader plasticity question, or change the Hypothesis standing.</p>
  </section>

  <section class="grid" aria-label="Episode summary">
    <article class="card">
      <div class="label">Operational result</div>
      <p class="metric">30 rows · 6 curves</p>
      <p>Every seed-arm trajectory contained updates 0 through 4, and both matching diagnostics
      remained below the declared 0.02 tolerance.</p>
    </article>
    <article class="card">
      <div class="label">Scientific standing</div>
      <p class="metric">0.153 slope gap</p>
      <p><strong>Qualified, inconclusive:</strong> the new Evidence is directly relevant but
      deliberately narrower than the research Hypothesis it informs.</p>
    </article>
  </section>

  <section class="scope" aria-labelledby="scope-title">
    <h2 id="scope-title">What remains open</h2>
    <ul>
      <li>Whether the measured difference generalizes beyond this matched synthetic setup.</li>
      <li>Whether search itself, rather than another correlated mechanism, causes the effect.</li>
      <li>Whether the result survives broader seeds, tasks, and real training trajectories.</li>
    </ul>
    <p><strong>Recommended next step:</strong> preserve the result as scoped Evidence and design a
    mechanism-separating follow-up before changing the Hypothesis standing.</p>
  </section>

  <p class="scope">This deterministic synthetic report demonstrates RCP's real episode wrap-up
  and sandboxed report lifecycle for the WebMCP challenge. It is not a new scientific claim.</p>
</main>
</body>
</html>
"""


_RESULT_ARTIFACT = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Held-out plasticity replicate</title>
<style>
  :root { color-scheme: light; font-family: ui-sans-serif, system-ui, sans-serif; }
  body { margin: 0; background: #f5f0e7; color: #29251f; }
  main { max-width: 760px; margin: 0 auto; padding: 44px 30px 56px; }
  h1 { max-width: 620px; margin: 0 0 10px; font: 700 42px/1.05 Georgia, serif; }
  .lede { max-width: 650px; color: #645c50; font-size: 18px; line-height: 1.55; }
  .card { margin-top: 28px; padding: 26px; border: 1px solid #cbbda8; background: #fffdf8; }
  .eyebrow { color: #8f3e32; font-size: 12px; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; }
  svg { display: block; width: 100%; height: auto; margin-top: 18px; }
  .axis { stroke: #8f8577; stroke-width: 1; }
  .search { fill: none; stroke: #9b3e32; stroke-width: 3; opacity: .72; }
  .value { fill: none; stroke: #536b73; stroke-width: 3; opacity: .72; }
  .legend { display: flex; gap: 24px; margin-top: 14px; font-weight: 700; }
  .swatch { display: inline-block; width: 18px; height: 4px; margin: 0 8px 4px 0; }
  .metrics { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 18px; }
  .metric { padding: 14px; background: #f5f0e7; }
  .metric strong { display: block; color: #8f3e32; font: 700 22px/1.2 Georgia, serif; }
  .metric span { color: #6f6659; font-size: 13px; line-height: 1.4; }
  .finding { margin-top: 24px; padding-left: 18px; border-left: 4px solid #9b3e32; line-height: 1.55; }
  .limit { margin-top: 18px; color: #6f6659; font-size: 14px; line-height: 1.5; }
  @media (max-width: 620px) { .metrics { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<main>
  <div class="eyebrow">Synthetic challenge result · completed replicate</div>
  <h1>Future learning separates after the second shift</h1>
  <p class="lede">Three fixed synthetic seeds show a recovered learning slope for the
  search-assisted arm while the matched value-only arm remains nearly flat.</p>
  <section class="card" aria-labelledby="chart-title">
    <h2 id="chart-title">Held-out second-shift return</h2>
    <svg viewBox="0 0 680 330" role="img"
      aria-label="All three search-assisted returns rise faster than all three value-only returns across five updates.">
      <line class="axis" x1="70" y1="275" x2="640" y2="275" />
      <line class="axis" x1="70" y1="35" x2="70" y2="275" />
      <polyline class="value" points="70,252 210,246 350,238 490,230 630,224" />
      <polyline class="value" points="70,255 210,249 350,244 490,235 630,227" />
      <polyline class="value" points="70,249 210,244 350,235 490,230 630,221" />
      <polyline class="search" points="70,252 210,218 350,173 490,114 630,52" />
      <polyline class="search" points="70,255 210,224 350,179 490,122 630,63" />
      <polyline class="search" points="70,249 210,213 350,165 490,105 630,43" />
      <text x="350" y="315" text-anchor="middle">Second-shift update (0–4)</text>
      <text x="18" y="160" text-anchor="middle" transform="rotate(-90 18 160)">Return</text>
    </svg>
    <div class="legend">
      <span><i class="swatch" style="background:#9b3e32"></i>Search-assisted</span>
      <span><i class="swatch" style="background:#536b73"></i>Matched value-only</span>
    </div>
    <div class="metrics" aria-label="Analysis diagnostics">
      <div class="metric"><strong>0.01</strong><span>maximum first-shift return gap</span></div>
      <div class="metric"><strong>0.002</strong><span>maximum first-shift policy-KL gap</span></div>
      <div class="metric"><strong>0.153</strong><span>mean second-shift slope difference</span></div>
    </div>
  </section>
  <p class="finding"><strong>Scoped result:</strong> the held-out replicate supports a
  difference in future learning under this matched setup. RCP records it as qualified,
  inconclusive Evidence rather than changing the Hypothesis standing.</p>
  <p class="limit">This is a deterministic synthetic artifact for the WebMCP challenge.
  It demonstrates RCP's real task, Patch, artifact, and Experiment lifecycle without claiming
  a new scientific finding.</p>
</main>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
