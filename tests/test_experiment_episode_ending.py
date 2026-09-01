from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from rcp.core.models import ExperimentDecisionPin
from rcp.limits import AGENT_TASK_RECEIPT_MAX_BYTES
from rcp.runs.experiment_loop import (
    commit_experiment_episode_binding,
    commit_experiment_episode_handoff,
    experiment_loop_ending_signal,
    experiment_loop_launch_failure_diagnostic,
    experiment_loop_semantic_ending,
    experiment_loop_wrapup_spec,
    prepare_experiment_watcher_records,
)
from rcp.service import GraphUpdateResult, RunRequest
from rcp.storage import AgentTaskRecord, WatcherContinuation
from rcp.watchers import ExperimentWatchSpec, WatcherBinding, WatcherCheckResult

_CONTROL_NODE_ID = "exp/evaluation"
_EPISODE_ID = "00000000-0000-4000-8000-000000000099"


def _patch(*ops: dict[str, object]) -> str:
    return json.dumps(
        {
            "summary": "Recorded the bounded Experiment result.",
            "repositories_read": ["repo-a"],
            "change_summary": ["Updated the Experiment control state."],
            "ops": list(ops),
        }
    )


def _proposal_operation() -> dict[str, object]:
    return {
        "op": "create_proposals",
        "proposals": [
            {
                "id": "prop/review-claim",
                "title": "Review the claim transition",
                "card": {
                    "situation_cold": "The experiment changed the evidence state.",
                    "why_human_now": "Belief status remains human-authoritative.",
                    "consequences": "Approval would promote the claim.",
                    "decision_needed": "Approve or reject the transition.",
                },
                "ops": [
                    {
                        "op": "update_nodes",
                        "intent": "status_change",
                        "nodes": [
                            {
                                "id": "hyp/claim",
                                "changes": {"status": "active"},
                                "cause": {
                                    "kind": "evidence_edge",
                                    "ref_id": "edge/result-supports-claim",
                                },
                            }
                        ],
                    }
                ],
            }
        ],
    }


def _blocker_operations() -> list[dict[str, object]]:
    return [
        {
            "op": "create_nodes",
            "nodes": [
                {
                    "id": "blk/missing-data",
                    "type": "blocker",
                    "title": "Missing evaluation data",
                    "description": "The authoritative result has not arrived.",
                    "blocker_type": "data",
                    "status": "open",
                    "resolution_condition": "The result becomes available.",
                }
            ],
        },
        {
            "op": "create_edges",
            "edges": [
                {
                    "source": _CONTROL_NODE_ID,
                    "target": "blk/missing-data",
                    "relation": "blocked_by",
                    "explanation": "The experiment cannot proceed without the result.",
                }
            ],
        },
    ]


def test_completion_has_priority_over_every_human_pause_signal() -> None:
    patch_text = _patch(
        {
            "op": "update_nodes",
            "nodes": [
                {
                    "id": _CONTROL_NODE_ID,
                    "changes": {"status": "completed", "next_action": None},
                },
                {
                    "id": "dec/evaluation-budget",
                    "changes": {"status": "ready"},
                },
            ],
        },
        _proposal_operation(),
        *_blocker_operations(),
    )

    ending = experiment_loop_semantic_ending(patch_text, _CONTROL_NODE_ID)

    assert ending is not None
    assert ending.ending == "completed"
    assert ending.partial is False
    assert ending.signals == (
        "experiment_completed",
        "proposal_created",
        "decision_awaits_human",
        "blocker_linked",
    )


@pytest.mark.parametrize(
    ("operations", "signal"),
    [
        ([_proposal_operation()], "proposal_created"),
        (
            [
                {
                    "op": "update_nodes",
                    "nodes": [
                        {
                            "id": "dec/evaluation-budget",
                            "changes": {"status": "ready"},
                        }
                    ],
                }
            ],
            "decision_awaits_human",
        ),
        (_blocker_operations(), "blocker_linked"),
    ],
)
def test_each_human_authority_signal_pauses_the_episode(
    operations: list[dict[str, object]],
    signal: str,
) -> None:
    ending = experiment_loop_semantic_ending(_patch(*operations), _CONTROL_NODE_ID)

    assert ending is not None
    assert ending.ending == "human_pause"
    assert ending.partial is True
    assert ending.signals == (signal,)


def test_revisit_decision_is_a_human_authority_pause() -> None:
    ending = experiment_loop_semantic_ending(
        _patch(
            {
                "op": "update_nodes",
                "nodes": [
                    {
                        "id": "dec/evaluation-budget",
                        "changes": {"status": "revisit"},
                    }
                ],
            }
        ),
        _CONTROL_NODE_ID,
    )

    assert ending is not None
    assert ending.ending == "human_pause"
    assert ending.signals == ("decision_awaits_human",)


def test_linking_an_existing_blocker_is_a_human_authority_pause() -> None:
    ending = experiment_loop_semantic_ending(
        _patch(
            {
                "op": "create_edges",
                "edges": [
                    {
                        "source": _CONTROL_NODE_ID,
                        "target": "blk/existing-infrastructure",
                        "relation": "blocked_by",
                        "explanation": "The existing infrastructure blocker prevents progress.",
                    }
                ],
            }
        ),
        _CONTROL_NODE_ID,
    )

    assert ending is not None
    assert ending.ending == "human_pause"
    assert ending.signals == ("blocker_linked",)


def test_ordinary_progress_has_no_semantic_ending() -> None:
    patch_text = _patch(
        {
            "op": "update_nodes",
            "nodes": [
                {
                    "id": _CONTROL_NODE_ID,
                    "changes": {
                        "status": "analyzing",
                        "current_summary": "The next comparison is still in progress.",
                        "next_action": "Analyze the held-out comparison.",
                    },
                }
            ],
        }
    )

    assert experiment_loop_semantic_ending(patch_text, _CONTROL_NODE_ID) is None


def test_ending_signal_is_compact_mode_specific_and_adapts_to_exact_continuation() -> None:
    attempts = [
        {
            "id": f"attempt-{index}",
            "sequence": index,
            "purpose": f"Measure configuration {index} " + ("purpose " * 100),
            "configuration": f"seed={index} " + ("configuration " * 100),
            "status": "completed",
            "outcome": f"Observed score {index} " + ("observation " * 100),
            "job_refs": [f"job-{index}-" + ("reference" * 30)],
            "debug": {
                "mechanical_fault": "fault " * 100,
                "change": "change " * 100,
                "predicted_effect": "effect " * 100,
            },
            "source_refs": [{"path": f"private-{index}"}],
            "transcript": f"do-not-copy-transcript-{index}",
        }
        for index in range(1, 11)
    ]
    control_snapshot: dict[str, object] = {
        "id": _CONTROL_NODE_ID,
        "type": "experiment",
        "title": "Evaluate adaptation",
        "objective": "Measure whether replanning restores adaptation. " + ("objective " * 100),
        "design": "Compare the bounded interventions on held-out seeds. " + ("design " * 100),
        "expected_outcomes": ["Replanning improves recovery. " + ("expected " * 100)] * 6,
        "interpretation_rules": ["Treat seed variance as uncertainty. " + ("interpretation " * 100)]
        * 6,
        "completion_criteria": ["All bounded comparisons are analyzed. " + ("criterion " * 100)]
        * 6,
        "current_summary": "Stale pre-turn summary.",
        "next_action": "Stale pre-turn action.",
        "attempts": [],
        "graph_path": "do-not-copy-graph-path",
        "research": "do-not-copy-research",
        "history": "do-not-copy-history",
        "repositories": ["do-not-copy-repository"],
    }
    patch_text = _patch(
        {
            "op": "update_nodes",
            "nodes": [
                {
                    "id": _CONTROL_NODE_ID,
                    "changes": {
                        "status": "completed",
                        "current_summary": "All bounded comparisons support replanning.",
                        "next_action": None,
                        "attempts": attempts,
                    },
                }
            ],
        }
    )
    semantic_ending = experiment_loop_semantic_ending(patch_text, _CONTROL_NODE_ID)
    assert semantic_ending is not None

    signal = experiment_loop_ending_signal(
        semantic_ending=semantic_ending,
        episode_id=_EPISODE_ID,
        control_node_id=_CONTROL_NODE_ID,
        invocation=3,
        invocation_ceiling=4,
        control_snapshot=control_snapshot,
        patch_text=patch_text,
        graph_update=GraphUpdateResult(status="applied", applied_revision=42),
        watcher_ids=[f"watcher-{index}-" + ("identifier" * 30) for index in range(20)],
        stopped_watcher_ids=[f"stopped-{index}-" + ("identifier" * 30) for index in range(18)],
        decision_bundle=[
            ExperimentDecisionPin(
                decision_id="dec/evaluation-budget",
                decision_revision=7,
                selected_option="bounded " + ("selection " * 100),
            )
        ],
    )

    assert set(signal) == {"episode_id", "ending", "partial", "receipt"}
    assert signal["episode_id"] == _EPISODE_ID
    assert signal["ending"] == "completed"
    assert signal["partial"] is False
    receipt = signal["receipt"]
    assert isinstance(receipt, dict)
    assert set(receipt) == {
        "control",
        "invocation",
        "attempt_observations",
        "omitted_attempt_count",
        "watcher_summary",
        "graph_result",
        "semantic_signals",
    }
    control = receipt["control"]
    assert isinstance(control, dict)
    assert control["current_summary"] == "All bounded comparisons support replanning."
    assert control["next_action"] is None
    observations = receipt["attempt_observations"]
    assert isinstance(observations, list)
    assert 1 <= len(observations) <= 8
    assert [item["id"] for item in observations] == [
        f"attempt-{index}" for index in range(11 - len(observations), 11)
    ]
    assert receipt["omitted_attempt_count"] + len(observations) == 10
    watcher_summary = receipt["watcher_summary"]
    assert isinstance(watcher_summary, dict)
    assert watcher_summary["armed_count"] == 20
    assert len(watcher_summary["armed_ids"]) == 4
    assert watcher_summary["stopped_count"] == 18
    assert len(watcher_summary["stopped_ids"]) == 4
    serialized = json.dumps(signal, sort_keys=True)
    for excluded in (
        "do-not-copy-graph-path",
        "do-not-copy-research",
        "do-not-copy-history",
        "do-not-copy-repository",
        "do-not-copy-transcript",
        "source_refs",
        "repositories_read",
    ):
        assert excluded not in serialized
    assert len(serialized.encode("utf-8")) <= AGENT_TASK_RECEIPT_MAX_BYTES

    spec = experiment_loop_wrapup_spec("continuation-operation", signal)
    assert spec.episode_id == _EPISODE_ID
    assert spec.ending == "completed"
    assert spec.partial is False
    assert spec.continuation_operation_id == "continuation-operation"
    assert spec.receipt == receipt


class _RecordingStore:
    def __init__(self, *, fail_commit: bool = False) -> None:
        self.fail_commit = fail_commit
        self.events: list[tuple[str, object]] = []

    def agent_task(self, operation_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            project_id="project-id",
            parent_operation_id=(
                "root-operation" if operation_id == "continuation-operation" else None
            ),
        )

    def experiment_episode(self, episode_id: str) -> None:
        return None

    def now(self) -> str:
        return "2026-08-19T00:00:00+00:00"

    def commit_experiment_episode_turn(self, **values: object) -> None:
        self.events.append(("commit", (values["operation_id"], values.get("ending_signal"))))
        if self.fail_commit:
            raise RuntimeError("handoff commit failed")

    def commit_experiment_episode_handoff(
        self, records: list[object], **values: object
    ) -> tuple[list[object], None]:
        binding = values["binding"]
        self.events.append(
            (
                "compound",
                (values["operation_id"], binding.origin_operation_id, len(records)),
            )
        )
        if self.fail_commit:
            raise RuntimeError("handoff commit failed")
        return records, None

    def record_agent_task_receipt(
        self,
        operation_id: str,
        category: str,
        payload: dict[str, object],
    ) -> None:
        self.events.append((category, (operation_id, payload)))


def _binding_request() -> RunRequest:
    return RunRequest(
        provider="codex",
        run_on="laptop",
        node_id=_CONTROL_NODE_ID,
        chat_id="episode-chat",
        mode="work",
        patch_kind="experiment_loop",
        control_node_id=_CONTROL_NODE_ID,
        control_episode_id=_EPISODE_ID,
        control_invocation=2,
        control_invocation_ceiling=4,
    )


def _minimal_ending_signal() -> dict[str, object]:
    return {
        "episode_id": _EPISODE_ID,
        "ending": "completed",
        "partial": False,
        "receipt": {"semantic_signals": ["experiment_completed"]},
    }


def _handoff_binding() -> WatcherBinding:
    return WatcherBinding(
        project_id="project-id",
        origin_operation_id="root-operation",
        origin_task_kind="node_chat",
        chat_id="episode-chat",
        node_id=_CONTROL_NODE_ID,
        episode_id=_EPISODE_ID,
        execution_host="",
        continuation=WatcherContinuation(
            provider="codex",
            run_on="laptop",
            patch_kind="experiment_loop",
            control_node_id=_CONTROL_NODE_ID,
            control_episode_id=_EPISODE_ID,
            control_invocation=2,
            control_invocation_ceiling=4,
        ),
    )


def test_exit_receipt_is_written_on_exact_operation_after_binding_commit() -> None:
    store = _RecordingStore()
    execution = SimpleNamespace(
        operation_id="continuation-operation", store=store, continuation="fresh"
    )

    commit_experiment_episode_binding(
        execution,
        _binding_request(),
        native_session_id="native-session",
        execution_host="",
        stage_host=None,
        stage_root="/tmp/exact-episode-stage",
        graph_result="applied as revision 42",
        watcher_ids=[],
        context_baseline={"revision": 42},
        ending_signal=_minimal_ending_signal(),
    )

    assert [event[0] for event in store.events] == [
        "commit",
        "experiment_episode_binding",
    ]
    operation_id, ending_signal = store.events[0][1]
    assert operation_id == "continuation-operation"
    assert ending_signal == _minimal_ending_signal()


def test_failed_binding_commit_cannot_publish_an_exit_receipt() -> None:
    store = _RecordingStore(fail_commit=True)
    execution = SimpleNamespace(
        operation_id="continuation-operation", store=store, continuation="fresh"
    )

    with pytest.raises(RuntimeError, match="handoff commit failed"):
        commit_experiment_episode_binding(
            execution,
            _binding_request(),
            native_session_id="native-session",
            execution_host="",
            stage_host=None,
            stage_root="/tmp/exact-episode-stage",
            graph_result="applied as revision 42",
            watcher_ids=[],
            context_baseline={"revision": 42},
            ending_signal=_minimal_ending_signal(),
        )

    assert store.events == [("commit", ("continuation-operation", _minimal_ending_signal()))]


def test_prepared_handoff_uses_rooted_watchers_and_commits_receipt_after_storage() -> None:
    store = _RecordingStore()
    execution = SimpleNamespace(
        operation_id="continuation-operation", store=store, continuation="fresh"
    )
    binding = _handoff_binding()
    prepared = prepare_experiment_watcher_records(
        execution,
        [ExperimentWatchSpec(check_command="true", log_path="/tmp/result.log", cwd="/tmp")],
        [WatcherCheckResult(state="complete", checked_at="2026-08-19T00:01:00+00:00")],
        binding,
    )

    assert len(prepared) == 1
    assert prepared[0].origin_operation_id == "root-operation"
    stored = commit_experiment_episode_handoff(
        execution,
        _binding_request(),
        prepared,
        binding,
        native_session_id="native-session",
        execution_host="",
        stage_host=None,
        stage_root="/tmp/exact-episode-stage",
        graph_result="applied as revision 42",
        context_baseline={"revision": 42},
    )

    assert stored == prepared
    assert store.events == [
        ("compound", ("continuation-operation", "root-operation", 1)),
        (
            "experiment_episode_binding",
            (
                "continuation-operation",
                {
                    "episode_id": _EPISODE_ID,
                    "invocation": 2,
                    "provider": "codex",
                    "execution_machine": "laptop",
                    "stage_host": None,
                    "stage_root": "/tmp/exact-episode-stage",
                    "graph_result": "applied as revision 42",
                    "watcher_ids": [prepared[0].watcher_id],
                    "binding_replaced": False,
                },
            ),
        ),
    ]


def _failed_launch_task(**fields: object) -> AgentTaskRecord:
    values: dict[str, object] = {
        "operation_id": "operation",
        "project_id": "project",
        "episode_id": _EPISODE_ID,
        "kind": "node_chat",
        "status": "failed",
        "request": {},
        "created_at": "2026-08-21T11:22:37+00:00",
        "updated_at": "2026-08-21T11:22:48+00:00",
        "status_message": "repository 'vista' does not match its project execution host",
    }
    values.update(fields)
    return AgentTaskRecord.model_validate(values)


def test_launch_failure_names_an_available_action_and_the_real_cause() -> None:
    diagnostic = experiment_loop_launch_failure_diagnostic(
        _failed_launch_task(error="repository 'vista' does not match its project execution host")
    )

    assert "before it started its agent session" in diagnostic
    assert "repository 'vista' does not match its project execution host" in diagnostic
    # The ending fence retires Stop loop, so the diagnostic must never send the
    # human to it, and it must not blame a pre-migration lineage.
    assert "Stop loop" not in diagnostic
    assert "pre-migration" not in diagnostic
    assert "Press Run" in diagnostic


def test_launch_failure_falls_back_to_the_status_message() -> None:
    diagnostic = experiment_loop_launch_failure_diagnostic(_failed_launch_task())

    assert "repository 'vista' does not match its project execution host" in diagnostic


def test_launch_failure_without_any_cause_still_explains_itself() -> None:
    diagnostic = experiment_loop_launch_failure_diagnostic(
        _failed_launch_task(status_message="", error=None)
    )

    assert diagnostic.endswith("Press Run to start a fresh episode.")
