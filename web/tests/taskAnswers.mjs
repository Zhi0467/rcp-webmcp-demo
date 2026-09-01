/**
 * Fill in the lifecycle answers the server derives from a task's status.
 *
 * Fixtures stand in for the projection, so they carry what it would have decided
 * rather than leaving the browser to work it out from `status` — which it can no
 * longer do, because the type is opaque.
 */
const LABELS = {
  queued: "Queued",
  running: "Running in the background",
  pausing: "Pausing",
  paused: "Paused at checkpoint",
  interrupted: "Interrupted",
};

export function withTaskAnswers(task) {
  const status = task.status;
  // Derived last: the answers always follow the status this fixture ended up with,
  // so a spread that changes the status cannot leave a stale answer behind.
  return {
    ...task,
    active: ["queued", "running", "pausing"].includes(status),
    queued: status === "queued",
    pausing: status === "pausing",
    awaiting_human: ["paused", "failed", "interrupted"].includes(status),
    paused: status === "paused",
    failed: status === "failed",
    settled: status === "succeeded",
    finished: ["succeeded", "failed", "interrupted"].includes(status),
    status_label:
      status === "succeeded"
        ? task.applied_revision
          ? `Completed at revision ${task.applied_revision}`
          : "Completed"
        : (LABELS[status] ?? "Failed"),
  };
}

/** The same answers for the loop turn the control projection reports. */
export function withTurnAnswers(operational) {
  const status = operational.current_status;
  return {
    ...operational,
    current_queued: status === "queued",
    current_active: ["queued", "running", "pausing"].includes(status),
    current_awaiting_human: ["paused", "failed", "interrupted"].includes(status),
  };
}

/** Fill the Experiment Runs answers a backend control fixture must publish. */
export function withExperimentControlAnswers(control) {
  return {
    health: "needs_action",
    recommendation: "start_episode",
    run_section: "actionable",
    live: false,
    can_start: true,
    can_stop: false,
    stop_pending: false,
    task_control: null,
    can_switch_provider: false,
    can_open_report: false,
    report_episode_id: null,
    node_closed: false,
    ...control,
  };
}
