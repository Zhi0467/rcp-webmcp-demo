from __future__ import annotations

from pydantic import ConfigDict, Field, model_validator

from rcp.service import RunRequest


class BranchMergeRunRequest(RunRequest):
    """Pinned, graph-only request for one human-dispatched branch merge task."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_dedicated_merge_shape(self) -> BranchMergeRunRequest:
        if (
            self.provider is None
            or self.model is None
            or self.reasoning is None
            or self.run_on is None
            or not self.run_truth_scope
        ):
            raise ValueError("branch merge requires one fully pinned orchestrator profile")
        if self.run_truth_scope != sorted(set(self.run_truth_scope)):
            raise ValueError("branch merge run truth scope must be sorted and unique")
        if self.mode != "work" or self.trigger != "human" or self.patch_kind != "work":
            raise ValueError("branch merge requires one human-triggered Work Patch contract")
        if (
            self.chat_scope != "project"
            or self.chat_id is not None
            or self.node_id is not None
            or self.message is not None
            or self.session_id is not None
            or self.result_view is not None
            or self.control_node_id is not None
            or self.control_revision is not None
            or self.control_episode_id is not None
            or self.control_invocation is not None
            or self.control_invocation_ceiling is not None
            or self.control_decision_bundle
            or self.control_completion_criteria
            or self.watcher_ids
            or self.attachments
            or self.attachment_set_id is not None
            or self.attachment_client_id is not None
            or self.attachment_batch_id is not None
        ):
            raise ValueError("branch merge request contains unrelated chat or control state")
        return self
