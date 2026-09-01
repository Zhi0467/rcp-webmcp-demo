from __future__ import annotations

import inspect
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute

from .helpers import create_named_app

RouteEntry = tuple[tuple[str, ...], str]


# This is the structural safety net for route extraction. Keep this literal
# unchanged when a handler moves: it records the route surface, not ownership.
_FROZEN_ROUTE_INVENTORY: tuple[RouteEntry, ...] = (
    (("GET", "HEAD"), "/openapi.json"),
    (("GET", "HEAD"), "/docs"),
    (("GET", "HEAD"), "/docs/oauth2-redirect"),
    (("GET", "HEAD"), "/redoc"),
    (("GET",), "/api/health"),
    (("GET",), "/api/server-status"),
    (("GET",), "/api/identity"),
    (("PATCH",), "/api/identity"),
    (("POST",), "/api/team/enroll"),
    (("POST",), "/api/team/session/exchange"),
    (("POST",), "/api/team/session/logout"),
    (("GET",), "/api/team/invitations"),
    (("POST",), "/api/team/invitations"),
    (("POST",), "/api/team/credential/rotate"),
    (("POST",), "/api/team/credential/revoke"),
    (("PATCH",), "/api/team/space"),
    (("GET",), "/api/projects"),
    (("GET",), "/api/episodes"),
    (("GET",), "/api/space/users"),
    (("GET",), "/api/project-invitations"),
    (("POST",), "/api/project-invitations/{invitation_id}/{response}"),
    (("GET",), "/api/providers"),
    (("POST",), "/api/projects"),
    (("POST",), "/api/project-setup/preflight"),
    (("POST",), "/api/project-setup/create"),
    (("POST",), "/api/project-provisioning/requests"),
    (("GET",), "/api/project-provisioning/requests"),
    (("GET",), "/api/project-provisioning/requests/{request_id}"),
    (("POST",), "/api/project-provisioning/requests/{request_id}/cancel"),
    (("POST",), "/api/project-provisioning/requests/{request_id}/complete"),
    (("POST",), "/api/project-transfers/incoming-provisioning-requests"),
    (("GET",), "/api/project-transfers/incoming-provisioning-requests"),
    (
        ("GET",),
        "/api/project-transfers/incoming-provisioning-requests/{request_id}",
    ),
    (("POST",), "/api/project-transfers/source-requests"),
    (("POST",), "/api/project-transfers/target-requests"),
    (("GET",), "/api/project-transfers/requests"),
    (("GET",), "/api/project-transfers/requests/{request_id}"),
    (("POST",), "/api/project-transfers/source-requests/{request_id}/link"),
    (("POST",), "/api/project-transfers/target-requests/{request_id}/admit"),
    (("POST",), "/api/project-transfers/source-requests/{request_id}/target-admission"),
    (("GET",), "/api/project-transfers/source-requests/{request_id}/release-boundary"),
    (("POST",), "/api/project-transfers/source-requests/{request_id}/release"),
    (("POST",), "/api/project-transfers/target-requests/{request_id}/restore-reentry"),
    (("POST",), "/api/project-transfers/target-requests/{request_id}/source-release"),
    (("POST",), "/api/project-transfers/requests/{request_id}/archive"),
    (
        ("GET",),
        "/api/native/project-transfers/source-requests/{request_id}/archive",
    ),
    (
        ("POST",),
        "/api/native/project-transfers/target-requests/{request_id}/cleanup-acknowledgment",
    ),
    (
        ("POST",),
        "/api/native/project-transfers/source-requests/{request_id}/target-activation-proof",
    ),
    (
        ("GET",),
        "/api/native/project-transfers/target-requests/{request_id}/activation-proof",
    ),
    (("DELETE",), "/api/caches"),
    (("GET",), "/api/skills/{kind}/{package_id}"),
    (("DELETE",), "/api/projects/{project_id}"),
    (("GET",), "/api/projects/{project_id}"),
    (("GET",), "/api/projects/{project_id}/members"),
    (("POST",), "/api/projects/{project_id}/invitations"),
    (("POST",), "/api/projects/{project_id}/leave"),
    (("GET",), "/api/projects/{project_id}/cached"),
    (("GET",), "/api/projects/{project_id}/cached/revision"),
    (("GET",), "/api/projects/{project_id}/readiness"),
    (("GET",), "/api/projects/{project_id}/graph"),
    (("GET",), "/api/projects/{project_id}/revision"),
    (("HEAD",), "/api/projects/{project_id}/repositories/files/preview"),
    (("GET",), "/api/projects/{project_id}/repositories/files/preview"),
    (("PUT",), "/api/projects/{project_id}/settings"),
    (("POST",), "/api/projects/{project_id}/machines/{machine_alias}/providers/{provider}/resolve"),
    (("GET",), "/api/projects/{project_id}/history"),
    (("GET",), "/api/projects/{project_id}/history/summaries"),
    (("GET",), "/api/projects/{project_id}/sources"),
    (("POST",), "/api/projects/{project_id}/sync"),
    (("GET",), "/api/projects/{project_id}/transition-manifest"),
    (("POST",), "/api/projects/{project_id}/sync/preview"),
    (("DELETE",), "/api/projects/{project_id}/caches"),
    (("POST",), "/api/projects/{project_id}/chats/{chat_id}/attachments"),
    (("DELETE",), "/api/projects/{project_id}/chats/{chat_id}/attachments/{attachment_id}"),
    (("POST",), "/api/projects/{project_id}/tasks/{kind}"),
    (("POST",), "/api/projects/{project_id}/experiments/{node_id:path}/run"),
    (("GET",), "/api/projects/{project_id}/tasks"),
    (("GET",), "/api/projects/{project_id}/usage"),
    (("GET",), "/api/projects/{project_id}/watchers"),
    (("POST",), "/api/projects/{project_id}/watchers/{watcher_id}/check"),
    (("POST",), "/api/projects/{project_id}/watchers/{watcher_id}/stop"),
    (("POST",), "/api/projects/{project_id}/experiments/{node_id:path}/watchers/stop"),
    (("POST",), "/api/projects/{project_id}/experiments/{node_id:path}/stop"),
    (("GET",), "/api/projects/{project_id}/chats"),
    (("GET",), "/api/projects/{project_id}/chats/{chat_id}"),
    (("GET",), "/api/projects/{project_id}/episodes"),
    (("POST",), "/api/projects/{project_id}/episodes"),
    (("POST",), "/api/projects/{project_id}/episodes/{episode_id}/stop"),
    (("POST",), "/api/projects/{project_id}/episodes/{episode_id}/merge"),
    (("POST",), "/api/projects/{project_id}/episodes/{episode_id}/reauthorize"),
    (("GET",), "/api/projects/{project_id}/episodes/{episode_id}/messages"),
    (("POST",), "/api/projects/{project_id}/episodes/{episode_id}/messages"),
    (("HEAD",), "/api/projects/{project_id}/episodes/{episode_id}/report/content"),
    (("GET",), "/api/projects/{project_id}/episodes/{episode_id}/report/content"),
    (("HEAD",), "/api/projects/{project_id}/episodes/{episode_id}/report/preview"),
    (("GET",), "/api/projects/{project_id}/episodes/{episode_id}/report/preview"),
    (("GET",), "/api/projects/{project_id}/episodes/{episode_id}/report/viewer"),
    (("GET",), "/api/projects/{project_id}/result-views"),
    (("HEAD",), "/api/projects/{project_id}/result-views/{view_id}/preview"),
    (("GET",), "/api/projects/{project_id}/result-views/{view_id}/preview"),
    (("POST",), "/api/projects/{project_id}/result-views/{view_id}/keep"),
    (("GET",), "/api/projects/{project_id}/tasks/{operation_id}"),
    (("HEAD",), "/api/projects/{project_id}/tasks/{operation_id}/artifacts/{artifact_id}/content"),
    (("GET",), "/api/projects/{project_id}/tasks/{operation_id}/artifacts/{artifact_id}/content"),
    (("HEAD",), "/api/projects/{project_id}/tasks/{operation_id}/artifacts/{artifact_id}/preview"),
    (("GET",), "/api/projects/{project_id}/tasks/{operation_id}/artifacts/{artifact_id}/preview"),
    (("HEAD",), "/api/projects/{project_id}/tasks/{operation_id}/artifacts/{artifact_id}/download"),
    (("GET",), "/api/projects/{project_id}/tasks/{operation_id}/artifacts/{artifact_id}/download"),
    (("GET",), "/api/projects/{project_id}/tasks/{operation_id}/artifacts/{artifact_id}/viewer"),
    (("POST",), "/api/projects/{project_id}/tasks/{operation_id}/artifacts/{artifact_id}/keep"),
    (("POST",), "/api/projects/{project_id}/tasks/{operation_id}/pause"),
    (("POST",), "/api/projects/{project_id}/tasks/{operation_id}/resume"),
    (("POST",), "/api/projects/{project_id}/tasks/{operation_id}/repair-graph-update"),
    (("POST",), "/api/projects/{project_id}/tasks/{operation_id}/retry"),
    (("GET",), "/api/projects/{project_id}/paper"),
    (("POST",), "/api/projects/{project_id}/paper/create"),
    (("PUT",), "/api/projects/{project_id}/paper"),
    (("GET",), "/api/projects/{project_id}/paper/sessions"),
)


# This map is intentionally independent from _FROZEN_ROUTE_INVENTORY. Update
# only this map as Phase 5 extracts handlers into their owning API modules.
_HANDLER_MODULE_MAP: dict[str, str] = {
    "agent_task": "src/rcp/api/tasks.py",
    "content_agent_artifact": "src/rcp/api/tasks.py",
    "content_episode_report": "src/rcp/api/episode_routes.py",
    "agent_tasks": "src/rcp/api/tasks.py",
    "agent_usage": "src/rcp/api/project_state.py",
    "answer_project_invitation": "src/rcp/api/index.py",
    "cached_project": "src/rcp/api/project_state.py",
    "cached_project_revision": "src/rcp/api/project_state.py",
    "chat": "src/rcp/api/chats.py",
    "chats": "src/rcp/api/chats.py",
    "check_watcher_now": "src/rcp/api/watchers.py",
    "clear_all_rebuildable_caches": "src/rcp/api/index.py",
    "clear_rebuildable_caches": "src/rcp/api/project_state.py",
    "complete_project_provisioning_request": "src/rcp/api/project_provisioning.py",
    "acknowledge_project_transfer_cleanup": "src/rcp/api/project_provisioning.py",
    "accept_source_project_transfer_release": "src/rcp/api/project_provisioning.py",
    "accept_target_project_transfer_admission": "src/rcp/api/project_provisioning.py",
    "admit_target_project_transfer_request": "src/rcp/api/project_provisioning.py",
    "bind_project_transfer_archive": "src/rcp/api/project_provisioning.py",
    "create_source_project_transfer_request": "src/rcp/api/project_provisioning.py",
    "create_target_project_transfer_request": "src/rcp/api/project_provisioning.py",
    "create_incoming_transfer_provisioning_request": ("src/rcp/api/project_provisioning.py"),
    "create_project_provisioning_request": "src/rcp/api/project_provisioning.py",
    "download_source_project_transfer_archive": "src/rcp/api/project_provisioning.py",
    "create_paper": "src/rcp/api/paper.py",
    "create_project": "src/rcp/api/index.py",
    "create_team_invitation": "src/rcp/api/team.py",
    "delete_project": "src/rcp/api/index.py",
    "download_agent_artifact": "src/rcp/api/tasks.py",
    "enroll_team_member": "src/rcp/api/team.py",
    "episode_messages": "src/rcp/api/episode_routes.py",
    "episodes": "src/rcp/api/episode_routes.py",
    "exchange_team_session": "src/rcp/api/team.py",
    "experiment_episodes": "src/rcp/api/index.py",
    "get_identity": "src/rcp/api/team.py",
    "get_paper": "src/rcp/api/paper.py",
    "graph": "src/rcp/api/project_state.py",
    "graph_transition_manifest": "src/rcp/api/history.py",
    "health": "src/rcp/api/health.py",
    "server_status": "src/rcp/api/server_status.py",
    "history": "src/rcp/api/history.py",
    "history_summaries": "src/rcp/api/history.py",
    "invite_project_member": "src/rcp/api/project_state.py",
    "keep_result_view": "src/rcp/api/result_views.py",
    "keep_agent_artifact": "src/rcp/api/tasks.py",
    "leave_project": "src/rcp/api/project_state.py",
    "logout_team_session": "src/rcp/api/team.py",
    "merge_episode_branch": "src/rcp/api/episode_routes.py",
    "paper_sessions": "src/rcp/api/paper.py",
    "pause_agent_task": "src/rcp/api/tasks.py",
    "preflight_project": "src/rcp/api/index.py",
    "preview_agent_artifact": "src/rcp/api/tasks.py",
    "preview_episode_report": "src/rcp/api/episode_routes.py",
    "view_episode_report": "src/rcp/api/episode_routes.py",
    "preview_graph_sync": "src/rcp/api/sync.py",
    "preview_repository_file": "src/rcp/api/project_state.py",
    "preview_result_view": "src/rcp/api/result_views.py",
    "view_agent_artifact": "src/rcp/api/tasks.py",
    "project": "src/rcp/api/project_state.py",
    "project_provisioning_request": "src/rcp/api/project_provisioning.py",
    "project_provisioning_requests": "src/rcp/api/project_provisioning.py",
    "project_transfer_request": "src/rcp/api/project_provisioning.py",
    "project_transfer_requests": "src/rcp/api/project_provisioning.py",
    "incoming_transfer_provisioning_request": "src/rcp/api/project_provisioning.py",
    "incoming_transfer_provisioning_requests": "src/rcp/api/project_provisioning.py",
    "project_invitations_for_me": "src/rcp/api/index.py",
    "project_members": "src/rcp/api/project_state.py",
    "project_readiness": "src/rcp/api/project_state.py",
    "project_revision": "src/rcp/api/project_state.py",
    "project_watchers": "src/rcp/api/watchers.py",
    "projects": "src/rcp/api/index.py",
    "providers": "src/rcp/api/index.py",
    "read_skill_package": "src/rcp/api/index.py",
    "reauthorize_episode": "src/rcp/api/episode_routes.py",
    "register_project": "src/rcp/api/index.py",
    "release_source_project_transfer_request": "src/rcp/api/project_provisioning.py",
    "read_source_project_transfer_release_boundary": "src/rcp/api/project_provisioning.py",
    "reenter_restored_target_project_transfer": "src/rcp/api/project_provisioning.py",
    "retrieve_target_activation_proof": "src/rcp/api/project_provisioning.py",
    "verify_target_activation_proof": "src/rcp/api/project_provisioning.py",
    "remove_chat_attachment": "src/rcp/api/chats.py",
    "repair_agent_task_graph_update": "src/rcp/api/tasks.py",
    "resolve_project_provider_path": "src/rcp/api/project_state.py",
    "result_views": "src/rcp/api/result_views.py",
    "resume_agent_task": "src/rcp/api/tasks.py",
    "retry_agent_task": "src/rcp/api/tasks.py",
    "revoke_team_credential": "src/rcp/api/team.py",
    "rotate_team_credential": "src/rcp/api/team.py",
    "run_experiment": "src/rcp/api/experiments.py",
    "save_paper": "src/rcp/api/paper.py",
    "send_episode_message": "src/rcp/api/episode_routes.py",
    "sources": "src/rcp/api/project_state.py",
    "space_users": "src/rcp/api/index.py",
    "start_agent_task": "src/rcp/api/tasks.py",
    "start_episode": "src/rcp/api/episode_routes.py",
    "stop_episode": "src/rcp/api/episode_routes.py",
    "stop_experiment_loop": "src/rcp/api/experiments.py",
    "stop_experiment_watchers": "src/rcp/api/experiments.py",
    "stop_watcher": "src/rcp/api/watchers.py",
    "sync_graph": "src/rcp/api/sync.py",
    "team_invitations": "src/rcp/api/team.py",
    "update_identity": "src/rcp/api/team.py",
    "update_project_settings": "src/rcp/api/project_state.py",
    "update_team_space": "src/rcp/api/team.py",
    "upload_chat_attachment": "src/rcp/api/chats.py",
    "cancel_project_provisioning_request": "src/rcp/api/project_provisioning.py",
    "link_source_project_transfer_request": "src/rcp/api/project_provisioning.py",
}


def _walk_routes(routes: Iterable[object]) -> Iterable[object]:
    """Yield actual routes, including routes nested by ``include_router``."""

    for route in routes:
        inner = getattr(route, "original_router", None)
        if inner is not None:
            yield from _walk_routes(inner.routes)
        elif hasattr(route, "methods"):
            yield route


def _route_entry(route: Any) -> RouteEntry:
    methods = tuple(sorted(route.methods))
    return methods, route.path


@pytest.fixture
def route_app(manifest, tmp_path: Path) -> FastAPI:
    return create_named_app(str(manifest.path), data_dir=tmp_path / "route-inventory")


def test_frozen_route_inventory(route_app: FastAPI) -> None:
    routes = list(_walk_routes(route_app.routes))
    entries = tuple(_route_entry(route) for route in routes)

    assert len(entries) == 118
    assert len(_FROZEN_ROUTE_INVENTORY) == 118
    # Registration order is not part of the route contract; membership is.
    assert frozenset(entries) == frozenset(_FROZEN_ROUTE_INVENTORY)

    # The count makes the application/generated split explicit. FastAPI's
    # built-in routes are ordinary Starlette Route objects, while application
    # routes are APIRoute objects (including those nested in the router).
    assert sum(isinstance(route, APIRoute) for route in routes) == 114
    assert len(routes) - sum(isinstance(route, APIRoute) for route in routes) == 4


def test_handler_module_map_is_separate_and_current(route_app: FastAPI) -> None:
    routes = list(_walk_routes(route_app.routes))
    application_routes = [route for route in routes if isinstance(route, APIRoute)]
    observed: dict[str, str] = {}
    repository_root = Path(__file__).resolve().parents[1]
    for route in application_routes:
        endpoint = route.endpoint
        source = inspect.getsourcefile(endpoint)
        assert source is not None
        observed[endpoint.__name__] = str(Path(source).resolve().relative_to(repository_root))

    assert len(observed) == 107
    assert observed == _HANDLER_MODULE_MAP
