"""Every RCP operational record, stored in one SQLite file.

`AppStore` is the whole public surface. Its behaviour is split across topic
mixins purely so no single module carries ten thousand lines; the split is an
implementation detail and nothing outside this package should import a mixin.
Record models and exceptions live in `models` and are re-exported here.
"""

from __future__ import annotations

from rcp.storage.agent_tasks import AgentTaskStoreMixin
from rcp.storage.auto_research import AutoResearchStoreMixin
from rcp.storage.auto_research_children import AutoResearchChildrenStoreMixin
from rcp.storage.base import AppStoreBase
from rcp.storage.episodes import EpisodeStoreMixin
from rcp.storage.experiments import ExperimentStoreMixin
from rcp.storage.models import *  # noqa: F401,F403
from rcp.storage.models import __all__ as _model_names
from rcp.storage.projects import ProjectStoreMixin
from rcp.storage.provisioning import ProjectProvisioningStoreMixin
from rcp.storage.restore_detachment import RestoreDetachmentStoreMixin
from rcp.storage.result_views import ResultViewStoreMixin
from rcp.storage.rows import RowMappingMixin
from rcp.storage.spaces import SpaceStoreMixin
from rcp.storage.transfer import ProjectTransferStoreMixin
from rcp.storage.watchers import WatcherStoreMixin


class AppStore(
    ProjectTransferStoreMixin,
    RestoreDetachmentStoreMixin,
    SpaceStoreMixin,
    ProjectProvisioningStoreMixin,
    ProjectStoreMixin,
    ResultViewStoreMixin,
    EpisodeStoreMixin,
    AutoResearchStoreMixin,
    AutoResearchChildrenStoreMixin,
    ExperimentStoreMixin,
    WatcherStoreMixin,
    AgentTaskStoreMixin,
    RowMappingMixin,
    AppStoreBase,
):
    """One SQLite file holding every operational record RCP keeps.

    The mixins do not override each other, so the inheritance order carries no
    precedence meaning; it only groups the methods by topic.
    """


__all__ = [*_model_names, "AppStore"]
