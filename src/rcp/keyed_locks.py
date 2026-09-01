from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Protocol

from rcp.core.models import Experiment


class _History(Protocol):
    def state(self): ...


class _ProjectService(Protocol):
    history: _History


class KeyedLocks:
    """A lock per key, created on first use."""

    def __init__(
        self,
        lock_factory: Callable[[], AbstractContextManager[object]] = threading.Lock,
    ) -> None:
        self._guard = threading.Lock()
        self._locks: dict[str, AbstractContextManager[object]] = {}
        self._lock_factory = lock_factory

    def __call__(self, key: str) -> AbstractContextManager[object]:
        with self._guard:
            return self._locks.setdefault(key, self._lock_factory())


class ExperimentAdmission:
    """Serialize bounded Experiment admission without owning request policy."""

    def __init__(
        self,
        locks: KeyedLocks,
        control_node_id: Callable[[object], str | None],
    ) -> None:
        self._locks = locks
        self._control_node_id = control_node_id

    @contextmanager
    def __call__(
        self,
        project_id: str,
        service: _ProjectService,
        request: object,
    ) -> Iterator[None]:
        with self._locks(project_id):
            self.require_current(service, request)
            yield

    def require_current(self, service: _ProjectService, request: object) -> None:
        """Validate one continuation when its caller already holds the project lock."""

        control_node_id = self._control_node_id(request)
        if control_node_id is None:
            return
        if not isinstance(service.history.state().nodes.get(control_node_id), Experiment):
            raise ValueError(
                f"Experiment {control_node_id} no longer exists; it cannot be continued."
            )
