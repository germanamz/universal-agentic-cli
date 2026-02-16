"""Blackboard persistence backends.

:class:`BlackboardBackend` defines the async storage protocol.
:class:`InMemoryBackend` provides a lightweight dict-based implementation
suitable for testing and single-process deployments.
"""

from typing import Protocol

from uac.core.blackboard.blackboard import Blackboard


class BlackboardBackend(Protocol):
    """Async persistence protocol for :class:`Blackboard` instances."""

    async def load(self, board_id: str) -> Blackboard | None:
        """Load a board by ID, or return ``None`` if it does not exist."""
        ...

    async def save(self, board_id: str, board: Blackboard) -> None:
        """Persist the board under the given ID (upsert semantics)."""
        ...

    async def delete(self, board_id: str) -> None:
        """Remove a board by ID (no-op if absent)."""
        ...

    async def exists(self, board_id: str) -> bool:
        """Return ``True`` if a board with the given ID is persisted."""
        ...


class InMemoryBackend:
    """Dict-backed :class:`BlackboardBackend` implementation.

    Stores boards as serialised JSON bytes so that each :meth:`load` returns
    a fresh, independent copy (mimicking a real persistence layer).
    """

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    async def load(self, board_id: str) -> Blackboard | None:
        data = self._store.get(board_id)
        if data is None:
            return None
        return Blackboard.restore(data)

    async def save(self, board_id: str, board: Blackboard) -> None:
        self._store[board_id] = board.snapshot()

    async def delete(self, board_id: str) -> None:
        self._store.pop(board_id, None)

    async def exists(self, board_id: str) -> bool:
        return board_id in self._store
