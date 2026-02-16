"""Tests for BlackboardBackend protocol and InMemoryBackend."""

from uac.core.blackboard.backend import InMemoryBackend
from uac.core.blackboard.blackboard import Blackboard
from uac.core.blackboard.models import StateDelta


class TestInMemoryBackend:
    def setup_method(self) -> None:
        self.backend = InMemoryBackend()

    async def test_load_missing_returns_none(self) -> None:
        result = await self.backend.load("nonexistent")
        assert result is None

    async def test_save_and_load(self) -> None:
        board = Blackboard(belief_state="saved")
        await self.backend.save("board-1", board)
        loaded = await self.backend.load("board-1")
        assert loaded is not None
        assert loaded.belief_state == "saved"

    async def test_load_returns_independent_copy(self) -> None:
        board = Blackboard(belief_state="original")
        await self.backend.save("b", board)
        copy1 = await self.backend.load("b")
        copy2 = await self.backend.load("b")
        assert copy1 is not copy2
        assert copy1 is not None
        copy1.belief_state = "mutated"
        assert copy2 is not None
        assert copy2.belief_state == "original"

    async def test_save_overwrites(self) -> None:
        board1 = Blackboard(belief_state="v1")
        board2 = Blackboard(belief_state="v2")
        await self.backend.save("b", board1)
        await self.backend.save("b", board2)
        loaded = await self.backend.load("b")
        assert loaded is not None
        assert loaded.belief_state == "v2"

    async def test_exists(self) -> None:
        assert await self.backend.exists("b") is False
        await self.backend.save("b", Blackboard())
        assert await self.backend.exists("b") is True

    async def test_delete(self) -> None:
        await self.backend.save("b", Blackboard())
        assert await self.backend.exists("b") is True
        await self.backend.delete("b")
        assert await self.backend.exists("b") is False

    async def test_delete_missing_is_noop(self) -> None:
        await self.backend.delete("nonexistent")

    async def test_full_workflow(self) -> None:
        board = Blackboard()
        board.apply(StateDelta(belief_state="working", artifacts={"step": 1}))
        await self.backend.save("session-1", board)

        loaded = await self.backend.load("session-1")
        assert loaded is not None
        assert loaded.belief_state == "working"
        assert loaded.artifacts["step"] == 1

        loaded.apply(StateDelta(belief_state="done"))
        await self.backend.save("session-1", loaded)

        final = await self.backend.load("session-1")
        assert final is not None
        assert final.belief_state == "done"

    async def test_multiple_boards(self) -> None:
        await self.backend.save("a", Blackboard(belief_state="alpha"))
        await self.backend.save("b", Blackboard(belief_state="beta"))
        a = await self.backend.load("a")
        b = await self.backend.load("b")
        assert a is not None and a.belief_state == "alpha"
        assert b is not None and b.belief_state == "beta"
