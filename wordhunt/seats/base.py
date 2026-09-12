"""Seat policy abstraction.

Every AI seat is a tile-level `Policy`: given the board and the hand's current path it returns
one action (`extend(tile)`, `submit`, `abort`) or `None` to keep thinking. The nano policy will
implement this directly. Word-level players (reflex bots, Gemma) subclass `WordQueuePolicy`,
which turns whole words into tile-by-tile actions so the hand controller sees the same interface.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

Kind = Literal["extend", "submit", "abort"]


@dataclass(frozen=True)
class Action:
    kind: Kind
    tile: int | None = None

    @staticmethod
    def extend(tile: int) -> "Action":
        return Action("extend", tile)


SUBMIT = Action("submit")
ABORT = Action("abort")


class Policy:
    """Tile-level decision maker. Stateless w.r.t. timing; the hand owns cadence and noise."""

    name = "policy"

    async def start_round(self, board: str, words: dict[str, list[int]]) -> None:
        """Called when a round starts. `words` is the solver output (oracle); a fair policy
        ignores it. Reflex bots and evals use it."""

    def next_action(self, board: str, path: list[int], found: set[str]) -> Action | None:
        raise NotImplementedError

    def on_result(self, word: str, ok: bool, reason: str) -> None:
        """Feedback after a submit is judged by the room."""

    async def end_round(self) -> None:
        pass


class WordQueuePolicy(Policy):
    """Adapter: subclass supplies whole-word paths via `next_word`; this emits tile actions."""

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()
        self._target: list[int] | None = None
        self._board = ""

    async def start_round(self, board: str, words: dict[str, list[int]]) -> None:
        self._board = board
        self._target = None

    def next_word(self, board: str, found: set[str]) -> list[int] | None:
        raise NotImplementedError

    def next_action(self, board: str, path: list[int], found: set[str]) -> Action | None:
        if self._target is None:
            self._target = self.next_word(board, found)
            if self._target is None:
                return None
        tgt = self._target
        # Path drifted off target (hand made a wrong move that it did not backtrack): abort.
        if path != tgt[: len(path)]:
            self._target = None
            return ABORT
        if len(path) == len(tgt):
            self._target = None
            return SUBMIT
        return Action.extend(tgt[len(path)])


class RandomSwiperPolicy(Policy):
    """Baseline for the nano gate: random walk over neighbors, submits at random length."""

    name = "random-swiper"

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def next_action(self, board: str, path: list[int], found: set[str]) -> Action | None:
        from ..solver import NEIGHBORS

        if not path:
            return Action.extend(self.rng.randrange(16))
        if len(path) >= 3 and self.rng.random() < 0.35:
            return SUBMIT
        opts = [j for j in NEIGHBORS[path[-1]] if j not in path]
        if not opts or len(path) >= 8:
            return SUBMIT if len(path) >= 3 else ABORT
        return Action.extend(self.rng.choice(opts))
