"""Hand controller: every AI seat acts through this.

Cadence cap: one tile per tick (~10 Hz), one word at a time. Human-ish noise: 150-300 ms reaction
lag before a word starts, an occasional wrong neighbor followed by a backtrack, and a think pause
between words. The policy decides *what*; the hand decides *when* and adds the noise.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .seats.base import Policy
from .solver import NEIGHBORS


@dataclass
class HandProfile:
    hz: float = 10.0
    lag: tuple[float, float] = (0.15, 0.30)       # reaction lag before starting a word
    think: tuple[float, float] = (1.0, 3.0)       # pause between words
    p_wrong: float = 0.08                         # chance of tapping a wrong neighbor
    p_hesitate: float = 0.05                      # chance of skipping a tick mid-word


@dataclass
class HandEvent:
    kind: str                     # "cursor" | "submit" | "abort"
    path: list[int] = field(default_factory=list)
    tile: int | None = None


class Hand:
    def __init__(self, policy: Policy, profile: HandProfile | None = None, rng: random.Random | None = None):
        self.policy = policy
        self.profile = profile or HandProfile()
        self.rng = rng or random.Random()
        self.board = ""
        self.path: list[int] = []
        self.found: set[str] = set()
        self.busy_until = 0.0
        self._backtrack = False
        self._ticks = 0

    @property
    def tick_period(self) -> float:
        return 1.0 / self.profile.hz

    async def start_round(self, board: str, words: dict[str, list[int]], now: float) -> None:
        self.board = board
        self.path = []
        self.found = set()
        self._backtrack = False
        self.busy_until = now + self.rng.uniform(0.5, 1.5)
        await self.policy.start_round(board, words)

    def on_result(self, word: str, ok: bool, reason: str) -> None:
        if ok:
            self.found.add(word)
        self.policy.on_result(word, ok, reason)

    def tick(self, now: float) -> list[HandEvent]:
        if now < self.busy_until:
            return []
        p = self.profile
        self.busy_until = now + self.tick_period

        if self._backtrack:
            self._backtrack = False
            if self.path:
                self.path.pop()
            return [HandEvent("cursor", list(self.path), self.path[-1] if self.path else None)]

        if self.path and self.rng.random() < p.p_hesitate:
            return []

        action = self.policy.next_action(self.board, list(self.path), self.found)
        if action is None:
            self.busy_until = now + self.rng.uniform(*p.lag)
            return []

        if action.kind == "extend":
            if not self.path:
                self.busy_until = now + self.rng.uniform(*p.lag)
            target = action.tile
            if self.path and self.rng.random() < p.p_wrong:
                wrong = [j for j in NEIGHBORS[self.path[-1]] if j not in self.path and j != target]
                if wrong:
                    target = self.rng.choice(wrong)
                    self._backtrack = True
            if target is None or target in self.path or (self.path and target not in NEIGHBORS[self.path[-1]]):
                self.path = []
                return [HandEvent("abort", [])]
            self.path.append(target)
            return [HandEvent("cursor", list(self.path), target)]

        if action.kind == "submit":
            path = list(self.path)
            self.path = []
            self.busy_until = now + self.rng.uniform(*p.think)
            return [HandEvent("submit", path, None), HandEvent("cursor", [], None)]

        self.path = []
        self.busy_until = now + self.rng.uniform(*p.lag)
        return [HandEvent("abort", []), HandEvent("cursor", [], None)]
