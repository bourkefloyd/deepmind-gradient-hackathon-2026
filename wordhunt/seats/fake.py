"""Heuristic "reflex bot" seats: pull words from the solver output with a frequency-ish bias
(short common words first) and feed them through the hand. Honest fallback for the nano seat.
"""
from __future__ import annotations

import math
import random

from ..hand import HandProfile
from .base import WordQueuePolicy


class ReflexPolicy(WordQueuePolicy):
    name = "reflex"

    def __init__(self, rank: dict[str, int], rng: random.Random | None = None,
                 length_bias: float = -0.6, rare_penalty: float = 0.02, temperature: float = 1.0):
        super().__init__(rng)
        self.rank = rank
        self.length_bias = length_bias        # <0 favors short words, >0 favors long
        self.rare_penalty = rare_penalty      # weight for words outside the common list
        self.temperature = temperature
        self._queue: list[list[int]] = []

    def _weight(self, w: str) -> float:
        r = self.rank.get(w)
        base = 1.0 / (1.0 + r / 2000.0) if r is not None else self.rare_penalty
        return base * math.exp(self.length_bias * (len(w) - 4))

    async def start_round(self, board: str, words: dict[str, list[int]]) -> None:
        await super().start_round(board, words)
        # Weighted shuffle without replacement (Efraimidis-Spirakis).
        keyed = []
        for w, path in words.items():
            wt = max(self._weight(w), 1e-9) ** (1.0 / self.temperature)
            keyed.append((-(self.rng.random() ** (1.0 / wt)), path))
        keyed.sort()
        self._queue = [p for _, p in keyed]

    def next_word(self, board: str, found: set[str]) -> list[int] | None:
        while self._queue:
            path = self._queue.pop(0)
            word = "".join(board[t] for t in path)
            if word not in found:
                return path
        return None


# Two fake seats with different tempos. Names are honest: these are heuristics, not the nano.
FAKE_SEATS = [
    {
        "seat_id": "ai:reflex-a",
        "name": "Reflex-A",
        "policy": lambda rank, rng: ReflexPolicy(rank, rng, length_bias=-0.7, temperature=1.0),
        "profile": HandProfile(hz=10, lag=(0.15, 0.25), think=(1.2, 2.6), p_wrong=0.06, p_hesitate=0.04),
    },
    {
        "seat_id": "ai:reflex-b",
        "name": "Reflex-B",
        "policy": lambda rank, rng: ReflexPolicy(rank, rng, length_bias=0.15, rare_penalty=0.08, temperature=1.3),
        "profile": HandProfile(hz=8, lag=(0.2, 0.3), think=(2.5, 4.5), p_wrong=0.12, p_hesitate=0.08),
    },
]
