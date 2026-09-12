"""Heuristic "reflex bot" seats: pull words from the solver output with a frequency-ish bias
(short common words first) and feed them through the hand. Honest fallback for the nano seat.
"""
from __future__ import annotations

import math
import os
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


def _pair(env: str, default: str) -> tuple[float, float]:
    v = [float(x) for x in os.environ.get(env, default).split(",")]
    return (v[0], v[-1])


def _f(env: str, default: float) -> float:
    return float(os.environ.get(env, default))


# Two fake seats with different tempos. Names are honest: these are heuristics, not the nano.
# Pace is env-tunable without a rebuild (seconds between words, hesitation/wrong-tap rates):
#   WH_REFLEX_A_THINK="5,9"  WH_REFLEX_B_THINK="7,12"  WH_REFLEX_HESITATE=0.15  WH_REFLEX_WRONG=0.1
# Defaults land each bot around 2-3k on rich boards so a casual human can beat them.
FAKE_SEATS = [
    {
        "seat_id": "ai:reflex-a",
        "name": "Reflex-A",
        "policy": lambda rank, rng: ReflexPolicy(rank, rng, length_bias=-0.7, temperature=1.0),
        "profile": HandProfile(hz=8, lag=(0.2, 0.35), think=_pair("WH_REFLEX_A_THINK", "5,9"),
                               p_wrong=_f("WH_REFLEX_WRONG", 0.1), p_hesitate=_f("WH_REFLEX_HESITATE", 0.15)),
    },
    {
        "seat_id": "ai:reflex-b",
        "name": "Reflex-B",
        "policy": lambda rank, rng: ReflexPolicy(rank, rng, length_bias=-0.4, rare_penalty=0.02, temperature=1.2),
        "profile": HandProfile(hz=7, lag=(0.25, 0.4), think=_pair("WH_REFLEX_B_THINK", "10,15"),
                               p_wrong=_f("WH_REFLEX_WRONG", 0.1) + 0.04, p_hesitate=_f("WH_REFLEX_HESITATE", 0.15)),
    },
]
