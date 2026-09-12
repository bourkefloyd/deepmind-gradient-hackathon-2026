"""Crowd seats: a pack of casual, human-ish reflex bots the host adds "for fun" before a round.

Same honest machinery as the reflex bots (solver-fed ReflexPolicy through the Hand) with a
random first name + emoji avatar, a per-seat pace sampled from WH_CROWD_THINK (default 4-14 s
between words), ~10% wrong taps / hesitation, and a short-word bias, so five of them look like a
table of casual humans rather than five copies of one bot. They are still `kind: "ai"` (🤖 badge).
"""
from __future__ import annotations

import os
import random

from ..hand import HandProfile
from .fake import ReflexPolicy

NAMES = ["Maya", "Leo", "Priya", "Sam", "Noor", "Jules", "Kai", "Ava", "Omar", "Zoe", "Theo", "Ines",
         "Ravi", "Luca", "Mei", "Tariq", "Nina", "Elio", "Sofia", "Dev", "Hana", "Milo", "Anya", "Femi",
         "Rosa", "Yuki", "Bea", "Idris", "Cleo", "Arjun", "Lena", "Otto", "Sana", "Nico", "Wren", "Diego"]
EMOJI = ["🦊", "🐼", "🐸", "🦄", "🐙", "🦉", "🐯", "🦋", "🐨", "🦩", "🐢", "🦔", "🐬", "🦜", "🐝", "🦁",
         "🐧", "🦖", "🐳", "🦒", "🐻", "🦭", "🐰", "🦚"]


def _pair(env: str, default: str) -> tuple[float, float]:
    v = [float(x) for x in os.environ.get(env, default).split(",")]
    return (v[0], v[-1])


THINK = _pair("WH_CROWD_THINK", "4,14")            # each seat's think range is sampled inside this
P_WRONG = float(os.environ.get("WH_CROWD_WRONG", 0.1))


def pick_name(rng: random.Random, taken: set[str]) -> str:
    for _ in range(60):
        n = f"{rng.choice(NAMES)} {rng.choice(EMOJI)}"
        if n not in taken:
            return n
    return f"{rng.choice(NAMES)} {rng.choice(EMOJI)} {rng.randrange(10, 99)}"


def make(solver, rng: random.Random):
    lo = rng.uniform(THINK[0], THINK[0] + 0.4 * (THINK[1] - THINK[0]))
    hi = rng.uniform(min(lo + 2.0, THINK[1]), THINK[1])
    policy = ReflexPolicy(solver.rank, rng, length_bias=rng.uniform(-1.0, -0.5), rare_penalty=0.01,
                          temperature=rng.uniform(1.0, 1.5))
    profile = HandProfile(hz=rng.uniform(5, 8), lag=(0.25, 0.5), think=(lo, hi),
                          p_wrong=P_WRONG * rng.uniform(0.7, 1.3), p_hesitate=rng.uniform(0.1, 0.2))
    return policy, profile
