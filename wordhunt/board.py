"""4x4 board generator with a dead-board filter, plus a packed-board picker."""
from __future__ import annotations

import json
import os
import random

from .scoring import score_word
from .solver import DATA_DIR, Solver

# Letter weights loosely following English frequency, nudged toward vowels; Q kept rare.
LETTER_WEIGHTS = {
    "a": 9, "b": 2, "c": 3, "d": 4, "e": 12, "f": 2, "g": 3, "h": 3, "i": 8, "j": 1,
    "k": 1, "l": 5, "m": 3, "n": 6, "o": 8, "p": 2, "q": 0.3, "r": 6, "s": 6, "t": 7,
    "u": 4, "v": 1, "w": 2, "x": 0.5, "y": 2, "z": 0.5,
}
_LETTERS = list(LETTER_WEIGHTS)
_WEIGHTS = [LETTER_WEIGHTS[c] for c in _LETTERS]


def random_board(rng: random.Random) -> str:
    return "".join(rng.choices(_LETTERS, weights=_WEIGHTS, k=16))


def board_stats(words: dict[str, list[int]]) -> dict:
    longest = max((len(w) for w in words), default=0)
    return {
        "n_words": len(words),
        "longest": longest,
        "total_score": sum(score_word(w) for w in words),
    }


def is_live(words: dict[str, list[int]], min_words: int = 60, min_longest: int = 6) -> bool:
    s = board_stats(words)
    return s["n_words"] >= min_words and s["longest"] >= min_longest


def generate_board(solver: Solver, rng: random.Random | None = None, tries: int = 200) -> tuple[str, dict[str, list[int]]]:
    """First random board passing the dead-board filter (>= 1 word of 6+ letters)."""
    rng = rng or random.Random()
    best = None
    for _ in range(tries):
        b = random_board(rng)
        words = solver.solve(b)
        if is_live(words):
            return b, words
        if best is None or len(words) > len(best[1]):
            best = (b, words)
    return best  # type: ignore[return-value]


def packed_board(solver: Solver, rng: random.Random | None = None, candidates: int = 40) -> tuple[str, dict[str, list[int]]]:
    """Best of N live boards by total score; used for stage boards."""
    rng = rng or random.Random()
    best = None
    for _ in range(candidates):
        b, words = generate_board(solver, rng)
        s = board_stats(words)
        if best is None or s["total_score"] > best[2]:
            best = (b, words, s["total_score"])
    return best[0], best[1]  # type: ignore[index]


def load_packed_boards() -> list[str]:
    p = os.path.join(DATA_DIR, "boards.json")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [x["board"] if isinstance(x, dict) else x for x in json.load(f)]


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    solver = Solver()
    rng = random.Random(2026)
    out = []
    for _ in range(n):
        b, words = packed_board(solver, rng)
        s = board_stats(words)
        out.append({"board": b, **s, "longest_words": sorted(words, key=len, reverse=True)[:5]})
        print(b, s)
    with open(os.path.join(DATA_DIR, "boards.json"), "w") as f:
        json.dump(out, f, indent=1)
