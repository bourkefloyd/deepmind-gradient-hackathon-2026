"""Pre-canned themed levels: load `data/levels.json`, hand them out in order, score boards for humans.

A level is {n, theme, board, seed_words, notes, stats}. Rooms play level 1 on round 1, the next
level on each rematch, and wrap after the last one. `WH_LEVELS` points at the JSON file (default
data/levels.json; set it empty or to "0" to fall back to packed/random boards). `WH_LEVEL_START=n`
starts the sequence at level n.

The "common" set used for the stats is the top COMMON_RANK words of data/common-30k.txt that are
also valid (enable1). Casual humans mostly find 3-4 letter words from that set.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache

from .scoring import score_word
from .solver import DATA_DIR, Solver, path_for_word

COMMON_RANK = 10_000
LEVELS_PATH = os.environ.get("WH_LEVELS", os.path.join(DATA_DIR, "levels.json"))
LEVEL_START = int(os.environ.get("WH_LEVEL_START", "1") or 1)


@dataclass
class Level:
    n: int
    theme: str
    board: str
    seed_words: list[str] = field(default_factory=list)
    notes: str = ""
    stats: dict = field(default_factory=dict)

    def public(self) -> dict:
        return {"n": self.n, "theme": self.theme}


def board_stats(board: str, solver: Solver, words: dict[str, list[int]] | None = None,
                seed_words: list[str] | None = None) -> dict:
    """Human-facing board stats: word counts by commonness/length, max score, casual estimate."""
    words = words if words is not None else solver.solve(board)
    common = [w for w in words if solver.rank.get(w, COMMON_RANK) < COMMON_RANK]
    c3 = sum(1 for w in common if len(w) == 3)
    c4 = sum(1 for w in common if len(w) == 4)
    c5 = sum(1 for w in common if len(w) >= 5)
    longest = max(words, key=lambda w: (len(w), -solver.rank.get(w, COMMON_RANK)), default="")
    out = {
        "n_words": len(words),
        "common_words": len(common),
        "common_3": c3,
        "common_4": c4,
        "common_5plus": c5,
        "max_score": sum(score_word(w) for w in words),
        "longest": longest,
        "human_target": human_target(c3, c4, c5),
    }
    if seed_words is not None:
        out["seeds_present"] = [w for w in seed_words if path_for_word(board, w) is not None]
    return out


def human_target(c3: int, c4: int, c5: int) -> int:
    """Rough score a casual player reaches in a 75 s round without effort.

    Model: they spot ~30% of the common 3-4 letter words, capped at 12 finds (one every ~6 s),
    and 4-letter words come at about a third of the rate of 3-letter ones. 12 finds of mostly 3s
    with a few 4s lands in the 1,200-2,000 range, which is the demo target. Common 5+ letter
    words (`common_5plus`) are upside on top of this, not counted here.
    """
    short = c3 + c4
    if short == 0:
        return 0
    finds = min(12.0, 0.3 * short)
    share4 = 0.35 * c4 / short
    pts = finds * ((1 - share4) * 100 + share4 * 400)
    return int(round(pts / 100.0) * 100)


def load_levels(path: str | None = None) -> list[Level]:
    """Levels from JSON, in file order (the `n` field is informational). [] when disabled/missing."""
    path = LEVELS_PATH if path is None else path
    if not path or path == "0" or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    levels = []
    for i, d in enumerate(raw, 1):
        board = str(d.get("board", "")).lower()
        if len(board) != 16 or not board.isalpha():
            continue
        levels.append(Level(int(d.get("n", i)), str(d.get("theme", f"Level {i}")), board,
                            [str(w).lower() for w in d.get("seed_words", [])], str(d.get("notes", "")),
                            dict(d.get("stats", {}))))
    return levels


@lru_cache(maxsize=1)
def get_levels() -> tuple[Level, ...]:
    return tuple(load_levels())


def level_for_round(levels: list[Level] | tuple[Level, ...], round_no: int, start: int | None = None) -> Level | None:
    """Round 1 -> level `start`, then the next level per round, wrapping around."""
    if not levels:
        return None
    start = LEVEL_START if start is None else start
    return levels[(start - 1 + round_no - 1) % len(levels)]
