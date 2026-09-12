#!/usr/bin/env python3
"""Generate the 15 themed levels in data/levels.json.

    python scripts/make_levels.py                 # regenerate all themes -> data/levels.json
    python scripts/make_levels.py --only Kitchen  # regenerate one theme in place
    python scripts/make_levels.py --seed 7 --budget 60

Per theme: pick 3-5 seed words from THEMES, trace them onto an empty 4x4 grid (words may share
tiles), fill the free tiles from a vowel-heavy bag (no Q/X/Z/J/V), then hill-climb the free tiles
to maximise the density of *common* 3-5 letter words (top 10k of data/common-30k.txt) using the
game's own solver. The best subset+fill per theme wins. Seeds stay fixed, so the theme words are
always findable on the final board.

Bourke curates by editing data/levels.json (theme titles, order, or a whole board) and redeploying;
`python scripts/levels_report.py` re-checks the numbers.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from wordhunt.levels import COMMON_RANK, board_stats  # noqa: E402
from wordhunt.solver import DATA_DIR, NEIGHBORS, Solver  # noqa: E402

THEMES: list[tuple[str, list[str]]] = [
    ("Kitchen", ["pan", "pot", "oven", "stove", "spoon", "fork", "knife", "plate", "bowl", "cup", "mug", "dish", "bake", "cook", "stir", "salt", "sink", "boil", "egg", "meal"]),
    ("Animals", ["cat", "dog", "cow", "pig", "bear", "lion", "deer", "goat", "bird", "fish", "hen", "rat", "bat", "owl", "ant", "bee", "mouse", "horse", "tiger", "wolf", "frog", "duck", "seal"]),
    ("Space", ["sun", "moon", "star", "mars", "orbit", "comet", "rocket", "alien", "planet", "earth", "nova", "ship", "dust", "ring", "sky", "light"]),
    ("Beach", ["sand", "wave", "sun", "tide", "shell", "surf", "tan", "sea", "swim", "crab", "boat", "salt", "towel", "shore", "pier", "heat", "dune", "shade"]),
    ("Music", ["song", "note", "tune", "beat", "band", "drum", "bass", "tone", "sing", "rock", "solo", "harp", "horn", "piano", "choir", "dance", "chord", "tempo", "hymn"]),
    ("Sports", ["ball", "goal", "team", "race", "bat", "net", "run", "swim", "golf", "kick", "score", "coach", "win", "game", "ski", "dive", "pass", "bike", "shot", "medal"]),
    ("Garden", ["seed", "rose", "soil", "weed", "tree", "leaf", "root", "rake", "grow", "plant", "bloom", "herb", "vine", "dirt", "bush", "pot", "hose", "lawn", "stem", "bud", "bean"]),
    ("Weather", ["rain", "snow", "wind", "sun", "hail", "fog", "mist", "cloud", "storm", "ice", "heat", "cold", "warm", "dew", "gust", "sleet", "sunny", "breeze", "damp"]),
    ("Colors", ["red", "blue", "tan", "gold", "pink", "gray", "teal", "lime", "rose", "navy", "olive", "green", "rust", "ruby", "black", "white", "brown", "amber", "coral", "plum"]),
    ("Body", ["arm", "leg", "ear", "eye", "toe", "nose", "hand", "foot", "lip", "hip", "knee", "chin", "neck", "bone", "skin", "hair", "heart", "rib", "shin", "palm", "brain", "heel"]),
    ("School", ["desk", "book", "test", "read", "math", "pen", "class", "note", "grade", "learn", "page", "art", "gym", "bell", "chalk", "essay", "lesson", "study", "teach", "pencil"]),
    ("City", ["road", "bus", "park", "mall", "cafe", "shop", "train", "street", "bank", "tower", "metro", "lane", "bar", "bridge", "hotel", "cab", "store", "siren", "crowd", "light"]),
    ("Farm", ["barn", "hay", "cow", "pig", "hen", "goat", "corn", "seed", "crop", "milk", "egg", "mule", "silo", "plow", "wheat", "horse", "lamb", "field", "sheep", "farmer", "gate"]),
    ("Ocean", ["wave", "tide", "fish", "reef", "deep", "salt", "whale", "shark", "coral", "kelp", "crab", "seal", "ship", "sail", "blue", "foam", "eel", "ray", "cod", "shore", "sea"]),
    ("Party", ["cake", "gift", "hat", "game", "dance", "song", "candle", "punch", "toast", "host", "fun", "wish", "band", "cheer", "guest", "drink", "treat", "card", "wine", "music"]),
]

# Filler bag: vowel-heavy, common consonants, no Q/X/Z/J/V.
FILL_WEIGHTS = {
    "a": 10, "e": 12, "i": 8, "o": 8, "u": 3,
    "r": 7, "s": 7, "t": 7, "n": 6, "l": 5, "d": 4, "m": 3, "c": 3, "h": 3, "p": 3,
    "g": 2, "b": 2, "y": 2, "k": 1, "f": 1, "w": 1,
}
_FILL = list(FILL_WEIGHTS)
_FILL_W = [FILL_WEIGHTS[c] for c in _FILL]


def place_words(words: list[str], rng: random.Random, tries: int = 60) -> list[str | None] | None:
    """Trace every word onto a grid (None = free tile); words may share matching tiles."""
    for _ in range(tries):
        grid: list[str | None] = [None] * 16
        ok = True
        for w in sorted(words, key=len, reverse=True):
            if not _trace(grid, w, rng):
                ok = False
                break
        if ok:
            return grid
    return None


def _trace(grid: list[str | None], word: str, rng: random.Random) -> bool:
    starts = [i for i in range(16) if grid[i] in (None, word[0])]
    rng.shuffle(starts)

    def dfs(i: int, k: int, path: list[int]) -> bool:
        path.append(i)
        if k == len(word) - 1:
            return True
        nxt = [j for j in NEIGHBORS[i] if j not in path and grid[j] in (None, word[k + 1])]
        rng.shuffle(nxt)
        for j in nxt:
            if dfs(j, k + 1, path):
                return True
        path.pop()
        return False

    for s in starts:
        path: list[int] = []
        if dfs(s, 0, path):
            for k, i in enumerate(path):
                grid[i] = word[k]
            return True
    return False


def objective(board: str, solver: Solver) -> tuple[float, dict]:
    words = solver.solve(board)
    c3 = c4 = c5 = 0
    for w in words:
        if solver.rank.get(w, COMMON_RANK) < COMMON_RANK:
            if len(w) == 3:
                c3 += 1
            elif len(w) == 4:
                c4 += 1
            else:
                c5 += 1
    # Casual humans live on 3-4 letter words; 5+ common words are the "nice one" moments.
    score = c3 + 1.5 * c4 + 3.0 * c5 + 0.02 * len(words)
    return score, {"c3": c3, "c4": c4, "c5": c5, "n": len(words)}


def optimise(grid: list[str | None], solver: Solver, rng: random.Random, iters: int) -> tuple[str, float]:
    free = [i for i in range(16) if grid[i] is None]
    tiles = [g if g is not None else rng.choices(_FILL, weights=_FILL_W)[0] for g in grid]
    best_s, _ = objective("".join(tiles), solver)
    if not free:
        return "".join(tiles), best_s
    for _ in range(iters):
        i = rng.choice(free)
        old = tiles[i]
        tiles[i] = rng.choices(_FILL, weights=_FILL_W)[0]
        if tiles[i] == old:
            continue
        s, _ = objective("".join(tiles), solver)
        if s >= best_s:
            best_s = s
        else:
            tiles[i] = old
    return "".join(tiles), best_s


def build_level(theme: str, candidates: list[str], solver: Solver, rng: random.Random,
                budget_s: float, iters: int = 300) -> dict:
    cands = [w for w in candidates if solver.is_word(w) and 3 <= len(w) <= 6]
    best: tuple[float, str, list[str]] | None = None
    t0 = time.time()
    attempts = 0
    while time.time() - t0 < budget_s or attempts < 8:
        attempts += 1
        k = rng.choice([3, 4, 4, 4, 5])
        seeds = rng.sample(cands, min(k, len(cands)))
        grid = place_words(seeds, rng)
        if grid is None:
            continue
        free = sum(1 for g in grid if g is None)
        if free < 3:          # no room to tune the filler
            continue
        board, s = optimise(grid, solver, rng, iters)
        s += 3.0 * (len(seeds) - 3)   # more theme on the board is worth a little density
        if best is None or s > best[0]:
            best = (s, board, seeds)
    assert best is not None, theme
    _, board, seeds = best
    words = solver.solve(board)
    seeds = sorted(seeds, key=lambda w: words[w][0] if w in words else 99)
    return {
        "theme": theme,
        "board": board,
        "seed_words": seeds,
        "notes": "",
        "stats": board_stats(board, solver, words, seeds),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "levels.json"))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--budget", type=float, default=20.0, help="seconds of search per theme")
    ap.add_argument("--iters", type=int, default=300, help="hill-climb steps per seed placement")
    ap.add_argument("--only", action="append", default=[], help="regenerate only these themes (repeatable)")
    a = ap.parse_args()

    solver = Solver()
    existing: dict[str, dict] = {}
    if a.only and os.path.exists(a.out):
        with open(a.out) as f:
            existing = {d["theme"]: d for d in json.load(f)}

    out = []
    for n, (theme, cands) in enumerate(THEMES, 1):
        if a.only and theme not in a.only and theme in existing:
            lvl = existing[theme]
        else:
            rng = random.Random(a.seed * 1000 + n)
            lvl = build_level(theme, cands, solver, rng, a.budget, a.iters)
        lvl = {"n": n, **{k: lvl[k] for k in ("theme", "board", "seed_words", "notes", "stats")}}
        out.append(lvl)
        st = lvl["stats"]
        print(f"{n:2d} {theme:<8} {lvl['board']}  words={st['n_words']:3d} common={st['common_words']:3d} "
              f"(3:{st['common_3']} 4:{st['common_4']} 5+:{st['common_5plus']}) max={st['max_score']:6d} "
              f"human~{st['human_target']}  seeds={','.join(lvl['seed_words'])}", flush=True)

    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"wrote {len(out)} levels -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
