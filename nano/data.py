"""Word Hunt training data: random boards -> solver -> frequency-weighted path sampling -> per-prefix soft targets.

  python -m nano.data --boards 200000 --paths 6 --out data/wh_200k.npz

Generative story behind the labels (PLAN.md section 3): a player samples a word from the board's words with
probability proportional to weight(word) (common words heavy, rare enable1 words light) and traces one of its
paths tile by tile. For every prefix along the way the soft target is the posterior of that process:
  target_dist[tile] = mass of completions that continue through `tile`          (pointer over 16 tiles)
  type_dist         = [extend, submit, abort] = [mass of longer completions, mass of the prefix itself, 0]
  value             = 1 (some completion exists)
Dead prefixes (a legal move that no word continues) are mined next to the sampled ones: type = abort, value = 0,
no pointer target (soft_ce skips all-zero rows).

Output arrays (N samples):
  board  (N,16) uint8   letter index 0..25, row-major
  path   (N,8)  int8    tile indices, -1 padded
  plen   (N,)   uint8
  target (N,16) float16
  ptype  (N,3)  float16
  value  (N,)   uint8
  bid    (N,)   int32   board id (samples of one board are contiguous; split train/val on this)
"""

from __future__ import annotations

import argparse
import os
import time
from collections import Counter
from typing import Iterable

import numpy as np

from .solver import MAX_LEN, NEIGHBORS, Solver, Trie, load_common_ranks

TYPES = ("extend", "submit", "abort")
UNCOMMON_WEIGHT = 0.01


def word_weight(word: str, ranks: dict[str, int]) -> float:
    r = ranks.get(word)
    if r is None:
        return UNCOMMON_WEIGHT
    return 1.0 / (1.0 + r / 3000.0)


def letter_distribution(words: Iterable[str]) -> np.ndarray:
    c = Counter("".join(words))
    p = np.array([c.get(chr(97 + i), 0) for i in range(26)], dtype=np.float64)
    return p / p.sum()


def random_board(rng: np.random.Generator, letter_p: np.ndarray) -> str:
    return "".join(chr(97 + i) for i in rng.choice(26, size=16, p=letter_p))


PrefixInfo = dict[tuple[int, ...], tuple[float, dict[int, float]]]


def annotate(board: str, solver: Solver, ranks: dict[str, int]) -> tuple[PrefixInfo, int]:
    """Every valid prefix path on the board -> (weight of the prefix as a word, {next tile: completion mass}).
    Returns (info, n_words). Prefix paths with zero completion mass are absent (= dead)."""
    info: PrefixInfo = {}
    n_words = 0

    def dfs(node: Trie, path: tuple[int, ...], used: int) -> float:
        nonlocal n_words
        w_word = 0.0
        if node.word is not None:
            w_word = word_weight(node.word, ranks)
            n_words += 1
        kids: dict[int, float] = {}
        if len(path) < MAX_LEN:
            for j in NEIGHBORS[path[-1]]:
                if used & (1 << j):
                    continue
                nxt = node.children.get(board[j])
                if nxt is None:
                    continue
                sub = dfs(nxt, path + (j,), used | (1 << j))
                if sub > 0:
                    kids[j] = sub
        total = w_word + sum(kids.values())
        if total > 0:
            info[path] = (w_word, kids)
        return total

    starts: dict[int, float] = {}
    for i in range(16):
        nxt = solver.trie.children.get(board[i])
        if nxt is None:
            continue
        sub = dfs(nxt, (i,), 1 << i)
        if sub > 0:
            starts[i] = sub
    if starts:
        info[()] = (0.0, starts)
    return info, n_words


def sample_path(info: PrefixInfo, rng: np.random.Generator) -> tuple[int, ...]:
    """One word path drawn from the generative process (submit vs. each continuation by mass)."""
    path: tuple[int, ...] = ()
    while True:
        w_word, kids = info[path]
        opts = list(kids.items())
        masses = np.array([w_word] + [m for _, m in opts], dtype=np.float64)
        k = int(rng.choice(len(masses), p=masses / masses.sum()))
        if k == 0:
            return path
        path = path + (opts[k - 1][0],)


def legal_moves(path: tuple[int, ...]) -> list[int]:
    if not path:
        return list(range(16))
    return [j for j in NEIGHBORS[path[-1]] if j not in path]


def board_samples(board: str, solver: Solver, ranks: dict[str, int], rng: np.random.Generator, n_paths: int, dead_frac: float, min_words: int) -> list[tuple] | None:
    info, n_words = annotate(board, solver, ranks)
    if n_words < min_words or () not in info:
        return None
    prefixes: set[tuple[int, ...]] = set()
    for _ in range(n_paths):
        p = sample_path(info, rng)
        for k in range(len(p) + 1):
            prefixes.add(p[:k])
    return rows_from_info(board, info, prefixes, rng, dead_frac)


def rows_from_info(board: str, info: PrefixInfo, prefixes: Iterable[tuple[int, ...]], rng: np.random.Generator | None = None, dead_frac: float = 0.0) -> list[tuple]:
    """Emit one soft-target row per prefix in `prefixes` (all must be keys of `info`), plus mined dead rows.
    Dead mining needs the full dictionary annotation; pass dead_frac=0 for partial word sets (live learning)."""
    rows = []
    letters = np.frombuffer(board.encode(), dtype=np.uint8) - 97
    for p in sorted(prefixes, key=lambda q: (len(q), q)):
        w_word, kids = info[p]
        tgt = np.zeros(16, np.float32)
        ext = 0.0
        for j, m in kids.items():
            tgt[j] = m
            ext += m
        if ext > 0:
            tgt /= ext
        ty = np.array([ext, w_word, 0.0], np.float32)
        ty /= ty.sum()
        rows.append((letters, p, tgt, ty, 1))
        if dead_frac > 0 and rng is not None and len(p) < MAX_LEN and rng.random() < dead_frac:
            dead = [j for j in legal_moves(p) if j not in kids]
            if dead:
                j = int(rng.choice(dead))
                rows.append((letters, p + (j,), np.zeros(16, np.float32), np.array([0.0, 0.0, 1.0], np.float32), 0))
    return rows


def word_paths(board: str, word: str) -> list[tuple[int, ...]]:
    """Every tile path spelling `word` on the board (no dictionary involved)."""
    board = board.lower()
    word = word.lower()
    out: list[tuple[int, ...]] = []
    stack = [((i,), 1 << i) for i in range(16) if board[i] == word[0]]
    while stack:
        path, used = stack.pop()
        k = len(path)
        if k == len(word):
            out.append(path)
            continue
        for j in NEIGHBORS[path[-1]]:
            if not used & (1 << j) and board[j] == word[k]:
                stack.append((path + (j,), used | (1 << j)))
    return out


def info_from_words(board: str, words: Iterable[str], ranks: dict[str, int]) -> PrefixInfo:
    """Prefix annotation built from a known word set only (validated words found in a round), same posterior
    semantics as `annotate` but restricted to those words; a word with several paths splits its mass."""
    acc: dict[tuple[int, ...], list] = {}  # path -> [w_word, {kid: mass}]
    for w in set(words):
        paths = word_paths(board, w)
        if not paths:
            continue
        m = word_weight(w, ranks) / len(paths)
        for p in paths:
            for k in range(len(p)):
                node = acc.setdefault(p[:k], [0.0, {}])
                node[1][p[k]] = node[1].get(p[k], 0.0) + m
            node = acc.setdefault(p, [0.0, {}])
            node[0] += m
    return {p: (v[0], v[1]) for p, v in acc.items()}


def pack(rows: list[tuple], bid: int) -> dict[str, np.ndarray]:
    n = len(rows)
    out = {
        "board": np.stack([r[0] for r in rows]).astype(np.uint8),
        "path": np.full((n, MAX_LEN), -1, np.int8),
        "plen": np.array([len(r[1]) for r in rows], np.uint8),
        "target": np.stack([r[2] for r in rows]).astype(np.float16),
        "ptype": np.stack([r[3] for r in rows]).astype(np.float16),
        "value": np.array([r[4] for r in rows], np.uint8),
        "bid": np.full(n, bid, np.int32),
    }
    for i, r in enumerate(rows):
        if r[1]:
            out["path"][i, : len(r[1])] = r[1]
    return out


_W: dict = {}


def _init_worker() -> None:
    solver = Solver()
    _W["solver"] = solver
    _W["ranks"] = load_common_ranks(valid=solver.word_set)
    _W["letter_p"] = letter_distribution(solver.words)


def _work(args: tuple) -> dict[str, np.ndarray]:
    seed, bid0, n_boards, n_paths, dead_frac, min_words = args
    if not _W:
        _init_worker()
    rng = np.random.default_rng(seed)
    parts = []
    bid = bid0
    made = 0
    while made < n_boards:
        board = random_board(rng, _W["letter_p"])
        rows = board_samples(board, _W["solver"], _W["ranks"], rng, n_paths, dead_frac, min_words)
        if rows is None:
            continue
        parts.append(pack(rows, bid))
        bid += 1
        made += 1
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def generate(n_boards: int, n_paths: int = 6, dead_frac: float = 0.3, min_words: int = 15, seed: int = 0, workers: int = 0, chunk: int = 500) -> dict[str, np.ndarray]:
    jobs = []
    for i, start in enumerate(range(0, n_boards, chunk)):
        jobs.append((seed * 1_000_003 + i, start, min(chunk, n_boards - start), n_paths, dead_frac, min_words))
    if workers <= 1:
        parts = [_work(j) for j in jobs]
    else:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
            parts = pool.map(_work, jobs)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def load(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", type=int, default=2000)
    ap.add_argument("--paths", type=int, default=6, help="word paths sampled per board (prefixes are deduped)")
    ap.add_argument("--dead-frac", type=float, default=0.3, help="P(mine one dead continuation next to each valid prefix)")
    ap.add_argument("--min-words", type=int, default=15, help="reject boards with fewer words")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    d = generate(a.boards, a.paths, a.dead_frac, a.min_words, a.seed, a.workers)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, **d)
    n = len(d["value"])
    print(f"{a.boards} boards -> {n} samples ({n / a.boards:.1f}/board; dead {1 - d['value'].mean():.2f}; "
          f"submit mass {float(d['ptype'][:, 1].astype(np.float32).mean()):.3f}) in {time.time() - t0:.1f}s -> {a.out} "
          f"({os.path.getsize(a.out) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
