"""Minimal Word Hunt solver: trie DFS over a 4x4 board, 3-8 letter words, 8-way adjacency, no tile reuse.

TODO(consolidate): `wordhunt/solver.py` did not exist on any branch when nano/ was ported (2026-09-12 12:10 PT).
When the game-server solver lands, make one of them import the other; the trie/DFS here is intentionally plain.

Board = 16 lowercase letters, row-major (index i -> row i // 4, col i % 4).
"""

from __future__ import annotations

import os
from typing import Iterable, Iterator

MIN_LEN = 3
MAX_LEN = 8
SCORE = {3: 100, 4: 400, 5: 800, 6: 1400, 7: 1800}

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "data")


def score_word(word: str) -> int:
    n = len(word)
    if n < MIN_LEN:
        return 0
    if n >= 7:
        return 1800 + 400 * (n - 7)
    return SCORE[n]


def neighbors() -> list[list[int]]:
    nb: list[list[int]] = []
    for i in range(16):
        r, c = divmod(i, 4)
        cur = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < 4 and 0 <= cc < 4:
                    cur.append(rr * 4 + cc)
        nb.append(cur)
    return nb


NEIGHBORS = neighbors()


class Trie:
    __slots__ = ("children", "word")

    def __init__(self) -> None:
        self.children: dict[str, Trie] = {}
        self.word: str | None = None

    def insert(self, w: str) -> None:
        node = self
        for ch in w:
            nxt = node.children.get(ch)
            if nxt is None:
                nxt = Trie()
                node.children[ch] = nxt
            node = nxt
        node.word = w


def load_words(path: str | None = None, min_len: int = MIN_LEN, max_len: int = MAX_LEN) -> list[str]:
    path = path or os.path.join(DATA_DIR, "enable1.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing; see data/README.md for the download command")
    out = []
    with open(path) as f:
        for line in f:
            w = line.strip().lower()
            if min_len <= len(w) <= max_len and w.isalpha():
                out.append(w)
    return out


def load_common_ranks(path: str | None = None, valid: Iterable[str] | None = None) -> dict[str, int]:
    """word -> rank (0 = most common) from the ~30k list, intersected with `valid` if given."""
    path = path or os.path.join(DATA_DIR, "common-30k.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing; see data/README.md for the download command")
    vs = set(valid) if valid is not None else None
    ranks: dict[str, int] = {}
    with open(path) as f:
        for i, line in enumerate(f):
            w = line.strip().lower()
            if not w or (vs is not None and w not in vs) or w in ranks:
                continue
            ranks[w] = i
    return ranks


def build_trie(words: Iterable[str]) -> Trie:
    t = Trie()
    for w in words:
        t.insert(w)
    return t


class Solver:
    def __init__(self, words: Iterable[str] | None = None):
        self.words = list(words) if words is not None else load_words()
        self.word_set = frozenset(self.words)
        self.trie = build_trie(self.words)

    def is_word(self, w: str) -> bool:
        return w in self.word_set

    def paths(self, board: str) -> Iterator[tuple[str, tuple[int, ...]]]:
        """Yield every (word, path) on the board; a word can appear with several paths."""
        board = board.lower()
        assert len(board) == 16
        stack: list[tuple[Trie, tuple[int, ...], int]] = []
        for i in range(16):
            nxt = self.trie.children.get(board[i])
            if nxt is not None:
                stack.append((nxt, (i,), 1 << i))
        while stack:
            node, path, used = stack.pop()
            if node.word is not None:
                yield node.word, path
            if len(path) >= MAX_LEN:
                continue
            for j in NEIGHBORS[path[-1]]:
                if used & (1 << j):
                    continue
                nxt = node.children.get(board[j])
                if nxt is not None:
                    stack.append((nxt, path + (j,), used | (1 << j)))

    def solve(self, board: str) -> dict[str, list[tuple[int, ...]]]:
        out: dict[str, list[tuple[int, ...]]] = {}
        for w, p in self.paths(board):
            out.setdefault(w, []).append(p)
        return out

    def words_on(self, board: str) -> set[str]:
        return {w for w, _ in self.paths(board)}


def is_valid_path(path: Iterable[int]) -> bool:
    p = list(path)
    if len(set(p)) != len(p):
        return False
    return all(b in NEIGHBORS[a] for a, b in zip(p, p[1:]))


def path_word(board: str, path: Iterable[int]) -> str:
    return "".join(board[i] for i in path)


if __name__ == "__main__":
    import sys
    import time

    s = Solver()
    b = sys.argv[1] if len(sys.argv) > 1 else "ratehunslentgoid"
    t0 = time.time()
    found = s.solve(b)
    dt = time.time() - t0
    top = sorted(found, key=lambda w: (-len(w), w))[:15]
    print(f"{len(found)} words in {dt*1000:.1f} ms; longest: {top}")
    print("score if all found:", sum(score_word(w) for w in found))
