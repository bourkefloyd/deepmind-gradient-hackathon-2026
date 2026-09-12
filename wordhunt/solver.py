"""Trie DFS solver over enable1 (3-8 letters) for a 4x4 board.

A board is a 16-char lowercase string, row-major. Tile i is at (i // 4, i % 4).
`solve(board)` returns {word: path} where path is the first tile-index path found.
"""
from __future__ import annotations

import os
from functools import lru_cache

DATA_DIR = os.environ.get("WH_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
MIN_LEN, MAX_LEN = 3, 8
END = "$"

NEIGHBORS: list[list[int]] = []
for _i in range(16):
    _r, _c = divmod(_i, 4)
    _n = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = _r + dr, _c + dc
            if 0 <= rr < 4 and 0 <= cc < 4:
                _n.append(rr * 4 + cc)
    NEIGHBORS.append(_n)


def _load_words(path: str) -> list[str]:
    words = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            w = line.strip().lower()
            if MIN_LEN <= len(w) <= MAX_LEN and w.isalpha():
                words.append(w)
    return words


class Trie:
    __slots__ = ("root", "size")

    def __init__(self, words: list[str]):
        self.root: dict = {}
        self.size = 0
        for w in words:
            node = self.root
            for ch in w:
                node = node.setdefault(ch, {})
            if END not in node:
                node[END] = True
                self.size += 1


class Solver:
    def __init__(self, dict_path: str | None = None, common_path: str | None = None):
        dict_path = dict_path or os.path.join(DATA_DIR, "enable1.txt")
        common_path = common_path or os.path.join(DATA_DIR, "common-30k.txt")
        self.words = _load_words(dict_path)
        self.wordset = set(self.words)
        self.trie = Trie(self.words)
        # rank: 0 = most common. Only words that are also valid.
        self.rank: dict[str, int] = {}
        if os.path.exists(common_path):
            with open(common_path, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    w = line.strip().lower()
                    if w in self.wordset and w not in self.rank:
                        self.rank[w] = i

    def is_word(self, w: str) -> bool:
        return w in self.wordset

    def solve(self, board: str) -> dict[str, list[int]]:
        board = board.lower()
        found: dict[str, list[int]] = {}
        root = self.trie.root
        path: list[int] = []
        used = [False] * 16

        def dfs(i: int, node: dict, prefix: str):
            used[i] = True
            path.append(i)
            if END in node and len(prefix) >= MIN_LEN and prefix not in found:
                found[prefix] = list(path)
            if len(prefix) < MAX_LEN:
                for j in NEIGHBORS[i]:
                    if not used[j]:
                        nxt = node.get(board[j])
                        if nxt is not None:
                            dfs(j, nxt, prefix + board[j])
            path.pop()
            used[i] = False

        for i in range(16):
            node = root.get(board[i])
            if node is not None:
                dfs(i, node, board[i])
        return found

    def valid_path(self, board: str, path: list[int]) -> str | None:
        """Return the word if the path is a legal board path spelling a dictionary word."""
        if not (MIN_LEN <= len(path) <= MAX_LEN) or len(set(path)) != len(path):
            return None
        for a, b in zip(path, path[1:]):
            if b not in NEIGHBORS[a]:
                return None
        if any(not (0 <= t < 16) for t in path):
            return None
        w = "".join(board[t] for t in path)
        return w if w in self.wordset else None


def path_for_word(board: str, word: str, rng=None) -> list[int] | None:
    """A legal tile path spelling `word` on `board`, or None if it cannot be traced."""
    board = board.lower()
    word = word.lower()
    if not word:
        return None
    starts = [i for i in range(16) if board[i] == word[0]]
    if rng is not None:
        rng.shuffle(starts)

    def dfs(i: int, k: int, path: list[int]) -> list[int] | None:
        path.append(i)
        if k == len(word) - 1:
            return list(path)
        for j in NEIGHBORS[i]:
            if j not in path and board[j] == word[k + 1]:
                r = dfs(j, k + 1, path)
                if r:
                    return r
        path.pop()
        return None

    for s in starts:
        r = dfs(s, 0, [])
        if r:
            return r
    return None


def best_effort_path(board: str, word: str, rng) -> list[int]:
    """Trace as much of `word` as the board allows, then wander to reach 3 tiles.
    Used so an AI's hallucinated word still produces a visible (and judged-as-miss) attempt."""
    board = board.lower()
    word = word.lower()
    for n in range(len(word), 0, -1):
        p = path_for_word(board, word[:n], rng)
        if p:
            break
    else:
        p = [rng.randrange(16)]
    while len(p) < 3:
        opts = [j for j in NEIGHBORS[p[-1]] if j not in p]
        if not opts:
            break
        p.append(rng.choice(opts))
    return p


@lru_cache(maxsize=1)
def get_solver() -> Solver:
    return Solver()
