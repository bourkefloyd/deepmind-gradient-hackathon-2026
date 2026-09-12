"""Minimal 4x4 Word Hunt board generator + solver over enable1.

Board = 16 lowercase letters, row-major. Tile index i -> (row i // 4, col i % 4).
Words are 3-8 letters, 8-way adjacency, no tile reuse. Scoring follows PLAN.md
section 2: 3=100, 4=400, 5=800, 6=1400, 7=1800, +400 per letter beyond 7.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

SIZE = 4
MIN_LEN = 3
MAX_LEN = 8

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_WORDLIST = Path(os.environ.get("WORDHUNT_WORDLIST", _DATA_DIR / "enable1.txt"))

# Boggle-style letter bag weights (roughly English frequency, lighter on the
# rare tail so random boards are not dead).
_LETTER_WEIGHTS = {
    "a": 8, "b": 2, "c": 3, "d": 4, "e": 12, "f": 2, "g": 3, "h": 3, "i": 8,
    "j": 1, "k": 1, "l": 5, "m": 3, "n": 6, "o": 7, "p": 3, "q": 1, "r": 6,
    "s": 6, "t": 7, "u": 4, "v": 1, "w": 2, "x": 1, "y": 2, "z": 1,
}
_LETTERS = list(_LETTER_WEIGHTS)
_WEIGHTS = [_LETTER_WEIGHTS[c] for c in _LETTERS]


def neighbors(i: int) -> list[int]:
    r, c = divmod(i, SIZE)
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < SIZE and 0 <= cc < SIZE:
                out.append(rr * SIZE + cc)
    return out


NEIGHBORS = [neighbors(i) for i in range(SIZE * SIZE)]


def score_word(word: str) -> int:
    n = len(word)
    if n < MIN_LEN:
        return 0
    table = {3: 100, 4: 400, 5: 800, 6: 1400, 7: 1800}
    return table.get(n, 1800 + 400 * (n - 7))


@dataclass(frozen=True)
class Dictionary:
    words: frozenset[str]
    prefixes: frozenset[str]

    def __contains__(self, word: str) -> bool:
        return word in self.words

    def __len__(self) -> int:
        return len(self.words)


def load_dictionary(path: Path | str = DEFAULT_WORDLIST) -> Dictionary:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Word list not found at {path}. See data/README.md (enable1 is not committed)."
        )
    words: set[str] = set()
    prefixes: set[str] = set()
    with path.open() as fh:
        for line in fh:
            w = line.strip().lower()
            if MIN_LEN <= len(w) <= MAX_LEN and w.isalpha() and w.isascii():
                words.add(w)
                for k in range(1, len(w)):
                    prefixes.add(w[:k])
    return Dictionary(frozenset(words), frozenset(prefixes))


def grid_rows(board: str) -> list[str]:
    return [board[r * SIZE : (r + 1) * SIZE] for r in range(SIZE)]


def grid_text(board: str, upper: bool = True) -> str:
    rows = grid_rows(board.upper() if upper else board)
    return "\n".join(" ".join(row) for row in rows)


def random_board(rng: random.Random) -> str:
    return "".join(rng.choices(_LETTERS, weights=_WEIGHTS, k=SIZE * SIZE))


def solve(board: str, dictionary: Dictionary) -> dict[str, list[int]]:
    """All valid words on the board -> one path (tile indices) that spells each."""
    board = board.lower()
    found: dict[str, list[int]] = {}
    path: list[int] = []
    used = [False] * (SIZE * SIZE)

    def dfs(i: int, prefix: str) -> None:
        prefix += board[i]
        if len(prefix) > MAX_LEN:
            return
        is_word = prefix in dictionary.words
        is_prefix = prefix in dictionary.prefixes
        if not is_word and not is_prefix:
            return
        path.append(i)
        used[i] = True
        if is_word and len(prefix) >= MIN_LEN and prefix not in found:
            found[prefix] = list(path)
        if is_prefix:
            for j in NEIGHBORS[i]:
                if not used[j]:
                    dfs(j, prefix)
        used[i] = False
        path.pop()

    for start in range(SIZE * SIZE):
        dfs(start, "")
    return found


def find_path(board: str, word: str) -> list[int] | None:
    """A tile path spelling `word` on the board (adjacency + no reuse), or None."""
    board = board.lower()
    word = word.lower()
    if not (MIN_LEN <= len(word) <= MAX_LEN):
        return None
    used = [False] * (SIZE * SIZE)
    path: list[int] = []

    def dfs(i: int, k: int) -> bool:
        if board[i] != word[k]:
            return False
        path.append(i)
        used[i] = True
        if k == len(word) - 1:
            return True
        for j in NEIGHBORS[i]:
            if not used[j] and dfs(j, k + 1):
                return True
        used[i] = False
        path.pop()
        return False

    for start in range(SIZE * SIZE):
        if dfs(start, 0):
            return path
    return None


def validate(board: str, word: str, dictionary: Dictionary) -> tuple[bool, str]:
    """(valid, reason) where reason in {'ok', 'short', 'not_word', 'not_on_board'}."""
    w = word.lower()
    if len(w) < MIN_LEN:
        return False, "short"
    if w not in dictionary:
        return False, "not_word"
    if find_path(board, w) is None:
        return False, "not_on_board"
    return True, "ok"


def packed_board(
    rng: random.Random,
    dictionary: Dictionary,
    min_words: int = 40,
    min_longest: int = 6,
    tries: int = 500,
) -> tuple[str, dict[str, list[int]]]:
    """A random board with at least `min_words` words and one of >= `min_longest` letters."""
    best: tuple[str, dict[str, list[int]]] | None = None
    for _ in range(tries):
        b = random_board(rng)
        sol = solve(b, dictionary)
        if best is None or len(sol) > len(best[1]):
            best = (b, sol)
        if len(sol) >= min_words and any(len(w) >= min_longest for w in sol):
            return b, sol
    assert best is not None
    return best


def total_score(words: list[str]) -> int:
    return sum(score_word(w) for w in set(words))


if __name__ == "__main__":
    import sys

    d = load_dictionary()
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    b, sol = packed_board(random.Random(seed), d)
    print(grid_text(b))
    print(f"{len(sol)} words, longest {max(len(w) for w in sol)}, max score {total_score(list(sol))}")
    print(", ".join(sorted(sol, key=lambda w: (-len(w), w))[:25]))
