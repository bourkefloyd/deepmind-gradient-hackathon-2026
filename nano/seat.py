"""NanoSeat: CPU inference wrapper the game server drives tile by tile.

  seat = NanoSeat("nano/checkpoints/d6_lambda.pt", temperature=1.0)   # d6_s0.pt = Mac-trained fallback
  seat.reset(board)                 # 16 lowercase letters, row-major
  seat.step() -> ("extend", tile) | ("submit",) | ("abort",)

The seat keeps its own copy of the path (mirrors what the hand is tracing) and the words it has already
submitted successfully. No word list at inference: the only inputs to the network are the 16 tiles and the path.
`next_action(board, path, found)` is the same decision in the game's `wordhunt.seats.base.Policy` shape
(returns a tuple; the adapter in wordhunt/seats/ maps it to their Action).

  python -m nano.seat runs/d6_s0/model.pt        # ms/action on CPU
"""

from __future__ import annotations

import time
from typing import Iterable, Optional

import numpy as np
import torch

from .model import NanoAgent
from .policy import StudentPolicy
from .solver import MIN_LEN, path_word

Step = tuple  # ("extend", tile) | ("submit",) | ("abort",)


class NanoSeat:
    name = "nano"

    def __init__(self, checkpoint: str, temperature: float = 1.0, seed: int = 0, threads: int = 1, device: str = "cpu"):
        if device == "cpu" and threads:
            torch.set_num_threads(threads)
        model, extra = NanoAgent.load(checkpoint)
        self.model = model
        self.extra = extra
        self.policy = StudentPolicy(model, torch.device(device), temperature=temperature, seed=seed)
        self.board = ""
        self.path: tuple[int, ...] = ()
        self.found: set[str] = set()
        self.n_actions = 0
        self.total_ms = 0.0
        self.last_info: dict = {}

    @property
    def n_params(self) -> int:
        return self.model.n_params()

    def reset(self, board: str, found: Optional[Iterable[str]] = None) -> None:
        assert len(board) == 16, "board is 16 letters, row-major"
        self.board = board.lower()
        self.path = ()
        self.found = set(found or ())

    def on_result(self, word: str, ok: bool, reason: str = "") -> None:
        """Feedback from the room after a submit; only the accepted set is remembered (no dictionary)."""
        if ok:
            self.found.add(word.lower())

    def step(self) -> Step:
        a = self.next_action(self.board, list(self.path), self.found)
        if a[0] == "extend":
            self.path = self.path + (a[1],)
        else:
            if a[0] == "submit":
                # optimistic: assume the room accepts it, so the seat does not re-trace the same word
                self.found.add(path_word(self.board, self.path))
            self.path = ()
        return a

    def next_action(self, board: str, path: list[int], found: set[str]) -> Step:
        t0 = time.perf_counter()
        p = tuple(path)
        d = self.policy.dists([board], [p])[0]
        # a word we already have gets no submit mass: extend or abort instead
        if len(p) >= MIN_LEN and path_word(board, p) in found:
            d["type"] = d["type"].copy()
            d["type"][1] = 0.0
            if d["type"].sum() <= 0:
                d["type"][2] = 1.0
        (atype, tile), conf = self.policy.pick(d, p)
        self.last_info = {"confidence": conf, "value": d["value"]}
        self.total_ms += (time.perf_counter() - t0) * 1000
        self.n_actions += 1
        return ("extend", tile) if atype == "extend" else (atype,)

    @property
    def ms_per_action(self) -> float:
        return self.total_ms / max(self.n_actions, 1)


def bench(checkpoint: str, n: int = 500, threads: int = 1) -> dict:
    seat = NanoSeat(checkpoint, threads=threads)
    rng = np.random.default_rng(0)
    board = "".join(chr(97 + int(i)) for i in rng.integers(0, 26, 16))
    seat.reset(board)
    for _ in range(20):  # warm-up
        seat.step()
    seat.n_actions, seat.total_ms = 0, 0.0
    t0 = time.perf_counter()
    for _ in range(n):
        seat.step()
    wall = (time.perf_counter() - t0) * 1000 / n
    return {"params_M": round(seat.n_params / 1e6, 2), "ms_per_action": round(seat.ms_per_action, 3), "wall_ms_per_action": round(wall, 3), "threads": threads}


if __name__ == "__main__":
    import sys

    ck = sys.argv[1] if len(sys.argv) > 1 else "nano/checkpoints/d6_lambda.pt"
    for th in (1, 4):
        print(bench(ck, threads=th))
