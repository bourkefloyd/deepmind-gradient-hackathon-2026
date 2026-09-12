"""Nano seat: wraps `nano.seat.NanoSeat` (the ~10M-param policy, CPU, same process) so it acts
through the hand like everyone else. The nano never sees the solver output; its inputs are the
16 tiles and the current path. See nano/README.md ("Plugging the seat into wordhunt/seats/").

Torch and the checkpoint are optional: if either is missing, `available()` is False and the seat
is hidden from the catalog (the reflex bots remain as the honest fallback).
"""
from __future__ import annotations

import os
import random
import time

from .base import ABORT, SUBMIT, Action, Policy

DEFAULT_CKPT = os.path.join(os.path.dirname(__file__), "..", "..", "nano", "checkpoints", "d6_s0.pt")


def ckpt_path() -> str:
    return os.environ.get("NANO_CKPT") or DEFAULT_CKPT


def temperature() -> float:
    return float(os.environ.get("WH_NANO_TEMPERATURE", os.environ.get("NANO_TEMPERATURE", "1.0")))


_avail: bool | None = None


def available() -> bool:
    global _avail
    if _avail is None:
        p = ckpt_path()
        ok = bool(p) and os.path.exists(p)
        if ok:
            try:
                import torch  # noqa: F401
                import nano.seat  # noqa: F401
            except Exception:
                ok = False
        _avail = ok
    return _avail


def load_seat(seed: int = 0):
    from nano.seat import NanoSeat

    return NanoSeat(ckpt_path(), temperature=temperature(), seed=seed, threads=1)


class NanoPolicy(Policy):
    name = "nano"

    def __init__(self, seat=None, rng: random.Random | None = None):
        self.rng = rng or random.Random()
        self.seat = seat or load_seat(seed=self.rng.randrange(1 << 30))
        self.stats = {"actions": 0, "ms": 0.0, "aborts": 0, "submits": 0}

    async def start_round(self, board: str, words: dict[str, list[int]]) -> None:
        # `words` is the oracle; the nano must not see it.
        self.stats = {"actions": 0, "ms": 0.0, "aborts": 0, "submits": 0}
        self.seat.reset(board)

    def next_action(self, board: str, path: list[int], found: set[str]) -> Action | None:
        t0 = time.perf_counter()
        a = self.seat.next_action(board, list(path), found)
        self.stats["ms"] += (time.perf_counter() - t0) * 1000
        self.stats["actions"] += 1
        if a[0] == "extend":
            return Action.extend(int(a[1]))
        if a[0] == "submit":
            self.stats["submits"] += 1
            return SUBMIT
        self.stats["aborts"] += 1
        return ABORT

    def on_result(self, word: str, ok: bool, reason: str) -> None:
        self.seat.on_result(word, ok, reason)

    def public_stats(self) -> dict:
        n = self.stats["actions"]
        return {"actions": n, "latency_ms": round(self.stats["ms"] / n, 2) if n else None,
                "aborts": self.stats["aborts"], "submits": self.stats["submits"],
                "tokens": 0, "cost_usd": 0.0, "params_M": round(self.seat.n_params / 1e6, 1)}
