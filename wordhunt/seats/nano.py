"""Nano seat: wraps a `nano.policy.Policy` (StudentPolicy / EscalatingPolicy) so it acts through
the hand like everyone else. The nano contract is `act(board, path_tuple) -> ((type, tile), info)`.

Torch and the checkpoint are optional: if `NANO_CKPT` is unset or torch is missing, `available()`
is False and the seat is hidden from the catalog. CPU inference, same process.
"""
from __future__ import annotations

import os
import random
import time

from .base import ABORT, SUBMIT, Action, Policy

_cache: dict[str, object] = {}


def ckpt_path() -> str:
    return os.environ.get("NANO_CKPT", "")


def available() -> bool:
    p = ckpt_path()
    if not p or not os.path.exists(p):
        return False
    try:
        import torch  # noqa: F401
        import nano.policy  # noqa: F401
    except Exception:
        return False
    return True


def load_student(temperature: float = 1.0, seed: int = 0):
    """Load (and cache) the nano model from NANO_CKPT; returns a nano StudentPolicy."""
    import torch
    from nano.model import NanoAgent
    from nano.policy import StudentPolicy

    key = ckpt_path()
    if key not in _cache:
        model, extra = NanoAgent.load(key, map_location="cpu")
        _cache[key] = (model, extra)
    model, _ = _cache[key]
    return StudentPolicy(model, torch.device("cpu"), temperature=temperature, seed=seed)


class NanoPolicy(Policy):
    name = "nano"

    def __init__(self, inner, rng: random.Random | None = None):
        self.inner = inner            # nano.policy.Policy
        self.rng = rng or random.Random()
        self.stats = {"actions": 0, "ms": 0.0, "escalated": 0}

    async def start_round(self, board: str, words: dict[str, list[int]]) -> None:
        self.stats = {"actions": 0, "ms": 0.0, "escalated": 0}
        if hasattr(self.inner, "reset"):
            self.inner.reset()

    def next_action(self, board: str, path: list[int], found: set[str]) -> Action | None:
        t0 = time.perf_counter()
        (kind, tile), info = self.inner.act(board, tuple(path))
        self.stats["actions"] += 1
        self.stats["ms"] += (time.perf_counter() - t0) * 1000
        if info.get("escalated"):
            self.stats["escalated"] += 1
        if kind == "extend":
            return Action.extend(int(tile))
        if kind == "submit":
            return SUBMIT
        return ABORT

    def public_stats(self) -> dict:
        n = self.stats["actions"]
        return {"actions": n, "latency_ms": round(self.stats["ms"] / n, 2) if n else None,
                "escalated": self.stats["escalated"], "tokens": 0, "cost_usd": 0.0}
