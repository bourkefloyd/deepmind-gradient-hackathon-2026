"""Catalog of AI seats a host can add to a room. Each entry builds a (policy, hand profile) pair."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Callable

from ..hand import HandProfile
from ..solver import Solver
from .base import Policy, RandomSwiperPolicy
from .fake import FAKE_SEATS
from .gemma import GemmaPolicy, gemma_seats_from_env
from . import crowd
from . import nano as nano_seat


@dataclass
class SeatSpec:
    id: str
    name: str
    label: str
    make: Callable[[Solver, random.Random], tuple[Policy, HandProfile]]
    available: bool = True
    default: bool = False       # seated when a room is created (overridden by WH_SEATS)
    namer: Callable[[random.Random, set[str]], str] | None = None   # per-seat display name (crowd)

    def public(self) -> dict:
        return {"id": self.id, "name": self.name, "label": self.label, "available": self.available}


_gthink = tuple(float(x) for x in os.environ.get("WH_GEMMA_THINK", "2,4").split(","))
GEMMA_PROFILE = HandProfile(hz=10, lag=(0.15, 0.3), think=(_gthink[0], _gthink[-1]), p_wrong=0.05, p_hesitate=0.04)
# The nano hesitates and backs out on its own (abort actions), so the hand adds no wrong taps.
# Between-word think time bounds its pace (PLAN: "speed is bounded"); WH_NANO_THINK="1.0,2.2" overrides.
_think = tuple(float(x) for x in os.environ.get("WH_NANO_THINK", "6,9").split(","))
NANO_PROFILE = HandProfile(hz=float(os.environ.get("WH_NANO_HZ", 10)), lag=(0.15, 0.3), think=(_think[0], _think[-1]),
                           p_wrong=0.0, p_hesitate=float(os.environ.get("WH_NANO_HESITATE", 0.05)))


def tuning() -> dict:
    """Live pace knobs, surfaced in /api/health so the deployed dial is visible."""
    keys = ("WH_SEATS", "NANO_CKPT", "WH_NANO_THINK", "WH_NANO_TEMPERATURE", "WH_NANO_HESITATE", "WH_NANO_HZ",
            "WH_REFLEX_A_THINK", "WH_REFLEX_B_THINK", "WH_REFLEX_HESITATE", "WH_REFLEX_WRONG", "WH_GEMMA_THINK", "GEMMA_MODELS",
            "WH_CROWD_THINK", "WH_CROWD_WRONG", "WH_MAX_AI", "WH_MAX_AI_CROWD")
    out = {k: os.environ.get(k, "(default)") for k in keys}
    out["nano_ckpt_resolved"] = nano_seat.ckpt_name()
    return out


def default_lineup() -> list[str]:
    """Spec ids seated in a new room. WH_SEATS="nano,reflex-a" overrides; default is the nano hero seat."""
    raw = os.environ.get("WH_SEATS")
    if raw is not None:
        return [x.strip() for x in raw.split(",") if x.strip()]
    return ["nano"] if nano_seat.available() else ["reflex-a", "reflex-b"]


def build_catalog() -> list[SeatSpec]:
    specs: list[SeatSpec] = []
    for f in FAKE_SEATS:
        specs.append(SeatSpec(f["seat_id"].split(":", 1)[1], f["name"], "heuristic",
                              (lambda f: lambda solver, rng: (f["policy"](solver.rank, rng), f["profile"]))(f)))
    specs.append(SeatSpec("random", "Random", "random swiper",
                          lambda solver, rng: (RandomSwiperPolicy(rng), HandProfile(hz=10, think=(0.3, 0.8), p_wrong=0.0))))

    lambda_ckpt = nano_seat.ckpt_name().startswith("d6_lambda")
    specs.append(SeatSpec("nano", os.environ.get("WH_NANO_SEAT_NAME") or ("Nano 10M (Lambda)" if lambda_ckpt else "Nano 10M"),
                          "10M params · $0 · ~2 ms" + (" · trained on Lambda A100" if lambda_ckpt else " · CPU-trained fallback"),
                          lambda solver, rng: (nano_seat.NanoPolicy(rng=rng), NANO_PROFILE),
                          available=nano_seat.available(), default=True))
    specs.append(SeatSpec("crowd", "Crowd", "casual bot · random name", crowd.make, namer=crowd.pick_name))

    for g in gemma_seats_from_env():
        specs.append(SeatSpec(g["id"], g.get("name", g["id"]), g.get("label", "gemma"),
                              (lambda g: lambda solver, rng: (GemmaPolicy(g["base_url"], g["model"], g.get("api_key", ""), rng,
                                                                          thinking=bool(g.get("thinking", False))), GEMMA_PROFILE))(g)))
    return specs


_catalog: list[SeatSpec] | None = None


def catalog() -> list[SeatSpec]:
    global _catalog
    if _catalog is None:
        _catalog = build_catalog()
    return _catalog


def lineup() -> list[SeatSpec]:
    out = []
    for sid in default_lineup():
        spec = find(sid)
        if spec and spec.available:
            out.append(spec)
    return out


def find(spec_id: str) -> SeatSpec | None:
    return next((s for s in catalog() if s.id == spec_id), None)
