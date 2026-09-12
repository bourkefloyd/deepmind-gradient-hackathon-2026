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
from . import nano as nano_seat


@dataclass
class SeatSpec:
    id: str
    name: str
    label: str
    make: Callable[[Solver, random.Random], tuple[Policy, HandProfile]]
    available: bool = True
    default: bool = False

    def public(self) -> dict:
        return {"id": self.id, "name": self.name, "label": self.label, "available": self.available}


GEMMA_PROFILE = HandProfile(hz=10, lag=(0.15, 0.3), think=(0.6, 1.4), p_wrong=0.05, p_hesitate=0.04)
NANO_PROFILE = HandProfile(hz=10, lag=(0.15, 0.3), think=(0.4, 1.0), p_wrong=0.0, p_hesitate=0.03)


def build_catalog() -> list[SeatSpec]:
    specs: list[SeatSpec] = []
    for f in FAKE_SEATS:
        specs.append(SeatSpec(f["seat_id"].split(":", 1)[1], f["name"], "heuristic",
                              (lambda f: lambda solver, rng: (f["policy"](solver.rank, rng), f["profile"]))(f), default=True))
    specs.append(SeatSpec("random", "Random", "random swiper",
                          lambda solver, rng: (RandomSwiperPolicy(rng), HandProfile(hz=10, think=(0.3, 0.8), p_wrong=0.0))))

    nano_ok = nano_seat.available()
    temp = float(os.environ.get("NANO_TEMPERATURE", "1.0"))
    specs.append(SeatSpec("nano", "Nano", "nano 10M · CPU",
                          lambda solver, rng: (nano_seat.NanoPolicy(nano_seat.load_student(temp, rng.randrange(1 << 30)), rng), NANO_PROFILE),
                          available=nano_ok))

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


def find(spec_id: str) -> SeatSpec | None:
    return next((s for s in catalog() if s.id == spec_id), None)
