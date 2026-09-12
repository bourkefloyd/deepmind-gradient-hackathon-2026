"""Room: code, seats, 20 s countdown, 75 s race, private word lists, ticker, rematch."""
from __future__ import annotations

import asyncio
import os
import random
import string
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from .board import generate_board
from .hand import Hand, HandProfile
from .scoring import score_word
from .seats.base import Policy
from .seats.fake import FAKE_SEATS
from .solver import Solver

COUNTDOWN_S = float(os.environ.get("WH_COUNTDOWN_S", 20.0))
RACE_S = float(os.environ.get("WH_RACE_S", 75.0))
TICKER_MAX = 60
COLORS = ["#ff5d5d", "#4da3ff", "#ffc93c", "#42d392", "#c77dff", "#ff9f43", "#2ec4b6", "#f368e0"]

Send = Callable[[dict], Awaitable[None]]


@dataclass
class Seat:
    seat_id: str
    name: str
    kind: str                       # "human" | "ai"
    color: str
    hand: Hand | None = None
    sockets: set[Any] = field(default_factory=set)
    found: dict[str, int] = field(default_factory=dict)   # word -> score
    queued: bool = False            # joined mid-round; plays from the next round
    cursor_path: list[int] = field(default_factory=list)
    label: str = ""                 # e.g. "heuristic", "nano 10M", "gemma-4-31b"

    @property
    def score(self) -> int:
        return sum(self.found.values())

    @property
    def connected(self) -> bool:
        return self.kind == "ai" or bool(self.sockets)

    def public(self, reveal_words: bool) -> dict:
        d = {
            "seat_id": self.seat_id, "name": self.name, "kind": self.kind, "color": self.color,
            "score": self.score, "n_words": len(self.found), "connected": self.connected,
            "queued": self.queued, "label": self.label,
        }
        if reveal_words:
            d["words"] = sorted(self.found.items(), key=lambda kv: (-kv[1], kv[0]))
        return d


def new_code(rng: random.Random) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(rng.choice(alphabet) for _ in range(4))


class Room:
    def __init__(self, code: str, solver: Solver, rng: random.Random | None = None):
        self.code = code
        self.solver = solver
        self.rng = rng or random.Random()
        self.seats: dict[str, Seat] = {}
        self.host_id: str | None = None
        self.state = "lobby"            # lobby | countdown | playing | results
        self.board = ""
        self.words: dict[str, list[int]] = {}
        self.round_no = 0
        self.ticker: list[dict] = []
        self.phase_ends_at = 0.0
        self.created_at = time.time()
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self.results: list[dict] = []
        for spec in FAKE_SEATS:
            self.add_ai(spec["seat_id"], spec["name"], spec["policy"](solver.rank, random.Random(self.rng.random())), spec["profile"], label="heuristic")

    # ---- seats -------------------------------------------------------------------------------
    def _color(self) -> str:
        used = {s.color for s in self.seats.values()}
        for c in COLORS:
            if c not in used:
                return c
        return self.rng.choice(COLORS)

    def add_ai(self, seat_id: str, name: str, policy: Policy, profile: HandProfile, label: str = "") -> Seat:
        seat = Seat(seat_id, name, "ai", self._color(), hand=Hand(policy, profile, random.Random(self.rng.random())), label=label)
        seat.queued = self.state in ("countdown", "playing")
        self.seats[seat_id] = seat
        return seat

    def add_human(self, player_id: str, name: str, ws: Any) -> Seat:
        seat = self.seats.get(player_id)
        if seat is None:
            seat = Seat(player_id, name, "human", self._color())
            seat.queued = self.state in ("countdown", "playing")
            self.seats[player_id] = seat
        if name:
            seat.name = name
        seat.sockets.add(ws)
        if self.host_id is None or not self._host_connected():
            self.host_id = player_id
        return seat

    def _host_connected(self) -> bool:
        h = self.seats.get(self.host_id or "")
        return bool(h and h.sockets)

    def drop_socket(self, ws: Any) -> None:
        for s in self.seats.values():
            s.sockets.discard(ws)

    def remove_seat(self, seat_id: str) -> None:
        self.seats.pop(seat_id, None)

    @property
    def humans_connected(self) -> int:
        return sum(1 for s in self.seats.values() if s.kind == "human" and s.sockets)

    # ---- snapshot / io ----------------------------------------------------------------------
    def now(self) -> float:
        return time.time()

    def snapshot(self) -> dict:
        reveal = self.state == "results"
        return {
            "type": "state",
            "code": self.code,
            "state": self.state,
            "round": self.round_no,
            "host_id": self.host_id,
            "board": self.board if self.state in ("playing", "results") else "",
            "seats": [s.public(reveal) for s in self.seats.values()],
            "ticker": self.ticker[-TICKER_MAX:],
            "phase_ends_at": self.phase_ends_at,
            "now": self.now(),
            "countdown_s": COUNTDOWN_S,
            "race_s": RACE_S,
            "results": self.results if reveal else [],
        }

    async def broadcast(self, msg: dict) -> None:
        dead = []
        for s in self.seats.values():
            for ws in list(s.sockets):
                try:
                    await ws.send_json(msg)
                except Exception:
                    dead.append(ws)
        for ws in dead:
            self.drop_socket(ws)

    async def send_state(self) -> None:
        await self.broadcast(self.snapshot())

    # ---- flow --------------------------------------------------------------------------------
    def can_control(self, player_id: str) -> bool:
        return player_id == self.host_id or not self._host_connected()

    async def start(self) -> None:
        if self.state not in ("lobby", "results"):
            return
        self._cancel_task()
        self._task = asyncio.create_task(self._run_round())

    def _cancel_task(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()

    async def _run_round(self) -> None:
        self.round_no += 1
        self.state = "countdown"
        self.board, self.words = generate_board(self.solver, self.rng)
        self.ticker = []
        self.results = []
        for s in self.seats.values():
            s.found = {}
            s.queued = False
            s.cursor_path = []
        self.phase_ends_at = self.now() + COUNTDOWN_S
        await self.send_state()
        await asyncio.sleep(COUNTDOWN_S)

        self.state = "playing"
        start = self.now()
        self.phase_ends_at = start + RACE_S
        for s in self.seats.values():
            if s.hand:
                await s.hand.start_round(self.board, self.words, start)
        await self.send_state()

        period = 0.05
        while self.now() < self.phase_ends_at:
            await self._tick_ais()
            await asyncio.sleep(period)

        self.state = "results"
        for s in self.seats.values():
            if s.hand:
                await s.hand.policy.end_round()
            s.cursor_path = []
        self.results = self._compute_results()
        await self.send_state()

    def _compute_results(self) -> list[dict]:
        board_words = sorted(self.words, key=lambda w: (-len(w), w))
        return [{
            "best_words": board_words[:12],
            "n_board_words": len(self.words),
            "max_score": sum(score_word(w) for w in self.words),
        }]

    async def _tick_ais(self) -> None:
        now = self.now()
        cursors: dict[str, dict] = {}
        ticks: list[dict] = []
        for s in self.seats.values():
            if not s.hand or s.queued:
                continue
            for ev in s.hand.tick(now):
                if ev.kind == "cursor":
                    s.cursor_path = ev.path
                    cursors[s.seat_id] = {"path": ev.path, "tile": ev.tile}
                elif ev.kind == "submit":
                    res = self.judge(s, ev.path)
                    s.hand.on_result(res["word"], res["ok"], res["reason"])
                    ticks.append(res)
                elif ev.kind == "abort":
                    s.cursor_path = []
                    cursors[s.seat_id] = {"path": [], "tile": None}
        if cursors:
            await self.broadcast({"type": "cursors", "cursors": cursors})
        for t in ticks:
            await self.broadcast({"type": "tick", "entry": t})

    def judge(self, seat: Seat, path: list[int]) -> dict:
        """Validate a path; update the seat's private list; return a ticker entry."""
        word = "".join(self.board[t] for t in path if 0 <= t < 16)
        entry = {"seat_id": seat.seat_id, "name": seat.name, "color": seat.color, "word": word.upper(),
                 "ok": False, "delta": 0, "reason": "", "t": round(self.now() - (self.phase_ends_at - RACE_S), 1)}
        if self.state != "playing" or seat.queued:
            entry["reason"] = "closed"
            return entry
        valid = self.solver.valid_path(self.board, path)
        if valid is None:
            entry["reason"] = "miss"
        elif valid in seat.found:
            entry["reason"] = "dup"
        else:
            pts = score_word(valid)
            seat.found[valid] = pts
            entry.update(ok=True, delta=pts)
        self.ticker.append(entry)
        if len(self.ticker) > TICKER_MAX * 2:
            self.ticker = self.ticker[-TICKER_MAX:]
        return entry

    async def human_submit(self, seat: Seat, path: list[int]) -> dict:
        res = self.judge(seat, path)
        res["total"] = seat.score
        await self.broadcast({"type": "tick", "entry": res})
        return res

    async def human_path(self, seat: Seat, path: list[int]) -> None:
        seat.cursor_path = path
        await self.broadcast({"type": "cursors", "cursors": {seat.seat_id: {"path": path, "tile": path[-1] if path else None}}})
