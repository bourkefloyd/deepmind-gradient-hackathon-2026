"""Room: code, seats, 20 s countdown, 75 s race, private word lists, ticker, rematch."""
from __future__ import annotations

import asyncio
import os
import random
import string
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from .board import generate_board, load_packed_boards
from .hand import Hand, HandProfile
from .scoring import score_word
from .seats import registry
from .seats.base import Policy
from .solver import Solver

COUNTDOWN_S = float(os.environ.get("WH_COUNTDOWN_S", 20.0))
RACE_S = float(os.environ.get("WH_RACE_S", 75.0))
TICKER_MAX = 60
MAX_HUMANS = int(os.environ.get("WH_MAX_HUMANS", 24))
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
    spec_id: str = ""

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
        if self.hand and hasattr(self.hand.policy, "public_stats"):
            try:
                d["stats"] = self.hand.policy.public_stats()
            except Exception:
                pass
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
        self.spectators: set[Any] = set()
        self.perf = {"broadcast_ms_max": 0.0, "broadcast_ms_last": 0.0, "tick_late_ms_max": 0.0}
        self._boards = load_packed_boards()
        self.rng.shuffle(self._boards)
        for spec in registry.catalog():
            if spec.default and spec.available:
                self.add_from_catalog(spec.id)

    # ---- seats -------------------------------------------------------------------------------
    def _color(self) -> str:
        used = {s.color for s in self.seats.values()}
        for c in COLORS:
            if c not in used:
                return c
        return self.rng.choice(COLORS)

    def add_ai(self, seat_id: str, name: str, policy: Policy, profile: HandProfile, label: str = "", spec_id: str = "") -> Seat:
        seat = Seat(seat_id, name, "ai", self._color(), hand=Hand(policy, profile, random.Random(self.rng.random())), label=label, spec_id=spec_id)
        seat.queued = self.state == "playing"
        self.seats[seat_id] = seat
        return seat

    def add_from_catalog(self, spec_id: str) -> Seat | None:
        spec = registry.find(spec_id)
        if spec is None or not spec.available or len(self.seats) >= 8:
            return None
        n = sum(1 for s in self.seats.values() if s.spec_id == spec_id)
        seat_id = f"ai:{spec_id}" + (f"-{n + 1}" if n else "")
        name = spec.name + (f" {n + 1}" if n else "")
        try:
            policy, profile = spec.make(self.solver, random.Random(self.rng.random()))
        except Exception:
            return None
        return self.add_ai(seat_id, name, policy, profile, label=spec.label, spec_id=spec_id)

    @property
    def n_humans(self) -> int:
        return sum(1 for s in self.seats.values() if s.kind == "human")

    def add_human(self, player_id: str, name: str, ws: Any) -> Seat | None:
        """Seat a human. Joining in lobby/countdown/results plays now; joining mid-race is queued
        for the next round (the client shows a banner). Returns None when the room is full."""
        seat = self.seats.get(player_id)
        if seat is None:
            if self.n_humans >= MAX_HUMANS:
                return None
            seat = Seat(player_id, name, "human", self._color())
            seat.queued = self._late_for_this_round()
            self.seats[player_id] = seat
        if name:
            seat.name = name
        seat.sockets.add(ws)
        if self.host_id is None or not self._host_connected():
            self.host_id = player_id
        return seat

    def _late_for_this_round(self) -> bool:
        """Joining in lobby/countdown/results, or within the first HOTJOIN_S of a race, plays now."""
        if self.state != "playing":
            return False
        elapsed = RACE_S - (self.phase_ends_at - self.now())
        return elapsed > HOTJOIN_S

    def _host_connected(self) -> bool:
        h = self.seats.get(self.host_id or "")
        return bool(h and h.sockets)

    def drop_socket(self, ws: Any) -> None:
        self.spectators.discard(ws)
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

    @staticmethod
    def mask_entry(entry: dict) -> dict:
        w = entry["word"]
        return {**entry, "word": w[:2] + "_" * max(0, len(w) - 2), "masked": True}

    def ticker_for(self, seat_id: str | None) -> list[dict]:
        """Ticker as seen by one seat: others' words masked until results; own words in clear."""
        entries = self.ticker[-TICKER_MAX:]
        if self.state == "results":
            return entries
        return [e if e["seat_id"] == seat_id else self.mask_entry(e) for e in entries]

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
            "ticker": self.ticker_for(None),
            "max_humans": MAX_HUMANS,
            "perf": dict(self.perf),
            "phase_ends_at": self.phase_ends_at,
            "now": self.now(),
            "countdown_s": COUNTDOWN_S,
            "race_s": RACE_S,
            "results": self.results if reveal else [],
            "catalog": [c.public() for c in registry.catalog()],
            "spectators": len(self.spectators),
        }

    async def broadcast(self, msg: dict) -> None:
        dead = []
        for ws in list(self.spectators):
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for s in self.seats.values():
            for ws in list(s.sockets):
                try:
                    await ws.send_json(msg)
                except Exception:
                    dead.append(ws)
        for ws in dead:
            self.drop_socket(ws)

    async def send_state(self) -> None:
        snap = self.snapshot()
        if self.state == "results":
            await self.broadcast(snap)
            return
        t0 = time.perf_counter()
        for ws in list(self.spectators):
            await self._send(ws, snap)
        for s in self.seats.values():
            if not s.sockets:
                continue
            mine = {**snap, "ticker": self.ticker_for(s.seat_id)}
            for ws in list(s.sockets):
                await self._send(ws, mine)
        self._note_broadcast(t0)

    async def _send(self, ws: Any, msg: dict) -> None:
        try:
            await ws.send_json(msg)
        except Exception:
            self.drop_socket(ws)

    def _note_broadcast(self, t0: float) -> None:
        ms = (time.perf_counter() - t0) * 1000
        self.perf["broadcast_ms_last"] = round(ms, 2)
        self.perf["broadcast_ms_max"] = max(self.perf["broadcast_ms_max"], round(ms, 2))

    async def broadcast_tick(self, entry: dict, owner: "Seat | None") -> None:
        """Owner sees the full word; everyone else sees it masked until results."""
        t0 = time.perf_counter()
        masked = {"type": "tick", "entry": self.mask_entry(entry)}
        full = {"type": "tick", "entry": entry}
        for ws in list(self.spectators):
            await self._send(ws, masked)
        for s in self.seats.values():
            msg = full if s is owner else masked
            for ws in list(s.sockets):
                await self._send(ws, msg)
        self._note_broadcast(t0)

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
        self.board, self.words = self._next_board()
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
            t0 = time.perf_counter()
            await self._tick_ais()
            await asyncio.sleep(period)
            late = (time.perf_counter() - t0 - period) * 1000
            if late > self.perf["tick_late_ms_max"]:
                self.perf["tick_late_ms_max"] = round(late, 2)

        self.state = "results"
        for s in self.seats.values():
            if s.hand:
                await s.hand.policy.end_round()
            s.cursor_path = []
        self.results = self._compute_results()
        await self.send_state()

    def _next_board(self) -> tuple[str, dict[str, list[int]]]:
        """Packed boards first (stage-safe, known long words), random live boards after."""
        while self._boards:
            b = self._boards.pop()
            words = self.solver.solve(b)
            if words:
                return b, words
        return generate_board(self.solver, self.rng)

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
            t0 = time.perf_counter()
            await self.broadcast({"type": "cursors", "cursors": cursors})
            self._note_broadcast(t0)
        for t in ticks:
            await self.broadcast_tick(t, None)

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
        await self.broadcast_tick(res, seat)
        return {**res, "total": seat.score}

    async def human_path(self, seat: Seat, path: list[int]) -> None:
        seat.cursor_path = path
        await self.broadcast({"type": "cursors", "cursors": {seat.seat_id: {"path": path, "tile": path[-1] if path else None}}})
