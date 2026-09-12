"""WebSocket bot: joins a room by code and plays Gemma's words through a hand.

    python -m gemma_seat.bot --room ABCD [--server ws://localhost:8000] [--modality text|image]
    python -m gemma_seat.bot --dry-run [--seed 3]      # no server: prints the hand timeline

The hand drains one word at a time: 150-300 ms reaction lag before each word,
then one tile step per tick at ~10 Hz (jittered), then submit. Words arrive
from Gemma as a stream so the finger starts on the first line, not the last.
Gemma is queried once per board (thinking off, hard timeout); the hand never
outruns the cadence cap, however fast the model is.

The wire protocol lives behind `ProtocolAdapter`. `WordhuntV1Adapter` follows
the draft on the game worker's branch (`wordhunt/server.py`, iteration 1):

    client -> {"type":"hello","player_id":..,"name":..}
    server -> {"type":"welcome","seat_id":..}
    server -> {"type":"state","state":"lobby|countdown|playing|results","board":"abcdefghijklmnop",
               "phase_ends_at":unix_s,"now":unix_s,...}
    client -> {"type":"path","path":[tile,...]}     live finger (broadcast as "cursors")
    client -> {"type":"submit","path":[tile,...]}   -> {"type":"mine","ok":bool,"word":..,"delta":..}

Swap or subclass the adapter when the final shapes land.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .boards import find_path, grid_text, score_word

log = logging.getLogger("gemma_seat.bot")


# --------------------------------------------------------------------------------------
# Protocol adapter
# --------------------------------------------------------------------------------------
@dataclass
class RoomEvent:
    kind: str  # "welcome" | "board" | "phase" | "result" | "tick" | "error" | "other"
    state: str = ""  # lobby | countdown | playing | results
    board: str = ""
    phase_ends_at: float = 0.0
    server_now: float = 0.0
    payload: dict = field(default_factory=dict)


class ProtocolAdapter:
    """Message shapes for one game-server protocol. Override for the real one."""

    def ws_url(self, server: str, room: str) -> str:
        raise NotImplementedError

    def hello(self, name: str, player_id: str) -> dict:
        raise NotImplementedError

    def path(self, tiles: list[int]) -> dict:
        raise NotImplementedError

    def submit(self, tiles: list[int]) -> dict:
        raise NotImplementedError

    def parse(self, msg: dict) -> RoomEvent:
        raise NotImplementedError


class WordhuntV1Adapter(ProtocolAdapter):
    def ws_url(self, server: str, room: str) -> str:
        return f"{server.rstrip('/')}/ws/{room.upper()}"

    def hello(self, name: str, player_id: str) -> dict:
        return {"type": "hello", "player_id": player_id, "name": name}

    def path(self, tiles: list[int]) -> dict:
        return {"type": "path", "path": tiles}

    def submit(self, tiles: list[int]) -> dict:
        return {"type": "submit", "path": tiles}

    def parse(self, msg: dict) -> RoomEvent:
        t = msg.get("type")
        if t == "welcome":
            return RoomEvent("welcome", payload=msg)
        if t == "state":
            board = (msg.get("board") or "").lower()
            ev = RoomEvent(
                "board" if board else "phase",
                state=str(msg.get("state", "")),
                board=board,
                phase_ends_at=float(msg.get("phase_ends_at") or 0.0),
                server_now=float(msg.get("now") or time.time()),
                payload=msg,
            )
            return ev
        if t == "mine":
            return RoomEvent("result", payload=msg)
        if t == "error":
            return RoomEvent("error", payload=msg)
        if t in ("tick", "cursors", "pong"):
            return RoomEvent("tick", payload=msg)
        return RoomEvent("other", payload=msg)


ADAPTERS: dict[str, type[ProtocolAdapter]] = {"v1": WordhuntV1Adapter}


# --------------------------------------------------------------------------------------
# Hand: cadence-capped finger
# --------------------------------------------------------------------------------------
@dataclass
class HandConfig:
    tick_hz: float = 10.0
    tick_jitter: float = 0.25  # +-25 % per step
    lag_min_s: float = 0.15
    lag_max_s: float = 0.30
    wrong_neighbor_p: float = 0.04  # occasional slip to a wrong neighbour, then backtrack


class Hand:
    """Drains a word queue tile by tile through `send`, one word at a time."""

    def __init__(self, cfg: HandConfig, rng: random.Random, send, clock=time.monotonic):
        self.cfg = cfg
        self.rng = rng
        self.send = send  # async (kind, tiles) -> None ; kind in {"path", "submit", "release"}
        self.clock = clock
        self.played: list[tuple[float, str, list[int]]] = []

    async def _tick(self) -> None:
        base = 1.0 / self.cfg.tick_hz
        await asyncio.sleep(base * (1 + self.rng.uniform(-self.cfg.tick_jitter, self.cfg.tick_jitter)))

    async def play_word(self, word: str, tiles: list[int], deadline: float) -> bool:
        if self.clock() + self.cfg.lag_min_s + len(tiles) / self.cfg.tick_hz > deadline:
            return False
        await asyncio.sleep(self.rng.uniform(self.cfg.lag_min_s, self.cfg.lag_max_s))
        path: list[int] = []
        for i, t in enumerate(tiles):
            if self.clock() >= deadline:
                await self.send("release", [])
                return False
            if i > 0 and self.rng.random() < self.cfg.wrong_neighbor_p:
                slip = self._wrong_neighbor(path, t)
                if slip is not None:
                    await self.send("path", path + [slip])
                    await self._tick()
                    await self.send("path", path)  # backtrack
                    await self._tick()
            path.append(t)
            await self.send("path", list(path))
            await self._tick()
        await self.send("submit", list(path))
        self.played.append((self.clock(), word, list(path)))
        return True

    def _wrong_neighbor(self, path: list[int], target: int) -> int | None:
        from .boards import NEIGHBORS

        if not path:
            return None
        options = [n for n in NEIGHBORS[path[-1]] if n != target and n not in path]
        return self.rng.choice(options) if options else None


# --------------------------------------------------------------------------------------
# Seat: Gemma -> queue -> hand
# --------------------------------------------------------------------------------------
class GemmaSeat:
    def __init__(self, client, modality: str, hand: Hand, rng: random.Random, clock=time.monotonic):
        self.client = client
        self.modality = modality
        self.hand = hand
        self.rng = rng
        self.clock = clock
        self.queue: asyncio.Queue[str | None] = asyncio.Queue()
        self.seen: set[str] = set()
        self.submitted: list[str] = []
        self.skipped: list[str] = []
        self.calls = 0

    def _fetch_words(
        self, board: str, loop: asyncio.AbstractEventLoop, budget_s: float, exclude: list[str]
    ) -> None:
        """Runs in a thread: streams Gemma's words into the asyncio queue."""
        try:
            for w in self.client.stream_words(board, self.modality, deadline_s=budget_s, exclude=exclude):
                loop.call_soon_threadsafe(self.queue.put_nowait, w)
        finally:
            loop.call_soon_threadsafe(self.queue.put_nowait, None)

    def _start_fetch(self, board: str, deadline: float) -> threading.Thread:
        loop = asyncio.get_running_loop()
        budget = max(1.0, deadline - self.clock() - 1.0)
        t = threading.Thread(
            target=self._fetch_words, args=(board, loop, budget, list(self.seen)), daemon=True
        )
        t.start()
        self.calls += 1
        return t

    async def play_round(self, board: str, deadline: float, max_calls: int = 8) -> None:
        """Drain Gemma's words through the hand until the deadline.

        Gemma is asked once, and re-asked (with the words already tried fed back)
        whenever its list runs dry and time remains, up to `max_calls` per round:
        a player keeps looking for the whole 75 s, and so does this seat.
        """
        self.queue = asyncio.Queue()
        self.seen.clear()
        self.submitted.clear()
        self.skipped.clear()
        self.calls = 0
        self._start_fetch(board, deadline)
        model_done = False
        while self.clock() < deadline:
            try:
                w = await asyncio.wait_for(self.queue.get(), timeout=max(0.05, deadline - self.clock()))
            except asyncio.TimeoutError:
                break
            if w is None:
                if self.calls < max_calls and deadline - self.clock() > 3.0:
                    log.info("list ran dry, asking again (%d/%d)", self.calls + 1, max_calls)
                    self._start_fetch(board, deadline)
                    continue
                model_done = True
                break
            if w in self.seen:
                continue
            self.seen.add(w)
            tiles = find_path(board, w)
            if tiles is None:
                # Not traceable on the board: a human would notice mid-swipe. Skip,
                # do not burn the cadence on it (the ticker shows misses only for
                # words the hand actually submits).
                self.skipped.append(w)
                log.info("skip %-10s (not on board)", w)
                continue
            ok = await self.hand.play_word(w, tiles, deadline)
            if ok:
                self.submitted.append(w)
                log.info("swipe %-10s +%d", w.upper(), score_word(w))
            else:
                break
        log.info(
            "round over: %d submitted, %d skipped, %d model calls, model_done=%s",
            len(self.submitted), len(self.skipped), self.calls, model_done,
        )


# --------------------------------------------------------------------------------------
# Bot: WebSocket glue
# --------------------------------------------------------------------------------------
async def run_bot(args: argparse.Namespace) -> None:
    import websockets

    from .client import GemmaSeatClient

    adapter = ADAPTERS[args.protocol]()
    rng = random.Random(args.seed)
    client = GemmaSeatClient(
        base_url=args.base_url, model=args.model, timeout_s=args.timeout, max_words=args.max_words
    )
    url = adapter.ws_url(args.server, args.room)
    player_id = args.player_id or f"gemma12b-{uuid.uuid4().hex[:6]}"
    log.info("connecting %s as %s (%s modality)", url, args.name, args.modality)

    async with websockets.connect(url, max_size=2**22) as ws:
        send_lock = asyncio.Lock()

        async def send_json(obj: dict) -> None:
            async with send_lock:
                await ws.send(json.dumps(obj))

        async def hand_send(kind: str, tiles: list[int]) -> None:
            if kind == "path":
                await send_json(adapter.path(tiles))
            elif kind == "submit":
                await send_json(adapter.submit(tiles))
            elif kind == "release":
                await send_json(adapter.path([]))

        hand = Hand(HandConfig(), rng, hand_send)
        seat = GemmaSeat(client, args.modality, hand, rng)
        await send_json(adapter.hello(args.name, player_id))

        current_board = ""
        round_task: asyncio.Task | None = None
        rounds = 0
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            ev = adapter.parse(msg)
            if ev.kind == "welcome":
                log.info("seated: %s", ev.payload)
            elif ev.kind == "error":
                log.error("server error: %s", ev.payload)
                if ev.payload.get("fatal"):
                    return
            elif ev.kind == "result":
                p = ev.payload
                log.info("judge %-10s ok=%s delta=%s %s", p.get("word"), p.get("ok"), p.get("delta"), p.get("reason", ""))
            elif ev.kind in ("board", "phase"):
                if ev.state == "playing" and ev.board and ev.board != current_board:
                    current_board = ev.board
                    # Convert the server clock to our monotonic clock.
                    remaining = ev.phase_ends_at - ev.server_now if ev.phase_ends_at else args.race_s
                    deadline = time.monotonic() + max(0.0, remaining) - args.safety_s
                    log.info("round %d board:\n%s\n(%.0fs left)", rounds + 1, grid_text(ev.board), remaining)
                    if round_task and not round_task.done():
                        round_task.cancel()
                    round_task = asyncio.create_task(seat.play_round(ev.board, deadline))
                    rounds += 1
                elif ev.state in ("results", "lobby", "countdown"):
                    if ev.state != "playing" and current_board and ev.state in ("results", "lobby"):
                        current_board = ""
                    if ev.state == "results" and round_task and round_task.done() and args.rounds and rounds >= args.rounds:
                        log.info("played %d rounds, leaving", rounds)
                        return


async def run_dry(args: argparse.Namespace) -> None:
    """No server: generate a board, stream Gemma, print the hand timeline."""
    from .boards import load_dictionary, packed_board, validate
    from .client import GemmaSeatClient

    d = load_dictionary()
    rng = random.Random(args.seed)
    board, sol = packed_board(rng, d)
    print(grid_text(board), f"\n({len(sol)} words on board)\n")
    t0 = time.monotonic()
    tiles_sent = 0

    async def hand_send(kind: str, tiles: list[int]) -> None:
        nonlocal tiles_sent
        if kind == "path":
            tiles_sent += 1
        elif kind == "submit":
            word = "".join(board[t] for t in tiles)
            ok, why = validate(board, word, d)
            print(f"{time.monotonic() - t0:6.2f}s  {word.upper():<10} {'+' + str(score_word(word)) if ok else why}")

    hand = Hand(HandConfig(), rng, hand_send)
    client = GemmaSeatClient(base_url=args.base_url, model=args.model, timeout_s=args.timeout, max_words=args.max_words)
    seat = GemmaSeat(client, args.modality, hand, rng)
    await seat.play_round(board, time.monotonic() + args.race_s)
    valid = [w for w in seat.submitted if validate(board, w, d)[0]]
    print(
        f"\n{len(seat.submitted)} swiped ({len(valid)} valid, score {sum(score_word(w) for w in valid)}), "
        f"{len(seat.skipped)} skipped as not-on-board, {tiles_sent} tile steps in {time.monotonic() - t0:.1f}s"
    )


def main(argv: list[str] | None = None) -> int:
    from .client import DEFAULT_BASE_URL, DEFAULT_MODEL

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--room", help="room code to join")
    p.add_argument("--server", default="ws://localhost:8000", help="game server ws base URL")
    p.add_argument("--protocol", choices=sorted(ADAPTERS), default="v1")
    p.add_argument("--name", default="Gemma 12B")
    p.add_argument("--player-id", default=None)
    p.add_argument("--modality", choices=["text", "image"], default="text")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--timeout", type=float, default=60.0, help="hard cap per model call")
    p.add_argument("--max-words", type=int, default=40)
    p.add_argument("--race-s", type=float, default=75.0, help="fallback race length if the server sends none")
    p.add_argument("--safety-s", type=float, default=0.5, help="stop swiping this long before the buzzer")
    p.add_argument("--rounds", type=int, default=0, help="leave after this many rounds (0 = stay)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(message)s")

    if args.dry_run:
        asyncio.run(run_dry(args))
        return 0
    if not args.room:
        p.error("--room is required unless --dry-run")
    asyncio.run(run_bot(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
