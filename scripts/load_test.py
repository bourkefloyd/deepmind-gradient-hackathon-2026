#!/usr/bin/env python3
"""Load test: N simulated humans in one Word Hunt VS room over WebSockets.

    python scripts/load_test.py --url https://iter4b---wordhunt-pngitthrva-uw.a.run.app --n 30
    python scripts/load_test.py --url http://127.0.0.1:8765 --n 30 --cap 40      # also tries a 41st join

Each client has its own player id/name, joins the room, and once the race starts plays like a human:
cursor moves at ~10 Hz along a real path on the board, one submitted word every 4-8 s (valid words
from the local solver, so the server has to judge + score + fan out real ticks). The first client hosts
and presses Start. Measured: join success, per-message latency (server `now` in state messages and
round-trip of our own `path` -> `cursors` echo), dropped sockets, results delivered to all, and server
CPU/RSS from /api/health when the build exposes `proc`.

Needs `websockets` (pip) and the word lists in data/ (make words) for the solver.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time
import urllib.request

import websockets

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from wordhunt.solver import get_solver  # noqa: E402


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def http_json(url: str, method: str = "GET") -> dict:
    req = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


class Client:
    def __init__(self, idx: int, ws_url: str, stats: dict, solver, args):
        self.idx, self.ws_url, self.stats, self.solver, self.args = idx, ws_url, stats, solver, args
        self.pid = f"lt_{args.run_id}_{idx:02d}"
        self.name = f"Load-{idx:02d}"
        self.joined = False
        self.dropped = False
        self.got_results = False
        self.n_state = 0
        self.n_cursor_msgs = 0
        self.n_ticks = 0
        self.words_ok = 0
        self.words_sent = 0
        self.board = ""
        self.state = ""
        self.candidates: list[tuple[str, list[int]]] = []
        self.path_sent_at: dict[str, float] = {}
        self.ws = None
        self.offset = 0.0
        self.err = ""

    async def run(self):
        try:
            async with websockets.connect(self.ws_url, open_timeout=20, ping_interval=20, max_queue=None) as ws:
                self.ws = ws
                t0 = time.time()
                await ws.send(json.dumps({"type": "hello", "player_id": self.pid, "name": self.name}))
                player = asyncio.create_task(self.play())
                try:
                    async for raw in ws:
                        m = json.loads(raw)
                        if m["type"] == "welcome":
                            self.joined = True
                            self.stats["join_s"].append(time.time() - t0)
                        elif m["type"] == "error":
                            self.err = m.get("error", "error")
                            if m.get("fatal"):
                                break
                        elif m["type"] == "state":
                            self.n_state += 1
                            if self.n_state == 1:
                                self.offset = m["now"] - time.time()
                            else:
                                # one-way latency of a state broadcast, corrected by the first-sample clock offset
                                self.stats["state_lat_ms"].append((time.time() + self.offset - m["now"]) * 1000)
                            self.state = m["state"]
                            if m["state"] == "playing" and m["board"] != self.board:
                                self.board = m["board"]
                                words = self.solver.solve(self.board)
                                self.candidates = [(w, p) for w, p in words.items() if len(w) >= 3]
                                random.shuffle(self.candidates)
                            if m["state"] == "results":
                                self.got_results = True
                                break
                        elif m["type"] == "cursors":
                            self.n_cursor_msgs += 1
                            mine = m["cursors"].get(self.pid)
                            if mine:
                                key = ",".join(map(str, mine.get("path") or []))
                                t = self.path_sent_at.pop(key, None)
                                if t is not None:
                                    self.stats["cursor_rtt_ms"].append((time.time() - t) * 1000)
                        elif m["type"] == "tick":
                            self.n_ticks += 1
                        elif m["type"] == "mine":
                            self.words_ok += 1 if m.get("ok") else 0
                finally:
                    player.cancel()
        except Exception as e:  # connection refused / closed
            if not self.got_results:
                self.dropped = True
                self.err = self.err or repr(e)

    async def send(self, o: dict):
        if self.ws:
            await self.ws.send(json.dumps(o))

    async def play(self):
        # wait to be seated + for the race
        while not (self.joined and self.state == "playing" and self.candidates):
            await asyncio.sleep(0.1)
        await asyncio.sleep(random.uniform(0.5, 2.5))
        while self.state == "playing":
            if not self.candidates:
                break
            word, path = self.candidates.pop()
            # trace the path at ~10 Hz like a thumb, then submit
            for k in range(1, len(path) + 1):
                sub = path[:k]
                key = ",".join(map(str, sub))
                self.path_sent_at[key] = time.time()
                await self.send({"type": "path", "path": sub})
                await asyncio.sleep(0.1)
            await self.send({"type": "submit", "path": path})
            self.words_sent += 1
            await self.send({"type": "path", "path": []})
            await asyncio.sleep(random.uniform(self.args.min_gap, self.args.max_gap))


async def host_start(ws_url: str, pid: str, name: str, n_wait: int, timeout: float, seated: asyncio.Event):
    """The host: seated first (the first human in becomes host), waits until n_wait humans are in the lobby, then Start."""
    async with websockets.connect(ws_url, open_timeout=20, ping_interval=20) as ws:
        await ws.send(json.dumps({"type": "hello", "player_id": pid, "name": name}))
        t0 = time.time()
        started = False
        async for raw in ws:
            m = json.loads(raw)
            if m["type"] == "welcome":
                seated.set()
            if m["type"] == "state":
                humans = sum(1 for s in m["seats"] if s["kind"] == "human")
                if not started and m["state"] == "lobby" and (humans >= n_wait or time.time() - t0 > timeout):
                    print(f"[host] {humans} humans seated after {time.time()-t0:.1f}s -> Start", flush=True)
                    await ws.send(json.dumps({"type": "start"}))
                    started = True
                if m["state"] == "results":
                    return m


async def cap_probe(ws_url: str, run_id: str) -> str:
    async with websockets.connect(ws_url, open_timeout=20) as ws:
        await ws.send(json.dumps({"type": "hello", "player_id": f"lt_{run_id}_cap", "name": "Cap-Probe"}))
        try:
            async for raw in ws:
                m = json.loads(raw)
                if m["type"] == "error":
                    return f"rejected: {m.get('error')} (reason={m.get('reason')})"
                if m["type"] == "welcome":
                    return "ACCEPTED (no cap hit)"
        except Exception as e:
            return f"closed: {e!r}"
    return "no answer"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--n", type=int, default=30, help="simulated humans incl. the host")
    ap.add_argument("--room", default="", help="join an existing room instead of creating one")
    ap.add_argument("--cap", type=int, default=0, help="after N joined, try (cap+1)-th join to verify the cap message")
    ap.add_argument("--min-gap", type=float, default=4.0)
    ap.add_argument("--max-gap", type=float, default=8.0)
    ap.add_argument("--join-timeout", type=float, default=25.0)
    ap.add_argument("--no-start", action="store_true", help="do not host/start; join a room someone else runs")
    args = ap.parse_args()
    args.run_id = f"{int(time.time()) % 100000:05d}"

    base = args.url.rstrip("/")
    ws_base = base.replace("https://", "wss://").replace("http://", "ws://")
    health0 = None
    try:
        health0 = http_json(base + "/api/health")
        print(f"[health] build={health0.get('build')} rooms={health0.get('rooms')} proc={health0.get('proc')}")
    except Exception as e:
        print(f"[health] unavailable: {e!r}")
    code = args.room or http_json(base + "/api/rooms", "POST")["code"]
    ws_url = f"{ws_base}/ws/{code}"
    print(f"[room] {code}  ->  {base}/r/{code}   (spectate: {base}/s/{code})")

    solver = get_solver()
    stats = {"join_s": [], "state_lat_ms": [], "cursor_rtt_ms": []}
    n_clients = args.n if args.no_start else args.n - 1
    clients = [Client(i + 1, ws_url, stats, solver, args) for i in range(n_clients)]
    host_task = None
    if not args.no_start:
        seated = asyncio.Event()
        host_task = asyncio.create_task(host_start(ws_url, f"lt_{args.run_id}_host", "Load-Host", args.n, args.join_timeout, seated))
        await asyncio.wait_for(seated.wait(), 20)
    tasks = [asyncio.create_task(c.run()) for c in clients]

    # wait for joins
    t0 = time.time()
    while time.time() - t0 < args.join_timeout and sum(c.joined for c in clients) + sum(bool(c.err) for c in clients) < n_clients:
        await asyncio.sleep(0.2)
    joined = sum(c.joined for c in clients) + (0 if args.no_start else 1)
    rejected = [c for c in clients if c.err and not c.joined]
    print(f"[join] {joined}/{args.n} seated in {time.time()-t0:.1f}s; rejected={len(rejected)}"
          + (f" e.g. {rejected[0].err!r}" if rejected else ""))
    if args.cap and joined >= args.cap:
        print(f"[cap] probe join #{args.cap + 1}: {await cap_probe(ws_url, args.run_id)}")

    t_race0 = time.time()
    cpu0 = (health0 or {}).get("proc", {}).get("cpu_s")
    wall0 = time.time()
    await asyncio.gather(*tasks, *( [host_task] if host_task else []), return_exceptions=True)
    race_s = time.time() - t_race0

    health1 = None
    try:
        health1 = http_json(base + "/api/health")
    except Exception:
        pass
    cpu_line = "server CPU: n/a (build does not expose proc)"
    if health1 and health1.get("proc") and cpu0 is not None:
        d_cpu = health1["proc"]["cpu_s"] - cpu0
        d_wall = time.time() - wall0
        cpu_line = f"server CPU: {d_cpu:.1f}s over {d_wall:.1f}s wall = {100*d_cpu/d_wall:.0f}% of one core; rss={health1['proc'].get('rss_mb')} MB"

    print("\n=== load test summary ===")
    print(f"target            {base}  room {code}  build {health1.get('build') if health1 else '?'}")
    print(f"clients           {args.n} humans ({'host+' if not args.no_start else ''}{n_clients} sim)")
    print(f"join success      {joined}/{args.n}   join time p50 {pct(stats['join_s'],50)*1000:.0f} ms  p95 {pct(stats['join_s'],95)*1000:.0f} ms")
    print(f"dropped sockets   {sum(c.dropped for c in clients)}")
    print(f"results delivered {sum(c.got_results for c in clients)}/{n_clients} sim clients (+host {'yes' if host_task and not host_task.cancelled() and host_task.exception() is None else 'n/a'})")
    print(f"state latency     p50 {pct(stats['state_lat_ms'],50):.0f} ms  p95 {pct(stats['state_lat_ms'],95):.0f} ms  (n={len(stats['state_lat_ms'])}, one-way, clock-offset corrected)")
    print(f"cursor echo RTT   p50 {pct(stats['cursor_rtt_ms'],50):.0f} ms  p95 {pct(stats['cursor_rtt_ms'],95):.0f} ms  (n={len(stats['cursor_rtt_ms'])}, own path -> cursors broadcast; includes server coalescing)")
    print(f"words             sent {sum(c.words_sent for c in clients)}  accepted {sum(c.words_ok for c in clients)}  ticks seen/client p50 {statistics.median([c.n_ticks for c in clients]) if clients else 0:.0f}  cursor msgs/client p50 {statistics.median([c.n_cursor_msgs for c in clients]) if clients else 0:.0f}")
    print(f"round wall        {race_s:.1f}s")
    print(cpu_line)
    if health1 and health1.get("room_list"):
        r = next((x for x in health1["room_list"] if x["code"] == code), None)
        if r:
            print(f"room after        {r}")


if __name__ == "__main__":
    asyncio.run(main())
