"""FastAPI app: static page, room API, WebSocket room. Single process, in-memory rooms."""
from __future__ import annotations

import asyncio
import os
import random
import re
import time

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .persist import store as room_store
from .room import Room, new_code
from .solver import get_solver

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")
INDEX = os.path.join(STATIC_DIR, "index.html")
LAB = os.path.join(STATIC_DIR, "lab", "index.html")
WORLD = os.path.join(STATIC_DIR, "world.html")
ROOM_TTL_S = 3 * 3600
NAME_RE = re.compile(r"[^\w \-\.]+")
CODE_RE = re.compile(r"^[A-Z0-9]{4}$")
WORLD_PAGE = 24
WORLD_PAGE_MAX = 48
# Reserved for BOU-31 (bot filler). The worker is not in this MVP; the knob is documented only.
WORLD_FILL_N = int(os.environ.get("WH_WORLD_FILL_N", "0") or 0)
_STATE_RANK = {"playing": 0, "countdown": 1, "results": 2, "lobby": 3}

app = FastAPI(title="Word Hunt VS")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

rooms: dict[str, Room] = {}
STARTED_AT = time.time()
INSTANCE_ID = os.environ.get("K_REVISION", "local") + "/" + "".join(random.choice("0123456789abcdef") for _ in range(6))
players: dict[str, str] = {}          # player_id -> display name (passkey can attach here later)
_rng = random.Random()


def clean_name(name: str, player_id: str) -> str:
    name = NAME_RE.sub("", (name or "")).strip()[:20]
    return name or f"Player-{player_id[:4]}"


def clean_id(pid: str) -> str:
    pid = re.sub(r"[^a-zA-Z0-9_\-]", "", pid or "")[:40]
    return pid or "anon" + "".join(_rng.choice("0123456789abcdef") for _ in range(8))


def gc_rooms() -> None:
    now = time.time()
    for code in list(rooms):
        r = rooms[code]
        if r.humans_connected == 0 and now - r.created_at > ROOM_TTL_S:
            r._cancel_task()
            del rooms[code]
            try:
                asyncio.get_running_loop().create_task(room_store().delete(code))
            except RuntimeError:
                pass


BUILD = os.environ.get("WH_BUILD", "dev")


@app.on_event("startup")
async def rehydrate_rooms() -> None:
    """Bring back rooms saved by the previous process/revision (deploys wipe memory)."""
    docs = await room_store().load_all()
    solver = get_solver()
    for d in docs:
        try:
            rooms[d["code"]] = Room.from_doc(d, solver)
        except Exception as e:
            __import__("logging").getLogger("wordhunt.server").warning("rehydrate %s failed: %s", d.get("code"), e)
    if docs:
        __import__("logging").getLogger("wordhunt.server").info("rehydrated %d rooms from %s", len(rooms), room_store().url)


def _respan_status() -> dict:
    try:
        from integrations import respan
        return respan.status()
    except Exception:  # noqa: BLE001 - health must never fail on an optional module
        return {"enabled": False}


def _proc_stats() -> dict:
    """Process CPU seconds + RSS so a load test can derive utilisation from two samples."""
    out = {"cpu_s": round(time.process_time(), 2), "uptime_s": round(time.time() - STARTED_AT, 1)}
    try:
        import resource
        out["rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    except Exception:
        pass
    return out



@app.get("/api/health")
async def healthz():
    s = get_solver()
    now = time.time()
    from .seats import registry
    return {"ok": True, "words": len(s.words), "build": BUILD, "rooms": len(rooms), "proc": _proc_stats(), "tuning": registry.tuning(),
            "uptime_s": int(now - STARTED_AT), "instance_id": INSTANCE_ID,
            "store": {"url": room_store().url, **room_store().stats},
            "lineup": [x.id for x in registry.lineup()], "respan": _respan_status(),
            "room_list": [{"code": r.code, "state": r.state, "round": r.round_no, "humans": r.humans_connected,
                           "seats": len(r.seats), "age_s": int(now - r.created_at)} for r in rooms.values()]}


@app.get("/api/build")
async def build_info():
    return {"build": BUILD}


@app.get("/")
async def index():
    return FileResponse(INDEX)


@app.get("/lab")
async def lab_page():
    return FileResponse(LAB)


@app.get("/world")
async def world_page():
    return FileResponse(WORLD)


def _query_int(raw: str | None, default: int) -> int:
    try:
        return int(raw) if raw is not None and raw != "" else default
    except (TypeError, ValueError):
        return default


def world_index(now: float | None = None, offset: int = 0, limit: int = WORLD_PAGE,
                state: str = "", quiet: str | None = None) -> dict:
    """Paginated cheap cards from the in-memory room dict. No Redis / World WS."""
    now = time.time() if now is None else now
    offset = max(0, offset)
    limit = max(1, min(limit, WORLD_PAGE_MAX))
    state = (state or "").strip().lower()
    cards = []
    for r in rooms.values():
        if state and r.state != state:
            continue
        if quiet == "1" and not r.quiet:
            continue
        if quiet == "0" and r.quiet:
            continue
        cards.append(r.world_card(now))
    cards.sort(key=lambda c: (_STATE_RANK.get(c["state"], 9), -c["humans"], -c["top_score"], c["code"]))
    total = len(cards)
    page = cards[offset:offset + limit]
    return {
        "ok": True,
        "rooms": page,
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": offset + limit < total,
        "filler": {"target": WORLD_FILL_N, "enabled": False},
    }


@app.get("/api/world")
async def api_world(request: Request):
    q = request.query_params
    return world_index(
        offset=_query_int(q.get("offset"), 0),
        limit=_query_int(q.get("limit"), WORLD_PAGE),
        state=q.get("state") or "",
        quiet=q.get("quiet"),
    )


@app.get("/r/{code}")
async def room_page(code: str):
    return FileResponse(INDEX)


@app.get("/s/{code}")
async def spectator_page(code: str):
    """Projector / spectator view: same page, no seat."""
    return FileResponse(INDEX)


@app.post("/api/rooms")
async def create_room(request: Request):
    """Create a room. Body {"quiet": true} suppresses Discord/Nango posts for this room (worker tests)."""
    gc_rooms()
    quiet = False
    try:
        body = await request.json()
        quiet = bool(body.get("quiet"))
    except Exception:
        pass
    quiet = quiet or request.query_params.get("quiet") == "1"
    solver = get_solver()
    for _ in range(50):
        code = new_code(_rng)
        if code not in rooms:
            break
    rooms[code] = Room(code, solver, random.Random(_rng.random()), quiet=quiet)
    return {"code": code, "quiet": quiet}


@app.get("/api/rooms/{code}")
async def room_info(code: str):
    r = rooms.get(code.upper())
    if not r:
        return JSONResponse({"error": "no such room"}, status_code=404)
    return r.snapshot()


@app.websocket("/ws/{code}")
async def ws_room(ws: WebSocket, code: str):
    code = code.upper()
    room = rooms.get(code)
    await ws.accept()
    if room is None and CODE_RE.match(code):
        doc = await room_store().load(code)
        if doc:
            room = Room.from_doc(doc, get_solver())
            rooms[code] = room
    if room is None and CODE_RE.match(code):
        # Unknown but well-formed code (typically a link shared before a deploy reset the rooms):
        # recreate a fresh lobby under the same code so the shared link keeps working.
        gc_rooms()
        room = Room(code, get_solver(), random.Random(_rng.random()), quiet=ws.query_params.get("quiet") == "1")
        room.recreated = True
        rooms[code] = room
    if room is None:
        await ws.send_json({"type": "error", "error": "no such room", "code": code, "reason": "not_found", "fatal": True})
        await ws.close()
        return
    seat = None
    spectator = False
    try:
        while True:
            msg = await ws.receive_json()
            t = msg.get("type")
            if t == "hello" and msg.get("role") == "spectator":
                spectator = True
                room.spectators.add(ws)
                await ws.send_json({"type": "welcome", "role": "spectator"})
                await ws.send_json(room.snapshot())
            elif t == "hello":
                pid = clean_id(msg.get("player_id", ""))
                name = clean_name(msg.get("name", ""), pid)
                players[pid] = name
                seat = room.add_human(pid, name, ws)
                if seat is None:
                    await ws.send_json({"type": "error", "error": f"Room is full ({room.n_humans} players)", "code": code, "reason": "full", "fatal": True})
                    await ws.close()
                    return
                await ws.send_json({"type": "welcome", "player_id": pid, "seat_id": seat.seat_id, "name": seat.name})
                await room.send_state()
            elif spectator:
                if t == "ping":
                    await ws.send_json({"type": "pong", "now": time.time()})
                elif t == "integration":
                    # Client-side commentator (Mac-side Gemma via Nango) reports its tool call; rebroadcast so
                    # every phone and the projector toast it, same shape as the server-side handler emits.
                    allowed = ("provider", "action", "event", "ok", "ms", "round", "by", "label", "error", "dry_run", "text", "model", "caller")
                    payload = {"type": "integration", **{k: msg[k] for k in allowed if k in msg}}
                    payload.setdefault("provider", "nango")
                    payload.setdefault("by", "commentator")
                    payload.setdefault("round", room.round_no)
                    payload["code"] = room.code
                    await room.broadcast(payload)
            elif seat is None:
                await ws.send_json({"type": "error", "error": "say hello first"})
            elif t == "name":
                seat.name = clean_name(msg.get("name", ""), seat.seat_id)
                players[seat.seat_id] = seat.name
                await room.send_state()
            elif t == "start":
                if room.can_control(seat.seat_id) and room.state == "lobby":
                    await room.start()
            elif t == "rematch":
                if room.can_control(seat.seat_id) and room.state == "results":
                    await room.start()
            elif t == "add_seat":
                if room.can_control(seat.seat_id) and room.state in ("lobby", "results"):
                    room.add_from_catalog(str(msg.get("spec_id", ""))[:40])
                    await room.send_state()
            elif t == "remove_seat":
                sid = str(msg.get("seat_id", ""))
                target = room.seats.get(sid)
                if target and target.kind == "ai" and room.can_control(seat.seat_id) and room.state in ("lobby", "results"):
                    room.remove_seat(sid)
                    await room.send_state()
            elif t == "submit":
                path = [int(x) for x in msg.get("path", [])][:8]
                res = await room.human_submit(seat, path)
                await ws.send_json({"type": "mine", **res})
            elif t == "path":
                path = [int(x) for x in msg.get("path", [])][:8]
                await room.human_path(seat, path)
            elif t == "ping":
                await ws.send_json({"type": "pong", "now": time.time()})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        room.drop_socket(ws)
        if room.state == "lobby" and seat is not None and not seat.sockets:
            room.remove_seat(seat.seat_id)
            if room.host_id == seat.seat_id:
                room.host_id = next((s.seat_id for s in room.seats.values() if s.kind == "human" and s.sockets), None)
        try:
            await room.send_state()
        except Exception:
            pass
