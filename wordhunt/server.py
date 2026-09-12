"""FastAPI app: static page, room API, WebSocket room. Single process, in-memory rooms."""
from __future__ import annotations

import os
import random
import re
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .room import Room, new_code
from .solver import get_solver

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")
INDEX = os.path.join(STATIC_DIR, "index.html")
ROOM_TTL_S = 3 * 3600
NAME_RE = re.compile(r"[^\w \-\.]+")
CODE_RE = re.compile(r"^[A-Z0-9]{4}$")

app = FastAPI(title="Word Hunt VS")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

rooms: dict[str, Room] = {}
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


BUILD = os.environ.get("WH_BUILD", "dev")


@app.get("/api/health")
async def healthz():
    s = get_solver()
    now = time.time()
    from .seats import registry
    return {"ok": True, "words": len(s.words), "build": BUILD, "rooms": len(rooms), "tuning": registry.tuning(),
            "lineup": [x.id for x in registry.lineup()],
            "room_list": [{"code": r.code, "state": r.state, "round": r.round_no, "humans": r.humans_connected,
                           "seats": len(r.seats), "age_s": int(now - r.created_at)} for r in rooms.values()]}


@app.get("/api/build")
async def build_info():
    return {"build": BUILD}


@app.get("/")
async def index():
    return FileResponse(INDEX)


@app.get("/r/{code}")
async def room_page(code: str):
    return FileResponse(INDEX)


@app.get("/s/{code}")
async def spectator_page(code: str):
    """Projector / spectator view: same page, no seat."""
    return FileResponse(INDEX)


@app.post("/api/rooms")
async def create_room():
    gc_rooms()
    solver = get_solver()
    for _ in range(50):
        code = new_code(_rng)
        if code not in rooms:
            break
    rooms[code] = Room(code, solver, random.Random(_rng.random()))
    return {"code": code}


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
        # Unknown but well-formed code (typically a link shared before a deploy reset the rooms):
        # recreate a fresh lobby under the same code so the shared link keeps working.
        gc_rooms()
        room = Room(code, get_solver(), random.Random(_rng.random()))
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
