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

app = FastAPI(title="Word Hunt arena")
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


@app.get("/healthz")
async def healthz():
    s = get_solver()
    return {"ok": True, "words": len(s.words), "rooms": len(rooms)}


@app.get("/")
async def index():
    return FileResponse(INDEX)


@app.get("/r/{code}")
async def room_page(code: str):
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
    if room is None:
        await ws.send_json({"type": "error", "error": "no such room", "fatal": True})
        await ws.close()
        return
    seat = None
    try:
        while True:
            msg = await ws.receive_json()
            t = msg.get("type")
            if t == "hello":
                pid = clean_id(msg.get("player_id", ""))
                name = clean_name(msg.get("name", ""), pid)
                players[pid] = name
                seat = room.add_human(pid, name, ws)
                await ws.send_json({"type": "welcome", "player_id": pid, "seat_id": seat.seat_id, "name": seat.name})
                await room.send_state()
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
