"""Wire game events to Discord: build a payload from a Room, format it, send via Nango.

`from_room(room)` is the only place that knows about `wordhunt.room.Room`; the hook in
room.py is `emit(event, from_room(self))`. Formatters and transport stay game-agnostic.
"""
from __future__ import annotations

import logging
from typing import Any

from . import config, discord, events, formatters

log = logging.getLogger("integrations.handlers")

_settings: config.Settings | None = None


def settings() -> config.Settings:
    global _settings
    if _settings is None:
        _settings = config.load()
    return _settings


def from_room(room: Any) -> dict[str, Any]:
    """Snapshot the bits of a Room the formatters need (plain dicts, no references)."""
    s = settings()
    seats = []
    for seat in room.seats.values():
        found: dict[str, int] = dict(getattr(seat, "found", {}) or {})
        words = sorted(found.items(), key=lambda kv: (-kv[1], kv[0]))
        invalid = sum(1 for t in room.ticker if t.get("seat_id") == seat.seat_id and not t.get("ok") and t.get("reason") == "miss")
        d = {
            "name": seat.name, "kind": seat.kind, "label": getattr(seat, "label", ""),
            "score": sum(found.values()), "n_words": len(found), "words": words[:12], "invalid": invalid,
        }
        if words:
            d["best_word"], d["best_pts"] = words[0]
        seats.append(d)
    p: dict[str, Any] = {
        "code": room.code, "round": room.round_no, "url": s.room_url(room.code), "seats": seats,
        "countdown_s": getattr(room, "countdown_s", None) or _module_const(room, "COUNTDOWN_S", 20),
        "race_s": getattr(room, "race_s", None) or _module_const(room, "RACE_S", 75),
        "quiet": bool(getattr(room, "quiet", False)),     # POST /api/rooms {quiet: true}: no Discord posts
    }
    level = getattr(room, "level", None)
    if level is not None:
        p["level"] = level.public()
    if getattr(room, "results", None):
        r0 = room.results[0]
        p["board_best"] = r0.get("best_words", [])
        p["max_score"] = r0.get("max_score")
        p["n_board_words"] = r0.get("n_board_words")
    return p


def from_snapshot(snap: dict[str, Any], public_url: str = "") -> dict[str, Any]:
    """Same payload as from_room, built from a `state` WebSocket message in the `results` phase
    (seats carry `words` only then). Used by the client-side commentator."""
    s = settings()
    code = snap.get("code", "")
    ticker = snap.get("ticker", [])
    seats = []
    for seat in snap.get("seats", []):
        words = [list(kv) for kv in seat.get("words", [])]
        words.sort(key=lambda kv: (-int(kv[1]), kv[0]))
        sid = seat.get("seat_id")
        invalid = sum(1 for t in ticker if t.get("seat_id") == sid and not t.get("ok") and t.get("reason") == "miss")
        d = {
            "name": seat.get("name", "?"), "kind": seat.get("kind", ""), "label": seat.get("label", ""),
            "score": int(seat.get("score", 0)), "n_words": int(seat.get("n_words", len(words))),
            "words": words[:12], "invalid": invalid,
        }
        if words:
            d["best_word"], d["best_pts"] = words[0][0], int(words[0][1])
        seats.append(d)
    url = (public_url.rstrip("/") + f"/r/{code}") if public_url else s.room_url(code)
    p: dict[str, Any] = {
        "code": code, "round": snap.get("round", 0), "url": url, "seats": seats,
        "countdown_s": snap.get("countdown_s", 20), "race_s": snap.get("race_s", 75),
        "quiet": bool(snap.get("quiet", False)),
    }
    if snap.get("level"):
        p["level"] = dict(snap["level"])
    if snap.get("results"):
        r0 = snap["results"][0]
        p["board_best"] = r0.get("best_words", [])
        p["max_score"] = r0.get("max_score")
        p["n_board_words"] = r0.get("n_board_words")
    return p


def _module_const(obj: Any, name: str, default: float) -> float:
    import sys
    mod = sys.modules.get(type(obj).__module__)
    return getattr(mod, name, default)


async def discord_recap(event_type: str, payload: dict[str, Any]) -> None:
    fmt = formatters.FORMATTERS.get(event_type)
    if fmt is None:
        return
    if payload.get("quiet"):
        log.info("%s: room %s is quiet, not posting", event_type, payload.get("code"))
        return
    text = fmt(payload)
    res = await discord.send_recap(text)
    if settings().dry_run:
        log.info("[dry-run] %s -> would post to Discord via %s:\n%s", event_type, settings().recap_action, text)
    else:
        log.info("%s -> posted via %s (%s)", event_type, settings().recap_action, _short(res))


def _short(res: Any) -> str:
    if isinstance(res, dict):
        return ", ".join(f"{k}={res[k]}" for k in ("id", "channelId", "ok", "status") if k in res) or "ok"
    return str(res)[:80]


def register() -> None:
    """Idempotent: install the v1 handler for every formatted event."""
    events._installed = True
    for ev in formatters.FORMATTERS:
        events.on(ev, discord_recap)
