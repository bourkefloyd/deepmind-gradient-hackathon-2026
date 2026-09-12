"""Message templates for game events -> Discord text (markdown).

Every message names the room code, because v1 posts everything to one channel (no threads).
Inputs are plain dicts built by `handlers.py` from `wordhunt.room.Room`, so these can be
unit-tested and re-skinned without a game running.

Payload shapes (all keys optional except code):
    room_created: {code, url, seats: [{name, kind, label}], countdown_s, race_s}
    round_started: {code, round, race_s, seats: [{name, kind, label}]}
    round_ended:   {code, round, url, seats: [{name, kind, label, score, n_words,
                    words: [[word, pts], ...], invalid: int, best_word, best_pts}],
                    board_best: [word, ...], max_score, n_board_words}
"""
from __future__ import annotations

SEAT_ICON = {"human": "🧑", "ai": "🤖"}
MEDALS = ["🥇", "🥈", "🥉"]


def _seat_name(s: dict) -> str:
    label = s.get("label") or ""
    icon = SEAT_ICON.get(s.get("kind", ""), "•")
    return f"{icon} **{s.get('name', '?')}**" + (f" _({label})_" if label else "")


def _seat_list(seats: list[dict]) -> str:
    return ", ".join(_seat_name(s) for s in seats) if seats else "_no seats yet_"


def room_created(p: dict) -> str:
    code = p["code"]
    lines = [
        f"🎮 **Room {code} created** — join at {p.get('url', '')}",
        f"Seats: {_seat_list(p.get('seats', []))}",
    ]
    if p.get("race_s"):
        lines.append(f"Format: {int(p['race_s'])} s race, same board for everyone, {int(p.get('countdown_s', 0))} s countdown.")
    return "\n".join(lines)


def round_started(p: dict) -> str:
    code, rnd = p["code"], p.get("round", 1)
    return "\n".join([
        f"🏁 **Room {code} · round {rnd} started** — board hidden, {int(p.get('race_s', 75))} s, go!",
        f"Racing: {_seat_list(p.get('seats', []))}",
    ])


def round_ended(p: dict) -> str:
    code, rnd = p["code"], p.get("round", 1)
    seats = sorted(p.get("seats", []), key=lambda s: -int(s.get("score", 0)))
    lines = [f"🔔 **Room {code} · round {rnd} results**"]
    for i, s in enumerate(seats):
        medal = MEDALS[i] if i < len(MEDALS) else f"{i + 1}."
        best = f"best `{s['best_word'].upper()}` +{s['best_pts']}" if s.get("best_word") else "no words"
        inv = int(s.get("invalid", 0))
        inv_txt = f", {inv} invalid" if inv else ""
        lines.append(f"{medal} {_seat_name(s)} — **{int(s.get('score', 0))}** ({int(s.get('n_words', 0))} words, {best}{inv_txt})")
    if p.get("board_best"):
        lines.append(f"Board had {p.get('n_board_words', '?')} words, max {p.get('max_score', '?')}; longest: "
                     + " ".join(f"`{w.upper()}`" for w in p["board_best"][:6]))
    if p.get("url"):
        lines.append(f"Rematch: {p['url']}")
    return "\n".join(lines)


def round_ended_short(p: dict) -> str:
    """One-liner (v2: posted in the room thread)."""
    seats = sorted(p.get("seats", []), key=lambda s: -int(s.get("score", 0)))
    top = f"{seats[0]['name']} {int(seats[0].get('score', 0))}" if seats else "nobody"
    return f"Round {p.get('round', 1)} over in room {p['code']}: {top} wins."


FORMATTERS = {
    "room_created": room_created,
    "round_started": round_started,
    "round_ended": round_ended,
}
