"""Message templates for game events -> Discord text (markdown).

Every message names the room code, because v1 posts everything to one channel (no threads).
Inputs are plain dicts built by `handlers.from_room` / `from_snapshot`, so these can be
unit-tested (`python -m integrations.test_formatters`) and re-skinned without a game running.

Payload shapes (all keys optional except code):
    room_created:  {code, url, seats: [{name, kind, label}], countdown_s, race_s}
    round_started: {code, round, race_s, seats: [{name, kind, label}]}
    round_ended:   {code, round, url, seats: [{name, kind, label, score, n_words,
                    words: [[word, pts], ...], invalid: int, best_word, best_pts}],
                    board_best: [word, ...], max_score, n_board_words, level: {n, theme}}
"""
from __future__ import annotations

SEAT_ICON = {"human": "🧑", "ai": "🤖"}
MEDALS = ["🥇", "🥈", "🥉"]
LONGEST_SHOWN = 6


def n(x) -> str:
    """Thousands separators: 4700 -> '4,700'."""
    try:
        return f"{int(x):,}"
    except (TypeError, ValueError):
        return str(x)


def seat_title(s: dict) -> str:
    """'🤖 **Reflex-B** · *heuristic*' / '🧑 **Player-5347**'."""
    icon = SEAT_ICON.get(s.get("kind", ""), "•")
    out = f"{icon} **{s.get('name', '?')}**"
    label = (s.get("label") or "").strip()
    if label:
        out += f" · *{label}*"
    return out


def _seat_inline(seats: list[dict]) -> str:
    return " · ".join(seat_title(s) for s in seats) if seats else "*no seats yet*"


def _ranked(seats: list[dict]) -> list[dict]:
    return sorted(seats, key=lambda s: (-int(s.get("score", 0)), -int(s.get("n_words", 0)), str(s.get("name", ""))))


def room_created(p: dict) -> str:
    code = p["code"]
    lines = [f"🎮 **Room {code}** is open 🎮", f"**Join →** {p.get('url', '')}"]
    if p.get("seats"):
        lines.append(f"Seats: {_seat_inline(p['seats'])}")
    if p.get("race_s"):
        lines.append(f"⏱️ {int(p['race_s'])} s race · same board for everyone · {int(p.get('countdown_s', 0))} s countdown")
    return "\n".join(lines)


def round_started(p: dict) -> str:
    code, rnd = p["code"], p.get("round", 1)
    return "\n".join([
        f"🏁 **Room {code} — Round {rnd}** is live 🏁",
        f"Board hidden · **{int(p.get('race_s', 75))} s** · go!",
        f"Racing: {_seat_inline(p.get('seats', []))}",
    ])


def seat_block(rank: int, s: dict) -> str:
    """One seat in the results post (rank is 1-based)."""
    medal = MEDALS[rank - 1] if rank <= len(MEDALS) else f"**{rank}.**"
    words_line = f"**{n(s.get('score', 0))} pts** · {n(s.get('n_words', 0))} words"
    inv = int(s.get("invalid", 0) or 0)
    if inv:
        words_line += f" · {inv} invalid"
    lines = [f"{medal} {seat_title(s)}", words_line]
    if s.get("best_word"):
        lines.append(f"Best: `{str(s['best_word']).upper()}` **+{n(s.get('best_pts', 0))}**")
    return "\n".join(lines)


def level_suffix(p: dict) -> str:
    """' — Level 3 · Kitchen' when the payload carries a themed level, else ''."""
    lvl = p.get("level") or {}
    if not lvl.get("theme"):
        return ""
    return f" — Level {lvl.get('n', '?')} · {lvl['theme']}"


def round_ended(p: dict) -> str:
    code, rnd = p["code"], p.get("round", 1)
    parts = [f"🎮 **Room {code} — Round {rnd} Results{level_suffix(p)}** 🎮"]
    for i, s in enumerate(_ranked(p.get("seats", [])), 1):
        parts.append(seat_block(i, s))
    stats = ["**Board Stats**"]
    if p.get("n_board_words") is not None:
        stats.append(f"📚 **{n(p['n_board_words'])}** possible words")
    if p.get("max_score") is not None:
        stats.append(f"💰 **{n(p['max_score'])}** max score")
    if p.get("board_best"):
        stats.append("🔠 Longest: " + " · ".join(f"`{w.upper()}`" for w in p["board_best"][:LONGEST_SHOWN]))
    if len(stats) > 1:
        parts.append("\n".join(stats))
    if p.get("url"):
        parts.append(f"⚔️ **Think you can beat the bots?**\n**Rematch →** {p['url']}")
    return "\n\n".join(parts)


def round_ended_short(p: dict) -> str:
    """One-liner (v2: posted in the room thread)."""
    seats = _ranked(p.get("seats", []))
    top = f"{seats[0]['name']} {n(seats[0].get('score', 0))}" if seats else "nobody"
    return f"Round {p.get('round', 1)} over in room {p['code']}: {top} wins."


FORMATTERS = {
    "room_created": room_created,
    "round_started": round_started,
    "round_ended": round_ended,
}
