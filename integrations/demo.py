"""Simulate the three game events with no game running.

    python -m integrations.demo                 # dry-run: prints the would-be Nango payloads
    python -m integrations.demo --live          # env set: fires ONE real send-discord-recap
    python -m integrations.demo --connections   # list Nango connections for the integration
    python -m integrations.demo --tool          # show the commentator tool schema + a fake tool call

Dry-run is automatic whenever NANGO_SECRET_KEY is unset.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from . import config, discord, events, formatters, handlers, nango, tools

SEATS = [
    {"name": "Bourke", "kind": "human", "label": ""},
    {"name": "Nano", "kind": "ai", "label": "nano 10M"},
    {"name": "Gemma 12B", "kind": "ai", "label": "gemma-4-12b"},
    {"name": "Reflex", "kind": "ai", "label": "heuristic"},
]


def sample_payloads(code: str = "ABCD", url: str = "") -> dict[str, dict]:
    s = config.load()
    url = url or s.room_url(code)
    ended_seats = [
        {**SEATS[0], "score": 4200, "n_words": 11, "best_word": "lantern", "best_pts": 1800, "invalid": 1,
         "words": [["lantern", 1800], ["antler", 1400], ["learn", 800]]},
        {**SEATS[1], "score": 3400, "n_words": 14, "best_word": "rental", "best_pts": 1400, "invalid": 0,
         "words": [["rental", 1400], ["tern", 400], ["ant", 100]]},
        {**SEATS[2], "score": 2900, "n_words": 6, "best_word": "eternal", "best_pts": 1800, "invalid": 3,
         "words": [["eternal", 1800], ["neat", 400]]},
        {**SEATS[3], "score": 1500, "n_words": 7, "best_word": "rant", "best_pts": 400, "invalid": 0,
         "words": [["rant", 400], ["tan", 100]]},
    ]
    return {
        "room_created": {"code": code, "url": url, "seats": SEATS, "countdown_s": 20, "race_s": 75},
        "round_started": {"code": code, "round": 1, "race_s": 75, "seats": SEATS},
        "round_ended": {"code": code, "round": 1, "url": url, "seats": ended_seats,
                        "board_best": ["eternal", "lantern", "antler", "rental"], "max_score": 41200, "n_board_words": 96},
    }


async def run_events(code: str) -> None:
    for ev, payload in sample_payloads(code).items():
        print(f"\n=== emit({ev!r})")
        n = events.emit(ev, payload)
        await events.drain()
        print(f"({n} handler(s) ran)")


async def run_live(code: str) -> int:
    s = config.load()
    missing = s.missing()
    if missing:
        print(f"cannot go live, missing env: {', '.join(missing)}", file=sys.stderr)
        return 2
    text = formatters.round_ended(sample_payloads(code)["round_ended"])
    print(f"POST {s.nango_base_url}/action/trigger  action={s.recap_action}  integration={s.integration_id}  connection={s.connection_id}")
    print(text)
    try:
        res = await discord.send_recap(text)
    except nango.NangoError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1
    print("OK:", json.dumps(res, indent=2, ensure_ascii=False)[:1500])
    return 0


def run_connections() -> int:
    s = config.load()
    if s.dry_run:
        print("NANGO_SECRET_KEY is unset; cannot list connections", file=sys.stderr)
        return 2
    try:
        conns = nango.client().list_connections(s.integration_id)
    except nango.NangoError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1
    if not conns:
        print(f"no connections for integration {s.integration_id!r}")
        return 1
    print(f"{len(conns)} connection(s) for {s.integration_id!r}:")
    for c in conns:
        err = f"  errors={c['errors']}" if c.get("errors") else ""
        print(f"  connection_id={c['connection_id']}  provider={c.get('provider')}  created={c.get('created')}{err}")
    if len(conns) == 1:
        print(f"\nexport NANGO_CONNECTION_ID={conns[0]['connection_id']}")
    return 0


async def run_tool(code: str) -> None:
    print(json.dumps(tools.TOOLS, indent=2))
    fake_call = {"id": "call_0", "type": "function", "function": {
        "name": "send_discord_recap",
        "arguments": json.dumps({"text": f"Room {code}: Bourke 4200 takes it with LANTERN. Gemma tried 'CTA' three times. Nano learning."}),
    }}
    print("\n=== dispatch(fake tool call)")
    print(json.dumps(await tools.dispatch([fake_call]), indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--live", action="store_true", help="fire one real send-discord-recap (needs env)")
    p.add_argument("--connections", action="store_true", help="list Nango connections for the integration")
    p.add_argument("--tool", action="store_true", help="print the commentator tool schema and dispatch a fake call")
    p.add_argument("--code", default="ABCD")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    s = config.load()
    mode = "DRY-RUN (NANGO_SECRET_KEY unset)" if s.dry_run else "LIVE"
    print(f"integrations demo: {mode}; integration={s.integration_id} connection={s.connection_id or '<unset>'} "
          f"action={s.recap_action} public_url={s.public_url}")
    if a.connections:
        return run_connections()
    if a.live:
        return asyncio.run(run_live(a.code))
    if a.tool:
        asyncio.run(run_tool(a.code))
        return 0
    handlers.register()
    asyncio.run(run_events(a.code))
    return 0


if __name__ == "__main__":
    sys.exit(main())
