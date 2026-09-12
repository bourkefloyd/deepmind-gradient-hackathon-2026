"""Gemma commentator: the model function-calls Nango at the buzzer.

Given a round_ended payload (the dict `handlers.from_room` / `from_snapshot` produce) and an
OpenAI-compatible endpoint, make ONE chat call with the `send_discord_recap` tool available.
If Gemma calls the tool within the deadline, `tools.dispatch()` triggers the Nango action with
the model's text. Otherwise the templated recap is posted so Discord never goes quiet.

    # in-process (server side, when the model endpoint is reachable from the game)
    await commentate(payload)

    # client side: Gemma on the Mac (mlx-vlm), game on Cloud Run
    python -m integrations.commentator --room ABCD --server https://wordhunt-....run.app
    python -m integrations.commentator --once            # one call on the demo payload

Env: GEMMA_BASE_URL (default http://localhost:8080/v1), GEMMA_MODEL
(default mlx-community/gemma-4-12B-it-4bit), GEMMA_API_KEY (optional), plus the Nango vars
from config.py. RESPAN_ENABLED=1 + RESPAN_API_KEY routes the call through the Respan gateway
tagged `commentator` (integrations/respan.py). Thinking is off: `reasoning_effort: "none"` and `enable_thinking: false`
(the same two knobs gemma_seat/client.py uses for mlx-vlm).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from . import config, discord, formatters, handlers, respan, tools

log = logging.getLogger("integrations.commentator")

DEFAULT_BASE_URL = os.environ.get("GEMMA_BASE_URL", os.environ.get("MLXVLM_BASE_URL", "http://localhost:8080/v1"))
DEFAULT_MODEL = os.environ.get("GEMMA_MODEL", os.environ.get("MLXVLM_MODEL", "mlx-community/gemma-4-12B-it-4bit"))
TOOL_DEADLINE_S = float(os.environ.get("GEMMA_TOOL_DEADLINE_S", "10"))

SYSTEM_PROMPT = (
    "You are the live commentator for a 75-second Word Hunt race between humans, a 10-million-"
    "parameter 'nano' model that plays on a CPU with zero tokens, and Gemma seats. A round just "
    "ended and you have the scores.\n"
    "Write the recap and POST IT by calling the send_discord_recap tool exactly once. Rules for "
    "the text: 2-4 short lines, Discord markdown ok, start with the room code, name the winner "
    "and their score, roast any seat that submitted invalid words (name the seat and the count), "
    "and mention the nano by its parameter count when it plays. Trash talk, not cruelty. "
    "No preamble outside the tool call."
)


@dataclass
class Outcome:
    posted_by: str                      # "model" | "template" | "none"
    text: str
    tool_call: dict[str, Any] | None = None
    model_latency_s: float = 0.0
    error: str = ""
    nango_result: Any = None
    raw_response: dict[str, Any] = field(default_factory=dict, repr=False)


def model_label(model: str) -> str:
    m = model.lower()
    for key, label in (("31b", "Gemma 31B"), ("27b", "Gemma 27B"), ("12b", "Gemma 12B"), ("e4b", "Gemma E4B"), ("e2b", "Gemma E2B")):
        if key in m:
            return label
    return model.rsplit("/", 1)[-1] or "Gemma"


def decorate_model_text(text: str, model: str, latency_s: float) -> str:
    """The Discord post itself is the proof: who wrote it, and that it arrived through a Nango tool call."""
    return (f"🤖 **{model_label(model)}** (via Nango tool call)\n{text.strip()}\n"
            f"via Nango → Discord · model chose the tool in {latency_s:.1f} s")


def decorate_template_text(text: str) -> str:
    text = text.rstrip()
    if text.endswith(formatters.FOOTER):          # swap the server footer for the fallback one
        text = text[: -len(formatters.FOOTER)].rstrip()
    return text + "\n\nvia Nango → Discord · templated"


def user_prompt(payload: dict[str, Any]) -> str:
    seats = sorted(payload.get("seats", []), key=lambda s: -int(s.get("score", 0)))
    lines = [f"Room {payload.get('code')} round {payload.get('round')} is over. Standings:"]
    for i, s in enumerate(seats, 1):
        label = f" [{s['label']}]" if s.get("label") else ""
        best = f", best {s['best_word'].upper()} +{s['best_pts']}" if s.get("best_word") else ""
        inv = f", {s.get('invalid', 0)} invalid submissions" if s.get("invalid") else ""
        lines.append(f"{i}. {s.get('name')}{label} ({s.get('kind')}): {s.get('score', 0)} pts, {s.get('n_words', 0)} words{best}{inv}")
    if payload.get("board_best"):
        lines.append("Longest words on the board: " + ", ".join(w.upper() for w in payload["board_best"][:5]))
    lines.append("Now call send_discord_recap with your recap.")
    return "\n".join(lines)


def build_request(payload: dict[str, Any], model: str, max_tokens: int = 300) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(payload)},
        ],
        "tools": tools.TOOLS,
        "tool_choice": "auto",
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "reasoning_effort": "none",
        "enable_thinking": False,
    }


def _chat(base_url: str, body: dict[str, Any], timeout_s: float, api_key: str = "",
          room: str = "", round_no: Any = None) -> dict[str, Any]:
    # RESPAN_ENABLED=1: the call goes to the Respan gateway tagged `commentator` (thread = room),
    # with the Gemma model name and a credential_override to GEMMA_BASE_URL. Otherwise straight to Gemma.
    r = respan.route("commentator", base_url, api_key, body["model"], thread=room or None,
                     metadata={"where": "commentator", "room": room or None, "round": round_no, "tool": "send_discord_recap"})
    if r.via_respan:
        body = {**body, "model": r.model, **r.body_extra}
        log.info("routing via %s", r.describe())
    req = urllib.request.Request(
        r.completions_url(),
        data=json.dumps(body).encode(),
        method="POST",
        headers={**r.auth_headers(), **({} if r.api_key else {"Authorization": "Bearer none"})},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = json.loads(resp.read())
    if respan.log_mode():
        # Direct call done; ship it to Respan as a span (tool call rendered as text so the trace shows what Gemma said).
        msg = ((data.get("choices") or [{}])[0].get("message") or {})
        out = msg.get("content") or "".join(json.dumps(tc.get("function", {})) for tc in msg.get("tool_calls") or [])
        res = respan.log_request("commentator", model=body["model"], messages=body["messages"], completion=out or "",
                                 latency_s=time.perf_counter() - t0, usage=data.get("usage"), thread=room or None,
                                 upstream_base_url=base_url, timeout_s=5.0,
                                 metadata={"where": "commentator", "room": room or None, "round": round_no, "tool": "send_discord_recap",
                                           "tool_called": bool(msg.get("tool_calls"))})
        log.info("respan log: ok=%s status=%s unique_id=%s", res["ok"], res["status"],
                 res["body"].get("unique_id") if isinstance(res["body"], dict) else "-")
    return data


def extract_tool_call(resp: dict[str, Any]) -> dict[str, Any] | None:
    """First `send_discord_recap` call in an OpenAI chat response, or None."""
    try:
        msg = resp["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        if fn.get("name") == "send_discord_recap":
            return tc
    return None


async def commentate(payload: dict[str, Any], base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL,
                     deadline_s: float = TOOL_DEADLINE_S, fallback: bool = True, api_key: str = "") -> Outcome:
    """One model call with the tool; dispatch the call; template fallback after `deadline_s`."""
    body = build_request(payload, model)
    t0 = time.perf_counter()
    resp: dict[str, Any] = {}
    error = ""
    try:
        resp = await asyncio.wait_for(asyncio.to_thread(_chat, base_url, body, deadline_s, api_key or os.environ.get("GEMMA_API_KEY", ""),
                                                        str(payload.get("code") or ""), payload.get("round")),
                                      timeout=deadline_s + 0.5)
    except asyncio.TimeoutError:
        error = f"model did not answer within {deadline_s:.0f}s"
    except urllib.error.HTTPError as e:
        error = f"model HTTP {e.code}: {e.read()[:200]!r}"
    except Exception as e:  # noqa: BLE001
        error = f"model call failed: {type(e).__name__}: {e}"
    latency = time.perf_counter() - t0

    tc = extract_tool_call(resp) if not error else None
    if tc is not None:
        log.info("MODEL CALLED NANGO TOOL (%.1fs): %s", latency, json.dumps(tc, ensure_ascii=False))
        try:
            args = json.loads(tc["function"].get("arguments") or "{}")
        except Exception:  # noqa: BLE001
            args = {}
        text = decorate_model_text(str(args.get("text", "")), model, latency)
        # Same call the model made, with the attribution header/footer wrapped around its text.
        posted_call = {**tc, "function": {**tc["function"], "arguments": json.dumps({**args, "text": text}, ensure_ascii=False)}}
        t1 = time.perf_counter()
        results = await tools.dispatch([posted_call])
        r = results[0]
        # In-process only (payload from handlers.from_room carries the room's broadcast); the
        # Mac-side spectator path has no way to push a toast into the room.
        await handlers.notify(payload, "commentator", ok="error" not in r, ms=int((time.perf_counter() - t1) * 1000),
                              result=r.get("result"), error=r.get("error", ""), by="gemma",
                              label=f"🤖 {model_label(model)} posted a recap via Nango")
        if "error" in r:
            log.warning("tool dispatch failed: %s", r["error"])
            return Outcome("none", text, tc, latency, r["error"], None, resp)
        return Outcome("model", text, tc, latency, "", r.get("result"), resp)

    content = ""
    try:
        content = (resp.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    except Exception:  # noqa: BLE001
        pass
    why = error or ("model answered with text but no tool call" + (f": {content[:120]!r}" if content else ""))
    log.warning("no tool call (%.1fs): %s", latency, why)
    if not fallback:
        return Outcome("none", "", None, latency, why, None, resp)
    text = decorate_template_text(formatters.round_ended(payload))
    try:
        res = await discord.send_recap(text)
    except Exception as e:  # noqa: BLE001
        log.warning("template fallback failed too: %s", e)
        return Outcome("none", text, None, latency, f"{why}; fallback: {e}", None, resp)
    log.info("posted templated recap instead")
    return Outcome("template", text, None, latency, why, res, resp)


# ---- client side: spectate a room over WebSocket, commentate at results -------------------------
async def spectate(server: str, room: str, base_url: str, model: str, rounds: int = 0,
                   public_url: str = "", deadline_s: float = TOOL_DEADLINE_S) -> int:
    import websockets  # only needed for this path; `pip install websockets`

    ws_base = server.rstrip("/").replace("https://", "wss://").replace("http://", "ws://")
    url = f"{ws_base}/ws/{room.upper()}"
    public_url = public_url or server.rstrip("/").replace("wss://", "https://").replace("ws://", "http://")
    log.info("spectating %s (model %s at %s)", url, model, base_url)
    done = 0
    last_round_seen = -1
    async with websockets.connect(url, max_size=2**22) as ws:
        await ws.send(json.dumps({"type": "hello", "role": "spectator"}))
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            t = msg.get("type")
            if t == "welcome":
                log.info("seated as %s", msg.get("role"))
            elif t == "error":
                log.error("server error: %s", msg)
                if msg.get("fatal"):
                    return 1
            elif t == "state":
                st, rnd = msg.get("state"), int(msg.get("round", 0))
                if st == "results" and rnd != last_round_seen:
                    last_round_seen = rnd
                    payload = handlers.from_snapshot(msg, public_url)
                    if payload.get("quiet"):
                        log.info("round %d over in %s; room is quiet, skipping", rnd, payload["code"])
                        continue
                    log.info("round %d over in %s; asking %s", rnd, payload["code"], model)
                    out = await commentate(payload, base_url, model, deadline_s)
                    log.info("outcome: posted_by=%s latency=%.1fs%s\n%s", out.posted_by, out.model_latency_s,
                             f" ({out.error})" if out.error else "", out.text)
                    done += 1
                    if rounds and done >= rounds:
                        return 0
                elif st in ("countdown", "playing"):
                    log.debug("room %s: %s", msg.get("code"), st)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--room", help="room code to spectate")
    p.add_argument("--server", default="http://localhost:8000", help="game server base URL (http/https)")
    p.add_argument("--public-url", default="", help="join URL base for the recap (defaults to --server)")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible endpoint")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--deadline", type=float, default=TOOL_DEADLINE_S, help="seconds to wait for the tool call")
    p.add_argument("--rounds", type=int, default=0, help="leave after this many recaps (0 = stay)")
    p.add_argument("--once", action="store_true", help="one commentator call on the demo payload, no game")
    p.add_argument("--no-fallback", action="store_true", help="do not post the template if the model skips the tool")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    s = config.load()
    print(f"commentator: nango={'DRY-RUN' if s.dry_run else 'LIVE'} integration={s.integration_id} "
          f"connection={s.connection_id or '<unset>'} model={a.model} endpoint={a.base_url}", flush=True)
    if a.once:
        from .demo import sample_payloads
        out = asyncio.run(commentate(sample_payloads()["round_ended"], a.base_url, a.model, a.deadline, not a.no_fallback))
        print(json.dumps({"posted_by": out.posted_by, "latency_s": round(out.model_latency_s, 2), "error": out.error,
                          "tool_call": out.tool_call, "text": out.text}, indent=2, ensure_ascii=False))
        return 0 if out.posted_by != "none" else 1
    if not a.room:
        p.error("--room is required unless --once")
    return asyncio.run(spectate(a.server, a.room, a.base_url, a.model, a.rounds, a.public_url, a.deadline))


if __name__ == "__main__":
    sys.exit(main())
