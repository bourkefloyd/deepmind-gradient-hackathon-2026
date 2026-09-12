"""OpenAI-style tool schema for the Gemma commentator.

One tool, mapped 1:1 to the Nango action `send-discord-recap(text)`. At the buzzer the game
(or a commentator loop) gives Gemma the round results and this tool; if Gemma emits a
`send_discord_recap` tool call, `dispatch()` triggers the Nango action. The model calling
the tool is the point (PLAN.md section 5), so nothing here posts on its own.

Usage with any OpenAI-compatible client (vLLM `--tool-call-parser gemma4`, mlx-vlm, Respan):

    from integrations import tools
    resp = client.chat.completions.create(model=..., messages=tools.commentator_messages(payload),
                                          tools=tools.TOOLS, tool_choice="auto")
    results = await tools.dispatch(resp.choices[0].message.tool_calls)
"""
from __future__ import annotations

import json
import logging
from typing import Any

from . import discord, formatters

log = logging.getLogger("integrations.tools")

SEND_DISCORD_RECAP = {
    "type": "function",
    "function": {
        "name": "send_discord_recap",
        "description": (
            "Post a short recap message to the Word Hunt Discord channel via Nango. "
            "Use it once at the end of a round to announce the result, name the room code, "
            "call out the best word and any embarrassing invalid words. Max 2000 characters."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The message to post. Discord markdown allowed. Start with the room code.",
                }
            },
            "required": ["text"],
        },
    },
}

TOOLS = [SEND_DISCORD_RECAP]

SYSTEM_PROMPT = (
    "You are the commentator for a 75-second Word Hunt race between humans, a 10M-parameter nano "
    "model and Gemma seats. When a round ends you receive the results. Write one punchy recap "
    "(2-4 lines, a little trash talk, name the room code) and post it by calling the "
    "send_discord_recap tool. Call the tool exactly once."
)


def commentator_messages(round_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Chat messages that hand Gemma the results as text (same payload as formatters.round_ended)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Round results:\n" + formatters.round_ended(round_payload)
                                    + "\n\nRaw JSON:\n" + json.dumps(round_payload, ensure_ascii=False)},
    ]


async def call(name: str, arguments: dict[str, Any]) -> Any:
    if name == "send_discord_recap":
        return await discord.send_recap(str(arguments.get("text", "")))
    raise ValueError(f"unknown tool {name!r}")


async def dispatch(tool_calls: list[Any] | None) -> list[dict[str, Any]]:
    """Execute OpenAI tool_calls (objects or dicts). Returns [{tool_call_id, name, result|error}]."""
    out: list[dict[str, Any]] = []
    for tc in tool_calls or []:
        fn = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", None)
        name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", "")
        raw_args = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", "{}")
        tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
            res = await call(name, args)
            out.append({"tool_call_id": tc_id, "name": name, "result": res})
            log.info("tool %s ok", name)
        except Exception as e:  # noqa: BLE001
            out.append({"tool_call_id": tc_id, "name": name, "error": f"{type(e).__name__}: {e}"})
            log.warning("tool %s failed: %s", name, e)
    return out
