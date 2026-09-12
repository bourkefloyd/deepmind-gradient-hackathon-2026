"""Discord operations, all routed through Nango.

v1 (live today): one Nango action, `send-discord-recap`, input `{"text": str}`. It posts plain
text to the channel the action is configured for. Every game event goes through `send_recap`,
so the tool surface for the Gemma commentator (tools.py) is exactly this one function.

v2 (documented, not wired): room thread flow via the Nango proxy or Nango's prebuilt Discord
actions (`create-message`, `create-thread-from-message`). Those need a bot token in the
connection metadata (`botToken`) and DISCORD_CHANNEL_ID; see README "v2: threads".
"""
from __future__ import annotations

import logging
from typing import Any

from . import nango

log = logging.getLogger("integrations.discord")

DISCORD_MAX_LEN = 2000


def _clip(text: str, limit: int = DISCORD_MAX_LEN) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ---- v1 -------------------------------------------------------------------------------------
async def send_recap(text: str, client: nango.NangoClient | None = None) -> Any:
    """Trigger `send-discord-recap` with {text}. Returns the action output (or a dry-run stub)."""
    c = client or nango.client()
    return await c.atrigger_action(c.s.recap_action, {"text": _clip(text)})


def send_recap_sync(text: str, client: nango.NangoClient | None = None) -> Any:
    c = client or nango.client()
    return c.trigger_action(c.s.recap_action, {"text": _clip(text)})


# ---- v2 (threads) ---------------------------------------------------------------------------
# Discord REST paths; Nango's Discord provider proxies to https://discord.com so the path
# includes /api/v10. The OAuth user token Nango holds cannot post messages; pass the bot token
# explicitly (Nango forwards `nango-proxy-Authorization`), which is what Nango's own prebuilt
# Discord actions do internally.

def _bot_headers(bot_token: str) -> dict[str, str]:
    return {"Authorization": f"Bot {bot_token}"}


async def send_message(channel_id: str, content: str | None = None, embed: dict | None = None,
                       bot_token: str = "", client: nango.NangoClient | None = None) -> str:
    """POST /channels/{id}/messages -> message id."""
    c = client or nango.client()
    body: dict[str, Any] = {}
    if content:
        body["content"] = _clip(content)
    if embed:
        body["embeds"] = [embed]
    res = await c.aproxy("POST", f"/api/v10/channels/{channel_id}/messages", body, _bot_headers(bot_token))
    return str((res or {}).get("id", ""))


async def create_thread(channel_id: str, message_id: str, name: str, bot_token: str = "",
                        client: nango.NangoClient | None = None, auto_archive_minutes: int = 60) -> str:
    """POST /channels/{id}/messages/{mid}/threads -> thread id."""
    c = client or nango.client()
    body = {"name": name[:100], "auto_archive_duration": auto_archive_minutes}
    res = await c.aproxy("POST", f"/api/v10/channels/{channel_id}/messages/{message_id}/threads", body,
                         _bot_headers(bot_token))
    return str((res or {}).get("id", ""))


async def post_in_thread(thread_id: str, content: str, bot_token: str = "",
                         client: nango.NangoClient | None = None) -> str:
    """A thread is a channel; same endpoint as send_message."""
    return await send_message(thread_id, content, bot_token=bot_token, client=client)
