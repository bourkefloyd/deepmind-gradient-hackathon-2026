"""Async fire-and-forget event dispatcher.

    from integrations.events import emit
    emit("round_ended", payload)          # returns immediately; never raises

Handlers are `async def handler(event_type, payload)`. Each runs in its own task under a
timeout; failures are logged and swallowed so the game loop is never blocked or broken by
an integration. `emit` is safe to call without a running loop (it then drops the event
with a debug log), and payloads are shallow-copied so handlers cannot mutate game state.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable

log = logging.getLogger("integrations.events")

Handler = Callable[[str, dict[str, Any]], Awaitable[None]]

HANDLER_TIMEOUT_S = 15.0

_handlers: dict[str, list[Handler]] = {}
_tasks: set[asyncio.Task] = set()
_installed = False


def on(event_type: str, handler: Handler) -> None:
    hs = _handlers.setdefault(event_type, [])
    if handler not in hs:
        hs.append(handler)


def clear() -> None:
    _handlers.clear()


def handlers_for(event_type: str) -> list[Handler]:
    return list(_handlers.get(event_type, [])) + list(_handlers.get("*", []))


def emit(event_type: str, payload: dict[str, Any] | None = None) -> int:
    """Schedule all handlers for `event_type`. Returns how many tasks were started."""
    global _installed
    if not _installed:
        _install_default_handlers()
    hs = handlers_for(event_type)
    if not hs:
        log.debug("no handlers for %s", event_type)
        return 0
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        log.debug("emit(%s) outside an event loop; dropped", event_type)
        return 0
    data = dict(payload or {})
    data.setdefault("ts", time.time())
    n = 0
    for h in hs:
        t = loop.create_task(_run(h, event_type, data))
        _tasks.add(t)
        t.add_done_callback(_tasks.discard)
        n += 1
    return n


async def _run(h: Handler, event_type: str, payload: dict[str, Any]) -> None:
    t0 = time.perf_counter()
    try:
        await asyncio.wait_for(h(event_type, payload), timeout=HANDLER_TIMEOUT_S)
        log.debug("%s -> %s ok in %.2fs", event_type, getattr(h, "__name__", h), time.perf_counter() - t0)
    except asyncio.TimeoutError:
        log.warning("%s -> %s timed out after %.0fs", event_type, getattr(h, "__name__", h), HANDLER_TIMEOUT_S)
    except Exception as e:  # noqa: BLE001 - integrations must never break the game
        log.warning("%s -> %s failed: %s: %s", event_type, getattr(h, "__name__", h), type(e).__name__, e)


async def drain(timeout: float = HANDLER_TIMEOUT_S) -> None:
    """Wait for in-flight handlers (tests, demo, graceful shutdown)."""
    if _tasks:
        await asyncio.wait(list(_tasks), timeout=timeout)


def _install_default_handlers() -> None:
    global _installed
    _installed = True
    ensure_logging()
    from . import handlers
    handlers.register()


def ensure_logging() -> None:
    """uvicorn configures only its own loggers; make `integrations.*` INFO visible in the game logs."""
    pkg = logging.getLogger("integrations")
    if not pkg.handlers and not logging.getLogger().handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        pkg.addHandler(h)
        pkg.setLevel(logging.INFO)
