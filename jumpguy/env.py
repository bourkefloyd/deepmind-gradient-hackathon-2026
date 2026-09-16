"""Playwright / CDP wrapper around the live Jump Guy origin.

The game is a Phaser canvas at https://game.jumpguy.net (not the Viva+ iframe
wrapper). We drive Space/tap, grab the canvas, and read score from the
server-authoritative /api/runs* traffic.
"""

from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse

import numpy as np

from .actions import Action
from .constants import CANVAS_H, CANVAS_W, GAME_URL, PLAYER_NAME
from .observe import FrameStack, hint_text_ratio
from .sim import GameState, StepResult


@dataclass
class LiveInfo:
    score: int = 0
    server_score: int = 0
    status: str = "ready"
    run_id: Optional[str] = None
    overlay_open: bool = False
    signed_in: bool = False
    latency_ms: float = 0.0
    grab_ms: float = 0.0
    action_ms: float = 0.0


class JumpGuyLive:
    """Realtime env: frame in, discrete action out, game clock runs on its own."""

    def __init__(
        self,
        url: str = GAME_URL,
        headless: bool = True,
        width: int = CANVAS_W,
        height: int = CANVAS_H,
        player_name: str = PLAYER_NAME,
        navigation_timeout_ms: int = 30_000,
        chrome_channel: str = "chrome",
    ):
        self.url = url
        self.headless = headless
        self.width = width
        self.height = height
        self.player_name = player_name
        self.navigation_timeout_ms = navigation_timeout_ms
        self.chrome_channel = chrome_channel
        self._pw = None
        self._browser = None
        self._context = None
        self._page = None
        self._canvas = None
        self.stacker = FrameStack()
        self.info = LiveInfo()
        self._started_running = False
        self._last_hint = 0.0
        self._on_log: list[str] = []

    def __enter__(self) -> "JumpGuyLive":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def start(self) -> None:
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        launch_kwargs: dict[str, Any] = {
            "headless": self.headless,
            "args": [
                "--disable-dev-shm-usage",
                "--no-first-run",
                "--autoplay-policy=no-user-gesture-required",
                f"--window-size={self.width},{self.height + 80}",
            ],
        }
        try:
            self._browser = self._pw.chromium.launch(channel=self.chrome_channel, **launch_kwargs)
        except Exception:
            self._browser = self._pw.chromium.launch(**launch_kwargs)
        self._context = self._browser.new_context(
            viewport={"width": self.width, "height": self.height},
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            ignore_https_errors=True,
        )
        self._page = self._context.new_page()
        self._page.set_default_timeout(self.navigation_timeout_ms)
        self._page.on("response", self._on_response)
        self._page.goto(self.url, wait_until="domcontentloaded")
        self._page.wait_for_selector("#game-root canvas", timeout=self.navigation_timeout_ms)
        # First click focuses the canvas so Space is delivered to Phaser.
        self._canvas = self._page.locator("#game-root canvas")
        self._canvas.click(timeout=self.navigation_timeout_ms)
        time.sleep(0.15)
        self.info = LiveInfo()
        self._started_running = False

    def close(self) -> None:
        for obj in (self._context, self._browser):
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
        if self._pw is not None:
            try:
                self._pw.stop()
            except Exception:
                pass
        self._pw = self._browser = self._context = self._page = self._canvas = None

    def _on_response(self, response) -> None:
        try:
            url = response.url
            path = urlparse(url).path
            status = response.status
            if "/api/runs" in path and response.request.method == "POST":
                if path.rstrip("/").endswith("/runs") and status < 400:
                    self.info.run_id = (path.split("/")[-1] if False else self.info.run_id)
                    try:
                        data = response.json()
                        if isinstance(data, dict) and data.get("runId"):
                            self.info.run_id = data["runId"]
                            self.info.status = "running"
                    except Exception:
                        pass
                elif path.endswith("/pass") and status < 400:
                    try:
                        data = response.json()
                        if isinstance(data, dict) and isinstance(data.get("score"), (int, float)):
                            self.info.server_score = int(data["score"])
                            self.info.score = self.info.server_score
                    except Exception:
                        pass
                elif path.endswith("/end"):
                    self.info.status = "gameover"
                    try:
                        data = response.json()
                        if isinstance(data, dict) and isinstance(data.get("score"), (int, float)):
                            self.info.server_score = int(data["score"])
                            self.info.score = self.info.server_score
                    except Exception:
                        pass
                elif path.endswith("/submit") and status < 400:
                    self._on_log.append("score submitted")
        except Exception:
            return

    def _press_jump(self) -> None:
        assert self._page is not None
        self._page.keyboard.down("Space")
        self._page.keyboard.up("Space")

    def _press_restart(self) -> None:
        assert self._page is not None
        self._page.keyboard.press("r")

    def _grab_frame(self) -> np.ndarray:
        assert self._page is not None
        t0 = time.perf_counter()
        # Canvas toDataURL is one round-trip and avoids a full-page screenshot.
        data_url = self._page.evaluate(
            """() => {
              const c = document.querySelector('#game-root canvas');
              if (!c) return null;
              return c.toDataURL('image/jpeg', 0.55);
            }"""
        )
        if not data_url:
            png = self._canvas.screenshot(type="jpeg", quality=55)
            frame = _jpeg_to_rgb(png)
        else:
            raw = data_url.split(",", 1)[1]
            frame = _jpeg_to_rgb(base64.b64decode(raw))
        self.info.grab_ms = (time.perf_counter() - t0) * 1000.0
        return frame

    def _poll_dom(self) -> None:
        assert self._page is not None
        overlay = self._page.evaluate(
            """() => {
              const el = document.querySelector('.score-overlay');
              const name = document.querySelector('.score-overlay__name');
              return {
                open: !!(el && !el.hidden),
                name: name ? name.textContent : '',
                canSubmit: !!(document.querySelector('.score-overlay__button[data-submit]:not([hidden])')),
              };
            }"""
        )
        if not overlay:
            return
        self.info.overlay_open = bool(overlay.get("open"))
        if overlay.get("name"):
            self.info.signed_in = True
        if self.info.overlay_open:
            self.info.status = "gameover"
            self._maybe_fill_name_and_submit()

    def _maybe_fill_name_and_submit(self) -> None:
        """Fill any name field with PLAYER_NAME. Official submit uses Viva+ username."""
        assert self._page is not None
        try:
            self._page.evaluate(
                """(name) => {
                  const root = document.querySelector('.score-overlay');
                  if (!root || root.hidden) return false;
                  const inputs = root.querySelectorAll('input, textarea, [contenteditable="true"]');
                  for (const el of inputs) {
                    if (el.isContentEditable) { el.textContent = name; }
                    else { el.value = name; el.dispatchEvent(new Event('input', {bubbles:true})); }
                  }
                  const named = root.querySelector('.score-overlay__name');
                  if (named && (named.tagName === 'INPUT' || named.tagName === 'TEXTAREA')) {
                    named.value = name;
                  }
                  return true;
                }""",
                self.player_name,
            )
        except Exception:
            pass

    def handle_game_over_ui(self, submit: bool = False) -> dict[str, Any]:
        """Best-effort overlay handling. Returns what we could do."""
        self._poll_dom()
        result = {
            "overlay_open": self.info.overlay_open,
            "player_name": self.player_name,
            "submitted": False,
            "skipped": False,
            "note": "",
        }
        if not self.info.overlay_open:
            result["note"] = "no overlay (guest / offline: name is Viva+ account, not a text field)"
            return result
        self._maybe_fill_name_and_submit()
        assert self._page is not None
        if submit:
            btn = self._page.locator(".score-overlay__button[data-submit]")
            try:
                if btn.is_visible():
                    btn.click()
                    result["submitted"] = True
                    result["note"] = f"clicked SUBMIT as {self.player_name} (server uses Viva+ username)"
                    return result
            except Exception as exc:
                result["note"] = f"submit click failed: {exc}"
        skip = self._page.locator(".score-overlay__button[data-skip]")
        try:
            if skip.is_visible():
                skip.click()
                result["skipped"] = True
                result["note"] = "clicked SKIP on score overlay"
        except Exception as exc:
            result["note"] = f"skip failed: {exc}"
        return result

    def _infer_status(self, frame: np.ndarray) -> None:
        hint = hint_text_ratio(frame)
        if not self._started_running:
            if hint < 0.004:
                self._started_running = True
                if self.info.status == "ready":
                    self.info.status = "running"
        else:
            # Hint band fills back in with GAME OVER text.
            if hint > 0.012 and self.info.status == "running":
                self.info.status = "gameover"
        self._last_hint = hint

    def reset(self) -> StepResult:
        if self._page is None:
            self.start()
        if self.info.status == "gameover":
            self.handle_game_over_ui(submit=False)
            self._press_restart()
            time.sleep(0.05)
            self._press_jump()
        self.info.status = "ready"
        self.info.score = 0
        self._started_running = False
        frame = self._grab_frame()
        stack = self.stacker.reset(frame)
        return StepResult(frame, stack, self._state(), 0.0, False, {"event": "reset"})

    def _state(self) -> GameState:
        return GameState(
            score=self.info.score,
            player_x=CANVAS_W * 0.22,
            player_y=0.0,
            player_vy=0.0,
            grounded=True,
            speed=0.0,
            status=self.info.status,
            obstacles=[],
            t=0.0,
            ticks=0,
        )

    def step(self, action: int | Action) -> StepResult:
        t0 = time.perf_counter()
        if self.info.status == "gameover":
            if int(action) == Action.JUMP:
                self.handle_game_over_ui(submit=False)
                self._press_jump()
                self.info.status = "ready"
                self.info.score = 0
                self._started_running = False
        elif int(action) == Action.JUMP:
            ta = time.perf_counter()
            self._press_jump()
            self.info.action_ms = (time.perf_counter() - ta) * 1000.0
            if self.info.status == "ready":
                self.info.status = "running"
        frame = self._grab_frame()
        self._poll_dom()
        self._infer_status(frame)
        stack = self.stacker.push(frame)
        done = self.info.status == "gameover"
        reward = 0.0
        self.info.latency_ms = (time.perf_counter() - t0) * 1000.0
        info = {
            "event": "dead" if done else "tick",
            "score": self.info.score,
            "server_score": self.info.server_score,
            "status": self.info.status,
            "latency_ms": self.info.latency_ms,
            "grab_ms": self.info.grab_ms,
            "action_ms": self.info.action_ms,
            "run_id": self.info.run_id,
            "hint": self._last_hint,
            "overlay": self.info.overlay_open,
        }
        return StepResult(frame, stack, self._state(), reward, done, info)


def _jpeg_to_rgb(data: bytes) -> np.ndarray:
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(data)).convert("RGB")
        return np.asarray(img, dtype=np.uint8)
    except Exception:
        # Minimal JPEG fallback: require Pillow in practice.
        raise RuntimeError("Pillow is required to decode live canvas JPEGs") from None
