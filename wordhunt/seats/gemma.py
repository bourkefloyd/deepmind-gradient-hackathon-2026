"""Gemma seat via any OpenAI-compatible chat endpoint (vLLM on Lambda, Respan gateway, mlx-vlm).

Text-grid input, thinking off. The model returns a burst of words; they become a queue the hand
drains at human speed (cadence cap lives in the hand, not here). Words the model claims but that
cannot be traced are still attempted with a best-effort path so the miss shows on the ticker.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
import urllib.error
import urllib.request

from .base import WordQueuePolicy
from ..solver import best_effort_path, path_for_word

try:  # integrations/ is copied into the image next to wordhunt/; optional in stripped checkouts
    from integrations import respan as _respan
except ImportError:  # pragma: no cover
    _respan = None

WORD_RE = re.compile(r"[A-Za-z]{3,8}")

PROMPT = """You are playing Word Hunt on a 4x4 letter board.
Rules: words have 3-8 letters, use adjacent tiles (8 directions, diagonals allowed), never reuse a tile.
Board (rows top to bottom):
{grid}
{found_line}{bad_line}List up to {n} valid English words you can trace on this board, most confident first.
Output ONLY the words, uppercase, one per line, no numbering, no commentary."""


class GemmaPolicy(WordQueuePolicy):
    name = "gemma"

    def __init__(self, base_url: str, model: str, api_key: str = "", rng: random.Random | None = None,
                 words_per_call: int = 10, max_calls: int = 12, thinking: bool = False, timeout: float = 30.0,
                 temperature: float = 0.7, max_tokens: int = 200):
        super().__init__(rng)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.words_per_call = words_per_call
        self.max_calls = max_calls
        self.thinking = thinking
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._queue: list[list[int]] = []
        self._seen: set[str] = set()
        self._bad: list[str] = []
        self._found: list[str] = []
        self._task: asyncio.Task | None = None
        self._round = 0
        self.stats = {"calls": 0, "errors": 0, "latency_ms": [], "prompt_tokens": 0, "completion_tokens": 0,
                      "words_claimed": 0, "words_traceable": 0}

    def _route(self, board: str):
        """Respan gateway when RESPAN_ENABLED=1 (tagged gemma-seat / where=server), else direct."""
        if _respan is None:
            return None
        return _respan.route("gemma-seat", self.base_url, self.api_key, self.model, thread=f"server-{board.upper()}",
                             metadata={"where": "server", "board": board.upper(), "round": self._round,
                                       "words_per_call": self.words_per_call})

    # ---- lifecycle --------------------------------------------------------------------------
    async def start_round(self, board: str, words: dict[str, list[int]]) -> None:
        await super().start_round(board, words)
        self._queue, self._seen, self._bad, self._found = [], set(), [], []
        self._round += 1
        self.stats = {k: ([] if k == "latency_ms" else 0) for k in self.stats}
        self._task = asyncio.create_task(self._loop(board))

    async def end_round(self) -> None:
        if self._task:
            self._task.cancel()
            self._task = None

    def on_result(self, word: str, ok: bool, reason: str) -> None:
        if ok:
            self._found.append(word.upper())
        elif reason == "miss" and word:
            self._bad.append(word.upper())

    def next_word(self, board: str, found: set[str]) -> list[int] | None:
        while self._queue:
            p = self._queue.pop(0)
            w = "".join(board[t] for t in p)
            if w not in found:
                return p
        return None

    # ---- model calls -------------------------------------------------------------------------
    async def _loop(self, board: str) -> None:
        for _ in range(self.max_calls):
            if len(self._queue) >= 4:
                await asyncio.sleep(0.5)
                continue
            try:
                text = await asyncio.to_thread(self._call, board)
            except Exception as e:  # network / parse; keep the seat alive
                self.stats["errors"] += 1
                await asyncio.sleep(2.0)
                continue
            n_new = 0
            for w in WORD_RE.findall(text):
                w = w.lower()
                if w in self._seen:
                    continue
                self._seen.add(w)
                self.stats["words_claimed"] += 1
                p = path_for_word(board, w, self.rng)
                if p:
                    self.stats["words_traceable"] += 1
                else:
                    p = best_effort_path(board, w, self.rng)
                self._queue.append(p)
                n_new += 1
            if n_new == 0:
                await asyncio.sleep(1.5)

    def _call(self, board: str) -> str:
        grid = "\n".join(" ".join(board[r * 4 + c].upper() for c in range(4)) for r in range(4))
        found_line = f"Already found (do not repeat): {', '.join(self._found[-20:])}\n" if self._found else ""
        bad_line = f"Rejected as not on the board or not words: {', '.join(self._bad[-10:])}\n" if self._bad else ""
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": PROMPT.format(grid=grid, found_line=found_line, bad_line=bad_line, n=self.words_per_call)}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": self.thinking},
        }
        url = self.base_url + "/chat/completions"
        headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})}
        r = self._route(board)
        if r is not None and r.via_respan:
            body = {**body, "model": r.model, **r.body_extra}
            url, headers = r.completions_url(), r.auth_headers()
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
        t0 = time.time()
        self.stats["calls"] += 1
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        self.stats["latency_ms"].append(round((time.time() - t0) * 1000))
        usage = data.get("usage") or {}
        self.stats["prompt_tokens"] += usage.get("prompt_tokens", 0)
        self.stats["completion_tokens"] += usage.get("completion_tokens", 0)
        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        if _respan is not None and _respan.log_mode():
            import threading
            threading.Thread(target=_respan.log_request, kwargs=dict(
                caller="gemma-seat", model=self.model, messages=body["messages"], completion=content,
                latency_s=self.stats["latency_ms"][-1] / 1000, usage=usage, thread=f"server-{board.upper()}",
                upstream_base_url=self.base_url, timeout_s=5.0,
                metadata={"where": "server", "board": board.upper(), "round": self._round, "words_per_call": self.words_per_call}),
                name="respan-log", daemon=True).start()
        return content

    def public_stats(self) -> dict:
        lat = self.stats["latency_ms"]
        return {
            "calls": self.stats["calls"], "errors": self.stats["errors"],
            "latency_ms": round(sum(lat) / len(lat)) if lat else None,
            "tokens": self.stats["prompt_tokens"] + self.stats["completion_tokens"],
            "claimed": self.stats["words_claimed"], "traceable": self.stats["words_traceable"],
        }


def gemma_seats_from_env() -> list[dict]:
    """Seat specs from env. Either GEMMA_SEATS (JSON list of {id,name,model,base_url,api_key,label})
    or GEMMA_BASE_URL + GEMMA_MODELS (comma-separated) + GEMMA_API_KEY."""
    raw = os.environ.get("GEMMA_SEATS")
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return []
    base = os.environ.get("GEMMA_BASE_URL")
    if not base:
        return []
    key = os.environ.get("GEMMA_API_KEY", "")
    out = []
    for m in [x.strip() for x in os.environ.get("GEMMA_MODELS", "").split(",") if x.strip()]:
        short = m.split("/")[-1]
        sid = re.sub(r"[^a-z0-9]+", "-", short.lower()).strip("-")
        out.append({"id": sid if sid.startswith("gemma") else "gemma-" + sid, "name": short, "model": m,
                    "base_url": base, "api_key": key, "label": "gemma · text grid"})
    return out
