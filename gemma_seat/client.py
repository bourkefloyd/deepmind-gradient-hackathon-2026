"""OpenAI-compatible client for the Gemma 4 12B seat (local mlx-vlm).

Settings are lifted as-is from ActionFleet's `gemma4-12b-mlxvlm` runner
(`backend/actionfleet/agents/runners.py`, `docs/gemma4-local-vision-findings.md`):

- model `mlx-community/gemma-4-12B-it-4bit` served by `mlx_vlm.server` on
  http://localhost:8080/v1 (`make mlxvlm-up` in actionfleet). Never the Ollama
  `gemma4:*-mlx` tags: those drop the vision tower.
- thinking OFF: `reasoning_effort="none"` plus mlx-vlm's own `enable_thinking=false`.
- temperature 0.2; image placed BEFORE the text in the user turn; PNG data URL.

Two input modalities against the same model: a text grid and a rendered
screenshot of the grid. Every call has a hard wall-clock timeout.
"""

from __future__ import annotations

import base64
import io
import os
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from PIL import Image, ImageDraw, ImageFont

from .boards import MIN_LEN, grid_rows, grid_text

DEFAULT_BASE_URL = os.environ.get("MLXVLM_BASE_URL", "http://localhost:8080/v1")
DEFAULT_MODEL = os.environ.get("MLXVLM_MODEL", "mlx-community/gemma-4-12B-it-4bit")
# 448 px board: comfortably below the 560-token vision budget the GUI records
# settled on for Gemma 4 pointing; letters stay large and unambiguous.
DEFAULT_IMAGE_PX = 448

SYSTEM_PROMPT = (
    "You are playing Word Hunt, a 4x4 letter-grid word game.\n"
    "Rules: a word is spelled by a path of adjacent tiles (horizontal, vertical or "
    "diagonal neighbours). Each tile may be used at most once per word. Words must "
    "have at least 3 letters and be ordinary English dictionary words (no proper "
    "nouns, no abbreviations). Longer words score far more: 3 letters=100, 4=400, "
    "5=800, 6=1400, 7+=1800.\n"
    "Answer with the words only, one per line, UPPERCASE, longest first. "
    "Each word exactly once; never repeat a word. When you run out of words, stop. "
    "No numbering, no commentary, no explanations."
)

TEXT_USER_PROMPT = (
    "Here is the 4x4 grid (rows top to bottom, letters left to right):\n\n{grid}\n\n"
    "List every valid word you can find on this grid, following the adjacency and "
    "no-reuse rules. Aim for up to {max_words} words. Words only, one per line."
)

IMAGE_USER_PROMPT = (
    "The image is a screenshot of a 4x4 Word Hunt grid. Read the 16 letters from the "
    "image, then list every valid word you can find on it, following the adjacency "
    "and no-reuse rules. Aim for up to {max_words} words. Words only, one per line."
)

_WORD_RE = re.compile(r"[A-Za-z]{3,}")


@dataclass
class CallResult:
    modality: str
    words: list[str]
    latency_s: float
    raw: str = ""
    error: str | None = None
    usage: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None


def parse_words(text: str) -> list[str]:
    """Distinct lowercase alphabetic tokens (>= MIN_LEN) in output order.

    Tolerates 'WORD', '1. WORD', 'WORD, WORD' and markdown bullets. Drops any
    <think>/<thought> block if a template ever leaks one.
    """
    text = re.sub(r"<(think|thought)>.*?</\1>", " ", text, flags=re.S | re.I)
    seen: set[str] = set()
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*•").strip()
        if not line:
            continue
        for tok in _WORD_RE.findall(line):
            w = tok.lower()
            if len(w) >= MIN_LEN and w not in seen:
                seen.add(w)
                out.append(w)
    return out


def render_board_png(board: str, px: int = DEFAULT_IMAGE_PX) -> bytes:
    """Render the grid as a Word-Hunt-ish screenshot (tan tiles, dark letters)."""
    img = Image.new("RGB", (px, px), (34, 92, 66))
    draw = ImageDraw.Draw(img)
    n = 4
    margin = px * 0.05
    gap = px * 0.03
    cell = (px - 2 * margin - (n - 1) * gap) / n
    font = _load_font(int(cell * 0.62))
    for r, row in enumerate(grid_rows(board.upper())):
        for c, ch in enumerate(row):
            x0 = margin + c * (cell + gap)
            y0 = margin + r * (cell + gap)
            draw.rounded_rectangle(
                (x0, y0, x0 + cell, y0 + cell),
                radius=cell * 0.15,
                fill=(233, 214, 168),
                outline=(120, 90, 40),
                width=max(2, px // 150),
            )
            bbox = draw.textbbox((0, 0), ch, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(
                (x0 + (cell - tw) / 2 - bbox[0], y0 + (cell - th) / 2 - bbox[1]),
                ch,
                fill=(40, 30, 20),
                font=font,
            )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _load_font(size: int):
    for name in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


class GemmaSeatClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout_s: float = 30.0,
        max_tokens: int = 400,
        temperature: float = 0.2,
        max_words: int = 40,
        image_px: int = DEFAULT_IMAGE_PX,
    ):
        from openai import OpenAI

        self.model = model
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_words = max_words
        self.image_px = image_px
        self._client = OpenAI(
            base_url=base_url,
            api_key=os.getenv("MLXVLM_API_KEY") or "mlx-vlm",
            timeout=timeout_s,
            max_retries=0,
        )

    def _messages(self, board: str, modality: str, exclude: list[str] | None = None) -> list[dict]:
        tail = ""
        if exclude:
            tail = (
                "\n\nYou already tried these words, do not list them again: "
                + ", ".join(w.upper() for w in exclude[-60:])
                + ". Find different words, including shorter ones."
            )
        if modality == "text":
            user_content: object = (
                TEXT_USER_PROMPT.format(grid=grid_text(board), max_words=self.max_words) + tail
            )
        elif modality == "image":
            b64 = base64.b64encode(render_board_png(board, self.image_px)).decode()
            user_content = [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": IMAGE_USER_PROMPT.format(max_words=self.max_words) + tail},
            ]
        else:
            raise ValueError(f"unknown modality {modality!r}")
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _request_kwargs(self, board: str, modality: str, exclude: list[str] | None = None) -> dict:
        return {
            "model": self.model,
            "messages": self._messages(board, modality, exclude),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "reasoning_effort": "none",
            # Gemma 4 at low temperature falls into WORD\nWORD\nWORD loops (the
            # 'repetition loops' failure of nanoagent record 0025); the penalties
            # break the loop, and the stream reader below cuts it off if not.
            "presence_penalty": 0.8,
            "extra_body": {"enable_thinking": False, "repetition_penalty": 1.15},
        }

    def words(self, board: str, modality: str = "text") -> CallResult:
        """One blocking call; never raises (errors land in CallResult.error).

        Streams internally so a repetition loop is cut after `repeat_stop`
        duplicate lines instead of burning the whole token budget.
        """
        t0 = time.perf_counter()
        state: dict = {"raw": "", "error": None}
        words = list(self._stream(board, modality, state=state))
        return CallResult(modality, words, time.perf_counter() - t0, state["raw"], state["error"])

    def stream_words(
        self,
        board: str,
        modality: str = "text",
        on_word: Callable[[str], None] | None = None,
        deadline_s: float | None = None,
        exclude: list[str] | None = None,
    ) -> Iterator[str]:
        """Yield words as the model emits lines, so a hand can start swiping early.

        Stops at `deadline_s` seconds of wall clock (defaults to timeout_s).
        `exclude` is fed back as "already tried" so a seat can re-ask mid-race.
        """
        for w in self._stream(board, modality, deadline_s=deadline_s, exclude=exclude):
            if on_word:
                on_word(w)
            yield w

    repeat_stop = 4

    def _stream(
        self,
        board: str,
        modality: str,
        deadline_s: float | None = None,
        state: dict | None = None,
        exclude: list[str] | None = None,
    ) -> Iterator[str]:
        deadline = time.perf_counter() + (deadline_s if deadline_s is not None else self.timeout_s)
        seen: set[str] = set()
        buf = ""
        raw = ""
        dup_run = 0

        def take(line: str) -> Iterator[str]:
            nonlocal dup_run
            toks = parse_words(line)
            if not toks:
                return
            new = [w for w in toks if w not in seen]
            dup_run = 0 if new else dup_run + 1
            for w in new:
                seen.add(w)
                yield w

        try:
            stream = self._client.chat.completions.create(
                **self._request_kwargs(board, modality, exclude), stream=True, timeout=self.timeout_s
            )
            for chunk in stream:
                if time.perf_counter() > deadline or dup_run >= self.repeat_stop:
                    stream.close()
                    break
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content or ""
                buf += delta
                raw += delta
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    yield from take(line)
        except Exception as exc:  # noqa: BLE001 - timeouts and server hiccups are data here
            if state is not None:
                state["error"] = f"{type(exc).__name__}: {exc}"
        if dup_run < self.repeat_stop:
            yield from take(buf)
        if state is not None:
            state["raw"] = raw


if __name__ == "__main__":
    import random
    import sys

    from .boards import load_dictionary, packed_board, validate

    d = load_dictionary()
    board, sol = packed_board(random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 0), d)
    print(grid_text(board), f"\n({len(sol)} words on board)\n")
    c = GemmaSeatClient()
    for modality in ("text", "image"):
        r = c.words(board, modality)
        marks = [f"{w}{'' if validate(board, w, d)[0] else '?'}" for w in r.words]
        print(f"[{modality}] {r.latency_s:.1f}s err={r.error} n={len(r.words)}: {' '.join(marks)}")
