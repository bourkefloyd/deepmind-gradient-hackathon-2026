# gemma_seat: Gemma 4 12B seat (local mlx-vlm)

Standalone bot client for the Word Hunt arena. Same model, two input
modalities (text grid vs. rendered screenshot), thinking off, one hand-paced
finger. Does not import `wordhunt/`.

## Settings (reused as-is from ActionFleet)

| | |
|---|---|
| Model | `mlx-community/gemma-4-12B-it-4bit` (12B Unified, encoder-free, plain 4-bit quantizes cleanly) |
| Server | `mlx_vlm.server` 0.6.3, `http://localhost:8080/v1`, started with `make mlxvlm-up` in the actionfleet checkout (~7.4 GB RSS) |
| Never | Ollama `gemma4:*-mlx` tags: vision tower dropped, the model is blind |
| Thinking | off: `reasoning_effort="none"` + mlx-vlm `enable_thinking=false` |
| Decoding | T=0.2, image before text in the user turn, PNG data URL; plus `presence_penalty=0.8`, `repetition_penalty=1.15` and a stream cut-off after 4 duplicate lines (Gemma 4 loops `WORD\nWORD\n...` at low T otherwise) |
| Image | 448 px board render, well under the 560-token vision budget the GUI records settled on |

## Run

```sh
uv venv .venv && uv pip install -p .venv/bin/python -r gemma_seat/requirements.txt
curl -sL -o data/enable1.txt https://raw.githubusercontent.com/dolph/dictionary/master/enable1.txt

.venv/bin/python -m gemma_seat.boards 3                 # solver smoke test
.venv/bin/python -m gemma_seat.client 3                 # one board, both modalities
.venv/bin/python -m gemma_seat.eval --n 20 --seed 0 --json out.json
.venv/bin/python -m gemma_seat.bot --dry-run --seed 3   # hand timeline, no server
.venv/bin/python -m gemma_seat.bot --room ABCD --server ws://localhost:8000 --modality text
```

## Eval, n=20 boards, seed 0 (`results/eval_n20_seed0.md`)

| metric | 12B text | 12B image |
|---|---:|---:|
| valid-word rate | 20% | 22% |
| mean word length (valid) | 3.79 | 3.57 |
| words / call (returned) | 25.5 | 20.4 |
| valid words / call | 5.1 | 4.4 |
| mean score / board | 1780 | 1200 |
| latency / call, mean | 2.1s | 2.2s |
| latency / call, p50 | 1.8s | 1.6s |
| invalid: not a word | 44 | 32 |
| invalid: not on board | 364 | 287 |
| errors / timeouts | 0 | 0 |

Reading: the 12B knows words but cannot trace adjacency; ~80% of its output is
real English that is not on the board. The screenshot seat reads the tiles fine
(vision intact) and lands in the same place as the text seat. Thinking off,
~2 s per call, so a 75 s race gets several re-asks.

## Bot

`bot.py` joins `ws://<server>/ws/<CODE>` and drains Gemma's words through
`Hand`: 150-300 ms reaction lag per word, one tile per ~10 Hz tick (jittered),
occasional wrong-neighbour slip + backtrack, then submit. Words that cannot be
traced on the board are skipped (a human would notice mid-swipe); traceable
non-words are swiped and judged as misses on the ticker. When Gemma's list runs
dry and time remains, it is re-asked with the words already tried fed back.

The protocol sits behind `ProtocolAdapter`; `WordhuntV1Adapter` follows the
iteration-1 draft (`hello` / `state` / `path` / `submit` / `mine`). Update it
when the game server's shapes are final.
