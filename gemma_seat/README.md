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
.venv/bin/python -m gemma_seat.eval --n 20 --seed 0 --modality text --thinking --max-tokens 2048 --json think.json
.venv/bin/python -m gemma_seat.eval --merge a.json b.json        # one table across runs
.venv/bin/python -m gemma_seat.bot --dry-run --seed 3   # hand timeline, no server
.venv/bin/python -m gemma_seat.bot --room ABCD --server ws://localhost:8000 --modality text
.venv/bin/python -m gemma_seat.bot --create-room --start --rounds 1   # make a room, host it, play one race
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

### Prompt x thinking A/B, same 20 boards (`results/eval_n20_seed0_ab.md`)

| | text (default) | image | text + thinking on | index (tile-path prompt) |
|---|---:|---:|---:|---:|
| valid words / call | 5.1 | 4.4 | 0.0 | 0.1 |
| mean score / board | 1780 | 1200 | 0 | 25 |
| latency mean / p95 | 2.1s / 5.4s max | 2.2s / 10.3s max | 30.0s / 30.0s | 15.5s / 21.4s |
| timeouts (30 s cap) | 0 | 0 | 20/20 | 0 |

- Thinking on (`enable_thinking=true`, max_tokens 2048): 20/20 calls hit the 30 s
  cap with zero output; uncapped it was still enumerating tiles at 180 s. Not a
  race seat, and not an exhibition seat either: the 75 s clock ends before the
  first word. Thinking-on is a 31B story (ActionFleet record 0029/0031), not 12B.
- Tile-index paths, validated client-side: the 12B reasons in-band instead of
  listing, almost nothing parses as a legal path. Kept as `--modality index`
  for reference; default stays `text`.

## Live race (local `wordhunt` server, room QDH9, 75 s)

`python -m gemma_seat.bot --create-room --start --rounds 1`: 15 words swiped,
15/15 judged ok on the ticker, seat total 7400 vs Reflex-A 3400 / Reflex-B 5900.
Text modality, thinking off, one call up front (~2 s) then re-asks.

## Bot

`bot.py` joins `<server>/ws/<CODE>` and drains Gemma's words through `Hand`:
150-300 ms reaction lag per word, one tile per ~10 Hz tick (jittered),
occasional wrong-neighbour slip + backtrack, then submit. When Gemma's list runs
dry and time remains, it is re-asked with the words already tried fed back.

**Raw by default, no dictionary at the seat.** Every distinct 3+ letter word
Gemma emits is swiped: on a legal path when one exists, otherwise on a
best-effort path that jumps to the nearest tile with the next letter, so the
server judges it and the ticker shows `Gemma 12B LIPE - miss`. The only drop is
a word whose letters are not on the board at all (a human would not try it
either); the count is logged per round. `--filter-solver` is the comparison
mode from the first live race (skip untraceable words, and non-enable1 words if
the list is present).

Live races, local server, 75 s, text modality:

| mode | judged | ok | valid rate | score | vs reflex bots |
|---|---:|---:|---:|---:|---|
| `--filter-solver` (room QDH9) | 15 | 15 | 100% (filtered) | 7400 | 3400 / 5900 |
| raw (room G6WK) | 65 | 18 | 28% | 6800 | 4000 / 6400 |
| raw (room DQ8L) | 101 | 39 | 39% | 15500 | - |

`--server` takes `ws://`, `wss://`, `http://` or `https://` (Cloud Run): the bot
runs on the Mac next to mlx-vlm and joins the remote room over wss.
`make gemma-seat ROOM=CODE SERVER=https://<cloud-run-host>` from the repo root
(`SEAT_ARGS="--modality image --rounds 1"` for variants).

The protocol sits behind `ProtocolAdapter`; `WordhuntV1Adapter` follows the
iteration-1 draft (`hello` / `state` / `path` / `submit` / `mine`). Update it
when the game server's shapes are final.
