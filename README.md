# Word Hunt VS (humans vs. models): nano is the hero

Open Model Hack, San Francisco, 2026-09-12. Full plan: [PLAN.md](PLAN.md).

**Pitch.** A joinable 75-second Word Hunt race on one shared board: one or two humans on their phones, a ~10M-parameter nano policy on CPU (ours), and Gemma 4 seats (E4B, 31B on Lambda) for comparison. Every AI seat acts through the same tile-by-tile "hand", so the projector shows four fingers racing on one board, and the same room with no humans is the eval harness.

One line: *10M params, zero tokens, human pace; 31B finds longer words at 1000x the compute.*

## Repo layout (proposed)

Locked stack (PLAN.md section 7): one FastAPI process, one WebSocket room, one static HTML page in vanilla JS (no build step), nano CPU inference in the same process. No coupling to the ActionFleet dashboard.

```
wordhunt/            game server package
  solver.py          trie DFS over enable1 (3-8 letters); also eval + dead-board filter
  board.py           4x4 board generator, seeds, packed-board picker
  scoring.py         3=100 4=400 5=800 6=1400 7+=1800+; common-words weighting
  hand.py            hand controller: 8-12 Hz, reaction lag, wrong-neighbor, backtrack, cadence cap
  room.py            room code, 75 s clock, seats, private word lists, ticker, rematch
  server.py          FastAPI app + WebSocket room; serves static/; runs nano in-process
  seats/
    nano.py          nano policy seat (StudentPolicy decode, temperature sampling)
    gemma.py         Gemma seats via Respan/vLLM, text-grid + screenshot, thinking off
    fake.py          heuristic "reflex bot" (fallback, same visuals) + random swiper
nano/                the hero model (see nano/README.md)
  data.py            boards -> words -> frequency-weighted paths -> soft targets
  model.py           depth-6 (~10M) pointer/type/value heads (copied from nanoagent)
  train.py           soft-CE + BCE(value), fp32, no AMP (copied from nanoagent)
  gate.py            50 unseen boards: >= 4x random swiper, beats E4B on valid-word rate
  lab_notebook.ipynb executable lab (architecture, train, inference, trajectories, plots); rendered html beside it
static/
  index.html         host view doubles as phone view; board, ticker, QR, join
data/                word lists + boards (see data/README.md; lists are not committed)
  enable1.txt
  common-30k.txt
  boards.json        8-10 pre-generated seeds with long words
eval/
  league.py          lunch league: 20 boards, all seats, one table
scripts/             run server, start trainer on Lambda, pull weights
```

## Build order

Cut from the bottom. **MP** = minimum playable (1 + 3 + two fake AIs). Owner letters from PLAN.md section 9.

- [ ] 1. Solver + board generator + scoring **(MP, A)**
- [ ] 2. Data generator -> `train.py` running on Lambda; one person, nothing else **(A)**
- [ ] 3. Room + timer + one shared page + hand controller **(MP, B)**
- [ ] 3b. Two fake AI seats through the hand (reflex bot + random swiper) **(MP, B)**
- [ ] 4. Nano seat plugged into the hand (fallback: heuristic hand, same visuals) **(A)**
- [ ] 5. Gemma seats through the same hand, thinking off, via Respan **(C)**
- [ ] 6. Lunch league, table, Respan on the side monitor **(C)**
- [ ] 7. Gemma function-calls Nango at the buzzer **(D)**
- [ ] 8. Live between-round learning **(A)**
- [ ] 9. Escalation seat (nano asks Gemma when unsure) **(A)**
- [ ] 10. LoRA: never today

## Day-of clock (re-timed from ~11:10 PT)

PLAN.md assumed a 10:30 start and "trainer running by 11:00". Actual start is ~11:10. Hard anchors unchanged: lunch 1:30, nano gate 1:30, rehearsal 2:30-3:30, backup video + QR by 3:30, demos 4:30. The squeeze lands on the 12:00-1:30 block, so the first training seed has ~50 min, not 90; the second seed only fits if the first starts by 12:10.

| When | What |
|---|---|
| 11:10-11:30 | Copy prior work in (below). Lambda box up, 31B pull started. Respan gateway question answered. Word lists downloaded. |
| 11:30-12:15 | A: solver, data gen, trainer running on Lambda by **12:10** (hard). B: room, timer, page, hand, human can play. |
| 12:15-1:30 | B: two fake AIs in (MP done). A: nano seat in from the first checkpoint. C: one real Gemma seat via Respan. D: Nango action tested, slide drafted. |
| 1:30 | **Nano gate decision.** Miss = ship reflex bot, labelled honestly. Lunch. |
| 1:30-2:30 | Lunch league running. Nango buzzer call. Respan on side monitor. Pack 8-10 boards. |
| 2:30-3:30 | Rehearse. Record 60 s backup video by **3:30**. Kill anything flaky. |
| 3:30-4:30 | QR up. Volunteer. Nano + two Gemmas. Rematch once. |
| 4:30 | Demos. |

## Prior work to copy in

PLAN.md references files from another repo that are **not** in this one. Build-order item 2 cannot start until the first two are here.

- [ ] `sandbox/nanoagent/nanoagent/model.py` -> `nano/model.py` (drop span heads; keep `target_logits`, `type_logits`, `value_logit`)
- [ ] `sandbox/nanoagent/train.py` -> `nano/train.py` (soft-CE + BCE value loss, `StudentPolicy`, `EscalatingPolicy`)
- [ ] `make check-mps` target (Mac training fallback) -> `scripts/` or a Makefile
- [ ] `make mlxvlm-up` target (laptop 12B 4-bit understudy, vision intact) -> `scripts/`
- [ ] `docs/gemma4-local-vision-findings.md` ("Ollama mlx tags are blind"; vision budget 560)
- [ ] `sandbox/nanoagent/records/HANDOFF.md` (records 0024-0029 context for the slide)

## Pre-event checklist (PLAN.md section 13)

- [ ] Luma approval confirmed (fully in-person; unapproved not admitted)
- [ ] HF token with Gemma 4 access (`google/gemma-4-E4B-it`, `-12B-it`, `-31B-it`)
- [ ] Lambda account, card, SSH key
- [ ] Respan account + `RESPAN_API_KEY`; one test call to `https://api.respan.ai/api/chat/completions`
- [ ] Nango account; Slack (or Linear) connected; 2-3 action functions enabled; `POST /action/trigger` tested
- [ ] `make mlxvlm-up` works on the laptop (12B 4-bit, vision intact)
- [ ] enable1 + a common-words list downloaded locally
- [ ] Adapter, dongle, battery, hotspot; phone for the human seat
- [ ] One slide: lineup table + "Ollama mlx tags are blind" + Fund V line

## Open picks (PLAN.md section 14)

Default unless overridden:

- **Tap-to-build path for humans** (not swipe). Swipe is fussy over venue Wi-Fi and on unknown phones; tap is one event per tile and matches the AI hand's tile-by-tile cadence (section 11).
- **Two human seats.** Two QR joiners is the spectacle the pitch promises ("two of you, play it") and keeps the humans-are-the-bar comparison honest (section 1).

## Hosting (Cloud Run)

`scripts/deploy.sh iter<N>` builds the image without Docker (crane onto `python:3.12-slim`) and deploys a tagged, no-traffic revision at `https://iter<N>---wordhunt-<hash>-uw.a.run.app`; `scripts/promote.sh iter<N>` moves the demo link. One instance, session affinity, 3600 s WebSocket timeout. **Rooms live in memory per revision**: each tag URL has its own rooms, and a deploy or promote resets them, so share the link of the build you are actually playing on and do not promote mid-round (`promote.sh` refuses while a room has connected humans; `FORCE=1` overrides). Health and the room list: `/api/health`.

Quiet rooms (no Discord posts, for worker/load tests): `POST /api/rooms` with body `{"quiet": true}` (or `?quiet=1`), or open `/r/NEW?quiet=1` in a browser. Room state is snapshotted to `WH_ROOM_STORE` (a GCS bucket, default `gs://<project>-wordhunt-rooms`) on every transition and every 5 s during a race, and rehydrated on startup or on an unknown-code lookup, so shared links survive deploys (lobby/results restore with scores; a mid-race room comes back as a lobby with the same seats). Rooms expire after 3 h.
