# Word Hunt VS — Humans vs AI

**TL;DR:** On-policy distillation of frontier AI agents into tiny, task-specialized action models.

Word Hunt VS (Humans vs AI) is a live multiplayer arena where humans, a frontier open model, and a tiny specialized model compete on the exact same 4x4 board under the same 75-second clock and the same physical action constraints ("Word Hunt").

Our smallest agent is an 11M-parameter transformer trained on a Lambda A100. At inference it sees only the board and its current path — no dictionary — and acts tile-by-tile through the same simulated "hand" as a human player. It competes live against Gemma 4 12B served with vLLM on Lambda, heuristic baselines, and people joining from their phones.

The larger idea is to use frontier models as teachers rather than permanent workers. A smaller policy acts inside an interactive environment, reaches the states it would naturally encounter, receives demonstrations or corrections from a stronger model, and is retrained on those trajectories. Repeating that loop can turn an expensive general-purpose model into a fast, cheap, specialized action policy.

Today's demo proves both ends of that loop: a frontier model operating in the environment and a tiny specialized policy operating in the same environment. Our current 11M model was trained from generated/self-solved trajectories rather than a fully closed-loop on-policy teacher process; closing that loop with teacher intervention is the next step (`nano/learn.py`).

Word Hunt also doubles as a human-calibrated evaluation arena. Every agent is measured on the same task for score, validity, latency, token usage, cost, and behavior under time pressure. Gemma calls are traced through Respan, and at the buzzer Gemma can invoke a Nango tool to publish the round recap directly to Discord.

**Same environment. Same actions. Same clock. Frontier teacher -> tiny policy.**

Team **wordhunt**, Open Model Hack, San Francisco, 2026-09-12. Submission: [docs/submission.md](docs/submission.md). Architecture: [docs/architecture.md](docs/architecture.md). Full plan: [PLAN.md](PLAN.md).

## Quickstart

Run locally: `make setup && make words && make serve` (players at `/r/CODE`, projector at `/s/CODE`); the nano seat needs torch + numpy (`make setup` installs them) and `NANO_CKPT=nano/checkpoints/d6_lambda.pt`. Live demo: https://wordhunt-pngitthrva-uw.a.run.app. Runbook: [docs/demo-runbook.md](docs/demo-runbook.md); more commands in [docs/architecture.md §6](docs/architecture.md#6-repo-map-runbook-local-vs-cloud).

## Original pitch and build plan (2026-09-12 morning)

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

- [ ] **Next step: closed-loop on-policy distillation with teacher intervention** (nano acts, Gemma corrects the states it actually reaches, retrain on those trajectories; `nano/learn.py` is the between-round hook)
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
