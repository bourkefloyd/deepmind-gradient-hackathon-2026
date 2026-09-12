# Submission: Word Hunt VS — Humans vs AI

Final form answers for the Open Model Hack (DeepMind / Gradient, 2026-09-12). Source of truth for the claims: [docs/judge-review.md](judge-review.md), [docs/architecture.md](architecture.md), [docs/league.md](league.md), [docs/gemma-lambda-vllm.md](gemma-lambda-vllm.md), [nano/README.md](../nano/README.md).

## Form fields

**Team name:** wordhunt

**Project name:** Word Hunt VS — Humans vs AI

**TL;DR:** On-policy distillation of frontier AI agents into tiny, task-specialized action models.

**Pitch (2 sentences):**
Word Hunt VS is a live multiplayer arena where humans on their phones, a frontier open model (Gemma 4 12B served with vLLM on Lambda) and a tiny specialized model (an 11-million-parameter transformer we trained today on a Lambda A100, no dictionary at inference) compete on the exact same 4x4 board under the same 75-second clock and the same tile-by-tile "hand", so the frontier teacher and the tiny policy are measured in one environment with the same actions. The larger idea is to use frontier models as teachers rather than permanent workers: a small policy acts, reaches the states it would naturally encounter, gets demonstrations or corrections from the stronger model, and is retrained on those trajectories; today's demo proves both ends of that loop live, with every Gemma call traced in Respan and Gemma itself calling a Nango tool at the buzzer to post the recap to Discord.

**Same environment. Same actions. Same clock. Frontier teacher -> tiny policy.**

**Where the loop stands today (honest):** the 11M model was trained from generated/self-solved trajectories (solver-annotated boards), not from a closed-loop on-policy teacher process. Both ends of the loop run in the same environment now (frontier model in the arena, tiny policy in the arena, one eval table: nano 16,550 a board at 99% valid words vs Gemma 12B 1,780 at 20%). Closing the loop with teacher intervention is the next step; the between-round learner (`nano/learn.py`: words found by any seat, human or Gemma, become a 50-step CPU update with held-out check and rollback) is measured but not wired into the live rematch.

**What it does:**
Create a room, share the QR / four-letter code, and up to 40 humans join from their phones while AI seats sit at the same table: `Nano 10M (Lambda)` (an 11.0M-parameter transformer that sees only the 16 tiles and its own path, no dictionary at inference), `Gemma 12B (Lambda)` (Gemma 4 12B via vLLM on a Lambda A100), plus heuristic Reflex bots and a random swiper as floors. The server runs a 20 Hz tick loop; every AI plays through a `Hand` (one tile per 100 ms, reaction lag, wrong-neighbour slips), so scores are comparable to a person's. The projector view (`/s/CODE`) shows every finger, per-seat mini-boards, and a masked live feed; results reveal all words, best words, and a "Posted to Discord" badge. `/lab` is the nano deep-dive (architecture, training curve, latency, trajectories, gate, league, live-learning curve).

## Sponsor tools and how each is used

| Sponsor | How it is used | Where in the repo |
|---|---|---|
| **Nango** | Discord integration `discord-wordhunt`, custom action `send-discord-recap`. The game server fires it on `room_created`, `round_started`, `round_ended` (leaderboard with medals). At the buzzer the Gemma commentator is handed the same action as an OpenAI **tool** and calls it itself (`send_discord_recap`), so the model, not our server, makes the tool call; the post is prefixed `(via Nango tool call)`, the terminal logs `MODEL CALLED NANGO TOOL`, and the game broadcasts an `integration` event that shows as a toast and a live "Posted to Discord" badge in the results screen. | `integrations/nango.py`, `integrations/handlers.py`, `integrations/tools.py`, `integrations/commentator.py`, `wordhunt/room.py` |
| **Lambda** | One `gpu_1x_a100_sxm4` box did two jobs. (1) **Trained the hero nano**: 1M self-solved boards → 27M samples, 16.3k steps × 1024, bf16 + compile, 33 minutes, about $2; `d6_lambda.pt` lifted the same 11M architecture from 11,245 to 16,550 a board (gate 14.3× random, 99.1% valid) and is the default checkpoint on the live service. (2) **Serves Gemma 4 12B with vLLM** (`google/gemma-4-12B-it`, bf16, `--reasoning-parser gemma4 --tool-call-parser gemma4`) to the in-server `Gemma 12B (Lambda)` seat, so Gemma is a first-class AI seat inside Cloud Run with calls, latency and tokens on its seat card. | `nano/README.md` (GPU training), `nano/checkpoints/d6_lambda.pt`, `docs/gemma-lambda-vllm.md`, `wordhunt/seats/gemma.py` |
| **Respan** | Every Gemma call goes through `integrations/respan.py`. The Lambda seat runs in **log mode** (direct vLLM call, then the finished call is posted to Respan `request-logs`), so every seat call is a span tagged `wordhunt-vs/gemma-seat`. The **commentator** and the nano-vs-Gemma eval run in **proxy mode** through the Respan gateway (`api.respan.ai/api`, hosted Gemma 4 31B), which gives cost, latency and fallbacks per call. Screenshot of the Logs page: `docs/respan-logs.png`. 17 unit tests on the routing (`integrations/test_respan.py`). | `integrations/respan.py`, `docs/architecture.md` §5b |
| **Gemma 4 (bonus: Gemma on Lambda)** | Gemma 4 12B-it served on the Lambda A100 with vLLM is the live `Gemma 12B (Lambda)` seat. We compared text grid vs screenshot vs thinking-on on 20 boards (`gemma_seat/results/`): thinking-on never answers inside a 75 s race, so it is off in-match; the bf16 vLLM model uses the whole 400-token budget (129 words per call vs 26 on the 4-bit Mac build) at the same ~20% valid rate. Gemma 4 31B via Respan writes the recap and calls the Nango tool. | `docs/gemma-lambda-vllm.md`, `gemma_seat/`, `integrations/commentator.py` |

## Multi-agent / multi-player

One room, one board, one clock: 25+ humans on phones (`WH_MAX_HUMANS` default 40; load-tested with 30 WebSocket humans making real swipes, ~750 MB RSS), plus `Nano 10M (Lambda)`, `Gemma 12B (Lambda)`, Reflex-A/B and a random swiper as AI seats, all acting through the same tile-by-tile hand and judged by the same solver. A Gemma commentator joins as a spectator agent and posts through Nango at the buzzer. The projector shows every finger and per-seat mini-boards at once; at 25+ seats the UI collapses to a top-5 leaderboard with you pinned. Between-round learning (`nano/learn.py`: humans and Gemma teach the nano, 50 BC steps in ~10 s with keep/rollback) is measured but not wired into the live rematch.

## Commercial angle

Frontier models are expensive general-purpose workers; most production agent tasks are narrow, repetitive and latency-bound. The business is the distillation loop: put a customer's frontier model into an interactive environment as the teacher, let a tiny policy act in that environment and reach the states it actually encounters, have the teacher demonstrate or correct, retrain, repeat, and ship the resulting action model (11M params, zero tokens, ~20 ms on one vCPU here) in place of the frontier model. Word Hunt is the first environment and the human-calibrated arena that makes the result measurable: every round is a labelled comparison (score, validity, latency, tokens, cost, behaviour under time pressure) of a frontier open model, the distilled policy and humans on the same board, with Respan traces on every teacher call. Model: a distillation service (bring your teacher and your environment, get a task-specialized policy plus the eval that proves it); paid eval seats for model teams (bring your endpoint, get a human-calibrated score and cost per action); free-to-play arenas as the top of funnel and the source of human baselines. Every fixed-rule casual game is a new environment and a new benchmark, and game studios pay for believable bots that play at human pace, which is exactly the hand controller.

## Links

- Repo: https://github.com/bourkefloyd/deepmind-gradient-hackathon-2026
- Live demo: https://wordhunt-pngitthrva-uw.a.run.app (create a room; players at `/r/CODE`, projector at `/s/CODE`)
- Nano lab (deep dive): https://wordhunt-pngitthrva-uw.a.run.app/lab
- Architecture: https://github.com/bourkefloyd/deepmind-gradient-hackathon-2026/blob/main/docs/architecture.md
- Eval league: https://github.com/bourkefloyd/deepmind-gradient-hackathon-2026/blob/main/docs/league.md
- Gemma on Lambda: https://github.com/bourkefloyd/deepmind-gradient-hackathon-2026/blob/main/docs/gemma-lambda-vllm.md
- Backup demo video: `demo-backup.mp4` (posted in the team Discord)

## 3-minute demo script

Setup: projector on `/s/CODE` of a fresh room on the demo URL with the QR visible; phone on `/r/CODE` as host; side monitor split between the Discord channel and the Respan Logs page; commentator running (`make commentator-demo ROOM=CODE SERVER=<demo-url>`); Lambda vLLM warm. Lineup: Nano 10M (Lambda), Gemma 12B (Lambda), Reflex-A.

| Clock | Screen | Say |
|---|---|---|
| 0:00–0:15 | Projector lobby: QR, room code, seat list `Nano 10M (Lambda)`, `Gemma 12B (Lambda)`, `Reflex-A`. | "Word Hunt VS. One board, 75 seconds, everyone at once. Scan this. You are playing an 11-million-parameter model we trained this afternoon on a Lambda A100, and Gemma 4 12B served from the same GPU." |
| 0:15–0:30 | Phone: Start. Projector countdown ring 20 → 0. | "Every AI plays through the same hand: one tile per tick, reaction lag, wrong-neighbour slips. The nano never sees a dictionary at inference; the words are in the weights." |
| 0:30–1:15 | Race. Projector: fingers, per-seat mini-boards, feed (`Nano 10M HUNT +400`, `Gemma 12B LIPE miss`, "thinking…"). Phone: swipe one word live. | "Nano takes the common short words first, that is the frequency-weighted training. Gemma names real English that is not on the board, watch the miss. Its calls, latency and tokens are on the seat card." |
| 1:15–1:45 | Side monitor: Respan Logs filtered to `wordhunt-vs`. | "Every Gemma call is a span in Respan: seat calls from the Lambda box in log mode, the commentator through the Respan gateway with cost per call. The nano row reads zero tokens, about 20 ms on one vCPU." |
| 1:45–2:15 | Buzzer. Results: leaderboard, best words, words possible, "Posted to Discord" badge and toast. Discord tab: server leaderboard post, then Gemma's recap. | "At the buzzer Gemma itself calls a Nango tool, `send_discord_recap`, and roasts the round. That is the model calling the action, not our server." Read one line of the post. |
| 2:15–2:45 | `/lab` or `docs/league.md`. | "Same room with no humans is the benchmark. Twenty boards: nano 16,550 at 99% valid; Gemma 12B 1,780 at 20%; random 1,180. Lambda training took the same 11M model from 11k to 16.5k in 33 minutes for two dollars." |
| 2:45–3:00 | Projector: tap Rematch, countdown running. | "11M params, zero tokens, human pace; 12B finds longer words at a thousand times the compute. Same environment, same actions, same clock: frontier teacher, tiny policy. Nango, Lambda, Respan, Gemma: all live in that round. Questions." |

Fallback: if the Lambda seat is unreachable, run the Mac bot (`make gemma-seat ROOM=CODE SERVER=<demo-url>`) and say so; do not claim what is not on screen.
