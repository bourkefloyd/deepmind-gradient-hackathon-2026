# Open Model Hack (2026-09-12): Word Hunt arena, nano is the hero

Plan for the Gradient x Google DeepMind Open Model Hack, San Francisco, Saturday 2026-09-12.
Hosts: Gradient, Lambda, Nango, Respan, GDM Gemma. Max team size 4. Prizes are Nango credits
($20k / $5k / $5k). Hacking window is roughly 10:30 to 4:30 (about six hours).

Event agenda: 9:30 doors, 10:00 opening, 10:30 team formation + hacking, 1:30 lunch, 4:30 demos,
6:00 closing + awards.

---

## 1. Pitch

**Can a 10M-parameter model play Word Hunt like a person?**

A joinable 75-second Word Hunt race. Same board, same clock, no turns. Seats: one or two humans on
their phones, a **nano policy** (~10M params, CPU, our own), and **Gemma 4** seats (E4B, 31B on
Lambda) for comparison. Every AI seat acts through the same "hand" that swipes tile by tile, so the
projector shows four fingers racing on one board.

- The nano is the hero: it plays at human pace, wanders, backs out of dead ends, finds `CAT`
  before `LANTERN`, and gets better between rounds.
- Gemma is the deliberate giant it is compared against, and the teacher whose words feed the nano.
- Humans are the bar.
- The arena is the eval harness. AI-only exhibition matches are the benchmark; matches with humans
  are the product.

One line for the close: *10M params, zero tokens, human pace; 31B finds longer words at 1000x the
compute.*

Why this wins the room: casual, spectatable, honest about model differences, and it doubles as an
eval harness for exactly the theme in the event copy ("open models are improving faster than most
teams can evaluate them").

Why this fits Gradient (Fund V): agentic systems, physical-AI infrastructure (perception, edge
inference, sim-to-real), and explicitly "open source / small models / hybrid compute"; no
foundation-model labs. Frontier open model as teacher, tiny student that acts at reflex speed on the
edge, measured against people. That is `sandbox/nanoagent` as a game.

---

## 2. Game rules

| | |
|---|---|
| Board | 4x4 letters. Words 3+ letters, 8-way adjacency, no tile reuse within a word. |
| Clock | 75 seconds. Everyone starts together. Simultaneous, not turn-based. |
| Seats | Human x1-2, Nano, Gemma 4 E4B, Gemma 4 31B (or 12B on the laptop as understudy). |
| Fairness | Same board, private word lists. |
| Score | 3=100, 4=400, 5=800, 6=1400, 7+=1800+. Highest score wins. |
| Dictionary | Valid = enable1 (3-8 letters). Scoring weight / "common" = ~30k common-words list so the AI does not win on `aalii`. |
| Join | Room code / QR. Late arrivals sit the next sprint (about a 30 s queue). No hot-join. |
| Spectator | One big board + live ticker (`31B LANTERN +1400`, `E4B CTA - miss`). Show everything; copying a 7-letter word in 75 s is hard anyway. |
| Rematch | One button, same seats, next packed board. |
| Boards | 8-10 pre-generated seeds known to have long words (no dead boards on stage). |

AI vs AI is the same room with no humans. AI vs human is the same room with a QR on the screen.

### The hand controller (core, not stretch)

Every AI seat, nano or Gemma, acts through a hand: tile-by-tile pointer moves on the shared board
at 8-12 Hz with human-ish noise (150-300 ms reaction lag, occasional wrong neighbor, backtrack).
Gemma's word bursts become a queue the hand drains at human speed. This is what removes the
"waiting for the model" feel; it is not a model-size problem first.

Cadence cap is a rule: one tile per tick, one word at a time. An AI that dumps 40 words in three
seconds is a solver, not a player.

---

## 3. The nano (hero seat)

### What "plays like us" means

| Human trait | Nano rule |
|---|---|
| Sees tiles, not a dictionary | Input is the 16 tiles + current path. No word list at inference. Words live in the weights. |
| Moves a finger | Acts tile by tile through the same hand controller as everyone else. |
| Hesitates, backs out | Actions: `extend(tile)`, `submit`, `abort`. Value head = "is this path still going somewhere". |
| Finds CAT before AALII | Train on a frequency-weighted word distribution. |
| Not deterministic | Sample with temperature > 0. Two nanos on one board diverge. |
| Gets better | Between rounds, add words found by anyone to its buffer; a few BC steps; rollback if held-out drops. |
| Speed is bounded | Reaction lag + one tile per tick. Cannot out-type a human by cheating on cadence. |

### Model (reuse `sandbox/nanoagent/nanoagent/model.py`)

- Tokens: `[CLS]` + 16 tile tokens (letter vocab + grid position) + up to 8 path tokens (tile
  index in order) + a path-length token. About 30 tokens.
- Heads: `target_logits` = pointer over the 16 tiles (next tile, or start tile when the path is
  empty). `type_logits` = `{extend, submit, abort}`. `value_logit` = P(prefix extends to a word).
  Span heads dropped.
- Size: depth 4 (~3M) or depth 6 (~10M). Depth 6 is the number said on stage.
- Policy: `StudentPolicy` decode + confidence carries over. `EscalatingPolicy` is the stretch
  (nano asks Gemma when its value is low; the $ is charged).

### Training

Word Hunt gives an exact oracle and infinite boards, which AndroidWorld never did.

1. Solver: trie DFS over enable1 (3-8 letters), plus the ~30k common-words list for weights.
2. Data: random boards -> all words -> sample paths weighted by word frequency. For every prefix
   emit soft targets: distribution over all valid next tiles (shared prefixes), `submit` mass when
   the prefix is a word, `abort` on mined dead prefixes, value = 1 if any completion exists. Same
   soft-CE + BCE(value) loss as `train.py`.
3. Scale: ~200k boards, 2-4M steps. Generation is minutes on CPU.
4. Train: `train.py` recipe, fp32, no AMP, 20-40 min on a Lambda 4090/A100, or the Mac after
   `make check-mps`. Start by 11:00 so a second seed fits. One MPS trainer at a time.
5. Serve: CPU inference in the game server, ~ms per action.

Expectation after 30 min: hundreds of common short words, occasional 5s, very few 7s. That is the
arc. The human beats it in round 1.

### Pre-registered gate

On 50 unseen boards:

- Must beat a random swiper by >= 4x score, and beat Gemma E4B on valid-word rate.
- Stretch target: >= 60% of the human median score on the same boards.
- If it misses at 1:30 PM, the seat ships as "reflex bot" (heuristic hand, same visuals) and the
  nano is one honest slide. Do not hide the swap.

### Live learning (stretch, but it is the thesis)

Between rounds: every validated word from any seat -> nano buffer -> 50-100 BC steps on CPU during
the 20 s rematch countdown -> held-out check -> keep or roll back. Teacher $0; Gemma becomes the
teacher on stage. Record the curve during lunch (AI-only league, 10 rounds) so the live version can
stall without cost.

---

## 4. Gemma seats (comparison + teacher)

| Seat | Serve | Character |
|---|---|---|
| Gemma 4 E4B | vLLM on Lambda (24 GB+), or laptop | hasty, short words, invalids |
| Gemma 4 12B Unified | laptop via mlx-vlm 4-bit (`make mlxvlm-up`), understudy for 31B | solid, a bit slow |
| Gemma 4 31B | vLLM on Lambda (80 GB), vision budget 560 | deliberate, long words |

Rules:

- Thinking **off** in-match for all seats (75 s is too short; thinking-on 31B can zero out).
  Thinking-on is an exhibition variant, run once.
- Input modality is part of the eval: run the same model on a screenshot seat and a text-grid seat.
  That is a comparison the event asked for and a clean fallback if vision is flaky.
- Do not use Ollama `gemma4:*-mlx` tags for vision; they drop the vision tower
  (`docs/gemma4-local-vision-findings.md`). mlx-vlm or vLLM only.
- vLLM flags from the Gemma 4 recipe: `--reasoning-parser gemma4`, `--tool-call-parser gemma4`,
  `--enable-auto-tool-choice`. Vision budget 560 (1120 halved pointing in record 0026).
- 31B weights are ~60 GB; start the pull at 10:30 sharp.

Teacher role: Gemma's validated words feed the nano buffer. Commentator role: at the buzzer Gemma
function-calls Nango (section 5).

---

## 5. Sponsor map

| Sponsor | Job in the match | Must be visible |
|---|---|---|
| Gemma 4 | Comparison seats, teacher, commentator | yes, on the board |
| Lambda | Serve 31B (and nano training) | architecture slide, endpoint URL |
| Respan | One endpoint for Gemma seats; traces, latency, $, fallbacks; side-by-side eval table | side monitor during the match |
| Nango | Gemma **function-calls** Nango at the buzzer: post a trash-talk recap to Slack, or file each E4B invalid word as a Linear issue ("bug: CTA is not a word"). Optional `/join` via Slack. | live at the buzzer |
| Gradient | Product: arena for open models vs people on casual games; small model + edge + distillation | the pitch |

Nango is the prize. A scorecard webhook is an integration; Gemma calling the tool itself is an
agent. Make the model call the tool.

Respan caveat: the gateway may not proxy an arbitrary vLLM URL on Lambda. Ask a Respan engineer at
9:45 before writing code. Fallback: Respan tracing SDK around our own OpenAI client (still gives
traces, latency, cost).

---

## 6. Eval headline (lunch league)

During lunch run an AI-only league in the background: 20 boards, seats = nano, E4B, 12B, 31B,
each on text grid and screenshot where applicable. One table on one slide:

score, valid-word rate, mean word length, words/min, $/match, latency per action, params.

That table is the research claim. The live match is the product claim.

---

## 7. Build order (cut from the bottom)

0. **Next step: closed-loop on-policy distillation with teacher intervention.** The nano acts in
   the room, the states it actually reaches go to Gemma for demonstrations or corrections, retrain
   on those trajectories between rounds (`nano/learn.py`), repeat. Today's 11M model is trained from
   self-solved trajectories; items 8-9 below are the hooks. Frontier teacher -> tiny policy.
1. Solver + board generator + scoring (also the eval and dead-board filter).
2. Data generator -> `train.py` running on Lambda by 11:00. One person owns this and nothing else.
3. Room + timer + one shared page (host view doubles as phone view) + hand controller.
4. Nano seat plugged into the hand (fallback: heuristic hand, same visuals).
5. Gemma seats through the same hand, thinking off, via Respan.
6. Lunch league, table, Respan on the side monitor.
7. Gemma function-calls Nango at the buzzer.
8. Live between-round learning.
9. Escalation seat (nano asks Gemma when unsure).
10. LoRA: never today (see section 10).

Minimum playable = 1 + 3 + two fake AIs. Everything else is afternoon.

Stack decision (made now, not at 10:30): one FastAPI process with a WebSocket room and a single
static HTML page (vanilla JS, no build step). New directory, no coupling to the ActionFleet
dashboard. Nano inference in the same process.

---

## 8. Day-of clock

| When | What |
|---|---|
| 9:30-10:30 | Credits, HF Gemma gate accepted, Respan key + gateway question, Slack workspace, Lambda box up and pulling 31B. Lock teammates. Do not invent a new game. |
| 10:30-12:00 | Solver, data, trainer running. Room, timer, hand, human play. |
| 12:00-1:30 | Nano seat in. One real Gemma seat. Fake AIs as fallback. |
| 1:30-2:30 | Nano gate decision. Lunch league running. Nango buzzer call. Respan visible. Pack 8-10 boards. |
| 2:30-3:30 | Rehearse. Record a 60 s backup video. Kill anything flaky. |
| 3:30-4:30 | QR up. Volunteer. Nano + two Gemmas. Rematch once. |

### 3-minute talk

1. Open the nano's finger on the board, not a table. "This is 10M parameters playing like you."
2. QR: "two of you, play it." Drop E4B and 31B in the other seats.
3. 75 s of ticker. Nano wanders and lands `HUNT`. E4B bricks `CTA`. 31B pauses then drops
   `LANTERN`. Someone in the room almost wins.
4. Side monitor: Respan cost/latency per word for Gemma; nano line reads $0, 2 ms.
5. Gemma calls Nango; the Slack card lands.
6. Lunch-league table. "Same match without humans is the eval. With humans is the product."

Framing that keeps DeepMind judges happy: Gemma is the teacher and the boss seat; the nano is what
we distill so a phone can play.

---

## 9. Roles (max 4)

| | |
|---|---|
| A | Solver, data, training, gate. Owns the nano. |
| B | Game, room, hand controller, projector. |
| C | Gemma seats via Respan/Lambda, lunch league table. |
| D | Nango, volunteers, backup video, the talk. |

Solo: A + B, Gemma from the laptop 12B, skip 6-9.

---

## 10. Stretch notes: LoRA (demoted to a slide)

Do not run the AndroidWorld recipe. Records 0027-0029 retired 0024-style LoRA on 31B GUI (inside
seed noise; Unsloth went backwards). If GPU minutes are spare, they buy the nano, which is our
thesis; LoRA is Gemma's.

If it ever happens (not today): task LoRA on the cheap seat (E4B or 12B Unified), text-grid input,
QLoRA r16, one epoch, <= 45 min, served with vLLM `--enable-lora` (no merges). Pre-registered gate
on 50 held-out boards: valid-word rate +15 points and no drop in mean word length, else discard.
Payoff is one exhibition match (E4B, E4B+LoRA, 31B). Never train thinking-on; thinking is an
inference knob.

Do not bring SWM or the JEPA encoder into today's loop: the board is our own data structure, so a
pixel encoder is overhead, and JEPA collapse debugging costs hours
(`.cursor/skills/toonblast-encoder-training/SKILL.md`). Mention JEPA / TTA / AdaJEPA
(`lab/swm/tta.py`, `research/adajepa`) as prior work on one slide.

---

## 11. Risks and answers

| Risk | Answer |
|---|---|
| Six hours is short | Minimum playable first; cut from the bottom of section 7. |
| Lambda queue / credits late | Laptop 12B via mlx-vlm, same OpenAI-compatible URL. |
| Gemma gated weights | Accept the license before doors. |
| Respan cannot proxy Lambda vLLM | Ask at 9:45; fall back to Respan tracing SDK. |
| Nano learns nothing in 30 min | Start training by 11:00; second seed; heuristic "reflex bot" fallback, labelled honestly. |
| AI bursts / silence break the race | Hand controller + cadence cap; thinking off in-match. |
| Vision misreads tiles | Text-grid seat as the paired comparison and fallback. |
| Dead boards | 8-10 pre-generated seeds with long words. |
| Swipe fussy on venue Wi-Fi | Tap-to-build path. |
| Shy room | You are the human. |
| Nango usage too thin for a Nango prize | Gemma function-calls Nango; not a server webhook. |

---

## 12. Not building today

Toon Blast as the main act. Crosswords. Turns. Elo. Accounts. Puzzle generation for crosswords.
A fourth GUI LoRA. A 120-episode AndroidWorld run. SWM / JEPA in the loop.

---

## 13. Pre-event checklist

- [ ] Luma approval confirmed (fully in-person; unapproved not admitted)
- [ ] HF token with Gemma 4 access (`google/gemma-4-E4B-it`, `-12B-it`, `-31B-it`)
- [ ] Lambda account, card, SSH key
- [ ] Respan account + `RESPAN_API_KEY`; one test call to `https://api.respan.ai/api/chat/completions`
- [ ] Nango account; Slack (or Linear) connected; 2-3 action functions enabled; `POST /action/trigger` tested
- [ ] `make mlxvlm-up` works on the laptop (12B 4-bit, vision intact)
- [ ] enable1 + a common-words list downloaded locally
- [ ] Adapter, dongle, battery, hotspot; phone for the human seat
- [ ] One slide: lineup table + "Ollama mlx tags are blind" + Fund V line

---

## 14. Decisions locked

- Nano depth 6, soft targets, frequency-weighted words, temperature sampling, ~10 Hz hand.
- Gate pre-registered as in section 3.
- Gemma stays: comparison seats + teacher + commentator. Thinking off in-match.
- Gemma calls Nango, not our server.
- The talk opens on the nano's finger.

Open picks: swipe vs tap-path for humans; one human vs two.

## References

- Event: https://luma.com/openmodelhack
- Gemma 4 model card: https://ai.google.dev/gemma/docs/core/model_card_4
- vLLM Gemma 4 recipe: https://github.com/vllm-project/recipes/blob/main/Google/Gemma4.md
- Respan gateway quickstart: https://respan.ai/docs/documentation/features/gateway/gateway-quickstart
- Nango tool calling: https://nango.dev/docs/getting-started/use-cases/tool-calling
- Repo prior work: `sandbox/nanoagent/README.md`, `sandbox/nanoagent/records/HANDOFF.md`,
  `docs/gemma4-local-vision-findings.md`
