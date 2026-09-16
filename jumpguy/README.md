# Jump Guy realtime agent

A local control loop that plays [Jump Guy](https://game.jumpguy.net) at game speed: **frame in → discrete jump/noop out**. Training happens on a physics clone of the live Phaser scene; evaluation can target the live site.

This package is isolated from Word Hunt (`wordhunt/`, `nano/`, `gemma_seat/`). It reuses the same ideas (tiny policy, checkpoint JSON, Lambda/SSH train script) without touching those paths.

Player name used on any score / achievement UI: **`Grok Bot Son`**.

## What the live game actually is

Investigated 2026-09-16 by fetching the Viva+ page and the playable origin (not guessed).

| | |
|---|---|
| Marketing page | https://vivaplus.tv/pages/jumpguy (Fourthwall, reCAPTCHA, HMAC iframe) |
| **Playable origin** | **https://game.jumpguy.net** — drive this URL, not the iframe wrapper |
| Engine | Phaser 3 Arcade, canvas **960×540**, pixel art |
| Input | **Space / tap / click** to jump. **R** or Space restarts after game over |
| Physics | gravity `y=1800`, jump velocity `-700`, buffer 130 ms, coyote 100 ms |
| Obstacles | tinted squares 28–34 px, spawn off the right, 260→520 px/s (+4.5 px/s²) |
| Score | +1 when an obstacle’s right edge passes the player’s left edge |
| Anti-cheat | server-authoritative `POST /api/runs`, `/pass?seq=`, `/end`, `/submit` |
| Leaderboard | top scores ~1430; submit requires a **Viva+ Apprentice** session. The overlay shows the account username (not a free-text field) |

CSP on the game origin: `frame-ancestors 'self' https://vivaplus.tv https://*.vivaplus.tv https://*.fourthwall.com`. Cloudflare sits in front (`__cf_bm`). Play the origin as a top-level page.

Constants live in `jumpguy/constants.py` (copied from the shipped `main-*.js` bundle).

## Folder layout

```
jumpguy/
  constants.py      live-game numbers + PLAYER_NAME
  actions.py        NOOP / JUMP
  sim.py            60 Hz physics clone + numpy renderer
  observe.py        84×84 grayscale stack
  policy.py         timed-jump heuristic (state or CV) + CNN wrapper
  model.py          JumpNet (small CNN, policy + value)
  env.py            Playwright live env (Phaser bind/rAF hook + tryJump + grabs)
  collect.py        teacher rollouts from the sim
  train.py          BC on rollouts, PPO on the sim
  eval.py           sim evaluation
  play.py           live realtime runner
  device.py         cuda / mps / cpu
  remote_train.sh   Lambda / RunPod SSH: clone | up | log | pull | smoke
  tests/            unittest suite (no live site required)
```

## Setup

From the repo root (Word Hunt `.venv` is fine; this does not change server deps):

```bash
python3 -m pip install -r jumpguy/requirements.txt
python3 -m playwright install chromium   # only needed for live play
```

`torch` is required to train / run the CNN. The heuristic teacher and the sim tests only need numpy.

## How to train

Fast local smoke (CPU is enough):

```bash
python -m jumpguy collect --episodes 20 --out jumpguy/data/heuristic.npz
python -m jumpguy train --bc jumpguy/data/heuristic.npz --steps 300 --out jumpguy/runs/bc_smoke
python -m jumpguy train --ppo --init jumpguy/runs/bc_smoke/model.pt --steps 2000 --out jumpguy/runs/ppo_smoke
python -m jumpguy eval --policy heuristic --episodes 15
python -m jumpguy eval --policy cnn --ckpt jumpguy/runs/bc_smoke/model.pt --episodes 10
```

### Lambda Labs (GPU)

We start instances through the [Lambda Cloud API](https://cloud.lambda.ai/api/v1). Launch **requires an SSH key *name* that is already registered on the account** (`GET /ssh-keys` → use the `name` field). That name is not a private-key file path and must not be invented. The API rejects a launch with an unknown name.

```bash
export LAMBDA_API_KEY=...          # existing secret; never commit

# 1) List registered keys — copy one "name":
curl -sS https://cloud.lambda.ai/api/v1/ssh-keys \
  -H "Authorization: Bearer $LAMBDA_API_KEY" \
  -H "Accept: application/json" \
  -H "User-Agent: cursor-cloud-agent/lambda-cloud"

# 2) Launch (exactly one registered name):
curl -sS -X POST https://cloud.lambda.ai/api/v1/instances \
  -H "Authorization: Bearer $LAMBDA_API_KEY" \
  -H "Accept: application/json" \
  -H "Content-Type: application/json" \
  -H "User-Agent: cursor-cloud-agent/lambda-cloud" \
  -d '{"region_name":"us-west-2",
       "instance_type_name":"gpu_1x_a100_sxm4",
       "ssh_key_names":["<name-from-GET-ssh-keys>"]}'

# 3) SSH as ubuntu@<ip> with the private key that matches that registered name.
```

Default Python `urllib` User-Agent is blocked by Cloudflare 1010; send a non-default `User-Agent` as above.

#### On the box (exact clone + branch + train)

```bash
git clone https://github.com/bourkefloyd/deepmind-gradient-hackathon-2026.git
cd deepmind-gradient-hackathon-2026
git checkout cursor/jumpguy-realtime-agent-7cc0
python3 -m pip install -r jumpguy/requirements.txt
python3 -m jumpguy device cuda
python3 -m jumpguy collect --episodes 80 --out jumpguy/data/heuristic.npz
python3 -m jumpguy train --bc jumpguy/data/heuristic.npz --steps 2000 --out jumpguy/runs/bc1
python3 -m jumpguy train --ppo --init jumpguy/runs/bc1/model.pt --steps 30000 --out jumpguy/runs/ppo1
python3 -m jumpguy eval --policy cnn --ckpt jumpguy/runs/bc1/model.pt --episodes 15
```

| Artifact | What |
|---|---|
| `jumpguy/data/heuristic.npz` | teacher rollouts |
| `jumpguy/runs/bc1/model.pt` | BC weights |
| `jumpguy/runs/bc1/train.json` | BC steps / val acc / device |
| `jumpguy/runs/ppo1/model.pt` | PPO weights |
| `jumpguy/runs/ppo1/train.json` | PPO env steps / mean+max score |

#### From a laptop that already has SSH

```bash
export JUMPGUY_SSH_KEY=$HOME/.ssh/<private-key-matching-the-registered-name>
export JUMPGUY_REMOTE=ubuntu@<instance-ip>
export JUMPGUY_DEVICE=cuda
# optional:
#   WANDB_API_KEY
#   EPISODES=80
#   BC_STEPS=2000
#   PPO_STEPS=30000
#   REPO_URL=https://github.com/bourkefloyd/deepmind-gradient-hackathon-2026.git
#   REPO_BRANCH=cursor/jumpguy-realtime-agent-7cc0
#   JUMPGUY_REMOTE_DIR=~/jumpguy_ws

jumpguy/remote_train.sh clone "$JUMPGUY_REMOTE"   # git clone + checkout the branch
jumpguy/remote_train.sh up    "$JUMPGUY_REMOTE"   # or ship the jumpguy/ tree + train
jumpguy/remote_train.sh log   "$JUMPGUY_REMOTE"
jumpguy/remote_train.sh pull  "$JUMPGUY_REMOTE"   # → jumpguy/runs/remote/{bc1,ppo1}/
```

`RUNPOD_POD_ID` is document-only; same Python commands work on a RunPod box after you SSH in. `python -m jumpguy device` prints the resolved torch device.

Live play does **not** need a GPU: the privileged-state heuristic calls `scene.tryJump()` on the game origin.

## How to eval against the live game

```bash
# privileged-state teacher (Phaser hook + tryJump). No checkpoint needed.
python -m jumpguy.play --policy heuristic --episodes 2 --max-seconds 90 --hz 60

# trained CNN (needs canvas grabs)
python -m jumpguy.play --policy cnn --ckpt jumpguy/runs/bc_smoke/model.pt --episodes 1 --grab-every 1

# watch it (needs a display)
python -m jumpguy.play --policy heuristic --headed --episodes 1
```

`--submit` clicks the score overlay SUBMIT if a Viva+ session can save. The displayed name is the Viva+ username; if any text field appears we fill **`Grok Bot Son`**. Guests cannot write the global board (the game says so).

Drive **`https://game.jumpguy.net`**, not the Viva+ marketing page. The wrapper adds reCAPTCHA / HMAC and the game refuses unexpected embedders.

## Success metrics

| Metric | Where | First pass | This follow-up |
|---|---|---|---|
| Sim heuristic | `eval --policy heuristic` | mean 33.7 / 45 s cap | **mean 33.9, max 35 / 45 s cap** (12 eps, all hit the tick cap) |
| CNN BC | `eval --policy cnn` | mean 12.7, max 22 | 16-ep collect + 1000-step BC, val acc **0.905**; greedy CNN still dies on cactus 1 (over-early jumps). Live play does **not** use the CNN. |
| CPU PPO | 4000 env steps from BC | — | mean **0.27**, max 1 (too short; collapsed) |
| Live loop latency | hooked `tryJump` + state read | p50 86 ms (screenshot loop) | **p50 6.8 ms**, p95 14.7 ms, ~60 Hz |
| Live score | `game.jumpguy.net` | server **4** | **69 and 70** on two 90 s hooked runs (still `running` at the time cap); 50 s snapshot **server 39**, run `e2f54a30-3bf1-4914-94b2-2cfad56601ad` |

Leaderboard (human) scores sit around 1400. The hooked teacher now matches sim quality on the live origin (≈1 cactus / 1.2 s → ~70 in 90 s). Long sessions are the remaining training problem.

## Blockers / live-site notes

- **Do not fake `/api/runs/*/pass`**. Seq tokens are server-checked; faking a score is both against the community rules and rejected.
- **Leaderboard name** is the signed-in Viva+ username. There is no free-text “player name” box for guests. We still write `Grok Bot Son` into any input we find.
- **CORS / iframe**: play the game origin as top-level. Embedding from localhost is blocked by CSP.
- **Canvas / WebGL**: Phaser `AUTO` picks WebGL. `canvas.toDataURL()` is **black** (no `preserveDrawingBuffer`). Screenshots use the composited layer when a frame is needed. This build never calls `Phaser.GAMES.push`. The live heuristic captures `Game` via `Function.prototype.bind` / `requestAnimationFrame` (constructor does `this.boot.bind(this)` and `this.loop.start(this.step.bind(this))`), then calls `scene.tryJump()` — the same function Space uses, no 70 ms pointer debounce — so the control loop is not screenshot-bound.
- **Fallback**: if the hook misses, we tap the canvas + send Space and run the CV blob teacher.
- **Audio**: jump/score/lose oggs; ignored by the agent.
- **Offline mode**: if `POST /api/runs` fails the game still plays locally and skips submit.

## Tests

```bash
python -m unittest discover -s jumpguy/tests -v
```

No Word Hunt server, GPU, or live site required.
