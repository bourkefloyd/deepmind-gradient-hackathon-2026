# Jump Guy realtime agent

A local control loop that plays [Jump Guy](https://game.jumpguy.net) at game speed: **frame in → discrete jump/noop out**. Training happens on a physics clone of the live Phaser scene; evaluation can target the live site.

This package is isolated from Word Hunt (`wordhunt/`, `nano/`, `gemma_seat/`). It reuses the same ideas (tiny policy, checkpoint JSON, SSH train script) without touching those paths. Remote GPU follows the Action Fleet pattern: rent a CUDA box on the **RunPod console**, then SSH/rsync and train. There is no in-repo RunPod config.

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
  remote_train.sh   RunPod-first SSH: clone | up | log | pull | smoke | env
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

### RunPod (primary GPU path — Action Fleet ops)

actionfleet has **no in-repo RunPod config**. GPU work is: rent a CUDA pod on the [RunPod console](https://www.runpod.io/console/pods), then SSH/rsync and train. Same pattern as ActionFleet `make train-swm SWM_DEVICE=cuda`: once the box exists, run collect → BC → PPO on it.

| Env | Required | What |
|---|---|---|
| `RUNPOD_API_KEY` | console/API only | Existing secret. **Never commit.** This script does not launch pods. |
| `RUNPOD_POD_ID` | after you rent | Pod id from the console. Document-only; SSH uses `JUMPGUY_REMOTE`. |
| `JUMPGUY_SSH_KEY` | yes | Path to the **private** key you added on the pod (e.g. `$HOME/.ssh/id_ed25519`) |
| `JUMPGUY_REMOTE` | yes | RunPod is often **`root@<pod-ip>`** |
| `JUMPGUY_SSH_PORT` | if not 22 | Exposed SSH port from the pod Connect panel |
| `JUMPGUY_DEVICE` | recommended | `cuda` |
| `WANDB_API_KEY` | optional | W&B |

```bash
# 1) Rent a CUDA / PyTorch pod on the RunPod console. Add your public SSH key.
# 2) Copy pod id, IP, and port from Connect.

export RUNPOD_API_KEY=...                 # existing secret; never commit
export RUNPOD_POD_ID=<pod-id>
export JUMPGUY_SSH_KEY=$HOME/.ssh/id_ed25519
export JUMPGUY_REMOTE=root@<pod-ip>      # RunPod often uses root
export JUMPGUY_SSH_PORT=22                # or the console's exposed port
export JUMPGUY_DEVICE=cuda
# optional: WANDB_API_KEY  EPISODES=80  BC_STEPS=2000  PPO_STEPS=30000
```

On the box (or via `jumpguy/remote_train.sh clone`):

```bash
git clone https://github.com/bourkefloyd/deepmind-gradient-hackathon-2026.git
cd deepmind-gradient-hackathon-2026
git checkout cursor/jumpguy-realtime-agent-7cc0
python3 -m pip install -r jumpguy/requirements.txt
python3 -m jumpguy device cuda
# ActionFleet parallel: make train-swm SWM_DEVICE=cuda
python3 -m jumpguy collect --episodes 80 --out jumpguy/data/heuristic.npz
python3 -m jumpguy train --bc jumpguy/data/heuristic.npz --steps 2000 --out jumpguy/runs/bc1
python3 -m jumpguy train --ppo --init jumpguy/runs/bc1/model.pt --steps 30000 --out jumpguy/runs/ppo1
python3 -m jumpguy eval --policy cnn --ckpt jumpguy/runs/bc1/model.pt --episodes 15
```

| Artifact | What |
|---|---|
| `jumpguy/data/heuristic.npz` | teacher rollouts |
| `jumpguy/runs/bc1/model.pt` | BC weights |
| `jumpguy/runs/bc1/train.json` | BC steps / val acc / sim score |
| `jumpguy/runs/ppo1/model.pt` | PPO weights |
| `jumpguy/runs/ppo1/train.json` | PPO env steps / mean+max score |

From a laptop that already has SSH:

```bash
jumpguy/remote_train.sh env   "$JUMPGUY_REMOTE"   # prints env; never prints secret values
jumpguy/remote_train.sh clone "$JUMPGUY_REMOTE"   # git clone + checkout the branch
jumpguy/remote_train.sh up    "$JUMPGUY_REMOTE"   # rsync jumpguy/ + collect/BC + PPO
jumpguy/remote_train.sh log   "$JUMPGUY_REMOTE"
jumpguy/remote_train.sh pull  "$JUMPGUY_REMOTE"   # → jumpguy/runs/remote/{bc1,ppo1}/
```

`python -m jumpguy device` prints the resolved torch device. Live play does **not** need a GPU: the privileged-state heuristic calls `scene.tryJump()` on the game origin.

### Lambda Labs (secondary)

Same SSH train flow if you already have a Lambda box (`JUMPGUY_REMOTE=ubuntu@<ip>`). Launch still needs a **registered SSH key name** on the account (`GET /ssh-keys` → `ssh_key_names` on `POST /instances`). Prefer RunPod for this package.

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
