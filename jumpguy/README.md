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
  env.py            Playwright live env (canvas grab + /api/runs hooks)
  collect.py        teacher rollouts from the sim
  train.py          BC on rollouts, PPO on the sim
  eval.py           sim evaluation
  play.py           live realtime runner
  device.py         cuda / mps / cpu
  remote_train.sh   Lambda / RunPod SSH: up | log | pull | smoke
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

### Remote GPU (Lambda Cloud or RunPod)

Same pattern as `nano/lambda_train.sh`. **Do not invent keys.** Use Bourke’s existing Lambda SSH key / RunPod pod.

```bash
export JUMPGUY_SSH_KEY=$HOME/.ssh/lambda_nano    # or your RunPod key
export JUMPGUY_REMOTE=ubuntu@<gpu-ip>            # Lambda: ubuntu@<ip>
# optional
export WANDB_API_KEY=                            # unused unless you add wandb
export JUMPGUY_DEVICE=cuda
export RUNPOD_POD_ID=                            # document-only; SSH to the pod

jumpguy/remote_train.sh up   "$JUMPGUY_REMOTE"   # ship + collect + BC + PPO
jumpguy/remote_train.sh log  "$JUMPGUY_REMOTE"
jumpguy/remote_train.sh pull "$JUMPGUY_REMOTE"   # -> jumpguy/runs/remote/
```

On the box the script runs:

1. `python -m jumpguy collect` (heuristic teacher on the sim, rendered frames)
2. `python -m jumpguy train --bc` (imitation)
3. `python -m jumpguy train --ppo --init …` (on-policy refinement)

A100 / 4090: keep `--steps` in the 20k–100k env-step range for PPO. The CNN is ~0.4 M params at `width=32`; a BC pass of a few thousand steps is seconds-to-minutes.

`python -m jumpguy device` prints the resolved torch device.

## How to eval against the live game

```bash
# heuristic CV controller (no checkpoint needed)
python -m jumpguy.play --policy heuristic --episodes 2 --max-seconds 45 --hz 30

# trained CNN
python -m jumpguy.play --policy cnn --ckpt jumpguy/runs/bc_smoke/model.pt --episodes 2

# watch it (needs a display)
python -m jumpguy.play --policy heuristic --headed --episodes 1
```

`--submit` clicks the score overlay SUBMIT if a Viva+ session can save. The displayed name is the Viva+ username; if any text field appears we fill **`Grok Bot Son`**. Guests cannot write the global board (the game says so).

Drive **`https://game.jumpguy.net`**, not the Viva+ marketing page. The wrapper adds reCAPTCHA / HMAC and the game refuses unexpected embedders.

## Success metrics

| Metric | Where | Measured (this PR) |
|---|---|---|
| Sim heuristic | `eval --policy heuristic` | **mean 33.7 / 45 s** (hits the tick cap; ~21k actions/s) |
| CNN after balanced BC | `eval --policy cnn` | **mean 12.7, max 22** on 6 sim episodes (weak, scoring) |
| Live loop latency | Playwright canvas screenshot | **p50 ~86 ms, p95 ~106 ms** (~12 Hz including grab) |
| Live score | `POST /api/runs/*/pass` | **server score 4** on a headless heuristic run |
| CNN BC val acc | balanced batches | **~0.96** (without balancing the net collapses to NOOP) |

Leaderboard (human) scores sit around 1400. Getting there is a training problem, not a missing env.

## Blockers / live-site notes

- **Do not fake `/api/runs/*/pass`**. Seq tokens are server-checked; faking a score is both against the community rules and rejected.
- **Leaderboard name** is the signed-in Viva+ username. There is no free-text “player name” box for guests. We still write `Grok Bot Son` into any input we find.
- **CORS / iframe**: play the game origin as top-level. Embedding from localhost is blocked by CSP.
- **Canvas / WebGL**: Phaser `AUTO` picks WebGL. `canvas.toDataURL()` is **black** (no `preserveDrawingBuffer`). The live env screenshots the composited canvas instead. Keyboard-only Space is flaky until focus; we tap the canvas (the game’s real pointer path) and also send Space.
- **Audio**: jump/score/lose oggs; ignored by the agent.
- **Offline mode**: if `POST /api/runs` fails the game still plays locally and skips submit.

## Tests

```bash
python -m unittest discover -s jumpguy/tests -v
```

No Word Hunt server, GPU, or live site required.
