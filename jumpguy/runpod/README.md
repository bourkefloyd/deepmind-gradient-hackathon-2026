# Jump Guy RunPod train (ActionFleet mirror)

Mirrors **actionfleet `sandbox/runpod-train/`** (main, commit `9a87b5d`, Blaise).

Hubs in bourkefloyd/actionfleet:

1. SWM: `sandbox/runpod-train/` — `make train-swm-runpod` → `make -C sandbox/runpod-train train`
2. nanoagent: `sandbox/nanoagent/scripts/runpod_train.sh` + `pod_serve.sh`

Jump Guy equivalent: `make -C jumpguy/runpod train` after `create`.

This hackathon repo could not fetch actionfleet (private). The scripts here follow that layout and the documented CLI/defaults — they do not invent other template IDs.

## Layout

```
jumpguy/runpod/
  Makefile
  README.md
  scripts/env.sh
  scripts/create-pod.sh
  scripts/destroy-pod.sh
  scripts/setup-host.sh
  scripts/sync-code.sh
  scripts/push-runs.sh
  scripts/train.sh
  scripts/logs.sh
  scripts/pull-ckpt.sh
  scripts/watch-train.sh
  out/                 gitignored state
    pod-id
    volume-id
    ssh.env            JUMPGUY_REMOTE / JUMPGUY_SSH_PORT / JUMPGUY_SSH_KEY
```

## Env / secret names (no values)

| Name | Role |
|---|---|
| `RUNPOD_API_KEY` | Required to create/destroy. **Never commit.** |
| `RUNPOD_SSH_KEY` | Cursor cloud secret → `~/.ssh/runpod_nano` (or a key path) |
| `RUNPOD_SSH_IDENTITY` | Path override for the private key |
| `JUMPGUY_SSH_KEY` | Resolved identity (see below) |
| `JUMPGUY_REMOTE` | From `runpodctl ssh info <id>` — usually `root@<ip>` |
| `JUMPGUY_SSH_PORT` | From `runpodctl ssh info <id>` |
| `WANDB_API_KEY` | optional |
| `JUMPGUY_DEVICE` | `cuda` |

SSH filenames (ActionFleet):

- `~/.ssh/runpod_ed25519` — SWM, nicknamed **actionfleet-runpod**
- `~/.ssh/runpod_nano` — nanoagent / Cursor `RUNPOD_SSH_KEY` bootstrap

`scripts/env.sh` picks `RUNPOD_SSH_IDENTITY`, else a path in `RUNPOD_SSH_KEY`, else those two files, else `id_ed25519`. It never prints key material.

## Non-secret defaults (from actionfleet)

| | |
|---|---|
| Template | `runpod-torch-v280` (min CUDA 12.8) — do not invent another id |
| GPU | `NVIDIA GeForce RTX 4090` (~$0.74/hr). A100 80GB is later nano records only |
| Cloud | `SECURE` |
| Region | `EU-RO-1` (SWM) |
| Pod name | `jumpguy-train` |
| Volume | 50 GB at `/workspace` (attach `JUMPGUY_VOLUME_ID` if you already have one, e.g. SWM `af-swm-data`) |
| CLI | `runpodctl config --apiKey`, `pod create/get/delete`, `ssh info/add-key` |

**Pods do NOT auto-stop.** Use `make watch` / `make destroy` (`pod-down`).

## Flow

```bash
export RUNPOD_API_KEY=...            # existing secret
# optional: RUNPOD_SSH_KEY / RUNPOD_SSH_IDENTITY

make -C jumpguy/runpod env
make -C jumpguy/runpod create        # writes out/pod-id + out/ssh.env from ssh info
make -C jumpguy/runpod sync          # rsync jumpguy/
make -C jumpguy/runpod train         # collect → BC → PPO   (SWM: make train-swm SWM_DEVICE=cuda)
make -C jumpguy/runpod logs
make -C jumpguy/runpod pull          # → jumpguy/runs/remote/{bc1,ppo1}/
make -C jumpguy/runpod watch
make -C jumpguy/runpod destroy
```

Thin wrappers (same names as before): `jumpguy/remote_train.sh create|clone|up|log|pull|watch|destroy|env`

After `create`, `JUMPGUY_REMOTE` / `JUMPGUY_SSH_KEY` come from `runpodctl ssh info <id>` (saved in `out/ssh.env`). You can also set them by hand:

```bash
runpodctl ssh info "$RUNPOD_POD_ID"
export JUMPGUY_REMOTE=root@<ip>
export JUMPGUY_SSH_PORT=<port>
export JUMPGUY_SSH_KEY=$HOME/.ssh/runpod_ed25519   # or runpod_nano
```

On the box, train is:

```bash
python3 -m jumpguy collect --episodes 80 --out jumpguy/data/heuristic.npz
python3 -m jumpguy train --bc jumpguy/data/heuristic.npz --steps 2000 --out runs/bc1
python3 -m jumpguy train --ppo --init runs/bc1/model.pt --steps 30000 --out runs/ppo1
```

Expected artifacts: `runs/bc1/{model.pt,train.json}`, `runs/ppo1/{model.pt,train.json}`.
