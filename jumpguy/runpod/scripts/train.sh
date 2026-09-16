#!/usr/bin/env bash
# collect → BC → PPO on the CUDA box.
# ActionFleet parallel: make train-swm SWM_DEVICE=cuda
#   (make train-swm-runpod → make -C sandbox/runpod-train train)
set -euo pipefail
# shellcheck source=env.sh
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
jumpguy_require_ssh

jumpguy_ssh "$JUMPGUY_REMOTE" bash -s <<EOF
set -e
cd ${JUMPGUY_REMOTE_DIR}
echo "ActionFleet parallel: make train-swm SWM_DEVICE=cuda  →  Jump Guy collect → BC → PPO"
python3 -m pip install -q -r jumpguy/requirements.txt
python3 -m jumpguy device cuda || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
export JUMPGUY_DEVICE=\${JUMPGUY_DEVICE:-cuda}
export WANDB_API_KEY=\${WANDB_API_KEY:-}
python3 -m jumpguy collect --episodes ${EPISODES} --out jumpguy/data/heuristic.npz
python3 -m jumpguy train --bc jumpguy/data/heuristic.npz --steps ${BC_STEPS} --out runs/bc1
mkdir -p runs
nohup python3 -m jumpguy train --ppo --init runs/bc1/model.pt --steps ${PPO_STEPS} --out runs/ppo1 > runs/ppo1.log 2>&1 &
echo "BC done; PPO logging to runs/ppo1.log"
echo "pods do NOT auto-stop — make -C jumpguy/runpod watch  then destroy"
tail -20 runs/bc1/train.json || true
EOF
