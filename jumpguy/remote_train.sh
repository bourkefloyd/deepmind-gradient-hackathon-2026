#!/usr/bin/env bash
# Train Jump Guy on a CUDA box you already have SSH to.
#
# Primary path = RunPod, Action Fleet ops pattern (Blaise):
#   actionfleet has NO in-repo RunPod config. GPU is:
#     1) rent a CUDA pod on the RunPod console
#     2) SSH / rsync
#     3) train
# Same idea as ActionFleet `make train-swm SWM_DEVICE=cuda`: once the box
# exists, collect → BC → PPO on that box. This script is the Jump Guy
# equivalent (clone | up | log | pull | smoke).
#
# ═══════════════════════════════════════════════════════════════════════════
# Env (never commit secrets)
# ═══════════════════════════════════════════════════════════════════════════
#
#   RUNPOD_API_KEY          RunPod console/API key. Never commit. Optional for
#                           this script (SSH does the work). Use it only if you
#                           list pods yourself; we do not launch pods here.
#   RUNPOD_POD_ID           Pod id from the RunPod console after you rent.
#                           Document-only here; SSH uses JUMPGUY_REMOTE.
#   JUMPGUY_SSH_KEY         Path to the *private* key you added on the pod
#                           (default: $HOME/.ssh/id_ed25519)
#   JUMPGUY_REMOTE          SSH target. RunPod is often root:
#                             root@<pod-ip>
#                           If the console shows a non-22 port:
#                             JUMPGUY_SSH_PORT=xxxxx
#   JUMPGUY_DEVICE          cuda  (default when training remotely)
#   WANDB_API_KEY           optional
#   EPISODES BC_STEPS PPO_STEPS REPO_URL REPO_BRANCH JUMPGUY_REMOTE_DIR
#
# ═══════════════════════════════════════════════════════════════════════════
# 1. Rent a CUDA box (RunPod console — not an in-repo config)
# ═══════════════════════════════════════════════════════════════════════════
#
#   Open https://www.runpod.io/console/pods → Deploy
#   Pick a CUDA GPU template (PyTorch). Add your public SSH key on the pod.
#   Copy: pod id → RUNPOD_POD_ID
#         IP + port from Connect / SSH over exposed TCP
#   Then:
#     export RUNPOD_API_KEY=...          # existing secret; never commit
#     export RUNPOD_POD_ID=<pod-id>
#     export JUMPGUY_SSH_KEY=$HOME/.ssh/id_ed25519
#     export JUMPGUY_REMOTE=root@<pod-ip>
#     export JUMPGUY_SSH_PORT=22         # or the console's exposed port
#     export JUMPGUY_DEVICE=cuda
#     # optional: WANDB_API_KEY
#
# ═══════════════════════════════════════════════════════════════════════════
# 2. On the box: same sequence as ActionFleet SWM-on-CUDA
#    (make train-swm SWM_DEVICE=cuda  →  collect → BC → PPO)
# ═══════════════════════════════════════════════════════════════════════════
#
#   git clone https://github.com/bourkefloyd/deepmind-gradient-hackathon-2026.git
#   cd deepmind-gradient-hackathon-2026
#   git checkout cursor/jumpguy-realtime-agent-7cc0
#   python3 -m pip install -r jumpguy/requirements.txt
#   python3 -m jumpguy device cuda
#   python3 -m jumpguy collect --episodes 80 --out jumpguy/data/heuristic.npz
#   python3 -m jumpguy train --bc jumpguy/data/heuristic.npz --steps 2000 --out jumpguy/runs/bc1
#   python3 -m jumpguy train --ppo --init jumpguy/runs/bc1/model.pt --steps 30000 --out jumpguy/runs/ppo1
#   python3 -m jumpguy eval --policy cnn --ckpt jumpguy/runs/bc1/model.pt --episodes 15
#
# Expected artifacts:
#   jumpguy/data/heuristic.npz
#   jumpguy/runs/bc1/model.pt + train.json
#   jumpguy/runs/ppo1/model.pt + train.json
#
# ═══════════════════════════════════════════════════════════════════════════
# 3. From a laptop that already has SSH (this script)
# ═══════════════════════════════════════════════════════════════════════════
#
#   jumpguy/remote_train.sh clone "$JUMPGUY_REMOTE"
#   jumpguy/remote_train.sh up    "$JUMPGUY_REMOTE"
#   jumpguy/remote_train.sh log   "$JUMPGUY_REMOTE"
#   jumpguy/remote_train.sh pull  "$JUMPGUY_REMOTE"   # → jumpguy/runs/remote/{bc1,ppo1}/
#   jumpguy/remote_train.sh smoke "$JUMPGUY_REMOTE"
#
# Lambda Labs is secondary (same SSH flow, user often ubuntu@). See README.
set -euo pipefail
cmd=${1:?clone|up|log|pull|smoke|env}
host=${2:-${JUMPGUY_REMOTE:?pass user@host or set JUMPGUY_REMOTE (RunPod: root@<pod-ip>)}}
KEY=${JUMPGUY_SSH_KEY:-$HOME/.ssh/id_ed25519}
PORT=${JUMPGUY_SSH_PORT:-22}
SSH=(ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=15)
SCP=(scp -i "$KEY" -P "$PORT" -o StrictHostKeyChecking=no)
EPISODES=${EPISODES:-80}
BC_STEPS=${BC_STEPS:-2000}
PPO_STEPS=${PPO_STEPS:-30000}
REPO_URL=${REPO_URL:-https://github.com/bourkefloyd/deepmind-gradient-hackathon-2026.git}
REPO_BRANCH=${REPO_BRANCH:-cursor/jumpguy-realtime-agent-7cc0}
REMOTE_DIR=${JUMPGUY_REMOTE_DIR:-~/jumpguy_ws}

_print_env() {
  echo "JUMPGUY_REMOTE=${host}"
  echo "JUMPGUY_SSH_KEY=${KEY} exists=$( [[ -f "$KEY" ]] && echo yes || echo no )"
  echo "JUMPGUY_SSH_PORT=${PORT}"
  echo "JUMPGUY_DEVICE=${JUMPGUY_DEVICE:-cuda}"
  echo "JUMPGUY_REMOTE_DIR=${REMOTE_DIR}"
  echo "RUNPOD_POD_ID=${RUNPOD_POD_ID:-<unset>}"
  if [[ -n "${RUNPOD_API_KEY:-}" ]]; then
    echo "RUNPOD_API_KEY=set (value not printed)"
  else
    echo "RUNPOD_API_KEY=unset"
  fi
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    echo "WANDB_API_KEY=set (value not printed)"
  else
    echo "WANDB_API_KEY=unset"
  fi
  echo "EPISODES=${EPISODES} BC_STEPS=${BC_STEPS} PPO_STEPS=${PPO_STEPS}"
  echo "REPO_BRANCH=${REPO_BRANCH}"
}

case "$cmd" in
  env)
    _print_env
    ;;
  clone)
    _print_env
    "${SSH[@]}" "$host" bash -s <<EOF
set -euo pipefail
mkdir -p ${REMOTE_DIR}
cd ${REMOTE_DIR}
if [ -d deepmind-gradient-hackathon-2026/.git ]; then
  cd deepmind-gradient-hackathon-2026
  git fetch origin ${REPO_BRANCH}
  git checkout ${REPO_BRANCH}
  git pull --ff-only origin ${REPO_BRANCH} || true
else
  git clone --branch ${REPO_BRANCH} --single-branch ${REPO_URL} deepmind-gradient-hackathon-2026
  cd deepmind-gradient-hackathon-2026
fi
git rev-parse --abbrev-ref HEAD
git log -1 --oneline
python3 -m pip install -q -r jumpguy/requirements.txt
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
EOF
    ;;
  up)
    _print_env
    "${SSH[@]}" "$host" "mkdir -p ${REMOTE_DIR}/jumpguy ${REMOTE_DIR}/runs"
    if command -v rsync >/dev/null 2>&1; then
      rsync -az --exclude '__pycache__' --exclude '*.pyc' --exclude 'data' --exclude 'runs' \
        -e "ssh -i ${KEY} -p ${PORT} -o StrictHostKeyChecking=no -o ConnectTimeout=15" \
        jumpguy/ "$host:${REMOTE_DIR}/jumpguy/"
    else
      tar czf /tmp/jumpguy_ship.tgz jumpguy/*.py jumpguy/requirements.txt jumpguy/README.md jumpguy/remote_train.sh
      "${SCP[@]}" /tmp/jumpguy_ship.tgz "$host":${REMOTE_DIR}/
      "${SSH[@]}" "$host" "cd ${REMOTE_DIR} && tar xzf jumpguy_ship.tgz"
    fi
    "${SSH[@]}" "$host" bash -s <<EOF
set -e
cd ${REMOTE_DIR}
echo "ActionFleet parallel: make train-swm SWM_DEVICE=cuda  →  Jump Guy collect → BC → PPO"
python3 -m pip install -q -r jumpguy/requirements.txt
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
export JUMPGUY_DEVICE=\${JUMPGUY_DEVICE:-cuda}
export WANDB_API_KEY=\${WANDB_API_KEY:-}
python3 -m jumpguy collect --episodes ${EPISODES} --out jumpguy/data/heuristic.npz
python3 -m jumpguy train --bc jumpguy/data/heuristic.npz --steps ${BC_STEPS} --out runs/bc1
nohup python3 -m jumpguy train --ppo --init runs/bc1/model.pt --steps ${PPO_STEPS} --out runs/ppo1 > runs/ppo1.log 2>&1 &
echo "BC done; PPO logging to runs/ppo1.log"
tail -20 runs/bc1/train.json || true
EOF
    ;;
  log)
    "${SSH[@]}" "$host" "cd ${REMOTE_DIR} && echo '== bc1' && (cat runs/bc1/train.json 2>/dev/null || cat deepmind-gradient-hackathon-2026/jumpguy/runs/bc1/train.json 2>/dev/null) | tail -20; echo '== ppo1'; tail -30 runs/ppo1.log 2>/dev/null; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader || true"
    ;;
  pull)
    mkdir -p jumpguy/runs/remote
    "${SCP[@]}" -r "$host":${REMOTE_DIR}/runs/bc1 jumpguy/runs/remote/ || echo "missing ${REMOTE_DIR}/runs/bc1"
    "${SCP[@]}" -r "$host":${REMOTE_DIR}/runs/ppo1 jumpguy/runs/remote/ || echo "missing ${REMOTE_DIR}/runs/ppo1"
    "${SCP[@]}" -r "$host":${REMOTE_DIR}/deepmind-gradient-hackathon-2026/jumpguy/runs/bc1 jumpguy/runs/remote/ 2>/dev/null || true
    "${SCP[@]}" -r "$host":${REMOTE_DIR}/deepmind-gradient-hackathon-2026/jumpguy/runs/ppo1 jumpguy/runs/remote/ 2>/dev/null || true
    ls -la jumpguy/runs/remote/*/model.pt 2>/dev/null || true
    ;;
  smoke)
    "${SSH[@]}" "$host" "cd ${REMOTE_DIR} && python3 -m jumpguy device cuda && python3 -m jumpguy collect --episodes 4 --out /tmp/jg.npz && python3 -m jumpguy train --bc /tmp/jg.npz --steps 40 --out /tmp/jg_bc"
    ;;
  *)
    echo "unknown cmd $cmd (clone|up|log|pull|smoke|env)" >&2
    exit 2
    ;;
esac
