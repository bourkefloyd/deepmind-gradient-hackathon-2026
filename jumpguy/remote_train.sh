#!/usr/bin/env bash
# Train Jump Guy on a Lambda Labs (or RunPod) GPU box that you already have SSH to.
#
# ── 1. Start a Lambda instance (API; do this on a machine that has LAMBDA_API_KEY) ──
# The launch body needs an SSH key *name already registered on the Lambda account*
# (GET https://cloud.lambda.ai/api/v1/ssh-keys). That name is not the private-key
# file path. Do not invent a name; list keys first.
#
#   export LAMBDA_API_KEY=...          # existing account secret
#   # GET /ssh-keys  → pick one "name"
#   # POST /instances  { region_name, instance_type_name, ssh_key_names: ["<that-name>"] }
#   # See jumpguy/README.md § "Lambda Labs"
#
# ── 2. On the box (Ubuntu, Lambda Stack torch+CUDA) ──
#   git clone https://github.com/bourkefloyd/deepmind-gradient-hackathon-2026.git
#   cd deepmind-gradient-hackathon-2026
#   git checkout cursor/jumpguy-realtime-agent-7cc0
#   python3 -m pip install -r jumpguy/requirements.txt
#
# ── 3. From your laptop, ship + train (or just run the python commands on the box) ──
#   export JUMPGUY_SSH_KEY=$HOME/.ssh/<private-key-that-matches-the-registered-name>
#   export JUMPGUY_REMOTE=ubuntu@<instance-ip>
#   jumpguy/remote_train.sh up   "$JUMPGUY_REMOTE"
#   jumpguy/remote_train.sh log  "$JUMPGUY_REMOTE"
#   jumpguy/remote_train.sh pull "$JUMPGUY_REMOTE"
#
# Expected artifacts after pull:
#   jumpguy/runs/remote/bc1/model.pt + train.json
#   jumpguy/runs/remote/ppo1/model.pt + train.json
#
# Optional env: JUMPGUY_DEVICE=cuda  WANDB_API_KEY=  EPISODES=80  BC_STEPS=2000  PPO_STEPS=30000
# RUNPOD_POD_ID is document-only; SSH to the pod the same way.
set -euo pipefail
cmd=${1:?up|log|pull|smoke}
host=${2:-${JUMPGUY_REMOTE:?pass user@host or set JUMPGUY_REMOTE}}
KEY=${JUMPGUY_SSH_KEY:-$HOME/.ssh/id_ed25519}
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=15)
SCP=(scp -i "$KEY" -o StrictHostKeyChecking=no)
EPISODES=${EPISODES:-80}
BC_STEPS=${BC_STEPS:-2000}
PPO_STEPS=${PPO_STEPS:-30000}
REPO_BRANCH=${REPO_BRANCH:-cursor/jumpguy-realtime-agent-7cc0}

case "$cmd" in
  up)
    tar czf /tmp/jumpguy_ship.tgz jumpguy/*.py jumpguy/requirements.txt jumpguy/README.md jumpguy/remote_train.sh
    "${SSH[@]}" "$host" 'mkdir -p ~/jumpguy_ws/jumpguy ~/jumpguy_ws/runs'
    "${SCP[@]}" /tmp/jumpguy_ship.tgz "$host":~/jumpguy_ws/
    "${SSH[@]}" "$host" bash -s <<EOF
set -e
cd ~/jumpguy_ws && tar xzf jumpguy_ship.tgz
echo "branch-hint ${REPO_BRANCH}  (this tarball is the jumpguy/ tree, not a full clone)"
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
    "${SSH[@]}" "$host" 'cd ~/jumpguy_ws && echo "== bc1"; cat runs/bc1/train.json 2>/dev/null | tail -20; echo "== ppo1"; tail -30 runs/ppo1.log 2>/dev/null; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader || true'
    ;;
  pull)
    mkdir -p jumpguy/runs/remote
    "${SCP[@]}" -r "$host":~/jumpguy_ws/runs/bc1 jumpguy/runs/remote/ || echo "missing bc1"
    "${SCP[@]}" -r "$host":~/jumpguy_ws/runs/ppo1 jumpguy/runs/remote/ || echo "missing ppo1"
    ls -la jumpguy/runs/remote/*/model.pt 2>/dev/null || true
    ;;
  smoke)
    "${SSH[@]}" "$host" 'cd ~/jumpguy_ws && python3 -m jumpguy device cuda && python3 -m jumpguy collect --episodes 4 --out /tmp/jg.npz && python3 -m jumpguy train --bc /tmp/jg.npz --steps 40 --out /tmp/jg_bc'
    ;;
  *)
    echo "unknown cmd $cmd" >&2
    exit 2
    ;;
esac
