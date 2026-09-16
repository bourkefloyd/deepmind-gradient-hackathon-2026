#!/usr/bin/env bash
# Train Jump Guy on a Lambda Labs (or RunPod) GPU box you already have SSH to.
#
# ═══════════════════════════════════════════════════════════════════════════
# Lambda Labs: how we start instances (API; SSH key *name*, not a file)
# ═══════════════════════════════════════════════════════════════════════════
#
# Instances are launched with POST https://cloud.lambda.ai/api/v1/instances
# The body requires ssh_key_names: an array with exactly ONE name that is
# already registered on the Lambda account. That name is NOT a private-key
# path and must not be invented.
#
#   export LAMBDA_API_KEY=...          # existing secret; never commit
#   # 1) List registered keys — use the "name" field:
#   curl -sS https://cloud.lambda.ai/api/v1/ssh-keys \
#     -H "Authorization: Bearer $LAMBDA_API_KEY" \
#     -H "Accept: application/json" \
#     -H "User-Agent: cursor-cloud-agent/lambda-cloud"
#   # 2) Launch (example). ssh_key_names must match a name from step 1:
#   curl -sS -X POST https://cloud.lambda.ai/api/v1/instances \
#     -H "Authorization: Bearer $LAMBDA_API_KEY" \
#     -H "Accept: application/json" \
#     -H "Content-Type: application/json" \
#     -H "User-Agent: cursor-cloud-agent/lambda-cloud" \
#     -d '{"region_name":"us-west-2",
#          "instance_type_name":"gpu_1x_a100_sxm4",
#          "ssh_key_names":["<name-from-GET-ssh-keys>"]}'
#   # 3) SSH as ubuntu@<ip> with the *private* key that matches that name.
#
# ═══════════════════════════════════════════════════════════════════════════
# On the box: exact clone + train (copy-paste)
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
# Expected artifacts on the box:
#   jumpguy/data/heuristic.npz          teacher rollouts
#   jumpguy/runs/bc1/model.pt           BC weights
#   jumpguy/runs/bc1/train.json         BC metrics
#   jumpguy/runs/ppo1/model.pt          PPO weights
#   jumpguy/runs/ppo1/train.json        PPO metrics
#
# ═══════════════════════════════════════════════════════════════════════════
# From a laptop that already has SSH (this script)
# ═══════════════════════════════════════════════════════════════════════════
#
#   export JUMPGUY_SSH_KEY=$HOME/.ssh/<private-key-matching-the-registered-name>
#   export JUMPGUY_REMOTE=ubuntu@<instance-ip>
#   export JUMPGUY_DEVICE=cuda          # optional; default cuda
#   # optional: WANDB_API_KEY  EPISODES=80  BC_STEPS=2000  PPO_STEPS=30000
#   # optional: REPO_URL  REPO_BRANCH  (defaults below)
#
#   jumpguy/remote_train.sh clone "$JUMPGUY_REMOTE"   # git clone + checkout
#   jumpguy/remote_train.sh up    "$JUMPGUY_REMOTE"   # ship tree + collect/BC + PPO
#   jumpguy/remote_train.sh log   "$JUMPGUY_REMOTE"
#   jumpguy/remote_train.sh pull  "$JUMPGUY_REMOTE"   # → jumpguy/runs/remote/{bc1,ppo1}/
#   jumpguy/remote_train.sh smoke "$JUMPGUY_REMOTE"
#
# RUNPOD_POD_ID is document-only; SSH to the pod the same way after you have a host.
set -euo pipefail
cmd=${1:?clone|up|log|pull|smoke}
host=${2:-${JUMPGUY_REMOTE:?pass user@host or set JUMPGUY_REMOTE}}
KEY=${JUMPGUY_SSH_KEY:-$HOME/.ssh/id_ed25519}
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=15)
SCP=(scp -i "$KEY" -o StrictHostKeyChecking=no)
EPISODES=${EPISODES:-80}
BC_STEPS=${BC_STEPS:-2000}
PPO_STEPS=${PPO_STEPS:-30000}
REPO_URL=${REPO_URL:-https://github.com/bourkefloyd/deepmind-gradient-hackathon-2026.git}
REPO_BRANCH=${REPO_BRANCH:-cursor/jumpguy-realtime-agent-7cc0}
REMOTE_DIR=${JUMPGUY_REMOTE_DIR:-~/jumpguy_ws}

case "$cmd" in
  clone)
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
    tar czf /tmp/jumpguy_ship.tgz jumpguy/*.py jumpguy/requirements.txt jumpguy/README.md jumpguy/remote_train.sh
    "${SSH[@]}" "$host" "mkdir -p ${REMOTE_DIR}/jumpguy ${REMOTE_DIR}/runs"
    "${SCP[@]}" /tmp/jumpguy_ship.tgz "$host":${REMOTE_DIR}/
    "${SSH[@]}" "$host" bash -s <<EOF
set -e
cd ${REMOTE_DIR} && tar xzf jumpguy_ship.tgz
echo "branch-hint ${REPO_BRANCH}  (this tarball is the jumpguy/ tree, not a full clone)"
echo "prefer: jumpguy/remote_train.sh clone  — then run the python commands in the clone"
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
    echo "unknown cmd $cmd (clone|up|log|pull|smoke)" >&2
    exit 2
    ;;
esac
