#!/usr/bin/env bash
# Ship jumpguy/ to a remote GPU box (Lambda Cloud or RunPod) and train.
#
#   export JUMPGUY_SSH_KEY=$HOME/.ssh/lambda_nano    # or your RunPod key
#   jumpguy/remote_train.sh up   ubuntu@<ip>
#   jumpguy/remote_train.sh log  ubuntu@<ip>
#   jumpguy/remote_train.sh pull ubuntu@<ip>
#
# Hooks (do not invent credentials; leave empty if unused):
#   JUMPGUY_SSH_KEY   SSH private key path
#   JUMPGUY_REMOTE    default user@host (optional if you pass it as $2)
#   RUNPOD_POD_ID     unused here; document-only. SSH to the pod yourself.
#   WANDB_API_KEY     optional; forwarded if set (trainer does not require it)
#   JUMPGUY_DEVICE    cuda|mps|cpu (default: auto)
set -euo pipefail
cmd=${1:?up|log|pull|smoke}
host=${2:-${JUMPGUY_REMOTE:?pass user@host or set JUMPGUY_REMOTE}}
KEY=${JUMPGUY_SSH_KEY:-$HOME/.ssh/id_ed25519}
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=15)
SCP=(scp -i "$KEY" -o StrictHostKeyChecking=no)
EPISODES=${EPISODES:-80}
BC_STEPS=${BC_STEPS:-2000}
PPO_STEPS=${PPO_STEPS:-30000}

case "$cmd" in
  up)
    tar czf /tmp/jumpguy_ship.tgz jumpguy/*.py jumpguy/requirements.txt jumpguy/README.md jumpguy/remote_train.sh
    "${SSH[@]}" "$host" 'mkdir -p ~/jumpguy_ws/jumpguy ~/jumpguy_ws/runs'
    "${SCP[@]}" /tmp/jumpguy_ship.tgz "$host":~/jumpguy_ws/
    "${SSH[@]}" "$host" bash -s <<EOF
set -e
cd ~/jumpguy_ws && tar xzf jumpguy_ship.tgz
python3 -m pip install -q -r jumpguy/requirements.txt
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
export JUMPGUY_DEVICE=\${JUMPGUY_DEVICE:-cuda}
export WANDB_API_KEY=\${WANDB_API_KEY:-}
python3 -m jumpguy collect --episodes ${EPISODES} --out jumpguy/data/heuristic.npz
nohup python3 -m jumpguy train --bc jumpguy/data/heuristic.npz --steps ${BC_STEPS} --out runs/bc1 > runs/bc1.log 2>&1
nohup python3 -m jumpguy train --ppo --init runs/bc1/model.pt --steps ${PPO_STEPS} --out runs/ppo1 > runs/ppo1.log 2>&1 &
sleep 5
tail -20 runs/bc1.log || true
EOF
    ;;
  log)
    "${SSH[@]}" "$host" 'cd ~/jumpguy_ws && tail -30 runs/bc1.log runs/ppo1.log 2>/dev/null; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader || true'
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
