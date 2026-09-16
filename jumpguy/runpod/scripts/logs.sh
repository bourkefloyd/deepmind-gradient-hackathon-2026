#!/usr/bin/env bash
# Tail BC/PPO logs on the pod (actionfleet logs.sh).
set -euo pipefail
# shellcheck source=env.sh
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
jumpguy_require_ssh

jumpguy_ssh "$JUMPGUY_REMOTE" "cd ${JUMPGUY_REMOTE_DIR} && echo '== bc1' && (cat runs/bc1/train.json 2>/dev/null || cat jumpguy/runs/bc1/train.json 2>/dev/null) | tail -20; echo '== ppo1'; tail -30 runs/ppo1.log 2>/dev/null; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader || true"
