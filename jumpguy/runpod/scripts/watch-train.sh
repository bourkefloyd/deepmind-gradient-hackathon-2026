#!/usr/bin/env bash
# Poll training logs. Pods do NOT auto-stop — destroy when the run is done.
set -euo pipefail
# shellcheck source=env.sh
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
jumpguy_require_ssh

echo "watching ${JUMPGUY_REMOTE}  (Ctrl-C does not stop the pod)"
echo "when finished: make -C jumpguy/runpod destroy"
for _ in $(seq 1 120); do
  jumpguy_ssh "$JUMPGUY_REMOTE" "cd ${JUMPGUY_REMOTE_DIR} && tail -5 runs/ppo1.log 2>/dev/null || tail -5 runs/bc1/train.json 2>/dev/null || echo waiting"
  if jumpguy_ssh "$JUMPGUY_REMOTE" "test -f ${JUMPGUY_REMOTE_DIR}/runs/ppo1/train.json"; then
    echo "PPO train.json present — pull with make -C jumpguy/runpod pull"
    echo "then: make -C jumpguy/runpod destroy"
    exit 0
  fi
  sleep 30
done
echo "still running. pods do NOT auto-stop. make -C jumpguy/runpod logs / destroy"
exit 1
