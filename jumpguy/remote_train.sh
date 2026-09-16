#!/usr/bin/env bash
# Thin wrapper around jumpguy/runpod (ActionFleet sandbox/runpod-train mirror).
#
#   jumpguy/remote_train.sh create | clone | up | log | pull | watch | destroy | env | smoke
#
# create needs RUNPOD_API_KEY. JUMPGUY_REMOTE / JUMPGUY_SSH_KEY are filled from
# `runpodctl ssh info <id>` (see jumpguy/runpod/out/ssh.env).
# Keep clone|up|log|pull. Pods do NOT auto-stop — destroy when done.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RP="$HERE/runpod/scripts"
cmd=${1:-env}
shift || true

# shellcheck source=runpod/scripts/env.sh
source "$RP/env.sh"
# Allow `remote_train.sh up root@ip` for a pre-existing box (Lambda / already-rented).
if [[ -n "${1:-}" && "$1" == *"@"* ]]; then
  JUMPGUY_REMOTE="$1"
  shift || true
fi

case "$cmd" in
  env) jumpguy_print_env ;;
  create|up-pod) exec bash "$RP/create-pod.sh" ;;
  destroy|down|pod-down) exec bash "$RP/destroy-pod.sh" ;;
  clone|sync) exec bash "$RP/sync-code.sh" ;;
  setup) exec bash "$RP/setup-host.sh" ;;
  push-runs) exec bash "$RP/push-runs.sh" ;;
  up)
    bash "$RP/sync-code.sh"
    bash "$RP/setup-host.sh"
    exec bash "$RP/train.sh"
    ;;
  log|logs) exec bash "$RP/logs.sh" ;;
  pull) exec bash "$RP/pull-ckpt.sh" ;;
  watch) exec bash "$RP/watch-train.sh" ;;
  smoke)
    jumpguy_require_ssh
    jumpguy_ssh "$JUMPGUY_REMOTE" "cd ${JUMPGUY_REMOTE_DIR} && python3 -m jumpguy device cuda && python3 -m jumpguy collect --episodes 4 --out /tmp/jg.npz && python3 -m jumpguy train --bc /tmp/jg.npz --steps 40 --out /tmp/jg_bc"
    ;;
  *)
    echo "usage: $0 create|clone|up|log|pull|watch|destroy|env|smoke  [user@host]" >&2
    exit 2
    ;;
esac
