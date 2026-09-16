#!/usr/bin/env bash
# Push local runs/data up to the pod (actionfleet push-runs.sh).
set -euo pipefail
# shellcheck source=env.sh
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
jumpguy_require_ssh

repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
jumpguy_ssh "$JUMPGUY_REMOTE" "mkdir -p ${JUMPGUY_REMOTE_DIR}/jumpguy/data ${JUMPGUY_REMOTE_DIR}/runs"
if [[ -d "${repo_root}/jumpguy/data" ]]; then
  rsync -az -e "ssh -i ${JUMPGUY_SSH_KEY} -p ${JUMPGUY_SSH_PORT} -o StrictHostKeyChecking=no" \
    "${repo_root}/jumpguy/data/" "$JUMPGUY_REMOTE:${JUMPGUY_REMOTE_DIR}/jumpguy/data/" || true
fi
if [[ -d "${repo_root}/jumpguy/runs" ]]; then
  rsync -az -e "ssh -i ${JUMPGUY_SSH_KEY} -p ${JUMPGUY_SSH_PORT} -o StrictHostKeyChecking=no" \
    "${repo_root}/jumpguy/runs/" "$JUMPGUY_REMOTE:${JUMPGUY_REMOTE_DIR}/runs/" || true
fi
echo "pushed local data/runs if present"
