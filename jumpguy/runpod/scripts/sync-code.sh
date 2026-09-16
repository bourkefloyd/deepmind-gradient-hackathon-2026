#!/usr/bin/env bash
# rsync jumpguy/ onto the pod. ActionFleet: sandbox/runpod-train/scripts/sync-code.sh
set -euo pipefail
# shellcheck source=env.sh
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
jumpguy_require_ssh

repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
jumpguy_ssh "$JUMPGUY_REMOTE" "mkdir -p ${JUMPGUY_REMOTE_DIR}/jumpguy ${JUMPGUY_REMOTE_DIR}/runs"
if command -v rsync >/dev/null 2>&1; then
  jumpguy_rsync "${repo_root}/jumpguy/" "$JUMPGUY_REMOTE:${JUMPGUY_REMOTE_DIR}/jumpguy/"
else
  tar czf /tmp/jumpguy_ship.tgz -C "$repo_root" jumpguy/*.py jumpguy/requirements.txt jumpguy/README.md jumpguy/remote_train.sh
  jumpguy_scp /tmp/jumpguy_ship.tgz "$JUMPGUY_REMOTE:${JUMPGUY_REMOTE_DIR}/"
  jumpguy_ssh "$JUMPGUY_REMOTE" "cd ${JUMPGUY_REMOTE_DIR} && tar xzf jumpguy_ship.tgz"
fi
echo "synced jumpguy/ → ${JUMPGUY_REMOTE}:${JUMPGUY_REMOTE_DIR}/jumpguy/"
echo "branch-hint ${REPO_BRANCH} (rsync of jumpguy/, not a full clone)"
echo "full clone: git clone --branch ${REPO_BRANCH} ${REPO_URL}"
