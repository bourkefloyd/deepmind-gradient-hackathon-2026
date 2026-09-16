#!/usr/bin/env bash
# Pull checkpoints back to the laptop (actionfleet pull-ckpt.sh).
set -euo pipefail
# shellcheck source=env.sh
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
jumpguy_require_ssh

repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
dest="${repo_root}/jumpguy/runs/remote"
mkdir -p "$dest"
jumpguy_scp -r "$JUMPGUY_REMOTE:${JUMPGUY_REMOTE_DIR}/runs/bc1" "$dest/" || echo "missing remote runs/bc1"
jumpguy_scp -r "$JUMPGUY_REMOTE:${JUMPGUY_REMOTE_DIR}/runs/ppo1" "$dest/" || echo "missing remote runs/ppo1"
ls -la "$dest"/*/model.pt 2>/dev/null || true
