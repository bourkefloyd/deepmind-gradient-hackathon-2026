#!/usr/bin/env bash
# Install Jump Guy deps on the pod (actionfleet setup-host.sh equivalent).
set -euo pipefail
# shellcheck source=env.sh
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
jumpguy_require_ssh

jumpguy_ssh "$JUMPGUY_REMOTE" bash -s <<EOF
set -euo pipefail
cd ${JUMPGUY_REMOTE_DIR}
python3 -m pip install -q -r jumpguy/requirements.txt
python3 -m jumpguy device cuda || python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
EOF
