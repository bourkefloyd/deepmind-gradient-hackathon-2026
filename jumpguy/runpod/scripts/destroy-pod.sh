#!/usr/bin/env bash
# Tear down the Jump Guy train pod. Pods do NOT auto-stop.
set -euo pipefail
# shellcheck source=env.sh
source "$(cd "$(dirname "$0")" && pwd)/env.sh"

jumpguy_require_api_key
pod_id="${RUNPOD_POD_ID:-}"
if [[ -z "$pod_id" ]]; then
  echo "no RUNPOD_POD_ID / out/pod-id — nothing to destroy"
  exit 0
fi
echo "deleting pod ${pod_id} (${JUMPGUY_POD_NAME})"
if runpodctl pod delete --help >/dev/null 2>&1; then
  runpodctl pod delete "$pod_id" || runpodctl pod delete --pod-id "$pod_id" || true
else
  runpodctl remove pod "$pod_id" || true
fi
rm -f "$JUMPGUY_RUNPOD_OUT/pod-id" "$JUMPGUY_RUNPOD_OUT/ssh.env"
echo "cleared $JUMPGUY_RUNPOD_OUT/pod-id and ssh.env (volume-id kept if present)"
