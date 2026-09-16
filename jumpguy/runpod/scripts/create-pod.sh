#!/usr/bin/env bash
# Create a Jump Guy train pod. Mirrors actionfleet sandbox/runpod-train/scripts/create-pod.sh
# Template/GPU defaults are from actionfleet main — do not invent other template IDs.
set -euo pipefail
# shellcheck source=env.sh
source "$(cd "$(dirname "$0")" && pwd)/env.sh"

jumpguy_require_api_key
jumpguy_print_env

if [[ -f "$JUMPGUY_RUNPOD_OUT/pod-id" ]]; then
  echo "pod-id already exists: $(cat "$JUMPGUY_RUNPOD_OUT/pod-id")"
  echo "destroy first (make -C jumpguy/runpod destroy) or reuse ssh.env"
  exit 0
fi

# ActionFleet CLI: runpodctl pod create/get/delete + ssh info/add-key
# Fall back to the older `create pod` surface if this runpodctl is older.
_create() {
  local args=(
    --name "$JUMPGUY_POD_NAME"
    --template-id "$JUMPGUY_RUNPOD_TEMPLATE"
    --gpu-id "$JUMPGUY_RUNPOD_GPU"
    --ports "22/tcp"
    --volume-mount-path "$JUMPGUY_VOLUME_MOUNT"
  )
  if [[ -n "${JUMPGUY_VOLUME_ID:-}" ]]; then
    args+=(--network-volume-id "$JUMPGUY_VOLUME_ID")
  else
    args+=(--volume-in-gb "$JUMPGUY_VOLUME_GB")
  fi
  if [[ "${JUMPGUY_RUNPOD_CLOUD}" == "SECURE" ]]; then
    args+=(--secure-cloud)
  fi
  if runpodctl pod create --help 2>&1 | grep -q -- '--data-center'; then
    args+=(--data-center-id "$JUMPGUY_RUNPOD_DC")
  fi
  if runpodctl pod create --help >/dev/null 2>&1; then
    runpodctl pod create "${args[@]}"
    return
  fi
  # Older runpodctl: create pod --gpuType --templateId --secureCloud
  local old=(
    --name "$JUMPGUY_POD_NAME"
    --gpuType "$JUMPGUY_RUNPOD_GPU"
    --templateId "$JUMPGUY_RUNPOD_TEMPLATE"
    --ports "22/tcp"
    --volumePath "$JUMPGUY_VOLUME_MOUNT"
    --volumeSize "$JUMPGUY_VOLUME_GB"
  )
  if [[ "${JUMPGUY_RUNPOD_CLOUD}" == "SECURE" ]]; then
    old+=(--secureCloud)
  fi
  runpodctl create pod "${old[@]}"
}

echo "creating pod ${JUMPGUY_POD_NAME} template=${JUMPGUY_RUNPOD_TEMPLATE} gpu=${JUMPGUY_RUNPOD_GPU}"
out="$(_create)"
printf '%s\n' "$out"
# Parse a pod id (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx or similar).
pod_id="$(printf '%s\n' "$out" | grep -Eo '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' | head -1 || true)"
if [[ -z "$pod_id" ]]; then
  pod_id="$(printf '%s\n' "$out" | awk '/id/ {print $NF; exit}')"
fi
if [[ -z "$pod_id" ]]; then
  echo "could not parse pod id from runpodctl output" >&2
  exit 1
fi
printf '%s\n' "$pod_id" > "$JUMPGUY_RUNPOD_OUT/pod-id"
echo "wrote $JUMPGUY_RUNPOD_OUT/pod-id"

# Register the public half of the identity (never print the private key).
if [[ -f "$JUMPGUY_SSH_KEY" ]]; then
  pub="${JUMPGUY_SSH_KEY}.pub"
  if [[ ! -f "$pub" ]] && command -v ssh-keygen >/dev/null; then
    ssh-keygen -y -f "$JUMPGUY_SSH_KEY" > "$pub" || true
  fi
  if runpodctl ssh add-key --help >/dev/null 2>&1; then
    if [[ -f "$pub" ]]; then
      runpodctl ssh add-key --key "$(cat "$pub")" || runpodctl ssh add-key || true
    else
      runpodctl ssh add-key || true
    fi
  fi
fi

echo "waiting for runpodctl ssh info ${pod_id} (pods do not auto-stop — destroy when done)"
ssh_line=""
for _ in $(seq 1 40); do
  info="$(runpodctl ssh info "$pod_id" 2>/dev/null || true)"
  ssh_line="$(printf '%s\n' "$info" | grep -E 'ssh root@|root@' | head -1 || true)"
  if [[ -n "$ssh_line" ]]; then
    break
  fi
  sleep 5
done
if [[ -z "$ssh_line" ]]; then
  echo "ssh info not ready. Later: runpodctl ssh info ${pod_id}" >&2
  echo "Then set JUMPGUY_REMOTE=root@<ip> JUMPGUY_SSH_PORT=<port> JUMPGUY_SSH_KEY=${JUMPGUY_SSH_KEY}" >&2
  exit 1
fi

remote="$(printf '%s\n' "$ssh_line" | grep -Eo 'root@[0-9.]+' | head -1)"
port="$(printf '%s\n' "$ssh_line" | grep -Eo -- '-p[ ]*[0-9]+' | grep -Eo '[0-9]+' | head -1 || true)"
{
  echo "RUNPOD_POD_ID=${pod_id}"
  echo "JUMPGUY_REMOTE=${remote}"
  echo "JUMPGUY_SSH_PORT=${port:-22}"
  echo "JUMPGUY_SSH_KEY=${JUMPGUY_SSH_KEY}"
} > "$JUMPGUY_RUNPOD_OUT/ssh.env"
echo "wrote $JUMPGUY_RUNPOD_OUT/ssh.env (from runpodctl ssh info)"
echo "JUMPGUY_REMOTE=${remote} JUMPGUY_SSH_PORT=${port:-22}"
echo "pods do NOT auto-stop. When training ends: make -C jumpguy/runpod destroy"
