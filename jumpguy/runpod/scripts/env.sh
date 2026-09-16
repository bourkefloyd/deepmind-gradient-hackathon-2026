#!/usr/bin/env bash
# Shared env for jumpguy/runpod — mirrors actionfleet sandbox/runpod-train/scripts/env.sh
# Never print secret values. Never invent keys.
#
# Secret NAMES (values come from the operator / Cursor cloud secrets):
#   RUNPOD_API_KEY       required to create/destroy pods
#   RUNPOD_SSH_KEY       Cursor cloud secret; cloud_bootstrap writes ~/.ssh/runpod_nano
#                        (may also be a path to a private key)
#   RUNPOD_SSH_IDENTITY  path override for the private key
#
# Non-secret defaults (actionfleet main, commit 9a87b5d / Blaise):
#   template runpod-torch-v280 (min CUDA 12.8)
#   GPU      NVIDIA GeForce RTX 4090
#   cloud    SECURE
#   region   EU-RO-1
#   pod name jumpguy-train
#   volume   50GB mounted at /workspace
#
# Gitignored state (same shape as sandbox/runpod-train/out/):
#   jumpguy/runpod/out/{pod-id,volume-id,ssh.env}

_JUMPGUY_RUNPOD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JUMPGUY_RUNPOD_OUT="${JUMPGUY_RUNPOD_OUT:-$_JUMPGUY_RUNPOD_ROOT/out}"
mkdir -p "$JUMPGUY_RUNPOD_OUT"

JUMPGUY_POD_NAME="${JUMPGUY_POD_NAME:-jumpguy-train}"
JUMPGUY_RUNPOD_TEMPLATE="${JUMPGUY_RUNPOD_TEMPLATE:-runpod-torch-v280}"
JUMPGUY_RUNPOD_GPU="${JUMPGUY_RUNPOD_GPU:-NVIDIA GeForce RTX 4090}"
JUMPGUY_RUNPOD_CLOUD="${JUMPGUY_RUNPOD_CLOUD:-SECURE}"
JUMPGUY_RUNPOD_DC="${JUMPGUY_RUNPOD_DC:-EU-RO-1}"
JUMPGUY_VOLUME_NAME="${JUMPGUY_VOLUME_NAME:-jumpguy-data}"
JUMPGUY_VOLUME_GB="${JUMPGUY_VOLUME_GB:-50}"
JUMPGUY_VOLUME_MOUNT="${JUMPGUY_VOLUME_MOUNT:-/workspace}"
JUMPGUY_REMOTE_DIR="${JUMPGUY_REMOTE_DIR:-/workspace/jumpguy_ws}"
JUMPGUY_DEVICE="${JUMPGUY_DEVICE:-cuda}"
REPO_URL="${REPO_URL:-https://github.com/bourkefloyd/deepmind-gradient-hackathon-2026.git}"
REPO_BRANCH="${REPO_BRANCH:-cursor/jumpguy-realtime-agent-7cc0}"
EPISODES="${EPISODES:-80}"
BC_STEPS="${BC_STEPS:-2000}"
PPO_STEPS="${PPO_STEPS:-30000}"

# Load last create-pod.sh output (JUMPGUY_REMOTE / port / key / pod id).
if [[ -f "$JUMPGUY_RUNPOD_OUT/ssh.env" ]]; then
  # shellcheck disable=SC1091
  source "$JUMPGUY_RUNPOD_OUT/ssh.env"
fi
if [[ -z "${RUNPOD_POD_ID:-}" && -f "$JUMPGUY_RUNPOD_OUT/pod-id" ]]; then
  RUNPOD_POD_ID="$(cat "$JUMPGUY_RUNPOD_OUT/pod-id")"
fi
if [[ -z "${JUMPGUY_VOLUME_ID:-}" && -f "$JUMPGUY_RUNPOD_OUT/volume-id" ]]; then
  JUMPGUY_VOLUME_ID="$(cat "$JUMPGUY_RUNPOD_OUT/volume-id")"
fi

jumpguy_resolve_identity() {
  if [[ -n "${RUNPOD_SSH_IDENTITY:-}" ]]; then
    printf '%s\n' "$RUNPOD_SSH_IDENTITY"
    return
  fi
  if [[ -n "${RUNPOD_SSH_KEY:-}" ]]; then
    if [[ -f "$RUNPOD_SSH_KEY" ]]; then
      printf '%s\n' "$RUNPOD_SSH_KEY"
      return
    fi
    # Cloud secret material: install once to the nanoagent filename. Do not echo it.
    local dest="${HOME}/.ssh/runpod_nano"
    mkdir -p "${HOME}/.ssh"
    chmod 700 "${HOME}/.ssh"
    if [[ ! -f "$dest" ]]; then
      umask 077
      printf '%s\n' "$RUNPOD_SSH_KEY" > "$dest"
      chmod 600 "$dest"
    fi
    printf '%s\n' "$dest"
    return
  fi
  local p
  for p in \
    "${HOME}/.ssh/runpod_ed25519" \
    "${HOME}/.ssh/runpod_nano" \
    "${HOME}/.ssh/id_ed25519"; do
    if [[ -f "$p" ]]; then
      printf '%s\n' "$p"
      return
    fi
  done
  printf '%s\n' "${HOME}/.ssh/id_ed25519"
}

JUMPGUY_SSH_KEY="${JUMPGUY_SSH_KEY:-$(jumpguy_resolve_identity)}"
JUMPGUY_SSH_PORT="${JUMPGUY_SSH_PORT:-22}"

jumpguy_ssh() {
  ssh -i "$JUMPGUY_SSH_KEY" -p "$JUMPGUY_SSH_PORT" \
    -o StrictHostKeyChecking=no -o ConnectTimeout=15 "$@"
}

jumpguy_scp() {
  scp -i "$JUMPGUY_SSH_KEY" -P "$JUMPGUY_SSH_PORT" \
    -o StrictHostKeyChecking=no "$@"
}

jumpguy_rsync() {
  rsync -az --exclude '__pycache__' --exclude '*.pyc' --exclude 'data' --exclude 'runs' --exclude 'runpod/out' \
    -e "ssh -i ${JUMPGUY_SSH_KEY} -p ${JUMPGUY_SSH_PORT} -o StrictHostKeyChecking=no -o ConnectTimeout=15" \
    "$@"
}

jumpguy_require_api_key() {
  if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
    echo "RUNPOD_API_KEY is unset. Export the existing RunPod secret (never commit it)." >&2
    echo "This VM will not invent a key. CPU train locally, or set the secret and re-run." >&2
    return 1
  fi
  if ! command -v runpodctl >/dev/null 2>&1; then
    echo "runpodctl is not on PATH. Install it, then: runpodctl config --apiKey \"\$RUNPOD_API_KEY\"" >&2
    return 1
  fi
  runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null
}

jumpguy_require_ssh() {
  if [[ -z "${JUMPGUY_REMOTE:-}" ]]; then
    echo "JUMPGUY_REMOTE is unset. Create a pod (make -C jumpguy/runpod create) or set it from:" >&2
    echo "  runpodctl ssh info <pod-id>" >&2
    echo "  → JUMPGUY_REMOTE=root@<ip>  JUMPGUY_SSH_PORT=<port>" >&2
    return 1
  fi
  if [[ ! -f "$JUMPGUY_SSH_KEY" ]]; then
    echo "SSH identity not found at $JUMPGUY_SSH_KEY" >&2
    echo "Expected ~/.ssh/runpod_ed25519 (SWM / actionfleet-runpod) or ~/.ssh/runpod_nano" >&2
    echo "or RUNPOD_SSH_IDENTITY / JUMPGUY_SSH_KEY." >&2
    return 1
  fi
}

jumpguy_print_env() {
  echo "JUMPGUY_POD_NAME=${JUMPGUY_POD_NAME}"
  echo "JUMPGUY_RUNPOD_TEMPLATE=${JUMPGUY_RUNPOD_TEMPLATE}"
  echo "JUMPGUY_RUNPOD_GPU=${JUMPGUY_RUNPOD_GPU}"
  echo "JUMPGUY_RUNPOD_CLOUD=${JUMPGUY_RUNPOD_CLOUD}"
  echo "JUMPGUY_RUNPOD_DC=${JUMPGUY_RUNPOD_DC}"
  echo "JUMPGUY_VOLUME_NAME=${JUMPGUY_VOLUME_NAME}"
  echo "JUMPGUY_VOLUME_MOUNT=${JUMPGUY_VOLUME_MOUNT}"
  echo "JUMPGUY_REMOTE_DIR=${JUMPGUY_REMOTE_DIR}"
  echo "JUMPGUY_DEVICE=${JUMPGUY_DEVICE}"
  echo "JUMPGUY_REMOTE=${JUMPGUY_REMOTE:-<unset>}"
  echo "JUMPGUY_SSH_PORT=${JUMPGUY_SSH_PORT}"
  echo "JUMPGUY_SSH_KEY=${JUMPGUY_SSH_KEY} exists=$( [[ -f "$JUMPGUY_SSH_KEY" ]] && echo yes || echo no )"
  echo "RUNPOD_POD_ID=${RUNPOD_POD_ID:-<unset>}"
  echo "JUMPGUY_VOLUME_ID=${JUMPGUY_VOLUME_ID:-<unset>}"
  echo "RUNPOD_SSH_IDENTITY=${RUNPOD_SSH_IDENTITY:-<unset>}"
  if [[ -n "${RUNPOD_API_KEY:-}" ]]; then echo "RUNPOD_API_KEY=set (value not printed)"; else echo "RUNPOD_API_KEY=unset"; fi
  if [[ -n "${RUNPOD_SSH_KEY:-}" ]]; then echo "RUNPOD_SSH_KEY=set (value not printed)"; else echo "RUNPOD_SSH_KEY=unset"; fi
  if [[ -n "${WANDB_API_KEY:-}" ]]; then echo "WANDB_API_KEY=set (value not printed)"; else echo "WANDB_API_KEY=unset"; fi
  echo "EPISODES=${EPISODES} BC_STEPS=${BC_STEPS} PPO_STEPS=${PPO_STEPS} REPO_BRANCH=${REPO_BRANCH}"
}
