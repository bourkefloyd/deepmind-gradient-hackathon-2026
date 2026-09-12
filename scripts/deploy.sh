#!/usr/bin/env bash
# Deploy a playtest build of the Word Hunt arena to Cloud Run as a tagged revision.
#
#   scripts/deploy.sh iter2          # deploys revision tagged "iter2"
#
# Every build keeps its own stable URL: https://iter2---wordhunt-<hash>-uw.a.run.app
# The first deploy of the service takes 100% traffic; later builds are deployed --no-traffic
# (reachable only via their tag URL) until scripts/promote.sh <tag> moves the demo link.
#
# Image build: no Docker daemon and no Cloud Build needed. The app layer (code, word lists,
# pip-installed deps for linux/x86_64 py3.12) is appended onto python:3.12-slim with `crane`
# and pushed to Artifact Registry with the deploying account. DEPLOY_MODE=source uses
# `gcloud run deploy --source` (Cloud Build) instead.
#
# Auth: set GCP_SA_KEY_JSON (service-account key as a JSON string) or be logged in already.
# Single instance, in-memory rooms, WebSockets (request timeout 3600 s), session affinity.
# Public access: the org policy forbids allUsers IAM bindings, so the service runs with
# --no-invoker-iam-check. Note: Google's frontend reserves /healthz; health is at /api/health.
set -euo pipefail

LABEL="${1:-}"
if [[ -z "$LABEL" ]]; then echo "usage: scripts/deploy.sh <label, e.g. iter2>" >&2; exit 2; fi
TAG="$(echo "$LABEL" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9-\n' '-')"
REGION="${REGION:-us-west1}"
SERVICE="${SERVICE:-wordhunt}"
REPO="${REPO:-wordhunt}"
DEPLOY_MODE="${DEPLOY_MODE:-image}"
BASE_IMAGE="${BASE_IMAGE:-python:3.12-slim}"
cd "$(dirname "$0")/.."
ROOT="$PWD"
SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
BUILD_LABEL="$TAG · $SHA"

if [[ -n "${GCP_SA_KEY_JSON:-}" ]]; then
  KEY_FILE="$(mktemp)"
  trap 'rm -f "$KEY_FILE"' EXIT
  printf '%s' "$GCP_SA_KEY_JSON" > "$KEY_FILE"
  gcloud auth activate-service-account --key-file="$KEY_FILE" --quiet
  PROJECT="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["project_id"])' "$KEY_FILE")"
  gcloud config set project "$PROJECT" --quiet
  # Run the revision as the deploying SA too, so secrets granted to it are readable at runtime.
  RUN_SA="${RUN_SA:-$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["client_email"])' "$KEY_FILE")}"
fi
RUN_SA="${RUN_SA:-$(gcloud config get-value account 2>/dev/null)}"
SA_FLAG=()
if [[ "$RUN_SA" == *.gserviceaccount.com ]]; then SA_FLAG=(--service-account "$RUN_SA"); fi
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
echo "project=$PROJECT region=$REGION service=$SERVICE tag=$TAG build='$BUILD_LABEL' mode=$DEPLOY_MODE"

gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com --quiet || \
  echo "warn: could not enable services (need serviceusage.serviceUsageAdmin); continuing"

# Env for the revision. '|' delimiter because GEMMA_MODELS is a comma list.
#   GEMMA_BASE_URL (OpenAI-compatible, e.g. http://LAMBDA_IP:8000/v1), GEMMA_API_KEY, GEMMA_MODELS
#   or GEMMA_SEATS (JSON list of {id,name,model,base_url,api_key,label}); NANO_CKPT for the nano seat.
#   WH_SEATS (default lineup, e.g. "nano,reflex-a"), WH_NANO_TEMPERATURE, WH_NANO_THINK
#   WH_INTEGRATIONS=1 mounts Secret Manager `nango-secret-key` as NANGO_SECRET_KEY and sets WH_PUBLIC_URL.
ENV_VARS="WH_BUILD=$BUILD_LABEL"
for v in GEMMA_BASE_URL GEMMA_API_KEY GEMMA_MODELS GEMMA_SEATS NANO_CKPT WH_NANO_TEMPERATURE WH_NANO_THINK WH_SEATS WH_MAX_HUMANS WH_INTEGRATIONS \
         NANGO_CONNECTION_ID NANGO_DISCORD_INTEGRATION_ID NANGO_PROVIDER_CONFIG_KEY NANGO_DISCORD_RECAP_ACTION NANGO_BASE_URL DISCORD_CHANNEL_ID WH_INTEGRATIONS_TIMEOUT_S; do
  if [[ -n "${!v:-}" ]]; then ENV_VARS="$ENV_VARS|$v=${!v}"; fi
done
EXISTING_URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)' 2>/dev/null || true)"
SECRET_FLAGS=()
if [[ "${WH_INTEGRATIONS:-0}" == "1" ]]; then
  ENV_VARS="$ENV_VARS|WH_PUBLIC_URL=${WH_PUBLIC_URL:-$EXISTING_URL}"
  SECRET_FLAGS=(--update-secrets "NANGO_SECRET_KEY=${NANGO_SECRET_NAME:-nango-secret-key}:latest")
fi

SOURCE_FLAGS=(--source .)
if [[ "$DEPLOY_MODE" == "image" ]]; then
  CRANE="${CRANE:-$(command -v crane || true)}"
  if [[ -z "$CRANE" ]]; then
    CRANE="/tmp/crane"
    if [[ ! -x "$CRANE" ]]; then
      echo "fetching crane"
      curl -sL https://github.com/google/go-containerregistry/releases/latest/download/go-containerregistry_Linux_x86_64.tar.gz | tar -xz -C /tmp crane
      chmod +x "$CRANE"
    fi
  fi
  AR_HOST="$REGION-docker.pkg.dev"
  IMAGE="$AR_HOST/$PROJECT/$REPO/$SERVICE:$TAG-$SHA"
  gcloud artifacts repositories describe "$REPO" --location "$REGION" >/dev/null 2>&1 || \
    gcloud artifacts repositories create "$REPO" --repository-format=docker --location="$REGION" --quiet

  BUILD_DIR="$(mktemp -d)"
  trap 'rm -rf "$BUILD_DIR" ${KEY_FILE:-}' EXIT
  mkdir -p "$BUILD_DIR/app/data"
  cp -r "$ROOT/wordhunt" "$ROOT/static" "$ROOT/nano" "$BUILD_DIR/app/"
  [[ -d "$ROOT/integrations" ]] && cp -r "$ROOT/integrations" "$BUILD_DIR/app/"
  find "$BUILD_DIR/app" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
  cp "$ROOT"/data/*.json "$BUILD_DIR/app/data/" 2>/dev/null || true
  # Word lists are not committed (data/README.md).
  if [[ -f "$ROOT/data/enable1.txt" ]]; then cp "$ROOT/data/enable1.txt" "$BUILD_DIR/app/data/"; else
    curl -sL -o "$BUILD_DIR/app/data/enable1.txt" https://raw.githubusercontent.com/dolph/dictionary/master/enable1.txt; fi
  if [[ -f "$ROOT/data/common-30k.txt" ]]; then cp "$ROOT/data/common-30k.txt" "$BUILD_DIR/app/data/"; else
    curl -sL https://raw.githubusercontent.com/arstgit/high-frequency-vocabulary/master/30k.txt | tr -d '\r\t' | awk 'NF' > "$BUILD_DIR/app/data/common-30k.txt"; fi
  # Deps as manylinux wheels for the base image's interpreter (no compilation here). Torch CPU wheels
  # are tagged manylinux_2_28; python:3.12-slim (Debian bookworm, glibc 2.36) runs them.
  python3 -m pip install -q --target "$BUILD_DIR/app/site" --python-version 3.12 --implementation cp \
    --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux_2_28_x86_64 \
    --only-binary=:all: -r "$ROOT/requirements.txt"
  find "$BUILD_DIR/app/site" -type d -name tests -prune -exec rm -rf {} + 2>/dev/null || true
  tar -C "$BUILD_DIR" -cf "$BUILD_DIR/layer.tar" app
  echo "pushing $IMAGE"
  gcloud auth print-access-token | "$CRANE" auth login "$AR_HOST" -u oauth2accesstoken --password-stdin >/dev/null
  "$CRANE" append -b "$BASE_IMAGE" -f "$BUILD_DIR/layer.tar" -t "$IMAGE" --platform linux/amd64 >/dev/null
  "$CRANE" mutate "$IMAGE" -t "$IMAGE" --workdir /app \
    --env PYTHONPATH=/app/site --env PYTHONUNBUFFERED=1 --env PORT=8080 --exposed-ports 8080 \
    --entrypoint '' \
    --cmd python --cmd -m --cmd uvicorn --cmd wordhunt.server:app --cmd --host --cmd 0.0.0.0 --cmd --port --cmd 8080 \
    --cmd --ws-ping-interval --cmd 20 --cmd --ws-ping-timeout --cmd 20 >/dev/null
  SOURCE_FLAGS=(--image "$IMAGE")
fi

TRAFFIC_FLAG=(--no-traffic)
if [[ -z "$EXISTING_URL" ]]; then
  echo "first deploy of $SERVICE: this revision takes traffic"
  TRAFFIC_FLAG=()
fi

gcloud run deploy "$SERVICE" \
  "${SOURCE_FLAGS[@]}" \
  --region "$REGION" \
  --allow-unauthenticated --no-invoker-iam-check "${SA_FLAG[@]}" \
  --tag "$TAG" "${TRAFFIC_FLAG[@]}" \
  --set-env-vars "^|^$ENV_VARS" "${SECRET_FLAGS[@]}" \
  --min-instances 1 --max-instances 1 \
  --session-affinity \
  --timeout 3600 \
  --cpu 1 --memory 1Gi \
  --quiet

SERVICE_URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')"
TAG_URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format=json \
  | python3 -c 'import json,sys;d=json.load(sys.stdin);print(next((t.get("url","") for t in d["status"].get("traffic",[]) if t.get("tag")==sys.argv[1]),""))' "$TAG")"
echo
echo "service (demo link): $SERVICE_URL"
echo "tag url ($TAG):      $TAG_URL"
curl -fsS "${TAG_URL:-$SERVICE_URL}/api/health" && echo
