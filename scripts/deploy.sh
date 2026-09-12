#!/usr/bin/env bash
# Deploy the Word Hunt arena to Cloud Run (single instance, in-memory rooms, WebSockets).
#
# Auth: set GCP_SA_KEY_JSON (service-account key as a JSON string) or be logged in already.
# Usage: scripts/deploy.sh            # build from source with Cloud Build and deploy
set -euo pipefail

REGION="${REGION:-us-west1}"
SERVICE="${SERVICE:-wordhunt}"
cd "$(dirname "$0")/.."

if [[ -n "${GCP_SA_KEY_JSON:-}" ]]; then
  KEY_FILE="$(mktemp)"
  trap 'rm -f "$KEY_FILE"' EXIT
  printf '%s' "$GCP_SA_KEY_JSON" > "$KEY_FILE"
  gcloud auth activate-service-account --key-file="$KEY_FILE" --quiet
  PROJECT="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["project_id"])' "$KEY_FILE")"
  gcloud config set project "$PROJECT" --quiet
fi
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
echo "project=$PROJECT region=$REGION service=$SERVICE"

gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com --quiet

# AI seat config is passed through as env vars when set locally:
#   GEMMA_BASE_URL (OpenAI-compatible, e.g. http://LAMBDA_IP:8000/v1), GEMMA_API_KEY, GEMMA_MODELS (comma list)
#   or GEMMA_SEATS (JSON list of {id,name,model,base_url,api_key,label}); NANO_CKPT for the nano seat.
ENV_VARS=""
for v in GEMMA_BASE_URL GEMMA_API_KEY GEMMA_MODELS GEMMA_SEATS NANO_CKPT NANO_TEMPERATURE; do
  if [[ -n "${!v:-}" ]]; then ENV_VARS="${ENV_VARS:+$ENV_VARS,}$v=${!v}"; fi
done
ENV_FLAG=()
if [[ -n "$ENV_VARS" ]]; then ENV_FLAG=(--set-env-vars "^,^$ENV_VARS"); fi

gcloud run deploy "$SERVICE" \
  --source . \
  "${ENV_FLAG[@]}" \
  --region "$REGION" \
  --allow-unauthenticated \
  --min-instances 1 --max-instances 1 \
  --session-affinity \
  --timeout 3600 \
  --cpu 1 --memory 512Mi \
  --quiet

URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')"
echo "URL=$URL"
curl -fsS "$URL/healthz" && echo
