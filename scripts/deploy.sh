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

gcloud run deploy "$SERVICE" \
  --source . \
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
