#!/usr/bin/env bash
# Deploy a playtest build of the Word Hunt arena to Cloud Run as a tagged revision.
#
#   scripts/deploy.sh iter2          # deploys revision tagged "iter2"
#
# Every build keeps its own stable URL: https://iter2---wordhunt-<hash>-uw.a.run.app
# The first deploy of the service takes 100% traffic; later builds are deployed --no-traffic
# (reachable only via their tag URL) until scripts/promote.sh <tag> moves the demo link.
#
# Auth: set GCP_SA_KEY_JSON (service-account key as a JSON string) or be logged in already.
# Single instance, in-memory rooms, WebSockets (request timeout 3600 s), session affinity.
set -euo pipefail

LABEL="${1:-}"
if [[ -z "$LABEL" ]]; then echo "usage: scripts/deploy.sh <label, e.g. iter2>" >&2; exit 2; fi
TAG="$(echo "$LABEL" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9-\n' '-')"
REGION="${REGION:-us-west1}"
SERVICE="${SERVICE:-wordhunt}"
cd "$(dirname "$0")/.."
SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
BUILD_LABEL="$TAG · $SHA"

if [[ -n "${GCP_SA_KEY_JSON:-}" ]]; then
  KEY_FILE="$(mktemp)"
  trap 'rm -f "$KEY_FILE"' EXIT
  printf '%s' "$GCP_SA_KEY_JSON" > "$KEY_FILE"
  gcloud auth activate-service-account --key-file="$KEY_FILE" --quiet
  PROJECT="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["project_id"])' "$KEY_FILE")"
  gcloud config set project "$PROJECT" --quiet
fi
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
echo "project=$PROJECT region=$REGION service=$SERVICE tag=$TAG build='$BUILD_LABEL'"

gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com --quiet || \
  echo "warn: could not enable services (need serviceusage.serviceUsageAdmin); continuing"

# Env for the revision. '|' delimiter because GEMMA_MODELS is a comma list.
#   GEMMA_BASE_URL (OpenAI-compatible, e.g. http://LAMBDA_IP:8000/v1), GEMMA_API_KEY, GEMMA_MODELS
#   or GEMMA_SEATS (JSON list of {id,name,model,base_url,api_key,label}); NANO_CKPT for the nano seat.
ENV_VARS="WH_BUILD=$BUILD_LABEL"
for v in GEMMA_BASE_URL GEMMA_API_KEY GEMMA_MODELS GEMMA_SEATS NANO_CKPT NANO_TEMPERATURE WH_MAX_HUMANS; do
  if [[ -n "${!v:-}" ]]; then ENV_VARS="$ENV_VARS|$v=${!v}"; fi
done

TRAFFIC_FLAG=(--no-traffic)
if ! gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)' >/dev/null 2>&1; then
  echo "first deploy of $SERVICE: this revision takes traffic"
  TRAFFIC_FLAG=()
fi

gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --tag "$TAG" "${TRAFFIC_FLAG[@]}" \
  --set-env-vars "^|^$ENV_VARS" \
  --min-instances 1 --max-instances 1 \
  --session-affinity \
  --timeout 3600 \
  --cpu 1 --memory 512Mi \
  --quiet

SERVICE_URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')"
TAG_URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format=json \
  | python3 -c 'import json,sys;d=json.load(sys.stdin);print(next((t.get("url","") for t in d["status"].get("traffic",[]) if t.get("tag")==sys.argv[1]),""))' "$TAG")"
echo
echo "service (demo link): $SERVICE_URL"
echo "tag url ($TAG):      $TAG_URL"
curl -fsS "${TAG_URL:-$SERVICE_URL}/healthz" && echo
