#!/usr/bin/env bash
# Move 100% of traffic on the main service URL to one tagged playtest build.
#   scripts/promote.sh iter3
set -euo pipefail
TAG="${1:-}"
if [[ -z "$TAG" ]]; then echo "usage: scripts/promote.sh <tag>" >&2; exit 2; fi
REGION="${REGION:-us-west1}"
SERVICE="${SERVICE:-wordhunt}"
gcloud run services update-traffic "$SERVICE" --region "$REGION" --to-tags "$TAG=100" --quiet
echo "demo link now serves $TAG: $(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')"
