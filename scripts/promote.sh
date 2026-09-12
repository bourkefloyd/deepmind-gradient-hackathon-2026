#!/usr/bin/env bash
# Move 100% of traffic on the main service URL to one tagged playtest build.
#   scripts/promote.sh iter3
set -euo pipefail
TAG="${1:-}"
if [[ -z "$TAG" ]]; then echo "usage: scripts/promote.sh <tag>" >&2; exit 2; fi
REGION="${REGION:-us-west1}"
SERVICE="${SERVICE:-wordhunt}"
URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')"
# Rooms live in memory per revision: promoting moves the demo link to a different process, so
# players in a live room on the old revision lose it. Treat as live: any room with connected humans,
# any room in lobby/countdown, or any room younger than 10 minutes. Refuse unless FORCE=1.
LIVE="$(curl -fsS "$URL/api/health" 2>/dev/null | python3 -c 'import json,sys;d=json.load(sys.stdin);print(",".join(r["code"] for r in d.get("room_list",[]) if r.get("humans",0)>0 or r.get("state") in ("lobby","countdown") or r.get("age_s",1e9)<600))' 2>/dev/null || true)"
if [[ -n "$LIVE" && "${FORCE:-0}" != "1" ]]; then
  echo "refusing to promote: rooms with connected humans on the live revision: $LIVE (FORCE=1 to override)" >&2
  exit 3
fi
gcloud run services update-traffic "$SERVICE" --region "$REGION" --to-tags "$TAG=100" --quiet
echo "demo link now serves $TAG: $(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')"
