# Demo runbook — 4:30 PM PT (cap 20 min)

Promoted game URL for this build: **https://iter4e---wordhunt-pngitthrva-uw.a.run.app**

Open these **before** the room goes live:

| Tab | URL |
|---|---|
| Projector (spectate) | `https://iter4e---wordhunt-pngitthrva-uw.a.run.app/s/<ROOM>` |
| Phone (host / play) | `https://iter4e---wordhunt-pngitthrva-uw.a.run.app/r/<ROOM>` |
| Discord | Nango `discord-wordhunt` channel (same channel the server posts `room_created` / `round_ended` into) |
| Respan traces | https://platform.respan.ai/platform/logs — filter **Customer ID** = `wordhunt-vs/commentator` or **Custom ID** = `commentator` |

Secrets (`respan-api-key`, `nango-secret-key`) are read from GCP Secret Manager project `actionfleet-live` by the Makefile targets below — no manual export needed on Bourke's Mac.

---

## 1. Commentator (primary path — Respan proxy, Gemma 4 31B)

In one terminal on the Mac (repo root, `.venv` linked or `make setup-server` done):

```bash
ROOM=$(curl -fsS -X POST https://iter4e---wordhunt-pngitthrva-uw.a.run.app/api/rooms | python3 -c 'import json,sys;print(json.load(sys.stdin)["code"])') && echo "room $ROOM"
```

Start the commentator **before** the round starts (leave running; `--rounds 0` = stay for every recap):

```bash
make commentator-demo ROOM=$ROOM SERVER=https://iter4e---wordhunt-pngitthrva-uw.a.run.app COMMENTATOR_ARGS="-v"
```

Drive a non-quiet round (host auto-starts; default lineup includes Nano on Cloud Run). From repo root, with word lists present (`make words` once if `scripts/load_test.py` errors on `data/enable1.txt`):

```bash
python3 scripts/load_test.py --url https://iter4e---wordhunt-pngitthrva-uw.a.run.app --room $ROOM --n 1
```

Or minimal host when word lists are missing (Nano plays alone; ~95 s):

```bash
python3 - <<'PY'
import asyncio, json, os, websockets
room = os.environ["ROOM"]
url = f"wss://iter4e---wordhunt-pngitthrva-uw.a.run.app/ws/{room}"
async def main():
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"type":"hello","player_id":"demo_host","name":"Demo Host"}))
        started = False
        async for raw in ws:
            m = json.loads(raw)
            if m.get("type")=="state" and m.get("state")=="lobby" and not started:
                await ws.send(json.dumps({"type":"start"})); started = True
            elif m.get("type")=="state" and m.get("state")=="results":
                break
asyncio.run(main())
PY
```

**Proof at the buzzer:** Discord gets a post starting with `🤖 **Gemma 31B** (via Nango tool call)`; terminal logs `MODEL CALLED NANGO TOOL`; Respan Logs shows a new span with `customer_identifier` = `wordhunt-vs/commentator`.

One-liner smoke (single recap, then exit):

```bash
make commentator-demo ROOM=$ROOM SERVER=https://iter4e---wordhunt-pngitthrva-uw.a.run.app COMMENTATOR_ARGS="--rounds 1 -v"
```

Routing (see `docs/architecture.md` §5b): `RESPAN_ENABLED=1 RESPAN_MODE=proxy RESPAN_MODEL=deepinfra/google/gemma-4-31B-it`; `GEMMA_BASE_URL` / `GEMMA_API_KEY` are ignored in proxy mode.

---

## 2. Gemma 12B seat fallback (Lambda vLLM down)

If the Cloud Run registry seat cannot reach Lambda vLLM, run the **Mac mlx-vlm bot** as a human seat instead:

```bash
make mlxvlm-up && make mlxvlm-status
```

```bash
make gemma-seat ROOM=$ROOM SERVER=https://iter4e---wordhunt-pngitthrva-uw.a.run.app SEAT_ARGS="--rounds 1 --start"
```

(`make gemma-seat` uses local mlx-vlm at `:8080`, not Respan proxy — seat traces use `RESPAN_MODE=log` on Cloud Run; the commentator stays on proxy per §5b.)

**Do not run** `make mlxvlm-stop` during the demo block.

---

## 3. Verify Respan span (optional CLI)

After a commentator recap (replace key via Secret Manager; never paste keys into chat):

```bash
curl -fsS -H "Authorization: Bearer $(gcloud secrets versions access latest --secret respan-api-key --project actionfleet-live)" \
  'https://api.respan.ai/api/request-logs/list/?customer_identifier=wordhunt-vs/commentator&limit=3'
```

Look for the newest `unique_id` / `span_name` = `commentator` and `thread_identifier` = `commentator:<ROOM>`.

---

## 4. Demo lineup (suggested)

Projector on `/s/$ROOM` with QR; phone hosts on `/r/$ROOM`; commentator terminal visible or tailed; Discord + Respan on the side monitor. Default Cloud Run lineup is Nano 10M — add seats from the host UI if you want Reflex-A or the registry Gemma seat (Lambda, log mode).

Do **not** redeploy or promote a new Cloud Run tag between now and show time.
