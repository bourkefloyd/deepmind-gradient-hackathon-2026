# integrations/ — Nango → Discord game events

Word Hunt room events posted to Discord through [Nango](https://nango.dev). Owner lane D
(PLAN.md section 5, 9). Nango is the prize: v1 is the plumbing (server-side events), the
commentator tool (`tools.py`) is how **Gemma** calls the same Nango action at the buzzer.

## What happens (v1)

| Game event | Where in `wordhunt/room.py` | Discord |
|---|---|---|
| `room_created` | `Room.__init__` | "Room ABCD created — join at `<WH_PUBLIC_URL>/r/ABCD`, seats: …" |
| `round_started` | after `state = "playing"` | "Room ABCD · round 1 started — board hidden, 75 s, go!" |
| `round_ended` | after results are computed | ranked scores, best word per seat, invalid count, longest board words |

All three go through **one Nango action**, `send-discord-recap` with input `{"text": "..."}`,
on integration `discord-wordhunt`. Every message names the room code because v1 posts to a
single channel (no threads yet; see v2).

The game never waits on Discord: `emit()` schedules a task, each handler has a 15 s cap,
every failure is a `WARNING` log line. `wordhunt/room.py` only imports `integrations` when
`WH_INTEGRATIONS=1`, and the import itself is inside a try/except.

## Env vars

| Var | Required | Default | Meaning |
|---|---|---|---|
| `WH_INTEGRATIONS` | yes (game) | unset | `1` turns the hooks in `room.py` on |
| `NANGO_SECRET_KEY` | yes (live) | unset → **dry-run** | Nango *environment* API key (Environment Settings → API Keys). Scopes: `environment:actions:execute`; add `environment:connections:list` for `--connections`, `environment:proxy` for v2 |
| `NANGO_DISCORD_INTEGRATION_ID` | no | `discord-wordhunt` | Nango integration unique key = `Provider-Config-Key` header |
| `NANGO_CONNECTION_ID` | yes (live) | unset | `Connection-Id` header. Find it with `python -m integrations.demo --connections` |
| `NANGO_DISCORD_RECAP_ACTION` | no | `send-discord-recap` | action name, input `{text}` |
| `WH_PUBLIC_URL` | recommended | `http://localhost:8000` | Cloud Run URL, used for the join link |
| `NANGO_BASE_URL` | no | `https://api.nango.dev` | self-hosted Nango |
| `WH_INTEGRATIONS_TIMEOUT_S` | no | `8` | per-request wall clock |
| `DISCORD_CHANNEL_ID` | v2 only | unset | channel for the thread flow |

Production: `NANGO_SECRET_KEY` is in GCP Secret Manager (`nango-secret-key`, project
`actionfleet-live`) and is mounted into the Cloud Run service as `NANGO_SECRET_KEY` together
with `WH_INTEGRATIONS=1`. The Dockerfile copies `integrations/` into the image.

## Test without a game (dry-run)

```
python -m integrations.demo            # three simulated events, prints the would-be POSTs
python -m integrations.demo --tool     # commentator tool schema + a fake tool call
```

With the env set (locally, or as a one-off Cloud Run job using the same image and secret):

```
python -m integrations.demo --connections    # GET /connections, filtered to discord-wordhunt
python -m integrations.demo --live           # fires ONE real send-discord-recap
```

Cloud Run one-off, same image + secret as the game:

```
gcloud run jobs create wh-nango-demo --image <game image> --region us-west1 \
  --set-secrets NANGO_SECRET_KEY=nango-secret-key:latest \
  --set-env-vars NANGO_CONNECTION_ID=<id>,WH_PUBLIC_URL=<url> \
  --command python --args -m,integrations.demo,--live
gcloud run jobs execute wh-nango-demo --region us-west1 --wait
```

## Nango API calls used (verified against docs.nango.dev, 2026-09)

```
POST https://api.nango.dev/action/trigger
  Authorization: Bearer $NANGO_SECRET_KEY
  Provider-Config-Key: discord-wordhunt
  Connection-Id: $NANGO_CONNECTION_ID
  {"action_name": "send-discord-recap", "input": {"text": "..."}}
```
Synchronous; the response body is the action's output. Errors: 400 bad input, 424 upstream
(Discord) error with `error.upstream.{status,body}`, 500 action error.

```
GET https://api.nango.dev/connections            (/connection is deprecated)
  Authorization: Bearer $NANGO_SECRET_KEY
```
No `provider_config_key` filter is documented, so the helper pages and filters client-side.

```
POST https://api.nango.dev/proxy/api/v10/channels/{id}/messages       (v2)
  Authorization / Provider-Config-Key / Connection-Id as above
  nango-proxy-Authorization: Bot <bot token>
```
Nango's Discord provider proxies to `https://discord.com`; caller headers prefixed
`nango-proxy-` are forwarded and override the OAuth bearer, which is what Nango's own
prebuilt Discord actions (`create-message`, `create-thread-from-message`) do internally
with `connection.metadata.botToken`.

## Nango dashboard checklist (already done by Bourke; here for a rebuild)

1. Integrations → Configure New Integration → Discord. Unique key `discord-wordhunt`.
   Discord OAuth app: redirect `https://api.nango.dev/oauth/callback`; scopes `identify bot`
   (bot is what lets the action post; the user token alone cannot create messages).
2. Connections → Add Test Connection → authorize with the server where the bot lives.
3. Functions → deploy the custom action `send-discord-recap` (input `{text}`), which posts to
   the configured channel with the bot token. For v2 also enable the prebuilt
   `create-message` and `create-thread-from-message` templates and set
   `metadata.botToken` on the connection (Connections → metadata, or `PATCH /connections/{id}/metadata`).
4. Environment Settings → API Keys → key with `environment:actions:execute`
   (+ `environment:connections:list`). That is `NANGO_SECRET_KEY`.

## Gemma commentator (the buzzer call)

`tools.py` exposes exactly one OpenAI tool, `send_discord_recap(text)`, mapped 1:1 to the
Nango action. `commentator.py` makes ONE chat call with that tool (thinking off:
`reasoning_effort: "none"` + `enable_thinking: false`), logs the raw tool call
(`MODEL CALLED NANGO TOOL …` — that line is the on-screen proof), executes it through
`tools.dispatch()` → `POST /action/trigger`, and posts the templated recap instead if the model
does not call the tool within 10 s (`GEMMA_TOOL_DEADLINE_S`) or answers with plain text.

Gemma runs on the Mac (mlx-vlm, `http://localhost:8080/v1`, `mlx-community/gemma-4-12B-it-4bit`)
and Cloud Run cannot reach it, so the commentator runs **client side**: it joins the room as a
spectator over wss (`{"type":"hello","role":"spectator"}`), waits for a `state` message with
`state == "results"`, rebuilds the payload with `handlers.from_snapshot`, and fires.

```
# on the Mac, with NANGO_SECRET_KEY + NANGO_CONNECTION_ID exported and mlx-vlm up
python -m integrations.commentator --room ABCD --server https://wordhunt-pngitthrva-uw.a.run.app
python -m integrations.commentator --once                # one call on the demo payload, no game
python -m integrations.commentator --once --base-url http://127.0.0.1:8811/v1 --model mock   # against a mock
```
Env: `GEMMA_BASE_URL`, `GEMMA_MODEL`, `GEMMA_API_KEY` (optional), `GEMMA_TOOL_DEADLINE_S`.
Needs `websockets` for the spectator path (`gemma_seat/requirements.txt` has it). vLLM seats need
`--tool-call-parser gemma4 --enable-auto-tool-choice` (PLAN.md section 4). Nothing posts on the
model's behalf except the explicit fallback, which is logged as `posted templated recap instead`.

Note: with `WH_INTEGRATIONS=1` on the server, `round_ended` also posts the templated leaderboard
server-side, so during a commentated match Discord gets the leaderboard (server) and the
trash talk (Gemma via the tool). Set `WH_INTEGRATIONS=0` if only the model should speak.

## v2: room threads (not wired)

Wanted flow: `room_created` → message in the main channel + thread from it; `round_started`
and a one-line `round_ended` in the thread; full leaderboard in the main channel. Needs message
and thread ids, which the text-only `send-discord-recap` cannot return. Options, code already
in `discord.py` (`send_message`, `create_thread`, `post_in_thread`, `formatters.round_ended_short`):

- Nango prebuilt actions `create-message` → returns `id`; `create-thread-from-message` → thread `id`;
  post into the thread with `create-message(channelId=thread_id)`. Requires `metadata.botToken`.
- Or the proxy with `nango-proxy-Authorization: Bot …` (same three Discord endpoints).

Then keep `{code: thread_id}` in `handlers.py` and route by event. Everything else stays.

## Layout

```
config.py      env → Settings; dry_run when NANGO_SECRET_KEY unset
nango.py       NangoClient: trigger_action, list_connections, proxy (+ async wrappers, stdlib urllib)
discord.py     send_recap (v1); send_message / create_thread / post_in_thread (v2)
formatters.py  room_created / round_started / round_ended → Discord markdown
events.py      on(), emit(), drain(); fire-and-forget with timeout
handlers.py    from_room(Room) → payload; discord_recap handler; register()
tools.py       OpenAI tool schema + dispatch for the Gemma commentator
demo.py        python -m integrations.demo [--live | --connections | --tool]
```
