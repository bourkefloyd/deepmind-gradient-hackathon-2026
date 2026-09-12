---
name: nango-game-hooks
description: Set up Nango Slack or Discord actions for the Word Hunt buzzer recap. Use when attaching Nango Management MCP, creating Slack/Discord integrations, minting connect links, deploying send-message/create-message, or triggering those actions for Gemma. Prefer MCP tools when the nango-management server is attached; otherwise use the REST fallback.
---

# Nango game hooks (Slack / Discord)

Prize rule from `PLAN.md`: **Gemma function-calls Nango at the buzzer**. A scorecard webhook from our server is the wrong shape. Do not post Slack/Discord from FastAPI/`wordhunt/server.py`.

Two Nango MCP servers exist. Do not mix them up:

| Server | URL | Who uses it |
| --- | --- | --- |
| Management MCP | `https://mcp.nango.dev/mcp` | This coding agent. Create integrations, connect sessions, deploy templates, proxy, logs. |
| Runtime / tool-calling MCP | `https://api.nango.dev/mcp` | Gemma (or any product agent). Headers pin `connection-id` and `provider-config-key`. Only enabled actions appear as tools. |

Auth for both: `Authorization: Bearer $NANGO_API_KEY`. Create the key at [app.nango.dev](https://app.nango.dev/) → Environment Settings → API Keys. Never print or commit it.

## Prefer Slack, Discord is fine

Slack can use Nango's shared OAuth app (`credential_source: "nango"`), so there is no Slack developer-portal work. Discord often needs your own Discord OAuth app (`credential_source: "own"`). User asked for Discord **or** Slack; pick whichever they authorize.

Buzzer action templates:

| Provider | Integration id | Template | Input |
| --- | --- | --- | --- |
| Slack | `slack` | `send-message` | `channel_id`, `text` |
| Discord | `discord` | `create-message` | channel + message fields from `providers_get` / the template |

Optional later: Linear `create-issue` for E4B invalids (`"bug: CTA is not a word"`).

## Current Nango environment (2026-09-12)

Management MCP is attached to Cloud Agents as namespace `nango`. Slack is created with Nango-shared OAuth (`credential_source: "nango"`). Discord create with `credential_source: "nango"` failed: Nango does not ship shared Discord app credentials.

| Item | Value |
| --- | --- |
| Slack integration id | `slack` |
| Slack connection id | `82f94a5c-4116-41b3-8afe-8aeed24f714b` |
| Slack test channel | `#new-channel` (`C0BUKC1AU06`) |
| Slack test | `send-message` ok, ts `1789242579.587919` |
| Discord guild | Open Model Hack - Gradient x Google Deepmind (`1547705063609470976`) |
| Discord invite | https://discord.gg/db3aDstTX (lands in `#general`) |
| Discord channel wanted | `wordhunt` (admin must create; you are not an admin) |
| Discord Nango | `discord-wordhunt` unauthenticated, connection `5ed2de56-09c0-4444-b53b-d4b74174078a`, action `send-discord-recap@v1.0.0` (posted test, HTTP 200) |

After the user authorizes Slack, `connections_list` with `end_user_id=wordhunt-buzzer` (or tag `purpose=wordhunt-buzzer`), then `actions_trigger` `list-channels` and a test `send-message`.

## Path A: Management MCP is attached

Use `nango-management` tools. Typical sequence (Slack shown; swap `slack` → `discord` if that is the target):

1. `integrations_list` — skip create if `slack` / `discord` already exists.
2. `integrations_create` with `provider`, `integration_id`, `display_name`, and `credential_source: "nango"` when the catalog allows it. If create fails on missing client id/secret, ask for their app credentials and retry with `credential_source: "own"`.
3. `connect_session_create` with `allowed_integrations` and a tag such as `end_user_id=wordhunt-buzzer`. Give the user the `connect_link`. It expires in ~30 minutes.
4. Wait until they authorize. Then `connections_list` filtered by that tag.
5. Smoke-test with `proxy_request` (Slack `GET /conversations.list`, Discord `GET /users/@me` or `GET /users/@me/guilds`).
6. `deploy_template` for `send-message` (Slack) or `create-message` (Discord), `function_type: "action"`. Poll `get_deployment_status`, then `functions_list`.
7. Trigger once with `actions_trigger` so a real message lands before Gemma is wired.

Do not enable a inbound Slack Events webhook as the buzzer path. Inbound webhooks are for `/join` or workspace events, not the prize recap.

## Path B: REST fallback (no Management MCP)

Same HTTP API the MCP wraps. Base URL `https://api.nango.dev`. Header `Authorization: Bearer $NANGO_API_KEY`.

Create integration (Slack shared app):

```bash
curl -sS -X POST https://api.nango.dev/integrations \
  -H "Authorization: Bearer $NANGO_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"provider":"slack","unique_key":"slack","display_name":"Slack","credentials":{"type":"OAUTH2"}}'
```

If that payload is rejected, search current Nango HTTP API docs (`https://nango.dev/docs/llms.txt`) and retry. Do not guess a body after one 4xx.

Mint a connect session:

```bash
curl -sS -X POST https://api.nango.dev/connect/sessions \
  -H "Authorization: Bearer $NANGO_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"allowed_integrations":["slack"],"tags":{"end_user_id":"wordhunt-buzzer"}}'
```

List connections, deploy templates, and trigger actions using the current public API. The smoke-test Gemma (or a seat) should call at the buzzer:

```bash
curl -sS -X POST https://api.nango.dev/action/trigger \
  -H "Authorization: Bearer $NANGO_API_KEY" \
  -H "Connection-Id: $NANGO_CONNECTION_ID" \
  -H "Provider-Config-Key: slack" \
  -H "Content-Type: application/json" \
  -d '{"action_name":"send-message","input":{"channel_id":"C0123456789","text":"Word Hunt recap: …"}}'
```

Runtime MCP equivalent (what Gemma should be given as a tool server):

```bash
curl -sS -X POST https://api.nango.dev/mcp \
  -H "Authorization: Bearer $NANGO_API_KEY" \
  -H "connection-id: $NANGO_CONNECTION_ID" \
  -H "provider-config-key: slack" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
```

## Wiring Gemma

vLLM flags already planned: `--tool-call-parser gemma4 --enable-auto-tool-choice`. Expose only Nango runtime tools (`send-message` or `create-message`), not Management MCP.

Store in Cloud Agent / Cloud Run secrets, not git:

- `NANGO_API_KEY`
- `NANGO_CONNECTION_ID` (after the user authorizes)
- `NANGO_INTEGRATION_ID` (`slack` or `discord`)
- Slack `NANGO_SLACK_CHANNEL_ID` or Discord channel id

## Docs

- Coding agent setup: https://nango.dev/docs/getting-started/coding-agent-setup
- Management MCP tools: https://nango.dev/docs/reference/backend/management-mcp
- Tool calling: https://nango.dev/docs/getting-started/use-cases/tool-calling
- Slack templates: https://nango.dev/docs/api-integrations/slack
- Discord templates: https://nango.dev/docs/api-integrations/discord
