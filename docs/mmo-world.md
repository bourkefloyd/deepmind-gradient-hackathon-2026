# Word Hunt MMO: world of rooms

Locked product and architecture for the live-match grid. Where this doc and a later PR disagree, **this file is the source of truth** until Bourke says otherwise.

Companion: [architecture.md](architecture.md) (today's single-process room), [PLAN.md](../PLAN.md) (hackathon pitch).

---

## 1. Locked shape: world of rooms

The product is a **grid of concurrent matches**. Each match is an existing `Room`: one 4x4 board, one 75 s clock, the existing WebSocket protocol (`/ws/CODE`).

This is **not** a mega-room with hundreds of seats on one board. Scale is "many small rooms," not "one huge board."

```
/world  →  paginated cards from GET /api/world
              │
              ├─ Spectate  →  existing /s/CODE  (projector / spectator)
              └─ Play      →  existing /r/CODE  (take a seat)
```

**Zoom** for this MVP is navigate into those routes. No embedded projector iframe, no World WebSocket, no Redis, no second Cloud Run instance.

The room remains the **sim shard**. The world index is a cheap read of the in-memory `rooms` dict already used by `/api/health` (`room_list`). Cards must stay cheap: no ticker, no cursors, no full seat lists.

---

## 2. MVP (this PR) — BOU-29 / BOU-30 / BOU-32

| Surface | What it does |
|---|---|
| `GET /api/world` | Lightweight cards: `code`, `state`, `round`, `humans`, `ai_count`, `top_score`, `leader`, `theme` / `level`, `quiet`, `age_s`, plus cheap extras (`remaining_s`, `spectators`, `joinable`). Query: `offset`, `limit` (default 24, max 48), `state`, `quiet=0\|1`. |
| `GET /world` | Vanilla JS grid (`static/world.html`). Filters, 4 s poll, pagination. Click **Spectate** → `/s/CODE`, **Play** → `/r/CODE`. Empty state when there are no rooms. |
| Home | Pill + "Browse live matches" link to `/world`. |
| `/api/health` `room_list` | Unchanged (smaller than world cards; still there for ops). |

Sort on the API: playing, then countdown, then results, then lobby; within a state, more humans and a higher top score first.

Quiet rooms (`POST /api/rooms` with `{"quiet": true}` or `?quiet=1`) already exist and skip Discord/Nango. They are the hook the bot filler will use. The world grid shows them with a `quiet` badge; `?quiet=0` hides them if you only want public rooms.

### Manual check

```bash
make setup-server && make words && make serve
# other terminal:
curl -sS -X POST http://localhost:8000/api/rooms | python3 -c 'import json,sys; print(json.load(sys.stdin)["code"])'
curl -sS 'http://localhost:8000/api/world'
# browser: http://localhost:8000/world  → card → Spectate or Play
# existing race path: http://localhost:8000/r/CODE  still starts / rematches over /ws/CODE
```

`make smoke` now also hits `/api/world` and `/world`.

---

## 3. Next milestone: bot filler (BOU-31)

`/world` is empty on a fresh process. A background worker should keep **~N public AI-only quiet races** warm so the grid never looks dead.

- Use the existing quiet-room path (`quiet: true` / `?quiet=1`) so filler rooms do not spam Discord.
- Prefer cheap seats (reflex / nano). Do **not** spin Gemma by default.
- Env knobs (reserved now, unused until BOU-31): `WH_WORLD_FILL_N` (target count; default `0`). `GET /api/world` already reports `filler: {target, enabled: false}`.
- Refill interval and "replace a finished quiet race" belong in that PR, not this one.

Done when: with zero humans, `/api/world` still shows several `playing` rooms.

---

## 4. After that: World WebSocket

Polling `/api/world` every few seconds is enough for a few dozen rooms on one process.

When the grid is large enough that poll+JSON is wasteful, add a **World WebSocket** (`/ws/world` or similar):

- Thumbnail deltas only (state, humans, leader, remaining_s) — not cursors or words.
- Client subscribes to the **visible page** plus at most one **focused** room.
- The per-room `/ws/CODE` protocol does not change. Zoom is still navigate (or, later, an optional embedded projector).

Do not build this until the poll is a measured problem.

---

## 5. Later: multi-instance (only if one process cannot hold every room)

Today: Cloud Run `--max-instances 1`, session affinity, in-memory `rooms`. That is correct for the demo.

Split only when one process cannot hold every live room (CPU/RAM from Hands + broadcast, not from the world card index).

Then, and only then:

1. Sticky affinity so `/ws/CODE` always hits the instance that owns that room.
2. A **shared world index** (Redis or equivalent) that instances update on room create/state/gc. `GET /api/world` reads the index, not a single process dict.
3. Still no mega-room. Still one board per room.

Redis, a second World process, and multi-instance are **out of this PR**.

---

## 6. Constraints

- Do **not** merge, force-push, buy domains, or spend. Open a PR; leave merge to Bourke.
- Do **not** reopen the mega-room vs world-of-rooms decision.
- Do **not** change the existing room WebSocket protocol to "make the world work."
- Evidence over theater: run the server, hit `/api/world`, click through `/world`.
- Hackathon Nango rule is unchanged: Gemma function-calls Nango at the buzzer. Do not add a server webhook that posts the scorecard.

---

## 7. Code map

| File | Role |
|---|---|
| `wordhunt/room.py` `Room.world_card()` | Cheap card from one room |
| `wordhunt/server.py` `GET /api/world`, `GET /world`, `world_index()` | Index + page |
| `static/world.html` | Grid UI |
| `static/index.html` | Home / room client; links to `/world` |
| `wordhunt/test_world.py` | Card + pagination + `/api/world` |

Room WebSocket, seats, Hands, and integrations are unchanged.
