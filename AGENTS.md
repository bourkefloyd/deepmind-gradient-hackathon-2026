# AGENTS.md

## Cursor Cloud specific instructions

Cloud Agents do **not** load project `.cursor/mcp.json`. That file is for the Cursor IDE and CLI only.

Personal Cursor accounts have **no** Dashboard → Integrations page. Team Integrations is team-admin only.

### Optional: attach Nango Management MCP

Do this on the **website**, not in the desktop Agents window (the desktop composer does not have this control).

1. Create an API key in [app.nango.dev](https://app.nango.dev/) → Environment Settings → API Keys. Store it as the Cloud Agent secret `NANGO_API_KEY`. Do not commit it.
2. Open [cursor.com/agents](https://cursor.com/agents) in a browser.
3. Click the **+** button to the **left of the prompt bar** (next to the model picker).
4. Hover **MCP Servers** → **Add MCP**.
5. Add a custom **HTTP** server (not Command/stdio). URL must be exactly `https://mcp.nango.dev/mcp`.
6. Add header `Authorization` with value `Bearer <your Nango API key>`. Paste JSON if the UI has Edit JSON:

```json
{
  "nango-management": {
    "url": "https://mcp.nango.dev/mcp",
    "headers": {
      "Authorization": "Bearer your-nango-api-key"
    }
  }
}
```

7. Save, then start a **new** Cloud Agent. An already-running agent cannot pick up MCP config mid-run.

### Nango without Management MCP

`NANGO_API_KEY` as an environment secret is enough to call `https://api.nango.dev` directly. Follow `.cursor/skills/nango-game-hooks/SKILL.md`.

Hackathon rule: Gemma **function-calls** Nango at the buzzer (Slack recap or Discord message). Do **not** add a server webhook that posts the scorecard; that is an integration, not an agent, and it weakens the Nango prize narrative.
