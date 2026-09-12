# AGENTS.md

## Cursor Cloud specific instructions

Cloud Agents do **not** load project `.cursor/mcp.json`. That file is for the Cursor IDE and CLI only.

Personal Cursor accounts have **no** Dashboard → Integrations page. Team Integrations is team-admin only.

### Optional: attach Lambda as a Cloud Agent MCP

Do this on the **website**, not in the desktop Agents window (the desktop composer does not have this control).

1. Open [cursor.com/agents](https://cursor.com/agents) in a browser.
2. Click the **+** button to the **left of the prompt bar** (next to the model picker). It is labeled for files, skills, and MCP servers.
3. Hover **MCP Servers** → **Add MCP**.
4. Add a custom **Command** server. Put **each** `npx` argument in its own Arguments row (or use Edit JSON). Do not put `-y` and the package name in one string.
5. Paste this JSON (keep your existing `LAMBDA_API_KEY` secret value):

```json
{
  "lambda": {
    "command": "npx",
    "args": ["-y", "@strand-ai/lambda-mcp"],
    "env": {
      "LAMBDA_API_KEY": "your-key"
    }
  }
}
```

Wrong: `"args": ["-y @strand-ai/lambda-mcp"]` (one string). That makes `npx` fail and Cloud Agents report “failed during live tool discovery.”
6. Save, then start a **new** Cloud Agent. This existing agent cannot pick up MCP config mid-run.

### Lambda Cloud without MCP

`LAMBDA_API_KEY` is already an environment secret. Follow `.cursor/skills/lambda-cloud/SKILL.md` and call `https://cloud.lambda.ai/api/v1` directly. Always send a non-default `User-Agent` (Python urllib is blocked by Cloudflare 1010).
