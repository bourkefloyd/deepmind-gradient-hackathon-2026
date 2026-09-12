# AGENTS.md

## Cursor Cloud specific instructions

Cloud Agents do **not** load project `.cursor/mcp.json`. That file is for the Cursor IDE and CLI only.

Personal Cursor accounts have **no** Dashboard → Integrations page. Team Integrations is team-admin only.

### Optional: attach Lambda as a Cloud Agent MCP

Do this on the **website**, not in the desktop Agents window (the desktop composer does not have this control).

1. Open [cursor.com/agents](https://cursor.com/agents) in a browser.
2. Click the **+** button to the **left of the prompt bar** (next to the model picker). It is labeled for files, skills, and MCP servers.
3. Hover **MCP Servers** → **Add MCP**.
4. Add a custom server:
   - Name: `lambda`
   - If the form offers **stdio** / Command: `npx`, args `-y` `@strand-ai/lambda-mcp`, env `LAMBDA_API_KEY` = the Cloud Agent secret
   - If the form only offers **HTTP URL**: skip MCP. There is no official Lambda HTTP MCP. Use the skill below instead.
5. Enable `lambda` for new runs. This existing agent cannot pick it up mid-run.

### Lambda Cloud without MCP

`LAMBDA_API_KEY` is already an environment secret. Follow `.cursor/skills/lambda-cloud/SKILL.md` and call `https://cloud.lambda.ai/api/v1` directly. Always send a non-default `User-Agent` (Python urllib is blocked by Cloudflare 1010).
