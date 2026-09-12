# AGENTS.md

## Cursor Cloud specific instructions

Cloud Agents do **not** load project `.cursor/mcp.json`. That file is for the Cursor IDE and CLI only.

To use Lambda Cloud (`https://cloud.lambda.ai/`) from a Cloud Agent:

1. Open [cursor.com/agents](https://cursor.com/agents) and add a personal MCP, or on a Team plan add it under [Dashboard → Integrations & MCP](https://cursor.com/dashboard/integrations).
2. Create a custom **stdio** server (HTTP/SSE/`mcp-remote` are not supported for this package):
   - Name: `lambda`
   - Command: `npx`
   - Args: `-y` `@strand-ai/lambda-mcp`
   - Env: `LAMBDA_API_KEY` = a key from [cloud.lambda.ai/api-keys/cloud-api](https://cloud.lambda.ai/api-keys/cloud-api)
3. Enable `lambda` for this environment before starting a new Cloud Agent.

`npx` is available in this environment. The unofficial MCP talks to `https://cloud.lambda.ai/api/v1`.

Once connected, the agent can call `list_gpu_types`, `check_availability`, `list_running_instances`, `start_instance`, `stop_instance`, and filesystem tools.
