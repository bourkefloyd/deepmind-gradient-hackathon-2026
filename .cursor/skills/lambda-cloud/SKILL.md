---
name: lambda-cloud
description: Manage Lambda Cloud GPU instances via the REST API when Lambda MCP tools are unavailable. Use when listing GPUs, checking capacity, launching or terminating instances, or listing SSH keys on cloud.lambda.ai.
---

# Lambda Cloud

Call the [Lambda Cloud API](https://docs-api.lambda.ai/api/cloud) with the `LAMBDA_API_KEY` environment secret. Do not print the key. Do not commit it.

There is no official Lambda MCP. Cloud Agents on a personal Cursor account often cannot attach the unofficial stdio MCP. Use this API path instead.

## Auth and headers

Base URL: `https://cloud.lambda.ai/api/v1`

Required headers:

- `Authorization: Bearer $LAMBDA_API_KEY`
- `Accept: application/json`
- `User-Agent: cursor-cloud-agent/lambda-cloud` (required; default Python/`urllib` is blocked with Cloudflare 1010)

## Endpoints

| Action | Method | Path |
| --- | --- | --- |
| List instance types / capacity | GET | `/instance-types` |
| List running instances | GET | `/instances` |
| Launch instance | POST | `/instances` |
| Terminate instances | POST | `/instance-operations/terminate` |
| List SSH keys | GET | `/ssh-keys` |
| List filesystems | GET | `/file-systems` |

Launch body (required): `region_name`, `instance_type_name`, `ssh_key_names` (exactly one key name). Optional: `name`, `file_system_names`.

Terminate body: `{ "instance_ids": ["..."] }`.

Do not launch or terminate unless the user explicitly asked. Prefer listing types and instances first.

## Example

```python
import json, os, urllib.request

key = os.environ["LAMBDA_API_KEY"]
req = urllib.request.Request(
    "https://cloud.lambda.ai/api/v1/instance-types",
    headers={
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
        "User-Agent": "cursor-cloud-agent/lambda-cloud",
    },
)
with urllib.request.urlopen(req, timeout=30) as resp:
    data = json.loads(resp.read().decode())["data"]
print(sorted(data))
```
