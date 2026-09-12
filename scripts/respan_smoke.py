#!/usr/bin/env python3
"""Fire ONE tagged chat call through the Respan gateway and print the trace id / URL.

    RESPAN_API_KEY=... python scripts/respan_smoke.py                       # hosted model on Respan (default)
    RESPAN_API_KEY=... python scripts/respan_smoke.py --model gemma-4-12b   # custom model registered on Respan
    RESPAN_API_KEY=... python scripts/respan_smoke.py --upstream http://LAMBDA_IP:8000/v1 --upstream-key tok \
                                                      --model google/gemma-4-12b-it   # per-request credential_override
    python scripts/respan_smoke.py --base-url http://127.0.0.1:8811/api --api-key x   # against a mock gateway

Forces RESPAN_ENABLED=1 for this process. The request is tagged like a real caller (default
`respan-smoke`, or --caller gemma-seat|commentator|nano-vs-gemma-eval) with a unique
custom_identifier so it is easy to find on https://platform.respan.ai/platform/logs.
Exit code 0 = the gateway answered 200 and returned a completion id; the key is never printed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from integrations import respan  # noqa: E402

# Small hosted Gemma slugs to try when no upstream/model is given (the catalog decides; first 2xx wins).
HOSTED_GEMMA_CANDIDATES = ["gemini/gemma-3-27b-it", "gemma-3-27b-it", "groq/gemma2-9b-it", "gemini/gemma-3-12b-it"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default=os.environ.get("RESPAN_BASE_URL", respan.DEFAULT_BASE_URL))
    p.add_argument("--api-key", default=os.environ.get("RESPAN_API_KEY", ""), help="Respan key (env RESPAN_API_KEY)")
    p.add_argument("--model", default=os.environ.get("RESPAN_MODEL", ""), help="model id to send (default: hosted Gemma candidates)")
    p.add_argument("--upstream", default="", help="upstream OpenAI-compatible base URL for a credential_override (vLLM / tunnel)")
    p.add_argument("--upstream-key", default=os.environ.get("GEMMA_API_KEY", ""))
    p.add_argument("--caller", default="respan-smoke")
    p.add_argument("--prompt", default="Board: S T O N / E A B C / D E F G / H I J K. List three English words you can trace. Words only, one per line.")
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--stream", action="store_true", help="use SSE streaming like the gemma seat does")
    p.add_argument("--mode", choices=["proxy", "log"], default=os.environ.get("RESPAN_MODE", "proxy"),
                   help="proxy: chat call through the gateway; log: POST a finished Gemma call to /request-logs/create/ (no upstream needed)")
    a = p.parse_args(argv)
    if not a.api_key:
        print("no RESPAN_API_KEY (env or --api-key)", file=sys.stderr)
        return 2

    env = {**os.environ, "RESPAN_ENABLED": "1", "RESPAN_API_KEY": a.api_key, "RESPAN_BASE_URL": a.base_url}
    if a.mode == "log":
        run_id = f"{a.caller}-{uuid.uuid4().hex[:8]}"
        model = a.model or os.environ.get("GEMMA_MODEL", "mlx-community/gemma-4-12B-it-4bit")
        print(f"-> POST {a.base_url.rstrip('/')}/request-logs/create/  model={model}  caller={a.caller}  run_id={run_id}")
        res = respan.log_request(a.caller, model=model, messages=[{"role": "user", "content": a.prompt}],
                                 completion="STONE\nNOTES\nTONES", latency_s=1.234, usage={"prompt_tokens": 64, "completion_tokens": 9},
                                 thread=run_id, upstream_base_url=a.upstream, env=env,
                                 metadata={"where": "smoke", "run_id": run_id, "mode": "log"})
        body = res["body"] if isinstance(res["body"], dict) else {}
        print(f"<- {res['status']} ok={res['ok']}  unique_id={body.get('unique_id')}  timestamp={body.get('timestamp')}")
        if not res["ok"]:
            print(f"FAILED: {res['body']}", file=sys.stderr)
            return 1
        print(f"trace: unique_id={body.get('unique_id')}  span_name={a.caller}  thread={a.caller}:{run_id}  custom_identifier={a.caller}")
        print(f"url:   {respan.LOGS_URL}   (filter Custom ID = {a.caller}, or custom property run_id = {run_id})")
        return 0
    if a.upstream:
        env["RESPAN_CREDENTIAL_OVERRIDE"] = "1"
    run_id = f"{a.caller}-{uuid.uuid4().hex[:8]}"
    candidates = [a.model] if a.model else HOSTED_GEMMA_CANDIDATES
    last_err = ""
    for model in candidates:
        env.pop("RESPAN_MODEL", None)
        r = respan.route(a.caller, a.upstream, a.upstream_key, model, thread=run_id,
                         metadata={"where": "smoke", "run_id": run_id, "argv": " ".join(sys.argv[1:])[:200]}, env=env)
        r.body_extra["custom_identifier"] = run_id          # unique per run: the thing to search for in the UI
        body = {"model": r.model, "messages": [{"role": "user", "content": a.prompt}], "max_tokens": 60,
                "temperature": 0.2, "stream": bool(a.stream), **r.body_extra}
        print(f"-> POST {r.completions_url()}  model={r.model}  caller={a.caller}  custom_identifier={run_id}")
        if "credential_override" in r.body_extra:
            print(f"   credential_override -> {a.upstream}")
        req = urllib.request.Request(r.completions_url(), data=json.dumps(body).encode(), method="POST", headers=r.auth_headers())
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=a.timeout) as resp:
                raw = resp.read()
                hdrs = {k.lower(): v for k, v in resp.headers.items()}
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code} for {model}: {e.read()[:300].decode(errors='replace')}"
            print(f"   {last_err}")
            continue
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            print(f"   {last_err}")
            continue
        dt = time.perf_counter() - t0
        text, cid, usage = "", "", {}
        if a.stream:
            for line in raw.decode(errors="replace").splitlines():
                if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                    ch = json.loads(line[6:])
                    cid = cid or ch.get("id", "")
                    for c in ch.get("choices") or []:
                        text += (c.get("delta") or {}).get("content") or ""
                    usage = ch.get("usage") or usage
        else:
            data = json.loads(raw)
            cid = data.get("id", "")
            usage = data.get("usage") or {}
            text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        trace_hdrs = {k: v for k, v in hdrs.items() if "respan" in k or "trace" in k or "request-id" in k or "keywords" in k}
        print(f"<- 200 in {dt:.2f}s  completion_id={cid}  usage={usage}")
        if trace_hdrs:
            print(f"   gateway headers: {trace_hdrs}")
        print(f"   reply: {text.strip()[:200]!r}")
        print(f"trace: custom_identifier={run_id}  span_name={a.caller}  model={r.model}")
        print(f"url:   {respan.trace_url(run_id)}")
        print("       (Logs page -> filter Custom ID / custom property run_id; the span carries metadata.trace)")
        return 0
    print(f"FAILED: {last_err}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
