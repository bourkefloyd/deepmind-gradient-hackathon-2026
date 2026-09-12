"""Respan gateway routing for every Gemma call (observability / traces / evals).

Respan (https://respan.ai) is an OpenAI-compatible AI gateway: point `base_url` at
`https://api.respan.ai/api`, authenticate with `Authorization: Bearer <RESPAN_API_KEY>`, and
every request is logged as a span with cost, latency, tokens and whatever tags we attach.
Self-hosted / custom upstreams (our Gemma on mlx-vlm or a Lambda vLLM box) are supported two
ways, both verified against the docs (2026-09):

- Providers page -> "Add Custom Provider" (name, base URL, API key) + a custom model whose ID is
  the model name the endpoint expects (docs/integrations/gateway/model-providers/custom).
- Per request: `credential_override: {"<model>": {"api_base": ..., "api_key": ...}}` sends this
  call to a different endpoint than the provider default (same page, "Override credentials").
  We attach it automatically when the upstream is not loopback (Respan's cloud cannot reach
  `localhost`; put a tunnel URL in MLXVLM_BASE_URL / GEMMA_BASE_URL for the Mac).

Tagging (docs/documentation/features/gateway/respan-params): `customer_identifier`,
`custom_identifier`, `thread_identifier`, `span_name`, `metadata` are accepted as top-level
body fields (OpenAI SDK: `extra_body`). Each caller passes a fixed name — `gemma-seat`,
`commentator`, `nano-vs-gemma-eval` — which becomes the span name, the customer id and
`metadata.trace`, so the three show up as separate, filterable groups on the Logs page.

Env:
  RESPAN_ENABLED=1        route through Respan (default off -> callers hit Gemma directly)
  RESPAN_API_KEY          Respan API key (required when enabled; otherwise silently disabled)
  RESPAN_BASE_URL         gateway base (default https://api.respan.ai/api)
  RESPAN_MODEL            optional: model ID to send instead of the upstream model name, e.g. a
                          custom model created on the Models page, or a hosted Gemma for the
                          fallback path (see FALLBACK in docs/architecture.md "Respan")
  RESPAN_CREDENTIAL_OVERRIDE  1/0 force the per-request upstream override on/off (default: on
                          when the upstream URL is not loopback and RESPAN_MODEL is unset)
  RESPAN_ENV              metadata.environment (default "dev"; deploy.sh sets "cloud-run")
  RESPAN_MODE             proxy (default: calls go through the gateway) | log (call Gemma directly, then
                          POST the finished call to /request-logs/create — Respan's "log without
                          proxying"; works for localhost Gemma and needs no custom model on Respan)

Stdlib only so `gemma_seat/`, `wordhunt/seats/` and `integrations/` can all import it.
"""
from __future__ import annotations

import base64
import json
import os
import platform
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

DEFAULT_BASE_URL = "https://api.respan.ai/api"
LOGS_URL = "https://platform.respan.ai/platform/logs"
APP = "wordhunt-vs"

CALLER_GEMMA_SEAT = "gemma-seat"
CALLER_COMMENTATOR = "commentator"
CALLER_EVAL = "nano-vs-gemma-eval"
# The in-server registry seat (wordhunt/seats/gemma.py) is the same product surface as the Mac bot,
# so it also reports as `gemma-seat`, distinguished by metadata.where = "server".

_TRUE = {"1", "true", "yes", "on"}


def enabled(env: dict[str, str] | None = None) -> bool:
    """True when RESPAN_ENABLED is truthy AND a key is present (no key -> never break a call)."""
    e = os.environ if env is None else env
    return e.get("RESPAN_ENABLED", "0").strip().lower() in _TRUE and bool(e.get("RESPAN_API_KEY", "").strip())


def base_url(env: dict[str, str] | None = None) -> str:
    e = os.environ if env is None else env
    return (e.get("RESPAN_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def api_key(env: dict[str, str] | None = None) -> str:
    e = os.environ if env is None else env
    return e.get("RESPAN_API_KEY", "").strip()


def _is_loopback(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1", "0.0.0.0") or host.endswith(".local")


@dataclass
class Route:
    """Where a chat-completions call goes and what to add to the body / headers."""

    base_url: str
    api_key: str
    model: str
    via_respan: bool
    upstream_base_url: str = ""
    upstream_model: str = ""
    body_extra: dict[str, Any] = field(default_factory=dict)   # merge at the top level of the JSON body
    headers: dict[str, str] = field(default_factory=dict)      # extra HTTP headers (Authorization is separate)

    @property
    def extra_body(self) -> dict[str, Any]:  # OpenAI SDK spelling
        return dict(self.body_extra)

    def auth_headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json", **self.headers}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def completions_url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    def describe(self) -> str:
        if not self.via_respan:
            return f"direct {self.base_url} model={self.model}"
        ov = "credential_override" if "credential_override" in self.body_extra else "provider default"
        return f"respan {self.base_url} model={self.model} upstream={self.upstream_base_url or '-'} ({ov})"


def respan_params(caller: str, *, upstream_model: str, upstream_base_url: str = "", thread: str | None = None,
                  metadata: dict[str, Any] | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    """The Respan tag block for one caller (also what the unit test asserts on)."""
    e = os.environ if env is None else env
    md: dict[str, Any] = {
        "app": APP,
        "trace": caller,
        "caller": caller,
        "upstream_model": upstream_model,
        "environment": e.get("RESPAN_ENV", "dev"),
        "host": platform.node() or "unknown",
    }
    if upstream_base_url:
        md["upstream_base_url"] = upstream_base_url
    build = e.get("WH_BUILD")
    if build:
        md["build"] = build
    if metadata:
        md.update({k: v for k, v in metadata.items() if v is not None})
    params: dict[str, Any] = {
        "customer_identifier": f"{APP}/{caller}",
        "custom_identifier": caller,
        "span_name": caller,
        "metadata": {k: str(v) for k, v in md.items()},
    }
    if thread:
        params["thread_identifier"] = f"{caller}:{thread}"
    return params


def route(caller: str, upstream_base_url: str, upstream_api_key: str, model: str, *, thread: str | None = None,
          metadata: dict[str, Any] | None = None, env: dict[str, str] | None = None) -> Route:
    """Decide where `caller`'s chat call goes.

    Disabled: `Route(upstream, upstream key, model, via_respan=False)` — current behaviour.
    Enabled: base_url/API key are Respan's; `model` is RESPAN_MODEL if set, else the upstream
    model name (must exist as a custom model on Respan, or be a hosted model); tags are added,
    plus a per-request `credential_override` to the upstream when that makes sense.
    """
    e = os.environ if env is None else env
    if not enabled(e) or log_mode(e):
        return Route(upstream_base_url.rstrip("/"), upstream_api_key, model, False, body_extra={})

    respan_model = (e.get("RESPAN_MODEL") or "").strip() or model
    body = respan_params(caller, upstream_model=model, upstream_base_url=upstream_base_url, thread=thread,
                         metadata=metadata, env=e)

    ov = e.get("RESPAN_CREDENTIAL_OVERRIDE", "").strip().lower()
    want_override = ov in _TRUE if ov else (bool(upstream_base_url) and not _is_loopback(upstream_base_url)
                                             and not e.get("RESPAN_MODEL"))
    if want_override and upstream_base_url:
        cred: dict[str, str] = {"api_base": upstream_base_url.rstrip("/")}
        if upstream_api_key:
            cred["api_key"] = upstream_api_key
        body["credential_override"] = {respan_model: cred}

    return Route(base_url(e), api_key(e), respan_model, True, upstream_base_url=upstream_base_url.rstrip("/"),
                 upstream_model=model, body_extra=body)


def log_mode(env: dict[str, str] | None = None) -> bool:
    """RESPAN_MODE=log: call Gemma directly and ship the span to Respan afterwards (`request-logs/create`).

    This is Respan's documented "Log without proxying" path for custom endpoints. It is the way to
    get traces when the gateway cannot reach the upstream (Gemma on a Mac at localhost) or the
    custom model is not yet resolvable by the gateway. Default mode is `proxy`."""
    e = os.environ if env is None else env
    return enabled(e) and e.get("RESPAN_MODE", "proxy").strip().lower() == "log"


def log_request(caller: str, *, model: str, messages: list[dict[str, Any]], completion: str, latency_s: float,
                usage: dict[str, Any] | None = None, thread: str | None = None, metadata: dict[str, Any] | None = None,
                upstream_base_url: str = "", status_code: int = 200, error: str = "", timeout_s: float = 10.0,
                env: dict[str, str] | None = None) -> dict[str, Any]:
    """POST one finished LLM call to Respan as a span (async logging; never raises).

    Returns {"ok": bool, "status": int, "custom_identifier": ..., "body": <response or error text>}.
    """
    import json as _json
    import urllib.error
    import urllib.request

    e = os.environ if env is None else env
    p = respan_params(caller, upstream_model=model, upstream_base_url=upstream_base_url, thread=thread, metadata=metadata, env=e)
    body: dict[str, Any] = {
        "model": model,
        "prompt_messages": messages,
        "completion_message": {"role": "assistant", "content": completion},
        "generation_time": round(latency_s, 3),
        "latency": round(latency_s, 3),
        "status_code": status_code,
        "customer_identifier": p["customer_identifier"],
        "custom_identifier": p["custom_identifier"],
        "span_name": p["span_name"],
        "metadata": p["metadata"],
        "customer_params": {"customer_identifier": p["customer_identifier"]},
    }
    if thread:
        body["thread_identifier"] = p["thread_identifier"]
    if usage:
        body["prompt_tokens"] = usage.get("prompt_tokens")
        body["completion_tokens"] = usage.get("completion_tokens")
    if error:
        body["error_message"] = error
    req = urllib.request.Request(base_url(e) + "/request-logs/create/", data=_json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {api_key(e)}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
            try:
                out = _json.loads(raw) if raw else {}
            except ValueError:
                out = {"raw": raw[:200].decode(errors="replace")}
            return {"ok": True, "status": resp.status, "custom_identifier": p["custom_identifier"], "body": out}
    except urllib.error.HTTPError as ex:
        return {"ok": False, "status": ex.code, "custom_identifier": p["custom_identifier"], "body": ex.read()[:300].decode(errors="replace")}
    except Exception as ex:  # noqa: BLE001 - logging must never break a game call
        return {"ok": False, "status": 0, "custom_identifier": p["custom_identifier"], "body": f"{type(ex).__name__}: {ex}"}


def params_header(params: dict[str, Any]) -> dict[str, str]:
    """`X-Data-Respan-Params` (base64 JSON) for clients whose body cannot be changed."""
    return {"X-Data-Respan-Params": base64.b64encode(json.dumps(params).encode()).decode()}


def trace_url(custom_identifier: str | None = None) -> str:
    """Logs page; Respan filters by custom property / Custom ID in the UI (no stable deep-link documented)."""
    return LOGS_URL if not custom_identifier else f"{LOGS_URL}?custom_identifier={custom_identifier}"


def status(env: dict[str, str] | None = None) -> dict[str, Any]:
    """For /api/health and the smoke script: never leaks the key."""
    e = os.environ if env is None else env
    return {
        "enabled": enabled(e),
        "requested": e.get("RESPAN_ENABLED", "0").strip().lower() in _TRUE,
        "key_present": bool(api_key(e)),
        "base_url": base_url(e),
        "mode": "log" if log_mode(e) else "proxy",
        "model_override": e.get("RESPAN_MODEL") or None,
    }
