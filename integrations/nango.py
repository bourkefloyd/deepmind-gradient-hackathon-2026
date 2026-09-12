"""Thin client for the Nango HTTP API (stdlib only, no new dependencies).

Verified against https://nango.dev/docs (2026-09):

    POST /action/trigger                headers: Authorization: Bearer <env API key>
                                                 Provider-Config-Key: <integration id>
                                                 Connection-Id: <connection id>
                                        body:    {"action_name": "...", "input": {...}}
                                        needs API key scope environment:actions:execute
    GET  /connections                   lists connections (no credentials); filter by
                                        provider_config_key client-side. /connection is deprecated.
                                        needs scope environment:connections:list
    {GET,POST,...} /proxy/<api path>    authenticated passthrough to the provider API
                                        (Discord base URL is https://discord.com, so the path
                                        starts with /api/v10/...). Extra headers are forwarded
                                        when prefixed `nango-proxy-`.
                                        needs scope environment:proxy

All calls are synchronous (urllib); the async wrappers run them in a worker thread with
a hard timeout so the caller can never block on Nango.
"""
from __future__ import annotations

import asyncio
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from . import config

log = logging.getLogger("integrations.nango")


class NangoError(RuntimeError):
    def __init__(self, status: int, body: Any, path: str):
        super().__init__(f"nango {path} -> HTTP {status}: {str(body)[:400]}")
        self.status = status
        self.body = body
        self.path = path


@dataclass
class DryRunRecord:
    """What would have been sent. Kept so the demo/tests can assert on it."""
    method: str
    path: str
    headers: dict[str, str]
    body: Any


class NangoClient:
    def __init__(self, settings: config.Settings | None = None):
        self.s = settings or config.load()
        self.dry_run_log: list[DryRunRecord] = []

    # ---- low level -----------------------------------------------------------------------
    def _auth_headers(self, with_connection: bool) -> dict[str, str]:
        h = {"Authorization": f"Bearer {self.s.nango_secret_key}", "Content-Type": "application/json"}
        if with_connection:
            h["Provider-Config-Key"] = self.s.integration_id
            h["Connection-Id"] = self.s.connection_id
        return h

    def request(self, method: str, path: str, body: Any = None, headers: dict[str, str] | None = None,
                with_connection: bool = True) -> Any:
        """One HTTP call. In dry-run it logs the payload and returns a stub instead."""
        hdrs = self._auth_headers(with_connection)
        if headers:
            hdrs.update(headers)
        if self.s.dry_run:
            rec = DryRunRecord(method, path, _redact(hdrs), body)
            self.dry_run_log.append(rec)
            log.info("[dry-run] %s %s%s headers=%s body=%d bytes", method, self.s.nango_base_url, path,
                     json.dumps(rec.headers), len(json.dumps(body, ensure_ascii=False)) if body is not None else 0)
            log.debug("[dry-run] body=%s", json.dumps(body, ensure_ascii=False))
            return {"dry_run": True, "method": method, "path": path, "body": body}
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.s.nango_base_url + path, data=data, method=method, headers=hdrs)
        try:
            with urllib.request.urlopen(req, timeout=self.s.timeout_s) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = raw.decode(errors="replace")
            raise NangoError(e.code, parsed, path) from None

    async def arequest(self, *args, **kwargs) -> Any:
        return await asyncio.wait_for(asyncio.to_thread(self.request, *args, **kwargs), timeout=self.s.timeout_s + 1)

    # ---- actions -------------------------------------------------------------------------
    def trigger_action(self, action_name: str, input: dict[str, Any]) -> Any:
        """POST /action/trigger (synchronous execution; result is the action's output)."""
        return self.request("POST", "/action/trigger", {"action_name": action_name, "input": input})

    async def atrigger_action(self, action_name: str, input: dict[str, Any]) -> Any:
        return await self.arequest("POST", "/action/trigger", {"action_name": action_name, "input": input})

    # ---- connections ---------------------------------------------------------------------
    def list_connections(self, provider_config_key: str | None = None) -> list[dict]:
        """GET /connections, filtered to one integration. Empty in dry-run (no key to call with)."""
        if self.s.dry_run:
            log.info("[dry-run] GET /connections needs NANGO_SECRET_KEY; nothing to list")
            return []
        key = provider_config_key or self.s.integration_id
        # The documented filters are connectionId/search/tags; provider_config_key is not one of
        # them, so page through and filter here (a hackathon environment has a handful).
        out: list[dict] = []
        page = 0
        while True:
            q = urllib.parse.urlencode({"limit": 100, "page": page})
            res = self.request("GET", f"/connections?{q}", with_connection=False)
            conns = (res or {}).get("connections", [])
            out.extend(c for c in conns if not key or c.get("provider_config_key") == key)
            if len(conns) < 100:
                return out
            page += 1

    # ---- proxy (v2: threads etc.) --------------------------------------------------------
    def proxy(self, method: str, api_path: str, body: Any = None, extra_headers: dict[str, str] | None = None) -> Any:
        """Authenticated passthrough. `api_path` is the provider path, e.g. /api/v10/channels/{id}/messages."""
        headers = {f"nango-proxy-{k}": v for k, v in (extra_headers or {}).items()}
        return self.request(method, "/proxy" + api_path, body, headers)

    async def aproxy(self, *args, **kwargs) -> Any:
        return await asyncio.wait_for(asyncio.to_thread(self.proxy, *args, **kwargs), timeout=self.s.timeout_s + 1)


def _redact(h: dict[str, str]) -> dict[str, str]:
    out = dict(h)
    if "Authorization" in out:
        out["Authorization"] = "Bearer ***" if out["Authorization"].strip() != "Bearer" else "Bearer <unset>"
    return out


_default: NangoClient | None = None


def client() -> NangoClient:
    global _default
    if _default is None:
        _default = NangoClient()
    return _default
