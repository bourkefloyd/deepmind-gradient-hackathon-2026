"""Respan gateway wiring against a mock gateway (no key, no network).

    python -m unittest integrations.test_respan       (or: python -m integrations.test_respan)

Spins up a local HTTP server that speaks just enough of /chat/completions (JSON + SSE stream),
records every request, and asserts that with RESPAN_ENABLED=1 each caller sends the Authorization
header, the upstream Gemma model name, the per-caller tags (span_name / customer_identifier /
custom_identifier / metadata.trace), and the credential_override to the upstream endpoint; and
that with RESPAN_ENABLED unset nothing changes (direct call, no Respan fields).
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from . import respan

UPSTREAM = "http://203.0.113.7:8000/v1"       # documentation range: never loopback, so the override applies
MODEL = "google/gemma-4-12b-it"
KEY = "rk-test-not-a-real-key"


class _Gateway(BaseHTTPRequestHandler):
    requests: list[dict] = []
    reply_text = "STONE\nNOTES\nTONES\n"
    tool_call = False

    def log_message(self, *a):  # quiet
        pass

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0")) or b"{}"))
        type(self).requests.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})
        if self.path.endswith("/request-logs/create/"):
            data = json.dumps({**body, "unique_id": "log-unique-1"}).encode()
            self.send_response(201)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for line in self.reply_text.splitlines(keepends=True):
                chunk = {"id": "chatcmpl-mock", "object": "chat.completion.chunk", "created": 0, "model": body["model"],
                         "choices": [{"index": 0, "delta": {"content": line}, "finish_reason": None}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        msg: dict = {"role": "assistant", "content": self.reply_text}
        if type(self).tool_call:
            msg = {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {
                "name": "send_discord_recap", "arguments": json.dumps({"text": "Room 9N3G: Reflex-B wins."})}}]}
        out = {"id": "chatcmpl-mock-1", "object": "chat.completion", "created": 0, "model": body["model"],
               "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class RespanGatewayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), _Gateway)
        cls.thread = threading.Thread(target=cls.srv.serve_forever, daemon=True)
        cls.thread.start()
        cls.gateway = f"http://127.0.0.1:{cls.srv.server_address[1]}/api"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        _Gateway.requests.clear()
        _Gateway.tool_call = False
        self.env = {"RESPAN_ENABLED": "1", "RESPAN_API_KEY": KEY, "RESPAN_BASE_URL": self.gateway, "RESPAN_ENV": "test"}

    def _last(self) -> dict:
        self.assertEqual(len(_Gateway.requests), 1, _Gateway.requests)
        return _Gateway.requests[0]

    def _assert_tagged(self, req: dict, caller: str, *, thread: str | None = None, model: str = MODEL):
        self.assertEqual(req["path"], "/api/chat/completions")
        self.assertEqual(req["headers"]["authorization"], f"Bearer {KEY}")
        body = req["body"]
        self.assertEqual(body["model"], model)
        self.assertEqual(body["span_name"], caller)
        self.assertEqual(body["custom_identifier"], caller)
        self.assertEqual(body["customer_identifier"], f"{respan.APP}/{caller}")
        md = body["metadata"]
        self.assertEqual(md["trace"], caller)
        self.assertEqual(md["upstream_model"], MODEL)
        self.assertEqual(md["environment"], "test")
        self.assertTrue(all(isinstance(v, str) for v in md.values()), md)
        self.assertEqual(body["credential_override"], {model: {"api_base": UPSTREAM, "api_key": "up-key"}})
        if thread:
            self.assertEqual(body["thread_identifier"], f"{caller}:{thread}")

    # ---- routing decisions (pure) --------------------------------------------------------------
    def test_disabled_is_passthrough(self):
        for env in ({}, {"RESPAN_ENABLED": "1"}, {"RESPAN_API_KEY": KEY}, {"RESPAN_ENABLED": "0", "RESPAN_API_KEY": KEY}):
            r = respan.route("gemma-seat", UPSTREAM, "up-key", MODEL, env=env)
            self.assertFalse(r.via_respan, env)
            self.assertEqual((r.base_url, r.api_key, r.model, r.body_extra), (UPSTREAM, "up-key", MODEL, {}))

    def test_enabled_route_and_override(self):
        r = respan.route("gemma-seat", UPSTREAM, "up-key", MODEL, thread="ABCD", env=self.env)
        self.assertTrue(r.via_respan)
        self.assertEqual(r.completions_url(), self.gateway + "/chat/completions")
        self.assertEqual(r.auth_headers()["Authorization"], f"Bearer {KEY}")
        self.assertEqual(r.body_extra["credential_override"], {MODEL: {"api_base": UPSTREAM, "api_key": "up-key"}})
        self.assertEqual(r.body_extra["thread_identifier"], "gemma-seat:ABCD")

    def test_loopback_upstream_skips_override_unless_forced(self):
        r = respan.route("gemma-seat", "http://localhost:8080/v1", "", MODEL, env=self.env)
        self.assertNotIn("credential_override", r.body_extra)
        r = respan.route("gemma-seat", "http://localhost:8080/v1", "", MODEL, env={**self.env, "RESPAN_CREDENTIAL_OVERRIDE": "1"})
        self.assertEqual(r.body_extra["credential_override"], {MODEL: {"api_base": "http://localhost:8080/v1"}})

    def test_respan_model_override_uses_hosted_model(self):
        r = respan.route("commentator", UPSTREAM, "up-key", MODEL, env={**self.env, "RESPAN_MODEL": "gemini/gemma-3-27b-it"})
        self.assertEqual(r.model, "gemini/gemma-3-27b-it")
        self.assertNotIn("credential_override", r.body_extra)      # hosted on Respan: no upstream to point at
        self.assertEqual(r.body_extra["metadata"]["upstream_model"], MODEL)

    def test_params_header_roundtrip(self):
        import base64
        p = respan.respan_params("commentator", upstream_model=MODEL, env=self.env)
        h = respan.params_header(p)
        self.assertEqual(json.loads(base64.b64decode(h["X-Data-Respan-Params"])), p)

    def test_status_never_leaks_key(self):
        s = respan.status(self.env)
        self.assertTrue(s["enabled"] and s["key_present"])
        self.assertNotIn(KEY, json.dumps(s))

    # ---- callers end to end against the mock gateway --------------------------------------------
    def test_gemma_seat_client_streams_through_gateway(self):
        from gemma_seat.client import GemmaSeatClient
        with mock.patch.dict(os.environ, {**self.env, "MLXVLM_API_KEY": "up-key"}, clear=False):
            c = GemmaSeatClient(base_url=UPSTREAM, model=MODEL, caller="gemma-seat", thread="ROOM1", tags={"where": "mac-bot"})
        self.assertTrue(c.route.via_respan)
        r = c.words("stonesabcdefghij"[:16], "text")
        self.assertIsNone(r.error, r.error)
        self.assertEqual(r.words, ["stone", "notes", "tones"])
        req = self._last()
        self._assert_tagged(req, "gemma-seat", thread="ROOM1")
        self.assertTrue(req["body"]["stream"])
        self.assertEqual(req["body"]["metadata"]["where"], "mac-bot")
        self.assertEqual(req["body"]["metadata"]["modality"], "text")
        self.assertEqual(req["body"]["metadata"]["board"], "STONESABCDEFGHIJ")
        self.assertEqual(req["body"]["repetition_penalty"], 1.15)       # upstream knobs still travel

    def test_eval_caller_tag(self):
        from gemma_seat.client import GemmaSeatClient
        with mock.patch.dict(os.environ, {**self.env, "MLXVLM_API_KEY": "up-key"}, clear=False):
            c = GemmaSeatClient(base_url=UPSTREAM, model=MODEL, caller="nano-vs-gemma-eval", thread="seed0-n20-text")
        c.words("stonesabcdefghij", "text")
        self._assert_tagged(self._last(), "nano-vs-gemma-eval", thread="seed0-n20-text")

    def test_gemma_seat_client_direct_when_disabled(self):
        from gemma_seat.client import GemmaSeatClient
        with mock.patch.dict(os.environ, {"RESPAN_ENABLED": "0", "MLXVLM_API_KEY": "up-key"}, clear=False):
            c = GemmaSeatClient(base_url=self.gateway, model=MODEL)   # mock plays the upstream here
        c.words("stonesabcdefghij", "text")
        req = self._last()
        self.assertEqual(req["headers"]["authorization"], "Bearer up-key")
        for k in ("span_name", "customer_identifier", "custom_identifier", "metadata", "credential_override"):
            self.assertNotIn(k, req["body"])

    def test_commentator_through_gateway(self):
        from . import commentator
        from .demo import sample_payloads
        _Gateway.tool_call = True
        payload = sample_payloads()["round_ended"]
        with mock.patch.dict(os.environ, self.env, clear=False), \
             mock.patch.object(commentator.tools, "dispatch", new=mock.AsyncMock(return_value=[{"result": {"ok": True}}])):
            out = asyncio.run(commentator.commentate(payload, UPSTREAM, MODEL, deadline_s=5, fallback=False, api_key="up-key"))
        self.assertEqual(out.posted_by, "model", out.error)
        req = self._last()
        self._assert_tagged(req, "commentator", thread=str(payload["code"]))
        self.assertEqual(req["body"]["metadata"]["room"], str(payload["code"]))
        self.assertEqual(req["body"]["tools"][0]["function"]["name"], "send_discord_recap")

    def test_commentator_direct_when_disabled(self):
        from . import commentator
        from .demo import sample_payloads
        _Gateway.tool_call = True
        with mock.patch.dict(os.environ, {"RESPAN_ENABLED": "0"}, clear=False), \
             mock.patch.object(commentator.tools, "dispatch", new=mock.AsyncMock(return_value=[{"result": {}}])):
            asyncio.run(commentator.commentate(sample_payloads()["round_ended"], self.gateway, MODEL, deadline_s=5, fallback=False))
        req = self._last()
        self.assertEqual(req["headers"]["authorization"], "Bearer none")
        self.assertNotIn("span_name", req["body"])

    def test_server_seat_through_gateway(self):
        from wordhunt.seats.gemma import GemmaPolicy
        with mock.patch.dict(os.environ, self.env, clear=False):
            p = GemmaPolicy(UPSTREAM, MODEL, api_key="up-key")
            p._round = 2
            text = p._call("stonesabcdefghij")
        self.assertIn("STONE", text)
        req = self._last()
        self._assert_tagged(req, "gemma-seat", thread="server-STONESABCDEFGHIJ")
        self.assertEqual(req["body"]["metadata"]["where"], "server")
        self.assertEqual(req["body"]["metadata"]["round"], "2")
        self.assertEqual(req["body"]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(p.stats["prompt_tokens"], 10)

    # ---- RESPAN_MODE=log: direct upstream call + async span to /request-logs/create/ -------------
    def _wait_for(self, n: int, timeout: float = 5.0):
        import time
        t0 = time.time()
        while len(_Gateway.requests) < n and time.time() - t0 < timeout:
            time.sleep(0.02)
        self.assertEqual(len(_Gateway.requests), n, _Gateway.requests)

    def _assert_logged(self, req: dict, caller: str, *, thread: str):
        self.assertEqual(req["path"], "/api/request-logs/create/")
        self.assertEqual(req["headers"]["authorization"], f"Bearer {KEY}")
        b = req["body"]
        self.assertEqual((b["model"], b["span_name"], b["custom_identifier"]), (MODEL, caller, caller))
        self.assertEqual(b["customer_identifier"], f"{respan.APP}/{caller}")
        self.assertEqual(b["thread_identifier"], f"{caller}:{thread}")
        self.assertEqual(b["metadata"]["trace"], caller)
        self.assertEqual(b["completion_message"]["role"], "assistant")
        self.assertTrue(b["prompt_messages"])

    def test_log_mode_gemma_seat_client(self):
        from gemma_seat.client import GemmaSeatClient
        env = {**self.env, "RESPAN_MODE": "log", "MLXVLM_API_KEY": "up-key"}
        with mock.patch.dict(os.environ, env, clear=False):
            c = GemmaSeatClient(base_url=self.gateway, model=MODEL, caller="gemma-seat", thread="R2")   # mock = upstream
            self.assertFalse(c.route.via_respan)
            r = c.words("stonesabcdefghij", "text")
        self.assertEqual(r.words, ["stone", "notes", "tones"])
        self._wait_for(2)
        direct, logged = _Gateway.requests
        self.assertEqual(direct["path"], "/api/chat/completions")
        self.assertEqual(direct["headers"]["authorization"], "Bearer up-key")
        self.assertNotIn("span_name", direct["body"])
        self._assert_logged(logged, "gemma-seat", thread="R2")
        self.assertIn("STONE", logged["body"]["completion_message"]["content"])
        self.assertEqual(logged["body"]["metadata"]["modality"], "text")

    def test_log_mode_commentator(self):
        from . import commentator
        from .demo import sample_payloads
        _Gateway.tool_call = True
        payload = sample_payloads()["round_ended"]
        with mock.patch.dict(os.environ, {**self.env, "RESPAN_MODE": "log"}, clear=False), \
             mock.patch.object(commentator.tools, "dispatch", new=mock.AsyncMock(return_value=[{"result": {}}])):
            out = asyncio.run(commentator.commentate(payload, self.gateway, MODEL, deadline_s=5, fallback=False, api_key="up-key"))
        self.assertEqual(out.posted_by, "model")
        self._wait_for(2)
        direct, logged = _Gateway.requests
        self.assertEqual(direct["headers"]["authorization"], "Bearer up-key")
        self._assert_logged(logged, "commentator", thread=str(payload["code"]))
        self.assertEqual(logged["body"]["metadata"]["tool_called"], "True")
        self.assertEqual(logged["body"]["prompt_tokens"], 10)

    def test_log_mode_server_seat(self):
        from wordhunt.seats.gemma import GemmaPolicy
        with mock.patch.dict(os.environ, {**self.env, "RESPAN_MODE": "log"}, clear=False):
            GemmaPolicy(self.gateway, MODEL, api_key="up-key")._call("stonesabcdefghij")
        self._wait_for(2)
        self._assert_logged(_Gateway.requests[1], "gemma-seat", thread="server-STONESABCDEFGHIJ")

    def test_log_request_never_raises(self):
        r = respan.log_request("gemma-seat", model=MODEL, messages=[], completion="", latency_s=0.1,
                               env={**self.env, "RESPAN_BASE_URL": "http://127.0.0.1:9/api"}, timeout_s=0.5)
        self.assertFalse(r["ok"])

    def test_server_seat_direct_when_disabled(self):
        from wordhunt.seats.gemma import GemmaPolicy
        with mock.patch.dict(os.environ, {"RESPAN_ENABLED": "0"}, clear=False):
            GemmaPolicy(self.gateway, MODEL, api_key="up-key")._call("stonesabcdefghij")
        req = self._last()
        self.assertEqual(req["headers"]["authorization"], "Bearer up-key")
        self.assertNotIn("customer_identifier", req["body"])


if __name__ == "__main__":
    unittest.main()
