"""Room persistence: durable room state as JSON objects in a GCS bucket (or a local directory).

Rooms live in memory per Cloud Run revision; a deploy or restart wipes them. This module snapshots
each room's durable state on every state transition (and every few seconds during a race) and
rehydrates it on startup or on an unknown-code lookup, so shared links survive deploys.

Backends (env WH_ROOM_STORE):
  gs://bucket[/prefix]   GCS via the JSON API; token from the metadata server on Cloud Run, or
                         `gcloud auth print-access-token` locally. No client library needed.
  file:/some/dir         local directory (dev/tests)
  (unset)                persistence off

Objects: <prefix>/rooms/<CODE>.json. Rooms older than WH_ROOM_TTL_S (default 3 h) are ignored.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

log = logging.getLogger("wordhunt.persist")

TTL_S = float(os.environ.get("WH_ROOM_TTL_S", 3 * 3600))
RACE_SAVE_EVERY_S = float(os.environ.get("WH_ROOM_SAVE_EVERY_S", 5))


class _Token:
    def __init__(self):
        self.value = ""
        self.expires = 0.0

    def get(self) -> str:
        if time.time() < self.expires - 60 and self.value:
            return self.value
        try:
            req = urllib.request.Request(
                "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
                headers={"Metadata-Flavor": "Google"})
            with urllib.request.urlopen(req, timeout=2) as r:
                d = json.loads(r.read())
            self.value, self.expires = d["access_token"], time.time() + float(d.get("expires_in", 3600))
        except Exception:
            out = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True, text=True, timeout=30)
            if out.returncode != 0:
                raise RuntimeError("no GCP access token (metadata server or gcloud)")
            self.value, self.expires = out.stdout.strip(), time.time() + 1800
        return self.value


class GcsBackend:
    def __init__(self, url: str):
        rest = url[len("gs://"):]
        self.bucket, _, prefix = rest.partition("/")
        self.prefix = (prefix.strip("/") + "/rooms/") if prefix.strip("/") else "rooms/"
        self.token = _Token()

    def _req(self, method: str, url: str, data: bytes | None = None, ctype: str | None = None) -> bytes:
        headers = {"Authorization": "Bearer " + self.token.get()}
        if ctype:
            headers["Content-Type"] = ctype
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.read()

    def _obj(self, code: str) -> str:
        return urllib.parse.quote(self.prefix + code + ".json", safe="")

    def put(self, code: str, doc: dict) -> None:
        url = f"https://storage.googleapis.com/upload/storage/v1/b/{self.bucket}/o?uploadType=media&name={self._obj(code)}"
        self._req("POST", url, json.dumps(doc).encode(), "application/json")

    def get(self, code: str) -> dict | None:
        url = f"https://storage.googleapis.com/storage/v1/b/{self.bucket}/o/{self._obj(code)}?alt=media"
        try:
            return json.loads(self._req("GET", url))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise

    def delete(self, code: str) -> None:
        url = f"https://storage.googleapis.com/storage/v1/b/{self.bucket}/o/{self._obj(code)}"
        try:
            self._req("DELETE", url)
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise

    def list_codes(self) -> list[str]:
        url = f"https://storage.googleapis.com/storage/v1/b/{self.bucket}/o?prefix={urllib.parse.quote(self.prefix, safe='')}&fields=items(name)"
        d = json.loads(self._req("GET", url))
        out = []
        for it in d.get("items", []):
            name = it["name"][len(self.prefix):]
            if name.endswith(".json"):
                out.append(name[:-5])
        return out


class FileBackend:
    def __init__(self, url: str):
        self.dir = url[len("file:"):]
        os.makedirs(self.dir, exist_ok=True)

    def _p(self, code: str) -> str:
        return os.path.join(self.dir, code + ".json")

    def put(self, code: str, doc: dict) -> None:
        tmp = self._p(code) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(doc, f)
        os.replace(tmp, self._p(code))

    def get(self, code: str) -> dict | None:
        try:
            with open(self._p(code)) as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    def delete(self, code: str) -> None:
        try:
            os.remove(self._p(code))
        except FileNotFoundError:
            pass

    def list_codes(self) -> list[str]:
        return [f[:-5] for f in os.listdir(self.dir) if f.endswith(".json")]


class RoomStore:
    """Async facade: writes are coalesced per room and run in a thread; never raises into the game loop."""

    def __init__(self, url: str | None):
        self.backend = None
        if url and url.startswith("gs://"):
            self.backend = GcsBackend(url)
        elif url and url.startswith("file:"):
            self.backend = FileBackend(url)
        self.url = url
        self._inflight: set[str] = set()
        self._dirty: dict[str, dict] = {}
        self.stats = {"saves": 0, "errors": 0, "loads": 0}

    @property
    def enabled(self) -> bool:
        return self.backend is not None

    def save(self, code: str, doc: dict) -> None:
        """Schedule a write; if one is in flight for this room, the latest doc is written after it."""
        if not self.enabled:
            return
        self._dirty[code] = doc
        if code in self._inflight:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._inflight.add(code)
        loop.create_task(self._drain(code))

    async def _drain(self, code: str) -> None:
        try:
            while code in self._dirty:
                doc = self._dirty.pop(code)
                try:
                    await asyncio.to_thread(self.backend.put, code, doc)
                    self.stats["saves"] += 1
                except Exception as e:
                    self.stats["errors"] += 1
                    log.warning("room save %s failed: %s", code, e)
        finally:
            self._inflight.discard(code)

    async def load(self, code: str) -> dict | None:
        if not self.enabled:
            return None
        try:
            doc = await asyncio.to_thread(self.backend.get, code)
        except Exception as e:
            self.stats["errors"] += 1
            log.warning("room load %s failed: %s", code, e)
            return None
        if doc and time.time() - doc.get("created_at", 0) > TTL_S:
            return None
        if doc:
            self.stats["loads"] += 1
        return doc

    async def load_all(self) -> list[dict]:
        if not self.enabled:
            return []
        try:
            codes = await asyncio.to_thread(self.backend.list_codes)
        except Exception as e:
            self.stats["errors"] += 1
            log.warning("room list failed: %s", e)
            return []
        out = []
        for c in codes:
            d = await self.load(c)
            if d:
                out.append(d)
        return out

    async def delete(self, code: str) -> None:
        if not self.enabled:
            return
        try:
            await asyncio.to_thread(self.backend.delete, code)
        except Exception as e:
            log.warning("room delete %s failed: %s", code, e)


_store: RoomStore | None = None


def store() -> RoomStore:
    global _store
    if _store is None:
        _store = RoomStore(os.environ.get("WH_ROOM_STORE"))
    return _store
