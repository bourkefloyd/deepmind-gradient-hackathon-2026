"""World-of-rooms cards and GET /api/world.

    python -m wordhunt.test_world
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from .levels import Level
from .room import Room, Seat
from . import server


class _DummySolver:
    words = []


def _room(code: str, *, state="lobby", quiet=False, humans=0, ai=1, scores=None, theme=None) -> Room:
    r = Room(code, _DummySolver(), seat_defaults=False, quiet=quiet)
    r.state = state
    r.round_no = 1 if state != "lobby" else 0
    for i in range(ai):
        sid = f"ai:{i}"
        r.seats[sid] = Seat(sid, f"Bot-{i}", "ai", "#4da3ff")
    for i in range(humans):
        sid = f"h{i}"
        seat = Seat(sid, f"Human-{i}", "human", "#2ee87a")
        seat.sockets.add(object())  # counts as connected
        r.seats[sid] = seat
    if scores:
        for sid, pts in scores.items():
            if sid in r.seats:
                r.seats[sid].found = {"cat": pts}
    if theme:
        r.level = Level(n=1, theme=theme, board="abcdefghijklmnop")
    return r


class WorldCard(unittest.TestCase):
    def test_fields_are_cheap_and_present(self):
        r = _room("AB12", state="playing", humans=2, ai=2, scores={"h0": 800, "ai:0": 400}, theme="Cats")
        card = r.world_card(now=r.created_at + 12)
        self.assertEqual(card["code"], "AB12")
        self.assertEqual(card["state"], "playing")
        self.assertEqual(card["round"], 1)
        self.assertEqual(card["humans"], 2)
        self.assertEqual(card["ai_count"], 2)
        self.assertEqual(card["top_score"], 800)
        self.assertEqual(card["leader"]["name"], "Human-0")
        self.assertEqual(card["theme"], "Cats")
        self.assertEqual(card["level"]["n"], 1)
        self.assertFalse(card["quiet"])
        self.assertEqual(card["age_s"], 12)
        self.assertTrue(card["joinable"])
        self.assertNotIn("seats", card)
        self.assertNotIn("ticker", card)
        self.assertNotIn("board", card)

    def test_quiet_and_no_leader_when_scores_are_zero(self):
        r = _room("Q9K2", quiet=True, ai=1)
        card = r.world_card()
        self.assertTrue(card["quiet"])
        self.assertEqual(card["top_score"], 0)
        self.assertIsNone(card["leader"])


class WorldIndex(unittest.TestCase):
    def setUp(self):
        self._saved = dict(server.rooms)
        server.rooms.clear()

    def tearDown(self):
        server.rooms.clear()
        server.rooms.update(self._saved)

    def test_sort_and_pagination(self):
        server.rooms["L1"] = _room("L1", state="lobby")
        server.rooms["P1"] = _room("P1", state="playing", humans=1, scores={"h0": 100})
        server.rooms["P2"] = _room("P2", state="playing", humans=3, scores={"h0": 50})
        server.rooms["C1"] = _room("C1", state="countdown")
        out = server.world_index(limit=2, offset=0)
        self.assertEqual(out["total"], 4)
        self.assertTrue(out["has_more"])
        self.assertEqual([c["code"] for c in out["rooms"]], ["P2", "P1"])
        page2 = server.world_index(limit=2, offset=2)
        self.assertEqual([c["code"] for c in page2["rooms"]], ["C1", "L1"])
        self.assertFalse(page2["has_more"])

    def test_state_and_quiet_filters(self):
        server.rooms["A"] = _room("AAAA", state="playing", quiet=True)
        server.rooms["B"] = _room("BBBB", state="playing", quiet=False)
        server.rooms["C"] = _room("CCCC", state="lobby")
        only_live = server.world_index(state="playing")
        self.assertEqual({c["code"] for c in only_live["rooms"]}, {"AAAA", "BBBB"})
        public = server.world_index(quiet="0")
        self.assertEqual({c["code"] for c in public["rooms"]}, {"BBBB", "CCCC"})

    def test_api_world_and_page(self):
        server.rooms["ZX9K"] = _room("ZX9K", state="results", humans=1, scores={"h0": 1400}, theme="Ocean")
        with patch("wordhunt.server.get_solver", return_value=_DummySolver()):
            with TestClient(server.app) as client:
                r = client.get("/api/world")
                self.assertEqual(r.status_code, 200)
                body = r.json()
                self.assertTrue(body["ok"])
                self.assertEqual(body["total"], 1)
                card = body["rooms"][0]
                self.assertEqual(card["code"], "ZX9K")
                self.assertEqual(card["leader"]["score"], 1400)
                self.assertEqual(card["theme"], "Ocean")
                self.assertFalse(body["filler"]["enabled"])
                page = client.get("/world")
                self.assertEqual(page.status_code, 200)
                self.assertIn("The <em>World</em>", page.text)
                self.assertIn("/api/world", page.text)


class RoomWebsocketUnchanged(unittest.TestCase):
    def setUp(self):
        self._saved = dict(server.rooms)
        server.rooms.clear()

    def tearDown(self):
        server.rooms.clear()
        server.rooms.update(self._saved)

    def test_hello_still_welcomes(self):
        server.rooms["WXYZ"] = _room("WXYZ")
        with patch("wordhunt.server.get_solver", return_value=_DummySolver()):
            with TestClient(server.app) as client:
                with client.websocket_connect("/ws/WXYZ") as ws:
                    ws.send_json({"type": "hello", "player_id": "p1", "name": "Tester"})
                    welcome = ws.receive_json()
                    self.assertEqual(welcome["type"], "welcome")
                    self.assertEqual(welcome["name"], "Tester")
                    state = ws.receive_json()
                    self.assertEqual(state["type"], "state")
                    self.assertEqual(state["code"], "WXYZ")


if __name__ == "__main__":
    unittest.main()
