"""Unit tests for the Discord templates on the demo payload.

    python -m integrations.test_formatters      (or: python -m unittest integrations.test_formatters)
"""
from __future__ import annotations

import unittest

from . import formatters
from .demo import sample_payloads

EXPECTED_RESULTS = """🎮 **Room 9N3G — Round 1 Results** 🎮

🥇 🤖 **Reflex-B** · *heuristic*
**4,700 pts** · 9 words
Best: `ANELES` **+1,400**

🥈 🤖 **Reflex-A** · *heuristic*
**3,400 pts** · 15 words
Best: `DATES` **+800**

🥉 🧑 **Player-5347**
**0 pts** · 0 words

**Board Stats**
📚 **236** possible words
💰 **154,000** max score
🔠 Longest: `DEPILATE` · `DEPLANES` · `ENDPLATE` · `SELENATE` · `TADPOLES` · `DELATES`

⚔️ **Think you can beat the bots?**
**Rematch →** https://wordhunt-pngitthrva-uw.a.run.app/r/9N3G"""

BOURKE_PAYLOAD = {
    "code": "9N3G", "round": 1, "url": "https://wordhunt-pngitthrva-uw.a.run.app/r/9N3G",
    "seats": [
        {"name": "Reflex-A", "kind": "ai", "label": "heuristic", "score": 3400, "n_words": 15, "best_word": "dates", "best_pts": 800},
        {"name": "Player-5347", "kind": "human", "label": "", "score": 0, "n_words": 0},
        {"name": "Reflex-B", "kind": "ai", "label": "heuristic", "score": 4700, "n_words": 9, "best_word": "aneles", "best_pts": 1400},
    ],
    "board_best": ["depilate", "deplanes", "endplate", "selenate", "tadpoles", "delates", "delapse", "elapsed"],
    "max_score": 154000, "n_board_words": 236,
}


class RoundEnded(unittest.TestCase):
    def test_matches_bourkes_sample_exactly(self):
        self.assertEqual(formatters.round_ended(BOURKE_PAYLOAD), EXPECTED_RESULTS)

    def test_demo_payload(self):
        p = sample_payloads("ABCD", "https://x.run.app/r/ABCD")["round_ended"]
        out = formatters.round_ended(p)
        self.assertTrue(out.startswith("🎮 **Room ABCD — Round 1 Results** 🎮\n\n🥇 🧑 **Bourke**\n**4,200 pts** · 11 words · 1 invalid\nBest: `LANTERN` **+1,800**"))
        self.assertIn("🥈 🤖 **Nano** · *nano 10M*", out)
        self.assertIn("🥉 🤖 **Gemma 12B** · *gemma-4-12b*", out)
        self.assertIn("**4.** 🤖 **Reflex** · *heuristic*", out)        # beyond 3rd: numbered
        self.assertIn("📚 **96** possible words\n💰 **41,200** max score", out)
        self.assertIn("🔠 Longest: `ETERNAL` · `LANTERN` · `ANTLER` · `RENTAL`", out)
        self.assertTrue(out.endswith("⚔️ **Think you can beat the bots?**\n**Rematch →** https://x.run.app/r/ABCD"))
        self.assertLess(len(out), 2000)

    def test_missing_board_stats_still_renders(self):
        p = {"code": "ZZZZ", "round": 2, "seats": [{"name": "A", "kind": "human", "score": 100, "n_words": 1}]}
        out = formatters.round_ended(p)
        self.assertIn("🥇 🧑 **A**\n**100 pts** · 1 words", out)
        self.assertNotIn("Board Stats", out)
        self.assertNotIn("Rematch", out)


class ShortPosts(unittest.TestCase):
    def test_room_created(self):
        p = sample_payloads("ABCD", "https://x.run.app/r/ABCD")["room_created"]
        out = formatters.room_created(p)
        self.assertEqual(out.splitlines()[0], "🎮 **Room ABCD** is open 🎮")
        self.assertIn("**Join →** https://x.run.app/r/ABCD", out)
        self.assertIn("🧑 **Bourke** · 🤖 **Nano** · *nano 10M*", out)

    def test_round_started(self):
        p = sample_payloads("ABCD")["round_started"]
        out = formatters.round_started(p)
        self.assertEqual(out.splitlines()[0], "🏁 **Room ABCD — Round 1** is live 🏁")
        self.assertIn("**75 s**", out)


if __name__ == "__main__":
    unittest.main()
