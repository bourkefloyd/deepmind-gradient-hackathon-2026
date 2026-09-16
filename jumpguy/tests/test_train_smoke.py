"""Tiny collect + BC so the train path is actually exercised."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jumpguy.collect import collect_sim
from jumpguy.train import train_bc


class TrainSmokeTests(unittest.TestCase):
    def test_collect_and_bc(self):
        with tempfile.TemporaryDirectory() as td:
            data = Path(td) / "rollouts.npz"
            meta = collect_sim(episodes=3, out=data, seed=1, max_ticks=800, render=True)
            self.assertGreater(meta["transitions"], 20)
            out = Path(td) / "run"
            bc = train_bc(data, out, steps=8, batch_size=16, device_name="cpu", width=16, val_every=8)
            self.assertTrue((out / "model.pt").exists())
            self.assertGreater(bc["params"], 10_000)
            self.assertEqual(bc["kind"], "bc")
