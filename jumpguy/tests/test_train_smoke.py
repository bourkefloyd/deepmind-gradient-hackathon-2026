"""Tiny collect + BC so the train path is actually exercised."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from jumpguy.collect import collect_sim
from jumpguy.train import expand_jump_labels, train_bc


class TrainSmokeTests(unittest.TestCase):
    def test_collect_and_bc(self):
        with tempfile.TemporaryDirectory() as td:
            data = Path(td) / "rollouts.npz"
            meta = collect_sim(episodes=3, out=data, seed=1, max_ticks=800, render=True)
            self.assertGreater(meta["transitions"], 20)
            out = Path(td) / "run"
            bc = train_bc(
                data, out, steps=8, batch_size=16, device_name="cpu", width=16, val_every=8, score_every=0
            )
            self.assertTrue((out / "model.pt").exists())
            self.assertGreater(bc["params"], 10_000)
            self.assertEqual(bc["kind"], "bc")

    def test_expand_jump_labels(self):
        actions = np.array([0, 0, 0, 1, 0, 0, 1, 0], dtype=np.int64)
        dones = np.array([0, 0, 0, 0, 0, 1, 0, 0], dtype=np.bool_)
        out = expand_jump_labels(actions, dones, radius=2)
        self.assertEqual(out.tolist(), [0, 1, 1, 1, 0, 0, 1, 0])
