from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from jumpguy.constants import FRAME_SIZE, FRAME_STACK
from jumpguy.model import JumpNet, JumpNetConfig, load_checkpoint, save_checkpoint
from jumpguy.observe import FrameStack, preprocess_frame


class ObserveTests(unittest.TestCase):
    def test_preprocess_and_stack(self):
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        frame[100:200, 100:200] = 200
        x = preprocess_frame(frame)
        self.assertEqual(x.shape, (FRAME_SIZE, FRAME_SIZE))
        self.assertLessEqual(float(x.max()), 1.0)
        stacker = FrameStack()
        s = stacker.reset(frame)
        self.assertEqual(s.shape, (FRAME_STACK, FRAME_SIZE, FRAME_SIZE))
        s2 = stacker.push(frame)
        self.assertEqual(s2.shape, s.shape)


class ModelTests(unittest.TestCase):
    def test_forward_and_checkpoint(self):
        import torch

        model = JumpNet(JumpNetConfig(width=16))
        x = torch.zeros(2, FRAME_STACK, FRAME_SIZE, FRAME_SIZE)
        logits, value = model(x)
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertEqual(tuple(value.shape), (2,))
        self.assertGreater(model.n_params(), 10_000)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.pt"
            save_checkpoint(path, model, extra={"kind": "test"})
            loaded, payload = load_checkpoint(path, device="cpu")
            self.assertEqual(payload.get("kind"), "test")
            y1, _ = model(x)
            y2, _ = loaded(x)
            self.assertTrue(torch.allclose(y1, y2))


if __name__ == "__main__":
    unittest.main()
