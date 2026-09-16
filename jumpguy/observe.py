"""Frame preprocessing shared by the sim renderer, live env, and the CNN."""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from .constants import FRAME_SIZE, FRAME_STACK


def rgb_to_gray(frame: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 -> uint8 HxW."""
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError(f"expected HxWx3, got {frame.shape}")
    r = frame[:, :, 0].astype(np.float32)
    g = frame[:, :, 1].astype(np.float32)
    b = frame[:, :, 2].astype(np.float32)
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)


def resize_nearest(image: np.ndarray, size: int = FRAME_SIZE) -> np.ndarray:
    """Nearest-neighbor resize (no extra deps). image is HxW or HxWxC."""
    if image.shape[0] == size and image.shape[1] == size:
        return image
    h, w = image.shape[:2]
    y = (np.arange(size) * h / size).astype(np.int32)
    x = (np.arange(size) * w / size).astype(np.int32)
    y = np.clip(y, 0, h - 1)
    x = np.clip(x, 0, w - 1)
    return image[y[:, None], x[None, :]]


def preprocess_frame(frame: np.ndarray, size: int = FRAME_SIZE) -> np.ndarray:
    """RGB uint8 -> float32 (size, size) in [0, 1]."""
    gray = rgb_to_gray(frame) if frame.ndim == 3 else frame
    small = resize_nearest(gray, size)
    return small.astype(np.float32) / 255.0


class FrameStack:
    """Keep the last K preprocessed frames, oldest first. Shape (K, S, S)."""

    def __init__(self, k: int = FRAME_STACK, size: int = FRAME_SIZE):
        self.k = k
        self.size = size
        self._buf: deque[np.ndarray] = deque(maxlen=k)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        x = preprocess_frame(frame, self.size)
        self._buf.clear()
        for _ in range(self.k):
            self._buf.append(x)
        return self.obs()

    def push(self, frame: np.ndarray) -> np.ndarray:
        if len(self._buf) == 0:
            return self.reset(frame)
        self._buf.append(preprocess_frame(frame, self.size))
        return self.obs()

    def obs(self) -> np.ndarray:
        if len(self._buf) < self.k:
            raise RuntimeError("FrameStack is empty; call reset() first")
        return np.stack(self._buf, axis=0)


def stack_to_nchw(stack: np.ndarray) -> np.ndarray:
    """(K, H, W) float32 -> (1, K, H, W) for a single-batch model forward."""
    if stack.ndim != 3:
        raise ValueError(f"expected (K,H,W), got {stack.shape}")
    return stack[None]


def encode_state_vector(
    score: float,
    player_y: float,
    player_vy: float,
    grounded: float,
    speed: float,
    obstacles: list,
    player_x: float,
    ground_y: float,
    n_obstacles: int = 3,
) -> np.ndarray:
    """Compact privileged vector for the optional state MLP / heuristic."""
    vec = np.zeros(5 + n_obstacles * 3, dtype=np.float32)
    vec[0] = score / 100.0
    vec[1] = (ground_y - player_y) / 200.0
    vec[2] = player_vy / 800.0
    vec[3] = 1.0 if grounded else 0.0
    vec[4] = speed / 520.0
    xs = sorted(obstacles, key=lambda o: o.x)
    for i, o in enumerate(xs[:n_obstacles]):
        base = 5 + i * 3
        vec[base] = (o.x - player_x) / 960.0
        vec[base + 1] = o.w / 40.0
        vec[base + 2] = 0.0 if o.passed else 1.0
    return vec


def hint_text_ratio(frame: np.ndarray) -> float:
    """Fraction of dark pixels in the Phaser hint band (GAME OVER / PRESS SPACE)."""
    h, w = frame.shape[:2]
    y0, y1 = int(0.22 * h), int(0.38 * h)
    x0, x1 = int(0.18 * w), int(0.82 * w)
    band = frame[y0:y1, x0:x1]
    if band.size == 0:
        return 0.0
    if band.ndim == 3:
        dark = band.mean(axis=2) < 70
    else:
        dark = band < 70
    return float(dark.mean())


def decode_score_digits(frame: np.ndarray) -> Optional[int]:
    """Best-effort SCORE readout from the top-left HUD. Returns None if unsure.

    The live HUD is `SCORE 00000` in Courier at (20, 16). We do not depend on
    this for the control loop (the /pass API is authoritative); it is a fallback
    for offline / guest sessions.
    """
    del frame
    return None
