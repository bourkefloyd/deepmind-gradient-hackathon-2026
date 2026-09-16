"""Policies: timed-jump heuristic (state or pixels) and a thin CNN wrapper."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .actions import Action
from .constants import (
    AIR_TIME_S,
    CANVAS_W,
    PLAYER_DISPLAY_W,
    PLAYER_X_FRAC,
)
from .sim import GameState, Obstacle


class HeuristicPolicy:
    """Jump when the nearest obstacle is about `lead` seconds from overlap.

    Tuned against the extracted Phaser numbers (air time ~0.78 s, overlap
    width = player + cactus). Works from privileged state (sim / hooked live
    game) or from a cheap color blob detector on RGB frames.
    """

    def __init__(self, lead_s: float = 0.18, min_lead_s: float = 0.08):
        self.lead_s = lead_s
        self.min_lead_s = min_lead_s
        self._prev_blob_x: Optional[float] = None
        self._est_speed = 260.0

    def reset(self) -> None:
        self._prev_blob_x = None
        self._est_speed = 260.0

    def act(self, state: Optional[GameState] = None, frame: Optional[np.ndarray] = None) -> int:
        if state is not None:
            return int(self._act_state(state))
        if frame is not None:
            return int(self._act_pixels(frame))
        return int(Action.NOOP)

    def _act_state(self, state: GameState) -> Action:
        if state.status in ("ready", "gameover"):
            return Action.JUMP
        player_right = state.player_x + PLAYER_DISPLAY_W / 2.0
        incoming = [
            o
            for o in state.obstacles
            if not o.passed and (o.x + o.w / 2.0) > state.player_x - PLAYER_DISPLAY_W / 2.0
        ]
        if not incoming:
            return Action.NOOP
        o = min(incoming, key=lambda z: z.x)
        overlap_x = o.x - o.w / 2.0
        dist = overlap_x - player_right
        speed = max(state.speed, 1.0)
        ttc = dist / speed
        overlap_dur = (PLAYER_DISPLAY_W + o.w) / speed
        max_lead = max(self.min_lead_s, AIR_TIME_S - overlap_dur - 0.04)
        lead = min(self.lead_s, max_lead)
        if -0.04 < ttc <= lead:
            return Action.JUMP
        return Action.NOOP

    def _act_pixels(self, frame: np.ndarray) -> Action:
        blobs = detect_obstacle_blobs(frame)
        if not blobs:
            # Nothing incoming: tap once to start / restart.
            self._prev_blob_x = None
            return Action.JUMP
        x = min(blobs)
        if self._prev_blob_x is not None:
            dx = self._prev_blob_x - x
            if 0.5 < dx < 40:
                # per-frame motion; assume ~60 Hz if we cannot measure dt.
                self._est_speed = 0.7 * self._est_speed + 0.3 * (dx * 60.0)
        self._prev_blob_x = x
        w = frame.shape[1]
        player_right = PLAYER_X_FRAC * w + (PLAYER_DISPLAY_W / CANVAS_W) * w / 1.0
        # PLAYER_DISPLAY_W is in game pixels; scale to this frame.
        scale = w / CANVAS_W
        player_right = PLAYER_X_FRAC * w + (PLAYER_DISPLAY_W * scale) / 2.0
        dist = x - player_right
        speed = max(self._est_speed * (w / CANVAS_W), 1.0)
        ttc = dist / speed
        if -0.05 < ttc <= self.lead_s:
            return Action.JUMP
        return Action.NOOP


def detect_obstacle_blobs(frame: np.ndarray) -> list[float]:
    """Left edges (px) of saturated blobs sitting on the ground, right of the player."""
    if frame.ndim != 3:
        raise ValueError("detect_obstacle_blobs expects HxWx3")
    h, w, _ = frame.shape
    gy = int(h * 0.72)
    y0, y1 = max(0, gy - 42), min(h, gy + 2)
    x0 = int(w * 0.28)
    band = frame[y0:y1, x0:]
    # Background is white; player is left of x0. Obstacles are saturated mid-tones.
    mn = band.min(axis=2)
    mx = band.max(axis=2)
    sat = (mx.astype(np.int16) - mn.astype(np.int16)) > 25
    darkish = mn < 230
    mask = sat & darkish
    cols = mask.mean(axis=0) > 0.18
    blobs: list[float] = []
    run = None
    for i, on in enumerate(cols):
        if on and run is None:
            run = i
        elif not on and run is not None:
            width = i - run
            if 3 <= width <= int(w * 0.12):
                blobs.append(float(x0 + run))
            run = None
    if run is not None:
        width = len(cols) - run
        if 3 <= width <= int(w * 0.12):
            blobs.append(float(x0 + run))
    return blobs


class RandomPolicy:
    def __init__(self, p_jump: float = 0.04, seed: int = 0):
        self.p_jump = p_jump
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def act(self, state: Optional[GameState] = None, frame: Optional[np.ndarray] = None) -> int:
        del state, frame
        return int(Action.JUMP if self.rng.random() < self.p_jump else Action.NOOP)


class CnnPolicy:
    """Greedy CNN. Loads a jumpguy checkpoint (`model.pt` + optional `train.json`)."""

    def __init__(self, path: str, device: str = "auto", temperature: float = 0.0):
        import torch

        from .device import torch_device
        from .model import JumpNet, load_checkpoint

        self.device = torch_device(device)
        self.model, self.meta = load_checkpoint(path, self.device)
        self.model.eval()
        self.temperature = temperature
        self._torch = torch

    def reset(self) -> None:
        pass

    def act(self, stack: np.ndarray, deterministic: Optional[bool] = None) -> int:
        torch = self._torch
        x = torch.as_tensor(stack, device=self.device, dtype=torch.float32)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.model(x)
            if deterministic or self.temperature <= 0:
                return int(logits.argmax(dim=-1).item())
            probs = torch.softmax(logits / self.temperature, dim=-1)
            return int(torch.multinomial(probs, 1).item())


def load_policy(name: str, ckpt: Optional[str] = None, device: str = "auto"):
    if name in ("heuristic", "cv", "teacher"):
        return HeuristicPolicy()
    if name == "random":
        return RandomPolicy()
    if name in ("cnn", "model"):
        if not ckpt:
            raise ValueError("cnn policy needs --ckpt")
        return CnnPolicy(ckpt, device=device)
    if name == "auto":
        if ckpt:
            try:
                return CnnPolicy(ckpt, device=device)
            except FileNotFoundError:
                pass
        return HeuristicPolicy()
    raise ValueError(f"unknown policy {name!r}")
