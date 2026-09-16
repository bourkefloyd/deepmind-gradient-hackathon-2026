"""Policies: timed-jump heuristic (state or pixels) and a thin CNN wrapper."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .actions import Action
from .constants import (
    AIR_TIME_S,
    CANVAS_W,
    PLAYER_DISPLAY_W,
    PLAYER_X_FRAC,
    TIME_TO_PEAK_S,
)
from .sim import GameState


class HeuristicPolicy:
    """Jump when the nearest obstacle is about `lead` seconds from overlap.

    Lead is derived from the live Phaser numbers: gravity 1800, jump -700
    (peak at ~0.389 s, air time ~0.778 s). Aim to put the peak near the
    middle of the player/cactus AABB overlap. Works from privileged state
    (sim / hooked live game) or from a cheap color blob detector on RGB frames.
    """

    def __init__(self, lead_s: float = 0.22, min_lead_s: float = 0.08, latency_s: float = 0.0):
        self.lead_s = lead_s
        self.min_lead_s = min_lead_s
        self.latency_s = latency_s
        self._prev_blob_x: Optional[float] = None
        self._prev_blob_t: Optional[float] = None
        self._est_speed = 260.0
        self._started = False
        self._did_jump = False
        self._prev_ttc = 1e9

    def reset(self) -> None:
        self._prev_blob_x = None
        self._prev_blob_t = None
        self._est_speed = 260.0
        self._started = False
        self._did_jump = False
        self._prev_ttc = 1e9

    def act(self, state: Optional[GameState] = None, frame: Optional[np.ndarray] = None) -> int:
        # Privileged Phaser / sim state beats pixels whenever it is real.
        if state is not None and (state.hooked or state.speed > 0 or state.obstacles or state.status != "ready"):
            return int(self._act_state(state))
        if state is not None and state.status in ("ready", "gameover"):
            return int(self._act_state(state))
        if frame is not None:
            return int(self._act_pixels(frame))
        return int(Action.NOOP)

    def _lead(self, speed: float, obs_w: float, player_w: float) -> float:
        """Seconds before AABB overlap to queue the jump (plus control latency)."""
        speed = max(speed, 1.0)
        overlap_dur = (player_w + obs_w) / speed
        # Latest safe takeoff: still airborne when the cactus exits the player.
        max_lead = max(self.min_lead_s, AIR_TIME_S - overlap_dur - 0.05)
        # Peak (v/g ≈ 0.389 s) over the middle of the overlap window.
        adaptive = TIME_TO_PEAK_S - 0.45 * overlap_dur
        preferred = self.lead_s if self.lead_s > 0 else adaptive
        chosen = max(min(preferred, max_lead), min(adaptive, max_lead))
        chosen = min(max(chosen, self.min_lead_s), max_lead)
        return chosen + self.latency_s

    def _act_state(self, state: GameState) -> Action:
        if state.status in ("ready", "gameover"):
            self._did_jump = False
            self._prev_ttc = 1e9
            return Action.JUMP
        player_right = (
            state.player_right
            if state.player_right is not None
            else state.player_x + PLAYER_DISPLAY_W / 2.0
        )
        player_left = (
            state.player_left
            if state.player_left is not None
            else state.player_x - PLAYER_DISPLAY_W / 2.0
        )
        incoming = []
        for o in state.obstacles:
            right = o.right if o.right is not None else o.x + o.w / 2.0
            if not o.passed and right > player_left:
                incoming.append(o)
        if not incoming:
            self._did_jump = False
            self._prev_ttc = 1e9
            return Action.NOOP
        o = min(incoming, key=lambda z: z.left if z.left is not None else z.x)
        overlap_x = o.left if o.left is not None else o.x - o.w / 2.0
        dist = overlap_x - player_right
        speed = max(state.speed, 1.0)
        ttc = dist / speed
        lead = self._lead(speed, o.w, PLAYER_DISPLAY_W)
        # Rising-edge: jump once as TTC enters the window (avoids 130 ms buffer-spam).
        if ttc > lead + 0.05:
            self._did_jump = False
        # Queue while airborne: Phaser jump buffer (130 ms) + coyote (100 ms)
        # fire on landing. Skipping here misses tight follow-up cacti.
        should = (not self._did_jump) and (-0.05 < ttc <= lead)
        self._prev_ttc = ttc
        if should:
            self._did_jump = True
            return Action.JUMP
        return Action.NOOP

    def _act_pixels(self, frame: np.ndarray) -> Action:
        blobs = detect_obstacle_blobs(frame)
        now = time.perf_counter()
        if not blobs:
            self._prev_blob_x = None
            self._prev_blob_t = None
            # One opening tap; do not spam-jump or the buffer lands us on the next cactus.
            if not self._started:
                self._started = True
                self._did_jump = True
                return Action.JUMP
            return Action.NOOP
        self._started = True
        x = min(blobs)
        if self._prev_blob_x is not None and self._prev_blob_t is not None:
            dt = now - self._prev_blob_t
            dx = self._prev_blob_x - x
            if 0.02 < dt < 0.30 and 1.0 < dx < 220:
                inst = dx / dt
                self._est_speed = 0.55 * self._est_speed + 0.45 * inst
        self._prev_blob_x = x
        self._prev_blob_t = now
        w = frame.shape[1]
        scale = w / CANVAS_W
        player_w = PLAYER_DISPLAY_W * scale
        player_right = PLAYER_X_FRAC * w + player_w / 2.0
        dist = x - player_right
        speed = max(self._est_speed * scale, 80.0)
        ttc = dist / speed
        lead = self._lead(speed / max(scale, 1e-6), PLAYER_DISPLAY_W, PLAYER_DISPLAY_W)
        if ttc > lead + 0.08:
            self._did_jump = False
        should = (not self._did_jump) and (-0.08 < ttc <= lead)
        if should:
            self._did_jump = True
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
