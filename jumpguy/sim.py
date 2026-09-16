"""Local Jump Guy physics that mirrors the live Phaser scene.

Collision and scoring use *display* AABBs (that is what `GameScene.update` does).
Jump / gravity use Arcade-style Euler integration at 60 Hz.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .actions import Action
from .constants import (
    AIR_TIME_S,
    CANVAS_H,
    CANVAS_W,
    COYOTE_MS,
    DT,
    GRAVITY_Y,
    GROUND_Y_FRAC,
    JUMP_BUFFER_MS,
    JUMP_VELOCITY,
    MAX_JUMP_HEIGHT,
    OBSTACLE_COLORS,
    OBSTACLE_SIZE_RANGE,
    PLAYER_DISPLAY_H,
    PLAYER_DISPLAY_W,
    PLAYER_HAT,
    PLAYER_PANTS,
    PLAYER_SHIRT,
    PLAYER_SHOES,
    PLAYER_SKIN,
    PLAYER_X_FRAC,
    SPAWN_COOLDOWN_BASE_MS,
    SPAWN_COOLDOWN_CLAMP_MS,
    SPAWN_COOLDOWN_JITTER,
    SPAWN_COOLDOWN_RESET_MS,
    SPAWN_X_JITTER,
    SPEED_ACCEL,
    SPEED_MAX,
    SPEED_START,
    TICK_HZ,
)
from .observe import FrameStack


@dataclass
class Obstacle:
    x: float
    w: float
    h: float
    passed: bool = False
    color: tuple[int, int, int] = OBSTACLE_COLORS[0]


@dataclass
class GameState:
    score: int
    player_x: float
    player_y: float
    player_vy: float
    grounded: bool
    speed: float
    status: str  # ready | running | gameover
    obstacles: list[Obstacle]
    t: float
    ticks: int


@dataclass
class StepResult:
    frame: np.ndarray
    stack: np.ndarray
    state: GameState
    reward: float
    done: bool
    info: dict


def _rects_overlap(ax: float, ay: float, aw: float, ah: float, bx: float, by: float, bw: float, bh: float) -> bool:
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by


class JumpGuySim:
    """Gym-like env with the live game's numbers. Fast enough for PPO on CPU."""

    def __init__(
        self,
        seed: int = 0,
        render: bool = True,
        max_ticks: int = int(3 * 60 * TICK_HZ),
        stack: bool = True,
        auto_restart: bool = False,
    ):
        self.rng = np.random.default_rng(seed)
        self.render_enabled = render
        self.max_ticks = max_ticks
        self.use_stack = stack
        self.auto_restart = auto_restart
        self.stacker = FrameStack() if stack else None
        self.ground_y = float(int(CANVAS_H * GROUND_Y_FRAC))
        self.player_x = CANVAS_W * PLAYER_X_FRAC
        self._frame = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        self.reset()

    def reset(self, seed: Optional[int] = None) -> StepResult:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.status = "ready"
        self.score = 0
        self.ticks = 0
        self.t = 0.0
        self.player_y = self.ground_y
        self.player_vy = 0.0
        self.grounded = True
        self.last_grounded_at = 0.0
        self.jump_queued_at = -1e9
        self.was_grounded = True
        self.speed = SPEED_START
        self.spawn_cooldown_s = SPAWN_COOLDOWN_RESET_MS / 1000.0
        self.obstacles: list[Obstacle] = []
        self.ground_pattern_offset = 0.0
        frame = self._render()
        stack = self.stacker.reset(frame) if self.stacker else frame
        return StepResult(frame, stack, self.state(), 0.0, False, {"event": "reset"})

    def state(self) -> GameState:
        return GameState(
            score=self.score,
            player_x=self.player_x,
            player_y=self.player_y,
            player_vy=self.player_vy,
            grounded=self.grounded,
            speed=self.speed,
            status=self.status,
            obstacles=[Obstacle(o.x, o.w, o.h, o.passed, o.color) for o in self.obstacles],
            t=self.t,
            ticks=self.ticks,
        )

    def _player_bounds(self) -> tuple[float, float, float, float]:
        return (
            self.player_x - PLAYER_DISPLAY_W / 2.0,
            self.player_y - PLAYER_DISPLAY_H,
            PLAYER_DISPLAY_W,
            PLAYER_DISPLAY_H,
        )

    def _can_jump(self, now_ms: float) -> bool:
        return now_ms - self.last_grounded_at <= COYOTE_MS

    def _queue_jump(self) -> None:
        if self.status != "gameover":
            self.jump_queued_at = self.t * 1000.0

    def _execute_jump(self) -> bool:
        now_ms = self.t * 1000.0
        if now_ms - self.jump_queued_at <= JUMP_BUFFER_MS and self._can_jump(now_ms):
            self.player_vy = JUMP_VELOCITY
            self.jump_queued_at = -1e9
            self.grounded = False
            return True
        return False

    def _spawn_obstacle(self) -> None:
        x = CANVAS_W + float(self.rng.integers(SPAWN_X_JITTER[0], SPAWN_X_JITTER[1] + 1))
        n = int(self.rng.integers(OBSTACLE_SIZE_RANGE[0], OBSTACLE_SIZE_RANGE[1] + 1))
        color = OBSTACLE_COLORS[int(self.rng.integers(0, len(OBSTACLE_COLORS)))]
        self.obstacles.append(Obstacle(x=x, w=float(n), h=float(n), passed=False, color=color))

    def _advance_obstacles(self, dt: float) -> None:
        if self.status != "running":
            return
        self.spawn_cooldown_s -= dt
        if self.spawn_cooldown_s <= 0:
            self._spawn_obstacle()
            jitter = int(self.rng.integers(SPAWN_COOLDOWN_JITTER[0], SPAWN_COOLDOWN_JITTER[1] + 1))
            raw = SPAWN_COOLDOWN_BASE_MS + jitter
            lo, hi = SPAWN_COOLDOWN_CLAMP_MS
            self.spawn_cooldown_s = float(np.clip(raw, lo, hi)) / 1000.0
        self.speed = min(self.speed + dt * SPEED_ACCEL, SPEED_MAX)
        keep: list[Obstacle] = []
        for o in self.obstacles:
            o.x -= self.speed * dt
            if o.x + o.w / 2.0 > -32:
                keep.append(o)
        self.obstacles = keep
        self.ground_pattern_offset += self.speed * dt

    def step(self, action: int | Action) -> StepResult:
        if self.status == "gameover":
            if int(action) == Action.JUMP or self.auto_restart:
                prev = self.reset()
                prev.info["event"] = "restart"
                return prev
            frame = self._render()
            stack = self.stacker.push(frame) if self.stacker else frame
            return StepResult(frame, stack, self.state(), 0.0, True, {"event": "dead"})

        if int(action) == Action.JUMP:
            if self.status == "ready":
                # First successful jump starts the run (mirrors startRunIfNeeded).
                self._queue_jump()
            else:
                self._queue_jump()

        jumped = self._execute_jump()
        if jumped and self.status == "ready":
            self.status = "running"

        self.player_vy += GRAVITY_Y * DT
        self.player_y += self.player_vy * DT
        if self.player_y >= self.ground_y:
            self.player_y = self.ground_y
            self.player_vy = 0.0
            self.grounded = True
        else:
            self.grounded = False
        if self.grounded:
            self.last_grounded_at = self.t * 1000.0

        self._advance_obstacles(DT)
        self.t += DT
        self.ticks += 1

        reward = 0.0
        done = False
        event = "tick"
        px, py, pw, ph = self._player_bounds()
        if self.status == "running":
            for o in self.obstacles:
                ox = o.x - o.w / 2.0
                oy = self.ground_y - o.h
                if _rects_overlap(px, py, pw, ph, ox, oy, o.w, o.h):
                    self.status = "gameover"
                    done = True
                    reward = -1.0
                    event = "dead"
                    break
                if not o.passed and (ox + o.w) < px:
                    o.passed = True
                    self.score += 1
                    reward += 1.0
                    event = "pass"
        if self.ticks >= self.max_ticks and not done:
            done = True
            event = "timeout"

        frame = self._render()
        stack = self.stacker.push(frame) if self.stacker else frame
        info = {
            "event": event,
            "jumped": jumped,
            "score": self.score,
            "speed": self.speed,
            "t": self.t,
        }
        return StepResult(frame, stack, self.state(), reward, done, info)

    def _render(self) -> np.ndarray:
        if not self.render_enabled:
            return self._frame
        img = self._frame
        img.fill(255)
        gy = int(self.ground_y)
        img[gy : gy + 2, :] = (58, 58, 58)
        # hash marks
        e = gy + 10
        t = int(self.ground_pattern_offset) % 22
        for n in range(-22, CANVAS_W + 22, 22):
            r = n - t
            if 0 <= r < CANVAS_W and 0 <= e < CANVAS_H:
                img[e, r : min(CANVAS_W, r + 4)] = (89, 89, 89)
        # obstacles
        for o in self.obstacles:
            x0 = int(round(o.x - o.w / 2.0))
            y0 = int(round(self.ground_y - o.h))
            x1 = int(round(o.x + o.w / 2.0))
            y1 = int(round(self.ground_y))
            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(CANVAS_W, x1)
            y1 = min(CANVAS_H, y1)
            if x1 > x0 and y1 > y0:
                img[y0:y1, x0:x1] = o.color
        # player (blocky stand-in matching the generated palette)
        px, py, pw, ph = self._player_bounds()
        x0, y0 = int(round(px)), int(round(py))
        x1, y1 = int(round(px + pw)), int(round(py + ph))
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(CANVAS_W, x1), min(CANVAS_H, y1)
        if x1 > x0 and y1 > y0:
            mid = y0 + (y1 - y0) // 3
            waist = y0 + 2 * (y1 - y0) // 3
            img[y0:mid, x0:x1] = PLAYER_HAT
            img[mid:waist, x0:x1] = PLAYER_SHIRT
            img[waist:y1, x0:x1] = PLAYER_PANTS
            # face sliver
            fx0 = x0 + (x1 - x0) // 4
            fx1 = x1 - (x1 - x0) // 4
            if fx1 > fx0:
                img[y0 + 4 : mid, fx0:fx1] = PLAYER_SKIN
            img[max(y0, y1 - 4) : y1, x0:x1] = PLAYER_SHOES
        self._frame = img
        return img.copy()


def theoretical_jump() -> dict[str, float]:
    """Sanity numbers from the closed-form jump (no collision)."""
    return {
        "air_time_s": AIR_TIME_S,
        "max_height_px": MAX_JUMP_HEIGHT,
        "tick_hz": TICK_HZ,
        "dt": DT,
    }
