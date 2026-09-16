from __future__ import annotations

import unittest

import numpy as np

from jumpguy.actions import Action
from jumpguy.constants import CANVAS_H, CANVAS_W, OBSTACLE_COLORS
from jumpguy.policy import HeuristicPolicy, detect_obstacle_blobs
from jumpguy.sim import GameState, Obstacle


class HeuristicTests(unittest.TestCase):
    def test_jumps_when_obstacle_close(self):
        policy = HeuristicPolicy(lead_s=0.2)
        state = GameState(
            score=0,
            player_x=211.2,
            player_y=388.0,
            player_vy=0.0,
            grounded=True,
            speed=260.0,
            status="running",
            obstacles=[Obstacle(x=211.2 + 36 + 30, w=30, h=30)],
            t=1.0,
            ticks=60,
        )
        self.assertEqual(policy.act(state=state), Action.JUMP)

    def test_noop_when_far(self):
        policy = HeuristicPolicy(lead_s=0.2)
        state = GameState(
            score=0,
            player_x=211.2,
            player_y=388.0,
            player_vy=0.0,
            grounded=True,
            speed=260.0,
            status="running",
            obstacles=[Obstacle(x=800, w=30, h=30)],
            t=1.0,
            ticks=60,
        )
        self.assertEqual(policy.act(state=state), Action.NOOP)

    def test_start_and_restart(self):
        policy = HeuristicPolicy()
        ready = GameState(0, 211.2, 388, 0, True, 260, "ready", [], 0, 0)
        dead = GameState(3, 211.2, 388, 0, True, 260, "gameover", [], 1, 60)
        self.assertEqual(policy.act(state=ready), Action.JUMP)
        self.assertEqual(policy.act(state=dead), Action.JUMP)

    def test_jumps_once_per_approach(self):
        policy = HeuristicPolicy(lead_s=0.2)
        close = GameState(
            score=0,
            player_x=211.2,
            player_y=388.0,
            player_vy=0.0,
            grounded=True,
            speed=260.0,
            status="running",
            obstacles=[Obstacle(x=211.2 + 36 + 30, w=30, h=30)],
            t=1.0,
            ticks=60,
        )
        self.assertEqual(policy.act(state=close), Action.JUMP)
        self.assertEqual(policy.act(state=close), Action.NOOP)

    def test_pixel_path_jumps_once(self):
        from jumpguy.constants import CANVAS_H, CANVAS_W, OBSTACLE_COLORS

        policy = HeuristicPolicy(lead_s=0.25, latency_s=0.0)
        policy._started = True
        frame = np.full((CANVAS_H, CANVAS_W, 3), 255, dtype=np.uint8)
        gy = int(CANVAS_H * 0.72)
        # Cactus just inside the ~0.17–0.22 s takeoff window at 260 px/s.
        x0 = int(CANVAS_W * 0.22 + 36 + 32)
        frame[gy - 30 : gy, x0 : x0 + 30] = OBSTACLE_COLORS[0]
        self.assertEqual(policy.act(frame=frame), Action.JUMP)
        self.assertEqual(policy.act(frame=frame), Action.NOOP)


class BlobTests(unittest.TestCase):
    def test_detects_colored_square(self):
        frame = np.full((CANVAS_H, CANVAS_W, 3), 255, dtype=np.uint8)
        gy = int(CANVAS_H * 0.72)
        x0, x1 = 500, 530
        frame[gy - 30 : gy, x0:x1] = OBSTACLE_COLORS[0]
        blobs = detect_obstacle_blobs(frame)
        self.assertTrue(blobs)
        self.assertGreater(blobs[0], 400)


if __name__ == "__main__":
    unittest.main()
