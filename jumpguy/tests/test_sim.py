"""Local sim matches the extracted Phaser numbers and is playable."""

from __future__ import annotations

import unittest

from jumpguy.actions import Action
from jumpguy.constants import AIR_TIME_S, MAX_JUMP_HEIGHT
from jumpguy.policy import HeuristicPolicy
from jumpguy.sim import JumpGuySim, theoretical_jump


class SimPhysicsTests(unittest.TestCase):
    def test_closed_form_jump(self):
        nums = theoretical_jump()
        self.assertAlmostEqual(nums["air_time_s"], AIR_TIME_S, places=5)
        self.assertGreater(nums["max_height_px"], 100)
        self.assertLess(nums["max_height_px"], 160)

    def test_jump_clears_obstacle_height(self):
        sim = JumpGuySim(seed=0, render=False, max_ticks=200)
        sim.reset()
        sim._queue_jump()
        peak = 0.0
        for _ in range(80):
            sim.step(Action.NOOP)
            peak = max(peak, sim.ground_y - sim.player_y)
            if sim.grounded and sim.ticks > 5:
                break
        self.assertGreater(peak, 30)  # obstacles are 28-34 px
        self.assertLess(abs(peak - MAX_JUMP_HEIGHT), 25)

    def test_first_jump_starts_run(self):
        sim = JumpGuySim(seed=1, render=False)
        step = sim.reset()
        self.assertEqual(step.state.status, "ready")
        step = sim.step(Action.JUMP)
        self.assertEqual(step.state.status, "running")

    def test_heuristic_scores(self):
        policy = HeuristicPolicy()
        sim = JumpGuySim(seed=7, render=False, max_ticks=4000)
        scores = []
        for ep in range(8):
            policy.reset()
            step = sim.reset(seed=7 + ep)
            while not step.done:
                step = sim.step(policy.act(state=step.state))
            scores.append(step.state.score)
        self.assertGreater(max(scores), 0)
        self.assertGreater(sum(scores) / len(scores), 1.0)

    def test_collision_ends_episode(self):
        sim = JumpGuySim(seed=0, render=False, max_ticks=500)
        step = sim.reset()
        sim.step(Action.JUMP)  # start
        # Plant an obstacle on the player.
        from jumpguy.sim import Obstacle

        sim.obstacles = [Obstacle(x=sim.player_x, w=32, h=32, passed=False)]
        sim.status = "running"
        step = sim.step(Action.NOOP)
        self.assertTrue(step.done)
        self.assertEqual(step.state.status, "gameover")
        self.assertLess(step.reward, 0)

    def test_pass_increments_score(self):
        sim = JumpGuySim(seed=0, render=False)
        sim.reset()
        sim.status = "running"
        from jumpguy.sim import Obstacle

        # Obstacle already left of the player.
        sim.obstacles = [Obstacle(x=20, w=30, h=30, passed=False)]
        step = sim.step(Action.NOOP)
        self.assertGreaterEqual(step.state.score, 1)
        self.assertGreaterEqual(step.reward, 1.0)


class RenderTests(unittest.TestCase):
    def test_frame_shape(self):
        sim = JumpGuySim(seed=0, render="small")
        step = sim.reset()
        self.assertEqual(step.frame.shape, (84, 84, 3))
        self.assertEqual(step.stack.shape, (4, 84, 84))

    def test_full_frame_shape(self):
        sim = JumpGuySim(seed=0, render="full")
        step = sim.reset()
        self.assertEqual(step.frame.shape[0], 540)
        self.assertEqual(step.frame.shape[1], 960)


if __name__ == "__main__":
    unittest.main()
