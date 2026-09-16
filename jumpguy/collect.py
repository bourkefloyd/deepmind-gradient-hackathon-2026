"""Collect teacher rollouts from the local sim (fast) or the live site."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .constants import FRAME_SIZE, FRAME_STACK
from .policy import HeuristicPolicy
from .sim import JumpGuySim


def collect_sim(
    episodes: int,
    out: Path,
    seed: int = 0,
    max_ticks: int = 20_000,
    render: bool = True,
) -> dict:
    policy = HeuristicPolicy()
    sim = JumpGuySim(seed=seed, render=render, max_ticks=max_ticks)
    frames: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    dones: list[np.bool_] = []
    scores: list[int] = []
    t0 = time.perf_counter()
    for ep in range(episodes):
        policy.reset()
        step = sim.reset(seed=seed + ep)
        ep_ret = 0.0
        while True:
            a = policy.act(state=step.state)
            frames.append(step.stack.astype(np.float32))
            actions.append(int(a))
            step = sim.step(a)
            rewards.append(float(step.reward))
            dones.append(np.bool_(step.done))
            ep_ret += step.reward
            if step.done:
                scores.append(int(step.state.score))
                break
    elapsed = time.perf_counter() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        frames=np.stack(frames).astype(np.float16),
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.bool_),
        scores=np.asarray(scores, dtype=np.int32),
    )
    meta = {
        "source": "sim",
        "episodes": episodes,
        "transitions": len(actions),
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "max_score": int(np.max(scores)) if scores else 0,
        "seconds": round(elapsed, 2),
        "steps_per_sec": round(len(actions) / max(elapsed, 1e-6), 1),
        "out": str(out),
        "frame_stack": FRAME_STACK,
        "frame_size": FRAME_SIZE,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return meta


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Collect Jump Guy teacher rollouts")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--out", default="jumpguy/data/heuristic.npz")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-ticks", type=int, default=12_000)
    p.add_argument("--no-render", action="store_true", help="state-only (cannot train a CNN)")
    args = p.parse_args(argv)
    meta = collect_sim(
        episodes=args.episodes,
        out=Path(args.out),
        seed=args.seed,
        max_ticks=args.max_ticks,
        render=not args.no_render,
    )
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
