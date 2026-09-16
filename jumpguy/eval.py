"""Evaluate a policy on the local sim (and optionally print live-play instructions)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .actions import Action
from .policy import CnnPolicy, HeuristicPolicy, RandomPolicy
from .sim import JumpGuySim


def eval_sim(
    policy_name: str,
    episodes: int = 20,
    seed: int = 0,
    ckpt: str = "",
    device: str = "auto",
    max_ticks: int = 2_700,
    render: bool = True,
) -> dict:
    if policy_name in ("heuristic", "teacher", "cv"):
        policy: object = HeuristicPolicy()
        kind = "heuristic"
    elif policy_name == "random":
        policy = RandomPolicy(seed=seed)
        kind = "random"
    elif policy_name in ("cnn", "model", "auto"):
        if not ckpt:
            raise SystemExit("cnn eval needs --ckpt")
        policy = CnnPolicy(ckpt, device=device)
        kind = "cnn"
    else:
        raise SystemExit(f"unknown policy {policy_name}")

    sim = JumpGuySim(seed=seed, render=render, max_ticks=max_ticks)
    scores: list[int] = []
    lengths: list[int] = []
    t0 = time.perf_counter()
    actions = 0
    for ep in range(episodes):
        if hasattr(policy, "reset"):
            policy.reset()
        step = sim.reset(seed=seed + ep)
        n = 0
        kicked = False
        while True:
            if kind == "cnn":
                a = policy.act(step.stack)  # type: ignore[attr-defined]
                if not kicked:
                    a = int(Action.JUMP)
                    kicked = True
            else:
                a = policy.act(state=step.state, frame=step.frame)  # type: ignore[attr-defined]
            step = sim.step(a)
            n += 1
            actions += 1
            if step.done:
                scores.append(int(step.state.score))
                lengths.append(n)
                break
    elapsed = time.perf_counter() - t0
    result = {
        "policy": kind,
        "episodes": episodes,
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "median_score": float(np.median(scores)) if scores else 0.0,
        "max_score": int(np.max(scores)) if scores else 0,
        "min_score": int(np.min(scores)) if scores else 0,
        "mean_ticks": float(np.mean(lengths)) if lengths else 0.0,
        "actions_per_sec": round(actions / max(elapsed, 1e-6), 1),
        "seconds": round(elapsed, 2),
        "scores": scores,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate a Jump Guy policy on the local sim")
    p.add_argument("--policy", default="heuristic", choices=["heuristic", "cnn", "random", "auto"])
    p.add_argument("--ckpt", default="")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-ticks", type=int, default=2700)
    p.add_argument("--out", default="")
    args = p.parse_args(argv)
    result = eval_sim(
        args.policy,
        episodes=args.episodes,
        seed=args.seed,
        ckpt=args.ckpt,
        device=args.device,
        max_ticks=args.max_ticks,
    )
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
