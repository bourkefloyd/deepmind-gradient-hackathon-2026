"""Play the live Jump Guy site in a local realtime control loop.

  python -m jumpguy.play --policy heuristic --episodes 3 --max-seconds 90
  python -m jumpguy.play --policy cnn --ckpt jumpguy/runs/bc_smoke/model.pt

Default player name for any score UI: Grok Bot Son
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .actions import Action
from .constants import GAME_URL, PLAYER_NAME
from .env import JumpGuyLive
from .policy import CnnPolicy, HeuristicPolicy


def play_live(
    policy_name: str,
    episodes: int = 1,
    ckpt: str = "",
    device: str = "auto",
    headless: bool = True,
    url: str = GAME_URL,
    max_seconds: float = 90.0,
    hz: float = 60.0,
    submit: bool = False,
    player_name: str = PLAYER_NAME,
    grab_every: int = 0,
) -> dict[str, Any]:
    if policy_name in ("cnn", "model") and ckpt:
        brain: Any = CnnPolicy(ckpt, device=device)
        kind = "cnn"
        if grab_every <= 0:
            grab_every = 1
        # CNN needs frames; keep a slower loop.
        hz = min(hz, 20.0)
    else:
        # Hooked tryJump is ~1 frame; CV/screenshot path needs more lead.
        brain = HeuristicPolicy(lead_s=0.22, latency_s=0.02)
        kind = "heuristic"

    env = JumpGuyLive(url=url, headless=headless, player_name=player_name, grab_every=grab_every)
    dt = 1.0 / max(hz, 1.0)
    latencies: list[float] = []
    episode_rows: list[dict[str, Any]] = []
    try:
        env.start()
        for ep in range(episodes):
            if hasattr(brain, "reset"):
                brain.reset()
            step = env.reset()
            step = env.step(Action.JUMP)
            t_end = time.perf_counter() + max_seconds
            n = 0
            jumps = 0
            while time.perf_counter() < t_end:
                loop_t0 = time.perf_counter()
                if kind == "cnn":
                    a = brain.act(step.stack)
                elif step.state.hooked:
                    a = brain.act(state=step.state)
                else:
                    a = brain.act(frame=step.frame)
                if int(a) == Action.JUMP:
                    jumps += 1
                step = env.step(a)
                n += 1
                latencies.append(float(step.info.get("latency_ms", 0.0)))
                if step.done:
                    break
                remain = dt - (time.perf_counter() - loop_t0)
                if remain > 0:
                    time.sleep(remain)
            overlay = env.handle_game_over_ui(submit=submit)
            episode_rows.append(
                {
                    "episode": ep,
                    "score": int(step.info.get("score", 0) or 0),
                    "server_score": int(step.info.get("server_score", 0) or 0),
                    "status": step.info.get("status"),
                    "actions": n,
                    "jumps": jumps,
                    "hooked": bool(step.info.get("hooked") or env.hooked),
                    "run_id": step.info.get("run_id"),
                    "overlay": overlay,
                }
            )
    finally:
        env.close()

    arr = np.asarray(latencies, dtype=np.float32) if latencies else np.zeros(1, dtype=np.float32)
    wall = sum(max(e["actions"], 1) for e in episode_rows)
    result = {
        "policy": kind,
        "url": url,
        "player_name": player_name,
        "episodes": episode_rows,
        "mean_score": float(np.mean([e["score"] for e in episode_rows])) if episode_rows else 0.0,
        "max_score": int(max((e["score"] for e in episode_rows), default=0)),
        "actions": int(sum(e["actions"] for e in episode_rows)),
        "actions_per_sec": None,
        "latency_ms": {
            "mean": float(arr.mean()),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(arr.max()),
        },
        "headless": headless,
        "hooked": any(e.get("hooked") for e in episode_rows),
    }
    # Prefer wall-clock Hz from measured step latency, not the sleep target.
    mean_lat_s = float(arr.mean()) / 1000.0 if latencies else dt
    result["actions_per_sec"] = round(1.0 / max(mean_lat_s, 1e-6), 1)
    del wall
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Play live Jump Guy in a realtime loop")
    p.add_argument("--policy", default="heuristic", choices=["heuristic", "cnn", "auto"])
    p.add_argument("--ckpt", default="")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--max-seconds", type=float, default=90.0)
    p.add_argument("--hz", type=float, default=60.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--url", default=GAME_URL)
    p.add_argument("--headed", action="store_true")
    p.add_argument("--submit", action="store_true", help="click SUBMIT if the overlay allows it")
    p.add_argument("--player-name", default=PLAYER_NAME)
    p.add_argument("--grab-every", type=int, default=0, help="screenshot every N steps (0=hooked heuristic skips grabs)")
    p.add_argument("--out", default="")
    args = p.parse_args(argv)
    result = play_live(
        args.policy,
        episodes=args.episodes,
        ckpt=args.ckpt,
        device=args.device,
        headless=not args.headed,
        url=args.url,
        max_seconds=args.max_seconds,
        hz=args.hz,
        submit=args.submit,
        player_name=args.player_name,
        grab_every=args.grab_every,
    )
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
