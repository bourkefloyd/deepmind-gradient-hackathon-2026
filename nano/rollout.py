"""Offline rollout: a policy plays N unseen boards under a tick budget (one action per tick, like the hand
controller), scored against the solver. Reports words found, valid-submit rate, score, vs. the random swiper.

  python -m nano.rollout --model runs/smoke_d4/model.pt --boards 20 --ticks 600 --temperature 1.0
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np

from .data import letter_distribution, random_board
from .device import torch_device
from .model import NanoAgent
from .policy import Policy, RandomSwiper, StudentPolicy
from .solver import Solver, path_word, score_word


def play(board: str, policy: Policy, solver: Solver, ticks: int, max_repeat: int = 3) -> dict[str, Any]:
    policy.reset()
    words_on = solver.words_on(board)
    found: set[str] = set()
    path: tuple[int, ...] = ()
    submits = valid = aborts = 0
    repeats: dict[str, int] = {}
    for _ in range(ticks):
        (atype, tile), _info = policy.act(board, path)
        if atype == "extend" and tile >= 0:
            path = path + (tile,)
        elif atype == "submit":
            submits += 1
            w = path_word(board, path)
            if w in words_on:
                valid += 1
                found.add(w)
                repeats[w] = repeats.get(w, 0) + 1
            path = ()
        else:
            aborts += 1
            path = ()
    score = sum(score_word(w) for w in found)
    return {
        "board": board,
        "n_words_on_board": len(words_on),
        "found": sorted(found, key=lambda w: (-len(w), w)),
        "n_found": len(found),
        "submits": submits,
        "valid_submits": valid,
        "valid_rate": valid / max(submits, 1),
        "aborts": aborts,
        "score": score,
        "max_score": sum(score_word(w) for w in words_on),
        "mean_len": float(np.mean([len(w) for w in found])) if found else 0.0,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "boards": len(rows),
        "score_mean": float(np.mean([r["score"] for r in rows])),
        "found_mean": float(np.mean([r["n_found"] for r in rows])),
        "valid_rate": float(sum(r["valid_submits"] for r in rows) / max(sum(r["submits"] for r in rows), 1)),
        "submits_mean": float(np.mean([r["submits"] for r in rows])),
        "aborts_mean": float(np.mean([r["aborts"] for r in rows])),
        "mean_len": float(np.mean([r["mean_len"] for r in rows if r["n_found"]] or [0.0])),
        "longest": max((w for r in rows for w in r["found"]), key=len, default=""),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--boards", type=int, default=20)
    ap.add_argument("--ticks", type=int, default=600, help="actions per board (75 s at ~8 Hz)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=10_000, help="board seed; keep away from the training seeds")
    ap.add_argument("--device", default="cpu", help="inference is CPU in the game server; mps also fine")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    solver = Solver()
    rng = np.random.default_rng(a.seed)
    letter_p = letter_distribution(solver.words)
    boards = []
    while len(boards) < a.boards:
        b = random_board(rng, letter_p)
        if len(solver.words_on(b)) >= 15:
            boards.append(b)
    device = torch_device(a.device)
    model, _ = NanoAgent.load(a.model)
    student = StudentPolicy(model, device, temperature=a.temperature, seed=a.seed)
    rnd = RandomSwiper(seed=a.seed)
    res: dict[str, Any] = {}
    for name, pol in (("student", student), ("random", rnd)):
        t0 = time.time()
        rows = [play(b, pol, solver, a.ticks) for b in boards]
        s = summarize(rows)
        s["sec_per_board"] = (time.time() - t0) / len(boards)
        res[name] = {"summary": s, "rows": rows}
        print(f"{name:8s} score {s['score_mean']:7.0f}  words {s['found_mean']:5.1f}  valid {s['valid_rate']:.3f}  "
              f"submits {s['submits_mean']:5.1f}  aborts {s['aborts_mean']:5.1f}  mean_len {s['mean_len']:.2f}  longest {s['longest']!r}  {s['sec_per_board']:.1f}s/board")
    ratio = res["student"]["summary"]["score_mean"] / max(res["random"]["summary"]["score_mean"], 1e-9)
    print(f"student/random score ratio {ratio:.2f}x (gate: >= 4x)")
    ex = res["student"]["rows"][0]
    print(f"example board {ex['board']}: found {ex['found'][:12]} ({ex['n_found']}/{ex['n_words_on_board']})")
    res["ratio"] = ratio
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
