"""Pre-registered nano gate (PLAN.md section 3) on N unseen boards, written as markdown to nano/results/.

  python -m nano.gate --model nano/checkpoints/d6_s0.pt --boards 50 --out nano/results/gate_d6_s0.md

Gate: score >= 4x the random swiper; valid-submit rate > Gemma 4 E4B (proxy today: the 12B mlx-vlm text seat,
20% valid-word rate on n=20 boards, `gemma_seat/results/eval_n20_seed0.md`). Also reports mean word length and
words per board.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .data import letter_distribution, random_board
from .device import torch_device
from .model import NanoAgent
from .policy import RandomSwiper, StudentPolicy
from .rollout import play, summarize
from .seat import bench
from .solver import Solver

GEMMA_PROXY = {"name": "Gemma 4 12B (mlx-vlm 4-bit, text grid, n=20)", "valid_rate": 0.20, "mean_len": 3.79, "score": 1780, "source": "gemma_seat/results/eval_n20_seed0.md"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--boards", type=int, default=50)
    ap.add_argument("--ticks", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20_000)
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
    model, extra = NanoAgent.load(a.model)
    student = StudentPolicy(model, torch_device("cpu"), temperature=a.temperature, seed=a.seed)
    rnd = RandomSwiper(seed=a.seed)
    t0 = time.time()
    s_rows = [play(b, student, solver, a.ticks) for b in boards]
    s_sec = time.time() - t0
    r_rows = [play(b, rnd, solver, a.ticks) for b in boards]
    S, R = summarize(s_rows), summarize(r_rows)
    ratio = S["score_mean"] / max(R["score_mean"], 1e-9)
    speed = bench(a.model, n=300, threads=1)
    pass_ratio = ratio >= 4.0
    pass_valid = S["valid_rate"] > GEMMA_PROXY["valid_rate"]
    verdict = "PASS" if (pass_ratio and pass_valid) else "FAIL"
    train = extra.get("train", {}) if isinstance(extra, dict) else {}
    lens = [len(w) for r in s_rows for w in r["found"]]
    hist = {n: lens.count(n) for n in range(3, 9)}
    longest = sorted({w for r in s_rows for w in r["found"]}, key=lambda w: (-len(w), w))[:10]

    md = [
        f"# Nano gate: `{a.model}` ({verdict})",
        "",
        f"{a.boards} unseen boards (seed {a.seed}, >= 15 words each), {a.ticks} actions per board (75 s at ~8 Hz), temperature {a.temperature}, CPU inference. "
        f"Pre-registered in PLAN.md section 3: score >= 4x random swiper AND valid-submit rate above the Gemma E4B seat (proxy: {GEMMA_PROXY['name']}, `{GEMMA_PROXY['source']}`).",
        "",
        "| metric | nano | random swiper | Gemma proxy | gate |",
        "|---|---:|---:|---:|---|",
        f"| score / board | **{S['score_mean']:.0f}** | {R['score_mean']:.0f} | {GEMMA_PROXY['score']} | {ratio:.2f}x random (>= 4x) {'PASS' if pass_ratio else 'FAIL'} |",
        f"| valid-submit rate | **{S['valid_rate']:.3f}** | {R['valid_rate']:.3f} | {GEMMA_PROXY['valid_rate']:.2f} | > {GEMMA_PROXY['valid_rate']:.2f} {'PASS' if pass_valid else 'FAIL'} |",
        f"| words found / board | {S['found_mean']:.1f} | {R['found_mean']:.1f} | 5.1 (per call) | |",
        f"| mean word length | {S['mean_len']:.2f} | {R['mean_len']:.2f} | {GEMMA_PROXY['mean_len']} | |",
        f"| submits / board | {S['submits_mean']:.1f} | {R['submits_mean']:.1f} | | |",
        f"| aborts / board | {S['aborts_mean']:.1f} | {R['aborts_mean']:.1f} | | |",
        f"| longest word | `{S['longest']}` | `{R['longest']}` | | |",
        f"| ms / action (CPU, 1 thread) | {speed['ms_per_action']} | | ~2000 per call | |",
        f"| params | {speed['params_M']}M | 0 | 12B | |",
        "",
        f"Word-length histogram (nano, all boards): {hist}. Longest: {', '.join(longest)}.",
        "",
        f"Training: depth {train.get('depth')}, {train.get('steps')} steps x batch {train.get('batch_size')} = {train.get('samples_seen')} samples on {train.get('n_train')} "
        f"({train.get('wall_sec', 0) / 60:.1f} min, {train.get('samples_per_sec', 0):.0f} samples/s, device {train.get('device')}); loss {train.get('first_loss', 0):.3f} -> {train.get('final_loss', 0):.3f}; "
        f"val {json.dumps({k: round(v, 3) for k, v in train.get('val', {}).items() if k != 'n'})}.",
        "",
        f"Rollout wall: {s_sec / a.boards:.1f} s/board for the nano. Stretch target (>= 60% of human median score) not measured: no human boards yet.",
        "",
        f"Example board `{s_rows[0]['board']}`: nano found {s_rows[0]['found'][:15]} ({s_rows[0]['n_found']}/{s_rows[0]['n_words_on_board']}); random found {r_rows[0]['found']}.",
    ]
    text = "\n".join(md) + "\n"
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as f:
            f.write(text)
        with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
            json.dump({"student": S, "random": R, "ratio": ratio, "verdict": verdict, "speed": speed, "boards": boards, "args": vars(a)}, f, indent=1)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
