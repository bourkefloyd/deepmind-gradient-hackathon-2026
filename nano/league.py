"""AI-only league table (PLAN.md section 6) on the gemma_seat seed-0 board set.

  python -m nano.league --out docs/league.md

Boards: the same 20 boards `gemma_seat.eval --n 20 --seed 0` used (regenerated via `gemma_seat.boards.packed_board`
with `random.Random(0)` and checked against the boards recorded in `gemma_seat/results/eval_n20_seed0.json`).
Nano and random rows are played here (750 actions = 75 s at the 10 Hz hand); Gemma rows are read from the recorded
evals (one call per board) - the mlx server is not touched.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any

import numpy as np

from .model import NanoAgent
from .policy import RandomSwiper, StudentPolicy
from .rollout import play, summarize
from .seat import bench
from .solver import Solver
from .device import torch_device

HAND_HZ = 10.0
MATCH_S = 75.0
ABORT_TICKS = 3  # the hand gives up on a word that is not on the board after a few taps


def league_boards(n: int = 20, seed: int = 0) -> list[str]:
    from gemma_seat.boards import load_dictionary, packed_board

    d = load_dictionary()
    rng = random.Random(seed)
    boards = [packed_board(rng, d, min_words=40)[0] for _ in range(n)]
    rec = "gemma_seat/results/eval_n20_seed0.json"
    if os.path.exists(rec):
        recorded = list(dict.fromkeys(r["board"] for r in json.load(open(rec))["rows"]))
        if recorded != boards:
            print(f"warning: regenerated boards differ from {rec}; using the recorded ones")
            boards = recorded
    return boards


def gemma_row(name: str, s: dict[str, Any], params: str, note: str) -> dict[str, Any]:
    """Words/min at hand cadence: one call (latency) yields `words_per_call` words; the hand traces the on-board ones
    at (len+1)/HAND_HZ s each and aborts the not-on-board ones after ABORT_TICKS; repeat calls assumed to add words."""
    calls = max(s.get("calls", 1), 1)
    not_on_board = s.get("invalid_not_on_board", 0) / calls
    traceable = s["words_per_call"] - not_on_board
    hand_s = traceable * (s["mean_word_len"] + 1) / HAND_HZ + not_on_board * ABORT_TICKS / HAND_HZ
    cycle = s["latency_mean_s"] + hand_s
    wpm = s["valid_words_per_call"] / cycle * 60 if cycle > 0 else 0.0
    return {
        "seat": name,
        "score": s["mean_score"],
        "valid_rate": s["valid_word_rate"],
        "mean_len": s["mean_word_len"],
        "wpm": wpm,
        "cost": "$0 (local mlx, Mac power)",
        "latency": f"{s['latency_mean_s']:.1f} s / call (~{s['words_per_call']:.0f} words)",
        "params": params,
        "note": note,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/league.md")
    ap.add_argument("--ticks", type=int, default=int(MATCH_S * HAND_HZ))
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra", action="append", default=[], metavar="NAME=PATH[=NOTE]", help="extra nano checkpoints to play as rows")
    a = ap.parse_args(argv)

    boards = league_boards()
    solver = Solver()
    device = torch_device("cpu")
    rows: list[dict[str, Any]] = []

    def played(name: str, pol, params: str, latency: str, note: str = "") -> None:
        rs = [play(b, pol, solver, a.ticks) for b in boards]
        S = summarize(rs)
        rows.append({"seat": name, "score": S["score_mean"], "valid_rate": S["valid_rate"], "mean_len": S["mean_len"],
                     "wpm": S["found_mean"] * 60 / MATCH_S, "cost": "$0", "latency": latency, "params": params, "note": note, "longest": S["longest"]})

    played("random swiper", RandomSwiper(seed=a.seed), "0", "0 ms", "floor")
    d4 = "runs/smoke_d4/model.pt"
    if os.path.exists(d4):
        m, _ = NanoAgent.load(d4)
        sp = bench(d4, n=200)["ms_per_action"]
        played("nano d4 smoke (300 steps, 2k boards)", StudentPolicy(m, device, a.temperature, a.seed), f"{m.n_params() / 1e6:.1f}M", f"{sp:.1f} ms / action", "smoke checkpoint")
    d6 = "nano/checkpoints/d6_s0.pt"
    m6, _ = NanoAgent.load(d6)
    sp6 = bench(d6, n=200)["ms_per_action"]
    played("nano d6_s0 (Mac MPS, 17k steps x 256, 200k boards, 30 min)", StudentPolicy(m6, device, a.temperature, a.seed), f"{m6.n_params() / 1e6:.1f}M", f"{sp6:.1f} ms / action", "first hero; kept as fallback")
    dl = "nano/checkpoints/d6_lambda.pt"
    ml, _ = NanoAgent.load(dl)
    spl = bench(dl, n=200)["ms_per_action"]
    played("**nano d6_lambda** (Lambda A100, 16.3k steps x 1024, 1M boards, len-bonus 1.6, 33 min)", StudentPolicy(ml, device, a.temperature, a.seed), f"{ml.n_params() / 1e6:.1f}M", f"{spl:.1f} ms / action", "the hero seat (default)")

    for spec in a.extra:
        name, path, *note = spec.split("=")
        me, _ = NanoAgent.load(path)
        played(name, StudentPolicy(me, device, a.temperature, a.seed), f"{me.n_params() / 1e6:.1f}M", f"{bench(path, n=200)['ms_per_action']:.1f} ms / action", note[0] if note else "")

    g = json.load(open("gemma_seat/results/eval_n20_seed0.json"))["summary"]
    rows.append(gemma_row("Gemma 4 12B, text grid", g["text"], "12B (4-bit)", "one call per board, T=0.2, thinking off"))
    rows.append(gemma_row("Gemma 4 12B, screenshot", g["image"], "12B (4-bit)", "one call per board, T=0.2, thinking off"))
    extra = []
    p = "gemma_seat/results/eval_n20_seed0_index.json"
    if os.path.exists(p):
        extra.append(gemma_row("Gemma 4 12B, tile-index prompt", json.load(open(p))["summary"]["index"], "12B (4-bit)", "index prompt: 15 s / call, ~0 words"))
    p = "gemma_seat/results/eval_n20_seed0_text_think.json"
    if os.path.exists(p):
        extra.append(gemma_row("Gemma 4 12B, text + thinking on", json.load(open(p))["summary"]["text+think"], "12B (4-bit)", "20/20 calls hit the 30 s timeout"))

    def fmt(r: dict[str, Any]) -> str:
        return (f"| {r['seat']} | {r['score']:.0f} | {r['valid_rate']:.2f} | {r['mean_len']:.2f} | {r['wpm']:.1f} | {r['cost']} | {r['latency']} | {r['params']} | {r['note']} |")

    md = [
        "# AI-only league (PLAN.md section 6)",
        "",
        f"Same 20 boards as `gemma_seat.eval --n 20 --seed 0` (`gemma_seat/boards.py`, `packed_board`, seed 0). Nano and random seats play {a.ticks} actions "
        f"= {MATCH_S:.0f} s at the {HAND_HZ:.0f} Hz hand (temperature {a.temperature}, CPU); Gemma numbers are the recorded one-call-per-board evals in "
        "`gemma_seat/results/` (not re-run). Words/min for Gemma assumes the same 10 Hz hand: one call's words traced at (len+1) ticks each, not-on-board "
        f"words aborted after {ABORT_TICKS} ticks, and every call adding new words (optimistic: repeat calls overlap). $/match: everything ran on this Mac.",
        "",
        "| seat | score / board | valid-word rate | mean word len | words / min (hand) | $ / match | latency | params | note |",
        "|---|---:|---:|---:|---:|---|---|---|---|",
        *[fmt(r) for r in rows],
        "| human (room, tonight) | | | | | $0 | reaction ~0.3 s | ~86B neurons | fill in from the room |",
    ]
    if extra:
        md += ["", "Gemma variants tried and parked (same boards):", "", "| seat | score / board | valid-word rate | mean word len | words / min (hand) | $ / match | latency | params | note |", "|---|---:|---:|---:|---:|---|---|---|---|", *[fmt(r) for r in extra]]
    nano = [r for r in rows if "d6_lambda" in r["seat"]][0]
    gt = rows[[i for i, r in enumerate(rows) if r["seat"].startswith("Gemma 4 12B, text")][0]]
    rnd = rows[0]
    md += [
        "",
        "## Reading it",
        "",
        f"1. The 11M nano scores {nano['score'] / gt['score']:.1f}x the 12B on the same boards with 1000x fewer parameters, deciding in {nano['latency']} where the 12B needs {gt['latency'].split()[0]} s per call; it is the only seat that plays at hand cadence without a word queue.",
        f"2. Valid-word rate is the honest split: nano {nano['valid_rate']:.2f}, Gemma 12B {gt['valid_rate']:.2f} (text) - the 12B hallucinates words that are not on the board ({g['text']['invalid_not_on_board'] / 20:.0f} of {g['text']['words_per_call']:.0f} per call); the nano only ever traces adjacent tiles.",
        f"3. Gemma finds longer words when it is right (mean length {gt['mean_len']:.2f} vs {nano['mean_len']:.2f}); the nano's score is volume of 3-4 letter words ({nano['wpm']:.0f} words/min), longest today `{nano.get('longest', '')}`.",
        f"4. Random swiper at {rnd['score']:.0f} is the floor; the nano is {nano['score'] / max(rnd['score'], 1):.1f}x it here (gate: >= 4x on 50 unseen boards passed at 14.3x, `nano/results/gate_d6_lambda.md`; the Mac-trained d6_s0 passed at 9.6x). Lambda A100 training (1M boards, 5x the samples, len-bonus 1.6 curriculum) lifted the same 11M architecture from 11245 to the hero row above at identical CPU cost; a 25.8M depth-8 twin scored about the same at 2x the ms/action (`nano/README.md`, GPU training).",
        "5. Thinking-on and the tile-index prompt are parked for the match: thinking timed out on 20/20 boards at 30 s, the index prompt returned ~0 words at 15 s per call. Thinking is an inference-budget knob, not a match setting.",
        "",
        "![league](league.png)",
    ]
    text = "\n".join(md) + "\n"
    print(text)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write(text)
    with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
        json.dump({"boards": boards, "rows": rows, "extra": extra}, f, indent=1)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [r["seat"].replace("**", "").split(" (")[0] for r in rows]
        fig_h = 3.6 + 0.35 * max(0, len(rows) - 5)
        fig, ax = plt.subplots(1, 2, figsize=(10, fig_h))
        colors = ["#999999"] + ["#1f77b4"] * (len(rows) - 3) + ["#ff7f0e", "#ff7f0e"]
        for i, r in enumerate(rows):
            if "d6_lambda" in r["seat"]:
                colors[i] = "#2ca02c"
        ax[0].barh(names, [r["score"] for r in rows], color=colors)
        ax[0].set_title("score / board (20 boards, 75 s)")
        ax[0].invert_yaxis()
        ax[1].barh(names, [r["valid_rate"] for r in rows], color=colors)
        ax[1].set_title("valid-word rate")
        ax[1].set_xlim(0, 1)
        ax[1].invert_yaxis()
        ax[1].set_yticklabels([])
        fig.tight_layout()
        fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=130)
    except Exception as e:
        print("no plot:", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
