"""Live learning between rounds (PLAN.md section 3, "Live learning"): validated words from any seat -> soft-target
samples -> a few BC steps on CPU mixed 1:1 with replay -> fixed held-out check -> keep or roll back.

  learner = OnlineLearner("nano/checkpoints/d6_lambda.pt")
  res = learner.update(board, ["hunt", "sent", "lantern"])   # {kept, held_out_before, held_out_after, seconds, ...}
  learner.seat / learner.model  -> the current (kept) weights; learner.save(path)

Budget: one update must fit inside the 20 s rematch countdown on CPU (measured in `nano/results/live_learning_curve.md`).
Held-out = a fixed 20-board set played with a batched rollout under common random numbers (same seed before/after).

  python -m nano.learn --checkpoint nano/checkpoints/d6_s0.pt --rounds 10 --out nano/results/live_learning_curve.md
"""

from __future__ import annotations

import argparse
import copy
import os
import time
from typing import Any, Iterable, Optional

import numpy as np
import torch

from .data import generate, info_from_words, letter_distribution, load, load_common_ranks, pack, random_board, rows_from_info
from .model import NanoAgent, compute_loss, encode, to_device
from .policy import StudentPolicy
from .solver import Solver, path_word, score_word
from .train import DeviceData, pre_encode


def play_batch(boards: list[str], student: StudentPolicy, words_on: list[set[str]], ticks: int, seed: int) -> dict[str, Any]:
    """All boards advance one action per tick with a single batched forward; common random numbers via `seed`."""
    student.rng = np.random.default_rng(seed)
    n = len(boards)
    paths: list[tuple[int, ...]] = [() for _ in range(n)]
    found: list[set[str]] = [set() for _ in range(n)]
    submits = np.zeros(n, int)
    valid = np.zeros(n, int)
    for _ in range(ticks):
        ds = student.dists(boards, paths)
        for i in range(n):
            (atype, tile), _ = student.pick(ds[i], paths[i])
            if atype == "extend":
                paths[i] = paths[i] + (tile,)
            elif atype == "submit":
                submits[i] += 1
                w = path_word(boards[i], paths[i])
                if w in words_on[i]:
                    valid[i] += 1
                    found[i].add(w)
                paths[i] = ()
            else:
                paths[i] = ()
    scores = np.array([sum(score_word(w) for w in f) for f in found], float)
    return {"score": float(scores.mean()), "words": float(np.mean([len(f) for f in found])), "valid_rate": float(valid.sum() / max(submits.sum(), 1)), "found": found}


class OnlineLearner:
    def __init__(
        self,
        checkpoint: str,
        replay: Optional[str] = None,
        held_out_boards: int = 20,
        held_out_seed: int = 30_000,
        held_out_ticks: int = 300,
        steps: int = 50,
        batch_size: int = 64,
        lr: float = 2e-5,
        tolerance: float = 0.03,
        self_mix: float = 0.5,
        temperature: float = 1.0,
        threads: int = 4,
        seed: int = 0,
    ):
        torch.set_num_threads(threads)
        self.device = torch.device("cpu")
        self.model, self.extra = NanoAgent.load(checkpoint)
        self.model.to(self.device).train()
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
        self.steps, self.batch_size, self.tolerance, self.temperature = steps, batch_size, tolerance, temperature
        # hinted self-distillation: new-sample targets = self_mix * base model's own prediction + (1 - self_mix) * taught
        # posterior, so 15 words cannot rewrite the type/pointer prior the model learned from 200k boards
        self.self_mix = self_mix
        self.seed = seed
        self.solver = Solver()  # training side only: scores the held-out set and weights words; never used to act
        self.ranks = load_common_ranks(valid=self.solver.word_set)
        self.student = StudentPolicy(self.model, self.device, temperature=temperature, seed=seed)
        # replay: the base distribution, so a handful of stage words cannot overwrite 200k boards of prior
        if replay and os.path.exists(replay):
            d = load(replay)
            keep = np.random.default_rng(seed).choice(len(d["value"]), size=min(len(d["value"]), 20_000), replace=False)
            d = {k: v[keep] for k, v in d.items()}
        else:
            d = generate(400, n_paths=6, seed=seed + 7, workers=1)
        self.replay = DeviceData(pre_encode(d), self.device)
        rng = np.random.default_rng(held_out_seed)
        lp = letter_distribution(self.solver.words)
        self.held_out: list[str] = []
        while len(self.held_out) < held_out_boards:
            b = random_board(rng, lp)
            if len(self.solver.words_on(b)) >= 15:
                self.held_out.append(b)
        self.held_out_words = [self.solver.words_on(b) for b in self.held_out]
        self.held_out_ticks = held_out_ticks
        self.held_out_seed = held_out_seed
        self.buffer: list[tuple[str, set[str]]] = []  # (board, words) of every round so far; recent rounds are replayed
        self.history: list[dict[str, Any]] = []
        self._held_out_cache: Optional[float] = None

    @property
    def seat(self):
        from .seat import NanoSeat

        s = NanoSeat.__new__(NanoSeat)
        s.model, s.extra, s.policy = self.model, self.extra, StudentPolicy(self.model, self.device, self.temperature, self.seed)
        s.board, s.path, s.found, s.n_actions, s.total_ms, s.last_info = "", (), set(), 0, 0.0, {}
        return s

    def evaluate(self) -> dict[str, Any]:
        self.model.eval()
        r = play_batch(self.held_out, self.student, self.held_out_words, self.held_out_ticks, self.held_out_seed)
        self.model.train()
        return r

    def samples(self, board: str, words: Iterable[str]) -> Optional[dict[str, np.ndarray]]:
        words = [w.lower() for w in words if 3 <= len(w) <= 8]
        info = info_from_words(board, words, self.ranks)
        if not info:
            return None
        rows = rows_from_info(board, info, info.keys())
        return pack(rows, 0)

    def update(self, board: str, words: Iterable[str], steps: Optional[int] = None) -> dict[str, Any]:
        t0 = time.time()
        words = set(w.lower() for w in words)
        self.buffer.append((board, words))
        # new samples: this round's words plus the last few rounds (a word seen once is not learned in 80 steps)
        parts = [s for b, ws in self.buffer[-2:] if (s := self.samples(b, ws)) is not None]
        if not parts:
            return {"kept": False, "reason": "no samples", "seconds": time.time() - t0}
        new = DeviceData(pre_encode({k: np.concatenate([p[k] for p in parts]) for k in parts[0]}), self.device)
        if self.self_mix > 0:
            self.model.eval()
            with torch.no_grad():
                b, _ = new.batch(torch.arange(new.n))
                out = self.model(b)
                p_type = torch.softmax(out["type_logits"], -1)
                p_tgt = torch.nan_to_num(torch.softmax(out["target_logits"], -1))
            self.model.train()
            has_tgt = (new.target.float().sum(-1, keepdim=True) > 0).float()
            new.ptype = (self.self_mix * p_type + (1 - self.self_mix) * new.ptype.float()).to(new.ptype.dtype)
            new.target = (has_tgt * (self.self_mix * p_tgt + (1 - self.self_mix) * new.target.float())).to(new.target.dtype)
        before = self._held_out_cache if self._held_out_cache is not None else self.evaluate()["score"]
        t_eval0 = time.time() - t0
        snapshot = copy.deepcopy(self.model.state_dict())
        opt_snapshot = copy.deepcopy(self.opt.state_dict())
        g = torch.Generator().manual_seed(self.seed + len(self.history))
        half = self.batch_size // 2
        losses = []
        self.model.train()
        for _ in range(steps or self.steps):
            i_new = torch.randint(0, new.n, (half,), generator=g)
            i_rep = torch.randint(0, self.replay.n, (self.batch_size - half,), generator=g)
            b1, l1 = new.batch(i_new)
            b2, l2 = self.replay.batch(i_rep)
            batch = {k: torch.cat([b1[k], b2[k]]) for k in b1}
            labels = {k: torch.cat([l1[k], l2[k]]) for k in l1}
            L = compute_loss(self.model(batch), labels)
            self.opt.zero_grad(set_to_none=True)
            L["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            losses.append(L["loss"].item())
        t_train = time.time() - t0 - t_eval0
        after_r = self.evaluate()
        after = after_r["score"]
        kept = after >= before * (1 - self.tolerance)
        if not kept:
            self.model.load_state_dict(snapshot)
            self.opt.load_state_dict(opt_snapshot)
            self._held_out_cache = before
        else:
            self._held_out_cache = after
        res = {
            "kept": bool(kept),
            "held_out_before": before,
            "held_out_after": after,
            "held_out_words": after_r["words"],
            "held_out_valid_rate": after_r["valid_rate"],
            "n_words": len(words),
            "n_samples": int(new.n),
            "steps": int(steps or self.steps),
            "loss_first": float(np.mean(losses[:10])),
            "loss_last": float(np.mean(losses[-10:])),
            "seconds": time.time() - t0,
            "seconds_train": t_train,
            "seconds_eval": time.time() - t0 - t_train,
        }
        self.history.append(res)
        return res

    def save(self, path: str) -> None:
        self.model.save(path, extra={**self.extra, "live_updates": self.history})


def simulate(checkpoint: str, rounds: int, out: Optional[str], teacher_top: int = 15, seed: int = 40_000, replay: Optional[str] = None, teacher_min_len: int = 4) -> list[dict[str, Any]]:
    """Stand-in for Gemma: the teacher's words are the solver's top-`teacher_top` common words of length >= `teacher_min_len`
    on each round board (Gemma's contribution on stage is the longer words the nano misses)."""
    learner = OnlineLearner(checkpoint, replay=replay)
    rng = np.random.default_rng(seed)
    lp = letter_distribution(learner.solver.words)
    t0 = time.time()
    base = learner.evaluate()
    base_sec = time.time() - t0
    learner._held_out_cache = base["score"]
    rows = [{"round": 0, "kept": True, "held_out_after": base["score"], "held_out_words": base["words"], "held_out_valid_rate": base["valid_rate"], "seconds": base_sec, "taught": 0, "recall_before": None, "recall_after": None}]
    print(f"round 0 held-out {base['score']:.0f} ({base['words']:.1f} words, valid {base['valid_rate']:.3f}) eval {base_sec:.1f}s")
    for r in range(1, rounds + 1):
        while True:
            board = random_board(rng, lp)
            words_on = learner.solver.words_on(board)
            if len(words_on) >= 15:
                break
        taught = sorted((w for w in words_on if len(w) >= teacher_min_len), key=lambda w: learner.ranks.get(w, 10**6))[:teacher_top]
        # does the nano find the taught words on this board before and after (same seed = common random numbers)?
        learner.model.eval()
        pre = play_batch([board], learner.student, [words_on], learner.held_out_ticks, seed + r)["found"][0]
        res = learner.update(board, taught)
        learner.model.eval()
        post = play_batch([board], learner.student, [words_on], learner.held_out_ticks, seed + r)["found"][0]
        learner.model.train()
        rec_b, rec_a = len(pre & set(taught)) / len(taught), len(post & set(taught)) / len(taught)
        rows.append({"round": r, **res, "taught": len(taught), "recall_before": rec_b, "recall_after": rec_a, "board": board, "words": taught})
        print(f"round {r} {'KEEP' if res['kept'] else 'ROLLBACK'} held-out {res['held_out_before']:.0f} -> {res['held_out_after']:.0f}  "
              f"taught {len(taught)} recall {rec_b:.2f} -> {rec_a:.2f}  loss {res['loss_first']:.3f}->{res['loss_last']:.3f}  {res['seconds']:.1f}s (train {res['seconds_train']:.1f}s)")
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        kept = [x for x in rows[1:] if x["kept"]]
        secs = [x["seconds"] for x in rows[1:]]
        md = [
            f"# Live learning curve: `{checkpoint}`, {rounds} AI-only rounds",
            "",
            f"Teacher stand-in for Gemma: the solver's top-{teacher_top} common words of length >= {teacher_min_len} on each round board. Learner: {learner.steps} BC steps per round on CPU "
            f"({torch.get_num_threads()} threads), batch {learner.batch_size} mixed 1:1 with replay from the base data, lr {learner.opt.param_groups[0]['lr']:.0e}, "
            f"the last 2 rounds' words stay in the new-sample pool, new-sample targets = {learner.self_mix} x base prediction + {1 - learner.self_mix} x taught posterior (hinted self-distillation). Held-out: {len(learner.held_out)} fixed boards, {learner.held_out_ticks} actions each, batched rollout with "
            f"common random numbers; keep if held-out score >= before x (1 - {learner.tolerance}).",
            "",
            "| round | kept | held-out score | words/board | valid rate | taught | recall before -> after | update s (train s) |",
            "|---:|---|---:|---:|---:|---:|---|---:|",
        ]
        for x in rows:
            rec = "" if x["recall_before"] is None else f"{x['recall_before']:.2f} -> {x['recall_after']:.2f}"
            secs_col = f"{x['seconds']:.1f}" if x["round"] == 0 else f"{x['seconds']:.1f} ({x['seconds_train']:.1f})"
            md.append(f"| {x['round']} | {'yes' if x['kept'] else 'rollback'} | {x['held_out_after']:.0f} | {x['held_out_words']:.1f} | {x['held_out_valid_rate']:.3f} | {x['taught']} | {rec} | {secs_col} |")
        final = learner._held_out_cache
        md += [
            "",
            f"Held-out score {rows[0]['held_out_after']:.0f} -> {final:.0f} ({(final / rows[0]['held_out_after'] - 1) * 100:+.1f}%), {len(kept)}/{rounds} updates kept. "
            f"Update wall time mean {np.mean(secs):.1f} s, max {np.max(secs):.1f} s (budget: 20 s rematch countdown). "
            f"Recall of taught words on the round board: mean {np.mean([x['recall_before'] for x in rows[1:]]):.2f} -> {np.mean([x['recall_after'] for x in rows[1:]]):.2f}.",
        ]
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
            ax[0].plot([x["round"] for x in rows], [x["held_out_after"] for x in rows], marker="o")
            for x in rows[1:]:
                if not x["kept"]:
                    ax[0].plot(x["round"], x["held_out_after"], "rx", ms=9)
            ax[0].set_title("held-out score (20 boards); x = rolled back")
            ax[0].set_xlabel("round")
            ax[1].plot([x["round"] for x in rows[1:]], [x["recall_before"] for x in rows[1:]], marker="o", label="before update")
            ax[1].plot([x["round"] for x in rows[1:]], [x["recall_after"] for x in rows[1:]], marker="o", label="after update")
            ax[1].set_title("taught-word recall on the round board")
            ax[1].set_xlabel("round")
            ax[1].legend()
            fig.tight_layout()
            png = os.path.splitext(out)[0] + ".png"
            fig.savefig(png, dpi=120)
            md.append(f"\n![curve]({os.path.basename(png)})")
        except Exception as e:  # matplotlib optional
            md.append(f"\n(no plot: {e})")
        with open(out, "w") as f:
            f.write("\n".join(md) + "\n")
        print("wrote", out)
    return rows


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="nano/checkpoints/d6_lambda.pt")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--teacher-top", type=int, default=15)
    ap.add_argument("--teacher-min-len", type=int, default=4)
    ap.add_argument("--replay", default="data/wh_200000.npz", help="base data for replay (falls back to generating 400 boards)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    simulate(a.checkpoint, a.rounds, a.out, a.teacher_top, replay=a.replay, teacher_min_len=a.teacher_min_len)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
