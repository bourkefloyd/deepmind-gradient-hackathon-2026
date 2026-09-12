"""Supervised training on Word Hunt soft targets. Same recipe as actionfleet nanoagent `train.py`:
soft-CE (type + pointer) + BCE(value), fp32, no AMP, AdamW(0.9, 0.95) wd 0.1, clip 1.0, cosine LR to 10%.

  python -m nano.train --data data/wh_2k.npz --steps 300 --depth 4 --out runs/smoke_d4
  python -m nano.train --data data/wh_200k.npz --steps 20000 --depth 6 --budget-min 30 --out runs/d6_s0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Any, Optional

import numpy as np
import torch

from .data import load
from .device import torch_device
from .model import ModelConfig, NanoAgent, compute_loss, encode


def pre_encode(d: dict[str, np.ndarray], chunk: int = 200_000) -> dict[str, np.ndarray]:
    """Tokenize the whole dataset once (small dtypes) so a training step is pure device indexing."""
    n = len(d["value"])
    out: dict[str, list[np.ndarray]] = {"ids": [], "attn": [], "letter": [], "tmask": []}
    for i in range(0, n, chunk):
        enc = encode(d["board"][i : i + chunk], d["path"][i : i + chunk], d["plen"][i : i + chunk])
        out["ids"].append(enc["ids"].astype(np.int16))
        out["attn"].append(enc["attn"])
        out["letter"].append(enc["letter"].astype(np.int8))
        out["tmask"].append(enc["tmask"])
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["target"] = d["target"]
    res["ptype"] = d["ptype"]
    res["value"] = d["value"]
    return res


class DeviceData:
    def __init__(self, enc: dict[str, np.ndarray], device: torch.device):
        from .model import _SEGS

        self.n = len(enc["value"])
        self.device = device
        self.ids = torch.as_tensor(enc["ids"], device=device)
        self.attn = torch.as_tensor(enc["attn"], device=device)
        self.letter = torch.as_tensor(enc["letter"], device=device)
        self.tmask = torch.as_tensor(enc["tmask"], device=device)
        self.target = torch.as_tensor(enc["target"], device=device)
        self.ptype = torch.as_tensor(enc["ptype"], device=device)
        self.value = torch.as_tensor(enc["value"], device=device)
        self.segs = torch.as_tensor(_SEGS, device=device)

    def batch(self, idx: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        B = idx.shape[0]
        batch = {
            "ids": self.ids[idx].long(),
            "segs": self.segs.unsqueeze(0).expand(B, -1),
            "attn": self.attn[idx],
            "letter": self.letter[idx].long(),
            "tmask": self.tmask[idx],
        }
        labels = {
            "type_dist": self.ptype[idx].float(),
            "target_dist": self.target[idx].float(),
            "value": self.value[idx].float(),
        }
        return batch, labels


@torch.no_grad()
def evaluate(model: NanoAgent, data: DeviceData, bs: int = 2048, max_n: int = 50_000) -> dict[str, float]:
    model.eval()
    n = min(data.n, max_n)
    tot = {"loss": 0.0, "type": 0.0, "target": 0.0, "value": 0.0}
    ok_t = ok_g = n_g = ok_v = 0
    for i in range(0, n, bs):
        idx = torch.arange(i, min(i + bs, n), device=data.device)
        b, lab = data.batch(idx)
        out = model(b)
        L = compute_loss(out, lab)
        k = idx.shape[0]
        for key in tot:
            tot[key] += float(L[key]) * k
        ok_t += int((out["type_logits"].argmax(-1) == lab["type_dist"].argmax(-1)).sum())
        has_g = lab["target_dist"].sum(-1) > 0
        n_g += int(has_g.sum())
        ok_g += int(((out["target_logits"].argmax(-1) == lab["target_dist"].argmax(-1)) & has_g).sum())
        ok_v += int(((out["value_logit"] > 0).float() == lab["value"]).sum())
    model.train()
    return {**{k: v / n for k, v in tot.items()}, "type_top1": ok_t / n, "target_top1": ok_g / max(n_g, 1), "value_acc": ok_v / n, "n": n}


def train(
    tr: DeviceData,
    va: Optional[DeviceData],
    depth: int,
    steps: int,
    batch_size: int = 256,
    lr: Optional[float] = None,
    budget_min: float = 0.0,
    seed: int = 0,
    log=print,
    val_every: int = 500,
    out_dir: Optional[str] = None,
    init: Optional[str] = None,
    amp: bool = False,
    compile: bool = False,
) -> tuple[NanoAgent, dict[str, Any]]:
    device = tr.device
    torch.manual_seed(seed)
    g = torch.Generator(device="cpu").manual_seed(seed)
    if init:
        model, _ = NanoAgent.load(init)
        model.to(device)
    else:
        model = NanoAgent(ModelConfig(depth=depth)).to(device)
    lr = lr or 3e-4 * (4 / depth) ** 0.5  # nanochat-style: smaller models take larger lr
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    # bf16 autocast only where it is a free win (CUDA), as in the nanoagent recipe; CPU/MPS stay fp32
    use_amp = bool(amp) and device.type == "cuda"
    fwd = torch.compile(model) if compile else model  # `model` keeps the plain state_dict for saving
    log(f"model depth {model.cfg.depth} d_model {model.cfg.d_model} params {model.n_params() / 1e6:.2f}M lr {lr:.2e} batch {batch_size} steps {steps} budget {budget_min} min amp {use_amp} compile {compile}")
    t0 = time.time()
    losses: list[float] = []
    curve: list[dict[str, Any]] = []
    val_curve: list[dict[str, Any]] = []
    nonfinite = 0
    budget_sec = budget_min * 60.0
    step = 0
    model.train()
    while step < steps:
        elapsed = time.time() - t0
        if budget_sec > 0 and elapsed > budget_sec:
            log(f"budget {budget_min} min reached at step {step}")
            break
        frac = step / steps
        if budget_sec > 0:
            frac = max(frac, elapsed / budget_sec)
        cur_lr = lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0))))
        for grp in opt.param_groups:
            grp["lr"] = cur_lr
        idx = torch.randint(0, tr.n, (batch_size,), generator=g).to(device)
        batch, labels = tr.batch(idx)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = fwd(batch)
            L = compute_loss(out, labels)
        if not torch.isfinite(L["loss"]):
            nonfinite += 1
            step += 1
            continue
        opt.zero_grad(set_to_none=True)
        L["loss"].backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gn):
            nonfinite += 1
            step += 1
            continue
        opt.step()
        losses.append(L["loss"].item())
        step += 1
        if step % 50 == 0 or step == 1:
            el = time.time() - t0
            rec = {"step": step, "loss": float(np.mean(losses[-50:])), "type": float(L["type"]), "target": float(L["target"]), "value": float(L["value"]), "lr": cur_lr, "sec": round(el, 1)}
            curve.append(rec)
            log(f"step {step} loss {rec['loss']:.4f} type {rec['type']:.3f} target {rec['target']:.3f} value {rec['value']:.3f} lr {cur_lr:.2e} {step * batch_size / max(el, 1e-9):.0f} samples/s")
        if va is not None and val_every and step % val_every == 0:
            ev = evaluate(model, va)
            val_curve.append({"step": step, **ev})
            log(f"  val@{step}: loss {ev['loss']:.4f} type_top1 {ev['type_top1']:.3f} target_top1 {ev['target_top1']:.3f} value_acc {ev['value_acc']:.3f} (n={ev['n']})")
            if out_dir:
                model.save(os.path.join(out_dir, "model.pt"), extra={"step": step, "val": ev})
    wall = time.time() - t0
    info: dict[str, Any] = {
        "depth": model.cfg.depth,
        "n_params": model.n_params(),
        "steps": step,
        "samples_seen": step * batch_size,
        "wall_sec": wall,
        "samples_per_sec": step * batch_size / max(wall, 1e-9),
        "first_loss": float(np.mean(losses[:50])) if losses else None,
        "final_loss": float(np.mean(losses[-50:])) if losses else None,
        "nonfinite_steps": nonfinite,
        "device": str(device),
        "amp": use_amp,
        "compile": bool(compile),
        "lr": lr,
        "batch_size": batch_size,
        "n_train": tr.n,
        "curve": curve,
    }
    if va is not None:
        info["val"] = evaluate(model, va)
        info["val_curve"] = val_curve
    return model, info


def split(enc: dict[str, np.ndarray], bid: np.ndarray, val_frac: float) -> tuple[dict, dict]:
    """Hold out the last `val_frac` of boards (samples of a board are contiguous, so no board leaks)."""
    n_boards = int(bid.max()) + 1
    cut = int(n_boards * (1 - val_frac))
    m = bid < cut
    return {k: v[m] for k, v in enc.items()}, {k: v[~m] for k, v in enc.items()}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help=".npz from nano.data")
    ap.add_argument("--out", default=None, help="checkpoint dir (default runs/<data-stem>_d<depth>)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--budget-min", type=float, default=0.0, help="stop (and finish the cosine) after this many minutes")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast (CUDA only; ignored elsewhere)")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (CUDA; ~20 s warm-up)")
    a = ap.parse_args(argv)

    device = torch_device(a.device)
    print(f"Device: {device} (requested {a.device!r})")
    if device.type == "cuda":
        # fp32 sgemm is the bottleneck on Ampere (19.5 TFLOPS); TF32 keeps fp32 storage/accumulation with 10-bit
        # mantissa inputs (8x on the matmuls). Still no bf16 autocast, so CPU/MPS/CUDA runs stay comparable.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("cuda: TF32 matmuls enabled")
    out_dir = a.out or os.path.join("runs", f"{os.path.splitext(os.path.basename(a.data))[0]}_d{a.depth}")
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "train.log"), "a")

    def log(msg: str) -> None:
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"Device: {device}")
    t0 = time.time()
    d = load(a.data)
    enc = pre_encode(d)
    tr_np, va_np = split(enc, d["bid"], a.val_frac) if a.val_frac > 0 else (enc, None)
    tr = DeviceData(tr_np, device)
    va = DeviceData(va_np, device) if va_np is not None and len(va_np["value"]) else None
    log(f"data {a.data}: {len(d['value'])} samples, train {tr.n} val {va.n if va else 0} (loaded+encoded in {time.time() - t0:.1f}s)")
    model, info = train(tr, va, a.depth, a.steps, a.batch_size, a.lr, a.budget_min, a.seed, log, a.val_every, out_dir, a.init, a.amp, a.compile)
    model.save(os.path.join(out_dir, "model.pt"), extra={"train": {k: v for k, v in info.items() if k != "curve"}, "args": vars(a)})
    with open(os.path.join(out_dir, "train.json"), "w") as f:
        json.dump({**info, "args": vars(a)}, f, indent=1)
    log(json.dumps({k: v for k, v in info.items() if k not in ("curve", "val_curve")}, indent=1))
    log(f"saved {out_dir}/model.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
