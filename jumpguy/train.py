"""Train JumpNet: behavioral cloning from teacher rollouts, then optional PPO on the sim.

  python -m jumpguy.collect --episodes 40 --out jumpguy/data/heuristic.npz
  python -m jumpguy.train --bc jumpguy/data/heuristic.npz --steps 400 --out jumpguy/runs/bc_smoke
  python -m jumpguy.train --ppo --init jumpguy/runs/bc_smoke/model.pt --steps 2000 --out jumpguy/runs/ppo_smoke
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .actions import Action
from .device import torch_device
from .model import JumpNet, JumpNetConfig, load_checkpoint, save_checkpoint
from .sim import JumpGuySim


def expand_jump_labels(actions: np.ndarray, dones: np.ndarray, radius: int = 4) -> np.ndarray:
    """Mark a few frames *before* each teacher jump as JUMP (takeoff window)."""
    out = actions.copy()
    if radius <= 0:
        return out
    n = len(out)
    for i in range(n):
        if int(actions[i]) != 1:
            continue
        lo = i
        for k in range(1, radius + 1):
            j = i - k
            if j < 0 or bool(dones[j]):
                break
            lo = j
        out[lo : i + 1] = 1
    return out


def _load_bc(path: Path, jump_window: int = 4) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    frames = data["frames"].astype(np.float32)
    actions = data["actions"].astype(np.int64)
    dones = data["dones"] if "dones" in data.files else np.zeros(len(actions), dtype=np.bool_)
    if jump_window > 0:
        actions = expand_jump_labels(actions, np.asarray(dones), radius=jump_window)
    return frames, actions


def _sim_score(model, device, episodes: int = 3, max_ticks: int = 900, seed: int = 0) -> float:
    """Cheap rollout used as the BC checkpoint metric (val loss is noop-heavy)."""
    model.eval()
    scores: list[int] = []
    for ep in range(episodes):
        sim = JumpGuySim(seed=seed + ep, render=True, max_ticks=max_ticks)
        step = sim.reset(seed=seed + ep)
        cool = 0
        kicked = False
        while True:
            if not kicked:
                a = int(Action.JUMP)
                kicked = True
                cool = 28
            elif cool > 0:
                a = int(Action.NOOP)
                cool -= 1
            else:
                x = torch.as_tensor(step.stack, device=device).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = model(x)
                    p = float(torch.softmax(logits, dim=-1)[0, 1])
                if p >= 0.55:
                    a = int(Action.JUMP)
                    cool = 28
                else:
                    a = int(Action.NOOP)
            step = sim.step(a)
            if step.done:
                scores.append(int(step.state.score))
                break
    model.train()
    return float(np.mean(scores)) if scores else 0.0


def train_bc(
    data_path: Path,
    out: Path,
    steps: int = 800,
    batch_size: int = 64,
    lr: float = 3e-4,
    device_name: str = "auto",
    width: int = 32,
    val_every: int = 100,
    jump_window: int = 4,
    score_every: int = 200,
) -> dict[str, Any]:
    device = torch_device(device_name)
    frames, actions = _load_bc(data_path, jump_window=jump_window)
    n = len(actions)
    if n < 8:
        raise SystemExit(f"not enough transitions in {data_path}: {n}")
    rng = np.random.default_rng(0)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(8, n // 10)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    jump_idx = train_idx[actions[train_idx] == 1]
    noop_idx = train_idx[actions[train_idx] == 0]
    counts = np.bincount(actions, minlength=2).astype(np.float32)
    weights = 1.0 / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    model = JumpNet(JumpNetConfig(width=width)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    w = torch.as_tensor(weights, device=device)
    t0 = time.perf_counter()
    history: list[dict[str, float]] = []
    best_loss = 1e9
    best_score = -1.0
    for step in range(1, steps + 1):
        model.train()
        # Jumps are ~1-2% of ticks. Mild upsample (≈1:7) avoids NOOP collapse
        # without teaching a 50% jump prior that lands on the next cactus.
        n_jump = max(1, batch_size // 8)
        if len(jump_idx) and len(noop_idx):
            b_j = rng.choice(jump_idx, size=min(n_jump, len(jump_idx)), replace=len(jump_idx) < n_jump)
            b_n = rng.choice(noop_idx, size=min(batch_size - len(b_j), len(noop_idx)), replace=False)
            b = np.concatenate([b_j, b_n])
            rng.shuffle(b)
        else:
            b = rng.choice(train_idx, size=min(batch_size, len(train_idx)), replace=False)
        x = torch.as_tensor(frames[b], device=device)
        y = torch.as_tensor(actions[b], device=device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits, y, weight=w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % val_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                vx = torch.as_tensor(frames[val_idx], device=device)
                vy = torch.as_tensor(actions[val_idx], device=device)
                vlogits, _ = model(vx)
                vloss = float(F.cross_entropy(vlogits, vy))
                acc = float((vlogits.argmax(-1) == vy).float().mean())
            rec = {"step": step, "train_loss": float(loss.detach()), "val_loss": vloss, "val_acc": acc}
            history.append(rec)
            print(f"bc step {step}/{steps} loss={loss:.4f} val_loss={vloss:.4f} acc={acc:.3f}", flush=True)
            rec["sim_score"] = -1.0
            if score_every > 0 and (step % score_every == 0 or step == steps):
                rec["sim_score"] = _sim_score(model, device)
                print(f"bc sim_score={rec['sim_score']:.2f}", flush=True)
            if rec["sim_score"] > best_score or (rec["sim_score"] < 0 and vloss < best_loss):
                if rec["sim_score"] > best_score:
                    best_score = rec["sim_score"]
                if vloss < best_loss:
                    best_loss = vloss
                save_checkpoint(
                    out / "model.pt",
                    model,
                    extra={
                        "kind": "bc",
                        "step": step,
                        "val_acc": acc,
                        "val_loss": vloss,
                        "sim_score": rec["sim_score"],
                        "n": n,
                    },
                )
            elif vloss < best_loss:
                best_loss = vloss
    meta = {
        "kind": "bc",
        "steps": steps,
        "n": n,
        "jump_frac": float((actions == 1).mean()),
        "class_weight": weights.tolist(),
        "device": str(device),
        "params": model.n_params(),
        "seconds": round(time.perf_counter() - t0, 2),
        "history": history,
        "best_val_loss": best_loss,
        "best_sim_score": best_score,
        "jump_window": jump_window,
        "data": str(data_path),
    }
    (out / "train.json").write_text(json.dumps(meta, indent=2))
    return meta


def _gae(rewards, values, dones, gamma: float, lam: float):
    adv = np.zeros_like(rewards, dtype=np.float32)
    last = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        next_v = 0.0 if dones[t] else (values[t + 1] if t + 1 < len(values) else 0.0)
        delta = rewards[t] + gamma * next_v - values[t]
        last = delta + gamma * lam * (0.0 if dones[t] else last)
        adv[t] = last
    ret = adv + values[: len(adv)]
    return adv, ret


def train_ppo(
    out: Path,
    steps: int = 8_000,
    horizon: int = 256,
    epochs: int = 3,
    minibatch: int = 64,
    lr: float = 2.5e-4,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip: float = 0.2,
    entropy_coef: float = 0.02,
    value_coef: float = 0.5,
    device_name: str = "auto",
    width: int = 32,
    init: Optional[Path] = None,
    seed: int = 0,
    render: bool = True,
    eval_every: int = 2_000,
) -> dict[str, Any]:
    device = torch_device(device_name)
    if init and Path(init).exists():
        model, _ = load_checkpoint(init, device)
        model.train()
    else:
        model = JumpNet(JumpNetConfig(width=width)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    sim = JumpGuySim(seed=seed, render=render, max_ticks=8_000)
    step = sim.reset(seed=seed)
    t0 = time.perf_counter()
    env_steps = 0
    history: list[dict[str, float]] = []
    ep_scores: list[int] = []
    while env_steps < steps:
        mb_obs, mb_act, mb_logp, mb_rew, mb_done, mb_val = [], [], [], [], [], []
        for _ in range(horizon):
            obs = torch.as_tensor(step.stack, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, value = model(obs)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)
            a = int(action.item())
            nxt = sim.step(a)
            mb_obs.append(step.stack)
            mb_act.append(a)
            mb_logp.append(float(logp.item()))
            mb_rew.append(float(nxt.reward))
            mb_done.append(bool(nxt.done))
            mb_val.append(float(value.item()))
            step = nxt
            env_steps += 1
            if nxt.done:
                ep_scores.append(int(nxt.state.score))
                step = sim.reset()
            if env_steps >= steps:
                break
        values = np.asarray(mb_val, dtype=np.float32)
        rewards = np.asarray(mb_rew, dtype=np.float32)
        dones = np.asarray(mb_done, dtype=np.bool_)
        adv, ret = _gae(rewards, values, dones, gamma, lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        obs_t = torch.as_tensor(np.stack(mb_obs), device=device)
        act_t = torch.as_tensor(np.asarray(mb_act), device=device, dtype=torch.int64)
        old_logp = torch.as_tensor(np.asarray(mb_logp), device=device)
        adv_t = torch.as_tensor(adv, device=device)
        ret_t = torch.as_tensor(ret, device=device)
        n = len(mb_act)
        idx = np.arange(n)
        last_loss = 0.0
        for _ in range(epochs):
            np.random.shuffle(idx)
            for s in range(0, n, minibatch):
                b = idx[s : s + minibatch]
                if len(b) < 8:
                    continue
                logits, value = model(obs_t[b])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(act_t[b])
                ratio = torch.exp(logp - old_logp[b])
                unclipped = ratio * adv_t[b]
                clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t[b]
                pg = -torch.min(unclipped, clipped).mean()
                vloss = F.mse_loss(value, ret_t[b])
                ent = dist.entropy().mean()
                loss = pg + value_coef * vloss - entropy_coef * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                last_loss = float(loss.detach())
        rec = {
            "env_steps": env_steps,
            "loss": last_loss,
            "mean_score": float(np.mean(ep_scores[-20:])) if ep_scores else 0.0,
            "episodes": len(ep_scores),
        }
        history.append(rec)
        print(
            f"ppo steps={env_steps}/{steps} loss={last_loss:.4f} "
            f"mean_score={rec['mean_score']:.2f} episodes={len(ep_scores)}",
            flush=True,
        )
        if env_steps % eval_every < horizon or env_steps >= steps:
            save_checkpoint(
                out / "model.pt",
                model,
                extra={"kind": "ppo", "env_steps": env_steps, "mean_score": rec["mean_score"]},
            )
    meta = {
        "kind": "ppo",
        "steps": env_steps,
        "device": str(device),
        "params": model.n_params(),
        "seconds": round(time.perf_counter() - t0, 2),
        "mean_score": float(np.mean(ep_scores)) if ep_scores else 0.0,
        "max_score": int(np.max(ep_scores)) if ep_scores else 0,
        "episodes": len(ep_scores),
        "history": history[-50:],
        "init": str(init) if init else None,
    }
    (out / "train.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train the Jump Guy vision policy")
    p.add_argument("--bc", default="", help="path to collect .npz for behavioral cloning")
    p.add_argument("--ppo", action="store_true", help="PPO on the local sim")
    p.add_argument("--init", default="", help="checkpoint to warm-start PPO")
    p.add_argument("--out", default="jumpguy/runs/latest")
    p.add_argument("--steps", type=int, default=0, help="BC optimizer steps or PPO env steps")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--jump-window", type=int, default=4, help="BC: extra frames before each teacher jump")
    p.add_argument("--entropy", type=float, default=0.02, help="PPO entropy coefficient")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.bc:
        steps = args.steps or 800
        meta = train_bc(
            Path(args.bc),
            out,
            steps=steps,
            batch_size=args.batch_size,
            device_name=args.device,
            width=args.width,
            jump_window=args.jump_window,
        )
        print(json.dumps({k: v for k, v in meta.items() if k != "history"}, indent=2))
        return 0
    if args.ppo:
        steps = args.steps or 8_000
        meta = train_ppo(
            out,
            steps=steps,
            device_name=args.device,
            width=args.width,
            init=Path(args.init) if args.init else None,
            seed=args.seed,
            render=not args.no_render,
            entropy_coef=args.entropy,
        )
        print(json.dumps({k: v for k, v in meta.items() if k != "history"}, indent=2))
        return 0
    raise SystemExit("pass --bc <npz> and/or --ppo")


if __name__ == "__main__":
    raise SystemExit(main())
