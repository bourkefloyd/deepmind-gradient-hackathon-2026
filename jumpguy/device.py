"""Torch device helper. Unlike nano/device.py this package allows CPU by default.

Set JUMPGUY_DEVICE=cuda|mps|cpu or pass --device. CUDA/MPS are used when present
and prefer=auto.
"""

from __future__ import annotations

import os
import sys
import time


def torch_device(prefer: str = "auto"):
    import torch

    if prefer == "auto":
        env = os.environ.get("JUMPGUY_DEVICE", "").strip()
        if env:
            prefer = env
    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(prefer)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if device.type == "mps" and not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
        raise RuntimeError("MPS requested but not available")
    return device


def smoke_test(prefer: str = "auto") -> dict:
    import torch

    device = torch_device(prefer)
    t0 = time.time()
    x = torch.randn(512, 512, device=device)
    y = x @ x
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    return {
        "device": str(device),
        "torch": torch.__version__,
        "matmul_ms": round((time.time() - t0) * 1000, 1),
        "checksum": float(y.sum()),
    }


if __name__ == "__main__":
    prefer = sys.argv[1] if len(sys.argv) > 1 else "auto"
    info = smoke_test(prefer)
    print(info)
    print(f"OK: {info['device']}")
