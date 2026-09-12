"""PyTorch device selection. Ported from actionfleet `lab/device.py`; `make check-mps` runs this file.

GUARD: on Apple Silicon, `prefer="auto"` REFUSES to fall back to CPU. MPS always exists on these machines, so
"auto resolved to cpu" means a degraded environment (a sandboxed shell that blocks Metal) - a silent ~16x
slowdown that once burned a whole night of runs. Pass `--device cpu` or set `SWM_ALLOW_CPU=1` for an
intentional CPU run.
"""

from __future__ import annotations

import os
import platform
import sys
import time


def _on_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def torch_device(prefer: str = "auto"):
    import torch

    if prefer == "auto":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _on_apple_silicon() and os.environ.get("SWM_ALLOW_CPU", "") != "1":
            raise RuntimeError(
                "Refusing to run on CPU: this is an Apple Silicon machine but MPS is unavailable "
                f"(mps_built={torch.backends.mps.is_built()}, mps_available={torch.backends.mps.is_available()}). "
                "You are likely in a sandboxed/degraded shell that blocks Metal - relaunch from a normal shell. "
                "To intentionally run on CPU, pass --device cpu or set SWM_ALLOW_CPU=1."
            )
        return torch.device("cpu")
    device = torch.device(prefer)
    if device.type == "mps" and not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
        raise RuntimeError("MPS requested but not available on this machine")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available on this machine")
    return device


def smoke_test(prefer: str = "auto") -> dict:
    """A real tensor op on the selected device, so a blocked Metal shell fails here, not an hour into training."""
    import torch

    device = torch_device(prefer)
    t0 = time.time()
    x = torch.randn(1024, 1024, device=device)
    y = x @ x
    if device.type == "mps":
        torch.mps.synchronize()
    return {"device": str(device), "torch": torch.__version__, "matmul_ms": round((time.time() - t0) * 1000, 1), "checksum": float(y.sum())}


if __name__ == "__main__":
    prefer = sys.argv[1] if len(sys.argv) > 1 else "auto"
    info = smoke_test(prefer)
    print(info)
    if info["device"] == "cpu" and _on_apple_silicon() and prefer == "auto":
        print("FAIL: training would land on CPU", file=sys.stderr)
        raise SystemExit(2)
    print(f"OK: {info['device']}")
