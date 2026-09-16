"""Small vision policy for 4x84x84 stacks. Fast enough for realtime CPU inference."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .actions import N_ACTIONS
from .constants import FRAME_SIZE, FRAME_STACK
from .device import torch_device


@dataclass
class JumpNetConfig:
    in_ch: int = FRAME_STACK
    frame: int = FRAME_SIZE
    n_actions: int = N_ACTIONS
    width: int = 32


class JumpNet(nn.Module):
    """Nature-DQN-ish CNN: policy logits + state-value.

    84x84 stack: conv 8s4 -> 20, conv 4s2 -> 9, conv 3s1 -> 7. ~400k params at width=32.
    """

    def __init__(self, cfg: Optional[JumpNetConfig] = None):
        super().__init__()
        self.cfg = cfg or JumpNetConfig()
        c = self.cfg.width
        self.conv1 = nn.Conv2d(self.cfg.in_ch, c, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(c, c * 2, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(c * 2, c * 2, kernel_size=3, stride=1)
        n_flat = (c * 2) * 7 * 7
        self.fc = nn.Linear(n_flat, 256)
        self.policy = nn.Linear(256, self.cfg.n_actions)
        self.value = nn.Linear(256, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dtype != torch.float32:
            x = x.float()
        z = F.relu(self.conv1(x))
        z = F.relu(self.conv2(z))
        z = F.relu(self.conv3(z))
        z = z.flatten(1)
        z = F.relu(self.fc(z))
        return self.policy(z), self.value(z).squeeze(-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def save_checkpoint(path: str | Path, model: JumpNet, extra: Optional[dict[str, Any]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"cfg": asdict(model.cfg), "state_dict": model.state_dict(), **(extra or {})}
    torch.save(payload, path)
    meta = {k: v for k, v in payload.items() if k != "state_dict"}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))


def load_checkpoint(path: str | Path, device=None) -> tuple[JumpNet, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    device = device or torch_device("auto")
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = JumpNetConfig(**payload.get("cfg", {}))
    model = JumpNet(cfg)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload
