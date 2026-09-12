"""Tiny pre-norm transformer with pointer, action-type and value heads, for Word Hunt.

Ported from actionfleet `sandbox/nanoagent/nanoagent/model.py` (span heads and image adapter dropped; the
tokenizer is now fixed-shape Word Hunt tokens instead of a11y trees).

One dial: `depth`. Width = 64 * depth, heads = depth (head dim 64), MLP = 4x.
depth 4 ~ 3M params, depth 6 ~ 10M.

Tokens (26, fixed layout; PLAN.md section 3):
  0        [CLS]
  1..16    tile tokens: letter embedding + grid-position embedding (segment TILE)
  17       path-length token (0..8)
  18..25   path tokens: tile-index embedding + that tile's letter + step-order position (segment PATH), padded

Heads read from [CLS]:
  target_logits (B, 16)  dot(query(cls), key(h[tile i])), illegal tiles (not adjacent to the path head, or
                         already used) masked to -inf
  type_logits   (B, 3)   {extend, submit, abort}
  value_logit   (B,)     P(this prefix still completes to a word)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .solver import MAX_LEN, NEIGHBORS

N_TILES = 16
N_TYPES = 3  # extend, submit, abort
SEQ_LEN = 1 + N_TILES + 1 + MAX_LEN  # 26

# token ids
TOK_PAD = 0
TOK_CLS = 1
TOK_LETTER = 2  # +0..25
TOK_TILE = TOK_LETTER + 26  # +0..15 (path tokens reference a tile index)
TOK_LEN = TOK_TILE + N_TILES  # +0..8
VOCAB_SIZE = TOK_LEN + MAX_LEN + 1

SEG_CLS, SEG_TILE, SEG_LEN, SEG_PATH = 0, 1, 2, 3
N_SEGMENTS = 4

_SEGS = np.array([SEG_CLS] + [SEG_TILE] * N_TILES + [SEG_LEN] + [SEG_PATH] * MAX_LEN, dtype=np.int64)
_NB_MASK = np.zeros((N_TILES, N_TILES), dtype=bool)
for _i, _js in enumerate(NEIGHBORS):
    _NB_MASK[_i, _js] = True


def encode(board: np.ndarray, path: np.ndarray, plen: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorised tokenizer. board (B,16) letter ids 0..25; path (B,8) tile ids, -1 pad; plen (B,).
    Returns ids, segs, attn (B,26); letter (B,26) letter id of the tile each path token points at (-1 if none);
    tmask (B,16) legal next tiles."""
    B = board.shape[0]
    ids = np.zeros((B, SEQ_LEN), np.int64)
    ids[:, 0] = TOK_CLS
    ids[:, 1 : 1 + N_TILES] = TOK_LETTER + board.astype(np.int64)
    ids[:, 1 + N_TILES] = TOK_LEN + plen.astype(np.int64)
    path = path.astype(np.int64)
    valid = path >= 0
    ids[:, 2 + N_TILES :] = np.where(valid, TOK_TILE + np.clip(path, 0, N_TILES - 1), TOK_PAD)
    attn = ids != TOK_PAD
    letter = np.full((B, SEQ_LEN), -1, np.int64)
    pl = np.take_along_axis(board.astype(np.int64), np.clip(path, 0, N_TILES - 1), axis=1)
    letter[:, 2 + N_TILES :] = np.where(valid, pl, -1)
    tmask = np.ones((B, N_TILES), dtype=bool)
    has = plen > 0
    if has.any():
        last = path[np.arange(B), np.clip(plen.astype(np.int64) - 1, 0, MAX_LEN - 1)]
        tmask[has] = _NB_MASK[last[has]]
        used = np.zeros((B, N_TILES), dtype=bool)
        rows = np.repeat(np.arange(B), MAX_LEN)
        used[rows[valid.ravel()], path.ravel()[valid.ravel()]] = True
        tmask &= ~used
    return {"ids": ids, "segs": np.broadcast_to(_SEGS, (B, SEQ_LEN)).copy(), "attn": attn, "letter": letter, "tmask": tmask}


def to_device(enc: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, device=device) for k, v in enc.items()}


@dataclass
class ModelConfig:
    depth: int = 6
    dropout: float = 0.0
    vocab_size: int = VOCAB_SIZE
    n_segments: int = N_SEGMENTS
    max_len: int = SEQ_LEN

    @property
    def d_model(self) -> int:
        return 64 * self.depth

    @property
    def n_heads(self) -> int:
        return self.depth


class Block(nn.Module):
    def __init__(self, d: int, h: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class NanoAgent(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.tok = nn.Embedding(cfg.vocab_size, d)
        self.seg = nn.Embedding(cfg.n_segments, d)
        self.pos = nn.Embedding(cfg.max_len, d)
        self.letter = nn.Embedding(26 + 1, d, padding_idx=26)  # letter of the tile a path token points at
        self.blocks = nn.ModuleList([Block(d, cfg.n_heads, cfg.dropout) for _ in range(cfg.depth)])
        self.ln_f = nn.LayerNorm(d)
        self.type_head = nn.Linear(d, N_TYPES)
        self.q_target = nn.Linear(d, d)
        self.k_target = nn.Linear(d, d)
        self.value_head = nn.Linear(d, 1)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ids, segs, attn = batch["ids"], batch["segs"], batch["attn"]
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0)
        x = self.tok(ids) + self.seg(segs) + self.pos(pos)
        letter = batch["letter"]
        x = x + self.letter(torch.where(letter >= 0, letter, torch.full_like(letter, 26)))
        kpm = ~attn
        for blk in self.blocks:
            x = blk(x, kpm)
        x = self.ln_f(x)
        cls = x[:, 0]
        d = x.shape[-1]
        ht = x[:, 1 : 1 + N_TILES]
        target_logits = (self.k_target(ht) * self.q_target(cls).unsqueeze(1)).sum(-1) / math.sqrt(d)
        target_logits = target_logits.masked_fill(~batch["tmask"], float("-inf"))
        return {
            "type_logits": self.type_head(cls),
            "target_logits": target_logits,
            "value_logit": self.value_head(cls).squeeze(-1),
            "hidden": x,
        }

    def save(self, path: str, extra: dict | None = None) -> None:
        torch.save({"cfg": asdict(self.cfg), "state": self.state_dict(), "extra": extra or {}}, path)

    @staticmethod
    def load(path: str, map_location="cpu") -> tuple["NanoAgent", dict]:
        ck = torch.load(path, map_location=map_location, weights_only=False)
        m = NanoAgent(ModelConfig(**ck["cfg"]))
        m.load_state_dict(ck["state"])
        return m, ck.get("extra", {})


def soft_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy against a soft distribution; rows whose target sums to 0 are skipped."""
    logp = F.log_softmax(logits, dim=-1)
    logp = torch.where(torch.isfinite(logp), logp, torch.zeros_like(logp))
    row_has = target.sum(-1) > 0
    loss = -(target * logp).sum(-1)
    if row_has.any():
        return loss[row_has].mean()
    return torch.zeros((), device=logits.device)


def compute_loss(out: dict[str, torch.Tensor], labels: dict[str, torch.Tensor], weights: dict[str, float] | None = None) -> dict[str, torch.Tensor]:
    """labels: type_dist (B,3), target_dist (B,16), value (B,) in {0,1}. Same recipe as nanoagent train.py:
    soft-CE on type and pointer, BCE on value (weight 0.5)."""
    w = {"type": 1.0, "target": 1.0, "value": 0.5, **(weights or {})}
    l_type = soft_ce(out["type_logits"], labels["type_dist"])
    l_target = soft_ce(out["target_logits"], labels["target_dist"])
    l_value = F.binary_cross_entropy_with_logits(out["value_logit"], labels["value"])
    total = w["type"] * l_type + w["target"] * l_target + w["value"] * l_value
    return {"loss": total, "type": l_type.detach(), "target": l_target.detach(), "value": l_value.detach()}
