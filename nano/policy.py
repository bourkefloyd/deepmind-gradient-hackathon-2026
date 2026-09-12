"""Acting policies for Word Hunt: StudentPolicy (NanoAgent, temperature sampling), RandomSwiper, and a stub
EscalatingPolicy. Ported from actionfleet `sandbox/nanoagent/nanoagent/policy.py`; the a11y/GUI parts are gone.

An action is (type, tile): type in {"extend", "submit", "abort"}; tile is the next tile index for `extend`, else -1.
The game/hand layer owns the board and the path; a policy only maps (board, path) -> action.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import torch

from .data import TYPES, legal_moves
from .model import NanoAgent, encode, to_device
from .solver import MAX_LEN, MIN_LEN

Action = tuple[str, int]


def board_to_ids(board: str) -> np.ndarray:
    return np.frombuffer(board.lower().encode(), dtype=np.uint8) - 97


def _temper(p: np.ndarray, T: float) -> np.ndarray:
    q = np.log(np.clip(p, 1e-9, None)) / T
    q = np.exp(q - q.max())
    return q / q.sum()


class Policy:
    name = "policy"

    def act(self, board: str, path: tuple[int, ...]) -> tuple[Action, dict[str, Any]]:
        raise NotImplementedError

    def reset(self) -> None:
        pass


class RandomSwiper(Policy):
    """The floor: walks random legal neighbours for 3-6 tiles, then submits."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self._goal = 0

    def act(self, board: str, path: tuple[int, ...]) -> tuple[Action, dict[str, Any]]:
        if not path:
            self._goal = int(self.rng.integers(MIN_LEN, 7))
        moves = legal_moves(path)
        if len(path) >= self._goal or not moves or len(path) >= MAX_LEN:
            return ("submit", -1), {}
        return ("extend", int(self.rng.choice(moves))), {}


class StudentPolicy(Policy):
    name = "student"

    def __init__(self, model: NanoAgent, device: torch.device, temperature: float = 1.0, seed: int = 0):
        self.model = model.to(device).eval()
        self.device = device
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def dists(self, boards: list[str], paths: list[tuple[int, ...]]) -> list[dict[str, Any]]:
        B = len(boards)
        board = np.stack([board_to_ids(b) for b in boards])
        path = np.full((B, MAX_LEN), -1, np.int8)
        plen = np.zeros(B, np.uint8)
        for i, p in enumerate(paths):
            if p:
                path[i, : len(p)] = p
            plen[i] = len(p)
        enc = encode(board, path, plen)
        out = self.model(to_device(enc, self.device))
        type_p = torch.softmax(out["type_logits"], -1).cpu().numpy()
        tgt_p = torch.softmax(out["target_logits"], -1).cpu().numpy()
        val = torch.sigmoid(out["value_logit"]).cpu().numpy()
        return [{"type": type_p[i], "target": np.nan_to_num(tgt_p[i]), "value": float(val[i]), "legal": enc["tmask"][i]} for i in range(B)]

    def pick(self, d: dict[str, Any], path: tuple[int, ...]) -> tuple[Action, float]:
        type_p = d["type"].copy()
        tgt_p = d["target"].copy()
        # legality: extend needs a legal tile; submit needs >= MIN_LEN letters; a full path cannot extend
        if not d["legal"].any() or len(path) >= MAX_LEN:
            type_p[TYPES.index("extend")] = 0.0
        if len(path) < MIN_LEN:
            type_p[TYPES.index("submit")] = 0.0
        if type_p.sum() <= 0:
            return ("abort", -1), 0.0
        type_p /= type_p.sum()
        if self.temperature > 0:
            t = int(self.rng.choice(len(TYPES), p=_temper(type_p, self.temperature)))
        else:
            t = int(type_p.argmax())
        atype = TYPES[t]
        tile = -1
        conf = float(type_p[t])
        if atype == "extend":
            tgt_p = tgt_p * d["legal"]
            if tgt_p.sum() <= 0:
                return ("abort", -1), 0.0
            tgt_p /= tgt_p.sum()
            tile = int(self.rng.choice(16, p=_temper(tgt_p, self.temperature))) if self.temperature > 0 else int(tgt_p.argmax())
            conf *= float(tgt_p[tile])
        return (atype, tile), conf

    def act(self, board: str, path: tuple[int, ...]) -> tuple[Action, dict[str, Any]]:
        d = self.dists([board], [path])[0]
        a, conf = self.pick(d, path)
        return a, {"confidence": conf, "value": d["value"], "type_dist": d["type"], "target_dist": d["target"]}


class EscalatingPolicy(Policy):
    """Stub (PLAN.md section 3 stretch): the student acts unless its value is low; then a teacher is asked.

    `teacher` is any callable (board, path) -> Action | None (e.g. a Gemma seat proposing the next tile or a word);
    None means the teacher had nothing, and the student's action stands. Cost accounting is the caller's job."""

    name = "escalating"

    def __init__(self, student: StudentPolicy, teacher: Optional[Callable[[str, tuple[int, ...]], Optional[Action]]] = None, value_threshold: float = 0.2, conf_threshold: float = 0.0):
        self.student = student
        self.teacher = teacher
        self.value_threshold = value_threshold
        self.conf_threshold = conf_threshold
        self.n_steps = 0
        self.n_escalated = 0

    def act(self, board: str, path: tuple[int, ...]) -> tuple[Action, dict[str, Any]]:
        a, info = self.student.act(board, path)
        self.n_steps += 1
        reason = None
        if info.get("value", 1.0) < self.value_threshold:
            reason = "value"
        elif info.get("confidence", 1.0) < self.conf_threshold:
            reason = "conf"
        if reason and self.teacher is not None:
            ta = self.teacher(board, path)
            if ta is not None:
                self.n_escalated += 1
                return ta, {**info, "escalated": True, "escalation_reason": reason}
        return a, {**info, "escalated": False}
