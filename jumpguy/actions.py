"""Discrete action space the live game actually accepts."""

from __future__ import annotations

from enum import IntEnum


class Action(IntEnum):
    NOOP = 0
    JUMP = 1  # Space / tap. Also restarts after game over (Space or R).


N_ACTIONS = 2
ACTION_NAMES = ("noop", "jump")
