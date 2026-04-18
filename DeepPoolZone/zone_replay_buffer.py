"""
zone_replay_buffer.py
---------------------
Experience replay buffer for the zone-based repositioning DQN.

Each transition stores what happened when one taxi chose a zone to move to:

    (state, action, reward, next_state, done)

    state      : (2, ROWS, COLS) float32 ndarray  — global demand+vehicle grid
                 at the moment the taxi made its repositioning decision
    action     : int  — zone_id the taxi was sent to
    reward     : float — scalar reward accumulated between this decision and
                         the next time this taxi becomes idle again
    next_state : (2, ROWS, COLS) float32 ndarray  — global grid when the taxi
                 becomes idle again (the "next decision point")
    done       : bool — True if the simulation ended before the taxi became
                        idle again (terminal transition)

All taxis share a single replay buffer and a single Q-network (parameter
sharing, same as DeepPool).  This greatly increases data efficiency because
every taxi's experience informs the shared policy.
"""

from __future__ import annotations

import random
from collections import deque
from typing import List, NamedTuple, Optional

import numpy as np


class Transition(NamedTuple):
    state:      np.ndarray   # (2, R, C) float32
    action:     int          # zone_id
    reward:     float
    next_state: np.ndarray   # (2, R, C) float32
    done:       bool


class ZoneReplayBuffer:
    """
    Fixed-capacity circular replay buffer.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.  Oldest transitions are
        discarded when the buffer is full (FIFO eviction).
    """

    def __init__(self, capacity: int = 10_000) -> None:
        self.capacity = capacity
        self._buf: deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add one transition.  Arrays are stored as-is (no copy on push)."""
        self._buf.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Transition]:
        """
        Sample a random mini-batch of transitions without replacement.

        Raises ValueError if fewer than batch_size transitions are stored.
        """
        if len(self._buf) < batch_size:
            raise ValueError(
                f"Buffer has only {len(self._buf)} transitions "
                f"but batch_size={batch_size} was requested."
            )
        return random.sample(list(self._buf), batch_size)

    def sample_arrays(self, batch_size: int):
        """
        Sample a mini-batch and return separate numpy arrays for each field.

        Returns
        -------
        states      : np.ndarray, shape (B, 2, R, C)
        actions     : np.ndarray, shape (B,), int64
        rewards     : np.ndarray, shape (B,), float32
        next_states : np.ndarray, shape (B, 2, R, C)
        dones       : np.ndarray, shape (B,), float32  (1.0 = done)
        """
        batch = self.sample(batch_size)
        states      = np.stack([t.state      for t in batch], axis=0)
        actions     = np.array([t.action     for t in batch], dtype=np.int64)
        rewards     = np.array([t.reward     for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch], axis=0)
        dones       = np.array([float(t.done) for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self._buf)

    def is_ready(self, batch_size: int) -> bool:
        """True once the buffer holds at least batch_size transitions."""
        return len(self._buf) >= batch_size
