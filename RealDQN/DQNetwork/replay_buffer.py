from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Sum Tree — O(log n) proportional sampling for Prioritized Replay
# ---------------------------------------------------------------------------

class SumTree:
    """Binary tree where each leaf holds a priority value and internal nodes
    hold the sum of their children.  Supports O(log n) proportional sampling
    and O(log n) priority updates."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._tree = np.zeros(2 * capacity, dtype=np.float64)
        self._min  = np.full(2 * capacity, float("inf"), dtype=np.float64)
        self._write_idx = 0
        self._size = 0

    @property
    def total(self) -> float:
        return float(self._tree[1])

    @property
    def min_priority(self) -> float:
        return float(self._min[1]) if self._size > 0 else 0.0

    def __len__(self) -> int:
        return self._size

    def add(self, priority: float) -> int:
        """Insert a new leaf (overwriting oldest if full). Returns leaf index."""
        idx = self._write_idx
        self._write_idx = (self._write_idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        self._update_node(idx, priority)
        return idx

    def update(self, leaf_idx: int, priority: float) -> None:
        self._update_node(leaf_idx, priority)

    def sample(self, value: float) -> int:
        """Walk down the tree to find the leaf whose cumulative priority
        contains *value* (sampled uniformly from [0, total])."""
        idx = 1
        while idx < self.capacity:
            left = idx << 1
            if value <= self._tree[left]:
                idx = left
            else:
                value -= self._tree[left]
                idx = left + 1
        return idx - self.capacity

    def _update_node(self, leaf_idx: int, priority: float) -> None:
        tree_idx = leaf_idx + self.capacity
        self._tree[tree_idx] = priority
        self._min[tree_idx] = priority
        tree_idx >>= 1
        while tree_idx >= 1:
            left = tree_idx << 1
            right = left + 1
            self._tree[tree_idx] = self._tree[left] + self._tree[right]
            self._min[tree_idx]  = min(self._min[left], self._min[right])
            tree_idx >>= 1


# ---------------------------------------------------------------------------
# Prioritized Replay Buffer  (Schaul et al., 2016)
# ---------------------------------------------------------------------------

@dataclass
class PrioritizedReplayBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    next_state_exists: torch.Tensor
    dones: torch.Tensor
    is_weights: torch.Tensor        # importance-sampling weights
    leaf_indices: np.ndarray         # for updating priorities after train_step


class PrioritizedReplayBuffer:
    """Drop-in replacement for ReplayBuffer with proportional prioritization.

    Hyperparameters
    ---------------
    alpha : float   How much prioritization to use (0 = uniform, 1 = full).
    beta  : float   Importance-sampling correction (0 = none, 1 = full).
                     Should be annealed from ~0.4 to 1.0 over training.
    eps   : float   Small constant added to TD errors so no transition has
                     zero probability of being sampled.
    """

    def __init__(
        self,
        capacity: int = 50_000,
        alpha: float = 0.6,
        beta: float = 0.4,
        eps: float = 1e-5,
    ) -> None:
        self.capacity = int(capacity)
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self._tree = SumTree(self.capacity)
        self._data: list[Transition | None] = [None] * self.capacity
        self._max_priority: float = 1.0      # new transitions get max priority

    def __len__(self) -> int:
        return len(self._tree)

    def add(self, transition: Transition) -> None:
        """Add a transition with max priority (ensures it gets sampled at
        least once before its priority is corrected)."""
        priority = self._max_priority ** self.alpha
        idx = self._tree.add(priority)
        self._data[idx] = transition

    def sample(self, batch_size: int, device: torch.device) -> PrioritizedReplayBatch:
        """Sample a batch proportional to stored priorities."""
        n = len(self._tree)
        indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)

        # Stratified sampling: divide [0, total] into batch_size segments
        segment = self._tree.total / batch_size
        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            value = random.uniform(lo, hi)
            leaf = self._tree.sample(value)
            # Safety: make sure we got a valid filled slot
            while self._data[leaf] is None:
                value = random.uniform(0, self._tree.total)
                leaf = self._tree.sample(value)
            indices[i] = leaf
            priorities[i] = self._tree._tree[leaf + self._tree.capacity]

        # Importance-sampling weights: w_i = (N * P(i))^{-beta} / max(w)
        probs = priorities / self._tree.total
        is_weights = (n * probs) ** (-self.beta)
        is_weights /= is_weights.max()            # normalise so max weight = 1

        batch = [self._data[i] for i in indices]
        states = _pad_state_batch([t.state for t in batch])
        next_states = _pad_state_batch([
            t.next_state if t.next_state is not None
            else np.zeros((1, batch[0].state.shape[1]), dtype=np.float32)
            for t in batch
        ])
        next_exists = np.asarray(
            [0.0 if t.next_state is None else 1.0 for t in batch],
            dtype=np.float32,
        )

        return PrioritizedReplayBatch(
            states=torch.from_numpy(states).to(device),
            actions=torch.tensor([t.action_index for t in batch], dtype=torch.int64, device=device),
            rewards=torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device),
            next_states=torch.from_numpy(next_states).to(device),
            next_state_exists=torch.from_numpy(next_exists).to(device),
            dones=torch.tensor([1.0 if t.done else 0.0 for t in batch], dtype=torch.float32, device=device),
            is_weights=torch.from_numpy(is_weights.astype(np.float32)).to(device),
            leaf_indices=indices,
        )

    def update_priorities(self, leaf_indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities using absolute TD errors from the last train_step."""
        for idx, td in zip(leaf_indices, td_errors):
            priority = (abs(td) + self.eps) ** self.alpha
            self._tree.update(int(idx), priority)
            self._max_priority = max(self._max_priority, abs(td) + self.eps)


# ---------------------------------------------------------------------------
# N-step return buffer
# ---------------------------------------------------------------------------

class NStepBuffer:
    """
    Converts 1-step transitions into N-step return transitions before they
    enter the main ReplayBuffer.

    For each step t, once N future rewards are known:
        G_t = r_t + γ·r_{t+1} + … + γ^{N-1}·r_{t+N-1}
    The stored transition is (s_t, a_t, G_t, s_{t+N}, done_{t+N}).

    At episode boundaries (done=True), the remaining buffered transitions are
    flushed with a shorter-than-N horizon (the episode ended early).

    Usage in the training loop:
        n_buf = NStepBuffer(n_steps=5, gamma=0.95)
        ...
        n_buf.push(Transition(state, action, reward, next_state, done))
        for t in n_buf.drain_ready():
            replay.add(t)
        # After the episode loop, drain any remaining transitions:
        for t in n_buf.drain_ready():
            replay.add(t)

    The Q-target must then use γ^N instead of γ:
        target = G_t + γ^N · (1 − done) · max Q(s_{t+N})
    Pass n_steps to DQNAgent.train_step so it applies the correct exponent.
    """

    def __init__(self, n_steps: int, gamma: float) -> None:
        self.n_steps = n_steps
        self.gamma = gamma
        self._pending: deque[Transition] = deque()
        self._ready: list[Transition] = []

    # ------------------------------------------------------------------

    def push(self, t: Transition) -> None:
        """Add a 1-step transition. Call drain_ready() afterwards."""
        self._pending.append(t)
        if t.done:
            self._flush_episode()
        elif len(self._pending) >= self.n_steps:
            self._emit_oldest()

    def drain_ready(self) -> list[Transition]:
        """Return all completed N-step transitions and clear the list."""
        ready = self._ready
        self._ready = []
        return ready

    # ------------------------------------------------------------------

    def _emit_oldest(self) -> None:
        """Emit an N-step transition rooted at the oldest pending entry."""
        buf = list(self._pending)
        G = sum((self.gamma ** k) * buf[k].reward for k in range(self.n_steps))
        last = buf[self.n_steps - 1]
        self._ready.append(Transition(
            state=buf[0].state,
            action_index=buf[0].action_index,
            reward=G,
            next_state=last.next_state,
            done=last.done,
        ))
        self._pending.popleft()

    def _flush_episode(self) -> None:
        """Episode ended — emit all remaining pending transitions with shorter horizons."""
        buf = list(self._pending)
        n = len(buf)
        for start in range(n):
            G = sum((self.gamma ** k) * buf[start + k].reward for k in range(n - start))
            self._ready.append(Transition(
                state=buf[start].state,
                action_index=buf[start].action_index,
                reward=G,
                next_state=None,   # episode ended, no valid next state
                done=True,
            ))
        self._pending.clear()


@dataclass
class Transition:
    state: np.ndarray
    action_index: int
    reward: float
    next_state: np.ndarray | None
    done: bool


@dataclass
class ReplayBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    next_state_exists: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int = 50000):
        self.capacity = int(capacity)
        self.buffer: deque[Transition] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int, device: torch.device) -> ReplayBatch:
        batch = random.sample(self.buffer, batch_size)
        states = _pad_state_batch([t.state for t in batch])
        next_states = _pad_state_batch([
            t.next_state if t.next_state is not None else np.zeros((1, batch[0].state.shape[1]), dtype=np.float32)
            for t in batch
        ])
        next_exists = np.asarray([0.0 if t.next_state is None else 1.0 for t in batch], dtype=np.float32)
        return ReplayBatch(
            states=torch.from_numpy(states).to(device),
            actions=torch.tensor([t.action_index for t in batch], dtype=torch.int64, device=device),
            rewards=torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device),
            next_states=torch.from_numpy(next_states).to(device),
            next_state_exists=torch.from_numpy(next_exists).to(device),
            dones=torch.tensor([1.0 if t.done else 0.0 for t in batch], dtype=torch.float32, device=device),
        )


def _pad_state_batch(states: Sequence[np.ndarray]) -> np.ndarray:
    max_cands = max(s.shape[0] for s in states)
    feat_dim = states[0].shape[1]
    out = np.zeros((len(states), max_cands, feat_dim), dtype=np.float32)
    for i, s in enumerate(states):
        out[i, : s.shape[0], :] = s
    return out
