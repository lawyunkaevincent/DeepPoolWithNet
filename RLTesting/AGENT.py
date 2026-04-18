# AGENT.py
import random
import collections
import os
import pickle

class SarsaAgent:
    def __init__(self, action_space, gamma=0.95, alpha=0.1, epsilon=0.1):
        """
        action_space: list (or iterable) of possible actions
        gamma: discount factor
        alpha: learning rate (if None -> 1/N)
        epsilon: ε-greedy exploration
        """
        self.action_space = list(action_space)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

        # Tabular Q and visit counts
        self.Q = {}  # key: (state, action)  -> value: float
        self.N = collections.Counter()

        # (Optional) book-keeping when loaded from checkpoint
        self._episodes_trained = 0
        self._extra_meta = {}

    def _key(self, state, action):
        # Ensure 'state' is hashable (tuple is fine).
        return (state, action)

    def get_Q(self, state, action):
        return self.Q.get(self._key(state, action), 0.0)

    def act(self, state, available_actions=None, epsilon=None):
        """
        ε-greedy policy w.r.t. current Q.
        """
        if epsilon is None:
            epsilon = self.epsilon
        acts = list(available_actions) if available_actions else self.action_space

        # Explore
        if random.random() < epsilon:
            return random.choice(acts)

        # Exploit
        qs = [self.get_Q(state, a) for a in acts]
        max_q = max(qs) if qs else 0.0
        best_actions = [a for a, q in zip(acts, qs) if q == max_q] or acts
        return random.choice(best_actions)

    def update(self, s, a, r, s_next, a_next, done):
        """
        SARSA(0) update rule.
        """
        key = self._key(s, a)
        q_sa = self.Q.get(key, 0.0)

        if done:
            target = r
        else:
            q_next = self.get_Q(s_next, a_next)
            target = r + self.gamma * q_next

        # Step-size
        if self.alpha is None:
            self.N[key] += 1
            alpha = 1.0 / self.N[key]
        else:
            alpha = self.alpha

        self.Q[key] = q_sa + alpha * (target - q_sa)

    # ---------------- Checkpointing ----------------

    def save(self, path, episodes_trained=0, extra_meta=None):
        """
        Save agent state as a pickle file.
        """
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        payload = {
            "version": 1,
            "action_space": list(self.action_space),
            "gamma": self.gamma,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "Q": self.Q,
            "N": self.N,
            "episodes_trained": episodes_trained,
            "extra_meta": extra_meta or {}
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            payload = pickle.load(f)
        agent = cls(
            action_space=payload["action_space"],
            gamma=payload["gamma"],
            alpha=payload["alpha"],
            epsilon=payload["epsilon"],
        )
        agent.Q = payload["Q"]
        agent.N = payload["N"]
        agent._episodes_trained = payload.get("episodes_trained", 0)
        agent._extra_meta = payload.get("extra_meta", {})
        return agent