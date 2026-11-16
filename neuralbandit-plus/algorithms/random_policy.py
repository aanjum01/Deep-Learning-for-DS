"""Uniform-random sanity baseline (no learning)."""

import numpy as np

from algorithms.base import Bandit


class RandomPolicy(Bandit):
    """Picks an action uniformly at random; sanity line only."""

    def __init__(self, n_actions, seed=0):
        super().__init__(n_actions, seed)
        self.reset()

    def _init_state(self):
        pass

    def select(self, context, candidates=None, avail=None):
        arms = np.arange(self.n_actions) if avail is None else np.flatnonzero(avail)  # avail: plan Q11
        self.last_probs = np.zeros(self.n_actions)
        self.last_probs[arms] = 1.0 / len(arms)
        return int(arms[self.rng.integers(len(arms))])

    def update(self, context, action, reward, candidates=None):
        pass
