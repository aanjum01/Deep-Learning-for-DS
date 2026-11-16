"""Hindsight popularity reference line for MIND (plan Q21): not a baseline, it never learns."""

import numpy as np

from algorithms.base import Bandit
from exploration.egreedy import argmax_random_tie


class PopularityPolicy(Bandit):
    """Plays the available arm with the highest click share over the whole run (hindsight).

    shares[a] = fraction of the run's impressions where arm a earned reward 1; computed by the runner
    from the same rows the run plays, so it uses future information — a reference line only.
    """

    def __init__(self, n_actions, shares, seed=0):
        super().__init__(n_actions, seed)
        self.shares = np.asarray(shares, dtype=np.float64)
        self.reset()

    def _init_state(self):
        pass

    def select(self, context, candidates=None, avail=None):
        s = self.shares if avail is None else np.where(avail, self.shares, -np.inf)
        a = argmax_random_tie(s, self.rng)
        self.last_probs = np.zeros(self.n_actions)
        self.last_probs[a] = 1.0
        return a

    def update(self, context, action, reward, candidates=None):
        pass
