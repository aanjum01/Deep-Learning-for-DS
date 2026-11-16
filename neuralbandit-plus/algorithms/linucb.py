"""Disjoint LinUCB baseline."""

import numpy as np

from algorithms.base import Bandit
from exploration.egreedy import argmax_random_tie


class LinUCB(Bandit):
    """Disjoint LinUCB [li2010, Alg. 1, p. 4].

    Per arm a: A_a = I, b_a = 0, theta_a = A_a^-1 b_a,
    score = theta_a.x + alpha * sqrt(x' A_a^-1 x); only the chosen arm is updated
    (A += xx', b += r x). A^-1 is kept directly and updated with Sherman-Morrison
    (same maths as Alg. 1, no matrix inverse per step).

    Decisions: D3 - no bias feature, no normalisation (x used as given);
    D4 - ties broken uniformly at random with self.rng ([li2010]: "arbitrarily").
    """

    def __init__(self, n_actions, d=94, alpha=1.0, seed=0):
        super().__init__(n_actions, seed)
        self.d = d
        self.alpha = alpha
        self.reset()

    def _init_state(self):
        self.A_inv = np.tile(np.eye(self.d), (self.n_actions, 1, 1))
        self.b = np.zeros((self.n_actions, self.d))

    def select(self, context, candidates=None, avail=None):
        x = context.astype(np.float64)
        Ax = self.A_inv @ x                                   # (K, d)
        score = (self.b * Ax).sum(1) + self.alpha * np.sqrt(np.maximum((Ax * x).sum(1), 0))
        if avail is not None:  # only the arms available this round (A_t) [li2010, §2.1] (plan Q11)
            score = np.where(avail, score, -np.inf)
        best = np.flatnonzero(score == score.max())
        self.last_probs = np.zeros(self.n_actions)
        self.last_probs[best] = 1.0 / len(best)
        return argmax_random_tie(score, self.rng)

    def update(self, context, action, reward, candidates=None):
        x = context.astype(np.float64)
        Ax = self.A_inv[action] @ x
        self.A_inv[action] -= np.outer(Ax, Ax) / (1.0 + x @ Ax)   # Sherman-Morrison
        self.b[action] += reward * x
