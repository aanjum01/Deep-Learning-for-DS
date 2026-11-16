"""EXP3 over M models [allesiardo2014, Eq. 2-3, p. 5], weights kept in log-space."""

import numpy as np


class EXP3:
    """Adversarial bandit that picks one of M models.

    Weights start at 1 (log_w = 0). log-space is the same maths as Eq. 2-3 and cannot overflow
    (Nov 12 decision): w_m / sum(w) = exp(log_w_m - max) / sum(exp(log_w - max)).
    """

    def __init__(self, M, gamma_model, rng):
        self.M, self.gamma_model, self.rng = M, gamma_model, rng
        self.log_w = np.zeros(M)

    def probs(self):
        """Eq. 2: P(m) = (1 - gamma_model) w_m / sum(w) + gamma_model / M."""
        w = np.exp(self.log_w - self.log_w.max())  # softmax, overflow-safe
        return (1 - self.gamma_model) * w / w.sum() + self.gamma_model / self.M

    def draw(self):
        """Sample m ~ probs() as the Eq. 2 mixture: uniform w.p. gamma_model, else m ~ w / sum(w).

        Same distribution as probs() without a softmax (~2x faster, matters for NB3's K draws/step).
        Inverse CDF on unnormalised weights; side="left" keeps the index <= M - 1.
        """
        if self.rng.random() < self.gamma_model:
            return int(self.rng.integers(self.M))
        c = np.cumsum(np.exp(self.log_w - self.log_w.max()))
        return int(np.searchsorted(c, self.rng.random() * c[-1]))

    def update(self, m, y):
        """Eq. 3: w_m <- w_m exp(gamma_model y / (P(m) M)); only the drawn model changes."""
        self.log_w[m] += self.gamma_model * y / (self.probs()[m] * self.M)
