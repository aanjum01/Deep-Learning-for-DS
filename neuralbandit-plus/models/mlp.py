"""Batched one-hidden-layer sigmoid networks, no biases [allesiardo2014, §3, p. 3]."""

import numpy as np
from scipy.special import expit


class MLPStack:
    """M models x K actions of networks d -> C -> 1, sigmoid on hidden and output units.

    No biases: the paper counts N = dim(X)*C + C connections [allesiardo2014, §3, p. 3] (plan D2).
    Weights start uniform in (-0.5, 0.5) [allesiardo2014, Alg. 1, p. 4]. Each model m has its own
    step size lam[m]. Only nonzero inputs are used (identical maths: zero inputs contribute nothing
    forward and get zero gradient).
    """

    def __init__(self, M, K, d, C, lam, rng):
        self.W1 = rng.uniform(-0.5, 0.5, (M, K, d, C))
        self.w2 = rng.uniform(-0.5, 0.5, (M, K, C))
        self.lam = np.asarray(lam, dtype=np.float64).reshape(M)

    def forward(self, x):
        """Return hidden h (M, K, C) and output o (M, K) for context x (d,)."""
        idx = np.flatnonzero(x)
        h = expit(x[idx] @ self.W1[:, :, idx])                  # (n,) @ (M, K, n, C)
        return h, expit((h * self.w2).sum(-1))

    def update(self, x, h, o, action, r, weight):
        """One plain gradient step on weight * 1/2 (o - r)^2 for network `action` of every model.

        Delta = lam * x * delta [allesiardo2014, §3, p. 3; Alg. 1, p. 4] (plan D6). No gating here:
        the caller applies Eq. 1's 1[k_hat = k] (plan D10) — NB2/NB3 must gate per model.
        """
        idx = np.flatnonzero(x)
        h, o = h[:, action], o[:, action]                        # (M, C), (M,)
        d_o = (o - r) * o * (1 - o)                              # (M,)
        d_h = d_o[:, None] * self.w2[:, action] * h * (1 - h)    # (M, C)
        s = self.lam * weight                                    # (M,)
        self.w2[:, action] -= (s * d_o)[:, None] * h
        self.W1[:, action, idx] -= s[:, None, None] * x[idx][None, :, None] * d_h[:, None, :]
