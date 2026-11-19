"""NeuralUCB's confidence bonus with a diagonal Z [zhou2020neural, Alg. 1 lines 3, 7, 11; §7.2]."""

import numpy as np


class DiagUCB:
    """bonus(a) = sqrt(g_a^T Z^-1 g_a / m), Z = diag, Z_0 = lam * I, Z += g g^T / m for the played arm.

    g = gradient of f at the current weights (ReLUNet). It factorises per arm: the first-layer part is
    x_j * u_i with u = sqrt(m) w2 * 1[z > 0], the output part is sqrt(m) * ReLU(z). Real-data runs keep only the
    diagonal of Z [zhou2020neural, §7.2]. The first-layer Z grows by g^2 / (2m), not g^2 / m: our V is the paired
    form of the paper's mirrored network on [x, x]/sqrt(2), whose two input copies each carry g/sqrt(2)
    (plan phase-6a R5) -- this gives exactly the paper's bonus.
    """

    def __init__(self, K, d, m, lam):
        self.m = m
        self.Z1 = np.full((K, d, m), float(lam))
        self.Z2 = np.full(m, float(lam))

    @staticmethod
    def grad_sq(x, z, w2):
        """Squared gradient pieces of every arm: (nonzero idx, x_idx^2, u^2 (K, m), output part^2 (K, m))."""
        idx = np.flatnonzero(x)
        m = w2.size
        return idx, x[idx] ** 2, m * w2 ** 2 * (z > 0), m * np.maximum(z, 0) ** 2

    def bonus(self, idx, x2, u2, o2):
        """Bonus of every arm (K,)."""
        s = np.einsum("j,kji->k", x2, u2[:, None, :] / self.Z1.take(idx, axis=1)) + (o2 / self.Z2).sum(1)
        return np.sqrt(s / self.m)

    def add(self, a, idx, x2, u2, o2):
        """Z += g g^T / m (diagonal) for the played arm a [Alg. 1 line 11]."""
        self.Z1[a, idx] += np.outer(x2, u2[a]) / (2 * self.m)
        self.Z2 += o2[a] / self.m
