"""NeuralUCB's two-layer ReLU network on the disjoint arm input [zhou2020neural, Eq. 2.3, §7.2]."""

import numpy as np


class ReLUNet:
    """f(x^(a)) = sqrt(m) * w2 . ReLU(V[a]^T x), no biases [zhou2020neural, Eq. 2.3, L = 2].

    Disjoint model [zhou2020neural, §7.2; li2010]: x^(a) = (0, ..., x, ..., 0), so the first layer of one d*K-input
    network is stored per arm as V[a] (d x m) and the output weights w2 (m) are shared by all arms.

    Mirrored init [zhou2020neural, §3, p. 2] in its paired form (plan phase-6a R5): the paper's W1 = diag(W, W) on the
    duplicated input [x, x]/sqrt(2), W ~ N(0, 4/m), w2 = (w, -w), w ~ N(0, 2/m), equals V[a] = [W_a, W_a]/sqrt(2) on
    the raw x (both input copies always get identical updates), so f(x; theta_0) = 0 for every arm.
    """

    def __init__(self, K, d, m, rng):
        assert m % 2 == 0, "mirrored init needs an even width m"
        W = rng.normal(0.0, np.sqrt(4 / m), (K, d, m // 2))
        w = rng.normal(0.0, np.sqrt(2 / m), m // 2)
        self.V = np.concatenate([W, W], axis=2) / np.sqrt(2)
        self.w2 = np.concatenate([w, -w])
        self.V0, self.w20 = self.V.copy(), self.w2.copy()
        self.sm = np.sqrt(m)

    def forward(self, x):
        """Return f (K,) for every arm and z (K, m), the hidden pre-activations (only nonzero inputs are used)."""
        idx = np.flatnonzero(x)
        z = x[idx] @ self.V.take(idx, axis=1)             # (n,) @ (K, n, m) -> (K, m)
        return self.sm * np.maximum(z, 0) @ self.w2, z

    def grads(self, X, A, R, reg):
        """Gradients of mean_i (f(x_i^(a_i)) - r_i)^2 / 2 + reg/2 * ||theta - theta_0||^2, and the data loss.

        [zhou2020neural, Alg. 2] divided by the number of stored examples t, with reg = m*lam/t (plan Q3).
        """
        o = np.argsort(A, kind="stable")                  # group the batch by arm once: contiguous slices
        X, A, R = X[o], A[o], R[o]
        arms, start = np.unique(A, return_index=True)
        groups = [(k, slice(i, j)) for k, i, j in zip(arms, start, list(start[1:]) + [len(A)])]
        z = np.empty((len(A), self.w2.size))
        for k, s in groups:
            z[s] = X[s] @ self.V[k]
        h = np.maximum(z, 0)
        e = self.sm * h @ self.w2 - R
        de = e / len(R)
        g_w2 = self.sm * h.T @ de + reg * (self.w2 - self.w20)
        dz = de[:, None] * (self.sm * self.w2) * (z > 0)
        g_V = reg * (self.V - self.V0)
        for k, s in groups:
            g_V[k] += X[s].T @ dz[s]
        return g_V, g_w2, 0.5 * np.mean(e * e)

    def sgd_step(self, X, A, R, step, reg):
        """One gradient step on the minibatch; returns its data loss (before the step)."""
        g_V, g_w2, loss = self.grads(X, A, R, reg)
        self.V -= step * g_V
        self.w2 -= step * g_w2
        return loss

    def finite(self):
        return bool(np.isfinite(self.V).all() and np.isfinite(self.w2).all())
