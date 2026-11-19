"""Sliding-window memory: the rows a learner saw in its last W steps (plan phase-6b, Q1; [gama2014, §3.1.1])."""

import numpy as np

FIELDS = ("X", "step", "a", "r", "w")


class Notebook:
    """Rows (step, x, action, reward, weight), oldest first, in preallocated arrays (plan R7).

    `trim(t, W)` keeps only rows with step > t - W (moved to the front); `add` doubles the arrays when full.
    Iterating yields (x, action, reward, weight) oldest first.
    """

    def __init__(self, d, dtype=np.float32):
        self.X = np.zeros((1024, d), dtype=dtype)
        self.step = np.zeros(1024, dtype=np.int64)
        self.a = np.zeros(1024, dtype=np.int16)
        self.r = np.zeros(1024, dtype=np.int8)
        self.w = np.zeros(1024)
        self.n = 0

    def __len__(self):
        return self.n

    def add(self, step, x, a, r, w):
        if self.n == len(self.step):
            for f in FIELDS:
                old = getattr(self, f)
                setattr(self, f, np.concatenate([old, np.zeros_like(old)]))
        i = self.n
        self.X[i], self.step[i], self.a[i], self.r[i], self.w[i] = x, step, a, r, w
        self.n += 1

    def trim(self, t, W):
        """Forget rows with step <= t - W."""
        k = int(np.searchsorted(self.step[:self.n], t - W, side="right"))
        if k:
            for f in FIELDS:
                arr = getattr(self, f)
                arr[:self.n - k] = arr[k:self.n]
            self.n -= k

    def __iter__(self):
        n = self.n
        return zip(self.X[:n], self.a[:n].tolist(), self.r[:n].tolist(), self.w[:n].tolist())
