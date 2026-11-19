"""NeuralUCB [zhou2020neural] — the network + UCB core of NeuralBandit++ (Phase 6a: no window, no controller)."""

import numpy as np

from algorithms.base import Bandit
from exploration.egreedy import argmax_random_tie
from exploration.ucb import DiagUCB
from models.relu_net import ReLUNet


class Diverged(FloatingPointError):
    """Raised when the network's weights stop being finite after a training call (plan phase-6a R2)."""

    def __init__(self, step):
        super().__init__(f"NeuralUCB weights became non-finite at step {step}")
        self.step = step


class NeuralUCB(Bandit):
    """NeuralUCB [zhou2020neural, Alg. 1-2; real-data setup §7.2], plan paper/plans/phase-6a.md.

    select: U_a = f(x^(a)) + gamma * sqrt(g^T Z^-1 g / m) over the available arms, ties at random [Alg. 1 l. 7-8].
    update: Z += g g^T / m for the played arm [l. 11]; store (x, a, r); TrainNN when t <= 2000 or t % 100 == 0
    (t = examples stored) [§7.2; R6]: J SGD steps, batch 500 drawn with replacement from all t examples [Alg. 2,
    §7.2], loss = batch mean (f - r)^2 / 2 + (m lam / t) / 2 ||theta - theta_0||^2 (= the paper's L / t, Q3).

    Deviations from the paper (plan §6): constant gamma (as the paper's own experiments, §7.1), diagonal Z (§7.2);
    J = 10 instead of 1000 (cost, Q2); warm start instead of restarting from theta_0 (Q1); SGD step eta / m
    (Thm 4.5's eta ~ 1/m; the raw grid diverges with Eq. 2.3's sqrt(m) output, R1); training every round before
    round 2000 (Alg. 1 + the authors' code; §7.2 is ambiguous, R6); raw inputs, not unit length (Q5).
    """

    def __init__(self, n_actions, d=94, gamma=1.0, lam=0.01, eta=0.02, J=10, batch=500, m=100, seed=0):
        super().__init__(n_actions, seed)
        self.d, self.gamma, self.lam, self.eta = d, gamma, lam, eta
        self.J, self.batch, self.m = J, batch, m
        self.reset()

    def _init_state(self):
        K, d, m = self.n_actions, self.d, self.m
        self.net = ReLUNet(K, d, m, self.rng)
        self.ucb = DiagUCB(K, d, m, self.lam)
        self.t = 0                                    # examples stored = steps played
        self.X = np.zeros((1024, d), dtype=np.uint8)  # 0/1 contexts stay uint8 (Forest); else float32
        self.A = np.zeros(1024, dtype=np.int16)
        self.R = np.zeros(1024)
        self._win = [0.0, 0, 0.0, 0]                  # this 1,000-step window: bonus sum, explored, loss sum, losses
        self._diag = []

    def select(self, context, candidates=None, avail=None):
        x = context.astype(np.float64)
        f, z = self.net.forward(x)
        self._g = DiagUCB.grad_sq(x, z, self.net.w2)
        self._b = self.ucb.bonus(*self._g)
        u = f + self.gamma * self._b
        if not np.isfinite(u).all():  # huge but finite weights can still overflow f or Z (Forest, eta 0.5): R2
            raise Diverged(self.t)
        if avail is not None:  # only the arms shown this round (Phase 5 masking)
            u, f = np.where(avail, u, -np.inf), np.where(avail, f, -np.inf)
        best = np.flatnonzero(u == u.max())
        self.last_probs = np.zeros(self.n_actions)
        self.last_probs[best] = 1.0 / len(best)
        a = argmax_random_tie(u, self.rng)
        self._explored = a != np.argmax(f)  # UCB's pick vs the network's best guess (first on ties)
        return a

    def update(self, context, action, reward, candidates=None):
        self.ucb.add(action, *self._g)
        self._store(context, action, reward)
        self._win[0] += self._b[action]
        self._win[1] += self._explored
        if self.t <= 2000 or self.t % 100 == 0:
            self._train()
        if self.t % 1000 == 0:
            self._flush()

    def _store(self, x, a, r):
        if self.t == len(self.A):  # grow by doubling
            self.X = np.concatenate([self.X, np.zeros_like(self.X)])
            self.A = np.concatenate([self.A, np.zeros_like(self.A)])
            self.R = np.concatenate([self.R, np.zeros_like(self.R)])
        if self.X.dtype == np.uint8 and not np.all((x == 0) | (x == 1)):
            self.X = self.X.astype(np.float32)
        self.X[self.t], self.A[self.t], self.R[self.t] = x, a, r
        self.t += 1

    def _train(self):
        t = self.t
        with np.errstate(over="ignore", invalid="ignore"):  # divergence is caught just below (R2)
            for _ in range(self.J):
                i = self.rng.integers(t, size=self.batch)
                loss = self.net.sgd_step(self.X[i].astype(np.float64), self.A[i], self.R[i],
                                         self.eta / self.m, self.m * self.lam / t)
        if not (self.net.finite() and np.isfinite(loss)):
            raise Diverged(t - 1)
        self._win[2] += loss
        self._win[3] += 1

    def _row(self, n):
        """Window means over its n steps: bonus of the played arm, exploration rate, training data loss."""
        b, e, l, k = self._win
        return b / n, e / n, l / k if k else np.nan

    def _flush(self):
        self._diag.append(self._row(1000))
        self._win = [0.0, 0, 0.0, 0]

    def diagnostics(self):
        """Per 1,000 steps (last window may be partial): bonus of the played arm, exploration rate, train loss (Q8)."""
        rows = self._diag + ([self._row(self.t % 1000)] if self.t % 1000 else [])
        b, e, l = np.array(rows, dtype=np.float64).reshape(-1, 3).T
        return {"bonus": b, "explore_rate": e, "train_loss": l}
