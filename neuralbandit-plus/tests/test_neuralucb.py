"""NeuralUCB (plan phase-6a §7.1): gradients, f = 0 at start, paired form = the paper's literal form, masking,
determinism, divergence."""

import numpy as np
import pytest

from algorithms.neuralucb import Diverged, NeuralUCB
from exploration.ucb import DiagUCB
from models.relu_net import ReLUNet

K, D, M = 3, 5, 8


def toy(seed=0, n=40):
    r = np.random.default_rng(seed)
    X = r.uniform(-1, 1, (n, D)) * (r.random((n, D)) < 0.7)  # some zero inputs
    return X, r.integers(K, size=n), r.integers(2, size=n).astype(float)


def test_f_is_zero_at_start():
    net = ReLUNet(K, D, M, np.random.default_rng(1))
    for x in toy()[0]:
        assert np.abs(net.forward(x)[0]).max() < 1e-12


def test_gradients_match_finite_differences():
    net = ReLUNet(K, D, M, np.random.default_rng(2))
    X, A, R = toy(3)
    net.sgd_step(X, A, R, 0.05, 0.0)  # move away from theta_0 so the penalty has a gradient
    reg = 0.3
    g_V, g_w2, _ = net.grads(X, A, R, reg)

    def loss():
        _, _, l = net.grads(X, A, R, reg)
        return l + reg / 2 * (((net.V - net.V0) ** 2).sum() + ((net.w2 - net.w20) ** 2).sum())

    eps = 1e-6
    for arr, g in ((net.V, g_V), (net.w2, g_w2)):
        for i in np.random.default_rng(4).choice(arr.size, min(15, arr.size), replace=False):
            ix = np.unravel_index(i, arr.shape)
            old = arr[ix]
            arr[ix] = old + eps; lp = loss()
            arr[ix] = old - eps; lm = loss()
            arr[ix] = old
            assert abs((lp - lm) / (2 * eps) - g[ix]) < 1e-6


class Literal:
    """The paper's literal form: W1[a] = diag(W_a, W_a) (2D x M) on [x, x]/sqrt(2), w2 = (w, -w), full-gradient Z
    of Alg. 1 (kept as its diagonal, built from outer products). Reference only."""

    def __init__(self, net, lam):
        W = net.V[:, :, :M // 2] * np.sqrt(2)
        self.W1 = np.zeros((K, 2 * D, M))
        self.W1[:, :D, :M // 2], self.W1[:, D:, M // 2:] = W, W
        self.w2 = net.w2.copy()
        self.W10, self.w20 = self.W1.copy(), self.w2.copy()
        self.Z = lam * np.eye(K * 2 * D * M + M)

    def g(self, x, a):
        xd = np.concatenate([x, x]) / np.sqrt(2)
        z = xd @ self.W1[a]
        gW = np.zeros((K, 2 * D, M))
        gW[a] = np.outer(xd, np.sqrt(M) * self.w2 * (z > 0))
        return np.sqrt(M) * np.maximum(z, 0) @ self.w2, np.concatenate([gW.ravel(), np.sqrt(M) * np.maximum(z, 0)])

    def sgd(self, X, A, R, step, reg):
        gW, gw = reg * (self.W1 - self.W10), reg * (self.w2 - self.w20)
        for x, a, r in zip(X, A, R):
            f, g = self.g(x, a)
            gW += (f - r) / len(R) * g[:-M].reshape(gW.shape)
            gw += (f - r) / len(R) * g[-M:]
        self.W1 -= step * gW
        self.w2 -= step * gw


def test_paired_form_equals_literal_paper_form():
    lam = 0.5
    net = ReLUNet(K, D, M, np.random.default_rng(5))
    ucb = DiagUCB(K, D, M, lam)
    ref = Literal(net, lam)
    X, A, R = toy(6)
    for step in range(5):
        for x, a in zip(X[:8], A[:8]):  # forward + bonus + Z update
            f, z = net.forward(x)
            gs = DiagUCB.grad_sq(x, z, net.w2)
            b = ucb.bonus(*gs)
            for k in range(K):
                fk, g = ref.g(x, k)
                assert abs(fk - f[k]) < 1e-10
                assert abs(np.sqrt(g @ (g / np.diag(ref.Z)) / M) - b[k]) < 1e-9
            ucb.add(a, *gs)
            _, g = ref.g(x, a)
            ref.Z += np.outer(g, g) / M      # Alg. 1 line 11, full matrix; only its diagonal is used
            ref.Z = np.diag(np.diag(ref.Z))
        net.sgd_step(X, A, R, 0.1, 0.2)
        ref.sgd(X, A, R, 0.1, 0.2)


def run(policy, n=600, seed=0, mask=True):
    r = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        x = (r.random(D) < 0.5).astype(np.float32)
        av = r.random(K) < 0.6 if mask else None
        if av is not None and not av.any():
            av[0] = True
        a = policy.select(x, avail=av)
        assert av is None or av[a]
        assert policy.last_probs[a] > 0
        policy.update(x, a, float(r.random() < 0.3 + 0.2 * a))
        out.append(a)
    return out


def test_masking_determinism_and_reset():
    p = NeuralUCB(K, d=D, m=M, gamma=0.5, lam=0.1, eta=0.1, batch=16, seed=3)
    first = run(p)
    assert first == run(NeuralUCB(K, d=D, m=M, gamma=0.5, lam=0.1, eta=0.1, batch=16, seed=3))
    p.reset()
    assert first == run(p)
    d = p.diagnostics()
    assert len(d["bonus"]) == 1 and np.isfinite(d["bonus"]).all() and 0 <= d["explore_rate"][0] <= 1


def test_training_schedule_and_diagnostics():
    p = NeuralUCB(K, d=D, m=M, batch=8, seed=0)
    calls = []
    p._train = lambda: calls.append(p.t)
    run(p, n=2500, mask=False)
    assert calls == list(range(1, 2001)) + [2100, 2200, 2300, 2400, 2500]
    assert len(p.diagnostics()["bonus"]) == 3  # 1000, 2000, 2500 (partial)


def test_float_contexts_are_not_rounded():
    p = NeuralUCB(K, d=D, m=M, batch=4, seed=0)
    x = np.ones(D, dtype=np.float32)
    p.select(x)
    p.update(x, 0, 1.0)
    x2 = np.full(D, 0.25, dtype=np.float32)
    p.select(x2)
    p.update(x2, 1, 0.0)
    assert p.X.dtype == np.float32 and p.X[1, 0] == 0.25 and p.X[0, 0] == 1


def test_divergence_raises():
    p = NeuralUCB(K, d=D, m=M, eta=1e6, lam=0.1, batch=16, seed=0)
    with pytest.raises(Diverged) as e:
        run(p, n=300, mask=False)
    assert e.value.step < 300


def test_nonfinite_scores_raise_diverged():
    p = NeuralUCB(K, d=D, m=M, seed=0)
    p.net.V[:] = 1e308  # finite weights whose output overflows
    with pytest.raises(Diverged):
        p.select(np.ones(D, dtype=np.float32))
