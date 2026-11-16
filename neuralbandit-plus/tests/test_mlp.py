"""Tests for models/mlp.py (hand-written backprop)."""

import warnings

import numpy as np
import pytest
from scipy.special import expit

from models.mlp import MLPStack


def context(rng, d, binary):
    x = np.zeros(d)
    idx = rng.choice(d, 12, replace=False)
    x[idx] = 1.0 if binary else rng.normal(size=12)
    return x


def loss(net, x, a, r):
    return 0.5 * (net.forward(x)[1][0, a] - r) ** 2


@pytest.mark.parametrize("C", [1, 5, 25, 50, 100])
@pytest.mark.parametrize("binary", [True, False])
def test_gradient_matches_finite_differences(C, binary):
    rng = np.random.default_rng(C)
    net = MLPStack(1, 7, 94, C, [0.3], rng)
    x, a, r, w = context(rng, 94, binary), 3, 1.0, 2.5
    W1, w2 = net.W1.copy(), net.w2.copy()
    net.update(x, *net.forward(x), a, r, w)
    g1, g2 = (W1 - net.W1) / (0.3 * w), (w2 - net.w2) / (0.3 * w)  # step = lam * w * grad
    net.W1, net.w2 = W1, w2
    eps = 1e-6
    rows = list(np.flatnonzero(x)[:4]) + [int(np.flatnonzero(x == 0)[0])]  # active rows + a zero row
    for P, G, entries in [(net.W1, g1, [(0, a, i, c) for i in rows for c in range(min(C, 3))]),
                          (net.w2, g2, [(0, a, c) for c in range(min(C, 5))])]:
        for e in entries:
            P[e] += eps
            lp = loss(net, x, a, r)
            P[e] -= 2 * eps
            lm = loss(net, x, a, r)
            P[e] += eps
            assert np.isclose(G[e], (lp - lm) / (2 * eps), rtol=1e-5, atol=1e-10)


@pytest.mark.parametrize("binary", [True, False])
def test_sparse_update_equals_dense(binary):
    rng = np.random.default_rng(1)
    net = MLPStack(1, 7, 94, 25, [0.5], rng)
    x, a, r, w = context(rng, 94, binary), 2, 0.0, 7.0
    W1, w2 = net.W1[0, a].copy(), net.w2[0, a].copy()
    h = expit(x @ W1)                                   # dense reference: all 94 inputs
    o = expit(h @ w2)
    d_o = (o - r) * o * (1 - o)
    d_h = d_o * w2 * h * (1 - h)
    net.update(x, *net.forward(x), a, r, w)
    assert np.allclose(net.w2[0, a], w2 - 0.5 * w * d_o * h, rtol=1e-12, atol=0)
    assert np.allclose(net.W1[0, a], W1 - 0.5 * w * np.outer(x, d_h), rtol=1e-12, atol=0)


def test_only_played_action_changes_and_batch_equals_one_by_one():
    rng = np.random.default_rng(2)
    lam = [0.01, 0.1, 1.0]
    net = MLPStack(3, 7, 94, 25, lam, rng)
    singles = [MLPStack(1, 7, 94, 25, [l], rng) for l in lam]
    for m, s in enumerate(singles):
        s.W1, s.w2 = net.W1[m:m + 1].copy(), net.w2[m:m + 1].copy()
    W1, w2 = net.W1.copy(), net.w2.copy()
    for _ in range(50):
        x, a, r, w = context(rng, 94, False), 4, float(rng.integers(2)), rng.uniform(1, 100)
        net.update(x, *net.forward(x), a, r, w)
        for s in singles:
            s.update(x, *s.forward(x), a, r, w)
    others = [k for k in range(7) if k != 4]
    assert np.array_equal(net.W1[:, others], W1[:, others])
    assert np.array_equal(net.w2[:, others], w2[:, others])
    assert not np.array_equal(net.W1[:, 4], W1[:, 4])
    for m, s in enumerate(singles):
        assert np.allclose(net.W1[m], s.W1[0], rtol=1e-12, atol=0)
        assert np.allclose(net.w2[m], s.w2[0], rtol=1e-12, atol=0)


def test_huge_weights_stay_finite_without_warnings():
    rng = np.random.default_rng(3)
    net = MLPStack(2, 7, 94, 25, [1.0, 1.0], rng)
    net.W1 = rng.choice([-1e4, 1e4], net.W1.shape)
    net.w2 = rng.choice([-1e4, 1e4], net.w2.shape)
    x = context(rng, 94, False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        h, o = net.forward(x)
        net.update(x, h, o, 1, 1.0, 1400.0)
    assert np.isfinite(h).all() and np.isfinite(o).all()
    assert np.isfinite(net.W1).all() and np.isfinite(net.w2).all()
