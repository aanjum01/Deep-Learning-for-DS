"""Tests for algorithms/linucb.py."""

import time

import numpy as np

from algorithms.linucb import LinUCB


def test_learns_toy_problem():
    rng = np.random.default_rng(0)
    p = LinUCB(3, d=5, alpha=1.0, seed=0)
    hits = []
    for _ in range(3000):
        k = rng.integers(3)
        x = np.zeros(5, dtype=np.float32)
        x[k] = 1
        a = p.select(x)
        p.update(x, a, float(a == k))
        hits.append(a == k)
    assert np.mean(hits[-500:]) > 0.9


def test_sherman_morrison_matches_inverse():
    rng = np.random.default_rng(1)
    p = LinUCB(3, d=10, seed=1)
    A = np.tile(np.eye(10), (3, 1, 1))
    for _ in range(2000):
        x = (rng.random(10) < 0.3).astype(np.float32)
        a = int(rng.integers(3))
        p.update(x, a, float(rng.integers(2)))
        A[a] += np.outer(x, x)
    for a in range(3):
        assert np.allclose(p.A_inv[a], np.linalg.inv(A[a]), rtol=1e-6)


def test_ties_broken_randomly():
    x = np.ones(5, dtype=np.float32)
    firsts = set()
    for s in range(20):
        p = LinUCB(3, d=5, seed=s)
        firsts.add(p.select(x))
    assert len(firsts) > 1
    assert np.isclose(p.last_probs.sum(), 1.0)


def test_speed():
    rng = np.random.default_rng(0)
    p = LinUCB(7, seed=0)
    xs = np.zeros((20_000, 94), dtype=np.float32)
    for row in xs:
        row[rng.choice(94, 12, replace=False)] = 1
    t0 = time.perf_counter()
    for x in xs:
        a = p.select(x)
        p.update(x, a, float(rng.integers(2)))
    us = (time.perf_counter() - t0) / len(xs) * 1e6
    print(f"LinUCB: {us:.1f} us/step")
    assert us < 5000  # loose: only catches a gross slowdown, not machine load
