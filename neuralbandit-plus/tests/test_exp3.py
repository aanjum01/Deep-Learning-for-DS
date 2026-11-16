"""Tests for exploration/exp3.py [allesiardo2014, Eq. 2-3, p. 5]."""

import numpy as np

from exploration.exp3 import EXP3


def test_probs_match_eq2():
    e = EXP3(5, 0.1, np.random.default_rng(0))
    e.log_w = np.log(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    w = np.exp(e.log_w)
    assert np.isclose(e.probs().sum(), 1.0)
    assert np.allclose(e.probs(), 0.9 * w / w.sum() + 0.1 / 5)


def test_log_space_equals_plain_eq3():
    rng = np.random.default_rng(1)
    e = EXP3(4, 0.1, np.random.default_rng(0))
    w = np.ones(4)
    for _ in range(50):
        m, y = e.draw(), float(rng.random() < 0.5)
        P = 0.9 * w / w.sum() + 0.1 / 4
        w[m] *= np.exp(0.1 * y / (P[m] * 4))   # Eq. 3 in plain space
        e.update(m, y)
    assert np.allclose(e.probs(), 0.9 * w / w.sum() + 0.1 / 4)


def test_draw_frequencies_match_probs():
    e = EXP3(4, 0.2, np.random.default_rng(0))
    e.log_w = np.array([0.0, 1.0, 2.0, -1.0])
    n = 50_000
    counts = np.bincount([e.draw() for _ in range(n)], minlength=4)
    assert np.allclose(counts / n, e.probs(), atol=0.01)


def test_finite_where_plain_space_overflows():
    e = EXP3(15, 0.1, np.random.default_rng(0))
    for _ in range(200_000):
        e.update(0, 1.0)
    assert e.log_w[0] > 710                      # exp() would overflow in plain space
    assert np.isfinite(e.log_w).all() and np.isfinite(e.probs()).all()
    assert np.isclose(e.probs()[0], 0.9 + 0.1 / 15)


def test_finds_best_arm():
    rng = np.random.default_rng(1)
    e = EXP3(2, 0.1, np.random.default_rng(0))
    for _ in range(3000):
        m = e.draw()
        e.update(m, float(rng.random() < (0.9 if m == 0 else 0.1)))
    assert e.probs()[0] > 0.9
