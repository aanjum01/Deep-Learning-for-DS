"""Tests for algorithms/random_policy.py."""

import numpy as np

from algorithms.random_policy import RandomPolicy


def test_uniform_actions_and_probs():
    p = RandomPolicy(7, seed=0)
    x = np.zeros(94, dtype=np.float32)
    actions = [p.select(x) for _ in range(70_000)]
    freq = np.bincount(actions) / len(actions)
    assert len(freq) == 7 and np.allclose(freq, 1 / 7, atol=0.01)
    assert np.isclose(p.last_probs.sum(), 1.0)
