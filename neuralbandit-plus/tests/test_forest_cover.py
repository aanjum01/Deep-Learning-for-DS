"""Phase 1 done-checks: Forest Cover data and streaming environment."""

import time
from collections import deque
from itertools import islice

import numpy as np
import pytest

from evaluation.environments import forest_cover_stream, load_forest_cover


@pytest.fixture(scope="module")
def data():
    return load_forest_cover()


def test_features(data):
    X, _ = data
    assert X.shape == (581_012, 94)
    assert X.max() == 1                      # uint8, so values are 0/1
    assert (X.sum(axis=1) == 12).all()       # 10 bins + 1 wilderness + 1 soil


def test_actions(data):
    _, y = data
    shares = np.bincount(y, minlength=7) / len(y)
    np.testing.assert_allclose(shares, [0.365, 0.488, 0.062, 0.005, 0.016, 0.030, 0.035], atol=0.001)


def test_seeding(data):
    X, y = data
    run = lambda s: [(x.tobytes(), a) for x, a in forest_cover_stream(X, y, 50, s)]
    assert run(1) == run(1)
    assert run(1) != run(2)


def test_wraps(data):
    X, y = data
    s = forest_cover_stream(X, y, len(X) + 1, 3)
    first, last = next(s)[0], deque(s, maxlen=1)[0][0]
    assert (first == last).all()             # one full loop later: the start row again


def test_drift(data):
    X, y = data
    n = 1_000_001
    plain = forest_cover_stream(X, y, n, 0)
    drift = forest_cover_stream(X, y, n, 0, drift_every=500_000)
    checks = {0: 0, 499_999: 0, 500_000: 1, 1_000_000: 2}
    for t, ((_, a), (_, b)) in enumerate(zip(plain, drift)):
        if t in checks:
            assert b == (a + checks[t]) % 7
    small = zip(forest_cover_stream(X, y, 80, 0), forest_cover_stream(X, y, 80, 0, drift_every=10))
    assert all(b == (a + t // 10) % 7 for t, ((_, a), (_, b)) in enumerate(small))  # incl. 7 shifts wrap to 0


def test_speed(data):
    X, y = data
    t0 = time.perf_counter()
    deque(forest_cover_stream(X, y, 2_250_000, 0, drift_every=500_000), maxlen=0)
    dt = time.perf_counter() - t0
    print(f"2.25M steps: {dt:.2f}s")
    assert dt < 10
