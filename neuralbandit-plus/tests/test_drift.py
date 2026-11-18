"""Phase 5d: planted shifts in the MIND stream and the recovery measures (plan phase-5d.md §4, §7)."""

from pathlib import Path

import numpy as np
import pytest

from algorithms.linucb import LinUCB
from evaluation.environments import load_mind, mind_rows, mind_stream
from evaluation.metrics import adapts_better, clicks_lost, recovery
from evaluation.runner import run

ROOT = Path(__file__).resolve().parent.parent
W = 10_000


def toy():
    """4 arms; row i shows arms {0,1,2} and rewards only arm i % 4."""
    n = 6
    R = np.zeros((n, 4), np.uint8)
    R[np.arange(n), np.arange(n) % 4] = 1
    avail = np.tile(np.array([1, 1, 1, 0], bool), (n, 1))
    return {"X": np.ones((n, 2), np.float32), "R": R, "avail": avail}


def test_shift_relabels_rewards_and_availability():
    d = toy()
    out = list(mind_stream(d, np.arange(6), shifts=[2, 4]))
    for t, (_, r, a) in enumerate(out):
        j = (t >= 2) + (t >= 4)
        for arm in range(4):  # after shift j, arm a serves category (a + j) mod 4
            assert r[arm] == d["R"][t, (arm + j) % 4] and a[arm] == d["avail"][t, (arm + j) % 4]


def test_no_shifts_is_unchanged():
    d = toy()
    for (x1, r1, a1), (x2, r2, a2) in zip(mind_stream(d, np.arange(6)), mind_stream(d, np.arange(6), [])):
        assert (r1 == r2).all() and (a1 == a2).all() and (x1 == x2).all()


def curve(*pieces):
    """Concatenate (value, length) pieces into a ratio curve."""
    return np.concatenate([np.full(n, v, float) for v, n in pieces])


def test_stopwatch_dip_and_recovery():
    r = curve((1.0, 50_000), (0.5, 30_000), (1.0, 100_000))
    out = recovery(r, 50_000, len(r))
    assert out["t1"] == 50_000 and out["t2"] == 80_000 and out["duration"] == 30_000 and out["recovered"]
    assert out["max_loss"] == pytest.approx(0.5) and out["entered_at"] == 1.0


def test_stopwatch_ignores_short_blip():
    r = curve((1.0, 50_000), (0.5, 20_000), (1.0, W - 1), (0.5, 20_000), (1.0, 100_000))
    out = recovery(r, 50_000, len(r))
    assert out["t2"] == 50_000 + 20_000 + (W - 1) + 20_000  # the 9,999-step blip does not count


def test_stopwatch_never_below_and_never_recovers():
    assert recovery(curve((1.0, 200_000)), 50_000, 200_000)["duration"] == 0
    out = recovery(curve((1.0, 50_000), (0.6, 100_000)), 60_000, 150_000)
    assert not out["recovered"] and out["duration"] == 150_000 - 60_000 and out["since_shift"] == 90_000


def test_stopwatch_recovery_stretch_must_end_before_end():
    r = curve((1.0, 50_000), (0.5, 30_000), (1.0, W - 1))  # back up, but data ends before a full window
    assert not recovery(r, 50_000, len(r))["recovered"]


def test_clicks_lost_is_signed():
    e3, e4 = np.ones(10, int), np.zeros(10, int)
    assert clicks_lost(e4, e3, 2, 5) == 5 and clicks_lost(e3, e4, 2, 5) == -5


def test_adapts_better_verdicts():
    a, b = [100, 110, 90, 105, 95], [500, 520, 480, 510, 490]
    assert adapts_better(a, b, [10] * 5, [9] * 5) == "better"
    assert adapts_better(a, b, [8] * 5, [9] * 5) == "trades performance for robustness"
    assert adapts_better(b, a, [10] * 5, [9] * 5) == "worse"
    assert adapts_better(a, [101, 108, 92, 104, 96], [10] * 5, [9] * 5) == "no clear difference"


def test_e4_equals_e3_before_first_shift():
    """Plan §7 check 3 on real data: same seed, same draws -> identical rewards until the first shift."""
    if not (ROOT / "data/processed/mind_small.npz").exists():
        pytest.skip("run scripts/preprocess_mind.py --size small first")
    d = load_mind("small")
    rows = mind_rows(d, 100)[:30_000]
    e3 = run(LinUCB(16, d=19, alpha=0.5, seed=100), mind_stream(d, rows))
    e4 = run(LinUCB(16, d=19, alpha=0.5, seed=100), mind_stream(d, rows, shifts=[20_000]))
    assert np.array_equal(e3[:20_000], e4[:20_000]) and not np.array_equal(e3[20_000:], e4[20_000:])
