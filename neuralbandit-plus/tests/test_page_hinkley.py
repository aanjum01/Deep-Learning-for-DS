"""Page-Hinkley (plan phase-6c §7): river's detector equals Gama et al.'s Alg. 1, drops version, written out by hand."""

import numpy as np

from nonstationary.drift import make_detector


def gama_ph_drop(xs, lam, delta=0.005, min_n=30):
    """[gama2014, App. A Alg. 1] on -x (a drop in x = a rise in -x): m_T = sum(-x_t + xbar_T - delta), alarm when m_T - min m >= lam.
    River tests only after min_n points and uses '>'; no reset needed here (first alarm only)."""
    s = m = 0.0
    m_min = np.inf
    for t, x in enumerate(xs, 1):
        s += x
        m += -(x - s / t) - delta
        m_min = min(m_min, m)
        if t >= min_n and m - m_min > lam:
            return t
    return None


def river_first_alarm(xs, lam):
    ph = make_detector(lam)
    for t, x in enumerate(xs, 1):
        ph.update(x)
        if ph.drift_detected:
            return t
    return None


def test_matches_gama_formula_on_noisy_drop():
    r = np.random.default_rng(0)
    xs = np.concatenate([r.normal(0, 0.5, 3000), r.normal(-0.3, 0.5, 3000)])
    for lam in (5, 20, 50, 80):  # 5 and 20 alarm early on noise; 50 and 80 only after the drop
        t = river_first_alarm(xs, lam)
        assert t is not None and t == gama_ph_drop(xs, lam)
        assert lam < 50 or t > 3000


def test_no_alarm_on_constant_stream_or_on_a_rise():
    assert river_first_alarm(np.zeros(5000), 1) is None
    assert river_first_alarm(np.r_[np.zeros(1000), np.ones(4000)], 5) is None  # mode "down": rises never alarm
