"""NB1 + fade + PH alarm + controller (plan phase-6c §7)."""

import numpy as np

import pytest

from algorithms.neuralbandit import NeuralBandit1Ctrl, NeuralBandit1Fade
from nonstationary.controller import CONTROLLERS

from tests.toy import D, K, play, stream

KW = dict(rho=1e-3, d=D, hidden=3, lam=0.5, gamma=0.2, seed=1)
BOOST = {"rho": 0.05, "length": 100}


@pytest.fixture(autouse=True)
def fake_controllers(monkeypatch):
    for name, d in {"none": None, "boost": BOOST, "bad_rho": {"rho": 1.5, "length": 100}, "bad_len": {"rho": 0.05}}.items():
        monkeypatch.setitem(CONTROLLERS, name, lambda _, d=d: d)


def ctrl(threshold=50.0, **kw):
    return NeuralBandit1Ctrl(K, threshold=threshold, **KW, **kw)


class Recorder:
    """Stands in for PH: records every value it is fed, never alarms."""
    drift_detected = False

    def __init__(self):
        self.fed = []

    def update(self, x):
        self.fed.append(x)


def test_never_boosting_controller_is_the_slow_fade_bit_for_bit():
    p, q = ctrl(0.5, controller="none"), NeuralBandit1Fade(K, **KW)
    assert play(p, 1500) == play(q, 1500)
    assert np.array_equal(p.net.W1, q.net.W1)
    assert p.alarms  # alarms fired, but no boost was applied
    assert (np.diff([a["step"] for a in p.alarms]) >= 30).all()  # river restarts PH after a no-change alarm (30-step warm-up)


@pytest.mark.parametrize("name", ["bad_rho", "bad_len"])
def test_invalid_decision_counts_as_no_change(name):
    p, q = ctrl(0.5, controller=name), ctrl(0.5, controller="none")
    assert play(p, 1500) == play(q, 1500) and np.isnan(p.diagnostics()["alarm_rho"]).all()


def test_rule_controller_uses_the_configured_boost():
    p = ctrl(0.5, boost_rho=0.01, boost_length=77)
    play(p, 1500)
    assert p.alarms[0]["rho"] == 0.01 and p.alarms[0]["length"] == 77


def test_boost_replaces_rho_for_its_length_then_base_returns():
    p = ctrl(0.5, controller="boost")
    rho = {}
    for x, av, click in stream(3000):
        a = p.select(x, avail=av)
        p.update(x, a, int(click[a]))
        rho[p.t] = p.rho
    t0 = p.alarms[0]["step"]
    assert [rho[t] for t in range(t0, t0 + 100)] == [0.05] * 100
    assert rho[t0 + 100] == 1e-3
    assert (np.diff([a["step"] for a in p.alarms]) >= 100).all()  # no alarm during a boost
    assert set(p.diagnostics()["alarm_length"]) == {100}


def test_ph_not_fed_during_boost_and_restarted_after():
    p, rec, steps = ctrl(0.5, controller="boost"), Recorder(), stream(3000)
    for x, av, click in steps:  # until the first alarm
        a = p.select(x, avail=av)
        p.update(x, a, int(click[a]))
        if p.alarms:
            break
    p.ph = rec  # swapped in right after the alarm: during the boost it must be fed nothing
    for x, av, click in steps:
        a = p.select(x, avail=av)
        p.update(x, a, int(click[a]))
        if p.t == p.alarms[0]["step"] + 100:
            break
    assert rec.fed == [] and p.ph is not rec and p.boost_end == 0  # boost over: a fresh PH replaced it


def test_surprise_is_reward_minus_pre_update_estimate_every_step():
    p = ctrl(1e9)
    p.ph = rec = Recorder()
    surprises = []
    for x, av, click in stream(200):
        a = p.select(x, avail=av)
        surprises.append(int(click[a]) - p._o[0, a])
        p.update(x, a, int(click[a]))
    assert np.allclose(rec.fed, surprises)


def test_same_seed_identical_rerun_and_reset():
    p, q = ctrl(1.0), ctrl(1.0)
    first = play(p, 1500)
    assert first == play(q, 1500)
    assert p.alarms == q.alarms
    p.reset()
    assert play(p, 1500) == first
