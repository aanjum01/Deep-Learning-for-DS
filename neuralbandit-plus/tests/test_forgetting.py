"""NB1 + gradual forgetting (plan phase-6b2 §7): the fade, rho = 0 = NB1, theta_0, every step, traces, metrics."""

import copy

import numpy as np
import yaml

from algorithms.neuralbandit import NeuralBandit1, NeuralBandit1Fade
from evaluation.metrics import beaten_windows, tradeoff_beats, trap_blocks
from nonstationary.forgetting import fade_toward_init

from tests.toy import D, K, play, stream  # noqa: F401


def fade(rho, seed=1):
    return NeuralBandit1Fade(K, rho=rho, d=D, hidden=3, lam=0.5, gamma=0.2, seed=seed)


def test_rho_zero_is_nb1_bit_for_bit():
    nb1 = NeuralBandit1(K, d=D, hidden=3, lam=0.5, gamma=0.2, seed=1)
    p = fade(0.0)
    assert play(p, 800) == play(nb1, 800)
    assert np.array_equal(p.net.W1, nb1.net.W1) and np.array_equal(p.net.w2, nb1.net.w2)


def test_theta0_is_initial_draw_without_rng_and_never_changes():
    p, q = fade(0.01), NeuralBandit1(K, d=D, hidden=3, lam=0.5, gamma=0.2, seed=1)
    assert np.array_equal(p.W1_0, q.net.W1) and p.rng.bit_generator.state == q.rng.bit_generator.state
    W0 = p.W1_0.copy()
    play(p, 300)
    assert np.array_equal(p.W1_0, W0)


def test_update_then_fade_on_every_arm_every_step():
    p = fade(0.05)
    for x, av, click in stream(200):
        a = p.select(x, avail=av)
        r = int(click[a])
        q = copy.deepcopy(p)
        NeuralBandit1.update(q, x, a, r)                # NB1's own update only
        fade_toward_init(q.net, q.W1_0, q.w2_0, 0.05)   # then every network fades
        p.update(x, a, r)
        assert np.array_equal(p.net.W1, q.net.W1) and np.array_equal(p.net.w2, q.net.w2)


def test_no_data_decays_geometrically():
    p = fade(0.02)
    p.net.W1 += 1.0
    gap = p.net.W1 - p.W1_0
    for _ in range(30):
        fade_toward_init(p.net, p.W1_0, p.w2_0, p.rho)
    assert np.allclose(p.net.W1 - p.W1_0, gap * 0.98 ** 30)


def test_rho_change_mid_run_and_reset():
    p = fade(0.0)
    first = play(p, 300)
    p.rho = 0.5
    W = p.net.W1.copy()
    play(p, 1, seed=9)
    assert not np.allclose(p.net.W1, W)
    p.rho = 0.0
    p.reset()
    assert play(p, 300) == first and p.t == 300


def test_determinism_and_traces():
    p = fade(1e-3)
    a = play(p, 2500)
    assert a == play(fade(1e-3), 2500)
    d = p.diagnostics()
    assert all(len(d[k]) == 2 for k in ("mean_abs_W1", "mean_abs_w2", "near_start_rate"))
    assert 0 <= d["near_start_rate"].min() <= d["near_start_rate"].max() <= 1


def test_rho_parses_as_float_from_config():
    assert isinstance(yaml.safe_load("rho: 0.00002")["rho"], float)
    assert isinstance(yaml.safe_load("rho: 1e-5")["rho"], str)  # the PyYAML trap the configs avoid (plan R10)


def test_trap_blocks():
    r = np.r_[np.ones(10_000), np.zeros(10_000), np.full(10_000, 0.3), np.zeros(10_000), np.ones(5_000)]
    assert trap_blocks(r) == 2
    assert trap_blocks(r, skip=[10_000]) == 1


def test_tradeoff_beats():
    rng = np.random.default_rng(0)
    base_rate, base_lost = 0.44 + rng.normal(0, 1e-4, 10), 300 + rng.normal(0, 50, 10)
    assert tradeoff_beats(base_rate + 0.01, base_lost, base_rate, base_lost)          # better clicks, same lost
    assert tradeoff_beats(base_rate, base_lost - 1000, base_rate, base_lost)          # fewer lost, same clicks
    assert not tradeoff_beats(base_rate + 0.01, base_lost + 1000, base_rate, base_lost)  # trade-off: no
    assert not tradeoff_beats(base_rate, base_lost, base_rate, base_lost)             # tie: no
    assert not tradeoff_beats([0.5], [0], [0.4], [9])                                 # undefined SE: tie


def test_beaten_windows_and_placebo_shape():
    rng = np.random.default_rng(1)
    pt = lambda rate, lost: (rate + rng.normal(0, 1e-4, 10), lost + rng.normal(0, 20, 10))
    pts = {"W10k": pt(0.433, 100), "W493k": pt(0.45, 400), "W493k_off": pt(0.426, -10_000),
           "fade": pt(0.44, 100), "NB1": pt(0.46, 5000)}
    targets = {"W10k": ["W10k"], "W493k": ["W493k", "W493k_off"]}
    assert beaten_windows(pts, ["fade"], targets) == {"W10k": ["fade"], "W493k": []}   # must beat both 493k schedules
    assert beaten_windows(pts, ["NB1"], targets) == {"W10k": [], "W493k": []}           # placebo: more clicks, more lost


def test_invalid_rho_raises():
    import pytest
    p = fade(1.5)
    with pytest.raises(ValueError):
        play(p, 1)
