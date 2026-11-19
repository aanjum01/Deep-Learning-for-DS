"""NB1 + sliding window (plan phase-6b §7.1): notebook, schedule, replay, bit-for-bit equality with NB1."""

import numpy as np

from algorithms.neuralbandit import NeuralBandit1, NeuralBandit1SW
from algorithms.random_policy import RandomPolicy
from models.mlp import MLPStack
from nonstationary.window import Notebook

from tests.toy import D, K, play, stream  # noqa: F401


def nb1sw(W, **kw):
    return NeuralBandit1SW(K, W=W, d=D, hidden=3, lam=0.5, gamma=0.2, seed=kw.pop("seed", 1), **kw)


def test_notebook_keeps_last_W_steps_oldest_first():
    nb = Notebook(2)
    for t in range(1, 3001):  # crosses several regrowths
        if t % 3:  # not every step is logged (Eq. 1 gate)
            nb.add(t, np.array([t, 0], np.float32), t % 5, t % 2, 1.5)
        if t % 400 == 0:
            nb.trim(t, 500)
    nb.trim(3000, 500)
    steps = [int(x[0]) for x, *_ in nb]
    assert steps == [s for s in range(2501, 3001) if s % 3]


def test_never_retraining_equals_nb1_bit_for_bit():
    a = play(nb1sw(W=10**9), 800)
    b = play(NeuralBandit1(K, d=D, hidden=3, lam=0.5, gamma=0.2, seed=1), 800)
    assert a == b


def test_equals_nb1_until_first_retrain_and_schedule():
    p, q = nb1sw(W=300), NeuralBandit1(K, d=D, hidden=3, lam=0.5, gamma=0.2, seed=1)
    a, b = play(p, 360)[0], play(q, 360)[0]
    assert a == b                                   # first retrain at t = 360 (first multiple of R = 60 above 300)
    assert p.retrain_steps == [360]
    play(p, 240, seed=5)
    assert p.retrain_steps == [360, 420, 480, 540, 600]
    p.W = 100                                       # a controller shrinks the window: R = 20 from the next check
    play(p, 45, seed=6)
    assert p.retrain_steps[-2:] == [620, 640]
    assert p.notebook.step[:len(p.notebook)].min() > 640 - 100


def test_retrain_equals_fresh_net_fed_logged_rows_in_order():
    p, q = nb1sw(W=200), nb1sw(W=200)
    play(p, 239)
    play(q, 239)
    x, av, click = next(stream(1, seed=9))
    a = p.select(x, avail=av)
    p.update(x, a, int(click[a]))                   # t = 240: retrain
    assert p.retrain_steps == [240]
    assert p.notebook.step[:len(p.notebook)].min() > 240 - 200
    q.select(x, avail=av)                           # same rng draws as p up to its retrain
    net = MLPStack(1, K, D, 3, [0.5], q.rng)
    for xx, aa, rr, ww in p.notebook:               # forward pass recomputed on the fresh network (plan R6)
        h, o = net.forward(xx)
        net.update(xx, h, o, aa, rr, ww)
    assert np.array_equal(net.W1, p.net.W1) and np.array_equal(net.w2, p.net.w2)


def test_only_greedy_played_rounds_are_logged_with_weight():
    p = nb1sw(W=10**9)
    for x, av, click in stream(500):
        a = p.select(x, avail=av)
        n0 = len(p.notebook)
        greedy, probs = p._greedy, p.last_probs.copy()
        p.update(x, a, int(click[a]))
        assert len(p.notebook) == n0 + (a == greedy)
        if a == greedy:
            assert p.notebook.w[len(p.notebook) - 1] == 1.0 / probs[a]


def test_retrain_draws_exactly_one_network():
    p = nb1sw(W=10**9)
    play(p, 50)
    s = p.rng.bit_generator.state
    p.retrain()
    r = np.random.default_rng(); r.bit_generator.state = s
    r.uniform(-0.5, 0.5, (1, K, D, 3)); r.uniform(-0.5, 0.5, (1, K, 3))
    assert p.rng.bit_generator.state == r.bit_generator.state


def test_determinism_and_reset():
    p = nb1sw(W=150)
    first = play(p, 600)
    assert first == play(nb1sw(W=150), 600)
    p.reset()
    assert p.t == 0 and len(p.notebook) == 0 and p.retrain_steps == []
    assert first == play(p, 600)


def test_small_window_is_finite_and_beats_random():
    p = nb1sw(W=200)
    got = np.mean(play(p, 4000, mask=False)[1])
    rnd = np.mean(play(RandomPolicy(K, seed=1), 4000, mask=False)[1])
    assert np.isfinite(p.net.W1).all() and got > rnd + 0.05


def test_offset_shifts_the_reset_schedule():
    p = nb1sw(W=300, offset=30)                     # R = 60: resets at 330, 390, ... (first above W)
    play(p, 400)
    assert p.retrain_steps == [330, 390]
