"""Tests for algorithms/neuralbandit.py and the runner's finite-weights check."""

import numpy as np

from algorithms.neuralbandit import NeuralBandit1, NeuralBandit2, NeuralBandit3
from evaluation.environments import forest_cover_stream, load_forest_cover
from evaluation.runner import check_finite, run


def test_probs():
    p = NeuralBandit1(7, gamma=0.005, seed=0)
    p.select((np.arange(94) % 8 == 0).astype(np.float32))
    k_hat = int(np.argmax(p.last_probs))
    assert np.isclose(p.last_probs.sum(), 1.0)
    assert np.isclose(p.last_probs[k_hat], 1 - 0.005 + 0.005 / 7)


def test_empirical_frequencies_match_probs():
    p = NeuralBandit1(7, gamma=0.3, seed=1)
    x = (np.arange(94) % 8 == 0).astype(np.float32)
    n = 50_000
    counts = np.bincount([p.select(x) for _ in range(n)], minlength=7)
    assert np.allclose(counts / n, p.last_probs, atol=0.006)


def test_learns_toy_problem():
    rng = np.random.default_rng(0)
    p = NeuralBandit1(3, d=10, hidden=5, lam=1.0, gamma=0.05, seed=0)
    hits = []
    for _ in range(5000):
        k = int(rng.integers(3))
        x = np.zeros(10, dtype=np.float32)
        x[k] = 1
        a = p.select(x)
        p.update(x, a, float(a == k))
        hits.append(a == k)
    assert np.mean(hits[-1000:]) > 0.8


def test_weights_finite_on_forest_cover():
    X, y = load_forest_cover()
    p = NeuralBandit1(7, hidden=25, lam=1.0, seed=0)
    run(p, forest_cover_stream(X, y, 20_000, seed=0))
    check_finite(p)  # harshest grid setting (lam = 1, 1/P up to 1400)


# ---------------------------------------------------------------- NB2 / NB3 (committees)
X0 = (np.arange(94) % 8 == 0).astype(np.float32)


def weights(p):
    return [(s.W1.copy(), s.w2.copy()) for s in p.stacks]


def changed_networks(p, before):
    """Set of action indices whose network changed, asserting it changed in every model."""
    out = set()
    for s, (W1, w2) in zip(p.stacks, before):
        for k in range(p.n_actions):
            diff = (s.w2[:, k] != w2[:, k]).any(1) | (s.W1[:, k] != W1[:, k]).any((1, 2))
            if diff.any():
                assert diff.all()            # every model of the stack (plan D5)
                out.add(k)
    return out


def test_nb2_greedy_step_updates_network_a_of_every_model():
    p = NeuralBandit2(7, gamma=0.0, seed=0)
    a = p.select(X0)
    assert a == p._greedy
    before, log_w = weights(p), p.exp3.log_w.copy()
    p.update(X0, a, 1)
    assert changed_networks(p, before) == {a}
    assert len(p.stacks) == 5 and all(s.W1.shape[0] == 3 for s in p.stacks)
    assert np.flatnonzero(p.exp3.log_w != log_w).tolist() == [p._m]


def test_nb2_exploration_step_updates_no_network():
    p = NeuralBandit2(7, seed=0)
    p.select(X0)
    before, log_w = weights(p), p.exp3.log_w.copy()
    p.update(X0, (p._greedy + 1) % 7, 1)
    assert changed_networks(p, before) == set()
    assert np.flatnonzero(p.exp3.log_w != log_w).tolist() == [p._m]   # EXP3 always updated


def test_nb3_gate_and_only_played_exp3_changes():
    p = NeuralBandit3(7, gamma=0.0, seed=0)
    a = p.select(X0)
    before, log_w = weights(p), np.array([e.log_w.copy() for e in p.exp3s])
    p.update(X0, a, 1)
    assert changed_networks(p, before) == {a}
    diff = np.array([e.log_w for e in p.exp3s]) != log_w
    assert np.argwhere(diff).tolist() == [[a, p._ms[a]]]

    p.select(X0)
    before = weights(p)
    p.update(X0, (p._greedy + 1) % 7, 1)
    assert changed_networks(p, before) == set()


def toy_run(p, steps, seed=0):
    """Reward 1 iff action == index of the one-hot context (learnable, K = d = 3)."""
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        k = int(rng.integers(3))
        x = np.zeros(3, dtype=np.float32)
        x[k] = 1
        a = p.select(x)
        assert np.isclose(p.last_probs.sum(), 1.0)
        p.update(x, a, float(a == k))
    check_finite(p)
    return p


def test_nb2_committee_picks_learning_model():
    # model 0 (lam ~ 0) never learns; model 1 (lam = 1) does -> EXP3 must concentrate on model 1 (D7)
    p = toy_run(NeuralBandit2(3, d=3, hidden=[5], lam=[1e-9, 1.0], gamma=0.05, seed=0), 3000)
    assert p.diagnostics()["model_probs"][1] > 0.8


def test_nb3_committee_favours_learning_model():
    p = toy_run(NeuralBandit3(3, d=3, hidden=[5], lam=[1e-9, 1.0], gamma=0.05, seed=0), 3000)
    assert p.diagnostics()["model_probs"].shape == (3, 2)
    # Weaker check than NB2 (plan D3 limitation): EXP3_k only sees rewards when k is played, so a
    # frozen model that always over-scores k earns the same reward rate on k as the learning model
    # and can win that one EXP3 (seen for 1 of 3 actions with seed 0). On average model 1 must win.
    assert p.diagnostics()["model_probs"][:, 1].mean() > 0.6
