"""Tests for algorithms/base.py — the contract every bandit policy inherits."""

import numpy as np
import pytest

from algorithms.base import Bandit


class DummyPolicy(Bandit):
    """Throwaway policy for testing the base class only (not a real baseline)."""

    def __init__(self, n_actions=None, seed=0):
        super().__init__(n_actions, seed)
        self.reset()

    def _init_state(self):
        self.n_updates = 0                      # stands in for "learned state"

    def select(self, context, candidates=None):
        k = self.n_choices(candidates)
        self.last_probs = np.full(k, 1.0 / k)
        return int(self.rng.integers(k))

    def update(self, context, action, reward, candidates=None):
        self.n_updates += 1


class IncompletePolicy(Bandit):
    """Forgets to implement update() — Python must refuse to create it."""

    def _init_state(self):
        pass

    def select(self, context, candidates=None):
        return 0


def run(policy, steps, candidates=None):
    """Play `steps` rounds with a zero context and reward 1; return the actions."""
    x = np.zeros(94)
    actions = []
    for _ in range(steps):
        a = policy.select(x, candidates)
        policy.update(x, a, 1.0, candidates)
        actions.append(a)
    return actions


# --- Group A: the contract is enforced ------------------------------------
def test_base_class_cannot_be_created():
    with pytest.raises(TypeError):
        Bandit(7)


def test_incomplete_policy_cannot_be_created():
    with pytest.raises(TypeError):
        IncompletePolicy(7)


# --- Group B: input checks -------------------------------------------------
@pytest.mark.parametrize("bad", [0, -3])
def test_invalid_n_actions_rejected(bad):
    with pytest.raises(ValueError):
        DummyPolicy(n_actions=bad)


def test_n_choices():
    assert DummyPolicy(7).n_choices() == 7
    assert DummyPolicy(7).n_choices(np.zeros((5, 3))) == 5
    with pytest.raises(ValueError):
        DummyPolicy(None).n_choices()


# --- Group C: the select -> update loop -------------------------------------
def test_last_probs_starts_empty():
    assert DummyPolicy(7).last_probs is None


def test_fixed_arm_round_trip():
    p = DummyPolicy(7)
    actions = run(p, 100)
    assert all(0 <= a < 7 for a in actions)
    assert len(p.last_probs) == 7
    assert np.isclose(p.last_probs.sum(), 1.0)
    assert p.n_updates == 100


def test_candidate_round_trip():
    p = DummyPolicy(None)
    cands = np.zeros((5, 3))
    actions = run(p, 20, cands)
    assert all(0 <= a < 5 for a in actions)
    assert len(p.last_probs) == 5
    assert np.isclose(p.last_probs.sum(), 1.0)


# --- Group D: reproducibility (most important) ------------------------------
def test_same_seed_same_run():
    assert run(DummyPolicy(7, seed=42), 50) == run(DummyPolicy(7, seed=42), 50)


def test_different_seed_different_run():
    assert run(DummyPolicy(7, seed=1), 50) != run(DummyPolicy(7, seed=2), 50)


def test_reset_replays_exactly():
    p = DummyPolicy(7, seed=3)
    first = run(p, 50)
    p.reset()
    assert p.n_updates == 0 and p.last_probs is None
    assert run(p, 50) == first


# --- Group E: labels --------------------------------------------------------
def test_default_name():
    assert DummyPolicy(7).name == "DummyPolicy"
