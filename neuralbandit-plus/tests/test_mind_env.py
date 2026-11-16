"""Phase 5b: masking (plan Q11), MIND stream (Q5-Q8), runner, and Forest Cover unchanged (Q18)."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from algorithms.linucb import LinUCB
from algorithms.neuralbandit import NeuralBandit1, NeuralBandit2, NeuralBandit3
from algorithms.popularity import PopularityPolicy
from algorithms.random_policy import RandomPolicy
from evaluation.environments import load_mind, mind_rows, mind_stream
from evaluation.runner import run, run_one
from exploration.egreedy import egreedy

ROOT = Path(__file__).resolve().parent.parent
K, D = 6, 5
POLICIES = [lambda: RandomPolicy(K), lambda: LinUCB(K, d=D, alpha=0.5),
            lambda: NeuralBandit1(K, d=D, hidden=5, lam=1), lambda: NeuralBandit2(K, d=D, hidden=(1, 5), lam=(1,)),
            lambda: NeuralBandit3(K, d=D, hidden=(1, 5), lam=(1,)), lambda: PopularityPolicy(K, np.arange(K))]


def test_egreedy_mask():
    rng = np.random.default_rng(0)
    avail = np.array([0, 1, 0, 1, 1, 0], bool)
    for _ in range(2000):
        a, k_hat, P = egreedy(rng.random(K), 0.5, rng, avail)
        assert avail[a] and avail[k_hat]
        assert P.sum() == pytest.approx(1) and (P[~avail] == 0).all() and P[k_hat] == pytest.approx(0.5 + 0.5 / 3)


@pytest.mark.parametrize("make", POLICIES)
def test_policies_only_pick_available(make):
    p, rng = make(), np.random.default_rng(1)
    for _ in range(300):
        avail = rng.random(K) < 0.4
        avail[rng.integers(K)] = True
        x = rng.random(D).astype(np.float32)
        a = p.select(x, avail=avail)
        assert avail[a]
        assert p.last_probs.sum() == pytest.approx(1) and (p.last_probs[~avail] == 0).all()
        p.update(x, a, int(rng.random() < 0.3))


def test_runner_mind_reward_is_chosen_arms_reward():
    stream = [(np.ones(D, np.float32), np.array([0, 1, 0, 0, 1, 0]), np.array([1, 1, 0, 0, 1, 0], bool))] * 50
    rewards = run(RandomPolicy(K, seed=3), stream)
    assert set(rewards) <= {0, 1} and 0 < rewards.mean() < 1  # picks among arms 0, 1, 4: reward on 1 and 4


@pytest.fixture(scope="module")
def mind():
    if not (ROOT / "data/processed/mind_small.npz").exists():
        pytest.skip("run scripts/preprocess_mind.py --size small first")
    return load_mind("small")


def test_mind_rows(mind):
    r0, r0b, r1 = mind_rows(mind, 0), mind_rows(mind, 0), mind_rows(mind, 1)
    assert np.array_equal(r0, r0b) and not np.array_equal(r0, r1)
    users = np.unique(mind["user"])
    assert len(np.unique(mind["user"][r0])) == round(0.9 * len(users))
    assert (np.diff(mind["time"][r0]) >= np.timedelta64(0)).all()  # time order kept


def test_mind_random_matches_analytic(mind):
    rows = mind_rows(mind, 0)[:20_000]
    rewards = run(RandomPolicy(mind["R"].shape[1], seed=0), mind_stream(mind, rows))
    analytic = (mind["R"][rows].sum(1) / mind["avail"][rows].sum(1)).mean()  # plan Q21 check 2
    assert rewards.mean() == pytest.approx(analytic, abs=0.01)


def test_forest_cover_unchanged():
    """Same seeds -> the first 5,000 rewards equal the saved Phase 2-4 runs (plan Q18)."""
    C = {"RandomPolicy": RandomPolicy, "LinUCB": LinUCB, "NeuralBandit1": NeuralBandit1,
         "NeuralBandit2": NeuralBandit2, "NeuralBandit3": NeuralBandit3}
    checked = 0
    for name in ["phase2_drift", "phase3_drift", "phase4_drift"]:
        cfg = yaml.safe_load(open(ROOT / f"configs/{name}.yaml"))
        for a in cfg["algorithms"]:
            saved = ROOT / f"results/raw/{name}/{a['label']}_seed0.npz"
            if saved.exists():
                new = run_one(C[a["class"]], a["params"], 0, 5_000, cfg["drift_every"])[0]
                assert np.array_equal(np.load(saved)["rewards"][:5_000], new), a["label"]
                checked += 1
    if not checked:
        pytest.skip("no saved Forest Cover runs in results/raw")
