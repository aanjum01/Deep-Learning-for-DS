import numpy as np

from evaluation.metrics import cumulative_regret, last_k_accuracy
from evaluation.runner import run


class AlwaysZero:
    def __init__(self):
        self.updates = []

    def select(self, x):
        return 0

    def update(self, x, a, r):
        self.updates.append((a, r))


def test_cumulative_regret():
    np.testing.assert_allclose(cumulative_regret([1, 0, 1, 1]), [-0.07, 0.86, 0.79, 0.72])


def test_last_k_accuracy():
    assert last_k_accuracy(np.array([0, 0, 1, 1, 0, 1]), k=3) == 2 / 3
    assert np.isnan(last_k_accuracy(np.array([1, 0]), k=100))  # run shorter than k


def test_run():
    policy = AlwaysZero()
    stream = [(np.zeros(2), c) for c in [0, 1, 0, 0, 2]]
    rewards = run(policy, stream)
    assert rewards.dtype == np.int8
    assert rewards.tolist() == [1, 0, 1, 1, 0]
    assert policy.updates == [(0, 1), (0, 0), (0, 1), (0, 1), (0, 0)]
