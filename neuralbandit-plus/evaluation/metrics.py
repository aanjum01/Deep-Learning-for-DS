"""Metrics for the Forest Cover experiments [allesiardo2014, §5]."""

import numpy as np


def cumulative_regret(rewards, ref=0.93):
    """Cumulative regret vs. a reference reward per step [§5, p. 5].

    ref: the paper's constant 0.93 (Forest Cover), or an array of each step's best available reward
    (MIND dynamic regret, plan Q17).
    """
    return np.cumsum(ref - np.asarray(rewards))


def last_k_accuracy(rewards, k=100_000):
    """Mean reward over the last k steps [§5, p. 6]; NaN if the run is shorter than k."""
    return float(np.mean(rewards[-k:])) if len(rewards) >= k else float("nan")
