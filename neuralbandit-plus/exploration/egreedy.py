"""Epsilon-greedy action probabilities of NeuralBandit [allesiardo2014, §3, Alg. 1, p. 4]."""

import numpy as np


def argmax_random_tie(scores, rng):
    """Index of the max score; ties broken uniformly at random (no rng draw when there is no tie)."""
    best = np.flatnonzero(scores == scores.max())
    return int(best[0] if len(best) == 1 else rng.choice(best))


def egreedy(scores, gamma, rng, avail=None):
    """Return (sampled action, k_hat, P) with P(k) = (1 - gamma) [k = k_hat] + gamma / K.

    Sampling: with prob. 1 - gamma play k_hat, else uniform over all K actions (the same
    distribution P, without rng.choice(p=P)).
    avail (bool mask, MIND): k_hat, exploration and P range over the available arms only, K = their
    number; P = 0 elsewhere (plan Q11). avail=None gives exactly the same draws as before.
    """
    arms = np.arange(len(scores)) if avail is None else np.flatnonzero(avail)
    K = len(arms)
    k_hat = int(arms[argmax_random_tie(scores[arms], rng)])
    P = np.zeros(len(scores))
    P[arms] = gamma / K
    P[k_hat] += 1 - gamma
    a = k_hat if rng.random() >= gamma else int(arms[rng.integers(K)])
    return a, k_hat, P
