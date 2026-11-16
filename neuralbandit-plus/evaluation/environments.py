"""Streaming environments: Forest Cover [allesiardo2014, §5] and MIND category arms [wu2020mind] (plan phase 5)."""

from pathlib import Path

import numpy as np

FOREST_COVER_N_ACTIONS = 7


def load_forest_cover():
    d = np.load(Path(__file__).resolve().parent.parent / "data/processed/forest_cover.npz")
    return d["X"], d["y"]


def forest_cover_stream(X, y, n_steps, seed, drift_every=None):
    """Yield (context, correct_action) for t = 0..n_steps-1.

    Data is played in a loop from a seeded random start row [§5]. With drift_every, the
    correct class rotates cyclically (+1 every drift_every steps, cumulative, from run start;
    plan decision D1) [§5, p. 6].
    """
    start = np.random.default_rng((seed, 1)).integers(len(X))  # own stream, not the policy's
    for t in range(n_steps):
        i = (start + t) % len(X)
        a = int(y[i])
        if drift_every:
            a = (a + t // drift_every) % FOREST_COVER_N_ACTIONS
        yield X[i].astype(np.float32), a  # float copy: safe math, can't corrupt shared X


def load_mind(size):
    """Arrays of data/processed/mind_{size}.npz (scripts/preprocess_mind.py), loaded into memory."""
    with np.load(Path(__file__).resolve().parent.parent / f"data/processed/mind_{size}.npz") as d:
        return {k: d[k] for k in d.files}


def mind_rows(data, seed, user_frac=0.9):
    """Row indices of one run: a seeded random user_frac of the users, all their rows, in time order (plan Q7)."""
    users = np.unique(data["user"])
    keep = np.random.default_rng((seed, 1)).permutation(users)[:round(user_frac * len(users))]
    return np.flatnonzero(np.isin(data["user"], keep))  # rows are already sorted by time


def mind_stream(data, rows):
    """Yield (context, reward per arm, available arms) for each row, once, in time order (plan Q5-Q11)."""
    X, R, avail = data["X"], data["R"], data["avail"]
    for i in rows:
        yield X[i].astype(np.float32), R[i], avail[i]
