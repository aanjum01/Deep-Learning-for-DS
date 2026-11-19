"""Toy bandit stream shared by the window and forgetting tests (K = 4 arms, D = 6 binary features)."""

import numpy as np

K, D = 4, 6


def stream(n, seed=0, mask=True):
    r = np.random.default_rng(seed)
    for _ in range(n):
        x = (r.random(D) < 0.5).astype(np.float32)
        av = r.random(K) < 0.7 if mask else None
        if av is not None and not av.any():
            av[0] = True
        yield x, av, r.random(K) < np.linspace(0.2, 0.7, K)  # arm k clicks with prob 0.2..0.7


def play(policy, n, seed=0, mask=True):
    """Run the loop; return (actions, rewards). Unavailable arms must never be picked."""
    acts, rew = [], []
    for x, av, click in stream(n, seed, mask):
        a = policy.select(x, avail=av)
        assert av is None or av[a]
        policy.update(x, a, int(click[a]))
        acts.append(a)
        rew.append(int(click[a]))
    return acts, rew
