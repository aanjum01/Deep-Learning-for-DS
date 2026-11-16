"""Run one policy on one stream: Forest Cover [allesiardo2014, §5] or MIND (plan phase 5, Q18)."""

import inspect
import time

import numpy as np
from threadpoolctl import threadpool_limits

from algorithms.popularity import PopularityPolicy
from evaluation.environments import (FOREST_COVER_N_ACTIONS, forest_cover_stream, load_forest_cover,
                                     load_mind, mind_rows, mind_stream)


def run(policy, stream):
    """Play the bandit loop; return int8 rewards (only the chosen arm's reward reaches the policy).

    Stream items: Forest Cover (context, correct_action) -> r = 1[a = correct];
    MIND (context, reward per arm, available arms) -> a chosen among available arms, r = rewards[a].
    """
    rewards = []
    for x, target, *avail in stream:
        if avail:
            a = policy.select(x, avail=avail[0])
            r = int(target[a])
        else:
            a = policy.select(x)
            r = int(a == target)
        policy.update(x, a, r)
        rewards.append(r)
    return np.array(rewards, dtype=np.int8)


def check_finite(policy):
    """Fail loudly if any float array reachable from the policy (lists, objects) is NaN/inf (plan D7)."""
    todo, seen = [(policy.name, policy)], set()
    while todo:
        path, obj = todo.pop()
        if id(obj) in seen:  # guards against reference cycles
            continue
        seen.add(id(obj))
        if isinstance(obj, np.ndarray):
            if obj.dtype.kind == "f" and not np.isfinite(obj).all():
                raise FloatingPointError(f"non-finite values in {path}")
        elif isinstance(obj, (list, tuple)):
            todo += [(f"{path}[{i}]", v) for i, v in enumerate(obj)]
        elif hasattr(obj, "__dict__"):
            todo += [(f"{path}.{k}", v) for k, v in vars(obj).items()]


def run_one(cls, params, seed, steps, drift_every, env="forest_cover"):
    """Worker job: build data/stream/policy, return (rewards, best, seconds, diagnostics).

    best = per-step reward of the best possible choice: Forest Cover uses the paper's constant 0.93
    reference instead (best=None); MIND = best available arm (dynamic regret, plan Q17).
    env "mind_small" / "mind_large": one pass over the seed's 90% of users (steps ignored); n_actions comes
    from the data (context size d is set in the config); the popularity reference line gets the run's own
    click shares (hindsight, Q21).
    """
    with threadpool_limits(1):  # one BLAS thread per parallel job
        if env == "forest_cover":
            X, y = load_forest_cover()
            stream, best = forest_cover_stream(X, y, steps, seed, drift_every), None
            n_actions = FOREST_COVER_N_ACTIONS
        else:
            data = load_mind(env.removeprefix("mind_"))
            rows = mind_rows(data, seed)
            R = data["R"][rows]
            stream, best, n_actions = mind_stream(data, rows), R.max(1), R.shape[1]
            if cls is PopularityPolicy:
                params = {**params, "shares": R.mean(0)}
            if "d" in inspect.signature(cls).parameters and params.get("d") != data["X"].shape[1]:
                raise ValueError(f"{cls.__name__}: set d: {data['X'].shape[1]} in the config (MIND context size)")
        policy = cls(n_actions=n_actions, seed=seed, **params)
        t0 = time.perf_counter()
        rewards = run(policy, stream)
        secs = time.perf_counter() - t0
        check_finite(policy)
        return rewards, best, secs, policy.diagnostics()
