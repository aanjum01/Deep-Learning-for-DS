"""Run all (algorithm, seed) pairs of a YAML config in parallel: python scripts/run_experiment.py --config configs/X.yaml"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from algorithms.linucb import LinUCB  # noqa: E402
from algorithms.neuralbandit import NeuralBandit1, NeuralBandit2, NeuralBandit3  # noqa: E402
from algorithms.popularity import PopularityPolicy  # noqa: E402
from algorithms.random_policy import RandomPolicy  # noqa: E402
from evaluation.metrics import cumulative_regret, last_k_accuracy  # noqa: E402
from evaluation.runner import run_one  # noqa: E402

CLASSES = {"RandomPolicy": RandomPolicy, "LinUCB": LinUCB, "NeuralBandit1": NeuralBandit1,
           "NeuralBandit2": NeuralBandit2, "NeuralBandit3": NeuralBandit3, "PopularityPolicy": PopularityPolicy}

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
cfg = yaml.safe_load(open(p.parse_args().config))
name, steps, env = cfg["name"], cfg.get("steps"), cfg.get("env", "forest_cover")  # MIND: one pass, no steps
shifts, drift_every = cfg.get("shifts"), cfg.get("drift_every")  # shifts: E4 planted relabelling steps (phase-5d)

raw = ROOT / "results/raw" / name
raw.mkdir(parents=True, exist_ok=True)


def job(a, seed):
    """Run one (algorithm, seed), saving as soon as it finishes; reuse a saved run if present."""
    path = raw / f"{a['label']}_seed{seed}.npz"
    # reuse only if settings match; Forest Cover keeps its old key so saved runs stay valid
    # append only non-default settings, so Forest Cover's and E3's saved runs keep their keys (phase-5d R10)
    extra = ((env,) if env != "forest_cover" else ()) + ((tuple(shifts),) if shifts else ())
    key = repr((a["class"], a["params"], steps, drift_every) + extra)
    if path.exists():
        with np.load(path) as d:
            if str(d["key"]) == key:
                return d["rewards"], d["best"] if "best" in d else None, float(d["seconds"])
    rewards, best, secs, diag = run_one(CLASSES[a["class"]], a["params"], seed, steps, drift_every, env, shifts)
    extra = {} if best is None else {"best": best}
    np.savez(path, rewards=rewards, seconds=secs, key=key, **extra, **diag)  # diag: NB2/NB3 model_probs
    return rewards, best, secs


jobs = [(a["label"], s, a) for a in cfg["algorithms"] for s in cfg["seeds"]]
outs = Parallel(n_jobs=-1)(delayed(job)(a, s) for _, s, a in jobs)

summary, regret = [], []
for (label, seed, _), (rewards, best, secs) in zip(jobs, outs):
    n = len(rewards)
    reg = cumulative_regret(rewards) if best is None else cumulative_regret(rewards, ref=best)  # plan Q17
    summary.append((label, seed, rewards.mean(), last_k_accuracy(rewards), reg[-1], secs, n / secs))
    idx = np.arange(999, n, 1000)  # every 1,000 steps [plan D8]
    regret.append(pd.DataFrame({"label": label, "seed": seed, "step": idx + 1, "cum_regret": reg[idx]}))

# steps/s is measured while all parallel jobs share the CPU: not a standalone per-algorithm speed
summary = pd.DataFrame(summary, columns=["label", "seed", "mean_reward", "last100k_acc", "final_regret",
                                         "seconds", "steps_per_sec_parallel"])
summary.to_csv(ROOT / f"results/{name}_summary.csv", index=False)
pd.concat(regret).to_csv(ROOT / f"results/{name}_regret.csv", index=False)

g = summary.groupby("label", sort=False)
print(pd.DataFrame({
    "mean_reward": g.mean_reward.mean(), "final_regret": g.final_regret.mean(),
    "last100k_acc_mean": g.last100k_acc.mean(), "last100k_acc_std": g.last100k_acc.std(),
    "steps_per_sec_parallel": g.steps_per_sec_parallel.mean(),
}).to_string())
