"""Run all (algorithm, seed) pairs of a YAML config in parallel: python scripts/run_experiment.py --config configs/X.yaml"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from algorithms.linucb import LinUCB  # noqa: E402
from algorithms.neuralbandit import NeuralBandit1, NeuralBandit1Ctrl, NeuralBandit1Fade, NeuralBandit1SW, NeuralBandit2, NeuralBandit3  # noqa: E402
from algorithms.neuralucb import Diverged, NeuralUCB  # noqa: E402
from algorithms.popularity import PopularityPolicy  # noqa: E402
from algorithms.random_policy import RandomPolicy  # noqa: E402
from evaluation.metrics import cumulative_regret, last_k_accuracy  # noqa: E402
from evaluation.runner import run_one  # noqa: E402

CLASSES = {"RandomPolicy": RandomPolicy, "LinUCB": LinUCB, "NeuralBandit1": NeuralBandit1,
           "NeuralBandit2": NeuralBandit2, "NeuralBandit3": NeuralBandit3, "PopularityPolicy": PopularityPolicy,
           "NeuralUCB": NeuralUCB, "NeuralBandit1SW": NeuralBandit1SW,
           "NeuralBandit1Fade": NeuralBandit1Fade, "NeuralBandit1Ctrl": NeuralBandit1Ctrl}

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--jobs", type=int, default=-1)  # fewer parallel jobs when memory is tight (phase-6b)
args = p.parse_args()
cfg = yaml.safe_load(open(args.config))
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
                return d["rewards"], d.get("best"), float(d["seconds"]), bool(d.get("diverged", False))
    t0 = time.perf_counter()
    try:
        rewards, best, secs, diag = run_one(CLASSES[a["class"]], a["params"], seed, steps, drift_every, env, shifts)
    except Diverged as e:  # weights went non-finite: keep the run as "diverged", no rewards (phase-6a R2)
        rewards, best, secs, diag = np.zeros(0, np.int8), None, time.perf_counter() - t0, {"diverged": True,
                                                                                            "diverged_step": e.step}
    extra = {} if best is None else {"best": best}
    np.savez(path, rewards=rewards, seconds=secs, key=key, **extra, **diag)  # diag: NB2/NB3 model_probs, NeuralUCB traces
    return rewards, best, secs, "diverged" in diag


jobs = [(a["label"], s, a) for a in cfg["algorithms"] for s in cfg["seeds"]]
outs = Parallel(n_jobs=args.jobs)(delayed(job)(a, s) for _, s, a in jobs)

summary, regret = [], []
for (label, seed, _), (rewards, best, secs, diverged) in zip(jobs, outs):
    if diverged:  # one NaN row, no regret rows
        summary.append((label, seed, np.nan, np.nan, np.nan, secs, np.nan, True))
        continue
    n = len(rewards)
    reg = cumulative_regret(rewards) if best is None else cumulative_regret(rewards, ref=best)  # plan Q17
    summary.append((label, seed, rewards.mean(), last_k_accuracy(rewards), reg[-1], secs, n / secs, False))
    idx = np.arange(999, n, 1000)  # every 1,000 steps [plan D8]
    regret.append(pd.DataFrame({"label": label, "seed": seed, "step": idx + 1, "cum_regret": reg[idx]}))

# steps/s is measured while all parallel jobs share the CPU: not a standalone per-algorithm speed
summary = pd.DataFrame(summary, columns=["label", "seed", "mean_reward", "last100k_acc", "final_regret",
                                         "seconds", "steps_per_sec_parallel", "diverged"])
if not summary.diverged.any():  # column only when something diverged: earlier phases' CSVs keep their shape
    summary = summary.drop(columns="diverged")
summary.to_csv(ROOT / f"results/{name}_summary.csv", index=False)
pd.concat(regret or [pd.DataFrame(columns=["label", "seed", "step", "cum_regret"])]).to_csv(
    ROOT / f"results/{name}_regret.csv", index=False)  # empty if every run diverged

g = summary.groupby("label", sort=False)
print(pd.DataFrame({
    "mean_reward": g.mean_reward.mean(), "final_regret": g.final_regret.mean(),
    "last100k_acc_mean": g.last100k_acc.mean(), "last100k_acc_std": g.last100k_acc.std(),
    "steps_per_sec_parallel": g.steps_per_sec_parallel.mean(),
}).to_string())
