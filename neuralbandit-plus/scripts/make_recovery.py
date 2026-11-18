"""Recovery after planted shifts (plan phase-5d.md §3, §6): python scripts/make_recovery.py {trial|e4|forest}

Pairs each shifted run with its no-shift twin (same algorithm, same seed) from results/raw/:
  trial  : phase5d_trial vs phase5d_trial_noshift (MIND-small, seed 100, shifts 50k/100k/150k, 50k windows)
  e4     : phase5d_mind_e4 vs phase5c_mind_e3 (MIND-large, seeds 0-9, shifts 300k/1M/1.5M, 500k windows)
  forest : phase{2,3,4}_drift vs phase{2,3,4}_stationary (Forest Cover E2, drifts every 500k)
Per algorithm x shift: clicks lost per seed (mean ± std; the deciding number), total shifted-run clicks in the
window, and on the seed-averaged 10k-trailing curves: entered-at, adaptation time (duration, since shift; a lower
bound when recovered = False), max loss. Writes results/{stem}_recovery.csv (+ _verdicts.csv, + _table.csv for
MIND) and results/figures/{fig}_recovery.png (+ {fig}_regret.png for MIND).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evaluation.environments import day_starts, load_mind  # noqa: E402
from evaluation.metrics import adapts_better, clicks_lost, cumulative_regret, recovery, trailing_mean  # noqa: E402

W = 10_000
MIND = ["Random", "Popularity", "LinUCB", "NB1", "NB2", "NB3"]
LEARNERS = ["LinUCB", "NB1", "NB2", "NB3"]
REFERENCE = ("Random", "Popularity")  # dashed lines: not learners
FOREST_PHASE = {"Random": "phase2", "LinUCB": "phase2", "NB1": "phase3", "NB2": "phase4", "NB3": "phase4"}
SETUPS = {  # dirs[label] = (shifted run dir, no-shift twin dir) under results/raw/
    "trial": dict(dirs={l: ("phase5d_trial", "phase5d_trial_noshift") for l in MIND}, seeds=[100],
                  config="phase5d_trial", length=50_000, stem="phase5d_trial", fig="mind_trial",
                  title="MIND-small rehearsal", mind=None),
    "e4": dict(dirs={l: ("phase5d_mind_e4", "phase5c_mind_e3") for l in MIND}, seeds=list(range(10)),
               config="phase5d_mind_e4", length=500_000, stem="phase5d_mind_e4", fig="mind_e4",
               title="MIND E4", mind="large"),
    "forest": dict(dirs={l: (f"{p}_drift", f"{p}_stationary") for l, p in FOREST_PHASE.items()}, seeds=list(range(10)),
                   config="phase4_drift", length=500_000, stem="forest_e2",
                   fig="forest_e2", title="Forest Cover E2"),
}
S = SETUPS[sys.argv[1]]
labels, seeds = list(S["dirs"]), S["seeds"]
cfg = yaml.safe_load(open(ROOT / f"configs/{S['config']}.yaml"))  # schedule from the config (CLAUDE.md: settings in configs/)
shifts = cfg.get("shifts") or list(range(cfg["drift_every"], cfg["steps"], cfg["drift_every"]))

runs = {(l, s): [np.load(ROOT / f"results/raw/{d}/{l}_seed{s}.npz")["rewards"] for d in S["dirs"][l]]
        for l in labels for s in seeds}  # (shifted, no-shift)
bad = [k for k, (a, b) in runs.items() if len(a) != len(b)]
assert not bad, f"shifted run and its no-shift twin differ in length: {bad}"  # same seed -> same rows
end = min(len(a) for a, _ in runs.values())  # cut every seed to the shortest run (plan S6)
bounds = shifts[1:] + [end]
length = [min(S["length"], end - sh) for sh in shifts]

# --- tables ---
rows, ratios, per_seed = [], {}, {}
for l in labels:
    shifted, plain = ([trailing_mean(runs[l, s][i][:end], W) for s in seeds] for i in (0, 1))
    ratios[l] = np.mean(shifted, 0) / np.mean(plain, 0)
    for k, (sh, nxt, n) in enumerate(zip(shifts, bounds, length)):
        lost = [clicks_lost(*runs[l, s], sh, n) for s in seeds]
        total = [int(runs[l, s][0][sh:sh + n].sum()) for s in seeds]
        per_seed[l, k] = lost, total
        rec = recovery(ratios[l], sh, nxt, W)
        rows.append({"algorithm": l, "shift": k + 1, "window": f"[{sh}, {sh + n})",
                     "clicks_lost_mean": np.mean(lost), "clicks_lost_std": np.std(lost, ddof=1) if len(lost) > 1 else np.nan,
                     "shifted_clicks_mean": np.mean(total), "click_rate": np.mean(total) / n,
                     **{key: rec[key] for key in ("entered_at", "max_loss", "recovered", "duration", "since_shift")}})
table = pd.DataFrame(rows)  # duration / since_shift are lower bounds where recovered is False
table.to_csv(ROOT / f"results/{S['stem']}_recovery.csv", index=False)
with pd.option_context("display.width", 200, "display.max_columns", 20):
    print(table.round(3).to_string(index=False))

if len(seeds) > 1:  # pre-registered pairwise verdicts among learners (plan R2, S2, S3)
    learners = [l for l in LEARNERS if l in labels]
    pd.DataFrame([{"shift": k + 1, "A": a, "B": b, "verdict": adapts_better(per_seed[a, k][0], per_seed[b, k][0],
                                                                         per_seed[a, k][1], per_seed[b, k][1])}
                  for k in range(len(shifts)) for a in learners for b in learners if a != b]
                 ).to_csv(ROOT / f"results/{S['stem']}_verdicts.csv", index=False)

if "mind" in S:  # MIND: overall results table (regret = steps - clicks, plan Q17)
    pd.DataFrame([{"algorithm": l, "click_rate": np.mean([runs[l, s][0][:end].mean() for s in seeds]),
                   "final_regret": np.mean([cumulative_regret(runs[l, s][0][:end], 1.0)[-1] for s in seeds])}
                  for l in labels]).to_csv(ROOT / f"results/{S['stem']}_table.csv", index=False)

# --- figures ---
fig, axes = plt.subplots(1, len(shifts), figsize=(4.2 * len(shifts), 3.6), sharey=True)
for ax, sh, nxt, k in zip(axes, shifts, bounds, range(len(shifts))):
    lo = max(sh - S["length"] // 10, W)
    for l in labels:
        ax.plot(np.arange(lo, nxt), ratios[l][lo:nxt], label=l, lw=1, ls="--" if l in REFERENCE else "-")
    ax.axhline(0.95, color="k", lw=0.8, ls=":")
    ax.axvline(sh, color="grey", lw=1)
    ax.set(title=f"shift {k + 1} at {sh:,}", xlabel="step")
axes[0].set_ylabel("shifted / no-shift click rate")
axes[-1].legend(fontsize=7)
fig.suptitle(f"{S['title']}: recovery after each shift (dotted = 95%)")
fig.tight_layout()
(ROOT / "results/figures").mkdir(parents=True, exist_ok=True)
fig.savefig(ROOT / f"results/figures/{S['fig']}_recovery.png", dpi=150)

if "mind" in S:  # cumulative regret (= steps - clicks), planted shifts solid, day boundaries dotted
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for l in labels:
        ax.plot(np.mean([cumulative_regret(runs[l, s][0][:end], 1.0) for s in seeds], 0), label=l,
                ls="--" if l in REFERENCE else "-")
    for sh in shifts:
        ax.axvline(sh, color="k", lw=1)
    if S["mind"]:
        for x in day_starts(load_mind(S["mind"]), seeds):
            ax.axvline(x, color="grey", lw=0.6, ls=":")
    ax.set(xlabel="time step (solid: planted shifts; dotted: day boundaries)", ylabel="cumulative regret",
           title=f"{S['title']}: cumulative regret = steps − clicks")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / f"results/figures/{S['fig']}_regret.png", dpi=150)
