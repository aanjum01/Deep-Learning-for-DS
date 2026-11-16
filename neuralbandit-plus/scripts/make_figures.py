"""Figure 1-style plot: cumulative regret vs. time, stationary (left) and drift (right) [allesiardo2014, Fig. 1].

Combines every results/*_stationary_regret.csv and results/*_drift_regret.csv, so later phases' runs
(e.g. NB1/2/3) land on the same figure as Random and LinUCB.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, kind, title in zip(axes, ["stationary", "drift"], ["Stationary", "Drift"]):
    files = sorted((ROOT / "results").glob(f"*_{kind}_regret.csv"))
    for f in files:
        for label, d in pd.read_csv(f).groupby("label", sort=False):
            g = d.groupby("step").cum_regret
            m, s = g.mean(), g.std()
            ax.plot(m.index, m, label=label)
            ax.fill_between(m.index, m - s, m + s, alpha=0.2)
    ax.set(title=title if files else f"{title} (no results)", xlabel="time step")
    if files:
        ax.legend()
axes[0].set_ylabel("cumulative regret")
(ROOT / "results/figures").mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(ROOT / "results/figures/figure1.png", dpi=150)
