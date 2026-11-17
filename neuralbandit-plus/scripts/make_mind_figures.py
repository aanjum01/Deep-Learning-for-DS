"""MIND E3 outputs (plan phase-5c.md Q8): regret figure, results table, per-day click rates, data drift chart.

Reads results/phase5c_mind_e3_{summary,regret}.csv, the per-step rewards in results/raw/phase5c_mind_e3/ and
data/processed/mind_large.npz. Aggregates only (MIND license: no data rows are written).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evaluation.environments import load_mind, mind_rows  # noqa: E402

NAME, FIG = "phase5c_mind_e3", ROOT / "results/figures"
FIG.mkdir(parents=True, exist_ok=True)
data = load_mind("large")
day = data["time"].astype("datetime64[D]")
days = np.unique(day)

# (4) Natural drift in the data: share of impressions with a click in each category, per day (top 8)
click_share = pd.DataFrame(data["R"], columns=data["arms"]).groupby(day).mean()
top = click_share.mean().sort_values(ascending=False).index[:8]
ax = click_share[top].plot(figsize=(8, 4), marker="o")
ax.set(xlabel="day (Nov 2019)", ylabel="share of impressions with a click", title="MIND: daily category clicks")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(FIG / "mind_daily_categories.png", dpi=150)
plt.close()

summary_csv = ROOT / f"results/{NAME}_summary.csv"
if not summary_csv.exists():
    sys.exit("no E3 results yet — only mind_daily_categories.png written")
summary = pd.read_csv(summary_csv)
labels = list(dict.fromkeys(summary.label))
seeds = sorted(summary.seed.unique())

# (2) Results table: click rate and final regret, mean ± std over seeds
g = summary.groupby("label", sort=False)
table = pd.DataFrame({"click_rate_mean": g.mean_reward.mean(), "click_rate_std": g.mean_reward.std(),
                      "final_regret_mean": g.final_regret.mean(), "final_regret_std": g.final_regret.std()})
table.to_csv(ROOT / f"results/{NAME}_table.csv")
print(table.round(4).to_string())

# (1) Cumulative regret vs step, mean ± std, day boundaries = mean day-start step over seeds (plan Q8)
rows = {s: mind_rows(data, s) for s in seeds}
starts = np.mean([np.flatnonzero(np.diff(day[r].astype(np.int64))) + 1 for r in rows.values()], axis=0)
fig, ax = plt.subplots(figsize=(8, 4.5))
for label, d in pd.read_csv(ROOT / f"results/{NAME}_regret.csv").groupby("label", sort=False):
    m, s = d.groupby("step").cum_regret.mean(), d.groupby("step").cum_regret.std()
    ax.plot(m.index, m, label=label, ls="--" if label in ("Random", "Popularity") else "-")
    ax.fill_between(m.index, m - s, m + s, alpha=0.2)
for x in starts:
    ax.axvline(x, color="grey", lw=0.6, ls=":")
ax.set(xlabel="time step (dotted lines: day boundaries, Nov 9–15)", ylabel="cumulative regret",
       title="MIND E3 (real week): cumulative regret = steps − clicks")
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "mind_e3_regret.png", dpi=150)
plt.close()

# (3) Click rate per day: each seed's rewards mapped back to its rows' days, then mean over seeds
daily = {label: np.mean([pd.Series(np.load(ROOT / f"results/raw/{NAME}/{label}_seed{s}.npz")["rewards"])
                         .groupby(day[rows[s]]).mean().to_numpy() for s in seeds], axis=0) for label in labels}
daily = pd.DataFrame(daily, index=days)
daily.to_csv(ROOT / f"results/{NAME}_daily.csv")
ax = daily.plot(figsize=(8, 4), marker="o")
ax.set(xlabel="day (Nov 2019)", ylabel="click rate (mean over seeds)", title="MIND E3: click rate per day")
plt.tight_layout()
plt.savefig(FIG / "mind_e3_daily.png", dpi=150)
plt.close()
print(daily.round(4).to_string())
