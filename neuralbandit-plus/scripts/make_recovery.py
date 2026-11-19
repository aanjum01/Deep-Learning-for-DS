"""Recovery after planted shifts (plan phase-5d.md §3, §6): python scripts/make_recovery.py {trial|e4|forest|trial_6a|e4_6a|forest_6a|trial_6b|e4_6b|e4_6b_off|trial_6b2|e4_6b2|trial_6c|e4_6c}

Pairs each shifted run with its no-shift twin (same algorithm, same seed) from results/raw/:
  trial  : phase5d_trial vs phase5d_trial_noshift (MIND-small, seed 100, shifts 50k/100k/150k, 50k windows)
  e4     : phase5d_mind_e4 vs phase5c_mind_e3 (MIND-large, seeds 0-9, shifts 300k/1M/1.5M, 500k windows)
  forest : phase{2,3,4}_drift vs phase{2,3,4}_stationary (Forest Cover E2, drifts every 500k)
Phase 6a setups add NeuralUCB (plan phase-6a.md section 4): e4_6a = e4 + NeuralUCB (phase6a_mind_e4 vs phase6a_mind_e3),
forest_6a = forest + NeuralUCB (phase6a_drift vs phase6a_stationary), trial_6a = NeuralUCB only (phase6a_trial vs
phase6a_trial_noshift). Phase 6b setups add NB1-SW (plan phase-6b.md): e4_6b = e4 + the 7 NB1-SW window sizes
(phase6b_mind_e4 vs phase6b_mind_e3), trial_6b = NB1-SW_W20k only (phase6b_trial vs phase6b_trial_noshift). Phase 6b2 setups add NB1-Fade (plan phase-6b2.md): e4_6b2 = e4 + the 7 windows +
NB1-SW_W493k_off + the 8 NB1-Fade rates (phase6b2_mind_e4 vs phase6b2_mind_e3), trial_6b2 = NB1-Fade_20k only
(phase6b2_trial vs phase6b2_trial_noshift). Phase 6c (plan phase-6c.md): e4_6c = e4 + slow fade 1/987k + NB1-Ctrl_A/B
(phase6c_mind_e4 vs phase6c_mind_e3), trial_6c = fade 1/100k + NB1-Ctrl_A/B (phase6c_trial vs phase6c_trial_noshift); e4_6b2 also writes {fig}_fade_vs_windows.png, results/phase6b2_tradeoff.csv and
results/phase6b2_decision.csv (the pre-registered trade-off rule). An algorithm with any diverged run is skipped with a printed note. MIND setups with NeuralUCB also
write results/figures/{fig}_exploration.png (NeuralUCB's exploration rate and bonus, shifted vs no-shift twin).
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
from evaluation.metrics import (adapts_better, clicks_lost, cumulative_regret, recovery, beaten_windows, two_se,  # noqa: E402
                                trailing_mean, trap_blocks)

W = 10_000
MIND = ["Random", "Popularity", "LinUCB", "NB1", "NB2", "NB3"]
SW = [f"NB1-SW_W{w}" for w in ("10k", "20k", "50k", "99k", "197k", "493k", "987k")]
OFF = "NB1-SW_W493k_off"  # reset-timing check (RESEARCH_LOG §6)
FADE = [f"NB1-Fade_{r}" for r in ("10k", "20k", "50k", "100k", "99k", "197k", "493k", "987k")]  # Phase 6b2
CTRL = ["NB1-Ctrl_A", "NB1-Ctrl_B"]  # Phase 6c: slow fade + PH alarm + rule (plan phase-6c.md)
LEARNERS = ["LinUCB", "NB1", "NB2", "NB3", "NeuralUCB", *SW, OFF, *FADE, *CTRL]
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
NU = "NeuralUCB"
SETUPS["e4_6a"] = {**SETUPS["e4"], "dirs": {**SETUPS["e4"]["dirs"], NU: ("phase6a_mind_e4", "phase6a_mind_e3")},
                   "config": "phase6a_mind_e4", "stem": "phase6a_mind_e4"}
SETUPS["forest_6a"] = {**SETUPS["forest"], "dirs": {**SETUPS["forest"]["dirs"], NU: ("phase6a_drift", "phase6a_stationary")},
                       "config": "phase6a_drift", "stem": "phase6a_forest_e2"}
SETUPS["trial_6a"] = {**SETUPS["trial"], "dirs": {NU: ("phase6a_trial", "phase6a_trial_noshift")},
                      "config": "phase6a_trial", "stem": "phase6a_trial", "fig": "mind_trial_6a"}
SETUPS["e4_6b"] = {**SETUPS["e4"], "dirs": {**SETUPS["e4"]["dirs"], **{l: ("phase6b_mind_e4", "phase6b_mind_e3") for l in SW}},
                   "config": "phase6b_mind_e4", "stem": "phase6b_mind_e4", "fig": "mind_e4_6b"}
SETUPS["e4_6b_off"] = {**SETUPS["e4"], "dirs": {**SETUPS["e4"]["dirs"], "NB1-SW_W493k": ("phase6b_mind_e4", "phase6b_mind_e3"),
                                                OFF: ("phase6b_offset_mind_e4", "phase6b_offset_mind_e3")},
                       "config": "phase6b_mind_e4", "stem": "phase6b_offset_mind_e4", "fig": "mind_e4_6b_offset"}
SETUPS["trial_6b"] = {**SETUPS["trial"], "dirs": {"NB1-SW_W20k": ("phase6b_trial", "phase6b_trial_noshift")},
                      "config": "phase6b_trial", "stem": "phase6b_trial", "fig": "mind_trial_6b"}
SETUPS["e4_6b2"] = {**SETUPS["e4"], "dirs": {**SETUPS["e4_6b"]["dirs"], "NB1-SW_W493k_off": ("phase6b_offset_mind_e4", "phase6b_offset_mind_e3"),
                                             **{l: ("phase6b2_mind_e4", "phase6b2_mind_e3") for l in FADE}},
                    "config": "phase6b2_mind_e4", "stem": "phase6b2_mind_e4", "fig": "mind_e4_6b2"}
SETUPS["trial_6b2"] = {**SETUPS["trial"], "dirs": {"NB1-Fade_20k": ("phase6b2_trial", "phase6b2_trial_noshift")},
                       "config": "phase6b2_trial", "stem": "phase6b2_trial", "fig": "mind_trial_6b2"}
SETUPS["e4_6c"] = {**SETUPS["e4"], "dirs": {**SETUPS["e4"]["dirs"], "NB1-Fade_987k": ("phase6b2_mind_e4", "phase6b2_mind_e3"),
                                            **{l: ("phase6c_mind_e4", "phase6c_mind_e3") for l in CTRL}},
                   "config": "phase6c_mind_e4", "stem": "phase6c_mind_e4", "fig": "mind_e4_6c"}
SETUPS["trial_6c"] = {**SETUPS["trial"], "dirs": {l: ("phase6c_trial", "phase6c_trial_noshift") for l in ["NB1-Fade_100k", *CTRL]},
                      "config": "phase6c_trial", "stem": "phase6c_trial", "fig": "mind_trial_6c"}
S = SETUPS[sys.argv[1]]
labels, seeds = list(S["dirs"]), S["seeds"]
cfg = yaml.safe_load(open(ROOT / f"configs/{S['config']}.yaml"))  # schedule from the config (CLAUDE.md: settings in configs/)
shifts = cfg.get("shifts") or list(range(cfg["drift_every"], cfg["steps"], cfg["drift_every"]))

runs = {(l, s): [np.load(ROOT / f"results/raw/{d}/{l}_seed{s}.npz")["rewards"] for d in S["dirs"][l]]
        for l in labels for s in seeds}  # (shifted, no-shift)
dropped = [l for l in labels if any(not len(r) for s in seeds for r in runs[l, s])]  # diverged runs have no rewards
if dropped:
    print(f"skipped (a run diverged): {dropped}")
labels = [l for l in labels if l not in dropped]
runs = {k: v for k, v in runs.items() if k[0] in labels}
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

windows = [l for l in SW if l in labels]
if "NB1" in labels and len(windows) > 1:  # one row per window size vs NB1 alone: the all-lines figure is unreadable
    fig, axes = plt.subplots(len(windows), len(shifts), figsize=(4.2 * len(shifts), 1.7 * len(windows)),
                             sharex="col", sharey=True, squeeze=False)
    for row, l in zip(axes, windows):
        for ax, sh, nxt in zip(row, shifts, bounds):
            lo = max(sh - S["length"] // 10, W)
            x = np.arange(lo, nxt)
            ax.plot(x, ratios["NB1"][lo:nxt], color="k", lw=1, label="NB1")
            ax.plot(x, ratios[l][lo:nxt], color="tab:orange", lw=1, label=l)
            ax.axhline(0.95, color="k", lw=0.8, ls=":")
            ax.axvline(sh, color="grey", lw=1)
        row[0].set_ylabel(l.replace("NB1-SW_", ""), rotation=0, ha="right", va="center")
    for ax, k, sh in zip(axes[0], range(len(shifts)), shifts):
        ax.set_title(f"shift {k + 1} at {sh:,}")
    for ax in axes[-1]:
        ax.set_xlabel("step")
    axes[0, -1].legend(["NB1", "window"], fontsize=7, loc="lower right")
    fig.suptitle(f"{S['title']}: shifted / no-shift click rate, NB1 (black) vs each window size (orange); dotted = 95%")
    fig.tight_layout()
    fig.savefig(ROOT / f"results/figures/{S['fig']}_nb1_vs_windows.png", dpi=150)

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

if NU in labels and "mind" in S:  # NeuralUCB's own traces (per 1,000 steps), shifted run vs no-shift twin, mean over seeds
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key in zip(axes, ("explore_rate", "bonus")):
        for d, name in zip(S["dirs"][NU], ("shifted", "no shift")):
            tr = np.mean([np.load(ROOT / f"results/raw/{d}/{NU}_seed{s}.npz")[key] for s in seeds], 0)
            ax.plot(np.arange(1, len(tr) + 1) * 1000, tr, label=name, lw=1)
        for sh in shifts:
            ax.axvline(sh, color="grey", lw=1)
        ax.set(title=key.replace("_", " "), xlabel="step (per-1,000-step means; grey: planted shifts)")
    axes[0].legend()
    fig.suptitle(f"{S['title']}: NeuralUCB exploration")
    fig.tight_layout()
    fig.savefig(ROOT / f"results/figures/{S['fig']}_exploration.png", dpi=150)

PAIRS = [("Fade_10k", "W10k"), ("Fade_20k", "W20k"), ("Fade_50k", "W50k"), ("Fade_100k", "W99k"), ("Fade_99k", "W99k"),
         ("Fade_197k", "W197k"), ("Fade_493k", "W493k"), ("Fade_987k", "W987k")]  # matched memory (plan Q8)
if sys.argv[1] == "e4_6b2":  # matched fade vs window rows: NB1 black, window orange, fade blue
    fig, axes = plt.subplots(len(PAIRS), len(shifts), figsize=(4.2 * len(shifts), 1.7 * len(PAIRS)),
                             sharex="col", sharey=True, squeeze=False)
    for row, (f, w) in zip(axes, PAIRS):
        for ax, sh, nxt in zip(row, shifts, bounds):
            lo = max(sh - S["length"] // 10, W)
            x = np.arange(lo, nxt)
            ax.plot(x, ratios["NB1"][lo:nxt], color="k", lw=1)
            ax.plot(x, ratios[f"NB1-SW_{w}"][lo:nxt], color="tab:orange", lw=1)
            ax.plot(x, ratios[f"NB1-{f}"][lo:nxt], color="tab:blue", lw=1)
            ax.axhline(0.95, color="k", lw=0.8, ls=":")
            ax.axvline(sh, color="grey", lw=1)
        row[0].set_ylabel(f"{f}\nvs {w}", rotation=0, ha="right", va="center")
    for ax, k, sh in zip(axes[0], range(len(shifts)), shifts):
        ax.set_title(f"shift {k + 1} at {sh:,}")
    for ax in axes[-1]:
        ax.set_xlabel("step")
    axes[0, -1].legend(["NB1", "window", "fade"], fontsize=7, loc="lower right")
    fig.suptitle(f"{S['title']}: shifted / no-shift click rate, NB1 (black), window (orange), fade (blue); dotted = 95%")
    fig.tight_layout()
    fig.savefig(ROOT / f"results/figures/{S['fig']}_fade_vs_windows.png", dpi=150)

    # trade-off rule (plan phase-6b2, second review): points = E3 click rate and mean clicks lost per shift, per seed
    pts = {l: (np.array([runs[l, s][1][:end].mean() for s in seeds]),
               np.array([np.mean([per_seed[l, k][0][i] for k in range(len(shifts))]) for i in range(len(seeds))]))
           for l in labels if l == "NB1" or l in SW or l == OFF or l in FADE}
    pd.DataFrame([{"setting": l, "e3_click_rate": pts[l][0].mean(), "e3_click_rate_2se": two_se(pts[l][0]),
                   "clicks_lost_per_shift": pts[l][1].mean(), "clicks_lost_per_shift_2se": two_se(pts[l][1]),
                   "trap_blocks_e3": sum(trap_blocks(runs[l, s][1][:end]) for s in seeds),
                   "trap_blocks_e4": sum(trap_blocks(runs[l, s][0][:end], skip=shifts) for s in seeds)}
                  for l in pts]).to_csv(ROOT / "results/phase6b2_tradeoff.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))  # the trade-off picture behind the decision rule
    for group, colour in ((["NB1"], "k"), ([l for l in pts if l in SW or l == OFF], "tab:orange"), (FADE, "tab:blue")):
        for l in group:
            (r, c) = pts[l]
            ax.errorbar(c.mean(), 100 * r.mean(), xerr=two_se(c), yerr=100 * two_se(r), fmt="o", color=colour, ms=4)
            ax.annotate(l.replace("NB1-SW_", "").replace("NB1-Fade_", "fade "), (c.mean(), 100 * r.mean()), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
    ax.set(xlabel="clicks lost per planted shift (mean of 3; lower = better)", ylabel="E3 click rate % (higher = better)",
           title="MIND forgetting trade-off: windows (orange), fades (blue), NB1 (black); bars 2×SE")
    fig.tight_layout()
    fig.savefig(ROOT / "results/figures/mind_6b2_tradeoff.png", dpi=150)

    targets = {"W10k": ["NB1-SW_W10k"], "W20k": ["NB1-SW_W20k"], "W50k": ["NB1-SW_W50k"], "W99k": ["NB1-SW_W99k"],
               "W197k": ["NB1-SW_W197k"], "W493k": ["NB1-SW_W493k", OFF], "W987k": ["NB1-SW_W987k"]}  # 493k: both schedules
    rows = []
    for who, cands in (("fades", FADE), ("NB1 (placebo)", ["NB1"])):
        for name, by in beaten_windows(pts, cands, targets).items():
            rows.append({"candidates": who, "window": name, "beaten": bool(by), "by": " ".join(by),
                         "decides": name in ("W10k", "W20k", "W50k")})
    res = pd.DataFrame(rows)
    n_fade, n_nb1 = (int(res[(res.candidates == who) & res.decides].beaten.sum()) for who in ("fades", "NB1 (placebo)"))
    res = pd.concat([res, pd.DataFrame([
        {"candidates": "fades", "window": "small windows beaten", "beaten": n_fade, "decides": True},
        {"candidates": "NB1 (placebo)", "window": "small windows beaten", "beaten": n_nb1, "decides": True},
        {"candidates": "decision", "window": "fade rate if >= 2 of 3 small windows beaten else window",
         "beaten": "fade rate" if n_fade >= 2 else "window", "decides": True},
        {"candidates": "placebo_ok", "window": "NB1 beats < 2 small windows", "beaten": n_nb1 < 2, "decides": True}])])
    res.to_csv(ROOT / "results/phase6b2_decision.csv", index=False)
    print(res.to_string(index=False))
