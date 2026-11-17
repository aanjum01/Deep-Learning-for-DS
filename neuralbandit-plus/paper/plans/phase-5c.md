# Phase 5c — Baselines on MIND, E3 (the real week)

**Status: DONE (2026-11-16).** NB1 tuned (C = 1, λ = 1, user-confirmed at the checkpoint); 60 E3 runs in
2 h 29 min; done-check §7 passed (all finite; Random − analytic ≤ 0.0003 every seed; 4 outputs written).
Results + findings (incl. the EXP3-weights answer): RESEARCH_LOG §4. Next: Phase 5d (E4).
**Order (user, Nov 16):** 5c (E3 baselines) → 5d (design E4 planted shifts + run baselines on it), both
before Phase 6's results (planned in another session), so E4 is fixed before the method is seen.

## 1. Goal and why

Run every required baseline on MIND-large, experiment E3 (the real week, chronological), so the new
method (Phase 6) only has to be added next to finished baseline results.

| Why | Source |
|---|---|
| Every baseline runs in every environment the method runs in | Course slides *Baselines*; scope decision Nov 13 |
| Baselines = NB1, NB2, NB3 + LinUCB (+ Random as sanity) | Scope decision Nov 13 |
| E3 = natural interest drift; worth more than expected (strong daily swings) | Phase 5 Q24 + data review (user, Nov 16) |
| Tune on MIND-small, report on MIND-large minus small's users | Phase 5 Q22 |

## 2. What is already fixed (Phase 5, no new decision needed)

Data and environment (`mind_large.npz`, 2,270,815 rows, 16 arms, d = 19, masking, one chronological pass,
each seed keeps 90% of users ≈ 2.04M steps); metric = cumulative dynamic regret (= steps − reward on MIND)
+ click rate (Q17); LinUCB α = 0.5 (tuned on MIND-small, Phase 5b); Forest Cover untouched.

## 3. Measured cost (one job alone, MIND-small, 20k steps, Nov 16)

| | µs/step | MIND-small pass (207k) | MIND-large pass (2.04M) |
|---|---|---|---|
| LinUCB | 29 | 0.1 min | 1.0 min |
| NB1 C=100 | 67 | 0.2 min | 2.3 min |
| NB2 | 408 | 1.4 min | 14 min |
| NB3 | 492 | 1.7 min | 17 min |

With 8 parallel jobs, Phase 4 ran ≈ 4× slower per job than alone → 10 seeds of all baselines ≈ 3 h.

## 4. Plan (all decided, §8)

| Step | What | Q |
|---|---|---|
| 1. Tune NB1 on MIND-small | paper grid C ∈ {1,5,25,50,100} × λ ∈ {0.01,0.1,1}, seeds 100–102, one pass, best total reward | Q1 |
| 2. **Checkpoint** | show the user the tuning table + winner before the big runs | Q10 |
| 3. E3 runs on MIND-large | Random, Popularity (reference), LinUCB α = 0.5, NB1 (tuned), NB2, NB3; seeds 0–9; ≈ 3 h background | Q2–Q7, Q10 |
| 4. Outputs | regret figure, results table, per-day chart, data drift chart, RESEARCH_LOG | Q8 |
| 5. Done-check, report | §7; mean ± std (tests later) | Q9, Q11 |

## 5. Files

| File | What it does |
|---|---|
| `configs/phase5c_tune_nb1.yaml` | env mind_small, seeds 100–102, the 15 NB1 settings, γ = 0.005, `d: 19` |
| `configs/phase5c_mind_e3.yaml` | env mind_large, seeds 0–9: Random, Popularity, LinUCB (α 0.5, d 19), NB1 (tuned C, λ; γ 0.005; d 19), NB2/NB3 (paper grid, γ 0.005, γ_model 0.1, d 19) |
| `scripts/make_mind_figures.py` | reads `results/phase5c_mind_e3_*` + `results/raw/phase5c_mind_e3/` + `mind_large.npz`; writes the 4 Q8 outputs to `results/figures/` and `results/phase5c_mind_e3_table.csv` |

No algorithm / environment code changes. File names don't match `*_stationary`/`*_drift`, so Forest
Cover's Figure 1 is unaffected (checked in `make_figures.py`). Machine: 8 cores (checked).

## 6. Settings and sources

| Setting | Value | Source |
|---|---|---|
| NB1 grid | C ∈ {1,5,25,50,100} × λ ∈ {0.01,0.1,1} | paper §5 p. 5; Q1 |
| NB1 selection | highest mean total reward over seeds 100–102 (near-ties shown at the checkpoint) | Q1, Phase 5 Q23, Q10 |
| γ, γ_model | 0.005, 0.1 | paper §5 p. 5; Q2 |
| NB2/NB3 | all 15 models | paper §4–5; Q3 |
| LinUCB α | 0.5 | Phase 5b tuning (RESEARCH_LOG) |
| d | 19 | Phase 5 Q13/Q16 (runner enforces) |
| Seeds | 0–9 reported; 100–102 tuning | Q6; Phase 2/3 convention |
| Data | MIND-large minus small's users, 90% users per seed, one chronological pass | Phase 5 Q5–Q8, Q22 |
| Metric | dynamic regret (= steps − reward) + click rate, mean ± std | Phase 5 Q17; Q11 |

## 7. Done-check (Q9)

1. NB1 tuning ran; winner logged and shown to the user (Q10).
2. All 60 E3 runs (6 algorithms × 10 seeds) complete; no NaN/inf (runner `check_finite`).
3. Random's mean reward = its analytic value (mean of clicked-shown ÷ shown over the seed's rows) ± 0.005,
   every seed.
4. The four Q8 outputs exist; results + findings in RESEARCH_LOG.
Findings (not pass/fail): NB1/2/3 vs LinUCB; vs popularity; NB3 lock-on; behaviour on spike days;
**NB2's final EXP3 model weights on the 2.04M-step runs** — do large networks win once runs are long
(→ NB1's C = 1 was a short-run tuning effect) or do small ones still win (→ MIND's category context is
mostly linear, so Phase 6 should frame MIND gains around adaptation, not non-linearity)? (Nov 16 note,
RESEARCH_LOG.)

## 8. Open questions

- **Q1 — DECIDED (user, Nov 16): re-tune NB1 on MIND-small**: paper grid C ∈ {1,5,25,50,100} × λ ∈
  {0.01,0.1,1} [allesiardo2014, §5, p. 5], tuning seeds 100–102, one pass each, pick by total reward
  (same rule as LinUCB's α, Phase 5 Q23). Why: Phase 5 Q22 (MIND settings chosen on small); fair vs. the
  re-tuned LinUCB.
- **Q2 — DECIDED (user, Nov 16): keep the paper's γ = 0.005, γ_model = 0.1** [allesiardo2014, §5, p. 5],
  as on Forest Cover. Why: NB stays the published algorithm; the abstract's gap is NB's *fixed* exploration;
  tuning covers only what the paper leaves open (NB1's C, λ; LinUCB's α). Paper sentence: "we use the
  paper's γ, γ_model and tune only the settings it leaves open".
- **Q3 — DECIDED (user, Nov 16): NB2/NB3 = the paper's full 15-model grid, unchanged, nothing tuned**
  (as Phase 4). If NB3's weak-model lock-on (Phase 4 finding) reappears on MIND, it is reported, not fixed.
- **Q4 — DECIDED (user, Nov 16, phase order):** E3 only; E4 is Phase 5d.
- **Q5 — DECIDED (scope decision Nov 13):** Random, LinUCB (α = 0.5), NB1 (tuned, Q1), NB2, NB3.
- **Q6 — DECIDED (user, Nov 16): 10 seeds, 0–9** (as Forest Cover and the paper's "10 runs", §5);
  ≈ 3 h on 8 cores; seeds 100–102 stay tuning-only.
- **Q7 — DECIDED (user, Nov 16): include the popularity line**, labelled "hindsight reference, not a
  baseline" (Phase 5 Q21) — shows what personalising/adapting adds over a fixed popularity rule.
- **Q8 — DECIDED (user, Nov 16): all four outputs**, separate from Forest Cover's Figure 1:
  (1) `mind_e3_regret.png` — cumulative regret vs step, mean ± std over 10 seeds, 6 lines, day boundaries
  marked; (2) results table (CSV + RESEARCH_LOG) — click rate + final regret, mean ± std; (3)
  `mind_e3_daily.png` — each algorithm's click rate per day (Nov 9–15); (4) `mind_daily_categories.png` —
  daily category click shares from the data (E3 motivation, Phase 5 Q24 note).
  - Day lines on (1) = mean day-start step over seeds 0–9 (measured Nov 16: starts ≈ 151k, 316k, 679k,
    1.05M, 1.41M, 1.74M; seeds differ by ≤ 2,436 steps = 0.12% of a run → one line per day is exact enough).
    (3) uses each seed's own day boundaries (rewards mapped back to rows via `mind_rows(data, seed)`).
- **Q9 — DECIDED (user, Nov 16): done-check §7 = (1) NB1 tuning ran, winner logged; (2) all 60 runs
  (6 × 10 seeds) complete, no NaN/inf; (3) Random ≈ analytic value within 0.005 on every seed; (4) the four
  Q8 outputs + RESEARCH_LOG entry.** Findings, not pass/fail: NB1/2/3 vs LinUCB, vs popularity, NB3
  lock-on, behaviour on spike days (Nov 10, 12, 15). No instructor criterion exists for MIND.
- **Q10 — DECIDED (user, Nov 16): tune NB1 on small → STOP, show the user the winner → launch the
  60 E3 runs on large in the background (≈ 3 h; finished runs are saved and reused, lid-close only
  pauses) → outputs + report.**
- **Q11 — DECIDED (user, Nov 16): mean ± std now; significance tests in the later "statistics" step**
  (one method for Forest Cover, E3, E4, ablations). Per-step rewards stay saved in `results/raw/` (local,
  gitignored), so tests need no re-runs.
