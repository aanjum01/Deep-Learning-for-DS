# Phase 1 — Forest Cover data + environment

**Status:** DONE (2026-11-15) — built, verified (18/18 tests), reviewed (`/simplify`, `/code-review`).
Review changes: env start row on its own RNG stream; contexts yielded as float32 copies.

## 1. Goal and why

Turn the raw Forest Cover data into the paper's 94-feature stream, in both the stationary and the
drifting version, so every algorithm in later phases plays the exact same game.

| Why | Source |
|---|---|
| **Course requirement:** "Implement the Neural Networks Committee and reproduce the main experimental results"; baselines must use "identical environments and evaluation protocols" | Course slides: *Final Project: Improve on Top of Existing Work*; *Baselines* |
| **Proposal:** "We will first reproduce the original NeuralBandit baseline" | Approved abstract (Oct 5, 2026) |
| **Research:** Figure 1 has two panels, stationary and drift; both need this environment. Drift recovery (75k–350k steps) is the gap our proposal targets | Base paper §5, pp. 5–7, Fig. 1 |
| **Instructor check (Nov 18):** NB2 must beat LinUCB on Figure 1's cumulative regret | Prof. Bouneffouf, Slack, Nov 12 |

**Out of scope here:** algorithms, regret/metrics, runner (Phase 2); MIND (later phase).

## 2. Files

| File | What it does |
|---|---|
| `scripts/preprocess_forest_cover.py` | Loads data with `fetch_covtype` → bins 10 numeric columns into 5 equal-frequency bins each (`pd.qcut` + one-hot) → keeps the 44 binary columns → labels 1–7 become actions 0–6 → shuffles once → saves `data/processed/forest_cover.npz` |
| `evaluation/environments.py` | One small function, `forest_cover_stream(X, y, n_steps, seed, drift_every=None)`, that yields `(context, correct_action)` per step: starts at a random row, loops over the data, and applies the class rotation if `drift_every` is set. The runner (Phase 2) computes reward = 1 if action == correct_action |
| `tests/test_forest_cover.py` | The done-checks below |

## 3. Settings and their sources

| Setting | Value | Source |
|---|---|---|
| Rows | 581,012 | UCI dataset page (paper says "581.000") |
| Features | 10 numeric × 5 bins = 50, + 4 wilderness + 40 soil = **94**, all 0/1 | Paper §5, p. 5 |
| Binning | Equal-frequency, 5 bins (`pd.qcut(col, 5)`) | Paper §5 ("using equal frequencies into five binary variables") |
| Actions | 7 (class 1–7 → action 0–6) | Paper §5 |
| Reward | 1 if chosen action is the correct class, else 0 | Paper §5 |
| Shuffle | Once, seed 0, before saving | Paper §5 ("it is shuffled"); seed = decision recorded Nov 12 |
| Stream | Loops over the data ("played in loop"); each run starts at a random row | Paper §5, p. 5 |
| Drift | Class rotation 1→2→…→7→1 every 500,000 steps | Paper §5, p. 6 |

**Facts checked in the data today (Nov 13):**
- All 10 numeric columns produce 5 bins. Bin sizes are about 116k each, except **Slope** (99k–142k), because many rows share the same slope value.
- Every row has exactly 1 wilderness and 1 soil column set, so every row has exactly **12 ones** (10 bins + 1 + 1).
- Class shares: 36.5%, 48.8%, 6.2%, 0.5%, 1.6%, 3.0%, 3.5%. Always guessing class 2 already scores 48.8%. Useful sanity line: anything near 49% has learned nothing beyond the majority class.

## 4. Done-check (Phase 1 is done when all pass)

1. Processed data is 581,012 × 94; all values 0/1; every row has exactly 12 ones.
2. Actions are 0–6 with the class shares above.
3. Same seed → same start row and same sequence; different seeds → different start rows.
4. The stream wraps around past the last row without breaking.
5. With drift: correct action unchanged before step 500k, shifted by 1 from 500k, by 2 from 1M, and wraps 6 → 0.
6. Streaming 2.25M steps takes only a few seconds, so the environment is never the bottleneck.

## 5. Who builds it

One **Sonnet** agent builds all three files (they're small and depend on each other). Then I verify,
run `/simplify` and `/code-review`, and report.

## 6. Decisions (user, 2026-11-13) — open questions: 0

- **D1 — Drift is cumulative:** +1 at 500k, +2 at 1M, …, counted from the start of each run (paper p. 6, "circular cycle").
- **D2 — Bin edges from all 581k rows** before streaming, as in the paper (feature values only, never labels).
- **D3 — Slope's uneven bins accepted** (tied values make exact equal frequency impossible).
- **D4 — Shared start rows:** for seed *s*, every algorithm starts at the same row ("identical environments").
- **D5 — Scope:** Forest Cover only; MIND gets its own phase.

## Note for later phases (not a Phase 1 question)

The abstract says the proposed method is evaluated **on MIND**. The README's E2 also runs NeuralUCB
and the controllers on **Forest Cover drift**, which the abstract doesn't mention. To settle in the
Phase 5 plan: keep E2 for the new method, or keep Forest Cover for the reproduction only.
