# Phase 2 — Random + LinUCB + runner + metrics + first regret plot

**Status:** DONE (2026-11-15) — all done-checks pass. LinUCB stationary 71.94% ± 0.14 (target 69–75%);
Random 14.25%; LinUCB regret ~3× steeper after the first drift; Figure 1 at `results/figures/figure1.png`.
α = 0.1 from tuning (all α within 71.7–71.9%). Reviewed with `/simplify` and `/code-review`.

## 1. Goal and why

Run the two simplest policies end to end on the Phase 1 environment, measure them the way the paper
does, and draw the first Figure 1-style plot. This builds the runner and metrics every later algorithm
reuses, and it gives the LinUCB line that NB2 must beat on Nov 18.

| Why | Source |
|---|---|
| **Course requirement:** "at least one classical contextual-bandit baseline"; "identical environments and evaluation protocols"; evaluation by cumulative reward / regret | Course slides: *Baselines*; *Evaluation: what counts as improvement?* |
| **Instructor check (Nov 18):** NeuralBandit must beat LinUCB on cumulative regret vs. Figure 1 | Prof. Bouneffouf, Slack, Nov 12 |
| **Proposal:** "We will first reproduce the original NeuralBandit baseline" and compare cumulative reward / regret | Approved abstract (Oct 5) |
| **Research:** Figure 1 plots cumulative regret vs. a 93% offline reference; the paper reports LinUCB at 72% (last 100k predictions) and says LinUCB "can't deal with non-stationarity" | Base paper §5, pp. 5–6 |
| **Planning:** steps/second for each policy tells us how long the full 10-seed runs take | Our Nov 13 plan (Saturday benchmark) |

**Out of scope here:** NB1/2/3 (Phases 3–4); dynamic regret and adaptation time (drift phase, see Q7); MIND.

## 2. Files

| File | What it does |
|---|---|
| `algorithms/random_policy.py` | `RandomPolicy(Bandit)`: uniform action, `last_probs` = 1/K |
| `algorithms/linucb.py` | `LinUCB(Bandit)`: disjoint LinUCB, Li et al. 2010 Alg. 1. All 7 arms batched in one array; keeps A⁻¹ updated with the Sherman–Morrison formula (same maths as Alg. 1, avoids a matrix inverse per step) |
| `evaluation/runner.py` | `run(policy, stream) -> rewards` (one array of 0/1 per step), plus a function that runs (policy, seed) pairs in parallel with `joblib` (already installed) and saves results |
| `evaluation/metrics.py` | cumulative regret vs. the 93% reference; last-100k accuracy |
| `scripts/run_experiment.py` + `configs/phase2_*.yaml` | Reads a config (algorithms, α, steps, seeds, drift) and runs it |
| `scripts/make_figures.py` | Figure 1-style plot: cumulative regret vs. step, mean over seeds, stationary and drift panels |
| `tests/test_random_policy.py`, `tests/test_linucb.py`, `tests/test_metrics.py` | Small checks (below) |

**Packages to add:** `matplotlib` (plots), `pyyaml` (configs — CLAUDE.md requires hyperparameters in `configs/`).
**Small cleanup from Phase 1 review:** rename `N_ACTIONS` → `FOREST_COVER_N_ACTIONS` so MIND won't collide.

## 3. Settings and their sources

| Setting | Value | Source |
|---|---|---|
| LinUCB init | A_a = I (94×94), b_a = 0 | Li et al. 2010, Alg. 1, lines 4–7 (p. 4) |
| LinUCB score | θ_a = A_a⁻¹b_a; p_a = θ_aᵀx + α·√(xᵀA_a⁻¹x) | Li et al. 2010, Alg. 1, lines 8–9 (p. 4); Eq. 5 (p. 3) |
| LinUCB update | chosen arm only: A += xxᵀ, b += r·x | Li et al. 2010, Alg. 1, lines 12–13 (p. 4) |
| LinUCB α | **open (Q2)** — paper gives no value; Li et al. tuned α on separate "tuning data" by plotting reward vs. α | Li et al. 2010 §5.2.1 (p. 6), §5.5.1 + Fig. 2 (pp. 7–8); Eq. 4 α = 1+√(ln(2/δ)/2) "may be conservatively large" (p. 3); base paper silent |
| Ties | **open (Q4)** — Li et al.: "ties broken arbitrarily" | Li et al. 2010, Alg. 1, line 11 (p. 4) |
| Constant feature | **open (Q3)** — Li et al. append a constant 1 | Li et al. 2010 §5.2.2 (p. 6) |
| Regret reference | **open (Q1)** — "computed from the rewards of an offline algorithm fitting the data with 93% of classification" | Base paper §5, p. 5 |
| Accuracy metric | Mean reward over the last 100,000 steps | Base paper §5, p. 6 |
| Steps / seeds / drift | 2.25M steps; seeds 0–9; drift every 500k for the drift panel | Base paper §5 + Fig. 1; Phase 1 decisions |

**Source:** Li et al. 2010, arXiv:1003.0146v2 PDF (10 pp.; user-provided copy, Nov 15). Page numbers refer to that PDF.

**Facts checked today (Nov 15):**
- The 94 features have rank 82 (each one-hot group sums to 1, so a constant is already implied by the data). This matters for Q3.
- A first LinUCB timing on this machine: ~0.2 ms per step → ~8 min per 2.25M-step run on one core. 10 seeds × 2 panels ≈ 2.7 h on one core, ~20–30 min if seeds run in parallel. Random is near-instant.
- Expected Random accuracy: 1/7 = 14.3%. Under a constant 0.93 reference, Random's regret grows ≈ 0.79 per step (≈ 1.8M at 2.25M steps) — consistent with Figure 1's Random line (~2M at the end of the axis).

## 4. Done-check

1. Tests: Random's actions uniform and `last_probs` sum to 1; LinUCB on a tiny toy problem with an obvious best arm learns it; LinUCB's Sherman–Morrison A⁻¹ matches `np.linalg.inv(A)` after many updates; metrics give known values on hand-made reward arrays.
2. Random's last-100k accuracy ≈ 14.3%.
3. LinUCB's stationary last-100k accuracy is close to the paper's 72% (**tolerance: Q6**).
4. Drift panel: LinUCB's regret slope gets steeper after the first drift (the paper's qualitative claim).
5. First Figure 1-style plot exists in `results/`, and the steps/second for each policy is logged.

## 5. Who builds it

- **Sonnet agent A:** `random_policy.py`, `linucb.py` + their tests.
- **Sonnet agent B (in parallel):** `metrics.py`, `runner.py`, `run_experiment.py`, configs, `make_figures.py` + metrics test.
- Then I verify, run the α tuning and the real runs, `/simplify`, `/code-review`, and report.

## 6. Decisions (user, 2026-11-15) — open questions: 0

- **D1 — Regret reference = constant 0.93 per step:** regret_t = 0.93 − reward_t, summed.
- **D2 — α tuning:** grid {0.01, 0.05, 0.1, 0.25, 0.5, 1, 2}; stationary stream; 500k steps; tuning
  seeds 100–102 (never used for results); pick the α with the best mean last-100k accuracy. That α is
  used for both panels. (Mirrors Li et al.'s separate tuning data, §5.2.1.)
- **D3 — No constant feature** for LinUCB: all algorithms use exactly the paper's 94 features.
- **D4 — Ties broken randomly** with the policy's own seeded rng.
- **D5 — Full runs now:** Random + LinUCB, 2.25M steps × seeds 0–9 × stationary and drift, seeds in parallel.
- **D6 — Tolerance:** LinUCB stationary last-100k accuracy within 72% ± 3 points (69–75%). Outside →
  investigate before Phase 3.
- **D7 — Metrics now:** cumulative regret vs. 0.93 and last-100k accuracy only; dynamic regret and
  adaptation time in the drift phase.
- **D8 — Git:** commit summary CSVs (cumulative regret every 1,000 steps, last-100k accuracy) and
  figures; per-step rewards go to `results/raw/` (gitignored — add to `.gitignore`).
