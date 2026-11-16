# Phase 3 — NeuralBandit1 (NB1)

**Status:** DONE (2026-11-15) — NB1 (Eq. 1 as printed, C = 100, λ = 1) 82.0% ± 0.2 stationary, 54.0% ± 21.5 drift;
beats LinUCB on cumulative regret in both panels. All done-checks pass. Reviewed (`/simplify`, `/code-review`).

## 1. Goal and why

Build NB1 — the paper's core algorithm — run it on both Forest Cover settings, and add it to Figure 1.
NB1 is also the building block of NB2 and NB3 (Phase 4), which must beat LinUCB by Nov 18, so the
network code here is written once and reused there.

| Why | Source |
|---|---|
| **Course requirement:** "Original NeuralBandit1" is a required baseline; "Implement the Neural Networks Committee and reproduce the main experimental results" | Course slides: *Baselines*; *Final Project: Improve on Top of Existing Work* |
| **Proposal:** "We will first reproduce the original NeuralBandit baseline"; the gap we target is NB's **fixed exploration γ** — NB1 is where that γ lives | Approved abstract (Oct 5) |
| **Research:** NB1 is Algorithm 1; NB2/NB3 are committees of NB1 instances | Base paper §3, pp. 3–4; §4, p. 5 |
| **Instructor check (Nov 18):** needs NB2, which needs NB1 first | Prof. Bouneffouf, Slack, Nov 12 |

**Out of scope here:** NB2, NB3, EXP3 (Phase 4); MIND.

## 2. What NB1 does (paper §3, Alg. 1, pp. 3–4)

- One small network **per action** (7), each: 94 inputs → C hidden units → 1 output; sigmoid
  activation; squared-error loss; weights start uniform in (−0.5, 0.5).
- Each step: every network scores the context; best = highest score (k̂).
  Play action k with probability P(k) = (1−γ)·[k = k̂] + γ/7, with **γ = 0.005**.
- Update **only the played action's network** with one gradient step of size λ, multiplied by
  1/P(played action) (importance weight; played-action reading of Eq. 1, decided Nov 12).
- 1/P is ≈1.0 when the greedy action is played, but **1,400** when an exploration action is played
  (7 / 0.005).

## 3. Files

| File | What it does |
|---|---|
| `models/mlp.py` | Batched one-hidden-layer sigmoid networks in **numpy**: forward, and a hand-written gradient step for one network. Built to hold several models × 7 actions so NB2/NB3 reuse it |
| `exploration/egreedy.py` | P(k) = (1−γ)·[k = k̂] + γ/K (CLAUDE.md: exploration logic lives in `exploration/`) |
| `algorithms/neuralbandit.py` | `NeuralBandit1(Bandit)` with params C, λ, γ, weighting mode |
| `configs/phase3_tune_nb1.yaml`, `phase3_stationary.yaml`, `phase3_drift.yaml` | Tuning grid and full runs |
| `tests/test_mlp.py`, `tests/test_neuralbandit.py` | Gradient vs. finite differences; learns a toy problem; weights stay finite; P sums to 1 |

`scripts/run_experiment.py` gets one new line in its class table. `make_figures.py` already picks up
`phase3_*` files, so NB1 lands on the same Figure 1 as Random and LinUCB.

## 4. Speed (measured today, numpy, real data, one core)

| | Plain numpy | Using only the 12 active inputs* |
|---|---|---|
| NB1, C = 50 | 66 µs/step | **36 µs/step** |
| NB1, C = 100 | 167 µs/step | **43 µs/step** |
| NB2-size committee (15 models) | 634 µs/step | **243 µs/step** |

\* Every row has exactly 12 ones out of 94, so the first layer only needs those 12 weight rows,
forward and backward. Identical maths (skips multiplications by zero).

**What that means:** one NB1 run (2.25M steps) ≈ 1.5 min alone; NB2/NB3 ≈ 9–10 min alone.
Phase 3 total (tuning + full runs) ≈ **30–45 min** on 8 cores. Phase 4 ≈ **2–3 h**, run in the
background/overnight with resume. The torch estimate was 0.3–1+ ms/step (not measured; torch isn't
installed) → hours for Phase 3 alone and possibly 12+ h for Phase 4.

## 5. Settings and their sources

| Setting | Value | Source |
|---|---|---|
| Networks | 1 per action, 1 hidden layer, sigmoid, squared error | Paper §3, p. 3 |
| Initial weights | Uniform (−0.5, 0.5) | Paper Alg. 1, p. 4 |
| γ | 0.005 | Paper §5, p. 5 |
| Update | played action only, step λ·x·δ, × 1/P(played) | Paper Alg. 1 + Theorem 1 (p. 4); played-action decision Nov 12 |
| C, λ for NB1 | **open (Q5)** — paper gives NB1 no setting; its model grid is C ∈ {1,5,25,50,100} × λ ∈ {0.01,0.1,1} | Paper §5, p. 5 |
| Biases | **open (Q2)** | Paper p. 3: N = dim(X)·C + C (no room for biases) |
| Output activation | **open (Q3)** | Paper p. 3: "a sigmoid activation function" |
| Importance weight | **open (Q4)** | Paper Eq. 1; our Nov 12 note: run weighted, capped, unweighted |

**Reference repo check (TheAmazingElys/NeuralBandit — the first author's unlicensed, unfinished
code; read only, nothing copied):**
- Regret = T × 0.93 − total reward → **confirms our constant 0.93 reference (Phase 2, D1).**
- Same binning (`pd.qcut(col, 5)` + one-hot) → confirms Phase 1.
- Its NB uses **biases** (torch `nn.Linear` default), **sigmoid on hidden and output**, MSE loss,
  **no importance weighting**, **SGD with momentum 0.9**, and an example γ = 0.05 with 64 hidden units.
  These differ from the paper's text in several places, and the repo's README calls itself work in
  progress (NB2/NB3 not implemented). Treated as a hint, not a source of truth.

## 6. Done-check

1. Correctness tests for the hand-written backprop:
   - gradient matches finite differences on random weights/contexts, C ∈ {1, 5, 25, 50, 100};
   - 12-active-inputs shortcut gives exactly the same update as the full 94-input update;
   - only the played action's network changes; batched update == updating networks one by one;
   - sigmoid via `scipy.special.expit` (overflow-safe under ×1,400 updates);
   - NB1 learns a toy problem; P sums to 1.
   - one-time cross-check vs. torch autograd in a throwaway env (`uv run --with torch`, never added
     to requirements.txt): same weights, 1,000 updates, final weights equal (D9).
2. No NaN/inf weights in any run (the ×1,400 updates are the risk) — checked at the end of every
   run; the run fails loudly otherwise.
3. NB1 clearly learns on Forest Cover: well above the 48.8% majority-class line and Random's regret.
4. Speed confirmed in the real runs (≈ the table above).
5. NB1 added to Figure 1, both panels; results + comparison with LinUCB logged. (**Target: Q7.**)

## 7. Who builds it

- **Opus agent:** `models/mlp.py`, `exploration/egreedy.py`, `algorithms/neuralbandit.py` + tests
  (hand-written gradients and the sparse trick are the error-prone part).
- Then I verify, run tuning + full runs, `/simplify`, `/code-review`, report.

## 8. Decisions (user, 2026-11-15)

- **D2 — No biases**, as the paper's connection count implies (N = dim(X)·C + C, p. 3).
- **D3 — Sigmoid on hidden and output units**; output = probability of reward.
- **D5 — C and λ tuned** on the paper's 15-setting grid (C ∈ {1,5,25,50,100} × λ ∈ {0.01,0.1,1}),
  tuning seeds 100–102, 500k steps, stationary; best mean last-100k accuracy wins.
- **D6 — Plain gradient step** Δ = λ·x·δ (paper p. 3), no momentum.
- **D7 — NB1 "works" if:** no NaN/inf weights; clearly above the 48.8% majority line; regret below
  Random. Comparison with LinUCB is logged, not required.
- **D8 — Scope:** NB1 only; NB2 and NB3 in Phase 4.
- **D4 — Weighted update (×1/P), exactly as in the paper**, is NB1 in Figure 1. Capped and
  unweighted variants are run **only if** weighted fails D7 (NaNs, or not clearly above 48.8%), and
  then reported as diagnostics, not as NB1.
- **D1 — numpy with hand-written backprop** (same maths as torch; ~10× faster at this size; maps
  line-for-line onto the paper's Δ = λ·x̂·δ, p. 3). No torch dependency.
- **D9 — One-time torch cross-check** of the hand-written gradients (see §6.1); torch is downloaded
  only into a temporary environment.
- **Loss convention:** ½(o − r)² — the usual backprop convention; the paper says only "quadratic
  error" (p. 3). Without the ½ the step is 2× larger, largely absorbed by tuning λ. Our choice; logged.
- **Faithfulness rule:** the 2014 paper is the source of truth. The author's GitHub repo is not used
  for any decision; where it differs from the paper, we follow the paper.

- **D10 — Update = Eq. 1 as printed (user, Nov 15; replaces D4's "played action" reading):**
  indicator 1[k̂_t = k] → only the greedy action's network is updated, only when it was played,
  weighted by 1/P(k̂) ≈ 1.004. Reason: the "played action × 1/P" reading collapses NB1 to 48.5% at
  every λ (×1,400 steps saturate the sigmoids), which cannot produce the paper's results; Eq. 1 as
  printed matches 2 of the paper's 3 formulas and reaches ~76% in a diagnostic. See RESEARCH_LOG §7.

## 9. Open questions

None.
