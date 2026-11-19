# Phase 6a — NeuralUCB alone (no window, no controller), E1–E4

**Status: CLOSED, no longer needed (2026-11-18).** The base learner changed to NB1, instructor-approved (RESEARCH_LOG §5,
Nov 18). Done: build, review, rehearsal, J check (J = 50), MIND tuning (winner γ 0.1, λ 0.1, η 0.2 = 0.4175 over 3 seeds,
below LinUCB 0.4367 / NB1 0.4460). Forest tuning was stopped by the user at 67/128 runs (saved in results/raw). The 40
full runs will not be run (user, Nov 18): NeuralUCB is reported only as the method tried and replaced.

*Was:* **APPROVED (user, 2026-11-18), rev. 2.** Decided with the user: Q1–Q10 (planning), R1–R6 (fresh-context
Opus review) — §8. Open questions: none. Walk-through given before approval. Order (user): build →
`/simplify` + `/code-review` → rehearsal + J check → tuning → **CHECKPOINT with the user** → 40 full runs →
outputs + report. Window, controllers and ablations are 6b–6e (not planned here).

## 1. Goal and why

Build NeuralUCB [zhou2020neural] as a reported method, tune it the way the baselines were tuned (per dataset, no
planted shifts or drift), and run it on E1–E4 with 10 seeds next to the finished baselines.

| Why | Source |
|---|---|
| Proposal: NeuralUCB replaces NB's fixed-γ ε-greedy with uncertainty-aware exploration | Approved abstract (Oct 5) |
| NeuralUCB alone is a reported method on E1–E4: separates "NeuralUCB is a better learner than NB" from "drift handling helps" | User, Nov 17 (RESEARCH_LOG §5, Phase 6 roadmap) |
| Every method runs in every environment, same protocol | Slides *Baselines*; scope Nov 13 |
| Tune only without shifts (Q10 rule); paired E3 + E4 runs per seed; shift times invisible to the method | Phase 5d Q10, R6 |
| Zhou et al. claim UCB beats NeuralBandit's ε-greedy but never run NB (C39) — E1–E4 test that claim directly | REFERENCES_VERIFICATION C39 |
| Research question: can the full model adapt faster after drift? 6a gives the "no drift handling" NeuralUCB reference that 6b–6e build on | README §1; RESEARCH_LOG §5 |

## 2. What the sources say (REFERENCES_VERIFICATION R2, C32–C40; read 2026-11-18)

- **Guarantee = stationary world** (C32, Eq. 2.2, p. 2): one fixed reward function h for every round; drift is
  never discussed. The bound also needs a very wide net, a set step size η ∝ 1/m and unit-norm duplicated inputs
  (C33, Assumption 4.2 / Thm 4.5, p. 4). The paper's experiments don't use the duplication (p = md + m, §7.1 p. 7);
  the authors' code does L2-normalise (C40). → The theory gives NeuralUCB no reason to adapt to drift.
- **Algorithm** (C34–C35, pp. 2–3): f = √m W₂ ReLU(W₁x), no biases; mirrored init so f(x; θ₀) = 0; UCB = f +
  γ √(gᵀZ⁻¹g / m); Z += g gᵀ/m for the played arm; TrainNN = J gradient steps on Σ(f − r)²/2 + mλ‖θ − θ₀‖²/2 over
  **all** past data.
- **Their real-data setup** (C37, §7.2 p. 8): disjoint arm input x⁽ᵃ⁾ = (0,…,x,…,0) [li2010]; covertype among the
  datasets; m = 100, two layers, diagonal Z, constant γ ∈ {0.01, 0.1, 1, 10}, λ ∈ {10⁻¹…10⁻⁴}, η ∈ {2, 5}·10⁻ⁱ
  (i = 1…4), J = 1000, SGD batch 500, train "every 100 rounds starting from round 2000" (⚠️ ambiguous: literally no
  training before 2000; Alg. 1 trains every round — R6), T = 15,000; best grid setting reported.
- **Authors' code** (uclaml = Prof. Gu's lab; **no license → reference only, nothing copied**) differs in init,
  regulariser, batch, bonus formula, and keeps only every 100th observation after round 2000 (C40). The paper is
  the source of truth.
- **Review finding (R1):** with Eq. 2.3's √m output, ‖g‖² at init ≈ m‖x‖² (≈ 1,700 Forest, ≈ 275 MIND), so the raw
  η grid diverges (6 of 8 values on Forest in scratch runs). Fixed by the step η/m (Thm 4.5's 1/m).

## 3. Measured cost (throwaway numpy prototype, paper form, real contexts, M2, Nov 18)

| | alone | 8 in parallel (measured) |
|---|---|---|
| select + Z update, MIND (K 16, d 19) | 85–94 µs/step | ≈ 420 µs |
| select + Z update, Forest (K 7, d 94) | 71 µs/step | ≈ 320 µs |
| one SGD step, batch 500, MIND / Forest | 0.86 / 0.9–1.0 ms | ≈ 4 ms |

Parallel slowdown ×4–5, as in Phases 4–5. Paper's schedule (J = 1000) ≈ 23 h per MIND-large run → infeasible.
With J = 10 (Q2), paired init without doubled input (R5):

| Job | Runs | Est. wall-clock (8 parallel) |
|---|---|---|
| J check: J 10 vs 50, seeds 100–102, MIND-small + Forest 500k stationary (R3) | 12 | ≈ 35 min |
| Rehearsal (§7.3) | 5 | ≈ 10 min |
| MIND-small tuning: 128 settings seed 100 + top 5 × seeds 101–102 (≈ 4 min/run) | 138 | ≤ 1.3 h |
| Forest tuning: 128 settings seed 100 + top 5 × seeds 101–102 (500k steps, ≈ 8–9 min/run) | 138 | ≤ 2.5 h |
| Full runs: E1, E2 (Forest 2.25M), E3, E4 (MIND-large ≈ 2.04M) × seeds 0–9 (≈ 30 min/run) | 40 | ≈ 2.5 h |

Total ≈ 7 h; diverging settings stop within minutes (R2), so tuning is likely shorter; tuning can run overnight.
**Memory:** the history buffer grows by doubling (Forest: 2²² rows × 94 B uint8 ≈ 394 MB + a temporary copy at
each resize; MIND: float32 ≈ 160 MB + ≈ 290 MB of loaded data per job) → ≈ 4–7 GB for 8 jobs, fine on 16 GB;
free memory is checked before every long batch (user rule; E4 lesson).

## 4. Files

| File | What it does | Builder |
|---|---|---|
| `models/relu_net.py` | `ReLUNet`: the paper's two-layer net on the disjoint arm input, stored per arm — first layer V[a] (d × m) per arm + shared output weights w₂ (m); f(x⁽ᵃ⁾) = √m·w₂ᵀReLU(V[a]ᵀx). **Paired mirrored init (R5):** per arm W_a with entries ~ N(0, 4/m), shape d × m/2, V[a] = [W_a, W_a]/√2 (columns = hidden units 1…50 and 51…100); w ~ N(0, 2/m), w₂ = (w, −w) → f = 0 for every arm at θ₀; equals the paper's diag(W, W) on [x, x]/√2 exactly. `forward(x)` → f for every arm + the pieces of g (only nonzero inputs used, as `MLPStack`); `sgd_step(X, A, R, step, reg)`: one step on batch-mean (f − r)²/2 + reg/2·‖θ − θ₀‖² (hand-written backprop); returns the batch-mean data loss | Opus |
| `exploration/ucb.py` | `DiagUCB`: diagonal Z for the first layer (per arm) and for w₂, both initialised to λ; `bonus` = √(Σ g²/Z / m); `add` for the played arm: first-layer Z += g²/(2m) (R5 equivalence), w₂ Z += g²/m (C35, C37). UCB logic lives only in `exploration/` (CLAUDE.md) | Opus |
| `algorithms/neuralucb.py` | `NeuralUCB(Bandit)`, params `gamma, lam, eta, J=10, batch=500, m=100, d`. select: U = f + γ·bonus over available arms, `argmax_random_tie` (egreedy.py; RNG drawn only on a tie); Z update for the played arm. update: store (x, a, r); t = examples stored; **train if t ≤ 2000 or t % 100 == 0** (R6, Q2): J SGD steps, step η/m (R1), batch 500 drawn uniformly with replacement from all t examples with `self.rng`, reg = mλ/t (Q3), warm start (Q1). After each training call: if any weight is non-finite → raise `Diverged` (R2). History buffer doubles when full; uint8 when the contexts are 0/1 (Forest), else float32. Diagnostics per 1,000 steps (Q8): mean bonus of the played arm; exploration rate = share of steps where the played arm ≠ argmax f over available arms (first index on ties, no RNG — at θ₀ all f = 0); mean data loss (no penalty) of the last SGD step of each training call in the window (NaN if none). Docstring cites [zhou2020neural, Eq. 2.3, Alg. 1–2, §7.2] and lists every deviation (§6) | Opus |
| `scripts/run_experiment.py` | register `NeuralUCB` in `CLASSES`; `job()` catches `Diverged`, saves `{label}_seed{s}.npz` with `diverged=True` + the step it stopped at (same key rule), returns NaN results; summary rows of diverged runs get NaN (no crash, R2). Baseline saved-run keys unchanged | Sonnet |
| `configs/phase6a_jcheck_mind.yaml`, `phase6a_jcheck_forest.yaml` | J 10 vs 50, γ 1, λ 10⁻², η 2·10⁻², seeds 100–102; MIND-small; Forest stationary 500k (R3) | Sonnet |
| `configs/phase6a_trial.yaml`, `phase6a_trial_noshift.yaml`, `phase6a_trial_fstat.yaml`, `phase6a_trial_frot.yaml` | rehearsal (§7.3), J-check setting (R4). Names must **not** end in `_stationary`/`_drift` (`make_figures.py` would add them to Figure 1) | Sonnet |
| `configs/phase6a_tune_mind.yaml`, `phase6a_tune_forest.yaml` (+ `_top5` follow-ups written after stage 1) | Q6/Q7 grids (128 settings, seed 100), then top 5 × seeds 101–102 | Sonnet |
| `configs/phase6a_stationary.yaml`, `phase6a_drift.yaml`, `phase6a_mind_e3.yaml`, `phase6a_mind_e4.yaml` | the 40 full runs, tuned settings, seeds 0–9; `_stationary`/`_drift` names put NeuralUCB into Figure 1 automatically; E4: `shifts: [300000, 1000000, 1500000]`, `d: 19` | Sonnet |
| `scripts/make_mind_figures.py` | result-set names as arguments (`make_mind_figures.py phase5c_mind_e3 phase6a_mind_e3`; default = Phase 5c as before; label → run-dir map; summaries merged; tables under the last name); writes `results/phase6a_mind_e3_{table,daily}.csv` and redraws `mind_e3_regret.png`, `mind_e3_daily.png` with 7 lines. Phase 5c CSVs untouched | Sonnet |
| `scripts/make_recovery.py` | new setups `e4_6a` (dirs: baselines as `e4`, NeuralUCB → (`phase6a_mind_e4`, `phase6a_mind_e3`); stem `phase6a_mind_e4`, fig `mind_e4`), `forest_6a` (`LEARNERS`/`FOREST_PHASE` + NeuralUCB → `phase6a_drift`/`phase6a_stationary`; stem `phase6a_forest_e2`, fig `forest_e2`), `trial_6a` (NeuralUCB `phase6a_trial` vs `phase6a_trial_noshift`, stem `phase6a_trial`); verdicts = every pair incl. NeuralUCB, per shift; an algorithm with any diverged run is skipped with a printed note (/simplify, Nov 18); new `mind_e4_exploration.png` (Q8 traces, E3 vs E4 mean over seeds, shifts marked). Phase 5d CSVs untouched | Sonnet |
| `tests/test_neuralucb.py` | §7.1 tests | Opus |

Parallel: Opus builds model + exploration + algorithm + tests; Sonnet does configs, runner and script changes at the
same time (no shared files). No change to any baseline algorithm or environment.

## 5. Design (decided, §8)

| Piece | Decision | Ref. |
|---|---|---|
| Network | two layers, ReLU, no biases, √m output scaling, width m = 100 | C34, C37 |
| Arm input | disjoint model (own first-layer weights per arm, shared output weights) | C37 [li2010] |
| Init | paper's mirrored init in its paired form (no doubled input) → f = 0 for every arm at the start | Q4, R5, C34 |
| Inputs | raw (same numbers as every baseline) | Q5 |
| Exploration | UCB, constant γ (tuned), diagonal Z₀ = λI, Z update for the played arm at θ_{t−1}; only available arms | C35–C37; Phase 5 masking |
| Training | warm start; train when t ≤ 2000 or t % 100 == 0; J = 10 SGD steps of size η/m; batch 500 uniform with replacement from **all** stored examples; loss = batch mean (f − r)²/2 + (mλ/t)/2 ‖θ − θ₀‖² | Q1–Q3, R1, R6, C37 |
| Divergence | fail fast after a training call; run saved as "diverged", excluded from tuning | R2 |
| Tuning | paper's grid γ {0.01, 0.1, 1, 10} × λ {10⁻¹…10⁻⁴} × η {2, 5}·10⁻ⁱ (128); MIND-small one pass / Forest stationary 500k; all 128 on seed 100 → top 5 (by total reward, non-diverged) on seeds 101–102 → best 3-seed mean among those not diverged; exact ties → first in grid order (γ, λ, η ascending); near-ties shown at the checkpoint; nothing tuned on drift/shifted data | Q6, Q7, R4; Phase 5d Q10 |
| Reported runs | E1 Forest stationary 2.25M, E2 Forest drift (rotation every 500k), E3 MIND-large, E4 MIND-large + shifts 300k/1M/1.5M; seeds 0–9; E3/E4 paired per seed | README; Phase 5d R6 |
| Saved per run | rewards (+ best on MIND) + Q8 traces per 1,000 steps (+ diverged flag) | Q8, R2 |

## 6. Settings and sources

| Setting | Value | Source |
|---|---|---|
| m, L | 100, 2 | paper §7.2 (C37) |
| Z | diagonal, Z₀ = λI | paper §7.2, Alg. 1 (C35, C37) |
| γ | constant, tuned in {0.01, 0.1, 1, 10} | paper §7.1/7.2 (C36–C37); Q6 |
| λ | tuned in {10⁻¹, 10⁻², 10⁻³, 10⁻⁴} | paper §7.2 (C37); Q6 |
| η | tuned in {2·10⁻ⁱ, 5·10⁻ⁱ}, i = 1…4; **applied as η/m** | paper §7.2 (C37); **deviation**, Thm 4.5 (p. 4) η ∝ 1/m; R1 |
| Batch, retrain interval | 500; every 100 rounds from round 2000 | paper §7.2 (C37) |
| Before round 2000 | train every round | Alg. 1 + authors' code (C40); §7.2 ambiguous; R6 |
| J | **10** (paper 1000) | **deviation**, cost (§3); Q2; J check (R3) |
| Warm start | continue from current θ (paper: from θ₀) | **deviation**, cost; Q1 |
| Loss scaling | paper's L divided by t | Q3 (same minimiser as C35) |
| Init | mirrored, paired form (= diag(W, W) on [x, x]/√2) | paper §2–3 (C34) + Assumption 4.2 p. 4; Q4, R5 |
| Input scaling | raw (theory: unit length) | **deviation** from Assumption 4.2 (theory only); Q5; D3 |
| Tuning seeds / rule | 100–102; best mean total reward; MIND-small, Forest stationary 500k | Phase 2/3/5 precedent; Q6, Q7, R4 |
| Report seeds | 0–9 | as all baselines |
| Not used | ν, δ, S, the theoretical γ_t formula (only needed when γ_t is not constant) | C36 |

## 7. Done-check (Q9, R2–R4)

**Pass/fail**
1. All tests pass, including the new ones: hand-written gradients = finite differences (+ one-time torch
   cross-check in a temporary `uv run --with torch` env, as Phase 3); f(x; θ₀) = 0 for every arm; the paired form
   equals the literal doubled-input form step for step (f, bonus, after several SGD steps) on a toy problem;
   diagonal Z = diagonal of Alg. 1's full Z on a toy problem; unavailable arms are never picked; same seed →
   identical rerun; `reset()` replays exactly; a diverging setting raises `Diverged` and the runner records it
   without crashing the batch; all existing tests (incl. the permanent Forest Cover exact-match test) pass.
2. **J check** (R3): J = 50 is "clearly better" only if its 3-seed mean total reward exceeds J = 10's by
   > 2 × √(s₅₀²/3 + s₁₀²/3); checked separately on MIND-small and Forest. **Updated (user, Nov 18, at approval):
   no stop — pick the clearly better J myself, unless J = 50 is "mega slow"**, defined (by me, user asleep) as a
   tuning estimate from the J check's measured run times above ≈ 12 h; then J = 10 stays and the gap is shown at the
   checkpoint. One J for both datasets (the larger one if either dataset says J = 50).
   **Result (Nov 18):** J = 50 clearly better on both (MIND +0.0051 > 0.0041; Forest +0.0130 > 0.0053); measured
   tuning ≈ 5.5 h < 12 h → **J = 50** for tuning and full runs (RESEARCH_LOG §8).
3. **Rehearsal:** MIND-small seed 100, NeuralUCB without and with shifts at 50k/100k/150k — pair identical before
   50k, `make_recovery.py trial_6a` produces every output; Forest seed 100, 50k stationary (`phase6a_trial_fstat`)
   and 50k with `drift_every: 25000` (`phase6a_trial_frot`, mechanics only) — identical before 25k, finite; every
   output script runs on the rehearsal files.
4. Tuning tables + winners shown at the **checkpoint**; no full run before the user's go.
5. All 40 runs complete, none diverged, no NaN/inf.
6. E4 = E3 bit-for-bit for the first 300,000 steps, all 10 seeds.
7. All Q10 outputs + RESEARCH_LOG entry; README roadmap + CLAUDE.md updated.

**Findings (not pass/fail):** NeuralUCB vs NB1, NB2, NB3, LinUCB on E1–E4 (regret, click rate / last-100k
accuracy); the pre-registered "adapts better" verdicts per shift vs every baseline (E4, Forest E2); does
NeuralUCB beat LinUCB on our Forest Cover as on the paper's covertype (C38)? Exploration rate over time and
whether it reacts to shifts (Q8); how many grid settings diverged.

## 8. Decisions (user, Nov 18)

**Planning questions**
- **Q1 — warm start.** Each TrainNN continues from the current θ, not from θ₀ (deviation from Alg. 1 line 12,
  C35; the authors' code also warm-starts, C40). Why: restarting needs J ≈ 1000 per retrain (≈ 23 h per MIND run,
  §3); the objective (incl. the pull toward θ₀) is unchanged.
- **Q2 — J = 10 SGD steps per TrainNN, batch 500, every 100 rounds from round 2000, every round before** (paper
  C37 except J = 1000 → 10). ≈ 30 min per full run in parallel → the 40 reported runs ≈ 2.5 h (not including
  the J check or tuning). Checked by the J check (R3).
- **Q3 — the paper's loss divided by t:** batch mean (f − r)²/2 + (mλ/t)·‖θ − θ₀‖²/2, t = stored examples. Same
  minimiser as the paper's L(θ) (C35). Not the repo's weight decay toward 0 (C40). Comparison note (user asked):
  fairness comes from the shared protocol (streams, seeds, metrics, tuning rule), not from matching internals;
  this keeps "NeuralUCB" = the published algorithm, so NeuralUCB vs NB tests Zhou et al.'s own claim (C39). With
  all past data and a fading pull, plain NeuralUCB keeps fitting the old world — the expected reference for 6b.
- **Q4 — paper's mirrored init** (f = 0 for every arm at the start, no arm favoured by chance). Built in the
  paired form (R5).
- **Q5 — raw inputs** (as every baseline, D3). Measured: Forest contexts all have length √12 (unit scaling = a
  constant absorbed by η); MIND lengths 1.00–1.41 (median 1.11).
- **Q6 — MIND tuning:** paper's 128-setting grid on MIND-small seed 100 (one pass, no shifts), top 5 re-run on
  seeds 101–102, winner = best 3-seed mean; table at the checkpoint. NeuralUCB's grid (128 screened on one seed)
  is larger than NB1's (15 × 3 seeds) and LinUCB's (7 × 3) — it is the paper's own grid; stated in the write-up.
- **Q7 — Forest Cover tuned separately**, same scheme, stationary 500k, seeds 100–102 (as LinUCB's α and NB1's
  C, λ were).
- **Q8 — save three traces per 1,000 steps:** bonus of the played arm, exploration rate (played ≠ argmax f;
  comparable to NB's fixed 0.5%), training data loss.
- **Q9 — order and done-check as §7;** tuning between rehearsal and checkpoint.
- **Q10 — NeuralUCB added as a 7th line to the existing figures** (Figure 1, E3 regret/daily, E4 regret/recovery,
  Forest E2 recovery); new `phase6a_*` tables; Phase 5 CSVs untouched; verdicts vs every baseline per shift; new
  `mind_e4_exploration.png`.

**Fresh-context review (Opus, Nov 18) — R1–R6**
- **R1 — step = η/m** with η from the paper's grid. Why: with Eq. 2.3's √m output, ‖g‖² ≈ m‖x‖² at init, and the raw
  grid diverges on most values (Forest: 6 of 8 η → NaN within a few rounds; MIND: 3 settings); Thm 4.5 itself sets
  η ∝ 1/m. Stable limits become ≈ 0.12 (Forest) / ≈ 0.7 (MIND).
- **R2 — fail fast on divergence:** NeuralUCB raises `Diverged` when its weights become non-finite after a training
  call; the runner saves the run as "diverged" (NaN in the summary) instead of crashing the batch; diverged
  settings are excluded from ranking (a top-5 setting that diverges on seed 101/102 drops out) and shown as
  "diverged" in tables. Small runner change (§4).
- **R3 — J check:** J 10 vs 50 at γ 1, λ 10⁻², η 2·10⁻², seeds 100–102, on MIND-small and Forest 500k; rule §7.2
  (pre-registered, same form as the E4 margin).
- **R4 — tuning ties:** exact ties → first in grid order (γ, λ, η ascending); near-ties shown at the checkpoint;
  rehearsal uses the J-check setting.
- **R5 — paired form of the mirrored init:** first layer [W; W]/√2 per arm on raw x, output (w, −w) — exactly the
  paper's diag(W, W) on [x, x]/√2 (both copies always get identical updates); bonus matches with first-layer
  Z += g²/(2m). Saves the measured +6% / +20–33%. Equivalence test in §7.1.
- **R6 — train every round before round 2000** (Alg. 1 + authors' code); §7.2's wording is an ambiguity (logged).
- **Plain corrections applied (no decision):** §2 wording (duplication is theory-only, not in the experiments; the
  authors' code normalises; disjoint input is §7.2/C37); memory estimate (≈ 4–7 GB for 8 jobs); rehearsal configs
  split and never named `*_stationary`/`*_drift`; output-script modes and file names spelled out; training trigger,
  exploration-rate tie rule and loss trace defined; tie-break via `argmax_random_tie`.

**Open questions: none.**
