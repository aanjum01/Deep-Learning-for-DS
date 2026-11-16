# Phase 4 — NeuralBandit2 + NeuralBandit3 (committees with EXP3)

**Status:** DONE (2026-11-15). NB2 80.8% ± 0.6 stationary, 69.3% ± 0.9 drift; regret 0.332M / 0.497M,
below LinUCB (0.475M / 1.157M) in both panels, so D4 passes. NB3 72.5% / 64.4%. 57 tests pass, no NaN/inf.
Reviewed (`/simplify`, `/code-review`). Results: RESEARCH_LOG §6.

## 1. Goal and why

Build NB2 and NB3. Both are committees of the 15 NB1 models on the paper's grid, and an adversarial
bandit (EXP3) picks among the models. Run both on Forest Cover (stationary and drift, 10 seeds) and
add them to Figure 1. That finishes the reproduction (E1/E2).

| Why | Source |
|---|---|
| **Instructor check (Wed Nov 18):** NB2 must beat LinUCB on cumulative regret against Figure 1 | Prof. Bouneffouf, Slack, Nov 12 |
| **Course requirement:** "Implement the Neural Networks Committee and reproduce the main experimental results"; NB1–3 are the required baselines | Course slides: *Final Project: Improve on Top of Existing Work*; *Baselines* |
| **Proposal:** "We will first reproduce the original NeuralBandit baseline". The committee is the paper's answer to "which model is best?", and our method has to be compared against it | Approved abstract (Oct 5) |
| **Research:** NB2 is the paper's best result (76% stationary). NB3 (73%) "seem[s] to be more robust to nonstationarity" (p. 7, hedged). The drift panel is where NeuralBandit++ must improve | Base paper §4–5, pp. 4–7 |

**Out of scope:** NeuralUCB, the sliding window, the LLM controller and MIND (later phases); the toolkit.

## 2. What the paper says (read in order: §3 Eq. 1, §4, Alg. 2–3, §5; pp. 4–7)

- **§4 (p. 4):** a "model" is an MLP with one set of parameters. The paper proposes "to train the
  models in parallel and to use the adversarial bandit algorithm EXP3 to choose the best model",
  because each model's performance changes over time.
- **EXP3** (Eq. 2–3, p. 5): weights start at 1.
  - P_model(m) = (1−γ_model)·w_m/Σw + γ_model/M
  - w_i ← w_i·exp(γ_model·1[i = m]·y / (P_model(i)·M)), so only the chosen model's weight changes.
  - The text gives γ_model ∈ [0, 1] and Alg. 2–3 give [0, 0.5]. This doesn't matter at 0.1.
- **NB2, prose (p. 5):** each NB1 instance "corresponds to an arm"; EXP3 chooses the arm; "each
  neural network corresponding to the played arm is updated, and the weights of EXP3 are updated".
- **NB2, Alg. 2 (p. 6):** the Data line lists only γ_model; EXP3 draws m_t; "The model m_t choose
  action k̃_t"; then "Update of each network corresponding to the action k̃_t **for each model** with
  (1)"; then EXP3 is updated with (3). **Alg. 2 has no k̂_t and no P_t line**, although Eq. 1 needs
  both.
- **NB3, prose (p. 5):** one EXP3 per action. NB2 should find a shared best model faster, because
  NB3's EXP3s have "less possibility to update".
- **NB3, Alg. 3 (p. 6):** the Data line lists γ and γ_model.
  1. For each action k, EXP3_k draws m_k, and the action's score is s_k = N^{m_k,k}(x).
  2. k̂_t = argmax s, and P_t(k) = (1−γ)·1[k = k̂_t] + γ/K. Draw k̃.
  3. "Update of each network corresponding to the action k̃_t for each model with (1)"; "Update of
     each EXP3 weight vectors with (3)".
- **§5 (p. 5):** settings and protocol:
  - 581k instances, shuffled, 94 binary features, 7 actions;
  - played in a loop, reward 0/1, regret vs. a 93% offline reference;
  - 10 runs from random start positions, "γ = 0.005, γ_model = 0.1";
  - grid C ∈ {1,5,25,50,100} × λ ∈ {0.01,0.1,1}.
- **Results (pp. 6–7):** last 100k steps, stationary:
  - NB2 76%, NB3 73%, CTS 73%, LinUCB 72%, Banditron 57%.
  - Drift: classes rotate every 500k steps. The model-selection methods need "between 75.000 and
    350.000 in the worst case to stabilize at each drift" (no criterion is given).

## 3. Ambiguities and decisions

Eq. 1 as printed (D10 from Phase 3) updates network k only if k = k̂, with weight 1/P(k̂). In a
committee it is not obvious whose k̂ and P apply, or which models are updated at all.

| Ambiguity | Decision (user, Nov 15) | Why |
|---|---|---|
| **NB2: which models learn.** The prose says "the played arm" (arm = model, so only m_t); Alg. 2 says "for each model" | **D5: Alg. 2, every model** | §4 says "train the models in parallel", and the conclusion calls the approach "trivially parallelizable". The conflict is logged |
| **NB2: whose k̂ / P gate the update.** Alg. 2 defines neither | **D1 (revised): the chosen model's** k̂ = k̂_{m_t} and P_{m_t}. Network k̃ of every model is updated iff k̃ = k̂_{m_t}, with weight 1/P_{m_t}(k̂) ≈ 1.004 | "The model m_t choose action k̃" ties k̂/P to m_t. It has the same shape as NB3 (D2), and every model trains on every greedy step, which avoids starving the models EXP3 rarely picks. This is an **interpretation**, not the literal text. The first D1 (each model's own k̂_m) was dropped because it was wrongly labelled "literal" |
| **NB2: does the chosen model use γ?** Alg. 2's Data line has no γ | **D6: yes, γ = 0.005**, ε-greedy as in NB1 | The models are "NeuralBandit1" instances (Alg. 1 has γ), and §5 runs use γ = 0.005. This is an **inference** |
| **NB3: whose k̂ / P** | **D2: Alg. 3's own** k̂_t and P_t. Network k̃ of every model is updated iff k̃ = k̂_t, with weight 1/P_t(k̂_t) | Alg. 3 defines exactly one k̂_t and P_t, then says "with (1)" |
| **NB3: EXP3 rewards for the 6 unplayed actions** | **D3: only the played action's EXP3 receives y**. For the others this is the same as y = 0 in Eq. 3 (no change) | Unplayed rewards are never seen. The p. 5 prose ("less possibility to update") fits. **Logged limitation:** EXP3_k doesn't correct for how often action k is played, so it may favour models that over-score k |
| **NB2 importance weight vs. the true action probability** | Weight is 1/P_{m_t}(k̂) (D1). Not the mixture probability Σ_m P_model(m)·P_m(k) | It follows Eq. 1 with the chosen model's P. Theorem 1's unbiasedness then holds only approximately. Logged |

**Settled without a choice (sourced):**
- Grid, γ = 0.005, γ_model = 0.1, and the EXP3 equations: §5, Eq. 2–3.
- EXP3 in log-space: same maths, avoids overflow (Nov 12 decision).
- EXP3 reward = y ∈ {0,1} (Eq. 3). No tuning phase, since the paper gives every setting.

**Implementation defaults (not in the paper; stated so nothing is guessed):**
- One seeded `self.rng` per policy feeds the 15 models' initial weights, EXP3 draws and ε-greedy
  draws (CLAUDE.md convention). The models' starting weights are **not** matched to the standalone NB1
  runs.
- Ties, including across models in NB3's argmax, are broken uniformly at random (`argmax_random_tie`).
- Drift stabilisation (the p. 7 "75k–350k") is judged by eye from the regret curves and stated as such.
  The paper gives no criterion.

## 4. Files

| File | What it does |
|---|---|
| `exploration/exp3.py` | `EXP3(M, gamma_model, rng)`: `probs()` implements Eq. 2, `draw()`, and `update(m, y)` implements Eq. 3, in log-space (`log_w[m] += γ_model·y/(P(m)·M)`; probs come from `scipy.special.softmax`) |
| `algorithms/neuralbandit.py` | adds `NeuralBandit2` and `NeuralBandit3`. The 15 models live in **5 `MLPStack`s, one per C**, each holding the 3 λ values (MLPStack already supports a per-model λ). Under D1/D2 the gate is the same for every model, so the existing `MLPStack.update` is called unchanged. `last_probs` = the true probability of each action (NB2: Σ_m P_model(m)·P_m(k); NB3: P_t) |
| `evaluation/runner.py` | `check_finite` also checks arrays inside lists (the committee's stacks and EXP3s). Carry-over from the Phase 3 review |
| `scripts/run_experiment.py` | adds NeuralBandit2 and NeuralBandit3 to `CLASSES` |
| `configs/phase4_stationary.yaml`, `phase4_drift.yaml` | NB2 and NB3: `hidden: [1,5,25,50,100]`, `lam: [0.01,0.1,1]`, `gamma: 0.005`, `gamma_model: 0.1`, seeds 0–9, 2.25M steps |
| `configs/phase4_check.yaml` | the short check: NB2 and NB3, 200k steps, seed 0 |
| `tests/test_exp3.py` | probs sum to 1 and match Eq. 2; the log-space update equals plain Eq. 3 on a short run; stays finite after 10M updates; **finds the clearly best arm on a toy problem** |
| `tests/test_neuralbandit.py` | NB2: every model's network k̃ is updated on a greedy step and nothing changes on an exploration step (D1, D5). NB3: the update happens only when k̃ = k̂_t, and only the played action's EXP3 changes (D2, D3). **A toy committee where one model is clearly best: EXP3 must concentrate on it.** last_probs sums to 1; finite |

`make_figures.py` already picks up `phase4_*`, so Figure 1 shows Random, LinUCB, NB1, NB2 and NB3.
NB1 stays in the figure because it is a required baseline (slides), even though the paper doesn't
plot it. The run script also saves EXP3's final model probabilities per run, so they can be logged.

## 5. Settings and their sources

| Setting | Value | Source |
|---|---|---|
| Model grid | C ∈ {1,5,25,50,100} × λ ∈ {0.01,0.1,1} = 15 models | Paper §5, p. 5 |
| γ, γ_model | 0.005 (also inside NB2's chosen model, D6), 0.1 | Paper §5, p. 5; user decision Nov 15 |
| EXP3 | Eq. 2–3, weights start at 1, log-space | Paper p. 5; our Nov 12 decision |
| Network, init, loss, step | the same as NB1 (Phase 3 D1–D3, D6, D10) | Phase 3 |
| Which models learn | all 15 (D5) | Alg. 2–3; user decision Nov 15 |
| Gate and weight | NB2: k̂_{m_t}, 1/P_{m_t}(k̂) (D1); NB3: k̂_t, 1/P_t(k̂_t) (D2) | Eq. 1 + Alg. 2–3; user decision Nov 15 |
| NB3 EXP3 rewards | only the played action's EXP3 (D3) | Alg. 3; user decision Nov 15 |
| Runs | 10 seeds (0–9), 2.25M steps, drift every 500k: the same settings as the LinUCB/NB1 runs (checked in the phase2/phase3 configs) | Paper pp. 5–6 |

## 6. Speed

The committee costs ~243 µs/step (measured Nov 15), so one run takes ~9–10 min alone. That gives 40
runs (2 algorithms × 2 panels × 10 seeds), **~2–3 h on 8 cores**, in the background with resume. The
cost is the same for NB2 and NB3: every model's network k̃ needs a forward pass for its update, and
NB3 needs every model's scores.

## 7. Order of work (user, Nov 15)

1. Build the code (agents) and run the tests.
2. **Stop:** the user looks at the code; `/simplify`, then `/code-review`, then re-run the tests.
3. **Short check:** NB2 and NB3 for 200k steps on seed 0, to confirm speed and that they learn.
4. Full runs (2–3 h, background), Figure 1, then the report.

## 8. Done-check (D4, D7)

1. **Tests pass**, including the EXP3 toy test and the committee toy test, plus the 45 existing tests.
2. **No NaN/inf** anywhere in the committee (the fixed `check_finite`).
3. **Instructor check (required, D4):** NB2's final cumulative regret (mean of 10 seeds) is below
   LinUCB's in **both** panels: 0.475M stationary and 1.157M drift.
4. **Logged, not required (D7):**
   - NB2 and NB3 against our best NB1 (82.0% / 0.311M stationary): does the committee help?
   - EXP3's final model probabilities: which models win?
   - NB3 against LinUCB.
   - NB2 against 76% and NB3 against 73%.
   - Drift recovery by eye against "75k–350k".
   - If NB2 is worse than NB1, that is a finding, not a failure: the paper never reports NB1.

## 9. Who builds it

- **Opus agent:** `exploration/exp3.py`, NB2 and NB3, and their tests (EXP3 and the gating are the
  error-prone part).
- **Sonnet agent, in parallel:** the `check_finite` fix with a test, the configs, the `CLASSES` line,
  and saving the EXP3 probabilities.
- Then follow the order in §7.

## 10. Decisions (user, 2026-11-15)

- **D1 (revised): NB2 gate.** The chosen model m_t's k̂ and P. Network k̃ of every model is updated
  iff k̃ = k̂_{m_t}, with weight 1/P_{m_t}(k̂). (It replaces "each model's own k̂_m", which was mislabelled
  "literal": Alg. 2 defines no k̂.)
- **D2: NB3 gate.** Alg. 3's k̂_t and P_t.
- **D3: NB3 EXP3 rewards.** Only the played action's EXP3 gets y.
- **D4:** NB2 must beat LinUCB on final cumulative regret in both panels.
- **D5:** NB2 updates every model (Alg. 2), not only the played model (p. 5 prose). The conflict is logged.
- **D6:** NB2's chosen model explores ε-greedily with γ = 0.005 (an inference; Alg. 2 lists no γ).
- **D7:** a required toy test (EXP3/committee finds the clearly best model). Logged: NB2/NB3 vs. NB1,
  and EXP3's final model weights.

Open questions: **none.**
