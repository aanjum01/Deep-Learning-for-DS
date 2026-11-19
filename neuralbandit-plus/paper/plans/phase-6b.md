# Phase 6b — NB1 + sliding window (no controller)

**Status: DONE (2026-11-18); the window was then rejected in favour of gradual forgetting (6b part 2, user Nov 18).** Results: RESEARCH_LOG §6 (Phase 6b). *Was:* **APPROVED (user, 2026-11-18), rev. 2.** Q1–Q7 and review R1–R18 settled; walk-through given before approval.
Order: build → `/simplify` + `/code-review` → tests → rehearsal → tuning → **CHECKPOINT with the user** → 140 full runs → outputs + report.

## 1. Goal and why

Give NB1 a sliding-window memory, so that it learns only from the most recent W interactions, and test it on MIND.
NB1 runs with its fixed γ = 0.005 and no controller.

| Why | Source |
|---|---|
| "The sliding window helps us train on only the most recent, relevant data" | Approved abstract (Oct 5) |
| Base learner = NB1 (NeuralUCB replaced) | Professor + user, Nov 18 (RESEARCH_LOG §5) |
| "Appropriate baselines and ablation studies to evaluate the contribution of each component" | Professor, Nov 18 |
| NB1 + window (fixed γ, no controller) = the **"−LLM controller" ablation** | Scope (CLAUDE.md), Nov 17/18 |
| E4 finding: adaptation slows with every shift (NB1 14k → 47k → 114k steps), consistent with lost plasticity; "if the sliding window flattens it, that is evidence for plasticity" | RESEARCH_LOG §5/§6 (Phase 5d) |
| Tune only without planted shifts; paired E3 + E4 runs per seed; shift times never visible to the method | Phase 5d Q10, R6 |
| The controller (6c/6d) must later be able to change W and trigger a retrain ("decide how much old data to forget") | Abstract; README §2 |

## 2. What the sources say

Read in full 2026-11-18 (REFERENCES_VERIFICATION R11–R14, C41–C47, C49). ⚠️ C41–C42 (Garivier & Moulines) and C45–C46
(Gama et al.) rest on **author manuscripts**, which the user accepted on Nov 18 as standing in for the published versions (cite sections, not pages); C48 (ADWIN) rests on a preprint and is not used in 6b.

- **NB1** [allesiardo2014, §3, Alg. 1, Eq. 1]: one network per action, learns online with one SGD step per round. The
  step is taken only when the greedy action was played, weighted by 1/P. **No stored data**, so a window has nothing
  to act on as NB1 stands. The window therefore needs a buffer plus a way to forget, which is new design.
- **Sliding-window bandits.** SW-UCB uses "only the τ last plays" [garivier2011, 2011 manuscript §2, Alg. 2, p. 3; C41].
  Cheung et al.'s SW-UCB for linear bandits (our label "SW-LinUCB") fits only rounds 1∨(t−w)…t−1 [cheung2019, §4 Eq. 2; C43]. Both papers choose the window from the horizon and
  the number of changes (Υ_T) or the size of the drift (B_T) [C42, C43]. We know neither, and the shift times must
  stay invisible, so W has to be tuned. Two source-based defaults exist but don't fit NB1: Cheung's B_T-agnostic
  w = ⌊(dT)^{2/3}⌋ (proven for linear bandits only) [C43], and Garivier's experiments use τ = 4√(T log T) (one value,
  not their own §3.1 formula) [C42]. Cheung et al.'s **Bandit-over-Bandit** tunes w online: EXP3 picks a window for each block, and a fresh
  learner restarts every block [C44]. This is the published precedent for letting a higher-level learner choose W, our controller's job
  in 6c/6d.
- **Drift survey** [gama2014, §3.1.1; C45]: a short window adapts fast during change but hurts in stable periods; a
  long window is the reverse. Retraining "discards the current model and builds a new model from scratch". It is
  **blind** when periodic, or **informed** when a change detector triggers it [§3.3.1–3.3.2; C46]. **6b = blind
  (periodic); 6c/6d add the informed trigger.** Blind methods "forget old concepts at a constant speed
  independently of whether changes are happening or not" (§3.3.2). That is the expected cost in calm periods.
  Forgetting is either abrupt (windows) or gradual (weighting by age); "The more abrupt is forgetting, the faster
  the reactivity, but also the higher the risk of capturing noise" (§3.1.2). These are Q1's options (a) and (b).
  Read from the accepted author manuscript, so cite sections, not pages.
- **Plasticity** [dohare2024; C47, C49; full paper read]: networks trained continually gradually lose the ability to
  learn, **small ones included**. A 1-hidden-layer, 5-unit learner degrades to the linear baseline, and the loss is
  "most pronounced at the smaller network sizes" (Permuted MNIST). Their fix re-initialises a tiny fraction of the
  least-used units at every step. **A full reinitialisation is only a comparison baseline**, and it does worse than
  continual backprop: on Permuted MNIST it stays at the first-task level, and on ImageNet it ends slightly below
  continual backprop and takes ≈ 10× longer to reach its best. Their full reinitialisation baseline starts the network over (in RL they call it practical only with large
  buffers). Our reset retrains on the
  last W interactions, which is closer to their "retrain from scratch" reference: on CIFAR that beat 100 classes of
  incremental training by 5%. Differences from us: many tasks (loss on CIFAR after 8 increments, ant after 1–2
  changes), and no 1-unit network is tested. E4 has 3 shifts, and NB1 on MIND has 1 hidden unit per arm.
  ⇒ It motivates testing a reset. It does not show that one helps NB1. Our E4 slowdown may also come from timing.
- **ADWIN** [bifet2007]: not used in 6b. The published SDM version is not accessible (user, Nov 18); only a 2006 author
  preprint was read. Revisit in 6c (rule-based trigger), where it would matter.

## 3. Measured cost (throwaway prototype, M2, one job alone, seed 100, Nov 18)

Prototype = option A of Q1 below: every R steps, a fresh NB1 network (paper init) is trained by one pass, in time
order, over the stored Eq. 1 updates of the last W steps. Online learning continues in between.

| Run | Click rate / accuracy | Time | of which retraining |
|---|---|---|---|
| MIND-small, NB1 (C 1, λ 1), no window | 0.4448 | 7.0 s (34 µs/step) | — |
| + window W 20k, R 5k | 0.4345 | 24.1 s | 17.2 s |
| + W 50k, R 10k | 0.4416 | 24.4 s | 17.6 s |
| + W 100k, R 25k | 0.4279 | 16.1 s | 9.2 s |
| Forest 200k, NB1 (C 100, λ 1), no window | 0.7127 | 11.9 s | — |
| + W 50k, R 10k | 0.6972 | 48.4 s | 36.7 s |

- One replayed update costs ≈ 23 µs (MIND) / ≈ 49 µs (Forest) alone. Extra cost per step = (W/R) replayed updates.
- In parallel, NB1 runs ≈ 2.6× slower than alone (Phase 5c: 180 s per 2.04M-step MIND-large run). With W/R = 5,
  one MIND-large run ≈ 12–15 min, so **E3 + E4 × 10 seeds (20 runs) ≈ 45–60 min**. Tuning on MIND-small is a few
  minutes.
- On MIND-small (one seed) every window scored **below** plain NB1. This is expected: a window throws away data,
  and it can only pay off where old data misleads, i.e. after drift. This makes the tuning rule (Q4) the central
  question.
- Memory: the buffer holds ≤ W rows (MIND: 19 floats; Forest: 94 bytes), a few MB.

## 4. Files

| File | What | Who |
|---|---|---|
| `nonstationary/window.py` | `Notebook`: preallocated arrays (x float32 rows × d, step int64, action int16, reward int8, weight float64) with a head pointer, grown by doubling; `add`, `trim(t, W)` (keep step > t − W), iterate oldest first. Arrays, not a list of tuples: ≈ 90 MB per job at W = 987k instead of ≈ 400 MB, and `check_finite` scans a few arrays, not 1M objects | Opus (me) |
| `algorithms/neuralbandit.py` | `NeuralBandit1SW(NeuralBandit1)`, params `W`, `R = W // 5`. After each Eq. 1 update it also stores that row. Every R steps (only once t > W) `retrain()` trims the notebook, builds a fresh `MLPStack` (paper init, from `self.rng`) and replays the rows once, oldest first, with the same `update`. `W` can be changed and `retrain()` called from outside (6c/6d). `diagnostics()` returns the retrain steps | Opus |
| `tests/test_window.py` | See §7.1 | Opus |
| `scripts/run_experiment.py` | Register `NeuralBandit1SW` | Opus |
| `configs/phase6b_*.yaml` | `tune_mind` (MIND-small, seeds 100–102: NB1 + 4 W); `trial`, `trial_noshift` (rehearsal); `mind_e3`, `mind_e4` (MIND-large, seeds 0–9) | Sonnet agent (routine), checked by me |
| `scripts/make_recovery.py` | New setups `e4_6b`, `trial_6b`: NB1-SW lines next to the Phase 5 baselines, same verdict rule | Sonnet agent, checked by me |
| `scripts/make_mind_figures.py` | Already takes result-set names: `make_mind_figures.py phase5c_mind_e3 phase6b_mind_e3` | — |

The output of earlier phases is untouched: Phase 5 CSVs are regenerated byte-identical (checked as in 6a).

## 5. Design (decided, §8)

- **What gets stored** (Eq. 1): only rounds where NB1 actually learned, i.e. the played action was the greedy one,
  with that round's weight 1/P(k̂). Explored rounds never update NB1, so they are not stored either.
- **Exact schedule** (R8):
  - t = the number of `update` calls so far (1-based after the first), reset to 0 with the notebook in `_init_state`.
  - At the **end** of `update`: if t % R == 0 and t > W, then `retrain()`. The new network is used from the next
    `select` on.
  - R = W // 5, recomputed from the current W. While t ≤ W nothing would be forgotten, so there is no retrain.
- **Retrain** (Q1, Q2):
  - Trim the rows with step ≤ t − W.
  - Build a fresh `MLPStack(1, K, d, C, [λ], self.rng)` (paper init U(−0.5, 0.5), no biases). This uses exactly one network's
    worth of rng draws (d·C·K + C·K).
  - For each stored row, oldest first: `h, o = net.forward(x)` **on the fresh network**, then `net.update(x, h, o, a,
    r, weight)` (R6). λ is a constant and avail isn't needed (the action is fixed), so neither is stored.
  - Between retrains NB1 keeps learning online as usual.
- **Randomness**: the fresh network's weights come from `self.rng`, so the run stays deterministic. Paired E3/E4 runs
  reset at the same steps, so E4 = E3 up to the first shift.
- **For 6c/6d**: `W` is a plain attribute. R and the t > W gate follow the current W at the next check. `retrain()` can
  be called at any time.
- **Labels** (R13): `NB1-SW_W<size>` in configs, CSVs and figures, e.g. `NB1-SW_W197k`. The headline
  "−LLM controller" comparison is NB1-SW (tuned W) vs NB1 under the pre-registered adapts_better rule; the other
  baselines are shown too.
- **Saved-run keys** (R12): the class name is part of the key, so old keys don't change. R is derived, not
  configured, and identical between rehearsal and full runs.
- **Not in 6b**: Forest Cover (Q5), any controller, any change to γ.

## 6. Settings and sources

| Setting | Value | Source |
|---|---|---|
| NB1 on MIND | C = 1, λ = 1, γ = 0.005, d = 19 | Phase 5c tuning (user-confirmed Nov 16); paper γ |
| Window type | Reset + retrain from the last W steps (abrupt forgetting) | Q1; gama2014 §3.1.1–3.1.2, §3.3.2 |
| Retrain pass | One, in time order, NB1's own rule | Q2 |
| Retrain interval | R = W/5 | Q3 |
| W grid (MIND-small) | {10k, 20k, 50k, 100k} | Q4 |
| Tuning | Seeds 100–102, one pass, no shifts; smallest windowed W within 2 × combined SE of the best | Q4, R2; Q10 rule (5d) |
| MIND-large W | Tuned W × 9.868, rounded to the nearest 1k: total rows of MIND-large (2,270,815) ÷ MIND-small (230,117), seed-free since the 90% user sampling cancels. 10k → 99k (R 19,800), 20k → 197k (39,400), 50k → 493k (98,600), 100k → 987k (197,400) | Q4; R9 |
| E3/E4 sizes | 7: 10k, 20k, 50k (count) + 99k, 197k, 493k, 987k (time span); all reported, none picked by E4 | Q4 + R1 (user) |
| Report runs | MIND-large E3 + E4, seeds 0–9, shifts 300k / 1M / 1.5M | Phase 5d; Q6 |


**Disclosure (R3):** the case for adding small windows to the sweep, and the R = W/5 argument (Q3), use knowledge of E4's
shift gaps and adaptation times. That is acceptable only because every size is reported and the headline W comes
from the shift-free tuning rule.

**Walk-through point (side-agent note + review R1, Nov 18): the converted window can be as long as the gaps between shifts.**
Converted sizes are 99k / 197k / 493k / 987k, all longer than NB1's own recovery times (14k–114k). Any W ≥ 250k
has no retrain before shift 1, so there it is plain NB1, bit-for-bit. After a shift, every retrain in the next ≈ W
steps rebuilds NB1 mostly from **old-world** rows and can undo what it had re-learned online.

**Decision R1 (user, Nov 18): run both scales, 7 sizes, on E3 + E4.**
- Count-based: 10k, 20k, 50k, the same number of interactions as tuned.
- Time-span: 99k, 197k, 493k, 987k, the same stretch of the week.

Neither conversion is backed by a source: W trades staleness (time) against estimation noise (sample count). All 7
are reported and none is picked by E4. The headline W is still the shift-free tuning rule (Q4) with the time-span
conversion. 140 runs, ≈ 3.5–4 h.

**Pre-registered reading:** "the window hurts at the shifts" is a possible outcome. If it happens it is reported as a
property of **blind** windows (an old-world rebuild after a change; Gama et al.'s "constant speed independently of
whether changes are happening"). It is the motivation for 6c/6d's triggered reset, not a reason to retune.

## 7. Done-check

**Pass/fail**
1. **Tests** (`tests/test_window.py`), plus all existing tests, including the permanent Forest Cover exact-match test:
   - the notebook keeps exactly the rows from the last W steps, oldest first;
   - with W ≥ the run length (never retrains), NB1-SW = NB1 bit-for-bit (same actions, same rng draws);
   - after a retrain, the network equals a fresh `MLPStack` (same rng state) fed the stored rows in order;
   - only greedy-played rounds are stored, with weight 1/P;
   - with W < run length, NB1-SW = NB1 bit-for-bit **up to the first retrain**;
   - the first retrain happens at the first multiple of R above W, and later ones at the right steps, also after W
     is changed;
   - a retrain uses exactly one fresh network's worth of rng draws;
   - `reset()` clears the notebook and t;
   - a small-W sanity run is finite and beats Random;
   - masking: unavailable arms are never picked;
   - same seed → identical rerun; `reset()` replays exactly.
2. **Rehearsal**: MIND-small seed 100, W = 20k, with and without shifts at 50k/100k/150k. The pair must be identical
   before 50k, and `make_recovery.py trial_6b` must produce every output. Also measure peak memory (RSS) and the time of one
   retrain at the largest W (R7, R11).
3. **Checkpoint (Q7)**: tuning table (NB1 + 4 W, 3 seeds), chosen W, conversion factor and E3/E4 sizes shown to the
   user. **No full run before the user's go.**
4. **Full runs**: every run complete and finite; E4 = E3 bit-for-bit for the first 300,000 steps, all seeds and sizes.
5. **Outputs**: NB1-SW added to the E3/E4 tables, figures, recovery table and verdicts (pre-registered rule vs NB1,
   NB2, NB3, LinUCB per shift); RESEARCH_LOG, README roadmap, CLAUDE.md updated; Phase 5 outputs unchanged.

**Findings (not pass/fail):** does the window cut clicks lost at shifts 2–3 below NB1's 5,544 / 6,180? Does it
flatten NB1's 14k → 47k → 114k adaptation times (the plasticity question)? What does it cost in total E3 clicks?
How does it change across window sizes?

**Order (Q7):** build → `/simplify` + `/code-review` → tests → rehearsal → tuning → **checkpoint** → full runs →
outputs + report → (optional) gradual forgetting. Estimated time: tuning ≈ 5 min. Full runs at 8 parallel jobs (as in Phase 5c): ≈ 13 min per run at most (34 + 5 × 23 µs
per step alone, × 2.6 in parallel); larger W retrain less often, so they are cheaper. One retrain at W = 987k ≈ 1M
replayed updates ≈ 60 s in parallel (logged).

## 8. Decisions

**Planning questions (asked one at a time)**

- **Q1 — reset + retrain (abrupt forgetting)** (user, Nov 18). NB1 keeps the last W interactions it learned from.
  Every R steps it starts a fresh network (paper init) and re-learns from those W only, then keeps learning online.
  Why: the abstract's "train on only the most recent data"; the survey's fixed sliding window, "often used as a
  baseline", and blind retraining [gama2014 §3.1.1, §3.3.2]; the controller can later set W and trigger a reset
  (6c/6d). Rejected: gradual forgetting (no real window, and the controller gets one fuzzy dial); reset only on
  drift (needs the 6c/6d detector). Known cost: in calm periods it throws away useful data (measured §3), and a
  scheduled reset fully clears the old world only W steps after a change.
- **Q2 — one pass, in time order, with NB1's own rule** (user, Nov 18). The notebook holds only the rounds NB1 actually
  learned from (Eq. 1: the greedy action was played), with that round's 1/P weight and λ. A retrain replays them once,
  oldest first, so the result is a fresh network trained by one in-order pass of the **logged** Eq. 1 updates from the last W
  steps. (Not literally "an NB1 that started W steps ago": that NB1 would gate on its own greedy action. The stored
  1/P = 1/(1 − γ + γ/K_t) is almost constant, so the difference is small.)
  Rejected: several passes (2–5× cost, breaks NB1's one-look learning, adds a setting); one shuffled pass (loses the
  recency order).
- **Q3 — R = W/5** (user, Nov 18, after asking whether W/10 is worth it). Old-world data leaves the notebook W steps after a
  change whatever R is; R only sets how soon after that the next reset comes (≤ 1.2W with W/5, ≤ 1.1W with W/10:
  ≤ 5k steps apart at W = 50k, vs measured adaptation times of 14k–114k). W/10 doubles the replay cost (+35–45 min per
  MIND batch, ≈ +2–3 h over 6b–6e). Tuning W absorbs the coarser schedule, and 6c/6d's drift-triggered reset removes the
  lag at its source. Cost: 5 replayed updates per step, the same for every W.
- **Q3b — gradual forgetting → Phase 6b, part 2** (user, Nov 18; first "own plan later, if time", review R4; then made
  definite and moved before 6c). It has its own short plan, `paper/plans/phase-6b2.md`, written after the 493k reset-timing check. Why it might be worth it: Gama et al. name
  abrupt vs gradual forgetting but report no comparison [gama2014 §3.1.2].
- **Q4 — tune on MIND-small, convert by time span, plus a sensitivity sweep** (user, Nov 18).
  - Grid W ∈ {10k, 20k, 50k, 100k} on MIND-small (≈ 8 h, 16 h, 1.7 d, 3.3 d of the week), seeds 100–102, one pass,
    no planted shifts (Q10 rule). Plain NB1 is run alongside as the "no window" reference.
  - Pick the **smallest windowed W whose 3-seed mean click rate is within 2 × combined SE of the best windowed W**
    (combined SE = √(s₁²/3 + s₂²/3), the same margin as the E4 rule) (review R2, user Nov 18). Why: bigger windows stay
    identical to NB1 for longer, so "best mean" mechanically favours them. "No window" is excluded because the method
    has a window by definition; its gap is reported. The checkpoint table shows per-seed values.
  - Convert to MIND-large by the same time span: W_large = W_small × (mean MIND-large stream length / mean MIND-small
    stream length) ≈ × 9.85, rounded to 1k. MIND-small and MIND-large cover the same week.
  - Sensitivity sweep: superseded by R1 (7 sizes on both scales). **All are reported, none is picked by E4.** The
    headline is the tuned W.
  - Known limits: on MIND-small a window costs clicks in calm stretches, so tuning leans toward big windows. W = 100k
    leaves only ≈ 100k windowed steps in a 207k-step pass.
- **Q5 — MIND only; the Forest window is decided in 6d** (user, Nov 18). The 6b version is the "−LLM controller" ablation,
  and ablations run on MIND E3 + E4 only (Nov 17). Stationary Forest data can't tune a window, since the biggest one always
  wins. The full model's Forest W is set in 6d, where the controller chooses it.
- **Q6 — full E3 + E4 runs now, in 6b** (user, Nov 18): the 7 sizes of R1, 10 seeds each, paired E3 + E4, so 140 runs
  (≈ 3.5–4 h). They are reused in 6e as the "−LLM controller" ablation (NB1 + window, fixed γ = 0.005). Why: we learn
  whether the window helps before building the controllers. Caveat: if 6d changes the full model's starting W or γ,
  this ablation is re-run to match.
- **Q7 — checkpoint before the full runs, like 6a** (user, Nov 18): tuning table + chosen W shown first. §7 is the done-check.

**Fresh-context review (Opus, Nov 18) — R1–R18**
- Decisions (user): **R1** run 7 sizes on both scales (§6); **R2** smallest W within 2 × SE of the best (Q4);
  **R3** the sweep is closed by R1, and the use of E4 knowledge is disclosed (§6); **R4** gradual forgetting gets its
  own plan later (Q3b).
- Plain corrections applied: R5 wording of the retrain; R6 forward pass recomputed during replay; R7 notebook as
  arrays plus an RSS measurement; R8 exact schedule; R9 seed-free conversion 9.868 and R per size; R10 seven more tests;
  R11 cost assumptions stated; R12 saved-run keys; R13 labels and headline comparison; R14–R18 literature wording.

**Open questions: none.** Approved by the user on 2026-11-18 after the walk-through.
