# Phase 6b part 2 — NB1 + gradual forgetting (no rebuilds, no controller)

**Status: DONE (2026-11-18).** Results: RESEARCH_LOG (Phase 6b part 2). Pre-registered pairwise rule: window. **User decision
(Nov 18): fading is NeuralBandit++'s forgetting method**, judged against the paper's goal (RESEARCH_LOG §5). *Was:* **APPROVED (user, 2026-11-18), rev. 3.** Q1–Q16, review R1–R17 and the second review (rules) settled; walk-throughs given.
Order: build → `/simplify` + `/code-review` → tests → rehearsal → tune → **CHECKPOINT** → 160 runs → outputs + report.

## 1. Goal and why

Give NB1 a *gradual* way to forget: after every step its weights are pulled a small fraction back toward their
starting values, so old lessons fade unless recent data keeps reinforcing them. Compare it with plain NB1 and with
6b's window (abrupt forgetting) on MIND E3 + E4.

| Why | Source |
|---|---|
| The abstract's memory component: "train on only the most recent, relevant data"; the memory-manager agent "decides how much old data to forget" | Approved abstract (Oct 5) |
| The two published ways to forget are abrupt (windows) and gradual (fading old data); "The more abrupt is forgetting, the faster the reactivity, but also the higher the risk of capturing noise". No comparison is reported | gama2014 §3.1.2 (C45) |
| 6b: the window makes the shifts nearly free, but costs 1–3 points of clicks and can fall into the **dead-arm trap** after a rebuild; the 493k result flipped when the rebuild timing moved | RESEARCH_LOG §6 (Phase 6b, timing check) |
| Gradual forgetting never rebuilds the network, so the *rebuild-induced* trap and rebuild timing don't apply (the trap metric is still measured) | Mechanism (§5); review R8 |
| Its result decides what the 6c/6d controller sets: a window size or a fade rate | User, Nov 18 |
| Tune only without planted shifts; paired E3 + E4 runs; shift times never visible | Phase 5d Q10, R6 |

## 2. What the sources say

Read in full 2026-11-18 (REFERENCES_VERIFICATION R13–R16, C45–C46, C49–C54).

- **The method has a name and a published precedent: "L2 Init"** [kumar2024, §3, App. A.3; C50]. It pulls every parameter
  toward its *own initial value* at every step. With SGD it is θ ← θ − ρ(θ − θ₀) − α∇L, ρ = step × penalty. Ours is
  L2 Init's SGD form up to an O(αρ) term (we apply NB1's step and then the fade, which also scales the gradient by 1 − ρ),
  applied at every interaction including steps with no gradient. The difference is negligible at ρ ≤ 1e-4 (review R8). They call it an "implicit, smooth reset" and apply it to all parameters.
- **Toward θ₀, not toward zero:** plain L2 / weight decay (toward 0) fails on several of their problems; pulling toward
  freshly resampled weights is much worse than toward the fixed θ₀ [C52]. Shrink & Perturb [ash2020; C53] shrinks toward
  zero and adds noise. It also helps, but is "consistently lower" than L2 Init [C52], and Dohare et al. found S&P sensitive to its noise
  level [C54].
- **Tuning precedent:** they tune the rate on a log grid {1e-2…1e-5}, 3 seeds, by total online accuracy, then run 10
  seeds [C51]. Same procedure as ours (3 → 10 seeds), different grid. The best rate is problem-dependent [C51].
- **Small networks:** they chose narrow nets on purpose because plasticity loss is worse with less capacity [C51].
  Dohare et al. see it even with 5 hidden units [C49].
- **Abrupt vs gradual forgetting:** "The more abrupt is forgetting, the faster the reactivity, but also the higher the
  risk of capturing noise"; no empirical comparison is reported [gama2014 §3.1.2; C45]. 6b part 2 supplies one.
- **Not in the sources:** none of them is a bandit, and none uses MIND or a one-hidden-unit network; whether L2 Init helps NB1 is
  what this phase tests.

## 3. Measured cost (throwaway prototype, one job alone, seed 100, Nov 18)

Prototype: plain NB1 plus, after every step, θ ← θ − ρ(θ − θ₀) on both layers of every arm's network.

| Run | Click rate / accuracy | µs/step | mean abs(w2) at the end |
|---|---|---|---|
| MIND-small NB1 (C 1, λ 1), no forgetting | 0.4448 | 38 | 2.31 |
| ρ = 1e-5 (memory ≈ 100k steps) | 0.4444 | 36 | 1.62 |
| ρ = 2e-5 (≈ 50k) | 0.4425 | 39 | 1.42 |
| ρ = 5e-5 (≈ 20k) | 0.4374 | 39 | 1.23 |
| ρ = 1e-4 (≈ 10k) | 0.4291 | 40 | 1.02 |
| ρ = 1e-3 (≈ 1k) | 0.3508 | 37 | 0.61 |
| Forest 200k, NB1 (C 100), no forgetting | 0.7127 | 70 | — |
| Forest, ρ = 1e-5 | 0.7070 | **184** | — |

- **MIND:** forgetting costs ≈ 1 µs per step (16 × 19 + 16 weights), so runs cost the same as plain NB1: ≈ 3 min per
  MIND-large run with 8 jobs, ≈ 10 min per E3 + E4 batch of 20 runs.
- **Forest:** ≈ 2.6× slower (66k weights to fade every step); relevant only if Forest is in scope (Q).
- **One seed, calm data:** forgetting costs clicks, and more so the faster it is, like the window. At a similar memory length it
  scored a bit above the window (≈ 50k: 0.4425 vs W 50k 0.4392 3-seed mean; ≈ 10k: 0.4291 vs W 10k 0.4257). Indicative only.
- "Memory ≈ 1/ρ" is a rough guide: a weight's offset from θ₀ halves every ≈ 0.69/ρ steps if no update reinforces it.

## 4. Files (Q16)

| File | What | Who |
|---|---|---|
| `nonstationary/forgetting.py` | `fade_toward_init(net, W1_0, w2_0, rho)`: in place, W1 −= ρ(W1 − W1₀), w2 −= ρ(w2 − w2₀) | Opus (me) |
| `algorithms/neuralbandit.py` | `NeuralBandit1Fade(NeuralBandit1)`, param `rho`: copies θ₀ in `_init_state`; `update` = `super().update` then the fade (every step, all networks); records the 3 traces per 1,000 steps; `rho` is a plain attribute | Opus |
| `tests/test_forgetting.py` | See §7 | Opus |
| `scripts/run_experiment.py` | Register the class | Opus |
| `configs/phase6b2_*.yaml` | `tune_mind` (MIND-small, seeds 100–102: NB1 + 4 rates), `trial` / `trial_noshift`, `mind_e3` / `mind_e4` (8 rates, seeds 0–9) | Sonnet agent, checked by me |
| `scripts/make_recovery.py` | Setup `e4_6b2` (the 8 fade labels next to the baselines and the 6b windows), `trial_6b2`; the per-window figure extended to fade-vs-matched-window rows | Sonnet agent, checked by me |

Labels: `NB1-Fade_<memory>`, with memory = 1/ρ: count-based `NB1-Fade_10k/20k/50k/100k`; time-span `NB1-Fade_99k/197k/493k/987k`.
Phase 5 and 6b outputs stay unchanged.

## 5. Design (decided, §8)

- **Update (Q10, Q11, Q2, Q3):** at the end of every `update`, after NB1's own Eq. 1 step (if any), every category's
  network (both layers) does θ ← θ − ρ(θ − θ₀), where θ₀ is an exact copy of that network's initial draw [kumar2024, L2 Init; C50].
- **No randomness, no stored data:** the fade uses no rng draws, so action choices match plain NB1's until the weights
  differ; E3/E4 twins are identical before the first shift. Memory = one copy of θ₀ (≈ 320 numbers on MIND).
- **ρ = 0 is plain NB1 bit-for-bit.** ρ can be changed during a run (Q12).
- **Traces (Q9):** per 1,000 steps: mean |W1|, mean |w2|, and the share of picks whose chosen category's estimate is within 0.02 of its own starting estimate (R7).
  The trap count is computed afterwards from the saved clicks.
- **Not in 6b-2:** Forest (Q1), any controller, any change to γ or other NB1 settings (Q13).

## 6. Settings and sources

| Setting | Value | Source |
|---|---|---|
| NB1 on MIND | C 1, λ 1, γ 0.005, d 19 | Phase 5c (user-confirmed Nov 16); Q13 |
| Forgetting | θ ← θ − ρ(θ − θ₀) every step, all networks, both layers, after NB1's update | Q2, Q3, Q10, Q11; kumar2024 §3, App. A.3 (C50) |
| Tuning grid (MIND-small) | ρ ∈ {1/10k, 1/20k, 1/50k, 1/100k} | Q4 |
| Tuning | Seeds 100–102, one pass, no shifts; headline = best 3-seed mean click rate (exact ties → faster fade) | Q5; kumar2024 §5 (C51); Q10 rule (5d) |
| E3/E4 rates | 8: the 4 tuned values (count) + the same ÷ 9.868 (time span: 1/99k, 1/197k, 1/493k, 1/987k) | Q6; 6b's seed-free ratio |
| Headline | The time-span version of the tuned rate | Q6 |
| Comparisons | vs NB1 and vs the 6b window of the same memory; NB2/NB3/LinUCB in the tables | Q8 |
| Decision rule | Trade-off rule: a fade beats a window if not worse on (E3 click rate, mean clicks lost per shift) and better on one (2 × unpaired SE); fading is 6c/6d's knob if it beats ≥ 2 of the 3 small windows (10k, 20k, 50k); the larger windows are reported only (trap-distorted clicks lost); NB1 placebo must fail | Second review, decisions 1–2 |
| Report runs | MIND-large E3 + E4, seeds 0–9, shifts 300k / 1M / 1.5M | Phase 5d |
| Parallel jobs | 8 (no notebook; memory as plain NB1), confirmed in the rehearsal | §3; Q14 |

## 7. Done-check

**Pass/fail**
1. **Tests** (`tests/test_forgetting.py`), plus all existing tests:
   - the helper computes θ − ρ(θ − θ₀) exactly;
   - ρ = 0 gives plain NB1 bit-for-bit (same actions and rewards);
   - θ₀ is the initial draw and never changes during the run;
   - every network fades every step, including steps where NB1 doesn't learn (explored arm);
   - with no data, the weights decay geometrically: θ − θ₀ shrinks by (1 − ρ)ⁿ;
   - changing `rho` mid-run takes effect at the next step;
   - masking holds; same seed → identical rerun; `reset()` replays exactly and re-copies θ₀;
   - the trace lengths are right.
2. **Rehearsal (Q14):** the MIND-small pair is identical before 50k; `make_recovery.py trial_6b2` produces every output;
   traces are present. One MIND-large run alone: time and peak memory noted.
3. **Checkpoint (Q7):** tuning table (NB1 + 4 rates × 3 seeds), headline rate and the 8 E3/E4 rates shown; **no full run before
   the user's go.**
4. **Full runs:** all 160 complete and finite; E4 = E3 bit-for-bit for the first 300,000 steps.
5. **Outputs:** fade lines added to the E3/E4 tables, recovery table, verdicts (vs NB1 and matched windows), the
   fade-vs-window figure, trap counts and traces; RESEARCH_LOG, README and CLAUDE.md updated; Phase 5 / 6b outputs unchanged.

6. **Placebo:** plain NB1 scored by the same trade-off function beats < 2 of the 3 small windows (else stop: the rule is broken; on the 6b numbers it beats 0).
7. **Pre-registration:** dated research-log entry with the decision rule, written before the 160 runs.

**Findings (not pass/fail):**
- The trade-off plot (E3 click rate vs mean clicks lost, 7 windows + 8 fades + NB1) and the decision.
- Does fading flatten NB1's 11k → 45k → 113k slowdown, as the small windows did?
- Clicks lost per shift vs NB1.
- Trap counts (expected ≈ 0).
- The near-50% pick rate (the side agent's concern).

**Order:** build → `/simplify` + `/code-review` → tests → rehearsal → tuning (≈ 2 min) → **checkpoint** → 160 runs (≈ 1–1.5 h) →
outputs + report.

## 8. Decisions

**Planning questions (asked one at a time)**
- **Q1 — MIND only** (user, Nov 18). Same scope as 6b: this is part of the "−controller" ablation family, and ablations run
  on MIND E3 + E4 only (Nov 17). Stationary Forest can't tune a forgetting rate, and fading is 2.6× slower there (C 100).
  The full model's Forest setting is decided in 6d.
- **Q2 — every category's network fades every step** (user, Nov 18), whether or not it learned that step. Forgetting is
  by time: a category not seen for a while drifts back toward its start (≈ 50% guess) and is re-tried when shown, with
  its learning signal intact.
  *Checked after a side-agent note (faded categories drift to ≈ 50% and may look falsely attractive):* MIND-small, seeds
  100–101. Fading all categories scored **higher** than fading only the category that learned (scaled × K to give the same
  average fade): ρ 1/50k 0.4425 / 0.4437 vs 0.4335 / 0.4376; ρ 1/10k 0.4291 / 0.4329 vs 0.4072 / 0.4135. It does pick
  near-50% categories more often at the slower rate (≈ 19.6% vs 11.3% of picks; a rough measure), but stale estimates for
  rare categories cost more. Decision kept.
- **Q4 — fade-rate grid matched to the window grid** (user, Nov 18): ρ ∈ {1/10k, 1/20k, 1/50k, 1/100k} on MIND-small.
  These are the same memory lengths as 6b's windows, so the comparison is like-for-like. L2 Init's grid {1e-2…1e-5} is not copied:
  it is a penalty multiplied by a small optimiser step per mini-batch, a different unit from our per-interaction ρ
  [C50, C51]. 1e-3 is already far too fast here (prototype: 0.3508). Their tuning *method* is copied (3 seeds, best total
  online score, then 10 seeds).
- **Q5 — paper's rule for the headline, and all 4 rates run on E3 + E4** (user, Nov 18, after asking why not the paper's
  rule). Headline = the best 3-seed mean click rate on MIND-small (L2 Init's rule [C51]; exact ties → the faster fade).
  Every rate is also run on E3 + E4 and reported; none is picked by its E4 result. Why: our tuning data has no planted
  shifts (fairness rule), so "best mean" measures calm-data performance only. A different tie-breaker ("fastest within
  noise", 6b's rule) would be a judgement call, not a better estimator. Running every rate keeps the paper's standard
  rule and loses no information (runs cost ≈ 3 min each). Note: 6b used "smallest within noise" for its headline; the
  write-up states the difference.
- **Q6 — both scales on E3 + E4** (user, Nov 18), as 6b did for windows:
  - count-based: ρ ∈ {1/10k, 1/20k, 1/50k, 1/100k} as tuned (same memory in interactions);
  - time-span: the same ÷ 9.868, i.e. 1/99k, 1/197k, 1/493k, 1/987k (same memory in time; the factor is 6b's seed-free row ratio).
  8 rates × E3 + E4 × 10 seeds = 160 runs, ≈ 3 min each, so ≈ 1–1.5 h with 8 jobs. The headline follows 6b: the time-span version of
  the tuned rate.
- **Q7 — checkpoint before the full runs** (user, Nov 18): build → `/simplify` + `/code-review` → tests → rehearsal → tune →
  **STOP** (tuning table and headline rate shown) → user's go → 160 runs.
- **Q8 — compare each fade rate with plain NB1 and with the 6b window of the same memory** (user, Nov 18), using the
  pre-registered adapts-better rule. Pairs: 1/10k ↔ W 10k, 1/20k ↔ W 20k, 1/50k ↔ W 50k, 1/100k ↔ W 99k; 1/99k ↔ W 99k,
  1/197k ↔ W 197k, 1/493k ↔ W 493k, 1/987k ↔ W 987k. NB2, NB3 and LinUCB are shown in the tables as before. 6b's saved window
  runs are reused, so no extra runs.
- **Q9 — three extra records per run** (user, Nov 18):
  - **trap count**: 10k-step blocks below 0.25 click rate (6b's measure), computed from the saved clicks;
  - **weight-size trace**: mean |W1| and mean |w2| every 1,000 steps, which shows the fade working and spots a category stuck near its start;
  - **near-50% pick rate per 1,000 steps** (superseded by R7: "back near its own start"): originally ≥ 0.48 (the side agent's
    concern; a rough measure).
  The traces are saved through `diagnostics()`.
- **Q10 — fade toward the network's own starting weights θ₀** (user, Nov 18): an exact copy of the initial draw, kept
  for the whole run (L2 Init [C50]). Rejected: toward zero (plain L2 fails on several problems [C52]; all-zero sigmoid
  weights give flat 50% outputs with dead gradients, close to the trap), and toward fresh random weights (much worse [C52]).
- **Q11 — the fade is applied after NB1's own update, at every step** (user, Nov 18): NB1 learns as usual (Eq. 1 gate), then
  every network does θ ← θ − ρ(θ − θ₀). Steps where NB1 doesn't learn still fade (Q2: forgetting by time).
- **Q12 — ρ can be changed during a run** (user, Nov 18): it is a plain attribute, so a 6c/6d controller can raise it after a
  detected shift. In 6b-2 itself it stays fixed for the whole run.
- **Q13 — NB1 otherwise unchanged** (user, Nov 18): C = 1, λ = 1, γ = 0.005, d = 19 (Phase 5c, user-confirmed Nov 16). The fade is
  the only difference from plain NB1.
- **Q14 — rehearsal as in 6b** (user, Nov 18): MIND-small seed 100, ρ = 1/20k, one shifted run (50k/100k/150k) and its
  no-shift twin, identical before 50k; every output script produces its files; traces saved. Plus one real MIND-large run alone
  to measure speed and memory (≈ 3 min).
- ~~SUPERSEDED by the trade-off rule below~~ **Q15 — pre-registered reading of "gradual beats abrupt"** (user, Nov 18). For each pair with the same memory (Q8), gradual
  forgetting counts as the better method only if all three hold: (1) at ≥ 2 of the 3 shifts, the adapts-better verdict vs the window is
  "better" or "no clear difference", or the fade has more post-shift clicks (R3); (2) a higher E3 click rate than the window (difference > 2 × combined SE over 10 seeds);
  (3) no more trap blocks (10k-step blocks below 0.25) than the window, summed over E3 + E4 (R2: originally "fewer"; for the 493k pair, against both window schedules). Partial outcomes are reported as they fall.
  Fading vs plain NB1 is judged by the per-shift adapts-better verdicts, as in 6b.
- **Q16 — code split like 6b** (user, Nov 18): the helper goes in `nonstationary/forgetting.py` and the class `NeuralBandit1Fade` in
  `algorithms/neuralbandit.py`; labels `NB1-Fade_<memory>`.
- **Q3 — both layers fade** (user, Nov 18): the detector weights (input → hidden) and the dial (hidden → output).

**Fresh-context review (Opus, Nov 18) — R1–R17** (decision rules later replaced; see the second review). Plain corrections applied:
- **R8:** §2 now says "L2 Init's SGD form up to an O(αρ) term"; §1 says the *rebuild-induced* trap.
- **R9:** ρ = 0 is bit-for-bit (−0 signs are harmless); no rng draws; the near-50% trace reads the chosen arm's cached
  estimate `self._o[0, a]` from `select`, so there is no extra forward pass; old run keys are untouched.
- **R10:** configs write rates as decimals (e.g. `rho: 0.00002`, comment `# 1/50k`); a test checks that `rho` is a float.
- **R11:** time-span rates are exactly 1/99,000, 1/197,000, 1/493,000 and 1/987,000, so each matches its 6b window.
- **R12:** `evaluation/metrics.py` gets `trap_blocks(rewards)`: non-overlapping 10k blocks from step 0 with mean < 0.25.
  `make_recovery.py` computes Q15's checks with it; the SE for check 2 is unpaired across seeds, as in Phase 5d S3. Both get unit tests.
- **R13:** traces = a snapshot at each 1,000-step boundary of mean |W1| and mean |w2| over all 16 arms, plus the block's share
  of played actions whose estimate is back near its own start (R7). Length = ⌊n / 1000⌋. "Most shifts" = ≥ 2 of 3.
- **R14:** extra tests: fade order (update then fade); θ₀ copy uses no rng, so the first actions equal NB1's; a non-played
  arm fades exactly as with no gradient; trap-count and Q15 functions; an E4 = E3 first-300k check in code on all pairs.
- **R15:** the fade-vs-window pairing is passed explicitly to the figure code (no reliance on the SW list).
- **R16:** literature wording fixed (C51 "same procedure, different grid"; R14's read status; C54's grid not used).
- **R17:** cost confirmed: ≈ 180 s per run in parallel, so 160 runs ≈ 1 h; peak RSS noted in the rehearsal.

Decisions for the user:
- **R5 — keep all 8 rates** (user, Nov 18, after hearing the extra cost is ≈ 7 min): 1/100k and 1/99k are both run and both
  paired with W 99k, so 160 runs.
- **R4 — keep the paper's tuning rule** (user, Nov 18). Stated up front: on calm data it will very likely pick the slowest fade
  (1/100k, headline 1/987k ≈ plain NB1). The two headlines are **never** compared with each other; 6b's 493k "win" also
  did not survive the timing check. Window vs fade is judged only within matched pairs. Plain NB1 (ρ = 0) is shown in the
  tuning table as the reference, but it is not a candidate.
- **R6 — keep ρ = 1/W as a *nominal* match** (user, Nov 18). The write-up states that the remembered data is on average ≈ W
  steps old under a fade and ≈ W/2 under a W-window, so a fade remembers somewhat more at the same nominal memory. That
  gives it an easier time on Q15 check 2 (E3 clicks). Readers can also compare a fade with the next-smaller window, since all
  8 rates are run.
- ~~SUPERSEDED by the trade-off rule below~~ **R2 — trap check = "no more trap blocks than the window"** (user, Nov 18), summed over E3 + E4 runs. For the 493k pair
  it is checked against **both** window schedules (original and offset, `NB1-SW_W493k_off`) and passes only if it passes
  against both; both are reported. The 1/493k fade's verdicts are also shown against both 493k schedules.
- ~~SUPERSEDED by the trade-off rule below~~ **R3 — check 1 of Q15 also accepts the twin-free post-shift clicks** (user, Nov 18). At ≥ 2 of 3 shifts, check 1 passes if
  EITHER the adapts-better verdict (fade vs window) is "better" or "no clear difference", OR the fade collects more clicks than the
  window in the 500k steps after that shift (difference > 2 × unpaired SE over 10 seeds; 6b's cleaner check). Stated in the
  write-up: clicks lost alone is lopsided against fading versus small windows, which lose almost nothing (W 10k: −26 / 68 / 303).
- ~~SUPERSEDED by the trade-off rule below~~ **R1 — what decides 6c/6d's knob** (user, Nov 18): the controller sets a **fade rate** instead of a window size only if fading
  passes all three Q15 checks in **at least 5 of the 8 matched pairs**; otherwise the window stays. Fixed before the runs so
  no single pair (e.g. the timing-sensitive 493k) can decide it. Pairs 4 and 5 share the W 99k window.
- **R7 — "back near its start" pick rate replaces "near 50%"** (user, Nov 18). The trace is the share of played actions whose
  cached estimate is within 0.02 of that arm's own starting estimate for the same context: o₀ = sigmoid(w2₀[a] ·
  sigmoid(x·W1₀[:, a])), computed for the chosen arm only, with no rng. Why: initial outputs range ≈ 0.38–0.62, so faded
  arms return to *their own* start, not to 0.5.

- ~~SUPERSEDED by the trade-off rule below~~ **Pair-win rule revised after the walk-through** (user, Nov 18, asked "is this too strict?"): in each matched pair,
  fading **wins** if it is **not worse on any check and clearly better on at least one**. The checks:
  - shift handling: at ≥ 2 of 3 shifts, adapts-better verdict or post-shift clicks;
  - E3 click rate;
  - trap blocks.
  "Worse" = significantly fewer clicks (> 2 × unpaired SE) or more trap blocks (for 493k, vs either window schedule).
  "Better" = significantly more post-shift clicks at ≥ 2 shifts, or significantly more E3 clicks, or strictly fewer trap blocks.
  This replaces Q15's "all three, with check 2 a clear win", which favoured keeping the window, since a tie on E3
  clicks failed. R1 unchanged: fading must win ≥ 5 of 8 pairs.

**Second fresh-context review (Opus, Nov 18; user asked for a look at every rule's strictness) — findings 1–12**
- **Decision 1 (user): the trade-off rule replaces the pair rules (Q15, R1–R3, the revised pair rule).** Why: under the revised
  rule even **plain NB1 (no forgetting) would "win" most pairs**. It beats every window on E3 clicks by far more than the tiny 2 × SE
  margin (≈ 0.0003 for small windows), and the rule had no "worse" for slow adaptation. So it measured "forgets less", not
  "gradual vs abrupt". The new rule:
  - Each setting (7 windows from 6b, 8 fades) is a point: **E3 click rate** (higher = better) and **clicks lost per
    planted shift**, the mean over the 3 shifts per seed (lower = better; from the existing recovery computation).
  - A fade **beats** a window if it is not worse on either axis and better on at least one. Better/worse = a difference of the 10-seed
    means larger than 2 × unpaired SE, √(s₁²/10 + s₂²/10) with ddof = 1, two-sided; smaller differences count as ties.
  - Fading "beats a window" if **any** of the 8 fades beats it. ~~≥ 4 of 7 windows~~ → **6c/6d's controller sets a fade
    rate if fading beats ≥ 2 of the 3 trustworthy small windows (W 10k, 20k, 50k)**; otherwise the window stays (user, Nov 18).
    Why: a check on the real 6b points (below) showed the larger windows' clicks-lost values are distorted by the trap in
    their no-shift twins (W 99k −2,537, W 197k −3,197, W 493k-offset −10,038, W 987k −12,095 per shift, all beyond 2 × SE
    below zero), so no fade could beat them and "4 of 7" would always say "keep the window". The larger windows are still
    plotted and reported, with this reason.
  - Real 6b points (E3 click rate ± 2SE; mean clicks lost per shift ± 2SE): NB1 0.4597 ± 0.0002, 5,056 ± 162; W 10k 0.4329 ± 0.0002,
    115 ± 119; W 20k 0.4393 ± 0.0002, 254 ± 132; W 50k 0.4393 ± 0.0007, −368 ± 513; W 99k 0.4375, −2,537 ± 529; W 197k 0.4403,
    −3,197 ± 1,237; W 493k 0.4508, 425 ± 984 (offset 0.4263, −10,038 ± 1,890); W 987k 0.4331, −12,095 ± 3,368. Placebo on these
    numbers: NB1 beats **0** windows (more clicks on E3, but far more lost at shifts), so the rule is not broken.
  - Caveat, stated: the small windows lose almost nothing at shifts, so fading may only *tie* them. Then the outcome is "stay
    with the window", reported as "gradual forgetting matched but did not beat the window".
  - **Placebo, built into the code:** plain NB1 (ρ = 0, i.e. Phase 5c/5d's NB1 runs) is scored by the same function. It must not beat
    the small windows (it loses 3.4k–6.2k clicks per shift vs ≈ 0 for W 10k–50k). If it beats ≥ 4 windows, the rule is broken:
    stop and report.
  - **Trap blocks and post-shift clicks are reported** for every setting **but do not decide** (traps without a margin are noisy;
    post-shift clicks largely repeat the E3 axis). Trap blocks are counted on E3 runs and on E4 runs *excluding the first 10k
    steps after each shift*, so a shift's own dip isn't counted as a trap.
- **Decision 2 (user): 493k window = both schedules.** Its original and offset-schedule runs are two points. The 493k window counts
  as beaten only if some fade beats **both**. It is reported only, since the decision now uses W 10k, 20k and 50k.
- **Known bias, stated:** for windows ≥ 197k, clicks lost is distorted by the trap in the no-shift twin (it can even be negative,
  e.g. W 197k shift 1 −10,906). This makes those windows look *better* on that axis, so it is conservative against fading.
- **Plain fixes applied:**
  - Superseded rules are marked in place (finding 8).
  - Shift handling is defined only through clicks lost (findings 2, 9).
  - A check that can't be computed (missing seeds, NaN or zero SE) is a tie and is reported (finding 10).
  - Fade vs plain NB1 is **descriptive** (per-shift adapts-better verdicts, E3 cost, adaptation times), not a decision (finding 11).
  - Tuning ties: "within 1e-6 → faster fade" (finding 12).
  - Multiple comparisons: 8 fades × 7 windows × 2 axes. The ≥ 4-of-7 rule and the 2 × SE margin limit chance wins; the
    write-up states that the 8 fades are correlated (same seeds and base, a smooth sequence of rates), so they are not independent
    evidence (findings 5, 6). The 1/100k ≈ 1/99k duplicate no longer double-counts: fades are candidates, and only windows are counted.
  - **Pre-registration (finding 7):** the rule was set after seeing 6b's window results and the MIND-small fade prototype (never E4 fade
    results). A dated log entry is written before the 160 runs start. The "compare with the next-smaller window" reading (R6) is descriptive only.

**Open questions: none.** Approved by the user on 2026-11-18.
