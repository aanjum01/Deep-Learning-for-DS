# Phase 5d — E4: planted shifts on MIND + baselines + a sourced adaptation measure

**Status: DONE (2026-11-17).** Done-checks 1–7 pass; 60 E4 runs (2 h 32 min); results in RESEARCH_LOG §6–§7.
Caveat: the 500k tv spike sits inside shift 1's window and drives LinUCB's negative clicks lost there (§7).
Previously: APPROVED (user, Nov 17), rev. 4. Order (user): build → `/simplify` + `/code-review` → MIND-small
rehearsal (S5) → **CHECKPOINT with the user** → full E4 run + Forest Cover E2 re-measure → report.
(Plan rev. 4 — Decided with the user: Q1–Q15 (planning),
R1–R11 (first fresh-context review), S1–S8 (second fresh-context review) — §8. §3–§7 reflect all of them.
Approved.)
**Order (user, Nov 16):** after 5c, before Phase 6's results, so E4 is fixed before the method is seen.

## 1. Goal and why

Design E4 — the same real MIND week with **concept shifts planted at known times** — run every baseline on
it, define how adaptation after drift is measured (from a published method), and apply the same measure to
Forest Cover's drift runs, so Phase 6 is judged on a fixed, blind, pre-registered test.

| Why | Source |
|---|---|
| Main goal: "improve adaptation time after drift"; metrics: cumulative reward, dynamic regret, adaptation time | Approved abstract (Oct 5) |
| "Adaptation time after drift" is a listed evaluation metric; "a good bandit must recover quickly after the environment changes" | Course slides 67, 60 |
| E4 = main MIND adaptation test (natural drift has no clean start moment) | Phase 5 Q24 (user) |
| Baselines run in every environment the method runs in | Slides *Baselines*; scope Nov 13 |
| No formal adaptation-time definition existed (paper: "75k–350k" without a criterion; Phase 4 used an ad-hoc measure) | RESEARCH_LOG §5 open item |

## 2. What the sources say

- **Base paper (§5, pp. 6–7):** drift = classes "swapped in a circular cycle" every 500k steps; model-selection
  variants need "between 75.000 and 350.000 in the worst case to stabilize" — no criterion given.
- **Shaker & Hüllermeier 2015 (R10)** — *recovery analysis* (read Nov 17, authors' draft PDF; C29–C31):
  semi-synthetic drift on real data; for classification a "cyclic shift of the class labels, replacing label
  i by 1 + i mod 5" (§5.2); drift speed slow / modest / **sudden** ("concept shift", §5); reference curves =
  the same learner on the pure streams before (α) and after (β) the change (§3.1); **duration** (t₂ − t₁)/T
  with t₁ = drop below 95% of the pre-change curve, t₂ = recovery to 95% of the post-change curve (§3.3);
  **maximum performance loss** = max_t (S(t) − S_C(t))/S(t), S = min(α, β) (Eq. 8); test-then-train
  evaluation; **curves averaged over repeated runs** (§3.5); evaluation window chosen per dataset (§5, Table 2).
- **Our setting:** relabelling MIND's 16 arms consistently (rewards and availability) from the start is a pure
  renaming for our arm-agnostic per-arm learners, so their pre- and post-change references coincide **in
  expectation** with the same seed's E3 run (not step by step) — hence we compare **seed-averaged** curves (R1).
  E3 contains the same natural daily drift, so E4 − E3 isolates the planted shift.
- Arms (sorted): autos, entertainment, finance, foodanddrink, health, kids, lifestyle, movies, music, news,
  northamerica, sports, travel, tv, video, weather. Day starts ≈ 151k, 316k, 679k, 1.05M, 1.41M, 1.74M steps;
  each seed's run ≥ 2.03M steps (5c; checked by the reviewer).
- **Evidence gathered during planning (Nov 17):** (a) saved Forest Cover stationary vs drift runs are identical
  for the first 500k steps and differ right after, for every algorithm × 10 seeds; (b) per-seed stopwatch on
  Forest Cover makes Random "dip" after every drift (noise), while on 10-seed-averaged curves Random never dips
  (min 95.7–97.1%) and the learners' dips remain; (c) averaged Forest Cover curves: LinUCB, NB1 never back to
  95% within the next drift; NB2 after drift 1 only (220k steps); NB3 after drifts 1–3 (339k, 368k, 442k);
  (d) reviewer's LinUCB seed-0 E4 trial with the planned roll: first 500k rewards bit-identical to E3; shifts
  1/2/3 lost 809 / ~16k / ~31k clicks, the third unrecovered.

## 3. Design

| Piece | Decision | Ref. |
|---|---|---|
| Shift | after shift j (j = number of shifts at or before the step), arm a receives the rewards **and** availability of category (a + j) mod 16; the context is unchanged | Q1, Q3 |
| Speed / schedule | sudden; 3 shifts; 0-based steps **300,000**, 1,000,000, 1,500,000 are the first steps of each new arrangement; same for every seed. Times from a data-only rule fixed before any E4 run: no shift within ±50k steps of a natural jump (a category's click rate when shown differs by ≥ 0.15 between the 50k steps before and after); latest passing time ≤ 500k → 300k; 1M and 1.5M pass | Q2, Q4, R7, S1 |
| Curves | per algorithm, the **10,000-step trailing mean reward** of each seed (point at step t = mean of steps t−9,999…t), each seed cut to the shortest run (2,042,231), then **averaged over seeds 0–9**; E4 mean ÷ E3 mean = the ratio curve | Q6, R1, R7, S6 |
| Adaptation time (per shift) | on the ratio curve from the shift step on: t₁ = first step with ratio < 0.95; t₂ = first step from which ratio ≥ 0.95 for 10,000 consecutive steps, the stretch ending before the next shift / end of data. Report **duration t₂ − t₁** (R10) and **time since shift t₂ − shift**; never below → 0; not recovered → "> (end − t₁)" and "> (end − shift)" | Q5, Q7, R1, R4, R7 |
| Max performance loss (per shift) | max over the stretch of (1 − ratio), on the averaged curves (R10 Eq. 8 with E3 as reference) | Q8, R1 |
| **Clicks lost (per shift) — the deciding number** | per seed: Σ(E3 reward − E4 reward) over exactly the 500,000 steps after the shift (windows 300k–800k, 1.0M–1.5M, 1.5M–2.0M; signed, never clipped); reported mean ± std over seeds, Random as noise floor | Q8, R2, R4, R7, S1, S3 |
| **"Adapts better" (pre-registered)** | per shift: fewer clicks lost by > 2 × combined standard error **and** at least as many total E4 clicks in the same window; else "no clear difference" or "trades normal performance for robustness" | R2, S2, S3 |
| Also per shift | E4 click rate (absolute); "entered at X% of normal" (averaged ratio just before the shift) | R2, S4 |
| Reporting | every table/figure per shift; shifts never averaged together | R4 |
| Forest Cover E2 | same measures on the saved Phase 2–4 drift runs vs the same seed's stationary runs; drifts at 500k, 1M, 1.5M, 2M (the last has 250k steps); clicks lost over the available stretch (500k, last 250k) | Q9 |
| Algorithms / settings | Random; Popularity (E3's click shares — "knows the old world, never adapts"); LinUCB α 0.5; NB1 C 1, λ 1; NB2/NB3 full paper grid; γ 0.005, γ_model 0.1; d 19 — the 5c settings, no re-tuning | Q10, Q11, R3 |
| Seeds | 0–9 (each E4 seed pairs with the E3 seed of the same number) | Q12 |
| Trial (full rehearsal) | before the full run: MIND-small, seed 100, fresh no-shift + shifted runs for all 6 algorithms, shifts 50k / 100k / 150k, clicks lost over 50k windows; checks mechanics + every output; nothing tuned from it | R5, S5 |
| Statistics | clicks lost mean ± std; adaptation time and max loss have no ± (averaged curve; bootstrap possible later); significance tests in the later statistics step | Q15, S7 |

## 4. Files

| File | What it does |
|---|---|
| `evaluation/environments.py` | `mind_stream(data, rows, shifts=None)`: for stream step t (0-based) with j = #shifts ≤ t, yields `x`, `np.roll(R[i], -j)`, `np.roll(avail[i], -j)` (arm a → category a + j); `shifts=None` = E3 exactly as before. Drift stays in the environment (CLAUDE.md) |
| `evaluation/runner.py` | passes `shifts` to `mind_stream`; Popularity's shares stay `R[rows].mean(0)` of the **unshifted** data (R3); `best = R[rows].max(1)` is unchanged by the roll |
| `scripts/run_experiment.py` | config key `shifts:`; added to the saved-run key **only when set**, so E3's saved runs keep their key (R10) |
| `evaluation/metrics.py` | `trailing_mean(r, w=10_000)`; `recovery(ratio, shift, end, w=10_000, thr=0.95)` → t₁, t₂, both times / "> X" bounds, max loss; `clicks_lost(e4, e3, shift, length)` |
| `configs/phase5d_mind_e4.yaml`, `configs/phase5d_trial*.yaml` | E4: env mind_large, `shifts: [300000, 1000000, 1500000]`, 6 algorithms, 5c settings, seeds 0–9. Trial: env mind_small, seed 100, a no-shift config and a `shifts: [50000, 100000, 150000]` config |
| `scripts/make_recovery.py` | E4 regret figure, recovery figure, per-shift recovery table, results table; Forest Cover E2 recovery figure + table (from saved runs) |
| `tests/test_drift.py` | toy stream: arm a serves category a + j after shift j; `shifts=None` unchanged; stopwatch on synthetic ratio curves with known answers (dip, short blip that must not count, never below → 0, never recovers → "> X"); clicks lost sign |

## 5. Settings and sources

| Setting | Value | Source |
|---|---|---|
| Shift | cyclic relabel, cumulative +1 | base paper §5 p. 6; R10 §5.2 (C31); Q1, Q3 |
| Speed / times | sudden; 300k, 1M, 1.5M | data-only calm-time rule (S1, ours); R10 §5 (sudden), §3.5 (stream must not end mid-recovery); Q2 |
| Threshold | 95% | R10 §3.3 (C30); Q5 |
| Window | 10,000-step trailing | **ours** — Phase 4's recovery measure (R10 sets its window per dataset); Q6, R8 |
| Curve averaging | mean of 10 seeds before measuring | R10 §3.5; R1 |
| Recovered | ≥ 95% for 10,000 consecutive steps, before the next shift | **ours** (R10 only says "recovers up to 95%"); Q7, R7 |
| Reference | the same seed's E3 run for both pre- and post-change curves | **ours** — valid in expectation for arm-agnostic learners (§2); R8 |
| Deciding rule | fewer clicks lost (> 2 × combined SE) and ≥ total E4 clicks, per shift | R2, S2, S3 (user, fixed before Phase 6; ours) |
| Algorithms / settings | as 5c | Q10 |
| Seeds | 0–9 | Q12 |

Compute: the 60 E4 runs ≈ 2.5 h on 8 cores (as 5c); trial = minutes; Forest Cover E2 analysis = minutes.

## 6. Outputs (Q13, R4)

`mind_e4_regret.png` (6 lines; shifts solid, day boundaries dotted); `mind_e4_recovery.png` (one panel per
shift: averaged ratio curve per algorithm, 95% line); per-shift recovery table for E4 and for Forest Cover E2
(clicks lost mean ± std with Random as the noise floor and the "adapts better" verdict, adaptation time —
both kinds, "> X" when not recovered — max loss, E4 click rate, "entered at X% of normal", exact step ranges)
as CSV + RESEARCH_LOG; `forest_e2_recovery.png`; E4 results table (click rate, regret per algorithm).

## 7. Done-check (Q14, updated by R1/R3/R5)

1. Tests pass (`tests/test_drift.py` as §4; all existing tests incl. Forest Cover unchanged).
2. Trial (R5, S5) runs end to end on MIND-small: each pair identical for the first 50k steps, learners dip
   after shift 1, every output produced.
3. **E4 = E3 exactly before the first shift:** first 300,000 rewards identical to the saved E3 run, every
   algorithm (Popularity included) × seed.
4. All 60 E4 runs complete; no NaN/inf.
5. Random shows no dip on the averaged curves at any shift (adaptation time 0).
6. Every learner (LinUCB, NB1, NB2, NB3) dips below 95% on the averaged curves after shift 1 (re-checked
   under the new times, S1).
7. All §6 outputs + RESEARCH_LOG entry; `/simplify` + `/code-review` before the E4 runs.
Findings (not pass/fail): clicks lost per shift and method; who recovers, how fast; vs the paper's
"75k–350k"; the new Forest Cover E2 numbers vs Phase 4's ad-hoc ones (R9).

## 8. Decisions (user, Nov 17 unless noted)

**Planning questions**
- **Q1** cyclic arm relabelling (rewards + availability; context unchanged). **Q2** sudden. **Q3** cumulative
  +1 per shift (each shift a new arrangement; known quirk: rare categories rotate too, e.g. after shift 1 the
  arm "news" serves northamerica). **Q4** 3 shifts at 500k / 1M / 1.5M, same for all seeds; README's "random
  times" wording to update.
- **Q5** adaptation time = R10's duration against each algorithm's own E3 run. **Q6** 10k-step trailing
  window (same as Phase 4). **Q7** recovered = ≥ 95% for one full window; never below → 0; not recovered →
  "> X", never a guessed number *(per-seed computation superseded by R1)*.
- **Q8** also max performance loss and clicks lost. **Q9** re-measure Forest Cover E2 (no new runs).
- **Q10** same 6 algorithms, 5c settings, no re-tuning; **rule for Phase 6: tune only on MIND-small without
  shifts**. **Q11** include Popularity as a "cannot adapt" reference *(shares fixed by R3)*. **Q12** seeds
  0–9 (forced by Q5). **Q13** all outputs. **Q14** done-check §7 + a fresh-context Opus plan review before
  approval. **Q15** mean ± std now, tests later (5c precedent).

**First fresh-context review (Opus) — R1–R11**
- **R1** measure adaptation time and max loss on the **10-seed-averaged** curves (R10 §3.5); clicks lost per
  seed. Verified: per-seed stopwatches catch noise; averaging removes it.
- **R2** per shift: **clicks lost decides** "adapts better" (fixed before Phase 6), plus adaptation time and
  the E4 click rate.
- **R3** Popularity uses E3's (unshifted) shares — "knows the old world, never adapts".
- **R4** per-shift reporting, never averaged across shifts; two times (t₂ − t₁ and t₂ − shift); clicks lost
  signed.
- **R5** quick trial on MIND-small (seed 100, shifts 50k/100k/150k) before the full run; nothing tuned.
- **R6** Phase 6 rules (also in RESEARCH_LOG): paired E3 + E4 runs per seed for the method and each ablation
  (budget runs and LLM cost); cache/replay LLM answers; shift times never visible to any method; the LLM sees
  arm numbers / neutral labels, not category names (or a stated limitation).
- **R7** exact counting rules (as §3): 0-based steps; trailing curves read from the shift step; recovery stretch
  ends before the next shift; "> X" bounds for both times; clicks lost over exactly 500k steps.
- **R8** (correction, no decision needed) the 10k window, the "stays recovered" rule and using E3 as both
  references are marked as **ours** in §5, not R10's.
- **R9** (information) the new Forest Cover E2 numbers will differ from Phase 4's ad-hoc ones (95% of a still-
  improving stationary run is a stricter bar); both definitions explained side by side, old numbers kept as history.
- **R10** (correction) `shifts` enters the saved-run key only when set, so E3's saved runs keep their key;
  done-check 3 depends on the local (gitignored) E3 files in `results/raw/phase5c_mind_e3/`.
- **R11** (correction) one sentence in the paper: the 1M shift sits ~50k steps before the Nov 12 day boundary;
  the E3 reference absorbs the natural change.

**Process (user, Nov 17):** this round answered → plan rewritten (this rev. 3) → **second fresh-context Opus
review** → approval → build → R5 trial. Trial finds a bug → fix the code; finds a design problem → back to the
user (and a focused re-review if large).

**Second fresh-context review (Opus, Nov 17) — S1–S8**
- **S1 — DECIDED (user, Nov 17): shifts at 300k, 1M, 1.5M, chosen by a data-only rule fixed now (blind to
  every method): no shift within ±50k steps of a natural jump, where a jump = some category's click rate
  (when shown) differs by ≥ 0.15 between the 50k steps before and after; among passing times ≤ 500k take the
  latest.** Why: the week's biggest natural event — tv click rate 0.08 → 0.52 → 0.14 over ≈ 460k–540k, same in
  every seed (verified seeds 0, 3, 7) — sat exactly on the old shift 1 (500k): in E3, LinUCB drops to 0.335 at
  510k while NB1/2/3 rise to ~0.52, and the relabelling pushed stale LinUCB onto the tv-serving arm (reviewer:
  +4,471 clicks windfall), biasing clicks lost against methods that track real drift. Scan (full data, 10k
  grid, run-step ≈ 0.9 × row): jumps ≥ 0.15 near 160–200k, 360–380k, 460–540k, 640k, 780–820k, 1.24–1.26M,
  1.94–1.96M; passing ≤ 500k: 60–100k, 270–300k; 1M and 1.5M pass. Clicks-lost windows 300k–800k, 1M–1.5M,
  1.5M–2.0M. The tv spike now lies inside shift 1's window (160–240k after it), shared with E3 — one sentence
  in the paper. Spacing is uneven (paper: every 500k), stated with the reason. *Supersedes Q4's times.* The R5
  trial keeps 50k / 100k / 150k on MIND-small (mechanics only).
- **S2 — DECIDED (user, Nov 17): "adapts better" (per shift) requires BOTH fewer clicks lost AND at least as
  many total E4 clicks in the same 500k window**; winning only the first = "trades normal performance for
  robustness", reported as such, not a win. Why: clicks lost is relative to each method's own no-shift run —
  Random loses ~0 and would "adapt best"; always-exploring or always-forgetting methods lower their own bar.
  *Refines R2.*
- **S3 — DECIDED (user, Nov 17): pre-registered margin — "A adapts better than B" on clicks lost only if the
  difference of their 10-seed means exceeds 2 × the combined standard error √(s_A²/10 + s_B²/10); otherwise
  "no clear difference".** Random's clicks lost is shown in every table as the noise floor (reviewer: Random
  per-seed clicks lost after shift 1 ranged −875 … +706, std ≈ 474). Formal tests still in the later
  statistics step.
- **S4 — DECIDED (user, Nov 17): carry-over made visible** — per shift, an "entered at X% of normal" column
  (the averaged E4/E3 ratio just before the shift: 100% = clean test; lower = still hurt from the last shift);
  clicks-lost windows labelled with their exact step ranges; paper states that shift 1 is the clean comparison
  while shifts 2–3 also include leftover damage for methods that hadn't recovered.
- **S5 — DECIDED (user, Nov 17): the R5 trial is a full rehearsal** — fresh no-shift AND shifted runs on
  MIND-small seed 100 for all 6 algorithms (same code), shifts 50k / 100k / 150k, clicks lost over 50k
  windows; checks: first 50k rewards identical in each pair, learners dip after shift 1, `make_recovery.py`
  produces every output. Nothing tuned from it. (Old seed-100 tuning runs not reused: different keys/dirs, and
  none exist for NB2/NB3.)
- **S6 — DECIDED:** before averaging, every seed is cut to the shortest run (2,042,231 steps); all measures end
  by 2.0M, so nothing is lost.
- **S7 — DECIDED:** adaptation time and max loss come from the averaged curve → one number per method × shift,
  no ± over seeds (stated in the paper); a bootstrap over seeds can add spread in the later statistics step.
  Only clicks lost carries mean ± std.
- **S8 — DECIDED:** rotating availability can make an arm almost unplayable (it inherits kids / northamerica,
  ≈ 0% shown), forcing exploration — unlike Forest Cover (all arms always available). Keep the design; one
  sentence in the paper and RESEARCH_LOG; E4 and Forest Cover E2 adaptation times are not compared
  like-for-like.
- **Tidy-ups:** done-check 6 (learners dip after shift 1) re-checked under the new shift times (trial + run);
  durations are reported in steps (R10 divides by stream length T).
