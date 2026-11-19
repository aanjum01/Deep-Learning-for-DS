# Phase 6c — Page-Hinkley alarm + rule-based controller (sets the fade boost; shares its interface with 6d's LLM)

**Status: DONE (2026-11-19).** Results: RESEARCH_LOG §6 (Phase 6c). *Was:* **APPROVED (user, 2026-11-18), rev. 2.** Length menu (one memory length per boost rate) confirmed by the user. Written after the 6c/6d design discussion (Nov 18). Q1–Q10 decided;
fresh-context review round 1 (R1–R7) applied (§8). No building before approval.
Order: γ check → read Page 1954 → build → `/simplify` + `/code-review` → tests → measurement → thresholds (pre-registered) →
rehearsal → **CHECKPOINT** → 40 MIND-large runs → outputs + report.

## 1. Goal and why

Build the non-LLM controller for NB1 + gradual forgetting: NB1 always fades slowly; a Page-Hinkley (PH) alarm watches NB1's surprise
for a drop; on an alarm a fixed rule boosts the fade rate for a while, then returns to the slow rate. Fix the alarm → action interface
that 6d's LLM controller will reuse unchanged.

| Why | Source |
|---|---|
| Abstract: agents "monitor drift, decide how much old data to forget, and adjust exploration settings"; ablations "measure the value of each component" | Approved abstract (Oct 5) |
| Rule-based counterpart isolates what the LLM adds ("LLM → rules" ablation) | Scope (Nov 13); professor, Nov 18 |
| Fixed fades form a trade-off: slow 1/987k keeps NB1's clicks (45.96%) but loses 3,172 per shift; fast 1/20k loses 504 but costs 1.7 pts. A controller can try for both: slow when calm, fast after a drift | RESEARCH_LOG §6 (6b part 2) |
| Detector = Page-Hinkley, not ADWIN: drops only; ADWIN's natural output is a window (rebuilds → dead-arm trap) | User, Nov 18 |
| Tune only without planted shifts; paired E3 + E4 runs; shift times and category names never visible | Phase 5d Q10, R6 |

## 2. Design (decided with the user, Nov 18)

- **Event-triggered:** between alarms nothing changes (slow base fade). On an alarm: decision → boost ρ for a length → back to base.
  A boost **replaces** ρ (it does not add to the base); θ₀ never changes.
- **Versions** (§3): A = rules, A-budget PH; B = rules, sensitive PH (diagnostic + second rival); C = sensitive PH + LLM (6d).
- **PH signal = NB1's surprise:** reward − NB1's estimate for the played arm, using the cached pre-update estimate (`self._o`), on
  every step including explored plays (Q2).
- **The LLM never changes PH's settings;** it only decides the response after an alarm, so C and B see the same alarms. LLM-tuned
  detection is future work.
- **γ:** stays a dial; expected finding: γ control adds little (exploration plays never train NB1: 0.41% of steps on MIND-small,
  verified Nov 18); forgetting is NB1's real re-try mechanism (fade lifts trained arms' scores 0.23 → 0.35 toward the ≈ 0.49 start).
  γ in the rules comparison follows the γ check (R2).
- **LLM interface (used in 6d):** called only at alarms; JSON of summary statistics (no step number, no category names) → {real
  shift / spike / unsure; ρ boost from the menu; length from the menu; optional γ; one-line reason}; schema-checked, clipped,
  invalid → no change. Fields fixed after step 4 (Q10); prompt, model (text LLM main; Jev = optional D), agent layout: 6d.
- **Boost menu** = the MIND-small-tuned rates {1/10k, 1/20k, 1/50k}; lengths = one memory length each (10k, 20k, 50k steps).
  Paper disclosure: fixed after the fixed-fade E4 results were known.
- **Dropped:** headlines / topics as controller input (clashes with R6).
- **"−detector" is a required ablation** (6d/6e): LLM on a fixed schedule, no PH, as many calls per run as C makes on alarms.
  Look for papers that tried something similar.

## 3. All versions

| Version | Forgetting | Notices drift | Decides response | Phase |
|---|---|---|---|---|
| NB1 | none | — | — | done (5c/5d) |
| −controller | fixed slow fade | — | — | done (6b part 2) |
| **A** rules | slow fade + boosts | PH, boosts ≤ 10% of a calm run | fixed rule | **6c** |
| **B** rules, sensitive | slow fade + boosts | PH, boosts ≤ 30% | same fixed rule | **6c** |
| C full model | slow fade + boosts | PH as B | LLM | 6d |
| D (optional) | as C | as C | Jev | 6d, if time |
| −detector | as C | LLM on a schedule | LLM | 6d/6e |
| −adaptive exploration | as C | as C | LLM, γ locked 0.005 | 6e |
| −forgetting | none (no base fade, no boosts) | as C | LLM sets γ only | 6e |

## 4. Steps

1. **γ check:** MIND-small, seeds 100–102, NB1 + fade 1/100k, γ ∈ {0, 0.005, 0.02, 0.05} (12 runs, ≈ 1 min). Applies R2.
2. **Read Page 1954 in full** (Q3; the user downloads it if paywalled); log in REFERENCES_VERIFICATION. The formula used is Gama et
   al.'s (§3.2.1, App. A Alg. 1; manuscript read in full; cite sections, not pages).
3. **Build** (§5) → review → tests.
4. **Measurement** (MIND-small, seeds 100–102, NB1 + fade 1/100k, no shifts): save the surprise stream (and the click stream,
   descriptive) once per seed; simulate PH **offline** over a log-spaced λ grid (with the controller's ignore-during-boost and reset
   logic) → boosted share vs λ, alarm positions vs MIND-small's natural jumps.
5. **Thresholds (pre-registered in RESEARCH_LOG before any MIND-large run):** λ_A = the smallest grid λ whose boosted share is
   ≤ 10% on **every** seed (100–102); λ_S likewise for ≤ 30%. Confirmed by one real controller run per seed (boosts change NB1's
   path); if a real run exceeds its budget, take the next grid λ up. Same λ on MIND-large (per-step rule, R1); MIND-large data is
   never used to set anything. Then fix the JSON fields (Q10) as a short plan revision for approval.
6. **Rehearsal** (MIND-small with the trial shifts, one seed): outputs produced; alarm log saved. Not used for any choice.
7. **Checkpoint**, then A and B on MIND-large E3 + E4, seeds 0–9 (40 runs, ≈ 20–30 min).

## 5. Files

| File | What | Who |
|---|---|---|
| `nonstationary/drift.py` | `make_detector(threshold)`: river `PageHinkley(mode="down", alpha=1, delta=0.005, min_instances=30, threshold=λ)` (R6) | Opus |
| `nonstationary/controller.py` | `rule_controller(alarm) → (rho, length)`: A/B's fixed rule (1/20k, 20k steps) | Opus |
| `algorithms/neuralbandit.py` | `NeuralBandit1Ctrl(NeuralBandit1Fade)`: after the update, feeds the surprise to PH (not during a boost); on an alarm calls `controller`, applies the boost (replaces ρ) for its length, then restores the base ρ and resets PH; `diagnostics()` saves alarm steps, decisions and boosted steps | Opus |
| `tests/test_page_hinkley.py`, `tests/test_controller.py` | §7 (`tests/test_drift.py` already exists: Phase 5d) | Opus |
| `configs/phase6c_*.yaml`, `scripts/run_experiment.py`, `scripts/make_recovery.py`, a small measurement script | γ check, measurement, rehearsal, E3/E4; labels `NB1-Ctrl_A`, `NB1-Ctrl_B` | Sonnet agent, checked |
| `requirements.txt` | add pinned `river` (works with Python 3.11 / numpy 2.4.6; checked at install) | Opus |

`summary()` (the LLM's JSON) is **not** built in 6c; 6d adds it to the same alarm → action hook (R7).

## 6. Settings and sources

| Setting | Value | Source |
|---|---|---|
| NB1 | C 1, λ 1, γ 0.005, d 19 | Phase 5c |
| Base fade | 1/100k (MIND-small), 1/987k (MIND-large) | 6b part 2 tuning + headline |
| Boost (A, B) | ρ = 1/20k for 20k steps, on both datasets; alarms ignored during a boost; PH reset after it | Q5, R1 |
| Boost menu (C, 6d) | {1/10k, 1/20k, 1/50k}, lengths 10k / 20k / 50k | Discussion; R1 |
| PH | river `PageHinkley`, mode "down", alpha 1 (Gama's formula has no fading), δ = 0.005 (library default; ≈ 1% of the surprise's ±0.5 per-step swing), min_instances 30 (default), λ from step 5 | Gama et al. §3.2.1, App. A; river docs/source; R6 |
| Signal | NB1's surprise, every step, pre-update estimate | Q2 |
| Thresholds | boosted share ≤ 10% (A) / ≤ 30% (B, C) on every MIND-small seed 100–102, no planted shifts | R1, R5 |
| γ check | γ ∈ {0, 0.005, 0.02, 0.05}, MIND-small, seeds 100–102 | Q8 |
| Runs | MIND-large E3 + E4, seeds 0–9, shifts 300k / 1M / 1.5M | Phase 5d |
| Judging | R3: primary numbers vs NB1 and the slow fade; 2 × SE labels "clear"; claims written in advance | R3 |

## 7. Done-check

1. **Tests:** river PH equals a hand-computed Gama Alg. 1 (drops version) on a fixed stream; no alarm on a constant stream; an alarm
   after a planted drop; ρ = boost during the boost and back to base after its length; no PH input during a boost and PH reset after
   it; a controller that never boosts = NB1 + slow fade bit-for-bit; same seed → identical rerun (PH deterministic); masking.
2. **Measurement + thresholds** logged and pre-registered before MIND-large runs.
3. **40 runs** complete and finite; E4 = E3 bit-for-bit before 300k.
4. **Research result** (R3), led by the primary question: A and B vs NB1 and vs the slow fade (1/987k) on E3 click rate, clicks
   lost per shift, adaptation time. Also: alarms and boosted share per run (→ C's and −detector's call budget), alarm positions vs
   the shifts and E3's natural jumps.
5. **Docs:** RESEARCH_LOG (design, γ check, measurement, results), README §2 ("every N rounds" → event-triggered; "Page-Hinkley /
   ADWIN" → Page-Hinkley) and §6, CLAUDE.md (6c description), REFERENCES_VERIFICATION (Page 1954, Gama PH, river).

## 8. Decisions

**Design-discussion questions (user, Nov 18):**
- **Q1** (budget; units superseded by R1) A sparser than the sensitive version; C must beat the rules (form: R3).
- **Q2** PH watches NB1's surprise, fixed before any data; the click stream is logged descriptively.
- **Q3** Page 1954 and the Gama survey read in full; the user downloads Page 1954 if paywalled.
- **Q4** Library (`river`), checked against the source formula by a test.
- **Q5** A's rule: 1/20k for 20k steps; alarms ignored during a boost; PH reset after it; no tuning.
- **Q6** (superseded by R1) boosts were to be time-converted.
- **Q7** A and B run on MIND-large in 6c.
- **Q8** γ grid {0, 0.005, 0.02, 0.05}.
- **Q9** MIND only.
- **Q10** JSON fields fixed after the measurement, as a short plan revision.

**Review round 1 (fresh-context Opus, Nov 18):**
- **R1 (user):** budget by boosted share — ≤ 10% (A) / ≤ 30% (B, C) of a MIND-small run without planted shifts; boosts 1/20k for 20k
  steps on both datasets (per step everywhere, so the calibration carries over without looking at MIND-large). Reason: a MIND-small
  day is ≈ 30k steps, so 20k-step boosts at 1–5 alarms/day would make A and B near-constant fast fades.
- **R2 (user):** after the γ check — if no γ beats 0.005, γ leaves C's menu (A and C then have the same dials; the γ check is the
  −adaptive-exploration evidence); if some γ helps, A gets a fixed γ rule (that γ during boosts).
- **R3 (user):** no new pass/fail gate. All versions reported on the same primary numbers vs NB1 and the slow fade; 2 × SE labels
  "clear". Claims written in advance: **"the controller helps"** if A, B or C loses clearly fewer clicks per shift than the slow fade
  with no clear drop in E3 clicks; **"the LLM helps"** if C does so better than both rules versions; otherwise "matched". The design
  stays NB1 + fade + controller (lesson from 6b part 2).
- **R4 (user):** the LLM never changes PH's settings.
- **R5 (applied):** threshold rule pinned (grid, every seed, real-run confirmation); MIND-large never used to set anything.
- **R6 (user):** PH = river defaults except mode "down", alpha 1, λ from the budget; δ = 0.005 is the library default (Gama gives no
  value; it calls δ "the tolerable magnitude of changes").
- **R7 (applied):** simplifications — offline PH simulation for the measurement; no `summary()` in 6c; rule controller in
  `nonstationary/`; new test file names (test_drift.py exists).
- **R8 (user, Nov 18, after the measurement):** MIND-large uses the MIND-small λ unchanged (A 219.9, B 90.9). If the boosted share
  there differs from 10% / 30%, it is reported as is. No rescaling, and no look at MIND-large first.
- **Measured (step 4–5, pre-registered):** λ_A = 219.9 (1 alarm, 9.6% boosted), λ_S = 90.9 (3 alarms, 29%), confirmed by real runs.
  **γ check (step 1):** no γ clearly beats 0.005 (γ = 0: +0.0005, within 2 × SE), so under R2 γ leaves C's menu.
- **R9 (user, Nov 18):** the γ result goes in the paper only (no separate note to the professor), worded precisely: fixed γ values
  were tested on calm data; because Eq. 1 never learns from exploration plays, raising γ after a drift cannot speed recovery either;
  re-exploration after drift comes from forgetting, so the controller adjusts only the fade rate.
- **Noted for the paper (review):** PH compares with its running mean since the last reset, so some "false" alarms on E3 are real
  natural events; the surprise is biased slightly negative (greedy picks favour over-estimated arms); the detection delay may eat
  the gain at shift 1, where NB1 already recovers in ≈ 11k steps.

## 9. Carried to 6d planning

- **Can the LLM's inputs tell a news spike from a real shift?** (side-agent note, Nov 19) At an alarm both look like "NB1's
  estimates went wrong". If the summary can't separate them, C just copies rule A. Candidate signals (no step number, no category
  names):
  - how many arms' surprise turned negative at once (a relabelling breaks every arm; a news spike moves a few);
  - whether the drop persists or reverts within a few thousand steps;
  - the share of plays on the top arm.
  
  Check on MIND-small (natural spikes only) before writing the prompt.
- The JSON summary fields (Q10) are decided in 6d with this in mind.
