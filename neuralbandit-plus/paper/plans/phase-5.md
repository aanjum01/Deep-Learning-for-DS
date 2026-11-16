# Phase 5 — MIND: data (5a) → data review → environment + sanity run (5b)

**Status: DONE (2026-11-16).** 5b built, reviewed (`/simplify`, `/code-review`), 73/73 tests, Forest
Cover step-for-step identical; done-checks §7 all pass (Random 0.199 = analytic 0.1995; LinUCB 0.437 >
popularity 0.360 > Random on every seed; best α = 0.5). Results: RESEARCH_LOG §4. Deviation from §3
wording: the runner keeps Forest Cover's (x, correct) items and branches on item length instead of
Forest Cover yielding one-hot reward vectors (keeps the Forest stream and its tests untouched).

**Earlier status:** rev. 3 (2026-11-16). **5a DONE**: MIND-small and MIND-large downloaded (`download_mind.py
--size small|large`), verified byte-identical to the official release (Q4), all facts counted
(`mind_facts.py`, `results/mind_{small,large}_facts.txt`). **Data review (§6b) DONE** — every decision
re-checked against the data; none changed (outcomes below §6b). **5b ready for approval.**
Change vs. rev. 2: building `mind_{small,large}.npz` (`preprocess_mind.py`) moves from 5a to 5b, since
its contents depended on the review.

## 1. Goal and why

Get MIND onto disk, turn it into one processed file that every algorithm reads, build a category-arm
bandit environment on it, and prove it works with a short Random + LinUCB (+ popularity line) run that
also tunes LinUCB's α.

| Why | Source |
|---|---|
| **Proposal:** "evaluate the proposed method on the MIND news recommendation dataset from Microsoft, where user interests naturally change over time"; metrics "cumulative reward, dynamic regret, and adaptation time after drift" | Approved abstract (Oct 5) |
| **Course:** every baseline runs in "identical environments and evaluation protocols" | Course slides: *Baselines*; scope decision Nov 13 |
| **Research:** E3/E4 (README §6) need a real non-stationary environment beyond Forest Cover | README §6 |

**Out of scope:** E4 shift injection (designed later), NB1–3 / NeuralUCB runs on MIND, the LLM
controller, the article-level formulation (future work, Q26), MIND-large download (before final runs, Q3).

## 2. What the sources say (read 2026-11-15/16)

**Access** (checked live): the official page msnews.github.io links to Hugging Face `yjw1029/MIND`
(**gated**, HTTP 401). `Recommenders/MIND` (used by Microsoft's `recommenders` library,
`recommenders/datasets/mind.py`) is **open**: `MINDsmall_train.zip` 52,994,575 B, `MINDsmall_dev.zip`
30,948,560 B; large train 531,360,717 B, dev 103,592,887 B. Old Azure blob: dead (HTTP 409). The two HF
copies' zips differ by ~0.08% (re-zipped) — contents [UNVERIFIED] until Q4's check.

**License** (MSR License Terms PDF, read Nov 16 — REFERENCES_VERIFICATION R7): research use, modify,
publish results with "no material portion of the Dataset"; **no distribution**; accepted by use. User agreed Nov 16.

**Format** (official `introduction.md`): each zip has `behaviors.tsv` (Impression ID, User ID, Time
`MM/DD/YYYY HH:MM:SS AM/PM`, History = earlier clicked news IDs in time order, Impressions =
`N…-1/0` shown news + click label, order shuffled) and `news.tsv` (News ID, Category, SubCategory, Title,
Abstract, URL, entities). MIND-small = 50,000 random users from MIND (C20); train + validation only.
Training = week 5 of Oct 12 – Nov 22, 2019; validation = last day of week 5; history from weeks 1–4;
only samples with non-empty history kept (C18, C19). → **≈ 1 week of impressions in small and large**;
large's test week has no public labels. MIND: 20 categories (C17).

**LinUCB** [li2010contextual]: defined for a per-trial arm set A_t, "works well for a dynamic arm set"
(C23); its news experiment used shares summing to 1 **plus a constant feature 1** (C24).

## 3. Design (all decided, §9)

| Piece | Decision | Q |
|---|---|---|
| Data source | script, open `Recommenders/MIND` mirror; SHA-256 + row counts recorded; one-time check vs the official copy | Q1, Q4 |
| Datasets | build & tune on **MIND-small**; final reported runs on **MIND-large minus small's users** | Q3, Q22 |
| Stream | train + dev glued, **chronological**, one pass, no looping | Q5, Q6, Q8 |
| Seeds | each seed keeps a random **90% of users** (same subset for all algorithms) + drives policy randomness | Q7 |
| Arms | every category shown somewhere in the data; count read from the file | Q9, Q19 |
| Reward | r_a = 1 if the user clicked any shown article of category a in this impression | Q10 |
| Availability | **masking**: policy chooses only among categories shown in this impression | Q11 |
| Zero-click impressions | kept | Q12 |
| Context | category shares of the user's history (sum 1) **+ constant 1**; history = file History + user's own logged clicks from earlier impressions, plain cumulative counts | Q13–Q16 |
| Regret | dynamic regret vs. best available arm: 1[impression has a click] − r_t; plus cumulative reward | Q17 |
| Runner | one runner for both datasets: env yields (context, reward per arm, available mask); `r = rewards[a]` | Q18, Q25 |
| Processed file | `mind_{small,large}.npz`: `X`, `R`, `avail`, `time`, `user`, `impression_id`, `categories` | Q20 |

## 4. Files

**5a**
| File | What it does |
|---|---|
| `scripts/download_mind.py` | downloads MIND-small train/dev zips from `Recommenders/MIND` (stdlib `urllib`, skip if present), unzips to `data/raw/mind/small/{train,dev}/`, prints SHA-256 of zips and unzipped files |
| `scripts/preprocess_mind.py` | parses both TSVs, merges, sorts by time, builds X / R / avail / time / user / impression_id / categories, saves `data/processed/mind_small.npz`, prints every §6 fact |
| `scripts/mind_facts.py` *(or inside preprocess)* | the counts and distributions for the data review (§6) |

**5b** (confirmed by the data review)
| File | What it does |
|---|---|
| `scripts/preprocess_mind.py --size small\|large` | line-by-line (memory-safe, like `mind_facts.py`): merge train + dev, drop duplicate impressions, sort by time, build X (16 shares + constant 1, history updated with the user's own earlier clicks) / R / avail / time / user / impression_id / categories → `data/processed/mind_{size}.npz` |
| `evaluation/environments.py` | `load_mind()`, `mind_stream(data, seed, user_frac=0.9)`; Forest Cover stream yields (x, one-hot rewards, all-true mask) |
| `evaluation/runner.py` | generic loop: `a = policy.select(x, avail)`, `r = rewards[a]`, regret from best available; env chosen by config |
| `algorithms/*.py`, `exploration/*.py` | `select` takes the availability mask: argmax / random exploration only over available arms; NB1/2/3's P(k) over available arms |
| `algorithms/popularity.py` (or in evaluation) | the hindsight popularity reference line (Q21) |
| `scripts/run_experiment.py` | dataset chosen in config |
| `configs/phase5_mind_tune.yaml` | Q23 sanity/tuning run |
| `tests/test_mind_env.py` | toy TSVs → expected X / R / avail; history update uses only earlier clicks; 90% subset deterministic per seed; masking never picks an unavailable arm; Random's mean ≈ analytic |
| existing tests | + Forest Cover exact-match check |

No new packages (stdlib + numpy/pandas already installed).

## 5. Settings and sources

| Setting | Value | Source |
|---|---|---|
| Users per seed | 90% | user decision Nov 16 (Q7) |
| Steps | one pass (small ≈ 207k [UNVERIFIED]) | Q8 |
| Context | K shares + constant 1 | Q13, Q16; C24 |
| Tuning seeds | 100, 101, 102 | as Forest Cover (Phase 2) |
| LinUCB α grid | 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2 | `configs/phase2_tune_alpha.yaml` |
| α selection | total reward over the whole pass | Q23 |
| Reported seeds | 0–9, MIND-large minus small's users (later phase) | Q22 |

## 6. Data facts to count and log (5a; RESEARCH_LOG + REFERENCES_VERIFICATION C21/C22)

Impressions (train, dev, overlap), users, news, first/last timestamp of each file, categories shown,
click share per category, categories shown per impression, clicks per impression, zero-click
impressions, history lengths, empty histories, history IDs missing from news.tsv, impressions per user
(distribution), CTR, category click share per day (= natural drift), whether the History column
changes between a user's impressions, steps per run after the 90% cut.

## 6b. Data review checkpoint (user, Nov 16) — MANDATORY between 5a and 5b

After 5a, **stop**: Claude shows what the data looks like and we re-check every decision against it.
Then this plan is updated and re-approved before 5b is built.

| Decision | What to check |
|---|---|
| Q3 small → large | impression counts; time one pass → estimate final-run hours |
| Q4 mirror | unzipped files identical to the official copy (once the user has downloaded it) |
| Q5 train + dev | overlapping impressions; date ranges; dev after train? |
| Q7 / Q8 | steps per run after the 90% cut |
| Q9 / Q19 arms | categories shown (small now, large later — may differ); click share; never-shown ones |
| Q11 masking | categories shown per impression (mean, distribution) |
| Q12 zero clicks | do they exist, how many |
| Q13 / Q14 context | does History change within the week? impressions per user (distribution); history length; IDs missing from news.tsv |
| Q14 small vs large | impressions per user, large vs small (large counted when downloaded) |
| Q16 constant | any empty histories? |
| Q22 split | small's user IDs ⊂ large's; impressions left after removing them (when large is downloaded) |
| Q24 E3 vs E4 | how much category click share moves day by day |
| Q17 metrics | every impression has ≥ 1 click → dynamic regret = steps − cumulative reward: one metric on MIND, not two (side-agent note, Nov 16) |

### Data review outcomes (user, Nov 16)

- **Q3 — confirmed.** Large minus small's users, 90% cut ≈ 2.04M steps/run (Forest: 2.25M); NB2 ≈ 40–50
  min/run in 8 parallel jobs (estimate; time a real run). Small's daily patterns match large's.
- **Q4 — done, passed.** All four official zips' contents are SHA-256-identical to our copy (C28).
- **Q5 — confirmed (a)**, plus drop large's 6 duplicate impressions. Small's last day = 32% of its
  stream (dev samples its own 50k users) — a small-only quirk; it affects every tuned setting equally,
  so it doesn't change which wins; noted in the paper. Large: last day 14%, normal.
- **Q7 / Q8 — confirmed.** ≈ 207k steps/run (small), ≈ 2.04M (large minus small's users); 10% left out
  ≈ 66k users per seed on large.
- **Q9 / Q19 — confirmed (a): 16 arms** (same in small and large). kids (0.09%) and northamerica (0.02%)
  stay — rule fixed before seeing the data; with masking they are almost never available, so they cost
  only compute.
- **Q11 — confirmed (masking).** 8.7 of 16 categories shown per impression (54%, below the 80% cut-off;
  10% of impressions show ≤ 3) → without masking ~half the arms would be fake zeros each step.
- **Q12 — moot.** No zero-click impressions in small or large (every impression has ≥ 1 click).
- **Q13 / Q14 — confirmed.** History never changes within the week (0 of 533,415 multi-impression users
  on large) → the online update is the only way context follows the week. All history IDs map to a
  category. Impressions per user (large): median 2, mean 3.48, p90 7, p99 17 → modest effect for most
  users, real for active ones.
- **Q16 — confirmed (a): constant 1, keep empty-history impressions.** 57,335 on large (2.2%), 5,452 on
  small, despite the MIND paper's non-empty rule (C26) → the constant is actually needed (cold start).
- **Q22 — confirmed.** All 94,057 small users are in large; large minus them = 2,270,818 impressions,
  656,377 users. Small matches large (16 categories, CTR 4.05% vs 4.06%, near-identical daily shares).
- **Q24 — confirmed (a), with a note (user): E3's findings are worth more than we expected.** Natural
  drift is strong and event-like (music 5% → 25% → 8%, finance 22% for one day, sports ×3 on Fri; same
  in small) → E3 is a real adaptation test in its own right, and the daily table goes in the paper as
  its motivation. E4 stays the adaptation-time test (natural spikes have no clean start moment).
- **Q17 — presentation settled.** Every impression has ≥ 1 click → on MIND dynamic regret = steps −
  cumulative reward (one metric). Paper: regret curves as the main MIND charts (Figure 1 style, bandit
  convention), click rates in tables/text, one sentence stating the equivalence; never claimed as two
  separate wins. Independent MIND evidence = adaptation time (E4) and reward around drifts (E3). Code
  unchanged (records reward).

**Review complete (Nov 16): no decision changed; Q5 adds "drop large's 6 duplicates"; Q24 adds the E3 note.**

## 7. Done-checks

**5a:** download + unzip work from a clean `data/`; checksums and row counts printed and logged; every
§6 fact logged (C21/C22 resolved); `mind_small.npz` loads; §6b review held with the user.

**5b (Q21):**
1. All tests pass; **Forest Cover reruns with the same seeds give identical numbers** to the saved Phase 2–4 results.
2. Random's mean reward ≈ its analytic value (mean of clicked-shown ÷ shown).
3. LinUCB (best α) > Random on every tuning seed.
4. Popularity reference line reported; LinUCB above it = the context carries personal signal (if not → back to the user).

## 8. Who builds it and order

1. **5a** — Sonnet agent: download + preprocess + facts scripts. Claude runs them and verifies counts.
   The user downloads the official zips by hand when convenient (Q4) → Claude compares.
2. **Data review (§6b)** with the user → update + re-approve 5b.
3. **5b** — Sonnet agent: environment, runner, configs, tests; Opus agent: masking in NB1/2/3 (Eq. 1's
   P(k), the error-prone part). Then `/simplify`, `/code-review`, tests, Forest Cover exact-match,
   sanity/tuning run, report.

Phase 4 runs are finished (Nov 16), so the CPU is free.

## 9. Decisions (user, Nov 15–16) — every question decided

**A. Data**
- **Q1:** script from the open `Recommenders/MIND` mirror, no data in GitHub (license forbids
  distribution; git keeps files forever; `data/*` gitignored). Manual download = fallback only.
- **Q2:** MSR license read and summarised (R7); **user agrees**. No raw/processed MIND data in git; only
  aggregate results; no real rows in the paper or demo.
- **Q3:** build and tune on small; final MIND experiments on large (same week, ≈ 10× impressions; one large
  pass ≈ 2.3M steps [UNVERIFIED] ≈ Forest Cover's 2.25M). Count both and time a run before final runs.
- **Q4:** (b) open mirror + SHA-256 + row counts; one-time check against the official gated copy that
  the user downloads by hand into `data/raw/mind_official/`; compare **unzipped** files. Differ → stop.

**B. Stream**
- **Q5:** train + dev concatenated, sorted by time (one continuous week); check overlap and dates.
- **Q6:** chronological for all MIND experiments; no shuffled panel.
- **Q7:** each seed keeps a random 90% of users (seeded), time order kept; same subset for every
  algorithm. Paper: "each run uses a random 90% subset of users".
- **Q8:** one pass, no looping (looping = fake end→start jump + replayed impressions).

**C. Arms and reward**
- **Q9:** arms = every category shown somewhere; never-shown dropped; no merging / top-K. Rule fixed
  before seeing data.
- **Q10:** r_a = 1 if any clicked shown article has category a. Limitations logged (counterfactual
  unshown; categories shown many times have more chances). Li et al. replay not used (logging not uniform).
- **Q11:** masking (available arm set A_t), as in LinUCB's own setting (C23). Reward then reflects user
  interest, not MSN's display choices; measured drift = interest drift (abstract: "changing user
  interests"). NB1/2/3's P(k) over available arms = documented adaptation. Forest Cover: all arms
  available → identical results. Limitation: MSN chose what was available.
- **Q12:** keep zero-click impressions (regret 0 there; true click rate). Check existence.

**D. Context**
- **Q13:** category shares of the user's history (fixed slot order, sum 1). MiniLM → article-level only.
- **Q14:** history = file History + the user's own logged clicks from earlier impressions (never the
  bandit's choices); plain cumulative counts, no recency weighting (recency = the method's sliding window).
- **Q15:** categories only, no subcategories (revisit only if results disappoint and time allows).
- **Q16:** + constant 1 on MIND for all algorithms, following LinUCB's news experiment (C24). Forest
  Cover unchanged. For NB it acts like a first-layer bias — documented as a MIND input choice.

**E. Metrics**
- **Q17:** dynamic regret vs. best available arm + cumulative reward. No made-up constant reference.

**F. Code**
- **Q18:** one runner for both datasets, Forest Cover exact-match check (confirmed by Q25).
- **Q19:** number of arms read from the processed file.
- **Q20:** npz layout in §3; store `impression_id`, not article IDs.

**G. Sanity run**
- **Q21:** done-checks §7 incl. the hindsight popularity reference line (not a baseline).
- **Q22:** small = tuning set; large minus small's users = evaluation set. All MIND settings tuned on
  small only. Algorithms learn from scratch in every run; only chosen settings carry over.
- **Q23:** tuning run = sanity run: seeds 100–102, Forest Cover's α grid, + Random + popularity; α by
  total reward over the whole pass.

**H. Scope**
- **Q24:** keep both MIND experiments: E4 (planted shifts, known times) = main adaptation test; E3 (real
  week) = real-behaviour check. May be revisited after §6b. (E1–E4 are our own README labels.)
- **Q25:** the new method also runs on Forest Cover drift (E2) — head-to-head with the paper's
  "75k–350k". Ablations on E2 or MIND only: decided when those phases are planned.
- **Q26:** article-level formulation = future work (README note with the Phase 5 report, on approval).
- **Q27:** moot — Phase 4 runs finished; 5a can start on approval.
- **Q28:** split: 5a → data review (§6b) → plan update → 5b.
