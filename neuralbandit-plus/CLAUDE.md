# CLAUDE.md — NeuralBandit++ (Deep Learning for DS final project)

## Project
Reproduce Allesiardo, Féraud & Bouneffouf (2014), "A Neural Networks Committee for the
Contextual Bandit Problem" (NeuralBandit1/2/3), then extend it as NeuralBandit++:
NeuralUCB + sliding-window memory + an LLM controller (drift monitor, memory manager,
exploration tuner agents) that adjusts settings every N rounds while NeuralUCB picks actions.
Goal: faster adaptation after concept drift. Datasets: UCI Forest Cover Type, Microsoft MIND-small.
Full plan, experiments (E1–E4) and metrics: see README.md.

## Deadlines
- Paper (6–8 pp, ACM WWW format) + this repo: working target Sun Nov 29, 2026, 11:59 PM (official: Dec 1)
- Presentation + demo: Mon Dec 14, 2026
- Checkpoint Wed Nov 18: NB2 must beat LinUCB on Forest Cover (instructor's success criterion:
  cumulative regret vs. paper Figure 1; NeuralBandit must do better than LinUCB)

## Scope (decided Nov 13, 2026 — short on time, do not add work beyond this)
- Baselines (course slide "Baselines"): NB1, NB2, NB3 + exactly ONE classical baseline = LinUCB
  (chosen because the instructor's check is "beat LinUCB"). Random kept as a sanity line only.
  Banditron and CTS are NOT built.
- Every baseline runs in every environment the method runs in (slide: "identical environments and
  evaluation protocols") — Forest Cover AND MIND, not just Figure 1.
- Ablations = core 4 only: −LLM controller; LLM → rule-based controller; −UCB; −sliding window.
  Run on MIND E3 + E4 only (user, Nov 17); E2 only if time allows. Full model + NeuralUCB alone run E1–E4.
- Phase 6 = modular sub-phases 6a–6f (user, Nov 17; RESEARCH_LOG §5): 6a NeuralUCB alone, 6b + window,
  6c rule-based controller, 6d LLM controller → full model, 6e ablations, 6f stats/figures. Optional last:
  window/LLM on NB1.
- "Common toolkit" integration: deferred — do not raise it until the user does.

## How we work (MANDATORY) — the phase loop
Work in small phases. Each phase runs: **Plan → Build → Verify → Review → Report → user commits.**

### Research log — `paper/RESEARCH_LOG.md` (update continuously, every phase)
The user writes the paper and presentation from this file. The moment anything paper-worthy
turns up, add a dated, sourced entry under the matching paper section: source facts, decisions
(and why), paper ambiguities, data facts (even small ones, e.g. uneven Slope bins), results with
numbers, failures and dead ends, speed/cost. If in doubt, log it.

### Ground truth: every decision traces to a source (never assume)
Before planning anything, check what the sources actually say. Lesson learned: we assumed all of
the paper's classical baselines were required; the slides only say "at least one" — so we build one.
1. **Course requirements** — final-project slides in `../Intro-Deep Learning for Data Science.pptx`
   (Baselines, Ablation studies, Final project workflow, What your final report must answer,
   Final deliverable) + instructor Slack (Nov 12: beat LinUCB on cumulative regret vs. Figure 1).
2. **The approved proposal** — `../Final Project Abstract - Google Docs.pdf`: what we promised to
   build and why (fixed γ, no drift handling, slow re-stabilization → NeuralUCB + sliding window +
   LLM controller; main goal = faster adaptation after drift).
3. **The base paper** — `../A Neural Networks Committee for the Contextual Bandit Problem.pdf`.
4. **Papers for methods we adopt** (NeuralUCB, LinUCB, EXP3, ...) — primary sources only.
5. **User decisions** — recorded with their date in the Scope section above.
If something is required by none of these, it is out of scope until the user says otherwise.

### 1. Plan — iterate until 95% ready (user approves)
- Draft the plan in `paper/plans/phase-N.md`. Every draft and every edit comes with clarifying
  questions; keep asking until nothing would have to be guessed. Never fill a gap with an assumption.
- Check the sources above first; bring the user only what the sources can't answer, with what was
  found and where (page/slide/section).
- A plan is ready only when it states, with nothing vague:
  1. Goal of the phase and **why**: which course requirement, which part of the proposal, and which
     research question it serves.
  2. Every file to build and what it does.
  3. Every setting/number with its source (paper page/§, slide, or "user decision, date").
  4. Done-check: the exact result or test that proves the phase worked (tied to the research,
     e.g. "LinUCB near the paper's 72%", "NB2 below LinUCB").
  5. Who builds what (Sonnet = routine; Opus = NB1/2/3, EXP3, NeuralUCB, LLM controller, debugging),
     and what runs in parallel.
  6. Open questions: **zero**. No building until this list is empty.

### 2. Build — agents
- Simplest correct version first; well-known library calls over hand-written code
  (e.g. sklearn `fetch_covtype`, not a custom downloader). Complexity only when a result demands it.
- Add a package only when the code that needs it is built; then re-freeze requirements.txt.

### 3. Verify — Claude
- Run the tests and a short real run; compare numbers with the paper / the done-check.
  Anything off is fixed before reporting.

### 4. Review
- `/simplify` (shortest, laziest-senior-engineer version; efficiency), then `/code-review`
  (correctness), then re-run tests.

### 5. Report — user reads
- Results vs. the done-check; one short paragraph per file + the 1–2 lines that matter
  (so the user can explain the code on Dec 14). Update README roadmap, this file, and
  `paper/RESEARCH_LOG.md` (check nothing from the phase is missing).
- Paper ambiguities (e.g. biases, NB2's P(k), NB3's EXP3 rewards) always go to the user as decisions.
- Never git commit, push, or init. The user commits after each phase.

## References & claims (MANDATORY — anti-hallucination)
- Source of truth for the base paper is the PDF the user was given (arXiv:1409.8191v1, 29 Sep 2014):
  authors exactly "Robin Allesiardo, Raphaël Féraud, Djallel Bouneffouf". Do not substitute other name forms.
- Never write a citation, author name, date, venue, or quoted number from memory into the paper,
  README, or code. Look it up in a primary source and log it in `paper/REFERENCES_VERIFICATION.md` first.
- Primary sources only: the provided PDFs, arXiv abstract pages, DOI/publisher pages, DBLP, official
  proceedings. Not blogs, not AI summaries. Bibliographic fields need ≥2 independent sources.
- Anything not yet verified is marked `[UNVERIFIED]` wherever it appears.
- If sources disagree, flag the discrepancy to the user — never silently pick one.
- Every claim about another paper (numbers, theorems, settings) records page/section/equation/table/figure.
- `paper/references.bib` holds only ✅ verified entries.
- Code that implements a paper cites it in the docstring as `[citekey, §/Eq./Alg.]`; deliberate
  deviations from the paper are stated with the reason.

## Environment
- Python 3.11.4 in `.venv/` (uv). Install: `uv pip install --python .venv/bin/python -r requirements.txt`
- Tests: `.venv/bin/pytest` (pytest.ini sets pythonpath=. — no `__init__.py` files; implicit namespace packages)
- Secrets: only `ANTHROPIC_API_KEY`, in gitignored `.env`, needed only for the LLM controller.

## Layout (follows course slide 71 — do not add top-level code folders)
algorithms/ models/ exploration/ nonstationary/ llm/ evaluation/
+ configs/ scripts/ data/ (gitignored) tests/ results/ (committed) paper/ notebooks/

## Code conventions
- Every policy subclasses `algorithms/base.py::Bandit`: implement `_init_state`, `select`
  (must set `self.last_probs`), `update`. Constructor: super().__init__ → store hyperparams → self.reset().
- All randomness through `self.rng` (seeded numpy Generator) — never global np.random / random.
- `update` only ever receives the chosen action's reward (bandit feedback).
- UCB logic lives only in `exploration/`; environments and drift never leak into algorithms.
- All hyperparameters in `configs/`; document any value the paper does not specify.

## Decided reproduction details (paper is ambiguous — keep consistent)
- Features: 10 continuous cols → 5 equal-frequency bins (one-hot) + 4 wilderness + 40 soil = 94 binary; 7 actions;
  reward 1 if correct class; data shuffled once (seed 0), streamed in a loop from a random start.
- γ = 0.005, γ_model = 0.1; model grid hidden {1,5,25,50,100} × λ {0.01,0.1,1}; sigmoid + squared error.
- NB update follows **Eq. 1 as printed**: 1[k̂ = k] → update only the greedy network, only when it was
  played, × 1/P(k̂) ≈ 1.004 (user, Nov 15). The "played action × 1/P" reading of Alg. 1 collapses NB1
  to 48.5% (×1,400 steps saturate the sigmoids) — a finding, not the method.
- NB2/NB3 (Phase 4, user Nov 15): NB2 = one EXP3 over 15 models; the chosen model plays ε-greedy
  (γ = 0.005, inferred) and its k̂/P gate the Eq. 1 update of **every** model (Alg. 2, not p. 5 prose).
  NB3 = one EXP3 per action; gate = Alg. 3's k̂_t/P_t; only the played action's EXP3 gets the reward
  (causes a weak-model lock-on for common actions — a finding). Details: `paper/plans/phase-4.md` §10.
- EXP3 in log-space. Drift: label rotation 1→2→…→7→1 every 500k steps. Regret vs. 93% offline reference.
- Report last-100k-step classification rate (paper metric) + cumulative regret.
- Regret = Σ(0.93 − reward) (constant reference). LinUCB: α = 0.1 (tuned, seeds 100–102), no bias
  feature, no normalisation, random tie-break. Tuning seeds 100+ never used for reported results.
- Runs: `scripts/run_experiment.py --config configs/X.yaml` (saves each run; reuses it on rerun only if
  settings match) → `scripts/make_figures.py` (plots every *_stationary / *_drift result file).
  Phase 3: call `torch.set_num_threads(1)` inside each parallel job.
- MIND (Phase 5, full decisions in `paper/plans/phase-5.md` §9): category arms (16 shown categories);
  reward = 1 if the user clicked any shown article of the chosen category; **masking** — `select(x,
  avail=mask)` picks only shown categories (avail=None = Forest Cover, unchanged); context d = 19 (18
  category shares of history + the user's earlier in-week clicks, + constant 1); chronological, one pass,
  each seed keeps a random 90% of users. Tune on `mind_small`, report on `mind_large` (= large minus
  small's users). Dynamic regret = steps − reward on MIND (every impression has a click). MIND configs
  must set `d: 19` (runner checks). LinUCB α = 0.5 on MIND (tuned Nov 16). NB1 on MIND: C = 1, λ = 1
  (tuned Nov 16, Phase 5c); NB2/NB3: paper grid, γ = 0.005, γ_model = 0.1. E3 results: phase5c_mind_e3.
- E4 (Phase 5d, `paper/plans/phase-5d.md`): shifts at 300k/1M/1.5M = cyclic arm relabelling (arm a → category
  (a+j) mod 16); every method needs a paired E3 + E4 run per seed. Deciding number = clicks lost per shift
  (500k window, vs same-seed E3) with the pre-registered `adapts_better` rule; adaptation time = R10 stopwatch
  on 10-seed mean curves. `scripts/make_recovery.py {e4|forest|trial}`. E4 results: phase5d_mind_e4.
- Reference repo TheAmazingElys/NeuralBandit: no license, unfinished (no NB2/NB3), never copy code,
  and **never a source for any decision** — the 2014 paper is the source of truth; where the repo
  differs (biases, no importance weighting, momentum, γ = 0.05), follow the paper (user, Nov 15).
