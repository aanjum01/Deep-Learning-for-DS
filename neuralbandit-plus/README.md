# NeuralBandit++: Drift-Adaptive Neural Contextual Bandits with Gradual Forgetting and an LLM Controller

Final project for **Deep Learning for Data Science** (Columbia University, Fall 2026).
Author: Ahad Anjum Sadaqat

> **Status:** 🚧 Under active development. The repository scaffold is in place; modules are being built one at a time. Sections marked *(planned)* describe the intended design and will be updated as each piece lands.

---

## 1. Overview

Contextual bandits make a sequence of decisions (e.g., which news article to recommend) using side information (context), while observing feedback only for the action they took. Real systems are **non-stationary**: user interests and reward patterns change over time, so a good bandit must keep learning *and* recover quickly after a change (concept drift).

This project:

1. **Reproduces** *A Neural Networks Committee for the Contextual Bandit Problem* (Robin Allesiardo, Raphaël Féraud & Djallel Bouneffouf, [arXiv:1409.8191v1](https://arxiv.org/abs/1409.8191), 2014; published at ICONIP 2014), which introduced **NeuralBandit1/2/3**.
2. **Identifies a gap:** NeuralBandit explores with a *fixed* rate γ, has no explicit mechanism to detect or respond to drift, and its model-selection variants need **75k–350k steps** to re-stabilize after a drift.
3. **Proposes NeuralBandit++**, which combines:
   - **NeuralBandit1** as the base learner (the best and fastest of NB1–3 on MIND);
   - **Gradual forgetting**: NB1's weights fade slowly back toward their starting values, so old lessons weaken unless recent data keeps reinforcing them ("L2 Init", Kumar, Marklund & Van Roy, CoLLAs 2024);
   - **A structured LLM workflow**: task-specific agents that periodically monitor drift, decide how much old data to forget, and tune NB1's exploration rate, so exploration adapts to drift instead of staying fixed. NB1 still chooses every action.

   > **Change from the approved abstract (Nov 18, 2026, approved by the instructor):** the abstract proposed NeuralUCB as the base learner. NeuralUCB was built and tuned (Phase 6a), but on MIND-small it scored 41.8% clicks vs. 43.7% for LinUCB and 44.6% for NB1, and its runs were far slower. The base is therefore NB1. Fixed exploration is replaced by *drift-adaptive* exploration (the controller sets γ), not NeuralUCB's per-context uncertainty bonus, and no regret guarantee is claimed.

   > **Second change (Nov 18, 2026):** the abstract's *sliding-window memory* was built and tested (Phase 6b) but replaced by *gradual forgetting* (Phase 6b part 2). On MIND the window adapted fast but cost 1–3 points of clicks, sometimes left NB1 stuck at random-level play after a rebuild ("dead-arm trap"), and its results depended on when its rebuilds happened. Gradual forgetting adapted faster than NB1 at every planted shift while keeping NB1's click rate, and never got stuck. The idea, forgetting old data so the model can follow drift, is unchanged.
4. **Evaluates** against the original NeuralBandit1/2/3 and the classical LinUCB baseline on Forest Cover Type and Microsoft **MIND**, using identical environments and evaluation protocols for every method, with ablations of each main component.

**Primary research question:** *Can NeuralBandit++ adapt faster after concept drift than the original NeuralBandit, without sacrificing cumulative reward?*

---

## 2. Method

```
            context x_t (+ candidate items)
                        │
                        ▼
   ┌──────────────────────────────────────────┐
   │ NB1  (one neural net per action)         │──► action a_t ──► reward r_t
   │ weights fade toward their start (ρ)      │                      │
   └──────────────────────────────────────────┘                      │
                        ▲                                            │
     fade rate ρ,       │                                            │
     exploration γ,     │   every N rounds: summary statistics       │
     reset signals      │   (recent reward, uncertainty, trends)     │
                        │                                            │
   ┌──────────────────────────────────────────┐                      │
   │ LLM controller                           │◄─────────────────────┘
   │  • Drift Monitor agent                   │
   │  • Memory Manager agent  (forgetting)    │
   │  • Exploration Tuner agent (γ)           │
   └──────────────────────────────────────────┘
```

- The controller is **event-triggered**: NB1 always fades slowly, and a Page-Hinkley test watches NB1's surprise (reward minus its own estimate) for a drop. Only on an alarm does the controller act, boosting the fade rate for a while. It reads only aggregate statistics and never the absolute time step, and its outputs are validated, with a no-change fallback. (Phase 6c found that changing the exploration rate γ never helps NB1, because NB1's update never learns from exploration plays, so the controller adjusts only forgetting.)
- A **rule-based controller** (the same Page-Hinkley alarm + a fixed boost: fade 1/20k for 20k steps) serves as the non-LLM counterpart, isolating what the LLM itself contributes.
- NeuralUCB (Zhou, Li & Gu, ICML 2020) was the abstract's original base learner; its Õ(d̃√T) regret guarantee (Theorem 4.5 / Remark 4.8) assumes a fixed reward function, i.e. no drift. It is reported only as the method tried and replaced (see the note in §1).

---

## 3. Repository Structure

The layout follows the course's suggested NeuralBandit++ toolkit architecture.

```
neuralbandit-plus/
├── algorithms/      Bandit policies: Random, LinUCB, NeuralBandit 1/2/3, NeuralUCB
├── models/          Neural reward models: batched per-arm MLPs, shared (user, item) scorer
├── exploration/     ε-greedy, EXP3 (log-space), UCB, Thompson sampling, uncertainty estimates
├── nonstationary/   Gradual forgetting (fade toward init), sliding window (tested, rejected), drift detectors
├── llm/             Article embeddings, cached LLM client, agents, LLM + rule-based controllers, prompts/
├── evaluation/      Environments (Forest Cover, MIND), experiment runner, metrics, statistics, plots
│
├── configs/         One YAML per experiment (all hyperparameters disclosed)
├── scripts/         Data download/preprocessing, experiment runner, figure generation
├── data/            Raw and processed data (gitignored; recreated by scripts/)
├── tests/           Unit tests for algorithms, exploration, environments, metrics
├── results/         Committed CSVs, figures, tables, and cached LLM responses
├── paper/           ACM Web Conference (WWW) format LaTeX source
├── notebooks/       Demo notebook
├── README.md
└── requirements.txt
```

---

## 4. Setup *(planned)*

Requirements: Python 3.11 and [uv](https://github.com/astral-sh/uv).

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

LLM controller experiments need an API key (see Section 8 for how cached responses make reruns key-free):

```bash
export ANTHROPIC_API_KEY=...
```

---

## 5. Datasets *(planned)*

| Dataset | Use | Preparation |
|---|---|---|
| **Forest Cover Type** (UCI, 581k rows) | Reproduction of the original paper; controlled drift | 10 continuous features → 5 equal-frequency bins each (one-hot) + 4 wilderness + 40 soil = **94 binary features**; **7 actions** (cover types); reward 1 if the correct class is chosen, else 0; shuffled once with a fixed seed, then streamed in a loop |
| **MIND** (Microsoft News, Nov 9–15 2019, train + dev glued in time order). **MIND-small** (230,117 impressions) is the tuning set; **MIND-large minus MIND-small's users** (2,270,815 impressions) is the evaluation set. Downloaded from the Microsoft Recommenders copy, verified byte-identical to the official release | Real news recommendation with changing user interests | **Category-arm bandit:** arms = the 16 news categories that are ever shown; the bandit may only pick categories shown in the current impression (masking, as in LinUCB's arm set A_t); reward = 1 if the user clicked any shown article of that category. Context (19 numbers) = the category shares of the user's click history, updated with their own earlier clicks during the week, + a constant 1. Played once, chronologically; each seed keeps a random 90% of users. (An article-level formulation is future work.) |

```bash
python scripts/preprocess_forest_cover.py   # downloads Forest Cover via scikit-learn if needed
python scripts/download_mind.py --size small     # and --size large (train + dev only; ~635 MB)
python scripts/preprocess_mind.py --size small   # and --size large (removes MIND-small's users)
python scripts/mind_facts.py --size small        # optional: data facts (aggregates only)
```

---

## 6. Experiments *(planned)*

| ID | Environment | Algorithms | Purpose |
|---|---|---|---|
| **E1** | Forest Cover, stationary | Random, LinUCB, NB1, NB2, NB3 | Reproduce the paper (reported: NB2 76%, NB3 73%, LinUCB 72%, measured as classification rate over the last 100k predictions) |
| **E2** | Forest Cover, class rotation (1→2→…→7→1) every 500k steps | E1 + full model (NB1 + fading + controller) | Compare recovery time with the paper's 75k–350k steps |
| **E3** | MIND, the real week in time order | LinUCB, NB1–3, full model + ablations | Natural interest drift (strong, event-like daily swings in category clicks) |
| **E4** | E3 plus 3 planted shifts (cyclic relabelling of the 16 category arms at steps 300k, 1M, 1.5M, fixed by a data-only rule before any run) | Same as E3 | Precise adaptation measurement vs the same seed's E3 run: clicks lost per shift + adaptation time (Shaker & Hüllermeier 2015) |

**Baselines.** Following the course requirements, the baselines are NeuralBandit1, NeuralBandit2, NeuralBandit3 and one classical contextual bandit, **LinUCB** (chosen because the instructor's success criterion is defined against it). Random is included only as a sanity reference. All baselines run in every environment (E1–E4) under the same evaluation protocol as the proposed method. Banditron and CTS, which appear in the original paper's Figure 1, are out of scope.

### Reproduction success criterion
Confirmed by the course instructor: reproduction is judged by **cumulative regret, compared against Figure 1 of the paper**, and the NeuralBandit variants **must achieve lower cumulative regret than LinUCB**.
- Output a Figure 1 equivalent: cumulative regret vs. time step (0 to ~2.25M) for Random, LinUCB, NB1, NB2, NB3 (Banditron and CTS from the original figure are not reproduced), averaged over 10 runs, in both stationary (E1) and drift (E2) panels. Regret is computed against the 93%-accuracy offline reference.
- **E1 (stationary) target:** NB2 below LinUCB, matching the paper. NB3 ≈ LinUCB is expected per the paper.
- **E2 (drift) target:** NB2 and NB3 both below LinUCB after the first drift.
- LinUCB's α is tuned with the same budget as the NeuralBandit settings, so the comparison is fair.

### Ablations
Each ablation removes one component from the full model, to measure what actually caused the improvement:
- Full NeuralBandit++ (reference)
- − controller (NB1 + a fixed fade rate, no controller)
- LLM controller → rule-based controller (Page-Hinkley / ADWIN)
- − adaptive exploration (the controller may not change NB1's γ; replaces the abstract's "− UCB")
- − forgetting (the controller sets γ only; no fade; replaces the abstract's "− sliding window")

The sliding window is not an ablation (it is not part of the final model); it appears as a design comparison: 7 window sizes vs 8 fade rates on MIND E3/E4 (Phases 6b and 6b part 2).

```bash
python scripts/run_experiment.py --config configs/e1_forest_stationary.yaml
python scripts/make_figures.py
python scripts/run_experiment.py --config configs/phase5d_mind_e4.yaml   # E4 (needs phase5c_mind_e3 too)
python scripts/make_recovery.py e4       # also: forest (E2 re-measure), trial (MIND-small rehearsal)
```

---

## 7. Evaluation Metrics

- **Cumulative reward** and **classification rate over the last 100k steps** (paper's metric)
- **Cumulative, instantaneous, and dynamic regret**, plus regret against the best fixed policy in each time window
- **Adaptation time after drift:** steps until rolling performance returns to a fixed fraction of its pre-drift level (threshold, window, and handling of non-recovery fixed before running)
- **Sample efficiency:** steps to reach a fixed fraction of final reward
- **Cost:** per-decision latency, peak memory, LLM calls, tokens, and dollar cost
- **Statistics:** mean ± confidence intervals across seeds, significance tests between methods

---

## 8. Reproducibility

- Every hyperparameter lives in `configs/`, including settings the original paper leaves unspecified, which are documented as our choices.
- Fixed seeds; 10 seeds for main results, 5 for ablations.
- LLM calls use a pinned model ID and temperature 0, and every response is cached and committed to `results/`, so experiments replay deterministically without an API key.
- `results/` contains the exact CSVs and figures used in the paper.

### Implementation notes on the original paper
Where the 2014 paper is ambiguous, this repository makes an explicit, documented choice:
- The update follows **Eq. 1 as printed** (indicator 1[k̂ = k]): a network is updated only when it was the greedy action and was played, weighted by 1/P(k̂) ≈ 1.004. Algorithm 1 writes an unmarked k_t; reading it as the *played* action makes explored updates ×1,400 (γ = 0.005), which saturates the sigmoid outputs and collapses NB1 to the majority class (48.5%). This is reported as a finding.
- NB2/NB3 update semantics for non-selected models and per-action EXP3 instances are specified in `algorithms/neuralbandit.py` and the paper.
- EXP3 weights are maintained in log-space for numerical stability over millions of steps.

A partial, unlicensed implementation by the original first author ([TheAmazingElys/NeuralBandit](https://github.com/TheAmazingElys/NeuralBandit)) was consulted as a reference only. No code is copied from it.

---

## 9. Roadmap

- [x] Repository scaffold
- [x] Environment and pinned dependencies
- [x] Core bandit API (`algorithms/base.py`)
- [x] Forest Cover data + streaming environment (stationary and drift)
- [x] Baselines (Random, LinUCB) + runner + metrics — LinUCB 71.9% vs. paper's 72%
- [x] NeuralBandit1 — 82.0% stationary; beats LinUCB on regret in both panels
- [x] NeuralBandit 2/3 + EXP3 → **E1/E2 reproduction**: NB2 80.8% (paper 76%), NB3 72.5% (paper 73%); NB2 beats LinUCB on regret in both panels
- [x] 6a NeuralUCB built + tuned on MIND-small (41.8% clicks, below LinUCB/NB1) → base switched to NB1 (Nov 18, instructor-approved); phase closed, no full runs
- [x] 6b NB1 + sliding window (reset + retrain on the last W steps; tuned W 50k on MIND-small → 493k on MIND-large; 7 sizes × E3/E4 × 10 seeds): small windows make the planted shifts nearly free (W 10k loses ≤ 0.3k clicks per shift vs NB1's 3.4k–6.2k) and recover in ≤ 10k steps at every shift, but cost 1–3 points of clicks on the real week; the tuned W 493k "adapted better" than NB1 at all 3 shifts, but that win vanished when its reset schedule was moved by half an interval (timing check), because resets can hit a "dead-arm trap" depending on where they land
- [x] 6b part 2: gradual forgetting (weights fade toward their initial values, "L2 Init"; 8 rates × E3/E4 × 10 seeds): never trapped; the slowest fade keeps NB1's click rate (45.96% vs 45.97%) while losing 37% fewer clicks per shift and adapts better than NB1 at all 3 shifts; chosen as NeuralBandit++'s forgetting method (Nov 18); a pre-registered pairwise rule favoured the smallest windows (which give up 2–3 points of clicks) and is reported as such
- [x] 6c rule-based controller (slow fade + Page-Hinkley alarm on NB1's surprise + fixed fade boost; thresholds set on MIND-small by an alarm budget): rule A loses 33% fewer clicks per shift than the slow fade and adapts faster at every shift, but costs 0.18 points on the real week, because about 5 alarms per run fire on natural news events, which the LLM controller (6d) must learn to skip. Changing γ never helps NB1, so the controller adjusts forgetting only
- [x] MIND environment (category arms, masking) — LinUCB 43.7% clicks vs. popularity 36.0%, Random 19.9% (MIND-small, α = 0.5)
- [ ] 6d LLM controller and agents → full model, E1–E4
- [x] Baselines on MIND E3 (real week, MIND-large, 10 seeds): NB1 46.0%, NB2 45.6%, LinUCB 44.7%, NB3 43.6% clicks (popularity 37.2%, Random 20.4%)
- [x] E4 (planted shifts) design + baselines: every learner recovers; after shifts 2–3 NB1/NB2 lose ~6k clicks vs LinUCB ~24–29k; adaptation slows with each shift (NB1 14k → 47k → 114k steps)
- [ ] 6e ablations (MIND E3 + E4) · 6f statistics, figures (optional if time: ablations on E2)
- [ ] Paper (ACM WWW format, due Dec 1, 2026)
- [ ] Presentation and demo (Dec 14, 2026)

---

## 10. References

Every entry below is checked against primary sources; see [`paper/REFERENCES_VERIFICATION.md`](paper/REFERENCES_VERIFICATION.md) for sources, dates, and claim locations.

- Robin Allesiardo, Raphaël Féraud, Djallel Bouneffouf. *A Neural Networks Committee for the Contextual Bandit Problem.* arXiv:1409.8191v1, 2014. [arXiv](https://arxiv.org/abs/1409.8191). Published in *Neural Information Processing* (ICONIP 2014), LNCS, Springer, pp. 374–381, DOI [10.1007/978-3-319-12637-1_47](https://doi.org/10.1007/978-3-319-12637-1_47).
- Dongruo Zhou, Lihong Li, Quanquan Gu. *Neural Contextual Bandits with UCB-based Exploration.* ICML 2020, PMLR 119:11492–11502. [arXiv:1911.04462](https://arxiv.org/abs/1911.04462)
- Lihong Li, Wei Chu, John Langford, Robert E. Schapire. *A Contextual-Bandit Approach to Personalized News Article Recommendation.* WWW 2010, ACM, pp. 661–670. DOI [10.1145/1772690.1772758](https://doi.org/10.1145/1772690.1772758)
- Shipra Agrawal, Navin Goyal. *Thompson Sampling for Contextual Bandits with Linear Payoffs.* ICML 2013, PMLR 28(3):127–135. [arXiv:1209.3352](https://arxiv.org/abs/1209.3352)
- Sham M. Kakade, Shai Shalev-Shwartz, Ambuj Tewari. *Efficient Bandit Algorithms for Online Multiclass Prediction* (Banditron). ICML 2008, ACM, pp. 440–447. DOI [10.1145/1390156.1390212](https://doi.org/10.1145/1390156.1390212)
- Peter Auer, Nicolò Cesa-Bianchi, Yoav Freund, Robert E. Schapire. *The Nonstochastic Multiarmed Bandit Problem* (EXP3). SIAM Journal on Computing 32(1):48–77, 2002. DOI [10.1137/S0097539701398375](https://doi.org/10.1137/S0097539701398375)
- Fangzhao Wu, Ying Qiao, Jiun-Hung Chen, Chuhan Wu, Tao Qi, Jianxun Lian, Danyang Liu, Xing Xie, Jianfeng Gao, Winnie Wu, Ming Zhou. *MIND: A Large-scale Dataset for News Recommendation.* ACL 2020, pp. 3597–3606. DOI [10.18653/v1/2020.acl-main.331](https://doi.org/10.18653/v1/2020.acl-main.331)
- Albert Bifet, Ricard Gavaldà. *Learning from Time-Changing Data with Adaptive Windowing* (ADWIN). SIAM SDM 2007, pp. 443–448. DOI [10.1137/1.9781611972771.42](https://doi.org/10.1137/1.9781611972771.42) — ⏳ second-source check pending
- E. S. Page. *Continuous Inspection Schemes* (Page-Hinkley test). Biometrika 41(1–2):100–115, 1954. DOI [10.1093/biomet/41.1-2.100](https://doi.org/10.1093/biomet/41.1-2.100)
