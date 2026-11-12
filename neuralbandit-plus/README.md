# NeuralBandit++: Drift-Adaptive Neural Contextual Bandits with NeuralUCB, Sliding-Window Memory, and an LLM Controller

Final project for **Deep Learning for Data Science** (Columbia University, Fall 2026).
Author: Ahad Anjum Sadaqat

> **Status:** 🚧 Under active development. The repository scaffold is in place; modules are being built one at a time. Sections marked *(planned)* describe the intended design and will be updated as each piece lands.

---

## 1. Overview

Contextual bandits make a sequence of decisions (e.g., which news article to recommend) using side information (context), while observing feedback only for the action they took. Real systems are **non-stationary**: user interests and reward patterns change over time, so a good bandit must keep learning *and* recover quickly after a change (concept drift).

This project:

1. **Reproduces** *A Neural Networks Committee for the Contextual Bandit Problem* (Allesiardo, Féraud & Bouneffouf, ICONIP 2014; [arXiv:1409.8191](https://arxiv.org/abs/1409.8191)), which introduced **NeuralBandit1/2/3**.
2. **Identifies a gap:** NeuralBandit explores with a *fixed* rate γ, has no explicit mechanism to detect or respond to drift, and its model-selection variants need **75k–350k steps** to re-stabilize after a drift.
3. **Proposes NeuralBandit++**, which combines:
   - **NeuralUCB**: uncertainty-aware exploration instead of fixed random exploration;
   - **Sliding-window memory**: trains on recent, relevant interactions only;
   - **A structured LLM workflow**: task-specific agents that periodically monitor drift, decide how much old data to forget, and tune exploration. NeuralUCB still chooses every action.
4. **Evaluates** against the original NeuralBandit variants and classical baselines on Forest Cover Type and Microsoft **MIND**, with ablations of every component.

**Primary research question:** *Can NeuralBandit++ adapt faster after concept drift than the original NeuralBandit, without sacrificing cumulative reward?*

---

## 2. Method

```
            context x_t (+ candidate items)
                        │
                        ▼
   ┌──────────────────────────────────────────┐
   │ NeuralUCB  (neural reward model + UCB)   │──► action a_t ──► reward r_t
   │ trained on a sliding window of history   │                      │
   └──────────────────────────────────────────┘                      │
                        ▲                                            │
     window size,       │                                            │
     exploration α,     │   every N rounds: summary statistics       │
     reset signals      │   (recent reward, uncertainty, trends)     │
                        │                                            │
   ┌──────────────────────────────────────────┐                      │
   │ LLM controller                           │◄─────────────────────┘
   │  • Drift Monitor agent                   │
   │  • Memory Manager agent  (forgetting)    │
   │  • Exploration Tuner agent (α)           │
   └──────────────────────────────────────────┘
```

- The controller runs every *N* rounds (not every round). It reads only aggregate statistics and never the absolute time step, and its outputs are schema-validated and clipped to safe bounds, with a no-change fallback.
- A **rule-based controller** (Page-Hinkley / ADWIN drift detector + fixed rules) serves as the non-LLM counterpart, isolating what the LLM itself contributes.
- NeuralUCB's Õ(√T) regret guarantee (Zhou, Li & Gu, ICML 2020) applies to its stationary setting. Here it motivates the design and is not claimed for the windowed, controller-tuned variant.

---

## 3. Repository Structure

The layout follows the course's suggested NeuralBandit++ toolkit architecture.

```
neuralbandit-plus/
├── algorithms/      Bandit policies: Random, Banditron, LinUCB, CTS, NeuralBandit 1/2/3, NeuralUCB, (Neural TS)
├── models/          Neural reward models: batched per-arm MLPs, shared (user, item) scorer
├── exploration/     ε-greedy, EXP3 (log-space), UCB, Thompson sampling, uncertainty estimates
├── nonstationary/   Sliding window, drift detectors (Page-Hinkley, ADWIN), drift generators
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
| **MIND-small** (Microsoft News, ~230k impressions) | Real news recommendation with changing user interests | Article = MiniLM title embedding; user = embedding of click history, updated online; reward = click label of the chosen candidate. Two formulations: **shared-network** (score each candidate) and **category-arm** (~17 news categories as fixed arms, so NB1/2/3 run unmodified) |

```bash
python scripts/download_forest_cover.py
python scripts/download_mind.py
python scripts/preprocess_forest_cover.py
python scripts/preprocess_mind.py      # includes embedding precompute
```

---

## 6. Experiments *(planned)*

| ID | Environment | Algorithms | Purpose |
|---|---|---|---|
| **E1** | Forest Cover, stationary | Random, Banditron, LinUCB, CTS, NB1, NB2, NB3 | Reproduce the paper (reported: NB2 76%, CTS 73%, NB3 73%, LinUCB 72%, Banditron 57%, measured as classification rate over the last 100k predictions) |
| **E2** | Forest Cover, class rotation (1→2→…→7→1) every 500k steps | E1 + NeuralUCB variants + rule / LLM controller | Compare recovery time with the paper's 75k–350k steps |
| **E3** | MIND, chronological order | LinUCB, NB1–3 (category-arm), shared-network NB, NeuralUCB variants | Natural interest drift |
| **E4** | MIND with category-preference shifts injected at random times | Same as E3 | Precise adaptation-time measurement |

### Reproduction success criterion
Confirmed by the course instructor: reproduction is judged by **cumulative regret, compared against Figure 1 of the paper**, and the NeuralBandit variants **must achieve lower cumulative regret than LinUCB**.
- Output a Figure 1 equivalent: cumulative regret vs. time step (0 to ~2.25M) for Random, Banditron, LinUCB, CTS, NB2, NB3, averaged over 10 runs, in both stationary (E1) and drift (E2) panels. Regret is computed against the 93%-accuracy offline reference.
- **E1 (stationary) target:** NB2 below LinUCB, matching the paper. NB3 ≈ LinUCB is expected per the paper.
- **E2 (drift) target:** NB2 and NB3 both below LinUCB after the first drift.
- LinUCB's α is tuned with the same budget as the NeuralBandit settings, so the comparison is fair.

### Ablations
- Full NeuralBandit++
- − LLM controller (no controller)
- LLM controller → rule-based controller (Page-Hinkley / ADWIN)
- − each agent individually (its setting frozen at the default)
- Three agents → one combined agent
- − UCB (fall back to ε-greedy exploration)
- − sliding window
- Sensitivity: window size, controller frequency N
- Diagonal vs. full uncertainty matrix in NeuralUCB

```bash
python scripts/run_experiment.py --config configs/e1_forest_stationary.yaml
python scripts/make_figures.py
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
- The importance-weighted update uses the **played** action (Algorithm 1), not the greedy action written in Eq. 1. The 1/P(k) factor (up to ~1400× at γ = 0.005) is reported both clipped and unweighted.
- NB2/NB3 update semantics for non-selected models and per-action EXP3 instances are specified in `algorithms/neuralbandit.py` and the paper.
- EXP3 weights are maintained in log-space for numerical stability over millions of steps.

A partial, unlicensed implementation by the original first author ([TheAmazingElys/NeuralBandit](https://github.com/TheAmazingElys/NeuralBandit)) was consulted as a reference only. No code is copied from it.

---

## 9. Roadmap

- [x] Repository scaffold
- [ ] Environment and pinned dependencies
- [ ] Core bandit API (`algorithms/base.py`)
- [ ] Forest Cover environment + baselines (Random, Banditron, LinUCB, CTS)
- [ ] NeuralBandit 1/2/3 + EXP3 → **E1/E2 reproduction**
- [ ] NeuralUCB + sliding window + rule-based controller
- [ ] MIND environments + embeddings
- [ ] LLM controller and agents
- [ ] E3/E4, ablations, statistics, figures
- [ ] Paper (ACM WWW format, due Dec 1, 2026)
- [ ] Presentation and demo (Dec 14, 2026)

---

## 10. References

- R. Allesiardo, R. Féraud, D. Bouneffouf. *A Neural Networks Committee for the Contextual Bandit Problem.* ICONIP 2014. [arXiv:1409.8191](https://arxiv.org/abs/1409.8191)
- D. Zhou, L. Li, Q. Gu. *Neural Contextual Bandits with UCB-based Exploration.* ICML 2020. [arXiv:1911.04462](https://arxiv.org/abs/1911.04462)
- L. Li, W. Chu, J. Langford, R. Schapire. *A Contextual-Bandit Approach to Personalized News Article Recommendation.* WWW 2010.
- S. Agrawal, N. Goyal. *Thompson Sampling for Contextual Bandits with Linear Payoffs.* ICML 2013.
- S. Kakade, S. Shalev-Shwartz, A. Tewari. *Efficient Bandit Algorithms for Online Multiclass Prediction (Banditron).* ICML 2008.
- P. Auer, N. Cesa-Bianchi, Y. Freund, R. Schapire. *The Nonstochastic Multiarmed Bandit Problem (EXP3).* SIAM J. Computing 2002.
- F. Wu et al. *MIND: A Large-scale Dataset for News Recommendation.* ACL 2020.
- A. Bifet, R. Gavaldà. *Learning from Time-Changing Data with Adaptive Windowing (ADWIN).* SDM 2007.
