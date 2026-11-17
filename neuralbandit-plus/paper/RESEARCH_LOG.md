# Research Log — NeuralBandit++

Everything worth knowing when writing the paper or presenting: findings, decisions, deviations,
results, failures and surprises. Each entry is dated and says where it comes from.
Sections follow the paper's structure. Newest entries go at the bottom of each section.

Legend: 📄 paper/source fact · ⚖️ decision · 🔍 finding · 📊 result · ❌ failure/surprise

---

## 1. Problem & motivation (paper §1, presentation "Problem & motivation")

- 📄 *(Nov 13)* The base paper's gap, in its own words: fixed exploration γ; model-selection variants
  "need between 75.000 and 350.000 in the worst case to stabilize at each drift" (p. 7).
- 📄 *(Nov 13)* Our proposal (abstract, Oct 5): NeuralUCB + sliding window + LLM workflow; main goal =
  faster adaptation after drift; evaluated on MIND; compare cumulative reward, dynamic regret,
  adaptation time; ablations of each component.

## 2. The original method (paper §2–3, presentation "Original paper")

- 📄 *(Nov 13)* NB1: one 1-hidden-layer network per action, sigmoid + squared error, weights init in
  (−0.5, 0.5), exploration P(k) = (1−γ)·1[k = best] + γ/K; update divided by P(k) (pp. 3–4, Alg. 1).
- 📄 *(Nov 13)* NB2: 15 NB1 models (hidden {1,5,25,50,100} × λ {0.01,0.1,1}), one EXP3 picks the
  model each step; all models update the played action's network (p. 5, Alg. 2).
- 📄 *(Nov 13)* NB3: same 15 models, one EXP3 per action (p. 6, Alg. 3). The paper argues NB3 is more
  robust to drift because one bad network doesn't penalise the whole model (p. 7).
- 📄 *(Nov 13)* Paper settings: γ = 0.005, γ_model = 0.1; 10 runs averaged; each run starts at a random
  position; data "played in loop" (p. 5).
- 📄 *(Nov 13)* Paper results, stationary, last-100k classification rate: NB2 76%, CTS 73%, NB3 73%,
  LinUCB 72%, Banditron 57% (p. 6). Drift results are qualitative only (pp. 6–7). NB1 is not plotted
  in Figure 1. Figure 1 x-axis runs to ~2.25M steps (stationary panel).

- 📄 *(Nov 15, audit Nov 16)* **The paper hedges NB3's robustness:** NB3 "seem[s] to be more robust to
  nonstationarity" (p. 7). Our results don't show an NB3 advantage: on Forest Cover drift NB3 beats
  LinUCB but trails NB2 (0.696M vs 0.497M regret), and on MIND E3 it is the weakest learner — cite the
  hedge, not a firm claim (plans/phase-4.md §1).
- 📄 *(audit Nov 16)* **The paper states NeuralBandit has no regret bound** (non-convex cost, §6, p. 7; C13)
  — the contrast with NeuralUCB's guarantee that motivates our method.
- 📄 *(Nov 15)* **EXP3 as used** (Eq. 2–3, p. 5): weights start at 1; P(m) = (1−γ_model)·w_m/Σw + γ_model/M;
  only the drawn model's weight changes, w_m ← w_m·exp(γ_model·y/(P(m)·M)), reward y ∈ {0,1}. NB2/NB3 need
  no tuning phase: the paper gives every setting (plans/phase-4.md §2–3).
- 📄 *(Nov 12, REFERENCES_VERIFICATION R2, C14–C16)* **NeuralUCB facts for the method section:** regret
  Õ(d̃√T) (Thm 4.5; headline Remark 4.8, Eq. 4.5; the abstract writes Õ(√T) with d̃ suppressed — our README
  had that wrong); effective dimension d̃ = log det(I + H/λ) / log(1 + TK/λ) (Def. 4.3, Eq. 4.1); on
  **real** data Zhou et al. approximate Z_t by its diagonal for cost, not on synthetic data (§7.2 vs §7.1)
  — relevant to our NeuralUCB build.

## 3. Reproduction: ambiguities in the paper and our choices (paper "Reproduction setup")

- 🔍 *(Nov 13; superseded Nov 15 — see §7)* **Eq. 1 vs. Algorithm 1 disagree:** Eq. 1 updates the *greedy* action; Alg. 1 updates
  the *played* action. Only the played action makes the paper's own Theorem 1 (unbiased update) hold.
  ⚖️ We use the played action.
- 🔍 *(Nov 13)* **Importance weight can reach ~1,400×:** 1/P(k) for a non-greedy action = K/γ =
  7/0.005 = 1,400. ⚖️ Run weighted, capped and unweighted variants and report all.
- 🔍 *(Nov 13)* **No bias terms implied:** the paper counts connections as N = dim(X)·C + C (p. 3),
  which leaves no room for biases. ❓ Undecided: copy exactly or add biases and say so.
- 🔍 *(Nov 13)* **NB2's update probability is unspecified:** all 15 models update with "÷ P(k)", but
  each model has its own P(k), and the action was really chosen via EXP3. ❓ Undecided.
- 🔍 *(Nov 13)* **NB3's EXP3 rewards are unspecified** for the 6 actions that weren't played.
  ❓ Undecided.
- 🔍 *(Nov 15, Phase 4 plan; rev. after a paper re-read + fresh-context review)* Committee ambiguities.
  - Under Eq. 1 as printed (D10), a committee has several k̂ and P. Alg. 2 (p. 6) defines **no** k̂_t
    or P_t line at all. It only says "The model m_t choose action k̃_t" and "update ... for each model
    with (1)". Alg. 3 (p. 6) defines exactly one k̂_t and one P_t.
  - **Prose vs. Alg. 2 conflict:** p. 5 says "each neural network corresponding to the played arm is
    updated", where arm = model, so only m_t would learn. Alg. 2 says "for each model". §4 ("train the
    models in parallel") and the conclusion ("trivially parallelizable") side with Alg. 2.
  - **γ in NB2:** Alg. 2's Data line lists only γ_model; Alg. 3 lists γ and γ_model. §5 says runs use
    γ = 0.005.
  - **Minor:** γ_model ∈ [0, 1] in the text (p. 5) vs. [0, 0.5] in Alg. 2–3. The p. 5 prose says NB3's
    EXP3s have "less possibility to update", which fits only the played action's EXP3 learning.
  - **Limitations of our reading:**
    - NB2's weight 1/P_{m_t}(k̂) ignores the true mixture probability, so Theorem 1 holds only
      approximately.
    - NB3's EXP3_k doesn't correct for how often action k is played, so it may favour models that
      over-score k.
  - **Process note:** the first draft called "each model gates on its own k̂_m" the literal reading of
    Alg. 2. The fresh-context reviewer showed Alg. 2 defines no k̂, so the user switched to the chosen
    model's k̂.
  - ⚖️ User decisions (Nov 15):
    - NB2 updates every model (Alg. 2).
    - Gate: the chosen model's k̂_{m_t}, weight 1/P_{m_t}(k̂) ≈ 1.004 (an interpretation).
    - The chosen model explores with γ = 0.005 (an inference).
    - NB3 gate: Alg. 3's k̂_t and P_t.
    - NB3: only the played action's EXP3 receives y.
    - Done-check: NB2 below LinUCB in both panels. Required toy tests: EXP3 and the committee find the
      clearly best model. Logged: NB2/NB3 vs. NB1 and EXP3's final weights.
- 🔍 *(Nov 15, Phase 4 build)* **A toy case of the D3 limitation (NB3).**
  - Setup: toy problem, K = 3, one-hot context, two models: λ = 1, and λ ≈ 0 (frozen).
  - NB3's EXP3 for one action locked onto the **frozen** model (P = 0.95) in 2 of 6 seeds. NB2 found
    the learning model (P ≈ 0.95) in all 6.
  - Likely reason (not verified): EXP3_k only sees a reward when action k is played. A frozen model
    that always ranks k on top gets k played often, and on those plays it earns about the same reward
    rate as the learning model, so EXP3_k can't tell them apart.
  - This fits the paper's own remark that NB3's EXP3s have "less possibility to update" (p. 5). Check
    for it in the real runs via the saved EXP3 model probabilities.
- ⚖️ *(Nov 12)* EXP3 weights kept in log-space (they overflow over millions of steps).
- ⚖️ *(Nov 15, Phase 3 plan)* NB1 follows the paper's text throughout: **no biases** (N = dim(X)·C + C,
  p. 3); sigmoid hidden **and** output; plain gradient step Δ = λ·x̂·δ (no momentum); **weighted**
  ×1/P update as in Eq. 1 / Alg. 1 (capped/unweighted only as diagnostics if weighted fails); loss
  ½(o − r)² (paper says only "quadratic error"; the ½ is the usual backprop convention — our choice);
  C and λ tuned on the paper's own 15-setting grid with tuning seeds 100–102 (paper gives NB1 no
  setting). NB1 counts as working if weights stay finite, it is clearly above the 48.8% majority line,
  and its regret is below Random's.
- ⚖️ *(Nov 15)* Implementation: numpy with hand-written backprop (maps line-for-line onto the paper's
  Δ = λ·x̂·δ), not torch — same maths, ~10× faster at this size. Verified by finite differences, a
  sparse-vs-dense equality test, and a one-time cross-check against torch autograd. The paper does
  not state its software (it predates TensorFlow/PyTorch); it describes backprop by hand, citing
  Rumelhart et al. [25].
- 🔍 *(Nov 15)* **First author's reference repo** (github.com/TheAmazingElys/NeuralBandit; unlicensed,
  self-described WIP, NB2/NB3 not implemented; read only). Confirms: regret = T·0.93 − total reward
  (our constant reference) and the same qcut-5 + one-hot binning. Differs from the paper's text: its
  NB networks have **biases**, sigmoid on hidden **and output**, MSE loss, **no importance weighting**,
  **SGD momentum 0.9**, example γ = 0.05 and 64 hidden units; it reshuffles the data every pass. The
  repo lists the author as "Elys Allesiardo" (README citation); we keep the PDF's "Robin Allesiardo".
  It also cites a follow-up, Allesiardo & Féraud, "Selection of Learning Experts", IJCNN 2017
  (DOI 10.1109/IJCNN.2017.7965962) `[UNVERIFIED]`, described as theory for the committee — possibly
  useful for NB2/NB3's open questions.
- 📄 *(Nov 13)* Baseline hyperparameters (LinUCB α etc.) are not given in the paper → our choice,
  must be reported.

- 📄 *(Nov 15)* LinUCB = Li, Chu, Langford, Schapire, "A Contextual-Bandit Approach to Personalized
  News Article Recommendation", WWW 2010 (arXiv:1003.0146; cited as ref. [17] in the base paper).
  Checked against the arXiv v2 PDF (user copy, 10 pp.; Nov 15). Alg. 1 (§3.1, p. 4): A_a = I,
  b_a = 0; θ_a = A_a⁻¹b_a; score θ_aᵀx + α√(xᵀA_a⁻¹x); ties "broken arbitrarily"; only the chosen
  arm updates (A += xxᵀ, b += r·x). Eq. 4 (p. 3): α = 1 + √(ln(2/δ)/2), which "may be conservatively
  large". α tuned on a separate day of "tuning data" (§5.2.1, p. 6; §5.5.1, p. 8); Fig. 2 (p. 7) shows
  an inverted-U of reward vs. α over roughly 0–1.4; chosen value not reported. They append a constant
  feature 1 and normalise feature vectors to unit length (§5.2.2, p. 6) — ⚖️ we do neither: every
  algorithm gets the base paper's same 94 binary features (decision D3, Nov 15).
- 🔍 *(Nov 15)* The base paper states the 94 binary features once for the whole experiment (§5,
  p. 5) and is silent on LinUCB's bias feature, normalisation and α — our choices don't contradict
  it but can't claim to match it. Note: every row has exactly 12 ones, so every context has the same
  length √12; unit-normalising would divide all contexts by one constant, which for LinUCB only
  amounts to a different ridge strength and a rescaled α (absorbed by α tuning).
- ⚖️ *(Nov 15, Phase 2 plan)* Regret = Σ(0.93 − reward) (constant reference). LinUCB α tuned on a grid
  {0.01…2} with separate tuning seeds 100–102, 500k steps. No constant feature (94 one-hot features
  already imply one: feature rank is 82). Random tie-breaking. LinUCB counts as reproduced if its
  stationary last-100k accuracy is within 72% ± 3 points.
- 🔍 *(Nov 15)* Under the 0.93 reference, Random's expected regret slope is 0.93 − 1/7 ≈ 0.79/step
  (≈1.8M at 2.25M steps), consistent with Figure 1's Random line ending near 2M.

- ⚖️ *(Nov 15, Phase 2 Q1 → D1)* **The regret reference is our reading:** the paper says regret is
  "computed from the rewards of an offline algorithm fitting the data with 93% of classification" (§5,
  p. 5) and never names that algorithm or says the reference is constant per step; we use a constant
  0.93 per step (user decision) — state it as an interpretation (plans/phase-2.md §3/§6; C2).
- ⚖️ *(Nov 15, Phase 4)* **NB2/NB3 implementation defaults not in the paper:** one seeded rng feeds the 15
  models' initial weights, EXP3 draws and ε-greedy draws; committee start weights are not matched to the
  standalone NB1 runs; ties (incl. across models in NB3's argmax) broken uniformly at random; NB2's
  logged action probability is the true mixture Σ_m P_model(m)·P_m(k), while its update weight uses
  1/P_{m_t}(k̂) (plans/phase-4.md §3–4).
- ⚖️ *(Nov 15, user)* **Faithfulness rule:** the 2014 paper is the source of truth; the author's repo is
  never a source for any decision — where it differs (biases, no importance weighting, momentum,
  γ = 0.05) we follow the paper. **Process rule:** every paper ambiguity goes to the user as a decision,
  never assumed (CLAUDE.md).
- ⚖️ *(Nov 15, Phase 5 Q16)* On MIND the constant-1 input is the only bias-like input of NB1/2/3, so it
  acts like a first-layer bias the 2014 networks don't have — a deliberate, documented MIND input choice
  (follows LinUCB's news experiment, C24), not a change to the networks.

## 4. Data & environment (paper "Experimental setup")

- 📄 *(Nov 13)* Forest Cover Type, UCI (Blackard, 1998; DOI 10.24432/C50K5N; CC BY 4.0):
  **581,012 rows**, 54 features, 7 classes. The paper rounds this to "581.000".
- ⚖️ *(Nov 13)* Downloaded with scikit-learn 1.9.1 `fetch_covtype` (simplest; same data).
- 🔍 *(Nov 13)* Paper's encoding gives exactly 94 features: 10 numeric × 5 equal-frequency bins = 50,
  + 4 wilderness + 40 soil. Every row has exactly one wilderness and one soil column set →
  **exactly 12 ones per row**.
- 🔍 *(Nov 13)* **Slope's bins are uneven** (99,685–141,865 rows vs ~116k for the others) because many
  rows share the same slope value; exact equal frequency is impossible with ties. ⚖️ Accepted.
- 🔍 *(Nov 13)* **Class imbalance:** shares 36.5%, 48.8%, 6.2%, 0.5%, 1.6%, 3.0%, 3.5%. Always guessing
  class 2 scores 48.8% — a useful floor when reading results (and context for Banditron's 57%).
- ⚖️ *(Nov 13, Phase 1 plan)* Drift = cumulative class rotation (+1 at 500k, +2 at 1M, …) counted
  from each run's start; bin edges computed once on all rows (feature values only); data shuffled
  once (seed 0); for each seed every algorithm starts at the same row ("identical environments").
- ⚖️ *(Nov 15, Phase 1 code review)* The environment's start row uses its own random stream
  (`default_rng((seed, 1))`), separate from the algorithm's `default_rng(seed)`, so a run's start row
  and the algorithm's exploration draws are independent. Contexts are handed to algorithms as float32
  copies (prevents integer wrap-around and protects the shared dataset from in-place edits).
- 📊 *(Nov 15)* Phase 1 done: processed data 581,012 × 94, all 0/1, exactly 12 ones per row; class
  shares match; seeded, looping, drifting stream verified (+1 at 500k, +2 at 1M, 7 shifts wrap to 0).

### MIND (Phase 5 plan, decided with the user Nov 15–16; full reasoning in `paper/plans/phase-5.md` §9)
- 📄 *(Nov 15)* **Access:** the official page links to a gated HF copy (`yjw1029/MIND`, login needed);
  `Recommenders/MIND` (used by Microsoft's `recommenders` library) is open; the old Azure blob is dead
  (HTTP 409). Zip sizes differ by ~0.08% between the two HF copies (re-zipped) → we use the open copy,
  record SHA-256 + row counts, and compare unzipped files once against the official copy.
- 📄 *(Nov 15)* **License** (MSR License Terms, read in full): research use and publishing results allowed
  if "no material portion of the Dataset" is included; **distribution forbidden**; accepted by use.
  User agreed. → no MIND data (raw or processed) in git; no real rows in the paper/demo.
- 📄 *(Nov 15)* **Time span:** small and large both have usable impressions only for **week 5** (≈ 1 week,
  Nov 2019); large's test week has no public labels. → natural drift may be small (side-agent note).
- ⚖️ *(Nov 16)* **Small vs large:** build and tune on MIND-small; final MIND results on MIND-large
  **minus small's 50k users** (small ⊂ MIND, C20) → tuning and evaluation never share a person. One large
  pass ≈ 2.3M steps [UNVERIFIED] ≈ Forest Cover's 2.25M-step runs.
- ⚖️ *(Nov 16)* **Stream:** train + dev glued, chronological, one pass (looping would fake an end→start
  drift), each seed keeps a random 90% of users (same subset for every algorithm).
- ⚖️ *(Nov 16)* **Category-arm bandit:** arms = shown categories; reward = 1 if the user clicked any
  shown article of the chosen category; **masking** — only categories shown in the impression can be
  chosen. Source: LinUCB is defined for a per-trial arm set A_t and "works well for a dynamic arm set"
  (Li et al. 2010 §2.1 p. 2, §3.1 p. 3; C23). Why it matters: a 0 then means "seen, not clicked", so
  measured drift is **user-interest drift**, not MSN changing what it displays — matches the abstract.
  Adaptation for NB1/2/3: Eq. 1's P(k) is over available arms (not in the 2014 paper). Limitation: MSN
  decided what was available.
- ⚖️ *(Nov 16)* **Context:** category shares of the user's click history + **constant 1**, as in LinUCB's
  own news experiment (shares "normalized so that they sum up to unity" + "a constant feature 1",
  Li et al. 2010 p. 6; C24). History grows during the week with the user's own logged clicks (plain
  counts; recency weighting would hand baselines part of our sliding window). MiniLM and subcategories
  not used: arms are categories, so within-category detail barely changes the decision.
- ⚖️ *(Nov 16)* **Metric:** dynamic regret vs. the best available arm (= 1 if the impression has any click)
  + cumulative reward — both named in the abstract. Zero-click impressions kept.
- ⚖️ *(Nov 16)* **Protocol:** one runner for both datasets (Forest Cover must reproduce exactly);
  α tuned on small (seeds 100–102, Forest Cover's grid) by total reward over the pass; sanity check
  includes a hindsight "most popular available category" reference line (not a baseline).
- ⚖️ *(Nov 16)* **Data review gate:** after download/counting, stop and re-check every MIND decision
  against the real data before building the environment (user).
- 📊 *(Nov 16, Phase 5a)* **MIND-small facts** (`results/mind_small_facts.txt`, aggregates only). Download
  9 s; zip sizes match the HF listing; SHA-256 recorded in the script output.
  - 230,117 impressions: train 156,965 (Nov 9–14, 2019), dev 73,152 (Nov 15); dev follows train; no
    duplicates. ≈ 207k steps per run after the 90% user cut.
  - 🔍 **The last day is 32% of the stream:** ~14k impressions/day on the weekend, ~32k on weekdays,
    73k on Nov 15 (dev). Dev samples its own 50,000 users for one day.
  - 🔍 Users: 50,000 per file but 94,057 overall (only 5,943 in both) — the site's "50,000 users" is per file.
    Impressions per user: median 2, mean 2.45, p99 13.
  - Every impression has ≥ 1 click (min 1, median 1, max 35) → **no zero-click impressions** (Q12 moot);
    the best-available-arm reward is always 1, so dynamic regret = steps − reward. CTR 4.05%.
    Articles shown per impression: median 24 (2–299).
  - **Arms:** 18 categories in news.tsv, **16 shown**. Two are almost never shown: kids (0.07% of
    impressions), northamerica (0.01%) — clicked in 1.9% / 3.4% of the few impressions that show them. The rest are
    shown in 35–96% of impressions.
  - **Categories shown per impression:** mean 8.8 of 16 (55%), median 10 → masking (Q11) matters.
  - Random's expected reward with masking: 0.1995. Always-best-available = 1.0.
  - History: median 19 clicks; **5,452 impressions have an empty history** (contradicts the MIND paper's
    "non-empty history" rule for full MIND); all history IDs are in news.tsv.
  - **History never changes within the week:** 0 of 47,827 multi-impression users have >1 distinct
    History → the Q14 online update is the only way context follows the week.
  - 🔍 **Natural drift is visible day to day** (share of impressions with a click in the category):
    tv 18.8% (Nov 9) → 5.0% (Nov 10); music 4.7% → 25.0% (Nov 10) → 7.5%; finance 21.9% on Nov 12 vs
    5–11% otherwise; sports 10.8% → 30.9% (Nov 15). Likely news events — strong, short spikes.
- 📊 *(Nov 16, Phase 5a)* **MIND-large facts** (`results/mind_large_facts.txt`; download 84 s, 1.7 GB
  unzipped; counting 90 s, 2.3 GB RAM after a line-by-line rewrite of the facts script — the first
  version kept every article list in memory, grew to 22 GB on a 16 GB Mac and stalled in swap).
  - 2,609,219 impressions: train 2,232,748 (Nov 9–14), dev 376,471 (Nov 15); 750,434 users (the 1M in
    the MIND paper includes test-week users); **6 impressions appear in both files** (drop duplicates).
  - 🔍 **Last day = 14% of the stream on large** (32% on small): small's dev over-weight is a small-only
    artifact of sampling 50k users per file.
  - Same 16 shown categories; kids and northamerica again almost never shown (0.09%, 0.02%).
    Categories shown per impression: mean 8.7 of 16 (54%). CTR 4.06%. No zero-click impressions.
    Random's expected reward 0.204 (small 0.200).
  - Impressions per user: median 2, mean 3.48, p90 7, p99 17 (small: mean 2.45, p90 5) — slightly more
    per user on large, not fewer as estimated. Empty histories: 57,335 (2.2%). History never changes
    within the week (0 of 533,415 multi-impression users).
  - 🔍 **Daily category shares on large match small almost exactly** (e.g. music 24.8% vs 25.0% on Nov 10,
    finance 21.9% on Nov 12 in both) → small is representative for tuning, and the drift is real.
  - ✅ **Q4 official-copy check passed:** every file inside all four official zips (small/large ×
    train/dev, downloaded by the user from msnews.github.io → gated HF) is SHA-256-identical to our open
    `Recommenders/MIND` copy. Official large zips add an empty `__placeholder__`; the zip-size
    difference is re-compression only. Paper wording: "obtained from the Microsoft Recommenders copy,
    verified byte-identical to the official release" (C28).
  - **Q22 split works:** all 94,057 small users are in large; large minus them = 2,270,818 impressions,
    656,377 users → ≈ 2.04M steps per run after the 90% cut (Forest Cover runs: 2.25M).

- 📄 *(audit Nov 16; REFERENCES_VERIFICATION C17–C20)* **MIND paper facts for context:** full MIND =
  1,000,000 users, 161,013 news, **20 categories**, 15,777,377 impressions, 24,155,470 clicks (R7 Table 2,
  p. 3600); 6 weeks Oct 12 – Nov 22 2019, users with ≥ 5 clicks; training = week 5, validation = its last
  day, test = week 6 (labels hidden), history from weeks 1–4, only non-empty histories kept (§3.1).
  Categories: **20 in the MIND paper (full MIND) vs 18 in our news.tsv vs 16 ever shown** (= our arms).
  The site's "50,000 users" for MIND-small (C20, one source) is per file — 94,057 distinct (C25); cite carefully.
- ⚖️ *(Nov 15, Phase 5 Q9, fixed before seeing data)* Arm rule = every category ever shown; no merging, no
  top-K. kids (0.09%) and northamerica (0.02%) stay — with masking they are almost never available, so they
  cost only compute.
- ⚖️ *(Nov 15, Phase 5 Q10)* **Reward-definition limitations to state:** what a user would have done with an
  unshown category is unknown (counterfactual); a category shown many times in one impression has more
  chances of a click; Li et al.'s replay estimator is not used because MSN's logging policy was not
  uniformly random.
- 🔍 *(Nov 16, data review)* **Masking evidence:** 8.7 of 16 categories shown per impression (54%, below
  the 80% cut-off agreed in Q11), 10% of impressions show ≤ 3 → without masking about half the arms would
  be fake zeros each step. **Constant-1 evidence:** 57,335 empty-history impressions on large (2.2%),
  5,452 on small → the constant is actually needed (cold start); those impressions are kept.
- ⚖️ *(Nov 15–16, Phase 5 Q7/Q22)* Protocol: algorithms learn from scratch in every run; only chosen
  settings carry over from small to large; each seed leaves out 10% of users (≈ 66k on large).

## 5. Scope & experimental design decisions

- ⚖️ *(Nov 12)* Instructor's success criterion: cumulative regret vs. the paper's Figure 1; NeuralBandit
  must beat LinUCB (Prof. Bouneffouf, Slack).
- ⚖️ *(Nov 13)* Baselines = NB1, NB2, NB3 + **one** classical baseline, **LinUCB** (slides require "at
  least one"; LinUCB because the instructor's check uses it). Random kept as a sanity line.
  Banditron and CTS not reproduced — our Figure 1 will omit them.
- ⚖️ *(Nov 13)* Every baseline runs in every environment (slides: "identical environments and
  evaluation protocols") — Forest Cover and MIND.
- ⚖️ *(Nov 13)* Ablations = core 4: −LLM controller; LLM → rule-based controller; −UCB; −sliding window
  (slides: "remove one component at a time").
- ❓ *(Nov 13)* Abstract evaluates the new method on MIND; README's E2 also runs it on Forest Cover
  drift. To settle in the Phase 5 plan.
- ⚖️ *(Nov 16, Phase 5 Q25 — resolves the ❓ above)* The new method runs on **both**: MIND (promised in
  the abstract) and Forest Cover drift (E2), for a head-to-head with the paper's "75k–350k" steps to
  stabilise — the abstract's own motivation. Whether ablations also run on E2: decided later (cost
  incl. LLM API calls).
- ⚖️ *(Nov 16, Phase 5 Q24)* MIND has two experiments: **E4** (planted shifts at known times) is the main
  adaptation-time test — adaptation time needs a known drift moment; **E3** (real week, untouched) is the
  real-behaviour check. Revisit after the data review.
- ⚖️ *(Nov 16, Phase 5 Q26)* Article-level (shared-network) MIND formulation → future work.
- ⚖️ *(Nov 16, Phase 5c plan, all decided with the user; full text in `paper/plans/phase-5c.md` §8)*
  **How the MIND baselines were run (paper "Experimental setup"):**
  - Tuning budget = only what the paper leaves open: NB1's C and λ re-tuned on MIND-small over the paper's
    15-setting grid (seeds 100–102, best total reward); LinUCB's α likewise (Phase 5b). The paper's
    γ = 0.005 and γ_model = 0.1 kept unchanged — NB stays the published algorithm, and the abstract's gap
    is exactly NB's *fixed* exploration. NB2/NB3 use the paper's full 15-model grid, nothing tuned.
    Paper sentence: "we use the paper's γ, γ_model and tune only the settings it leaves open".
  - 10 seeds (0–9) as on Forest Cover; popularity line included as a labelled hindsight reference.
  - Success on MIND is not pass/fail on "NB beats LinUCB" (no instructor criterion exists for MIND; making
    it pass/fail would invite tuning toward a result). Done-check = runs complete and finite, Random ≈ its
    analytic value, outputs produced.
  - Statistics: mean ± std now; significance tests in one later step for all experiments (paired over
    seeds is possible: every algorithm sees the same users within a seed; per-step rewards are kept in
    `results/raw/`, local only).
  - Figures: day boundaries on the regret plot = mean day-start step over seeds (seeds differ by ≤ 2,436
    steps = 0.12% of a run); the per-day chart uses each seed's own boundaries.
- ⚖️ *(Nov 16)* **Phase order:** 5c = baselines on MIND E3 → 5d = design E4 (planted shifts) + baselines
  on it → Phase 6 = the new method (planned in a separate session). E4 is designed **before** the method's
  results exist, so the main drift test can't be shaped (even unintentionally) to suit the method.
- ⚖️ *(Nov 16, data review)* **E3 is worth more than expected** (user): the real week has strong,
  event-like interest drift (daily category click shares: music 5% → 25% → 8% around Nov 10, finance 22%
  on Nov 12, sports 11% → 31% on Nov 15; identical pattern in small and large). E3 is therefore a real
  adaptation test, not only a sanity check; the daily table motivates it in the paper. E4 stays the
  precise adaptation-time test (natural spikes have no clean start moment at our daily resolution).
- ⚖️ *(Nov 16, data review — side-agent note)* On MIND every impression has ≥ 1 click, so dynamic regret
  = steps − cumulative reward: **one metric, not two**. Paper: MIND regret curves (Figure 1 style) as
  the main charts, click rates in tables/text, one sentence stating the equivalence; never presented as
  two independent wins. On Forest Cover they differ (regret uses the 0.93 reference).
- ⚖️ *(Nov 16)* **Data review result:** all MIND decisions re-checked against small and large; none
  changed. Additions: drop large's 6 duplicate impressions; E3 upgraded in importance.
- ⚖️ *(Nov 16, Phase 5b build)* Context = shares over **all 18** news categories (incl. the 2 never shown,
  so no past click is discarded) + constant 1 → **d = 19**; arms = the 16 shown categories. Processed:
  `mind_small.npz` 230,117 rows; `mind_large.npz` 2,270,815 rows (small's users and 3 duplicate pairs
  removed). Masking implemented in every policy; with no mask (Forest Cover) every code path and random
  draw is unchanged — **Forest Cover reruns are step-for-step identical** to the saved Phase 2–4 runs
  (all algorithms, both panels, 2 seeds × 50k steps; a 5k-step version is a permanent test).
- 📊 *(Nov 16, Phase 5b)* **MIND-small sanity + LinUCB α tuning** (`results/phase5_mind_tune_summary.csv`;
  tuning seeds 100–102, ≈ 207k steps each, 27 runs in 75 s). Mean reward (click rate of the chosen
  category):

  | | seed 100 | 101 | 102 |
  |---|---|---|---|
  | Random | 0.199 | 0.199 | 0.201 |
  | Popularity (hindsight reference) | 0.359 | 0.360 | 0.360 |
  | LinUCB α = 0.01 … 2 | 0.431–0.437 | 0.433–0.437 | 0.431–0.437 |
  | **LinUCB α = 0.5 (best)** | **0.437** | **0.437** | **0.437** |

  - ✅ Random 0.1993 = its analytic value 0.1995 → reward wiring and masking are correct.
  - ✅ Every LinUCB α beats Random (×2.2) **and the popularity line** (0.43 vs 0.36) on every seed →
    the category-share context carries real personal signal.
  - **α = 0.5** is best on all 3 seeds (chosen by total reward, plan Q23). As on Forest Cover, α barely
    matters (0.432–0.437). Use α = 0.5 for LinUCB on MIND.
  - LinUCB runs at ~8k steps/s with 9 parallel jobs → a 2.04M-step MIND-large run ≈ 4–5 min.
- 📊 *(Nov 16, Phase 5c)* **NB1 tuning on MIND-small** (`results/phase5c_tune_nb1_summary.csv`; paper grid,
  seeds 100–102, one pass ≈ 207k steps, 45 runs in 4 min). Mean click rate: **C = 1, λ = 1: 44.6%** (best
  on all 3 seeds) ≈ C = 5, λ = 1: 44.5% > C = 25, λ = 1: 44.0% … C = 100, λ = 1: 40.0%; λ = 0.01 is worst
  for small nets (C = 1: 36.7%). → **NB1 on MIND = C = 1, λ = 1** (user confirmed at the checkpoint).
  - 🔍 The **opposite** of Forest Cover (C = 100, λ = 1 best there). With one hidden unit and no biases,
    NB1 is close to a per-arm logistic model of the category shares — the MIND category context looks
    mostly linear. Tuned NB1 (44.6%) already edges LinUCB (43.7%) on small.
  - ⚠️ Caveat: tuning passes are ~10× shorter than the final MIND-large runs (≈ 207k vs 2.04M steps), which
    may favour small, fast-learning networks; Forest Cover had the same issue (500k vs 2.25M).
  - ❓ **For Phase 6 and the paper's framing (side-agent note, Nov 16):** if MIND's category context is
    mostly linear, NeuralUCB's edge over LinUCB on MIND may come mainly from **adapting to drift**, not
    from being neural — frame MIND results around adaptation, and don't promise non-linear gains there.
    **Evidence to check after the E3 runs:** NB2's EXP3 model weights on the full 2.04M-step runs
    (saved per run) show whether large networks win once runs are long — that tests the caveat above.
- 📊 *(Nov 16, Phase 5c)* **MIND-large E3 (real week), baselines** (`results/phase5c_mind_e3_table.csv`,
  `_daily.csv`; figures `mind_e3_regret.png`, `mind_e3_daily.png`, `mind_daily_categories.png`). 60 runs
  (6 × seeds 0–9), ≈ 2.04M steps each, 2 h 29 min on 8 cores (M2: 4 performance + 4 efficiency cores).
  Done-check passed: all runs finite; Random − analytic ≤ 0.0003 on every seed.

  | | click rate (mean ± std) | final regret |
  |---|---|---|
  | **NB1** (C = 1, λ = 1) | **46.0% ± 0.04** | **1.104M** |
  | NB2 | 45.6% ± 0.06 | 1.111M |
  | LinUCB (α = 0.5) | 44.7% ± 0.03 | 1.130M |
  | NB3 | 43.6% ± 0.14 | 1.153M |
  | Popularity (hindsight reference) | 37.2% ± 0.01 | 1.283M |
  | Random | 20.4% ± 0.02 | 1.626M |

  - 🔍 **NB1 and NB2 beat LinUCB on MIND too** (+1.3 and +0.9 points), as on Forest Cover; NB3 is below
    LinUCB. All learners beat the fixed popularity rule by 6–9 points → personalising and adapting pay.
    Seed spread is tiny (≤ 0.14 points), so the ranking is stable (formal tests: later statistics step).
  - 🔍 **Per day:** LinUCB is best on the first day (learns fastest), NB1/NB2 lead from Nov 10 on. Every
    learner peaks on **Nov 10** (the music spike). On **Nov 15** (sports spike) LinUCB stays at 41.8%
    while NB1/2/3 jump to 45–46%; the fixed popularity rule falls steadily (41% → 31%) as interests move
    away from its whole-week ranking — the clearest picture of natural drift hurting a non-adaptive rule.
  - 🔍 **Answer to the "is MIND linear?" question:** on the full 2.04M-step runs, **NB2's EXP3 puts 51% of
    its weight on C = 25** (λ = 1), 16–17% each on C = 1 and C = 5, 14% on C = 50, **2% on C = 100**
    (argmax: C = 25 in 6/10 seeds). So with long runs a **mid-size network is preferred** over C = 1 —
    the tuning winner (C = 1) was partly a short-run effect, and MIND is **not purely linear**. But very
    large networks gain nothing, the margins are small (NB1 C = 1 still has the best overall click rate,
    NB2 pays for running 15 models), and λ = 1 wins everywhere. **For Phase 6:** frame MIND gains mainly
    around adaptation; some, but modest, non-linear structure exists.
  - 🔍 **NB3's lock-on reappears, stronger:** 69% of its per-category EXP3s end on the weakest model
    (C = 1, λ = 0.01) — same mechanism as Phase 4 (only the played category's EXP3 is rewarded) → NB3 is
    the weakest learner on MIND.
- 📝 *(Nov 16, user: keep for the paper)* **Two definitions to state in the paper:**
  1. On MIND every impression has ≥ 1 click, so **dynamic regret = steps − cumulative reward** — one
     metric shown two ways, never two separate wins (regret charts; click rates in tables/text).
  2. The **popularity reference line** ranks categories by their share of **all** impressions in which
     they were clicked (unclicked or unshown = 0), not by click rate when shown; it plays the highest-
     ranked category among those shown. Hindsight (whole run), non-learning, not a baseline.
- 📊 *(Nov 16)* **NB2 speed on MIND** (measured, 20k steps, one job alone): 408 µs/step vs 325 µs on
  Forest Cover (×1.26) — 16 arms vs 7 cost more, partly offset by fewer non-zero inputs (median 8 vs 12;
  the networks only process non-zero inputs). → one MIND-large run ≈ 14 min alone, ≈ 60–70 min with 8 in
  parallel (Phase 4 NB2: 46–54 min) → ≈ 2.5 h for 10 seeds of NB2. (Side-agent note predicted ×2; the
  earlier "40–50 min" estimate ignored the arm count.)

- 📝 *(CLAUDE.md, "Ground truth")* **Workflow lesson:** we first assumed all of the paper's classical
  baselines were required; the slides only say "at least one", so we build one (LinUCB). Since then every
  decision traces to a source, in this order: course slides + instructor Slack > approved abstract >
  base paper > primary papers of adopted methods > user decisions; anything none of them requires is out
  of scope. (Good methodology point for the presentation.)
- ⚖️ *(Nov 13)* "Common toolkit" integration deferred — not raised until the user does.
- 📅 **Deadlines:** instructor checkpoint Wed Nov 18 (NB2 must beat LinUCB — passed, Phase 4); paper
  working target Sun Nov 29 (official Dec 1); presentation + demo Mon Dec 14.
- ⚖️ *(Nov 15, Phase 2 D7)* Forest Cover metrics so far: cumulative regret vs 0.93 + last-100k accuracy;
  dynamic regret and a formal **adaptation-time definition are still open** — the only recovery measure
  so far is our ad-hoc one in §6 (Phase 4). To define in Phase 5d (E4) and apply to E2 as well.
- ⚖️ *(Nov 16, Phase 5c Q3)* If NB3's lock-on reappeared on MIND it would be reported, not fixed (the
  baseline is NB3 as published) — it did reappear (69%).
- ⚖️ *(Nov 16, Phase 5 Q17)* Because regret = steps − reward on MIND, the **independent** MIND evidence of
  adaptation is adaptation time (E4) and reward around drifts (E3).

## 6. Results

- 📊 *(Nov 15)* **NB1 tuning** (Eq. 1 as printed; stationary, 500k steps, tuning seeds 100–102;
  `results/phase3_tune_nb1_summary.csv`). Last-100k accuracy, mean over 3 seeds:

  | C \ λ | 0.01 | 0.1 | 1 |
  |---|---|---|---|
  | 1 | 58.2% (± 8.6) | 60.0% (± 9.6) | 60.8% (± 10.6) |
  | 5 | 69.0% | 72.4% | 73.8% |
  | 25 | 69.6% | 73.6% | 77.0% |
  | 50 | 69.8% | 74.0% | 77.5% |
  | 100 | 70.2% | 74.1% | **77.5%** |

  ⚖️ C = 100, λ = 1 chosen (best mean, per D5; C = 50, λ = 1 within 0.1 points). 🔍 Clear pattern:
  **larger λ is always better** (the paper's largest grid value wins every row), and accuracy rises
  with C up to ~25–50 then flattens. C = 1 is unstable across seeds (± 9–11 points). Even at 500k
  steps NB1 is already ~5.6 points above tuned LinUCB (71.9%) and above the paper's NB2 (76%).
  Compute: 45 runs × 500k in 8 min on 8 cores.
- 📊 *(Nov 15)* **NB1 main runs** (Eq. 1 as printed, C = 100, λ = 1, γ = 0.005; 2.25M steps, seeds 0–9;
  `results/phase3_*_summary.csv`, Figure 1). Last-100k accuracy, mean ± std:

  | | Stationary | Drift | Final cum. regret (stat. / drift) |
  |---|---|---|---|
  | Random | 14.3% | 14.3% | 1.771M / 1.771M |
  | LinUCB | 71.9% ± 0.1 | 27.7% ± 10.6 | 0.475M / 1.157M |
  | **NB1** | **82.0% ± 0.2** | 54.0% ± 21.5 | **0.311M / 0.689M** |

  ✅ NB1 passes D7 (finite weights in all runs, far above 48.8%, far below Random's regret) and
  **beats LinUCB on cumulative regret in both panels** (−35% stationary, −40% drift) — the
  instructor's criterion, already met by NB1 alone.
- 🔍 *(Nov 15)* **NB1 vs. the paper:** our NB1 (82.0%) is above the paper's best number (NB2 76%);
  the paper reports no NB1 number. Possible reasons: tuning NB1 on its own (the paper's NB2 must pick
  among 15 models online, which costs learning time), the update reading, or details the paper
  leaves out. ❓ Worth a careful sentence; don't claim "we beat the paper" — different protocol.
- 📊 *(Nov 15)* **NB1 under drift gets worse with every drift.** Regret per 500k segment: 91k (before)
  → 109k → 173k → 211k after the 1st/2nd/3rd drift (LinUCB: 106k → 307k → 294k → 268k). NB1 recovers
  much better than LinUCB at first but degrades with each drift, and seeds diverge (± 21.5 points at
  the end). Fits the paper's gap (fixed γ; networks don't "unlearn") and motivates our proposal.
  Hypothesis to check later: under Eq. 1 the networks only learn from greedy plays, so after a drift a
  network that is no longer greedy never gets corrected unless exploration makes it greedy again.

- 📊 *(Nov 15)* **LinUCB α tuning** (stationary, 500k steps, tuning seeds 100–102; `results/phase2_tune_alpha_summary.csv`):
  last-100k accuracy (mean ± std over 3 seeds):

  | α | 0.01 | 0.05 | 0.1 | 0.25 | 0.5 | 1 | 2 |
  |---|---|---|---|---|---|---|---|
  | acc | 71.80 ± 0.14 | 71.81 ± 0.20 | **71.88 ± 0.07** | 71.83 ± 0.15 | 71.87 ± 0.06 | 71.81 ± 0.12 | 71.74 ± 0.06 |

  🔍 LinUCB is **almost insensitive to α** on Forest Cover: all values within 0.15 points, inside seed
  noise. No inverted U as in Li et al.'s Fig. 2. ⚖️ α = 0.1 chosen (best mean, per D2). Even untuned,
  LinUCB already lands at ~71.8% — right at the paper's 72%. Likely reason (to discuss): with
  binary one-hot features and 7 fixed arms, the ridge estimate converges quickly and exploration
  bonuses shrink fast, so the exploration scale matters little after the first few thousand steps.
- 📊 *(Nov 15)* **Phase 2 main runs** (2.25M steps, seeds 0–9, LinUCB α = 0.1; `results/phase2_*_summary.csv`,
  `results/figures/figure1.png`). Last-100k accuracy, mean ± std over 10 seeds:

  | | Stationary | Drift (every 500k) | Paper (stationary) |
  |---|---|---|---|
  | Random | 14.25% ± 0.12 | 14.28% ± 0.08 | — (1/7 = 14.3%) |
  | LinUCB | **71.94% ± 0.14** | 27.69% ± 10.64 | 72% |

  ✅ LinUCB reproduces the paper's 72% (inside the ±3-point target). Final cumulative regret at 2.25M
  (mean): Random 1.771M (stationary) / 1.771M (drift); LinUCB 0.475M / 1.157M.
- 📊 *(Nov 15)* **Drift hurts LinUCB, as the paper says.** LinUCB regret per 500k-step segment:
  106k (before drift) → 307k → 294k → 268k after the 1st/2nd/3rd drifts — ~3× steeper and it never
  recovers to the pre-drift rate (it never forgets old data). Random is flat at ~394k per segment.
  Its drift accuracy also varies a lot across seeds (± 10.6 points). This is exactly the gap our
  proposal targets.
- 🔍 *(Nov 15)* Rough comparison with Figure 1 (read by eye): Random's line matches (~1.8M at 2.25M).
  Our stationary LinUCB (~0.48M) is close to or slightly below the paper's LinUCB line (~0.5M). Our
  drift LinUCB (~1.16M at 2.25M) looks lower than the paper's (~1.4M+ near the end of its axis) —
  same qualitative behaviour, less severe. Possible reasons: different α (unreported in the paper),
  bias/normalisation choices, or how exactly classes were swapped. ❓ Worth a sentence in the paper;
  not blocking.
- 📊 *(Nov 15)* **Phase 4 main runs: NB2 and NB3** (committee of 15 models, C ∈ {1,5,25,50,100} ×
  λ ∈ {0.01,0.1,1}, γ = 0.005, γ_model = 0.1; 2.25M steps, seeds 0–9; `results/phase4_*_summary.csv`,
  `results/figures/figure1.png`). Mean ± std over 10 seeds:

  | | Last-100k acc. (stat.) | Last-100k acc. (drift) | Final cum. regret (stat. / drift) | Paper (stat.) |
  |---|---|---|---|---|
  | Random | 14.3% | 14.3% | 1.771M / 1.771M | — |
  | LinUCB | 71.9% ± 0.1 | 27.7% ± 10.6 | 0.475M / 1.157M | 72% |
  | NB1 | 82.0% ± 0.2 | 54.0% ± 21.5 | 0.311M / 0.689M | — |
  | **NB2** | **80.8% ± 0.6** | **69.3% ± 0.9** | **0.332M ± 0.005 / 0.497M ± 0.021** | 76% |
  | NB3 | 72.5% ± 1.4 | 64.4% ± 1.7 | 0.525M ± 0.013 / 0.696M ± 0.022 | 73% |

  ✅ **Instructor check (D4) passes:** NB2's regret is below LinUCB's in both panels: −30% stationary
  (0.332M vs. 0.475M) and −57% drift (0.497M vs. 1.157M). NB2 is below LinUCB in every seed (worst
  seed 0.340M / 0.532M). No NaN/inf in any run (D7).
  🔍 NB2 is above the paper's 76% (80.8%); NB3 is near the paper's 73% (72.5%), and its last-100k
  accuracy ties LinUCB's (72.5% vs. 71.9%), but its regret is 11% higher because it learns slower
  early on (regret per 500k: NB3 134k vs. LinUCB 106k in the first segment). This fits the paper's
  ordering NB2 > NB3 ≈ LinUCB.
- 📊 *(Nov 15)* **The committee vs. a single tuned NB1.** Stationary: NB2 is slightly worse than our
  tuned NB1 (80.8% vs. 82.0%; 0.332M vs. 0.311M regret), because EXP3 spends time on weak models
  before it finds the good ones. **Drift: NB2 is much better than NB1** (0.497M vs. 0.689M regret;
  69.3% ± 0.9 vs. 54.0% ± 21.5 at the end). NB1 gets worse with each drift and seeds diverge; NB2's
  regret per 500k segment stays flat after drifts (95k → 106k → 114k → 111k; NB1: 91k → 109k → 173k →
  211k). The committee is what makes the method robust to drift, and the seed spread drops 20×.
  ❓ Paper sentence: the paper never reports NB1, so "NB2 < tuned NB1 on stationary data" is our finding.
- 📊 *(Nov 15)* **Which models EXP3 picks (final probabilities, argmax per seed).** Model index = 3·(C
  position) + λ position.
  - NB2 stationary: C = 100, λ = 1 in 6 seeds and C = 50, λ = 1 in 4. This agrees with NB1's tuning
    (largest λ, large C).
  - NB2 drift: C = 50, λ = 1 in 7 seeds; C = 25, λ = 1 in 2; C = 100, λ = 1 in 1. Still always λ = 1.
  - NB3 stationary: the EXP3s for actions 0, 1 (1 = the 48.8% majority class) and 2 end on model 0
    (C = 1, λ = 0.01, the weakest model) in 10/10, 9/10 and 10/10 seeds; the EXP3 for action 5 also
    does in 9/10. Action 6 picks larger models (C = 25–100); actions 3–4 vary. **The toy lock-on (D3
    limitation) shows up on real data in every seed.** Likely reason (inference): under D3 only the
    played action's EXP3 is rewarded, and a weak model that scores high on common classes gets played
    and rewarded often, so its weight keeps growing.
  - NB3 drift: less lock-on on actions 1–2 (they move to C = 5 models), but actions 0 and 5 still end
    on model 0 in 9/10 and 10/10 seeds.
- 📊 *(Nov 15)* **Drift recovery (paper p. 7: "between 75.000 and 350.000 in the worst case").** Our
  rough measure: steps until the 10k-step moving average of accuracy (mean over seeds) gets back
  within 2 points of the 100k steps before the drift. NB2: ~249k, ~294k and ~166k after the 1st/2nd/3rd
  drift; NB3: ~348k, ~239k, ~169k. The 4th drift (at 2M) has only 250k steps left, and neither gets
  back in that time. Both are inside the paper's 75k–350k range. By eye, Figure 1's drift panel shows
  the same: NB2's slope goes back to its pre-drift slope within a few hundred thousand steps.
  ⚠️ The paper gives no criterion; this measure is ours.
- 📊 *(Nov 16)* **MIND results so far** — full tables and findings are in §4 under "MIND"; summary here:
  - **Sanity / tuning on MIND-small** (Phase 5b, 5c): LinUCB best α = 0.5 (43.7% clicks); NB1 best
    C = 1, λ = 1 (44.6%); popularity 36.0%; Random 19.9% = analytic.
  - **E3 on MIND-large, baselines** (Phase 5c, 10 seeds): NB1 46.0% > NB2 45.6% > LinUCB 44.7% > NB3
    43.6% > popularity 37.2% > Random 20.4%. Figures: `mind_e3_regret.png`, `mind_e3_daily.png`,
    `mind_daily_categories.png` (day starts ≈ 151k, 316k, 679k, 1.05M, 1.41M, 1.74M steps). Key findings: NB beats LinUCB on MIND too; NB2's EXP3 prefers mid-size
    networks (C = 25) on long runs (MIND not purely linear, modest gain); NB3 locks onto the weakest model
    (69%); the fixed popularity rule decays 41% → 31% across the week (natural drift).
  - **E4 (planted shifts):** Phase 5d, not yet designed. **New method:** Phase 6.

## 7. Failures, surprises & dead ends

- ❌ *(Nov 15)* **NB1 with the ×1/P update on the played action collapses to the majority class.**
  Last-100k accuracy after 300k steps (C = 25, tuning seeds 100/101): **48.4–48.6% for every λ in
  {0.01, 0.1, 1}** — exactly "always guess class 2". Final network outputs ≈ [0, 1, 0, 0, 0, 0, 0].
  Mechanism: with γ = 0.005 an explored action has P = γ/7 ≈ 0.0007, so its update is ×1,400. The
  first explored action with reward 0 gets its sigmoid output driven to ≈ 0; then o(1 − o) ≈ 0, its
  gradient vanishes and it never recovers. Six of seven networks die this way. Weights stay finite,
  so it's silent (no NaN). Found by the Phase 3 build agent; reproduced independently.
- 🔍 *(Nov 15)* **Re-reading the paper's notation (PDF p. 4):** Eq. 1 and the proof of Theorem 1 both
  write the indicator as **1[k̂_t = k]** (k̂ = the *greedy* action); Algorithm 1 writes **1[k_t = k]**
  with an unmarked k_t (it uses k̃_t for the played action elsewhere). Our Nov 12 "played action"
  reading came from the unbiasedness argument of Theorem 1, not from the symbols. Read literally,
  Eq. 1 updates network k only when k is the greedy action **and** was played (its reward is only
  known then), with weight 1/P(k̂) ≈ 1.004.
- 📊 *(Nov 15)* Same 300k-step diagnostic, three readings (C = 25; seeds 100/101):

  | Reading | λ = 0.01 | λ = 0.1 | λ = 1 |
  |---|---|---|---|
  | Played action × 1/P (our Nov 12 reading of Alg. 1) | 48.5% | 48.5% | 48.5% |
  | Eq. 1 as printed (greedy-and-played only, × 1/P(k̂)) | 69.0% | 72.3% | **75.8%** |
  | Played action, no weight (not in the paper) | 69.0% | 72.1% | 75.5% |

  The literal Eq. 1 reading works and reaches the paper's NB2 level (76%) in one config. It never
  learns from exploration steps (0.43% of steps), which is why it behaves like the unweighted variant.
  ⚖️ *(Nov 15, user)* **NB1 uses Eq. 1 as printed.** Reasoning for the paper: the paper's own settings
  (γ = 0.005, K = 7) force ×1,400 steps under the "played action" reading, which provably saturates
  sigmoid networks in any implementation (our numpy matched torch autograd to 1e-15 including ×1,400
  steps), yet the paper reports NB2 76% / NB3 73% — so the authors cannot have run that reading.
  Eq. 1 as printed is consistent with the text (Eq. 1, Theorem 1; Alg. 2 says "update each network
  corresponding to the action k̃_t for each model with (1)", and (1) contains the k̂ gate) and with
  the results. Unstated details on the authors' side (e.g. weight clipping) can't be ruled out; the
  paper mentions none. Supersedes the Nov 12 "played action" decision in §3.

## 8. Engineering notes (cost, speed, reproducibility)

- 📊 *(Nov 13)* `algorithms/base.py` contract: 12 tests pass (same seed → identical run; `reset()`
  replays exactly).
- 📊 *(Nov 13)* First `fetch_covtype` load ≈ 1.5 min (one-time conversion); cached afterwards.
- 📊 *(Nov 15)* Environment streams 2.25M steps in ~1.0 s (uint8 view) → ~4.2 s with the float32
  copy per step (~1.4 µs/step). Negligible next to neural-network steps; the stream is not a bottleneck.
- 📊 *(Nov 15)* Test suite: 18 tests pass in ~9.5 s.
- 📊 *(Nov 15)* First LinUCB timing (batched, Sherman–Morrison): ~0.2 ms/step → ~8 min per 2.25M-step
  run on one core.
- 📊 *(Nov 15)* Actual compute: α tuning (21 runs × 500k) took 4 min 50 s; each full config
  (20 runs × 2.25M) took ~10.5 min on 8 cores (4 performance + 4 efficiency). LinUCB alone runs at
  ~18.6k steps/s (54 µs/step) after a review speed-up; inside 8 parallel jobs the summary CSV shows
  ~7k steps/s per job — ⚠️ the CSV's `steps_per_sec` is measured under contention, so don't quote it
  as LinUCB's speed.
- 🔍 *(Nov 15, Phase 2 reviews)* Fixes: runs are saved as each finishes and reused on rerun only if
  algorithm, params, steps and drift all match (a crash no longer loses hours of NB runs); LinUCB skips
  the random tie-break when there is no tie (verified bit-identical results on 300k steps); LinUCB's
  confidence term is floored at 0 (guards against a NaN from rounding); the figure script now plots
  every `*_stationary`/`*_drift` result file, so Phase 3 NB runs join the same Figure 1.
- 📊 *(Nov 15)* NB step-cost benchmark (numpy, real data, one core): NB1 C=50 66 µs → **36 µs** and
  C=100 167 µs → **43 µs**; NB2-size committee (15 models) 634 µs → **243 µs**, by using only the 12
  active inputs of each row in the first layer (identical maths). ⇒ NB1 run ≈ 1.5 min, NB2/NB3 run
  ≈ 9–10 min alone. torch not benchmarked (not installed); estimated 0.3–1+ ms/step.
- 📊 *(Nov 15)* NB1 compute: tuning 8 min; each full config (10 runs × 2.25M) ~8–10 min on 8 cores.
  Torch cross-check of the hand-written backprop: max |diff| 1.8e-15 (W1), 8.0e-15 (w2) after 1,000
  updates incl. ×1,400-weighted ones. Review refactors verified bit-identical on saved runs.
- 📊 *(Nov 15, Phase 4 short check)* NB2 and NB3, 200k steps, stationary, seed 0. Both ran in
  parallel: 67 s and 74 s, about 330–370 µs/step. No NaN/inf.

  | Seed 0, first 200k steps | Regret at 200k | Accuracy, steps 150–200k |
  |---|---|---|
  | LinUCB | 42,645 | 72.2% |
  | NB1 | 42,704 | 74.9% |
  | NB2 | 42,764 | 74.1% |
  | NB3 | **59,410** | **66.2%** |

  - **NB2 is still climbing:** accuracy per 50k block was 67.7 → 71.8 → 72.9 → 74.1%.
  - **NB2's EXP3** favours the λ = 1 models: C = 25 at 0.43, C = 50 at 0.39, C = 100 at 0.10. This
    matches NB1 tuning, where λ = 1 was best in every row.
  - **NB3** is behind but still learning: 60.1 → 66.2%.
  - **NB3's per-action EXP3s:**
    - Action 1, the 48.8% majority class, put 0.91 on model 0 (C = 1, λ = 0.01), the smallest and
      slowest model. This looks like the toy-problem lock-on (D3 limitation) showing up on real data.
    - Actions 3–6 are rare classes. Their EXP3s are still near uniform, because they are rarely played.
  - One seed only: indicative, not a result.
- ⚠️ *(Nov 15, Phase 3 review → Phase 4)* `MLPStack.update` applies to network `action` of *every*
  model; Eq. 1's gate must be applied **per model** in NB2/NB3 (each model has its own k̂), otherwise
  NB2 reintroduces the collapse. Also `check_finite` only looks one level deep — make it see the
  committee's stacks.
- ⚠️ *(Nov 15)* For Phase 3 (only if torch is used): torch must be limited to 1 thread per parallel job
  (`torch.set_num_threads(1)`), otherwise 8 jobs oversubscribe the cores. Reviewer estimate for NB:
  0.3–1 ms/step → 10–40 min per 2.25M-step run.
- 🔍 *(Nov 15)* Review workflow (`/simplify` + `/code-review`) caught two latent bugs before any
  algorithm existed: uint8 contexts (would wrap around / break torch) and shared RNG stream between
  environment and policy (seeds would not be independent).
- 📊 *(Nov 15)* **Phase 4 compute:** 40 runs × 2.25M steps (NB2 + NB3, 10 seeds, 2 settings), 8 parallel
  jobs. Stationary 1 h 50 min, drift 2 h 0 min. Per run inside 8 parallel jobs: NB2 ≈ 46–54 min
  (~700–950 steps/s), NB3 ≈ 21–46 min. That's faster than the 2.5 h-per-setting estimate.
- 🔍 *(Nov 16, Phase 5b build)* **Masking without touching Forest Cover:** every `select` takes an optional
  `avail` mask; with `avail=None` the code path and the order of random draws are exactly the old ones
  (egreedy draws over `arange(K)`, Random's single `rng.integers`), so all saved Forest Cover runs reproduce
  step for step (verified: all algorithms, both panels, 2 seeds × 50k steps; a 5k-step version is a
  permanent test). NB1/2/3's Eq. 1 weight uses P(k̂) over the available arms (documented adaptation).
- ⚖️ *(Nov 16, Phase 5b)* **Deviation from the plan's wording (accepted):** the runner keeps Forest Cover's
  (context, correct class) items and branches on item length, instead of Forest Cover yielding one-hot
  reward vectors like MIND — avoids rewriting the verified Forest Cover stream and its tests for a
  cosmetic gain. Cost: 5 lines; a third environment would have to pick one of the two formats.
- 🔍 *(Nov 16, Phase 5b review)* `/simplify` (4 reviewers): processing script shortened (plain tables
  instead of bit masks, vectorised per-user history update) — output **byte-identical** on small and large.
  `/code-review` (6 findings): fixed (1) the runner now **refuses a MIND config without `d: 19`** — NB would
  otherwise run silently with the default 94 inputs; (2) preprocessing asserts every impression has ≥ 1
  available arm (masking would crash otherwise). Skipped, with reasons: hash-only dedup (collision
  ≈ 4e-7), loading unused arrays (≈ 38 MB/job), the runner's two formats (above).
- ❌ *(Nov 16, Phase 5a)* **Memory lesson:** the first MIND facts script kept every impression's article
  list in Python lists → 22 GB on a 16 GB Mac, stalled in swap; rewritten line-by-line with a few numbers
  per impression → 90 s, 2.3 GB. Same compact design used for `preprocess_mind.py` (large: 2 min).
- 📊 *(Nov 16)* **Measured per-step cost on MIND** (one job alone, 20k steps): LinUCB 29 µs, NB1 C = 1
  32 µs, NB1 C = 100 67 µs, NB2 408 µs, NB3 492 µs (NB2 on Forest Cover: 325 µs). Machine: Apple M2,
  4 performance + 4 efficiency cores, 16 GB; with 8 parallel jobs each runs ≈ 4× slower than alone.
  **Phase 5c compute:** NB1 tuning 45 runs in 4 min; E3 60 runs on MIND-large in **2 h 29 min**.
  (Two earlier estimates were wrong and corrected by measuring: "40–50 min per NB2 run" ignored the arm
  count; "7–8 h for 5c" wrongly assumed NB1 costs as much as NB2.)
- 🔍 *(Nov 16)* Practicalities: closing the laptop lid pauses runs (sleep), it doesn't kill them;
  `run_experiment.py` saves each run as it finishes and reuses it if the settings match, so an interrupted
  batch only redoes unfinished runs.
- 📊 **Test counts:** Phase 1: 18; Phase 4: 57; Phase 5b: 73 (incl. the permanent 5k-step Forest Cover
  exact-match test).
- 🔍 *(Phase 3–4)* Overflow-safe sigmoid (`scipy.special.expit`), so even ×1,400 updates can't overflow;
  `check_finite` extended in Phase 4 to look inside the committee's stacks and EXP3s (fixes the Phase 3
  review warning above). Committee layout: NB2/NB3's 15 models live in 5 `MLPStack`s (one per C, 3 λ each);
  one gate applies to all models, so `MLPStack.update` is unchanged.
- ⚖️ *(Nov 15, Phase 2 D8; Phase 5 Q1–Q2)* **Git policy:** commit summary CSVs and figures; per-step rewards
  go to gitignored `results/raw/`; no MIND data (raw or processed) in git (MSR license).
- ⚖️ *(Nov 15, Phase 3 D9)* PyTorch stays out of requirements.txt: the one-time gradient cross-check ran in a
  temporary `uv run --with torch` environment.
- 🔍 **Environment:** Python 3.11.4 in `.venv` (uv), pinned requirements (scikit-learn 1.9.1, numpy 2.4.6,
  pandas 3.0.5 …), pytest with pythonpath=.; folder layout follows course slide 71.
- 📝 **Workflow (methodology slide):** each phase = plan to zero open questions (asked one by one, every
  setting sourced) → build (Sonnet for routine code; Opus for NB1/2/3, EXP3, gating) → verify → `/simplify`
  → `/code-review` → report → user commits. Phase 4 added a stop after the build for the user to read the
  code, then a 200k-step short check before the full runs. Data-dependent plans get a data-review stop
  (Phase 5).

## 9. References & verification (paper references section)

- 📄 *(Nov 12, user decision)* **Base paper authors:** the provided PDF (arXiv:1409.8191v1) prints "Robin
  Allesiardo, Raphaël Féraud, Djallel Bouneffouf"; current arXiv/Springer/Crossref metadata say "Elys
  Allesiardo" (and drop the accent in Féraud). We cite as printed in the PDF; discrepancy recorded.
- 📄 **Published version:** *Neural Information Processing* (ICONIP 2014), LNCS, Springer, pp. 374–381,
  DOI 10.1007/978-3-319-12637-1_47 — LNCS volume and editors `[UNVERIFIED]` (Springer page needs login).
- 📄 **CTS:** the base paper cites the 2012 CoRR version (its ref. [19]); we cite ICML 2013 (PMLR 28(3),
  pp. 127–135).
- ⏳ **ADWIN (R8)** has only one independent source (Crossref; SIAM and DBLP blocked automated access on
  Nov 12) → not in references.bib yet. Page–Hinkley (R9) verified.
- 📝 **Corrections the audit triggered (Part 3):** README had NeuralUCB's bound as Õ(√T) (→ Õ(d̃√T)); "MIND-
  small ~230k impressions" unverified (now counted: 230,117); base paper cited as ICONIP without the arXiv
  v1 we used; README reference list was written from memory.
- 📝 **Citation rule:** never cite from memory; bibliographic fields need ≥ 2 primary sources; references.bib
  holds only ✅ entries; anything unverified is marked `[UNVERIFIED]`; discrepancies go to the user.
- 📄 *(Nov 15)* **MIND license** (MSR License Terms): allowed — teaching, academic research, public
  demonstrations, personal experimentation, publishing results without "material portion of the
  Dataset"; forbidden — distributing it, altering copyright notices, implying Microsoft endorsement,
  malicious use; accepted by use; §7 US binding arbitration; §11–12 "as is", liability capped at $5; no
  citation requirement (the site asks for one; we cite R7). **User agreed Nov 15.**
- 📄 **LinUCB versions:** the §3 LinUCB notes come from the arXiv **v2** PDF (user copy, read Nov 15); C23/C24
  (arm set A_t, constant feature) were read in **v1** (Nov 16). Same content for these passages; cite one
  version in the paper (decide when writing references).
