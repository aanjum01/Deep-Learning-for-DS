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
- ⚖️ *(Nov 17, Phase 5d R2)* **What decides "adapts better" (fixed before Phase 6):** per shift, **clicks
  lost** (extra regret vs the no-shift run over the 500k steps after the shift) decides; adaptation time
  (R10 stopwatch, 95%, on 10-seed-averaged curves, "> X" if not recovered) and the click rate are reported
  alongside. Reason: on Forest Cover the 95% bar is often never reached (LinUCB, NB1 "> 500k" after every
  drift), so the stopwatch can't rank methods.
  **Refined (S2, Nov 17):** "adapts better" needs fewer clicks lost **and** at least as many total E4 clicks in
  the same window — otherwise "trades normal performance for robustness" (Random loses ~0 clicks by never learning).
  **Margin (S3, Nov 17):** a difference counts only if the 10-seed means differ by > 2 × combined standard
  error; otherwise "no clear difference"; Random shown as the noise floor.
- 🔍 *(Nov 17, Phase 5d planning, verified on saved runs)* **Forest Cover recovery under the new, sourced
  measure** (R10 stopwatch on 10-seed-averaged curves vs the stationary run): LinUCB and NB1 never get back to
  95% before the next drift; NB2 recovers after drift 1 only (220k steps); NB3 after drifts 1–3 (339k, 368k,
  442k). This is **stricter** than Phase 4's ad-hoc "within 2 points of the pre-drift level" (§6), because the
  stationary reference keeps improving — the paper must explain both and use the new one. Per-seed stopwatches
  were rejected: they make Random "dip" after every drift on every seed (noise); averaging over seeds removes it.
- ⚖️ *(Nov 17, Phase 5d S1)* **E4 shift times = 300k, 1M, 1.5M**, from a data-only rule fixed before any E4
  run: no shift within ±50k steps of a natural jump (a category's click rate when shown changes by ≥ 0.15
  between the 50k steps before and after). The originally planned 500k sat on the week's biggest natural
  event — **tv click rate 0.08 → 0.52 → 0.14 over ≈ 460k–540k**, in every seed — where stale LinUCB (E3 drops
  to 0.335) got a relabelling windfall while NB1/2/3 (rising to ~0.52 in E3) didn't; that biases clicks lost
  against methods that track real drift. Other natural jumps ≥ 0.15: ≈ 160–200k (music), 360–380k, 640k,
  780–820k, 1.24–1.26M, 1.94–1.96M. Uneven spacing (paper: every 500k), stated with this reason.
- 🔍 *(Nov 17, Phase 5d S8)* On MIND a planted shift can make an arm almost unplayable (it inherits the
  availability of kids / northamerica, ≈ 0% shown), forcing exploration — Forest Cover has no such effect (all
  arms always available). E4 and Forest Cover adaptation times are therefore **not compared like-for-like**.
- ⚖️ *(Nov 17, Phase 5d R6)* **Rules for Phase 6 from the E4 design:** (1) every method/ablation needs a
  paired E3 + E4 run per seed (adaptation is measured against the same seed's no-shift run) — budget runs and
  LLM API cost; (2) cache/replay LLM answers so runs are repeatable; (3) shift times are never visible to any
  method (settings, prompts, code) — it must detect the change; (4) the LLM sees arm numbers / neutral labels,
  not category names (relabelling is only a pure renaming for arm-agnostic learners), or this is a stated
  limitation.
- ⚖️ *(Nov 17, after Phase 5d results, user agreed)* **What Phase 6 must keep in mind:** (1) the bar is NB1/NB2,
  not LinUCB — "adapts better" needs fewer than ≈ 5.5–6.5k clicks lost at shifts 2–3 **and** at least as many
  total clicks; (2) little room at shift 1 (NB1/NB2 already recover in 14–18k steps) — the chance to show a
  gain is at the later shifts (47k–116k); (3) gains may be small in absolute terms (6k ≈ under 3% of the
  ≈ 220k clicks in a window), so a marginal gain may not clear the 2 × SE margin with 10 seeds; (4) the
  "slower each shift" trend mixes lost plasticity with later, weaker days — if the sliding window flattens
  it, that is evidence for plasticity; (5) only 3 shifts — a stated limitation.
- ⚖️ *(Nov 17, user)* **What the ablations answer:** which component made a meaningful difference. The core 4
  (Nov 13) are leave-one-out from the full model: −LLM controller (= NeuralUCB + window), LLM → rule-based
  controller, −UCB (ε-greedy network + window + LLM), −sliding window (= NeuralUCB + LLM). "NeuralUCB alone"
  (no window, no controller) is **not** among the core 4; whether to add it is a Phase 6 planning question
  (it would separate "NeuralUCB vs NB" from "drift handling"). Each needs paired E3 + E4 runs (rule above).
- ❓ *(Nov 17, user ideas for Phase 6 planning — not decided)* (1) NeuralUCB's guarantees appear to be for
  stationary data [UNVERIFIED — check the NeuralUCB paper, log in REFERENCES_VERIFICATION before any claim];
  the window/LLM may be what makes it work under drift. (2) Since NB1 is the best baseline, also try the
  sliding window and/or LLM controller **on NB1** — this would show whether drift handling helps any learner
  or only NeuralUCB. Out of the Nov 13 scope until the user decides; it adds paired runs. (3) **Make Phase 6
  modular:** Phase 5 grew into 5, 5b, 5c, 5d, each needing a full plan; plan Phase 6 as small sub-phases
  (e.g. NeuralUCB alone → + sliding window → rule-based controller → LLM controller → ablation runs), each
  with its own plan, checks and commit, so a deadline slip cuts the last pieces rather than everything.
- ⚖️ *(Nov 17, user)* **Phase 6 decisions and roadmap** (resolves the open items above and the Nov 13 "ablations
  on E2?" question):
  - **Ablations run on MIND E3 + E4 only**; E2 only at the end if time allows. Why: an ablation asks "which
    component matters" — E4 (precise clock) + E3 (real drift) answer it; 4 ablations × E1/E2 × 10 seeds = 80
    more 2.25M-step runs + LLM calls for little new information. The **full model** still runs on E1–E4
    (every baseline did; the proposal compares with the paper's 75k–350k on Forest Cover).
  - **NeuralUCB alone becomes a reported method** (E1–E4): it is built anyway (6a) and separates "NeuralUCB is
    a better learner than NB" from "drift handling helps".
  - **Window / LLM on NB1: deferred**, last optional sub-phase, dropped if short on time (outside the Nov 13
    scope; NB1 learns online with no stored data, so a window changes its algorithm — new design work).
  - **Sub-phases** (each: plan → build → verify → review → report → commit). Build order adds pieces; the
    reported ablations remove one piece from the full model.
    - 6a NeuralUCB alone: build, tune on MIND-small without shifts, full runs E1–E4.
    - 6b + sliding window: build + short checks + tune the window (full E3/E4 runs of this version happen
      in 6e as the "−LLM controller" ablation).
    - 6c rule-based controller: build + short checks + tune (full runs in 6e as "LLM → rule-based").
    - 6d LLM controller → full model: build, then full runs E1–E4.
    - 6e the 4 ablations on MIND E3 + E4.
    - 6f statistics, figures, paper tables.
    - Optional if time: ablations on E2; window/LLM on NB1.
  - ❓ For 6b planning: the window size must be tuned without planted shifts (Q10 fairness rule), but on
    shift-free data the best window may simply be "as large as possible" — decide how to tune it then.
    User idea (Nov 18): test several window sizes. Fair if the size used for the headline comparison is
    chosen without shifted data and the other sizes are shown as a sensitivity sweep (all reported, none
    picked by its E4 result). Also: the user has ideas for the LLM controller — raise them in 6d planning. baselines on E4 keep their 5c
  settings, tuned on MIND-small **without** planted shifts; the new method must be tuned the same way —
  never on shifted data — otherwise it would get a drift-specific advantage the baselines didn't.
- 📄 *(Nov 18, Phase 6a planning; REFERENCES_VERIFICATION C32–C40)* **NeuralUCB paper read in full (arXiv v3,
  checked against the PMLR version) — answers the Nov 17 "[UNVERIFIED]" idea:**
  - ✅ **Its guarantee is for a stationary world:** rewards come from **one fixed unknown function h** for every
    round plus zero-mean noise (Eq. 2.2, p. 2); contexts may be any sequence, but the context → reward map never
    changes. The paper never mentions drift. So E4's relabelling (and E3's natural drift) break the assumption
    behind Õ(d̃√T) — the theory gives NeuralUCB **no** reason to adapt; any adaptation must come from training.
    Paper sentence: "NeuralUCB's bound assumes a fixed reward function (Zhou et al., Eq. 2.2), which concept
    drift violates; this motivates adding a sliding window and a controller."
  - 🔍 It also trains on **all past data, restarting from θ₀ each time** (Alg. 1–2, p. 3) — with long runs this
    remembers the old world, which is exactly what 6b's window will cut.
  - 🔍 The bound also needs a very wide network, a specific step size and unit-norm duplicated inputs
    (Assumption 4.2, Thm 4.5); the authors' own experiments relax all of these (m = 100, diagonal Z, constant γ,
    tuned η) — so in practice NeuralUCB is a heuristic with a theoretically-motivated bonus, as ours will be.
  - 🔍 The paper names NeuralBandit [allesiardo2014] directly (p. 3): "UCB … more effective than ϵ-greedy", one
    network instead of K — but never runs it as a baseline. **Our E1–E4 runs are a direct test of that claim.**
  - 🔍 Their real-data runs are **T = 15,000** steps (§7.2) with J = 1000 SGD steps of batch 500 every 100 rounds
    — 5,000 gradient evaluations per new observation; our runs are 2.0–2.25M steps (×150).
  - 🔍 The authors' code (uclaml = Prof. Gu's lab; no license → reference only) differs from the paper in init,
    regulariser, batch size, bonus formula, and keeps only every 100th observation after round 2000 (C40).
- 📊 *(Nov 18, Phase 6a planning)* **Measured NeuralUCB cost** (throwaway numpy prototype, paper form: disjoint
  arms, one hidden ReLU layer, m = 100, diagonal Z; real contexts; M2): one job alone — selection + Z update
  85 µs/step (MIND-small, K = 16, d = 19) / 71 µs (Forest Cover, K = 7, d = 94); one SGD step 0.12 ms (batch 1),
  0.39 ms (batch 50), 0.8–0.9 ms (batch 500). **8 jobs in parallel: ×4–5 slower** (selection ≈ 420 / 320 µs,
  batch-500 step ≈ 4 ms) — same factor as Phases 4–5. ⇒ the paper's schedule (J = 1000, batch 500, every 100
  rounds) would take **≈ 23 h per MIND-large run** in parallel — infeasible; the authors' repo schedule ≈ 2–5 h
  per run. Training budget is therefore a planning decision (phase-6a plan).
- 🔍 *(Nov 18, Phase 6a review, fresh-context Opus; checked analytically)* **The paper's step-size grid does not fit
  its own network form:** with f = √m·w₂ᵀReLU(W₁x) (Eq. 2.3) the gradient at init has ‖g‖² ≈ m‖x‖² (≈ 1,700 on
  Forest Cover, ≈ 275 on MIND), so a plain gradient step is stable only for η ≲ 1e-3 (Forest) / 7e-3 (MIND),
  while the paper's grid is η ∈ {2, 5}·10⁻ⁱ, i = 1…4 (up to 0.5). Scratch runs: on Forest 6 of 8 η values → NaN
  within a few rounds. The grid likely belongs to an implementation like the authors' code (no √m, default init,
  normalised inputs). ⚖️ **Decision (user, Phase 6a R1):** step = η/m, following Thm 4.5's η ∝ 1/m (p. 4);
  divergent settings fail fast and count as failed (R2). Paper sentence: "the paper's η grid is applied as η/m,
  the 1/m scaling of its Theorem 4.5, because with the √m output scaling of Eq. 2.3 the raw grid diverges."
- 🔍 *(Nov 18, Phase 6a review)* **Paper ambiguity — training before round 2000:** §7.2 trains "every 100 rounds
  starting from round 2000" (literally: none before), Alg. 1 trains every round; the authors' code trains every
  round before 2000. ⚖️ We train every round before 2000 (user, R6) — 0.1% of a run.
- 🔍 *(Nov 18, Phase 6a review, maths checked)* The paper's mirrored init on a doubled input [x, x]/√2 equals a
  width-m net on raw x with mirrored hidden pairs (first layer [W; W]/√2, output (w, −w)): both input copies always
  get identical updates. Same f, training path and penalty; the diagonal-Z bonus matches with g²/(2m) for the first
  layer. Built this way (user, R5) — no doubled input.
- ⚖️ *(Nov 18, professor via Slack + user)* **The base learner changes from NeuralUCB to NB1.** The user asked the
  professor about compute time and NeuralUCB's MIND result. The question gave him: tuned NeuralUCB 41.8% clicks on MIND-small vs
  LinUCB 43.7% and NB1 44.6%; about 5 h per full set of NeuralUCB runs; about 30 h of compute left before Dec 1. His
  reply: "switching to NB1 is reasonable given the computational constraints", and "make sure to include appropriate
  baselines and ablation studies to evaluate the contribution of each component."
  Why NB1 and not NB2/NB3: it is the best of the three on MIND E3 (46.0% vs NB2 45.6%, NB3 43.6%), and as one
  network instead of a 15-model committee it is ≈ 15× faster per run (≈ 3 min vs ≈ 50 min on E3, Phase 5c summary).
  - **Why we considered switching, and why we switched (for the paper):**
    1. *Compute.* The paper's training schedule (J = 1000 SGD steps of batch 500 every 100 rounds) would take ≈ 23 h
       per MIND-large run (measured, entry above). On top of that, the paper's runs are 15k steps and ours are 2–2.25M. We cut
       the schedule to J = 50 (J check) and it still cost ≈ 5 h per set of 40 full runs. NeuralUCB was the base of every later phase
       (window, controllers, ablations): ≈ 30 h of compute left on one laptop before Dec 1, every batch run overnight,
       no room for reruns.
    2. *Results.* Even tuned, it scored below the simpler methods on MIND-small (41.8% vs LinUCB 43.7%, NB1 44.6%).
       Likely causes: the cut-down training means it is undertrained; and it retrains on all past data equally, only every
       100 rounds, so it is slow to follow MIND's natural drift. LinUCB also assumes a fixed world but updates exactly
       after every click. Raising the training budget to close the gap would slow every later phase by the same factor.
    3. *Theory.* Its regret guarantee assumes a fixed reward function (Eq. 2.2), so the theoretical reason for picking it
       does not cover drift, which is the project's main question.
    4. *The alternative.* NB1 is already the best baseline on MIND and ≈ 15× faster than NB2/NB3. Having the controller set
       its exploration rate keeps the abstract's aim of replacing fixed exploration.
    The instructor agreed: "switching to NB1 is reasonable given the computational constraints."
    Suggested paper sentence: "We first implemented NeuralUCB as the base learner, but under our compute budget it had
    to be trained far less than in its original paper, and it underperformed simpler baselines on MIND-small (41.8% vs
    44.6% clicks for NB1). We therefore build on NB1 and make its exploration rate adaptive instead."
  - **New full model:** NB1 + sliding window + LLM controller. The controller also sets NB1's exploration rate γ
    (fixed at 0.005 in the paper), so exploration adapts to drift instead of staying fixed.
  - **Deviations from the approved abstract (Oct 5), to state in the paper:**
    1. NeuralUCB is no longer the base learner.
    2. "Uncertainty-aware exploration" becomes **drift-adaptive exploration**: the controller changes one global γ.
       This is not a per-context uncertainty bonus.
    3. The near-√T regret guarantee is no longer claimed. It covered stationary data only anyway (entry above).
    4. Ablation "−UCB" becomes **"−adaptive exploration"**: the controller is not allowed to change γ.
  - Unchanged: the sliding window, the LLM controller with its agents, the rule-based counterpart, MIND, E1–E4,
    the metrics, the main goal (faster adaptation after drift), and the baselines (NB1/2/3, LinUCB; Random and
    Popularity as sanity lines).
  - **Ablations now (leave one out of the full model):** −LLM controller (NB1 + window); LLM → rule-based
    controller; −adaptive exploration; −sliding window.
  - ⚖️ *(user, Nov 18)* NeuralUCB's Forest tuning was **stopped** at 67 of 128 runs, which stay saved in results/raw.
    Forest Cover adds nothing to the main claim (adaptation on MIND). In the paper NeuralUCB is the method we tried
    and replaced, reported with the MIND tuning numbers. **Phase 6a is closed (user, Nov 18):** no NeuralUCB full runs
    on any dataset. The code, tests and tuning results stay in the repo as the record.
  - ❓ For 6b planning: NB1 learns online and stores no past data, so "a sliding window on NB1" needs new design
    work (noted Nov 17). Also: −LLM = the 6b runs and LLM → rules = the 6c runs, so only −adaptive exploration
    and −window may need new runs (to confirm in 6e planning).
- ⚖️ *(Nov 18, Phase 6b plan approved; `paper/plans/phase-6b.md`)* **NB1 + sliding window.**
  - **Mechanism (Q1, Q2, Q3):** reset + retrain (abrupt forgetting). NB1 stores its logged Eq. 1 updates. Every R = W/5 steps,
    once t > W, a fresh network learns them once, oldest first. Gradual forgetting and reset-only-on-drift were
    rejected for 6b. Sources: Gama et al. 2014 §3.1.1–3.1.2, §3.3.1–3.3.2 (fixed sliding window = standard baseline;
    blind vs informed adaptation).
  - **W tuning (Q4, R2):** grid 10k/20k/50k/100k on MIND-small, seeds 100–102, no shifts. Pick the smallest windowed W within
    2 × combined SE of the best (bigger windows stay identical to NB1 longer, so they are favoured mechanically).
    Converted to MIND-large by time span × 9.868 (total rows 2,270,815 / 230,117).
  - **E3 + E4 runs (Q6, R1):** 7 sizes, 10k/20k/50k (count) + 99k/197k/493k/987k (time span), 10 seeds, 140 runs.
    Why both scales: every time-span size is longer than NB1's own recovery (14k–114k), and W ≥ 250k doesn't reset
    before shift 1. Pre-registered: "the window hurts at shifts" is a possible result, read as a property of blind
    windows (a rebuild from old-world rows) and the motivation for 6c/6d's triggered reset.
  - MIND only (Q5); checkpoint before the full runs (Q7); gradual forgetting gets its own plan later (Q3b).
  - 🔍 Literature, all papers read in full: SW-UCB (Garivier & Moulines 2011) and SW-UCB for linear bandits (Cheung et al.
    2019) set the window from the horizon and the amount of change, which we can't know, so W is tuned. Cheung's
    Bandit-over-Bandit (EXP3 picks the window per block, with a fresh learner each block) is the published precedent for
    a higher-level learner choosing W (6c/6d). Dohare et al. 2024: small networks also lose plasticity, but a full
    reinitialisation is only their (weaker) baseline. This motivates testing a reset; it doesn't show one helps NB1.
- ⚖️ *(Nov 18, Phase 6b part 2 plan approved; `paper/plans/phase-6b2.md` rev. 3; this entry also pre-registers the decision rule, before any E3/E4 fade run)* **NB1 + gradual forgetting.**
  - **Mechanism:** after NB1's update, at every step, every category's network (both layers) does θ ← θ − ρ(θ − θ₀) toward its
    own initial draw. This is "L2 Init" [kumar2024 §3, App. A.3] up to an O(αρ) term. Rejected: toward zero, toward fresh
    weights (both worse in kumar2024 §5.3), and fading only the arm that learned (scored lower on MIND-small: 0.4335 vs 0.4425 at 1/50k).
  - **Tuning:** ρ ∈ {1/10k, 1/20k, 1/50k, 1/100k} on MIND-small, seeds 100–102, no shifts; headline = best mean (the paper's
    rule; expected to pick the slowest fade, stated).
  - **E3 + E4 runs:** 8 rates (the 4 as tuned, plus ÷ 9.868 = 1/99k, 1/197k, 1/493k, 1/987k), 10 seeds, 160 runs.
  - **Decision rule (pre-registered):** each setting is a point (E3 click rate, mean clicks lost per shift). A fade beats a window
    if it is not worse on either and better on one (2 × unpaired SE, ddof 1). **6c/6d's controller sets a fade rate if some fade beats ≥ 2 of
    the windows 10k / 20k / 50k**; otherwise the window stays. The larger windows' clicks-lost values are distorted by the trap
    in their no-shift twins (−2.5k to −12k per shift), so they are reported but don't decide. Placebo: plain NB1 must beat < 2 small
    windows (on the 6b numbers it beats 0). Traps and post-shift clicks are reported, not deciding. The rule was set after seeing 6b's window
    results and the MIND-small fade prototype, never any E4 fade result.
  - How the rule got here: a first pair rule was too strict (a tie on E3 clicks failed). The revision was too lenient (plain NB1 would have won
    most pairs). The trade-off rule, restricted to trustworthy windows, came out of two fresh-context reviews and the user's decisions.
- ⚖️ *(Nov 18, user, after the 6b part 2 results)* **Gradual forgetting becomes NeuralBandit++'s forgetting method; the
  sliding window is the method tried and rejected.**
  - **Judged against the paper's goal** (adapt faster after drift than NB1 without losing clicks; abstract, README §1):
    - The slow fade (1/987k) adapts better than NB1 at all 3 shifts (recovery 10k / 15k / 42k vs 11k / 45k / 113k) and keeps
      NB1's clicks on the real week (45.96% vs 45.97%). It gets **more** total clicks on the drift data (E4 45.47% vs 45.18%) and
      never traps.
    - The windows adapt fast but cost 1–3 points of clicks, hit the dead-arm trap, and depend on rebuild timing (493k check).
  - **New full model:** NB1 + gradual forgetting (fade toward θ₀) + a controller that sets the **fade rate** and the
    exploration rate γ (6c rule-based, 6d LLM).
  - **The pre-registered window-vs-fade rule** (which favoured the smallest windows; they give up 2–3 points of clicks) is reported
    as one honest line in the methods, not a headline:
    "a pre-registered pairwise rule favoured the smallest windows, which give up 2–3 points of clicks; we chose fading because it
    meets the paper's goal". The rule was a planning tool for the knob choice; the data answering the primary question decide.
  - **Another deviation from the abstract** (after the NB1 switch): "sliding window memory" → **gradual forgetting**. The idea
    (forget old data so the model can follow drift) is unchanged.
  - **Ablations** (each removes one piece of the final model):
    - **−controller** = NB1 + fixed fade (already run: 6b part 2);
    - **LLM → rule-based** (6c);
    - **−adaptive exploration** (the controller sets the fade only);
    - **−forgetting** (the controller sets γ only, no fade; replaces "−sliding window").
    The window is **not** re-tested in the ablations, because it is not part of the final model. It appears as a design comparison
    (6b vs 6b part 2: 7 window sizes vs 8 fade rates, E3 + E4, 10 seeds each).
  - The user has questions about the LLM and rule-based controllers to raise in a separate 6c planning session.
- ⚖️ *(Nov 18, Phase 6c design discussion + plan approved; `paper/plans/phase-6c.md` rev. 2)* **The controller design.**
  - 🔍 **Two findings that shaped it** (MIND-small, seed 100, checked Nov 18):
    - **γ is close to a dead knob in NB1 as published.** Random exploration plays are 0.41% of steps, and under Eq. 1 as printed
      they never train a network. Raising γ after a drift costs clicks and teaches NB1 nothing.
    - **Forgetting is NB1's real re-try mechanism.** Arms start at an estimate of ≈ 0.49 (0.42–0.58). Trained arms sit far lower
      (mean of shown arms 0.23 with plain NB1). Fading lifts them toward the start (0.35 at fade 1/20k), so arms nothing has
      reinforced lately outrank the favourite and get re-tried. Same mechanism as the dead-arm trap (a stuck arm at ≈ 0.5).
  - **Design:**
    - **Three layers:** NB1 always fades slowly; Page-Hinkley (PH) watches NB1's **surprise** (reward − its estimate for the played
      arm) for a drop; on an alarm a decision-maker boosts the fade rate for a while, then returns to the slow rate.
    - **Event-triggered:** this replaces README's "every N rounds", and the LLM is called only on alarms (cost).
    - **PH, not ADWIN:** PH detects drops only. ADWIN's natural output is a window, and windows mean rebuilds and the dead-arm trap.
      Our design is the gradual-forgetting counterpart of ADWIN's adaptive idea, with no rebuild (paper line).
  - **Versions:**
    - **A** = rules with sparse PH (boosts ≤ 10% of a calm MIND-small run);
    - **B** = rules with sensitive PH (≤ 30%);
    - **C** = sensitive PH + LLM (6d);
    - **D** = Jev (optional);
    - **−detector** = LLM on a schedule with matched calls, **required**;
    - **−adaptive exploration**;
    - **−forgetting**.
  - **Rules:** boost 1/20k for 20k steps on both datasets; alarms ignored during a boost; PH reset after it.
  - **LLM's menu:** {1/10k, 1/20k, 1/50k}, each for one memory length. Disclosure: fixed after the fixed-fade E4 results were known.
  - **PH settings:** river defaults (δ = 0.005, min_instances 30) except mode "down", alpha = 1 (Gama et al.'s formula has no
    fading), and λ from the boosted-share budget. Gama et al. (§3.2.1, App. A Alg. 1) give no value for δ; river's docs give no
    source for their defaults.
  - **The LLM never changes PH's settings** (C and B see the same alarms). LLM-tuned detection is future work.
  - **Judging (no new pass/fail gate; lesson from 6b part 2):** every version is reported on E3 clicks, clicks lost per shift and
    adaptation time vs NB1 and the slow fade. Claims written in advance:
    - "the controller helps" if a version loses clearly fewer clicks per shift than the slow fade (2 × SE) with no clear click drop;
    - "the LLM helps" if C does so better than both rules versions;
    - otherwise "matched".
  - **Dropped:** headlines/topics as LLM input (clashes with R6: no category names). Jev (TypeSafe, Sept 2026) reviewed: cheap and
    typed, but its vendor says it is weak at arithmetic, it is not deterministic, and it has no paper. So it is an optional extra
    controller; a text LLM is the main one (6d).
  - **Why per-step budgets:** a MIND-small day is ≈ 30k steps, so per-day budgets with 20k-step boosts would have made A and B
    near-constant fast fades (found by the fresh-context review).
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
  - **E4 (planted shifts):** Phase 5d, results below. **New method:** Phase 6.
- 📊 *(Nov 17, Phase 5d)* **E4 on MIND-large, baselines** (10 seeds, shifts at 300k / 1M / 1.5M, each a cyclic
  relabelling of the 16 category arms; measured against the same seed's E3 run; plan `phase-5d.md`).
  Outputs: `results/phase5d_mind_e4_{recovery,verdicts,table}.csv`, `figures/mind_e4_{recovery,regret}.png`.
  - **Done-checks all pass:** E4 = E3 bit-for-bit for the first 300,000 steps in all 60 (algorithm × seed)
    pairs; all 60 runs complete and finite; Random never dips (worst 2%, adaptation time 0 at every shift);
    every learner dips below 95% after shift 1 (worst loss LinUCB 27%, NB3 22%, NB2 12%, NB1 10%).
  - **Overall E4 click rate:** NB1 45.2% > NB2 44.8% > LinUCB 42.1% ≈ NB3 42.0% > Popularity 23.9% > Random
    20.4% (E3: 46.0 / 45.6 / 44.7 / 43.6 / 37.2 / 20.4). The shifts cost NB1/NB2 under 1 point, LinUCB 2.6,
    NB3 1.6; same ranking as E3 except LinUCB falls to NB3's level.
  - **Clicks lost per shift** (500k-step window, mean ± std over seeds — the pre-registered deciding number):

    | | Shift 1 (300k) | Shift 2 (1M) | Shift 3 (1.5M) |
    |---|---|---|---|
    | NB1 | 3,443 ± 659 | **5,544 ± 221** | **6,180 ± 253** |
    | NB2 | 3,129 ± 655 | 6,017 ± 770 | 6,515 ± 686 |
    | NB3 | 3,186 ± 2,051 | 14,238 ± 2,468 | 13,894 ± 2,521 |
    | LinUCB | −2,546 ± 1,011 ⚠️ | 24,288 ± 795 | 29,487 ± 713 |
    | Popularity | 79,987 | 86,950 | 67,499 |
    | Random | −109 | −99 | −53 |

    ⚠️ LinUCB's negative number at shift 1 is the 500k "tv" spike (see §7), not robustness.
  - **Adaptation time** (R10 stopwatch on the 10-seed mean curves; steps from the shift until back ≥ 95% of
    E3 for 10k straight steps): NB1 14k → 47k → 114k; NB2 18k → 51k → 116k; NB3 23k → 178k → 127k;
    LinUCB 20k → 219k → 236k. **Every learner recovers from every shift** (unlike Forest Cover E2).
  - **Verdicts (pre-registered rule):** shifts 2 and 3 — NB1 and NB2 adapt better than LinUCB and NB3, NB3
    better than LinUCB, NB1 vs NB2 no clear difference. Shift 1 — NB1/NB2/NB3 mutually no clear difference;
    vs LinUCB the rule says "LinUCB trades performance for robustness" (vs NB1, NB2) and "LinUCB better" (vs
    NB3), but this is driven by the 500k spike (§7) — report with that caveat.
  - **Main findings for the paper:** (1) the NB committee/MLPs lose 4–5× fewer clicks than LinUCB after a
    shift once the run is long (shifts 2–3); (2) **adaptation gets slower with every shift for every
    learner** (NB1 14k → 114k) — consistent with learners that lose plasticity as they accumulate data
    (LinUCB's A matrix only grows; the MLPs never forget). ⚠️ Inference: later shifts also fall on
    different days with lower click rates, so this design can't separate plasticity from timing. This is the
    gap the proposal's sliding window targets — Phase 6 should test whether it flattens this trend;
    (3) NB3 is the least stable NB variant (seed std 3–4× the others), matching its lock-on (§6, Phase 4).
  - **vs the paper's "between 75.000 and 350.000 in the worst case" (p. 7, Forest Cover):** on MIND the NB
    methods recover in 14k–178k steps — faster, but not like-for-like (S8: a shift on MIND can make an arm
    nearly unplayable; Forest Cover's arms are always available).
- 📊 *(Nov 17, Phase 5d R9)* **Forest Cover E2 under the new measure** (`results/forest_e2_{recovery,
  verdicts}.csv`, `figures/forest_e2_recovery.png`; same stopwatch, vs the stationary run): only NB2 (drift 1,
  221k steps) and NB3 (drifts 1–3: 340k, 369k, 443k) ever get back to 95%; LinUCB and NB1 never do before
  the next drift. Clicks lost per 500k window: LinUCB 129k–201k, NB1 38k–153k (grows each drift), NB2 31k–47k,
  NB3 34k–49k. Verdicts: NB2 and NB3 adapt better than NB1 and LinUCB from drift 2 on; at drift 1 every NB
  beats LinUCB. **Compared with Phase 4's ad-hoc measure** (NB2 ~249k/294k/166k, NB3 ~348k/239k/169k, all
  "inside 75k–350k"): the new measure is stricter because the stationary reference keeps improving, so most
  drifts count as "never recovered", and NB3's 443k after drift 3 falls **outside** the paper's range. The
  paper should report the new measure and say why the numbers differ.
- 📊 *(Nov 17, Phase 5d S5)* **MIND-small rehearsal** (seed 100, shifts 50k/100k/150k, 3.5 min): every check
  passed (pairs identical before shift 1, every output produced). Shift 1 bit weakly on small (NB1 stayed
  above 95%) and Random showed one single-seed noise blip at shift 3 — both expected with one seed; the
  10-seed large run shows neither.

## 7. Failures, surprises & dead ends

- 🔍 *(Nov 17, Phase 5d)* **The 500k "tv" spike leaks into shift 1's window.** S1 moved shift 1 to 300k so the
  shift itself avoids the spike, but the 500k-step measurement window [300k, 800k) still contains it. Around
  500k the shifted run's click rate jumps to ~1.5× the no-shift run for LinUCB and Popularity (figure
  `mind_e4_recovery.png`). Split of LinUCB's clicks lost: [300k, 480k) +1,014; [480k, 560k) **−3,537**;
  [560k, 800k) −24. The NB methods lose clicks in all three parts (NB1 +1,352 / +910 / +1,181). So LinUCB's
  −2,546 at shift 1 comes from its no-shift twin doing badly during the spike, not from robustness. The
  pre-registered rule is kept as is (not changed after seeing results); the paper reports shift 1 with this
  caveat and leans on shifts 2–3. ⚖️ *(user, Nov 17)* The paper shows the split as **one sentence with the
  numbers** (not a figure/table), e.g. "LinUCB's negative value at shift 1 comes entirely from steps
  480k–560k (−3,537 clicks), where its no-shift run drops during a news-event spike; outside it, it loses
  ≈ 1,000 clicks." Why: without the numbers our own rule appears to call LinUCB more robust; with them the
  claim is checkable, and keeping the pre-registered rule while explaining it reads as honest.
- 🔍 *(Nov 17, Phase 5d)* LinUCB recovers from shift 3 (236k) and then falls below 95% again at ~1.97M; the
  stopwatch counts only the first recovery (R10 definition), so this second dip shows only in the figure.
  It lines up with the natural jump at ≈ 1.94–1.96M listed under S1 (§5) — inference: a real news event
  hits the relabelled LinUCB harder than its no-shift twin.
- 📝 *(Nov 17, user)* **Paper framing — natural vs planted drift (E3 vs E4):** E3 has real, short news-event
  spikes (music Nov 10, finance Nov 12, sports Nov 15, the tv spike at ≈ 460k–540k steps; §4 and S1 in §5)
  with no clean start moment; E4 adds planted shifts with exact start times on top of them. The two
  interact: the tv spike distorts shift 1's measurement (above) and the 1.94M jump gives LinUCB a second dip.
  Talking points: (1) on natural spikes NB1/2 rise and LinUCB stays flat (Nov 15: LinUCB 41.8% vs NB 45–46%);
  (2) planted shifts confirm the same order with a precise clock; (3) real drift can't be fully separated
  from planted drift on real data — a limitation and the reason E4 shift times avoid natural jumps.

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
- 📊 *(Nov 17, Phase 5d)* **E4 compute:** 60 runs on MIND-large in **2 h 32 min** (Random/Popularity/LinUCB/NB1
  done in 12 min; NB2/NB3 ≈ 50 min per wave of 8). `make_recovery.py e4` takes 47 s. Lesson: other apps
  (Docker's VM, Chrome) had filled RAM and swap, and the workers ran at ~30% CPU until they were closed —
  check free memory before a long batch.
- 📊 *(Nov 18, Phase 6a build)* **NeuralUCB built** (`models/relu_net.py`, `exploration/ucb.py`,
  `algorithms/neuralucb.py`; numpy, hand-written backprop). Checks: hand gradients = finite differences, and =
  torch autograd to ≤ 1.7e-16 on a MIND-sized net (K 16, d 19, m 100; one-time `uv run --with torch`, torch not
  added to requirements); the paired init = the paper's literal doubled-input form step for step (f, bonus,
  several SGD steps) in a test. Speed after round 2000 (one job alone, MIND-small): ≈ 160–180 µs/step (the first
  2,000 rounds train every round: ≈ 17 s per run). Sanity (MIND-small seed 100, steps 2k–32k, J 10, λ 10⁻², η 2·10⁻²):
  γ = 0.01 → 42.5% clicks, exploration rate 0.6%; **γ = 0.1 → 45.4% in steps 22–32k** (LinUCB's whole-pass rate on
  small: 43.7%), 7.9%; γ = 1 → 34.7%, 45% — the bonus must be tuned (as planned).
- 🔍 *(Nov 18, Phase 6a review)* `/simplify` (4 reviewers): one diagnostics helper, LinUCB-style masking in select
  (same RNG draws), batch grouped by arm once (≈ 17% faster SGD step), `.take` gathers, int16 arm buffer; divergence
  handling made general (`diverged` column only when something diverged → earlier CSVs keep their shape;
  `make_recovery` drops an algorithm with any diverged run; `make_mind_figures.py` takes result-set names).
  Phase 5 CSVs re-generated byte-identical. `/code-review` (5 findings): fixed — J, batch, m written into every 6a
  config so the saved-run key records them (else a changed code default could reuse stale runs); assert even m
  (paired init). Skipped with reasons: test writes into `results/` (cleaned in `finally`); diverged runs' seconds
  include data loading; UCB score in the algorithm (bonus is in `exploration/`, as LinUCB's precedent).
- 📊 *(Nov 18, Phase 6a rehearsal, `results/phase6a_rehearsal_jcheck.log`)* MIND-small seed 100 no-shift vs shifts
  50k/100k/150k: rewards identical before 50k, differ after; Forest 50k stationary vs rotation at 25k: identical before
  25k; `make_recovery.py trial_6a` wrote every output. Mechanics only (γ = 1 setting; nothing tuned from it).
- 📊 *(Nov 18, Phase 6a J check — pre-registered rule, R3)* γ 1, λ 10⁻², η 2·10⁻², seeds 100–102, 6 jobs in parallel:

  | | J = 10 | J = 50 | diff | 2 × SE | s/run (J 10 / 50) |
  |---|---|---|---|---|---|
  | MIND-small click rate | 0.3594 | 0.3644 | +0.0051 | 0.0041 | 130 / 258 |
  | Forest 500k accuracy | 0.5949 | 0.6079 | +0.0130 | 0.0053 | 330 / 647 |

  → **J = 50 is clearly better on both** (more training per retrain pays). User's rule (Nov 18, at approval): take the
  clearly better J unless "mega slow" (tuning > ≈ 12 h); measured J = 50 tuning ≈ 5.5 h → **J = 50 used** for tuning
  and the full runs (paper: J = 1000; still a cost deviation). Full runs at J = 50 ≈ 4.5–5 h (vs 2.5 h planned).
- 📊 *(Nov 18, Phase 6a MIND tuning stage 1)* 128 settings, seed 100, J 50, 8 jobs in parallel: **0 of 128 diverged**
  (the η/m step, R1, holds across the full paper grid). ≈ 570 s/run with 8 jobs in parallel (vs 258 s with 6 jobs at the J check),
  128 runs in ≈ 2.5 h. Top 5 by mean reward (seed 100): γ 0.1 λ 0.1 η 0.2 = 0.4186; γ 0.1 λ 1e-4 η 0.05 = 0.4162;
  γ 0.01 λ 0.01 η 0.2 = 0.4158; γ 0.01 λ 0.1 η 0.2 = 0.4157; γ 0.1 λ 0.1 η 0.05 = 0.4156. These are near-ties
  (spread 0.003). Small γ (≤ 0.1) clearly beats γ ≥ 1: γ 10 gives ≈ 0.19–0.20, so heavy exploration is costly on MIND.
  Stage 2 (seeds 101–102) has started: `configs/phase6a_tune_mind_top5.yaml`.
- 📊 *(Nov 18, Phase 6a MIND tuning stage 2, seeds 100–102)* 3-seed means: **γ 0.1 λ 0.1 η 0.2 = 0.4175 (SD 0.0010) → MIND
  setting**; γ 0.1 λ 0.1 η 0.05 = 0.4158; γ 0.01 λ 0.1 η 0.2 = 0.4156; γ 0.1 λ 1e-4 η 0.05 = 0.4145; γ 0.01 λ 0.01 η 0.2 = 0.4115.
  The seed-100 winner holds up on the other seeds.
  ⚠️ **Finding:** tuned NeuralUCB is *below* the tuned baselines on MIND-small in the same tuning protocol: LinUCB
  α 0.5 = 0.4367 (Phase 5), NB1 C 1 λ 1 = 0.4460 (Phase 5c), against NeuralUCB 0.4175. γ 0.01 and γ 0.1 score about the same,
  so the gap points to the network fit rather than to exploration. Candidate causes: J = 50 instead of the paper's 1000, and
  retraining only every 100 steps after round 2000. For the user at the checkpoint.
- 🔍 *(Nov 18, Phase 6a Forest tuning bug)* Forest stage 1 crashed at once (ZeroDivisionError in `select`). With η = 0.5,
  the grid's largest step, Forest's 94 inputs (vs 19 on MIND) drive the weights so high that they stay finite while f or Z
  overflow, so u = NaN. The weight check in `_train` missed this. Reproduced at 3,000 steps: every η = 0.5 setting
  fails within the first steps (4 are caught by the weight check, 11 by NaN u). Smaller η is fine. Fix, under the approved R2 fail-fast rule:
  `select` raises `Diverged` if any u is non-finite. Added test `test_nonfinite_scores_raise_diverged`, so 91 tests now.
  MIND results are unaffected: a NaN u always crashed the run, and every MIND run completed. Forest stage 1 has been relaunched.
- 📊 *(Nov 18, Phase 6b planning)* **Measured cost of a window on NB1** (throwaway prototype: every R steps a fresh
  network is trained by one pass, in time order, over the stored Eq. 1 updates of the last W steps; one job alone, seed 100).
  One replayed update ≈ 23 µs (MIND, C 1) / ≈ 49 µs (Forest, C 100); extra cost per step = W/R replayed updates.
  MIND-small: plain NB1 0.4448 in 7 s; W 20k/R 5k 0.4345; W 50k/R 10k 0.4416 (24 s); W 100k/R 25k 0.4279.
  Forest 200k: 0.7127 → 0.6972 (W 50k/R 10k; 12 s → 48 s). Estimate with W/R = 5: ≈ 12–15 min per MIND-large run in
  parallel, so E3 + E4 × 10 seeds ≈ 45–60 min.
  🔍 Every window lost to plain NB1 on MIND-small (one seed). Expected: a window pays off only where old data misleads.
  The tuning rule for W is therefore the central 6b question.
- 📊 *(Nov 18, Phase 6b build)* `nonstationary/window.py` (Notebook: arrays, trim keeps step > t − W) and
  `NeuralBandit1SW` (logs NB1's Eq. 1 updates; retrain at the end of update when t % (W//5) = 0 and t > W; fresh
  `MLPStack` from self.rng; one in-order replay with the forward pass recomputed). 8 new tests, 99 in total. /simplify:
  simpler Notebook (trim moves the kept rows to the front) and a bulk read during replay. /code-review: 7 findings,
  1 fixed (W < 5 raises), 6 skipped with reasons. **Rehearsal (MIND-small seed 100, W 20k):** shifted/unshifted pair
  identical before 50k, first retrain at 24k as planned, 46 retrains, `make_recovery.py trial_6b` produced every
  output. Click rate 0.4344 (no shift) vs plain NB1's 0.4448 on the same seed. **Largest window (synthetic, W 987k):**
  one retrain of 987k rows takes 22.8 s alone (23 µs/row, as measured in planning); peak RSS 630 MB with the notebook's
  capacity at 2.1M rows.
- 📊 *(Nov 18, Phase 6b tuning, MIND-small, seeds 100–102, no shifts)* 3-seed mean click rate (SD):
  NB1 (no window) 0.4460 (0.0010); W 10k 0.4257 (0.0015); W 20k 0.4342 (0.0002); W 50k 0.4392 (0.0021); W 100k 0.4400 (0.0010).
  Every window is below plain NB1, more so the smaller it is (expected on shift-free data). Rule (Q4/R2): the best windowed W is 100k.
  50k is within 2 × combined SE (diff 0.0008 < 0.0027) and 20k is not (0.0058 > 0.0012) → **W = 50k → MIND-large 493k** (× 9.868).
- 🔍 *(Nov 18, Phase 6b, found before the checkpoint)* **A dead-arm trap after a reset.** A real MIND-large run (seed 0,
  W 987k, alone) is identical to NB1 until its first reset (1.18M), then averages 0.360 / **0.206** (Random level) /
  0.402 / 0.459 over the next four reset intervals, vs NB1's 0.437 / 0.444 / 0.401 / 0.461. Whole run: 0.428 vs 0.460.
  - A control run that resets at the same steps but replays **all** history matches NB1 exactly (0.4600 vs 0.4600), so
    the replay is correct and the loss comes from forgetting.
  - Mechanism, from inspecting the weights: after the reset at 1,381,800 some arms' 1-hidden-unit nets end with output
    weight ≈ 0 and hidden unit ≈ 0. Their output is then stuck near sigmoid(0) = 0.5, above every other arm (≈ 0.43), so
    they are picked whenever shown, and their gradients (∝ h, ∝ w2) are ≈ 0, so they never correct. The run stays stuck
    until the next reset. Online NB1, trained from the start of the week, had those arms' w2 at −2 to −3.
  - MIND-small tuning runs show no stuck stretch (no 10k block below 0.25). The trap was seen on MIND-large only.
  - Memory: one MIND-large job at W 987k peaks at 1.04 GB RSS (183 s alone); 8 parallel ≈ 8 GB on a 16 GB Mac.
- ❓ *(Nov 18, user idea, optional, not decided)* **Window on NB2 as a small extra at the end.** Reasoning (untested):
  NB2's committee holds slow learners (λ 0.01, 0.1) and big networks (C up to 100) that keep old data far more than NB1
  (λ = 1), so a reset could restore more plasticity. EXP3 could also steer around a model stuck in the dead-arm trap.
  Cost (estimate, not measured): replay through 15 models makes one windowed NB2 MIND-large run ≈ 4–5 h, so one W × E3 + E4
  × 10 seeds ≈ 18 h. Cheaper variants: 5 seeds (≈ 9 h), reset every W instead of W/5 (≈ 6–7 h), or both (≈ 3–4 h).
  It would use NB1's tuned W (not tuned for NB2, to be stated) and needs its own short plan with a timing run first. To be
  decided only after 6c/6d, if time allows. Otherwise it goes in the paper as a limitation / future work: "the window was
  tested on NB1, the fastest-adapting learner; slower committees like NB2 may benefit more."
- 📊 *(Nov 18, Phase 6b results: MIND-large, 10 seeds, E3 + E4, 140 runs, 6 jobs, ≈ 3.5 h)* Outputs:
  `results/phase6b_mind_{e3,e4}_*.csv`, `phase6b_mind_e4_{recovery,verdicts,table}.csv`,
  `figures/mind_e4_6b_{recovery,regret}.png`, `figures/mind_e3_6b_{daily,regret}.png`. The Phase 5c figures are kept unchanged.
  - **Done-checks pass:** all 140 runs are full length and finite; E4 = E3 bit-for-bit for the first 300k steps in all 70 pairs.
  - **E3 click rate** (NB1 46.0%): W 10k 43.3, 20k 43.9, 50k 43.9, 99k 43.8, 197k 44.0, **493k 45.1**, 987k 43.3.
    Every fixed window costs click rate on the real week. Around E3's natural jumps the windows mostly fall *further*
    behind NB1 (e.g. W 50k: −1.3 → −8.4 points around ≈ 370k). The exception is the TV spike, where small windows close part of the gap.
  - **Clicks lost per planted shift** (vs the same seed's E3 run; NB1 3,443 / 5,544 / 6,180): W 10k −26 / 68 / 303,
    W 20k 33 / 337 / 392, W 50k −1,000 / −561 / 455. Small windows make the planted shifts nearly free.
  - **Adaptation time** (R10 stopwatch; NB1 10.7k → 45.4k → 113k, slower with every shift): W 10k 0 / 0 / 3k,
    20k 4k / 0 / 10k, 50k 10k / 8k / 10k, 99k 1k / 9k / 9k, 197k 10k / 9k / 29k, 493k 11k / 11k / 56k, 987k 11k / 45k / 51k.
    **Windows ≤ 99k flatten NB1's slowdown**: they recover within ≈ 10k at every shift. This is evidence for the plasticity
    explanation (Phase 5d), although the timing confound (later, weaker days) still applies.
  - **Verdicts (pre-registered rule) vs NB1:** W 10k–197k "trade performance for robustness" at every shift (fewer clicks
    lost, but fewer total E4 clicks: −3k to −10k per window; 197k is level at shift 3). **The headline W 493k "adapts
    better" than NB1, NB2 and NB3 at all 3 shifts**, and than LinUCB at shifts 2–3 (shift 1 vs LinUCB = the 500k TV-spike
    caveat, §7). Independent check that needs no twin: W 493k's E4 clicks per post-shift window vs NB1's are
    +74 ± 441 / **+2,229 ± 213 / +2,368 ± 271** (2 × SE), i.e. level at shift 1 (no reset before 591.6k) and more at shifts 2–3.
    Its overall E4 click rate equals NB1's (45.2%).
  - 🔍 **The dead-arm trap is systematic, not random.** It hits the same reset point in (nearly) every seed, because it
    is tied to the data slice the network is rebuilt from:
    - **W 987k:** stuck ≈ 1.28M and 1.38–1.48M in all 10 E3 runs (135 blocks of 10k steps), never in E4.
    - **W 197k:** 14 blocks in 9/10 E3 runs vs 3 in E4.
    - **W 493k:** one 10k block at ≈ 890k (just after the reset at 887,400) in all 10 E4 runs; 2 in E3.
    - **W 50k:** 2–3 blocks. **10k, 20k and 99k:** none.
    - Plain NB1: none.
    Because the planted relabelling changes which arms the slice favours, the trap can hit an E3 run and not its E4
    twin, which makes "clicks lost" negative (W 197k shift 1 −10,906; W 987k shift 2 −34,097). For the large windows,
    clicks lost is therefore confounded by the trap. The twin-free E4-clicks comparison above is the cleaner check.
  - **Reading (as pre-registered):** a fixed blind window buys near-free recovery at the planted shifts, but loses clicks
    the rest of the time. Small windows pay too much, large windows risk the trap. The tuned 493k is the one fixed size
    that comes out ahead of NB1 after shifts while matching it overall on E4 (but 45.1% vs 46.0% on E3). This motivates
    6c/6d: reset when a change is detected, keeping the window's recovery without its constant cost and without needless resets.
- ⚠️ *(Nov 18, side-agent note, agreed)* **The W 493k "adapts better" result may partly be reset timing.** Resets
  follow a fixed schedule (every 98,600 steps) that never lines up with the shifts, and the dead-arm trap hits specific
  resets in almost every seed. So which size comes out ahead depends partly on where its resets happen to fall on the
  data, which may be why 493k beats NB1 while 197k and 987k don't. **For the paper:** report it as one window size on one reset
  schedule, not as a general property of W ≈ 493k. A check is possible but not run: the same W with the schedule offset
  (e.g. the first reset shifted by R/2), E3 + E4, 10 seeds, ≈ 40 min. **Run (user, Nov 18):** `NeuralBandit1SW(offset=…)`
  resets when (t − offset) % R == 0 (default 0, so earlier saved runs are unchanged; 1 new test, 100 in total).
  Configs `phase6b_offset_mind_{e3,e4}` use W 493k with offset 49,300 = R/2, label `NB1-SW_W493k_off`; outputs via
  `make_recovery.py e4_6b_off` (vs NB1-SW_W493k and every baseline). Same rule, seeds and shifts.
  📊 **Result (17:02, 20 runs): the 493k win depends on reset timing.**
  - **Original schedule:** E3 45.1%, E4 45.2%; post-shift E4 clicks vs NB1 +74 / +2,229 / +2,368; trap blocks E3/E4 2 / 10;
    verdicts vs NB1 better / better / better.
  - **Schedule moved by R/2:** E3 **42.6%**, E4 44.1%; post-shift E4 clicks vs NB1 −9 (±438) / **−14,312 (±3,639)** / +2,638 (±246);
    trap blocks E3/E4 **94 / 69**; verdicts trades / trades / better.
  - Same window size and data, rebuilds at different moments: the trap hits far more often and the result flips at shift 2.
  - ⇒ The paper must **not** claim "a 493k window adapts better". Correct statement: a fixed blind window's outcome is
    dominated by where its rebuilds land, through the dead-arm trap.
  - The robust 6b findings are the small-window ones (10k–50k: near-zero clicks lost, recovery ≤ 10k, never or rarely trapped,
    1–3 points lower click rate).
  - This strengthens the case for 6b part 2 (gradual forgetting, no rebuilds).
- 📊 *(Nov 18)* New figure `figures/mind_e4_6b_nb1_vs_windows.png`: one row per window size, NB1 (black) vs that window
  (orange) at each shift. The all-lines `mind_e4_6b_recovery.png` is unreadable with 13 lines. Values above 1.0 are the
  no-shift twin falling into the trap, not real gains.
- ⚖️ *(Nov 18, user)* **No fix for the dead-arm trap inside the window.** Options weighed: a validation check that keeps
  the old net unless the rebuilt one predicts recent held-out data better; several rebuilds keeping the best; pessimistic
  init; re-initialising a stuck arm. All were rejected because they overlap with the controllers' job (deciding when and
  whether to reset). The trap is reported as a 6b finding. The only related option kept is gradual forgetting (pull the
  weights toward their initial values, close to the "shrink" in Dohare et al.'s Shrink and Perturb). It is already Q3b: its
  own short plan after 6b, done only if time allows.
  **Update (user, Nov 18): gradual forgetting becomes Phase 6b, part 2, and will definitely be done before 6c**, right after the
  493k reset-timing check, with its own short plan (`paper/plans/phase-6b2.md`). Why it belongs in 6b: it is the other way to
  forget old data (Gama et al. §3.1.2: abrupt vs gradual). If it wins, the 6c/6d controller sets a fade rate instead of W
  (a documented deviation from the abstract's "sliding window"; ablation "−sliding window" → "−forgetting"). Estimated cost ≈ 2–3 h of work, < 1 h of compute (runs cost about as much as plain NB1).
- 📊 *(Nov 18, Phase 6b part 2 build + rehearsal + tuning)*
  - **Build:** `nonstationary/forgetting.py` (`fade_toward_init`, rejects ρ ∉ [0, 1)), `NeuralBandit1Fade`, and in
    `evaluation/metrics.py` `trap_blocks`, `two_se`, `tradeoff_beats`, `beaten_windows` (the pre-registered rule, now tested).
    Tests: 111 (12 new; shared toy stream in `tests/toy.py`).
  - **/simplify:** shared test helpers, a simpler update-then-fade test, `two_se`. Skipped: moving the figure code and the label tables.
    **/code-review:** 6 findings; 3 fixed (decision rule moved into a tested function, ρ validated, test added), 3 skipped (minor).
  - **Rehearsal** (MIND-small seed 100, ρ = 1/20k): pair identical before 50k, traces saved (207 blocks), every output produced.
    One MIND-large run alone: 88 s (43 µs/step), peak RSS 450 MB, so 8 parallel jobs are fine.
  - **Tuning** (MIND-small, seeds 100–102, 3-seed mean ± SD):
    - plain NB1 0.4460 ± 0.0010 (reference);
    - fade 1/100k 0.4451 ± 0.0006; 1/50k 0.4432 ± 0.0006; 1/20k 0.4384 ± 0.0009; 1/10k 0.4307 ± 0.0020.
    - As stated in advance, the paper's rule picks the slowest fade: **headline 1/100k → 1/987k on MIND-large**.
    - At the same *nominal* memory each fade beats the matching window on calm data (window 10k 0.4257, 20k 0.4342, 50k 0.4392, 100k 0.4400).
      This is consistent with R6 (a fade keeps more old data than a window of the same nominal memory). The trade-off rule decides, not this.
- 📊 *(Nov 18, Phase 6b part 2 results: MIND-large, 8 fade rates × E3 + E4 × 10 seeds, 160 runs, 8 jobs, ≈ 1.3 h)* Outputs:
  `results/phase6b2_{tradeoff,decision}.csv`, `phase6b2_mind_e4_{recovery,verdicts,table}.csv`,
  `figures/mind_6b2_tradeoff.png`, `figures/mind_e4_6b2_fade_vs_windows.png`.
  - **Done-checks pass:** all 160 runs are full length and finite; E4 = E3 bit-for-bit for the first 300k steps in all 80 pairs; Phase 5 / 6b
    tracked outputs are unchanged.
  - **Trade-off points** (E3 click rate; mean clicks lost per shift):
    - NB1 45.97%, 5,056.
    - Fades: 1/10k 43.33%, 44; 1/20k 44.25%, 504; 1/50k 45.01%, 858; 1/100k 45.45%, 1,367; 1/99k 45.45%, 1,341;
      1/197k 45.71%, 1,769; 1/493k 45.91%, 2,564; **1/987k 45.96%, 3,172**.
    - Windows: see the 6b entries.
  - **Pre-registered decision: "window".** Fading beats only 1 of the 3 small windows (W 20k, by fade 1/20k: +0.32 points of E3
    clicks, clicks lost a tie). W 10k: tie on both axes (fade 1/10k). W 50k: the fades with more clicks also lose more at shifts.
    **Placebo OK:** plain NB1 beats 0 windows. So 6c/6d's controller sets the **window size**, as pre-registered.
  - **The trap never fired for any fade:** 0 trap blocks in all 160 runs (windows: up to 135).
  - **Fading vs plain NB1** (descriptive, Q11):
    - Fades 1/197k, 1/493k and 1/987k "adapt better" than NB1 at **all 3 shifts** (pre-registered adapts-better rule).
    - The headline 1/987k keeps NB1's E3 click rate (45.96% vs 45.97%) and loses **37% fewer clicks per shift** (3,172 vs 5,056).
      Under the trade-off rule it **beats plain NB1**: equal clicks, fewer lost.
    - Adaptation time (NB1 10.7k → 45.4k → 113k): 1/987k 10.1k / 15.5k / 42.3k; 1/99k 8.4k / 8.6k / 15.6k;
      1/20k 3.1k / 6.8k / 10.5k. Every fade flattens NB1's slowdown, smoothly with its rate.
  - **Reading:** the fades form a smooth curve from NB1 (no forgetting) to the small windows. They give up very little normal-time
    performance for large gains at shifts, with no trap and no timing sensitivity. Against the *small* windows (which lose almost nothing at
    shifts) fading does not dominate, so the pre-registered rule keeps the window for 6c/6d. Fading is, however, the **first component
    that beats plain NB1 robustly** (no schedule luck). To raise in 6c planning (user decision): the pre-registered answer stands, but
    whether 6c/6d should also offer the fade rate as a knob is a fair new question. **Resolved (user, Nov 18): fading is the
  forgetting method and the controller sets the fade rate (§5 decision entry).**
- 📊 *(Nov 18, Phase 6c step 1)* **Fixed-γ check** (`results/phase6c_gamma_check_summary.csv`; MIND-small, seeds 100–102,
  NB1 + fade 1/100k, no shifts). 3-seed mean click rate ± SD:
  - γ = 0: 0.4456 ± 0.0007;
  - γ = 0.005 (paper): 0.4451 ± 0.0006;
  - γ = 0.02: 0.4409 ± 0.0005;
  - γ = 0.05: 0.4333 ± 0.0002.

  Reading:
  - **More exploration only costs clicks**, as expected: exploration plays never train NB1.
  - **γ = 0 vs 0.005: no clear difference.** The gap is +0.0005 (2 × combined SE ≈ 0.0011), though γ = 0 is ahead on every one
    of the 3 seeds.
  - Under plan rule R2, no γ clearly beats 0.005, so **γ leaves the LLM's action menu**. A and C then have the same dials, and this
    check is the "−adaptive exploration" evidence: in NB1 as published, the controller has nothing to gain from γ.
- 🔍 *(Nov 18, Phase 6c build)* **river's Page-Hinkley equals Gama et al.'s Alg. 1 exactly** (drops version: mode "down",
  alpha = 1). The first alarm step is identical at every λ tried on a synthetic drop (5, 10, 20, 30, 50, 80). The test is permanent
  (`tests/test_page_hinkley.py`). On noise of ±0.5 per step, λ ≤ 20 already raises false alarms, so λ must be set by the budget.
- 📊 *(Nov 18, Phase 6c steps 4–5)* **PH measurement and thresholds, PRE-REGISTERED before any MIND-large controller run**
  (`scripts/measure_ph.py` → `results/phase6c_measure.csv`; `configs/phase6c_confirm_small.yaml`). MIND-small, seeds 100–102,
  NB1 + fade 1/100k, no planted shifts.
  - **Offline simulation over 25 λ values (10–2000):**
    - **Rule:** the smallest λ with boosted share ≤ 10% / ≤ 30% on every seed.
    - **Result:** **λ_A = 219.9**, 1 alarm per run (at ≈ 48k); **λ_S = 90.9**, 3 alarms (≈ 18k, 65k, 105k).
  - **Confirmed by real controller runs** (boosts change NB1's path):
    - **Boosted share:** A 9.6–9.7%, B 28.9–29.0%. Alarms land at the same steps as the offline simulation.
    - **Calm-data cost vs the slow fade (0.4451):** A 0.4442 (−0.09 points), B 0.4429 (−0.22 points).
  - 🔍 **Alarms track the week's real events:** at a given λ they fire at nearly the same step in all 3 seeds (≈ 17–18k, 48k, 65k,
    88k, 105k, 127k, 148k), so they follow MIND's natural drift, not noise.
  - 🔍 **Alarm count is not monotonic in λ** (λ 219.9 gives 1 alarm, 274.2 gives 2): an early alarm's 20k-step boost can swallow a
    later one. The pre-stated "smallest λ" rule is applied as written.
  - 🔍 **NB1's surprise is slightly positive on average** (+0.00 to +0.06 per 10k block, seed 100): NB1 slightly *under*-estimates
    the arms it plays. The review expected a negative bias; it isn't there.
  - ⚠️ **Open (side-agent note):** the per-step threshold may not carry over to MIND-large unchanged. The same week has ≈ 9.9× the
    steps, and the base fade is 1/987k there. MIND-large must not be used to adjust it, so a fallback must be decided before the runs.
- 📄 *(Nov 18, Phase 6c step 2)* **Page 1954 read in full** (published JSTOR version; REFERENCES_VERIFICATION R9, C55–C58).
  - **It supports our threshold rule:** Page measures a detector by its average run length (A.R.L.). On good quality the A.R.L. is
    the cost of false alarms, which are unavoidable ("probability of one"). Page picks a scheme by fixing the A.R.L. on good
    quality and making detection fast on bad quality (§1.2, §5.2). That is our alarm budget on data without planted shifts.
  - **One-sided by design** (§2.1), as our drops-only `mode="down"`.
  - ⚠️ **Citation detail:** the Page-Hinkley form (deviations from a running mean, tolerance δ) is not in Page 1954. Page's CUSUM
    uses fixed scores chosen to drift down when all is well. We cite PH as Gama et al. (§3.2.1, App. A) and Page 1954 as the
    CUSUM origin.
- 📊 *(Nov 19, Phase 6c results: MIND-large, A and B × E3 + E4 × 10 seeds, 40 runs, 8 jobs)* Outputs: `results/phase6c_mind_{e3,e4}_*.csv`,
  `phase6c_mind_e4_{recovery,verdicts,table}.csv`, `figures/mind_e4_6c_{recovery,regret}.png`; log `results/phase6c_full.log`.
  - **Done-checks pass:** 40 runs complete and finite; E4 = E3 bit-for-bit for the first 300k steps in all 20 pairs.
  - **Clicks lost per planted shift** (mean ± SD; 2 × SE vs the slow fade):

    | | Shift 1 | Shift 2 | Shift 3 | Mean |
    |---|---|---|---|---|
    | NB1 | 3,443 | 5,544 | 6,180 | 5,056 |
    | Slow fade 1/987k | 2,552 ± 396 | 3,171 ± 214 | 3,794 ± 209 | 3,172 |
    | **A** | 1,910 ± 747 | 2,573 ± 230 | 1,888 ± 369 | **2,124** |
    | **B** | 1,932 ± 1,175 | 3,195 ± 622 | 1,884 ± 310 | 2,337 |

    - A loses **clearly fewer** clicks than the slow fade at all 3 shifts: −33% on average, and −58% vs plain NB1.
    - B is clearly better at shifts 1 and 3 and level at shift 2.
  - **Adaptation time** (R10):
    - NB1: 10.7k / 45.4k / 113k;
    - slow fade: 10.1k / 15.5k / 42.3k;
    - **A: 10.0k / 10.5k / 36.4k**;
    - B: 9.9k / 10.5k / 28.7k.
  - **Click rate:**
    - **E3 (real week):** A 45.79%, B 45.74% vs slow fade 45.97% and NB1 45.97%. That is −0.18 / −0.22 points, both **clear**
      (2 × SE ≈ 0.03).
    - **E4 (drift data):** A 45.47% = slow fade 45.47% (no clear difference); B 45.39% (−0.08, clear).
  - **Alarms on MIND-large** (λ from MIND-small, unchanged, R8):
    - A: 5.4 per E3 run, 5.5 per E4 run; boosted share 5.3% (target 10%).
    - B: 7.3 / 10.1 per run; boosted share 7–10% (target 30%).
    - The MIND-small calibration carried over **conservatively**, with fewer boosts than budgeted.
    - **C's call budget ≈ 7–10 LLM calls per run**, so −detector gets the same.
  - **Reading against the pre-written claims (R3):**
    - **"The controller helps" is NOT met as written.** A loses clearly fewer clicks at every shift and adapts faster, but costs a
      clear 0.18 points on the real week. It is a **trade**: on the drift data overall it ties the slow fade.
    - The cost on E3 comes from boosts at **natural news events**: ≈ 5 alarms per E3 run, where the week has no planted shift.
    - This is exactly the gap the LLM is meant to close (6d): **keep A's shift gains while skipping the natural-event boosts.**
      The bar for C: E3 clicks within noise of the slow fade (45.97%), and clicks lost per shift ≤ A's ≈ 2.1k.
    - B is no better than A: more alarms, more cost, same gain. Extra alarm sensitivity alone doesn't help the rules.
- 📊 **Test counts:** Phase 6c build + review: 121 (code-review fixes: invalid decisions count as no change; controllers are named
  and the boost is a config param, so both are in the saved-run key; no-change alarms are logged as NaN). Phase 6b part 2: 111. Phase 6b: 99. Phase 6a: 91. Phase 5d: 82. Phase 1: 18; Phase 4: 57; Phase 5b: 73 (incl. the permanent 5k-step Forest Cover
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

**Reference map — what each source is used for in this project** (full bibliographic records and
checks: `paper/REFERENCES_VERIFICATION.md`; ✅ = verified in ≥ 2 independent sources):

| # | Reference | ✓ | What we use it for / what it inspired |
|---|---|---|---|
| R1 | Allesiardo, Féraud & Bouneffouf 2014, *A Neural Networks Committee for the Contextual Bandit Problem* (arXiv:1409.8191v1; ICONIP 2014) | ✅ | The base paper we reproduce: NB1/2/3, Forest Cover setup, 0.93 reference, drift = class rotation every 500k, "75k–350k to stabilize" — the gap our method targets |
| R2 | Zhou, Li & Gu 2020, *Neural Contextual Bandits with UCB-based Exploration* (NeuralUCB), ICML 2020 | ✅ | Core of NeuralBandit++ (Phase 6): uncertainty-aware exploration replacing NB's fixed γ; regret Õ(d̃√T) the 2014 paper lacks |
| R3 | Li, Chu, Langford & Schapire 2010, *A Contextual-Bandit Approach to Personalized News Article Recommendation* (LinUCB), WWW 2010 | ✅ | The classical baseline (instructor's "beat LinUCB"); also justifies MIND **masking** (per-trial arm set A_t, C23) and the **constant-1** context feature (C24) |
| R4 | Agrawal & Goyal 2013, *Thompson Sampling for Contextual Bandits with Linear Payoffs* (CTS), ICML 2013 | ✅ | Appears in the base paper's Figure 1; **not reproduced** (scope: one classical baseline) |
| R5 | Kakade, Shalev-Shwartz & Tewari 2008, *Efficient Bandit Algorithms for Online Multiclass Prediction* (Banditron), ICML 2008 | ✅ | Appears in the base paper's Figure 1 (57%); **not reproduced** |
| R6 | Auer, Cesa-Bianchi, Freund & Schapire 2002, *The Nonstochastic Multiarmed Bandit Problem* (EXP3), SIAM J. Comput. | ✅ | EXP3 inside NB2/NB3 (model selection over the committee) |
| R7 | Wu et al. 2020, *MIND: A Large-scale Dataset for News Recommendation*, ACL 2020 | ✅ | The real-world dataset (E3/E4); facts, splits, license |
| R8 | Bifet & Gavaldà 2007, *Learning from Time-Changing Data with Adaptive Windowing* (ADWIN), SIAM SDM 2007 | ⏳ | Planned drift detector for the rule-based controller / drift monitor (Phase 6); one source so far → not citable yet |
| R9 | Page 1954, *Continuous Inspection Schemes* (Page-Hinkley test origin), Biometrika | ✅ | Planned drift detector for the rule-based controller (Phase 6) |
| R10 | Shaker & Hüllermeier 2015, *Recovery analysis for adaptive learning from non-stationary data streams: Experimental design and case study*, Neurocomputing 150:250–264 | ✅ | **Defines how we measure adaptation** (Phase 5d): reference-curve comparison, "duration" with 95% thresholds, maximum performance loss (C30); supports the **cyclic label shift** used for E4 (C31) and semi-synthetic drift on real data (C29). Read Nov 17 (authors' draft) |
| R11 | Garivier & Moulines 2011, *On Upper-Confidence Bound Policies for Switching Bandit Problems*, ALT 2011 (LNCS) | ✅ meta | Classic **sliding-window / discounted UCB** for abruptly changing bandits — background for our sliding-window component (Phase 6). Metadata verified; **paper not read yet** — read before citing any claim from it |
| — | Course slides (Intro-Deep Learning for Data Science.pptx) | — | Requirements: baselines (slide 69), ablations (68), evaluation metrics incl. "adaptation time after drift" (67), concept drift (60), report questions (73), repo layout (71) |
| — | Approved abstract (Oct 5) | — | Our promised method, datasets and metrics |

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
