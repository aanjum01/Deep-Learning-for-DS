# Reference & Claim Verification Log

Rules (see CLAUDE.md): nothing is cited from memory; every bibliographic field is checked against
≥2 independent primary sources; every claim about another paper records its exact location;
unverified items are marked `[UNVERIFIED]`; discrepancies are flagged, never silently resolved.
Only ✅ entries may go into `references.bib`.

Status legend: ✅ verified · ⏳ pending · ⚠️ discrepancy (see note)

---

## Part 1 — Bibliographic records

### R1 `allesiardo2014neural` — base paper ✅ (with a recorded name discrepancy)
| Field | Value | Source |
|---|---|---|
| Title | A Neural Networks Committee for the Contextual Bandit Problem | provided PDF; arXiv abs; Crossref |
| Authors (as cited) | **Robin Allesiardo, Raphaël Féraud, Djallel Bouneffouf** | **provided PDF (source of truth)** |
| arXiv | 1409.8191, v1 submitted 29 Sep 2014; cs.NE (primary), cs.LG | provided PDF margin; arxiv.org/abs/1409.8191 |
| Published version | *Neural Information Processing* (ICONIP 2014, 21st Int. Conf.), LNCS, Springer, 2014, pp. 374–381, DOI 10.1007/978-3-319-12637-1_47 | Crossref; arXiv comment field |
| LNCS volume, editors | `[UNVERIFIED]` — not in Crossref record; Springer page requires login | — |
| Verified | 2026-11-12 | |

⚠️ **Name discrepancy (user decision: the provided PDF is the source of truth).** The PDF (arXiv v1) prints
"Robin Allesiardo". The current arXiv metadata and the Springer/Crossref record list "Elys Allesiardo";
the arXiv author line also omits the accent ("Raphael Feraud"). We cite the work as printed in the PDF.
**Decision (confirmed by user 2026-11-12):** cite the arXiv v1 document we actually used, with the PDF's author names.

### R2 `zhou2020neural` — NeuralUCB ✅
| Field | Value | Source |
|---|---|---|
| Title | Neural Contextual Bandits with UCB-based Exploration | arXiv abs; PMLR |
| Authors | Dongruo Zhou, Lihong Li, Quanquan Gu | arXiv abs; PMLR |
| Venue | Proceedings of the 37th International Conference on Machine Learning (ICML 2020), PMLR vol. 119 | PMLR (proceedings.mlr.press/v119/zhou20a.html) |
| Pages | 11492–11502 | PMLR; corroborated by a Crossref reference list |
| arXiv | 1911.04462 (v1 11 Nov 2019; v3 2 Jul 2020) | arxiv.org/abs/1911.04462 |
| Verified | 2026-11-12 | |

### R3 `li2010contextual` — LinUCB ✅
| Field | Value | Source |
|---|---|---|
| Title | A Contextual-Bandit Approach to Personalized News Article Recommendation | arXiv abs; Crossref |
| Authors | Lihong Li, Wei Chu, John Langford, Robert E. Schapire | arXiv abs; Crossref |
| Venue | Proceedings of the 19th International Conference on World Wide Web (WWW 2010), Raleigh, NC, ACM | arXiv journal-ref; Crossref |
| Pages / DOI | 661–670 / 10.1145/1772690.1772758 | Crossref; arXiv "Related DOI" |
| arXiv | 1003.0146 (v1 28 Feb 2010) | arxiv.org/abs/1003.0146 |
| Verified | 2026-11-12 | |

### R4 `agrawal2013thompson` — Contextual Thompson Sampling (CTS) ✅
| Field | Value | Source |
|---|---|---|
| Title | Thompson Sampling for Contextual Bandits with Linear Payoffs | arXiv abs; PMLR |
| Authors | Shipra Agrawal, Navin Goyal | arXiv abs; PMLR |
| Venue | Proceedings of the 30th International Conference on Machine Learning (ICML 2013), PMLR 28(3) | PMLR (proceedings.mlr.press/v28/agrawal13.html) |
| Pages | 127–135 | PMLR |
| arXiv | 1209.3352 (v1 15 Sep 2012) | arxiv.org/abs/1209.3352 |
| Note | The base paper cites the 2012 CoRR version (its ref. [19]); we cite the ICML 2013 version. | provided PDF p. 8 |
| Verified | 2026-11-12 | |

### R5 `kakade2008efficient` — Banditron ✅
| Field | Value | Source |
|---|---|---|
| Title | Efficient Bandit Algorithms for Online Multiclass Prediction | Crossref; provided PDF ref. [20] |
| Authors | Sham M. Kakade, Shai Shalev-Shwartz, Ambuj Tewari | Crossref; provided PDF ref. [20] |
| Venue | Proceedings of the 25th International Conference on Machine Learning (ICML '08), Helsinki, ACM | Crossref; provided PDF ref. [20] |
| Pages / DOI | 440–447 / 10.1145/1390156.1390212 | Crossref; provided PDF ref. [20] (pages) |
| Verified | 2026-11-12 | |

### R6 `auer2002nonstochastic` — EXP3 ✅
| Field | Value | Source |
|---|---|---|
| Title | The Nonstochastic Multiarmed Bandit Problem | Crossref; provided PDF ref. [6] |
| Authors | Peter Auer, Nicolò Cesa-Bianchi, Yoav Freund, Robert E. Schapire | Crossref; provided PDF ref. [6] |
| Venue | SIAM Journal on Computing, 32(1) | Crossref; provided PDF ref. [6] |
| Pages / Year / DOI | 48–77 / 2002 / 10.1137/S0097539701398375 | Crossref; provided PDF ref. [6] |
| Verified | 2026-11-12 | |

### R7 `wu2020mind` — MIND dataset ✅
| Field | Value | Source |
|---|---|---|
| Title | MIND: A Large-scale Dataset for News Recommendation | ACL Anthology; Crossref; msnews.github.io |
| Authors | Fangzhao Wu, Ying Qiao, Jiun-Hung Chen, Chuhan Wu, Tao Qi, Jianxun Lian, Danyang Liu, Xing Xie, Jianfeng Gao, Winnie Wu, Ming Zhou | ACL Anthology; Crossref; msnews.github.io |
| Venue | Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020), Online | ACL Anthology 2020.acl-main.331 |
| Pages / DOI | 3597–3606 / 10.18653/v1/2020.acl-main.331 | ACL Anthology; Crossref |
| Dataset license | Microsoft Research License Terms, research use | msnews.github.io; license PDF read 2026-11-15 (see below) |
| Verified | 2026-11-12 | |

**MIND license — read 2026-11-15** (`MSR License_Data.pdf`, github.com/msnews/MIND, 3 pp.; linked from msnews.github.io).
- §1 Allowed: use and modify for non-commercial research ("teaching, academic research, public
  demonstrations and personal experimentation"); analysing/testing; publish/present results
  "provided that no material portion of the Dataset is included".
- §2 Forbidden: distribute the Dataset; alter copyright notices; imply Microsoft endorsement; malicious use.
- Acceptance is by use: "BY USING THE DATASET, YOU ACCEPT THESE TERMS." §7 US binding arbitration; §11–12 "as is", liability capped at $5.
- No citation requirement in the license (the MIND site asks for one; we cite R7).
- **User agreed to the terms: 2026-11-15** (Phase 5 Q2).
- Consequences for this project: no raw or processed MIND data in git (`data/*`, `results/raw/` gitignored);
  only aggregate results (curves, rates, figures, tables) are committed; the paper and the Dec 14 demo
  show no real data rows (use Microsoft's documented example row or invented ones).

### R8 `bifet2007learning` — ADWIN ⏳ (only one independent source so far)
| Field | Value | Source |
|---|---|---|
| Title | Learning from Time-Changing Data with Adaptive Windowing | Crossref |
| Authors | Albert Bifet, Ricard Gavaldà | Crossref |
| Venue | Proceedings of the 2007 SIAM International Conference on Data Mining (SDM) | Crossref |
| Pages / DOI | 443–448 / 10.1137/1.9781611972771.42 | Crossref |
| Needed | A second source (SIAM page / DBLP — both blocked automated access on 2026-11-12) | — |

### R9 `page1954continuous` — Page-Hinkley test origin ✅
| Field | Value | Source |
|---|---|---|
| Title | Continuous Inspection Schemes | Crossref (OUP record); Crossref (JSTOR record) |
| Author | E. S. Page | both records |
| Venue | Biometrika, 41(1–2) | both records |
| Pages / Year | 100–115 / 1954 | OUP record (JSTOR record gives start page 100 only) |
| DOI | 10.1093/biomet/41.1-2.100 (OUP); also 10.2307/2333009 (JSTOR) | Crossref |
| Verified | 2026-11-12 | |

---

## Part 2 — Claims about other papers (exact locations)

Page numbers for R1 refer to the provided PDF (arXiv v1).

| # | Claim | Source & location | Status |
|---|---|---|---|
| C1 | Context: continuous variables recoded into 5 equal-frequency binary variables, categorical into binary → 94 binary variables; 7 classes = actions; 581,000 instances, shuffled; played in a loop; reward 1 if correct class else 0 | R1 §5, p. 5 | ✅ |
| C2 | Regret computed against an offline algorithm with 93% classification | R1 §5, p. 5 | ✅ |
| C3 | Curves average 10 runs, γ = 0.005, γ_model = 0.1, each run starts at a random position | R1 §5, p. 5 | ✅ |
| C4 | Model grid: hidden layer size {1, 5, 25, 50, 100} × λ {0.01, 0.1, 1} | R1 §5, p. 5 | ✅ |
| C5 | One one-hidden-layer network per action; quadratic error, sigmoid activation | R1 §3, p. 3 | ✅ |
| C6 | Exploration P_t(k) = (1−γ)·1[k = k̂_t] + γ/K | R1 §3, p. 3 | ✅ |
| C7 | Importance-weighted update Eq. (1) writes 1[k̂_t = k]; Algorithm 1 writes 1[k_t = k] (inconsistency in the paper) | R1 Eq. (1) and Alg. 1, p. 4 | ✅ |
| C8 | Weights initialised in ]−0.5, 0.5[ | R1 Alg. 1, p. 4 | ✅ |
| C9 | EXP3 model probability Eq. (2) and weight update Eq. (3) | R1 §4, p. 5 | ✅ |
| C10 | Stationary results (last 100,000 predictions): Banditron 57%, LinUCB 72%, CTS 73%, NB3 73%, NB2 76% | R1 §5, p. 6 | ✅ |
| C11 | Non-stationarity: classes swapped in a circular cycle (1→2, …, 7→1) every 500,000 iterations | R1 §5, p. 6 | ✅ |
| C12 | Model-selection algorithms need 75,000–350,000 steps (worst case) to stabilise after each drift | R1 §5, p. 7 | ✅ |
| C13 | Paper states NeuralBandit has no regret bound (non-convex cost) | R1 §6, p. 7 | ✅ |
| C14 | NeuralUCB regret Õ(d̃√T): Theorem 4.5 (§4); headline form Remark 4.8, Eq. (4.5); Intro & Conclusion state Õ(d̃√T); abstract states Õ(√T) with d̃ suppressed | R2 §4 | ✅ |
| C15 | Effective dimension d̃ = log det(I + H/λ) / log(1 + TK/λ) | R2 Def. 4.3, Eq. (4.1) | ✅ |
| C16 | Real-data experiments approximate Z_t by its diagonal for computational cost; synthetic experiments do not | R2 §7.2 (vs. §7.1) | ✅ |
| C17 | MIND: 1,000,000 users, 161,013 news, 20 categories, 15,777,377 impressions, 24,155,470 clicks | R7 Table 2, p. 3600 | ✅ |
| C18 | MIND collected from Microsoft News logs over 6 weeks, Oct 12 – Nov 22, 2019; users with ≥5 clicks | R7 §3.1, p. 3599 | ✅ |
| C19 | MIND splits: training = 5th week (validation = last day of 5th week), test = last week; click history from earlier weeks; only samples with non-empty history kept | R7 §3.1, p. 3600 | ✅ |
| C20 | MIND-small = random sample of 50,000 users and their behavior logs from MIND | msnews.github.io | ✅ (one source) |
| C21 | MIND-small: train 156,965 impressions (2019-11-09 → 11-14), dev 73,152 (2019-11-15); 0 duplicates across files; 65,238 news | counted from the files, `scripts/mind_facts.py`, 2026-11-16 (`Recommenders/MIND` copy) | ✅ |
| C22 | MIND-small CTR = 4.05% (clicked / shown articles) | counted from the files, 2026-11-16 | ✅ |
| C27 | MIND-large (train + dev): train 2,232,748 impressions (Nov 9–14), dev 376,471 (Nov 15), 750,434 users, 104,151 news, CTR 4.06%; 6 impressions appear in both files; all 94,057 MIND-small users are in MIND-large | counted from the files, `scripts/mind_facts.py --size large`, 2026-11-16 | ✅ |
| C28 | The open `Recommenders/MIND` copy is content-identical to the official gated copy linked from msnews.github.io: SHA-256 of every file inside MINDsmall_train, MINDsmall_dev, MINDlarge_train, MINDlarge_dev matches (behaviors.tsv, news.tsv, entity/relation embeddings). Official large zips add an empty 0-byte `__placeholder__` file; zip sizes differ only through re-compression | official zips downloaded by the user, compared 2026-11-16 | ✅ |
| C25 | ⚠️ MIND-small: **50,000 users in each file**, 94,057 distinct overall (5,943 in both) — the site says "50,000 users" (C20) | counted 2026-11-16 | discrepancy, flagged to user |
| C26 | ⚠️ MIND-small has **5,452 impressions with an empty history**, although the MIND paper says only non-empty-history samples were kept (C19, full MIND) | counted 2026-11-16 | discrepancy, flagged to user |
| C24 | LinUCB's Yahoo! news experiment: raw features normalised to unit length and "augmented … with a constant feature of value 1"; the final user feature is 5 cluster memberships "normalized so that they sum up to unity" **plus** "a constant feature 1" (six-vector) | R3 §5 (feature construction), p. 6 (arXiv 1003.0146v1, read 2026-11-16) | ✅ |
| C23 | LinUCB is defined for a per-trial arm set A_t ("observes the current user u_t and a set A_t of arms"); motivation: "dynamically changing pools of content"; LinUCB "works well for a dynamic arm set" (editors add/remove articles); with fixed A_t of K arms the bound is Õ(√(KdT)) | R3 abstract p. 1; §2.1 p. 2; §3.1 p. 3 (arXiv 1003.0146v1, read 2026-11-16) | ✅ |

---

## Part 3 — Corrections triggered by this audit
- README stated NeuralUCB's guarantee as Õ(√T); the theorem form is Õ(d̃√T) (C14). → fix in README.
- README stated "MIND-small (~230k impressions)"; not verified (C21). → mark `[UNVERIFIED]` until counted.
- README cited the base paper as "ICONIP 2014" without the arXiv v1 we used. → cite as R1 decision above.
- README reference list was written from memory; all entries now checked above (R8 still ⏳).
