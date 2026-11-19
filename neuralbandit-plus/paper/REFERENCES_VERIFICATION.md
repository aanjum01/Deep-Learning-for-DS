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
| Re-verified (Phase 6a) | 2026-11-18: title, 3 authors, v1 11 Nov 2019 / v3 2 Jul 2020 and comment "27 pages … In ICML 2020" from the **arXiv abs page**; title, authors, pp. 11492–11502, ICML, PMLR 119 from the **PMLR page** (proceedings.mlr.press/v119/zhou20a.html). The two user links (arxiv.org/pdf/1911.04462 and /abs/1911.04462) are the **same arXiv record** = one source. DBLP blocks scripts (anti-bot page) — not used. | arXiv abs; PMLR |
| Read | arXiv v3 PDF, 27 pp. (main text pp. 1–9, App. E NeuralUCB₀ pp. 26–27), read in full for §2–§7, 2026-11-18. The PMLR PDF (11 pp., main text) was checked against v3: identical on every setting used here (init, Alg. 1–2, §7.1–7.2 settings, the Allesiardo comparison) | arXiv PDF; PMLR PDF |
| Authors' code | github.com/uclaml/NeuralUCB — account of the UCLA lab of Prof. Quanquan Gu (GitHub profile), README: "our pytorch implementation of NeuralUCB in the paper"; one commit, 4 Jul 2020; **no license** (GitHub API: none) → reference only, no code copied; differences from the paper: C40 | GitHub, read 2026-11-18 |

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

### R8 `bifet2007learning` — ADWIN ✅ (second source found 2026-11-18)
| Field | Value | Source |
|---|---|---|
| Title | Learning from Time-Changing Data with Adaptive Windowing | Crossref |
| Authors | Albert Bifet, Ricard Gavaldà | Crossref |
| Venue | Proceedings of the 2007 SIAM International Conference on Data Mining (SDM) | Crossref |
| Pages / DOI | 443–448 / 10.1137/1.9781611972771.42 | Crossref |
| Second source | OpenAlex agrees on title, authors, venue, pp. 443–448, 2007 (checked 2026-11-18). SIAM epubs is still a bot page | OpenAlex |
| Third source | SIAM's own citation export (.bib, supplied by the user 2026-11-18): same title, authors, booktitle "Proceedings of the 2007 SIAM International Conference on Data Mining (SDM)", pp. 443–448, DOI. The abstract names the algorithm **ADWIN2** (the time- and memory-efficient version) and tests it with Naïve Bayes | SIAM epubs |
| Read | Author-hosted preprint (cs.upc.edu/~gavalda/papers/adwin06.pdf, dated 17 Oct 2006, ≈ 17 pp.; longer than the 6-page SDM paper). Its §/page numbers may differ from SDM: [UNVERIFIED] for the SDM version | |

### R9 `page1954continuous` — Page-Hinkley test origin ✅
| Field | Value | Source |
|---|---|---|
| Title | Continuous Inspection Schemes | Crossref (OUP record); Crossref (JSTOR record) |
| Author | E. S. Page | both records |
| Venue | Biometrika, 41(1–2) | both records |
| Pages / Year | 100–115 / 1954 | OUP record (JSTOR record gives start page 100 only) |
| DOI | 10.1093/biomet/41.1-2.100 (OUP); also 10.2307/2333009 (JSTOR) | Crossref |
| Verified | 2026-11-12 | |
| Read | **Published version read in full, 2026-11-18** (JSTOR PDF, stable URL jstor.org/stable/2333009, downloaded by the user via Columbia; 16 pp. + an appendix table on p. 17 that is a scanned image and could not be read: [UNVERIFIED]). JSTOR's cover page also gives the month: **June 1954** | JSTOR PDF |
| ⚠️ Note | Page 1954 defines the CUSUM scheme (cumulative scores, act when S_n − min S_i ≥ h). The "Page-Hinkley" form with deviations from a **running mean** and a tolerance δ is **not** in this paper; we take it from Gama et al. 2014 (§3.2.1, App. A Alg. 1), who cite Page 1954 and Mouss et al. 2004 for it (Mouss not read). Cite PH as [gama2014] with [page1954] for the CUSUM origin | |

### R10 `shaker2015recovery` — Recovery analysis (adaptation-time measure) ✅
| Field | Value | Source |
|---|---|---|
| Title | Recovery analysis for adaptive learning from non-stationary data streams: Experimental design and case study | Crossref; OpenAlex; authors' draft PDF (title page) |
| Authors | Ammar Shaker, Eyke Hüllermeier | Crossref; OpenAlex; draft PDF |
| Venue | Neurocomputing, vol. 150 | Crossref; OpenAlex; draft PDF ("to be published in Neurocomputing") |
| Pages / Year | 250–264 / 2015 (volume issued Feb 2015) | Crossref; OpenAlex (pages) |
| DOI | 10.1016/j.neucom.2014.09.076 | Crossref; OpenAlex |
| Earlier version | CORES 2013 (Adv. Intell. Syst. Comput. 226), pp. 289–298, DOI 10.1007/978-3-319-00969-8_28 — "Recovery Analysis for Adaptive Learning from Non-stationary Data Streams" | Crossref |
| Discrepancy | OpenAlex gives publication date 2014-10-18 (online first); Crossref's issued date is 2015-02 (volume). Cite **2015** (volume year) | |
| Read | authors' draft PDF, cs.uni-paderborn.de/…/Recovery-draft.pdf, 35 pp., read 2026-11-17 (§3, §5) | |
| Not reachable automatically | ScienceDirect (403), Paderborn RIS (connection refused), DBLP (blocks scripts) | |
| Verified | 2026-11-17 | |

### R11 `garivier2011upper` — Switching bandits: sliding-window / discounted UCB ✅ (paper not yet read)
| Field | Value | Source |
|---|---|---|
| Title | On Upper-Confidence Bound Policies for Switching Bandit Problems | Crossref; OpenAlex |
| Authors | Aurélien Garivier, Eric Moulines (OpenAlex: "Éric") | Crossref; OpenAlex; arXiv |
| Venue | Algorithmic Learning Theory (ALT 2011), Lecture Notes in Computer Science | Crossref; OpenAlex |
| Pages / Year | 174–188 / 2011 | Crossref; OpenAlex |
| DOI | 10.1007/978-3-642-24412-4_16 | Crossref; OpenAlex |
| LNCS volume | **6925** (LoC also lists it as LNAI). Not in Crossref, OpenAlex or the Springer export | Library of Congress record (LCCN 2011936965); Open Library (ISBN 9783642244117), checked 2026-11-18 |
| Preprint | arXiv:0805.3415 [math.ST], v1 22 May 2008, 24 pp., titled "On Upper-Confidence Bound Policies for **Non-Stationary** Bandit Problems", same authors, describes discounted and sliding-window UCB — title differs; that it is the same work as the ALT paper is **not confirmed** | arXiv abs page |
| Verified | 2026-11-17 metadata; 2026-11-18 LNCS volume. Content read from **arXiv v1 only**: very likely an earlier version of the ALT paper (same contributions in the abstract), but the ALT full text was not reachable. Theorem/remark numbers are from v1; that they match ALT is [UNVERIFIED] | |

### R12 `cheung2019learning` — SW-LinUCB + Bandit-over-Bandit ✅ (⚠️ see notes)
| Field | Value | Source |
|---|---|---|
| Title | Learning to Optimize under Non-Stationarity | PMLR page; arXiv 1810.03024 |
| Authors | Wang Chi Cheung, David Simchi-Levi, Ruihao Zhu | PMLR page; arXiv |
| Venue | Proc. 22nd AISTATS, PMLR vol. 89, published 2019-04-11 | PMLR page; arXiv journal_ref (AISTATS 2019) |
| Pages | 1079–1087 (**PMLR page only**; Crossref has no record) | PMLR |
| Journal version | arXiv 1903.01461 "Hedging the Drift: Learning to Optimize under Non-Stationarity" (same three authors, posted 2019-03-04) says it is the journal version of this AISTATS paper and fixes a proof error. Not read; we cite the AISTATS version | arXiv API, 2026-11-18 |
| ⚠️ | OpenAlex merges this arXiv id with a different work ("Hedging the Drift…", arXiv 1903.01461): **do not use OpenAlex here**. arXiv v6 (2021) fixes an error in the proof of Lemma 1 | |
| Read | PMLR PDF (PDF p. n = proceedings p. 1078 + n), 2026-11-18 | |

### R13 `gama2014survey` — Concept-drift adaptation survey ✅
| Field | Value | Source |
|---|---|---|
| Title | A survey on concept drift adaptation | Crossref; OpenAlex |
| Authors | João Gama, Indrė Žliobaitė, Albert Bifet, Mykola Pechenizkiy, Abdelhamid Bouchachia | Crossref; OpenAlex |
| Venue | ACM Computing Surveys 46(4), pp. 1–37, 2014; DOI 10.1145/2523813 | Crossref; OpenAlex |
| ⚠️ | Crossref: online 2014-03, print 2014-04. Article number not in Crossref: [UNVERIFIED] | |
| Read in full (2026-11-18) | PDF supplied by the user: an **author manuscript in ACM's template** with placeholder metadata ("Vol. 1, No. 1, Article 1", DOI 10.1145/0000000.0000000, "January 2013", 44 pp. incl. appendix; pdfTeX 2013-09-09; cites grant ICT-2013-612944, so likely the revised/accepted version). **Not ACM's typeset version**: cite sections, not pages; bibliographic fields come from Crossref/OpenAlex above |
| Read | Bournemouth eprints **submitted version** (header "Vol. 1, No. 1, Article 1, January 2013, 35 pages"): **page numbers differ from the published version**. Cite sections, not pages | |

### R14 `dohare2024loss` — Loss of plasticity ✅
| Field | Value | Source |
|---|---|---|
| Title | Loss of plasticity in deep continual learning | Crossref; OpenAlex; PDF |
| Authors | Shibhansh Dohare, J. Fernando Hernandez-Garcia, Qingfeng Lan, Parash Rahman, A. Rupam Mahmood, Richard S. Sutton | Crossref; OpenAlex; PDF |
| Venue | Nature 632(8026): 768–774; online 2024-08-21; DOI 10.1038/s41586-024-07711-7 | Crossref; OpenAlex; PDF header |
| Read | Full published PDF (main text, Methods, Extended Discussion, ED captions), 2026-11-18 | |

### R15 `ash2020warm` — Shrink and Perturb ✅
| Field | Value | Source |
|---|---|---|
| Title | On Warm-Starting Neural Network Training | arXiv API (1910.08475); NeurIPS page metadata; PDF |
| Authors | Jordan T. Ash, Ryan P. Adams | arXiv API; NeurIPS metadata; OpenAlex ("Ryan Prescott Adams") |
| Venue | Advances in Neural Information Processing Systems 33 (NeurIPS 2020), pp. 3884–3894 | NeurIPS metadata; OpenAlex (pages); arXiv journal_ref |
| ⚠️ | OpenAlex gives 2019 (the arXiv date); the venue is 2020. Crossref has no usable record | |
| Read | NeurIPS published paper (11 pp.), in full, 2026-11-18; appendix from arXiv v3 (pp. 12–21; figures are images, only their captions read) | |

### R16 `kumar2024maintaining` — L2 Init (regenerative regularisation) ✅
| Field | Value | Source |
|---|---|---|
| Title | Maintaining Plasticity in Continual Learning via Regenerative Regularization | arXiv API (2308.11958); PMLR page; PDF |
| Authors | Saurabh Kumar*, Henrik Marklund* (equal contribution), Benjamin Van Roy | arXiv API; PMLR page; OpenAlex; PDF |
| Venue | Proc. 3rd Conference on Lifelong Learning Agents (CoLLAs 2024), PMLR vol. 274, pp. 410–430 (pages: PMLR page only) | PMLR page; PDF header |
| ⚠️ | OpenAlex lists only the arXiv version; PMLR published 17 Feb 2025; arXiv v1 23 Aug 2023, v3 24 Oct 2024 | |
| Read | PMLR PDF in full, 2026-11-18 (text identical to arXiv v3 apart from the arXiv stamp) | |

---

## Part 2 — Claims about other papers (exact locations)

> ⚠️ **2026-11-18 audit (user rule: read the full published paper, never just the abstract or a preprint):** C41, C42 re-checked on 2026-11-18 against the authors' 2011 manuscript (full read; accepted by the user Nov 18). C48 rests on a preprint (Bifet & Gavaldà 2006) ; C45, C46 re-checked by a full read of Gama et al.'s accepted manuscript (2026-11-18). These are **[UNVERIFIED] for the published versions** until the user downloads them. C43, C44, C47, C49: published PDFs read in full on 2026-11-18 (C43 corrected).

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
| C29 | Recovery analysis: semi-synthetic drift on real data; mixture stream with sigmoid λ(t) = (1+exp(4(t−t₀)/w))⁻¹; speeds slow θ=π/75, modest π/30, sudden π/2 ("concept shift"); reference curves α(t), β(t) = same learner on the pure streams; test-then-train evaluation, averaged over repeated runs | R10 §3.1 Eq. 1–2, §3.5, §5 (draft pp. 5–6, 9–10, 12) | ✅ |
| C30 | "Duration" = (t₂ − t₁)/T, t₁ = when the drifting curve drops below 95% of the pre-change curve, t₂ = when it recovers to 95% of the post-change curve; "maximum performance loss" = max_t (S(t) − S_C(t))/S(t), S = min(S_A, S_B) | R10 §3.3, Eq. 8 (draft p. 8) | ✅ |
| C31 | Drift in multiclass data simulated by "a cyclic shift of the class labels, replacing label i by 1 + i mod 5" (Page Blocks; same for Letter Recognition) | R10 §5.2 (draft p. 14) | ✅ |
| C23 | LinUCB is defined for a per-trial arm set A_t ("observes the current user u_t and a set A_t of arms"); motivation: "dynamically changing pools of content"; LinUCB "works well for a dynamic arm set" (editors add/remove articles); with fixed A_t of K arms the bound is Õ(√(KdT)) | R3 abstract p. 1; §2.1 p. 2; §3.1 p. 3 (arXiv 1003.0146v1, read 2026-11-16) | ✅ |
| C32 | **NeuralUCB's guarantee assumes a stationary reward function:** r_{t,a_t} = h(x_{t,a_t}) + ξ_t with **one fixed unknown h** (0 ≤ h ≤ 1) for every round, ξ_t ν-sub-Gaussian with mean 0 given the past (Eq. 2.2); horizon T known; regret vs a*_t = argmax_a E[r_{t,a}] (Eq. 2.1). Contexts may be any sequence (no i.i.d. assumption), but the context→reward map never changes. The paper never discusses drift or non-stationarity (full-text search of all 27 pp.: no "stationar", "drift", "adversar") | R2 §2, p. 2 | ✅ |
| C33 | Theorem 4.5 also needs: Assumption 4.2 (NTK matrix H ⪰ λ₀I over all TK contexts; ‖x‖₂ = 1; [x]_j = [x]_{j+d/2}), width m ≥ poly(T, L, K, λ⁻¹, λ₀⁻¹, S⁻¹, log 1/δ) (Eq. 4.2), η = C₁(mTL + mλ)⁻¹, λ ≥ max{1, S⁻²}, S ≥ √(2hᵀH⁻¹h); footnote 1: experiments use m "relatively large, but not as large as theory suggests" | R2 §4, p. 4; §7.1 footnote, p. 7 | ✅ |
| C34 | Network (Eq. 2.3): f(x; θ) = √m W_L σ(… σ(W₁x)), ReLU, no biases, p = m + md + m²(L−1). Initialisation: W_l = diag(W, W), entries of W ~ N(0, 4/m); W_L = (wᵀ, −wᵀ), w ~ N(0, 2/m); with Assumption 4.2's duplicated input this gives f(x; θ₀) = 0 | R2 §2–3, p. 2; §4, p. 4 | ✅ |
| C35 | Algorithm 1: Z₀ = λI; U_{t,a} = f(x_{t,a}; θ_{t−1}) + γ_{t−1} √(gᵀZ_{t−1}⁻¹g / m); play argmax; Z_t = Z_{t−1} + g gᵀ/m (played arm, at θ_{t−1}); θ_t = TrainNN on **all** t past (x, r), started from θ₀. Algorithm 2: J gradient-descent steps of size η on L(θ) = Σᵢ (f(xᵢ; θ) − rᵢ)²/2 + mλ‖θ − θ₀‖²/2 | R2 Alg. 1–2, p. 3 | ✅ |
| C36 | Synthetic experiments: **γ_t = γ held constant**, grid {0.01, 0.1, 1, 10}; two-layer net, m = 20; TrainNN every 50 rounds, SGD batch 50, J = t, η ∈ {0.001, 0.01, 0.1}; 10 repetitions | R2 §7.1, p. 7–8 | ✅ |
| C37 | Real-data experiments: covertype (54 features, 7 classes, 581,012 rows), magic, statlog, mnist; **T = 15,000**; disjoint model x⁽ᵃ⁾ = (0, …, x, …, 0) ∈ R^{dk} [li2010]; regret 0 if correct class else 1; two-layer, **m = 100**; **Z_t approximated by its diagonal** (cost); TrainNN **every 100 rounds starting from round 2000**; λ ∈ {10⁻¹, 10⁻², 10⁻³, 10⁻⁴}; η ∈ {2·10⁻ⁱ, 5·10⁻ⁱ}, i = 1…4; **J = 1000, SGD batch 500**; "rest of parameters" (γ) tuned as §7.1; best grid setting reported; 10 reshuffled runs. Text does not say whether contexts were normalised. ⚠️ **Ambiguity:** "every 100 rounds starting from round 2000" read literally = no training before round 2000, while Alg. 1 trains every round (the authors' code trains every round before 2000, C40). The experiments' net has p = md + m (§7.1, p. 7) — **no input duplication** in the experiments (Assumption 4.2's duplication is theory only, p. 4) | R2 §7.2, p. 8; §7.1, p. 7 | ✅ |
| C38 | Results: NeuralUCB has the lowest regret on the real datasets incl. covertype (Fig. 2, read qualitatively); "LinUCB fails to learn them for nearly all tasks"; Neural ε-greedy lies between; BootstrappedNN "can be competitive" | R2 §7.3, p. 9; Fig. 2, p. 8 | ✅ |
| C39 | The paper contrasts itself with Allesiardo et al. (2014): NeuralBandit = K networks + ε-greedy; NeuralUCB = one network + UCB, "more effective than ϵ-greedy", "can be computationally more efficient". NeuralBandit is **not** among its experimental baselines (§7) | R2 §3, p. 3; §7 p. 7 | ✅ |
| C41 | SW-UCB uses "a local empirical average of the observed rewards, using only the τ last plays": mean over s = t−τ+1…t plus padding c_t(τ,i) = B√(ξ log(t∧τ)/N_t(τ,i)) (D-UCB has 2B); Thm 2 needs ξ > 1/2 | R11 author manuscript (2011-05-17, 15 pp.) §2, Alg. 2, p. 3; Thm 2 Eq. 4, p. 4 | ✅ full read 2026-11-18 (author manuscript; accepted by the user Nov 18 as the published version) |
| C42 | With T and Υ_T known in advance: τ = 2B√(T log T/Υ_T) → O(√(Υ_T T log T)); "matches the lower-bound of Theorem 3 up to a factor √log T". The lower bound (Thm 3, Cor. 1) is Ω(√T), from a two-breakpoint construction, so the match is in T only. With Υ_T ≤ rT, τ stays bounded: τ = 2B√(−log r / r). No breakpoints → τ = T (= UCB, Remark 1). Window tuning is left open ("on-going research", §8). Experiments: one τ only (4√(T log T), not the §3.1 formula), no window sweep; D-UCB ≈ SW-UCB; both beat UCB-1 and EXP3.S | R11 manuscript §3.1 p. 5; Thm 2 p. 4; §5 Thm 3/Cor. 1 pp. 9–10; Remarks 1–2 p. 5; §6 Figs 1–2 pp. 11–12; §8 p. 14 | ✅ full read 2026-11-18 (author manuscript; accepted by the user Nov 18) |
| C43 | SW-UCB (the paper's name; "SW-LinUCB" is ours): ridge regression on rounds 1∨(t−w)…t−1 only; regret Õ(w·B_T + d·T/√w). Known B_T: w = ⌊d^{2/3}T^{2/3}B_T^{−2/3}⌋ → Õ(d^{2/3}B_T^{1/3}T^{2/3}), matching the lower bound (Thm 1). Unknown B_T: w = ⌊(dT)^{2/3}⌋ → **Õ(d^{2/3}(B_T+1)T^{2/3})**, linear in B_T and meaningless when B_T = Ω(T^{1/3}) | R12 §4 Eq. 2–3 p. 3; Eq. 11 p. 4; §5.1 p. 4; Thm 3 + §6 p. 5 (PDF pages; the PDF prints no proceedings page numbers) | ✅ full read 2026-11-18 (corrected: unknown-B_T order) |
| C44 | Bandit-over-Bandit: blocks of length H = ⌊d^{2/3}T^{1/2}⌋; EXP3 picks each block's window from a geometric grid J = {1, ⌊H^{1/Δ}⌋, …, H}; a **fresh SW-UCB restarts every block**, and the rescaled block reward is EXP3's feedback; nothing depends on B_T; Õ(d^{2/3}(B_T+1)^{1/4}T^{3/4}). Setting: linear bandit, θ_t drifts adversarially with total variation ≤ B_T (no assumption of gradual vs abrupt). Experiments: synthetic only (2 arms, sinusoidal θ_t); **no window-size sweep**; BOB beat the fixed B_T-agnostic window | R12 §6.2 Eq. 13–22 pp. 5–7, Alg. 2 + Thm 4 pp. 7–8; §2.2 Eq. 1 p. 2; §7 Figs 1–2 p. 8 | ✅ full read 2026-11-18 |
| C45 | Window trade-off: "A short window reflects the current distribution more accurately"; it adapts fast during change but "worsens the performance" in stable periods; a long window is the reverse. A fixed-size FIFO window ("the oldest one is discarded") "is often used as a baseline". Window size is "The key challenge"; variable windows shrink when a detector signals change. Abrupt forgetting (windows) vs gradual forgetting (weighting by age, linear or exponential decay): "The more abrupt is forgetting, the faster the reactivity, but also the higher the risk of capturing noise"; no empirical comparison | R13 §3.1.1 p. 1:12; §3.1.2 pp. 1:12–13 (manuscript pages) | ✅ full read 2026-11-18 (author manuscript; accepted by the user Nov 18) |
| C46 | Retraining "discards the current model and builds a new model from scratch using buffered data"; "They fall into two major types: blind and informed": blind methods "periodically retrain the model with the latest w examples" and "forget old concepts at a constant speed independently of whether changes are happening or not" (main limitation: slow reaction); informed methods act when a trigger (change detector) fires; Global Replacement: "The full model is deleted and a new model is started from scratch". Drift types: sudden/abrupt, incremental, gradual, reoccurring (§2.2) | R13 §3.3.1 p. 1:19; §3.3.2 pp. 1:20–21; §2.2 pp. 1:5–6 (manuscript pages) | ✅ full read 2026-11-18 (author manuscript; accepted by the user Nov 18) |
| C47 | Deep nets trained continually lose plasticity "until they learn no better than a shallow network"; continual backprop reinitialises "a tiny proportion of less-used units" each step (in practice < 1 unit per step), chosen by contribution utility after a maturity threshold; in RL used together with L2 | R14 abstract p. 768; Discussion p. 773; Methods Alg. 1 (PDF p. 8, unnumbered) | ✅ full read 2026-11-18 |
| C49 | Settings: continual ImageNet (CNN), class-incremental CIFAR-100 (ResNet-18), Permuted MNIST (3 × 2,000 units, also 100–10,000), Slowly-Changing Regression (**1 hidden layer × 5 units**), RL ant (PPO 2 × 256; friction changes every 2M steps). **Small nets lose plasticity too**: the 5-unit learner's error rises to the linear baseline; in Permuted MNIST the loss is "most pronounced at the smaller network sizes". No single-hidden-unit net is tested. Full reinitialisation is a **comparison baseline, never their method**: in Permuted MNIST it stays at first-task level and continual backprop beats it; on ImageNet continual backprop ends slightly better and is ≈ 10× faster to its best accuracy; in RL it is practical only with large buffers. CIFAR: incremental training 5% below retraining from scratch after 100 classes; the loss starts after 40 classes (8 increments); ant: PPO degrades after the first change. "May not occur at all for small numbers of tasks" | R14 pp. 768–773; Methods/Extended Discussion PDF pp. 8–15; ED Tables 3–4 (PDF pp. 24–25) | ✅ full read 2026-11-18. Issue no. 8026 is not printed on the PDF |
| C50 | L2 Init: L_reg(θ) = L_train(θ) + λ‖θ − θ₀‖² with θ₀ = the actual initial draw, applied at every gradient step, to all parameters (no layer-specific variant); an "implicit, smooth reset". With SGD it is θ_{t+1} = (1 − αλ)θ_t + αλθ₀ − α∇L, which is **our θ ← θ − ρ(θ − θ₀) with ρ = αλ** | R16 §3 p. 4; App. A.3 pp. 18–19 | ✅ full read |
| C51 | L2 Init tuning: λ ∈ {1e-2, 1e-3, 1e-4, 1e-5}, 3 seeds, picked by total average online accuracy, then re-run on 10 seeds; problem-sensitive, mostly 1e-2 chosen (1e-3 Continual ImageNet; 1e-5 on 5+1 CIFAR with SGD). Small nets on purpose (MLP 2 × 100 ReLU) to study plasticity loss "exacerbated due to capacity constraints" | R16 §5 p. 7; App. A.1.3 p. 12; A.2.5 Fig. 12; Tables 3–4 | ✅ |
| C52 | Results: L2 Init is top-3 on all five problems and matches Continual Backprop on 4 of 5; plain L2 (toward 0) fails on the random-label problems and 5+1 CIFAR; Shrink & Perturb mitigates plasticity loss but is "consistently lower" than L2 Init; regularising toward **freshly resampled** weights is much worse than toward the fixed θ₀ | R16 §5.1 Figs 1–2; §5.3 | ✅ |
| C53 | Shrink and Perturb: θ ← λθ + p, p ~ N(0, σ²), 0 < λ < 1, shrinking toward **zero**, applied once per data arrival before retraining (not every step); in practice the noise is a scaled fresh initialisation; fixes worse generalisation of warm-started nets (ResNet-18 CIFAR-10 51.7 vs 56.2); applied every SGD step it acts like "an aggressive, noisy L2 regularization" | R15 §4 p. 5–6; §2 Table 1 pp. 2–3; §4.1 p. 7 | ✅ |
| C54 | Dohare et al.'s baselines: L2 moves weights toward zero "at each step"; S&P = L2 plus small random changes at each step; both maintain plasticity on Continual ImageNet; on Permuted MNIST S&P "almost completely" mitigates plasticity loss but is sensitive to the noise SD, and both work only in a narrow hyperparameter range; no formulas given | R14 p. 772; Methods (unnumbered) | ✅ (grid values in ED Table 2 are an image: [UNVERIFIED]) |
| C55 | Page's sequential CUSUM (Rule 2): assign a score x_k to each observation, S_n = Σ x_k, take action when S_n − min_{0≤i<n} S_i ≥ h. Scores are chosen so the mean path goes **down** when quality is satisfactory and **up** when it is not; the rule is a chain of Wald tests with boundaries (0, h), restarted from 0 | R9 §2.2 pp. 103–104, Eqs. (5)–(7) | ✅ full read 2026-11-18 |
| C56 | One-sided schemes detect deviations in one direction only; two directions need two one-sided schemes run together (Rule 4) | R9 §2.1 p. 102; §3.1 p. 106 | ✅ |
| C57 | False alarms are unavoidable: "there is a probability of one that some point will eventually fall outside the limits … even when there is no change". The **average run length (A.R.L.)** on good quality measures the cost of false alarms; on bad quality it measures the detection delay | R9 §1.2 p. 101 (Definition) | ✅ |
| C58 | Choosing a scheme: e.g. a specified A.R.L. on good quality (rare stoppages) with the smallest A.R.L. on bad quality, or the reverse, depending on which matters more | R9 §5.2 p. 109 | ✅ |
| C48 | ADWIN: the window grows while the data look stationary; when two large enough sub-windows' means differ by more than ε_cut, "the older portion of the window is dropped" | R8 author preprint §3.2 p. 3, Thm 3.1, Fig. 1 | ⏳ published version not read |
| C40 | **Authors' code (uclaml/NeuralUCB) differs from the paper:** PyTorch default init with biases, no √m factor (vs C34); loss (f − r)² + `weight_decay = λ` toward 0 (vs mλ‖θ − θ₀‖²); SGD batch 1, lr 0.01, ≤ 1000 steps per call, stops early when an epoch's mean loss ≤ 1e-3 (vs batch 500, J = 1000); trains every round for t < 2000, then every 100 rounds — **and after round 2000 only every 100th observation is stored** (vs all past data, C35); bonus √(λν Σ g²/U) with U = λ + Σ g² (no 1/m) (vs C35); covertype from OpenML, rows L2-normalised; disjoint input of size d·K | `learner_diag.py`, `train.py`, `data_multi.py` (commit 9a74044, read 2026-11-18) | ✅ (code facts) |

---

## Part 3 — Corrections triggered by this audit
- README stated NeuralUCB's guarantee as Õ(√T); the theorem form is Õ(d̃√T) (C14). → fix in README.
- README stated "MIND-small (~230k impressions)"; not verified (C21). → mark `[UNVERIFIED]` until counted.
- README cited the base paper as "ICONIP 2014" without the arXiv v1 we used. → cite as R1 decision above.
- README reference list was written from memory; all entries now checked above (R8 still ⏳).
