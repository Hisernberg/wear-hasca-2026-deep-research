# 🏆 WEAR @HASCA 2026 — Deep Research, Ecosystem Analysis & Improvement Plan

> **3rd WEAR Dataset Challenge @HASCA 2026** (Kaggle) — multimodal Human Activity Recognition:
> 1-second inertial windows (50 Hz, one random limb per window) + VideoMAEv2-Base egocentric video features
> → 19-class activity label → **macro F1**.
>
> This repo is a **full-stack research dossier**: analysis of our own submissions, every public notebook,
> all competition discussions, prior WEAR/HAR challenges, cross-competition engineering techniques
> (timeline reconstruction, Hungarian linking, count calibration, greedy anchored voting, category alignment),
> a complete *what-hurts / what-improves* matrix, bugs & lessons, and a prioritized improvement plan
> covering **algorithmic, non-LLM, LLM-enabled and engineering** tracks.

---

## TL;DR — The 12 findings that matter

| # | Finding | Evidence |
|---|---------|----------|
| 1 | **Sequence reconstruction is the dominant lever** — reconstructing the shuffled test windows into timelines via learned successor links + Hungarian 1:1 assignment + cycle breaking is worth **≈ +0.07 LB by itself** (0.8365 → 0.9077) and **+0.18–0.20 total** when combined with count calibration | yeashusemwal LB log; woominyo/udaken10 stack OOF ladder 0.725 → 0.905 |
| 2 | **Per-subject count calibration** (sharpen T=0.5 → Sinkhorn to ≈97 tiles per exercise with null floor 5%) is the second pillar — it fixes the null-vs-exercise painting pathology that naive chaining creates | all 0.90+ solutions; akhyar's failed pre-Sinkhorn attempts (LB 0.72–0.75) |
| 3 | **Window-level accuracy is NOT the bottleneck.** Best window models cap at OOF macro-F1 ≈ 0.725; post-processing lifts this to ≈ 0.93. Adding window-model capacity is nearly dead weight vs. upgrading links/calibration | goodpjw2008 prints (0.7253 → 0.9300 OOF) |
| 4 | **The exact-successor recall of the link model (66%) is the ceiling** — every +1% of exact-successor accuracy tracks ≈ +0.01–0.02 OOF F1. This is where the next wins live | goodpjw2008 L0→L3 progression (32.9% → 66%) |
| 5 | **A third of all errors lie within 1 s of a true activity boundary** — boundary refiners (LightGBM flip classifiers over 8 timeline hypotheses) recovered +0.003 OOF; boundary localization is the main open problem | yeashusemwal; goodpjw2008 refiner |
| 6 | **Learned per-(subject,class) count priors** (regress expected tile counts from sorted log-margin profiles) beat the fixed-97 assumption: +0.006 OOF; oracle counts → +0.013 | goodpjw2008; honghanhhh 3-pass variant |
| 7 | **The video–inertial "oracle late fusion" gap is huge and unclaimed** — WEAR paper: oracle late fusion hits 91–94 F1 vs ~75 realizable. Confidence-gated late fusion / greedy anchored voting is the biggest theoretical headroom | arXiv:2304.05088 (Bock et al., IMWUT 2024) |
| 8 | **Video features help mainly through geometry, not classification** — their biggest value is kNN-graph embeddings + boundary linking; video towers add little to window logits (IMU-only head: 0.60 vs fusion 0.72) | all top solutions' architecture prints |
| 9 | **Test-like OOF validation is non-negotiable** — non-overlapping 1-s tiles, one random *valid* limb, majority label, subject-wise folds. Solutions with mismatched CV (segment-only windows, round-robin sensors, 1-fold) collapsed to 0.61–0.67 LB | nomannic19, avikdas567, swathia case studies |
| 10 | **Transductive per-subject statistics win** — per-(subject,limb) IMU normalization and per-subject video centering computed label-free on test; global stats under-perform | top-4 solutions |
| 11 | **Cross-competition priors transfer**: SHL winners' location-conditional training (+10 F1) maps directly to per-limb calibration; Child Mind's Viterbi decoding and Ion Switching's HMM emissions-from-confusion-matrix are the proven sequence-decoding recipes | Kalabakov et al. (Sensors 2022); Kaggle CMI/Ion solutions |
| 12 | **Validation honesty compounds** — a leaked count model (through a second calibration pass) and a dev-vs-LB divergence (+0.0030 dev vs +0.0001 LB whitening) both documented publicly; OOF-gated changes ("if OOF doesn't improve, don't ship") are the standard | honghanhhh; yeashusemwal audit |

---

## The scoreboard (public LB, Oct 2026 snapshot)

| Rank | Team | Public LB | Entries | Approach (where known) |
|---:|---|---:|---:|---|
| 1 | Anonym | **0.94535** | 43 | unknown (private) |
| 2 | ANMOL GARG | 0.94355 | 6 | unknown |
| 3 | Nicolas Krusche | 0.93849 | 60 | unknown |
| 4 | Santiago Maniches | 0.93762 | 161 | unknown |
| 5 | Mateo Allmer | 0.93661 | 68 | unknown |
| 6 | Koushik Rudra | 0.93541 | 150 | unknown |
| 7 | Sameerk | 0.93527 | 45 | unknown |
| … | … | … | … | [full 97-row table](artifacts/leaderboard_public_top97.csv) |
| 11 | goodpjw2008 | 0.92942 | 28 | Learned Links + Counts (open) |
| 12–15 | rupsa roy / Kilian Z. / Akhyar A. / Mathieu W. | 0.92875 | 13–48 | fork cluster incl. Ranked Timelines |
| 19 | Hanh Tran | 0.92718 | 39 | Hungarian Chain Viterbi lineage |
| 29 | Jeki Wan Taufik | 0.92485 | 3 | honghanhhh LB 0.9 fork |
| 44–48 | jiweiliu cluster | 0.90770 | 1–10 | public fast GPU inference (open) |
| **49** | **really? (us — udaken10)** | **0.90759** | 11 | Timeline Reconstruction + Graph (open) |

> ⚠️ **Competition-meta reality check** (from the official discussion threads — see [artifacts/discussions](artifacts/discussions)):
> the **final ranking was frozen on July 5, 2026** (private LB at the technical-report deadline). The current Kaggle LB
> no longer affects prizes. The host explicitly invites **arXiv preprints** (offers endorsement) or long forum
> write-ups as the path to share solutions now. Our improvement plan is therefore framed for: (a) the arXiv
> technical report, (b) the 4th WEAR challenge / future multimodal HAR work.

---

## Repo map

```
docs/
├── 01-my-submission-analysis.md   ← our 6 notebooks, diff-by-diff, what moved/hurt
├── 02-competitor-deep-dive.md     ← all 29 public notebooks dissected (arch/recipe/postproc/scores/bugs)
├── 03-external-research.md        ← WEAR 2024/2025 winners, SHL, CMI Viterbi, LLM-for-HAR, fusion SOTA
├── 04-techniques-matrix.md        ← 45+ techniques: improves ✅ / hurts ❌ / risky ⚠️ with evidence
├── 05-bugs-setbacks-lessons.md    ← the bug hall of fame: official video bug, leaks, fake timelines, …
├── 06-improvement-plan.md         ← Track A algorithmic · Track B LLM · Track C engineering · Track D meta
└── 07-roadmap.md                  ← prioritized 4-week execution plan + validation protocol

artifacts/
├── leaderboard_public_top97.csv   ← full public LB snapshot (rank|team|score|entries)
├── discussions/                   ← all 5 competition threads, full text
└── notebooks-index.md             ← every public notebook + LB + one-line technique summary
```

## How to read this repo in 10 minutes

1. Read the TL;DR table above.
2. Skim [04-techniques-matrix.md](docs/04-techniques-matrix.md) — the one-stop what-works list.
3. Read [06-improvement-plan.md](docs/06-improvement-plan.md) Track A items 1–5.
4. If you want the full story of *why*, dive into 01–03.

## Key numbers cheat-sheet (from our own pipeline logs)

| Stage | Metric | Value |
|---|---|---|
| Window ensemble (4 fusion + 2 IMU) | OOF macro-F1 | 0.7194 (fusion) / 0.5995 (IMU) / **0.7253 (blend)** |
| L0 links (LGBM pair scorer + Hungarian) | linked / exact-successor / same-label | 0.924 / 0.330 / 0.887 |
| Ridge boundary predictors (video→video) | R² | ≈ 0.58 |
| Combined-link scorer | rows / pos-rate / candidate recall | 1,173,696 / 0.0549 / 0.930 |
| Test linking coverage | linked / median raw score | 0.951 / 0.52 |
| Null share per test subject | sbj 22 / 23 / 24 / 25 | 0.66 / 0.43 / 0.15 / 0.09 |
| End-to-end | OOF 0.905 → LB | **0.90315–0.90759** |

## Citation / credits

- WEAR dataset & challenge: Bock et al., [arXiv:2304.05088](https://arxiv.org/abs/2304.05088) (IMWUT 2024); challenge by M. Bock.
- Public notebooks analyzed: goodpjw2008, honghanhhh (+Hanh Tran lineage), yeashusemwal, jiweiliu, woominyo/akhyar2612 ("Briano" base), avikdas567, nomannic19, and the rest — see [notebooks-index](artifacts/notebooks-index.md).
- Prior challenge reports: Signal Sleuths ([arXiv:2408.03947](https://arxiv.org/abs/2408.03947)), 3KA ([arXiv:2511.23173](https://arxiv.org/abs/2511.23173)), HARMA (UbiComp/ISWC '25 adjunct).
