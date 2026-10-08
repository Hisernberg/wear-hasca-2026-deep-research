# 03 — External Research: Prior WEAR Challenges, HAR Competition Playbook, LLM Angles

## 1. Prior WEAR challenges (verified reports)

### 1.1 Format evolution
- **1st WEAR Challenge @HASCA 2024**: inertial-only, all 4 accelerometers available at test; 18 workout activities + null; macro-F1.
- **2nd WEAR Challenge @HASCA 2025** (Kaggle, Apr–Oct 2025): switched to the hard **single random sensor location per test window** setting; 72 entrants / 630 submissions.
- **3rd (this one, 2026)**: first multimodal edition — 50 Hz accel windows + pre-extracted VideoMAEv2 features; 22 train subjects, 4 unseen test subjects.

### 1.2 Team Signal Sleuths — 2024 exemplar report (arXiv:2408.03947)
Van Der Donckt ×2 + Van Hoecke (Ghent). The organizers' cited "example report". Pipeline:
- **Data audit first**: found intra/inter-participant **wearable orientation inconsistencies** (inverted left/right wearables, worn-on-wrong-limb) — this drove everything.
- **Multi-resolution features** over windows {1, 2, 4, 8, 16, 32} s × 14 features per axis (time + frequency, spectral entropy, fractal, Hjorth via `antropy`; extraction via `tsflex`) = 1,992 features.
- **Rotation-invariant axis aggregations** (stat2/stat3/sort) + Signal Magnitude Vector.
- **CatBoost** (depth 5) — a GBDT beat deep models.
- **Augmentation-as-pairing**: left-right swapping (F1 91.30%) and upper-lower limb pairing (91.87%) vs raw 90.01%.
- **Post-processing**: k-fold probability majority voting (+~1 pt), temporal smoothing (label autocorrelation), rule-based activity boosting exploiting the ≥50 s-per-activity protocol (organizer-acknowledged exploit).

### 1.3 Team 3KA — 2025 report (arXiv:2511.23173)
Single random sensor setting: angle + SMV features, statistical + **fractal/spectral** + higher-order differentials; **data-side fusion** of left/right limbs; **sensor rotation + axis-inversion augmentation**; soft voting HistGradientBoosting + XGBoost → **58.83% macro-F1** (CV). ANOVA-F: fractal/spectral features matter for arm data, least for leg — **feature–location interactions are real**.

### 1.4 Team HARMA — 2025 (UbiComp/ISWC '25 adjunct)
"Mitigating Null-Class Dominance": confirms **null dominance** was *the* 2025 difficulty; class-imbalance handling is first-class.

## 2. The WEAR benchmark ceiling (Bock et al.)

- WEAR paper (arXiv:2304.05088, IMWUT 2024): LOSO F1 @1 s — ActionFormer inertial 74.67, camera 66.40, **concat fusion 75.26**. **Oracle late fusion O-LF(I,C) = 91.26; O-LF(I,C,I+C) = 93.99** — the paper's own headline is that modalities have "complementary strengths" and **naive fusion captures almost none of the +16–19 point gap**. This is the single biggest theoretical headroom in the challenge.
- **DeepConvContext** (arXiv:2505.20894, ISWC 2026, same authors): processes **sequences of windows** for inter-window context — +5% F1 avg / +18 mAP across 6 HAR benchmarks. Directly applicable upgrade to our window model (it currently sees one window in isolation).
- **WHAR Arena** (arXiv:2606.13194): 30 datasets × 17 architectures × 4,760 runs — top models cluster within ~1–2 macro-F1; **random forests sit on the practical Pareto frontier**; big recurrent models add cost, not accuracy.
- **Weak-annotation with vision foundation models** (arXiv:2408.05169): VLMs weakly label inertial data — a video→label bridge directly relevant here (we have VideoMAE features, not raw video).

## 3. HAR competition playbook (cross-competition engineering)

### 3.1 SHL Prediction Challenge — the definitive post-mortem (Kalabakov et al., Sensors 2022, PMC9145859)
Jožef Stefan team — won 2018 & 2019, 3rd in 2020:
- **Training-data selection matched to test distribution ("location-specific" training): up to +10 F1** — biggest lever. Maps to WEAR: per-limb conditional training/calibration.
- **HMM temporal smoothing: up to ~+10 points when the data are temporally ordered** — they had to *disable* it in 2020 when the test design scrambled order. Key warning for us: our timeline reconstruction exists precisely because Kaggle windows are shuffled; once we relink, HMM/Viterbi-style constraints become applicable again *on the reconstructed sequences*.
- Separate models for locomotion vs vehicle classes: +1 pt → WEAR analogue: arm-dominant vs leg-dominant specialists.
- Semi-supervised learning on unlabeled test-like data: +1 pt.
- Feature selection & person clustering: inconclusive.

### 3.2 Child Mind Institute — Detect Sleep States (Kaggle 2023–24)
Top solutions = segmentation (U-Net heatmaps over downsampled series) + **Viterbi-style DP decoding** enforcing valid event structure (alternation, min separation, expected densities). 1st place: github.com/sakami0000/child-mind-institute-detect-sleep-states-1st-place. Lesson: **structured decoding of overlapping window probabilities beats peak-picking** — the same logic that makes our Sinkhorn+timeline stack win.

### 3.3 Liverpool Ion Switching (Kaggle 2020)
3rd place (Vandewiele, Vo, Zidmie): **HMM over per-timestep classifier logits** with adjacent-state transition bandwidth, emissions validated by CV LB probing. Canonical "NN/GBM + HMM" recipe. For WEAR: estimate the transition matrix from train label sequences, set emissions from our OOF confusion matrix (SHL recipe) and decode each reconstructed timeline.

### 3.4 Google Brain — Ventilator Pressure (Kaggle 2021) — "category alignment"
Winners exploited deterministic control-input patterns: **align test rows to identical train "categories" and submit aligned train statistics** — the cleanest documented "category alignment" win. WEAR analogue: align test windows to *identical train contexts* (same subject-protocol position, same limb, same bout neighborhood) and use aligned train statistics instead of global ones.

### 3.5 "Greedy anchored voting" — verification & mechanics
The exact phrase has no public Kaggle write-up (it appears once, for LLM inference: run one **greedy (deterministic) pass as the anchor**; sampled passes may override only under consensus). The well-documented analogues:
1. **Anchor-and-override ensembling** (this repo's working definition): designate one high-precision anchor predictor (our inertial+timeline pipeline); a second predictor (video-side) may overturn the anchor **only** when ≥2 independent video-side models agree with margin > τ. Directly targets finding #7 (oracle-LF gap) while protecting precision.
2. **Greedy box/mask consensus** (segmentation comps): seed-consensus merging with greedy order — maps to greedy merging of overlapping *timeline hypotheses* (we already have 8 matchings; add greedy consensus before the geometric mean).
3. **Greedy per-class threshold optimization** (RSNA-style): greedy per-class logit bias — we already ship this (coordinate ascent on OOF); it is exactly "greedy anchored" threshold search per class.

## 4. Multimodal video+IMU fusion SOTA (2024–2026)

- **Concat fusion ≈ +0.5–1 F1; oracle late fusion ≈ +16–19** (WEAR paper). With pre-extracted VideoMAE features, the realistic fusion menu: (i) **per-sample confidence-gated late fusion** (estimate per-modality reliability, gate the vote), (ii) **video→IMU distillation** — **COMODO** (arXiv:2503.07259, IMWUT 2026) does label-free cross-modal distillation with a frozen video encoder + instance queue, matching supervised IMU models, (iii) **gated MoE per modality-reliability**.
- **PIM** (arXiv:2503.17978): physics-informed multi-task SSL pretexts (movement speed, joint angles, inter-sensor symmetry) — up to ~+10 macro-F1 few-label, ~+3% full-data, evaluated on WEAR.
- **Masked Video+IMU autoencoder** (arXiv:2407.06628, ECCV'24 WS); **Cross-modal Transfer Through Time** (arXiv:2407.16803).
- **Test-time adaptation for wearable HAR**: COA-HAR (ESWA 2025); optimization-free TTA (IMWUT, DOI 10.1145/3631450); "Temporal Structure Matters for Efficient TTA" (arXiv:2605.04617). Supports per-window limb detection + limb-conditional calibration at test time.

## 5. LLM / foundation-model angles (2025–2026) — what has real evidence

| Approach | Evidence | Verdict for us |
|---|---|---|
| **SensorLM** (Google, NeurIPS 2025, arXiv:2506.09108) | sensor–language foundation model; zero-shot HAR via text alignment | class-name/text embeddings usable for label-structure priors; zero-shot IMU classification itself is not competitive here |
| **SLIP** (arXiv:2603.11950) | language-informed sensor pretraining | promising; weights/publicity still thin |
| **LLM late fusion** (Demirel et al., OpenReview) | LLM fuses audio+motion *predictions* | directly matches our finding #7: use an LLM (or LLM-style gated voter) as the **decision-level fusion layer** over per-modality probabilities + class semantics |
| Direct LLM classification of serialized IMU (NTU 2026, Claude-on-IMU) | feasible but weak | not competitive as backbone; useful for error analysis only |
| **TSFMs on IMU** (MOMENT/Chronos/Timer; DeepSenseMoE AAAI 2026) | mixed vs tuned GBM/CNN baselines (WHAR Arena agrees) | use as *extra ensemble members / SSL initializers*, never the backbone bet |
| **VLM weak labels** (Bock et al., arXiv:2408.05169) | VFM pseudo-labels inject video info into inertial training | we only have VideoMAE features (not raw video) — a KNN/cluster→label-semantics bridge is the feasible slice |
| LLM as meta-engineer (mining discussions/logs, generating refiner features) | standard 2026 practice | cheap, do it (this repo is partially that) |

## 6. Top-15 transferable techniques (cross-competition ranking)

| # | Technique | Source | Expected gain context | Effort |
|---|---|---|---|---|
| 1 | Confidence-gated late fusion of inertial + video decisions (oracle-LF attack) | WEAR paper O-LF 91–94 vs 75; anchor-override analogue | biggest identified headroom | Med |
| 2 | Per-window limb detection → limb-conditional calibration/training | SHL +10 p.p.; 3KA arm/leg split | top domain-shift lever; we already feed limb embeddings — add limb-conditional priors | Low–Med |
| 3 | HMM/Viterbi decode on *reconstructed* timelines, emissions = own OOF confusion | SHL 2019; Ion Switching; CMI | decisive once sequences are relinked (we have them) | Med |
| 4 | Multi-resolution {1–32 s} features + rotation-invariant aggregation + GBDT | Signal Sleuths (won 2024) | cheap, proven backbone for the inertial side | Low |
| 5 | Sensor rotation + axis-inversion augmentation | Signal Sleuths; 3KA | WEAR has documented orientation inconsistencies | Low |
| 6 | Null-class handling: reweighting + greedy per-class thresholds for macro-F1 | HARMA 2025; RSNA-style | null dominance is measurable in our null-share prints | Low |
| 7 | CV-probability majority voting + temporal smoothing | Signal Sleuths (+1 pt each) | low-risk +1–2 | Low |
| 8 | DeepConvContext-style window-sequence context model | arXiv:2505.20894 (+5% F1) | same authors' architecture for exactly this data | Med |
| 9 | Left/right + upper/lower data-side fusion (train-time) | Signal Sleuths; 3KA | cross-limb consistency training even though test = 1 limb | Low |
| 10 | Physics-informed / masked SSL on unlabeled inertial streams | PIM; masked AE; COMODO | free signal from 22 subjects' continuous recordings | Med–High |
| 11 | Semi-supervised self-training on test-like windows | SHL (+1 p.p.) | we already pseudo-label in graph rounds; extend to window model | Low |
| 12 | Class-name semantic embeddings + VLM pseudo-labels | SensorLM; SLIP; Bock weak-annotation | label-semantics enabler, mostly for the report | Med |
| 13 | Activity-group specialists (arm- vs leg-dominant) | SHL +1; 3KA ANOVA | matches measured feature–location interactions | Low |
| 14 | Protocol-derived duration/count priors (label-statistically defensible only) | Signal Sleuths ≥50 s rule | our Sinkhorn-97 is already this; make counts learned | Low |
| 15 | TSFM/MoE ensemble members as diversity | DeepSenseMoE; WHAR Arena | diversity where GBM/CNN/TSFM disagree least | Med |
