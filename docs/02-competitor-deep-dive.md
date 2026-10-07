# 02 — Competitor Deep-Dive (all 29 public notebooks)

> Ecosystem truth first: **the 0.90+ notebooks are not independent solutions** — they are a fork-lineage
> ("Briano" Timeline+Graph base, itself building on honghanhhh's Hungarian Chain lineage) plus three teams'
> add-on layers. Treat the public code as one evolving stack, not 29 ideas.

## 0. Ecosystem map

```
akhyar2612 "Hungarian Chain Viterbi" (0.72916) / "Chain" (0.71512)   ← earliest public decode
   └─ honghanhhh "LB 0.9" (0.92485, 27 votes)                        ← L3 links, bagging, learned counts, null specialist
   └─ "Briano" Timeline+Graph base = woominyo (0.890/0.90315) ≡ akhyar2612_lb-0-891 (byte-identical) ≡ udaken10_* (ours)
         ├─ jiweiliu "fast GPU inference" (0.9077)                   ← same stack, weights-only port
         ├─ goodpjw2008 "Learned Links + Counts" (0.92942)           ← best public: +7 layers on the base
         └─ yeashusemwal "Ranked Timelines + Boundary Refiner" (0.92875) ← artifacts-only + ranked timelines
independent designs: avikdas567 cross-attention (no LB stated, window ≈0.67), nomannic19 (0.61278),
swathiha (0.671, broken timeline), sibamsamanta07 (0.652), evelynyang02, lakhindarpal, hmnshudhmn24, stmugiwara_*
```

## 1. goodpjw2008 — "Learned Links + Counts" (0.92942, best public)

The reference implementation. Same window model as ours (identical `Net`, 5 subject-folds, 69,326 test-like
OOF windows, EMA, modality dropout, per-limb normalization; blend `fus + 0.3·imu`; window OOF 0.7253).
Their seven additions, each individually documented:

1. **L3 links — two-tower neural matcher**: BiGRU over video frames (tail reads last step, head reads first)
   + IMU ConvBlocks over 8 edge samples + per-limb `same` embeddings; contrastive CE over 512-tile blocks;
   2 seeds. Plus **self-supervised per-subject ridge** (within-tile frame→frame+14 mapping, applied across
   tiles — works label-free on test subjects). Successor top-1: 32.9% (L0) → 56.2% (L2) → 62.0% (+tower)
   → 65.0% (+SSL) → **66.0% (+2nd seed)**. OOF 0.9040 → 0.9138.
2. **Link bagging**: 8 Hungarian matchings = clean + 7 Gumbel(τ=0.3)-perturbed; graph decoded per matching,
   geometric mean. +0.0014 OOF.
3. **kNN leans on links**: video embedding smoothed along links before neighbor search (`g` 0.5 → 1.5).
   +0.0018 OOF.
4. **Learned count prior**: per-(subject,class) features from sorted log-margin profiles (ranks 60–140) →
   RidgeCV regresses expected tile counts (clip 70–135), trained only on true counts ∈ (55,150), leave-one-
   fold-out. Count MAE 9.8 → 7.3 tiles vs fixed 97. **OOF 0.9170 → 0.9232 (oracle true counts: 0.936!)**.
   Two passes (re-propagate with new targets).
5. **Pseudo-label adaptation of tabular expert T**: refit on train tiles + pipeline-pseudo-labeled OOF/test
   tiles (weight 3.0), cross-fitted halves. Tile-level F1 0.64 → 0.83; +0.0039 OOF.
6. **Chain boundary refiner**: at every label change along each of the 8 matchings, windows within ±3 tiles
   get ~37 features (offsets, null flags, log-prob/logit/Q margins, link scores, video motion, neighbor
   marginals, timeline-consistency flags); LGBM binary "should flip", applied if mean flip-prob > 0.5.
   +0.0029 OOF → **0.9300 OOF / 0.92942 LB**.
7. **Engineering**: float16 storage everywhere, float64 Gram on GPU, dual-GPU longest-first queues with
   staggered loads, fp16-retry-fp32 safety, SHA1-pinned OOF sampling.

Documented data findings: two `null`-labeled session tails where the participant exercises (`sbj_10` tail
2566 s, `sbj_2` tail 3452 s), one session whose labels lag 10 s (`sbj_7` from s=1368). Cleaning lifts OOF but
not LB (ships off by default, `WEAR_CLEAN`).

## 2. honghanhhh — "LB 0.9" (0.92485, most popular, 27 votes)

Same skeleton. Deltas vs goodpjw2008:
- **soft_bout_feats** (7 features: soft same-bout probability Σ√(p_i·p_j), both-null, both-exercise, argmax
  agreement, video cos, margins) replace the second tower seed; single L3 scorer.
- **Link bagging ×15** (τ ∈ {0.2, 0.3} × 7 + clean).
- **Learned counts**: 0.55·RidgeCV + 0.45·LGBM with run-length features (longest run, n_runs), link
  consistency (P(successor same class)), null share; **3 passes**.
- **Gated null/boundary specialist**: LGBM is-null on tile features; flips only if gate margins satisfied;
  **OOF gate — if OOF doesn't improve, keep pre-refine labels** (their log shows "boundary refine GATE FAIL"
  discipline).
No kNN-lean (g stays 0.5), no chain refiner → lands ≈0.005 under goodpjw2008.

## 3. yeashusemwal — "Ranked Timelines + Boundary Refiner" (0.92875)

Thin inference notebook + private artifacts; the interesting content is the **LB experiment log** (rare
honesty):

```
0.6646 baseline repro → 0.6894 sensor specialists + per-participant activity minimum
→ 0.7775 VideoMAE similarity graph + 90s quota → 0.8092 supervised video head
→ 0.8280 CatBoost-LGBM blend → 0.8365 cross-sensor graph
→ 0.9077 TIMELINE RECONSTRUCTION on public stack  (+0.071 — biggest single lever)
→ 0.9190 SSL links + adapted teacher + two-pass counts → 0.9237 confidence-adaptive link blend
→ 0.9272 8 ranked timelines → 0.9280 consensus refiner → 0.9286 4 refiner seeds → 0.92875 whitening
```

Techniques unique here: **LambdaRank timeline links** (score candidates within the candidate set; confidence
= margin over runner-up — "a link score needs a competitor"); **8 ranked timeline hypotheses** with
cross-hypothesis consistency features; **whitened graph geometry** (per-participant centering + whitening^0.5
of embeddings, kNN τ=0.3); **two-pass learned counts**; **consensus boundary refiner** (89 label-free
features, proposals supported by several hypotheses are more reliable — "support enters as features rather
than as a vote").
Rejected & documented: 16 timelines (−0.0013), single timeline (−0.0009), graph smoothing 1.5 (inconsistent),
supernode restart (rescued 321, broke 324), direct macro-F1 regret optimization (overfit).
**Validation-honesty audit**: a count model trained on another fold's features leaked through the second
count pass; excluding both outer and profile folds in *both* passes made it invariant. Whitening: dev +0.0030
but LB +0.0001.

## 4. jiweiliu — "public fast GPU inference" (0.9077)

Weights-only port of the base stack. Contributions: **prepared-input cache** verified `max_abs < 1e-5`
against the source path, per-batch timing, run-set split across 2 GPUs, `source_precision.txt` to pin
fp32-vs-AMP numerics, single `test_emb` computation. Use as the template for our submission runtime.

## 5. avikdas567 — "Multimodal VideoMAE & Inertial Cross-Attention HAR" (21 votes)

Beautiful notebook, capped design: `EgoInertialNet` (~0.79M params) — IMU SE-Residual CNN → 2-layer BiGRU;
video Linear+pos → 2-layer Transformer; **bidirectional cross-attention** + gated bilinear fusion. But:
**single fold** (`next(GroupKFold.split)`), 8 epochs, zero augmentation, null subsampled 1-in-8, round-robin
sensor assignment (↔ label-position correlation), 80%-homogeneous windows only, raw unnormalized features,
**no post-processing at all** (just 0.6·NN + 0.4·LGBM, temperature tuning, argmax). Val macro-F1 0.6734.
Lesson: an elegant fusion head cannot compensate for missing sequence machinery; also 1-fold variance is
unusable for decisions.

## 6. nomannic19 — "Temporal Fusion Ensemble" (0.61278) — post-mortem

1. No sequence reconstruction at all → leaves ~0.2 LB on the table (window ceiling here ≈ 0.60–0.72).
2. Train windows cut only inside pure segments (stride 25) vs arbitrary test tiles → boundary mismatch.
3. **Sensor-geometry mismatch**: trains on all 4 limbs simultaneously `(B,4,3,50)` with per-limb logits; test
   is one random limb `(1,3,50)`. (Same bug family as hmnshudhmn24's right-arm-only RF.)
4. 150 windows/subject/class cap + stride 0.5 s overlap → diversity-starved; 3–4 epochs, no normalization,
   no OOF monitoring, hand-set `NULL_BIAS` ×2.12 on null prob.

## 7. The weak cluster — failure taxonomy

| Notebook | LB | Fatal flaw |
|---|---:|---|
| swathia "Aligned Video–Inertial Fusion" | 0.67126 | "timeline" links built on test `id` column — **ids are shuffled**, so it smooths among random rows (no-op dressed as sequence modeling) |
| sibamsamanta07 | 0.65265 | hand-crafted features + hierarchical LGBM, no deep model, no sequence machinery |
| akhyar Chain/Viterbi (pre-Sinkhorn era) | 0.715–0.729 | raw-cosine chaining folds during still stretches + hard 80 s fill painted exercise onto null — the pathology Sinkhorn later fixes |
| evelynyang02 | ~0.7 | window-only LGBM; nice trick though: **left-limb mirroring** (x-axis sign flip onto right arm) |
| lakhindarpal | ~0.5s | CatBoost on 12 stats; probe-based feature selection on full data (mild leakage) |
| hmnshudhmn24 | ~0.0–0.4 | right_arm-only means — ignores random-limb test design |
| stmugiwara v5–v7 | ≤0.05 | string labels in CSV, syntax error, 18-class head, mean-over-time padded test features |
| stmugiwara dummy | 0.0 | copies sample submission |

## 8. What the 0.92+ share that 0.60–0.75 lack (the five pillars)

1. **Sequence reconstruction of shuffled windows** — learned links → Hungarian 1:1 → cycle-break (+0.071 LB alone).
2. **Per-subject count calibration** — sharpen T=0.5 + Sinkhorn to ≈97 tiles/exercise, null floor 5%.
3. **Test-like OOF validation** — non-overlapping 1-s tiles, one random *valid* limb, subject-wise folds; every stacking input out-of-fold.
4. **Graph label propagation + self-training** with count-calibrated pseudo-labels, config ensembles.
5. **Ensembling discipline** — 4–8 window-model members, 8–15 link baggings, 5–10 graph configs, 3–4 refiner seeds, log-linear blends with OOF-tuned weights — plus transductive per-subject normalization.

## 9. Ranked technique table (LB-impact order)

| # | Technique | Evidence | Used by |
|---|---|---|---|
| 1 | Learned successor links + Hungarian + cycle-break | +0.071 LB; exact-succ 33→66% tracks +0.01–0.02 OOF per step | all 0.90+ |
| 2 | Per-subject count calibration (Sinkhorn-97) | inside the 0.725→0.847 step; learned counts +0.006; oracle +0.013 | all 0.90+ |
| 3 | Video-kNN label propagation (+ prediction similarity) | core of the +0.122 step | all 0.90+ |
| 4 | Dual-tower window model + per-limb normalization + modality dropout + seed blend | ceiling 0.725 vs 0.60 | top-4 |
| 5 | Graph self-training rounds w/ count-calibrated pseudo-labels | +0.028 | all 0.90+ |
| 6 | Tabular experts S3/T + blend | +0.030 | woominyo family |
| 7 | Link bagging (Gumbel matchings, geometric mean) | +0.0014–0.004 | goodpjw, honghanhhh, yeashu |
| 8 | Learned count prior (rank-profile ridge) | +0.006 OOF (MAE 9.8→7.3) | goodpjw, honghanhhh, yeashu |
| 9 | Two-tower matcher + SSL per-subject ridge | +0.0098 OOF | goodpjw, honghanhhh |
| 10 | Pseudo-label adaptation of tabular expert | +0.0039 OOF | goodpjw, honghanhhh |
| 11 | Boundary refiner (chain / consensus / gated-null) | +0.0008–0.003 | goodpjw, yeashu, honghanhhh |
| 12 | kNN embedding smoothed along links (g 1.5) | +0.0018 OOF | goodpjw |
| 13 | Per-subject quantile normalization of link scores | enabler of #7/#2 | goodpjw, honghanhhh, jiweiliu |
| 14 | Whitened graph geometry | dev +0.0030 / LB +0.0001 | yeashu |
| 15 | LambdaRank-within-candidates + runner-up margin confidence | part of 0.9237→0.9272 | yeashu |
| — | Elegant-but-unpaired fusion heads, sensor-geometry mismatch, fake timelines from `id`, no CV | 0.61–0.67 | the rest |
