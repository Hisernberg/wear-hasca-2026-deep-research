# 01 — Our Submission Analysis (udaken10 / team "really?")

## 1. Trajectory overview

Six public notebooks form our lineage. Public LB evolution:

| # | Notebook | What it is | Public LB |
|---|---|---|---:|
| 1 | `base-line-liner-model` | EDA (Japanese annotations) + `LogisticRegression(C=1.0)` on basic per-sensor features. Starting point; learned the data shape (T=3466400 samples, 24 recordings, 22 subjects; label distribution plots) | — |
| 2–5 | `hey_gpu`, `timeline`, `timelinereconstruction`, `timelinereconstructiongraph` | Adoption & hardening of the public "Briano Timeline+Graph" stack (credits inside: link code adapted from honghanhhh's Hungarian Chain lineage). Four near-identical variants differing in: local-run path handling (`WEAR_DATA`/repo-root detection), Kaggle time budget guards, CPU-fallback runs | 0.88653 → 0.89577 → 0.90315 → **0.90759** |
| 6 | `boosted_boosted_boosted` | The [BOOSTED] iteration: 8-seed ensemble plan, segment time-masking augmentation, LS 0.1→0.12, OOF-tuned blend weights, +5 graph configs, more LGBM rounds | run locally (8h budget) |

Ranking context: 0.90759 = **rank 49 public** of 205 teams. The Kaggle API reports our `userRank: 6` —
consistent with a **better private-only submission** (public notebooks ≠ our best submission) or private-LB
position; the exact split needs the submissions tab (our KGAT token lacks the `competitions.participate`
scope to read it — see §6).

## 2. Anatomy of our final pipeline (the timeline* family)

```
W1 preprocess ──► W2 test-like OOF windows ──► W3 window models (Net) ──► W4 blend
                                                                      │
        ┌─────────────────────────────────────────────────────────────┘
        ▼
L0 links (LGBM pair scorer → Hungarian → cycle-break)
        ▼
Plp (per-subject label propagation)
        ▼
tabular experts S3/T (hand-crafted IMU + video PCA + neighbor blocks)
        ▼
combined links A–E (ridge boundary predictors + 49+16 feat LGBM → qnorm)
        ▼
graph stage (kNN graph + link graph + self-training rounds, 5–10 configs)
        ▼
finish: sharpen T=0.5 → per-subject Sinkhorn count calibration → argmax → submission.csv
```

### W3 window model `Net` (the ceiling-setter)
- **Inputs**: IMU `(B,50,3)` of one limb + sensor embedding `Embedding(4→32)` tiled as extra channels; video `(B,15,768)`.
- **IMU tower**: 5 dilated residual Conv1d blocks (4+32→64 k7 → 128 k5 → 128 k5 d2 → d k3 d4 → d k3 d8) → mean+max pool → `Linear(2d→d)`.
- **Video tower**: `Dropout(0.5)→Linear(768→d)→GELU→Dropout(0.2)`; "pool" head = mean+std; "tf" head = 2-layer TransformerEncoder(4 heads, pre-norm) → mean+max.
- **Fusion**: `Linear(2d+32→d)→GELU→Dropout(0.3)→Linear(19)` + aux heads `h_imu`, `h_vid`.
- **Training**: AdamW lr 1e-3, wd 0.05, OneCycle, EMA 0.999, grad-clip 2.0, bs 512, LS 0.12 ([BOOSTED] from 0.1), loss = CE + 0.3·aux_i + 0.3·aux_v (modality-dropout-masked), class-balanced sampler ∝ freq^-0.5, 5 subject-folds greedy-balanced by window count (fold sizes pinned [14353, 12995, 14625, 12768, 14585]).
- **Augmentation**: Rodrigues rotation ≤20°, scale 0.9–1.1, Gaussian σ=0.02, video noise σ=0.1, 10% frame dropout, modality dropout (20% no-video, 10% no-IMU), [BOOSTED] segment time-masking (zero a random ≤20% span).
- **Normalisation**: per-(subject×limb) IMU stats, per-subject video stats — **transductive** (test-subject stats computed label-free). Global mode exists but subject mode is what ships.
- **Ensemble**: 4 fusion seeds + 2 TF-variant + 2 IMU-only (30 ep) runs; blend `logp = fus + imu_w·imu`, [BOOSTED] tunes `imu_w ∈ [0.1, 0.6)` on OOF argmax-F1.

### The post-processing chain (where +0.18 comes from)
1. **L0 links**: candidate successors = union of top-k of three video similarity matrices (tail3·head k=40, last·first k=15, mean·mean k=15); 12 pair features (fwd/rev cos, dominance margins row/col, log-ranks, same-limb, IMU endpoint gap + linear-extrapolation error); LightGBM 200→400 rounds ([BOOSTED]); **Hungarian 1:1 assignment → drop log-odds < −1 → break every cycle at its weakest edge**. OOF: linked 0.924, exact-successor 0.330, same-label 0.887.
2. **Ridge boundary predictors**: per-subject sufficient statistics (Gram matrices, float64 on GPU) over tile pairs; forward/backward ridge predicting the next tile's head video block from the current tail block (5×768 blocks: First/Head3/Tail3/Last/Mean, subject-centered); per-(limb×limb) IMU 19→11 ridge with precision-sqrt (Mahalanobis log-likelihood). R² ≈ 0.58.
3. **Combined links**: 11 candidate sources (th, lf, mm, lfc, imu, extlin, ext2, dual[=Sinkhorn-style 2L−r−c], lfz + ridge top-20s), cap 160/row; 49 base + 16 ridge features; LGBM 300 rounds; true-successor + 16 random negatives/tile; OOF rows 1,173,696, pos-rate 0.0549, candidate recall 0.930. → Hungarian → cycle-break → **per-subject quantile normalization** against a 1001-point OOF reference table.
4. **Tabular experts**: ~130 IMU features per window (time/freq/bands/autocorr/entropy/grav) + [BOOSTED] cross-time features (half-vs-half means/stds, peak-to-peak, frac>1.2g) + z-versions + video mean/motion/PCA-48 (per-subject-centered, PCA fitted label-free on OOF); LGBM "S3" (with linked-neighbor block + kNN block, 533 cols) and "T" (361 cols); blend `0.55·S3 + 0.25·T` ([BOOSTED] grid on OOF).
5. **Graph stage**: per-subject kNN graph on video embedding (optionally smoothed along links) mixed with link-row propagation; label propagation α≈0.93–0.97, 20 iters; **2–4 self-training rounds** where propagated probabilities are count-calibrated pseudo-labels mixed back at w≈0.5; 5→10 config ensemble ([BOOSTED]).
6. **finish**: sharpen `P^(1/0.5)` → **Sinkhorn per subject**: 18 exercise columns each constrained to `per_ex × sets` tiles (97, sets={0:2, 14:2} for the double-recorded train subjects; `{}` for test), null gets the rest (floor 5%) → argmax.

### Reference numbers from our own logs
- Window-level OOF: fusion 0.7194, IMU-only 0.5995, blend 0.7253.
- OOF ladder of the base stack: 0.725 → 0.847 (prop + link smoothing + sharpen/Sinkhorn) → 0.877 (+tabular) → 0.905 (+combined links + graph self-training) → LB 0.90315; our best variant **0.90759**.
- Test null shares: sbj 22→0.66, 23→0.43, 24→0.15, 25→0.09 (Sinkhorn absorbs this huge variance).

## 3. What our iterations actually changed (diff-by-diff)

| Change (timeline → boosted) | Rationale | Expected effect |
|---|---|---|
| `WINDOW_BUDGET_MIN` 480 → 360 min; 8-seed sequential plan on 1 GPU | fit 8 seeds into one T4 session | ⚠️ double-edged: more seeds but the time-guard **skips** later runs when the estimate exceeds budget → seed set becomes nondeterministic between sessions |
| Segment time-masking ≤20% zeroed | robustness to sensor dropout windows in test | small + on OOF (masking also helps calibration by preventing over-confidence) |
| LS 0.1 → 0.12 | macro-F1 likes smoother distributions | ±0.001–0.003 OOF; monitor boundary-window confusion |
| L0 rounds 200→400; tab S3 60→200, T 113→350 | link quality & tabular capacity | +0.001–0.005 OOF (link L0 exact ↑) |
| OOF-tuned `imu_w`, `S3/T` weights (grid) | removes hand-tuned constants | +0.001–0.004 OOF, reduces one overfit source |
| TF video head (mean+max) as extra ensemble members | decorrelate video pooling | small + if seeds are blended geometrically |
| 10 graph configs (α 0.93–0.97, rounds 2–4, link bias −2/−3) | config diversity ≈ free ensemble | +0.001–0.003 OOF; watch runtime |
| soft-asserts, local-path resolution, `/tmp` scratch, RSS/stage logging | reproducibility engineering | no LB effect, big iteration-speed effect |

**What did NOT change (and should, see 06-improvement-plan):** candidate-generation diversity, exact-successor
ceiling (66% best-known), boundary refiner, learned count prior, link bagging, evidence-blending of an external
teacher, whitening of embeddings, per-limb specialist models.

## 4. What hurt us / risks in our stack

1. **qnorm coupling**: our 1001-point `QN_REF` quantile table is a snapshot of *one* run's OOF link-score
   distribution. Re-running with different seeds/rounds silently shifts the meaning of every sigmoid weight.
   Fix: regenerate the table inside each run (we already compute OOF links — use them).
2. **Time-budget nondeterminism**: the `run_queue` guard can silently drop `imu_s1` or `final_s1_tf`,
   changing the ensemble between submissions. Fix: make the guard *downgrade* (shorter epochs) rather than skip.
3. **fp16/GradScaler on T4**: mixed precision failed at least once and the auto-fp32-retry then downgrades the
   whole queue. On T4 consider `bf16` via `torch.autocast(dtype=torch.bfloat16, ...)` emulation checks, or
   keep fp16 but pin per-run fallback instead of queue-wide.
4. **Sinkhorn on test without sets knowledge**: `sets={}` assumes every test subject recorded each exercise
   once — correct per the protocol, but if a test subject repeated/skipped an activity the column targets are
   wrong. The learned-count prior (goodpjw2008) is the robust replacement.
5. **No boundary refiner**: ~1/3 of our remaining errors sit within 1 s of a true boundary (best-documented
   number in the ecosystem). We currently ship raw argmax at boundaries.
6. **Video tower is near-dead weight for logits** (video-only head ≈ 0.45–0.5 vs IMU 0.60): its value flows
   through embeddings/links. Consider freezing/shrinking it and reallocating capacity to IMU context (see plan).

## 5. Where our rank sits and why

- Our 0.90759 = fully-reproduced public stack + our [BOOSTED] tuning. The fork cluster (0.90770 jiweiliu) is
  the same base. The next tier (0.923–0.929) is exactly the six increments we have not yet ported:
  L3/two-tower+SSL links, link bagging, kNN-leans-on-links (g=1.5), learned counts, adapted tabular expert,
  boundary refiner. goodpjw2008 documents each as +0.001–0.010 OOF; their cumulative OOF 0.9040→0.9300.
- The 0.935–0.945 tier (top-7) is presumably the same stack plus private add-ons (better window ensemble,
  more careful boundary work, maybe ensemble of independent pipelines). The public code ceiling observed is
  0.92942; the delta from 0.929 → 0.945 is our realistic "unknown unknowns" zone.

## 6. Open items / blockers

- [ ] Kaggle submissions tab: our KGAT token cannot list submissions (`competitions.participate` scope missing).
      Either mint a new token with that scope or check the "Submissions" tab manually for: which submission is
      selected for private LB, per-submission public/private scores, and whether our best is already selected.
- [ ] Private LB will use the unseen 50% of test. Selection strategy: prefer the submission with best
      *subject-balanced* OOF, not best public LB (public/private split is by test rows; our OOF by subjects is
      the safer proxy).
- [ ] The train VideoMAE `.npy` set disappeared from a later dataset version (host thread, unanswered) — pin the
      dataset version we used and archive a copy locally.
- [ ] arXiv path: host offers endorsement for preprints → write the technical report from 06-improvement-plan
      results (HASCA format ≤ 6 pages).
