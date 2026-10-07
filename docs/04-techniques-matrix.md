# 04 — Techniques Matrix: every layer × hurts ❌ / helps ✅ / risky ⚠️

Legend: **Δ** = measured or strongly inferred effect. "OOF" = our-style subject-wise out-of-fold macro F1;
"LB" = public leaderboard. Sources: our logs, competitor prints, prior-challenge reports.

## Layer 0 — Data handling & hygiene

| Technique | Verdict | Δ / Evidence |
|---|---|---|
| Test-like OOF windows (non-overlapping 1 s, one random *valid* limb, majority label) | ✅ critical | all 0.90+; mismatched CV = 0.61–0.67 cluster |
| Subject-wise folds (greedy bin-pack by window count) | ✅ critical | leakage-free CV; fold sizes pinned [14353,12995,14625,12768,14585] |
| NaN interpolation (limit_direction=both) + missing-window masks | ✅ | ~62k windows usable; `W_ok` per-limb validity drives sensor choice |
| Transductive per-(subject,limb) IMU stats & per-subject video stats (label-free, test incl.) | ✅ | global-stats mode under-performs; SHL/3KA agree |
| Label-lag & mislabeled-null cleanup (sbj_10/2/7) | ⚠️ | OOF ↑ but LB ≈ flat (goodpjw measured) — ship off by default |
| `(N,768,15)→(N,15,768)` test-video transpose guard | ✅ (bug fix) | official April-2025 video bug family |
| Pin dataset version (train `.npy` disappeared in later release) | ✅ process | host thread unanswered; archive locally |
| Excluding boundary-straddling windows from training | ❌ | creates train/test mismatch (nomannic19: −0.1 LB class) |
| Null subsampling (1-in-8) without per-subject correction | ❌ | destroys null prior (avikdas567) |
| Windows cut only inside pure segments | ❌ | model never sees boundaries; test tiles straddle them |

## Layer 1 — Window model

| Technique | Verdict | Δ / Evidence |
|---|---|---|
| Dilated residual Conv1d IMU tower (d 1/2/4/8) + sensor-embedding channels | ✅ | backbone of 0.725 window ceiling |
| Aux heads (IMU-only, video-only) with modality-dropout-masked CE | ✅ | +0.3·aux each; enables the fus+imu blend |
| Modality dropout (20% no-video / 10% no-IMU) | ✅ | robustness to test sensor variety |
| Rotation (Rodrigues ≤20°), scale, Gaussian σ=0.02, video noise σ=0.1, 10% frame dropout | ✅ | standard here; 3KA/Signal Sleuths equivalents |
| Segment time-masking (≤20% zeroed) | ✅ small | our [BOOSTED]; helps calibration (anti-overconfidence) |
| Class-balanced sampler ∝ freq^-0.5 | ✅ | macro-F1 aligned; full-balancing (∝ 1/freq) overcorrects |
| Label smoothing 0.10–0.12 | ✅ small | 0.12 slightly better OOF in our diff |
| EMA 0.999 + OneCycle + AdamW wd 0.05 + grad clip 2.0 | ✅ | stability at 16 ep × 300 steps × bs 512 |
| Video tower "tf" variant (2-layer transformer, mean+max) as extra ensemble member | ✅ small | decorrelation; video-only head ≈ 0.45–0.5 (near-dead for logits) |
| Per-class logit bias tuned by greedy coordinate ascent on OOF | ✅ | free +0.001–0.003; classic "greedy anchored" threshold search |
| kNN-smoothing of window logp (video emb, k from {1,3,5,7,10}) | ✅ small | mostly superseded by graph stage |
| Single fold / 8 epochs / no aug (elegant cross-attention net) | ❌ | avikdas567: caps at 0.67 window |
| Training all 4 limbs simultaneously, testing 1 random limb | ❌ | geometry mismatch (nomannic19) |
| Bigger/deeper window towers beyond ~0.8M params | ⚠️ | WHAR Arena: ceiling effect; capacity better spent on links/context |
| DeepConvContext-style window-sequence context model | ✅ (unexploited) | +5% F1 avg on 6 HAR benchmarks (arXiv:2505.20894) — next big window-side win |

## Layer 2 — Fusion (inertial × video)

| Technique | Verdict | Δ / Evidence |
|---|---|---|
| Video value flows via **geometry** (kNN embeddings + boundary linking), not logits | ✅ | top-4 all use video this way; video tower adds ~nothing to logits |
| `logp = fus + imu_w·imu` with OOF-tuned imu_w | ✅ | removes hand constant; +0.001–0.004 |
| Concat-fusion of modalities inside the net | ⚠️ | WEAR paper: ≈ +0.5–1 only |
| Confidence-gated late fusion / anchor-override voting | ✅ (unexploited) | oracle O-LF = 91–94 vs 75 realizable — biggest headroom |
| Cross-attention fusion | ⚠️ | pretty, but without the five pillars caps at 0.67 |
| Video→IMU distillation (COMODO-style) | ✅ (unexploited) | label-free; matches supervised IMU models cross-dataset |

## Layer 3 — Sequence reconstruction (the crown jewels)

| Technique | Verdict | Δ / Evidence |
|---|---|---|
| Candidate successors from video boundary similarities (tail3·head k40, last·first k15, mean·mean k15) | ✅ | recall 0.930 |
| 12→49→65 pair features (cos, dom margins, log-ranks, same-limb, IMU gap/ext, ridge ll, soft-bout) | ✅ | each layer of features tracked +0.005–0.01 OOF |
| LightGBM pair scorer (true-succ + 16 random negatives) | ✅ | pos-rate 0.0549, 1.17M rows |
| **Hungarian 1:1 assignment → drop weak → break cycles at weakest edge** | ✅ critical | +0.071 LB alone; raw-cosine chaining without it *loses* 0.01 (akhyar 0.72 era) |
| Ridge boundary predictors (video blocks fwd/bwd + per-(limb,limb) IMU Mahalanobis ll) | ✅ | R²≈0.58; +16 features |
| Two-tower neural matcher (BiGRU tails/heads + contrastive) | ✅ | exact-succ 56→62% |
| Self-supervised per-subject ridge (within-tile mapping, applied across tiles) | ✅ | 62→65% exact-succ; label-free on test subjects |
| 2nd seed averaging on link scorer | ✅ | 65→66% |
| Link bagging: Gumbel-perturbed matchings, geometric mean of decoded graphs | ✅ | +0.0014–0.004 |
| LambdaRank-within-candidates + runner-up-margin confidence | ✅ | part of yeashu 0.9237→0.9272 |
| Per-subject quantile normalization of link scores | ✅ (enabler) | makes sigmoid weights comparable across subjects |
| qnorm reference table frozen from an old run | ⚠️ | silent drift when re-running (our risk) |
| Dual-softmax "2L−r−c" candidate source | ✅ | cheap diversity in candidate union |
| Link score floor / hard fill of unlinked windows | ❌ | painted exercise onto null (akhyar 0.72 era pathology) |

## Layer 4 — Graph & propagation

| Technique | Verdict | Δ / Evidence |
|---|---|---|
| Per-subject kNN graph on video emb (+ prediction similarity, use_p≈0.5) | ✅ | core of the +0.122 step |
| kNN graph built on **link-smoothed** embeddings (g 0.5→1.5) | ✅ | +0.0018 OOF |
| Mixing link-row propagation into kNN propagation (β 0.6–0.8, α 0.93–0.97) | ✅ | standard in 0.92+ |
| Self-training rounds with count-calibrated pseudo-labels (2–4 rounds, w 0.5–0.7) | ✅ | +0.028 |
| Config ensembles (5–10 graph configs, geometric/mean) | ✅ | +0.001–0.003 |
| Whitened graph geometry (per-participant centering + whitening^0.5, τ 0.3) | ⚠️ | dev +0.0030 → LB +0.0001 (yeashu, honest) |
| Supernode restart | ❌ | rescued 321, broke 324 (yeashu) |
| 16 timelines / single timeline instead of 8 | ❌ | −0.0013 / −0.0009 |
| Direct macro-F1 regret optimization on OOF | ❌ | overfit (yeashu, documented) |

## Layer 5 — Count calibration & final decode

| Technique | Verdict | Δ / Evidence |
|---|---|---|
| Sharpen `P^(1/T)`, T=0.5 | ✅ | inside 0.725→0.847 |
| **Sinkhorn per subject to ≈97 tiles/exercise (null floor 5%)** | ✅ critical | the fix for null-painting; all 0.90+ |
| TRAIN_SETS {0:2, 14:2} for train; `sets={}` for test | ✅/⚠️ | protocol-correct but brittle if a test subject repeats/skips |
| **Learned per-(subject,class) count prior** (rank-profile RidgeCV/LGBM, clip 70–135, LOFO) | ✅ | +0.006 OOF; count MAE 9.8→7.3; 2–3 passes |
| Oracle counts | — | OOF 0.936 vs 0.930 — the remaining +0.006 is count accuracy |
| Duration-band decode + hard fill (pre-Sinkhorn era) | ❌ | the 0.72–0.75 pathology |
| NULL_BIAS ×2.12 hand multiplier | ❌ | nomannic19; superseded by Sinkhorn |
| Greedy per-class logit bias (coordinate ascent) | ✅ | free; keep |

## Layer 6 — Refinement & meta

| Technique | Verdict | Δ / Evidence |
|---|---|---|
| Boundary refiner (LGBM flip classifier over ±3 tiles of label changes, chain or consensus features) | ✅ | +0.0008–0.003; "1/3 of errors within 1 s of a boundary" |
| Gated refiner (only ship if OOF improves) | ✅ process | honghanhhh "GATE FAIL" discipline |
| Pseudo-label adaptation of tabular expert (cross-fitted, weight 3.0) | ✅ | tile F1 0.64→0.83; +0.0039 |
| Evidence blend with external teacher (0.2/0.5/0.3) | ✅ | yeashu stack |
| Two-pass count calibration with leak-checked features | ✅ | leaks through second pass documented & fixed (honghanhhh) |
| OOF-tuned blend weights everywhere (no hand constants) | ✅ | removes silent overfit vectors |
| Mixed-precision fp16 on T4 with auto-fp32 retry | ⚠️ | needed but queue-wide downgrade risk (our incident) |
| fp32-only / CPU fallback runs | ⚠️ | keeps notebook alive; score collapses if it silently engages |
| Time-budget guards that *skip* runs | ⚠️ | nondeterministic ensembles; prefer degrade-over-skip |
| Fast-inference caches (prepared inputs, verified <1e-5) | ✅ | jiweiliu; submission runtime safety |
| SHA1-pinned OOF sampling & stage timing | ✅ process | reproducibility |

## One-line summary

**Ship order by ROI**: links (L3+bagging) → learned counts → boundary refiner → graph tweaks → window-model
context (DeepConvContext) → fusion gating (oracle-LF attack). **Never**: fake timelines from `id`, hard fills,
segment-only training windows, geometry-mismatched sensor handling, single-fold decisions, unvalidated "cute"
graph surgery.
