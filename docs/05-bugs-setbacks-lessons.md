# 05 — Bugs, Setbacks & Lessons (the hall of fame)

A curated catalogue of every documented failure in this competition's ecosystem + ours, with root cause and
the lesson each one teaches. Use it as a pre-flight checklist for any future HAR pipeline.

## A. Official / dataset bugs

| Bug | What happened | Lesson |
|---|---|---|
| **Video feature generation bug** (official, Apr 2025) | Host fixed a bug in test VideoMAE feature creation; ordered re-download. Submissions not using video features were unaffected | Modality-value isolation pays: keep an inertial-only path that survives modality bugs |
| **Video tensor shape ambiguity** `(N,768,15)` vs `(N,15,768)` | Multiple notebooks carry a transpose guard | Assert shapes + a smoke-tile unit test before any run |
| **Train VideoMAE `.npy` disappeared** in a later dataset version (host thread, unanswered) | Downstream teams (e.g. Bina Salama's 62k-window Inception+Transformer) blocked on regeneration | Pin dataset versions; archive inputs locally; keep feature-extraction repro scripts |
| **Session label defects** (goodpjw2008 findings) | `sbj_10` tail 2566 s & `sbj_2` tail 3452 s labeled `null` while the participant exercises; `sbj_7` labels lag 10 s from s=1368 | Audit labels against physics (acc magnitude) before trusting them; cleaning helped OOF but **not** LB — CV gains ≠ LB gains |

## B. "Sequence reconstruction done wrong" bugs

| Bug | What happened | Lesson |
|---|---|---|
| **Fake timeline from shuffled `id`** (swathiha, 0.671) | Linked windows by `id` order — test ids are shuffled → smoothing among random rows, a no-op | Verify that any "time" column is actually temporal (monotonic vs meta; sanity-plot autocorrelation) |
| **Raw-cosine chaining** (akhyar pre-Hungarian era, 0.72) | Greedy nearest-cosine chaining folded onto itself during still stretches; a hard 80 s fill then painted exercise onto null | Naive greedy linking is worse than none; need 1:1 assignment (Hungarian) + cycle breaking + count calibration |
| **Hard fills of unlinked windows** (same era) | Long still stretches force-filled with exercise classes | Never hard-fill; leave null-floor + Sinkhorn to decide |
| **Cycles in successor chains** | 1:1 assignment creates cycles (A→B→A) | Break every cycle at its weakest link edge (standard in all 0.90+ code) |

## C. Validation & leakage bugs

| Bug | What happened | Lesson |
|---|---|---|
| **Two-pass count leak** (honghanhhh) | Count model trained on another fold's features leaked through the second calibration pass — changing one fold's targets moved 32 held-out predictions | Out-of-fold EVERY stacking input, including "calibration" stages; exclude both outer and profile folds in *all* passes |
| **Probe selection on full data** (lakhindarpal) | 300-feature probe selection before CV → mild leakage | Feature selection belongs inside folds |
| **Single-fold decisions** (avikdas567) | `next(GroupKFold.split)` — one fold, high variance, no seed ensemble | Decision-grade CV needs ≥5 folds × ≥2 seeds |
| **CV that doesn't mimic test geometry** (nomannic19, swathia) | Segment-only windows, round-robin limbs, all-4-limb inputs vs 1 random test limb | Reproduce the test distribution *exactly* in OOF (we do: 69,326 tiles, random valid limb, SHA1-pinned) |
| **Dev-vs-LB divergence** (yeashu whitening) | +0.0030 dev, +0.0001 LB | Report both; prefer changes justified by mechanism, not just dev noise |
| **OOF-tuned constants everywhere without gates** | Silent overfit accumulation | OOF-gate every change ("if OOF doesn't improve, don't ship" — honghanhhh's GATE FAIL log) |

## D. Engineering incidents (ours)

| Incident | Root cause | Fix / mitigation |
|---|---|---|
| **P100 "visible but unusable"** | torch build without sm_60 kernels; `cuda.is_available()` True but first op fails | GPU sanity check in a subprocess running a real op; fall back cleanly (we ship this) |
| **fp16 + GradScaler failure on T4** | mixed-precision path died mid-run | Auto-retry in fp32 — but it then downgrades the *whole queue*: make fallback per-run, and test bf16 emulation first |
| **Time-budget guard skipping runs** | `WINDOW_BUDGET_MIN` estimate exceeded → `imu_s1` silently dropped in some sessions | Degrade (shorter epochs) instead of skip; print a "dropped run" manifest next to submission.csv |
| **qnorm table coupling** | 1001-point `QN_REF` frozen from one run's OOF distribution | Regenerate the reference table from the current run's OOF links (we compute them anyway) |
| **Scratch space as notebook output** | 6.5 GB of intermediates would blow Kaggle output limits | `/tmp` scratch + `WEAR_CLEANUP`; stage-level RSS/peak logging |
| **Local vs Kaggle path drift** | `WEAR_DATA`/repo-root/`/kaggle/input` resolution differences across our 4 variants | One resolver function with explicit search order (later variants fixed this) |

## E. Design anti-patterns observed across the ecosystem

1. **Beautiful-model syndrome**: elegant cross-attention/gated-fusion architectures with broken validation
   foundations (avikdas567) — architecture is the *last* differentiator here.
2. **Right-arm-only assumptions** (hmnshudhmn24): the test set's defining quirk is the random limb — any
   single-limb assumption is fatal.
3. **Hand-tuned magic numbers** (NULL_BIAS ×2.12, hard duration bands): every one of them was later beaten by
   a learned/OOF-tuned equivalent.
4. **Untested refactors** (stmugiwara: string labels in submission CSV, syntax error in cell, 18-class head
   for 19 classes): always smoke-test (`WEAR_SMOKE=1` mode exists for this reason) and assert the label set.
5. **Treating leaderboard as validation**: 4,301 submissions vs ~200 teams — the winners used OOF discipline;
   the LB-chasers burned entries on noise.

## F. Distilled lessons (top 10)

1. **Reconstruct sequences, then decode them** — the whole competition is won in post-processing (+0.18–0.20).
2. **OOF must be a mirror of the test distribution** — same windowing, same limb randomness, same labels.
3. **1:1 assignment beats greedy chaining; count calibration beats hard fills.**
4. **Video's value is geometric (links, graphs), not logit-level** — allocate accordingly.
5. **Out-of-fold everything, including calibration stages** — leaks hide in second passes.
6. **OOF-gate every change and log the gate result** — discipline compounds.
7. **Transductive per-subject statistics are free accuracy** — label-free stats on test are legitimate and big.
8. **Ensemble at every level** (seeds, folds, matchings, configs, refiner seeds) but geometric-mean the
   probability outputs, and OOF-tune blend weights.
9. **Engineering determinism is a scoring feature** — pinned sampling, pinned versions, staged budgets,
   manifests next to submissions.
10. **Clean ≠ better**: some label cleanings help CV and not LB; trust mechanism + LB-verified ablations
    (when entries were free) and keep cleanups behind a flag.
