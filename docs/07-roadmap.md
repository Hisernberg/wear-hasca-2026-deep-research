# 07 — Roadmap & Validation Protocol

## Week 1 — Foundation & ports (P1)
- [ ] Rebuild environment from `timeline.py` (boosted variant) as a clean repo (`wear-timeline`), keep
      SHA1-pinned OOF sampling; smoke-run 5 subjects end-to-end.
- [ ] Port A1-1 L3 links (two-tower + SSL ridge + 2 seeds). **Gate**: OOF links exact-successor ≥ 0.62,
      OOF F1 +0.005 vs 0.9076 stack, else revert.
- [ ] Port A1-2 link bagging (8 matchings). Gate: OOF +0.001.
- [ ] Port A1-3 kNN-lean g=1.5. Gate: OOF +0.001.
- [ ] Port A1-4 learned counts (2 passes). Gate: count MAE < 9.0, OOF +0.004.
- [ ] Port A1-5 adapted tabular T. Gate: tile F1 > 0.80, OOF +0.003.
- [ ] Port A1-6 chain refiner (3 seeds). Gate: OOF +0.002.
- [ ] Engineering: per-run precision fallback, run manifest, QN_REF regeneration.

## Week 2 — Ceiling attacks (P2–P4)
- [ ] A2: candidate sources v2 (limb-aware extrapolation, jerk continuity, cluster anchors) + LambdaRank
      scorer + 2-opt swap pass. Target: exact-successor ≥ 0.70.
- [ ] A4: count model v2 (quantile targets, duration percentiles, learned null floor).
- [ ] A3: boundary program (soft boundary labels → retrain window model; consensus refiner v2 with
      duration-prior + Sinkhorn-sensitivity features; mini-DP at flagged clusters).
- [ ] Freeze "v2 submission" = best OOF-gated composite. Submit ≥ 3 seeds; record manifest per submission.

## Week 3 — Fusion & context (P5–P7)
- [ ] A5: confidence-gated late fusion + greedy anchored voting (video specialist override). Gate per class:
      only keep override where OOF class-F1 improves.
- [ ] A6: window-sequence context model (DeepConvContext-style) retrained on protocol-ordered train windows.
- [ ] A7: per-limb calibration + limb-pairing augmentation; A8: SSL loop closure (test-half tabular T).

## Week 4 — Semantics, hardening, publication (P8–P10)
- [ ] B2: hierarchical class-semantics smoothing/calibration.
- [ ] B1 (optional): LLM tie-breaker on low-margin windows; measure on OOF first, ship only if +.
- [ ] C: full CI smoke, artifact pinning, public repo polish (`wear-timeline`).
- [ ] D: technical report draft from the OOF ladder; request arXiv endorsement from the host.

## Validation protocol (every change, no exceptions)
1. **Same OOF**: 69,326 test-like windows, 5 subject-folds, seed 1234+fold, SHA1 check
   `f81d90085315` / `e4177f2589d7`.
2. **Out-of-fold everything**: links per-fold, counts LOFO, refiner fold-exclusive, second calibration
   pass double-excluded (the honghanhhh leak).
3. **Gate on OOF macro-F1** with a minimum margin of +0.0005 (below = noise) before any change ships.
4. **Seed sensitivity**: any accepted change must survive 2 seeds (±0.001 tolerance).
5. **Submission policy**: only composites of gated changes; ≥2 submissions per composite (seed jitter);
   manifest recorded.
6. **Leak audit**: after every stacking change, perturb one fold's upstream targets and verify held-out
   predictions don't move (yeashu's audit method).
7. **Report dev and LB together**; investigate any divergence > +0.002 (whitening case).

## Risk register
| Risk | Likelihood | Mitigation |
|---|---|---|
| Count prior wrong for a test subject (protocol deviation) | Med | quantile targets; learned null floor; graceful degradation to fixed-97 |
| qnorm/manifest drift across reruns | Med | in-run regeneration + manifest |
| fp16 instability on T4 | Med | per-run fallback; bf16 emulation check |
| Kaggle runtime budget cuts seed set | High (single GPU) | degrade-over-skip; prioritize essential runs first (final_s0, imu_s0, tf variants) |
| Train `.npy` availability | Med | local archive; feature-extraction script ready |
| Overfitting OOF via many gated changes | Med | seed sensitivity + minimum margin + leak audits; report honest CV-LB deltas |
| LLM tie-breaker inconsistency | Med | OOF-gated; tie-break only on top-5% uncertain windows |

## Definition of done (for the 4th-challenge / arXiv goal)
- [ ] `wear-timeline` public repo: one-command run, tests, manifests, model cards.
- [ ] OOF ladder documented: 0.725 → 0.905 → 0.930 → target ≥ 0.940 with every rung's ablation.
- [ ] arXiv preprint submitted (endorsement requested).
- [ ] Reusable lessons: 05-bugs doc translated into a `pre-flight checklist` in the repo.
