# 08 — BATTLE PLAN: "Operation Beat-Anonym" (target ≥ 0.9454, 10 submissions)

> Mission: beat **Anonym 0.94535** (#1) and **ANMOL GARG 0.94355** (#2) on the public LB tomorrow.
> Honest framing first: the **prize ranking was frozen July 5** (private LB at report deadline) — public #1
> now = credibility, arXiv report strength, and 4th-challenge seeding. Still worth winning. Here's the shot.

## 1. Where the points must come from (the math)

| Lever | Expected Δ (public LB) | Source |
|---|---:|---|
| Rerun goodpjw2008 stack | 0.92942 baseline | measured |
| + yeashusemwal artifacts stack | 0.92875 (member) | measured |
| + probability-level ensemble of members | **+0.004–0.010** | standard ensemble gain over best member |
| + boundary refiner on the ensemble | +0.002–0.004 | goodpjw/yeashu measured |
| + learned-counts v2 (quantile targets, 2-pass) | +0.002–0.005 | goodpjw measured; oracle headroom +0.013 |
| + confidence-gated video override (anchored voting) | +0.002–0.006 | A5 in plan 06; unclaimed headroom |
| + per-class logit bias ascent + link bagging | +0.002–0.004 | measured |
| **Cumulative optimistic** | **0.944–0.952** | vs needed 0.9454 |
| **Cumulative mid-estimate** | **0.938–0.942** | → likely #2–#5 territory |
| Floor (if ensemble under-delivers) | 0.933 | → top-10 |

**P(#1) ≈ 20–30% · P(top-3) ≈ 45–55% · P(top-5) ≈ 70%+.** The only realistic path to #1 is
*ensemble-first + refiner/counts on top* — no single rerun gets there.

## 2. Tonight (0 GPU minutes, pure setup — do before sleeping)

1. **Fork all 4 backbone notebooks** into your account (forks inherit attached inputs + accelerator):
   - `goodpjw2008/wear-hasca-learned-links-counts-lb-0-92942` ← the anchor
   - `yeashusemwal/ranked-timelines-boundary-refiner-lb-0-928` ← artifacts-based, fast
   - `jiweiliu/public-fast-gpu-inference` (weights already public: dataset `jiweiliu/wear-timeline-graph-models`, 238 MB)
   - your `boosted_boosted_boosted` (set ESSENTIAL-only if the 12 h window is tight)
2. **Check your submission quota** in the Submissions tab (left panel: "x remaining today"). The plan
   below lists 10 slots in priority order — if the cap is 5/day, run slots 1–5 only.
3. **Settings check per fork**: accelerator = GPU T4 ×2, internet OFF, competition attached, persistent
   /kaggle/working. Verify the input panel still resolves (dataset `jiweiliu/wear-timeline-graph-models`,
   competition files).
4. **Save & Run All (Commit)** on goodpjw first (longest: ~6–8 h on T4×2), then queue yeashu (~1 h,
   artifacts) and jiweiliu (~1–2 h). Kaggle free tier runs limited concurrent GPU sessions — serialize:
   goodpjw overnight; yeashu + jiweiliu tomorrow early morning.
5. **Pre-write the ensemble notebook** now (see `code/wear_ensemble_toolkit.py`): a fresh notebook with
   (a) the three pipeline outputs attached as notebook-inputs, (b) the toolkit as a utility-script cell.
   When the runs finish tomorrow, you hit "Save & Run All" and it produces every ensemble variant at once.

## 3. Tomorrow — submission ladder (run top-down; ship only if OOF gate passes)

| Slot | What | Expected LB | Decision rule |
|---:|---|---:|---|
| 0 | goodpjw rerun finishes → **submit as-is** | 0.92942±0.001 | validates fork repro; anchor for everything |
| 1 | yeashu rerun → **submit as-is** | 0.92875 | second member |
| 2 | jiweiliu fast rerun (optional if slots tight) | 0.9077 | skip if quota < 6 |
| 3 | **E3 = geo-mean of P_test of {goodpjw, yeashu, ours}** (weights OOF-tuned via member OOF artifacts; toolkit `tune_weights_oof`) | 0.933–0.940 | submit only if OOF(E3) > OOF(best member) + 0.0005 |
| 4 | **E3 + chain boundary refiner** (flip classifier on ±3 tiles of label changes; features: E3 margins, per-member agreement, link scores, duration priors) | +0.002–0.004 | OOF-gated; 3 refiner seeds |
| 5 | **+ learned-counts v2** (quantile count targets p25/p75 → two Sinkhorns averaged; learned null floor per subject) | +0.002–0.005 | count MAE < 9.0 on member OOF |
| 6 | **+ confidence-gated video override** (anchored voting: video-heads of ≥2 members agree w/ margin > τ on E3-low-margin windows only) | +0.002–0.006 | τ swept on OOF; ship only per-class-positive |
| 7 | Best-so-far + per-class logit-bias coordinate ascent (OOF) | +0.001–0.003 | free, always gated |
| 8 | Best-so-far with 15-matchings link bagging (vs 8) | +0.001–0.003 | OOF-gated |
| 9 | **FINAL = best subject-balanced-OOF composite** (not best public LB!) | — | selection rule below |

**Anti-noise rules:** minimum OOF margin +0.0005 to ship any change; every shipped composite gets a manifest
(git SHA/notebook version, member list, weights, OOF F1); never submit two composites that differ only by
seed noise — probe with intent (the slots ARE your LB-probing budget).

## 4. The OOF problem & how we solve it honestly

Ensemble weights and gates need out-of-fold probabilities of every member:
- goodpjw's notebook **computes and ships OOF artifacts in-run** (`oof_raw.npz`, links, counts) — the fork's
  output carries them; attach it as input to the ensemble notebook.
- yeashu's notebook is artifacts-only and prints per-component OOF F1; if its OOF `.npz` are attached, use
  them; else fix its weight at the OOF-optimal constant from its own log (0.3 teacher share) and gate the
  ensemble on goodpjw's OOF only.
- OOF metric = subject-balanced macro-F1 on the 69,326 test-like windows — the same protocol as our stack.

## 5. Private-LB / final-selection discipline (still matters for the report)

- Select the final submission by **subject-balanced OOF**, never by public LB (public/private split is by
  rows; subjects are the honest generalization axis; yeashu's whitening case proved dev≠LB).
- Keep the last slot for the OOF-best, not the public-best.

## 6. Contingencies

| If... | Then... |
|---|---|
| goodpjw run fails / OOM | rerun with ESSENTIAL-only queue (final_s0, imu_s0) — expect −0.004 vs full |
| Quota is 5/day | run slots 0,1,3,4,9 (skip jiweiliu, bagging, bias) |
| E3 ≤ best member on OOF | submit best member + refiner/counts directly (slots 4–5 off the member, not E3) |
| We're at 0.942 by slot 7 and #1 needs 0.9454 | spend remaining slots on the video-override τ sweep — it's the only lever with +0.006 upside left |
| A member's OOF artifacts are missing | treat it as label-only member (majority vote, tie→anchor) — ensemble gain halves |

## 7. After the dust settles

Write the outcome into `docs/02` (LB log like yeashu's — full transparency), update the techniques matrix
with measured deltas, and start the arXiv report from the OOF ladder. Whatever rank lands, the dossier is
the asset that compounds into the 4th challenge.
