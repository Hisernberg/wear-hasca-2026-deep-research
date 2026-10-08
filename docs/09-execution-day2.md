# 09 — Execution Day (Oct 8): cross-family falsification + saturation proof

> Operator: automated session via Kaggle API (KGAT token, account `koushikrudra`).
> Slots used today: **1 of 10** (B2 probe). 9 held in reserve.
> Verdict: **the 0.935-family is saturated at label level; every remaining lever is internal (video override, counts v2, link bagging, refiner-on-ensemble) — none executable from outside the local stack.**

## 1. What happened today (evidence trail)

### 1.1 Asset recovery
| Asset | Source | Value |
|---|---|---|
| 157 own submissions list + 150 valid label files | `DownloadSubmission` signed-URL route | full fusion-era history |
| goodpjw2008 kernel output | `kernels output` | `submission.csv` + `final_probabilities.npz` (oof 69,326×19, test 12,234×19) |
| yeashusemwal kernel output | `kernels output` | the 0.92875 artifacts file (never in user's fusion pool) |
| honghanhhh + akhyar outputs | `kernels output` | 2 more independent families |
| user's own base-model artifacts | zip hidden in submission 56842474 | `oof_logp` (69,326×19, honest OOF 0.7458), `test_logp`, 2 extra fits (`new_oof/new_test`) |
| user's own train.py (Kaggle-patched) | submission 56842464 | window classifier, 5-fold by subject, VideoMAE+IMU |

### 1.2 The falsified hypothesis — B2 cross-family consensus flip (slot 1)
**Design (greedy anchored voting):** anchor = `f4n_clog` (0.93541). Flip window → class c where the two
independent public families BOTH disagree with the anchor and agree with each other
(`goodpjw` ∧ `yeashu` ≠ anchor). Volume: **119 windows**.

**Result: LB 0.93236 (−0.00305).** The flips were mostly wrong.

**Interpretation.** The own stack's decode (links + counts + log-count prior, self-trained on test-like
windows) is right on exactly the windows where external families reach a different consensus. The public
pipelines share the generic prior; the refined own decode carries specialized structure they cannot see.
This **falsifies the entire cross-family flip direction** (B1/B2s/B3/E die with it), consistent with the
user's own earlier probe: "K7+K9 chain-link fusion + public v4 run Q" (LB 0.93087 < 0.93426 without it).

### 1.3 The saturation proof (no slot needed)
Three independent checks, all negative:

1. **Committee unanimity vs decode.** Broad 31-member own committee (0.930+): the decode deviates from
   committee-top on only **43 windows**, and **never** when the committee is (near-)unanimous
   (margin ≥ 28/31 → 0 windows). The "decode-override correction" candidate is **empty** — the decode
   already defers to internal unanimity perfectly. There is no internal inconsistency to exploit.
2. **Committee split volume.** 97.9 % of windows are 16/16 unanimous in the top-16 committee →
   probability-level reweighting can only act on ~130 windows, and only via exact ties (~50).
   The E-Coal candidate changes 50 windows — pure tie-break, no upside mechanism.
3. **Vote saturation.** All 9 of the user's own vote experiments (5–7 members, same pool) scored
   0.9335–0.9352 ≤ best single member 0.93541. Label voting inside the family is exhausted.

### 1.4 Count-lever check
User's own probes already falsified count variations: `b4wa` log-count ridge (0.93145 < 0.93278),
count-target variants cluster below the log-count prior line. Post-hoc count nudging = public-LB probing
with no OOF gate. Rejected.

## 2. Slot ledger & discipline

| Date | Used | Notes |
|---|---|---|
| Oct 8 (today) | 1 / 10 | B2 probe (−0.003, bought falsification of a whole direction) |
| Reserve | 9 today + ~4×10 later | spend only on OOF-gated internal levers |

Anti-noise rules (unchanged from docs/08): minimum subject-balanced-OOF margin **+0.0005** to ship any
change; public-LB reads of ±0.0005 are noise at 12,234 windows; final 2 picks by OOF, not public LB.

## 3. What would actually move the needle (all internal — run on your machine tonight)

Ranked by expected Δ on subject-balanced OOF (reference: mv4 OOF 0.9364 → LB 0.93516):

| # | Lever | Expected Δ OOF | Gate |
|---|---|---:|---|
| 1 | **Confidence-gated video override** (video-head of ≥2 runs, margin > τ, applied ONLY to low-margin fused windows; anchored voting, τ swept on OOF) | +0.002–0.006 | per-class positive on OOF; ~1–3 % windows touched |
| 2 | **Learned-counts v2**: quantile count targets (p25/p75) → two Sinkhorn passes averaged; learned null floor per subject | +0.002–0.005 | count MAE < 9.0 on member OOF |
| 3 | **Refiner on the fused result** (flip classifier on ±3 tiles of label changes; features: fused margins, member agreement, link scores, duration priors) | +0.002–0.004 | 3 seeds, OOF-gated |
| 4 | **Link bagging**: 15 matchings vs 8 in graph + refiner | +0.001–0.003 | OOF-gated |

Sum of mid-estimates: **+0.007–0.018 OOF** → 0.943–0.954 OOF band; LB translation is noisy but a real
+0.005 OOF gain should read on the public LB and, more importantly, hold on the private half.

**Do NOT re-try:** public-family members in the fusion (run-Q evidence), cross-family flips (today's B2),
count-ridge variants (b4wa evidence), more 5–7-member label votes (saturation).

## 4. If quota must be spent before internal results exist

Acceptable information probes, in order (each bounded, mechanistically new):
1. F = weighted 6-vote {mv4, mv6, f4n}×2 + {gp, ys, hh}×1 (30 windows vs anchor) — tests diversity
   tie-breaking at exact internal ties. Expected ±0.001; only run if a slot would otherwise expire unused.
2. Nothing else. No flip variants, no count nudges, no public-file re-submissions.

## 5. Leaderboard context (Oct 8, 07:55 UTC)

| Rank | Team | Score | Gap to us |
|---|---|---:|---:|
| 1 | Anonym | 0.94535 | +0.0099 |
| 3 | Nicolas Krusche | 0.93849 | +0.0031 |
| 4 | Santiago Maniches | 0.93762 | +0.0022 |
| 5 | Mateo Allmer | 0.93661 | +0.0012 |
| **6** | **Koushik Rudra** | **0.93541** | — |
| 7 | Sameerk | 0.93527 | −0.0001 (active; will pass us without a real gain) |

Reaching #3–#5 requires the internal levers above; nothing externally available produces that delta
(today's evidence).
