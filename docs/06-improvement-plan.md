# 06 — Improvement Plan (top-notch, multi-track)

> Framing: the July 5 ranking freeze makes this plan serve three goals — (G1) the **arXiv technical report**
> (host offers endorsement), (G2) **dominance in the remaining public-LB window** and any private-LB selection
> we still control, (G3) **the 4th WEAR challenge / any future multimodal HAR competition**.
> Each item: [expected Δ OOF] [effort] [risk]. Order inside tracks = build order.

## Track A — Algorithmic / non-LLM (the points live here)

### A1. Port the six documented increments from the 0.92+ stack [Δ +0.02–0.03] [Low] [Low]
Straight ports, each individually OOF-gated (in impact order):
1. **L3 links**: two-tower matcher (BiGRU tails/heads, contrastive over 512-tile blocks) + **SSL per-subject
   ridge** + 2 seeds → exact-successor 33→66% path is mapped; [Δ +0.0098 alone].
2. **Link bagging**: clean + 7×Gumbel(τ=0.3) matchings, geometric mean of decoded graphs [+0.0014–0.004].
3. **kNN leans on links** (embedding smoothed along links, g=1.5) [+0.0018].
4. **Learned count prior** (sorted log-margin profile → RidgeCV counts, clip 70–135, LOFO, 2–3 passes)
   [+0.006; oracle shows +0.013 available].
5. **Pseudo-label adaptation of tabular expert T** (cross-fitted, weight 3.0) [+0.0039].
6. **Chain boundary refiner** (LGBM flip at label changes, ±3 tiles, ~37 features, 3–4 seeds) [+0.003].

### A2. Attack the exact-successor ceiling (66% → 75%+) [Δ +0.01–0.02] [Med] [Med]
Every +1% exact-successor ≈ +0.01–0.02 OOF. Ideas beyond the public stack:
- **Candidate generation**: add (a) sensor/limb-aware extrapolation (per-limb velocity-matched candidates,
  not just same-limb), (b) IMU-jerk continuity candidates, (c) cluster-anchored candidates (anchor = cluster
  centroid of the last frames of bout-shaped runs).
- **Loss**: replace binary CE with **listwise LambdaRank within candidate sets** (yeashu evidence) + the
  runner-up margin as a confidence feature.
- **Global assignment**: after Hungarian, run a **local 2-opt swap pass** on low-margin links (swap
  successors of i and j if both margins improve) — cheap, deterministic.
- **Self-supervised pretraining of the towers** on *all* train recordings (incl. subject 0/14 duplicates)
  with the within-tile→cross-tile frame mapping pretext (goodpjw's SSL generalized).

### A3. Boundary localization program [Δ +0.003–0.008] [Med] [Med]
"1/3 of errors lie within 1 s of a true boundary" — the main open problem:
- **Boundary-aware window labels**: train window model with soft targets at straddling windows
  (fraction-of-window label mixture) instead of majority label — directly improves boundary logits.
- **Consensus refiner v2**: features from all 8–15 hypotheses (cross-hypothesis support as features, not
  votes — yeashu), plus duration-prior features (expected bout-length distributions per class from train),
  plus Sinkhorn-sensitivity features (does flipping change column sums?).
- **Boundary-specific mini-decoder**: for flagged boundary clusters, run a tiny DP over the ±5 tiles with
  transition penalties learned from train label runs (mini-Viterbi inside the refiner).

### A4. Count model v2 [Δ +0.003–0.006] [Med] [Low]
- Add features: per-class **duration percentiles** of the subject's current runs, bout-shape statistics,
  link-consistency per class, per-class video-cluster sizes (label-free), sensor-mix per class.
- Replace point regression with **quantile regression** (predict p10/p50/p90 of counts) and feed the
  Sinkhorn targets as soft ranges (two Sinkhorn runs at p25/p75, average) — robustness for test subjects
  whose true counts fall outside train support.
- **Sanity prior**: total non-null tiles per subject ≈ session exercise time (protocol); null floor as
  learned per-subject share (0.09–0.66 observed spread!) rather than fixed 5%.

### A5. Oracle-late-fusion attack (the unclaimed headroom) [Δ +0.005–0.015, unbounded upside] [Med] [Med]
WEAR paper: O-LF(I,C) = 91.26 vs realizable ~75–93 (ours). Mechanisms:
- **Confidence-gated late fusion**: per window, estimate per-modality reliability (window-model entropy per
  head, kNN agreement per modality, neighbor variance); fuse `logp = w_i(s)·imu + w_v(s)·video` with
  per-sample weights, w's from a small OOF-trained gate net.
- **Greedy anchored voting** (our working definition, see 03 §3.5): inertial+timeline pipeline = anchor;
  a video-side specialist may override only when ≥2 independent video models agree with margin > τ.
  τ tuned on OOF; expected to help exactly on visually-distinct classes (jogging variants, burpees).
- **Video→IMU distillation** (COMODO-style, label-free): train the IMU tower to match video kNN geometry —
  merges the modality strengths without needing raw video.

### A6. Window-model context upgrade (DeepConvContext-style) [Δ +0.005–0.01 window-level] [Med] [Med]
Replace the single-window Net with a window-sequence model for the *train* side (process windows in
protocol order with dilated temporal context), keeping test inference per-window (links carry context there).
Same authors, +5% F1 avg on 6 benchmarks — this is the legitimate window-side capacity upgrade.

### A7. Limb-conditional modeling (SHL transfer) [Δ +0.002–0.005] [Low] [Low]
- Per-limb specialist heads (we already feed limb embedding — go further: per-limb output calibration
  vectors, per-limb class-prior logit adjustments tuned on OOF).
- Train-time **data-side limb pairing** (Signal Sleuths: UL-pairing +0.6 over LR-swap): pair left/right
  windows of the same timestamp as augmentation views.
- Axis-inversion / rotation augmentation ×2 stronger for leg-placed windows (3KA: feature–location
  interaction).

### A8. Semi-supervised loop closure [Δ +0.002–0.004] [Low] [Low]
Current graph self-training propagates labels; extend: (1) feed final calibrated pseudo-labels back as
weights for one extra window-model fine-tune epoch (careful: fold-clean, weight 0.3, 1 epoch only);
(2) use pseudo-labeled *test* tiles (transductive) to refit tabular expert T (goodpjw already does the OOF
half — add the test half with weight 1.5–3.0).

## Track B — LLM-enabled (semantics & meta, where LLMs have evidence)

### B1. LLM decision-level late fusion [Δ unknown, orthogonal] [Med] [Med]
Demirel et al. (OpenReview) show LLMs can perform **late fusion of sensor modality predictions**. Concrete
recipe: serialize per-window compact evidence (top-3 classes + margins per modality head, limb id, session
position, link confidence) → LLM returns a distribution-adjusted label with rationale. Practical mode:
**batch offline for the report** (not per-window latency-critical), or as the tie-breaker on low-margin
windows only (top-5% uncertain). Risk: LLM inconsistency — measure on OOF before shipping anything.

### B2. Class-semantics layer [Δ +0.001–0.003] [Low] [Low]
19 class names have structure (null / jogging ×5 / stretching ×5 / push-ups ×2 / sit-ups ×2 / burpees /
lunges ×2 / bench-dips). Use text embeddings of class names to (a) define **hierarchical label smoothing**
(within-family smoothing > cross-family), (b) initialize per-class logit biases from family priors,
(c) group-wise calibration (calibrate family totals then within-family splits — fits Sinkhorn naturally).

### B3. LLM as meta-engineer (this repo's own method) [process] [Low] [—]
Mining discussions/logs/notebooks, generating refiner feature candidates, writing the technical report
draft, generating unit tests for the pipeline (shape/label asserts caught 3 ecosystem bugs cited in 05).

### B4. VLM pseudo-label bridge (limited feasibility) [Δ unknown] [High] [High]
We hold VideoMAE *features*, not raw video — full VLM captioning is out. Feasible slice: cluster video
features per subject, name clusters via nearest-neighbor transfer from confident windows, use cluster
pseudo-labels as additional graph seeds. Only pursue if A-track items plateau.

## Track C — Engineering & reproducibility

| Item | Why |
|---|---|
| Manifest next to every submission (git SHA, seeds, run list incl. skipped, config dump, OOF F1) | the time-guard nondeterminism incident |
| Regenerate `QN_REF` inside each run | qnorm coupling risk |
| Per-run (not per-queue) precision fallback | fp16 incident |
| `WEAR_SMOKE` end-to-end CI on 5 subjects, asserted shapes/labels | stmugiwara-class bugs are embarrassing and avoidable |
| Prepared-input inference cache (jiweiliu) with `max_abs < 1e-5` verification | submission runtime safety |
| Artifact versioning: pin dataset version; archive train `.npy` copies | official dataset drift |
| Selection protocol: pick private-LB candidate by **subject-balanced OOF**, not public LB | public/private split is by rows; OOF by subjects is the safer proxy |

## Track D — Competition-meta & publication

1. **Technical report skeleton** (HASCA 2-col, ≤6 pp + refs): problem → data quirks we found (label lags,
   null tails, video bug) → pipeline (window model + timeline reconstruction + counts + refiner) → ablation
   table (the OOF ladder IS the ablation) → honest CV-LB divergence notes → limits (single random limb,
   count prior brittleness).
2. **arXiv preprint** — host explicitly offers endorsement on request; do this for G1/G3 credibility.
3. **Public code release** (this repo's companion `wear-timeline` repo): cleaned, tested, single-command
   pipeline — visibility for the 4th challenge.
4. **Watch the host's next-year redesign** (their thread: "exploring how we could change things for next
   year") — likely to close the protocol exploits (count priors); design the count model to degrade
   gracefully when counts are unknown.

## Priority matrix (build order)

| Prio | Item | Track | Δ (OOF) | Effort |
|---|---|---|---:|---|
| 1 | A1 six increments port | A | +0.02–0.03 | Low |
| 2 | A2 successor-ceiling attack | A | +0.01–0.02 | Med |
| 3 | A4 count model v2 | A | +0.003–0.006 | Med |
| 4 | A3 boundary program | A | +0.003–0.008 | Med |
| 5 | A5 oracle-LF attack | A | +0.005–0.015 | Med |
| 6 | A6 context window model | A | +0.005–0.01 (window) | Med |
| 7 | A7/A8 limb-conditional + SSL loop | A | +0.004–0.009 | Low |
| 8 | B2 class-semantics layer | B | +0.001–0.003 | Low |
| 9 | C engineering hardening | C | process | Low |
| 10 | B1 LLM late fusion / B4 VLM bridge | B | unknown | Med–High |

Cumulative realistic target: **0.930 → 0.940–0.945 OOF** (= top-3 public-LB territory) with items 1–5 alone.
