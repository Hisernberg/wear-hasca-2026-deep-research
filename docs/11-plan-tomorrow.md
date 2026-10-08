# 11 — Tomorrow's Plan (Oct 9): the 10-slot ladder to rank higher

> Current: **#7 at 0.93597** (`Fm1_margin1_only`, ref 56944621). Quota resets to 10 at UTC midnight.
> Deadline ~Oct 13 (verify on competition page). Final-2 selection is MANUAL in the UI.
> **Hard truth from today's evidence: every label-level lever is closed (M2 committee-refresh
> produced 0 new flips). Tomorrow's rank is decided by tonight's local GPU work — the 10 slots
> are only the delivery vehicle.**

## 0. What today's learning dictates (the targeting brief)

The 8 winning margin-1 flips were **all exercise↔rest boundary corrections**:
`situp-cx→null, null→situp, jog→jog-butt, null→burpee, pushup-cx→null, situp→null, null→jog-butt, str-lung→null`.

⇒ Video evidence is most decisive at null boundaries; the refiner's ±3-tile band is exactly where
the LB pays. **Every lever below must prioritize null-boundary windows.**

Three laws to respect (docs/10):
1. Flips need ≥1 internal family + full external consensus (never external-only).
2. Margin gradient: committee margin 1 = safe; margin 2–3 = net harm; margin ≥4 = B2 territory.
3. Blips are real signal — never remove isolated 1-second exercise tiles (−0.01655 lesson).

## 1. TONIGHT (user's machine, GPU, start by 22:00 — runs take 6–8 h)

Ordered by expected Δ-OOF; every lever is OOF-gated before shipping (gate: subject-balanced OOF
must improve ≥ +0.0005; per-class checks must be positive).

| Lever | Recipe | Expected Δ-OOF | Ship file |
|---|---|---:|---|
| **L1 video override** (targeted) | ≥2 run video-heads; flip ONLY fused low-margin windows (margin < θ) at null-boundaries where video-heads consensus class c has margin > τ; sweep θ∈{0.1,0.2,0.3}, τ∈{0.3,0.5,0.7} on OOF; per-class positive gate | +0.002–0.006 | `V1.csv` |
| **L2 counts v2** | quantile count targets (p25/p75) → two Sinkhorn passes averaged; learned null-floor per subject; gate: count MAE < 9.0 on member OOF | +0.002–0.005 | `C1.csv` |
| **L3 refiner on fused result** | LightGBM flip classifier on ±3-tile band; features: fused margins, member agreement, link scores, duration priors; 3 seeds; null-boundary upweighting | +0.002–0.004 | `R1.csv` |
| **L4 link bagging** | 15 perturbed matchings vs current 8, feeding the same decode | +0.001–0.003 | `B1.csv` |
| **S1 full stack** | f4n decode → L1 → L2 → L3 → L4, OOF-gate at each step (abort the step if gate fails) | +0.005–0.018 | `S1.csv` |

Do NOT re-try (evidenced dead): public-family members in fusion (run-Q), cross-family label flips
(B2), count-ridge post-hoc (b4wa), 5–7-member label votes (saturation), any smoothing/null-repair
(D_esmooth), margin≥2 consensus flips (today's F_m23 read).

## 2. MORNING RECON (08:00–09:00, no slots)

1. LB movement: Sameerk (0.93527, active) / GALABA / Mateo deltas; recompute gaps.
2. New public kernels scan (sujan appeared mid-day today — check for 0.936+ forks; if found,
   read its output — 1 slot max, replaces ladder slot 6).
3. Pull user's overnight files; run `python scripts/validate_any.py <file>` on each (format +
   id-order + flip-class report). Report OOF deltas; rank files by gated OOF gain.
4. Refresh committee with any new own outputs → recompute margin-1 set (today: 0 new flips, but a
   new GPU decode changes the committee fundamentally).

## 3. THE 10-SLOT LADDER (adaptive; every slot validated + logged via `slot_run.py`)

| Slot | Submission | Condition / gate | Why this order |
|---|---|---|---|
| 1 | **S1** full stack | OOF ≥ +0.005 | lock the big gain early; if S1 missing → V1 |
| 2 | **V1** video override | OOF-gated | most independent lever; isolates its LB value |
| 3 | **C1** counts v2 | OOF-gated | second independent read |
| 4 | **R1** refiner | OOF-gated | third read; boundary-focused |
| 5 | **S2** re-tuned stack | built from slots 1–4 LB reads: keep only levers whose solo read ≥ their OOF promise; re-compose with tightened gates | the composition the evidence actually supports |
| 6 | **B1** link bagging (or new-kernel read if recon found one) | OOF-gated | last individual lever |
| 7 | τ-neighbor of day's best (weaker gate, e.g. θ+0.1) | only if its parent won | parameter bracket around the winner |
| 8 | τ-neighbor of day's best (stronger gate, e.g. θ−0.1) | only if its parent won | bracket other side; pick bracket max |
| 9 | **Best pair composition** (e.g. V1+R1) if two singles won | LB-evidence subset | levers may stack sub-additively; read it |
| 10 | **Closer**: highest-OOF file of the day not yet submitted, or S3 = day's winning config re-composed at bracket-optimal τ | — | end the day at the max |

Decision rules between slots:
- If a solo lever beats S1's implied contribution → composition gates were mis-tuned; rebuild S2
  with that lever's solo-optimal τ (this becomes the new flagship).
- If S1 ≤ 0.93597 and all solos ≤ 0.93597 → the OOF→LB translation failed; STOP stacking, fall back
  to bracket reads on the single best-OOF lever (slots 5–10 become its τ-sweep), keep final-2 =
  F_m1 + f4n.
- Never submit a file that fails `validate_any.py` or lacks an OOF report.

## 4. FALLBACK LADDER (only if NO GPU file exists by 12:00)

Slots do not roll over — but spending them on evidenced-dead mechanisms cannot raise the rank.
Ranked residue:
1. M2-style committee refresh **if any new own file appeared** (today it gave 0 flips — needs a new member to move).
2. Semi-consensus micro-read: margin-1 windows where gp∧ys agree, hh disagrees, ak breaks the tie (volume ~2–5, expected ±0.0002 — noise).
3. Hold remaining slots for late-day GPU files; if the day ends with none, accept 0.93597 —
   **a wasted slot costs nothing; a wrong submission costs nothing either (best is auto-kept), but noise probes add no information beyond today's closure.**

## 5. RANK MATH (what each outcome buys)

| Target | Gap from 0.93597 | Needed |
|---|---:|---|
| Mateo #6 (0.93661) | +0.00064 | any single lever translating at ~30 % |
| GALABA #5 (0.93694) | +0.00097 | one full lever (L1 or L3) |
| Santiago #4 (0.93766) | +0.00169 | two levers composing |
| Krusche #3 (0.93849) | +0.00252 | S2 stack mid-range |
| ANMOL #2 (0.94355) | +0.00758 | full stack top-range + luck |
| Anonym #1 (0.94535) | +0.00938 | everything lands at upper band |

Realistic band tomorrow: **#4–#6**; stretch #3; #2 requires the full +0.018 OOF tail.

## 6. Final-2 discipline (before deadline, UI)

- Primary: best OOF-gated file of the final day; Secondary: `f4n` (independent decode, 0.93541).
- OOF over public LB for the private half (public = 4 subjects only).
- Re-verify selection the last evening; Kaggle does NOT auto-pick optimally for private if you
  have multiple close scores.

## 7. Automation ready now (no user action needed)

- `scripts/validate_any.py` — instant format + flip-class audit for any file you produce.
- `scripts/slot_run.py` — validate → submit → poll → ledger (used for all 9 slots today).
- Candidate builders + toolkit under `scripts/` and repo `code/`.
- Ledger of all 10 today's submissions: `wear_research/slot_ledger.json`.
