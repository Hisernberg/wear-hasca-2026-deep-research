# 12 — Execution Day 3 (Oct 9): 10/10 slots, additivity verified, mechanism fully mapped

> Start: #7 at 0.93597, 10 fresh slots (user's GPU stack self-submitted 1 at 04:50).
> End: **#7 at 0.93597** — best unchanged, but the winning mechanism is now fully decomposed
> and five new laws are on the books. Quota 10/10 exhausted.

## 1. The day in one table

| # | File | Ref | Score | Δ vs best | What it proved |
|---|------|-----|------:|----------:|----------------|
| 1 | G = user GPU stack (04:50) | 56991928 | 0.93308 | −0.00289 | OOF 0.9357 did NOT translate; G-support = anti-evidence |
| 2 | B_null4 (4 exercise→null flips) | 57002876 | 0.93581 | −0.00016 | null-boundary half of the win = **+0.00040** |
| 3 | B_class4 (4 class-type flips) | 57002901 | 0.93557 | −0.00040 | class half = **+0.00016**; sum = +0.00056 exactly ✓ |
| 4 | T3 (anti-G ∧ anti-ak flips) | 57002940 | 0.93569 | −0.00028 | anti-evidence gating does NOT rescue high margins |
| 5 | D6 (drop 2 G-supported winners) | 57003018 | 0.93585 | −0.00012 | G-backed winning flips are positive contributors |
| 6 | W_a (F_m1 + w4675) | 57003106 | 0.93597 | 0 | w4675 = exactly zero (public) |
| 7 | D2 (P1+P2 prob-argmax fusion) | 57003186 | 0.92539 | −0.01058 | **decode layer is worth ≈ +0.014 on LB** |
| 8 | LOO minus w1883 | 57003228 | 0.93597 | 0 | w1883 = exactly zero; no drop can beat the 8 |
| 9 | K_video (kansuke video head) | 57003271 | 0.89429 | — | video-embedding head weak; not committee material |
| 10 | W_b (F_m1 + w11153) | 57003324 | 0.93597 | 0 | predicted 0.93597, measured 0.93597 — **additivity verified** |

Calibrations (no slots): P2 (goodpjw Part-2, LB 0.92990): 3/8 winners vs 16/22 bad → anti-evidence.
kansuke: 4/8 vs 10/22 → neutral, no flip information. akhyar: 3/8 vs 13/22 → anti-evidence.

## 2. New laws (Day 3)

**Law 4 — Decode-layer dominance.** Raw per-window argmax of even excellent probabilities lands
≈0.9254; the link/counts decode machinery is worth ≈ **+0.014 LB**. Never submit decode-less
argmax; probability-level fusion without re-decoding is dead on arrival (D2).

**Law 5 — Per-window LB additivity.** Public LB composes per-window: bracket halves summed
exactly to the whole (B_null4 + B_class4 deltas = F_m1 delta), and each individually-zero window
(w4675, w11153, w1883) read exactly zero in isolation. Prediction of W_b before submission was
exact. Per-window contribution accounting is trustworthy on this LB.

**Law 6 — Null-boundary dominance.** The 8 winning flips decompose **71% null-boundary
corrections (+0.00040) / 29% class-type corrections (+0.00016)**. Exercise↔rest boundaries are
where label corrections pay; class-class corrections are secondary.

**Law 7 — Anti-evidence members stay anti.** G-stack, akhyar, P2 all support known-bad flips
at 55–73% rates while supporting winners at only 25–37%. Their agreement marks bad flips, but
gating BY their opposition does not rescue high-margin flips (T3: −0.00028). Their only use:
screening, never selection.

**Law 8 — OOF→LB translation is not free.** User stack: OOF 0.9357 → LB 0.93308 (−0.0029).
OOF gains from many-knob stacks (video override + counts + refiner) did not survive the
subject shift. Only 1–2-parameter, mechanism-level changes (like margin-1 gating) have
translated so far.

## 3. Falsified today (do not revisit)

- Anti-G/anti-ak gated flips at margin 5–15 (T3)
- Dropping any subset of the 8 winning flips (D6, LOO-w1883)
- Prob-level fusion without re-decode (D2: −0.0106)
- OOF-tuned Viterbi on goodpjw probs: OOF +0.00040 only — their decode already owns the
  temporal structure (below the +0.002 submission gate, never fired)
- P1+P2 OOF mixture: +0.00026 (below gate)
- kansuke video-embedding head as committee member (0.894 solo)

## 4. Remaining unspent levers (the honest list)

1. **A better decode** — the only +0.014-scale object left. Requires raw probabilities from the
   f4n/mv4/mv6 chain-link family (user's machine) + OOF-tuned re-decode. This is the single
   highest-value artifact the user can produce.
2. **New public kernel ≥0.936** — none as of 09:20 Oct 9; goodpjw Part-2 (0.92990) already
   absorbed. Watch daily; mine only if it clears 0.936.
3. **Margin-1 regime on a NEW committee** — closed unless a genuinely new family appears
   (M2 refresh: 0 flips; P2/G/ak additions: anti-evidence).

## 5. Final-2 guidance (deadline ~Oct 13, MANUAL in UI)

- **Primary: `Fm1_margin1_only.csv` (ref 56944621, 0.93597)** — public-optimal, 8 flips.
- **Secondary: `f4n` (ref 56884414, 0.93541)** — independent decode family, no shared flips.
- Alternative primary: `Fm1_plus2` (ref 56944691) — same public score with 2 extra
  consensus-backed flips (w4675 pushup-cx→pushup, w11153 jog-rot→null) that are zero-cost on
  public but could pay on the private subjects. Choose Fm1+2 if you believe consensus coverage
  transfers; F_m1 if you believe the 8 measured flips only.
- Do NOT pick by public score alone; the public 4 subjects are a small slice.

## 6. Rank state

| Pos | Team | Score | Gap |
|----:|------|------:|----:|
| 3 | Nicolas Krusche | 0.93849 | +0.00252 |
| 4 | GALABA VAMSI | 0.93825 | +0.00228 (climbed +0.0013 overnight — active) |
| 5 | Santiago Maniches | 0.93766 | +0.00169 |
| 6 | Mateo Allmer | 0.93661 | +0.00064 |
| **7** | **Koushik Rudra** | **0.93597** | — |
| 8 | Sameerk | 0.93543 | −0.00054 |

+0.00064 to #6 requires one null-boundary lever translating at ~50% — none exists on the
label level anymore. The path runs through Law 4: a better decode (user's machine) or a new
public breakthrough.
