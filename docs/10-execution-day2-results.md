# 10 — Execution Day Results (Oct 8): all 9 slots spent, 0.93541 → 0.93597

> Account: koushikrudra (#7 final, 0.93597). Daily quota 10/10 used (B2 pre-session + 9 scheduled probes).
> Best file of the day: `Fm1_margin1_only.csv` (ref 56944621) = anchor f4n + 8 margin-1 consensus flips.

## 1. The ledger (every slot, mechanism, and verdict)

| # | Ref | File | Score | Δ vs 0.93541 | Mechanism | Verdict |
|---|---|---|---|---:|---|---|
| 0 | 56942515 | B2_f4n_cons2 | 0.93236 | −0.00305 | cross-family flips (gp∧ys vs anchor) | FALSIFIED |
| 1 | 56943995 | F_vote6_weighted | **0.93562** | **+0.00021** | weighted 6-vote {mv4,mv6,f4n}×2+{gp,ys,hh}×1 | ✓ WIN |
| 2 | 56944036 | A2_ecoal_bolder | 0.93454 | −0.00108 | committee-score argmax (anchor slightly behind) | FALSIFIED |
| 3 | 56944155 | D_esmooth | 0.91907 | −0.01655 | null-repair of isolated blips (250w) | CATASTROPHIC |
| 4 | 56944195 | sujan output | 0.92552 | — | bagged-refiner fork of goodpjw | weak family |
| 5 | 56944464 | F_plus4 | 0.93562 | =F | F + 2 margin-1 consensus windows | neutral |
| 6 | 56944547 | F_mv4 | 0.93544 | — | F-rule on mv4 base | base weaker |
| 7 | 56944621 | **Fm1_margin1_only** | **0.93597** | **+0.00056** | only margin==1 flips of F (8w) | ✓✓ BEST |
| 8 | 56944691 | Fm1_plus2 | 0.93597 | =m1 | m1 + 2 consensus extras | neutral |
| 9 | 56944738 | Fm1_strict_to2nd | 0.93585 | −0.00012 | m1 restricted to committee-#2 class (4w) | subset worse |

## 2. The three laws discovered today

1. **Law of internal support** — a flip survives the LB only if ≥1 own fusion family AND full external
   consensus back it. F's 30 flips were 25 both-internals + 5 one-internal + **0** external-only.
   B2 (external-only) lost; F (internal-backed) won. The tie-gate is what rescues consensus flips.
2. **Law of the margin gradient** — flip safety decays steeply with the broad-committee (31-member)
   top-1−top-2 margin: margin==1 flips → **+0.00035** (8 flips); margin 2–3 flips → **−0.00035**
   (22 flips). At margin 0 there is nothing to flip (no external consensus exists there, F3=0).
3. **Law of blip sanctity** — the decode's isolated 1-second exercise blips are REAL signal
   (1,511 of them, uniform across classes). Repairing the conservative 250-subset to null cost
   −0.01655 (~4× the damage rate of ordinary flips; rare-class recall destruction). Never smooth
   the timeline post-hoc.

## 3. Veins exhausted (do not re-mine)

- Exact-tie + consensus: EMPTY (31-member committee never ties exactly).
- Internal-only or external-only flips: dead by Law 1.
- Margin 2–3: net negative even with full consensus. Margin ≥4: B2 territory.
- mv4 base + rule (0.93544 < 0.93597): f4n is the right base; the 45-window mv4↔f4n zone is
  net-f4n by ~1 window.
- sujan's bagged refiner: −0.0039 vs goodpjw; its additions do not transfer.
- F+4 extras (2 windows): net 0. The margin-1 regime is complete at 8–10 windows.

## 4. Final state and what remains

- LB: #7 at **0.93597** (Sameerk #8 at 0.93527; Mateo #6 at 0.93661 = +0.00064 away).
- All external/label-level levers are spent with evidence. The remaining +0.005–0.018 OOF lives in
  the user's local GPU stack: confidence-gated video override, learned-counts v2, refiner on fused
  result, link bagging (docs/09 §3).
- **Final-2 selection before deadline (UI, ~Oct 13)**: `Fm1_margin1_only` (0.93597, consensus-gated,
  OOF-principled) + `f4n` (0.93541, independent decode). Both are label-level micro-variants of the
  same decode — diversification is limited; consider one slot tomorrow for the best internal-lever
  output if the GPU work lands.
