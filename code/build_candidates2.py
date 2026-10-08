#!/usr/bin/env python3
"""Final candidate suite with 3-family consensus + risk stratification by own-committee support."""
import json, csv, os
import numpy as np

BASE = "/home/z/my-project/wear_research"
OUT = f"{BASE}/candidates"
os.makedirs(OUT, exist_ok=True)
N = 12234

def load_labels(path):
    rows = list(csv.reader(open(path, newline="")))
    ids = np.array([int(r[0]) for r in rows[1:]]); y = np.array([int(r[1]) for r in rows[1:]])
    o = np.argsort(ids); return ids[o], y[o]

def save_sub(path, ids, y):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["id", "target_feature"])
        for a, b in zip(ids.tolist(), y.tolist()):
            w.writerow([a, b])

def main():
    data = np.load(f"{BASE}/members/label_matrix.npz", allow_pickle=True)
    refs = data["refs"]; scores = data["scores"]; ys = data["ys"]
    meta = {m["ref"]: m for m in json.load(open(f"{BASE}/members/uniq_meta.json"))}
    ref2idx = {int(r): i for i, r in enumerate(refs)}

    ids, y_f4 = load_labels(f"{BASE}/members/own_subs/sub_56884414.csv")   # 0.93541 anchor
    _, y_mv4 = load_labels(f"{BASE}/members/own_subs/sub_56911837.csv")    # 0.93516 anchor (OOF 0.9364)
    _, y_gp = load_labels(f"{BASE}/members/goodpjw_out/submission.csv")
    _, y_ys = load_labels(f"{BASE}/members/yeashu_out/submission.csv")
    _, y_hh = load_labels(f"{BASE}/members/honghanhhh_out/submission.csv")
    z = np.load(f"{BASE}/members/goodpjw_out/final_probabilities.npz")
    p_gp = z["test"].astype(np.float64)
    srt = np.sort(p_gp, 1); gp_margin = srt[:, -1] - srt[:, -2]

    print("== family disagreements vs f4n anchor ==")
    for nm, v in [("gp", y_gp), ("ys", y_ys), ("hh", y_hh)]:
        print(f"  {nm}: {(v != y_f4).sum()} ({(v != y_f4).mean()*100:.2f}%)  vs mv4: {(v != y_mv4).sum()}")

    # own committee (16 best, Oct6+)
    committee_refs = [int(r) for r in refs
                      if meta[int(r)]["score"] and float(meta[int(r)]["score"]) >= 0.9330
                      and meta[int(r)]["date"] >= "2026-10-06"]
    C = np.array([ys[ref2idx[r]] for r in committee_refs])
    print(f"own committee: {len(committee_refs)} members")

    def committee_support(flip_mask, flip_cls):
        """how many own-committee members support the flip class at those windows"""
        sup = np.zeros(N, dtype=np.int16)
        for c in range(19):
            m = flip_mask & (flip_cls == c)
            if m.any():
                sup[m] = (C[m] == c).sum(1)
        return sup

    # ---------- consensus definitions vs f4n anchor ----------
    cons2 = (y_gp == y_ys) & (y_gp != y_f4)                       # gp+ys
    cons3_2of3 = (((y_gp == y_ys) | (y_gp == y_hh) | (y_ys == y_hh)) &
                  ((y_gp != y_f4) | (y_ys != y_f4) | (y_hh != y_f4)))
    # careful: require the majority class != anchor and majority agrees on ONE class
    maj_cls = np.where(y_gp == y_ys, y_gp, np.where(y_gp == y_hh, y_gp, y_ys))
    n_agree = ((y_gp == maj_cls).astype(int) + (y_ys == maj_cls).astype(int) + (y_hh == maj_cls).astype(int))
    cons3 = (n_agree >= 2) & (maj_cls != y_f4)

    print(f"\nconsensus gp==ys != f4n:            {cons2.sum()}")
    print(f"consensus 2-of-3{{gp,ys,hh}} != f4n:  {cons3.sum()}")

    sup2 = None
    if cons2.any():
        sup_full = np.zeros(N, dtype=np.int16)
        for c in range(19):
            m = cons2 & (y_gp == c)
            if m.any():
                sup_full[m] = (C[:, m] == c).sum(0)
        sup2 = sup_full
        for lo, hi, tag in [(0, 0, "sup0"), (1, 1, "sup1"), (2, 3, "sup2-3"), (4, 99, "sup4+")]:
            mm = cons2 & (sup_full >= lo) & (sup_full <= hi)
            print(f"    stratum {tag}: {mm.sum()} flips")

    # ---------- build candidates ----------
    # B2: f4n + all cons2 flips
    y = y_f4.copy(); y[cons2] = y_gp[cons2]
    save_sub(f"{OUT}/B2_f4n_cons2.csv", ids, y)

    # B2-safe: only flips with >=1 own-committee supporter OR high gp margin
    mq = np.quantile(gp_margin[cons2], 0.35) if cons2.any() else 0
    safe = cons2 & ((sup_full >= 2) | (gp_margin >= mq))
    y = y_f4.copy(); y[safe] = y_gp[safe]
    save_sub(f"{OUT}/B2s_f4n_cons2_safe.csv", ids, y)
    print(f"\nB2-safe flips (sup>=2 or top-65% margin): {safe.sum()}")

    # B1: mv4 anchor + cons2 flips (recompute vs mv4)
    cons2m = (y_gp == y_ys) & (y_gp != y_mv4)
    y = y_mv4.copy(); y[cons2m] = y_gp[cons2m]
    save_sub(f"{OUT}/B1_mv4_cons2.csv", ids, y)
    print(f"B1 flips vs mv4: {cons2m.sum()}")

    # B3: f4n + 2-of-3 flips (bolder)
    y = y_f4.copy(); y[cons3] = maj_cls[cons3]
    save_sub(f"{OUT}/B3_f4n_cons3_2of3.csv", ids, y)
    print(f"B3 flips (2-of-3): {cons3.sum()}")

    # E: B2 + gp-solo high-margin flips (top quantile of disagreement margins), excluding cons2 windows
    dis_gp = (y_gp != y_f4) & (~cons2)
    if dis_gp.any():
        thr = np.quantile(gp_margin[dis_gp], 0.70)  # top 30% margin
        solo = dis_gp & (gp_margin >= thr)
    else:
        solo = np.zeros(N, dtype=bool)
    yE = y_f4.copy(); yE[cons2] = y_gp[cons2]; yE[solo] = y_gp[solo]
    save_sub(f"{OUT}/E_f4n_cons2_gpsolo.csv", ids, yE)
    print(f"E: cons2 + gp-solo top-30% margin flips: {solo.sum()} (total changes {(yE != y_f4).sum()})")

    # F: committee-weighted fusion on mv4? skip. Instead: CONSENSUS-VOTE composite:
    #    majority vote of {mv4, mv6, f4n, gp, ys, hh} with weights {2,2,2,1,1,1}
    _, y_mv6 = load_labels(f"{BASE}/members/own_subs/sub_56911845.csv")
    M = np.stack([y_mv4, y_mv6, y_f4, y_gp, y_ys, y_hh])
    W = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
    S = np.zeros((N, 19))
    for i in range(6):
        S[np.arange(N), M[i]] += W[i]
    yF = S.argmax(1)
    # tie-break toward the strongest own member's label
    tie = (S[np.arange(N), yF] == S.max(1) - 1e-9)
    yF2 = np.where(S[np.arange(N), y_f4] >= S[np.arange(N), yF] - 1e-9, y_f4, yF)
    save_sub(f"{OUT}/F_vote6_weighted.csv", ids, yF2)
    print(f"F weighted vote changes vs f4n: {(yF2 != y_f4).sum()}  vs mv4: {(yF2 != y_mv4).sum()}")

    # ---------- summary ----------
    print("\n== candidate change-matrix vs anchors ==")
    import glob
    for f in sorted(glob.glob(f"{OUT}/[BEF]*.csv")):
        _, yv = load_labels(f)
        nm = os.path.basename(f)
        print(f"  {nm:32s} dF4n={(yv != y_f4).sum():5d} dMv4={(yv != y_mv4).sum():5d} dGp={(yv != y_gp).sum():5d} dYs={(yv != y_ys).sum():5d} dHh={(yv != y_hh).sum():5d}")

if __name__ == "__main__":
    main()
