#!/usr/bin/env python3
"""Probe D: decode-override correction (align anchor to unanimous committee where decode deviated)
+ probe F (diversity tie-break) volumes + final ladder decision data."""
import json, csv
import numpy as np

BASE = "/home/z/my-project/wear_research"
OUT = f"{BASE}/candidates"
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

    ids, y_f4 = load_labels(f"{BASE}/members/own_subs/sub_56884414.csv")
    _, y_mv4 = load_labels(f"{BASE}/members/own_subs/sub_56911837.csv")
    _, y_gp = load_labels(f"{BASE}/members/goodpjw_out/submission.csv")
    _, y_ys = load_labels(f"{BASE}/members/yeashu_out/submission.csv")
    _, y_hh = load_labels(f"{BASE}/members/honghanhhh_out/submission.csv")

    # committee of RAW decodes: use all own members 0.930+ from Oct 2-7 (broad, 25 members)
    broad = [int(r) for r in refs if meta[int(r)]["score"] and float(meta[int(r)]["score"]) >= 0.930]
    C = np.array([ys[ref2idx[r]] for r in broad])
    print(f"broad own committee: {len(broad)} members (0.930+)")

    counts = np.zeros((N, 19), dtype=np.int16)
    for c in range(19):
        counts[:, c] = (C == c).sum(0)
    c_top = counts.argmax(1)
    srt = np.sort(counts, 1)[:, ::-1]
    margin = srt[:, 0] - srt[:, 1]
    K = len(broad)

    # windows where the FINAL decode (f4n anchor) deviates from committee-top
    dev = y_f4 != c_top
    print(f"\ndecode deviates from committee-top on {dev.sum()} windows")
    for lo, hi, tag in [(K - 1, K, "unanimous-top"), (K - 3, K - 2, "near-unanimous"),
                        (K // 2, K - 4, "majority"), (0, K // 2 - 1, "minority")]:
        m = dev & (margin >= lo) & (margin <= hi)
        print(f"   committee support {tag} [{lo}-{hi}]: {m.sum()} windows")

    # PROBE D: align anchor to committee-top where committee is (near-)unanimous (margin >= K-3)
    dmask = dev & (margin >= K - 3)
    yD = y_f4.copy(); yD[dmask] = c_top[dmask]
    save_sub(f"{OUT}/D2_f4n_committee_override_fix.csv", ids, yD)
    print(f"\n[D2] committee-override corrections: {dmask.sum()} windows")
    diff_vs_b2 = (yD != y_f4).sum()
    print(f"    total changes vs anchor: {diff_vs_b2}")

    # variant D2-tight: only unanimous (margin >= K-1)
    dmask_t = dev & (margin >= K - 1)
    yDt = y_f4.copy(); yDt[dmask_t] = c_top[dmask_t]
    save_sub(f"{OUT}/D2t_f4n_unanimous_only.csv", ids, yDt)
    print(f"[D2t] unanimous-only corrections: {dmask_t.sum()} windows")

    # same probe on mv4 anchor
    dev_m = y_mv4 != c_top
    dmask_m = dev_m & (margin >= K - 3)
    yDm = y_mv4.copy(); yDm[dmask_m] = c_top[dmask_m]
    save_sub(f"{OUT}/D2m_mv4_committee_override_fix.csv", ids, yDm)
    print(f"[D2m] mv4 version corrections: {dmask_m.sum()} windows")

    # what do public families say at D2 windows? (informational)
    if dmask.sum():
        gp_agree_decode = (y_gp[dmask] == y_f4[dmask]).mean()
        ys_agree_decode = (y_ys[dmask] == y_f4[dmask]).mean()
        print(f"    at D2 windows: gp agrees with decode {gp_agree_decode*100:.0f}%, ys {ys_agree_decode*100:.0f}%")

    # F candidate stats recap
    S = np.zeros((N, 19))
    W = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
    M = np.stack([y_mv4, ys[ref2idx[56911845]], y_f4, y_gp, y_ys, y_hh])
    for i in range(6):
        S[np.arange(N), M[i]] += W[i]
    yF = S.argmax(1)
    yF2 = np.where(S[np.arange(N), y_f4] >= S[np.arange(N), yF] - 1e-9, y_f4, yF)
    print(f"[F] weighted-vote changes vs f4n: {(yF2 != y_f4).sum()}")

if __name__ == "__main__":
    main()
