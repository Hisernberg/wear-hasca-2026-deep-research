#!/usr/bin/env python3
"""Master analysis: parse all submissions, dedupe, agreement structure, goodpjw OOF calibration."""
import os, json, csv, glob
import numpy as np

BASE = "/home/z/my-project/wear_research"
N_TEST = 12234

def load_labels(path):
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    ids = np.array([int(r[0]) for r in rows[1:]])
    y = np.array([int(r[1]) for r in rows[1:]])
    order = np.argsort(ids)
    return ids[order], y[order]

def macro_f1(y_true, y_pred, n_cls=19):
    f1s = []
    for c in range(n_cls):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(f1s)), f1s

def main():
    subs = json.load(open(f"{BASE}/kaggle/subs_full_157.json"))
    meta = {s["ref"]: s for s in subs}

    # ---- parse all valid own submissions ----
    members = []   # (ref, score, date, desc, y)
    for f in sorted(glob.glob(f"{BASE}/members/own_subs/sub_*.csv")):
        ref = int(f.split("_")[-1].split(".")[0])
        try:
            _, y = load_labels(f)
        except Exception:
            continue
        if len(y) != N_TEST:
            continue
        m = meta.get(ref, {})
        members.append(dict(ref=ref, score=m.get("publicScoreNullable"), date=m.get("date", ""),
                            desc=m.get("descriptionNullable") or "", y=y))
    print(f"own members parsed: {len(members)}")

    # ---- dedupe ----
    seen, uniq = {}, []
    for m in members:
        key = m["y"].tobytes()
        if key in seen:
            seen[key].append(m["ref"])
        else:
            seen[key] = [m["ref"]]
            uniq.append(m)
    print(f"unique label vectors: {len(uniq)}")

    # ---- public members ----
    _, y_ys = load_labels(f"{BASE}/members/yeashu_out/submission.csv")
    _, y_gp = load_labels(f"{BASE}/members/goodpjw_out/submission.csv")
    _, y_gp_pub = load_labels(f"{BASE}/members/goodpjw_out/submission_public_pipeline.csv")

    # ---- goodpjw npz ----
    z = np.load(f"{BASE}/members/goodpjw_out/final_probabilities.npz", allow_pickle=True)
    p_oof, oof_labels, p_test = z["oof"], z["oof_labels"], z["test"]
    f1_oof, _ = macro_f1(oof_labels, p_oof.argmax(1))
    print(f"\ngoodpjw argmax(p_oof) vs oof_labels: macro F1 = {f1_oof:.5f}")
    agree_argmax = (p_test.argmax(1) == y_gp).mean()
    print(f"goodpjw submission vs argmax(p_test) agreement: {agree_argmax:.4f}")
    agree_pub = (y_gp == y_gp_pub).mean()
    print(f"goodpjw submission vs public_pipeline agreement: {agree_pub:.4f}")

    # margin distribution on OOF + precision vs margin
    srt = np.sort(p_oof, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    correct = (p_oof.argmax(1) == oof_labels)
    print("\nmargin-precision curve (OOF):")
    for tau in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]:
        sel = margin > tau
        if sel.sum() > 100:
            print(f"  margin>{tau:.1f}: n={sel.sum():6d} ({sel.mean()*100:5.1f}%)  precision={correct[sel].mean():.4f}")
    # fraction of test windows above taus
    srt_t = np.sort(p_test, axis=1)
    margin_t = srt_t[:, -1] - srt_t[:, -2]
    for tau in [1.0, 1.5, 2.0, 2.5, 3.0]:
        sel = margin_t > tau
        print(f"  TEST margin>{tau:.1f}: {sel.mean()*100:.1f}%")

    # ---- top members table ----
    scored = [m for m in uniq if m["score"]]
    scored.sort(key=lambda m: -float(m["score"]))
    print("\n== TOP 25 unique own members ==")
    for m in scored[:25]:
        print(f"  {m['score']}  ref={m['ref']}  {m['date'][:16]}  {m['desc'][:70]}")

    # ---- agreement of top members with each other / public members ----
    top = scored[:12]
    print("\n== pairwise agreement (top members + public) ==")
    names = [f"own_{m['score']}" for m in top] + ["goodpjw", "yeashu"]
    vecs = [m["y"] for m in top] + [y_gp, y_ys]
    A = np.array(vecs)
    n = len(names)
    print("              " + " ".join(f"{x[:9]:>9}" for x in names))
    for i in range(n):
        row = []
        for j in range(n):
            row.append(f"{(A[i]==A[j]).mean():9.4f}")
        print(f"{names[i][:12]:>12} " + " ".join(row))

    # ---- where do public members disagree with the own best ----
    best_own = top[0]["y"]
    for nm, v in [("goodpjw", y_gp), ("yeashu", y_ys)]:
        dis = v != best_own
        print(f"\n{nm} disagrees with best own ({scored[0]['score']}) on {dis.sum()} windows ({dis.mean()*100:.2f}%)")

    # save state for next step
    np.savez(f"{BASE}/members/label_matrix.npz",
             refs=np.array([m["ref"] for m in uniq]),
             scores=np.array([-1 if m["score"] is None else float(m["score"]) for m in uniq]),
             ys=np.array([m["y"] for m in uniq]))
    json.dump([{k: m[k] for k in ("ref", "score", "date", "desc")} for m in uniq],
              open(f"{BASE}/members/uniq_meta.json", "w"), indent=1)
    print("\nsaved label_matrix.npz + uniq_meta.json")

if __name__ == "__main__":
    main()
