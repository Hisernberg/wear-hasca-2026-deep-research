#!/usr/bin/env python3
"""Analyze recovered probability artifacts: quality, alignment with goodpjw OOF, fusion potential."""
import json, csv
import numpy as np

BASE = "/home/z/my-project/wear_research"

def macro_f1(y_true, y_pred, n_cls=19):
    f1s = []
    for c in range(n_cls):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(f1s))

def main():
    oof_logp = np.load(f"{BASE}/own_code/train_zip/oof_logp.npy")
    test_logp = np.load(f"{BASE}/own_code/train_zip/test_logp.npy")
    new_oof = np.load(f"{BASE}/own_code/train_zip/new_oof.npy")
    new_test = np.load(f"{BASE}/own_code/train_zip/new_test.npy")
    for nm, a in [("oof_logp", oof_logp), ("test_logp", test_logp), ("new_oof", new_oof), ("new_test", new_test)]:
        print(f"{nm}: shape={a.shape} dtype={a.dtype} min={a.min():.3f} max={a.max():.3f}")

    z = np.load(f"{BASE}/members/goodpjw_out/final_probabilities.npz")
    gp_oof, gp_oof_labels, gp_test = z["oof"], z["oof_labels"], z["test"]

    rows = list(csv.reader(open(f"{BASE}/members/own_subs/sub_56884414.csv")))
    y_f4 = np.array([int(r[1]) for r in rows[1:]])

    # alignment check: user oof (69326,19) vs goodpjw oof
    if oof_logp.shape[0] == gp_oof.shape[0]:
        # if both are log-probs/probs on same window set, their argmax agreement tells alignment
        am_u = oof_logp.argmax(1)
        agree_am = (am_u == gp_oof.argmax(1)).mean()
        f1_u = macro_f1(gp_oof_labels, am_u)
        print(f"\nuser oof_logp argmax vs goodpjw oof argmax agreement: {agree_am:.4f}")
        print(f"user oof_logp argmax vs goodpjw oof_labels macro F1: {f1_u:.5f}")
        f1_gp = macro_f1(gp_oof_labels, gp_oof.argmax(1))
        print(f"goodpjw oof argmax vs oof_labels macro F1: {f1_gp:.5f}")
        # new_oof quality (may be non-2D)
        print(f"new_oof raw shape: {new_oof.shape}")
        am_n = new_oof.reshape(new_oof.shape[0], -1).argmax(1) if new_oof.ndim != 2 else new_oof.argmax(1)
        if am_n.shape == am_u.shape:
            print(f"new_oof argmax vs oof_labels F1: {macro_f1(gp_oof_labels, am_n):.5f}  (agreement w/ user oof argmax: {(am_n==am_u).mean():.4f})")
        else:
            print(f"new_oof argmax shape {am_n.shape} != oof {am_u.shape}; skipping")
        # correlation between user-oof and gp-oof probability matrices (same window order?)
        # compare row-wise cosine on a sample
        rng = np.random.default_rng(1)
        idx = rng.choice(len(gp_oof), 2000, replace=False)
        a = oof_logp[idx] - oof_logp[idx].max(1, keepdims=True)
        b = gp_oof[idx] - gp_oof[idx].max(1, keepdims=True)
        # softmax-normalize both
        pa = np.exp(a); pa /= pa.sum(1, keepdims=True)
        pb = np.exp(np.clip(b, -50, 50)); pb /= pb.sum(1, keepdims=True)
        cos = (pa * pb).sum(1) / (np.linalg.norm(pa, axis=1) * np.linalg.norm(pb, axis=1) + 1e-12)
        print(f"row-wise cosine(user_oof_probs, gp_oof_probs) on sample: mean={cos.mean():.4f} median={np.median(cos):.4f}")

    # test-side: user test_logp vs gp p_test
    am_ut = test_logp.argmax(1)
    am_gt = gp_test.argmax(1)
    print(f"\ntest: user argmax vs goodpjw argmax agreement: {(am_ut == am_gt).mean():.4f}")
    # new_test: likely (6117,2,19) — decode layout
    print(f"\nnew_test raw shape: {new_test.shape}, new_oof raw shape: {new_oof.shape}")
    if new_test.ndim == 3:
        # try: (N2, 2, 19) where flatten gives 12234 rows in some order
        flat = new_test.reshape(-1, 19)
        print(f"flat shape: {flat.shape}")
        if flat.shape[0] == 12234:
            for tag, order in [("direct", np.arange(12234)), ("chunk-swap", np.concatenate([np.arange(6117,12234), np.arange(6117)]))]:
                f2 = flat[order] if tag == "chunk-swap" else flat
                print(f"  new_test[{tag}] vs user test argmax agreement: {(f2.argmax(1) == am_ut).mean():.4f}")
                print(f"  new_test[{tag}] vs final anchor agreement: {(f2.argmax(1) == y_f4).mean():.4f}")
    elif new_test.ndim == 2 and new_test.shape == test_logp.shape:
        print(f"new_test argmax vs user test argmax: {(new_test.argmax(1) == am_ut).mean():.4f}")
        print(f"new_test argmax vs final anchor: {(new_test.argmax(1) == y_f4).mean():.4f}")

    # how close is user's base test argmax to their FINAL decoded anchor?
    print(f"(y_f4 loaded: {y_f4.shape})")
    print(f"user base test argmax vs final anchor f4n: {(am_ut == y_f4).mean():.4f} (diff {(am_ut != y_f4).sum()} windows)")

    # meta info about the artifact submissions
    subs = json.load(open(f"{BASE}/kaggle/subs_full_157.json"))
    for s in subs:
        if s["ref"] in (56842464, 56842474, 56606305, 56481287, 56481150, 56638047, 56426550):
            print(f"\nref {s['ref']} | {s['date'][:16]} | score={s.get('publicScoreNullable')} | {(s.get('descriptionNullable') or '')[:110]}")

if __name__ == "__main__":
    main()
