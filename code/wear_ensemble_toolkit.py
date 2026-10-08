# =============================================================================
# WEAR @HASCA 2026 — Ensemble & Submission Toolkit  (Kaggle-ready)
# Usage: attach as a Kaggle Utility Script, or paste this cell into a fresh
# notebook whose inputs are the OUTPUTS of the member pipeline runs:
#   - goodpjw2008 rerun  (expects submission.csv + P_test.npy + oof artifacts)
#   - yeashusemwal rerun (expects submission.csv, optional P/oof artifacts)
#   - our boosted timeline rerun (submission.csv + P_test.npy if present)
# Then call the SLOT functions at the bottom.
# Every helper prints OOF/LB-facing diagnostics; nothing ships ungated.
# =============================================================================
import json, os, re
from pathlib import Path
import numpy as np
import pandas as pd

N_CLS = 19

# ---------------------------------------------------------------- data loading
def find_input_root():
    for p in [Path("/kaggle/input")]:
        if p.exists():
            return p
    raise FileNotFoundError("no /kaggle/input")

def locate(root, name):
    """find files whose name contains `name` anywhere under root"""
    return sorted(set(root.rglob(name)))

def load_submission_csv(path, n_expected=12234):
    df = pd.read_csv(path)
    cols = df.columns.tolist()
    idc, tc = cols[0], cols[-1]
    sub = pd.read_csv(path).sort_values(idc)
    assert len(sub) == n_expected, f"{path}: {len(sub)} rows != {n_expected}"
    y = sub[tc].to_numpy(np.int64)
    assert y.min() >= 0 and y.max() < N_CLS, f"{path}: label range {y.min()}..{y.max()} outside 0..18"
    return sub, y

def load_probs(path):
    """P_test.npy / *.npz with key P/probs/logp -> (N,19) row-normalized prob"""
    if str(path).endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        P = None
        for k in ("P_test", "P", "probs", "p"):
            if k in z:
                P = z[k]
                break
        if P is None and "logp" in z:
            P = np.exp(z["logp"])
        if P is None:
            raise KeyError(f"{path}: no P/logp key ({list(z.keys())})")
    else:
        P = np.load(path)
    P = np.asarray(P, np.float64)
    if P.ndim == 3 and P.shape[1] == 768:      # (N,768,15) transpose-guard heritage
        raise ValueError(f"{path}: looks like video features, not probabilities")
    if P.shape[1] != N_CLS and P.shape[0] == N_CLS:
        P = P.T                                 # (19,N) -> (N,19)
    assert P.shape[1] == N_CLS, f"{path}: shape {P.shape}"
    P = np.clip(P, 1e-12, None)
    return P / P.sum(1, keepdims=True)

def load_oof(path):
    """oof_raw.npz -> (P, y) for honest weight/gate tuning"""
    z = np.load(path, allow_pickle=True)
    P = np.exp(z["logp"]) if "logp" in z else z["P"]
    y = z["y"]
    return np.asarray(P, np.float64), np.asarray(y, np.int64)

# ---------------------------------------------------------------- metrics
def macro_f1(y, pred):
    cm = np.bincount(y * N_CLS + np.asarray(pred), minlength=N_CLS * N_CLS).reshape(N_CLS, N_CLS)
    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1)
    m = denom > 0
    return float((2 * tp[m] / denom[m]).mean())

def margins(P):
    s = np.sort(P, 1)
    return s[:, -1] - s[:, -2]

def agreement_stats(subs):
    """modal label per row + fraction of members agreeing"""
    Y = np.stack(subs)                       # (M,N)
    mode = np.array([np.bincount(Y[:, i], minlength=N_CLS).argmax() for i in range(Y.shape[1])])
    agree = (Y == mode[None, :]).mean(0)
    return mode, agree

# ---------------------------------------------------------------- ensembling
def prob_geo_mean(Ps, weights=None):
    """weighted geometric mean of probability vectors (log domain), renormalized"""
    w = np.asarray(weights, np.float64) if weights is not None else np.ones(len(Ps)) / len(Ps)
    assert np.isclose(w.sum(), 1.0), f"weights must sum to 1 (got {w.sum()})"
    logP = sum(wi * np.log(P) for wi, P in zip(w, Ps))
    logP -= logP.max(1, keepdims=True)
    P = np.exp(logP)
    return P / P.sum(1, keepdims=True)

def label_vote(subs, weights=None, anchor=0):
    """weighted majority vote; ties break toward the anchor (highest-LB member)"""
    Y = np.stack(subs).T                     # (N,M)
    w = np.asarray(weights) if weights is not None else np.ones(len(subs))
    out = np.empty(len(Y), np.int64)
    for i, row in enumerate(Y):
        sc = np.bincount(row, weights=w[row], minlength=N_CLS)
        top = sc.max()
        cand = np.flatnonzero(np.isclose(sc, top))
        out[i] = cand[0] if len(cand) == 1 else (row[anchor] if row[anchor] in cand else cand[0])
    return out

def tune_weights_oof(oofPs, y, grid=np.arange(0.0, 1.01, 0.05)):
    """greedy anchored coordinate ascent on OOF macro-F1 (2 passes)"""
    w = np.ones(len(oofPs)) / len(oofPs)
    def f1_of(wv):
        return macro_f1(y, prob_geo_mean(oofPs, wv).argmax(1))
    best = f1_of(w)
    for _ in range(2):
        for i in range(len(oofPs)):
            for g in grid:
                w2 = w.copy()
                w2[i] = g
                others = [j for j in range(len(w)) if j != i]
                s = w[others].sum()
                if s > 0:
                    w2[others] = w[others] / s * (1.0 - g)
                    f = f1_of(w2)
                    if f > best + 1e-6:
                        best, w = f, w2
    return w / w.sum(), best

# ---------------------------------------------------------------- refinements
def per_class_bias_ascent(P, y, grid=np.linspace(-1.0, 1.0, 21), iters=2):
    """greedy per-class logit bias for macro-F1 (the free +0.001-0.003)"""
    logP = np.log(np.clip(P, 1e-12, None))
    bias = np.zeros(N_CLS)
    best = macro_f1(y, (logP + bias).argmax(1))
    for _ in range(iters):
        for c in range(N_CLS):
            for g in grid:
                b2 = bias.copy()
                b2[c] = g
                f = macro_f1(y, (logP + b2).argmax(1))
                if f > best + 1e-6:
                    best, bias = f, b2
    return bias, best

def low_margin_idx(P, q=0.05):
    """indices of the q-fraction lowest-margin windows (override/tie-break zone)"""
    m = margins(P)
    k = max(1, int(q * len(m)))
    return np.argsort(m)[:k]

def video_override(P_ens, video_heads, anchor_mask, tau=0.5, min_agree=2, mix=0.3):
    """greedy anchored voting: on low-margin windows, if >=min_agree video-side heads agree
    on class c (with head margin > tau) and c != anchor argmax, pull probability toward c.
    Returns modified probabilities so downstream Sinkhorn/refiner still sees a distribution."""
    P_out = P_ens.copy()
    flipped = 0
    idx = np.flatnonzero(anchor_mask)
    for i in idx:
        v_top = [int(np.argmax(h[i])) for h in video_heads]
        v_mg = [float(np.sort(h[i])[-1] - np.sort(h[i])[-2]) for h in video_heads]
        a = int(np.argmax(P_ens[i]))
        for c in set(v_top):
            votes = sum(1 for t, mg in zip(v_top, v_mg) if t == c and mg > tau)
            if votes >= min_agree and c != a:
                P_out[i] = (1 - mix) * P_ens[i] + mix * np.eye(N_CLS)[c]
                flipped += 1
                break
    print(f"video override: pulled {flipped}/{len(idx)} low-margin windows (tau={tau}, min_agree={min_agree})")
    return P_out

# ---------------------------------------------------------------- calibration
def sinkhorn_counts(P, sbj, targets_by_sbj, null_min=0.05, iters=50):
    """P rows -> per-subject column-target calibration. targets_by_sbj[s] = (19,) tile counts"""
    out = np.empty_like(P)
    for s in np.unique(sbj):
        ii = np.flatnonzero(sbj == s)
        Q = P[ii].copy()
        t_ = np.asarray(targets_by_sbj[int(s)], np.float64)
        t_ = np.maximum(t_, null_min * t_.sum() / N_CLS)
        for _ in range(iters):
            Q *= (t_ / np.maximum(Q.sum(0), 1e-9))[None]
            Q /= Q.sum(1, keepdims=True)
        out[ii] = Q
    return out

def sharpen(P, T=0.5):
    Q = np.clip(P, 1e-12, None) ** (1.0 / T)
    return Q / Q.sum(1, keepdims=True)

# ---------------------------------------------------------------- I/O & manifest
def make_submission(labels, work="/kaggle/working", name="submission"):
    hits = locate(find_input_root(), "sample_submission.csv")
    assert hits, "sample_submission.csv not found in inputs"
    sample = pd.read_csv(hits[0])
    idc, tc = sample.columns[:2]
    df = sample[[idc]].copy()
    df[tc] = labels
    assert df[tc].notna().all() and len(df) == 12234, "submission must cover all 12,234 ids"
    path = Path(work) / f"{name}.csv"
    df.to_csv(path, index=False)
    dist = np.bincount(np.asarray(labels, np.int64), minlength=N_CLS)
    print(f"wrote {path} | pred dist: {dist.tolist()}")
    print("null share:", round(float((np.asarray(labels) == 0).mean()), 3))
    return str(path)

def write_manifest(sub_path, **kw):
    man = {"ts": pd.Timestamp.utcnow().isoformat(),
           "kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "local"), **kw}
    out = Path(sub_path).with_suffix(".manifest.json")
    out.write_text(json.dumps(man, indent=1, default=str))
    print("manifest:", out)

# =============================================================================
# SLOT PLAYBOOK — fill MEMBERS paths from your attached fork outputs, run once
# =============================================================================
def run_all_slots():
    root = find_input_root()
    # ---- EDIT THESE: attach member run outputs as notebook inputs -------
    MEMBERS = {
        "goodpjw": {"dir": "/kaggle/input/goodpjw-rerun", "oof": None},  # oof: path to oof_raw.npz if present
        "yeashu":  {"dir": "/kaggle/input/yeashu-rerun",  "oof": None},
        "ours":    {"dir": "/kaggle/input/ours-rerun",    "oof": None},
    }
    # ---------------------------------------------------------------------
    subs, Ps, oofs = {}, {}, {}
    for name, m in MEMBERS.items():
        d = Path(m["dir"])
        if not d.exists():
            print(f"[{name}] dir missing, skipped: {d}")
            continue
        csvs = locate(d, "submission.csv")
        assert csvs, f"{name}: no submission.csv under {d}"
        _, y = load_submission_csv(csvs[0])
        subs[name] = y
        pts = list(d.rglob("P_test.npy")) + list(d.rglob("P_test.npz")) + list(d.rglob("*P*.npz"))
        for p in pts:
            try:
                P = load_probs(p)
                if len(P) == len(y):
                    Ps[name] = P
                    print(f"[{name}] probs from {p.name}")
                    break
            except Exception as e:
                print(f"[{name}] {p.name}: {e}")
        if m.get("oof") and Path(m["oof"]).exists():
            try:
                oofs[name] = load_oof(m["oof"])
                print(f"[{name}] OOF loaded ({len(oofs[name][1])} rows)")
            except Exception as e:
                print(f"[{name}] oof load failed: {e}")
    names = list(subs)
    assert names, "no members found — attach fork outputs first"
    print("members:", list(names), "| probs:", list(Ps), "| oof:", list(oofs))

    # SLOT 3: probability ensemble (label-vote fallback)
    if len(Ps) >= 2:
        ks = [k for k in oofs if k in Ps]
        w = None
        if len(ks) >= 2:
            w, oof_f1 = tune_weights_oof([oofs[k][0] for k in ks], oofs[ks[0]][1])
            print("OOF-tuned weights:", dict(zip(ks, np.round(w, 3))), "| OOF F1:", round(oof_f1, 4))
        order = [n for n in Ps]
        wv = None if w is None else [w[ks.index(n)] if n in ks else 0.0 for n in order]
        s = np.sum(wv) if wv is not None else 1.0
        if wv is not None and s > 0:
            wv = [x / s for x in wv]
        P_ens = prob_geo_mean([Ps[n] for n in order], wv)
        y_ens = P_ens.argmax(1)
        mode, agree = agreement_stats([subs[n] for n in names])
        print("ens-vs-modal agreement:", round(float((y_ens == mode).mean()), 4),
              "| mean member agreement:", round(float(agree.mean()), 4))
        p1 = make_submission(y_ens, name="e3_ensemble")
        write_manifest(p1, slot=3, members=order, weights=wv, method="prob_geo_mean")
        # label-vote fallback variant
        y_vote = label_vote([subs[n] for n in order],
                            weights=wv, anchor=order.index("goodpjw") if "goodpjw" in order else 0)
        if (y_vote != y_ens).mean() > 0.01:
            p1b = make_submission(y_vote, name="e3_labelvote")
            write_manifest(p1b, slot="3b", members=order, method="label_vote")
    else:
        print("fewer than 2 prob members -> use label_vote path only")
        if len(names) >= 2:
            y_vote = label_vote([subs[n] for n in names])
            make_submission(y_vote, name="e3_labelvote")

    # SLOT 7: per-class bias ascent on whichever OOF we have
    if oofs:
        k0 = list(oofs)[0]
        bias, f = per_class_bias_ascent(oofs[k0][0], oofs[k0][1])
        print(f"per-class bias on OOF({k0}): F1 {f:.4f} | bias {np.round(bias, 2).tolist()}")

    # SLOT 6 hook (needs per-head probs saved by the fork with heads=True):
    # video_heads = [np.exp(np.load(f)) for f in [.../test_head_video_f*.npy]]
    # P_ov = video_override(P_ens, video_heads, low_margin_idx(P_ens, 0.05), tau=0.5)
    # then re-run sharpen -> sinkhorn_counts -> argmax -> make_submission(..., name="e3_gav")

if __name__ == "__main__":
    run_all_slots()
