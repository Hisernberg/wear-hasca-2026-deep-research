#!/usr/bin/env python3
"""Submit a candidate file to the competition and wait for the score."""
import os, sys, time, json

os.environ.setdefault("KAGGLE_API_TOKEN", "KGAT_e52c1bf267c579fdacc20157f79b2737")
COMP = "3rd-wear-dataset-challenge-hasca-2026"

def submit(path, message, wait_secs=420):
    from kaggle.api import kaggle_api_extended as K
    api = K.KaggleApi(); api.authenticate()
    print(f"submitting {os.path.basename(path)}: {message}")
    resp = api.competition_submit(path, message, COMP)
    print("submit response:", resp)
    # poll for score
    deadline = time.time() + wait_secs
    ref = None
    while time.time() < deadline:
        time.sleep(20)
        subs = api.competition_submissions(COMP)
        subs = list(subs)
        if subs:
            s = subs[0]  # latest
            st = getattr(s, "status", None)
            sc = getattr(s, "public_score", None) or getattr(s, "public_score_nullable", None)
            has = getattr(s, "has_public_score", None)
            print(f"  poll: status={st} score={sc} ref={getattr(s,'ref',None)}")
            if st and "complete" in str(st).lower() and sc:
                return {"ref": getattr(s, "ref", None), "status": str(st), "score": str(sc)}
            if st and ("error" in str(st).lower() or "invalid" in str(st).lower()):
                # fetch error description
                err = getattr(s, "error_description", None)
                return {"ref": getattr(s, "ref", None), "status": str(st), "error": str(err)}
    return {"status": "timeout"}

if __name__ == "__main__":
    path = sys.argv[1]
    msg = sys.argv[2]
    out = submit(path, msg)
    print(json.dumps(out, indent=1))
