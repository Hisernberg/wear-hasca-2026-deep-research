#!/usr/bin/env python3
"""Probe Kaggle competition API: submission limits + safe upload test (no actual submission)."""
import os, sys, json
os.environ.setdefault("KAGGLE_API_TOKEN", "KGAT_e52c1bf267c579fdacc20157f79b2737")
COMP = "3rd-wear-dataset-challenge-hasca-2026"

def main():
    from kaggle.api import kaggle_api_extended as K
    api = K.KaggleApi()
    api.authenticate()

    # 1. Submission limits (quota check)
    try:
        lim = api.competition_get_submission_limits(COMP)
        print("LIMITS:", json.dumps({
            "num_today": getattr(lim, "num_today", None),
            "num_total": getattr(lim, "num_total", None),
            "num_allowed_now": getattr(lim, "num_allowed_now", None),
            "limited_by_total": getattr(lim, "limited_by_total", None),
        }))
    except Exception as e:
        print("LIMITS FAILED:", type(e).__name__, str(e)[:500])

    # 2. Safe upload probe: start upload of a tiny file, DON'T create submission
    try:
        probe = "/tmp/probe_submission.csv"
        with open(probe, "w") as f:
            f.write("id,target_feature\n0,0\n")
        req = __import__("kagglesdk").competitions.types.ApiStartSubmissionUploadRequest()
        req.competition_name = COMP
        req.file_name = "probe_never_submitted.csv"
        req.content_length = os.path.getsize(probe)
        req.last_modified_epoch_seconds = int(os.path.getmtime(probe))
        with api.build_kaggle_client() as client:
            resp = client.competitions.competition_api_client.start_submission_upload(req)
        print("UPLOAD PROBE OK: token=", str(resp.token)[:20], "create_url=", str(resp.create_url)[:80])
        print("=> SUBMIT CAPABILITY CONFIRMED (upload path works; actual submission NOT created)")
    except Exception as e:
        print("UPLOAD PROBE FAILED:", type(e).__name__, str(e)[:500])

if __name__ == "__main__":
    main()
