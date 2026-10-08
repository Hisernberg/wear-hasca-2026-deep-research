#!/usr/bin/env python3
"""Download all own submissions (raw files) via DownloadSubmission redirect."""
import os, json, subprocess, sys

os.environ.setdefault("KAGGLE_API_TOKEN", "KGAT_e52c1bf267c579fdacc20157f79b2737")
COMP = "3rd-wear-dataset-challenge-hasca-2026"
OUT_DIR = "/home/z/my-project/wear_research/members/own_subs"
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    # 1. full list with pageSize=250
    import urllib.request
    req = urllib.request.Request(
        f"https://www.kaggle.com/api/v1/competitions/submissions/list/{COMP}?pageSize=250",
        headers={"Authorization": f"Bearer {os.environ['KAGGLE_API_TOKEN']}"})
    subs = json.loads(urllib.request.urlopen(req, timeout=60).read())
    json.dump(subs, open("/home/z/my-project/wear_research/kaggle/subs_full_157.json", "w"), indent=1)
    print("total submissions in list:", len(subs))

    from kaggle.api import kaggle_api_extended as K
    from kagglesdk.competitions.types.competition_api_service import ApiDownloadSubmissionRequest
    api = K.KaggleApi(); api.authenticate()

    ok, fail = 0, 0
    for s in subs:
        ref = s["ref"]
        dest = f"{OUT_DIR}/sub_{ref}.csv"
        if os.path.exists(dest) and os.path.getsize(dest) > 5000:
            ok += 1
            continue
        try:
            req = ApiDownloadSubmissionRequest(); req.submission_id = ref
            with api.build_kaggle_client() as client:
                resp = client.competitions.competition_api_client.download_submission(req)
            url = getattr(resp, "url", None) or str(resp)
            r = subprocess.run(["curl", "-s", "-L", "-m", "120", "-o", dest, url], capture_output=True)
            head = open(dest).readline()
            if "target" in head or "id" in head:
                ok += 1
            else:
                fail += 1
                print("BAD", ref, head[:60])
        except Exception as e:
            fail += 1
            print("FAIL", ref, type(e).__name__, str(e)[:120])
    print(f"downloaded ok={ok} fail={fail}")

if __name__ == "__main__":
    main()
