"""Small Kaggle API helper (kernels, submissions, leaderboard, private datasets).

Auth: KAGGLE_API_TOKEN from the environment (a Kaggle API token, sent as a Bearer token). Without it the
requests carry "Bearer proxy", which only works behind the cloud-session proxy that injects the real token.

  python tools/kaggle_api.py status <slug>            # kernel session status
  python tools/kaggle_api.py wait <slug> [grep]        # poll every 2 min until done, then print the log tail
  python tools/kaggle_api.py log <slug> [grep]         # log of the last completed run
  python tools/kaggle_api.py subs [n]                  # our latest submissions and scores
  python tools/kaggle_api.py submit <slug> <version> <description>
  python tools/kaggle_api.py lb [team-substring]       # public leaderboard: our rank and the medal lines
  python tools/kaggle_api.py upload <dataset-slug> <file> [<file> ...]   # create/version a PRIVATE dataset
"""
import json, os, re, sys, time
import requests

USER = os.environ.get("KAGGLE_USER", "xaxipirulazo")
COMP = "enveda-CASMI26-molecule-id-mass-spectra"
API = "https://api.kaggle.com/v1/"
H = {"Authorization": "Bearer " + os.environ.get("KAGGLE_API_TOKEN", "proxy")}


def post(method, body, timeout=120):
    return requests.post(API + method, json=body, headers=H, timeout=timeout)


def status(slug):
    return post("kernels.KernelsApiService/GetKernelSessionStatus", {"userName": USER, "kernelSlug": slug}, 60).json().get("status")


def log(slug, pat=None):
    o = post("kernels.KernelsApiService/ListKernelSessionOutput", {"userName": USER, "kernelSlug": slug}).json()
    L = json.loads(o.get("log") or "[]")
    out = "".join(e["data"] for e in L if e.get("stream_name") == "stdout")
    err = "".join(e["data"] for e in L if e.get("stream_name") == "stderr")
    if pat:
        return "\n".join(l for l in (out + err).splitlines() if re.search(pat, l))
    return out[-5000:] + "\n--- stderr ---\n" + err[-1500:]


def main(cmd, *a):
    if cmd == "status":
        print(status(a[0]))
    elif cmd == "wait":
        while True:
            s = status(a[0]); print(time.strftime("%H:%M"), s, flush=True)
            if s in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"):
                break
            time.sleep(120)
        print(log(a[0], a[1] if len(a) > 1 else None))
    elif cmd == "log":
        print(log(a[0], a[1] if len(a) > 1 else None))
    elif cmd == "subs":
        for s in post("competitions.CompetitionApiService/ListSubmissions", {"competitionName": COMP, "pageSize": int(a[0]) if a else 10}).json().get("submissions", []):
            print(s.get("ref"), (s.get("date") or "")[:16], s.get("status"), s.get("publicScore"), (s.get("description") or "")[:70])
    elif cmd == "submit":
        r = post("competitions.CompetitionApiService/CreateCodeSubmission", {"competitionName": COMP, "kernelOwner": USER, "kernelSlug": a[0],
                 "kernelVersion": int(a[1]), "fileName": "submission.csv", "submissionDescription": a[2]})
        print(r.status_code, r.text[:300])
    elif cmd == "lb":
        rows, tok = [], None
        while True:
            body = {"competitionName": COMP, "pageSize": 200}
            if tok:
                body["pageToken"] = tok
            j = post("competitions.CompetitionApiService/GetLeaderboard", body, 90).json()
            rows += j.get("submissions", []); tok = j.get("nextPageToken")
            if not tok:
                break
        n = len(rows); sc = [float(x.get("score") or 0) for x in rows]
        print(f"{n} teams | top {sc[:3]}")
        for name, k in (("gold", 10 + int(0.002 * n)), ("silver", int(0.05 * n)), ("bronze", int(0.10 * n))):
            print(f"  {name} line (rank {k}): {sc[k - 1]}")
        me = (a[0] if a else "xaxi").lower()
        for i, x in enumerate(rows):
            if me in str(x.get("teamName", "")).lower():
                print(f"  {x.get('teamName')}: rank {i + 1}, {x.get('score')}")
    elif cmd == "upload":
        slug, files, toks = a[0], a[1:], []
        for f in files:
            j = post("blobs.BlobApiService/StartBlobUpload", {"type": "DATASET", "name": os.path.basename(f), "contentLength": os.path.getsize(f),
                     "contentType": "application/octet-stream", "lastModifiedEpochSeconds": int(os.path.getmtime(f))}).json()
            with open(f, "rb") as fh:
                u = requests.put(j["createUrl"], data=fh, headers={"Content-Type": "application/octet-stream"}, timeout=3600)
            print("put", f, u.status_code); toks.append({"token": j["token"]})
        r = post("datasets.DatasetApiService/CreateDataset", {"ownerSlug": USER, "slug": slug, "title": slug, "isPrivate": True,
                 "licenseName": "unknown", "files": toks})
        print("create", r.status_code, r.text[:300])
        if r.status_code == 200 and (r.json().get("error") or "already" in r.text.lower()):
            r = post("datasets.DatasetApiService/CreateDatasetVersion", {"ownerSlug": USER, "datasetSlug": slug,
                     "body": {"versionNotes": "update", "files": toks, "deleteOldVersions": True}})
            print("version", r.status_code, r.text[:300])
    else:
        print(__doc__)


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["help"]))
