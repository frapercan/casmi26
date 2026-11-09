"""Dump the competition's discussion forum (topic list + each topic's first message) to local files.

Uses Kaggle's public web endpoints (anonymous session); comments are not readable this way.

  python tools/kaggle_forum.py --out work/forum
"""
import argparse, json, os, re, time
import requests

COMP = "enveda-CASMI26-molecule-id-mass-spectra"
U = "https://www.kaggle.com/api/i/"


def session():
    s = requests.Session(); s.headers["User-Agent"] = "Mozilla/5.0 (X11; Linux x86_64)"
    s.get(f"https://www.kaggle.com/competitions/{COMP}/discussion", timeout=30)
    s.headers.update({"x-xsrf-token": s.cookies.get("XSRF-TOKEN"), "content-type": "application/json"})
    return s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="work/forum"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    s = session()
    fid = s.post(U + "competitions.CompetitionService/GetCompetition", json={"competitionName": COMP}, timeout=30).json()["forumId"]
    topics, page = [], 1
    while True:
        r = s.post(U + "discussions.DiscussionsService/GetTopicListByForumId", json={"forumId": fid, "page": page}, timeout=30).json()
        if not r.get("topics"):
            break
        topics += r["topics"]; page += 1
    print(f"{len(topics)} topics")
    lines = []
    for t in sorted(topics, key=lambda t: -t.get("votes", 0)):
        p = f"{a.out}/{t['id']}.json"
        if not os.path.exists(p):
            j = s.post(U + "discussions.DiscussionsService/GetForumTopicById", json={"forumTopicId": t["id"], "includeComments": True},
                       timeout=30).json().get("forumTopic", {})
            json.dump(j, open(p, "w")); time.sleep(0.3)
        j = json.load(open(p))
        body = (j.get("firstMessage") or {}).get("rawMarkdown") or re.sub(r"<[^>]+>", " ", (j.get("firstMessage") or {}).get("content", ""))
        lines.append(f"### [{t['id']}] {t['title']}\nvotes={t.get('votes')} author={t['authorUser']['displayName']} ({t.get('authorType')}) "
                     f"date={t.get('postDate', '')[:10]} url=https://www.kaggle.com{t.get('topicUrl', '')}\n\n{body.strip()}\n")
    open(f"{a.out}/ALL.md", "w").write("\n\n".join(lines))
    json.dump(topics, open(f"{a.out}/topics.json", "w"))
    print("wrote", f"{a.out}/ALL.md", sum(len(l) for l in lines), "chars")


if __name__ == "__main__":
    main()
