"""Fetch random real-world photos of people wearing eyeglasses from Wikimedia
Commons and run the current model on them — a per-round reality check.

Usage:
    python scripts/web_test.py --checkpoint runs/R4/best.pt [--n 5]
        [--query "man wearing eyeglasses portrait"]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

UA = {"User-Agent": "glasses-detector-test/1.0 (research; contact: local)"}


def commons_search(query: str, limit: int = 20) -> list[dict]:
    params = urllib.parse.urlencode({
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": query, "gsrnamespace": 6,
        "gsrlimit": limit, "prop": "imageinfo", "iiprop": "url|size",
        "iiurlwidth": 1024,
    })
    req = urllib.request.Request(
        f"https://commons.wikimedia.org/w/api.php?{params}", headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    pages = data.get("query", {}).get("pages", {})
    out = []
    for p in pages.values():
        info = p.get("imageinfo", [{}])[0]
        url = info.get("thumburl") or info.get("url")
        bare = url.split("?")[0].lower() if url else ""
        if bare.endswith((".jpg", ".jpeg", ".png")):
            out.append({"title": p["title"], "url": url})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--query", default="portrait wearing eyeglasses")
    ap.add_argument("--out-dir", default="data/webtest")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    import random
    import time
    from glasses_detector.predict import GlassesDetector

    sys.stdout.reconfigure(errors="replace")

    results = commons_search(args.query, limit=40)
    if args.seed is not None:
        random.seed(args.seed)
    random.shuffle(results)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detector = GlassesDetector(args.checkpoint, device="cpu")

    tested = 0
    for item in results:
        if tested >= args.n:
            break
        name = "".join(c if c.isalnum() or c in "._-" else "_"
                       for c in item["title"].removeprefix("File:"))[:80]
        dest = out_dir / name
        try:
            time.sleep(3)  # stay under Commons rate limits
            req = urllib.request.Request(item["url"], headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                dest.write_bytes(r.read())
            res = detector.predict(str(dest))
        except Exception as e:  # noqa: BLE001 - skip bad downloads, keep testing
            print(f"  skip {name}: {e}")
            continue
        tested += 1
        verdict = ("NO FACE" if not res.face_found
                   else "GLASSES" if res.wearing_glasses else "no-glasses")
        print(f"{verdict:10s} p={res.probability:.3f} "
              f"probs={res.class_probs} det={res.det_score:.2f}  {name}")
    print(f"\ntested {tested} web photos with {args.checkpoint}")


if __name__ == "__main__":
    main()
