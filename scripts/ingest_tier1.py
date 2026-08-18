"""Ingest Tier-1 (mantasu glasses-detector corpus) and Tier-2 (MeGlass).

Label derivation for the glasses-detector classification tree:
    eyeglasses/<src>/<split>/eyeglasses/*      -> eyeglasses
    sunglasses/<src>/<split>/sunglasses/*      -> sunglasses
    anyglasses/<src>/<split>/no_anyglasses/*   -> none
An image claimed as BOTH eyeglasses and sunglasses is ambiguous -> dropped.
Source 'glasses-and-coverings' is skipped (it is our Tier-0).

MeGlass: meta.txt gives 1=black-eyeglasses, 0=no-eyeglasses; identity comes
from the filename (flickrid@identity_N@photo). No sunglasses class.

Every image passes the production SCRFD detector; no-detections are dropped.
Output: one manifest CSV per tier with the same schema as Tier-0 plus a
`group` column (identity where known, else empty).

Usage:
    python scripts/ingest_tier1.py --gd-root E:/datasets/glasses-detector/classification \
        --out data/manifest_tier1.csv
    python scripts/ingest_tier1.py --meglass-root E:/datasets/meglass \
        --out data/manifest_meglass.csv
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_manifest import LABEL_IDS, phash64  # noqa: E402

_DET = None


def _worker_init(scrfd_path: str):
    global _DET
    from glasses_detector.scrfd import SCRFD
    _DET = SCRFD(scrfd_path)


def _process_one(task):
    path, source, label, group = task
    bgr = cv2.imread(path)
    if bgr is None or min(bgr.shape[:2]) < 40:
        return None
    result = _DET.detect(bgr)
    if result is None:
        return None
    _, score, kps = result
    return {
        "path": path, "source": source, "label": label,
        "label_id": LABEL_IDS[label], "det_score": round(score, 4),
        **{f"kp{i}{ax}": round(float(kps[i, j]), 2)
           for i in range(5) for j, ax in enumerate("xy")},
        "phash": int(phash64(bgr)), "group": group,
    }


def collect_gd_tasks(root: Path) -> list:
    claims: dict[tuple, dict] = {}
    spec = [("eyeglasses", "eyeglasses", "eyeglasses"),
            ("sunglasses", "sunglasses", "sunglasses"),
            ("anyglasses", "no_anyglasses", "none")]
    for task_dir, class_dir, label in spec:
        for src_dir in sorted((root / task_dir).iterdir()):
            if not src_dir.is_dir() or src_dir.name == "glasses-and-coverings":
                continue
            for split_dir in src_dir.iterdir():
                cdir = split_dir / class_dir
                if not cdir.is_dir():
                    continue
                for img in cdir.iterdir():
                    key = (src_dir.name, img.name)
                    entry = claims.setdefault(
                        key, {"path": str(img), "source": f"gd:{src_dir.name}",
                              "labels": set()})
                    entry["labels"].add(label)
    tasks, ambiguous = [], 0
    for entry in claims.values():
        labels = entry["labels"]
        if "eyeglasses" in labels and "sunglasses" in labels:
            ambiguous += 1
            continue
        label = ("eyeglasses" if "eyeglasses" in labels
                 else "sunglasses" if "sunglasses" in labels else "none")
        tasks.append((entry["path"], entry["source"], label, ""))
    print(f"glasses-detector: {len(tasks)} tasks, {ambiguous} ambiguous dropped")
    return tasks


def collect_meglass_tasks(root: Path) -> list:
    meta = pd.read_csv(root / "repo" / "meta.txt", sep=" ",
                       names=["file", "glasses"])
    tasks = []
    for file, is_glasses in zip(meta["file"], meta["glasses"]):
        path = root / "MeGlass_ori" / file
        identity = "@".join(file.split("@")[:2])          # flickrid@identity_N
        label = "eyeglasses" if is_glasses == 1 else "none"
        tasks.append((str(path), "meglass", label, f"meglass:{identity}"))
    print(f"meglass: {len(tasks)} tasks")
    return tasks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gd-root")
    ap.add_argument("--meglass-root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scrfd", default="models/det_500m.onnx")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    tasks = []
    if args.gd_root:
        tasks += collect_gd_tasks(Path(args.gd_root))
    if args.meglass_root:
        tasks += collect_meglass_tasks(Path(args.meglass_root))

    rows, dropped = [], 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init,
                             initargs=(args.scrfd,)) as ex:
        for i, res in enumerate(ex.map(_process_one, tasks, chunksize=64)):
            if res is None:
                dropped += 1
            else:
                rows.append(res)
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{len(tasks)} processed "
                      f"({dropped} dropped)", flush=True)

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"kept {len(df)} / dropped {dropped}")
    print(df.groupby(["source", "label"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
