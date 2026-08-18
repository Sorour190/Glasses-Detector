"""Pre-crop cache: decode each image once, save an expanded ROI crop.

The cached crop uses ROI-v1 geometry expanded 1.4x (width 4.2d x height 2.8d)
at 224px, so training-time landmark jitter (+-10% scale) always stays inside.
Landmarks are re-projected into cache coordinates, so ManifestDataset works on
cached manifests unchanged. Cuts dataloader JPEG-decode cost ~30x.

Usage:
    python scripts/build_crop_cache.py --manifest data/manifest_r4.csv \
        --cache-dir E:/datasets/crop_cache --out data/manifest_r4_cached.csv
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_EXPAND = 1.4
_OUT = 224
_KP_COLS = [f"kp{i}{ax}" for i in range(5) for ax in "xy"]


def _process(task):
    path, kps_flat, cache_dir = task
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    kps = np.array(kps_flat, dtype=np.float64).reshape(5, 2)

    le, re = kps[0], kps[1]
    d = float(np.linalg.norm(re - le))
    if d < 1e-3:
        return None
    c = (le + re) / 2.0
    ex = (re - le) / d
    ey = np.array([-ex[1], ex[0]])
    cc = c + 0.10 * d * ey
    half_w, half_h = 1.5 * _EXPAND * d, 1.0 * _EXPAND * d
    src = np.float32([cc - half_w * ex - half_h * ey,
                      cc + half_w * ex - half_h * ey,
                      cc - half_w * ex + half_h * ey])
    dst = np.float32([[0, 0], [_OUT, 0], [0, _OUT]])
    m = cv2.getAffineTransform(src, dst)
    crop = cv2.warpAffine(bgr, m, (_OUT, _OUT), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)

    new_kps = (np.hstack([kps, np.ones((5, 1))]) @ m.T)
    name = hashlib.sha1(path.encode()).hexdigest()[:20] + ".jpg"
    out_path = str(Path(cache_dir) / name[:2] / name)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path, out_path, new_kps.flatten().round(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    tasks = [(row["path"], [row[c] for c in _KP_COLS], args.cache_dir)
             for _, row in df.iterrows()]
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, res in enumerate(ex.map(_process, tasks, chunksize=128)):
            if res is not None:
                results[res[0]] = res
            if (i + 1) % 10000 == 0:
                print(f"  {i + 1}/{len(tasks)}", flush=True)

    ok = df["path"].isin(results)
    df = df[ok].copy()
    df[_KP_COLS] = np.stack([results[p][2] for p in df["path"]])
    df["path"] = [results[p][1] for p in df["path"]]
    df.to_csv(args.out, index=False)
    print(f"cached {len(df)} crops -> {args.out}")


if __name__ == "__main__":
    main()
