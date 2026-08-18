"""Build the Tier-0 training manifest from E:/archive/glasses-and-coverings.

For every image: map folder -> label, run SCRFD (production detector) to get
det_score + 5 landmarks (images with no detection are dropped — production
input always has a detection), compute a 64-bit pHash, cluster near-duplicates
(Hamming <= 6, union-find), and assign cluster-wise train/val/cal splits so
near-dups can never straddle a split boundary.

Usage:
    python scripts/build_manifest.py --data-root E:/archive/glasses-and-coverings \
        --out data/manifest_tier0.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glasses_detector.scrfd import SCRFD  # noqa: E402

# folder -> (label, tier-0 source tag)
FOLDER_LABELS = {
    "glasses": "eyeglasses",
    "plain": "none",
    "covering": "none",          # negatives; audit pass pulls out glasses-wearers
    "sunglasses": "sunglasses",
    "sunglasses-imagenet": "sunglasses",
}
LABEL_IDS = {"none": 0, "eyeglasses": 1, "sunglasses": 2}

SPLIT_FRACS = {"train": 0.80, "val": 0.12, "cal": 0.08}


def phash64(bgr: np.ndarray) -> np.uint64:
    """64-bit perceptual hash: 32x32 gray -> DCT -> top-left 8x8 (minus DC) vs median."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(gray)[:8, :8].flatten()[1:]
    bits = dct > np.median(dct)
    return np.uint64(sum(int(b) << i for i, b in enumerate(bits)))


def hamming_clusters(hashes: np.ndarray, max_dist: int = 6) -> np.ndarray:
    """Union-find clusters over 64-bit hashes with Hamming distance <= max_dist."""
    n = len(hashes)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    xor = hashes[:, None] ^ hashes[None, :]
    dist = np.zeros((n, n), dtype=np.uint8)
    v = xor.copy()
    for _ in range(64):
        dist += (v & np.uint64(1)).astype(np.uint8)
        v >>= np.uint64(1)
    close = np.argwhere((dist <= max_dist) & (np.arange(n)[:, None] < np.arange(n)[None, :]))
    for i, j in close:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    return np.array([find(i) for i in range(n)])


def assign_splits(df: pd.DataFrame, seed: int = 42) -> pd.Series:
    """Cluster-wise split assignment, roughly stratified by label."""
    rng = np.random.default_rng(seed)
    split = pd.Series(index=df.index, dtype=object)
    for _, group in df.groupby("label"):
        clusters = group["cluster"].unique()
        rng.shuffle(clusters)
        sizes = group.groupby("cluster").size()
        total = len(group)
        budget = {k: v * total for k, v in SPLIT_FRACS.items()}
        filled = {k: 0 for k in SPLIT_FRACS}
        for cl in clusters:
            # put this cluster in the most-underfilled split
            target = max(SPLIT_FRACS, key=lambda k: budget[k] - filled[k])
            split.loc[group.index[group["cluster"] == cl]] = target
            filled[target] += int(sizes[cl])
    return split


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="data/manifest_tier0.csv")
    ap.add_argument("--scrfd", default="models/det_500m.onnx")
    ap.add_argument("--det-size", type=int, default=160)
    args = ap.parse_args()

    det = SCRFD(args.scrfd, det_size=(args.det_size, args.det_size))
    root = Path(args.data_root)
    rows, dropped = [], []
    for folder, label in FOLDER_LABELS.items():
        for path in sorted((root / folder).iterdir()):
            bgr = cv2.imread(str(path))
            if bgr is None:
                dropped.append((str(path), "unreadable"))
                continue
            result = det.detect(bgr)
            if result is None:
                dropped.append((str(path), "no_face"))
                continue
            bbox, score, kps = result
            rows.append({
                "path": str(path), "source": folder, "label": label,
                "label_id": LABEL_IDS[label], "det_score": round(score, 4),
                **{f"kp{i}{ax}": round(float(kps[i, j]), 2)
                   for i in range(5) for j, ax in enumerate("xy")},
                "phash": int(phash64(bgr)),
            })
    df = pd.DataFrame(rows)
    df["cluster"] = hamming_clusters(df["phash"].to_numpy(dtype=np.uint64))
    df["split"] = assign_splits(df)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"kept {len(df)} / dropped {len(dropped)}")
    drop_df = pd.DataFrame(dropped, columns=["path", "reason"])
    drop_df.to_csv(out.with_name(out.stem + "_dropped.csv"), index=False)
    print(drop_df["reason"].value_counts().to_string())
    print("\nby source:")
    print(df.groupby(["source", "split"]).size().unstack(fill_value=0).to_string())
    print("\nby label/split:")
    print(df.groupby(["label", "split"]).size().unstack(fill_value=0).to_string())
    n_multi = (df.groupby("cluster").size() > 1).sum()
    print(f"\nnear-dup clusters with >1 image: {n_multi}")


if __name__ == "__main__":
    main()
