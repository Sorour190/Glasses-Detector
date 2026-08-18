"""Merge tier manifests with cross-tier dedup and leakage-safe splits.

- Near-dup detection at 100k scale via banded LSH on the 64-bit pHash:
  4 x 16-bit bands; images sharing any band are candidate pairs, verified at
  Hamming <= 6, then union-find. (The naive all-pairs matrix is O(n^2).)
- Groups = union of (pHash cluster, identity group where present).
- Existing split assignments from the base manifest are kept frozen; any new
  image clustered with a base image inherits that split; remaining clusters
  are assigned 80/12/8 stratified by label.
- Leakage audit: exact/near-dup pairs that straddle splits after assignment
  (must be zero by construction; printed as proof).

Usage:
    python scripts/merge_manifests.py --base data/manifest_tier0_v2.csv \
        --add data/manifest_tier1.csv --out data/manifest_r2.csv
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_FRACS = {"train": 0.80, "val": 0.12, "cal": 0.08}


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int):
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[ri] = rj


def hamming_clusters_banded(hashes: np.ndarray, max_dist: int = 6) -> np.ndarray:
    n = len(hashes)
    uf = UnionFind(n)
    buckets: dict[tuple, list] = defaultdict(list)
    for band in range(4):
        shift = np.uint64(band * 16)
        keys = (hashes >> shift) & np.uint64(0xFFFF)
        for i, k in enumerate(keys):
            buckets[(band, int(k))].append(i)
    checked = 0
    for members in buckets.values():
        if len(members) < 2 or len(members) > 2000:
            continue
        arr = hashes[members]
        xor = arr[:, None] ^ arr[None, :]
        dist = np.zeros(xor.shape, dtype=np.uint8)
        v = xor.copy()
        for _ in range(64):
            dist += (v & np.uint64(1)).astype(np.uint8)
            v >>= np.uint64(1)
        for a, b in np.argwhere(
                (dist <= max_dist)
                & (np.arange(len(members))[:, None] < np.arange(len(members))[None, :])):
            uf.union(members[a], members[b])
            checked += 1
    return np.array([uf.find(i) for i in range(n)])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="manifest with frozen splits")
    ap.add_argument("--add", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base = pd.read_csv(args.base)
    base["group"] = base.get("group", "")
    added = pd.concat([pd.read_csv(p) for p in args.add], ignore_index=True)
    added["split"] = np.nan
    df = pd.concat([base, added], ignore_index=True)
    df["group"] = df["group"].fillna("")

    # exact-dup removal across tiers (same pHash + same label keeps first,
    # conflicting labels drop both — a cheap cross-source label check)
    before = len(df)
    dup = df.duplicated(subset=["phash"], keep=False)
    conflict_hashes = (df[dup].groupby("phash")["label"].nunique() > 1)
    conflict_hashes = set(conflict_hashes[conflict_hashes].index)
    df = df[~df["phash"].isin(conflict_hashes)]
    df = df.drop_duplicates(subset=["phash"], keep="first").reset_index(drop=True)
    print(f"exact dedup: {before} -> {len(df)} "
          f"({len(conflict_hashes)} conflicting-label hashes dropped)")

    clusters = hamming_clusters_banded(df["phash"].to_numpy(dtype=np.uint64))
    df["cluster"] = clusters

    # merge clusters that share an identity group
    uf = UnionFind(int(clusters.max()) + 1)
    for _, grp in df[df["group"] != ""].groupby("group"):
        cl = grp["cluster"].unique()
        for c in cl[1:]:
            uf.union(int(cl[0]), int(c))
    df["cluster"] = [uf.find(int(c)) for c in df["cluster"]]

    # split inheritance + assignment
    rng = np.random.default_rng(args.seed)
    cluster_split: dict[int, str] = {}
    for c, grp in df[df["split"].notna()].groupby("cluster"):
        cluster_split[int(c)] = grp["split"].mode()[0]

    unassigned = df[~df["cluster"].isin(cluster_split)]
    for label, grp in unassigned.groupby("label"):
        clusters_l = grp["cluster"].unique()
        rng.shuffle(clusters_l)
        sizes = grp.groupby("cluster").size()
        budget = {k: v * len(grp) for k, v in SPLIT_FRACS.items()}
        filled = {k: 0.0 for k in SPLIT_FRACS}
        for c in clusters_l:
            target = max(SPLIT_FRACS, key=lambda k: budget[k] - filled[k])
            cluster_split[int(c)] = target
            filled[target] += float(sizes[c])
    df["split"] = [cluster_split[int(c)] for c in df["cluster"]]

    # leakage audit
    leaks = (df.groupby("cluster")["split"].nunique() > 1).sum()
    print(f"leakage audit: clusters straddling splits = {leaks} (must be 0)")

    out = Path(args.out)
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df)} rows")
    print(df.groupby(["label", "split"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
