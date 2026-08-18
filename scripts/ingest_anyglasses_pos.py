"""Ingest the unused anyglasses-POSITIVE tree (eyewear type unknown).

These ~26k images are known to contain worn eyewear but not which kind.
They get label 'unknown' here; scripts then pseudo-label them with the
current best model (confident cases) or route them to review sheets
(uncertain cases = concentrated hard positives: rimless / clear-frame /
thin wire).
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.ingest_tier1 import _process_one, _worker_init  # noqa: E402
import scripts.build_manifest as bm  # noqa: E402

bm.LABEL_IDS["unknown"] = -1

ROOT = Path("E:/datasets/glasses-detector/classification/anyglasses")
SKIP = {"glasses-and-coverings", "specs-on-faces"}


def main():
    tasks = []
    for src_dir in sorted(ROOT.iterdir()):
        if not src_dir.is_dir() or src_dir.name in SKIP:
            continue
        for split_dir in src_dir.iterdir():
            cdir = split_dir / "anyglasses"
            if cdir.is_dir():
                for img in cdir.iterdir():
                    tasks.append((str(img), f"gd:{src_dir.name}", "unknown", ""))
    print(f"{len(tasks)} anyglasses-positive candidates")

    rows, dropped = [], 0
    with ProcessPoolExecutor(max_workers=8, initializer=_worker_init,
                             initargs=("models/det_500m.onnx",)) as ex:
        for i, res in enumerate(ex.map(_process_one, tasks, chunksize=64)):
            if res is None:
                dropped += 1
            else:
                rows.append(res)
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{len(tasks)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("data/manifest_anyglasses_pos.csv", index=False)
    print(f"kept {len(df)} / dropped {dropped}")


if __name__ == "__main__":
    main()
