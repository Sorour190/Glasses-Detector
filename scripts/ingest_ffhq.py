"""Ingest FFHQ-256 with the Zenodo eyeglasses-extension labels.

Positives = images with a LabelMe JSON in the Zenodo annotation set (16,923,
single class 'Glasses' -> type pseudo-labeled downstream). Negatives = a
seeded sample of unannotated FFHQ images (we already have many negatives).
Every image passes SCRFD as usual.
"""

from __future__ import annotations

import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.ingest_tier1 import _process_one, _worker_init  # noqa: E402
import scripts.build_manifest as bm  # noqa: E402

bm.LABEL_IDS["unknown"] = -1

IMAGES = Path("E:/datasets/ffhq256/resized")
LABELS = Path("E:/datasets/ffhq_labels/FFHQ_Eyeglasses_Detection_labels")
N_NEGATIVES = 12000


def main():
    pos_ids = {p.stem for p in LABELS.glob("*.json")}
    tasks = []
    neg_candidates = []
    for img in IMAGES.iterdir():
        if img.stem in pos_ids:
            tasks.append((str(img), "ffhq", "unknown", ""))
        else:
            neg_candidates.append(str(img))
    random.seed(42)
    for p in random.sample(neg_candidates, min(N_NEGATIVES, len(neg_candidates))):
        tasks.append((p, "ffhq", "none", ""))
    print(f"{len(pos_ids)} positives, {len(tasks) - len(pos_ids)} sampled negatives")

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
    df.to_csv("data/manifest_ffhq.csv", index=False)
    print(f"kept {len(df)} / dropped {dropped}")
    print(df.label.value_counts().to_string())


if __name__ == "__main__":
    main()
