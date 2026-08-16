"""Build the glasses/no_glasses training layout from CelebA.

CelebA ships an "Eyeglasses" attribute for ~200k celebrity face images,
which makes it a convenient starting dataset for this task. Download the
aligned images (img_align_celeba) and list_attr_celeba.txt from
https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html (or the Kaggle mirror),
then run:

    python scripts/prepare_celeba.py \
        --images img_align_celeba \
        --attrs list_attr_celeba.txt \
        --out data

Note: CelebA is heavily imbalanced (~6% eyeglasses), so this script caps
the negative class at --neg-ratio times the positive count.

For best production accuracy, fine-tune afterwards on face crops from your
own capture pipeline (same camera framing the onboarding flow produces).
"""

import argparse
import csv
import random
import shutil
from pathlib import Path


def read_attrs(attrs_path: Path) -> dict:
    """Returns {filename: has_glasses}. Supports both the original txt and CSV formats."""
    labels = {}
    with open(attrs_path) as f:
        if attrs_path.suffix == ".csv":
            reader = csv.DictReader(f)
            for row in reader:
                labels[row["image_id"]] = int(row["Eyeglasses"]) == 1
        else:
            f.readline()  # image count
            header = f.readline().split()
            idx = header.index("Eyeglasses")
            for line in f:
                parts = line.split()
                labels[parts[0]] = int(parts[idx + 1]) == 1
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="Directory of aligned CelebA images")
    parser.add_argument("--attrs", required=True, help="list_attr_celeba.txt or .csv")
    parser.add_argument("--out", default="data")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--neg-ratio", type=float, default=1.5,
                        help="Max negatives per positive (class balancing)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    labels = read_attrs(Path(args.attrs))

    positives = [f for f, glasses in labels.items() if glasses]
    negatives = [f for f, glasses in labels.items() if not glasses]
    rng.shuffle(negatives)
    negatives = negatives[: int(len(positives) * args.neg_ratio)]
    print(f"{len(positives)} glasses / {len(negatives)} no_glasses images")

    images_dir = Path(args.images)
    out = Path(args.out)
    for class_name, files in (("glasses", positives), ("no_glasses", negatives)):
        rng.shuffle(files)
        n_val = int(len(files) * args.val_fraction)
        splits = {"val": files[:n_val], "train": files[n_val:]}
        for split, split_files in splits.items():
            dest = out / split / class_name
            dest.mkdir(parents=True, exist_ok=True)
            copied = 0
            for name in split_files:
                src = images_dir / name
                if src.exists():
                    shutil.copy2(src, dest / name)
                    copied += 1
            print(f"{split}/{class_name}: {copied} images")


if __name__ == "__main__":
    main()
