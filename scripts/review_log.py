"""Review frames logged by the API (GLASSES_LOG_DIR) and tune the quality gate.

    python scripts/review_log.py logs/live [--out review] [--min-blur 20]

Prints:
  * per-frame confusion  truth x prediction   (only rows with truth != unknown)
  * burst confusion      truth x batch verdict
  * blur / det / eye-dist distributions for correct vs wrong frames
  * a suggested min_blur: smallest value that rejects >= 90% of the dangerous
    misses (truth=glasses predicted none) while keeping >= 95% of correct frames
Writes <out>/mismatches.png — a contact sheet of the wrong frames (dangerous
misses first), each crop annotated with p and blur, so you can see what the
model saw.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

TILE, COLS = 160, 8


def pred_label(row) -> str:
    a = row["pred_action"]
    return {"pass": "none", "remove_glasses": "glasses"}.get(a, "unsure")


def table(df: pd.DataFrame, rows: str, cols: str, title: str) -> None:
    if df.empty:
        print(f"\n{title}: (no rows)")
        return
    print(f"\n{title}")
    print(pd.crosstab(df[rows], df[cols], margins=True).to_string())


def describe(s: pd.Series) -> str:
    if s.empty:
        return "n=0"
    q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).values
    return f"n={len(s):4d}  p5={q[0]:7.1f} p25={q[1]:7.1f} med={q[2]:7.1f} p75={q[3]:7.1f} p95={q[4]:7.1f}"


def suggest_min_blur(wrong: pd.Series, right: pd.Series) -> float | None:
    if wrong.empty or right.empty:
        return None
    cands = np.unique(np.concatenate([wrong.values, right.values]))
    for t in cands:
        if (wrong < t).mean() >= 0.9 and (right >= t).mean() >= 0.95:
            return float(t)
    return None


def contact_sheet(rows: pd.DataFrame, path: Path, max_tiles: int = 96) -> None:
    rows = rows.head(max_tiles)
    tiles = []
    for _, r in rows.iterrows():
        cp = r["crop_path"]
        img = cv2.imread(str(cp)) if isinstance(cp, str) and cp else None
        if img is None:
            img = np.zeros((TILE, TILE, 3), np.uint8)
        img = cv2.resize(img, (TILE, TILE))
        txt = f"ew={float(r['eyewear']):.2f} b={float(r['blur_score']):.0f}"
        cv2.rectangle(img, (0, TILE - 18), (TILE, TILE), (0, 0, 0), -1)
        cv2.putText(img, txt, (3, TILE - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        color = (0, 0, 255) if r["truth"] == "glasses" else (255, 160, 0)
        cv2.rectangle(img, (0, 0), (TILE - 1, TILE - 1), color, 2)
        tiles.append(img)
    if not tiles:
        print("no mismatches to draw")
        return
    n_rows = math.ceil(len(tiles) / COLS)
    sheet = np.zeros((n_rows * TILE, COLS * TILE, 3), np.uint8)
    for i, t in enumerate(tiles):
        y, x = divmod(i, COLS)
        sheet[y * TILE:(y + 1) * TILE, x * TILE:(x + 1) * TILE] = t
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)
    print(f"\nwrote {path} ({len(tiles)} tiles; red border = truth glasses, blue = truth none)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir")
    ap.add_argument("--out", default="review")
    ap.add_argument("--min-blur", type=float, default=None,
                    help="evaluate what this gate would have rejected")
    args = ap.parse_args()

    csv_path = Path(args.log_dir) / "frames.csv"
    df = pd.read_csv(csv_path)
    if "eyewear" not in df.columns:                      # logs written before the eyewear column
        df["eyewear"] = 1.0 - df["none"]
    print(f"{len(df)} frames from {csv_path}  ·  truth counts: {dict(Counter(df['truth']))}")
    df["pred"] = df.apply(pred_label, axis=1)
    lab = df[df["truth"].isin(["glasses", "none"])].copy()
    if lab.empty:
        print("no frames with ground truth — set the truth toggle in the Live tab and capture again")
        return

    table(lab, "truth", "pred", "per-frame: truth x prediction")
    bursts = lab[lab["batch_action"].notna() & (lab["batch_action"] != "")]
    if not bursts.empty:
        b = bursts.groupby(["session_id", "ts"]).first().reset_index()
        table(b, "truth", "batch_action", "per-burst: truth x verdict")
        table(b, "truth", "batch_reason", "per-burst: truth x reason")

    lab["correct"] = lab["truth"] == lab["pred"]
    danger = lab[(lab["truth"] == "glasses") & (lab["pred"] == "none")]
    other_wrong = lab[(lab["truth"] == "none") & (lab["pred"] == "glasses")]
    abstain = lab[lab["pred"] == "unsure"]          # no face / inside the band: not a mistake
    right = lab[lab["correct"]]
    print(f"\ndangerous misses (truth glasses -> predicted none): {len(danger)}   "
          f"false alarms (truth none -> glasses): {len(other_wrong)}   "
          f"abstained (no face / unsure): {len(abstain)}   correct: {len(right)}")
    for name, col in (("blur", "blur_score"), ("det", "det_score"), ("eye_dist", "eye_dist")):
        print(f"  {name:8s} correct : {describe(right[col])}")
        print(f"  {name:8s} danger  : {describe(danger[col])}")

    sug = suggest_min_blur(danger["blur_score"], right["blur_score"])
    print("\nsuggested min_blur:", f"{sug:.1f}" if sug is not None else
          "none found that rejects >=90% of misses and keeps >=95% of correct frames "
          "(blur alone does not separate them — look at the contact sheet)")
    if args.min_blur is not None:
        t = args.min_blur
        print(f"with min_blur={t}: rejects {(danger['blur_score'] < t).mean():.0%} of misses, "
              f"{(right['blur_score'] < t).mean():.0%} of correct frames")

    wrong = pd.concat([danger.sort_values("blur_score"), other_wrong])
    contact_sheet(wrong, Path(args.out) / "mismatches.png")


if __name__ == "__main__":
    main()
