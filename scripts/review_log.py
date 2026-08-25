"""Review frames logged by the API (GLASSES_LOG_DIR) and tune the quality gate.

    python scripts/review_log.py logs/live [--out review] [--profile bad_phone]

Prints:
  * per-frame confusion  truth x prediction   (only rows with truth != unknown)
  * burst confusion      truth x batch verdict
  * blur / exposure / contrast / detector / eye-distance distributions
  * quality-gate impact by quality version and bad-camera profile
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

from glasses_detector.aggregate import AggregateConfig

TILE, COLS = 160, 8


def pred_label(row, vote_on: str = "eyewear", t_low: float = 0.2,
               t_high: float = 0.5) -> str:
    # Use model telemetry, not the post-quality action: rejected model misses
    # must remain visible when evaluating candidate quality thresholds.
    if float(row.get("det_score", 0.0)) <= 0.0:
        return "unsure"
    value = row.get("eyewear") if vote_on == "eyewear" else row.get("p")
    if pd.notna(value):
        p = float(value)
        return "glasses" if p >= t_high else ("none" if p < t_low else "unsure")
    a = row["pred_action"]
    return {"pass": "none", "remove_glasses": "glasses"}.get(a, "unsure")


def table(df: pd.DataFrame, rows: str, cols: str, title: str) -> None:
    if df.empty:
        print(f"\n{title}: (no rows)")
        return
    print(f"\n{title}")
    print(pd.crosstab(df[rows], df[cols], margins=True).to_string())


def describe(s: pd.Series) -> str:
    s = s.dropna()
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


def quality_gate_reasons(df: pd.DataFrame, *, min_blur: float,
                         min_brightness: float, max_brightness: float,
                         min_contrast: float) -> pd.Series:
    """Return one quality-v2 rejection reason per frame, or None when usable."""
    reasons = pd.Series([None] * len(df), index=df.index, dtype=object)
    checks = (
        (df["blur_score"] < min_blur, "blurry"),
        (df["brightness"] < min_brightness, "too_dark"),
        (df["brightness"] > max_brightness, "too_bright"),
        (df["contrast"] < min_contrast, "low_contrast"),
    )
    for condition, reason in checks:
        reasons.loc[reasons.isna() & condition.fillna(False)] = reason
    return reasons


def effective_min_blur(profile: str, cli_override: float | None,
                       cfg: AggregateConfig) -> float:
    """Resolve the review threshold exactly as runtime profile config does."""
    return cli_override if cli_override is not None else cfg.for_profile(profile).min_blur


def gate_impact(labeled: pd.DataFrame, reasons: pd.Series) -> dict:
    """Summarize the safety/availability trade-off of candidate thresholds."""
    rejected = reasons.notna()
    danger = (labeled["truth"] == "glasses") & (labeled["pred"] == "none")
    correct = labeled["truth"] == labeled["pred"]
    danger_n, correct_n = int(danger.sum()), int(correct.sum())
    danger_rejected = int((danger & rejected).sum())
    correct_kept = int((correct & ~rejected).sum())
    return {
        "dangerous_misses": danger_n,
        "dangerous_rejected": danger_rejected,
        "dangerous_reject_rate": danger_rejected / danger_n if danger_n else 0.0,
        "correct_frames": correct_n,
        "correct_kept": correct_kept,
        "correct_keep_rate": correct_kept / correct_n if correct_n else 0.0,
        "all_reject_rate": float(rejected.mean()) if len(rejected) else 0.0,
    }


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
    ap.add_argument("--profile", choices=("clean", "mild", "bad_phone", "extreme"),
                    help="only analyze frames captured with this simulation profile")
    ap.add_argument("--min-blur", type=float, default=None,
                    help="override every profile's configured blur threshold")
    ap.add_argument("--min-brightness", type=float, default=35.0)
    ap.add_argument("--max-brightness", type=float, default=220.0)
    ap.add_argument("--min-contrast", type=float, default=25.0)
    ap.add_argument("--vote-on", choices=("eyewear", "eyeglasses"), default="eyewear")
    ap.add_argument("--vote-low", type=float, default=0.2)
    ap.add_argument("--vote-high", type=float, default=0.5)
    args = ap.parse_args()
    runtime_cfg = AggregateConfig.load()

    csv_path = Path(args.log_dir) / "frames.csv"
    df = pd.read_csv(csv_path)
    if "eyewear" not in df.columns:                      # logs written before the eyewear column
        df["eyewear"] = 1.0 - df["none"]
    if "quality_version" not in df.columns:
        df["quality_version"] = "quality-v1"
    if "degradation_profile" not in df.columns:
        df["degradation_profile"] = "clean"
    if args.profile:
        df = df[df["degradation_profile"] == args.profile].copy()
    print(f"{len(df)} frames from {csv_path}  ·  truth counts: {dict(Counter(df['truth']))}")
    df["pred"] = df.apply(
        lambda row: pred_label(row, args.vote_on, args.vote_low, args.vote_high), axis=1
    )
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
    metric_columns = [("blur", "blur_score"), ("det", "det_score"),
                      ("eye_dist", "eye_dist")]
    if {"brightness", "contrast"}.issubset(lab.columns):
        metric_columns.extend((("brightness", "brightness"), ("contrast", "contrast")))
    for name, col in metric_columns:
        print(f"  {name:8s} correct : {describe(right[col])}")
        print(f"  {name:8s} danger  : {describe(danger[col])}")

    for (version, profile), group in lab.groupby(["quality_version", "degradation_profile"],
                                                 dropna=False):
        print(f"\nquality group: {version} / {profile} · n={len(group)}")
        group_danger = group[(group["truth"] == "glasses") & (group["pred"] == "none")]
        group_right = group[group["truth"] == group["pred"]]
        sug = suggest_min_blur(group_danger["blur_score"], group_right["blur_score"])
        print("  blur-only suggestion:", f"{sug:.2f}" if sug is not None else
              "none meets 90% dangerous-miss rejection and 95% correct-frame retention")
        if version == "quality-v2" and {"brightness", "contrast"}.issubset(group.columns):
            min_blur = effective_min_blur(str(profile), args.min_blur, runtime_cfg)
            reasons = quality_gate_reasons(
                group, min_blur=min_blur, min_brightness=args.min_brightness,
                max_brightness=args.max_brightness, min_contrast=args.min_contrast,
            )
            impact = gate_impact(group, reasons)
            print(f"  candidate gate: blur>={min_blur:g}, brightness="
                  f"[{args.min_brightness:g},{args.max_brightness:g}], "
                  f"contrast>={args.min_contrast:g}")
            print(f"  blocks {impact['dangerous_rejected']}/{impact['dangerous_misses']} "
                  f"dangerous misses ({impact['dangerous_reject_rate']:.0%}); keeps "
                  f"{impact['correct_kept']}/{impact['correct_frames']} correct frames "
                  f"({impact['correct_keep_rate']:.0%}); rejects "
                  f"{impact['all_reject_rate']:.0%} overall")
            print(f"  rejection reasons: {dict(Counter(reasons.dropna()))}")

    wrong = pd.concat([danger.sort_values("blur_score"), other_wrong])
    contact_sheet(wrong, Path(args.out) / "mismatches.png")


if __name__ == "__main__":
    main()
