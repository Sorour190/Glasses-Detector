"""Opt-in per-frame logging for tuning the quality gate / aggregation rule.

Enabled by GLASSES_LOG_DIR=<dir>. Writes <dir>/frames.csv (one row per frame)
and the 160x160 ROI crop as <dir>/crops/<ts>_<idx>.jpg. Set GLASSES_LOG_FULL=1
to also keep the full submitted frame under <dir>/full/. Review with
scripts/review_log.py.
"""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .predict import GlassesResult

COLUMNS = ["ts", "session_id", "frame_idx", "truth", "pred_action", "p", "eyewear",
           "none", "eyeglasses", "sunglasses", "det_score", "blur_score", "eye_dist",
           "valid", "reject_reason", "batch_action", "batch_reason", "crop_path", "full_path"]


class FrameLogger:
    def __init__(self, log_dir: str | Path, save_full: bool = False):
        self.dir = Path(log_dir)
        self.crops = self.dir / "crops"
        self.full = self.dir / "full"
        self.crops.mkdir(parents=True, exist_ok=True)
        if save_full:
            self.full.mkdir(parents=True, exist_ok=True)
        self.save_full = save_full
        self.csv_path = self.dir / "frames.csv"
        self._lock = threading.Lock()
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="") as f:
                csv.writer(f).writerow(COLUMNS)

    def log(self, result: GlassesResult, *, pred_action: str, truth: str = "unknown",
            session_id: str = "", frame_idx: int = 0, valid: Optional[bool] = None,
            reject_reason: Optional[str] = None, batch_action: str = "",
            batch_reason: str = "", full_bgr: Optional[np.ndarray] = None,
            ts: Optional[float] = None) -> None:
        ts = time.time() if ts is None else ts
        stem = f"{int(ts * 1000)}_{frame_idx}"
        crop_path = full_path = ""
        if result.crop is not None:
            crop_path = str(self.crops / f"{stem}.jpg")
            cv2.imwrite(crop_path, result.crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if self.save_full and full_bgr is not None:
            full_path = str(self.full / f"{stem}.jpg")
            cv2.imwrite(full_path, full_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        none_p, eye_p, sun_p = (list(result.class_probs) + [0.0, 0.0, 0.0])[:3]
        row = [f"{ts:.3f}", session_id, frame_idx, truth, pred_action,
               f"{result.probability:.4f}", f"{result.eyewear_prob:.4f}", f"{none_p:.4f}", f"{eye_p:.4f}", f"{sun_p:.4f}",
               f"{result.det_score:.4f}", f"{result.blur_score:.2f}", f"{result.eye_dist_px:.1f}",
               "" if valid is None else int(valid), reject_reason or "",
               batch_action, batch_reason, crop_path, full_path]
        with self._lock, self.csv_path.open("a", newline="") as f:
            csv.writer(f).writerow(row)
