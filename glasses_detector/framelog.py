"""Opt-in per-frame logging for tuning the quality gate / aggregation rule.

Enabled by GLASSES_LOG_DIR=<dir>. Writes request, frame/profile, decision, and
runtime records plus the 160x160 ROI crop under <dir>/crops/. Set
GLASSES_LOG_FULL=1 to preserve the exact uploaded bytes under <dir>/sources/
and every degraded full frame under <dir>/full/. request_id links one captured
source to its profile rows; window_id links requests/frames to the completed
decision. Frames with ground truth are additionally sorted into
<dir>/correct/ (raw model vote matches truth) or <dir>/mislabelled/ (raw vote
disagrees). Review with scripts/review_log.py.
"""

from __future__ import annotations

import csv
import json
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .predict import QUALITY_VERSION, GlassesResult
from .degrade import DEGRADE_VERSION
from .preprocess import PREPROCESS_VERSION

COLUMNS = ["ts", "request_id", "session_id", "source_frame_id", "window_id",
           "frame_idx", "truth", "raw_vote",
           "pred_action", "p", "eyewear", "none", "eyeglasses", "sunglasses",
           "det_score", "blur_score", "brightness", "contrast", "eye_dist",
           "quality_version", "preprocess_version", "degradation_version",
           "degradation_profile", "min_det_score", "min_blur", "min_eye_dist",
           "min_brightness", "max_brightness", "min_contrast", "vote_on",
           "vote_low", "vote_high", "valid", "reject_reason", "inference_ms",
           "batch_action", "batch_reason", "combined_action", "combined_reason",
           "source_path", "crop_path", "full_path", "sorted_path"]

DECISION_COLUMNS = ["ts", "session_id", "window_id", "decision_no", "request_kind",
                    "truth", "n_source_frames", "degradation_profile",
                    "profile_verdicts", "combined_verdict"]

REQUEST_COLUMNS = ["ts", "request_id", "session_id", "source_frame_id", "window_id",
                   "frame_idx", "endpoint", "request_kind", "truth",
                   "degradation_profile", "width", "height", "encoded_bytes",
                   "status", "error", "total_ms", "source_path"]


class FrameLogger:
    def __init__(self, log_dir: str | Path, save_full: bool = False):
        self.dir = Path(log_dir)
        self.crops = self.dir / "crops"
        self.full = self.dir / "full"
        self.sources = self.dir / "sources"
        self.correct = self.dir / "correct"
        self.mislabelled = self.dir / "mislabelled"
        self.crops.mkdir(parents=True, exist_ok=True)
        self.correct.mkdir(parents=True, exist_ok=True)
        self.mislabelled.mkdir(parents=True, exist_ok=True)
        if save_full:
            self.full.mkdir(parents=True, exist_ok=True)
            self.sources.mkdir(parents=True, exist_ok=True)
        self.save_full = save_full
        self.csv_path = self.dir / "frames.csv"
        self.decisions_path = self.dir / "decisions.csv"
        self.requests_path = self.dir / "requests.csv"
        self.runtime_path = self.dir / "runtime.jsonl"
        self._lock = threading.Lock()
        if self.csv_path.exists():
            self._migrate_schema()
        else:
            with self.csv_path.open("w", newline="") as f:
                csv.writer(f).writerow(COLUMNS)
        if not self.decisions_path.exists():
            with self.decisions_path.open("w", newline="") as f:
                csv.writer(f).writerow(DECISION_COLUMNS)
        if not self.requests_path.exists():
            with self.requests_path.open("w", newline="") as f:
                csv.writer(f).writerow(REQUEST_COLUMNS)

    def _migrate_schema(self) -> None:
        with self.csv_path.open(newline="") as file:
            reader = csv.DictReader(file)
            old_columns = reader.fieldnames or []
            if old_columns == COLUMNS:
                return
            if not set(old_columns).issubset(COLUMNS):
                unknown = sorted(set(old_columns) - set(COLUMNS))
                raise RuntimeError(f"cannot migrate frames.csv with unknown columns: {unknown}")
            rows = list(reader)

        backup = self.csv_path.with_name("frames.pre-debug-v3.csv")
        if not backup.exists():
            shutil.copy2(self.csv_path, backup)
        temp = self.csv_path.with_suffix(".csv.tmp")
        with temp.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=COLUMNS)
            writer.writeheader()
            for row in rows:
                migrated = {column: row.get(column, "") for column in COLUMNS}
                migrated["quality_version"] = migrated["quality_version"] or "quality-v1"
                migrated["degradation_profile"] = migrated["degradation_profile"] or "clean"
                writer.writerow(migrated)
        temp.replace(self.csv_path)

    def log(self, result: GlassesResult, *, pred_action: str, truth: str = "unknown",
            session_id: str = "", source_frame_id: str = "", frame_idx: int = 0,
            window_id: str = "", request_id: str = "", source_path: str = "",
            raw_vote: str = "", valid: Optional[bool] = None,
            reject_reason: Optional[str] = None, batch_action: str = "",
            batch_reason: str = "", degradation_profile: str = "clean",
            combined_action: str = "", combined_reason: str = "",
            quality_thresholds: Optional[dict] = None,
            inference_ms: Optional[float] = None,
            full_bgr: Optional[np.ndarray] = None,
            ts: Optional[float] = None) -> None:
        ts = time.time() if ts is None else ts
        with self._lock:
            safe_profile = "".join(c for c in degradation_profile if c.isalnum() or c in "_-")
            stem = f"{int(ts * 1000)}_{frame_idx}_{safe_profile}_{uuid.uuid4().hex[:12]}"
            crop_path = full_path = sorted_path = ""
            if result.crop is not None:
                crop_path = str(self.crops / f"{stem}.png")
                cv2.imwrite(crop_path, result.crop, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            if self.save_full and full_bgr is not None:
                full_path = str(self.full / f"{stem}.png")
                cv2.imwrite(full_path, full_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            if not raw_vote:
                raw_vote = ("no_face" if not result.face_found else
                            ("glasses" if result.wearing_glasses else "none"))
            if truth in ("glasses", "none") and result.face_found and result.crop is not None:
                sorted_dir = self.correct if raw_vote == truth else self.mislabelled
                sorted_path = str(sorted_dir / f"{stem}.png")
                cv2.imwrite(sorted_path, result.crop, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            none_p, eye_p, sun_p = (list(result.class_probs) + [0.0, 0.0, 0.0])[:3]
            thresholds = quality_thresholds or {}
            row = [f"{ts:.3f}", request_id, session_id, source_frame_id, window_id,
                   frame_idx, truth, raw_vote,
                   pred_action,
                   f"{result.probability:.4f}", f"{result.eyewear_prob:.4f}", f"{none_p:.4f}", f"{eye_p:.4f}", f"{sun_p:.4f}",
                   f"{result.det_score:.4f}", f"{result.blur_score:.2f}",
                   f"{result.brightness:.2f}", f"{result.contrast:.2f}",
                   f"{result.eye_dist_px:.1f}", QUALITY_VERSION, PREPROCESS_VERSION,
                   DEGRADE_VERSION, degradation_profile,
                   thresholds.get("min_det_score", ""), thresholds.get("min_blur", ""),
                   thresholds.get("min_eye_dist", ""), thresholds.get("min_brightness", ""),
                   thresholds.get("max_brightness", ""), thresholds.get("min_contrast", ""),
                   thresholds.get("vote_on", ""), thresholds.get("vote_low", ""),
                   thresholds.get("vote_high", ""),
                   "" if valid is None else int(valid), reject_reason or "",
                   "" if inference_ms is None else f"{inference_ms:.2f}",
                   batch_action, batch_reason, combined_action, combined_reason,
                   source_path, crop_path, full_path, sorted_path]
            with self.csv_path.open("a", newline="") as f:
                csv.writer(f).writerow(row)

    def save_source(self, bgr: np.ndarray, *, request_id: str, frame_idx: int = 0,
                    ts: Optional[float] = None) -> str:
        """Save one original decoded source frame and return its path."""
        if not self.save_full:
            return ""
        ts = time.time() if ts is None else ts
        safe_request = "".join(c for c in request_id if c.isalnum() or c in "_-")
        path = self.sources / f"{int(ts * 1000)}_{frame_idx}_{safe_request}.jpg"
        with self._lock:
            if not cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"failed to save source frame: {path}")
        return str(path)

    def save_source_bytes(self, data: bytes, *, filename: str, request_id: str,
                          frame_idx: int = 0, ts: Optional[float] = None) -> str:
        """Preserve the exact uploaded image bytes without JPEG recompression."""
        if not self.save_full:
            return ""
        ts = time.time() if ts is None else ts
        safe_request = "".join(c for c in request_id if c.isalnum() or c in "_-")
        suffix = Path(filename).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            suffix = ".img"
        path = self.sources / f"{int(ts * 1000)}_{frame_idx}_{safe_request}{suffix}"
        with self._lock:
            with path.open("wb") as file:
                file.write(data)
        return str(path)

    def log_request(self, *, request_id: str, session_id: str, source_frame_id: str,
                    window_id: str, frame_idx: int, endpoint: str,
                    request_kind: str, truth: str, degradation_profile: str,
                    width: int, height: int, encoded_bytes: int, status: int,
                    error: str, total_ms: float, source_path: str,
                    ts: Optional[float] = None) -> None:
        ts = time.time() if ts is None else ts
        row = [f"{ts:.3f}", request_id, session_id, source_frame_id, window_id,
               frame_idx, endpoint, request_kind, truth, degradation_profile,
               width, height, encoded_bytes, status, error, f"{total_ms:.2f}",
               source_path]
        with self._lock:
            with self.requests_path.open("a", newline="") as f:
                csv.writer(f).writerow(row)

    def log_runtime(self, payload: dict, *, ts: Optional[float] = None) -> None:
        ts = time.time() if ts is None else ts
        record = dict(payload)
        record["ts"] = ts
        with self._lock:
            with self.runtime_path.open("a") as f:
                f.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    def log_decision(self, *, session_id: str, window_id: str, decision_no: int,
                     request_kind: str, truth: str, n_source_frames: int,
                     degradation_profile: str, profile_verdicts: dict,
                     combined_verdict: dict, ts: Optional[float] = None) -> None:
        """Append one machine-readable verdict for a completed request/window."""
        ts = time.time() if ts is None else ts
        row = [
            f"{ts:.3f}", session_id, window_id, decision_no, request_kind, truth,
            n_source_frames, degradation_profile,
            json.dumps(profile_verdicts, sort_keys=True, separators=(",", ":")),
            json.dumps(combined_verdict, sort_keys=True, separators=(",", ":")),
        ]
        with self._lock:
            with self.decisions_path.open("a", newline="") as f:
                csv.writer(f).writerow(row)
