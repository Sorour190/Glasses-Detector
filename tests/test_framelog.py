import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from glasses_detector.framelog import COLUMNS, FrameLogger
from glasses_detector.predict import GlassesResult


def test_frame_log_keeps_degradation_and_quality_metrics(tmp_path):
    logger = FrameLogger(tmp_path)
    result = GlassesResult(
        False, 0.9, 0.1, False, True, 0.88, (0.9, 0.1, 0.0),
        eyewear_prob=0.1, blur_score=7.25, eye_dist_px=64.0,
        brightness=81.5, contrast=42.25,
        crop=np.full((24, 24, 3), 128, dtype=np.uint8),
    )

    logger.log(
        result, pred_action="pass", truth="none", degradation_profile="bad_phone",
        request_id="req-7", source_path="sources/req-7.jpg",
        source_frame_id="session:7", raw_vote="none", valid=True,
        batch_action="pass", batch_reason="ok",
        combined_action="pass", combined_reason="ok",
        quality_thresholds={
            "min_det_score": 0.6, "min_blur": 1.9, "min_eye_dist": 24,
            "min_brightness": 35, "max_brightness": 220, "min_contrast": 25,
            "vote_on": "eyewear", "vote_low": 0.2, "vote_high": 0.5,
        },
        inference_ms=12.34, ts=123.0,
    )

    with (tmp_path / "frames.csv").open(newline="") as file:
        row = next(csv.DictReader(file))
    assert row["degradation_profile"] == "bad_phone"
    assert row["quality_version"] == "quality-v2"
    assert row["brightness"] == "81.50"
    assert row["contrast"] == "42.25"
    assert row["source_frame_id"] == "session:7"
    assert row["request_id"] == "req-7"
    assert row["source_path"] == "sources/req-7.jpg"
    assert row["min_blur"] == "1.9"
    assert row["vote_on"] == "eyewear"
    assert row["vote_low"] == "0.2"
    assert row["vote_high"] == "0.5"
    assert row["inference_ms"] == "12.34"
    assert row["raw_vote"] == "none"
    assert row["batch_action"] == "pass"
    assert row["combined_action"] == "pass"
    assert row["combined_reason"] == "ok"
    assert row["sorted_path"]
    assert (tmp_path / "correct") in Path(row["sorted_path"]).parents


def test_request_source_and_runtime_records_form_a_complete_debug_trail(tmp_path):
    logger = FrameLogger(tmp_path, save_full=True)
    source = np.full((48, 64, 3), 77, dtype=np.uint8)

    source_path = logger.save_source(source, request_id="request-1", frame_idx=3, ts=10.0)
    logger.log_request(
        request_id="request-1", session_id="session-1",
        source_frame_id="session-1:3", window_id="session-1:w1", frame_idx=3,
        endpoint="/validate/glasses/stream", request_kind="stream", truth="none",
        degradation_profile="all", width=64, height=48, encoded_bytes=4567,
        status=200, error="", total_ms=87.65, source_path=source_path, ts=10.0,
    )
    logger.log_runtime({
        "checkpoint": "models/glasses_v1.pt",
        "quality_version": "quality-v2",
        "profiles": {"clean": None, "bad_phone": {"jpeg_quality": 25}},
    }, ts=9.0)

    assert source_path
    assert cv2.imread(source_path).shape == source.shape
    with (tmp_path / "requests.csv").open(newline="") as file:
        row = next(csv.DictReader(file))
    assert row == {
        "ts": "10.000", "request_id": "request-1", "session_id": "session-1",
        "source_frame_id": "session-1:3", "window_id": "session-1:w1",
        "frame_idx": "3", "endpoint": "/validate/glasses/stream",
        "request_kind": "stream", "truth": "none", "degradation_profile": "all",
        "width": "64", "height": "48", "encoded_bytes": "4567", "status": "200",
        "error": "", "total_ms": "87.65", "source_path": source_path,
    }
    runtime = json.loads((tmp_path / "runtime.jsonl").read_text().strip())
    assert runtime["ts"] == 9.0
    assert runtime["checkpoint"] == "models/glasses_v1.pt"
    assert runtime["profiles"]["bad_phone"]["jpeg_quality"] == 25


def test_encoded_source_upload_is_preserved_byte_for_byte(tmp_path):
    logger = FrameLogger(tmp_path, save_full=True)
    source = np.full((32, 40, 3), 91, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", source)
    assert ok

    source_path = logger.save_source_bytes(
        encoded.tobytes(), filename="phone-frame.png",
        request_id="exact-source", frame_idx=2, ts=11.0,
    )

    assert Path(source_path).suffix == ".png"
    assert Path(source_path).read_bytes() == encoded.tobytes()


def test_model_crop_and_degraded_input_are_saved_losslessly(tmp_path):
    logger = FrameLogger(tmp_path, save_full=True)
    rng = np.random.default_rng(42)
    full = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    crop = rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
    result = GlassesResult(
        False, 0.9, 0.1, False, True, 0.8, (0.9, 0.1, 0.0), crop=crop,
    )

    logger.log(result, pred_action="pass", truth="none", full_bgr=full, ts=12.0)

    with (tmp_path / "frames.csv").open(newline="") as file:
        row = next(csv.DictReader(file))
    assert Path(row["crop_path"]).suffix == ".png"
    assert Path(row["full_path"]).suffix == ".png"
    assert np.array_equal(cv2.imread(row["crop_path"]), crop)
    assert np.array_equal(cv2.imread(row["full_path"]), full)
    assert np.array_equal(cv2.imread(row["sorted_path"]), crop)


def test_existing_log_schema_is_migrated_before_new_rows_are_appended(tmp_path):
    old_columns = [
        "ts", "session_id", "frame_idx", "truth", "pred_action", "p", "eyewear",
        "none", "eyeglasses", "sunglasses", "det_score", "blur_score", "eye_dist",
        "valid", "reject_reason", "batch_action", "batch_reason", "crop_path", "full_path",
    ]
    old_values = ["1.000", "old", "0", "none", "pass", "0.1", "0.1", "0.9",
                  "0.1", "0.0", "0.8", "100", "60", "1", "", "pass", "ok", "", ""]
    with (tmp_path / "frames.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(old_columns)
        writer.writerow(old_values)

    logger = FrameLogger(tmp_path)
    result = GlassesResult(False, 0.9, 0.1, False, False, 0.0, (0.9, 0.1, 0.0))
    logger.log(result, pred_action="pass", degradation_profile="mild", ts=2.0)

    with (tmp_path / "frames.csv").open(newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    assert reader.fieldnames == COLUMNS
    assert rows[0]["session_id"] == "old"
    assert rows[0]["degradation_profile"] == "clean"
    assert rows[0]["quality_version"] == "quality-v1"
    assert rows[1]["degradation_profile"] == "mild"
    assert rows[1]["quality_version"] == "quality-v2"


def test_labeled_frames_are_sorted_into_correct_and_mislabelled_folders(tmp_path):
    logger = FrameLogger(tmp_path)
    image = np.full((24, 24, 3), 128, dtype=np.uint8)

    def result(wearing: bool) -> GlassesResult:
        return GlassesResult(wearing, 0.9, 0.9 if wearing else 0.1, False, True,
                             0.88, (0.1, 0.9, 0.0), crop=image)

    logger.log(result(True), pred_action="remove_glasses", truth="glasses", ts=1.0)
    logger.log(result(False), pred_action="pass", truth="glasses", ts=2.0)
    logger.log(result(True), pred_action="remove_glasses", truth="unknown", ts=3.0)

    assert len(list((tmp_path / "correct").glob("*.jpg"))) == 1
    assert len(list((tmp_path / "mislabelled").glob("*.jpg"))) == 1


def test_concurrent_logs_with_same_timestamp_never_share_image_paths(tmp_path):
    logger = FrameLogger(tmp_path, save_full=True)

    def write_frame(value: int):
        image = np.full((24, 24, 3), value, dtype=np.uint8)
        result = GlassesResult(
            False, 0.9, 0.1, False, True, 0.8, (0.9, 0.1, 0.0), crop=image,
        )
        logger.log(result, pred_action="pass", session_id="same/session", frame_idx=0,
                   full_bgr=image, ts=123.0)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(write_frame, (20, 60, 100, 140)))

    with (tmp_path / "frames.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    crop_paths = [row["crop_path"] for row in rows]
    full_paths = [row["full_path"] for row in rows]
    assert len(set(crop_paths)) == 4
    assert len(set(full_paths)) == 4
    assert all(cv2.imread(path) is not None for path in crop_paths + full_paths)


def test_no_face_variant_still_logs_its_degraded_full_frame(tmp_path):
    logger = FrameLogger(tmp_path, save_full=True)
    full = np.full((24, 24, 3), 40, dtype=np.uint8)
    result = GlassesResult(
        False, 0.0, 0.0, True, False, 0.0, (0.0, 0.0, 1.0), crop=None,
    )

    logger.log(
        result,
        pred_action="retry_capture",
        truth="glasses",
        source_frame_id="session:8",
        raw_vote="no_face",
        degradation_profile="extreme",
        valid=False,
        reject_reason="no_face",
        full_bgr=full,
        ts=124.0,
    )

    with (tmp_path / "frames.csv").open(newline="") as file:
        row = next(csv.DictReader(file))
    assert row["crop_path"] == ""
    assert row["sorted_path"] == ""
    assert row["full_path"]
    assert cv2.imread(row["full_path"]) is not None
