import csv
import json
import io
import threading
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi import HTTPException, UploadFile

from glasses_detector import api
from glasses_detector.aggregate import AggregateConfig
from glasses_detector.degrade import BAD_CAMERA_PROFILES, apply_bad_camera
from glasses_detector.framelog import FrameLogger
from glasses_detector.predict import GlassesResult


class ProfileSelectParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_profile_select = False
        self.options = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if attributes.get("id"):
            self.ids.add(attributes["id"])
        if tag == "select" and attributes.get("id") == "degrade":
            self.in_profile_select = True
        elif tag == "option" and self.in_profile_select:
            self.options.append((attributes.get("value"), "selected" in attributes))

    def handle_endtag(self, tag):
        if tag == "select" and self.in_profile_select:
            self.in_profile_select = False


class RecordingDetector:
    def __init__(self):
        self.frames = []
        self.band = (0.15, 0.235)
        self.threshold = 0.235
        self.uncertainty_band = 0.15
        self.result = GlassesResult(
            False, 0.95, 0.05, False, True, 0.9, (0.95, 0.05, 0.0),
            eyewear_prob=0.05, blur_score=10.0, eye_dist_px=60.0,
            brightness=100.0, contrast=80.0,
        )

    def predict(self, bgr):
        self.frames.append(bgr.copy())
        return self.result


class BlockingDetector(RecordingDetector):
    def __init__(self):
        super().__init__()
        self.first_entered = threading.Event()
        self.second_entered = threading.Event()
        self.release_first = threading.Event()
        self._calls = 0
        self._calls_lock = threading.Lock()

    def predict(self, bgr):
        with self._calls_lock:
            self._calls += 1
            call = self._calls
        if call == 1:
            self.first_entered.set()
            assert self.release_first.wait(timeout=5)
        else:
            self.second_entered.set()
        return super().predict(bgr)


def _frame(value: int) -> np.ndarray:
    yy, xx = np.indices((96, 128))
    gray = ((xx * 3 + yy * 5 + value) % 256).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _upload(frame: np.ndarray, name: str = "frame.jpg") -> UploadFile:
    ok, encoded = cv2.imencode(".png", frame)
    assert ok
    return UploadFile(io.BytesIO(encoded.tobytes()), filename=name)


def _install_detector(monkeypatch) -> RecordingDetector:
    detector = RecordingDetector()
    monkeypatch.setattr(api, "_detector", detector)
    monkeypatch.setattr(api, "_logger", None)
    monkeypatch.setattr(api, "_agg_cfg", AggregateConfig())
    return detector


def test_startup_logs_complete_runtime_configuration(monkeypatch, tmp_path):
    class StartupDetector:
        def __init__(self, checkpoint, device, threshold):
            self.checkpoint = checkpoint
            self.device = device
            self.threshold = threshold
            self.band = (0.15, 0.235)
            self.temperature = 1.1
            self.uncertainty_band = 0.15

    monkeypatch.setattr(api, "GlassesDetector", StartupDetector)
    monkeypatch.setattr(api, "_logger", None)
    monkeypatch.setenv("GLASSES_CHECKPOINT", "models/test-runtime.pt")
    monkeypatch.setenv("GLASSES_DEVICE", "cpu")
    monkeypatch.setenv("GLASSES_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("GLASSES_LOG_FULL", "1")

    api.load_model()

    runtime = json.loads((tmp_path / "runtime.jsonl").read_text().strip())
    assert runtime["checkpoint"] == "models/test-runtime.pt"
    assert runtime["device"] == "cpu"
    assert runtime["save_full"] is True
    assert runtime["quality_version"] == "quality-v2"
    assert runtime["preprocess_version"] == "roi-v1"
    assert runtime["degradation_version"] == "degrade-v1"
    assert runtime["degradation_profiles"]["extreme"]["jpeg_quality"] == 20
    assert runtime["aggregate"]["profile_quality"]["bad_phone"]["min_blur"] == 1.9
    assert runtime["model_band"] == [0.15, 0.235]


def test_single_frame_profile_is_applied_before_detection(monkeypatch):
    detector = _install_detector(monkeypatch)
    frame = _frame(7)

    response = api.validate_glasses(
        _upload(frame), truth="none", session_id="s1", degradation_profile="bad_phone"
    )

    assert np.array_equal(detector.frames[0], apply_bad_camera(frame, "bad_phone", index=0))
    assert response.degradation_profile == "bad_phone"
    assert response.quality_version == "quality-v2"
    assert response.brightness == 100.0
    assert response.contrast == 80.0


def test_single_source_frame_is_evaluated_at_all_four_qualities(monkeypatch):
    detector = _install_detector(monkeypatch)
    frame = _frame(8)

    response = api.validate_glasses(
        _upload(frame), truth="none", session_id="all-one", degradation_profile="all"
    )

    assert len(detector.frames) == 4
    for index, profile in enumerate(BAD_CAMERA_PROFILES):
        expected = apply_bad_camera(frame, profile, index=0)
        assert np.array_equal(detector.frames[index], expected)
    assert list(response.profile_results) == list(BAD_CAMERA_PROFILES)
    assert response.degradation_profile == "all"
    assert response.action == "pass"


def test_all_quality_variants_and_combined_result_are_fully_logged(monkeypatch, tmp_path):
    detector = _install_detector(monkeypatch)
    detector.result.crop = np.full((24, 24, 3), 128, dtype=np.uint8)
    monkeypatch.setattr(api, "_logger", FrameLogger(tmp_path, save_full=True))

    response = api.validate_glasses(
        _upload(_frame(9)), truth="none", session_id="logged", degradation_profile="all"
    )

    with (tmp_path / "frames.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 4
    assert [row["degradation_profile"] for row in rows] == list(BAD_CAMERA_PROFILES)
    assert {row["source_frame_id"] for row in rows} == {"logged:0"}
    assert len({row["ts"] for row in rows}) == 1
    assert {row["raw_vote"] for row in rows} == {"none"}
    assert {row["combined_action"] for row in rows} == {response.action}
    assert {row["batch_action"] for row in rows} == {""}
    assert all(Path(row["crop_path"]).is_file() for row in rows)
    assert all(Path(row["full_path"]).is_file() for row in rows)
    assert all(Path(row["sorted_path"]).parent == tmp_path / "correct" for row in rows)


def test_single_all_links_source_request_frames_and_effective_thresholds(monkeypatch, tmp_path):
    detector = _install_detector(monkeypatch)
    detector.result.crop = np.full((24, 24, 3), 128, dtype=np.uint8)
    monkeypatch.setattr(api, "_logger", FrameLogger(tmp_path, save_full=True))
    monkeypatch.setattr(api, "_agg_cfg", AggregateConfig(profile_quality={
        "clean": {"min_blur": 2.2}, "mild": {"min_blur": 2.1},
        "bad_phone": {"min_blur": 1.9}, "extreme": {"min_blur": 1.7},
    }))

    api.validate_glasses(
        _upload(_frame(19)), truth="none", session_id="debug", degradation_profile="all"
    )

    with (tmp_path / "requests.csv").open(newline="") as file:
        requests = list(csv.DictReader(file))
    with (tmp_path / "frames.csv").open(newline="") as file:
        frames = list(csv.DictReader(file))
    assert len(requests) == 1
    request = requests[0]
    assert request["endpoint"] == "/validate/glasses"
    assert request["status"] == "200"
    assert request["error"] == ""
    assert request["width"] == "128" and request["height"] == "96"
    assert int(request["encoded_bytes"]) > 0
    assert float(request["total_ms"]) >= 0
    assert Path(request["source_path"]).is_file()
    assert {row["request_id"] for row in frames} == {request["request_id"]}
    assert {row["source_path"] for row in frames} == {request["source_path"]}
    assert {row["window_id"] for row in frames} == {request["window_id"]}
    assert {row["degradation_profile"]: float(row["min_blur"]) for row in frames} == {
        "clean": 2.2, "mild": 2.1, "bad_phone": 1.9, "extreme": 1.7,
    }
    assert {row["vote_on"] for row in frames} == {"eyewear"}
    assert all(float(row["inference_ms"]) >= 0 for row in frames)


def test_invalid_image_request_is_logged_with_error_status(monkeypatch, tmp_path):
    _install_detector(monkeypatch)
    monkeypatch.setattr(api, "_logger", FrameLogger(tmp_path, save_full=True))
    upload = UploadFile(io.BytesIO(b"definitely-not-an-image"), filename="broken.jpg")

    with pytest.raises(HTTPException) as error:
        api.validate_glasses(
            upload, truth="unknown", session_id="broken", degradation_profile="all"
        )

    assert error.value.status_code == 400
    with (tmp_path / "requests.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert rows[0]["status"] == "400"
    assert rows[0]["error"] == "File is not a valid image"
    assert rows[0]["encoded_bytes"] == str(len(b"definitely-not-an-image"))
    assert rows[0]["source_path"] == ""


def test_batch_degrades_every_frame_with_its_own_deterministic_index(monkeypatch):
    detector = _install_detector(monkeypatch)
    frames = [_frame(10), _frame(20)]

    response = api.validate_glasses_batch(
        [_upload(frame, f"{i}.png") for i, frame in enumerate(frames)],
        truth="none", session_id="batch", degradation_profile="mild",
    )

    assert len(detector.frames) == 2
    assert np.array_equal(detector.frames[0], apply_bad_camera(frames[0], "mild", index=0))
    assert np.array_equal(detector.frames[1], apply_bad_camera(frames[1], "mild", index=1))
    assert all(frame.degradation_profile == "mild" for frame in response.frames)


def test_batch_evaluates_every_source_frame_at_all_qualities(monkeypatch):
    detector = _install_detector(monkeypatch)
    frames = [_frame(31), _frame(32), _frame(33)]

    response = api.validate_glasses_batch(
        [_upload(frame, f"{i}.png") for i, frame in enumerate(frames)],
        truth="none",
        session_id="all-batch",
        degradation_profile="all",
    )

    assert len(detector.frames) == 12
    assert list(response.profile_frames) == list(BAD_CAMERA_PROFILES)
    assert all(len(items) == 3 for items in response.profile_frames.values())
    assert {name: verdict.action for name, verdict in response.profile_verdicts.items()} == {
        profile: "pass" for profile in BAD_CAMERA_PROFILES
    }
    assert response.verdict.action == "pass"


def test_batch_all_logs_one_source_request_and_four_frames_per_capture(monkeypatch, tmp_path):
    detector = _install_detector(monkeypatch)
    detector.result.crop = np.full((24, 24, 3), 128, dtype=np.uint8)
    monkeypatch.setattr(api, "_logger", FrameLogger(tmp_path, save_full=True))

    api.validate_glasses_batch(
        [_upload(_frame(80 + index), f"{index}.png") for index in range(3)],
        truth="none", session_id="debug-batch", degradation_profile="all",
    )

    with (tmp_path / "requests.csv").open(newline="") as file:
        requests = list(csv.DictReader(file))
    with (tmp_path / "frames.csv").open(newline="") as file:
        frames = list(csv.DictReader(file))
    assert len(requests) == 3
    assert len(frames) == 12
    assert {row["endpoint"] for row in requests} == {"/validate/glasses/batch"}
    assert len({row["window_id"] for row in requests + frames}) == 1
    for request in requests:
        matching = [row for row in frames if row["request_id"] == request["request_id"]]
        assert len(matching) == 4
        assert {row["source_frame_id"] for row in matching} == {request["source_frame_id"]}
        assert {row["source_path"] for row in matching} == {request["source_path"]}
        assert Path(request["source_path"]).is_file()


def test_batch_decode_error_logs_every_uploaded_source(monkeypatch, tmp_path):
    _install_detector(monkeypatch)
    monkeypatch.setattr(api, "_logger", FrameLogger(tmp_path, save_full=True))
    broken = UploadFile(io.BytesIO(b"broken-batch-image"), filename="broken.jpg")

    with pytest.raises(HTTPException) as error:
        api.validate_glasses_batch(
            [_upload(_frame(91), "valid.png"), broken], truth="none",
            session_id="broken-batch", degradation_profile="all",
        )

    assert error.value.status_code == 400
    with (tmp_path / "requests.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert {row["status"] for row in rows} == {"400"}
    assert {row["error"] for row in rows} == {"File is not a valid image"}
    assert Path(rows[0]["source_path"]).is_file()
    assert rows[1]["source_path"] == ""
    assert rows[1]["encoded_bytes"] == str(len(b"broken-batch-image"))


def test_batch_inference_error_is_logged_after_source_is_saved(monkeypatch, tmp_path):
    detector = _install_detector(monkeypatch)
    monkeypatch.setattr(api, "_logger", FrameLogger(tmp_path, save_full=True))

    def fail_predict(_bgr):
        raise RuntimeError("detector exploded")

    detector.predict = fail_predict
    with pytest.raises(RuntimeError, match="detector exploded"):
        api.validate_glasses_batch(
            [_upload(_frame(92))], truth="none", session_id="failed-inference",
            degradation_profile="all",
        )

    with (tmp_path / "requests.csv").open(newline="") as file:
        row = next(csv.DictReader(file))
    assert row["status"] == "500"
    assert row["error"] == "RuntimeError: detector exploded"
    assert Path(row["source_path"]).is_file()


def test_invalid_profile_returns_a_client_error(monkeypatch):
    _install_detector(monkeypatch)

    with pytest.raises(HTTPException) as error:
        api.validate_glasses(_upload(_frame(1)), truth="unknown", degradation_profile="potato")

    assert error.value.status_code == 400
    assert "unknown bad-camera profile" in error.value.detail


def test_single_frame_bad_quality_retries_even_when_model_votes_none(monkeypatch):
    detector = _install_detector(monkeypatch)
    detector.result.brightness = 20.0

    response = api.validate_glasses(
        _upload(_frame(2)), truth="unknown", degradation_profile="clean"
    )

    assert response.action == "retry_capture"
    assert response.uncertain is True
    assert response.quality_valid is False
    assert response.quality_reject_reason == "too_dark"


def test_single_profile_response_uses_its_profile_specific_blur_gate(monkeypatch):
    detector = _install_detector(monkeypatch)
    detector.result.blur_score = 2.0
    monkeypatch.setattr(
        api,
        "_agg_cfg",
        AggregateConfig(profile_quality={"bad_phone": {"min_blur": 1.9}}),
    )

    clean = api.validate_glasses(
        _upload(_frame(12)), truth="none", degradation_profile="clean"
    )
    bad_phone = api.validate_glasses(
        _upload(_frame(12)), truth="none", degradation_profile="bad_phone"
    )

    assert clean.action == "retry_capture" and clean.quality_reject_reason == "blurry"
    assert bad_phone.action == "pass" and bad_phone.quality_valid is True


def test_rejected_frame_keeps_raw_glasses_result_for_telemetry(monkeypatch):
    detector = _install_detector(monkeypatch)
    detector.result = GlassesResult(
        True, 0.9, 0.8, False, True, 0.9, (0.1, 0.8, 0.1),
        eyewear_prob=0.9, blur_score=10.0, eye_dist_px=60.0,
        brightness=20.0, contrast=80.0,
    )

    response = api.validate_glasses(
        _upload(_frame(5)), truth="glasses", degradation_profile="clean"
    )

    assert response.wearing_glasses is True
    assert response.action == "retry_capture"
    assert response.quality_valid is False


def test_test_page_defaults_to_evaluating_all_qualities():
    parser = ProfileSelectParser()
    parser.feed(api.index())

    assert parser.options == [
        ("all", True),
        ("clean", False),
        ("mild", False),
        ("bad_phone", False),
        ("extreme", False),
    ]
    assert {"photo-profiles", "lv-profiles", "vprofiles"}.issubset(parser.ids)


def test_switching_stream_profile_starts_a_fresh_decision_window(monkeypatch):
    _install_detector(monkeypatch)
    monkeypatch.setattr(api, "_sessions", {})

    first = api.validate_glasses_stream(
        _upload(_frame(3)), session_id="stream", truth="none", reset=True,
        degradation_profile="clean",
    )
    second = api.validate_glasses_stream(
        _upload(_frame(4)), session_id="stream", truth="none", reset=False,
        degradation_profile="bad_phone",
    )

    assert first.window == 1
    assert second.window == 1


def test_stream_combines_four_profile_windows_into_one_decision(monkeypatch):
    detector = _install_detector(monkeypatch)
    monkeypatch.setattr(api, "_sessions", {})

    responses = [
        api.validate_glasses_stream(
            _upload(_frame(40 + index)),
            session_id="all-stream",
            truth="none",
            reset=index == 0,
            degradation_profile="all",
        )
        for index in range(5)
    ]

    assert len(detector.frames) == 20
    assert all(list(response.profile_frames) == list(BAD_CAMERA_PROFILES)
               for response in responses)
    assert all(response.source_action == "pass" for response in responses)
    final = responses[-1]
    assert final.window == 0
    assert final.decision_no == 1
    assert {name: verdict.action for name, verdict in final.profile_verdicts.items()} == {
        profile: "pass" for profile in BAD_CAMERA_PROFILES
    }
    assert final.verdict.action == "pass"


def test_stream_logs_profile_batch_verdicts_only_when_window_completes(monkeypatch, tmp_path):
    detector = _install_detector(monkeypatch)
    detector.result.crop = np.full((24, 24, 3), 128, dtype=np.uint8)
    monkeypatch.setattr(api, "_logger", FrameLogger(tmp_path, save_full=True))
    monkeypatch.setattr(api, "_sessions", {})

    for index in range(5):
        api.validate_glasses_stream(
            _upload(_frame(50 + index)),
            session_id="logged-stream",
            truth="none",
            reset=index == 0,
            degradation_profile="all",
        )

    with (tmp_path / "frames.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    with (tmp_path / "requests.csv").open(newline="") as file:
        requests = list(csv.DictReader(file))
    assert len(rows) == 20
    assert len(requests) == 5
    assert {row["endpoint"] for row in requests} == {"/validate/glasses/stream"}
    for index in range(5):
        source_rows = [row for row in rows if row["frame_idx"] == str(index)]
        assert len(source_rows) == 4
        assert len({row["source_frame_id"] for row in source_rows}) == 1
        assert len({row["ts"] for row in source_rows}) == 1
        expected = "pass" if index == 4 else ""
        assert {row["batch_action"] for row in source_rows} == {expected}

    assert len({row["window_id"] for row in rows}) == 1
    window_id = rows[0]["window_id"]
    assert window_id
    assert {row["window_id"] for row in requests} == {window_id}
    for request in requests:
        matching = [row for row in rows if row["request_id"] == request["request_id"]]
        assert len(matching) == 4
        assert {row["source_path"] for row in matching} == {request["source_path"]}
        assert Path(request["source_path"]).is_file()
    with (tmp_path / "decisions.csv").open(newline="") as file:
        decisions = list(csv.DictReader(file))
    assert len(decisions) == 1
    assert decisions[0]["window_id"] == window_id
    assert decisions[0]["n_source_frames"] == "5"
    profile_verdicts = json.loads(decisions[0]["profile_verdicts"])
    combined = json.loads(decisions[0]["combined_verdict"])
    assert {name: item["action"] for name, item in profile_verdicts.items()} == {
        profile: "pass" for profile in BAD_CAMERA_PROFILES
    }
    assert combined["action"] == "pass"


def test_stream_inference_error_logs_request_and_exact_source(monkeypatch, tmp_path):
    detector = _install_detector(monkeypatch)
    monkeypatch.setattr(api, "_logger", FrameLogger(tmp_path, save_full=True))
    monkeypatch.setattr(api, "_sessions", {})

    def fail_predict(_bgr):
        raise RuntimeError("stream detector exploded")

    detector.predict = fail_predict
    upload = _upload(_frame(62), "stream-source.png")
    with pytest.raises(RuntimeError, match="stream detector exploded"):
        api.validate_glasses_stream(
            upload, session_id="broken-stream", truth="none", reset=True,
            degradation_profile="all",
        )

    with (tmp_path / "requests.csv").open(newline="") as file:
        row = next(csv.DictReader(file))
    assert row["status"] == "500"
    assert row["error"] == "RuntimeError: stream detector exploded"
    assert Path(row["source_path"]).suffix == ".png"
    assert Path(row["source_path"]).is_file()


def test_stream_reset_gives_reused_session_a_new_source_generation(monkeypatch, tmp_path):
    detector = _install_detector(monkeypatch)
    detector.result.crop = np.full((24, 24, 3), 128, dtype=np.uint8)
    monkeypatch.setattr(api, "_logger", FrameLogger(tmp_path))
    monkeypatch.setattr(api, "_sessions", {})

    for value in (70, 71):
        api.validate_glasses_stream(
            _upload(_frame(value)),
            session_id="reset-session",
            truth="none",
            reset=True,
            degradation_profile="all",
        )

    with (tmp_path / "frames.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    source_ids = [rows[0]["source_frame_id"], rows[4]["source_frame_id"]]
    assert source_ids[0] != source_ids[1]
    assert all(sum(row["source_frame_id"] == source_id for row in rows) == 4
               for source_id in source_ids)


def test_same_session_stream_requests_are_serialized_across_inference(monkeypatch):
    detector = BlockingDetector()
    monkeypatch.setattr(api, "_detector", detector)
    monkeypatch.setattr(api, "_logger", None)
    monkeypatch.setattr(api, "_agg_cfg", AggregateConfig())
    monkeypatch.setattr(api, "_sessions", {})

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(
            api.validate_glasses_stream, _upload(_frame(6)), "concurrent",
            "none", True, "clean",
        )
        assert detector.first_entered.wait(timeout=2)
        second_future = pool.submit(
            api.validate_glasses_stream, _upload(_frame(7)), "concurrent",
            "none", False, "bad_phone",
        )
        try:
            assert not detector.second_entered.wait(timeout=0.2)
        finally:
            detector.release_first.set()
        first = first_future.result(timeout=5)
        second = second_future.result(timeout=5)

    assert first.window == 1
    assert second.window == 1


def test_in_flight_stream_session_is_not_evicted_after_ttl(monkeypatch):
    detector = BlockingDetector()
    now = {"value": 0.0}
    monkeypatch.setattr(api, "_detector", detector)
    monkeypatch.setattr(api, "_logger", None)
    monkeypatch.setattr(api, "_agg_cfg", AggregateConfig())
    monkeypatch.setattr(api, "_sessions", {})
    monkeypatch.setattr(api, "SESSION_TTL_S", 1.0)
    monkeypatch.setattr(api.time, "time", lambda: now["value"])

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(
            api.validate_glasses_stream, _upload(_frame(60)), "ttl-session",
            "none", True, "clean",
        )
        assert detector.first_entered.wait(timeout=2)
        now["value"] = 2.0
        second_future = pool.submit(
            api.validate_glasses_stream, _upload(_frame(61)), "ttl-session",
            "none", False, "clean",
        )
        try:
            assert not detector.second_entered.wait(timeout=0.2)
        finally:
            detector.release_first.set()
        first = first_future.result(timeout=5)
        second = second_future.result(timeout=5)

    assert first.frame_no == 1
    assert second.frame_no == 2
