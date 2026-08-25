"""FastAPI app: upload a photo, stream a live camera, or scan a video for a
glasses / no-glasses verdict.

Run from the repo root:
    uvicorn glasses_detector.api:app --host 127.0.0.1 --port 8000
then open http://127.0.0.1:8000 in a browser and drop a photo in.

Phone / live camera: browsers only expose the camera on HTTPS, so serve with
    scripts/serve_https.sh            (self-signed cert, port 8443)
and open https://<this-machine-ip>:8443 on the phone (accept the cert warning).
The Live and Video tabs post JPEG frames to /validate/glasses one at a time.

Env overrides:
    GLASSES_CHECKPOINT  path to a .pt checkpoint, or a HuggingFace checkpoint
                        directory containing config.json + model.safetensors
                        (default: models/glasses_v1.pt, else newest runs/*/best.pt)
    GLASSES_HF_INPUT    crop fed to an HF checkpoint: face (SCRFD box + 30%
                        margin, default) | roi (eye-region crop) | full (frame)
    GLASSES_THRESHOLD   fallback decision threshold (default 0.5) — only used when
                        models/threshold.json is absent; the calibrated band wins otherwise
    GLASSES_DEVICE      cuda | cpu (default: cpu, so a training run can keep the GPU)
    GLASSES_AGG_*       multi-frame rule overrides, e.g. GLASSES_AGG_MIN_BLUR=4.5
                        (see aggregate.AggregateConfig)
    GLASSES_LOG_DIR     write runtime, request, per-profile frame, decision, and
                        optional ground-truth records for scripts/review_log.py
    GLASSES_LOG_FULL=1  preserve exact uploads plus every degraded full frame

Endpoints:
    GET  /models                  registered models (default checkpoint + any
                                  models/<dir>/ HF checkpoint) and which is active
    POST /models/select           swap the active model (the UI dropdown uses this)
    POST /validate/glasses        one frame  -> per-frame verdict (kept for old clients)
    POST /validate/glasses/batch  a burst of frames (5 at 5 fps ≈ 1 s) -> per-frame
                                  results + ONE aggregated verdict (stateless).
    POST /validate/glasses/stream one frame at a time with a session_id -> per-frame
                                  result on every call + a verdict on every 5th frame
                                  (tumbling window). This is what the Live tab and a
                                  5 fps production client use.

The full production path runs on every upload: optional deterministic full-frame
bad-camera simulation (or all four profiles from one source frame) -> SCRFD face/landmark detection
-> ROI-v1 eye-region crop -> 3-class model -> eyewear = P(eyeglasses)+P(sunglasses).
Sunglasses count as glasses for the verification gate (AggregateConfig.vote_on="eyewear";
set "eyeglasses" for the legacy P(eyeglasses)-only semantics).
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .aggregate import (AggregateConfig, AggregateVerdict, aggregate,
                        combine_profile_verdicts, judge_frame)
from .degrade import BAD_CAMERA_PROFILES, DEGRADE_VERSION, apply_bad_camera
from .framelog import FrameLogger
from .preprocess import PREPROCESS_VERSION
from .predict import QUALITY_VERSION, GlassesDetector, GlassesResult

app = FastAPI(title="Glasses Detection", version="2.1.0")

_detector: GlassesDetector | None = None
_agg_cfg: AggregateConfig = AggregateConfig()
_logger: FrameLogger | None = None
MAX_BATCH = 10
TRUTH_VALUES = ("glasses", "none", "unknown")

# Switchable model registry: name -> checkpoint path. The default checkpoint is
# registered at startup; models/<dir>/ containing config.json + model.safetensors
# (HuggingFace checkpoints) are auto-discovered. GET /models lists them and
# POST /models/select swaps the active detector (the UI has a dropdown for this).
_model_registry: dict[str, str] = {}
_loaded_detectors: dict[str, GlassesDetector] = {}
_active_model: str = ""
_model_lock = threading.Lock()
_device: str = "cpu"
_fallback_threshold: float = 0.5


class ValidationResponse(BaseModel):
    wearing_glasses: bool      # raw configured model vote; action may still retry on quality
    confidence: float
    probability: float         # P(eyeglasses) — legacy field, telemetry
    eyewear_prob: float = 0.0  # P(eyeglasses) + P(sunglasses): what the decision uses
    uncertain: bool
    face_found: bool
    det_score: float
    class_probs: dict          # {none, eyeglasses, sunglasses} — telemetry
    action: str                # "pass" | "remove_glasses" | "retry_capture"
    blur_score: float = 0.0    # denoised Laplacian variance of the eye crop
    eye_dist_px: float = 0.0   # inter-ocular distance in the submitted frame
    brightness: float = 0.0    # mean grayscale value of the model crop
    contrast: float = 0.0      # grayscale p95-p5 dynamic range
    quality_version: str = QUALITY_VERSION
    quality_valid: bool = False
    quality_reject_reason: Optional[str] = None
    degradation_profile: str = "clean"
    profile_results: dict[str, dict] = Field(default_factory=dict)


class BatchFrame(ValidationResponse):
    index: int
    valid: bool                # passed the quality gate and voted
    reject_reason: Optional[str] = None   # face/distance/blur/exposure/contrast reason
    vote: Optional[str] = None            # glasses | none | unsure


class BatchVerdict(BaseModel):
    action: str                # "pass" | "remove_glasses" | "retry_capture"
    reason: str                # ok | glasses | mixed | dominant quality rejection reason
    n_frames: int
    n_valid: int
    glasses_votes: int
    none_votes: int
    unsure_votes: int
    mean_p: float
    max_p: float
    rejected: dict


class BatchResponse(BaseModel):
    session_id: str
    verdict: BatchVerdict
    frames: List[BatchFrame]
    profile_frames: dict[str, List[BatchFrame]] = Field(default_factory=dict)
    profile_verdicts: dict[str, BatchVerdict] = Field(default_factory=dict)
    config: dict               # the AggregateConfig used, for telemetry


def _default_checkpoint() -> str:
    if Path("models/glasses_v1.pt").exists():          # shipped release model
        return "models/glasses_v1.pt"
    runs = sorted(Path("runs").glob("*/best.pt"), key=os.path.getmtime)
    if not runs:
        raise FileNotFoundError("no model found; train one or fetch models/glasses_v1.pt")
    return str(runs[-1])


def _model_name(path: str) -> str:
    p = Path(path)
    return p.stem if p.suffix else p.name


def _discover_models(default_checkpoint: str) -> None:
    _model_registry[_model_name(default_checkpoint)] = default_checkpoint
    models_dir = Path("models")
    if models_dir.is_dir():
        for entry in sorted(models_dir.iterdir()):
            if (entry.is_dir() and (entry / "config.json").exists()
                    and (entry / "model.safetensors").exists()):
                _model_registry.setdefault(entry.name, str(entry))


@app.on_event("startup")
def load_model():
    global _detector, _agg_cfg, _logger, _active_model, _device, _fallback_threshold
    checkpoint = os.environ.get("GLASSES_CHECKPOINT") or _default_checkpoint()
    threshold = float(os.environ.get("GLASSES_THRESHOLD", "0.5"))
    device = os.environ.get("GLASSES_DEVICE", "cpu")
    _detector = GlassesDetector(checkpoint, device=device, threshold=threshold)
    _device, _fallback_threshold = device, threshold
    _discover_models(checkpoint)
    _active_model = _model_name(checkpoint)
    _loaded_detectors[_active_model] = _detector
    _agg_cfg = AggregateConfig.load()
    log_dir = os.environ.get("GLASSES_LOG_DIR")
    if log_dir:
        _logger = FrameLogger(log_dir, save_full=os.environ.get("GLASSES_LOG_FULL") == "1")
        _logger.log_runtime({
            "app_version": app.version,
            "checkpoint": checkpoint,
            "device": device,
            "save_full": _logger.save_full,
            "quality_version": QUALITY_VERSION,
            "preprocess_version": PREPROCESS_VERSION,
            "degradation_version": DEGRADE_VERSION,
            "degradation_profiles": BAD_CAMERA_PROFILES,
            "aggregate": _agg_cfg.to_dict(),
            "model_threshold": _detector.threshold,
            "model_band": _detector.band,
            "temperature": _detector.temperature,
            "pid": os.getpid(),
        })
    print(f"loaded {checkpoint} on {device} · aggregate={_agg_cfg.to_dict()}"
          + (f" · logging to {log_dir}" if log_dir else ""))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def list_models():
    return {
        "active": _active_model,
        "models": [{"name": name, "path": path, "loaded": name in _loaded_detectors}
                   for name, path in _model_registry.items()],
    }


@app.post("/models/select")
def select_model(name: str = Form(...)):
    """Swap the active detector. First selection of a model loads it (a few s)."""
    global _detector, _active_model
    if name not in _model_registry:
        raise HTTPException(status_code=404,
                            detail=f"unknown model {name!r}; have {sorted(_model_registry)}")
    with _model_lock:
        detector = _loaded_detectors.get(name)
        if detector is None:
            try:
                detector = GlassesDetector(_model_registry[name], device=_device,
                                           threshold=_fallback_threshold)
            except Exception as exc:
                raise HTTPException(status_code=500,
                                    detail=f"failed to load {name}: {exc}") from exc
            _loaded_detectors[name] = detector
        _detector = detector
        _active_model = name
    if _logger:
        _logger.log_runtime({
            "event": "model_switch",
            "model": name,
            "checkpoint": _model_registry[name],
            "model_threshold": detector.threshold,
            "model_band": detector.band,
            "temperature": detector.temperature,
        })
    return {"active": name, "checkpoint": _model_registry[name]}


def _band() -> tuple[float, float]:
    """(t_low, t_high) actually used by the detector."""
    if _detector.band is not None:
        return _detector.band
    t, b = _detector.threshold, _detector.uncertainty_band
    return t - b, t + b


def _decode(data: bytes) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="File is not a valid image")
    return bgr


def _frame_decision(r: GlassesResult) -> tuple[bool, bool]:
    """Raw model decision before capture-quality rejection."""
    if not r.face_found:
        return False, True
    if _agg_cfg.vote_on == "eyewear":
        p = r.eyewear_prob
        return p >= _agg_cfg.eyewear_t_high, _agg_cfg.eyewear_t_low <= p < _agg_cfg.eyewear_t_high
    return r.wearing_glasses, r.uncertain


def _quality_state(r: GlassesResult,
                   cfg: Optional[AggregateConfig] = None) -> tuple[bool, Optional[str]]:
    t_low, t_high = _band()
    frame = judge_frame(r, 0, t_low, t_high, cfg or _agg_cfg)
    return frame.valid, frame.reject_reason


def _frame_action(r: GlassesResult, cfg: Optional[AggregateConfig] = None) -> str:
    wearing, uncertain = _frame_decision(r)
    valid, _ = _quality_state(r, cfg)
    if not valid or uncertain:
        return "retry_capture"
    return "remove_glasses" if wearing else "pass"


def _raw_vote(r: GlassesResult) -> str:
    if not r.face_found:
        return "no_face"
    wearing, uncertain = _frame_decision(r)
    if uncertain:
        return "unsure"
    return "glasses" if wearing else "none"


def _to_response(r: GlassesResult, action: str, degradation_profile: str = "clean",
                 cfg: Optional[AggregateConfig] = None) -> dict:
    wearing, model_uncertain = _frame_decision(r)
    quality_valid, reject_reason = _quality_state(r, cfg)
    return dict(
        wearing_glasses=wearing,
        confidence=round(r.confidence, 4),
        probability=round(r.probability, 4),
        eyewear_prob=round(r.eyewear_prob, 4),
        uncertain=model_uncertain or not quality_valid,
        face_found=r.face_found,
        det_score=round(r.det_score, 4),
        class_probs=dict(zip(("none", "eyeglasses", "sunglasses"), r.class_probs)),
        action=action,
        blur_score=round(r.blur_score, 2),
        eye_dist_px=round(r.eye_dist_px, 1),
        brightness=round(r.brightness, 2),
        contrast=round(r.contrast, 2),
        quality_version=QUALITY_VERSION,
        quality_valid=quality_valid,
        quality_reject_reason=reject_reason,
        degradation_profile=degradation_profile,
    )


def _single_result_verdict(r: GlassesResult, profile: str) -> AggregateVerdict:
    cfg = _agg_cfg.for_profile(profile)
    t_low, t_high = _band()
    vote = judge_frame(r, 0, t_low, t_high, cfg)
    if vote.vote == "glasses":
        action, reason = "remove_glasses", "glasses"
    elif vote.vote == "none":
        action, reason = "pass", "ok"
    else:
        action, reason = "retry_capture", vote.reject_reason or "mixed"
    p = r.eyewear_prob if cfg.vote_on == "eyewear" else r.probability
    return AggregateVerdict(
        action=action,
        reason=reason,
        n_frames=1,
        n_valid=int(vote.valid),
        glasses_votes=int(vote.vote == "glasses"),
        none_votes=int(vote.vote == "none"),
        unsure_votes=int(vote.vote == "unsure"),
        mean_p=p if vote.valid else 0.0,
        max_p=p if vote.valid else 0.0,
        rejected={} if vote.valid else {reason: 1},
    )


def _evaluate_all_profiles(
    bgr: np.ndarray, index: int,
) -> tuple[dict, dict, dict[str, AggregateVerdict], AggregateVerdict]:
    evaluated = {}
    payloads = {}
    profile_verdicts = {}
    for profile in BAD_CAMERA_PROFILES:
        degraded = _degrade_frame(bgr, profile, index)
        inference_started = time.perf_counter()
        result = _detector.predict(degraded)
        inference_ms = (time.perf_counter() - inference_started) * 1000
        cfg = _agg_cfg.for_profile(profile)
        action = _frame_action(result, cfg)
        evaluated[profile] = (degraded, result, cfg, action, inference_ms)
        payloads[profile] = _to_response(result, action, profile, cfg)
        profile_verdicts[profile] = _single_result_verdict(result, profile)
    return evaluated, payloads, profile_verdicts, combine_profile_verdicts(profile_verdicts)


def _log_profile_evaluations(evaluated: dict, combined: AggregateVerdict, *, truth: str,
                             session_id: str, source_frame_id: str,
                             frame_idx: int, window_id: str = "",
                             request_id: str = "", source_path: str = "",
                             profile_verdicts: Optional[dict] = None,
                             ts: Optional[float] = None) -> None:
    if not _logger:
        return
    ts = time.time() if ts is None else ts
    t_low, t_high = _band()
    for profile, (bgr, result, cfg, action, inference_ms) in evaluated.items():
        vote = judge_frame(result, frame_idx, t_low, t_high, cfg)
        profile_verdict = (profile_verdicts or {}).get(profile)
        _logger.log(
            result,
            pred_action=action,
            truth=truth,
            session_id=session_id,
            source_frame_id=source_frame_id,
            window_id=window_id,
            request_id=request_id,
            source_path=source_path,
            frame_idx=frame_idx,
            raw_vote=_raw_vote(result),
            valid=vote.valid,
            reject_reason=vote.reject_reason,
            batch_action=profile_verdict.action if profile_verdict else "",
            batch_reason=profile_verdict.reason if profile_verdict else "",
            degradation_profile=profile,
            combined_action=combined.action,
            combined_reason=combined.reason,
            quality_thresholds=_threshold_log(cfg),
            inference_ms=inference_ms,
            full_bgr=bgr,
            ts=ts,
        )


def _threshold_log(cfg: AggregateConfig) -> dict:
    t_low, t_high = _band()
    vote_low, vote_high = ((cfg.eyewear_t_low, cfg.eyewear_t_high)
                           if cfg.vote_on == "eyewear" else (t_low, t_high))
    return {
        "min_det_score": cfg.min_det_score,
        "min_blur": cfg.min_blur,
        "min_eye_dist": cfg.min_eye_dist,
        "min_brightness": cfg.min_brightness,
        "max_brightness": cfg.max_brightness,
        "min_contrast": cfg.min_contrast,
        "vote_on": cfg.vote_on,
        "vote_low": vote_low,
        "vote_high": vote_high,
    }


def _combined_validation_response(payloads: dict[str, dict],
                                  verdict: AggregateVerdict) -> ValidationResponse:
    combined = dict(payloads["clean"])
    combined.update(
        wearing_glasses=verdict.action == "remove_glasses",
        uncertain=verdict.action == "retry_capture",
        action=verdict.action,
        quality_valid=verdict.action != "retry_capture",
        quality_reject_reason=verdict.reason if verdict.action == "retry_capture" else None,
        degradation_profile="all",
        profile_results=payloads,
    )
    return ValidationResponse(**combined)


def _batch_verdict(v: AggregateVerdict) -> BatchVerdict:
    payload = v.to_dict()
    payload.pop("frames")
    return BatchVerdict(**payload)


def _verdict_log_dict(verdict: AggregateVerdict) -> dict:
    payload = verdict.to_dict()
    payload.pop("frames")
    return payload


def _log_decision_record(*, session_id: str, window_id: str, decision_no: int,
                         request_kind: str, truth: str, n_source_frames: int,
                         degradation_profile: str,
                         profile_verdicts: dict[str, AggregateVerdict],
                         combined: AggregateVerdict,
                         ts: Optional[float] = None) -> None:
    if not _logger:
        return
    _logger.log_decision(
        session_id=session_id,
        window_id=window_id,
        decision_no=decision_no,
        request_kind=request_kind,
        truth=truth,
        n_source_frames=n_source_frames,
        degradation_profile=degradation_profile,
        profile_verdicts={name: _verdict_log_dict(verdict)
                          for name, verdict in profile_verdicts.items()},
        combined_verdict=_verdict_log_dict(combined),
        ts=ts,
    )


def _batch_frame(r: GlassesResult, profile: str, index: int,
                 cfg: AggregateConfig) -> BatchFrame:
    t_low, t_high = _band()
    vote = judge_frame(r, index, t_low, t_high, cfg)
    action = _frame_action(r, cfg)
    return BatchFrame(
        **_to_response(r, action, profile, cfg),
        index=index,
        valid=vote.valid,
        reject_reason=vote.reject_reason,
        vote=vote.vote,
    )


def _degrade_frame(bgr: np.ndarray, profile: str, index: int) -> np.ndarray:
    try:
        return apply_bad_camera(bgr, profile, index=index)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _check_truth(truth: str) -> str:
    if truth not in TRUTH_VALUES:
        raise HTTPException(status_code=400, detail=f"truth must be one of {TRUTH_VALUES}")
    return truth


def _log_request_records(records: list[dict], *, session_id: str, window_id: str,
                         endpoint: str, request_kind: str, truth: str,
                         degradation_profile: str, request_started: float,
                         request_ts: float, status: int = 200,
                         error: str = "") -> None:
    if not _logger:
        return
    total_ms = (time.perf_counter() - request_started) * 1000
    for record in records:
        _logger.log_request(
            request_id=record["request_id"], session_id=session_id,
            source_frame_id=record["source_frame_id"], window_id=window_id,
            frame_idx=record["frame_idx"], endpoint=endpoint,
            request_kind=request_kind, truth=truth,
            degradation_profile=degradation_profile,
            width=record.get("width", 0), height=record.get("height", 0),
            encoded_bytes=record.get("encoded_bytes", 0), status=status,
            error=error, total_ms=total_ms,
            source_path=record.get("source_path", ""), ts=request_ts,
        )


@app.post("/validate/glasses", response_model=ValidationResponse)
def validate_glasses(face_image: UploadFile = File(...),
                     truth: str = Form("unknown"), session_id: str = Form(""),
                     degradation_profile: str = Form("clean")):
    """Single-frame verdict (legacy / photo upload). Sync def -> runs in the threadpool."""
    request_started = time.perf_counter()
    request_ts = time.time()
    request_id = uuid.uuid4().hex
    log_session = session_id or uuid.uuid4().hex[:12]
    window_id = f"{log_session}:single:{uuid.uuid4().hex[:8]}"
    source_frame_id = f"{log_session}:0"
    encoded = b""
    decoded = None
    source_path = ""
    request_status = 200
    request_error = ""
    try:
        truth = _check_truth(truth)
        encoded = face_image.file.read()
        decoded = _decode(encoded)
        if _logger:
            source_path = _logger.save_source_bytes(
                encoded, filename=face_image.filename or "frame.img",
                request_id=request_id, frame_idx=0, ts=request_ts,
            )
        if degradation_profile == "all":
            evaluated, payloads, profile_verdicts, combined = _evaluate_all_profiles(
                decoded, index=0
            )
            _log_profile_evaluations(
                evaluated,
                combined,
                truth=truth,
                session_id=log_session,
                source_frame_id=source_frame_id,
                frame_idx=0,
                window_id=window_id,
                request_id=request_id,
                source_path=source_path,
                ts=request_ts,
            )
            _log_decision_record(
                session_id=log_session,
                window_id=window_id,
                decision_no=1,
                request_kind="single",
                truth=truth,
                n_source_frames=1,
                degradation_profile="all",
                profile_verdicts=profile_verdicts,
                combined=combined,
                ts=request_ts,
            )
            return _combined_validation_response(payloads, combined)

        bgr = _degrade_frame(decoded, degradation_profile, index=0)
        inference_started = time.perf_counter()
        result = _detector.predict(bgr)
        inference_ms = (time.perf_counter() - inference_started) * 1000
        profile_cfg = _agg_cfg.for_profile(degradation_profile)
        action = _frame_action(result, profile_cfg)
        frame_verdict = _single_result_verdict(result, degradation_profile)
        if _logger:
            _logger.log(
                result, pred_action=action, truth=truth, session_id=log_session,
                source_frame_id=source_frame_id, window_id=window_id,
                request_id=request_id, source_path=source_path, frame_idx=0,
                raw_vote=_raw_vote(result), valid=bool(frame_verdict.n_valid),
                reject_reason=(None if frame_verdict.n_valid else frame_verdict.reason),
                batch_action=frame_verdict.action, batch_reason=frame_verdict.reason,
                combined_action=frame_verdict.action, combined_reason=frame_verdict.reason,
                degradation_profile=degradation_profile,
                quality_thresholds=_threshold_log(profile_cfg),
                inference_ms=inference_ms, full_bgr=bgr, ts=request_ts,
            )
            _log_decision_record(
                session_id=log_session, window_id=window_id, decision_no=1,
                request_kind="single", truth=truth, n_source_frames=1,
                degradation_profile=degradation_profile,
                profile_verdicts={degradation_profile: frame_verdict},
                combined=frame_verdict, ts=request_ts,
            )
        return ValidationResponse(
            **_to_response(result, action, degradation_profile, profile_cfg)
        )
    except HTTPException as exc:
        request_status = exc.status_code
        request_error = str(exc.detail)
        raise
    except Exception as exc:
        request_status = 500
        request_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if _logger:
            height, width = decoded.shape[:2] if decoded is not None else (0, 0)
            _logger.log_request(
                request_id=request_id, session_id=log_session,
                source_frame_id=source_frame_id, window_id=window_id, frame_idx=0,
                endpoint="/validate/glasses", request_kind="single", truth=truth,
                degradation_profile=degradation_profile, width=width, height=height,
                encoded_bytes=len(encoded), status=request_status, error=request_error,
                total_ms=(time.perf_counter() - request_started) * 1000,
                source_path=source_path, ts=request_ts,
            )


@app.post("/validate/glasses/batch", response_model=BatchResponse)
def validate_glasses_batch(face_images: List[UploadFile] = File(...),
                           truth: str = Form("unknown"), session_id: str = Form(""),
                           degradation_profile: str = Form("clean")):
    """Burst verdict: send the 5 frames captured at 5 fps, get ONE decision.

    Frames failing the quality gate (face, distance, blur, exposure, or contrast)
    do not vote. See aggregate.aggregate() for the rule.
    """
    request_started = time.perf_counter()
    request_ts = time.time()
    truth = _check_truth(truth)
    if not 1 <= len(face_images) <= MAX_BATCH:
        raise HTTPException(status_code=400, detail=f"send 1..{MAX_BATCH} frames")
    session_id = session_id or uuid.uuid4().hex[:12]
    t_low, t_high = _band()

    encoded_frames = [file.file.read() for file in face_images]
    source_names = [file.filename or "frame.img" for file in face_images]
    window_id = f"{session_id}:batch:{uuid.uuid4().hex[:8]}"
    request_records = [{
            "request_id": uuid.uuid4().hex,
            "source_frame_id": f"{session_id}:{index}",
            "frame_idx": index,
            "width": 0,
            "height": 0,
            "encoded_bytes": len(encoded),
            "source_path": "",
        } for index, encoded in enumerate(encoded_frames)]
    decoded = []
    try:
        for record, encoded, source_name in zip(
            request_records, encoded_frames, source_names
        ):
            bgr = _decode(encoded)
            decoded.append(bgr)
            record["width"], record["height"] = bgr.shape[1], bgr.shape[0]
            if _logger:
                record["source_path"] = _logger.save_source_bytes(
                    encoded, filename=source_name, request_id=record["request_id"],
                    frame_idx=record["frame_idx"], ts=request_ts,
                )
    except HTTPException as exc:
        _log_request_records(
            request_records, session_id=session_id, window_id=window_id,
            endpoint="/validate/glasses/batch", request_kind="batch", truth=truth,
            degradation_profile=degradation_profile, request_started=request_started,
            request_ts=request_ts, status=exc.status_code, error=str(exc.detail),
        )
        raise
    except Exception as exc:
        _log_request_records(
            request_records, session_id=session_id, window_id=window_id,
            endpoint="/validate/glasses/batch", request_kind="batch", truth=truth,
            degradation_profile=degradation_profile, request_started=request_started,
            request_ts=request_ts, status=500,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    if degradation_profile == "all":
        source_evaluations = []
        results_by_profile = {profile: [] for profile in BAD_CAMERA_PROFILES}
        try:
            for index, bgr in enumerate(decoded):
                evaluated, _, _, _ = _evaluate_all_profiles(bgr, index)
                source_evaluations.append(evaluated)
                for profile, (_, result, _, _, _) in evaluated.items():
                    results_by_profile[profile].append(result)
        except Exception as exc:
            _log_request_records(
                request_records, session_id=session_id, window_id=window_id,
                endpoint="/validate/glasses/batch", request_kind="batch", truth=truth,
                degradation_profile="all", request_started=request_started,
                request_ts=request_ts, status=500,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

        profile_verdict_objects = {
            profile: aggregate(results, t_low, t_high, _agg_cfg.for_profile(profile))
            for profile, results in results_by_profile.items()
        }
        combined = combine_profile_verdicts(profile_verdict_objects)
        profile_frames = {
            profile: [
                _batch_frame(result, profile, index, _agg_cfg.for_profile(profile))
                for index, result in enumerate(results)
            ]
            for profile, results in results_by_profile.items()
        }

        for index, evaluated in enumerate(source_evaluations):
            _log_profile_evaluations(
                evaluated,
                combined,
                truth=truth,
                session_id=session_id,
                source_frame_id=f"{session_id}:{index}",
                frame_idx=index,
                window_id=window_id,
                request_id=request_records[index]["request_id"],
                source_path=request_records[index]["source_path"],
                profile_verdicts=profile_verdict_objects,
                ts=request_ts,
            )
        _log_decision_record(
            session_id=session_id, window_id=window_id, decision_no=1,
            request_kind="batch", truth=truth, n_source_frames=len(decoded),
            degradation_profile="all", profile_verdicts=profile_verdict_objects,
            combined=combined, ts=request_ts,
        )
        _log_request_records(
            request_records, session_id=session_id, window_id=window_id,
            endpoint="/validate/glasses/batch", request_kind="batch", truth=truth,
            degradation_profile="all", request_started=request_started,
            request_ts=request_ts,
        )
        return BatchResponse(
            session_id=session_id,
            verdict=_batch_verdict(combined),
            frames=profile_frames["clean"],
            profile_frames=profile_frames,
            profile_verdicts={name: _batch_verdict(v)
                              for name, v in profile_verdict_objects.items()},
            config={
                "base": _agg_cfg.to_dict(),
                "profiles": {name: _agg_cfg.for_profile(name).to_dict()
                             for name in BAD_CAMERA_PROFILES},
            },
        )

    try:
        frames_bgr = [_degrade_frame(bgr, degradation_profile, i)
                      for i, bgr in enumerate(decoded)]
        results = []
        inference_times = []
        for bgr in frames_bgr:
            inference_started = time.perf_counter()
            results.append(_detector.predict(bgr))
            inference_times.append((time.perf_counter() - inference_started) * 1000)
    except HTTPException as exc:
        _log_request_records(
            request_records, session_id=session_id, window_id=window_id,
            endpoint="/validate/glasses/batch", request_kind="batch", truth=truth,
            degradation_profile=degradation_profile, request_started=request_started,
            request_ts=request_ts, status=exc.status_code, error=str(exc.detail),
        )
        raise
    except Exception as exc:
        _log_request_records(
            request_records, session_id=session_id, window_id=window_id,
            endpoint="/validate/glasses/batch", request_kind="batch", truth=truth,
            degradation_profile=degradation_profile, request_started=request_started,
            request_ts=request_ts, status=500,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    profile_cfg = _agg_cfg.for_profile(degradation_profile)
    verdict = aggregate(results, t_low, t_high, profile_cfg)

    frames = []
    for r, v, bgr, inference_ms in zip(
        results, verdict.frames, frames_bgr, inference_times
    ):
        action = _frame_action(r, profile_cfg)
        frames.append(BatchFrame(**_to_response(r, action, degradation_profile, profile_cfg), index=v.index, valid=v.valid,
                                 reject_reason=v.reject_reason, vote=v.vote))
        if _logger:
            _logger.log(r, pred_action=action, truth=truth, session_id=session_id,
                        source_frame_id=f"{session_id}:{v.index}", window_id=window_id,
                        request_id=request_records[v.index]["request_id"],
                        source_path=request_records[v.index]["source_path"],
                        frame_idx=v.index,
                        raw_vote=_raw_vote(r), valid=v.valid, reject_reason=v.reject_reason,
                        batch_action=verdict.action, batch_reason=verdict.reason,
                        combined_action=verdict.action, combined_reason=verdict.reason,
                        degradation_profile=degradation_profile,
                        quality_thresholds=_threshold_log(profile_cfg),
                        inference_ms=inference_ms, full_bgr=bgr, ts=request_ts)

    _log_decision_record(
        session_id=session_id, window_id=window_id, decision_no=1,
        request_kind="batch", truth=truth, n_source_frames=len(decoded),
        degradation_profile=degradation_profile,
        profile_verdicts={degradation_profile: verdict}, combined=verdict, ts=request_ts,
    )
    _log_request_records(
        request_records, session_id=session_id, window_id=window_id,
        endpoint="/validate/glasses/batch", request_kind="batch", truth=truth,
        degradation_profile=degradation_profile, request_started=request_started,
        request_ts=request_ts,
    )

    vd = verdict.to_dict(); vd.pop("frames")
    return BatchResponse(session_id=session_id, verdict=BatchVerdict(**vd),
                         frames=frames, config=profile_cfg.to_dict())


# ---------------------------------------------------------------------------
# Streaming mode: frames arrive one at a time (5 fps on the line); the server
# keeps a small per-session window and emits ONE decision every n_frames frames.
# ---------------------------------------------------------------------------
class StreamResponse(BaseModel):
    session_id: str
    frame_no: int                      # 1-based count of frames in this session
    frame: BatchFrame                  # this frame's result + quality-gate vote
    window: int                        # frames in the current (incomplete) window
    decision_every: int                # = AggregateConfig.n_frames
    verdict: Optional[BatchVerdict] = None     # set on every n_frames-th frame only
    profile_frames: dict[str, BatchFrame] = Field(default_factory=dict)
    profile_verdicts: dict[str, BatchVerdict] = Field(default_factory=dict)
    source_action: str = "retry_capture"    # combined action for the current source frame
    decision_no: int = 0               # how many decisions this session has produced


class _Session:
    __slots__ = ("results", "profile_results", "frame_no", "decision_no",
                 "last_seen", "active_requests", "profile", "lock", "check_id")

    def __init__(self):
        self.results: list[GlassesResult] = []
        self.profile_results: dict[str, list[GlassesResult]] = {}
        self.frame_no = 0
        self.decision_no = 0
        self.last_seen = time.time()
        self.active_requests = 0
        self.profile = ""
        self.lock = threading.Lock()
        self.check_id = uuid.uuid4().hex[:8]


_sessions: dict[str, _Session] = {}
_sessions_lock = threading.Lock()
SESSION_TTL_S = 60.0


@contextmanager
def _session_scope(session_id: str):
    now = time.time()
    with _sessions_lock:
        # cheap GC: drop sessions idle for longer than SESSION_TTL_S
        for sid in [s for s, v in _sessions.items()
                    if v.active_requests == 0 and now - v.last_seen > SESSION_TTL_S]:
            _sessions.pop(sid, None)
        s = _sessions.get(session_id)
        if s is None:
            s = _sessions[session_id] = _Session()
        s.last_seen = now
        s.active_requests += 1
    try:
        yield s
    finally:
        with _sessions_lock:
            s.active_requests -= 1
            s.last_seen = time.time()


@app.post("/validate/glasses/stream", response_model=StreamResponse)
def validate_glasses_stream(face_image: UploadFile = File(...), session_id: str = Form(...),
                            truth: str = Form("unknown"), reset: bool = Form(False),
                            degradation_profile: str = Form("clean")):
    """Send frames one by one with the same session_id; every n_frames-th frame
    returns a `verdict` for that window (tumbling window, no overlap). Per-frame
    results come back on every call so the UI can show live feedback.
    Pass reset=true on the first frame of a new check to clear the window."""
    request_started = time.perf_counter()
    request_ts = time.time()
    request_id = uuid.uuid4().hex
    encoded = b""
    bgr = None
    source_path = ""
    source_frame_id = f"{session_id}:pending"
    window_id = ""
    frame_idx = 0
    request_status = 200
    request_error = ""
    try:
        truth = _check_truth(truth)
        encoded = face_image.file.read()
        bgr = _decode(encoded)
        if _logger:
            source_path = _logger.save_source_bytes(
                encoded, filename=face_image.filename or "frame.img",
                request_id=request_id, frame_idx=0, ts=request_ts,
            )
        with _session_scope(session_id) as sess:
            response, window_id, source_frame_id, frame_idx = _validate_glasses_stream_frame(
                bgr, sess, session_id, truth, reset, degradation_profile,
                request_id=request_id, source_path=source_path, request_ts=request_ts,
            )
        return response
    except HTTPException as exc:
        request_status = exc.status_code
        request_error = str(exc.detail)
        raise
    except Exception as exc:
        request_status = 500
        request_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if _logger:
            height, width = bgr.shape[:2] if bgr is not None else (0, 0)
            _logger.log_request(
                request_id=request_id, session_id=session_id,
                source_frame_id=source_frame_id, window_id=window_id,
                frame_idx=frame_idx, endpoint="/validate/glasses/stream",
                request_kind="stream", truth=truth,
                degradation_profile=degradation_profile, width=width, height=height,
                encoded_bytes=len(encoded), status=request_status, error=request_error,
                total_ms=(time.perf_counter() - request_started) * 1000,
                source_path=source_path, ts=request_ts,
            )


def _validate_glasses_stream_frame(bgr: np.ndarray, sess: _Session, session_id: str,
                                   truth: str, reset: bool,
                                   degradation_profile: str, *, request_id: str = "",
                                   source_path: str = "",
                                   request_ts: Optional[float] = None
                                   ) -> tuple[StreamResponse, str, str, int]:
    t_low, t_high = _band()
    n = max(1, _agg_cfg.n_frames)

    if degradation_profile != "all" and degradation_profile not in BAD_CAMERA_PROFILES:
        raise HTTPException(status_code=400,
                            detail=f"unknown bad-camera profile: {degradation_profile}")

    with sess.lock:
        if reset:
            sess.results.clear(); sess.profile_results.clear()
            sess.frame_no = 0; sess.decision_no = 0
            sess.check_id = uuid.uuid4().hex[:8]
        elif sess.profile and sess.profile != degradation_profile:
            sess.results.clear(); sess.profile_results.clear()
            sess.check_id = uuid.uuid4().hex[:8]
        sess.profile = degradation_profile
        check_id = sess.check_id
        window_id = f"{session_id}:{check_id}:w{sess.decision_no + 1}"
        degradation_index = sess.frame_no
        if degradation_profile == "all":
            evaluated, _, source_profile_verdicts, source_combined = _evaluate_all_profiles(
                bgr, degradation_index
            )
            if not sess.profile_results:
                sess.profile_results = {profile: [] for profile in BAD_CAMERA_PROFILES}
            for profile, (_, result, _, _, _) in evaluated.items():
                sess.profile_results[profile].append(result)
            sess.frame_no += 1
            frame_no = sess.frame_no
            profile_frames = {
                profile: _batch_frame(result, profile, frame_no - 1, cfg)
                for profile, (_, result, cfg, _, _) in evaluated.items()
            }
            profile_verdict_objects = {}
            verdict = None
            if len(sess.profile_results["clean"]) >= n:
                profile_verdict_objects = {
                    profile: aggregate(results, t_low, t_high,
                                       _agg_cfg.for_profile(profile))
                    for profile, results in sess.profile_results.items()
                }
                verdict = combine_profile_verdicts(profile_verdict_objects)
                sess.profile_results.clear()
                sess.decision_no += 1
            decision_no = sess.decision_no
            window = len(sess.profile_results.get("clean", []))
        else:
            profile_cfg = _agg_cfg.for_profile(degradation_profile)
            bgr = _degrade_frame(bgr, degradation_profile, degradation_index)
            inference_started = time.perf_counter()
            result = _detector.predict(bgr)
            inference_ms = (time.perf_counter() - inference_started) * 1000
            action = _frame_action(result, profile_cfg)
            sess.results.append(result)
            sess.frame_no += 1
            frame_no = sess.frame_no
            verdict = None
            if len(sess.results) >= n:
                verdict = aggregate(sess.results, t_low, t_high, profile_cfg)
                sess.results.clear()
                sess.decision_no += 1
            decision_no = sess.decision_no
            window = len(sess.results)
        source_frame_id = f"{session_id}:{check_id}:{frame_no - 1}"

    if degradation_profile == "all":
        log_combined = verdict or source_combined
        ts = time.time() if request_ts is None else request_ts
        _log_profile_evaluations(
            evaluated,
            log_combined,
            truth=truth,
            session_id=session_id,
            source_frame_id=source_frame_id,
            frame_idx=frame_no - 1,
            window_id=window_id,
            request_id=request_id,
            source_path=source_path,
            profile_verdicts=profile_verdict_objects or None,
            ts=ts,
        )
        if verdict:
            _log_decision_record(
                session_id=session_id, window_id=window_id,
                decision_no=decision_no, request_kind="stream", truth=truth,
                n_source_frames=n, degradation_profile="all",
                profile_verdicts=profile_verdict_objects,
                combined=verdict, ts=ts,
            )
        response = StreamResponse(
            session_id=session_id,
            frame_no=frame_no,
            frame=profile_frames["clean"],
            profile_frames=profile_frames,
            window=window,
            decision_every=n,
            verdict=_batch_verdict(verdict) if verdict else None,
            profile_verdicts={name: _batch_verdict(v)
                              for name, v in profile_verdict_objects.items()},
            source_action=source_combined.action,
            decision_no=decision_no,
        )
        return response, window_id, source_frame_id, frame_no - 1

    # per-frame quality vote (same gate the window uses)
    fv = judge_frame(result, frame_no - 1, t_low, t_high, profile_cfg)
    frame = BatchFrame(**_to_response(result, action, degradation_profile, profile_cfg), index=frame_no - 1, valid=fv.valid,
                       reject_reason=fv.reject_reason, vote=fv.vote)
    if _logger:
        ts = time.time() if request_ts is None else request_ts
        _logger.log(result, pred_action=action, truth=truth, session_id=session_id,
                    source_frame_id=source_frame_id, window_id=window_id,
                    request_id=request_id, source_path=source_path,
                    frame_idx=frame_no - 1,
                    raw_vote=_raw_vote(result), valid=fv.valid, reject_reason=fv.reject_reason,
                    batch_action=verdict.action if verdict else "",
                    batch_reason=verdict.reason if verdict else "",
                    combined_action=action,
                    combined_reason=(fv.reject_reason or fv.vote or "mixed"),
                    degradation_profile=degradation_profile,
                    quality_thresholds=_threshold_log(profile_cfg),
                    inference_ms=inference_ms, full_bgr=bgr, ts=ts)
        if verdict:
            _log_decision_record(
                session_id=session_id, window_id=window_id,
                decision_no=decision_no, request_kind="stream", truth=truth,
                n_source_frames=n, degradation_profile=degradation_profile,
                profile_verdicts={degradation_profile: verdict},
                combined=verdict, ts=ts,
            )
    vd = None
    if verdict:
        d = verdict.to_dict(); d.pop("frames"); vd = BatchVerdict(**d)
    response = StreamResponse(
        session_id=session_id, frame_no=frame_no, frame=frame, window=window,
        decision_every=n, verdict=vd, source_action=action, decision_no=decision_no,
    )
    return response, window_id, source_frame_id, frame_no - 1


_PAGE_PATH = Path(__file__).parent / "static" / "index.html"


@app.get("/", response_class=HTMLResponse)
def index():
    """Test UI. Read per request so frontend edits show without a server restart."""
    return _PAGE_PATH.read_text(encoding="utf-8")
